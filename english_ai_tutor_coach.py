import json
import os
import random
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher

import streamlit as st
from openai import OpenAI

APP_TITLE = "영어 튜터 코치"
DB_PATH = "english_tutor_coach.db"
DEFAULT_MODEL = "gpt-4o-mini"


# -----------------------------
# DB
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn



def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_text TEXT NOT NULL,
            expression TEXT NOT NULL,
            meaning_kr TEXT NOT NULL,
            nuance_kr TEXT NOT NULL,
            origin_kr TEXT NOT NULL,
            similar_json TEXT NOT NULL,
            examples_json TEXT NOT NULL,
            accepted_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stats (
            note_id INTEGER PRIMARY KEY,
            seen_count INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            wrong_count INTEGER NOT NULL DEFAULT 0,
            weight INTEGER NOT NULL DEFAULT 1,
            streak INTEGER NOT NULL DEFAULT 0,
            last_result TEXT,
            last_tested_at TEXT,
            FOREIGN KEY(note_id) REFERENCES notes(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS coach_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            ai_message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


# -----------------------------
# Helpers
# -----------------------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")



def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)



def json_loads(value, default):
    try:
        return json.loads(value)
    except Exception:
        return default



def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())



def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()



def runtime_client():
    api_key = st.session_state.get("runtime_api_key", "").strip() or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# -----------------------------
# OpenAI features
# -----------------------------
def fallback_analysis(text: str):
    cleaned = text.strip()
    return {
        "source_text": cleaned,
        "expression": cleaned,
        "meaning_kr": "API 키를 넣으면 더 정확한 분석이 됩니다.",
        "nuance_kr": "현재는 기본 분석 모드입니다.",
        "origin_kr": "어원/유래는 API 연결 시 더 정확하게 생성됩니다.",
        "similar_expressions": ["related expression 1", "related expression 2"],
        "example_sentences": [
            f"I want to learn how to use '{cleaned}' naturally.",
            f"Can you give me a natural example of '{cleaned}'?",
        ],
        "accepted_answers": [cleaned],
    }



def analyze_text(text: str):
    client = runtime_client()
    if client is None:
        return fallback_analysis(text), "API 키가 없어 기본 분석 모드로 진행했습니다."

    prompt = f"""
다음 영어 표현, 문장, 또는 질문을 영어 학습용 오답노트 형태로 분석하세요.

입력:
{text}

반드시 JSON만 반환하세요.
형식:
{{
  "source_text": "원문",
  "expression": "핵심 표현 또는 학습 대상",
  "meaning_kr": "핵심 뜻",
  "nuance_kr": "뉘앙스와 사용 상황",
  "origin_kr": "유래 또는 기억법",
  "similar_expressions": ["비슷한 표현1", "비슷한 표현2", "비슷한 표현3"],
  "example_sentences": ["예문1", "예문2", "예문3"],
  "accepted_answers": ["대표 정답", "허용 답안1", "허용 답안2"]
}}

규칙:
- explanation 계열은 모두 한국어
- example_sentences는 영어
- accepted_answers에는 핵심 표현을 반드시 포함
- JSON 외 텍스트 금지
"""
    try:
        response = client.responses.create(model=st.session_state.get("model_name", DEFAULT_MODEL), input=prompt)
        data = json.loads(response.output_text)
        return data, "AI 분석이 완료되었습니다."
    except Exception as e:
        return fallback_analysis(text), f"AI 분석에 실패해 기본 분석으로 전환했습니다. ({str(e)[:120]})"



def coach_reply(user_text: str):
    client = runtime_client()
    if client is None:
        return None, "API 키가 없어 코치 대화를 사용할 수 없습니다."

    history = get_recent_coach_history(limit=8)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal English speaking and writing coach for a Korean learner. "
                "Reply mainly in Korean, but always include natural English corrections/examples when helpful. "
                "Train the user. Do not just explain; coach them with short drills, corrections, and one follow-up task."
            ),
        }
    ]
    for row in reversed(history):
        messages.append({"role": "user", "content": row["user_message"]})
        messages.append({"role": "assistant", "content": row["ai_message"]})
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.responses.create(model=st.session_state.get("model_name", DEFAULT_MODEL), input=messages)
        return response.output_text.strip(), "코치 답변이 생성되었습니다."
    except Exception as e:
        return None, f"코치 답변 생성에 실패했습니다. ({str(e)[:120]})"



def grade_answer(prompt_text: str, accepted_answers: list[str], user_answer: str, expression: str):
    # Fast local check first
    user_norm = normalize_text(user_answer)
    accepted_norm = [normalize_text(a) for a in accepted_answers]
    if user_norm in accepted_norm:
        return True, "정답", accepted_answers[0], "대표 정답과 일치합니다."
    if any(similarity(user_norm, a) >= 0.92 for a in accepted_norm):
        return True, "거의 정답", accepted_answers[0], "표현이 거의 동일합니다."

    client = runtime_client()
    if client is None:
        return False, "오답", accepted_answers[0], "API 키가 없어서 정밀 채점을 하지 않았습니다."

    prompt = f"""
다음 영어 학습 문제의 답안을 채점하세요.

문제: {prompt_text}
대표 표현: {expression}
허용 답안: {json.dumps(accepted_answers, ensure_ascii=False)}
사용자 답안: {user_answer}

반드시 JSON만 반환하세요.
형식:
{{
  "correct": true 또는 false,
  "label": "정답/부분정답/오답",
  "best_answer": "가장 대표 정답",
  "feedback_kr": "왜 맞는지/틀리는지 짧은 한국어 설명"
}}
"""
    try:
        response = client.responses.create(model=st.session_state.get("model_name", DEFAULT_MODEL), input=prompt)
        data = json.loads(response.output_text)
        return bool(data.get("correct", False)), data.get("label", "채점결과"), data.get("best_answer", accepted_answers[0]), data.get("feedback_kr", "")
    except Exception:
        return False, "오답", accepted_answers[0], "정밀 채점 실패로 오답 처리했습니다."


# -----------------------------
# DB CRUD
# -----------------------------
def save_note(data: dict):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM notes WHERE lower(expression)=lower(?)", (data["expression"].strip(),))
    found = cur.fetchone()

    payload = (
        data["source_text"].strip(),
        data["expression"].strip(),
        data["meaning_kr"].strip(),
        data["nuance_kr"].strip(),
        data["origin_kr"].strip(),
        json_dumps(data["similar_expressions"]),
        json_dumps(data["example_sentences"]),
        json_dumps(data["accepted_answers"]),
    )

    if found:
        note_id = found["id"]
        cur.execute(
            """
            UPDATE notes
            SET source_text=?, expression=?, meaning_kr=?, nuance_kr=?, origin_kr=?,
                similar_json=?, examples_json=?, accepted_json=?
            WHERE id=?
            """,
            (*payload, note_id),
        )
    else:
        cur.execute(
            """
            INSERT INTO notes (
                source_text, expression, meaning_kr, nuance_kr, origin_kr,
                similar_json, examples_json, accepted_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*payload, now_iso()),
        )
        note_id = cur.lastrowid
        cur.execute(
            "INSERT OR IGNORE INTO stats (note_id, seen_count, correct_count, wrong_count, weight, streak) VALUES (?, 0, 0, 0, 1, 0)",
            (note_id,),
        )

    conn.commit()
    conn.close()



def list_notes(search_text="", sort_by="weight"):
    conn = get_conn()
    cur = conn.cursor()
    order_clause = {
        "weight": "s.weight DESC, s.wrong_count DESC, n.created_at DESC",
        "wrong": "s.wrong_count DESC, s.weight DESC, n.created_at DESC",
        "latest": "n.created_at DESC",
        "name": "n.expression COLLATE NOCASE ASC",
    }.get(sort_by, "s.weight DESC, s.wrong_count DESC, n.created_at DESC")

    keyword = f"%{search_text.strip()}%"
    cur.execute(
        f"""
        SELECT n.*, s.seen_count, s.correct_count, s.wrong_count, s.weight, s.streak
        FROM notes n
        LEFT JOIN stats s ON n.id = s.note_id
        WHERE n.expression LIKE ? OR n.meaning_kr LIKE ? OR n.source_text LIKE ?
        ORDER BY {order_clause}
        """,
        (keyword, keyword, keyword),
    )
    rows = cur.fetchall()
    conn.close()
    return rows



def get_note_count():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM notes")
    cnt = cur.fetchone()["cnt"]
    conn.close()
    return cnt



def get_stats_summary():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_notes,
            COALESCE(SUM(seen_count), 0) AS total_seen,
            COALESCE(SUM(correct_count), 0) AS total_correct,
            COALESCE(SUM(wrong_count), 0) AS total_wrong
        FROM stats
        """
    )
    summary = cur.fetchone()
    cur.execute(
        """
        SELECT n.expression, s.weight, s.wrong_count, s.correct_count
        FROM stats s
        JOIN notes n ON s.note_id = n.id
        ORDER BY s.weight DESC, s.wrong_count DESC, n.expression COLLATE NOCASE ASC
        LIMIT 10
        """
    )
    weak = cur.fetchall()
    conn.close()
    return summary, weak



def update_stat(note_id: int, correct: bool):
    conn = get_conn()
    cur = conn.cursor()
    if correct:
        cur.execute(
            """
            UPDATE stats
            SET seen_count = seen_count + 1,
                correct_count = correct_count + 1,
                streak = streak + 1,
                weight = CASE WHEN weight > 1 THEN weight - 1 ELSE 1 END,
                last_result = 'correct',
                last_tested_at = ?
            WHERE note_id = ?
            """,
            (now_iso(), note_id),
        )
    else:
        cur.execute(
            """
            UPDATE stats
            SET seen_count = seen_count + 1,
                wrong_count = wrong_count + 1,
                streak = 0,
                weight = weight + 2,
                last_result = 'wrong',
                last_tested_at = ?
            WHERE note_id = ?
            """,
            (now_iso(), note_id),
        )
    conn.commit()
    conn.close()



def save_coach_history(user_message: str, ai_message: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO coach_history (user_message, ai_message, created_at) VALUES (?, ?, ?)",
        (user_message, ai_message, now_iso()),
    )
    conn.commit()
    conn.close()



def get_recent_coach_history(limit=10):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM coach_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows



def build_questions(limit=10):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT n.*, s.weight, s.wrong_count, s.correct_count
        FROM notes n
        JOIN stats s ON n.id = s.note_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    pool = []
    for row in rows:
        pool.extend([dict(row)] * max(1, row["weight"]))

    picked = []
    used = set()
    while pool and len(picked) < min(limit, len(rows)):
        row = random.choice(pool)
        if row["id"] not in used:
            used.add(row["id"])
            picked.append(row)
        pool = [r for r in pool if r["id"] not in used]

    questions = []
    for row in picked:
        accepted = json_loads(row["accepted_json"], [row["expression"]])
        examples = json_loads(row["examples_json"], [])
        questions.append(
            {
                "note_id": row["id"],
                "expression": row["expression"],
                "prompt": f"다음 한국어 뜻에 맞는 영어 표현을 쓰세요: {row['meaning_kr']}",
                "accepted": accepted,
                "examples": examples,
                "meaning": row["meaning_kr"],
                "nuance": row["nuance_kr"],
            }
        )
        if examples:
            questions.append(
                {
                    "note_id": row["id"],
                    "expression": row["expression"],
                    "prompt": f"다음 예문의 빈칸에 들어갈 표현을 쓰세요: {examples[0].replace(row['expression'], '_____', 1)}",
                    "accepted": accepted,
                    "examples": examples,
                    "meaning": row["meaning_kr"],
                    "nuance": row["nuance_kr"],
                }
            )

    random.shuffle(questions)
    return questions[:limit]


# -----------------------------
# UI state
# -----------------------------
def init_state():
    defaults = {
        "analysis_result": None,
        "analysis_status": None,
        "coach_answer": None,
        "coach_status": None,
        "quiz_items": [],
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_feedback": None,
        "quiz_done": False,
        "model_name": DEFAULT_MODEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def reset_quiz():
    st.session_state.quiz_items = build_questions(limit=10)
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = None
    st.session_state.quiz_done = False


# -----------------------------
# Pages
# -----------------------------
def page_input_analysis():
    st.header("오늘의 학습 입력")
    text = st.text_area(
        "모르는 표현, 문장, 질문을 붙여넣으세요",
        placeholder="예: hit the spot / walk on eggshells / Why do people say address the problem?",
        height=140,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("AI 자동 분석", use_container_width=True):
            if text.strip():
                result, status = analyze_text(text)
                st.session_state.analysis_result = result
                st.session_state.analysis_status = status
            else:
                st.warning("먼저 입력해 주세요.")
    with c2:
        if st.button("오답노트 저장", use_container_width=True):
            result = st.session_state.analysis_result
            if result:
                save_note(result)
                st.success("오답노트에 저장했습니다.")
            else:
                st.warning("먼저 분석을 실행해 주세요.")
    with c3:
        if st.button("초기화", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.analysis_status = None
            st.rerun()

    if st.session_state.analysis_status:
        if "실패" in st.session_state.analysis_status or "기본" in st.session_state.analysis_status:
            st.warning(st.session_state.analysis_status)
        else:
            st.success(st.session_state.analysis_status)

    result = st.session_state.analysis_result
    if result:
        st.subheader("분석 결과")
        st.write(f"**핵심 표현**: {result['expression']}")
        st.write(f"**뜻**: {result['meaning_kr']}")
        st.write(f"**뉘앙스**: {result['nuance_kr']}")
        st.write(f"**유래/기억법**: {result['origin_kr']}")
        st.write("**비슷한 표현**")
        for x in result["similar_expressions"]:
            st.write(f"- {x}")
        st.write("**예문**")
        for x in result["example_sentences"]:
            st.write(f"- {x}")



def page_wrong_note():
    st.header("오답노트")
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("검색", placeholder="표현, 뜻, 원문")
    with c2:
        sort_by = st.selectbox("정렬", ["weight", "wrong", "latest", "name"], format_func=lambda x: {
            "weight": "복습 우선순위",
            "wrong": "자주 틀린 순",
            "latest": "최신 순",
            "name": "알파벳 순",
        }[x])

    rows = list_notes(q, sort_by)
    st.caption(f"총 {len(rows)}개")
    if not rows:
        st.info("저장된 노트가 없습니다.")
        return

    for row in rows:
        with st.expander(f"{row['expression']} | 가중치 {row['weight']} | 오답 {row['wrong_count']}회 | 연속정답 {row['streak']}"):
            st.write(f"**원문**: {row['source_text']}")
            st.write(f"**뜻**: {row['meaning_kr']}")
            st.write(f"**뉘앙스**: {row['nuance_kr']}")
            st.write(f"**유래/기억법**: {row['origin_kr']}")
            st.write("**비슷한 표현**")
            for x in json_loads(row["similar_json"], []):
                st.write(f"- {x}")
            st.write("**예문**")
            for x in json_loads(row["examples_json"], []):
                st.write(f"- {x}")



def page_test():
    st.header("테스트")
    if get_note_count() == 0:
        st.info("먼저 오답노트에 표현을 저장하세요.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if st.button("10문제 시작", use_container_width=True):
            reset_quiz()
    with c2:
        if st.button("다시 출제", use_container_width=True):
            reset_quiz()

    items = st.session_state.quiz_items
    if not items:
        st.write("아직 테스트가 시작되지 않았습니다.")
        return

    idx = st.session_state.quiz_index
    if idx >= len(items):
        st.session_state.quiz_done = True

    if st.session_state.quiz_done:
        st.success(f"테스트 종료. 점수: {st.session_state.quiz_score} / {len(items)}")
        return

    item = items[idx]
    st.progress(idx / len(items), text=f"진행 {idx}/{len(items)}")
    st.subheader(f"문제 {idx + 1}")
    st.write(item["prompt"])
    answer = st.text_input("영어로 답하세요", key=f"ans_{idx}")

    if st.button("채점", type="primary", use_container_width=True):
        if answer.strip():
            correct, label, best_answer, feedback = grade_answer(item["prompt"], item["accepted"], answer, item["expression"])
            update_stat(item["note_id"], correct)
            if correct:
                st.session_state.quiz_score += 1
                st.session_state.quiz_feedback = (True, f"{label}: {feedback or '좋습니다.'}", best_answer)
            else:
                st.session_state.quiz_feedback = (False, f"{label}: {feedback}", best_answer)
        else:
            st.warning("답을 입력해 주세요.")

    if st.session_state.quiz_feedback:
        ok, msg, best = st.session_state.quiz_feedback
        if ok:
            st.success(msg)
        else:
            st.error(f"{msg} / 대표 정답: {best}")
        with st.expander("해설"):
            st.write(f"**핵심 표현**: {item['expression']}")
            st.write(f"**뜻**: {item['meaning']}")
            st.write(f"**뉘앙스**: {item['nuance']}")
            for ex in item["examples"]:
                st.write(f"- {ex}")
        if st.button("다음 문제", use_container_width=True):
            st.session_state.quiz_index += 1
            st.session_state.quiz_feedback = None
            st.rerun()



def page_coach():
    st.header("코치 모드")
    st.write("질문만 설명하지 않고, 훈련까지 시키는 개인 영어 코치입니다.")
    text = st.text_area(
        "예: 이 문장 자연스러워? / 이 표현으로 대화 훈련 시켜줘 / 내 답변 고쳐줘",
        height=120,
    )
    if st.button("코치에게 요청", use_container_width=True):
        if text.strip():
            answer, status = coach_reply(text)
            st.session_state.coach_answer = answer
            st.session_state.coach_status = status
            if answer:
                save_coach_history(text, answer)
        else:
            st.warning("먼저 입력해 주세요.")

    if st.session_state.coach_status:
        if "실패" in st.session_state.coach_status or "없어" in st.session_state.coach_status:
            st.warning(st.session_state.coach_status)
        else:
            st.success(st.session_state.coach_status)

    if st.session_state.coach_answer:
        st.subheader("코치 답변")
        st.write(st.session_state.coach_answer)

    st.subheader("최근 코칭 기록")
    rows = get_recent_coach_history(limit=6)
    if not rows:
        st.info("아직 코칭 기록이 없습니다.")
    else:
        for row in rows:
            with st.expander(row["user_message"]):
                st.write(row["ai_message"])



def page_stats():
    st.header("학습 통계")
    summary, weak = get_stats_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("저장된 표현", summary["total_notes"])
    c2.metric("총 출제 수", summary["total_seen"])
    c3.metric("정답 수", summary["total_correct"])
    c4.metric("오답 수", summary["total_wrong"])

    st.subheader("약한 표현 TOP 10")
    if not weak:
        st.info("아직 데이터가 없습니다.")
        return
    for i, row in enumerate(weak, start=1):
        st.write(f"{i}. {row['expression']} | 가중치 {row['weight']} | 오답 {row['wrong_count']}회 | 정답 {row['correct_count']}회")


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")
    init_db()
    init_state()

    st.title(APP_TITLE)
    st.caption("오답노트 · 분석 · 테스트 · 코칭 중심의 개인 영어 훈련 앱")

    with st.sidebar:
        st.header("AI 설정")
        st.text_input("OpenAI API Key", type="password", key="runtime_api_key")
        st.selectbox("모델", ["gpt-4o-mini", "gpt-4.1-mini"], key="model_name", index=0)
        st.markdown("---")
        st.write("- 주 기능: 오답노트 / 분석 / 테스트")
        st.write("- 부 기능: 자유 코칭")
        st.write("- API 키가 없으면 분석은 기본 모드로 동작")

    menu = st.sidebar.radio("메뉴", ["오늘의 학습 입력", "오답노트", "테스트", "코치 모드", "학습 통계"])

    if menu == "오늘의 학습 입력":
        page_input_analysis()
    elif menu == "오답노트":
        page_wrong_note()
    elif menu == "테스트":
        page_test()
    elif menu == "코치 모드":
        page_coach()
    else:
        page_stats()


if __name__ == "__main__":
    main()
