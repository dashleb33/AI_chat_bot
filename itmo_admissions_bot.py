#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITMO MSc Admissions Helper Bot

Функционал:
- Парсит страницы двух магистратур ITMO (ИИ и AI Product)
- Извлекает учебные планы/дисциплины, формирует корпус знаний
- Диалоговая система (Telegram) отвечает на вопросы по содержимому
- Рекомендации по выборным дисциплинам на основе бэкграунда
- Жёсткая фильтрация по релевантности: бот отвечает только по двум программам

Зависимости (установите перед запуском):
    pip install python-telegram-bot==21.4 requests beautifulsoup4 lxml scikit-learn rapidfuzz python-dotenv

Переменные окружения:
    BOT_TOKEN — токен Telegram-бота (обязателен)

Запуск:
    python itmo_admissions_bot.py

Примечание:
- Скрипт сам скачивает страницы при первом запуске и кэширует их в data/*.html
- Если структура страниц поменяется, парсер попытается извлечь максимум возможного (best-effort)
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Telegram ---
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler,
    ContextTypes, filters, CallbackQueryHandler
)

# -----------------------------
# Конфигурация и константы
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_TTL = 60 * 60 * 24  # 24 часа

PROGRAMS = {
    "ai": {
        "name": "Искусственный интеллект",
        "url": "https://abit.itmo.ru/program/master/ai",
    },
    "ai_product": {
        "name": "AI Product Management",
        "url": "https://abit.itmo.ru/program/master/ai_product",
    },
}

ALLOWED_KEYWORDS = [
    "итмо", "магистратура", "магистерская программа", "ai", "искусственный интеллект",
    "ai product", "product management", "учебный план", "дисциплины", "курсы", "семестр",
    "ECTS", "зачётные единицы", "профиль", "поступление", "абитуриент", "выборные дисциплины",
]

# Разрешённая тематика — только про две программы
SCOPE_HINT = (
    "Я отвечаю только на вопросы по двум программам магистратуры ИТМО: \n"
    "— Искусственный интеллект (https://abit.itmo.ru/program/master/ai)\n"
    "— AI Product Management (https://abit.itmo.ru/program/master/ai_product).\n"
    "Сформулируйте вопрос о дисциплинах, учебных планах, сроках, форматах обучения, нагрузке, выборных курсах и т.п."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Вспомогательные структуры
# -----------------------------
@dataclass
class Course:
    title: str
    semester: Optional[str] = None
    ects: Optional[str] = None
    kind: Optional[str] = None  # 'core' | 'elective' | None
    description: Optional[str] = None

@dataclass
class ProgramPlan:
    key: str
    name: str
    url: str
    courses: List[Course] = field(default_factory=list)
    raw_text_blocks: List[str] = field(default_factory=list)  # дополнительные материалы

# -----------------------------
# Загрузка и парсинг
# -----------------------------

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def fetch_url(url: str, cache_path: str, ttl: int = CACHE_TTL) -> str:
    """Скачивает страницу с кэшем на диск."""
    ensure_dir(DATA_DIR)
    if os.path.isfile(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < ttl:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    html = resp.text
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or " ").strip()
    return s


def parse_program_page(key: str, name: str, url: str) -> ProgramPlan:
    """Best-effort парсер страницы программы.
    Пытается извлечь таблицы учебного плана и сопутствующий текст.
    """
    cache_path = os.path.join(DATA_DIR, f"{key}.html")
    html = fetch_url(url, cache_path)
    soup = BeautifulSoup(html, "lxml")

    plan = ProgramPlan(key=key, name=name, url=url)

    # 1) Собираем все тексты в осмысленные блоки для QA
    texts = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "div"]):
        txt = clean_text(tag.get_text(" "))
        if len(txt) > 40 and any(k.lower() in txt.lower() for k in ["дисциплин", "учебн", "курс", "семестр", "ECTS", "зачёт"]):
            texts.append(txt)
    plan.raw_text_blocks = list(dict.fromkeys(texts))  # убираем дубликаты

    # 2) Пытаемся распарсить таблицы с дисциплинами (варианты разметки)
    tables = soup.find_all("table")
    for tbl in tables:
        headers = [clean_text(th.get_text(" ")).lower() for th in tbl.find_all("th")]
        # Частые варианты заголовков
        has_title = any(h for h in headers if "дисципл" in h or "курс" in h or "наименование" in h)
        if not headers:
            # Иногда заголовков нет — проверяем первую строку
            first_row = tbl.find("tr")
            if first_row:
                headers = [clean_text(td.get_text(" ")).lower() for td in first_row.find_all(["td", "th"]) ]
                # если это реально заголовки, пропустим первую строку при парсинге
        if not (has_title or headers):
            continue

        # Считаем, что первая строка — заголовки, если она содержит текстовые метки
        rows = tbl.find_all("tr")
        start_idx = 1 if rows and rows[0].find_all("th") else 0
        for tr in rows[start_idx:]:
            cols = [clean_text(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            if not cols or all(len(c) == 0 for c in cols):
                continue
            title = None
            semester = None
            ects = None
            kind = None

            # эвристики по колонкам
            for i, c in enumerate(cols):
                lc = c.lower()
                if title is None and ("дисцип" in lc or "курс" in lc or len(c) > 5):
                    title = c
                if semester is None and ("семест" in lc or re.fullmatch(r"\d+", lc)):
                    semester = c
                if ects is None and ("ects" in lc or "зач" in lc or re.fullmatch(r"\d+(?:[,.]\d+)?", lc)):
                    ects = c
                if kind is None and any(k in lc for k in ["выбор", "электив", "обяз", "блок"]):
                    kind = c
            if title and len(title) >= 2:
                plan.courses.append(Course(title=title, semester=semester, ects=ects, kind=kind))

    # 3) Fallback: ищем списки дисциплин
    if not plan.courses:
        for li in soup.find_all("li"):
            txt = clean_text(li.get_text(" "))
            if 5 < len(txt) < 200 and any(w in txt.lower() for w in ["курс", "дисципл", "интеллект", "data", "ml", "product"]):
                plan.courses.append(Course(title=txt))

    # Нормализуем семестры и типы
    for c in plan.courses:
        if c.semester:
            m = re.search(r"(\d+)", c.semester)
            if m:
                c.semester = m.group(1)
        if c.kind:
            lk = c.kind.lower()
            if "выбор" in lk or "электив" in lk:
                c.kind = "elective"
            elif "обяз" in lk or "core" in lk or "баз" in lk:
                c.kind = "core"
            else:
                c.kind = None

    return plan


# -----------------------------
# Корпус знаний и QA
# -----------------------------
class Corpus:
    def __init__(self, plans: List[ProgramPlan]):
        self.plans = plans
        self.docs: List[str] = []
        self.meta: List[Tuple[str, str]] = []  # (program_key, origin)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._build()

    def _build(self):
        docs = []
        meta = []
        for p in self.plans:
            # Документы: описания курсов и текстовые блоки
            for c in p.courses:
                parts = [c.title]
                if c.description: parts.append(c.description)
                if c.semester: parts.append(f"Семестр: {c.semester}")
                if c.ects: parts.append(f"ECTS: {c.ects}")
                if c.kind: parts.append(f"Тип: {c.kind}")
                docs.append(" | ".join([clean_text(x) for x in parts if x]))
                meta.append((p.key, f"course:{c.title}"))
            for block in p.raw_text_blocks:
                docs.append(block)
                meta.append((p.key, "page:text"))
        # если пусто — хотя бы что-то
        if not docs:
            for p in self.plans:
                docs.append(f"Страница программы {p.name}: {p.url}")
                meta.append((p.key, "page:fallback"))
        self.docs = docs
        self.meta = meta
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def query(self, question: str, top_k: int = 5):
        if not self.docs:
            return []
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({
                "text": self.docs[i],
                "program_key": self.meta[i][0],
                "origin": self.meta[i][1],
                "score": float(sims[i])
            })
        return results


# -----------------------------
# Рекомендации
# -----------------------------
class Recommender:
    def __init__(self, plans: Dict[str, ProgramPlan]):
        self.plans = plans

    @staticmethod
    def _match_course(course: Course, keywords: List[str]) -> int:
        text = " ".join(filter(None, [course.title or "", course.description or ""]))
        text = text.lower()
        score = 0
        for kw in keywords:
            score = max(score, fuzz.partial_ratio(kw.lower(), text))
        return score

    def recommend(self, program_key: str, background: Dict[str, Any], top_n: int = 6) -> List[Tuple[Course, int]]:
        plan = self.plans.get(program_key)
        if not plan:
            return []
        # Из интересов абитуриента строим ключевые слова
        interests = (background.get("interests") or "") + " " + (background.get("goals") or "")
        stack = (background.get("stack") or "")
        seniority = (background.get("seniority") or "")
        text = " ".join([interests, stack, seniority]).lower()

        # Базовые наборы ключей по направлениям
        buckets = {
            "ml": ["machine learning", "ml", "обучение с учителем", "обучение без учителя", "регрессия", "классификация"],
            "dl": ["deep learning", "нейронн", "dl", "pytorch", "tensorflow"],
            "cv": ["computer vision", "компьютерное зрение", "segmentation", "detection"],
            "nlp": ["nlp", "natural language", "обработка естественного языка", "текст", "LLM"],
            "data": ["data", "аналит", "sql", "etl", "data engineering"],
            "mops": ["mlops", "mLOps", "prod", "deploy", "оркестрация"],
            "prod": ["product", "продукт", "маркет", "управлен", "roadmap", "ux", "user research"],
            "math": ["матан", "алгебра", "вероят", "статист"],
        }
        keys = []
        for k, kws in buckets.items():
            if any(w in text for w in kws):
                keys.extend(kws)
        if not keys:
            # универсальные ключи
            keys = ["advanced", "практика", "проект", "research", "seminar", "введение"]

        elect = [c for c in plan.courses if (c.kind == "elective") or (c.kind is None)]  # часто тип не размечен
        # Если помеченных нет — берём все
        if not elect:
            elect = list(plan.courses)
        scored = [(c, self._match_course(c, keys)) for c in elect]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x for x in scored[:top_n] if x[1] > 40] or scored[:top_n]


# -----------------------------
# Хранение профилей пользователей
# -----------------------------
PROFILE_PATH = os.path.join(DATA_DIR, "profiles.json")

def load_profiles() -> Dict[str, Any]:
    if os.path.isfile(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_profiles(data: Dict[str, Any]):
    ensure_dir(DATA_DIR)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------
# Инициализация данных
# -----------------------------

def build_plans() -> Dict[str, ProgramPlan]:
    plans = {}
    for key, cfg in PROGRAMS.items():
        try:
            plan = parse_program_page(key, cfg["name"], cfg["url"])
            plans[key] = plan
            logger.info("Parsed %s: %d courses, %d text blocks", cfg["name"], len(plan.courses), len(plan.raw_text_blocks))
        except Exception as e:
            logger.exception("Failed to parse %s: %s", cfg["url"], e)
            plans[key] = ProgramPlan(key=key, name=cfg["name"], url=cfg["url"], courses=[], raw_text_blocks=[])
    return plans


# -----------------------------
# Релевантность вопросов
# -----------------------------

def is_in_scope(text: str) -> bool:
    tl = text.lower()
    if any(k in tl for k in ["https://abit.itmo.ru/program/master/ai", "https://abit.itmo.ru/program/master/ai_product"]):
        return True
    # Быстрая евристика по ключевым словам
    score = 0
    for kw in ALLOWED_KEYWORDS:
        score = max(score, fuzz.partial_ratio(kw.lower(), tl))
    return score >= 45


# -----------------------------
# Telegram: состояния для опроса
# -----------------------------
(ASK_PROGRAM, ASK_SENIORITY, ASK_STACK, ASK_INTERESTS, ASK_GOALS) = range(5)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton(PROGRAMS['ai']['name'], callback_data='ai')],
        [InlineKeyboardButton(PROGRAMS['ai_product']['name'], callback_data='ai_product')],
    ]
    await update.message.reply_text(
        "Привет! Я помогу выбрать между двумя магистерскими программами ИТМО и спланировать обучение.\n"
        "Для начала — какая программа вам ближе?",
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return ASK_PROGRAM


async def pick_program(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    program_key = query.data
    context.user_data["program_key"] = program_key
    await query.edit_message_text(
        f"Отлично! Вы выбрали: {PROGRAMS[program_key]['name']}.\n"
        "Какой у вас уровень опыта? (например: студент, джун, мидл, сеньор)"
    )
    return ASK_SENIORITY


async def ask_seniority(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["seniority"] = clean_text(update.message.text)
    await update.message.reply_text("Какими технологиями и языками владеете? (например: Python, SQL, PyTorch, аналитика)" )
    return ASK_STACK


async def ask_stack(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["stack"] = clean_text(update.message.text)
    await update.message.reply_text("Какие темы особенно интересуют? (ML, DL, CV, NLP, продакт-менеджмент, аналитика и т.п.)")
    return ASK_INTERESTS


async def ask_interests(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["interests"] = clean_text(update.message.text)
    await update.message.reply_text("Какова цель обучения? (академическая карьера, R&D, продакт/менеджмент, переход в Data/ML и т.п.)")
    return ASK_GOALS


async def ask_goals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["goals"] = clean_text(update.message.text)

    # Сохраняем профиль
    profiles = load_profiles()
    uid = str(update.message.from_user.id)
    profiles[uid] = {
        "program_key": context.user_data.get("program_key"),
        "seniority": context.user_data.get("seniority"),
        "stack": context.user_data.get("stack"),
        "interests": context.user_data.get("interests"),
        "goals": context.user_data.get("goals"),
    }
    save_profiles(profiles)

    await update.message.reply_text(
        "Спасибо! Профиль сохранён. Можете задать вопрос по программе, команду /plan для учебного плана или /recommend для рекомендаций по выборным курсам."
    )
    return ConversationHandler.END


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Загружаем планы из контекста приложения
    plans: Dict[str, ProgramPlan] = context.application.bot_data.get("plans", {})
    # Определяем программу пользователя
    profiles = load_profiles()
    uid = str(update.message.from_user.id)
    program_key = profiles.get(uid, {}).get("program_key") or context.user_data.get("program_key")
    if not program_key:
        await update.message.reply_text("Сначала выберите программу командой /start.")
        return
    plan = plans.get(program_key)
    if not plan or not plan.courses:
        await update.message.reply_text(
            "Не удалось извлечь учебный план автоматически. Проверьте страницы программ: "
            f"{PROGRAMS[program_key]['url']}"
        )
        return

    # Формируем краткий план
    lines = [f"Учебный план — {plan.name}"]
    grouped: Dict[str, List[Course]] = {}
    for c in plan.courses:
        sem = c.semester or "?"
        grouped.setdefault(sem, []).append(c)
    for sem in sorted(grouped.keys(), key=lambda x: (x=='?', int(x) if x.isdigit() else 99)):
        lines.append(f"\nСеместр {sem}:")
        for c in grouped[sem]:
            tag = "(электив)" if c.kind == "elective" else ""
            ects = f" — {c.ects} ECTS" if c.ects and re.match(r"^\d", c.ects) else (f" — {c.ects}" if c.ects else "")
            lines.append(f" • {c.title}{ects} {tag}")

    text = "\n".join(lines)
    # Телеграм ограничивает 4096 символов — режем на блоки
    for chunk in split_long(text, limit=3800):
        await update.message.reply_text(chunk)


async def cmd_recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    plans: Dict[str, ProgramPlan] = context.application.bot_data.get("plans", {})
    profiles = load_profiles()
    uid = str(update.message.from_user.id)
    profile = profiles.get(uid)
    if not profile:
        await update.message.reply_text("Сначала заполните профиль: /start")
        return
    program_key = profile.get("program_key")
    plan = plans.get(program_key)
    if not plan:
        await update.message.reply_text("План не найден. Попробуйте /plan или /start")
        return

    rec = Recommender(plans).recommend(program_key, profile, top_n=8)
    if not rec:
        await update.message.reply_text("Не удалось подобрать рекомендации — возможно, парсер не нашёл выборные дисциплины. Попробуйте задать интересы подробнее командой /start.")
        return

    lines = [f"Персональные рекомендации по элективам — {PROGRAMS[program_key]['name']}"]
    for course, score in rec:
        sem = f" (семестр {course.semester})" if course.semester else ""
        lines.append(f" • {course.title}{sem} — релевантность {score}%")
    for chunk in split_long("\n".join(lines), limit=3500):
        await update.message.reply_text(chunk)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = clean_text(update.message.text)

    if not is_in_scope(question):
        await update.message.reply_text(SCOPE_HINT)
        return

    # Корпус
    corpus: Corpus = context.application.bot_data.get("corpus")
    plans: Dict[str, ProgramPlan] = context.application.bot_data.get("plans", {})

    # Постараемся определить, о какой программе вопрос
    program_key = detect_program_key(question)

    results = corpus.query(question, top_k=5)
    # Фильтрация по релевантности (TF-IDF + эвристика)
    results = [r for r in results if r["score"] >= 0.15]

    if program_key:
        results = [r for r in results if r["program_key"] == program_key] or results

    if not results:
        await update.message.reply_text("Не нашёл информации по вопросу в учебных планах. Попробуйте переформулировать или спросить про дисциплины/семестры/форматы обучения.")
        return

    # Синтез ответа из 1–3 лучших фрагментов
    answer_parts = []
    for r in results[:3]:
        answer_parts.append(f"• {r['text']}")
    answer = "\n".join(answer_parts)
    # Добавим ссылку на программу-источник
    src_keys = {r['program_key'] for r in results[:3]}
    src_lines = [f"Источник: {PROGRAMS[k]['name']} — {PROGRAMS[k]['url']}" for k in src_keys]
    answer += "\n\n" + "\n".join(src_lines)

    for chunk in split_long(answer, limit=3800):
        await update.message.reply_text(chunk)


def detect_program_key(text: str) -> Optional[str]:
    lt = text.lower()
    if "product" in lt or "продакт" in lt or "управлен" in lt:
        return "ai_product"
    if "искусственный интеллект" in lt or "нейронн" in lt or "ml" in lt or "машинн" in lt:
        return "ai"
    return None


def split_long(text: str, limit: int = 3800) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    current = []
    size = 0
    for line in text.split("\n"):
        if size + len(line) + 1 > limit:
            parts.append("\n".join(current))
            current = [line]
            size = len(line) + 1
        else:
            current.append(line)
            size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


# -----------------------------
# Точка входа (Telegram)
# -----------------------------
async def on_startup(app):
    # Строим планы и корпус при старте бота
    plans = build_plans()
    app.bot_data["plans"] = plans
    corpus = Corpus(list(plans.values()))
    app.bot_data["corpus"] = corpus
    logger.info("Bot data ready: %d documents", len(corpus.docs))


def main():
    from dotenv import load_dotenv
    load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN (переменная окружения)")

    application = ApplicationBuilder().token(token).build()

    # Диалог старта/профиля
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_PROGRAM: [CallbackQueryHandler(pick_program)],
            ASK_SENIORITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_seniority)],
            ASK_STACK: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_stack)],
            ASK_INTERESTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_interests)],
            ASK_GOALS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_goals)],
        },
        fallbacks=[],
        allow_reentry=True,
    )

    application.add_handler(conv)
    application.add_handler(CommandHandler("plan", cmd_plan))
    application.add_handler(CommandHandler("recommend", cmd_recommend))

    # Свободные вопросы
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    application.post_init = on_startup

    logger.info("Starting bot...")
    application.run_polling()


if __name__ == "__main__":
    main()
