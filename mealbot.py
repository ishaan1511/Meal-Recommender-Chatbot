#!/usr/bin/env python
# coding: utf-8

# # Load dataset

# In[1]:


import pandas as pd
import numpy as np

structured_food_df = pd.read_csv("structured_food_dataset.csv", low_memory=False)


# In[2]:


structured_food_df = structured_food_df.rename(columns={
    "fdc_id": "food_id",
    "data_type": "source_type",
    "description": "food_name",
    "Energy": "Energy",
    "Protein": "protein_g",
    "Total lipid (fat)": "fat_g",
    "Carbohydrate, by difference": "carbs_g",
    "Fiber, total dietary": "fiber_g",
    "Sugars, Total": "sugar_g",
    "Cholesterol": "cholesterol_mg",
    "Sodium, Na": "sodium_mg",
    "Potassium, K": "potassium_mg",
    "Calcium, Ca": "calcium_mg",
    "Iron, Fe": "iron_mg",
    "Vitamin B-12": "vitamin_b12_mcg",
    "Vitamin C, total ascorbic acid": "vitamin_c_mg",
    "Vitamin D (D2 + D3)": "vitamin_d_mcg",
    "Vitamin E (alpha-tocopherol)": "vitamin_e_mg",
    "portion_description": "portion_desc",
    "gram_weight": "portion_grams"})
structured_food_df.head()


# In[3]:


get_ipython().system('pip -q install transformers accelerate sentencepiece')


# In[4]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).eval()

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=96,      # ↓ fewer tokens
    temperature=0.0,        # ↓ disables sampling overhead
    do_sample=False,        # greedy decoding = faster
    batch_size=1)


# In[11]:


import json
import re
import torch

# ----- Nutrients you want supported (must match your dataframe column names) -----
NUTRIENT_KEYS = [
    "Energy",
    "protein_g",
    "fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "cholesterol_mg",
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "iron_mg",
    "vitamin_b12_mcg",
    "vitamin_c_mg",
    "vitamin_d_mcg",
    "vitamin_e_mg",
]

# ----- Default request object -----
DEFAULT_REQ = {
    "calorie_band": None,   # "small"|"medium"|"large"|None
    "avoid": set(),         # set[str]
    "nutrients": {k: {"min": None, "max": None, "target": None, "priority": 0} for k in NUTRIENT_KEYS},
}

# Precompile once (faster)
JSON_RE = re.compile(r"\{[\s\S]*\}")

def _coerce_number(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def _coerce_priority(x):
    try:
        x = int(x)
        return x if x in (0, 1, 2, 3) else 0
    except (TypeError, ValueError):
        return 0

def llm_parse_request(user_text: str):
    # Keep prompt compact (faster) and explicit (more reliable)
    schema_str = (
        '{'
        '"calorie_band": null|"small"|"medium"|"large",'
        '"avoid": [],'
        '"nutrients": {'
        + ",".join([f'"{k}":{{"min":null,"max":null,"target":null,"priority":0}}' for k in NUTRIENT_KEYS])
        + "}"
        + "}"
    )

    prompt = (
        "Return ONLY valid JSON. No extra text.\n"
        "Rules:\n"
        "- If user asks for 'high'/'more' of a nutrient: set priority=2 or 3 and set min/target if stated.\n"
        "- If user asks for 'low'/'less' of a nutrient: set priority=2 or 3 and set max/target if stated.\n"
        "- If user gives a number, fill min/max/target accordingly.\n"
        "- 'avoid' is a list of foods/keywords to exclude.\n"
        f"Schema: {schema_str}\n"
        f"User: {user_text}\n"
        "JSON:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=120,      # slightly higher because schema is larger
                do_sample=False,         # greedy = faster + deterministic
                temperature=0.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "JSON:" in text:
            text = text.split("JSON:", 1)[-1].strip()

    except Exception:
        return DEFAULT_REQ.copy()

    m = JSON_RE.search(text)
    if not m:
        return DEFAULT_REQ.copy()

    raw = m.group(0).replace("\n", " ").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_REQ.copy()

    # ----- Build a sanitized output that ALWAYS matches your expected structure -----
    out = DEFAULT_REQ.copy()
    out["avoid"] = set(data.get("avoid", []) or [])
    out["calorie_band"] = data.get("calorie_band")

    nutrients_in = data.get("nutrients", {}) or {}
    nutrients_out = {k: {"min": None, "max": None, "target": None, "priority": 0} for k in NUTRIENT_KEYS}

    for k in NUTRIENT_KEYS:
        v = nutrients_in.get(k, {}) or {}
        nutrients_out[k] = {
            "min": _coerce_number(v.get("min")),
            "max": _coerce_number(v.get("max")),
            "target": _coerce_number(v.get("target")),
            "priority": _coerce_priority(v.get("priority")),
        }

    out["nutrients"] = nutrients_out
    return out



# In[6]:


def llm_format_reply(user_text, constraints, recs_df):
    recs = []
    for _, r in recs_df.iterrows():
        recs.append({
            "name": r["product_name"],
            "kcal": float(r["Energy (kcal)"]),
            "protein_g": float(r.get("Protein (g)", 0) or 0),
            "sodium_mg": float(r.get("Sodium, Na (mg)", 0) or 0),
            "fiber_g": float(r.get("Fiber, total dietary (g)", 0) or 0),
            "added_sugar_g": float(r.get("Sugars, added (g)", 0) or 0),
        })

    prompt = f"""
You are a helpful meal recommendation chatbot.
You MUST only talk about the recommended items provided.
Explain briefly why each fits, and ask for feedback (e.g., "too salty", "too sweet", "more protein", "avoid bars").
User said: {user_text}

Extracted constraints: {json.dumps(constraints)}

Recommendations (from dataset): {json.dumps(recs)}

Reply:
""".strip()

    return llm(prompt)[0]["generated_text"].split("Reply:", 1)[-1].strip()


# In[7]:


def init_memory():
    return {
        "avoid_keywords": set(),
        "w_protein": 1.0,
        "w_fiber": 0.6,
        "w_sodium_penalty": 0.002,
        "w_sugar_penalty": 0.6
    }

memory = init_memory()

def apply_feedback(memory, feedback_text):
    t = feedback_text.lower()

    if "too salty" in t:
        memory["w_sodium_penalty"] *= 1.3
    if "too sweet" in t:
        memory["w_sugar_penalty"] *= 1.3
    if "more protein" in t:
        memory["w_protein"] *= 1.2
    if "more fiber" in t:
        memory["w_fiber"] *= 1.2

    m = re.findall(r"(?:avoid|no|hate|don't like)\s+([a-zA-Z]+)", t)
    for w in m:
        memory["avoid_keywords"].add(w.lower())


# In[13]:


# Map common user phrases -> your dataframe nutrient keys
NUTRIENT_SYNONYMS = {
    "Energy": ["calorie", "calories", "kcal", "energy"],
    "protein_g": ["protein", "proteins"],
    "fat_g": ["fat", "fats", "lipid", "lipids"],
    "carbs_g": ["carb", "carbs", "carbohydrate", "carbohydrates"],
    "fiber_g": ["fiber", "fibre"],
    "sugar_g": ["sugar", "sugars"],
    "cholesterol_mg": ["cholesterol"],
    "sodium_mg": ["sodium", "salt", "salty"],
    "potassium_mg": ["potassium"],
    "calcium_mg": ["calcium"],
    "iron_mg": ["iron"],
    "vitamin_b12_mcg": ["b12", "vit b12", "vitamin b12", "cobalamin"],
    "vitamin_c_mg": ["vit c", "vitamin c", "ascorbic", "ascorbic acid"],
    "vitamin_d_mcg": ["vit d", "vitamin d", "cholecalciferol", "ergocalciferol"],
    "vitamin_e_mg": ["vit e", "vitamin e", "tocopherol", "tocopherols"],
}

# Default minimums to avoid “high X” returning foods with 0/NA.
# Tune these to your dataset’s basis (often per 100g).
DEFAULT_MIN_IF_HIGH = {
    "protein_g": 10,
    "fiber_g": 5,
    "vitamin_c_mg": 10,
    "iron_mg": 2,
    "calcium_mg": 100,
    "potassium_mg": 200,
    "vitamin_b12_mcg": 0.5,
    "vitamin_d_mcg": 2,
    "vitamin_e_mg": 1,
    "sodium_mg": None,        # handled via "low" (max)
    "sugar_g": None,          # handled via "low" (max)
    "cholesterol_mg": None,   # handled via "low" (max)
    "fat_g": None,            # depends; leave None
    "carbs_g": None,          # depends; leave None
    "Energy": None,           # handled via calorie_band etc
}

# Regex helpers
WORD = r"(?:\b{}\b)"
def _has_any(t: str, phrases) -> bool:
    return any(re.search(WORD.format(re.escape(p)), t) for p in phrases)

def apply_nutrient_shortcuts(req, user_text: str):
    """
    General shortcut rules:
    - "high/more/rich/good source of X" => priority=3 and set min if not set
    - "low/less/reduce/avoid X" => priority=3 and set max if not set (when sensible)
    - Also supports nutrient name alone: "iron foods", "high protein"
    """
    t = user_text.lower()

    # intent words (expand as you like)
    HIGH_CUES = ["high", "more", "rich", "good source", "increase", "boost", "lots of"]
    LOW_CUES  = ["low", "less", "reduce", "lower", "avoid", "cut down", "not too much", "no"]

    # Ensure structure exists
    if "nutrients" not in req or not isinstance(req["nutrients"], dict):
        return req

    for nutrient_key, synonyms in NUTRIENT_SYNONYMS.items():
        if nutrient_key not in req["nutrients"]:
            continue

        mentioned = _has_any(t, synonyms)
        if not mentioned:
            continue

        # Determine direction (high vs low). If ambiguous, default to "high" for vitamins/minerals/protein/fiber.
        high_intent = any(cue in t for cue in HIGH_CUES)
        low_intent  = any(cue in t for cue in LOW_CUES)

        # Special handling for salt/sodium: "salty" usually means "low sodium"
        if nutrient_key == "sodium_mg" and ("salty" in t or "too salty" in t):
            low_intent = True
            high_intent = False

        spec = req["nutrients"][nutrient_key]

        # If both cues appear, keep priority but don't set min/max automatically
        spec["priority"] = max(int(spec.get("priority", 0) or 0), 3)

        if high_intent and not low_intent:
            # set a sensible min if none exists (to avoid 0/NA results)
            if spec.get("min") is None and DEFAULT_MIN_IF_HIGH.get(nutrient_key) is not None:
                spec["min"] = DEFAULT_MIN_IF_HIGH[nutrient_key]

        if low_intent and not high_intent:
            # set sensible max defaults for “low” nutrients when not provided
            if spec.get("max") is None:
                if nutrient_key == "sodium_mg":
                    spec["max"] = 200   # mg (tune to your basis)
                elif nutrient_key == "sugar_g":
                    spec["max"] = 5     # g
                elif nutrient_key == "cholesterol_mg":
                    spec["max"] = 50    # mg
                elif nutrient_key == "fat_g":
                    spec["max"] = 10    # g (optional)
                # otherwise leave max None

        req["nutrients"][nutrient_key] = spec

    return req


# In[14]:


def calorie_range(band):
    if band == "small":
        return (200, 450), 300
    if band == "medium":
        return (500, 900), 700
    if band == "large":
        return (900, 1500), 1100
    return None, None


def recommend(df, req, memory, top_k=5):
    d = df.copy()

    # --- basic columns ---
    if "food_name" not in d.columns:
        return None, "Missing food_name column."

    d["food_name"] = d["food_name"].astype(str)
    d["food_name_l"] = d["food_name"].str.lower()

    # --- avoid terms ---
    avoid_all = set(map(str.lower, memory.get("avoid_keywords", set()))) | set(map(str.lower, req.get("avoid", set())))
    for w in avoid_all:
        if w:
            d = d[~d["food_name_l"].str.contains(re.escape(w), na=False)]

    if d.empty:
        return None, "No matching foods."

    # --- calories ---
    cal_col = "calories_kcal" if "calories_kcal" in d.columns else "Energy"

    if req.get("calorie_min") is not None:
        d = d[d[cal_col] >= req["calorie_min"]]

    rng, target = calorie_range(req.get("cal_band"))
    if rng:
        lo, hi = rng
        d = d[(d[cal_col] >= lo) & (d[cal_col] <= hi)]

    if d.empty:
        return None, "No matching foods."

    # --- safe nutrient access ---
    def s(col):
        return d[col].fillna(0) if col in d.columns else pd.Series(0, index=d.index)

    protein = s("protein_g")
    fiber   = s("fiber_g")
    sodium  = s("sodium_mg")
    sugar   = s("sugar_g")

    # --- protein targeting ---
    protein_target = req.get("protein_target_g")
    if protein_target is not None:
        d = d[
            (protein >= protein_target * 0.7) &
            (protein <= protein_target * 1.3)
        ]
        protein = protein.loc[d.index]
        fiber   = fiber.loc[d.index]
        sodium  = sodium.loc[d.index]
        sugar   = sugar.loc[d.index]

    if d.empty:
        return None, "No foods match requested protein level."

    # --- weights ---
    priorities = req.get("priorities", set())

    w_p   = memory.get("w_protein", 1.0) * (1.3 if "high_protein" in priorities else 1.0)
    w_f   = memory.get("w_fiber", 0.6)   * (1.2 if "high_fiber" in priorities else 1.0)
    w_sod = memory.get("w_sodium_penalty", 0.002) * (1.4 if "low_sodium" in priorities else 1.0)
    w_sug = memory.get("w_sugar_penalty", 0.6)    * (1.4 if "low_sugar" in priorities else 1.0)

        # --- MUST-HAVE FILTERS for high-priority nutrients ---
    for nut, spec in req.get("nutrients", {}).items():
        if spec.get("priority", 0) >= 3:
            # treat missing as missing, not zero
            d[nut] = pd.to_numeric(d[nut], errors="coerce")

            # if user wants "high", we must exclude zero/missing
            d = d[d[nut].notna() & (d[nut] > 0)]

            # if a min is specified, enforce it
            if spec.get("min") is not None:
                d = d[d[nut] >= spec["min"]]

    if d.empty:
        return None, "No foods found with the requested nutrient data."
    # --- scoring ---
    cal_target = (
        req.get("calorie_min")
        if req.get("calorie_min") is not None
        else (target if target is not None else d[cal_col].median())
    )

    score = (
        -np.abs(d[cal_col] - cal_target) / 50.0
        - (np.abs(protein - protein_target) if protein_target else 0)
        + w_f * fiber
        - w_sod * sodium
        - w_sug * sugar
    )

    out = (
        d.assign(score=score)
         .sort_values("score", ascending=False)
         .head(top_k)
         .reset_index(drop=True)
    )

    return out, None



# In[15]:


print("MealBot is ready.")
print("Ask for food recommendations. Type 'quit' to stop.\n")

EXCLUDE_COLS = {"food_name", "food_name_l"}

while True:
    user = input("You: ").strip()

    if user.lower() in ("quit", "exit"):
        break

    # --- Handle feedback ---
    if any(k in user.lower() for k in ["too salty", "too sweet", "more protein", "avoid", "hate"]):
        apply_feedback(memory, user)
        print("Bot: Noted.\n")
        continue

    # --- Parse request ---
    req = llm_parse_request(user)
    req = apply_nutrient_shortcuts(req, user)

    # --- Recommend ---
    recos, err = recommend(structured_food_df, req, memory)

    if err:
        print(f"Bot: {err}\n")
        continue

    if recos.empty:
        print("Bot: No suitable recommendations found.\n")
        continue

    print("Bot: Recommended options:\n")

    for idx, row in recos.iterrows():
        print(f"{idx + 1}. {row['food_name']}")

        for col, val in row.items():
            if col in EXCLUDE_COLS or pd.isna(val):
                continue

            if isinstance(val, (int, float)):
                print(f"   - {col}: {val:.2f}")
            else:
                print(f"   - {col}: {val}")

        print()

    print("Bot: Provide feedback or ask for another recommendation.\n")



