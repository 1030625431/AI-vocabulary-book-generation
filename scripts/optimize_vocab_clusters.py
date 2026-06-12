from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path(__file__).resolve().parent
INPUT_WORDS = BASE / "output" / "word_filtered_3500_keep500.xlsx"
SOURCE = BASE / "output" / "deepseek_vocab_cluster_keep500" / "deepseek_vocab_clusters.xlsx"
OUT_DIR = BASE / "output" / "deepseek_vocab_cluster_keep500_optimized"


SPLIT_RULES: dict[str, list[tuple[str, list[str], str, str]]] = {
    "D0614": [
        ("en-/em- make into", ["embed", "embody", "enable", "ensure", "entitle"], "en-/em- = make into, put into", "都表示'使成为/放入某种状态'，适合做成一个使动前缀画面。"),
        ("en- intensify or apply", ["enforce", "enlarge", "enlighten", "enrich", "enroll"], "en- = make stronger/larger/brighter/richer or put on a roll", "都能画成'给对象加上一层力量/光/财富/名单'的动作。"),
    ],
    "D0801": [
        ("gene family", ["gene", "genetic"], "gen- = birth, kind", "基因和遗传直接同族。"),
        ("generate family", ["generate", "generation", "generator"], "gen- = produce", "围绕'产生'从动词到名词再到机器。"),
        ("general family", ["generalize", "generally"], "genus -> general kind", "都从'类别/一般'引申。"),
        ("gen- nature words", ["generous", "genius", "genuine"], "gen- = born nature", "都可联想到一个人与生俱来的品质：慷慨、天赋、真实。"),
    ],
    "D0808": [
        ("gl- light", ["gleam", "glitter", "glorious", "glow"], "gl- often suggests light or shining", "全都能放进'发光'画面。"),
        ("glance/glimpse", ["glance", "glimpse"], "gl- + quick look", "两词都表示很快地看一眼。"),
        ("glide", ["glide"], "gl- + smooth motion", "glide 是滑动，和发光组关联弱，单独记更干净。"),
    ],
    "D1284": [
        ("over-position", ["overcoat", "overhead", "overseas", "overnight"], "over- = above, across, through time", "都能用'越过/在上方/过一夜'的空间时间画面记。"),
        ("over-crossing", ["overflow", "overlap", "overtake", "overthrow"], "over- = across or beyond limit", "都带'越过界线'或'翻过去'的动态。"),
        ("over-perception", ["overhear", "overlook"], "over- in perception", "一个是不小心听到，一个是看漏/俯视，都是感知角度。"),
        ("over-excess/whole", ["overall", "overtime", "overwhelm"], "over- = excess or whole", "整体、超时、压倒，核心都是'超过平常范围'。"),
    ],
    "D1451": [
        ("radiate light", ["radiant", "radiate", "radiation"], "radi- = ray", "光芒向外射出：形容词、动词、名词。"),
        ("radioactive family", ["radioactive", "radioactivity", "radium"], "radio-/radium = radioactive ray", "放射性元素和性质放在一页。"),
        ("radius", ["radius"], "radius = ray-like line from center", "radius 是从中心射出的半径，和辐射词族有关但画面可单独处理。"),
    ],
    "D1466": [
        ("rule family", ["regular", "regularly", "regulate", "regulation"], "reg- = rule, make straight", "规则、规律、调节、规章是一条线。"),
        ("reign/region family", ["realm", "regime", "region", "reign"], "reg-/reign = rule, ruled area", "统治者、政权、疆域、统治动作适合画成王国。"),
    ],
    "D1784": [
        ("sub-under place", ["submarine", "submerge", "subway", "suburb"], "sub- = under or below", "水下、淹没、地下铁、城市边缘，都是'在下面/边缘'。"),
        ("sub-lower rank", ["subordinate", "subsidiary", "subsidy"], "sub- = lower, supporting", "下属、附属、补助，核心是下级支撑。"),
        ("sub-after/subtle", ["subsequent", "subtle"], "sub- = under, close behind", "随后与微妙都不适合硬塞大组，做小联想即可。"),
        ("substance family", ["substance", "substantial"], "substance = what stands under", "实质和大量/实质性的派生关系清晰。"),
        ("subtract", ["subtract"], "sub- + tract = draw away", "subtract 是'向下/拿走'，单独记更稳。"),
        ("subjective", ["subjective"], "subject + -ive", "主观的来自 subject，和 sub- 专题联系弱。"),
    ],
    "D1785": [
        ("success family", ["successfully", "succession", "successive", "successor"], "succed-/success = follow after", "成功/继承/连续来自'接着发生'的核心。"),
        ("suffice family", ["suffice", "sufficient", "sufficiently"], "suf- + fic = do enough", "足够：动词、形容词、副词。"),
        ("suspend/suppress", ["suppress", "suspend", "suspension"], "sup-/sus- = under, hold down/up", "压下与悬起都能画成'从下方控制'。"),
        ("suspicion family", ["suspicion", "suspicious"], "suspicion -> suspicious", "名词和形容词直接派生。"),
        ("suggestion", ["suggestion"], "suggest -> suggestion", "表中只有名词，可补 suggest。"),
        ("summon", ["summon"], "summon = call up", "召唤义独立。"),
        ("surge", ["surge"], "surge = rise suddenly", "浪涌/激增单独画面更强。"),
        ("sustain", ["sustain"], "sustain = hold up/support", "维持、支撑，单独画面更清楚。"),
    ],
    "D1793": [
        ("super-above", ["superficial", "supermarket", "supersonic", "supervise"], "super- = above, over", "表面、超市、超音速、监督，都有'在上/超过'。"),
        ("superiority/surpass", ["superiority", "surpass"], "super-/sur- = above, beyond", "优越和超过是同一高低画面。"),
        ("supplement family", ["supplement", "supplementary"], "supplement -> supplementary", "补充与补充的。"),
        ("surprise family", ["surprise", "surprising", "surprisingly"], "surprise -> surprising -> surprisingly", "惊讶的词性链。"),
        ("surround/survey", ["surroundings", "survey"], "sur- = over/around", "环绕环境和俯瞰调查可放在一个'从上环视'画面。"),
        ("surname", ["surname"], "sur- = over/additional name", "姓氏是附加在名上的家族名。"),
        ("surrender", ["surrender"], "surrender = give over", "投降/交出，单独记。"),
    ],
    "D1864": [
        ("trans-send/carry", ["transfer", "transmission", "transmit", "transaction"], "trans- = across + carry/send/do", "都表示跨越后把东西送过去或完成交易。"),
        ("transform family", ["transformation", "transformer"], "transform -> transformation/transformer", "改变形态和变压器直接成组。"),
        ("translate family", ["translate", "translation"], "translate -> translation", "翻译的动词和名词。"),
        ("transit/transport", ["transient", "transit", "transition", "transportation", "traverse"], "trans- = across, through", "穿过、过渡、运输、横越，都是移动画面。"),
        ("trans-beyond/through", ["transcend", "transparent", "transplant"], "trans- = beyond/through/across", "超越、透明、移植都表现'穿过边界'。"),
        ("transistor", ["transistor"], "transfer + resistor", "晶体管词源较专门，单独记更准确。"),
    ],
    "D1891": [
        ("un-fortune", ["unfortunate", "unfortunately", "unlucky"], "un- = not + fortune/luck", "倒霉、不幸一条线。"),
        ("un-comfort", ["unbearable", "uncomfortable", "uneasy"], "un- = not + comfort/ease/bear", "都能画成难受、无法忍受。"),
        ("un-truth/justice/kindness", ["unexpected", "unjust", "unkind", "unlikely"], "un- = not/opposite", "非预期、不公、不友善、不可能，都是否定判断。"),
        ("un-conscious/reverse", ["unconscious", "uncover"], "un- = not or reverse action", "unconscious 是否定，uncover 是反向动作，放小组说明差异。"),
    ],
    "D1892": [
        ("under-place", ["underground", "underneath", "underlying", "underline"], "under- = below", "地下、下面、底层、下划线都是'在下'。"),
        ("under-weakening", ["underestimate", "undermine"], "under- = below/too little", "低估和削弱都可画成从下面挖空。"),
        ("under-take/go", ["undergo", "undertake", "undertaking"], "under- = take on, go through", "经历、承担、事业/任务是一条动作链。"),
        ("undergraduate", ["undergraduate"], "undergraduate = before graduate", "本科生是未毕业阶段，单独记更清楚。"),
    ],
}


EXTRA_FORMS: dict[str, list[str]] = {
    "abolish": ["abolition", "abolitionist"],
    "abrupt": ["abruptly", "abruptness"],
    "accord": ["accordant"],
    "analyse": ["analysis", "analyst"],
    "analytic": ["analytically"],
    "assign": ["assignable"],
    "brace": ["bracing"],
    "breed": ["breeding", "breeder"],
    "capable": ["capability", "capably"],
    "doctrine": ["doctrinal"],
    "document": ["documentation", "documentary"],
    "fate": ["fateful"],
    "fatal": ["fatally", "fatality"],
    "frequency": ["frequent"],
    "grocer": ["groceries"],
    "invest": ["investor"],
    "notable": ["notably", "note"],
    "noticeable": ["noticeably"],
    "notify": ["notification"],
    "notorious": ["notoriously", "notoriety"],
    "production": ["produce", "producer"],
    "productive": ["productively"],
    "productivity": ["product"],
    "stiff": ["stiffly", "stiffness"],
    "transformation": ["transform"],
    "transformer": ["transform"],
    "transaction": ["transact"],
    "transmission": ["transmissive"],
    "translate": ["translator"],
    "transition": ["transitional"],
    "transient": ["transience"],
    "transparent": ["transparency", "transparently"],
    "transplant": ["transplantation"],
    "transcend": ["transcendent", "transcendence"],
    "radioactive": ["radioactively"],
    "radiant": ["radiantly"],
    "radiate": ["radiative"],
    "radius": ["radial"],
    "regular": ["irregular", "regularity"],
    "regulate": ["regulator", "regulatory"],
    "region": ["regional"],
    "reign": ["sovereign"],
    "gene": ["genome"],
    "genetic": ["genetics", "genetically"],
    "generate": ["generative"],
    "generation": ["generational"],
    "generous": ["generosity", "generously"],
    "genuine": ["genuinely"],
    "genius": ["ingenious"],
    "submarine": ["submariner"],
    "submerge": ["submersion"],
    "subordinate": ["subordination"],
    "subsidiary": ["subsidiarity"],
    "subsidy": ["subsidize"],
    "substance": ["substantiate"],
    "substantial": ["substantially"],
    "subtract": ["subtraction"],
    "subjective": ["subjectively", "subjectivity"],
    "successor": ["succeed", "success"],
    "succession": ["successive"],
    "sufficient": ["sufficiency"],
    "suppress": ["suppression"],
    "suspend": ["suspense"],
    "suspicious": ["suspiciously"],
    "suggestion": ["suggest", "suggestive"],
    "sustain": ["sustainable", "sustainability"],
    "supervise": ["supervisor", "supervision"],
    "superficial": ["superficially"],
    "superiority": ["superior"],
    "supplement": ["supplemental"],
    "surpass": ["surpassing"],
    "surprise": ["surprised"],
    "surrender": ["surrendered"],
    "survey": ["surveyor"],
    "unfortunate": ["fortune", "fortunate"],
    "unconscious": ["conscious", "consciousness"],
    "unexpected": ["expect", "expectation"],
    "underestimate": ["estimation"],
    "undermine": ["undermining"],
    "undertake": ["undertaken"],
    "undergraduate": ["graduate"],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_excel(SOURCE).fillna("")
    input_words = read_input_words()
    rows: list[dict[str, Any]] = []

    for _, row in source.iterrows():
        words = split_words(row["words"])
        rules = SPLIT_RULES.get(str(row["group_id"]))
        if rules:
            used: set[str] = set()
            for title, part_words, root, explanation in rules:
                selected = [word for word in part_words if word in words]
                if not selected:
                    continue
                used.update(selected)
                rows.append(make_row(row, selected, input_words, title, root, explanation, "split_large_group"))
            for leftover in sorted(set(words) - used):
                rows.append(make_row(row, [leftover], input_words, leftover, "", "从过大的前缀组中拆出的单词，单独成页更利于配图。", "split_leftover"))
        else:
            rows.append(make_row(row, words, input_words, str(row["headword"]) or words[0], str(row["root_affix"]), str(row["group_explanation"]), "kept"))

    out = pd.DataFrame(rows)
    out = out.sort_values(["source_words"], key=lambda s: s.str.lower()).reset_index(drop=True)
    out.insert(0, "optimized_group_id", [f"O{i:04d}" for i in range(1, len(out) + 1)])
    out.to_excel(OUT_DIR / "optimized_vocab_clusters.xlsx", index=False)
    out.to_csv(OUT_DIR / "optimized_vocab_clusters.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_groups": len(source),
        "optimized_groups": len(out),
        "source_unique_words": len(input_words),
        "covered_source_words": len({word for value in out["source_words"] for word in split_words(value)}),
        "multiword_groups": int((out["source_count"] > 1).sum()),
        "singleton_groups": int((out["source_count"] == 1).sum()),
        "max_source_group_size": int(out["source_count"].max()),
        "groups_with_added_forms": int((out["added_count"] > 0).sum()),
        "total_added_forms": int(out["added_count"].sum()),
    }
    (OUT_DIR / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")


def read_input_words() -> set[str]:
    df = pd.read_excel(INPUT_WORDS)
    word_col = "单词" if "单词" in df.columns else df.columns[1]
    return {str(word).strip().lower() for word in df[word_col].dropna() if str(word).strip()}


def make_row(
    source_row: pd.Series,
    words: list[str],
    input_words: set[str],
    headword: str,
    root_affix: str,
    explanation: str,
    decision: str,
) -> dict[str, Any]:
    words = sorted(dict.fromkeys(word.strip().lower() for word in words if word.strip()))
    added = collect_added_forms(words, input_words)
    source_defs = filter_definitions(str(source_row.get("definitions", "")), set(words))
    memory_words = words + added
    return {
        "headword": headword if headword else words[0],
        "source_count": len(words),
        "source_words": ", ".join(words),
        "added_count": len(added),
        "added_forms": ", ".join(added),
        "memory_word_count": len(memory_words),
        "all_memory_words": ", ".join(memory_words),
        "source_group_id": source_row.get("group_id", ""),
        "decision": decision,
        "root_affix": root_affix,
        "group_explanation": explanation,
        "memory_link": source_row.get("memory_link", ""),
        "mnemonic": source_row.get("mnemonic", ""),
        "collocations": source_row.get("collocations", ""),
        "related_recommendations": source_row.get("related_recommendations", ""),
        "definitions": source_defs,
    }


def collect_added_forms(words: list[str], input_words: set[str]) -> list[str]:
    added: list[str] = []
    current = set(words)
    for word in words:
        candidates = EXTRA_FORMS.get(word, []) + conservative_derivatives(word)
        for candidate in candidates:
            candidate = candidate.lower().strip()
            if not candidate or candidate in current or candidate in input_words or candidate in added:
                continue
            if not re.fullmatch(r"[a-z][a-z-]*", candidate):
                continue
            added.append(candidate)
    return added[:8]


def conservative_derivatives(word: str) -> list[str]:
    # Keep this deliberately conservative. Broad suffix rules produce too many
    # fake or low-value forms, so high-confidence additions live in EXTRA_FORMS.
    return []


def filter_definitions(definitions: str, wanted: set[str]) -> str:
    pieces = [piece.strip() for piece in definitions.split("|") if piece.strip()]
    kept = []
    for piece in pieces:
        word = piece.split(":", 1)[0].strip().lower()
        if word in wanted:
            kept.append(piece)
    return " | ".join(kept)


def split_words(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [word.strip().lower() for word in str(value).split(",") if word.strip()]


if __name__ == "__main__":
    main()
