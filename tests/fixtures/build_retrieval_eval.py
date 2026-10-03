# -*- coding: utf-8 -*-
"""生成检索评测集。calibration 与 holdout 使用不同事实。"""

import json
from pathlib import Path

from lightrag.search_strategies import tokenize

KB = "kb-main"
OTHER = "kb-other"
OWNER = "user-a"
OWNER_B = "user-b"


def doc(doc_id, name, chunks, kb=KB, owner=OWNER):
    return {
        "id": doc_id,
        "name": name,
        "kb_id": kb,
        "owner_id": owner,
        "chunks": [{"unit_id": unit, "text": text} for unit, text in chunks],
    }


def question(
    qid,
    split,
    track,
    query,
    answerable,
    docs,
    units,
    kb=KB,
    owner=OWNER,
    multihop=False,
    path_names=None,
):
    return {
        "id": qid,
        "split": split,
        "track": track,
        "query": query,
        "answerable": answerable,
        "kb_id": kb,
        "owner_id": owner,
        "expected_docs": docs,
        "expected_units": units,
        "multihop": multihop,
        "path_names": path_names or [],
    }


documents = [
    doc(
        "doc-edu",
        "resume.txt",
        [("t0001", "林知夏2018至2022年就读津门财经大学信息管理专业，导师是周衡。")],
    ),
    doc(
        "doc-cook",
        "pork.txt",
        [("t0001", "家常红烧肉先焯水，再放酱油和冰糖炖四十分钟。")],
    ),
    doc("doc-spare", "ribs.txt", [("t0001", "糖醋排骨先炸至外酥，再淋糖醋汁翻炒。")]),
    doc(
        "doc-trip",
        "hangzhou.txt",
        [("t0001", "西湖在杭州，春季可沿苏堤骑行，游船从断桥出发。")],
    ),
    doc(
        "doc-filter",
        "filter.txt",
        [("t0001", "净水器滤芯型号JW-220，每九十天更换一次。")],
    ),
    doc(
        "doc-clinic", "clinic.txt", [("t0001", "病历写明对青霉素过敏，就诊号MZ-3301。")]
    ),
    doc(
        "doc-pet",
        "pet.txt",
        [("t0001", "狗狗豆豆于2024年11月2日接种狂犬疫苗，芯片号P8891。")],
    ),
    doc(
        "doc-invoice",
        "invoice.txt",
        [("t0001", "发票号码INV-88421，税额一百二十元，销方北窗书店。")],
    ),
    doc(
        "doc-budget",
        "budget.txt",
        [("t0001", "三月家庭预算里通勤支出列为四百元，科目代码YS-0312。")],
    ),
    doc(
        "doc-plant",
        "plant.txt",
        [("t0001", "阳台绿萝每星期三浇水一次，花盆放在朝南窗台。")],
    ),
    doc(
        "doc-school",
        "school.txt",
        [
            (
                "t0001",
                "档案记载求职者在津门财经院校完成本科信息管理学业，入学年份为二零一八。",
            )
        ],
    ),
    doc(
        "doc-meet",
        "meeting.txt",
        [
            ("t0001", "极光项目安排评审会。"),
            ("t0002", "评审会位于三号会议室，时间是三月十二日下午三点。"),
        ],
    ),
    doc(
        "doc-river",
        "river.txt",
        [("t0001", "长江发源于唐古拉山，全长约六千三百公里。")],
    ),
    doc("doc-bridge", "bridge.txt", [("t0001", "长江大桥位于南京，桥面可以通汽车。")]),
    doc(
        "doc-bill-a",
        "bill-a.txt",
        [("t0001", "账单ZD-77821金额为三百元，收款方是晨光文具。")],
    ),
    doc(
        "doc-bill-b",
        "bill-b.txt",
        [("t0001", "账单ZD-77822金额为三百五十元，收款方是晨光文具。")],
    ),
    doc(
        "doc-intern",
        "intern.txt",
        [("t0001", "林知夏2023年在港湾物流公司实习六个月，岗位是仓储数据核对。")],
    ),
    doc(
        "doc-fish",
        "fish.txt",
        [("t0001", "清蒸鲈鱼需要姜丝和料酒，蒸八分钟后淋热油。")],
    ),
    doc(
        "doc-chengdu",
        "chengdu.txt",
        [("t0001", "十月去成都大熊猫基地，早上八点半入园，门票凭身份证换票。")],
    ),
    doc(
        "doc-print",
        "print.txt",
        [("t0001", "采购合同编号HT-2025-1107，乙方青柠印刷，价款一万二千。")],
    ),
    doc(
        "doc-ht18",
        "ht18.txt",
        [("t0001", "服务合同编号HT-2024-0918，甲方海盐工作室，价款八千元。")],
    ),
    doc(
        "doc-ht19",
        "ht19.txt",
        [("t0001", "补充协议编号HT-2024-0919，甲方海盐工作室，价款九千元。")],
    ),
    doc(
        "doc-air",
        "air.txt",
        [("t0001", "空气净化器型号AP-9，滤网每半年清洗，指示灯变红时更换。")],
    ),
    doc(
        "doc-cycle",
        "cycle.txt",
        [("t0001", "水循环包括蒸发、凝结和降水三个环节，课本页码是第48页。")],
    ),
    doc(
        "doc-guitar",
        "guitar.txt",
        [("t0001", "吉他练习记录写着每周三晚上练习扫弦二十分钟。")],
    ),
    doc(
        "doc-policy",
        "policy.txt",
        [("t0001", "意外险保单POL-55290，受益人写的是林知夏，保额三十万。")],
    ),
    doc(
        "doc-syrup",
        "syrup.txt",
        [("t0001", "糕点上色使用葡萄糖浆，烘焙温度一百八十度。")],
    ),
    doc(
        "doc-bio",
        "bio.txt",
        [
            ("t0001", "光合作用产生葡萄糖。"),
            ("t0002", "葡萄糖为细胞呼吸提供能量。"),
        ],
    ),
    doc(
        "doc-sweet",
        "sweet.txt",
        [("t0001", "葡萄糖导致甜味增加。")],
        kb=OTHER,
        owner=OWNER_B,
    ),
]

relations = [
    {
        "src": "极光项目",
        "relation": "安排",
        "tgt": "评审会",
        "document_id": "doc-meet",
        "unit_id": "t0001",
        "kb_id": KB,
        "owner_id": OWNER,
    },
    {
        "src": "评审会",
        "relation": "位于",
        "tgt": "三号会议室",
        "document_id": "doc-meet",
        "unit_id": "t0002",
        "kb_id": KB,
        "owner_id": OWNER,
    },
    {
        "src": "光合作用",
        "relation": "产生",
        "tgt": "葡萄糖",
        "document_id": "doc-bio",
        "unit_id": "t0001",
        "kb_id": KB,
        "owner_id": OWNER,
    },
    {
        "src": "葡萄糖",
        "relation": "提供",
        "tgt": "能量",
        "document_id": "doc-bio",
        "unit_id": "t0002",
        "kb_id": KB,
        "owner_id": OWNER,
    },
    {
        "src": "葡萄糖",
        "relation": "导致",
        "tgt": "甜味",
        "document_id": "doc-sweet",
        "unit_id": "t0001",
        "kb_id": OTHER,
        "owner_id": OWNER_B,
    },
]

questions = []


def add(
    split,
    track,
    query,
    docs,
    units,
    answerable=True,
    kb=KB,
    owner=OWNER,
    multihop=False,
    path_names=None,
):
    questions.append(
        question(
            f"q{len(questions)+1:03d}",
            split,
            track,
            query,
            answerable,
            docs,
            units,
            kb,
            owner,
            multihop,
            path_names,
        )
    )


lexical = [
    ("calibration", "林知夏的导师是谁", ["doc-edu"], ["t0001"]),
    ("calibration", "林知夏2018至2022年就读津门财经大学", ["doc-edu"], ["t0001"]),
    ("calibration", "家常红烧肉先焯水再放酱油和冰糖", ["doc-cook"], ["t0001"]),
    ("calibration", "家常红烧肉要炖四十分钟", ["doc-cook"], ["t0001"]),
    ("calibration", "西湖在杭州春季可沿苏堤骑行", ["doc-trip"], ["t0001"]),
    ("calibration", "西湖游船从断桥出发", ["doc-trip"], ["t0001"]),
    ("calibration", "净水器滤芯型号JW-220每九十天更换", ["doc-filter"], ["t0001"]),
    ("calibration", "病历写明对青霉素过敏，就诊号MZ-3301", ["doc-clinic"], ["t0001"]),
    ("calibration", "狗狗豆豆于2024年11月2日接种狂犬疫苗", ["doc-pet"], ["t0001"]),
    ("calibration", "芯片号P8891对应哪只狗狗", ["doc-pet"], ["t0001"]),
    ("calibration", "发票号码INV-88421的税额是多少", ["doc-invoice"], ["t0001"]),
    ("calibration", "发票INV-88421的销方是北窗书店", ["doc-invoice"], ["t0001"]),
    ("calibration", "三月家庭预算里通勤支出列为四百元", ["doc-budget"], ["t0001"]),
    ("calibration", "科目代码YS-0312是哪一项支出", ["doc-budget"], ["t0001"]),
    ("calibration", "阳台绿萝每星期三浇水一次", ["doc-plant"], ["t0001"]),
    ("calibration", "绿萝花盆放在朝南窗台", ["doc-plant"], ["t0001"]),
    ("holdout", "林知夏2023年在港湾物流公司实习多久", ["doc-intern"], ["t0001"]),
    ("holdout", "港湾物流公司的实习岗位是仓储数据核对", ["doc-intern"], ["t0001"]),
    ("holdout", "清蒸鲈鱼需要姜丝和料酒", ["doc-fish"], ["t0001"]),
    ("holdout", "清蒸鲈鱼蒸八分钟后淋热油", ["doc-fish"], ["t0001"]),
    ("holdout", "十月去成都大熊猫基地早上几点入园", ["doc-chengdu"], ["t0001"]),
    ("holdout", "成都大熊猫基地门票凭身份证换票", ["doc-chengdu"], ["t0001"]),
    ("holdout", "采购合同编号HT-2025-1107的乙方是谁", ["doc-print"], ["t0001"]),
    ("holdout", "青柠印刷这份采购合同价款一万二千", ["doc-print"], ["t0001"]),
    ("holdout", "空气净化器型号AP-9的滤网多久清洗", ["doc-air"], ["t0001"]),
    ("holdout", "AP-9指示灯变红时更换滤网", ["doc-air"], ["t0001"]),
    ("holdout", "水循环包括蒸发、凝结和降水", ["doc-cycle"], ["t0001"]),
    ("holdout", "水循环课本页码是第48页", ["doc-cycle"], ["t0001"]),
    ("holdout", "吉他练习记录写着每周三晚上练习扫弦", ["doc-guitar"], ["t0001"]),
    ("holdout", "扫弦练习每次二十分钟", ["doc-guitar"], ["t0001"]),
    ("holdout", "意外险保单POL-55290的受益人是谁", ["doc-policy"], ["t0001"]),
    ("holdout", "保单POL-55290的保额是三十万", ["doc-policy"], ["t0001"]),
]
for split, query, docs, units in lexical:
    add(split, "lexical", query, docs, units)

extra_lexical = [
    ("calibration", "周衡指导林知夏的信息管理专业", ["doc-edu"], ["t0001"]),
    ("calibration", "冰糖和酱油一起用于家常红烧肉", ["doc-cook"], ["t0001"]),
    ("calibration", "苏堤是西湖春季骑行的路线", ["doc-trip"], ["t0001"]),
    ("calibration", "JW-220是净水器滤芯型号", ["doc-filter"], ["t0001"]),
    ("calibration", "MZ-3301这次就诊记录了青霉素过敏", ["doc-clinic"], ["t0001"]),
    ("calibration", "P8891是豆豆的芯片号", ["doc-pet"], ["t0001"]),
    ("calibration", "北窗书店开出号码INV-88421", ["doc-invoice"], ["t0001"]),
    ("holdout", "仓储数据核对是港湾物流的实习岗位", ["doc-intern"], ["t0001"]),
    ("holdout", "料酒和姜丝用于清蒸鲈鱼", ["doc-fish"], ["t0001"]),
    ("holdout", "八点半进入成都大熊猫基地", ["doc-chengdu"], ["t0001"]),
    ("holdout", "合同HT-2025-1107价款一万二千", ["doc-print"], ["t0001"]),
    ("holdout", "型号AP-9的滤网每半年清洗", ["doc-air"], ["t0001"]),
    ("holdout", "凝结属于水循环的环节", ["doc-cycle"], ["t0001"]),
    ("holdout", "每周三晚上的扫弦练习二十分钟", ["doc-guitar"], ["t0001"]),
]
for split, query, docs, units in extra_lexical:
    add(split, "lexical", query, docs, units)

semantic = [
    ("calibration", "这位候选人的大学是哪一所？", ["doc-school"], ["t0001"]),
    ("calibration", "这位候选人读完高等学历的地方叫什么？", ["doc-school"], ["t0001"]),
    ("holdout", "这道鱼肴要蒸多久？", ["doc-fish"], ["t0001"]),
    ("holdout", "起锅之后浇上去的滚烫油脂叫什么？", ["doc-fish"], ["t0001"]),
    ("calibration", "窗边那盆观叶植物隔几天补水？", ["doc-plant"], ["t0001"]),
    ("holdout", "这份人身保障合同赔多少？", ["doc-policy"], ["t0001"]),
    ("calibration", "上下班那一项每个月列了多少钱？", ["doc-budget"], ["t0001"]),
    ("holdout", "当天要用什么证件兑换通行证？", ["doc-chengdu"], ["t0001"]),
]
for split, query, docs, units in semantic:
    add(split, "semantic", query, docs, units)

add("calibration", "hard_negative", "家常红烧肉先焯水还是先炸", ["doc-cook"], ["t0001"])
add("calibration", "hard_negative", "糖醋排骨淋的是什么汁", ["doc-spare"], ["t0001"])
add(
    "holdout",
    "hard_negative",
    "服务合同编号HT-2024-0918价款八千元吗",
    ["doc-ht18"],
    ["t0001"],
)
add(
    "holdout",
    "hard_negative",
    "补充协议编号HT-2024-0919价款是多少",
    ["doc-ht19"],
    ["t0001"],
)

add("calibration", "near_id", "账单ZD-77821金额是多少", ["doc-bill-a"], ["t0001"])
add(
    "calibration",
    "near_id",
    "账单ZD-77822的收款方金额是多少",
    ["doc-bill-b"],
    ["t0001"],
)
add("holdout", "near_id", "编号HT-2024-0918的价款", ["doc-ht18"], ["t0001"])
add("holdout", "near_id", "编号HT-2024-0919的价款", ["doc-ht19"], ["t0001"])

add("calibration", "same_name", "长江发源于哪里", ["doc-river"], ["t0001"])
add("calibration", "same_name", "长江大桥位于哪座城市", ["doc-bridge"], ["t0001"])
add(
    "holdout",
    "same_name",
    "细胞呼吸的能量由光合作用产生的糖提供吗",
    ["doc-bio"],
    ["t0001", "t0002"],
)
add(
    "holdout",
    "same_name",
    "糕点上色使用葡萄糖浆时烘焙温度是多少",
    ["doc-syrup"],
    ["t0001"],
)

add(
    "calibration",
    "multihop",
    "极光项目安排的评审会位于哪里",
    ["doc-meet"],
    ["t0001", "t0002"],
    multihop=True,
    path_names=["极光项目", "评审会", "三号会议室"],
)
add(
    "holdout",
    "multihop",
    "光合作用如何为细胞提供能量",
    ["doc-bio"],
    ["t0001", "t0002"],
    multihop=True,
    path_names=["光合作用", "葡萄糖", "能量"],
)

unanswerable = [
    "轨道电梯缆绳的屈服强度是多少",
    "火星移民舱的备件清单在哪一页",
    "深海潜标的电池仓怎么拆",
    "冰川钻孔岩芯的取样深度",
    "超导计算封装厂的门牌",
    "候鸟环志脚环的回收邮箱",
    "古琴减字谱里勾剔的指法图",
    "橡木发酵桶收在哪一窖",
    "滑雪镜防雾涂层的保修期",
    "蜂箱隔王板的安装间隙",
    "邮票齿孔度数怎么测量",
    "风洞试验段的风速上限",
    "陶轮转速刻度怎么校准",
    "灯塔透镜的焦距登记表",
    "马术障碍杯的赛道图",
    "珊瑚移植框的绑扎材料",
    "热气球燃烧器的年检章",
    "竹笛膜孔应该贴多厚",
    "天文台圆顶的开合密码",
    "雪线以上用什么容器采冰",
    "帆船帆号喷涂模板在哪",
    "地铁盾构机刀盘的扭矩",
    "蜡染冰纹的防染配方",
    "鸽子脚环的归属俱乐部",
    "火山灰样本的保管柜号",
    "钢琴调律师的上门档期",
    "沙漠公路里程桩的缺失段",
    "刺绣绷架的标准宽度",
    "观星营地何时必须关灯",
    "冷库月台的限高标志",
    "陶土釉料的批号对照",
    "信鸽公棚的归巢名次",
]
for index, query in enumerate(unanswerable):
    add(
        "calibration" if index < 16 else "holdout",
        "unanswerable",
        query,
        [],
        [],
        answerable=False,
    )


def tokens(text):
    return set(tokenize(text))


by_id = {item["id"]: item for item in documents}
failures = []
for item in questions:
    if item["track"] != "semantic":
        continue
    doc_tokens = set()
    for doc_id in item["expected_docs"]:
        for chunk in by_id[doc_id]["chunks"]:
            doc_tokens |= tokens(chunk["text"])
    shared = tokens(item["query"]) & doc_tokens
    if shared:
        failures.append((item["id"], item["query"], sorted(shared)))

import difflib

cal = [item["query"] for item in questions if item["split"] == "calibration"]
hold = [item["query"] for item in questions if item["split"] == "holdout"]
close = []
for left in cal:
    for right in hold:
        if difflib.SequenceMatcher(None, left, right).ratio() >= 0.8:
            close.append((left, right))

kb_text = {}
for item in documents:
    kb_text.setdefault(item["kb_id"], set())
    for chunk in item["chunks"]:
        kb_text[item["kb_id"]] |= tokens(chunk["text"])
leaks = []
for item in questions:
    if item["answerable"]:
        continue
    shared = tokens(item["query"]) & kb_text.get(item["kb_id"], set())
    if shared:
        leaks.append((item["query"], sorted(shared)))

print("docs", len(documents))
print("questions", len(questions))
print("unanswerable", sum(not item["answerable"] for item in questions))
print("semantic_overlap", failures)
print("close_pairs", close)
print("unanswerable_overlap", leaks)
if failures or close or leaks:
    raise SystemExit(1)

target = Path(__file__).with_name("retrieval_eval.json")
target.write_text(
    json.dumps(
        {"documents": documents, "relations": relations, "questions": questions},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(target)
