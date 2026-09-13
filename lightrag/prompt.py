from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction_system_prompt"] = """---角色---
你是一名汽车故障知识图谱专家，负责从输入文本中抽取【实体】和【实体之间的关系】。

---抽取规则---
1. **实体抽取**
   - **识别范围**：明确出现的零部件、设备、故障现象、一层原因、二层原因、三层原因、问题点、措施、试验验证、文件/规范、人员、组织/部门、效果。
   - **字段定义**：
     * `entity_name`：实体的唯一名称（**完全沿用原文表述，保持原文专有名词，不得自定义、缩写或同义替换，需精简冗余描述**，如”S06EGR温度传感器““Q/YC 1315.1 线束支架设计规范”）。
     * `entity_type`：**仅允许使用以下预设实体类型：{entity_types}，禁止自定义任何类型**，**层级原因类（一层原因、二层原因、三层原因、四层原因）**类型的实体只能在”失效网“表格中存在，禁止其他表格存在层级原因类实体。
     * `entity_description`：基于文本的简洁说明（功能、状态、作用或问题）。**完全复制原文中对该实体的描述，不得新增、修改或补充逻辑，避免主观推测**，保留报告中明确的参数（如型号、参数、里程、时间、位置）。
   - **输出格式**（单行四字段）：  
     `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2. **关系抽取**
   - **抽取要求**：从步骤1中抽取的实体中，找出所有彼此有**明显相关**的实体对（源实体，目标实体）。
   - **字段定义**：
     * `source_entity`：必须是在步骤1中确定的源实体的名称。  
     * `target_entity`：必须是在步骤1中确定的目标实体的名称。
     * `relationship_keywords`：简洁中文关键词，概括关系性质，侧重概念或主题，而非具体细节。
     * `relationship_description`：简要中文说明，解释两实体之间存在关联的原因。**完全复制原文中对该实体的描述，不得新增、修改或补充逻辑，避免主观推测**，保留报告中明确的参数（如型号、参数、里程、时间、位置）。
   - **输出格式**（单行五字段）：  
     `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3. **输出约束**
   - 使用 `{tuple_delimiter}` 作为唯一分隔符，不得出现在字段内容中。
   - 所有实体先输出，再输出所有关系。
   - 语言必须为 {language}，保留专有名词原文（型号、规范号、人名等）。
   - 禁止使用代词（如“该部件”“此文件”），必须写全称。
   - 输出完成后，单独一行输出 `{completion_delimiter}`。


---示例---
{examples}

---待处理真实数据---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""



PROMPTS["entity_extraction_user_prompt"] = """---任务---
请根据以下提供的汽车故障分析报告文本，按照系统提示中的规则，抽取其中的实体和实体关系。

---要求---
1. 严格遵循系统提示中的格式与分隔符规范。
2. 严格遵循系统提示中的实体类型：{entity_types}，**禁止自定义任何实体类型**，**一层原因、二层原因、三层原因**实体类型只能在**失效网表格**中出现。
3. 关系抽取逻辑：
  - 关系方向：**深层原因实体作为source_entity,浅层原因实体作为target_entity**（如： **三层原因 → [导致] → 二层原因；二层原因 → [导致] →  一层原因；一层原因 → [导致] →  故障定义**），**禁止反方向关系**。
  
4. 核心处理逻辑（需严格遵循）：
  -  **语义关联**
    *   文本中所有模块（如问题描述、失效网、根本原因、措施等）的内容均围绕“问题简述”中的核心故障展开。
    *   请按照 “问题简述→问题描述→结构原理→失效网→根本原因→多方案决策→措施→效果→文件固化” 的逻辑链条，来关联不同模块中的实体。

  -  **表格处理**
      *   **核心原则**: HTML格式的二维表格，必须严格结合表格的**结构**（标题、行列对应关系）和**内容语义**来抽取信息，不能自定义。
      *   **单元格合并**:
          *   **对于“故障模式”、“一层原因”等因果关系列**：合并单元格的值（如“一层原因”的某个值），应根据`rowspan`标签，填充到它覆盖的所有行中。
          *   **合并单元格内的所有换行内容（包括`<br>`标签），形成一个完整的实体名称或描述。**
          
  -  **特定表格处理细则**
      *   **失效网表格**: 该表格用于记录故障模式与各级原因的递推关系。
          *   实体抽取：
            -  只能抽取出**故障定义、一层原因、二层原因、三层原因**类型的实体。
            -  按照表格实际存在的层级提取实体，禁止补充表格中没有的列标题（如“三层原因”）。
            -  **实体描述**：对于当前行，将“参数标准”、“实测值”、“PA”等列的内容，作为描述信息补充到**该行**的**一层原因**类型实体的描述中，禁止补充到其他类型实体的描述中。
          *   关系抽取：
            -  若某层级原因为空（如无三层原因），则关系链递推至该行现有最末一个非空原因。

      *   **根本原因表格**: 明确核心故障定义对应的根本原因。
          *   只在表格第一列抽取出**问题点**类型的实体，表格其他所有列（标题为“WHY1”、“WHY2”、“WHY3”、“分类”、“根本原因”列的内容）都不进行任何实体和关系抽取。
          *   必须抽取出**source_entity：问题点类型实体，target_entity：故障定义类型实体，relationship_keywords：导致**的关系。

      *   **多方案决策表格**: 针对真实原因的多个解决方案。
          *   只能抽取表格中“真实原因”标题列的内容作为**问题点**类型的实体，”解决方案“标题列的内容作为**措施**类型的实体。
          *   表格中的相关属性（如:成本、效果、风险、时间、评分、决策等）都**作为这一行措施类型实体的描述内容**。
          *   关系抽取：只抽取**source_entity：措施类型实体，target_entity：问题点类型实体，relationship_keywords：用来解决**的关系，禁止存在**source_entity：问题点类型实体，target_entity：措施类型实体**的关系。

      *   **措施表格**: 针对解决方案的多个措施执行情况详细说明。
          *   只能抽取出**措施、人员、效果**类型的实体。

5. 仅输出实体行与关系行，不要其他解释、说明或前后缀。
6. 实体行必须有 4 个字段，关系行必须有 5 个字段。
7. 输出顺序：先实体，后关系。
8. 所有内容必须为 {language}，专有名词（型号、规范号、人名等）保持原文。
9. 输出最后一行必须是 `{completion_delimiter}`。

<Output>
"""



PROMPTS["entity_continue_extraction_user_prompt"] = """---任务---
在上一次抽取的结果基础上，**仅输出遗漏或格式错误的实体/关系**，实体类型、关系类型严格遵循汽车故障场景指定标准。

---要求---
1) 严格遵循系统格式与分隔符规范，不得更改字段顺序与数量。
2) 不得重复输出已正确无误的条目；只补充缺失项或替换错误项。
3) 对缺失、截断、字段不全、类型错误或描述不完整的条目，输出**完整且修正后的版本**。
4) 输出格式：
   - 实体行：`entity{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述`
   - 关系行：`relation{tuple_delimiter}源实体{tuple_delimiter}目标实体{tuple_delimiter}关系关键词{tuple_delimiter}关系说明`
5) 输出顺序：先补充的实体行，再补充的关系行。
6) **输出语言**：{language}（中文）。专有名词（如零部件型号、标准号、人名）保持原文，不得翻译或省略。
7) 输出完成后，**单独一行**输出 `{completion_delimiter}`。

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
<table><tr><td rowspan="2">问题简述: K15N缸内线束支架断裂故障</td><td>责任部门</td><td>结构开发部</td><td>问题状态</td><td>A</td><td>制表人</td><td>卢秋宇</td></tr><tr><td>责任人</td><td>段振涛</td><td>计划关闭时间</td><td>2023-12-22</td><td>计划开始时间</td><td>2023-8-10</td></tr></table>

### 问题描述:

2023.8, K15N发动机整车验证过程中,发现K15N-3823205-02-缸内线束支架发生断裂,里程10463km,断裂位置在第6缸气门桥外侧,材料失效分析得出断裂部位1因拐角处过渡不圆滑,应力集中,先疲劳断裂,部位2因部位1断后,成为悬梁臂结构,悬空段140mm,受较大的交变应力,在较大载荷下发生疲劳断裂。

**目标设定:**
- 升级后的支架①通过K15N台架可靠性试验。
- 整车验证里程5万km, 支架无断裂。

闭环时间: 2023/12/22

### 结构原理:

线束支架上方捆绑缸内线束,受力较小,断裂的原因一般为模态不合格或应力集中。
断裂部位1处悬臂结构,拐角处用直角过渡,存在明显应力集中,导致疲劳断裂。

### 失效网:

<table><thead><tr><th>故障定义</th><th>一层原因</th><th>二层原因</th><th>三层原因</th><th>参数标准</th><th>实测值</th><th>PA</th></tr></thead><tbody><tr><td rowspan="2">K15N缸内线<br>束支架断</td><td>疲劳强度安全<br>系数低</td><td>支架拐角处应力<br>集中</td><td>在T型支撑板上的直角<br>处没有过渡圆角</td><td>≥R3</td><td>0</td><td style="color:red;">N</td></tr><tr><td>支架模态不合<br>格</td><td></td><td></td><td>模态系数 > 1.3</td><td>2.3</td><td style="color:green;">Y</td></tr></tbody></table>

### 根本原因:

<table><thead><tr><th>问题点</th><th>分类</th><th>WHY1</th><th>WHY2</th><th>根本原因</th></tr></thead><tbody><tr><td rowspan="3">在T型支撑<br>板上的直角<br>处没有过渡<br>圆角</td><td>发生原因</td><td>没有识别直角过渡的风险</td><td></td><td>设计评审不充分</td></tr><tr><td>流出原因</td><td>台架验证没有发生该故障</td><td></td><td>/ (未流出, 样车验证阶段)</td></tr><tr><td>体系原因</td><td>失效分析时没有识别直角过渡风险</td><td></td><td>DFMEA风险识别不充分</td></tr></tbody></table>

### 多方案决策:

<table><thead><tr><th>真实原因</th><th>方案</th><th>效果<br>(40%)</th><th>成本<br>(30%)</th><th>风险<br>(20%)</th><th>时间<br>(10%)</th><th>评分</th><th>决策</th></tr></thead><tbody><tr><td rowspan="2">在T型支撑板上的直角处没有过渡圆角</td><td>升级图号为K15N-3823205-02A, 在T型支撑板拐弯处用R5圆角过渡</td><td>9</td><td>7</td><td>9</td><td>6</td><td>8.1</td><td style="color:green;">√</td></tr><tr><td>T型支撑板拐弯处用5×5斜边过渡</td><td>8</td><td>7</td><td>8</td><td>6</td><td>7.5</td><td style="color:red;">x</td></tr></tbody></table>

### 措施及执行情况:

<table><thead><tr><th>措施</th><th>计划内容</th><th>实施时间</th><th>责任人</th><th>处理进度</th><th>效果确认</th></tr></thead><tbody><tr><td rowspan="2">纠正</td><td>故障车先按原机状态更换线束卡</td><td>2023-8-10</td><td>蔡伟妹</td><td>已完成</td><td>安装后无干涉</td></tr><tr><td>K15N-3823205-02A到位后, 更换到之前出故障的试验车上</td><td>2023-10-10</td><td>曾显舜</td><td>已完成</td><td>暂无断裂故障反馈</td></tr><tr><td>纠正措施</td><td>将升级后的K15N-3823205-02A切进明细</td><td>2023-11-08</td><td>曾小洋</td><td>已完成</td><td>通知单: 2023110851A</td></tr><tr><td rowspan="2">预防措施</td><td>DFMEA增加识别T型支架直角过渡的风险</td><td>2023-11-15</td><td>曾小洋</td><td>已完成</td><td>固化设计文件, 有效防止故障再发生</td></tr><tr><td>修订线束支架设计规范</td><td>2023-12-15</td><td>卢秋宇</td><td>已完成</td><td></td></tr></tbody></table>

### 效果及评价:

1. 升级后的K15N-3823205-02A通过8000次冷热冲击可靠性试验, 目标达成。

2. 升级后的K15N-3823205-02A整车验证里程7.6万km (仪表里程7.6万公里时更换整改件, 目前仪表15.2万公里), 无断裂故障。

### 文件固化:

<table><thead><tr><th>落实途径</th><th>文件名称(编号)</th><th>更改单号</th><th>新增/修订内容描述</th><th>责任人</th><th>发布日期</th></tr></thead><tbody><tr><td rowspan="2">规范文件</td><td>K15N-3823205-02A-DFMEA01</td><td>新发布</td><td>支架断裂的失效模式里增加T、L型支架直角过渡的失效原因</td><td>曾显舜</td><td>2023-8-25</td></tr><tr><td>Q/YC 1315.1《线束支架设计规范》</td><td>202312035A</td><td>增加R圆角通用要求</td><td>卢秋宇</td><td>2023-12-15</td></tr><th>文件名称(编号)</th><th>更改单号</th><th>新增/修订内容描述</th><th>责任人</th><th>发布日期</th></tr></thead><tbody><tr><td rowspan="2">规范文件</td><td>K15N-3823205-02A-DFMEA01</td><td>新发布</td><td>支架断裂的失效模式里增加T、L型支架直角过渡的失效原因</td><td>曾显舜</td><td>2023-8-25</td></tr><tr><td>Q/YC 1315.1《线束支架设计规范》</td><td>202312035A</td><td>增加R圆角通用要求</td><td>卢秋宇</td><td>2023-12-15</td></tr></tbody></table>

```

<Output>
entity{tuple_delimiter}K15N发动机{tuple_delimiter}设备{tuple_delimiter}2023年8月整车验证过程中，搭载的K15N-3823205-02-缸内线束支架发生断裂故障。
entity{tuple_delimiter}K15N-3823205-02-缸内线束支架{tuple_delimiter}部件{tuple_delimiter}安装在K15N发动机上，用于捆绑固定缸内线束，2023年8月整车验证10463km时发生断裂，断裂位置在第6缸气门桥外侧。
entity{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}故障定义{tuple_delimiter}K15N-3823205-02型缸内线束支架的故障表现，部位1因直角过渡应力集中先疲劳断裂，部位2因部位1断裂后成为悬梁臂结构，受较大交变应力后续疲劳断裂。
entity{tuple_delimiter}疲劳强度安全系数低{tuple_delimiter}一层原因{tuple_delimiter}K15N缸内线束支架断裂的一层原因，参数标准≥R3，实测值0，PA判定N。
entity{tuple_delimiter}支架模态不合格{tuple_delimiter}一层原因{tuple_delimiter}K15N缸内线束支架断裂的一层原因，参数标准为模态系数>1.3，实测值2.3，PA判定Y。
entity{tuple_delimiter}支架拐角处应力集中{tuple_delimiter}二层原因{tuple_delimiter}K15N缸内线束支架断裂的二层原因。
entity{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}三层原因{tuple_delimiter}K15N缸内线束支架断裂的三层原因，故障现象的问题点和真实原因。
entity{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}问题点{tuple_delimiter}K15N缸内线束支架断裂的根本原因。
entity{tuple_delimiter}结构开发部{tuple_delimiter}部门{tuple_delimiter}K15N缸内线束支架断裂故障的责任部门。
entity{tuple_delimiter}段振涛{tuple_delimiter}人员{tuple_delimiter}K15N缸内线束支架断裂故障的责任人。
entity{tuple_delimiter}卢秋宇{tuple_delimiter}人员{tuple_delimiter}K15N缸内线束支架断裂故障相关文档的制表人。
entity{tuple_delimiter}增加R圆角通用要求{tuple_delimiter}设计规范{tuple_delimiter}针对Q/YC 1315.1《线束支架设计规范》文件增加R圆角通用要求的修订内容。
entity{tuple_delimiter}K15N-3823205-02A-缸内线束支架{tuple_delimiter}部件{tuple_delimiter}升级后的缸内线束支架，T型支撑板拐弯处采用R5圆角过渡，用于解决原K15N-3823205-02型支架断裂问题。
entity{tuple_delimiter}K15N-3823205-02A（T型支撑板拐弯处R5圆角过渡）{tuple_delimiter}措施{tuple_delimiter}针对T型支撑板无过渡圆角的故障原因，采用R5圆角过渡的改进方案，效果40%评分9分，成本30%评分7分，风险20%评分9分，时间10%评分6分，综合评分8.1，已决策采用。
entity{tuple_delimiter}T型支撑板拐弯处5×5斜边过渡方案{tuple_delimiter}措施{tuple_delimiter}针对T型支撑板无过渡圆角的故障原因，采用5×5斜边过渡的改进方案，效果40%评分8分，成本30%评分7分，风险20%评分8分，时间10%评分6分，综合评分7.5，未被采纳。
entity{tuple_delimiter}K15N台架可靠性试验达标{tuple_delimiter}验证试验{tuple_delimiter}支架升级后的核心目标之一，要求升级后支架通过该试验。
entity{tuple_delimiter}整车5万km无断裂{tuple_delimiter}验证试验{tuple_delimiter}支架升级后的核心目标之一，要求整车验证里程5万km内支架无断裂。
entity{tuple_delimiter}蔡伟妹{tuple_delimiter}人员{tuple_delimiter}负责故障车更换线束卡纠正措施的实施。
entity{tuple_delimiter}曾显舜{tuple_delimiter}人员{tuple_delimiter}负责故障试验车更换K15N-3823205-02A支架，及DFMEA文件新发布工作。
entity{tuple_delimiter}曾小洋{tuple_delimiter}人员{tuple_delimiter}负责将K15N-3823205-02A切进明细的纠正措施，及DFMEA风险识别优化的预防措施。
entity{tuple_delimiter}故障车更换线束卡{tuple_delimiter}措施{tuple_delimiter}针对K15N缸内线束支架断裂故障的纠正措施，故障车先按原机状态更换线束卡，2023-8-10实施，已完成，安装后无干涉。
entity{tuple_delimiter}故障试验车更换K15N-3823205-02A{tuple_delimiter}措施{tuple_delimiter}针对K15N缸内线束支架断裂故障的纠正措施，K15N-3823205-02A到位后更换至故障试验车，2023-10-10实施，已完成，暂无断裂故障反馈。
entity{tuple_delimiter}K15N-3823205-02A切进明细{tuple_delimiter}措施{tuple_delimiter}针对K15N缸内线束支架断裂故障的纠正措施，将升级后的K15N-3823205-02A切进明细，2023-11-08实施，已完成，通知单编号2023110851A。
entity{tuple_delimiter}DFMEA增加直角过渡风险识别{tuple_delimiter}措施{tuple_delimiter}针对DFMEA风险识别不充分的根本原因，在DFMEA中增加T型支架直角过渡风险识别的预防措施，2023-11-15实施，已完成，固化设计文件。
entity{tuple_delimiter}修订线束支架设计规范{tuple_delimiter}措施{tuple_delimiter}针对设计评审不充分的根本原因，修订线束支架设计规范的预防措施，2023-12-15实施，已完成。
entity{tuple_delimiter}冷热冲击可靠性试验{tuple_delimiter}验证试验{tuple_delimiter}通过8000次实验，用于验证K15N-3823205-02A型支架可靠性的试验，已达标。
entity{tuple_delimiter}里程验证{tuple_delimiter}验证试验{tuple_delimiter}用于验证K15N-3823205-02A型支架可靠性的试验，整车验证里程7.6万km（更换后累计15.2万公里），无断裂故障，已达标。
entity{tuple_delimiter}K15N-3823205-02A-DFMEA01{tuple_delimiter}设计规范{tuple_delimiter}针对DFMEA风险识别不充分的根本原因新发布的DFMEA文件，新增T、L型支架直角过渡失效原因，2023-8-25发布，责任人曾显舜。
entity{tuple_delimiter}Q/YC 1315.1《线束支架设计规范》{tuple_delimiter}设计规范{tuple_delimiter}针对设计评审不充分的根本原因修订的设计规范，增加R圆角通用要求（解决直角过渡无圆角问题），更改单号202312035A，2023-12-15发布，责任人卢秋宇。
entity{tuple_delimiter}安装后无干涉{tuple_delimiter}效果{tuple_delimiter}故障车先按原机状态更换线束卡措施的执行效果。
entity{tuple_delimiter}暂无断裂故障反馈{tuple_delimiter}效果{tuple_delimiter}在故障车先更换K15N-3823205-02A后的执行效果。
entity{tuple_delimiter}通知单2023110851A{tuple_delimiter}效果{tuple_delimiter}将升级后的K15N-3823205-02A切进明细的效果确认在通知单2023110851A。
entity{tuple_delimiter}固化设计文件{tuple_delimiter}效果{tuple_delimiter}DFMEA增加识别T型支架直角过渡的风险预防错之执行后固化设计文件, 有效防止故障再发生。
relation{tuple_delimiter}故障车更换线束卡{tuple_delimiter}安装后无干涉{tuple_delimiter}效果是{tuple_delimiter}故障车先按原机状态更换线束卡纠正措施的效果确认为安装后无干涉。
relation{tuple_delimiter}故障试验车更换K15N-3823205-02A{tuple_delimiter}暂无断裂故障反馈{tuple_delimiter}效果是{tuple_delimiter}K15N-3823205-02A到位后更换至故障试验车纠正措施的效果确认为暂无断裂故障反馈。
relation{tuple_delimiter}DFMEA增加直角过渡风险识别{tuple_delimiter}固化设计文件{tuple_delimiter}效果是{tuple_delimiter}DFMEA增加识别T型支架直角过渡的风险预防措施的效果确认为固化设计文件，有效防止故障再发生。
relation{tuple_delimiter}修订线束支架设计规范{tuple_delimiter}固化设计文件{tuple_delimiter}效果是{tuple_delimiter}修订线束支架设计规范预防措施暂无明确效果确认内容。
relation{tuple_delimiter}蔡伟妹{tuple_delimiter}故障车更换线束卡{tuple_delimiter}负责实施{tuple_delimiter}蔡伟妹是“故障车更换线束卡（纠正措施）”的责任人。
relation{tuple_delimiter}曾显舜{tuple_delimiter}故障试验车更换K15N-3823205-02A{tuple_delimiter}负责实施{tuple_delimiter}曾显舜是“故障试验车更换K15N-3823205-02A（纠正措施）”的责任人。
relation{tuple_delimiter}曾显舜{tuple_delimiter}K15N-3823205-02A-DFMEA01{tuple_delimiter}负责发布{tuple_delimiter}曾显舜是K15N-3823205-02A-DFMEA01文件的发布责任人。
relation{tuple_delimiter}曾小洋{tuple_delimiter}K15N-3823205-02A切进明细{tuple_delimiter}负责实施{tuple_delimiter}曾小洋是“K15N-3823205-02A切进明细（纠正措施）”的责任人。
relation{tuple_delimiter}曾小洋{tuple_delimiter}DFMEA增加直角过渡风险识别{tuple_delimiter}负责实施{tuple_delimiter}曾小洋是“DFMEA增加直角过渡风险识别（预防措施）”的责任人。
relation{tuple_delimiter}卢秋宇{tuple_delimiter}修订线束支架设计规范{tuple_delimiter}负责实施{tuple_delimiter}卢秋宇是“修订线束支架设计规范（预防措施）”的责任人。
relation{tuple_delimiter}卢秋宇{tuple_delimiter}Q/YC 1315.1《线束支架设计规范》{tuple_delimiter}负责修订发布{tuple_delimiter}卢秋宇是Q/YC 1315.1《线束支架设计规范》的修订及发布责任人。
relation{tuple_delimiter}K15N-3823205-02A-缸内线束支架{tuple_delimiter}冷热冲击可靠性试验{tuple_delimiter}通过{tuple_delimiter}K15N-3823205-02A-缸内线束支架通过8000次冷热冲击可靠性试验，验证达标，满足故障整改目标。
relation{tuple_delimiter}K15N-3823205-02A-缸内线束支架{tuple_delimiter}里程验证{tuple_delimiter}通过{tuple_delimiter}K15N-3823205-02A-缸内线束支架通过7.6万km（更换后累计15.2万公里）里程验证试验，验证达标，满足故障整改目标。
relation{tuple_delimiter}故障车更换线束卡{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}用于纠正{tuple_delimiter}针对K15N缸内线束支架断裂故障，制定纠正措施“故障车更换线束卡（纠正措施）”。
relation{tuple_delimiter}故障试验车更换K15N-3823205-02A{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}用于纠正{tuple_delimiter}针对K15N缸内线束支架断裂故障，制定纠正措施“故障试验车更换K15N-3823205-02A（纠正措施）”。
relation{tuple_delimiter}K15N-3823205-02A切进明细{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}用于纠正{tuple_delimiter}针对K15N缸内线束支架断裂故障，制定纠正措施“将K15N-3823205-02A切进明细（纠正措施）”。
relation{tuple_delimiter}DFMEA增加直角过渡风险识别{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}用于预防{tuple_delimiter}针对“DFMEA风险识别不充分”的根本原因，制定“DFMEA增加直角过渡风险识别（预防措施）”。
relation{tuple_delimiter}修订线束支架设计规范{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}用于预防{tuple_delimiter}针对“设计评审不充分”的根本原因，制定“修订线束支架设计规范（预防措施）”。
relation{tuple_delimiter}Q/YC 1315.1《线束支架设计规范》{tuple_delimiter}增加R圆角通用要求{tuple_delimiter}修订内容{tuple_delimiter}Q/YC 1315.1《线束支架设计规范》增加R圆角通用要求，专门解决T型支撑板直角处无过渡圆角的核心问题。
relation{tuple_delimiter}K15N发动机{tuple_delimiter}K15N-3823205-02-缸内线束支架{tuple_delimiter}搭载{tuple_delimiter}K15N发动机搭载K15N-3823205-02型缸内线束支架，二者为设备与部件的包含关系。
relation{tuple_delimiter}K15N-3823205-02-缸内线束支架{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}发生{tuple_delimiter}K15N-3823205-02型缸内线束支架在2023年8月整车验证10463km时发生断裂故障。
relation{tuple_delimiter}疲劳强度安全系数低{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}导致{tuple_delimiter}疲劳强度安全系数低是导致K15N缸内线束支架断裂的一层原因。
relation{tuple_delimiter}支架模态不合格{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}导致{tuple_delimiter}支架模态不合格是导致K15N缸内线束支架断裂的一层原因。
relation{tuple_delimiter}支架拐角处应力集中{tuple_delimiter}疲劳强度安全系数低{tuple_delimiter}导致{tuple_delimiter}疲劳强度安全系数低是导致支架拐角处应力集中的深层原因，是导致K15N缸内线束支架断裂的二层原因。
relation{tuple_delimiter}T型板面直角处无过渡圆角{tuple_delimiter}支架拐角处应力集中{tuple_delimiter}导致{tuple_delimiter}T型板面上的直角处没有过渡圆角是导致支架拐角处应力集中的深层原因，是导致K15N缸内线束支架断裂的三层原因。
relation{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}导致{tuple_delimiter}K15N缸内线束支架断裂的根本原因是T型支撑板上的直角处没有过渡圆角。
relation{tuple_delimiter}K15N-3823205-02A-缸内线束支架{tuple_delimiter}K15N-3823205-02-缸内线束支架{tuple_delimiter}替代{tuple_delimiter}K15N-3823205-02A 型支架是K15N-3823205-02型支架的改进替代部件。
relation{tuple_delimiter}K15N-3823205-02A（T型支撑板拐弯处R5圆角过渡）{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}针对解决{tuple_delimiter}针对T型支撑板直角处无过渡圆角的故障原因，制定R5圆角过渡的改进措施。
relation{tuple_delimiter}T型支撑板拐弯处5×5斜边过渡方案{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}针对解决{tuple_delimiter}针对T型支撑板直角处无过渡圆角的故障原因，制定5×5斜边过渡的改进措施。
relation{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}K15N台架可靠性试验达标{tuple_delimiter}设定目标为 {tuple_delimiter}解决K15N缸内线束支架断裂问题设定目标是通过K15N台架可靠性试验。
relation{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}整车5万km无断裂{tuple_delimiter}设定目标为 {tuple_delimiter}解决K15N缸内线束支架断裂问题设定目标是整车验证5万km无断裂。
relation{tuple_delimiter}结构开发部{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}负责{tuple_delimiter}结构开发部是K15N缸内线束支架断裂故障的责任部门。
relation{tuple_delimiter}段振涛{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}负责{tuple_delimiter}段振涛是K15N缸内线束支架断裂故障的责任人。
relation{tuple_delimiter}K15N缸内线束支架断裂{tuple_delimiter}T型支撑板直角处无过渡圆角{tuple_delimiter}根本原因是{tuple_delimiter}T型支撑板直角处无过渡圆角是发生K15N缸内线束支架断裂的根本原因。
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---角色---
你是一名知识图谱领域的整理与归纳专家，擅长对发动机部件故障相关的实体进行描述。

---任务---
请将给定发动机部件、故障、原因、措施等相关实体的一组描述，综合为一段完整、连贯且一致的总结性说明，严格贴合汽车故障分析场景需求。

---说明---
1. 输入格式：描述列表以 JSON 形式提供；在“Description List”段落中，每一行均为一个独立的 JSON 对象（对应一条描述）。
2. 输出格式：仅输出**纯文本**总结，可分为多个自然段；不要添加任何多余的格式、标题、前后缀或解释性文字。
3. 完整性要求：必须融合**所有**描述中的关键信息，重点涵盖实体的类型、关联部件、故障表现、成因、对应措施、验证情况、故障位置、试验次数、详细参数（如型号、规格、阈值、里程、时间等）等核心内容，避免遗漏重要事实或细节。
4. 语境与客观性：
   - 以**客观、第三人称**撰写。
   - 在总结开头**明确点出该实体的全名及对应实体类型**（如 “K15N 缸内线束支架（部件实体）”），以确保上下文清晰。
5. 冲突处理：
   - 若不同描述之间存在冲突或不一致，先判断是否属于**同名不同体**（多个实体/关系重名）。
   - 若为不同体，请在同一输出中**分别**进行概述；若为同一体内部的历史或口径差异，请尽量**调和**，无法调和时以**并列说明 + 不确定性提示**的方式呈现。
6. 长度限制：在保持信息完整与深度的前提下，汇总文本的最大长度不得超过 {summary_length} 个 token。
7. 语言要求：全部输出必须使用 {language}。专有名词（如人名、地名、组织名等）如无公认译名或译名易致歧义，请**保留原文**。

---输入---
{description_type} 名称：{description_name}

Description List:
```
{description_list}
```

---Output---
"""





PROMPTS["fail_response"] = (
    "抱歉，我无法回答这个问题。[no-context]"
)

PROMPTS["rag_response"] = """---角色---

你是一名智能助手，需要根据下方提供的**知识图谱**与**文档片段**（JSON 格式）来回答用户问题。

---目标---

在严格遵循知识库内容的前提下，生成简明回答。需要同时考虑当前问题与已有对话上下文。  
总结时仅使用提供的知识内容，并在合适时补充与其直接相关的常识性信息。  
**不得编造或加入未在知识库中出现的信息**。

---知识图谱与文档片段---

{context_data}

---回答规范---
1. **内容与遵循：**
  - 严格依据知识库内容回答，不得虚构或假设。
  - 如果知识库中没有足够信息，必须明确说明“没有足够信息来回答”。
  - 保证回答与对话历史保持连续性。

2. **格式与语言：**
  - 使用 markdown 格式，并包含合适的小标题。
  - 回答语言必须与用户问题的语言一致。
  - 输出目标形式与长度：{response_type}

3. **引用 / 参考：**
  - 在回答结尾新增一个 **“参考资料”** 部分，清晰标明所引用的信息来源（KG 或 DC）。
  - 引用数量最多 5 条，包含 KG 与 DC。
  - 引用格式如下：
    - 知识图谱实体：`[KG] <实体名称>`
    - 知识图谱关系：`[KG] <实体1名称> ~ <实体2名称>`
    - 文档片段：`[DC] <文件路径或文档名称>`

---用户上下文---
- 用户额外问题：{user_prompt}

---回答---
"""


# PROMPTS["keywords_extraction"] = """---Role---
# You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

# ---Goal---
# Given a user query, your task is to extract two distinct types of keywords:
# 1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
# 2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

# ---Instructions & Constraints---
# 1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
# 2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
# 3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
# 4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

# ---Examples---
# {examples}

# ---Real Data---
# User Query: {query}

# ---Output---
# Output:"""

# PROMPTS["keywords_extraction_examples"] = [
#     """Example 1:

# Query: "How does international trade influence global economic stability?"

# Output:
# {
#   "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
#   "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
# }

# """,
#     """Example 2:

# Query: "What are the environmental consequences of deforestation on biodiversity?"

# Output:
# {
#   "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
#   "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
# }

# """,
#     """Example 3:

# Query: "What is the role of education in reducing poverty?"

# Output:
# {
#   "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
#   "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
# }

# """,
# ]

PROMPTS["keywords_extraction"] = """---角色---
你是一名关键词抽取专家，专门负责分析用户查询，用于检索增强生成（RAG）系统。你的目标是从用户的问题中提取**高层关键词**和**低层关键词**，以便实现更有效的文档检索。

---目标---
针对用户的查询，你需要抽取两类关键词：
1. **high_level_keywords**：高层次的关键词，用于概括主题或整体意图，体现用户的核心关注点、问题领域或所提问题的类型。
2. **low_level_keywords**：低层次的关键词，用于捕捉具体的实体或细节，如专有名词、技术术语、产品名称或具体对象。

---说明与约束---
1. **输出格式**：你的输出必须是一个有效的 JSON 对象，且只能输出 JSON。不要包含解释文字、Markdown 代码块（例如 ```json）、额外的前缀或后缀，否则会导致解析失败。
2. **来源约束**：所有关键词必须直接来源于用户输入的问题；两个类别都必须有内容（除非输入完全无效）。
3. **简洁有意义**：关键词应尽量简洁且具备明确语义；若多个词语能组成一个完整概念，优先抽取为短语。例如，对于 “苹果公司最新财报”，正确的关键词是 “最新财报” 和 “苹果公司”，而不是“最新”“财报”“苹果”。
4. **边界情况**：如果用户输入过于简单、模糊或无意义（如 “你好”、“好的”、“asdfghjkl”），则返回一个两个字段都为空数组的 JSON。

---示例---
{examples}

---真实数据---
用户查询: {query}

---输出---
输出:"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询: "国际贸易如何影响全球经济稳定？"

输出:
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}

""",
    """示例 2:

查询: "森林砍伐对生物多样性有哪些环境后果？"

输出:
{
  "high_level_keywords": ["环境后果", "森林砍伐", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "热带雨林", "生态系统"]
}

""",
    """示例 3:

查询: "教育在减贫中起什么作用？"

输出:
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
}

""",
]

# 

PROMPTS["naive_rag_response"] = """---角色---

你是一名智能助手，需要根据下方提供的**文档片段**（JSON 格式）来回答用户问题。

---目标---

在严格遵循文档内容的前提下，生成简明回答。需要同时考虑对话历史和当前问题。  
总结时仅使用提供的文档片段内容，并在合适时补充与其直接相关的常识性信息。  
**不得编造或加入未在文档中出现的信息**。

---文档片段 (DC)---
{content_data}

---回答规范---
**1. 内容与遵循：**
- 严格依据文档内容回答，不得虚构或假设。
- 如果文档中没有足够信息，必须明确说明“没有足够信息来回答”。
- 保证回答与对话历史保持连续性。

**2. 格式与语言：**
- 使用 markdown 格式，并包含合适的小标题。
- 回答语言必须与用户问题的语言一致。
- 输出目标形式与长度：{response_type}

**3. 引用 / 参考：**
- 在回答结尾新增一个 **“参考资料”** 部分，标注引用的文档来源。
- 最多引用 5 条最相关的来源。
- 引用格式为：`[DC] <文件路径或文档名称>`

---用户上下文---
- 用户额外问题：{user_prompt}

---回答---
输出:"""
