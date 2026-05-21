# 纯 LLM 二分类完整指南（不含 RAG）

> **场景**：仅用 Gemma 4 E4B + Few-shot In-context Learning 做 Drebin-215 二分类
> **目标读者**：第一次用 LLM 做分类任务的初学者
> **配套**：`plan/项目方案.md` 中"消融实验 Exp 1"对应的实现

---

## 目录

1. [核心概念：什么是 In-context Learning](#一核心概念什么是-in-context-learning)
2. [K 选多少（关键决策）](#二k-选多少关键决策)
3. [公平对比的关键设定](#三公平对比的关键设定)
4. [Prompt 模板设计](#四prompt-模板设计)
5. [完整可运行代码](#五完整可运行代码)
6. [常见问题排查](#六常见问题排查)
7. [评估与可视化](#七评估与可视化)

---

## 一、核心概念：什么是 In-context Learning

### 1.1 与"训练"的本质区别

| 维度 | 传统 ML 训练（RF/SVM） | LLM In-context Learning |
|------|----------------------|------------------------|
| 权重是否更新 | ✅ 是 | ❌ 否 |
| 数据用法 | 用整个训练集反向传播更新参数 | 把 K 个例子放进 Prompt |
| 学到的东西放在哪 | 模型参数里 | **Prompt 里**（推理完就忘） |
| 每次预测新样本 | 加载训练好的模型即可 | **每次都要把 K 个例子再传一次** |

### 1.2 一个直观例子

**传统 ML**：
```
1. 把 10 个样本（标签）喂给 RandomForest.fit() → 模型权重更新
2. 推理：model.predict(new_sample) → 标签
   （新样本来时不需要再看那 10 个旧样本）
```

**LLM ICL**：
```
1. 每次推理前，把那 10 个旧样本拼到 Prompt 开头
2. 推理：ollama.chat(model, prompt=[10 examples] + [new_sample]) → 标签
   （新样本来时必须再传 10 个例子，因为 LLM 不记得它们）
```

**所以 ICL 的"K"不是训练数据量，是"提示词里的例子数量"。**

### 1.3 为什么 ICL 能 work

Gemma 4 在预训练时见过：
- 海量 Android 文档 / 安全博客
- 大量 "权限 → 应用类型" 的语料

所以即使你不告诉它什么是恶意软件，它已经"隐隐知道" `SEND_SMS + READ_CONTACTS + INTERNET` 看着不对劲。

**K 个例子的作用**：
1. 告诉它你想要的输出格式（`S` 还是 `Malware`）
2. 校准它对"恶意"的判断阈值
3. 帮它适应 Drebin 特有的特征命名

---

## 二、K 选多少（关键决策）

### 2.1 我的推荐

**主实验用 K=5**，附加跑 K=1 / K=10 做"K 的影响"折线图。

理由：
- K=5 是 Few-shot 的甜区
- 跟 PPT 默认示意一致
- 不会爆 Gemma 4 的上下文（每个例子约 300 字 × 10 = 3000 字，远小于 128K）

### 2.2 K 取值建议表

| K | Prompt 总 token 估计 | 适用 | 备注 |
|---|---------------------|------|------|
| K=0 | ~500 | 基线对比（zero-shot） | 测 LLM 先验能力 |
| K=3 | ~2000 | 极少样本 | 验证 K=3 是否够 |
| **K=5** | **~3500** | **主实验** | **推荐起步** |
| K=10 | ~7000 | 对比实验 | 测增益是否收敛 |
| K=20 | ~14000 | 上限测试 | 可能开始下降 |

### 2.3 为什么 K 太大反而差

你笔记第二讲讲过的现象，全部在这里复现：

| 问题 | 表现 | 解决 |
|------|------|------|
| **Lost in the middle** | K=20 时中间例子被忽略 | K ≤ 10 |
| **Context Rot** | K=30+ 时整体准确率下降 | K ≤ 10 |
| **格式漂移** | 例子太多，LLM 输出格式不稳 | 加强 system prompt |
| **推理变慢** | Prompt 越长越慢 | K 适中 + 量化模型 |

### 2.4 "Few-shot 性能曲线"做实验

跑 K ∈ {0, 1, 3, 5, 10, 20}，画一张折线图，X 轴 K，Y 轴 Accuracy。**这张图是报告的亮点**。

---

## 三、公平对比的关键设定

### 3.1 你的核心疑问

> "RF 用 80% 训练能到 98-99%，LLM 只用 K=5 会很差，能比吗？"

**答案：不能直接这样比，必须公平。**

### 3.2 PPT 强制要求 Few-shot

PPT 第 5 页明确："每类恶意软件仅选择 **K 个样本** 作为训练/示例"。

这意味着 **所有方法**（RF/纯 LLM/LLM+RAG）都必须在同样 K-shot 设定下做。

### 3.3 正确的对比表（推荐做法）

| 方法 | K=5 训练样本 | K=10 训练样本 | K=20 训练样本 | 80% 训练 |
|------|------------|--------------|--------------|---------|
| **Random Forest** | 训练 → 测试 | 训练 → 测试 | 训练 → 测试 | 上限参考（不参与主对比） |
| **SVM** | 训练 → 测试 | 训练 → 测试 | 训练 → 测试 | 上限参考 |
| **纯 LLM Few-shot** | ICL → 测试 | ICL → 测试 | ICL → 测试 | N/A |
| **LLM+RAG** | ICL+规则 → 测试 | ICL+规则 → 测试 | ICL+规则 → 测试 | N/A |

### 3.4 预期实验故事

| K=5 时 |
|-------|
| RF: 60% 左右（训练数据太少） |
| SVM: 65% 左右 |
| 纯 LLM: 80% 左右 |
| LLM+RAG: 88% 左右 |

**报告主结论**：
> "在少样本（K=5）设定下，传统机器学习方法因训练数据不足表现欠佳，而基于大语言模型的方法通过 In-context Learning 利用预训练知识达到了显著更高的准确率，进一步引入 RAG 知识增强后准确率进一步提升 X 个百分点。这验证了 PPT 提出的 '少样本场景下知识增强方法优于纯数据驱动方法' 的假设。"

### 3.5 "80% 训练"的 RF 怎么办

不要扔！在报告里作为**"上限参考"**单独提：

> "若放开 K-shot 限制，使用 80% 训练数据，RF 可达 98.9%。
> 这说明在数据充足时传统方法仍有强大表现，但本作业关注的是少样本场景，
> 不属于公平对比范畴。"

---

## 四、Prompt 模板设计

### 4.1 三个核心要素

```
┌─────────────────────────────┐
│  ① System Prompt            │  ← 定义角色 + 输出格式
└─────────────────────────────┘
┌─────────────────────────────┐
│  ② K 个 Few-shot Examples   │  ← 良性/恶意交替排列
└─────────────────────────────┘
┌─────────────────────────────┐
│  ③ Test Sample              │  ← 你要分类的新样本
└─────────────────────────────┘
```

### 4.2 System Prompt（建议）

```
You are an Android malware security analyst.
Given an Android app's features (permissions, API calls, intents),
classify it as Malware ("S") or Benign ("B").

Rules:
- Output STRICT JSON only, no markdown, no prose.
- The "label" field MUST be exactly "S" or "B".
- Base your judgment on the labeled examples below.
```

### 4.3 用户 Prompt 模板

```
Below are 10 labeled examples (5 Benign, 5 Malware) from the Drebin dataset:

[Example 1] Label: B (Benign)
Permissions: INTERNET, ACCESS_NETWORK_STATE, VIBRATE
API: bindService, transact, ServiceConnection

[Example 2] Label: S (Malware)
Permissions: SEND_SMS, RECEIVE_SMS, READ_PHONE_STATE, INTERNET
API: SmsManager.sendTextMessage, TelephonyManager.getDeviceId
Intent: android.provider.Telephony.SMS_RECEIVED

[Example 3] Label: B (Benign)
...

[Example 10] Label: S (Malware)
...

---

Now classify the following new sample:

Permissions: SEND_SMS, READ_CONTACTS, INTERNET
API: HttpGet.init, SmsManager
Intent: android.intent.action.BOOT_COMPLETED

Output JSON only:
{"label": "S" or "B", "evidence": ["feature1", ...], "explanation": "<one-sentence Chinese explanation>"}
```

### 4.4 关键设计原则

| 原则 | 为什么 |
|------|-------|
| **良性/恶意交替** | 避免 Recency Bias（模型偏向最后看到的标签） |
| **格式完全一致** | 所有例子用同样模板，模型才能学到格式 |
| **特征精简** | 每例子 Top-15 权限 + Top-10 API，避免太长 |
| **明确 label 取值** | 写"S or B"而不是"yes or no"，避免歧义 |
| **强制 JSON 输出** | Ollama 用 `format='json'` 强制 |

---

## 五、完整可运行代码

### 5.1 数据准备

```python
import pandas as pd
import numpy as np

DREBIN_PATH = ('data/Android Malware Dataset for Machine Learning/'
               'drebin-215-dataset-5560malware-9476-benign.csv')
CATS_PATH = ('data/Android Malware Dataset for Machine Learning/'
             'dataset-features-categories.csv')

df = pd.read_csv(DREBIN_PATH)
cats = pd.read_csv(CATS_PATH, header=None, names=['feature', 'category'])

# 1. 清理 NaN（有少量行的 class 列是空的）
df = df[df['class'].isin(['S', 'B'])].reset_index(drop=True)

# 2. K-shot 采样（每类 K 个）
def sample_kshot(df, k=5, seed=42):
    benign = df[df['class'] == 'B'].sample(k, random_state=seed)
    malware = df[df['class'] == 'S'].sample(k, random_state=seed)
    return pd.concat([benign, malware]).reset_index(drop=True), \
           pd.concat([
               df[df['class'] == 'B'].drop(benign.index),
               df[df['class'] == 'S'].drop(malware.index),
           ])

K = 5
train_df, rest_df = sample_kshot(df, k=K, seed=42)

# 3. 测试集（200 个，类别平衡）
test_df = pd.concat([
    rest_df[rest_df['class'] == 'B'].sample(100, random_state=42),
    rest_df[rest_df['class'] == 'S'].sample(100, random_state=42),
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"训练（K-shot 例子）: {len(train_df)} 行 = {K}×2")
print(f"测试集: {len(test_df)} 行")
```

### 5.2 特征转文本

```python
def sample_to_text(row, cats_df, max_perm=15, max_api=10, max_intent=5):
    """把一行 0/1 特征转换为自然语言描述"""
    feat2cat = dict(zip(cats_df['feature'], cats_df['category']))
    buckets = {'Permission': [], 'API': [], 'Intent': [], 'Commands': []}

    for col, val in row.items():
        if col == 'class' or val != 1:
            continue
        cat = feat2cat.get(col, '')
        if 'Permission' in cat:
            buckets['Permission'].append(col)
        elif 'API' in cat:
            buckets['API'].append(col)
        elif 'Intent' in cat:
            buckets['Intent'].append(col)
        elif 'Commands' in cat:
            buckets['Commands'].append(col)

    parts = []
    if buckets['Permission']:
        parts.append(f"Permissions: {', '.join(buckets['Permission'][:max_perm])}")
    if buckets['API']:
        parts.append(f"API: {', '.join(buckets['API'][:max_api])}")
    if buckets['Intent']:
        parts.append(f"Intent: {', '.join(buckets['Intent'][:max_intent])}")
    if buckets['Commands']:
        parts.append(f"Commands: {', '.join(buckets['Commands'])}")
    return '\n'.join(parts)
```

### 5.3 构建 Few-shot Prompt

```python
def build_examples_block(train_df, cats_df):
    """把 K-shot 样本拼成 Few-shot examples 字符串"""
    # 交替排列良性和恶意，避免 Recency Bias
    benign = train_df[train_df['class'] == 'B'].reset_index(drop=True)
    malware = train_df[train_df['class'] == 'S'].reset_index(drop=True)

    interleaved = []
    for i in range(max(len(benign), len(malware))):
        if i < len(benign):
            interleaved.append(benign.iloc[i])
        if i < len(malware):
            interleaved.append(malware.iloc[i])

    blocks = []
    for i, row in enumerate(interleaved, start=1):
        text = sample_to_text(row, cats_df)
        label = row['class']
        label_word = 'Malware' if label == 'S' else 'Benign'
        blocks.append(f"[Example {i}] Label: {label} ({label_word})\n{text}")
    return '\n\n'.join(blocks)


def build_prompt(test_row, train_df, cats_df):
    examples = build_examples_block(train_df, cats_df)
    test_text = sample_to_text(test_row, cats_df)
    return f"""Below are {len(train_df)} labeled examples from the Drebin dataset:

{examples}

---

Now classify the following new sample:

{test_text}

Output JSON only:
{{"label": "S" or "B", "evidence": ["feature1", ...], "explanation": "<one-sentence Chinese explanation>"}}"""
```

### 5.4 推理 + 解析

```python
import ollama, json, re

SYSTEM_PROMPT = """You are an Android malware security analyst.
Given an Android app's features (permissions, API calls, intents),
classify it as Malware ("S") or Benign ("B").

Rules:
- Output STRICT JSON only, no markdown, no prose.
- The "label" field MUST be exactly "S" or "B".
- Base your judgment on the labeled examples provided."""


def llm_predict(test_row, train_df, cats_df, model='gemma4:e4b'):
    prompt = build_prompt(test_row, train_df, cats_df)
    resp = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        format='json',
        options={'temperature': 0.1, 'num_predict': 200},
    )
    raw = resp['message']['content']
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), raw
            except Exception:
                pass
        return None, raw
```

### 5.5 批量跑测试集

```python
from tqdm import tqdm

def run_experiment(test_df, train_df, cats_df, save_path='exp_pure_llm.jsonl'):
    results = []
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df),
                             desc='Pure LLM Few-shot'):
            parsed, raw = llm_predict(row, train_df, cats_df)
            rec = {
                'idx': int(idx),
                'true_label': row['class'],
                'pred_label': parsed['label'] if parsed else None,
                'explanation': parsed.get('explanation') if parsed else None,
                'parse_ok': parsed is not None,
                'raw': raw,
            }
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return results
```

### 5.6 评估

```python
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

def evaluate(results):
    valid = [r for r in results if r['pred_label'] in ('S', 'B')]
    y_true = [r['true_label'] for r in valid]
    y_pred = [r['pred_label'] for r in valid]
    print(f"有效预测: {len(valid)}/{len(results)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("混淆矩阵 [行=真, 列=预测]:")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=['S', 'B']),
                       index=['真 S', '真 B'], columns=['预测 S', '预测 B']))

# 运行
results = run_experiment(test_df, train_df, cats_df,
                          save_path='results/predictions/exp1_pure_llm_k5.jsonl')
evaluate(results)
```

---

## 六、常见问题排查

### 6.1 LLM 输出不是 JSON

| 原因 | 解决 |
|------|------|
| 没用 `format='json'` | 加上 |
| temperature 太高 | 设 0.1 |
| Prompt 末尾没明确"Output JSON only:" | 加上 |
| 模型偶尔出错 | 用 `re.search(r'\{.*\}')` 兜底 |

### 6.2 LLM 总是输出 S 或总是输出 B

| 原因 | 解决 |
|------|------|
| Few-shot 例子不平衡（如 9B+1S） | 严格每类 K 个 |
| 例子排序集中（所有 S 在最后） | **交替排列** |
| 测试集本身不平衡 | 测试集每类 100 个 |

### 6.3 预测速度慢

| 原因 | 解决 |
|------|------|
| Prompt 太长 | 减少 K 或减少特征数量 |
| 没用 GPU | `ollama ps` 检查 |
| 每次都重新加载模型 | 第一次推理后会驻留 5 分钟 |

200 样本 CPU 模式 ≈ 15-30 分钟；GPU 模式 ≈ 2-5 分钟。

### 6.4 不同 seed 准确率波动大

K-shot 采样的具体 10 个例子对结果影响很大。**建议跑 3 次（seed=42/0/1）取均值**，方差也写进报告。

---

## 七、评估与可视化

### 7.1 必做指标

| 指标 | 公式 | 用途 |
|------|------|------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 整体准确率 |
| Precision（按类） | TP/(TP+FP) | 预测为该类的正确率 |
| Recall（按类） | TP/(TP+FN) | 实际该类的召回率 |
| F1（按类） | 2PR/(P+R) | 综合指标 |
| Macro-F1 | 两类 F1 平均 | 不平衡时更可靠 |
| Confusion Matrix | 2×2 矩阵 | 错误类型分析 |

### 7.2 必做图表

1. **K vs Accuracy 折线图**：K ∈ {0, 3, 5, 10} 各跑一次，画一条线
2. **方法对比柱状图**：RF / SVM / 纯 LLM / LLM+RAG 在 K=5 下的 Accuracy
3. **混淆矩阵热力图**：纯 LLM K=5 的 2×2 混淆矩阵
4. **错误案例特征分布**：选 5 个错分样本，分析它们的权限组合

### 7.3 报告里怎么写

```
4.2 纯 LLM Few-shot 二分类（Experiment 1）

我们使用 Gemma 4 E4B 模型，在 K-shot In-context Learning 设定下进行二分类。
具体地，从 Drebin-215 数据集中每类随机选取 K=5 个样本（共 10 个）作为
Few-shot examples，将其拼接到 Prompt 中，让模型根据这些示例对新样本进行
分类。测试集包含 200 个类别平衡的样本（每类 100 个）。

我们设置 temperature=0.1 以保证输出可复现，并使用 Ollama 的 JSON 强制
输出模式（format='json'）。完整 Prompt 模板见附录 A。

为了考察 K 的影响，我们额外跑了 K ∈ {1, 3, 10} 的实验，结果见图 X。
为了考察随机性，每个 K 值跑 3 次取均值，方差以 ± 形式标注。
```

---

## 附录 A：完整 Prompt 示例（K=5）

> 这是真实跑起来时塞给 Gemma 的内容（节选）

```
[SYSTEM]
You are an Android malware security analyst.
Given an Android app's features (permissions, API calls, intents),
classify it as Malware ("S") or Benign ("B").

Rules:
- Output STRICT JSON only, no markdown, no prose.
- The "label" field MUST be exactly "S" or "B".

[USER]
Below are 10 labeled examples from the Drebin dataset:

[Example 1] Label: B (Benign)
Permissions: INTERNET, ACCESS_NETWORK_STATE, VIBRATE, WAKE_LOCK
API: bindService, transact, ServiceConnection, attachInterface

[Example 2] Label: S (Malware)
Permissions: SEND_SMS, RECEIVE_SMS, READ_PHONE_STATE, INTERNET, READ_CONTACTS
API: SmsManager.sendTextMessage, TelephonyManager.getDeviceId, HttpGet.init
Intent: android.provider.Telephony.SMS_RECEIVED, android.intent.action.BOOT_COMPLETED

[Example 3] Label: B (Benign)
Permissions: INTERNET, ACCESS_NETWORK_STATE, READ_PHONE_STATE
API: bindService, attachInterface

[Example 4] Label: S (Malware)
Permissions: INSTALL_PACKAGES, DELETE_PACKAGES, INTERNET, SYSTEM_ALERT_WINDOW
API: DexClassLoader, System.loadLibrary, Runtime.getRuntime
Intent: android.intent.action.PACKAGE_ADDED

... (省略 Example 5-10)

---

Now classify the following new sample:

Permissions: SEND_SMS, READ_CONTACTS, INTERNET, RECEIVE_BOOT_COMPLETED
API: HttpGet.init, SmsManager, TelephonyManager.getDeviceId
Intent: android.provider.Telephony.SMS_RECEIVED

Output JSON only:
{"label": "S" or "B", "evidence": ["feature1", ...], "explanation": "<one-sentence Chinese explanation>"}

[ASSISTANT - 期望输出]
{
  "label": "S",
  "evidence": ["SEND_SMS", "TelephonyManager.getDeviceId", "SMS_RECEIVED 监听"],
  "explanation": "同时具备短信收发能力和设备指纹收集，符合短信木马典型行为模式。"
}
```

---

## 附录 B：与 RF 公平对比的代码

```python
from sklearn.ensemble import RandomForestClassifier

# 用同样的 K-shot 训练集
X_train = train_df.drop(columns=['class']).values
y_train = train_df['class'].values
X_test = test_df.drop(columns=['class']).values
y_test = test_df['class'].values

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"RF (K={K}) Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
# 在 K=5 下，RF 通常会很低（55-70%）
```

---

## 附录 C：本指南的关键 Take-aways

1. ✅ **K=5 起步**，跑 K ∈ {0, 3, 5, 10} 画曲线
2. ✅ **良性/恶意例子交替排列**，不要堆在一起
3. ✅ **测试集类别平衡**（200 个，每类 100）
4. ✅ **temperature=0.1 + format='json'**，保证稳定可复现
5. ✅ **不同 seed 跑 3 次**取均值，写进报告
6. ✅ **与 RF 公平对比**：RF 也只用 K-shot 训练
7. ✅ **80% 训练的 RF 作为"上限参考"**，不参与主对比
