from pydantic import BaseModel
from typing import Optional, List, Any, Dict, Union

# =============== 饼图专用数据项 ===============
class PieDatasetItem(BaseModel):
    value: Union[int, float]
    name: str

# =============== 柱状图/折线图专用数据集 ===============
class BarLineDataset(BaseModel):
    label: str
    data: List[Union[int, float]]

# =============== 图表数据详情（联合类型） ===============
class ChartDataDetailBarLine(BaseModel):
    labels: List[str]
    datasets: List[BarLineDataset]

class ChartDataDetailPie(BaseModel):
    # 饼图通常不需要 labels 字段，直接 datasets 包含 name/value
    datasets: List[PieDatasetItem]

# =============== 顶层 ChartData：通过 type 区分结构 ===============
class ChartData(BaseModel):
    type: str  # "bar", "line", "pie"
    title: str
    data: Union[ChartDataDetailBarLine, ChartDataDetailPie]

    # 可选：添加 validator 确保结构匹配 type（进阶）
    # 但为简化，此处依赖逻辑层正确构造

# =============== 分析结果 ===============
class AnalysisResult(BaseModel):
    chartData: Optional[ChartData] = None
    markdownReport: str

# =============== 请求/响应 ===============
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    status: str
    sql: str
    result: Optional[AnalysisResult] = None
    message: str