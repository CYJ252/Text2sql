from pydantic import BaseModel
from typing import Optional, List, Any, Dict, Union

# 1. 图表数据结构 (对应 ECharts/Chart.js 结构)
class ChartDataset(BaseModel):
    label: str
    data: List[Union[int, float]]

class ChartDataDetail(BaseModel):
    labels: List[str]
    datasets: List[ChartDataset]

class ChartData(BaseModel):
    type: str  # bar, line, pie, scatter
    title: str
    data: ChartDataDetail

# 2. 核心结果结构 (包含图表 + Markdown报告)
class AnalysisResult(BaseModel):
    chartData: Optional[ChartData] = None  # 如果不需要图表，允许为 null
    markdownReport: str


# ====== 请求/响应模型 ======
class QueryRequest(BaseModel):
    question: str

# 3. 顶层 API 响应结构
class QueryResponse(BaseModel):
    status: str  # success / error
    sql: str
    result: Optional[AnalysisResult] = None
    message: str = ""