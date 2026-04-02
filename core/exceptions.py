class ResearchCopilotError(Exception):
    """项目基础异常。"""


class ConfigurationError(ResearchCopilotError):
    """配置异常。"""


class LLMClientError(ResearchCopilotError):
    """模型调用异常。"""


class WorkflowError(ResearchCopilotError):
    """工作流异常。"""
