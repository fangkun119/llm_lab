import os
from enum import Enum
from typing import Optional, Tuple

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


class LLMVendorProp:
    def __init__(self, vendor_name, api_key_name,
                 default_model, default_temperature, default_base_url) -> None:
        self._vendor_name: str = vendor_name
        self._api_key_name: str = api_key_name
        self._default_model: str = default_model
        self._default_temperature: str = default_temperature
        self._default_base_url: str = default_base_url

    @property
    def vendor_name(self):
        return self._vendor_name

    @property
    def api_key_name(self):
        return self._api_key_name

    @property
    def default_model(self):
        return self._default_model

    @property
    def default_temperature(self):
        return self._default_temperature

    @property
    def default_base_url(self):
        return self._default_base_url


class LLMVendor(Enum):
    ZHIPU = 1,
    KIMI = 2


vendor_map = {
    # https://open.bigmodel.cn/pricing
    # GLM-4-0520：128K, 旗舰，0.1 元 / 千tokens
    # GLM-4-AirX: 8K fast, 极速推理，0.01 元 / 千tokens
    # GLM-4-Air: 128k，高性价比，0.001 元 / 千tokens
    # GLM-4-Long: 1M，超长输入，0.001 元 / 千tokens
    # GLM-4-Flash: 128K，低价极速，0.0001 元 / 千tokens
    LLMVendor.ZHIPU: LLMVendorProp('ZHIPU', 'ZHIPU_API_KEY', 'glm-4-air', '0', 'https://open.bigmodel.cn/api/paas/v4'),
    # moonshot-v1-8k: 它是一个长度为 8k 的模型，适用于生成短文本。
    # moonshot-v1-32k: 它是一个长度为 32k 的模型，适用于生成长文本。
    # moonshot-v1-128k: 它是一个长度为 128k 的模型，适用于生成超长文本。
    LLMVendor.KIMI: LLMVendorProp('KIMI', 'KIMI_API_KEY', 'moonshot-v1-32k', '0', 'https://api.moonshot.cn/v1')
}


class EmbeddingUtil:
    @staticmethod
    def getModel(vendor: LLMVendor) -> Optional[Embeddings]:
        vendor_prop = vendor_map.get(vendor)
        api_key = os.environ[vendor_prop.api_key_name]
        print(f"{vendor_prop.api_key_name}\t: {api_key[:5]}... ")
        if vendor == LLMVendor.ZHIPU:
            return ZhipuAIEmbeddings(api_key=api_key)
        else:
            return None


class ChatModelUtil:
    @staticmethod
    def getChatOpenAIModel(
            vendor: LLMVendor,
            temperature: Optional[int] = None,
            model: Optional[str] = None,
            base_url: Optional[str] = None) -> Optional[BaseChatModel]:
        vendor_prop = vendor_map.get(vendor)
        api_key = os.environ[vendor_prop.api_key_name]
        default_temperature = vendor_prop.default_temperature
        default_model = vendor_prop.default_model
        default_base_url = vendor_prop.default_base_url
        if vendor == LLMVendor.ZHIPU:
            # https://open.bigmodel.cn/dev/api#langchain_sdk
            return ChatOpenAI(
                temperature=default_temperature if (temperature is None) else temperature,
                model=default_model if (model is None) else model,
                openai_api_key=api_key,
                openai_api_base=default_base_url if (base_url is None) else default_base_url
            )
        if vendor == LLMVendor.KIMI:
            # https://python.langchain.com/v0.1/docs/integrations/chat/moonshot/
            # https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.moonshot.MoonshotChat.html
            return MoonshotChat(
                model=default_model if (model is None) else model,
                base_url=default_base_url if (base_url is None) else base_url,
                moonshot_api_key=api_key,
                temperature=default_temperature if (temperature is None) else temperature
            )
        else:
            return None
