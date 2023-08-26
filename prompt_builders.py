# Prompt builders for different types of prompt models
from abc import ABC, abstractmethod


class Llama2ChatPrompt():
    """Llama2 Chat prompt https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    def __init__(self, system_msg: str = None) -> None:
        """Initiate Llama2 prompt builder

        Args:
            system_msg (str): System message. If None, then message from the arxiv:2307.09288 paper is used
        """
        self.system_msg = system_msg if system_msg is not None else "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        # {0} - system message
        # {1} - user prompt
        self.prompt_template = "<s>[INST] <<SYS>>\n{0}\n<</SYS>>\n\n{1} [/INST]"
        # {0} - raw last model reply
        # {1} - user prompt
        self.prompt_multi_template = "{0}<s>[INST] {1} [/INST]"

    def make(self, user_text: str, model_reply: str = None) -> str:
        """Make a prompt from a user text

        Args:
            user_text (str): User prompt
            model_reply (str, optional): Include last model reply. If None - construct single-turn message

        Returns:
            str: Llama2 prompt string
        """
        if model_reply is None:
            return self.prompt_template.format(self.system_msg, user_text)
        else:
            return self.prompt_multi_template.format(model_reply, user_text)
        
    def refine_output(self, raw_model_reply: str) -> str:
        """Get model reply only from the raw model output

        Args:
            raw_model_reply (str): Raw Llama 2 output

        Returns:
            str: Refined model output
        """
        return raw_model_reply.split("[/INST]")[-1].strip().split("</s>")[0]


class CodeLlamaPrompt():
    """CodeLlama prompt builder - https://huggingface.co/blog/codellama#code-completion
    """
    def __init__(self, system_msg: str = None):
        # {0} - system message (optional)
        # {1} - user prompt
        if system_msg is None:
            self.prompt_template = "<s>[INST] {0} [/INST]"
        else:
            self.prompt_template = "<s>[INST] <<SYS>>\n{0}\n<</SYS>>\n\n{1} [/INST]"

        # {0} - raw last model reply
        # {1} - user prompt
        self.prompt_multi_template = "{0}<s>[INST] {1} [/INST]\n"
        
        self.system_msg = system_msg
    
    def make(self, user_text: str, model_reply: str = None) -> str:
        """Make a prompt from a user text or code

        Args:
            user_text (str): User prompt
            model_reply (str, optional): Include last model reply. If None - construct single-turn/first message

        Returns:
            str: CodeLlama prompt string
        """
        if model_reply is None:
            return self.prompt_template.format(self.system_msg, user_text)
        else:
            return self.prompt_multi_template(model_reply, user_text)
        
    def refine_output(raw_model_reply: str) -> str:
        """Get model reply only from the raw model output

        Args:
            raw_model_reply (str): Raw CodeLlama output

        Returns:
            str: Refined model output
        """
        return raw_model_reply.split("[/INST]")[-1].strip().split("</s>")[0]
        

if __name__ == "__main__":
    prompt_builder = Llama2ChatPrompt("SYS_MSG")
    print(prompt_builder.make("USR_MSG_1", None))
    print("\n---\n")
    print(prompt_builder.make("USR_MSG_2", prompt_builder.make("USR_MSG_1", None) + " ML_REPLY_1 </s>"))
    print("\n---\n")
    
    mdl_reply = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Assistant is an expert statistician.

<</SYS>>

Sampling methods in statistics are natural and artificial. The former method means that all sample elements have been formed outside of the researcher control.
Artificial sampling methods further divide between representative and discriminatory. The latter method means that all sample elements must follow some kind of criteria to be included or being excluded from the population.
Representative sampling method means that all sample elements reflect on the properties of the population.
Your task is to identify which sampling type from described above have been used in the text below. Explain your decision.

---
Data Source

Data were obtained from the Corporate Data Warehouse (CDW) of the Veterans Health Administration (VHA), the largest integrated health system in the United States(13). Inpatient, outpatient, demographic, pharmacy, and vital sign files for the fiscal years of 2008–2015 were obtained. The study was approved by the Central Arkansas Veterans Healthcare System Institutional Review Board. The aims and general analytic approach were pre-specified in the application to the Institutional Review Board as well as in a grant application to the National Institute On Drug Abuse; however, these aims and analysis plan were not registered in a publicly available trial registry prior to executing the study so the results could be considered exploratory.

Study Design and Subjects

Using a retrospective cohort study design, Veterans with chronic, non-cancer pain (CNCP) were identified. CNCP was defined as having at least one diagnosis for one of the following 5 major conditions: arthritis, back pain, neck pain, neuropathic pain, or headache/migraine from 10/1/2008 to 9/30/2015(14). Among those with CNCP, Veterans were further required to be to receive at least a 90 days’ supply of non-parenteral opioids without a 30 day or more gap in supply within two consecutive 180 day periods(15). The first period served as the baseline period, and the second was used to determine if the Veteran escalated their dose or maintained their dose. Opioids were defined by the VA Drug Class Code CN101 corresponding to ‘Opioid Analgesics’ (Appendix 1).
---
 [/INST]  The study utilizes an artificial sampling method, specifically a representative sampling method. The study aims to identify Veterans with chronic, non-cancer pain (CNCP) and further filters this population to include only those who receive a 90-day supply of non-parenteral opioids without a 30-day gap in supply within two consecutive 180-day periods. This selection process ensures that the sample reflects the properties of the population, making it a representative sample.

In representative sampling, the goal is to select a sample that accurately represents the target population's characteristics. By using specific inclusion and exclusion criteria, the study ensures that the sampled population has the same properties as the larger population of Veterans with CNCP. For instance, the requirement of a 90-day supply of non-parenteral opioids without a 30-day gap in supply within two consecutive 180-day periods helps to ensure that the sampled population has similar patterns of opioid use as the larger population.

Therefore, based on the given description of the sampling method used in the study, it falls under the category of representative sampling, which aligns with the principles of good statistical practice.</s>"""
    # mdl_reply_refined = mdl_reply.split("[/INST]")[-1].strip().split("</s>")[0]
    mdl_reply_refined = prompt_builder.refine_output(mdl_reply)
    print(mdl_reply_refined)
