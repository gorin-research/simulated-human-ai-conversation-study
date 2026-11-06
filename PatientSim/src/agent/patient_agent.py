import os
import json
import logging

from utils import file_to_string, prompt_valid_check, process_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


class PatientAgent:
    def __init__(
        self,
        patient_profile,
        backend_str="gpt4",
        backend_api_type="gpt_azure",
        prompt_dir=None,
        prompt_file=None,
        num_word_sample=3,
        cefr_type=None,
        personality_type=None,
        recall_level_type=None,
        dazed_level_type=None,
        client_params=None,
        verbose=False,
    ):
        self.prompt_dir = prompt_dir
        self.prompt_file = prompt_file
        self.backend = backend_str  # language model backend for patient agent
        self.backend_api_type = backend_api_type  # language model backend for patient agent
        self.num_word_sample = num_word_sample
        self.client_params = client_params if client_params is not None else {}
        self.verbose = verbose
        
        self.client = get_response_method(self.backend_api_type)
        self.model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend

        if verbose:
            logging.info(f"Setting patient agent with backend: {self.model} ({self.backend_api_type})")

        # Load patient profile & setting bias
        # Use triage-adapted persona files when they exist alongside the prompt file
        self.patient_profile = patient_profile

        def _load_bias(name):
            triage_path = os.path.join(self.prompt_dir, f"{name}_triage.json")
            default_path = os.path.join(self.prompt_dir, f"{name}.json")
            path = triage_path if ("triage" in (self.prompt_file or "") and os.path.exists(triage_path)) else default_path
            with open(path, "r") as f:
                return json.load(f)

        self.bias_prompt_dict = {
            "personality": _load_bias("personality_type"),
            "cefr_level": _load_bias("cefr_type"),
            "recall_level": _load_bias("recall_level_type"),
            "dazed_level": _load_bias("dazed_level_type"),
        }
        self.sentence_limit = json.load(open(os.path.join(self.prompt_dir, "sentence_length_limit.json"), "r"))

        # Set persona of patient 
        if verbose:
            logging.info(f"Setting patient profile: {patient_profile['hadm_id']} - {patient_profile['diagnosis']}")

        patient_profile['cefr_option'] = patient_profile.pop('cefr') if cefr_type is None else cefr_type
        patient_profile['personality_option'] = patient_profile.pop('personality') if personality_type is None else personality_type
        patient_profile['recall_level_option'] = patient_profile.pop('recall_level') if recall_level_type is None else recall_level_type
        patient_profile['dazed_level_option'] = patient_profile.pop('dazed_level') if dazed_level_type is None else dazed_level_type
        
        self.cefr_type = patient_profile["cefr_option"]
        self.personality_type = patient_profile["personality_option"]
        self.recall_level_type = patient_profile["recall_level_option"]
        self.dazed_level_type = patient_profile["dazed_level_option"]

        self.check_valid_argument()
        if verbose:
            logging.info(f" - CEFR Level: {patient_profile['cefr_option']}")
            logging.info(f" - Personality Type: {patient_profile['personality_option']}")
            logging.info(f" - Memory Recall Level: {patient_profile['recall_level_option']}")
            logging.info(f" - Dazed Level: {patient_profile['dazed_level_option']}")
            
        # Set CEFR bias
        cefr_levels = ["A", "B", "C"]
        current_index = cefr_levels.index(self.cefr_type)
        higher_level = cefr_levels[current_index + 1] if self.cefr_type != "C" else None

        self.patient_profile["understand_words"] = ", ".join(self.patient_profile[f"cefr_{self.cefr_type}1"].split(", ")[: self.num_word_sample])
        self.patient_profile["misunderstand_words"] = ", ".join(self.patient_profile[f"cefr_{self.cefr_type}2"].split(", ")[: self.num_word_sample])
        self.patient_profile["understand_med_words"] = ", ".join(self.patient_profile[f"med_{self.cefr_type}"].split(", ")[: self.num_word_sample])
        self.patient_profile["misunderstand_med_words"] = (
            ", ".join(self.patient_profile[f"med_{higher_level}"].split(", ")[: self.num_word_sample]) if higher_level is not None else ""
        )
        self.patient_profile["cefr"] = "\n\t\t" + "\n\t\t\t".join(self.bias_prompt_dict["cefr_level"][self.cefr_type].split("\n\t")[1:]).format(**self.patient_profile)
        self.patient_profile["personality"] = "\n\t\t"  + "\n\t\t".join(self.bias_prompt_dict["personality"][self.personality_type].split("\n\t")[1:])
        self.patient_profile["personality"] += "\n\t\tIMPORTANT: Ensure that your personality is clearly represented throughout the conversation, while allowing your emotional tone and style to vary naturally across turns." if self.personality_type != "plain" else ""
        
        self.patient_profile["memory_recall_level"] = f"{self.recall_level_type.capitalize()}\n\t\t" + "\n\t\t".join(
            self.bias_prompt_dict["recall_level"][self.recall_level_type].split("\n\t")[1:]
        )

        dazed_levels = ["high", "moderate", "normal"]
        dazed_states = ["initial", "intermediate", "later"]

        dazed_index = dazed_levels.index(self.dazed_level_type)
        dazed_description = ""

        if self.dazed_level_type != "normal":
            dazed_description += (
                f"\n\tThe patient's initial dazed level is {self.dazed_level_type}. "
                "The dazedness should gradually fade throughout the conversation as the doctor continues to reassure them. "
                "Transitions should feel smooth and natural, rather than abrupt. "
                "While the change should be subtle and progressive, the overall dazed level is expected to decrease noticeably every 4-5 turns, following the instructions for each level below."
            )

            for _dazed_index in range(dazed_index, len(dazed_levels)):
                dazed_description += f"\n\t{dazed_levels[_dazed_index].capitalize()} Dazedness ({dazed_states[_dazed_index].capitalize()} Phase)\n\t\t" + "\n\t\t".join(
                    self.bias_prompt_dict["dazed_level"][dazed_levels[_dazed_index]].split("\n\t")[1:]
                )

            dazed_description += "\n\tNote: Dazedness reflects the patient's state of confusion and inability in following the conversation, independent of their language proficiency."
        else:
            dazed_description = f"{self.dazed_level_type.capitalize()}\n\t\t" + "\n\t\t".join(self.bias_prompt_dict["dazed_level"][self.dazed_level_type].split("\n\t")[1:])

        self.patient_profile["dazed_level"] = dazed_description
        self.patient_profile["reminder"] = (
            "You should act like "
            + self.bias_prompt_dict["cefr_level"][self.cefr_type].split("\n\t")[0]
            + " You are "
            + self.bias_prompt_dict["personality"][self.personality_type].split("\n\t")[0]
            + ". Also, you "
            + self.bias_prompt_dict["recall_level"][self.recall_level_type].split("\n\t")[0].lower()
        )
        self.patient_profile["reminder"] += " " + self.bias_prompt_dict["dazed_level"][self.dazed_level_type].split("\n\t")[0]
        self.patient_profile["sent_limit"] = self.sentence_limit[self.personality_type] if self.personality_type is not None else "3"

        # Load prompt text file
        prompt_file = self.prompt_file
        if self.patient_profile["diagnosis"] == "Urinary tract infection":
            prompt_file += "_uti"
        self.system_prompt_text = file_to_string(os.path.join(self.prompt_dir, prompt_file + ".txt"))

        # Set gt diagnosis labels
        self.diagnosis = patient_profile["diagnosis"]
        self.reset()
        if verbose:
            logging.info(f" - Prompt file: {self.prompt_file}")
            logging.info(f" - System prompt: {self.system_prompt}")

    
    def check_valid_argument(self):
        assert self.cefr_type in list(self.bias_prompt_dict["cefr_level"].keys()), f"Invalid CEFR type: {self.cefr_type}"
        assert self.personality_type in list(self.bias_prompt_dict["personality"].keys()), f"Invalid personality type: {self.personality_type}"
        assert self.recall_level_type in list(self.bias_prompt_dict["recall_level"].keys()), f"Invalid recall level type: {self.recall_level_type}"
        assert self.dazed_level_type in list(self.bias_prompt_dict["dazed_level"].keys()), f"Invalid dazed level type: {self.dazed_level_type}"
    
    def set_system_prompt(self) -> None:
        self.system_prompt = self.system_prompt_text.format(**self.patient_profile)
        prompt_valid_check(self.system_prompt, self.patient_profile)

    def reset(self) -> None:
        self.set_system_prompt()
        system_message = {"role": "system", "content": self.system_prompt}
        self.messages = [system_message]
        self.token_log = {"prompt_tokens": [], "completion_tokens": [], "total_tokens": [], "extra_info": {}}

    def log_token_usage(self, response) -> None:
        token_usage = get_token_log(response)
        self.token_log["prompt_tokens"].append(token_usage["prompt_tokens"])
        self.token_log["completion_tokens"].append(token_usage["completion_tokens"])
        self.token_log["total_tokens"].append(token_usage["total_tokens"])
        if "extra_info" in token_usage:
            for key, value in token_usage["extra_info"].items():
                if key not in self.token_log["extra_info"]:
                    self.token_log["extra_info"][key] = [value]
                else:
                    self.token_log["extra_info"][key].append(value)

    def inference(self, question) -> str:
        answer = str()
        self.messages.append({"role": "user", "content": f"{question}"})
        response = self.client(self.messages, model=self.model, **self.client_params)
        answer = get_answer(response)
        answer = process_string(answer)
        self.log_token_usage(response)
        self.messages.append({"role": "assistant", "content": f"{answer}"})
        return answer
