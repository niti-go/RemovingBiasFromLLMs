I am recreating some results from the [BiasDPO paper](https://arxiv.org/abs/2407.13928) so I can learn more about LLMs. 
The BiasDPO paper introduces a method for Direct Preference Optimization (DPO) fine-tuning using the BiasDPO dataset. 
This dataset contains LLM prompts paired with preferred responses, which are more aligned with inclusive human values,
and less preferred responses, which may amplify harmful stereotypes related to gender, race, and religion.
The paper fine-tunes the Microsoft Phi-2 model to reduce the likelihood of biased completions.

I learned about DPO in my Intro to Reinforcement Learning course last year. 
By recreating some of BiasDPO's experiments, I aim to gain familiarity using DPO in practice and learn how to evaluate a fine-tuned model.
This work is in progress!

