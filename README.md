# CL-417A-Final-Project-Trans-Tokenization

### Installation:
  
    mamba env create -f environment.yaml

### Description:
  This code analyzes the differences between two mBART-large-50 models fine-tuned on a parallel corpus of Spanish-Nahuatl (specifically, the Thermostatic/Axolotl_Classical_Nahuatl_INALI dataset) to determine whether or not trans-tokenization improves the translational ability of mBART-large-50 from Spanish to Nahuatl. It performs trans-tokenization on one model prior to fine-tuning, and the other is left as the control or vanilla model. Perplexity and chrF are calculated, with perplexity run on a monolingual Spanish corpus (weshamhadd14/spanishNLP), and exact calques and loops (3 or more repetitions in predictions that do not exist in the labels) are recorded. A count is also performed on certain grammatical features of Nahuatl, including 7 adverbial affixes, 6 prepositions, and 2 pronouns. These results are then plotted.

### Running:
  Run create_vanilla_ft.py, create_transtokenizer_ft.py, and generate_predictions.py, then the *test_type*_test.py files (perplexity_test.py, chrF_test.py, repetition_test.py, calques_test.py, grammatical_features_count_test.py). In generate_predictions.py, beam=3 can be changed to beam=1 or beam=9, to create 3 separate csv files, and manually use a hierarchy of 3 > 1 > 9 to recreate the smooth generation mentioned in the study.   

### References:
  Dyer, C., Chahuneau, V., & Smith, N. (2013). A Simple, Fast, and Effective Reparameterization of IBM Model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, (pp. 644–648), Atlanta, Georgia. Association for Computational Linguistics. https://aclanthology.org/N13-1073/
  
  Remy, F., Delobelle, P., Avetisyan, H., Khabibullina, A., de Lhoneux, M., & Demeester, T. (2024). Trans-tokenization and cross-lingual vocabulary transfers: Language adaptation of LLMs for low-resource NLP. In Proceedings of the 1st Conference on Language Modeling (COLM 2024), (pp. 1-28). arXiv:2408.04303. https://arxiv.org/abs/2408.04303      
  
  Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Gan, A. (2020). Multilingual translation with extensible multilingual pretraining and finetuning. arXiv preprint arXiv:2008.00401. https://doi.org/10.48550/arXiv.2008.00401
  
  Thermostatic. (2023). Axolotol Classical Nahuatl INALI [Dataset]. Hugging Face. https://huggingface.co/datasets/Thermostatic/Axolotl_Classical_Nahuatl_INALI
  
   Wesamhaddad14. (n.d.). Large Spanish Corpus [Dataset]. Hugging Face. https://huggingface.co/datasets/josecannete/large_spanish_corpus

### Citation:
  
    Godfrey, M. (2025). Trans-tokenization as an Instrument to Improve LLM Modeling of Nahuatl.
