# Text2SQL

A [FLAN-T5](https://huggingface.co/google/flan-t5-base) fine tuned on spider dataset for text to SQL conversion. During fine-tuning 
 the input was preprocessed using one of the techniques proposed in [Improving Generalization in Language Model-Based Text-to-SQL Semantic Parsing: Two Simple Semantic Boundary-Based Techniques](https://arxiv.org/abs/2305.17378) paper.
In particular, they propose to split the database schema and SQL query components in such a way that
the tokenization of the input aligns with the words' semantics. Besides that, Natural SQL was used as an intermediate representation. The [TokenPreprocessor](https://github.com/ManukyanD/text-to-sql/blob/main/src/dataset/token_preprocessor.py) as well as the NatSQL to SQL conversion codes (in [natsql](https://github.com/ManukyanD/text-to-sql/tree/main/src/natsql) directory) were 
taken from the GitHub [repo](https://github.com/Dakingrai/ood-generalization-semantic-boundary-techniques) of the paper.
Additionally, train_spider-natsql.json and tables_for_natsql.json (both in [data](https://github.com/ManukyanD/text-to-sql/tree/main/data) directory)
were taken from the GitHub [repo](https://github.com/ygan/NatSQL) of the 
[Natural SQL: Making SQL Easier to Infer from Natural Language Specifications](https://arxiv.org/abs/2109.05153) paper.
The former is used to get the NatSQL labels during training. The latter is used during inference to provide database schema for NatSQL to SQL conversion.
Additional tokens were added for < and <= symbols as T5 tokenizer does not have them. Besides a token was added for 'id', because it happens in every input example but the T5 tokenizer 
splits it into separate tokens 'i' and 'd'. 

The evaluation is performed using the official evaluation scripts of Spider dataset (taken from [here](https://github.com/taoyds/test-suite-sql-eval)).
Two fine-tuning strategies were tried, namely full fine-tuning and LoRA. Full fine-tuning resulted in better performance, so that one was published to HuggingFace.
The model accuracy values are
 - Execution accuracy - 64.5 %
 - Exact matching accuracy - 57 %