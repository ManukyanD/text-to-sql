import json

from src.dataset.token_preprocessor import TokenPreprocessor
from src.natsql.natsql2sql import Args
from src.natsql.natsql_parser import create_sql_from_natSQL


class OutputPostprocessor:
    def __init__(self, args):
        self.token_processor = TokenPreprocessor()
        self.args = args

    def get_output(self, output, db_id):
        output = self.token_processor.postprocess(output)
        output = self.convert_to_sql(output, db_id)
        return output

    def convert_to_sql(self, pred, db_id):
        # args = self.construct_hyper_param()
        natsql2sql_args = Args()
        natsql2sql_args.not_infer_group = True  # verified
        tables = json.load(open(self.args.tables_path))
        table_dict = {}
        for t in tables:
            table_dict[t["db_id"]] = t

        try:
            query, _, __ = create_sql_from_natSQL(pred, db_id,
                                                  "data/spider/database/" + db_id + "/" + db_id + ".sqlite",
                                                  table_dict[db_id], None,
                                                  remove_values=False, remove_groupby_from_natsql=False,
                                                  args=natsql2sql_args)
            return query.strip()
        except:
            return "None"
