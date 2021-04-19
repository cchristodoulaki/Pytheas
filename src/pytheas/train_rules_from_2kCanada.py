import pandas as pd
from dotmap import DotMap
import os, argparse, sys
from psycopg2 import connect
import pytheas

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--database", default="ground_truth_2k_canada", help="database for experimentation with ground truth")
    parser.add_argument("-u", "--user", default="christina", help="user for the database connection")
    parser.add_argument("-p", "--port", default=5532, help="port that postgresql database listens to")
    parser.add_argument("-n", "--num_processors", default = 64, type=int, help="number of processors to be used")
    parser.add_argument("-t", "--top_level_dir", default="/home/christina/OPEN_DATA_CRAWL_2018", help="path to Open Data Crawl")
    
    args = parser.parse_args(sys.argv[1:])
    num_processors=min(args.num_processors,pytheas.available_cpu_count())
    top_level_dir = args.top_level_dir

    # Database connection credentials
    db_cred = DotMap()
    db_cred.user = args.user
    db_cred.database = args.database
    db_cred.port = args.port
    db_cred.password = ''
    
    pytheas_model = pytheas.PYTHEAS()

    pytheas_model.collect_rule_activation(db_cred, num_processors, top_level_dir)


    print('\nLoading CACHED Training Data...')

    con=connect(dbname=db_cred.database, 
                user=db_cred.user, 
                host = 'localhost', 
                password=db_cred.password, 
                port = db_cred.port)

    undersampled_cell_data = pd.read_sql_query(
            sql = f"SELECT * FROM pat_data_cell_rules WHERE undersample=True", con=con)
    con.close()

    con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
    undersampled_line_data = pd.read_sql_query(
            sql = f"SELECT * FROM pat_data_line_rules WHERE undersample=True", con=con)
    con.close()

    con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
    undersampled_cell_not_data = pd.read_sql_query(
            sql = f"SELECT * FROM pat_not_data_cell_rules WHERE undersample=True", con=con)
    con.close()

    con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
    undersampled_line_not_data = pd.read_sql_query(
            sql = f"SELECT * FROM pat_not_data_line_rules WHERE undersample=True", con=con)
    con.close()

    print('\nTraining model...')
    pytheas_model.train_rules(undersampled_cell_data, 
                                undersampled_cell_not_data, 
                                undersampled_line_data, 
                                undersampled_line_not_data)

    pytheas_model.save_trained_rules()