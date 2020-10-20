import sys, argparse
from dotmap import DotMap 
import json
import pytheas 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-w", "--weights", default = "pytheas/trained_rules.json")#, description="Filepath to pre-trained rule weights")
    parser.add_argument("-i", "--input_portals", nargs="*", default=[])
    parser.add_argument("-p", "--NPROC", type = int, default = pytheas.available_cpu_count())
    parser.add_argument("-m", "--max_lines",  type = int, default = 10000)
    parser.add_argument("-c", "--db_cred_file", default="database_credentials.json")
    args = parser.parse_args(sys.argv[1:])

    with open(args.db_cred_file) as f:
        credentials = json.load(f)

    args = parser.parse_args(sys.argv[1:])

    # Database connection credentials 
    db_cred = DotMap() 
    db_cred.user = credentials["user"]
    db_cred.password = credentials["password"]
    db_cred.database = credentials["ground_truth_db"]
    db_cred.opendata_database = credentials["profile_db"]
    db_cred.port = credentials["port"]

    Pytheas = pytheas.PYTHEAS()
    Pytheas.load_weights(args.weights)


    pytheas.process_endpoint(args.portals,
                            db_cred, 
                            NPROC=args.NPROC,
                            max_lines = args.max_lines)