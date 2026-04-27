import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

def get_postgres_db_config():
    # get db values from env variables
    db_config = {
        "host": os.getenv("P_DB_HOST", "default_localhost"),
        "port": os.getenv("P_DB_PORT"),
        "database": os.getenv("P_DB_NAME"),
        "user": os.getenv("P_DB_USER"),
        "password": os.getenv("P_DB_PASSWORD")
    }

    return db_config



# print(db_config)


# test the connection
# if __name__ == "__main__":
#     conn = get_postgres_db_connection()
#     print(conn)