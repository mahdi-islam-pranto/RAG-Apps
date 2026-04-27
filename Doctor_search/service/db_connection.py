import psycopg2
import os, sys
from configuration.postgress_config import get_postgres_db_config

# execute db connection + operation
def db_executed(operation, sql = None, values = None):
    conn = None
    db_config = get_postgres_db_config()

    #  get db values
    host = db_config["host"]
    port = db_config["port"]
    database = db_config["database"]
    user = db_config["user"]
    password = db_config["password"]

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        # print("Connection to the database established successfully.")

        #3 create a cursor object
        cursor = conn.cursor()

        if operation == "insert":
            cursor = conn.cursor()
            cursor.execute(sql, values)   # ← THIS prevents SQL injection
            conn.commit()
            # print("Record inserted successfully")


    except Exception as e:
        print(f"An error occurred while connecting to the database: {e}")
    finally:
        if cursor:
            cursor.close()
            # print("Database cursor closed.")

        if conn:
            conn.close()
            # print("Database connection closed.")    