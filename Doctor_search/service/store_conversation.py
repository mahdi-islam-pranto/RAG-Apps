# store conversation in the database
# from db_connection import de_executed
from service.db_connection import db_executed


def store_conversation(data):
    values = (
        data["thread_id"],
        data["user_id"],
        data["message"],
        data["stage_id"]
    )

    # print("values: ", values)

    sql = """
        INSERT INTO conversation (thread_id, user_id, message, stage_id)
        VALUES (%s, %s, %s, %s)
    """

    db_executed("insert", sql, values)

