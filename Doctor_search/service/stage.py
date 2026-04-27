# insert data in stages table
def create_table(cursor, conn):
    cursor.execute('''CREATE TABLE stages
      (id INT PRIMARY KEY NOT NULL,
      name TEXT NOT NULL);''')
    
    print("Table created successfully")

    conn.commit()

# insert data in stages table
def insert_stage(cursor, conn):
    # cursor.execute("INSERT INTO stages (id, name) VALUES (1, 'greeting')")
    cursor.execute("INSERT INTO stages (id, name) VALUES (2, 'name')")
    cursor.execute("INSERT INTO stages (id, name) VALUES (3, 'age')")
    cursor.execute("INSERT INTO stages (id, name) VALUES (4, 'date_of_birth')")

    print("Stage records created successfully")

    conn.commit()

# fetch data from stages table
def fetch_stage(cursor, conn):
    cursor.execute("SELECT id, name from stages")
    rows = cursor.fetchall()
    print(rows)

    # for row in rows:
    #     print("ID = ", row[0])
    #     print("NAME = ", row[1], "\n")

    print("Operation done success")
