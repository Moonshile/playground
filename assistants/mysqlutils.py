import mysql.connector
import json


with open('.key/assistants.json', 'r') as f:
    assistants_conf = json.load(f)



def assistants_conn():
    """
    Create a MySQL connection.
    """
    conn = mysql.connector.connect(
        host=assistants_conf['host'],
        port=assistants_conf['port'],
        user=assistants_conf['user'],
        password=assistants_conf['password'],
        database=assistants_conf['database']
    )
    return conn



if __name__ == "__main__":
    conn = assistants_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ai_assistant_message")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    cursor.close()
    conn.close()

