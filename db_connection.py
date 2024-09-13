import psycopg2
def get_db_connection():
    return psycopg2.connect(
        user="ng_read_all_data",
        password=f"[ng()r%aD]",
        host="ebmsngdev16.postgres.database.azure.com",
        port="5432",
        database="eBMSHybridMain"
    )