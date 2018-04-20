import pymysql.cursors
# from model.LoadData import read_json

# Function return a connection.


def getConnection(server_data):
    """
    parameters already fixed inside the function
    :return: the connection to the server
    """
    connection = pymysql.connect(charset='utf8', cursorclass=pymysql.cursors.DictCursor, **server_data)
    return connection


