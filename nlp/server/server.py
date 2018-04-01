import sys
import socket
from _thread import start_new_thread

from nlp.search.search import Search

HOST = ''   # All avaliable interfaces
PORT = 8888  # Arbitrary non-priveliged port


def launch_socket():
    """
    Creates and binds a server socket to listen in on the host and port.

    Returns:
    --------
    s: socket object
        The binded socket that listenes to the local host and port.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    # Bind socket to local host and port
    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('Bind failed. Error Code : ' +
              str(msg[0]) + ' Message ' + str(msg[1]))
        sys.exit()
    print('Socket bind complete.')

    # Start listening on socket
    s.listen(10)
    print('Socket now listening...')

    return s


def client_thread(conn):
    """Handles incoming connections. Useful for creating new threads."""
    # Sending message to connected client
    conn.send('Welcome to the server. Type something and hit enter\n'.encode(
        'utf-8'))  # Send only takes string

    # Infinite loop so that function does not terminate and thread does not
    # end.
    while True:
        # Receiving from client and decode (for processing)
        data = conn.recv(1024).decode('utf-8')
        reply = Search(data).to_json()  # Process requested search query
        if not data:
            break
        conn.sendall(reply.encode('utf-8'))

    # If connection is terminated
    conn.close()


if __name__ == "__main__":
    # Launch socket server
    s = launch_socket()

    # Keep talking with the client
    while 1:
        # Wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))

        # Start new thread: takes 1st argument as a function to be run, second
        # is the tuple of arguments to the function.
        start_new_thread(client_thread, (conn,))

    s.close()
