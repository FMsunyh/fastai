import pydevd
pydevd.settrace('192.168.38.133', port=22, stdoutToServer=True, stderrToServer=True)

print('hello')