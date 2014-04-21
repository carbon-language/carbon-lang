import os
import os.path

def _get_lldb_gdbserver_from_lldb(lldb_exe):
    lldb_gdbserver = os.path.join(os.path.dirname(lldb_exe), "lldb-gdbserver")
    if os.path.exists(lldb_gdbserver):
        return lldb_gdbserver
    else:
        return None

def get_lldb_gdbserver_exe():
    # check for --lldb-gdbserver='{some-path}' in args
    lldb_exe = os.environ["LLDB_EXEC"]
    if not lldb_exe:
        return None
    else:
        return _get_lldb_gdbserver_from_lldb(lldb_exe)

if __name__ == '__main__':
    import sys
    exe = get_lldb_gdbserver_exe()
    if exe:
        print "lldb-gdbserver exe at: {}".format(exe)
    else:
        print "lldb-gdbserver not specified"
