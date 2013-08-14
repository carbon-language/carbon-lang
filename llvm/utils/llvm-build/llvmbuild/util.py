import os
import sys

def _write_message(kind, message):
    program = os.path.basename(sys.argv[0])
    sys.stderr.write('%s: %s: %s\n' % (program, kind, message))

note = lambda message: _write_message('note', message)
warning = lambda message: _write_message('warning', message)
error = lambda message: _write_message('error', message)
fatal = lambda message: (_write_message('fatal error', message), sys.exit(1))

__all__ = ['note', 'warning', 'error', 'fatal']
