import inspect
import os
import sys

def _write_message(kind, message):
    # Get the file/line where this message was generated.
    f = inspect.currentframe()
    # Step out of _write_message, and then out of wrapper.
    f = f.f_back.f_back
    file,line,_,_,_ = inspect.getframeinfo(f)
    location = '%s:%d' % (os.path.basename(file), line)

    print >>sys.stderr, '%s: %s: %s' % (location, kind, message)

note = lambda message: _write_message('note', message)
warning = lambda message: _write_message('warning', message)
error = lambda message: _write_message('error', message)
fatal = lambda message: (_write_message('fatal error', message), sys.exit(1))

__all__ = ['note', 'warning', 'error', 'fatal']
