import lldb_shared
import six

def command(debugger, command, result, internal_dict):
	result.PutCString(six.u("hello world B"))
	return None
