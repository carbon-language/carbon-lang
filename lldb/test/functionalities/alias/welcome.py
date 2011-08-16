import sys

def welcome_impl(debugger, args, stream, dict):
    stream.Printf('Hello ' + args + ', welcome to LLDB');
    return None;

def target_name_impl(debugger, args, stream, dict):
    target = debugger.GetSelectedTarget()
    file = target.GetExecutable()
    stream.Printf('Current target ' + file.GetFilename())
    if args == 'fail':
        return 'a test for error in command'
    else:
        return None