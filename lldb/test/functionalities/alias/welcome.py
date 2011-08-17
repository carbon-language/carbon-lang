import sys

def welcome_impl(debugger, args, result, dict):
    """
        Just a docstring for welcome_impl
        A command that says hello to LLDB users
    """
    result.Printf('Hello ' + args + ', welcome to LLDB');
    return None;

def target_name_impl(debugger, args, result, dict):
    target = debugger.GetSelectedTarget()
    file = target.GetExecutable()
    result.PutCString('Current target ' + file.GetFilename())
    if args == 'fail':
        return 'a test for error in command'
    else:
        return None

def print_wait_impl(debugger, args, result, dict):
    print 'Trying to do long task..';
    import time
    time.sleep(1)
    print 'Still doing long task..';
    time.sleep(1)
    result.PutCString('Done; if you saw the delays I am doing OK')
    return None