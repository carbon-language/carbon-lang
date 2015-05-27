import lldb, sys

class WelcomeCommand(object):
    def __init__(self, debugger, session_dict):
        pass
    
    def get_short_help(self):
        return "Just a docstring for welcome_impl\nA command that says hello to LLDB users"
        
    def __call__(self, debugger, args, exe_ctx, result):
        print >>result,  ('Hello ' + args + ', welcome to LLDB');
        return None;

class TargetnameCommand(object):
    def __init__(self, debugger, session_dict):
        pass

    def __call__(self, debugger, args, exe_ctx, result):
        target = debugger.GetSelectedTarget()
        file = target.GetExecutable()
        print >>result,  ('Current target ' + file.GetFilename())
        if args == 'fail':
            result.SetError('a test for error in command')
    
    def get_flags(self):
        return lldb.eCommandRequiresTarget

def print_wait_impl(debugger, args, result, dict):
    result.SetImmediateOutputFile(sys.stdout)
    print >>result,  ('Trying to do long task..')
    import time
    time.sleep(1)
    print >>result,  ('Still doing long task..')
    time.sleep(1)
    print >>result,  ('Done; if you saw the delays I am doing OK')

def check_for_synchro(debugger, args, result, dict):
    if debugger.GetAsync() == True:
        print >>result,  ('I am running async')
    if debugger.GetAsync() == False:
        print >>result,  ('I am running sync')

def takes_exe_ctx(debugger, args, exe_ctx, result, dict):
    print >>result, str(exe_ctx.GetTarget())

