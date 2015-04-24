//===-- SWIG Interface for SBExecutionContext ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
class SBExecutionContext
{
public:
    SBExecutionContext();
    
    SBExecutionContext (const lldb::SBExecutionContext &rhs);
    
    SBExecutionContext (const lldb::SBTarget &target);
    
    SBExecutionContext (const lldb::SBProcess &process);
    
    SBExecutionContext (lldb::SBThread thread); // can't be a const& because SBThread::get() isn't itself a const function
    
    SBExecutionContext (const lldb::SBFrame &frame);
    
    ~SBExecutionContext();
    
    SBTarget
    GetTarget () const;
    
    SBProcess
    GetProcess () const;
    
    SBThread
    GetThread () const;
    
    SBFrame
    GetFrame () const;
    
    %pythoncode %{
        __swig_getmethods__["target"] = GetTarget
        if _newclass: target = property(GetTarget, None, doc='''A read only property that returns the same result as GetTarget().''')

        __swig_getmethods__["process"] = GetProcess
        if _newclass: process = property(GetProcess, None, doc='''A read only property that returns the same result as GetProcess().''')

        __swig_getmethods__["thread"] = GetThread
        if _newclass: thread = property(GetThread, None, doc='''A read only property that returns the same result as GetThread().''')
            
        __swig_getmethods__["frame"] = GetFrame
        if _newclass: frame = property(GetFrame, None, doc='''A read only property that returns the same result as GetFrame().''')
    %}

};
    
} // namespace lldb
