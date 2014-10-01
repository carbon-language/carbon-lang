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
};
    
} // namespace lldb
