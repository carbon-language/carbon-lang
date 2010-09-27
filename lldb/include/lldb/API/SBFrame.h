//===-- SBFrame.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFrame_h_
#define LLDB_SBFrame_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBValueList.h"

namespace lldb {

class SBValue;

class SBFrame
{
public:
    SBFrame ();

   ~SBFrame();

    bool
    IsValid() const;

    uint32_t
    GetFrameID () const;

    lldb::addr_t
    GetPC () const;

    bool
    SetPC (lldb::addr_t new_pc);

    lldb::addr_t
    GetSP () const;

    lldb::addr_t
    GetFP () const;

    lldb::SBAddress
    GetPCAddress () const;

    lldb::SBSymbolContext
    GetSymbolContext (uint32_t resolve_scope) const;

    lldb::SBModule
    GetModule () const;

    lldb::SBCompileUnit
    GetCompileUnit () const;

    lldb::SBFunction
    GetFunction () const;

    // Gets the deepest block that contains the frame PC
    lldb::SBBlock
    GetBlock () const;

    // Gets the lexical block that defines the stack frame. Another way to think
    // of this is it will return the block that contains all of the variables
    // for a stack frame. Inlined functions are represented as SBBlock objects
    // that have inlined function information: the name of the inlined function,
    // where it was called from. The block that is returned will be the first 
    // block at or above the block for the PC (SBFrame::GetBlock()) that defines
    // the scope of the frame. When a function contains no inlined functions,
    // this will be the top most lexical block that defines the function. 
    // When a function has inlined functions and the PC is currently
    // in one of those inlined functions, this method will return the inlined
    // block that defines this frame. If the PC isn't currently in an inlined
    // function, the lexical block that defines the function is returned.
    lldb::SBBlock
    GetFrameBlock () const;

    lldb::SBLineEntry
    GetLineEntry () const;

    lldb::SBThread
    GetThread () const;

    const char *
    Disassemble () const;

    void
    Clear();

#ifndef SWIG
    bool
    operator == (const lldb::SBFrame &rhs) const;

    bool
    operator != (const lldb::SBFrame &rhs) const;

#endif

    lldb::SBValueList
    GetVariables (bool arguments,
                  bool locals,
                  bool statics,
                  bool in_scope_only);

    lldb::SBValueList
    GetRegisters ();

    lldb::SBValue
    LookupVar (const char *var_name);

    lldb::SBValue
    LookupVarInScope (const char *var_name, const char *scope);

    bool
    GetDescription (lldb::SBStream &description);

protected:
    friend class SBValue;

    lldb_private::StackFrame *
    GetLLDBObjectPtr ();

private:
    friend class SBThread;
    friend class lldb_private::ScriptInterpreterPython;

#ifndef SWIG

    lldb_private::StackFrame *
    operator->() const;

    // Mimic shared pointer...
    lldb_private::StackFrame *
    get() const;

#endif


    SBFrame (const lldb::StackFrameSP &lldb_object_sp);

    void
    SetFrame (const lldb::StackFrameSP &lldb_object_sp);

    lldb::StackFrameSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBFrame_h_
