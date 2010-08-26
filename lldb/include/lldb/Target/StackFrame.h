//===-- StackFrame.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StackFrame_h_
#define liblldb_StackFrame_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackID.h"

namespace lldb_private {

class StackFrame :
    public ExecutionContextScope
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StackFrame (lldb::user_id_t frame_idx, lldb::user_id_t concrete_frame_idx, Thread &thread, lldb::addr_t cfa, lldb::addr_t pc, const SymbolContext *sc_ptr);
    StackFrame (lldb::user_id_t frame_idx, lldb::user_id_t concrete_frame_idx, Thread &thread, const lldb::RegisterContextSP &reg_context_sp, lldb::addr_t cfa, lldb::addr_t pc, const SymbolContext *sc_ptr);
    StackFrame (lldb::user_id_t frame_idx, lldb::user_id_t concrete_frame_idx, Thread &thread, const lldb::RegisterContextSP &reg_context_sp, lldb::addr_t cfa, const Address& pc, const SymbolContext *sc_ptr);
    virtual ~StackFrame ();

    Thread &
    GetThread()
    { return m_thread; }

    const Thread &
    GetThread() const
    { return m_thread; }

    StackID&
    GetStackID();

    Address&
    GetFrameCodeAddress();
    
    void
    ChangePC (lldb::addr_t pc);

    const SymbolContext&
    GetSymbolContext (uint32_t resolve_scope);

    bool
    GetFrameBaseValue(Scalar &value, Error *error_ptr);

    RegisterContext *
    GetRegisterContext ();

    const lldb::RegisterContextSP &
    GetRegisterContextSP () const
    {
        return m_reg_context_sp;
    }

    VariableList *
    GetVariableList ();

    bool
    HasDebugInformation ();

    ValueObjectList &
    GetValueObjectList();

    const char *
    Disassemble ();

    void
    Dump (Stream *strm, bool show_frame_index);

    uint32_t
    GetFrameIndex () const
    {
        return m_frame_index;
    }

    uint32_t
    GetConcreteFrameIndex () const
    {
        return m_concrete_frame_index;
    }
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual Target *
    CalculateTarget ();

    virtual Process *
    CalculateProcess ();

    virtual Thread *
    CalculateThread ();

    virtual StackFrame *
    CalculateStackFrame ();

    virtual void
    Calculate (ExecutionContext &exe_ctx);

protected:
    friend class StackFrameList;

    //------------------------------------------------------------------
    // Classes that inherit from StackFrame can see and modify these
    //------------------------------------------------------------------
    void
    SetInlineBlockID (lldb::user_id_t inline_block_id)
    {
        m_id.SetInlineBlockID(inline_block_id);
    }

private:
    //------------------------------------------------------------------
    // For StackFrame only
    //------------------------------------------------------------------
    Thread &m_thread;
    uint32_t m_frame_index;
    uint32_t m_concrete_frame_index;
    lldb::RegisterContextSP m_reg_context_sp;
    StackID m_id;
    Address m_pc;   // PC as a section/offset address
    SymbolContext   m_sc;
    Flags m_flags;
    Scalar m_frame_base;
    Error m_frame_base_error;
    lldb::VariableListSP m_variable_list_sp;
    ValueObjectList m_value_object_list;
    StreamString m_disassembly;
    DISALLOW_COPY_AND_ASSIGN (StackFrame);
};

} // namespace lldb_private

#endif  // liblldb_StackFrame_h_
