//===-- UnwindAssemblyInstEmulation.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UnwindAssemblyInstEmulation_h_
#define liblldb_UnwindAssemblyInstEmulation_h_

#include "lldb/lldb-private.h"
#include "lldb/Target/UnwindAssembly.h"
#include "lldb/Target/Thread.h"

class UnwindAssemblyInstEmulation : public lldb_private::UnwindAssembly
{
public:

    virtual
    ~UnwindAssemblyInstEmulation () 
    {
    }

    virtual bool
    GetNonCallSiteUnwindPlanFromAssembly (lldb_private::AddressRange& func, 
                                          lldb_private::Thread& thread, 
                                          lldb_private::UnwindPlan& unwind_plan);

    virtual bool
    GetFastUnwindPlan (lldb_private::AddressRange& func, 
                       lldb_private::Thread& thread, 
                       lldb_private::UnwindPlan &unwind_plan);

    // thread may be NULL in which case we only use the Target (e.g. if this is called pre-process-launch).
    virtual bool
    FirstNonPrologueInsn (lldb_private::AddressRange& func, 
                          lldb_private::Target& target, 
                          lldb_private::Thread* thread, 
                          lldb_private::Address& first_non_prologue_insn);

    static lldb_private::UnwindAssembly *
    CreateInstance (const lldb_private::ArchSpec &arch);


    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
private:

    // Call CreateInstance to get an instance of this class
    UnwindAssemblyInstEmulation(int cpu) : 
          lldb_private::UnwindAssembly(), m_cpu(cpu) 
    {
    }

    int m_cpu;
};

#endif // liblldb_UnwindAssemblyInstEmulation_h_
