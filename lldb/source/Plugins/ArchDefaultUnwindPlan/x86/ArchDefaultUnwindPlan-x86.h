//===-- ArchDefaultUnwindPlan-x86.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ArchDefaultUnwindPlan_x86_h_
#define liblldb_ArchDefaultUnwindPlan_x86_h_

#include "lldb/lldb-private.h"
#include "lldb/Target/ArchDefaultUnwindPlan.h"
#include "lldb/Target/Thread.h"
#include "lldb/Symbol/UnwindPlan.h"

namespace lldb_private {
    
class ArchDefaultUnwindPlan_x86_64 : public lldb_private::ArchDefaultUnwindPlan
{
public:

    ~ArchDefaultUnwindPlan_x86_64 () { }

    virtual lldb::UnwindPlanSP
    GetArchDefaultUnwindPlan (Thread& thread, Address current_pc);

    static lldb_private::ArchDefaultUnwindPlan *
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
    ArchDefaultUnwindPlan_x86_64();        // Call CreateInstance instead.

    lldb::UnwindPlanSP m_unwind_plan_sp;
};

class ArchDefaultUnwindPlan_i386 : public lldb_private::ArchDefaultUnwindPlan
{
public:

    ~ArchDefaultUnwindPlan_i386 () { }

    virtual lldb::UnwindPlanSP
    GetArchDefaultUnwindPlan (Thread& thread, Address current_pc);

    static lldb_private::ArchDefaultUnwindPlan *
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
    ArchDefaultUnwindPlan_i386();        // Call CreateInstance instead.

    lldb::UnwindPlanSP m_unwind_plan_sp;
};


} // namespace lldb_private

#endif // liblldb_UnwindAssembly_x86_h_
