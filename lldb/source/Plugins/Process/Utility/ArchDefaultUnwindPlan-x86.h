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
#include "lldb/Utility/ArchDefaultUnwindPlan.h"
#include "lldb/Target/Thread.h"
#include "lldb/Symbol/UnwindPlan.h"

namespace lldb_private {
    
class ArchDefaultUnwindPlan_x86 : public lldb_private::ArchDefaultUnwindPlan
{
public:

    ~ArchDefaultUnwindPlan_x86 () { }

    virtual lldb_private::UnwindPlan*
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
    
    virtual void
    GetPluginCommandHelp (const char *command, lldb_private::Stream *strm);
    
    virtual lldb_private::Error
    ExecutePluginCommand (lldb_private::Args &command, lldb_private::Stream *strm);
    
    virtual lldb_private::Log *
    EnablePluginLogging (lldb_private::Stream *strm, lldb_private::Args &command);

private:
    ArchDefaultUnwindPlan_x86(int cpu);        // Call CreateInstance instead.

    int m_cpu;
    lldb_private::UnwindPlan m_32bit_default;
    lldb_private::UnwindPlan m_64bit_default;
};


} // namespace lldb_private

#endif // liblldb_UnwindAssemblyProfiler_x86_h_
