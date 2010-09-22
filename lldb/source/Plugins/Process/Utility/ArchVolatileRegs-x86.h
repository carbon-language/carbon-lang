//===-- ArchVolatileRegs-x86.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ArchVolatileRegs_x86_h_
#define liblldb_ArchVolatileRegs_x86_h_

#include "lldb/lldb-private.h"
#include "lldb/Utility/ArchVolatileRegs.h"
#include <set>

namespace lldb_private {
    
class ArchVolatileRegs_x86 : public lldb_private::ArchVolatileRegs
{
public:

    ~ArchVolatileRegs_x86 () { }

    bool
    RegisterIsVolatile (lldb_private::Thread& thread, uint32_t regnum);

    static lldb_private::ArchVolatileRegs *
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
    ArchVolatileRegs_x86(int cpu);        // Call CreateInstance instead.

    void initialize_regset(lldb_private::Thread& thread);

    int m_cpu;
    std::set<int> m_non_volatile_regs;
};


} // namespace lldb_private

#endif // liblldb_ArchVolatileRegs_x86_h_
