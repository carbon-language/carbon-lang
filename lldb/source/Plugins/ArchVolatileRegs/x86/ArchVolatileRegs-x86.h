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
#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/ArchVolatileRegs.h"
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
    
private:
    ArchVolatileRegs_x86(llvm::Triple::ArchType cpu);        // Call CreateInstance instead.

    void initialize_regset(lldb_private::Thread& thread);

    llvm::Triple::ArchType m_cpu;
    std::set<int> m_non_volatile_regs;
};


} // namespace lldb_private

#endif // liblldb_ArchVolatileRegs_x86_h_
