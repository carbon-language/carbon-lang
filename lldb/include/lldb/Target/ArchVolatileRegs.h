//===-- ArchVolatileRegs.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_ArchVolatileRegs_h_
#define utility_ArchVolatileRegs_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

class ArchVolatileRegs :
   public PluginInterface
{
public:

    virtual
    ~ArchVolatileRegs();

    // Given a register number (in the eRegisterKindLLDB register numbering 
    // scheme), returns true if the register is defined to be "volatile" in
    // this architecture -- that is, a function is not required to preserve
    // the contents of the register.  
    // If r8 is defined to be volatile, it means that a function can put 
    // values in that register without saving the previous contents.
    // If r8 is defined to be non-volatile (preseved), a function must save
    // the value in the register before it is used.

    // The thread reference is needed to get a RegisterContext to look up by
    // register names.  

    virtual bool
    RegisterIsVolatile (lldb_private::Thread& thread, uint32_t regnum) = 0;

    static ArchVolatileRegs*
    FindPlugin (const ArchSpec &arch);

protected:
    ArchVolatileRegs();
private:
    DISALLOW_COPY_AND_ASSIGN (ArchVolatileRegs);
};

} // namespace lldb_private

#endif //utility_ArchVolatileRegs_h_

