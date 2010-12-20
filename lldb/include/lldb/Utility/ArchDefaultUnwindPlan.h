//===---------------------ArchDefaultUnwindPlan.h ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_ArchDefaultUnwindPlan_h_
#define utility_ArchDefaultUnwindPlan_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

class ArchDefaultUnwindPlan :
   public PluginInterface
{
public:

    virtual
    ~ArchDefaultUnwindPlan();

    virtual lldb_private::UnwindPlan*
    GetArchDefaultUnwindPlan (Thread& thread, Address current_pc) = 0;

    static ArchDefaultUnwindPlan*
    FindPlugin (const ArchSpec &arch);

protected:
    ArchDefaultUnwindPlan();
private:
    DISALLOW_COPY_AND_ASSIGN (ArchDefaultUnwindPlan);
};

} // namespace lldb_private

#endif //utility_ArchDefaultUnwindPlan_h_


