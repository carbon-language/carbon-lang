//===-- CommandContext.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandContext_h_
#define liblldb_CommandContext_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/ValueObjectList.h"

namespace lldb_private {

class CommandContext
{
public:
    CommandContext ();

    ~CommandContext ();

    void
    Update (ExecutionContext *override_context = NULL);

    Target *
    GetTarget();

    ExecutionContext &
    GetExecutionContext();

private:
    ExecutionContext m_exe_ctx;
};

} // namespace lldb_private

#endif // liblldb_CommandContext_h_
