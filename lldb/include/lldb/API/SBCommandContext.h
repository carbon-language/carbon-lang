//===-- SBCommandContext.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandContext_h_
#define LLDB_SBCommandContext_h_


#include <LLDB/SBDefines.h>

namespace lldb {

class SBCommandContext
{
public:

    SBCommandContext (lldb_private::CommandContext *lldb_object);

    ~SBCommandContext ();

    bool
    IsValid () const;

private:

    lldb_private::CommandContext *m_lldb_object;
};

} // namespace lldb

#endif // LLDB_SBCommandContext_h_
