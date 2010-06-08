//===-- SBBlock.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBlock_h_
#define LLDB_SBBlock_h_

#include <LLDB/SBDefines.h>

namespace lldb {

class SBBlock
{
public:

    SBBlock ();

    ~SBBlock ();

    bool
    IsValid () const;

    void
    AppendVariables (bool can_create, bool get_parent_variables, lldb_private::VariableList *var_list);

private:
    friend class SBFrame;
    friend class SBSymbolContext;

    SBBlock (lldb_private::Block *lldb_object_ptr);


    lldb_private::Block *m_lldb_object_ptr;
};


} // namespace lldb

#endif // LLDB_SBBlock_h_
