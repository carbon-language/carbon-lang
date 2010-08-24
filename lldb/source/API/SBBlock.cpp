//===-- SBBlock.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBBlock.h"
#include "lldb/Symbol/Block.h"

using namespace lldb;


SBBlock::SBBlock () :
    m_opaque_ptr (NULL)
{
}

SBBlock::SBBlock (lldb_private::Block *lldb_object_ptr) :
    m_opaque_ptr (lldb_object_ptr)
{
}

SBBlock::~SBBlock ()
{
    m_opaque_ptr = NULL;
}

bool
SBBlock::IsValid () const
{
    return m_opaque_ptr != NULL;
}

void
SBBlock::AppendVariables (bool can_create, bool get_parent_variables, lldb_private::VariableList *var_list)
{
    if (IsValid())
    {
        bool show_inline = true;
        m_opaque_ptr->AppendVariables (can_create, get_parent_variables, show_inline, var_list);
    }
}



