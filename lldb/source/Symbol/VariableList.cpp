//===-- VariableList.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/VariableList.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/CompileUnit.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// VariableList constructor
//----------------------------------------------------------------------
VariableList::VariableList() :
    m_variables()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
VariableList::~VariableList()
{
}


void
VariableList::AddVariable(const VariableSP &variable_sp)
{
    m_variables.push_back(variable_sp);
}


void
VariableList::AddVariables(VariableList *variable_list)
{
    std::copy(  variable_list->m_variables.begin(), // source begin
                variable_list->m_variables.end(),   // source end
                back_inserter(m_variables));        // destination
}


void
VariableList::Clear()
{
    m_variables.clear();
}



VariableSP
VariableList::GetVariableAtIndex(uint32_t idx)
{
    VariableSP variable_sp;
    if (idx < m_variables.size())
        variable_sp = m_variables[idx];
    return variable_sp;
}



VariableSP
VariableList::FindVariable(const ConstString& name)
{
    VariableSP var_sp;
    iterator pos, end = m_variables.end();
    for (pos = m_variables.begin(); pos != end; ++pos)
    {
        if ((*pos)->GetName() == name)
        {
            var_sp = (*pos);
            break;
        }
    }
    return var_sp;
}

uint32_t
VariableList::FindIndexForVariable (Variable* variable)
{
    VariableSP var_sp;
    iterator pos;
    const iterator begin = m_variables.begin();
    const iterator end = m_variables.end();
    for (pos = m_variables.begin(); pos != end; ++pos)
    {
        if ((*pos).get() == variable)
            return std::distance (begin, pos);
    }
    return UINT32_MAX;
}

size_t
VariableList::MemorySize() const
{
    size_t mem_size = sizeof(VariableList);
    const_iterator pos, end = m_variables.end();
    for (pos = m_variables.begin(); pos != end; ++pos)
        mem_size += (*pos)->MemorySize();
    return mem_size;
}

size_t
VariableList::GetSize() const
{
    return m_variables.size();
}


void
VariableList::Dump(Stream *s, bool show_context) const
{
//  s.Printf("%.*p: ", (int)sizeof(void*) * 2, this);
//  s.Indent();
//  s << "VariableList\n";

    const_iterator pos, end = m_variables.end();
    for (pos = m_variables.begin(); pos != end; ++pos)
    {
        (*pos)->Dump(s, show_context);
    }
}

