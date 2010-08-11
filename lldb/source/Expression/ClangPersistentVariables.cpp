//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"

using namespace lldb_private;
using namespace clang;

ClangPersistentVariables::ClangPersistentVariables () :
    m_variables(),
    m_result_counter(0)
{
}

ClangPersistentVariable *
ClangPersistentVariables::CreateVariable (ConstString name, 
                                          TypeFromUser user_type)
{    
    ClangPersistentVariable new_var(user_type);
    
    if (m_variables.find(name) != m_variables.end())
        return NULL;
    
    m_variables[name] = new_var;
    
    return &m_variables[name];
}

ClangPersistentVariable *
ClangPersistentVariables::CreateResultVariable (TypeFromUser user_type)
{    
    StreamString s;
    s.Printf("$%llu", m_result_counter);
    ConstString name(s.GetString().c_str());
    
    ClangPersistentVariable *ret = CreateVariable (name, user_type);
    
    if (ret != NULL)
        ++m_result_counter;
    
    return ret;
}

ClangPersistentVariable *
ClangPersistentVariables::GetVariable (ConstString name)
{    
    if (m_variables.find(name) == m_variables.end())
        return NULL;
    
    return &m_variables[name];
}
