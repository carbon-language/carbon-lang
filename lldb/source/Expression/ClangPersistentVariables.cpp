//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Value.h"

using namespace lldb_private;

ClangPersistentVariables::ClangPersistentVariables () :
    ClangExpressionVariableStore()
{
    m_result_counter = 0;
}

void
ClangPersistentVariables::GetNextResultName (std::string &name)
{
    StreamString s;
    s.Printf("$%llu", m_result_counter);
    
    m_result_counter++;
    
    name = s.GetString();
}

bool
ClangPersistentVariables::CreatePersistentVariable(const char *name,
                                                   TypeFromUser user_type)
{
    if (GetVariable(name))
            return false;

    ClangExpressionVariable &pvar (VariableAtIndex(CreateVariable()));

    pvar.m_name = name;
    pvar.m_user_type = user_type;
    // TODO: Sean, why do we need to call this?, we can just make it below
    // and we aren't checking the result or anything... Is this cruft left
    // over from an old code re-org?
    //pvar.EnableDataVars();
    pvar.m_data_sp.reset(new DataBufferHeap(pvar.Size(), 0));
    
    return true;
}