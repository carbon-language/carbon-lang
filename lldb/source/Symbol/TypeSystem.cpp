//
//  TypeSystem.cpp
//  lldb
//
//  Created by Ryan Brown on 3/29/15.
//
//

#include "lldb/Symbol/TypeSystem.h"

using namespace lldb_private;

TypeSystem::TypeSystem(LLVMCastKind kind) :
    m_kind (kind),
    m_sym_file (nullptr)
{
}

TypeSystem::~TypeSystem()
{
}
