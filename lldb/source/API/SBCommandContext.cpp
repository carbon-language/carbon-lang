//===-- SBCommandContext.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandContext.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/API/SBCommandContext.h"


using namespace lldb;
using namespace lldb_private;


SBCommandContext::SBCommandContext (CommandContext *lldb_object) :
    m_lldb_object (lldb_object)
{
}

SBCommandContext::~SBCommandContext ()
{
}

bool
SBCommandContext::IsValid () const
{
    return m_lldb_object != NULL;
}

