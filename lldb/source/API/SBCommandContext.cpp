//===-- SBCommandContext.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/API/SBCommandContext.h"


using namespace lldb;
using namespace lldb_private;


SBCommandContext::SBCommandContext (Debugger *lldb_object) :
    m_opaque (lldb_object)
{
}

SBCommandContext::~SBCommandContext ()
{
}

bool
SBCommandContext::IsValid () const
{
    return m_opaque != NULL;
}

