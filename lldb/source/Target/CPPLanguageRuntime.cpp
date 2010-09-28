//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CPPLanguageRuntime::~CPPLanguageRuntime()
{
}

CPPLanguageRuntime::CPPLanguageRuntime (Process *process) :
    LanguageRuntime (process)
{

}

bool
CPPLanguageRuntime::GetObjectDescription (Stream &str, ValueObject &object, ExecutionContextScope *exe_scope)
{
    // C++ has no generic way to do this.
    return false;
}
