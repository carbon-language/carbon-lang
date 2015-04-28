//===-- SBLanguageRuntime.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/Target/LanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;

lldb::LanguageType
SBLanguageRuntime::GetLanguageTypeFromString (const char *string)
{
    return LanguageRuntime::GetLanguageTypeFromString(string);
}

const char *
SBLanguageRuntime::GetNameForLanguageType (lldb::LanguageType language)
{
    return LanguageRuntime::GetNameForLanguageType(language);
}
