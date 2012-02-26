//===-- Symbols.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Symbols.h"

using namespace lldb;
using namespace lldb_private;

#if !defined (__APPLE__)

FileSpec
Symbols::LocateExecutableObjectFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::LocateExecutableSymbolFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

#endif
