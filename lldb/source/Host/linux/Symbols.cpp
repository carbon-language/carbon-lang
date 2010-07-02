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

FileSpec
Symbols::LocateExecutableObjectFile (const FileSpec *exec_fspec, const ArchSpec* arch, const UUID *uuid)
{
    // FIXME: Proper impliementation?
    return FileSpec();
}

FileSpec
Symbols::LocateExecutableSymbolFile (const FileSpec *exec_fspec, const ArchSpec* arch, const UUID *uuid)
{
    // FIXME: Proper impliementation?
    return FileSpec();
}
