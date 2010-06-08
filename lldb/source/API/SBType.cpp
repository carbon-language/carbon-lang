//===-- SBType.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBType.h"
#include "lldb/Symbol/ClangASTContext.h"

using namespace lldb;
using namespace lldb_private;


bool
SBType::IsPointerType (void *opaque_type)
{
    return ClangASTContext::IsPointerType (opaque_type);
}


