//===-- ThreadStateCoordinatorTestMock.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file provides a few necessary functions to link
// ThreadStateCoordinatorTest.cpp Bringing in the real implementations results
// in a cascade of dependencies that pull in all of lldb.

#include "lldb/Core/Log.h"

using namespace lldb_private;

void
lldb_private::Log::Error (char const*, ...)
{
}

void
lldb_private::Log::Printf (char const*, ...)
{
}

Log*
lldb_private::GetLogIfAnyCategoriesSet (unsigned int)
{
    return nullptr;
}
