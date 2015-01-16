//===-- SocketTestMock.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file provides a few necessary functions to link SocketTest.cpp
// Bringing in the real implementations results in a cascade of dependencies
// that pull in all of lldb.

#include "lldb/Core/Log.h"

#ifdef _WIN32
#include <windows.h>
#endif

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

#include "lldb/Host/FileSystem.h"

#ifdef _WIN32

Error
FileSystem::Unlink(const char *path)
{
    Error error;
    BOOL result = ::DeleteFile(path);
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

#else

Error
FileSystem::Unlink (const char *path)
{
    Error error;
    if (::unlink (path) == -1)
        error.SetErrorToErrno ();
    return error;
}

#endif

