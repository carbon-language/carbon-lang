//===-- NullLog.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/NullLog.h"

using namespace lldb_private;

NullLog::NullLog()
{
}
NullLog::~NullLog()
{
}

void
NullLog::PutCString(const char *cstr)
{
}

void
NullLog::Printf(const char *format, ...)
{
}

void
NullLog::VAPrintf(const char *format, va_list args)
{
}

void
NullLog::LogIf(uint32_t mask, const char *fmt, ...)
{
}

void
NullLog::Debug(const char *fmt, ...)
{
}

void
NullLog::DebugVerbose(const char *fmt, ...)
{
}

void
NullLog::Error(const char *fmt, ...)
{
}

void
NullLog::FatalError(int err, const char *fmt, ...)
{
}

void
NullLog::Verbose(const char *fmt, ...)
{
}

void
NullLog::Warning(const char *fmt, ...)
{
}

void
NullLog::WarningVerbose(const char *fmt, ...)
{
}
