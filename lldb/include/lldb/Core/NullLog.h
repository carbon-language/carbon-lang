//===-- NullLog.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Core_NullLog_H_
#define liblldb_Core_NullLog_H_

#include "lldb/Core/Log.h"

//----------------------------------------------------------------------
// Logging Functions
//----------------------------------------------------------------------
namespace lldb_private
{

class NullLog : public Log
{
    NullLog(NullLog &) = delete;
    NullLog &operator=(NullLog &) = delete;

  public:
    //------------------------------------------------------------------
    // Member functions
    //------------------------------------------------------------------
    NullLog();
    ~NullLog();

    void PutCString(const char *cstr) override;

    void Printf(const char *format, ...) override __attribute__((format(printf, 2, 3)));

    void VAPrintf(const char *format, va_list args) override;

    void LogIf(uint32_t mask, const char *fmt, ...) override __attribute__((format(printf, 3, 4)));

    void Debug(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));

    void DebugVerbose(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));

    void Error(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));

    void FatalError(int err, const char *fmt, ...) override __attribute__((format(printf, 3, 4)));

    void Verbose(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));

    void Warning(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));

    void WarningVerbose(const char *fmt, ...) override __attribute__((format(printf, 2, 3)));
};

} // namespace lldb_private

#endif // liblldb_Core_NullLog_H_
