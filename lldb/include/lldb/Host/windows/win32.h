//===-- lldb-win32.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_win32_h_
#define LLDB_lldb_win32_h_

#include <stdarg.h>

// posix utilities
int vasprintf(char **ret, const char *fmt, va_list ap);
char * strcasestr(const char *s, const char* find);
char* realpath(const char * name, char * resolved);

#define PATH_MAX MAX_PATH

#define O_NOCTTY    0
#define SIGTRAP     5
#define SIGKILL     9
#define SIGSTOP     20

#endif  // LLDB_lldb_win32_h_
