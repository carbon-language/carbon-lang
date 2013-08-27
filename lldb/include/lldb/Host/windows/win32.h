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
#define O_NONBLOCK  0
#define SIGTRAP     5
#define SIGKILL     9
#define SIGSTOP     20

#if defined(_MSC_VER)
# define S_IRUSR  S_IREAD                      /* read, user */
# define S_IWUSR  S_IWRITE                     /* write, user */
# define S_IXUSR  0                            /* execute, user */
#endif
#define S_IRGRP  0                            /* read, group */
#define S_IWGRP  0                            /* write, group */
#define S_IXGRP  0                            /* execute, group */
#define S_IROTH  0                            /* read, others */
#define S_IWOTH  0                            /* write, others */
#define S_IXOTH  0                            /* execute, others */
#define S_IRWXU  0
#define S_IRWXG  0
#define S_IRWXO  0

#endif  // LLDB_lldb_win32_h_
