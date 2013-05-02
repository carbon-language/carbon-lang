// RUN: %clang -fsyntax-only -target i386-pc-win32 %s

#if defined(_WCHAR_T_DEFINED)
#error "_WCHAR_T_DEFINED should not be defined in C99"
#endif

#include <stddef.h>

#if !defined(_WCHAR_T_DEFINED)
#error "_WCHAR_T_DEFINED should have been set by stddef.h"
#endif

#if defined(_NATIVE_WCHAR_T_DEFINED)
#error "_NATIVE_WCHAR_T_DEFINED should not be defined"
#endif
