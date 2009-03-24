// RUN: clang-cc %s -fsyntax-only -print-stats &&
// RUN: clang-cc -x c-header -o %t %s && clang-cc -token-cache %t %s
#ifdef __APPLE__
#include <Carbon/Carbon.h>
#endif

