// RUN: clang %s -fsyntax-only -print-stats &&
// RUN: clang -x c-header -o %t %s && clang -token-cache %t %s
#ifdef __APPLE__
#include <Carbon/Carbon.h>
#endif

