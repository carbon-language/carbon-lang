// RUN: clang-cc %s -print-stats &&
// RUN: clang-cc %s -disable-free
#ifdef __APPLE__
#include <Carbon/Carbon.h>
#endif

