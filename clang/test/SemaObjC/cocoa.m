// RUN: clang-cc %s -print-stats &&
// RUN: clang-cc %s -disable-free
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

