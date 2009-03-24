// RUN: clang-cc %s -print-stats &&
// RUN: clang-cc -x objective-c++-header -o %t %s && 
// RUN: clang-cc -token-cache %t %s
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif
