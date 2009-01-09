// RUN: clang %s -print-stats &&
// RUN: clang -x objective-c++-header -o %t %s && clang -token-cache %t %s
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif
