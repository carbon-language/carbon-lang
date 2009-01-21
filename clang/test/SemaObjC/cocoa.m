// RUN: clang %s -print-stats &&
// RUN: clang -x objective-c-header -o %t %s && clang -token-cache %t %s &&
// RUN: clang -x objective-c-header -o %t %s && clang -token-cache %t %s -E %s -o /dev/null
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

