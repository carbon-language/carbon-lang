// RUN: %clang_cc1 -target-cpu pentium4 %s -print-stats
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

