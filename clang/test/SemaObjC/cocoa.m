// RUN: %clang_cc1 -mcpu pentium4 %s -print-stats
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

