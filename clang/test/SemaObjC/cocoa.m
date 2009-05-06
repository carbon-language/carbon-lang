// RUN: clang-cc -mcpu=pentium4 %s -print-stats
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

