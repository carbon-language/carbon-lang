// RUN: clang-cc -mcpu=pentium4 -emit-pth -o %t %s && 
// RUN: clang-cc -mcpu=pentium4 -token-cache %t %s &&
// RUN: clang-cc -mcpu=pentium4 -token-cache %t %s -E %s -o /dev/null
#ifdef __APPLE__
#include <Carbon/Carbon.h>
#endif

