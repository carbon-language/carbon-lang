// RUN: %llvmgcc -xc %s -S -o /dev/null 2>&1 | grep fluffy | grep '2006-09-25-DebugFilename.c.tr'
#include "2006-09-25-DebugFilename.h"
int func1() { return hfunc1(); }
int func2() { fluffy; return hfunc1(); }

