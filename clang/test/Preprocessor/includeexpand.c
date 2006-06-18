// RUN: clang %s -fno-caret-diagnostics 2>&1 | grep 'file successfully included' | wc -l | grep 3

// XX expands to nothing.
#define XX

#define FILE "file_to_include.h"
#include XX FILE

#include FILE


#include "file_to_include.h"
