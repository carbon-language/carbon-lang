// RUN: clang -E %s -fno-caret-diagnostics 2>&1 >/dev/null | grep 'file successfully included' | count 3

// XX expands to nothing.
#define XX

#define FILE "file_to_include.h"
#include XX FILE

#include FILE


#include "file_to_include.h"
