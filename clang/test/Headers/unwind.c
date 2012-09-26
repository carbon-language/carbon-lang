// RUN: %clang -target arm-unknown-linux-gnueabi %s -fsyntax-only -o -
// RUN: %clang -target i686-unknown-linux %s -fsyntax-only -o -

#include "unwind.h"
// CHECK-NOT: error
// CHECK-NOT: warning
