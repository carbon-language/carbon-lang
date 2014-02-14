
// This uses a headermap with this entry:
//   Foo.h -> Foo/Foo.h

// RUN: %clang_cc1 -E %s -o %t.i -I %S/foo.hmap -F %S
// RUN: FileCheck %s -input-file %t.i

// CHECK: Foo.h is parsed

#include "Foo.h"
