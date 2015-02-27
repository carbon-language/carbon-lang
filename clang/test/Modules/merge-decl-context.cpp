// RUN: rm -rf %t

// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fmodule-name=b -o %t/b.pcm -fmodule-maps \
// RUN:     -emit-module %S/Inputs/merge-decl-context/merge-decl-context.modulemap -I%S/Inputs \
// RUN:     -I %S/Inputs/merge-decl-context
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fmodule-name=c -o %t/c.pcm -fmodule-maps \
// RUN:     -fmodule-file=%t/b.pcm -fno-implicit-modules \
// RUN:     -emit-module %S/Inputs/merge-decl-context/merge-decl-context.modulemap -I%S/Inputs \
// RUN:     -I %S/Inputs/merge-decl-context
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fmodule-name=d -o %t/d.pcm -fmodule-maps \
// RUN:     -fmodule-file=%t/b.pcm -fno-implicit-modules \
// RUN:     -emit-module %S/Inputs/merge-decl-context/merge-decl-context.modulemap -I%S/Inputs \
// RUN:     -I %S/Inputs/merge-decl-context

// Use the two modules in a single compile.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fmodule-file=%t/c.pcm -fmodule-file=%t/b.pcm \
// RUN:     -fmodule-file=%t/d.pcm -fno-implicit-modules \
// RUN:     -fmodule-map-file=%S/Inputs/merge-decl-context/merge-decl-context.modulemap -I%S/Inputs \
// RUN:     -emit-llvm -o %t/test.o %s

#include "Inputs/merge-decl-context/a.h"
#include "Inputs/merge-decl-context/b.h"
#include "Inputs/merge-decl-context/c.h"
#include "Inputs/merge-decl-context/d.h"

void t() {
  ff(42);
}
