/* Compile with:
   cat >modules.modulemap <<EOF
     module Foo {
       header "Foo.h"
       export *
     }
     module Bar {
       header "Bar.h"
       export *
     }
   EOF
   clang -D BAR_H -E -o Bar.h
   clang -D FOO_H -E -o Foo.h
   clang -cc1 -emit-obj -fmodules -fmodule-map-file=modules.modulemap \
     -fmodule-format=obj -g -dwarf-ext-refs -fmodules-cache-path=. \
     -fdisable-module-hash modules.m -o 1.o
*/

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-dump=info - | FileCheck %s

// ---------------------------------------------------------------------
#ifdef BAR_H
// ---------------------------------------------------------------------
// CHECK: DW_TAG_compile_unit
// CHECK:   DW_TAG_module
// CHECK-NEXT: DW_AT_name {{.*}}"Bar"
// CHECK:   DW_TAG_member
// CHECK:     DW_AT_name {{.*}}"value"

struct Bar {
  int value;
};

#else
// ---------------------------------------------------------------------
#ifdef FOO_H
// ---------------------------------------------------------------------
// CHECK: 55{{.*}}DW_TAG_compile_unit
// CHECK:   DW_TAG_module
// CHECK-NEXT: DW_AT_name {{.*}}"Foo"
// CHECK:      DW_TAG_typedef
@import Bar;
typedef struct Bar Bar;
struct S {};

// ---------------------------------------------------------------------
#else
// ---------------------------------------------------------------------

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_name {{.*}}"main"
@import Foo;
int main(int argc, char **argv) {
  Bar bar;
  bar.value = 42;
  return bar.value;
}
#endif
#endif
