/* Compile with:
   cat >modules.modulemap <<EOF
   module Parent {
     module Child {
       header "Child.h"
     }
   }
EOF
   clang -D CHILD_H -E -o Child.h submodules.m
   clang -cc1 -emit-obj -fmodules -fmodule-map-file=modules.modulemap \
     -fmodule-format=obj -g -dwarf-ext-refs -fmodules-cache-path=. \
     -fdisable-module-hash submodules.m -o 1.o
*/

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/submodules \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-dump=info - | FileCheck %s

// ---------------------------------------------------------------------
#ifdef CHILD_H
// ---------------------------------------------------------------------
// CHECK:            DW_TAG_compile_unit
// CHECK-NOT:        DW_TAG
// CHECK:              DW_TAG_module
// CHECK-NEXT:           DW_AT_name{{.*}}"Parent"
// CHECK:                DW_TAG_module
// CHECK-NEXT:             DW_AT_name{{.*}}"Child"
// CHECK:                  DW_TAG_structure_type
// CHECK-NOT:                DW_TAG
// CHECK:                    DW_AT_name {{.*}}"PruneMeNot"

struct PruneMeNot;

// ---------------------------------------------------------------------
#else
// ---------------------------------------------------------------------

@import Parent.Child;
int main(int argc, char **argv) { return 0; }
#endif
