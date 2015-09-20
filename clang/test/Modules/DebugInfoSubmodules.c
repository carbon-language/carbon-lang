// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-format=obj -g -dwarf-ext-refs \
// RUN:     -fimplicit-module-maps -x c -fmodules-cache-path=%t -I %S/Inputs \
// RUN:     %s -mllvm -debug-only=pchcontainer 2>&1 | FileCheck %s
#include "DebugSubmoduleA.h"

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "A",
// CHECK-SAME:             scope: ![[SUBMODULEA:[0-9]+]]
// CHECK: ![[SUBMODULEA]] = !DIModule(
// CHECK-SAME:                        name: "DebugSubmodules.DebugSubmoduleA",

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B",
// CHECK-SAME:             scope: ![[SUBMODULEB:[0-9]+]]
// CHECK: ![[SUBMODULEB]] = !DIModule(
// CHECK-SAME:                        name: "DebugSubmodules.DebugSubmoduleB",
