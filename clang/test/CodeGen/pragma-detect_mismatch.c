// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

#pragma detect_mismatch("test", "1")

#define BAR "2"
#pragma detect_mismatch("test2", BAR)

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = metadata !{i32 6, metadata !"Linker Options", metadata ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = metadata !{metadata ![[test:[0-9]+]], metadata ![[test2:[0-9]+]]}
// CHECK: ![[test]] = metadata !{metadata !"/FAILIFMISMATCH:\22test=1\22"}
// CHECK: ![[test2]] = metadata !{metadata !"/FAILIFMISMATCH:\22test2=2\22"}
