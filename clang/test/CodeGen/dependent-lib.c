// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple i686-pc-win32 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple x86_64-pc-win32 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple i686-pc-linux -emit-llvm -o - | FileCheck -check-prefix LINUX %s

// CHECK: !llvm.module.flags = !{!0}
// CHECK: !0 = metadata !{i32 6, metadata !"Linker Options", metadata ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = metadata !{metadata ![[msvcrt:[0-9]+]]}
// CHECK: ![[msvcrt]] = metadata !{metadata !"/DEFAULTLIB:msvcrt.lib"}

// LINUX: !llvm.module.flags = !{!0}
// LINUX: !0 = metadata !{i32 6, metadata !"Linker Options", metadata ![[link_opts:[0-9]+]]}
// LINUX: ![[link_opts]] = metadata !{metadata ![[msvcrt:[0-9]+]]}
// LINUX: ![[msvcrt]] = metadata !{metadata !"-lmsvcrt"}

int f();
