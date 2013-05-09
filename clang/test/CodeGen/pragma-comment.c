// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-linux -fms-extensions -emit-llvm -o - | FileCheck -check-prefix LINUX %s

#pragma comment(lib, "msvcrt.lib")

#define BAR "2"
#pragma comment(linker," /bar=" BAR)

// CHECK: !llvm.module.flags = !{!0}
// CHECK: !0 = metadata !{i32 6, metadata !"Linker Options", metadata ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = metadata !{metadata ![[msvcrt:[0-9]+]], metadata ![[bar:[0-9]+]]}
// CHECK: ![[msvcrt]] = metadata !{metadata !"/DEFAULTLIB:msvcrt.lib"}
// CHECK: ![[bar]] = metadata !{metadata !" /bar=2"}

// LINUX: metadata !{metadata !"-lmsvcrt.lib"}
// LINUX: metadata !{metadata !" /bar=2"}
