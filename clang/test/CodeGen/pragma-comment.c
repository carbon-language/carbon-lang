// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-linux -fms-extensions -emit-llvm -o - | FileCheck -check-prefix LINUX %s

#pragma comment(lib, "msvcrt.lib")
#pragma comment(lib, "kernel32")
#pragma comment(lib, "USER32.LIB")

#define BAR "2"
#pragma comment(linker," /bar=" BAR)

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 6, !"Linker Options", ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = !{![[msvcrt:[0-9]+]], ![[kernel32:[0-9]+]], ![[USER32:[0-9]+]], ![[bar:[0-9]+]]}
// CHECK: ![[msvcrt]] = !{!"/DEFAULTLIB:msvcrt.lib"}
// CHECK: ![[kernel32]] = !{!"/DEFAULTLIB:kernel32.lib"}
// CHECK: ![[USER32]] = !{!"/DEFAULTLIB:USER32.LIB"}
// CHECK: ![[bar]] = !{!" /bar=2"}

// LINUX: !{!"-lmsvcrt.lib"}
// LINUX: !{!"-lkernel32"}
// LINUX: !{!"-lUSER32.LIB"}
// LINUX: !{!" /bar=2"}
