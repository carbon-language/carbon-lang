// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-linux -fms-extensions -emit-llvm -o - | FileCheck -check-prefix LINUX %s

#pragma comment(lib, "msvcrt.lib")
#pragma comment(lib, "kernel32")
#pragma comment(lib, "USER32.LIB")

#define BAR "2"
#pragma comment(linker," /bar=" BAR)

// CHECK: !llvm.module.flags = !{!0}
// CHECK: !0 = metadata !{i32 6, metadata !"Linker Options", metadata ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = metadata !{metadata ![[msvcrt:[0-9]+]], metadata ![[kernel32:[0-9]+]], metadata ![[USER32:[0-9]+]], metadata ![[bar:[0-9]+]]}
// CHECK: ![[msvcrt]] = metadata !{metadata !"/DEFAULTLIB:msvcrt.lib"}
// CHECK: ![[kernel32]] = metadata !{metadata !"/DEFAULTLIB:kernel32.lib"}
// CHECK: ![[USER32]] = metadata !{metadata !"/DEFAULTLIB:USER32.LIB"}
// CHECK: ![[bar]] = metadata !{metadata !" /bar=2"}

// LINUX: metadata !{metadata !"-lmsvcrt.lib"}
// LINUX: metadata !{metadata !"-lkernel32"}
// LINUX: metadata !{metadata !"-lUSER32.LIB"}
// LINUX: metadata !{metadata !" /bar=2"}
