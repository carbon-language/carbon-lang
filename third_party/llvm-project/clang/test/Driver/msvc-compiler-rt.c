// RUN: %clang -target x86_64-pc-windows-msvc --rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix MSVC-COMPILER-RT
// RUN: not %clang %s -target x86_64-pc-windows-msvc --rtlib=libgcc 2>&1 | FileCheck %s -check-prefix CHECK-ERROR

// MSVC-COMPILER-RT: "{{.*}}clang_rt.builtins{{.*}}"
// CHECK-ERROR: unsupported runtime library 'libgcc' for platform 'MSVC'
