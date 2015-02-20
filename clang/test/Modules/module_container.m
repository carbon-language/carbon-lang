@import DependsOnModule;

// RUN: rm -rf %t-MachO %t-ELF %t-COFF
// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fmodules -fdisable-module-hash -fmodules-cache-path=%t-MachO -F %S/Inputs %s
// RUN: %clang_cc1 -triple=x86_64-linux-elf -fmodules -fdisable-module-hash -fmodules-cache-path=%t-ELF -F %S/Inputs %s
// RUN: %clang_cc1 -triple=x86_64-windows-coff -fmodules -fdisable-module-hash -fmodules-cache-path=%t-COFF -F %S/Inputs %s

// RUN: llvm-objdump -section-headers %t-MachO/DependsOnModule.pcm %t-ELF/DependsOnModule.pcm %t-COFF/DependsOnModule.pcm | FileCheck %s
// CHECK: file format Mach-O 64-bit x86-64
// CHECK: __clangast   {{[0-9a-f]+}} {{[0-9a-f]+}} DATA
// CHECK: file format ELF64-x86-64
// CHECK: __clangast   {{[0-9a-f]+}} {{[0-9a-f]+}} DATA
// CHECK: file format COFF-x86-64
// CHECK: clangast   {{[0-9a-f]+}} {{[0-9a-f]+}}
