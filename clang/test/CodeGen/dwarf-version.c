// RUN: %clang -target x86_64-linux-gnu -gdwarf-2 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target x86_64-linux-gnu -gdwarf-3 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER3
// RUN: %clang -target x86_64-linux-gnu -gdwarf-4 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-linux-gnu -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-apple-darwin -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-unknown-openbsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-unknown-freebsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target i386-pc-solaris -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
int main (void) {
  return 0;
}

// VER2: metadata !{i32 2, metadata !"Dwarf Version", i32 2}
// VER3: metadata !{i32 2, metadata !"Dwarf Version", i32 3}
// VER4: metadata !{i32 2, metadata !"Dwarf Version", i32 4}
