// RUN: %clang -target x86_64-linux-gnu -gdwarf-2 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -gdwarf-3 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER3
// RUN: %clang -target x86_64-linux-gnu -gdwarf-4 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
int main (void) {
  return 0;
}

// CHECK: metadata !{i32 2, metadata !"Dwarf Version", i32 2}
// VER3: metadata !{i32 2, metadata !"Dwarf Version", i32 3}
// VER4: metadata !{i32 2, metadata !"Dwarf Version", i32 4}
