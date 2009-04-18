// Test this without pch.
// RUN: clang-cc -fblocks -include %S/blocks.h -fsyntax-only -emit-llvm -o - %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/blocks.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -emit-llvm -o - %s 

int do_add(int x, int y) { return add(x, y); }

int do_scaled_add(int a, int b, int s) {
  return scaled_add(a, b, s);
}
