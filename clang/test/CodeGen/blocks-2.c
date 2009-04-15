// RUN: clang-cc -g %s -emit-llvm -o %t -fblocks 
// RUN: grep "func.start" %t | count 4
// 1 declaration, 1 bar, 1 test_block_dbg and 1 for the block.

static __inline__ __attribute__((always_inline)) int bar(int va, int vb) { return (va == vb); }

int test_block_dbg() {
  extern int g;
  static int i = 1;
  ^(int j){ i = bar(3,4); }(0);
  return i + g;
}

