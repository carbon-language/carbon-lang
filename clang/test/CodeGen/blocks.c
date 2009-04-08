// RUN: clang-cc %s -emit-llvm -o %t -fblocks
void (^f)(void) = ^{};

// rdar://6768379
int f0(int (^a0)()) {
  return a0(1, 2, 3);
}
