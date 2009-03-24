// RUN: clang-cc %s -emit-llvm -o %t -fblocks
void (^f)(void) = ^{};
