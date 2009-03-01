// RUN: clang %s -emit-llvm -o %t -fblocks
void (^f)(void) = ^{};
