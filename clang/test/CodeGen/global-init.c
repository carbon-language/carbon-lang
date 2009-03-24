// RUN: clang-cc -emit-llvm -o - %s | not grep "common"

// This checks that the global won't be marked as common. 
// (It shouldn't because it's being initialized).

int a;
int a = 242;
