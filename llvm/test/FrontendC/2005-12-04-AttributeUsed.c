// RUN: %llvmgcc %s -S -emit-llvm -o - | llvm-as | llvm-dis | \
// RUN:   grep llvm.used | grep foo | grep X
// FIXME - Unxfail this when I know what's going on.
// XFAIL: *

int X __attribute__((used));
int Y;

__attribute__((used)) void foo() {}

