// Check that -emit-llvm [-S] works with -opt.

// RUN: llvmc -c -opt -emit-llvm -o - %s | llvm-dis | grep "@f0()" | count 1
// RUN: llvmc -c -opt -emit-llvm -S -o - %s | grep "@f0()" | count 1
// RUN: llvmc --dry-run -c -opt -emit-llvm %s |& grep "^opt"
// XFAIL: vg_leak

int f0(void) {
}
