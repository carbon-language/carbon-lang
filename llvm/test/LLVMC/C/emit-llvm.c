// Check that -emit-llvm [-S] works correctly.

// RUN: llvmc -c -emit-llvm -o - %s | llvm-dis | grep "@f0()" | count 1
// RUN: llvmc -c -emit-llvm -S -o - %s | grep "@f0()" | count 1
// XFAIL: vg_leak

int f0(void) {
}
