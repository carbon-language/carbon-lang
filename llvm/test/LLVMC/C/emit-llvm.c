// RUN: llvmc -c -emit-llvm -o - %s | llvm-dis | grep "@f0()" | count 1
// XFAIL: vg_leak

int f0(void) {
}
