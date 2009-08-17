// RUN: %llvmgxx %s -S -emit-llvm -o - | grep _Z1az\(\.\.\.\)
// XFAIL: *
// PR4678
void a(...) {
}
