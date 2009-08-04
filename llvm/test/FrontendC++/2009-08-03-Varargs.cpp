// RUN: %llvmgxx %s -S -emit-llvm -o - | grep _Z1az\(\.\.\.\)
// PR4678
void a(...) {
}
