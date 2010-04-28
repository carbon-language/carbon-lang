// RUN: %llvmgcc -S %s -fasm-blocks -o - -O | grep naked
// PR2094

asm int f() {
  xyz
}
