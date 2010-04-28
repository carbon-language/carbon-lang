// RUN: %llvmgcc -S %s -fasm-blocks -o - -O | grep naked
// 7533078 (partial).

asm int f() {
  xyz
}
