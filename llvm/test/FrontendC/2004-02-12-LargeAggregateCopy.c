// RUN: %llvmgcc -xc %s -S -o - | grep llvm.memcpy

struct X { int V[10000]; };
struct X Global1, Global2;
void test() {
  Global2 = Global1;
}

