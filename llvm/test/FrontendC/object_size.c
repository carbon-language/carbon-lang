// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep ret | grep {\\-1} | count 1
// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep ret | grep {0}  | count 1
// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep ret | grep {8}  | count 1

unsigned t1(void *d) {
  return __builtin_object_size(d, 0);
}

unsigned t2(void *d) {
  return __builtin_object_size(d, 2);
}

char buf[8];
unsigned t3() {
  return __builtin_object_size(buf, 0);
}
