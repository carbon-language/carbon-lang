// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep call | not grep memset_chk
// rdar://6728562

void t(void *ptr) {
  __builtin___memset_chk(ptr, 0, 32, __builtin_object_size (ptr, 0));
}
