// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep call | grep memcpy_chk | count 3
// RUN: %llvmgcc -S -emit-llvm -O1 %s -o - | grep call | grep {llvm.memcpy} | count 3
// rdar://6716432

void *t1(void *d, void *s) {
  return __builtin___memcpy_chk(d, s, 16, 0);
}

void *t2(void *d, void *s) {
  return __builtin___memcpy_chk(d, s, 16, 10);
}

void *t3(void *d, void *s) {
  return __builtin___memcpy_chk(d, s, 16, 17);
}

void *t4(void *d, void *s, unsigned len) {
  return __builtin___memcpy_chk(d, s, len, 17);
}

char buf[10];
void *t5(void *s, unsigned len) {
  return __builtin___memcpy_chk(buf, s, 5, __builtin_object_size(buf, 0));
}

void *t6(void *d, void *s) {
  return __builtin___memcpy_chk(d, s, 16, __builtin_object_size(d, 0));
}
