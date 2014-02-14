// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t

int main(void) {
  int i;
  __sync_lock_test_and_set(&i, 0);
  return i;
}
