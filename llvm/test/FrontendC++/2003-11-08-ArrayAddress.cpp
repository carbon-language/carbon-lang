// RUN: %llvmgxx -xc++ %s -S -o - | grep getelementptr

struct foo {
  int array[100];
  void *getAddr(unsigned i);
};

void *foo::getAddr(unsigned i) {
  return &array[i];
}
