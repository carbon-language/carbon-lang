// RUN: clang-cc -emit-llvm -o - %s -std=c++0x
void *f1(unsigned long l) {
  return reinterpret_cast<void *>(l);
}

unsigned long f2() {
  return reinterpret_cast<unsigned long>(nullptr);
}

unsigned long f3(void *p) {
  return reinterpret_cast<unsigned long>(p);
}