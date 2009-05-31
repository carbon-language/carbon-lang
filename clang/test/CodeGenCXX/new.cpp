// RUN: clang-cc %s -emit-llvm -o %t

void t1() {
  int* a = new int;
}

// Placement.
void* operator new(unsigned long, void*) throw();

void t2(int* a) {
  int* b = new (a) int;
}
