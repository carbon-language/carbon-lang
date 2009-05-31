// RUN: clang-cc %s -emit-llvm -o %t

void t1() {
  int* a = new int;
}

// Placement.
void* operator new(unsigned long, void*) throw();

void t2(int* a) {
  int* b = new (a) int;
}

struct S {
  int a;
};

void t3() {
  int *a = new int(10);
  _Complex int* b = new _Complex int(10i);
  
  S s;
  s.a = 10;
  S *sp = new S(s);
}
