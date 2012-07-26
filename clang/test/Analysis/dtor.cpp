// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store region -analyzer-ipa=inlining -cfg-add-implicit-dtors -cfg-add-initializers -verify %s

class A {
public:
  ~A() { 
    int *x = 0;
    *x = 3; // expected-warning{{Dereference of null pointer}}
  }
};

int main() {
  A a;
}


typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

class SmartPointer {
  void *X;
public:
  SmartPointer(void *x) : X(x) {}
  ~SmartPointer() {
    free(X);
  }
};

void testSmartPointer() {
  char *mem = (char*)malloc(4);
  {
    SmartPointer Deleter(mem);
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}
