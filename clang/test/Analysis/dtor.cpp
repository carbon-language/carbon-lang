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


void doSomething();
void testSmartPointer2() {
  char *mem = (char*)malloc(4);
  {
    SmartPointer Deleter(mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


class Subclass : public SmartPointer {
public:
  Subclass(void *x) : SmartPointer(x) {}
};

void testSubclassSmartPointer() {
  char *mem = (char*)malloc(4);
  {
    Subclass Deleter(mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


class MultipleInheritance : public Subclass, public SmartPointer {
public:
  MultipleInheritance(void *a, void *b) : Subclass(a), SmartPointer(b) {}
};

void testMultipleInheritance1() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(mem, 0);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}

void testMultipleInheritance2() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(0, mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}

void testMultipleInheritance3() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(mem, mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
    // expected-warning@25 {{Attempt to free released memory}}
  }
}
