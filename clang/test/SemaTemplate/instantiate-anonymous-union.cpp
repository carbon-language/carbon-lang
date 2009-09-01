// RUN: clang-cc -fsyntax-only %s -Wall

template <typename T> class A { struct { }; };

A<int> a0;

template <typename T> struct B {
  union {
    int a;
    void* b;
  };
    
  void f() {
    a = 10;
    b = 0;
  }
};

B<int> b0;

template <typename T> struct C {
  union {
    int a;
    void* b;
  };

  C(int a) : a(a) { }
  C(void* b) : b(b) { }
};

C<int> c0(0);
