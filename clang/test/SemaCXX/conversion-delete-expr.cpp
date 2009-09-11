// RUN: clang-cc -fsyntax-only -verify -std=c++0x %s

// Test1
struct B {
  operator char *();
};

struct D : B {
  operator int *();
};

void f (D d)
{
   delete d; // expected-error {{cannot delete expression of type 'struct D'}}
}

// Test2
struct B1 {
  operator int *();
};

struct D1 : B1 {
  operator int *();
};

void f1 (D1 d)
{
   delete d;
}

// Test3
struct B2 {
  operator const int *();
};

struct D2 : B2 {
  operator int *();
};

void f2 (D2 d)
{
   delete d; // expected-error {{cannot delete expression of type 'struct D2'}}
}

// Test4

struct B3 {
  operator const int *();
};

struct A3 {
  operator const int *();
};

struct D3 : A3, B3 {
};

void f3 (D3 d)
{
   delete d; // expected-error {{cannot delete expression of type 'struct D3'}}
}

// Test5
struct X {
   operator int();
   operator int*();
};

void f4(X x) { delete x; delete x; }

// Test6

struct X1 {
   operator int();
   operator int*();
   template<typename T> operator T*() const; // converts to any pointer!
};

void f5(X1 x) { delete x; } // FIXME. May have to issue error here too.




