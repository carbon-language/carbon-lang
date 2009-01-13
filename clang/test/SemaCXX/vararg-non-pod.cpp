// RUN: clang -fsyntax-only -verify -fblocks %s

class C {
public:
  C(int);
  void g(int a, ...);
  static void h(int a, ...);
};

void g(int a, ...);

void t1()
{
  C c(10);
  
  g(10, c); // expected-warning{{cannot pass object of non-POD type 'class C' through variadic function; call will abort at runtime}}
}

void t2()
{
  C c(10);

  c.g(10, c); // expected-warning{{cannot pass object of non-POD type 'class C' through variadic method; call will abort at runtime}}
  
  C::h(10, c); // expected-warning{{cannot pass object of non-POD type 'class C' through variadic function; call will abort at runtime}}
}

int (^block)(int, ...);

void t3()
{
  C c(10);
  
  block(10, c); // expected-warning{{cannot pass object of non-POD type 'class C' through variadic block; call will abort at runtime}}
}

class D {
public:
  void operator() (int a, ...);
};

void t4()
{
  C c(10);

  D d;
  
  d(10, c); // expected-warning{{Line 48: cannot pass object of non-POD type 'class C' through variadic method; call will abort at runtime}}
}
