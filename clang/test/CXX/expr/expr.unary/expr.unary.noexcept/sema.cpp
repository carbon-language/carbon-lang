// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x -fms-extensions %s

#define P(e) static_assert(noexcept(e), "expected nothrow")
#define N(e) static_assert(!noexcept(e), "expected throw")

void simple() {
  P(0);
  P(0 + 0);
  int i;
  P(i);
  P(sizeof(0));
  P(static_cast<int>(0));
  N(throw 0);
}

void nospec();
void allspec() throw(...);
void intspec() throw(int);
void emptyspec() throw();

void call() {
  N(nospec());
  N(allspec());
  N(intspec());
  P(emptyspec());
}

void (*pnospec)();
void (*pallspec)() throw(...);
void (*pintspec)() throw(int);
void (*pemptyspec)() throw();

void callptr() {
  N(pnospec());
  N((*pnospec)());
  N(pallspec());
  N((*pallspec)());
  N(pintspec());
  N((*pintspec)());
  P(pemptyspec());
  P((*pemptyspec)());
}

struct S1 {
  void nospec();
  void allspec() throw(...);
  void intspec() throw(int);
  void emptyspec() throw();
};

void callmem() {
  S1 s;
  N(s.nospec());
  N(s.allspec());
  N(s.intspec());
  P(s.emptyspec());
}

void (S1::*mpnospec)();
void (S1::*mpallspec)() throw(...);
void (S1::*mpintspec)() throw(int);
void (S1::*mpemptyspec)() throw();

void callmemptr() {
  S1 s;
  N((s.*mpnospec)());
  N((s.*mpallspec)());
  N((s.*mpintspec)());
  P((s.*mpemptyspec)());
}

struct S2 {
  S2();
  S2(int, int) throw();
  void operator +();
  void operator -() throw();
  void operator +(int);
  void operator -(int) throw();
  operator int();
  operator float() throw();
};

void *operator new(__typeof__(sizeof(int)) sz, int) throw();

void implicits() {
  N(new int);
  P(new (0) int);
  N(S2());
  P(S2(0, 0));
  S2 s;
  N(+s);
  P(-s);
  N(s + 0);
  P(s - 0);
  N(static_cast<int>(s));
  P(static_cast<float>(s));
  // FIXME: test destructors of temporaries
}

struct V {
  virtual ~V() throw();
};
struct D : V {};

void dyncast() {
  V *pv = 0;
  D *pd = 0;
  P(dynamic_cast<V&>(*pd));
  P(dynamic_cast<V*>(pd));
  N(dynamic_cast<D&>(*pv));
  P(dynamic_cast<D*>(pv));
}

namespace std {
  struct type_info {};
}

void idtype() {
  P(typeid(V));
  P(typeid((V*)0));
  P(typeid(*(S1*)0));
  N(typeid(*(V*)0));
}

void uneval() {
  P(sizeof(typeid(*(V*)0)));
  P(typeid(typeid(*(V*)0)));
}
