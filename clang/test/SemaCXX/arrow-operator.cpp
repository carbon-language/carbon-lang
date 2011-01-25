// RUN: %clang_cc1 -fsyntax-only -verify %s
struct T { 
  void f();
};

struct A {
  T* operator->(); // expected-note{{candidate function}}
};

struct B {
  T* operator->(); // expected-note{{candidate function}}
};

struct C : A, B {
};

struct D : A { };

struct E; // expected-note {{forward declaration of 'E'}}

void f(C &c, D& d, E& e) {
  c->f(); // expected-error{{use of overloaded operator '->' is ambiguous}}
  d->f();
  e->f(); // expected-error{{incomplete definition of type}}
}

// rdar://8875304
namespace rdar8875304 {
class Point {};
class Line_Segment{ public: Line_Segment(const Point&){} };
class Node { public: Point Location(){ Point p; return p; } };

void f()
{
   Node** node1;
   Line_Segment(node1->Location()); // expected-error {{not a structure or union}}
}
}
