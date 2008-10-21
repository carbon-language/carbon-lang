// RUN: clang -fsyntax-only -verify %s 
void f();
void f(int);
void f(int, float); 
void f(int, int);
void f(int, ...);

typedef float Float;
void f(int, Float); // expected-error {{error: previous declaration is here}}

int f(int, Float); // expected-error {{error: functions that differ only in their return type cannot be overloaded}}

void g(void); // expected-error {{error: previous declaration is here}}
int g(); // expected-error {{error: functions that differ only in their return type cannot be overloaded}}

class X {
  void f();
  void f(int);

  // FIXME: can't test this until we can handle const methods.
  //   void f() const;

  void g(int); // expected-error {{error: previous declaration is here}}
  void g(int, float); // expected-error {{error: previous declaration is here}}
  int g(int, Float); // expected-error {{error: functions that differ only in their return type cannot be overloaded}}

  static void g(float);
  static void g(int); // expected-error {{error: static and non-static member functions with the same parameter types cannot be overloaded}}
};
