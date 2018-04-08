// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

class B {
public:
 int i;
 struct {  struct { union { int j; }; };  };
};

class X : public B { };
class Y : public B { };

class Z : public X, Y {
public:
 int a() { return X::i; }
 int b() { return X::j; }
 int c() { return this->X::j; }
};
