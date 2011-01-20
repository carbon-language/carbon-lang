// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s
struct S {
  virtual void final() final;
  virtual void override() override;
  virtual void n() new;
  int i : 3 new;
  int j new;
};

struct T {
  // virt-specifier-seq is only valid in member-declarators, and a function definition is not a member-declarator.
  virtual void f() const override { } // expected-error {{expected ';' at end of declaration list}}
};
