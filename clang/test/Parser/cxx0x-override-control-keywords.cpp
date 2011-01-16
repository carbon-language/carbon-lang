// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s
struct S {
  virtual void final() final;
  virtual void override() override;
  virtual void n() new;
};
