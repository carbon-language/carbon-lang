// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

struct Base {
  virtual void override();
};

struct S : Base {
  virtual void final() final;
  virtual void override() override;
  virtual void n() new;
  int i : 3 new;
  int j new;
};

struct T {
  // virt-specifier-seq is only valid in member-declarators, and a function definition is not a member-declarator.
  // FIXME: This currently doesn't work.
  // virtual void f() const override { } 
};

struct override;
struct Base2 {
  virtual override override(int override);
};

struct A : Base2 {
  virtual struct override override(int override) override;
};
