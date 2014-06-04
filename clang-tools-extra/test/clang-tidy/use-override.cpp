// RUN: $(dirname %s)/check_clang_tidy_output.sh %s misc-use-override %t
// REQUIRES: shell

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases() {}
  // CHECK: warning: Prefer using 'override' or 'final' instead of 'virtual'

  void a();
  // CHECK: warning: Prefer using
  virtual void b();
  // CHECK: warning: Prefer using
  void c() override;
  // CHECK-NOT: warning:
  void d() final;
  // CHECK-NOT: warning:
};
