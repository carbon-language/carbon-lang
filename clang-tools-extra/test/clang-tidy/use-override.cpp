// RUN: $(dirname %s)/check_clang_tidy_output.sh %s misc-use-override %t
// REQUIRES: shell

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
  virtual void e();
  virtual void f();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases() {}
  // CHECK: :[[@LINE-1]]:11: warning: Prefer using 'override' or 'final' instead of 'virtual'

  void a();
  // CHECK: :[[@LINE-1]]:8: warning: Use exactly
  virtual void b();
  // CHECK: :[[@LINE-1]]:16: warning: Prefer using
  virtual void c() override;
  // CHECK: :[[@LINE-1]]:16: warning: Use exactly
  void d() override final;
  // CHECK: :[[@LINE-1]]:8: warning: Use exactly
  void e() override;
  // CHECK-NOT: :[[@LINE-1]]:{{.*}} warning:
  void f() final;
  // CHECK-NOT: :[[@LINE-1]]:{{.*}} warning:
};
