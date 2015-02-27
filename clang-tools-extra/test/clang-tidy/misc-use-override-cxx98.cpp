// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-use-override %t -- -std=c++98
// REQUIRES: shell

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-FIXES: {{^}}  virtual ~SimpleCases();

  void a();
  // CHECK-FIXES: {{^}}  void a();

  virtual void b();
  // CHECK-FIXES: {{^}}  virtual void b();
};
