// RUN: %check_clang_tidy %s modernize-use-override %t -- -std=c++98

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
