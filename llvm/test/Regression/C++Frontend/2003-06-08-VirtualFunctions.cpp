

struct foo {
  int y;
  foo();
  virtual int T() = 0;
};

struct bar : public foo {
  bar();
  int T() {}
};

foo::foo() : y(4) { }
bar::bar() { }
int main() { return 0; }
