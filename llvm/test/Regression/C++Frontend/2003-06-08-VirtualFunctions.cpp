

struct foo {
  int y;
  foo();
  virtual int T() = 0;
};

struct bar : public foo {
  //int x;
  bar();
  int T() {}
};

//int bar::X() { return 0; }

foo::foo() : y(4) {

}

bar::bar() {

}
