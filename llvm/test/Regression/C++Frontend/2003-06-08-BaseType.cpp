

struct foo {
  int y;
  foo();
};

struct bar : public foo {
  //int x;
  bar();
};

bar::bar() { }
foo::foo() { }

int main() { return 0; }
