

struct foo {
  int y;
  foo();
};

struct bar : public foo {
  //int x;
  bar();
};

//int bar::X() { return 0; }

bar::bar() {

}

int main() { return 0; }
