class Empty {};

void foo(Empty E);

void bar() {
  foo(Empty());
}

