

static int foo(int);

static int foo(C)
char C;
{
  return C;
}

void test() {
  foo(7);
}
