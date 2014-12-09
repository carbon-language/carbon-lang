/* For compilation instructions see basic1.c. */

static int baz = 42;
static int private_int;
extern volatile int val;
int unused_data = 1;

int bar(int);

void unused1() {
  bar(baz);
}

static int inc() {
  return ++private_int;
}

__attribute__((noinline))
int foo(int arg) {
  return bar(arg+val) + inc() + baz++;
}

