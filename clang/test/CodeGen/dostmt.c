// RUN: clang %s -emit-llvm

int bar();
int foo() {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    i = bar();
  } while(0);
  return i;
}


int foo1() {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      break;
    i = bar();
  } while(1);
  return i;
}


int foo2() {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      continue;
    i = bar();
  } while(1);
  return i;
}


int foo3() {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      break;
  } while(0);
  return i;
}


int foo4() {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      continue;
  } while(0);
  return i;
}
