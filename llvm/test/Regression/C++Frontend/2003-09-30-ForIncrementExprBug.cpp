struct C {};

C &foo();

void foox() {
  for (; ; foo());
}

