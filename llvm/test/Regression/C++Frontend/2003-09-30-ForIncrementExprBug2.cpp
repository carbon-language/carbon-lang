// Test with an opaque type

struct C;

C &foo();

void foox() {
  for (; ; foo());
}

