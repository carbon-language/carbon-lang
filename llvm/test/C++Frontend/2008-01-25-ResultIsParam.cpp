// RUN: %llvmgcc %s -S -o - | not grep {@_ZN3fooC1Ev.*result}
// PR1942

class foo
{
public:
  int a;
  int b;

  foo(void) : a(0), b(0) {}

  foo(int aa, int bb) : a(aa), b(bb) {}

  const foo operator+(const foo& in) const;

};

const foo foo::operator+(const foo& in) const {
  foo Out;
  Out.a = a + in.a;
  Out.b = b + in.b;
  return Out;
}
