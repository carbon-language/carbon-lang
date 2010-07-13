// RUN: %clang_cc1 -g -S -masm-verbose -x c++ -o %t %s
// RUN: grep DW_TAG_volatile_type %t | count 3
// RUN: grep DW_TAG_const_type %t | count 3
// one for decl, one for def, one for abbrev

namespace A {
  class B {
  public:
    void dump() const volatile;
  };
}

int main () {
  using namespace A;
  B b;
  return 0;
}

void A::B::dump() const volatile{
}
