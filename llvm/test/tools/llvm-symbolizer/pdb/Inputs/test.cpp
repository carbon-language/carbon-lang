// To generate the corresponding EXE/PDB, run:
// cl /Zi test.cpp

namespace NS {
struct Foo {
  void bar() {}
};
}

void foo() {
}

int main() {
  foo();
  
  NS::Foo f;
  f.bar();
}
