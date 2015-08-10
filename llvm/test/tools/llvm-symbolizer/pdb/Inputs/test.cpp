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

extern "C" {
void __cdecl foo_cdecl() {}
void __stdcall foo_stdcall() {}
void __fastcall foo_fastcall() {}
void __vectorcall foo_vectorcall() {}
}
