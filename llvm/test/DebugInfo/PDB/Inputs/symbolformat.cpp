// Compile with "cl /c /Zi /GR- symbolformat.cpp"
// Compile symbolformat-fpo.cpp (see file for instructions)
// Link with "link symbolformat.obj symbolformat-fpo.obj /debug /nodefaultlib
//    /entry:main /out:symbolformat.exe"

int __cdecl _purecall(void) { return 0; }

struct A {
  virtual void PureFunc() = 0 {}
  virtual void VirtualFunc() {}
  void RegularFunc() {}
};

struct B : public A {
  void PureFunc() override {}
};

int main(int argc, char **argv) {
  B b;
  auto PureAddr = &B::PureFunc;
  auto VirtualAddr = &A::PureFunc;
  auto RegularAddr = &A::RegularFunc;
  return 0;
}
