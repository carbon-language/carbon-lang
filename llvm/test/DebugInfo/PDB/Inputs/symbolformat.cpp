// Compile with "cl /c /Zi /GR- symbolformat.cpp"
// Compile symbolformat-fpo.cpp (see file for instructions)
// Link with "link symbolformat.obj symbolformat-fpo.obj /debug /nodefaultlib
//    /entry:main /out:symbolformat.exe"

int __cdecl _purecall(void) { return 0; }

enum TestEnum {
  Value,
  Value10 = 10
};

enum class TestEnumClass {
  Value,
  Value10 = 10
};

struct A {
  virtual void PureFunc() = 0 {}
  virtual void VirtualFunc() {}
  void RegularFunc() {}
};

struct VirtualBase {
};

struct B : public A, protected virtual VirtualBase {
  void PureFunc() override {}

  enum NestedEnum {
    FirstVal,
    SecondVal
  };

  typedef int NestedTypedef;
  NestedEnum EnumVar;
  NestedTypedef TypedefVar;
};

typedef int IntType;
typedef A ClassAType;

int main(int argc, char **argv) {
  B b;
  auto PureAddr = &B::PureFunc;
  auto VirtualAddr = &A::PureFunc;
  auto RegularAddr = &A::RegularFunc;
  TestEnum Enum = Value;
  TestEnumClass EnumClass = TestEnumClass::Value10;
  IntType Int = 12;
  ClassAType *ClassA = &b;
  return 0;
}
