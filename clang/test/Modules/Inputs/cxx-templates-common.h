template<typename T> struct SomeTemplate {};

struct DefinedInCommon {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInCommon);
};
