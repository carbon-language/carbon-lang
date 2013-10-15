template<typename T> struct SomeTemplate {};

struct DefinedInCommon {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInCommon);
};

template<typename T> struct CommonTemplate {
  enum E { a = 1, b = 2, c = 3 };
};
