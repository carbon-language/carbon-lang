struct DefinedInBImpl {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInBImpl);
};
