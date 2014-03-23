template<typename T> struct SomeTemplate {};

struct DefinedInCommon {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInCommon);
};

template<typename T> struct CommonTemplate {
  enum E { a = 1, b = 2, c = 3 };
};

namespace Std {
  template<typename T> struct WithFriend {
    friend bool operator!=(const WithFriend &A, const WithFriend &B) { return false; }
  };
}

namespace Std {
  template<typename T> void f() {
    extern T g();
  }
}
