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

template<typename T> struct TemplateInstantiationVisibility { typedef int type; };

template<typename T> struct Outer {
  template<typename U> struct Inner {
    static constexpr int f();
    static constexpr int g();
  };
};

template<typename T> struct WithPartialSpecialization {};
typedef WithPartialSpecialization<int*> WithPartialSpecializationUse;

template<typename T> struct WithExplicitSpecialization;
typedef WithExplicitSpecialization<int> WithExplicitSpecializationUse;
