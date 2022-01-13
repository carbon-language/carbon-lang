// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

template<typename T> void capture(const T&);

class NonCopyable {
  NonCopyable(const NonCopyable&); // expected-note 2 {{implicitly declared private here}}
public:
  void foo() const;
};

class NonConstCopy {
public:
  NonConstCopy(NonConstCopy&); // expected-note{{would lose const}}
};

void capture_by_copy(NonCopyable nc, NonCopyable &ncr, const NonConstCopy nco) {
  (void)[nc] { }; // expected-error{{capture of variable 'nc' as type 'NonCopyable' calls private copy constructor}}
  (void)[=] { // expected-error{{capture of variable 'ncr' as type 'NonCopyable' calls private copy constructor}}
    ncr.foo();
  }();

  [nco] {}(); // expected-error{{no matching constructor for initialization of 'const NonConstCopy'}}
}

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &);
  ~NonTrivial();
};

struct CopyCtorDefault {
  CopyCtorDefault();
  CopyCtorDefault(const CopyCtorDefault&, NonTrivial nt = NonTrivial());

  void foo() const;
};

void capture_with_default_args(CopyCtorDefault cct) {
  (void)[=] () -> void { cct.foo(); };
}

struct ExpectedArrayLayout {
  CopyCtorDefault array[3];
};

void capture_array() {
  CopyCtorDefault array[3];
  auto x = [=]() -> void {
    capture(array[0]);
  };
  static_assert(sizeof(x) == sizeof(ExpectedArrayLayout), "layout mismatch");
}

// Check for the expected non-static data members.

struct ExpectedLayout {
  char a;
  short b;
};

void test_layout(char a, short b) {
  auto x = [=] () -> void {
    capture(a);
    capture(b);
  };
  static_assert(sizeof(x) == sizeof(ExpectedLayout), "Layout mismatch!");
}

struct ExpectedThisLayout {
  ExpectedThisLayout* a;
  void f() {
    auto x = [this]() -> void {};
    static_assert(sizeof(x) == sizeof(ExpectedThisLayout), "Layout mismatch!");
  }
};

struct CaptureArrayAndThis {
  int value;

  void f() {
    int array[3];
    [=]() -> int {
      int result = value;
      for (unsigned i = 0; i < 3; ++i)
        result += array[i];
      return result;
    }();
  }
};

namespace rdar14468891 {
  class X {
  public:
    virtual ~X() = 0; // expected-note{{unimplemented pure virtual method '~X' in 'X'}}
  };

  class Y : public X { };

  void capture(X &x) {
    [x]() {}(); // expected-error{{by-copy capture of value of abstract type 'rdar14468891::X'}}
  }
}

namespace rdar15560464 {
  struct X; // expected-note{{forward declaration of 'rdar15560464::X'}}
  void foo(const X& param) {
    auto x = ([=]() {
        auto& y = param; // expected-error{{by-copy capture of variable 'param' with incomplete type 'const rdar15560464::X'}}
      });
  }
}
