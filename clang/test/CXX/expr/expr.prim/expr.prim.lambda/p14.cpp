// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

template<typename T> void capture(const T&);

class NonCopyable {
  NonCopyable(const NonCopyable&); // expected-note 2 {{implicitly declared private here}}
};

void capture_by_copy(NonCopyable nc, NonCopyable &ncr) {
  // FIXME: error messages should talk about capture
  (void)[nc] { }; // expected-error{{field of type 'NonCopyable' has private copy constructor}}
  (void)[ncr] { }; // expected-error{{field of type 'NonCopyable' has private copy constructor}}
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
