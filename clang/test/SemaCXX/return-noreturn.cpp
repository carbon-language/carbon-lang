// RUN: %clang_cc1 %s -fsyntax-only -verify -Wmissing-noreturn -Wno-unreachable-code

// A destructor may be marked noreturn and should still influence the CFG.
namespace PR6884 {
  struct abort_struct {
    abort_struct() {} // Make this non-POD so the destructor is invoked.
    ~abort_struct() __attribute__((noreturn));
  };

  // FIXME: Should either of these actually warn, since the destructor is
  //  marked noreturn?

  int f() {
    abort_struct();
  } // expected-warning{{control reaches end of non-void function}}

  int f2() {
    abort_struct s;
  } // expected-warning{{control reaches end of non-void function}}
}
