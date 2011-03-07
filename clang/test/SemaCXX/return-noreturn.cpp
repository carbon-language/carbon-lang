// RUN: %clang_cc1 %s -fsyntax-only -verify -Wreturn-type -Wmissing-noreturn -Wno-unreachable-code

// A destructor may be marked noreturn and should still influence the CFG.
void pr6884_abort() __attribute__((noreturn));

struct pr6884_abort_struct {
  pr6884_abort_struct() {}
  ~pr6884_abort_struct() __attribute__((noreturn)) { pr6884_abort(); }
};

int pr6884_f(int x) {
  switch (x) { default: pr6884_abort(); }
}

int pr6884_g(int x) {
  switch (x) { default: pr6884_abort_struct(); }
}

int pr6884_g_positive(int x) {
  switch (x) { default: ; }
} // expected-warning {{control reaches end of non-void function}}

int pr6884_h(int x) {
  switch (x) {
    default: {
      pr6884_abort_struct a;
    }
  }
}

// PR9380
struct PR9380 {
  ~PR9380();
};
struct PR9380_B : public PR9380 {
  PR9380_B( const PR9380& str );
};
void test_PR9380(const PR9380& aKey) {
  const PR9380& flatKey = PR9380_B(aKey);
}

// Array of objects with destructors.  This is purely a coverage test case.
void test_array() {
  PR9380 a[2];
}

// Test classes wrapped in typedefs.  This is purely a coverage test case
// for CFGImplictDtor::getDestructorDecl().
void test_typedefs() {
  typedef PR9380 PR9380_Ty;
  PR9380_Ty test;
  PR9380_Ty test2[20];
}

// PR9412 - Handle CFG traversal with null successors.
enum PR9412_MatchType { PR9412_Exact };

template <PR9412_MatchType type> int PR9412_t() {
  switch (type) {
    case PR9412_Exact:
    default:
        break;
  }
} // expected-warning {{control reaches end of non-void function}}

void PR9412_f() {
    PR9412_t<PR9412_Exact>(); // expected-note {{in instantiation of function template specialization 'PR9412_t<0>' requested here}}
}

