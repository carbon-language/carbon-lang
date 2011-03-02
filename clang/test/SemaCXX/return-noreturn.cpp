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
