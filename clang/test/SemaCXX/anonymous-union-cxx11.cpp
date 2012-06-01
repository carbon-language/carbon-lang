// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s

namespace PR12866 {
  struct bar {
    union {
      int member;
    };
  };

  void foo( void ) {
    (void)sizeof(bar::member);
  }
}
