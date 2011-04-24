// no PCH
// RUN: %clang_cc1 -include %s -include %s -fsyntax-only %s
// with PCH
// RUN: %clang_cc1 -chain-include %s -chain-include %s -fsyntax-only %s
#if !defined(PASS1)
#define PASS1

namespace ns {}

#elif !defined(PASS2)
#define PASS2

namespace ns {
  namespace {
    extern int x;
  }
}

namespace {
  extern int y;
}

#else

namespace ns {
  namespace {
    int x;
  }
  void test() {
    (void)x;
  }
}

namespace {
  int y;
}
void test() {
  (void)y;
}

#endif
