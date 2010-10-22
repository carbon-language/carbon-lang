// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR8430
namespace N {
  class A { };
}

namespace M { }

using namespace M;

namespace N {
  namespace M {
  } 
}

namespace M {
  namespace N {
  }
}

namespace N {
  A *getA();
}
