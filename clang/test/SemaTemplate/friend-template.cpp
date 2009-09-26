// RUN: clang-cc -fsyntax-only %s

// PR5057
namespace std {
  class X {
  public:
    template<typename T>
    friend struct Y;
  };
}

namespace std {
  template<typename T>
  struct Y
  {
  };
}
