// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int N>
struct X {
  struct __attribute__((__aligned__((N)))) Aligned { }; // expected-error{{'aligned' attribute requires integer constant}}

  int __attribute__((__address_space__(N))) *ptr; // expected-error{{attribute requires 1 argument(s)}}
};

namespace PR7102 {

  class NotTpl {
  public:
    union {
      char space[11];
      void* ptr;
    }  __attribute__((packed));
  };
  template<unsigned N>
  class Tpl {
  public:
    union {
      char space[N];
      void* ptr;
    }  __attribute__((packed));
  };

  int array[sizeof(NotTpl) == sizeof(Tpl<11>)? 1 : -1];
}
