// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

namespace std {
  struct A { A(); };
  
  // CHECK: define void @_ZNSt1AC1Ev
  // CHECK: define void @_ZNSt1AC2Ev
  A::A() { }
};

namespace std {
  template<typename> struct allocator { };
}

// CHECK: define void @_Z1fSaIcESaIiE
void f(std::allocator<char>, std::allocator<int>) { }

namespace std {
  template<typename, typename, typename> struct basic_string { };
}

// CHECK: define void @_Z1fSbIcciE
void f(std::basic_string<char, char, int>) { }

namespace std {
  template<typename> struct char_traits { };
  
  typedef std::basic_string<char, std::char_traits<char>, std::allocator<char> > string;
}

// CHECK: _Z1fSs
void f(std::string) { }

namespace std {
  template<typename, typename> struct basic_istream { };
  template<typename, typename> struct basic_ostream { };
  template<typename, typename> struct basic_iostream { };
}

// CHECK: _Z1fSi
void f(std::basic_istream<char, std::char_traits<char> >) { }

// CHECK: _Z1fSo
void f(std::basic_ostream<char, std::char_traits<char> >) { }

// CHECK: _Z1fSd
void f(std::basic_iostream<char, std::char_traits<char> >) { }

extern "C++" {
namespace std
{
  typedef void (*terminate_handler) ();
  
  // CHECK: _ZSt13set_terminatePFvvE
  terminate_handler set_terminate(terminate_handler) { return 0; }
}
}
