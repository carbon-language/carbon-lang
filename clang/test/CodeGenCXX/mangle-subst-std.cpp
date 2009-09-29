// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

namespace std {
  struct A { A(); };
  
  // CHECK: define void @_ZNSt1AC1Ev
  // CHECK: define void @_ZNSt1AC2Ev
  A::A() { }
};

namespace std {
  template<typename T> struct allocator { };
}

// CHECK: define void @_Z1fSaIcESaIiE
void f(std::allocator<char>, std::allocator<int>) { }

namespace std {
  template<typename T, typename U, typename V> struct basic_string { };
}

// CHECK: define void @_Z1fSbIcciE
void f(std::basic_string<char, char, int>) { }

namespace std {
  template<typename T> struct char_traits { };
  
  typedef std::basic_string<char, std::char_traits<char>, std::allocator<char> > string;
}

// CHECK: _Z1fSs
void f(std::string) { }
