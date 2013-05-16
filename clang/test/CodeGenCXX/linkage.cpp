// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -O1 -disable-llvm-optzns %s -o - | FileCheck %s

// CHECK: define internal void @_ZN5test31fIZNS_1gEvE1SEEvT_(
// CHECK: define linkonce_odr void @_ZN5test41fIZNS_1gILi1EEEPvvE1SEEvT_(
// CHECK: define linkonce_odr void @_ZN5test51fIZNS_1gILi1EEEPvvE1S_1EEvT_(
// CHECK: define internal void @_ZN5test71fIZZNS_1gEvEN1S1hEvE1T_4EEvv(
// CHECK: define linkonce_odr void @_ZN5test61fIZZNS_1gEvEN1S1hE_2vE1T_3EEvv(
// CHECK: define internal void @_ZN5test21fIZNS_L1gEvE1S_0EEvT_(
// CHECK: define linkonce_odr void @_ZN5test11fIZNS_1gEvE1SEEvT_(

namespace test1 {
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test2 {
  template <typename T> void f(T) {}
  static inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test3 {
  template <typename T> void f(T) {}
  void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test4 {
  template <typename T> void f(T) {}
  template <int N> inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  extern template void *g<1>();
  template void *g<1>();
}

namespace test5 {
  template <typename T> void f(T) {}
  template <int N> inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  extern template void *g<1>();
  void *h() { return g<1>(); }
}

namespace test6 {
  template <typename T> void f() {}

  inline void *g() {
    struct S {
      void *h() {
        struct T {
        };
        return (void *)f<T>;
      }
    } s;
    return s.h();
  }

  void *h() { return g(); }
}

namespace test7 {
  template <typename T> void f() {}

  void *g() {
    struct S {
      void *h() {
        struct T {
        };
        return (void *)f<T>;
      }
    } s;
    return s.h();
  }

  void *h() { return g(); }
}
