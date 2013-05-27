// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -O1 -disable-llvm-optzns %s -o - | FileCheck %s

namespace test1 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test11fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test2 {
  // CHECK-DAG: define internal void @_ZN5test21fIZNS_L1gEvE1S_0EEvT_(
  template <typename T> void f(T) {}
  static inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test3 {
  // CHECK-DAG: define internal void @_ZN5test31fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test4 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test41fIZNS_1gILi1EEEPvvE1SEEvT_(
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
  // CHECK-DAG: define linkonce_odr void @_ZN5test51fIZNS_1gILi1EEEPvvE1S_1EEvT_(
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
  // CHECK-DAG: define linkonce_odr void @_ZN5test61fIZZNS_1gEvEN1S1hE_2vE1T_3EEvv(
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
  // CHECK-DAG: define internal void @_ZN5test71fIZZNS_1gEvEN1S1hEvE1T_4EEvv(
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

namespace test8 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test81fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    enum S {
    };
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test9 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test91fIPZNS_1gEvE1S_5EEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S*>);
  }
  void *h() { return g(); }
}

namespace test10 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test101fIPFZNS_1gEvE1S_6vEEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    typedef S(*ftype)();
    return reinterpret_cast<void *>(f<ftype>);
  }
  void *h() { return g(); }
}

namespace test11 {
  // CHECK-DAG: define internal void @_ZN6test111fIPFZNS_1gEvE1S_7PNS_12_GLOBAL__N_11IEEEEvT_(
  namespace {
    struct I {
    };
  }

  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    };
    typedef S(*ftype)(I * x);
    return reinterpret_cast<void *>(f<ftype>);
  }
  void *h() { return g(); }
}

namespace test12 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test123fooIZNS_3barIZNS_3zedEvE2S2EEPvvE2S1EEvv
  template <typename T> void foo() {}
  template <typename T> inline void *bar() {
    enum S1 {
    };
    return reinterpret_cast<void *>(foo<S1>);
  }
  inline void *zed() {
    enum S2 {
    };
    return reinterpret_cast<void *>(bar<S2>);
  }
  void *h() { return zed(); }
}
