// RUN: clang -fsyntax-only -verify %s

// PR5727
namespace test0 {
  template<typename> struct RefPtr { };
  template<typename> struct PtrHash {
    static void f() { }
  };
  template<typename T> struct PtrHash<RefPtr<T> > : PtrHash<T*> {
    using PtrHash<T*>::f;
    static void f() { f(); }
  };
}
