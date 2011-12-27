// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - | FileCheck %s
#include <typeinfo>

namespace Test1 {

struct Item {
  const std::type_info &ti;
  const char *name;
  void *(*make)();
};

template<typename T> void *make_impl() { return new T; }
template<typename T> constexpr Item item(const char *name) {
  return { typeid(T), name, make_impl<T> };
}

struct A { virtual ~A(); };
struct B : virtual A {};
struct C { int n; };

// FIXME: check we produce a constant array for this, once we support IRGen of
// folded structs and arrays.
constexpr Item items[] = {
  item<A>("A"), item<B>("B"), item<C>("C"), item<int>("int")
};

// CHECK: @_ZN5Test11xE = constant %"class.std::type_info"* bitcast (i8** @_ZTIN5Test11AE to %"class.std::type_info"*), align 8
constexpr auto &x = items[0].ti;

}
