#ifndef INPUTS_BASIC_H
#define INPUTS_BASIC_H

#include "memory_stub.h"

// Instrumentation for auto_ptr_ref test
// @{
struct Base {};
struct Derived : Base {};
std::auto_ptr<Derived> create_derived_ptr();
// CHECK: std::unique_ptr<Derived> create_derived_ptr();
// }

// Test function return values (declaration)
std::auto_ptr<char> f_5();
// CHECK: std::unique_ptr<char> f_5()

// Test function parameters
void f_6(std::auto_ptr<int>);
// CHECK: void f_6(std::unique_ptr<int>);
void f_7(const std::auto_ptr<int> &);
// CHECK: void f_7(const std::unique_ptr<int> &);

// Test on record type fields
struct A {
  std::auto_ptr<int> field;
  // CHECK: std::unique_ptr<int> field;

  typedef std::auto_ptr<int> int_ptr_type;
  // CHECK: typedef std::unique_ptr<int> int_ptr_type;
};

// Test template WITH instantiation
template <typename T> struct B {
  typedef typename std::auto_ptr<T> created_type;
  // CHECK: typedef typename std::unique_ptr<T> created_type;

  created_type create() { return std::auto_ptr<T>(new T()); }
  // CHECK: created_type create() { return std::unique_ptr<T>(new T()); }
};

// Test 'using' in a namespace (declaration)
namespace ns_1 {
// Test multiple using declarations
using std::auto_ptr;
using std::auto_ptr;
// CHECK: using std::unique_ptr;
// CHECK-NEXT: using std::unique_ptr;
}

namespace ns_2 {
template <typename T> struct auto_ptr {};
// CHECK: template <typename T> struct auto_ptr {};
}

#endif // INPUTS_BASIC_H
