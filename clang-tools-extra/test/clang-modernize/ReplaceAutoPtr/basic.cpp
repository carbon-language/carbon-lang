// RUN: mkdir -p %T/Inputs
//
// Without inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic.h > %T/Inputs/basic.h
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/memory_stub.h > %T/Inputs/memory_stub.h
// RUN: clang-modernize -include=%T -replace-auto_ptr %t.cpp -- \
// RUN:               -std=c++11 -I %T
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: FileCheck -input-file=%T/Inputs/basic.h %S/Inputs/basic.h
//
// With inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic.h > %T/Inputs/basic.h
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/memory_stub.h > %T/Inputs/memory_stub.h
// RUN: clang-modernize -include=%T -replace-auto_ptr %t.cpp -- \
// RUN:               -DUSE_INLINE_NAMESPACE=1 -std=c++11 -I %T
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: FileCheck -input-file=%T/Inputs/basic.h %S/Inputs/basic.h

#include "Inputs/basic.h"

void f_1() {
  std::auto_ptr<int> a;
  // CHECK: std::unique_ptr<int> a;

  // check that spaces aren't modified unnecessarily
  std:: auto_ptr <int> b;
  // CHECK: std:: unique_ptr <int> b;
  std :: auto_ptr < char > c(new char());
  // CHECK: std :: unique_ptr < char > c(new char());

  // Test construction from a temporary
  std::auto_ptr<char> d = std::auto_ptr<char>();
  // CHECK: std::unique_ptr<char> d = std::unique_ptr<char>();

  typedef std::auto_ptr<int> int_ptr_t;
  // CHECK: typedef std::unique_ptr<int> int_ptr_t;
  int_ptr_t e(new int());
  // CHECK: int_ptr_t e(new int());

  // Test pointers
  std::auto_ptr<int> *f;
  // CHECK: std::unique_ptr<int> *f;

  // Test 'static' declarations
  static std::auto_ptr<int> g;
  // CHECK: static std::unique_ptr<int> g;

  // Test with cv-qualifiers
  const std::auto_ptr<int> h;
  // CHECK: const std::unique_ptr<int> h;
  volatile std::auto_ptr<int> i;
  // CHECK: volatile std::unique_ptr<int> i;
  const volatile std::auto_ptr<int> j;
  // CHECK: const volatile std::unique_ptr<int> j;

  // Test auto and initializer-list
  auto k = std::auto_ptr<int>{};
  // CHECK: auto k = std::unique_ptr<int>{};
  std::auto_ptr<int> l{std::auto_ptr<int>()};
  // CHECK: std::unique_ptr<int> l{std::unique_ptr<int>()};

  // Test interlocked auto_ptr
  std::auto_ptr<std::auto_ptr<int> > m;
  // CHECK: std::unique_ptr<std::unique_ptr<int> > m;

  // Test temporaries
  std::auto_ptr<char>();
  // CHECK: std::unique_ptr<char>();

  // Test void-specialization
  std::auto_ptr<void> n;
  // CHECK: std::unique_ptr<void> n;

  // Test template WITH instantiation (instantiation)
  B<double> o;
  std::auto_ptr<double> p(o.create());
  // CHECK: std::unique_ptr<double> p(o.create());

  // Test 'using' in a namespace ("definition")
  ns_1::auto_ptr<int> q;
  // CHECK: ns_1::unique_ptr<int> q;

  // Test construction with an 'auto_ptr_ref'
  std::auto_ptr<Base> r(create_derived_ptr());
  // CHECK: std::unique_ptr<Base> r(create_derived_ptr());
}

// Test without the nested name specifiers
void f_2() {
  using namespace std;

  auto_ptr<int> a;
  // CHECK: unique_ptr<int> a;
}

// Test using declaration
void f_3() {
  using std::auto_ptr;
  // CHECK: using std::unique_ptr;

  auto_ptr<int> a;
  // CHECK: unique_ptr<int> a;
}

// Test messing-up with macros
void f_4() {
#define MACRO_1 <char>
  std::auto_ptr MACRO_1 p(new char());
// CHECK: std::unique_ptr MACRO_1 p(new char());
#define MACRO_2 auto_ptr
  std::MACRO_2<int> q;
// CHECK: #define MACRO_2 unique_ptr
#define MACRO_3(Type) std::auto_ptr<Type>
  MACRO_3(float)r(new float());
// CHECK: #define MACRO_3(Type) std::unique_ptr<Type>
#define MACRO_4 std::auto_ptr
  using MACRO_4;
// CHECK: #define MACRO_4 std::unique_ptr
#undef MACRO_1
#undef MACRO_2
#undef MACRO_3
#undef MACRO_4
}

// Test function return values (definition)
std::auto_ptr<char> f_5()
// CHECK: std::unique_ptr<char> f_5()
{
  // Test constructor
  return std::auto_ptr<char>(new char());
  // CHECK: return std::unique_ptr<char>(new char());
}

// Test that non-std auto_ptr aren't replaced
void f_8() {
  ns_2::auto_ptr<char> a;
  // CHECK: ns_2::auto_ptr<char> a;
  using namespace ns_2;
  auto_ptr<int> b;
  // CHECK: auto_ptr<int> b;
}

namespace std {
template <typename T> using aaaaaaaa = auto_ptr<T>;
}
// We want to avoid replacing 'aaaaaaaa' by unique_ptr here. It's better to
// change the type alias directly.
// XXX: maybe another test will be more relevant to test this potential error.
std::aaaaaaaa<int> d;
// CHECK: std::aaaaaaaa<int> d;
