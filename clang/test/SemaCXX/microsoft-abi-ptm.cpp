// RUN: %clang_cc1 -triple %ms_abi_triple -std=c++11 -fsyntax-only -fms-extensions -fms-compatibility %s

namespace Test1 {
struct A;
void f(int A::*mp);
struct A { };
static_assert(sizeof(int A::*) == sizeof(int), "pointer-to-member should be sizeof(int)");
}

namespace Test2 {
struct A;
void f(int A::*mp);
static_assert(sizeof(int A::*) == sizeof(int) * 3, "pointer-to-member should be sizeof(int) * 3");
struct A { };
static_assert(sizeof(int A::*) == sizeof(int) * 3, "pointer-to-member should still be sizeof(int) * 3");
}

namespace Test3 {
struct A;
typedef int A::*p;
struct __single_inheritance A;
p my_ptr;
static_assert(sizeof(int A::*) == sizeof(int), "pointer-to-member should be sizeof(int)");
}
