// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-ELIDE-NOTREE
// RUN: %clang_cc1 -fsyntax-only %s -fno-elide-type -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-NOELIDE-NOTREE
// RUN: %clang_cc1 -fsyntax-only %s -fdiagnostics-show-template-tree -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-ELIDE-TREE
// RUN: %clang_cc1 -fsyntax-only %s -fno-elide-type -fdiagnostics-show-template-tree -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-NOELIDE-TREE

// PR9548 - "no known conversion from 'vector<string>' to 'vector<string>'"
// vector<string> refers to two different types here.  Make sure the message
// gives a way to tell them apart.
class versa_string;
typedef versa_string string;

namespace std {template <typename T> class vector;}
using std::vector;

void f(vector<string> v);

namespace std {
  class basic_string;
  typedef basic_string string;
  template <typename T> class vector {};
  void g() {
    vector<string> v;
    f(v);
  }
} // end namespace std
// CHECK-ELIDE-NOTREE: no matching function for call to 'f'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<class std::basic_string>' to 'vector<class versa_string>' for 1st argument 
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<class std::basic_string>' to 'vector<class versa_string>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [class std::basic_string != class versa_string]>
// CHECK-NOELIDE-TREE: no matching function for call to 'f'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [class std::basic_string != class versa_string]>

template <int... A>
class I1{};
void set1(I1<1,2,3,4,2,3,4,3>) {};
void test1() {
  set1(I1<1,2,3,4,2,2,4,3,7>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set1'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I1<[5 * ...], 2, [2 * ...], 7>' to 'I1<[5 * ...], 3, [2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set1'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I1<1, 2, 3, 4, 2, 2, 4, 3, 7>' to 'I1<1, 2, 3, 4, 2, 3, 4, 3, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set1'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I1<
// CHECK-ELIDE-TREE:     [5 * ...],
// CHECK-ELIDE-TREE:     [2 != 3],
// CHECK-ELIDE-TREE:     [2 * ...],
// CHECK-ELIDE-TREE:     [7 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set1'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I1<
// CHECK-NOELIDE-TREE:     1,
// CHECK-NOELIDE-TREE:     2,
// CHECK-NOELIDE-TREE:     3,
// CHECK-NOELIDE-TREE:     4,
// CHECK-NOELIDE-TREE:     2,
// CHECK-NOELIDE-TREE:     [2 != 3],
// CHECK-NOELIDE-TREE:     4,
// CHECK-NOELIDE-TREE:     3,
// CHECK-NOELIDE-TREE:     [7 != (no argument)]>

template <class A, class B, class C = void>
class I2{};
void set2(I2<int, int>) {};
void test2() {
  set2(I2<double, int, int>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set2'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I2<double, [...], int>' to 'I2<int, [...], (default) void>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set2'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I2<double, int, int>' to 'I2<int, int, (default) void>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set2'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I2<
// CHECK-ELIDE-TREE:     [double != int],
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [int != (default) void]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set2'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I2<
// CHECK-NOELIDE-TREE:     [double != int],
// CHECK-NOELIDE-TREE:     int,
// CHECK-NOELIDE-TREE:     [int != (default) void]>

int V1, V2, V3;
template <int* A, int *B>
class I3{};
void set3(I3<&V1, &V2>) {};
void test3() {
  set3(I3<&V3, &V2>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set3'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I3<&V3, [...]>' to 'I3<&V1, [...]>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set3'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I3<&V3, &V2>' to 'I3<&V1, &V2>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set3'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I3<
// CHECK-ELIDE-TREE:     [&V3 != &V1]
// CHECK-ELIDE-TREE:     [...]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set3'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I3<
// CHECK-NOELIDE-TREE:     [&V3 != &V1]
// CHECK-NOELIDE-TREE:     &V2>

template <class A, class B>
class Alpha{};
template <class A, class B>
class Beta{};
template <class A, class B>
class Gamma{};
template <class A, class B>
class Delta{};

void set4(Alpha<int, int>);
void test4() {
  set4(Beta<void, void>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set4'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set4'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set4'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-NOELIDE-TREE: no matching function for call to 'set4'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument

void set5(Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>);
void test5() {
  set5(Alpha<Beta<Gamma<void, void>, double>, double>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Gamma<void, void>, double>, double>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Gamma<void, void>, double>, double>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set5'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Alpha<
// CHECK-ELIDE-TREE:     Beta<
// CHECK-ELIDE-TREE:       Gamma<
// CHECK-ELIDE-TREE:         [void != Delta<int, int>],
// CHECK-ELIDE-TREE:         [void != int]>
// CHECK-ELIDE-TREE:       [double != int]>
// CHECK-ELIDE-TREE:     [double != int]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set5'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Alpha<
// CHECK-NOELIDE-TREE:     Beta<
// CHECK-NOELIDE-TREE:       Gamma<
// CHECK-NOELIDE-TREE:         [void != Delta<int, int>],
// CHECK-NOELIDE-TREE:         [void != int]>
// CHECK-NOELIDE-TREE:       [double != int]>
// CHECK-NOELIDE-TREE:     [double != int]>

void test6() {
  set5(Alpha<Beta<Delta<int, int>, int>, int>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Delta<int, int>, [...]>, [...]>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, [...]>, [...]>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Delta<int, int>, int>, int>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set5'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Alpha<
// CHECK-ELIDE-TREE:     Beta<
// CHECK-ELIDE-TREE:       [Delta<int, int> != Gamma<Delta<int, int>, int>],
// CHECK-ELIDE-TREE:       [...]>
// CHECK-ELIDE-TREE:     [...]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set5'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Alpha<
// CHECK-NOELIDE-TREE:     Beta<
// CHECK-NOELIDE-TREE:       [Delta<int, int> != Gamma<Delta<int, int>, int>],
// CHECK-NOELIDE-TREE:       int>
// CHECK-NOELIDE-TREE:     int>

int a7, b7;
int c7[] = {1,2,3};
template<int *A>
class class7 {};
void set7(class7<&a7> A) {}
void test7() {
  set7(class7<&a7>());
  set7(class7<&b7>());
  set7(class7<c7>());
  set7(class7<nullptr>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<&b7>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<c7>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<nullptr>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<&b7>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<c7>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<nullptr>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [&b7 != &a7]>
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [c7 != &a7]>
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [nullptr != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [&b7 != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [c7 != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [nullptr != &a7]>

template<typename ...T> struct S8 {};
template<typename T> using U8 = S8<int, char, T>;
int f8(S8<int, char, double>);
int k8 = f8(U8<char>());
// CHECK-ELIDE-NOTREE: no matching function for call to 'f8'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'S8<[2 * ...], char>' to 'S8<[2 * ...], double>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f8'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'S8<int, char, char>' to 'S8<int, char, double>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f8'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   S8<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [char != double]>
// CHECK-NOELIDE-TREE: no matching function for call to 'f8'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   S8<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     char, 
// CHECK-NOELIDE-TREE:     [char != double]>

template<typename ...T> struct S9 {};
template<typename T> using U9 = S9<int, char, T>;
template<typename T> using V9 = U9<U9<T>>;
int f9(S9<int, char, U9<const double>>);
int k9 = f9(V9<double>());

// CHECK-ELIDE-NOTREE: no matching function for call to 'f9'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'S9<[2 * ...], S9<[2 * ...], double>>' to 'S9<[2 * ...], S9<[2 * ...], const double>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f9'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'S9<int, char, S9<int, char, double>>' to 'S9<int, char, S9<int, char, const double>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f9'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   S9<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     S9<
// CHECK-ELIDE-TREE:       [2 * ...], 
// CHECK-ELIDE-TREE:       [double != const double]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'f9'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   S9<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     char, 
// CHECK-NOELIDE-TREE:     S9<
// CHECK-NOELIDE-TREE:       int, 
// CHECK-NOELIDE-TREE:       char, 
// CHECK-NOELIDE-TREE:       [double != const double]>>

template<typename ...A> class class_types {};
void set10(class_types<int, int>) {}
void test10() {
  set10(class_types<int>());
  set10(class_types<int, int, int>());
}

// CHECK-ELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<[...], (no argument)>' to 'class_types<[...], int>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<[2 * ...], int>' to 'class_types<[2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<int, (no argument)>' to 'class_types<int, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<int, int, int>' to 'class_types<int, int, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set10'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_types<
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [(no argument) != int]>
// CHECK-ELIDE-TREE: no matching function for call to 'set10'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_types<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [int != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set10'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_types<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     [(no argument) != int]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set10'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_types<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     [int != (no argument)]>

template<int ...A> class class_ints {};
void set11(class_ints<2, 3>) {}
void test11() {
  set11(class_ints<1>());
  set11(class_ints<0, 3, 6>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<1, (no argument)>' to 'class_ints<2, 3>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<0, [...], 6>' to 'class_ints<2, [...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<1, (no argument)>' to 'class_ints<2, 3>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<0, 3, 6>' to 'class_ints<2, 3, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set11'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ints<
// CHECK-ELIDE-TREE:     [1 != 2], 
// CHECK-ELIDE-TREE:     [(no argument) != 3]>
// CHECK-ELIDE-TREE: no matching function for call to 'set11'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ints<
// CHECK-ELIDE-TREE:     [0 != 2], 
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [6 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set11'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ints<
// CHECK-NOELIDE-TREE:     [1 != 2], 
// CHECK-NOELIDE-TREE:     [(no argument) != 3]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set11'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ints<
// CHECK-NOELIDE-TREE:     [0 != 2], 
// CHECK-NOELIDE-TREE:     3, 
// CHECK-NOELIDE-TREE:     [6 != (no argument)]>

template<template<class> class ...A> class class_template_templates {};
template<class> class tt1 {};
template<class> class tt2 {};
void set12(class_template_templates<tt1, tt1>) {}
void test12() {
  set12(class_template_templates<tt2>());
  set12(class_template_templates<tt1, tt1, tt1>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt2, template (no argument)>' to 'class_template_templates<template tt1, template tt1>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<[2 * ...], template tt1>' to 'class_template_templates<[2 * ...], template (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt2, template (no argument)>' to 'class_template_templates<template tt1, template tt1>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt1, template tt1, template tt1>' to 'class_template_templates<template tt1, template tt1, template (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set12'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_template_templates<
// CHECK-ELIDE-TREE:     [template tt2 != template tt1], 
// CHECK-ELIDE-TREE:     [template (no argument) != template tt1]>
// CHECK-ELIDE-TREE: no matching function for call to 'set12'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_template_templates<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [template tt1 != template (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set12'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_template_templates<
// CHECK-NOELIDE-TREE:     [template tt2 != template tt1], 
// CHECK-NOELIDE-TREE:     [template (no argument) != template tt1]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set12'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_template_templates<
// CHECK-NOELIDE-TREE:     template tt1, 
// CHECK-NOELIDE-TREE:     template tt1, 
// CHECK-NOELIDE-TREE:     [template tt1 != template (no argument)]>

double a13, b13, c13, d13;
template<double* ...A> class class_ptrs {};
void set13(class_ptrs<&a13, &b13>) {}
void test13() {
  set13(class_ptrs<&c13>());
  set13(class_ptrss<&a13, &b13, &d13>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&c13, (no argument)>' to 'class_ptrs<&a13, &b13>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<[2 * ...], &d13>' to 'class_ptrs<[2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&c13, (no argument)>' to 'class_ptrs<&a13, &b13>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&a13, &b13, &d13>' to 'class_ptrs<&a13, &b13, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set13'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ptrs<
// CHECK-ELIDE-TREE:     [&c13 != &a13], 
// CHECK-ELIDE-TREE:     [(no argument) != &b13]>
// CHECK-ELIDE-TREE: no matching function for call to 'set13'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ptrs<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [&d13 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set13'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ptrs<
// CHECK-NOELIDE-TREE:     [&c13 != &a13], 
// CHECK-NOELIDE-TREE:     [(no argument) != &b13]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set13'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ptrs<
// CHECK-NOELIDE-TREE:     &a13, 
// CHECK-NOELIDE-TREE:     &b13, 
// CHECK-NOELIDE-TREE:     [&d13 != (no argument)]>


// CHECK-ELIDE-NOTREE: {{[0-9]*}} errors generated.
// CHECK-NOELIDE-NOTREE: {{[0-9]*}} errors generated.
// CHECK-ELIDE-TREE: {{[0-9]*}} errors generated.
// CHECK-NOELIDE-TREE: {{[0-9]*}} errors generated.
