// RUN: %clang_cc1 -no-opaque-pointers %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -triple=i686-pc-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-pc-windows-msvc  -fms-extensions -emit-llvm -o - | FileCheck %s

struct S {
  S();
  ~S();
};

template <typename T> struct __declspec(dllexport) ExportedTemplate {
  static S s;
};
template <typename T> S ExportedTemplate<T>::s;
void useExportedTemplate(ExportedTemplate<int> x) {
  (void)x.s;
}
int f();
namespace selectany_init {
// MS don't put selectany static var in the linker directive, init routine
// f() is not getting called if x is not referenced.
int __declspec(selectany) x = f();
inline int __declspec(selectany) x1 = f();
}

namespace explicit_template_instantiation {
template <typename T> struct A { static  int x; };
template <typename T> int A<T>::x = f();
template struct A<int>;
}

namespace implicit_template_instantiation {
template <typename T> struct A { static  int x; };
template <typename T>  int A<T>::x = f();
int g() { return A<int>::x; }
}


template <class T>
struct X_ {
  static T ioo;
  static T init();
};
template <class T> T X_<T>::ioo = X_<T>::init();
template struct X_<int>;

template <class T>
struct X {
  static T ioo;
  static T init();
};
// template specialized static data don't need in llvm.used,
// the static init routine get call from _GLOBAL__sub_I_ routines.
template <> int X<int>::ioo = X<int>::init();
template struct X<int>;
class a {
public:
  a();
};
// For the static var inside unnamed namespace, the object is local to TU.
// No need to put static var in the linker directive.
// The static init routine is called before main.
namespace {
template <int> class aj {
public:
  static a al;
};
template <int am> a aj<am>::al;
class b : aj<3> {
  void c();
};
void b::c() { al; }
}

// C++17, inline static data member also need to use
struct A
{
  A();
  ~A();
};

struct S1
{
  inline static A aoo; // C++17 inline variable, thus also a definition
};

int foo();
inline int zoo = foo();
inline static int boo = foo();

// CHECK: @llvm.used = appending global [8 x i8*] [i8* bitcast (i32* @"?x@selectany_init@@3HA" to i8*), i8* bitcast (i32* @"?x1@selectany_init@@3HA" to i8*), i8* bitcast (i32* @"?x@?$A@H@explicit_template_instantiation@@2HA" to i8*), i8* bitcast (i32* @"?ioo@?$X_@H@@2HA" to i8*), i8* getelementptr inbounds (%struct.A, %struct.A* @"?aoo@S1@@2UA@@A", i32 0, i32 0), i8* bitcast (i32* @"?zoo@@3HA" to i8*), i8* getelementptr inbounds (%struct.S, %struct.S* @"?s@?$ExportedTemplate@H@@2US@@A", i32 0, i32 0), i8* bitcast (i32* @"?x@?$A@H@implicit_template_instantiation@@2HA" to i8*)], section "llvm.metadata"
