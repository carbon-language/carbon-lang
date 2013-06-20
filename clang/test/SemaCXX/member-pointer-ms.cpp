// RUN: %clang_cc1 -std=c++11 -cxx-abi microsoft -fms-compatibility -fsyntax-only -triple=i386-pc-win32 -verify %s
// RUN: %clang_cc1 -std=c++11 -cxx-abi microsoft -fms-compatibility -fsyntax-only -triple=x86_64-pc-win32 -verify %s
//
// This file should also give no diagnostics when run through cl.exe from MSVS
// 2012, which supports C++11 and static_assert.  It should pass for both 64-bit
// and 32-bit x86.
//
// Test the size of various member pointer combinations:
// - complete and incomplete
// - single, multiple, and virtual inheritance (and unspecified for incomplete)
// - data and function pointers
// - templated with declared specializations with annotations
// - template that can be instantiated

// http://llvm.org/PR12070
struct Foo {
  typedef int Foo::*FooInt;
  int f;
};

enum {
  kSingleDataSize             = 1 * sizeof(int),
  kSingleFunctionSize         = 1 * sizeof(void *),
  kMultipleDataSize           = 1 * sizeof(int),
  kMultipleFunctionSize       = 2 * sizeof(void *),
  kVirtualDataSize            = 2 * sizeof(int),
  kVirtualFunctionSize        = 2 * sizeof(int) + 1 * sizeof(void *),
  // Unspecified is weird, it's 1 more slot than virtual.
  kUnspecifiedDataSize        = kVirtualDataSize + 1 * sizeof(int),
  kUnspecifiedFunctionSize    = kVirtualFunctionSize + 1 * sizeof(void *),
};

// incomplete types
class __single_inheritance IncSingle;
class __multiple_inheritance IncMultiple;
class __virtual_inheritance IncVirtual;
static_assert(sizeof(int IncSingle::*)        == kSingleDataSize, "");
static_assert(sizeof(int IncMultiple::*)      == kMultipleDataSize, "");
static_assert(sizeof(int IncVirtual::*)       == kVirtualDataSize, "");
static_assert(sizeof(void (IncSingle::*)())   == kSingleFunctionSize, "");
static_assert(sizeof(void (IncMultiple::*)()) == kMultipleFunctionSize, "");
static_assert(sizeof(void (IncVirtual::*)())  == kVirtualFunctionSize, "");

// An incomplete type with an unspecified inheritance model seems to take one
// more slot than virtual.  It's not clear what it's used for yet.
class IncUnspecified;
static_assert(sizeof(int IncUnspecified::*) == kUnspecifiedDataSize, "");
static_assert(sizeof(void (IncUnspecified::*)()) == kUnspecifiedFunctionSize, "");

// complete types
struct B1 { };
struct B2 { };
struct Single { };
struct Multiple : B1, B2 { };
struct Virtual : virtual B1 { };
static_assert(sizeof(int Single::*)        == kSingleDataSize, "");
static_assert(sizeof(int Multiple::*)      == kMultipleDataSize, "");
static_assert(sizeof(int Virtual::*)       == kVirtualDataSize, "");
static_assert(sizeof(void (Single::*)())   == kSingleFunctionSize, "");
static_assert(sizeof(void (Multiple::*)()) == kMultipleFunctionSize, "");
static_assert(sizeof(void (Virtual::*)())  == kVirtualFunctionSize, "");

// Test both declared and defined templates.
template <typename T> class X;
template <> class __single_inheritance   X<IncSingle>;
template <> class __multiple_inheritance X<IncMultiple>;
template <> class __virtual_inheritance  X<IncVirtual>;
// Don't declare X<IncUnspecified>.
static_assert(sizeof(int X<IncSingle>::*)           == kSingleDataSize, "");
static_assert(sizeof(int X<IncMultiple>::*)         == kMultipleDataSize, "");
static_assert(sizeof(int X<IncVirtual>::*)          == kVirtualDataSize, "");
static_assert(sizeof(int X<IncUnspecified>::*)      == kUnspecifiedDataSize, "");
static_assert(sizeof(void (X<IncSingle>::*)())      == kSingleFunctionSize, "");
static_assert(sizeof(void (X<IncMultiple>::*)())    == kMultipleFunctionSize, "");
static_assert(sizeof(void (X<IncVirtual>::*)())     == kVirtualFunctionSize, "");
static_assert(sizeof(void (X<IncUnspecified>::*)()) == kUnspecifiedFunctionSize, "");

template <typename T>
struct Y : T { };
static_assert(sizeof(int Y<Single>::*)        == kSingleDataSize, "");
static_assert(sizeof(int Y<Multiple>::*)      == kMultipleDataSize, "");
static_assert(sizeof(int Y<Virtual>::*)       == kVirtualDataSize, "");
static_assert(sizeof(void (Y<Single>::*)())   == kSingleFunctionSize, "");
static_assert(sizeof(void (Y<Multiple>::*)()) == kMultipleFunctionSize, "");
static_assert(sizeof(void (Y<Virtual>::*)())  == kVirtualFunctionSize, "");

struct A { int x; void bar(); };
struct B : A { virtual void foo(); };
static_assert(sizeof(int B::*) == kSingleDataSize, "");
// A non-primary base class uses the multiple inheritance model for member
// pointers.
static_assert(sizeof(void (B::*)()) == kMultipleFunctionSize, "");

struct AA { int x; virtual void foo(); };
struct BB : AA { void bar(); };
struct CC : BB { virtual void baz(); };
static_assert(sizeof(void (CC::*)()) == kSingleFunctionSize, "");

// We start out unspecified.
struct ForwardDecl1;
struct ForwardDecl2;

// Re-declare to force us to iterate decls when adding attributes.
struct ForwardDecl1;
struct ForwardDecl2;

typedef int ForwardDecl1::*MemPtr1;
typedef int ForwardDecl2::*MemPtr2;
MemPtr1 variable_forces_sizing;

struct ForwardDecl1 : B {
  virtual void foo();
};
struct ForwardDecl2 : B {
  virtual void foo();
};

static_assert(sizeof(variable_forces_sizing) == kUnspecifiedDataSize, "");
static_assert(sizeof(MemPtr1) == kUnspecifiedDataSize, "");
// FIXME: Clang fails this assert because it locks in the inheritance model at
// the point of the typedef instead of the first usage, while MSVC does not.
//static_assert(sizeof(MemPtr2) == kSingleDataSize, "");

struct MemPtrInBody {
  typedef int MemPtrInBody::*MemPtr;
  int a;
  operator MemPtr() const {
    return a ? &MemPtrInBody::a : 0;
  }
};

static_assert(sizeof(MemPtrInBody::MemPtr) == kSingleDataSize, "");

// Passing a member pointer through a template should get the right size.
template<typename T>
struct SingleTemplate;
template<typename T>
struct SingleTemplate<void (T::*)(void)> {
  static_assert(sizeof(int T::*) == kSingleDataSize, "");
  static_assert(sizeof(void (T::*)()) == kSingleFunctionSize, "");
};

template<typename T>
struct UnspecTemplate;
template<typename T>
struct UnspecTemplate<void (T::*)(void)> {
  static_assert(sizeof(int T::*) == kUnspecifiedDataSize, "");
  static_assert(sizeof(void (T::*)()) == kUnspecifiedFunctionSize, "");
};

struct NewUnspecified;
SingleTemplate<void (IncSingle::*)()> tmpl_single;
UnspecTemplate<void (NewUnspecified::*)()> tmpl_unspec;

struct NewUnspecified { };

static_assert(sizeof(void (NewUnspecified::*)()) == kUnspecifiedFunctionSize, "");

template <typename T>
struct MemPtrInTemplate {
  // We can't require that the template arg be complete until we're
  // instantiated.
  int T::*data_ptr;
  void (T::*func_ptr)();
};

int Virtual::*CastTest = reinterpret_cast<int Virtual::*>(&AA::x);
  // expected-error@-1 {{cannot reinterpret_cast from member pointer type}}
