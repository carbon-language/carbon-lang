// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

// Simple key function test
struct testa { virtual void a(); };
void testa::a() {}

// Simple key function test
struct testb { virtual void a() {} };
testb *testbvar = new testb;

// Key function with out-of-line inline definition
struct testc { virtual void a(); };
inline void testc::a() {}

// Key functions with inline specifier (PR5705)
struct testd { inline virtual void a(); };
void testd::a() {}

// Key functions with inline specifier (PR5705)
struct teste { inline virtual void a(); };
teste *testevar = new teste;

// Key functions with namespace (PR5711)
namespace {
  struct testf { virtual void a(); };
}
void testf::a() {}

// Key functions with namespace (PR5711)
namespace {
  struct testg { virtual void a(); };
}
testg *testgvar = new testg;

// FIXME: The checks are extremely difficult to get right when the globals
// aren't alphabetized
// CHECK: @_ZTV5testa = constant [3 x i8*] [i8* null
// CHECK: @_ZTV5testc = weak_odr constant [3 x i8*] [i8* null
// CHECK: @_ZTVN12_GLOBAL__N_15testgE = internal constant [3 x i8*] [i8* null
// CHECK: @_ZTV5teste = weak_odr constant [3 x i8*] [i8* null
// CHECK: @_ZTV5testb = weak_odr constant [3 x i8*] [i8* null

