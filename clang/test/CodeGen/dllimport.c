// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c11 -O1 -o - %s | FileCheck --check-prefix=O1 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c11 -O1 -o - %s | FileCheck --check-prefix=O1 %s

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define USEVAR(var) int JOIN(use, __LINE__)() { return var; }
#define USE(func) void JOIN(use, __LINE__)() { func(); }



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
// CHECK: @ExternGlobalDecl = external dllimport global i32
__declspec(dllimport) extern int ExternGlobalDecl;
USEVAR(ExternGlobalDecl)

// dllimport implies a declaration.
// CHECK: @GlobalDecl = external dllimport global i32
__declspec(dllimport) int GlobalDecl;
USEVAR(GlobalDecl)

// Redeclarations
// CHECK: @GlobalRedecl1 = external dllimport global i32
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;
USEVAR(GlobalRedecl1)

// CHECK: @GlobalRedecl2 = external dllimport global i32
__declspec(dllimport) int GlobalRedecl2;
__declspec(dllimport) int GlobalRedecl2;
USEVAR(GlobalRedecl2)

// NB: MSVC issues a warning and makes GlobalRedecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// CHECK: @GlobalRedecl3 = external global i32
__declspec(dllimport) extern int GlobalRedecl3;
                      extern int GlobalRedecl3; // dllimport ignored
USEVAR(GlobalRedecl3)

// Redeclaration in local context.
// CHECK: @GlobalRedecl4 = external dllimport global i32
__declspec(dllimport) int GlobalRedecl4;
int functionScope() {
  extern int GlobalRedecl4; // still dllimport
  return GlobalRedecl4;
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration.
// CHECK-DAG: declare dllimport void @decl()
__declspec(dllimport) void decl(void);

// Initialize use_decl with the address of the thunk.
// CHECK-DAG: @use_decl = global void ()* @decl
void (*use_decl)(void) = &decl;

// Import inline function.
// CHECK-DAG: declare dllimport void @inlineFunc()
// O1-DAG: define available_externally dllimport void @inlineFunc()
__declspec(dllimport) inline void inlineFunc(void) {}
USE(inlineFunc)

// inline attributes
// CHECK-DAG: declare dllimport void @noinline()
// O1-DAG: define available_externally dllimport void @noinline()
// CHECK-NOT: @alwaysInline()
// O1-NOT: @alwaysInline()
__declspec(dllimport) __attribute__((noinline)) inline void noinline(void) {}
__declspec(dllimport) __attribute__((always_inline)) inline void alwaysInline(void) {}
USE(noinline)
USE(alwaysInline)

// Redeclarations
// CHECK-DAG: declare dllimport void @redecl1()
__declspec(dllimport) void redecl1(void);
__declspec(dllimport) void redecl1(void);
USE(redecl1)

// NB: MSVC issues a warning and makes redecl2/redecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// CHECK-DAG: declare void @redecl2()
__declspec(dllimport) void redecl2(void);
                      void redecl2(void);
USE(redecl2)

// CHECK-DAG: define void @redecl3()
__declspec(dllimport) void redecl3(void);
                      void redecl3(void) {} // dllimport ignored
USE(redecl3)
