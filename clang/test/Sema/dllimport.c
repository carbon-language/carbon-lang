// RUN: %clang_cc1 -triple i686-win32     -fsyntax-only -verify -std=c99 %s
// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -verify -std=c11 %s
// RUN: %clang_cc1 -triple i686-mingw32   -fsyntax-only -verify -std=c11 %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -verify -std=c99 %s

// Invalid usage.
__declspec(dllimport) typedef int typedef1; // expected-warning{{'dllimport' attribute only applies to variables and functions}}
typedef __declspec(dllimport) int typedef2; // expected-warning{{'dllimport' attribute only applies to variables and functions}}
typedef int __declspec(dllimport) typedef3; // expected-warning{{'dllimport' attribute only applies to variables and functions}}
typedef __declspec(dllimport) void (*FunTy)(); // expected-warning{{'dllimport' attribute only applies to variables and functions}}
enum __declspec(dllimport) Enum { EnumVal }; // expected-warning{{'dllimport' attribute only applies to variables and functions}}
struct __declspec(dllimport) Record {}; // expected-warning{{'dllimport' attribute only applies to variables and functions}}



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
__declspec(dllimport) extern int ExternGlobalDecl;

// dllimport implies a declaration.
__declspec(dllimport) int GlobalDecl;
int **__attribute__((dllimport))* GlobalDeclChunkAttr;
int GlobalDeclAttr __attribute__((dllimport));

// Not allowed on definitions.
__declspec(dllimport) extern int ExternGlobalInit = 1; // expected-error{{definition of dllimport data}}
__declspec(dllimport) int GlobalInit1 = 1; // expected-error{{definition of dllimport data}}
int __declspec(dllimport) GlobalInit2 = 1; // expected-error{{definition of dllimport data}}

// Declare, then reject definition.
__declspec(dllimport) extern int ExternGlobalDeclInit;
int ExternGlobalDeclInit = 1; // expected-error{{definition of dllimport data}}

__declspec(dllimport) int GlobalDeclInit;
int GlobalDeclInit = 1; // expected-error{{definition of dllimport data}}

int *__attribute__((dllimport)) GlobalDeclChunkAttrInit;
int *GlobalDeclChunkAttrInit = 0; // expected-error{{definition of dllimport data}}

int GlobalDeclAttrInit __attribute__((dllimport));
int GlobalDeclAttrInit = 1; // expected-error{{definition of dllimport data}}

// Redeclarations
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;

__declspec(dllimport) int GlobalRedecl2a;
__declspec(dllimport) int GlobalRedecl2a;

int *__attribute__((dllimport)) GlobalRedecl2b;
int *__attribute__((dllimport)) GlobalRedecl2b;

int GlobalRedecl2c __attribute__((dllimport));
int GlobalRedecl2c __attribute__((dllimport));

// Import in local scope.
__declspec(dllimport) float LocalRedecl1; // expected-note{{previous definition is here}}
__declspec(dllimport) float LocalRedecl2; // expected-note{{previous definition is here}}
__declspec(dllimport) float LocalRedecl3; // expected-note{{previous definition is here}}
void functionScope() {
  __declspec(dllimport) int LocalRedecl1; // expected-error{{redefinition of 'LocalRedecl1' with a different type: 'int' vs 'float'}}
  int *__attribute__((dllimport)) LocalRedecl2; // expected-error{{redefinition of 'LocalRedecl2' with a different type: 'int *' vs 'float'}}
  int LocalRedecl3 __attribute__((dllimport)); // expected-error{{redefinition of 'LocalRedecl3' with a different type: 'int' vs 'float'}}

  __declspec(dllimport)        int LocalVarDecl;
  __declspec(dllimport)        int LocalVarDef = 1; // expected-error{{definition of dllimport data}}
  __declspec(dllimport) extern int ExternLocalVarDecl;
  __declspec(dllimport) extern int ExternLocalVarDef = 1; // expected-error{{definition of dllimport data}}
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration. Check different placements.
__attribute__((dllimport)) void decl1A(); // Sanity check with __attribute__
__declspec(dllimport)      void decl1B();

void __attribute__((dllimport)) decl2A();
void __declspec(dllimport)      decl2B();

// Not allowed on function definitions.
__declspec(dllimport) void def() {} // expected-error{{'dllimport' attribute can be applied only to symbol declaration}}

// Import inline function.
__declspec(dllimport) inline void inlineFunc1() {} // expected-warning{{'dllimport' attribute ignored}}
inline void __attribute__((dllimport)) inlineFunc2() {} // expected-warning{{'dllimport' attribute ignored}}

// Redeclarations
__declspec(dllimport) void redecl1();
__declspec(dllimport) void redecl1();

__declspec(dllimport) void redecl3();
                      void redecl3() {} // expected-warning{{'redecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
