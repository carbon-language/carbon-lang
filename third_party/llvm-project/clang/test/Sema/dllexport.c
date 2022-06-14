// RUN: %clang_cc1 -triple i686-win32     -fsyntax-only -fms-extensions -verify -std=c99 %s
// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -fms-extensions -verify -std=c11 %s
// RUN: %clang_cc1 -triple i686-mingw32   -fsyntax-only -fms-extensions -verify -std=c11 %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -fms-extensions -verify -std=c99 %s

// Invalid usage.
__declspec(dllexport) typedef int typedef1;
// expected-warning@-1{{'dllexport' attribute only applies to functions, variables, classes, and Objective-C interfaces}}
typedef __declspec(dllexport) int typedef2;
// expected-warning@-1{{'dllexport' attribute only applies to}}
typedef int __declspec(dllexport) typedef3;
// expected-warning@-1{{'dllexport' attribute only applies to}}
typedef __declspec(dllexport) void (*FunTy)(void);
// expected-warning@-1{{'dllexport' attribute only applies to}}
enum __declspec(dllexport) Enum { EnumVal };
// expected-warning@-1{{'dllexport' attribute only applies to}}
struct __declspec(dllexport) Record {};
// expected-warning@-1{{'dllexport' attribute only applies to}}



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Export declaration.
__declspec(dllexport) extern int ExternGlobalDecl;

// dllexport implies a definition.
__declspec(dllexport) int GlobalDef;

// Export definition.
__declspec(dllexport) int GlobalInit1 = 1;
int __declspec(dllexport) GlobalInit2 = 1;

// Declare, then export definition.
__declspec(dllexport) extern int GlobalDeclInit;
int GlobalDeclInit = 1;

// Redeclarations
__declspec(dllexport) extern int GlobalRedecl1;
__declspec(dllexport)        int GlobalRedecl1;

__declspec(dllexport) extern int GlobalRedecl2;
                             int GlobalRedecl2;

                      extern int GlobalRedecl3; // expected-note{{previous declaration is here}}
int useGlobalRedecl3(void) { return GlobalRedecl3; }
__declspec(dllexport) extern int GlobalRedecl3; // expected-error{{redeclaration of 'GlobalRedecl3' cannot add 'dllexport' attribute}}

                      extern int GlobalRedecl4; // expected-note{{previous declaration is here}}
__declspec(dllexport) extern int GlobalRedecl4; // expected-warning{{redeclaration of 'GlobalRedecl4' should not add 'dllexport' attribute}}


// External linkage is required.
__declspec(dllexport) static int StaticGlobal; // expected-error{{'StaticGlobal' must have external linkage when declared 'dllexport'}}

// Thread local variables are invalid.
__declspec(dllexport) __thread int ThreadLocalGlobal; // expected-error{{'ThreadLocalGlobal' cannot be thread local when declared 'dllexport'}}

// Export in local scope.
void functionScope(void) {
  __declspec(dllexport)        int LocalVarDecl; // expected-error{{'LocalVarDecl' must have external linkage when declared 'dllexport'}}
  __declspec(dllexport)        int LocalVarDef = 1; // expected-error{{'LocalVarDef' must have external linkage when declared 'dllexport'}}
  __declspec(dllexport) extern int ExternLocalVarDecl;
  __declspec(dllexport) static int StaticLocalVar; // expected-error{{'StaticLocalVar' must have external linkage when declared 'dllexport'}}
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Export function declaration. Check different placements.
__attribute__((dllexport)) void decl1A(void); // Basic check with __attribute__
__declspec(dllexport)      void decl1B(void);

void __attribute__((dllexport)) decl2A(void);
void __declspec(dllexport)      decl2B(void);

// Export function definition.
__declspec(dllexport) void def(void) {}

// Export inline function.
__declspec(dllexport) inline void inlineFunc1(void) {}
extern void inlineFunc1(void);

inline void __attribute__((dllexport)) inlineFunc2(void) {}
extern void inlineFunc2(void);

// Redeclarations
__declspec(dllexport) void redecl1(void);
__declspec(dllexport) void redecl1(void);

__declspec(dllexport) void redecl2(void);
                      void redecl2(void);

__declspec(dllexport) void redecl3(void);
                      void redecl3(void) {}

                      void redecl4(void); // expected-note{{previous declaration is here}}
void useRedecl4(void) { redecl4(); }
__declspec(dllexport) void redecl4(void); // expected-error{{redeclaration of 'redecl4' cannot add 'dllexport' attribute}}

                      void redecl5(void); // expected-note{{previous declaration is here}}
void useRedecl5(void) { redecl5(); }
__declspec(dllexport) inline void redecl5(void) {} // expected-error{{redeclaration of 'redecl5' cannot add 'dllexport' attribute}}

// Allow with a warning if the decl hasn't been used yet.
                      void redecl6(void); // expected-note{{previous declaration is here}}
__declspec(dllexport) void redecl6(void); // expected-warning{{redeclaration of 'redecl6' should not add 'dllexport' attribute}}


// External linkage is required.
__declspec(dllexport) static int staticFunc(void); // expected-error{{'staticFunc' must have external linkage when declared 'dllexport'}}

// Static locals don't count as having external linkage.
void staticLocalFunc(void) {
  __declspec(dllexport) static int staticLocal; // expected-error{{'staticLocal' must have external linkage when declared 'dllexport'}}
}



//===----------------------------------------------------------------------===//
// Precedence
//===----------------------------------------------------------------------===//

// dllexport takes precedence over dllimport if both are specified.
__attribute__((dllimport, dllexport))       extern int PrecedenceExternGlobal1A; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllimport) __declspec(dllexport) extern int PrecedenceExternGlobal1B; // expected-warning{{'dllimport' attribute ignored}}

__attribute__((dllexport, dllimport))       extern int PrecedenceExternGlobal2A; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllexport) __declspec(dllimport) extern int PrecedenceExternGlobal2B; // expected-warning{{'dllimport' attribute ignored}}

__attribute__((dllimport, dllexport))       int PrecedenceGlobal1A; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllimport) __declspec(dllexport) int PrecedenceGlobal1B; // expected-warning{{'dllimport' attribute ignored}}

__attribute__((dllexport, dllimport))       int PrecedenceGlobal2A; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllexport) __declspec(dllimport) int PrecedenceGlobal2B; // expected-warning{{'dllimport' attribute ignored}}

__declspec(dllexport) extern int PrecedenceExternGlobalRedecl1;
__declspec(dllimport) extern int PrecedenceExternGlobalRedecl1; // expected-warning{{'dllimport' attribute ignored}}

__declspec(dllimport) extern int PrecedenceExternGlobalRedecl2; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllexport) extern int PrecedenceExternGlobalRedecl2;

__declspec(dllexport) extern int PrecedenceGlobalRedecl1;
__declspec(dllimport)        int PrecedenceGlobalRedecl1; // expected-warning{{'dllimport' attribute ignored}}

__declspec(dllimport) extern int PrecedenceGlobalRedecl2; // expected-warning{{'dllimport' attribute ignored}}
__declspec(dllexport)        int PrecedenceGlobalRedecl2;

void __attribute__((dllimport, dllexport))       precedence1A(void) {} // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllimport) __declspec(dllexport) precedence1B(void) {} // expected-warning{{'dllimport' attribute ignored}}

void __attribute__((dllexport, dllimport))       precedence2A(void) {} // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllexport) __declspec(dllimport) precedence2B(void) {} // expected-warning{{'dllimport' attribute ignored}}

void __declspec(dllimport) precedenceRedecl1(void); // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllexport) precedenceRedecl1(void) {}

void __declspec(dllexport) precedenceRedecl2(void);
void __declspec(dllimport) precedenceRedecl2(void) {} // expected-warning{{'dllimport' attribute ignored}}
