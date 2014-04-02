// RUN: %clang_cc1 -triple i686-win32     -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -verify -std=c++1y %s
// RUN: %clang_cc1 -triple i686-mingw32   -fsyntax-only -verify -std=c++1y %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -verify -std=c++11 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Exported {};
struct ExplicitDecl_Exported {};
struct ExplicitInst_Exported {};
struct ExplicitSpec_Exported {};
struct ExplicitSpec_Def_Exported {};
struct ExplicitSpec_InlineDef_Exported {};
struct ExplicitSpec_NotExported {};
namespace { struct Internal {}; }
struct External { int v; };


// Invalid usage.
__declspec(dllexport) typedef int typedef1; // expected-warning{{'dllexport' attribute only applies to variables and functions}}
typedef __declspec(dllexport) int typedef2; // expected-warning{{'dllexport' attribute only applies to variables and functions}}
typedef int __declspec(dllexport) typedef3; // expected-warning{{'dllexport' attribute only applies to variables and functions}}
typedef __declspec(dllexport) void (*FunTy)(); // expected-warning{{'dllexport' attribute only applies to variables and functions}}
enum __declspec(dllexport) Enum {}; // expected-warning{{'dllexport' attribute only applies to variables and functions}}
#if __has_feature(cxx_strong_enums)
  enum class __declspec(dllexport) EnumClass {}; // expected-warning{{'dllexport' attribute only applies to variables and functions}}
#endif



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
__declspec(dllexport) extern int GlobalRedecl3; // expected-error{{redeclaration of 'GlobalRedecl3' cannot add 'dllexport' attribute}}

// External linkage is required.
__declspec(dllexport) static int StaticGlobal; // expected-error{{'StaticGlobal' must have external linkage when declared 'dllexport'}}
__declspec(dllexport) Internal InternalTypeGlobal; // expected-error{{'InternalTypeGlobal' must have external linkage when declared 'dllexport'}}
namespace    { __declspec(dllexport) int InternalGlobal; } // expected-error{{'(anonymous namespace)::InternalGlobal' must have external linkage when declared 'dllexport'}}
namespace ns { __declspec(dllexport) int ExternalGlobal; }

__declspec(dllexport) auto InternalAutoTypeGlobal = Internal(); // expected-error{{'InternalAutoTypeGlobal' must have external linkage when declared 'dllexport'}}
__declspec(dllexport) auto ExternalAutoTypeGlobal = External();

// Export in local scope.
void functionScope() {
  __declspec(dllexport)        int LocalVarDecl; // expected-error{{'LocalVarDecl' must have external linkage when declared 'dllexport'}}
  __declspec(dllexport)        int LocalVarDef = 1; // expected-error{{'LocalVarDef' must have external linkage when declared 'dllexport'}}
  __declspec(dllexport) extern int ExternLocalVarDecl;
  __declspec(dllexport) static int StaticLocalVar; // expected-error{{'StaticLocalVar' must have external linkage when declared 'dllexport'}}
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Export function declaration. Check different placements.
__attribute__((dllexport)) void decl1A(); // Sanity check with __attribute__
__declspec(dllexport)      void decl1B();

void __attribute__((dllexport)) decl2A();
void __declspec(dllexport)      decl2B();

// Export function definition.
__declspec(dllexport) void def() {}

// extern "C"
extern "C" __declspec(dllexport) void externC() {}

// Export inline function.
__declspec(dllexport) inline void inlineFunc1() {} // expected-warning{{'dllexport' attribute ignored}}
inline void __attribute__((dllexport)) inlineFunc2() {} // expected-warning{{'dllexport' attribute ignored}}

__declspec(dllexport) inline void inlineDecl(); // expected-warning{{'dllexport' attribute ignored}}
                             void inlineDecl() {}

__declspec(dllexport) void inlineDef();
               inline void inlineDef() {}

// Redeclarations
__declspec(dllexport) void redecl1();
__declspec(dllexport) void redecl1() {}

__declspec(dllexport) void redecl2();
                      void redecl2() {}

                      void redecl3(); // expected-note{{previous declaration is here}}
__declspec(dllexport) void redecl3(); // expected-error{{redeclaration of 'redecl3' cannot add 'dllexport' attribute}}

// Friend functions
struct FuncFriend {
  friend __declspec(dllexport) void friend1();
  friend __declspec(dllexport) void friend2();
  friend                       void friend3(); // expected-note{{previous declaration is here}}
};
__declspec(dllexport) void friend1() {}
                      void friend2() {}
__declspec(dllexport) void friend3() {} // expected-error{{redeclaration of 'friend3' cannot add 'dllexport' attribute}}

// Implicit declarations can be redeclared with dllexport.
__declspec(dllexport) void* operator new(__SIZE_TYPE__ n);

// External linkage is required.
__declspec(dllexport) static int staticFunc(); // expected-error{{'staticFunc' must have external linkage when declared 'dllexport'}}
__declspec(dllexport) Internal internalRetFunc(); // expected-error{{'internalRetFunc' must have external linkage when declared 'dllexport'}}
namespace    { __declspec(dllexport) void internalFunc() {} } // expected-error{{'(anonymous namespace)::internalFunc' must have external linkage when declared 'dllexport'}}
namespace ns { __declspec(dllexport) void externalFunc() {} }



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Export function template declaration. Check different placements.
template<typename T> __declspec(dllexport) void funcTmplDecl1();
template<typename T> void __declspec(dllexport) funcTmplDecl2();

// Export function template definition.
template<typename T> __declspec(dllexport) void funcTmplDef() {}

// Redeclarations
template<typename T> __declspec(dllexport) void funcTmplRedecl1();
template<typename T> __declspec(dllexport) void funcTmplRedecl1() {}

template<typename T> __declspec(dllexport) void funcTmplRedecl2();
template<typename T>                       void funcTmplRedecl2() {}

template<typename T>                       void funcTmplRedecl3(); // expected-note{{previous declaration is here}}
template<typename T> __declspec(dllexport) void funcTmplRedecl3(); // expected-error{{redeclaration of 'funcTmplRedecl3' cannot add 'dllexport' attribute}}

// Function template friends
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllexport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllexport) void funcTmplFriend2();
  template<typename T> friend                       void funcTmplFriend3(); // expected-note{{previous declaration is here}}
};
template<typename T> __declspec(dllexport) void funcTmplFriend1() {}
template<typename T>                       void funcTmplFriend2() {}
template<typename T> __declspec(dllexport) void funcTmplFriend3() {} // expected-error{{redeclaration of 'funcTmplFriend3' cannot add 'dllexport' attribute}}

// External linkage is required.
template<typename T> __declspec(dllexport) static int staticFuncTmpl(); // expected-error{{'staticFuncTmpl' must have external linkage when declared 'dllexport'}}
template<typename T> __declspec(dllexport) Internal internalRetFuncTmpl(); // expected-error{{'internalRetFuncTmpl' must have external linkage when declared 'dllexport'}}
namespace    { template<typename T> __declspec(dllexport) void internalFuncTmpl(); } // expected-error{{'(anonymous namespace)::internalFuncTmpl' must have external linkage when declared 'dllexport'}}
namespace ns { template<typename T> __declspec(dllexport) void externalFuncTmpl(); }


template<typename T> void funcTmpl() {}
template<typename T> __declspec(dllexport) void exportedFuncTmplDecl();
template<typename T> __declspec(dllexport) void exportedFuncTmpl() {}

// Export implicit instantiation of an exported function template.
void useFunTmplDecl() { exportedFuncTmplDecl<ImplicitInst_Exported>(); }
void useFunTmplDef() { exportedFuncTmpl<ImplicitInst_Exported>(); }

// Export explicit instantiation declaration of an exported function template.
extern template void exportedFuncTmpl<ExplicitDecl_Exported>();
       template void exportedFuncTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of an exported function template.
template void exportedFuncTmpl<ExplicitInst_Exported>();

// Export specialization of an exported function template.
template<> __declspec(dllexport) void exportedFuncTmpl<ExplicitSpec_Exported>();
template<> __declspec(dllexport) void exportedFuncTmpl<ExplicitSpec_Def_Exported>() {}
template<> __declspec(dllexport) inline void exportedFuncTmpl<ExplicitSpec_InlineDef_Exported>() {} // expected-warning{{'dllexport' attribute ignored}}

// Not exporting specialization of an exported function template without
// explicit dllexport.
template<> void exportedFuncTmpl<ExplicitSpec_NotExported>() {}


// Export explicit instantiation declaration of a non-exported function template.
extern template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();
       template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of a non-exported function template.
template __declspec(dllexport) void funcTmpl<ExplicitInst_Exported>();

// Export specialization of a non-exported function template.
template<> __declspec(dllexport) void funcTmpl<ExplicitSpec_Exported>();
template<> __declspec(dllexport) void funcTmpl<ExplicitSpec_Def_Exported>() {}
template<> __declspec(dllexport) inline void funcTmpl<ExplicitSpec_InlineDef_Exported>() {} // expected-warning{{'dllexport' attribute ignored}}



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

void __attribute__((dllimport, dllexport))       precedence1A() {} // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllimport) __declspec(dllexport) precedence1B() {} // expected-warning{{'dllimport' attribute ignored}}

void __attribute__((dllexport, dllimport))       precedence2A() {} // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllexport) __declspec(dllimport) precedence2B() {} // expected-warning{{'dllimport' attribute ignored}}

void __declspec(dllimport) precedenceRedecl1(); // expected-warning{{'dllimport' attribute ignored}}
void __declspec(dllexport) precedenceRedecl1() {}

void __declspec(dllexport) precedenceRedecl2();
void __declspec(dllimport) precedenceRedecl2() {} // expected-warning{{'dllimport' attribute ignored}}
