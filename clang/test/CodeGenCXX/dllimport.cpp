// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -O0 -o - %s -DMSABI | FileCheck --check-prefix=MSC --check-prefix=M32 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++1y -O0 -o - %s -DMSABI | FileCheck --check-prefix=MSC --check-prefix=M64 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c++1y -O0 -o - %s         | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -emit-llvm -std=c++1y -O0 -o - %s         | FileCheck --check-prefix=GNU --check-prefix=G64 %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -O1 -o - %s -DMSABI | FileCheck --check-prefix=MO1 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c++1y -O1 -o - %s         | FileCheck --check-prefix=GO1 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Imported {};
struct ImplicitInst_NotImported {};
struct ExplicitDecl_Imported {};
struct ExplicitInst_Imported {};
struct ExplicitSpec_Imported {};
struct ExplicitSpec_Def_Imported {};
struct ExplicitSpec_InlineDef_Imported {};
struct ExplicitSpec_NotImported {};

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEVARTYPE(type, var) type UNIQ(use)() { return var; }
#define USEVAR(var) USEVARTYPE(int, var)
#define USE(func) void UNIQ(use)() { func(); }



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
// MSC-DAG: @"\01?ExternGlobalDecl@@3HA" = external dllimport global i32
// GNU-DAG: @ExternGlobalDecl            = external dllimport global i32
__declspec(dllimport) extern int ExternGlobalDecl;
USEVAR(ExternGlobalDecl)

// dllimport implies a declaration.
// MSC-DAG: @"\01?GlobalDecl@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalDecl            = external dllimport global i32
__declspec(dllimport) int GlobalDecl;
USEVAR(GlobalDecl)

// Redeclarations
// MSC-DAG: @"\01?GlobalRedecl1@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl1            = external dllimport global i32
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;
USEVAR(GlobalRedecl1)

// MSC-DAG: @"\01?GlobalRedecl2a@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl2a            = external dllimport global i32
__declspec(dllimport) int GlobalRedecl2a;
__declspec(dllimport) int GlobalRedecl2a;
USEVAR(GlobalRedecl2a)

// M32-DAG: @"\01?GlobalRedecl2b@@3PAHA"   = external dllimport global i32*
// M64-DAG: @"\01?GlobalRedecl2b@@3PEAHEA" = external dllimport global i32*
// GNU-DAG: @GlobalRedecl2b                = external dllimport global i32*
int *__attribute__((dllimport)) GlobalRedecl2b;
int *__attribute__((dllimport)) GlobalRedecl2b;
USEVARTYPE(int*, GlobalRedecl2b)

// MSC-DAG: @"\01?GlobalRedecl2c@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl2c            = external dllimport global i32
int GlobalRedecl2c __attribute__((dllimport));
int GlobalRedecl2c __attribute__((dllimport));
USEVAR(GlobalRedecl2c)

// NB: MSC issues a warning and makes GlobalRedecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// MSC-DAG: @"\01?GlobalRedecl3@@3HA" = external global i32
// GNU-DAG: @GlobalRedecl3            = external global i32
__declspec(dllimport) extern int GlobalRedecl3;
                      extern int GlobalRedecl3; // dllimport ignored
USEVAR(GlobalRedecl3)

// MSC-DAG: @"\01?ExternalGlobal@ns@@3HA" = external dllimport global i32
// GNU-DAG: @_ZN2ns14ExternalGlobalE      = external dllimport global i32
namespace ns { __declspec(dllimport) int ExternalGlobal; }
USEVAR(ns::ExternalGlobal)



//===----------------------------------------------------------------------===//
// Variable templates
//===----------------------------------------------------------------------===//

// Import declaration.
// MSC-DAG: @"\01??$ExternVarTmplDecl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z17ExternVarTmplDeclI21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) extern int ExternVarTmplDecl;
USEVAR(ExternVarTmplDecl<ImplicitInst_Imported>)

// dllimport implies a declaration.
// MSC-DAG: @"\01??$VarTmplDecl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z11VarTmplDeclI21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) int VarTmplDecl;
USEVAR(VarTmplDecl<ImplicitInst_Imported>)

// Redeclarations
// MSC-DAG: @"\01??$VarTmplRedecl1@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z14VarTmplRedecl1I21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;
USEVAR(VarTmplRedecl1<ImplicitInst_Imported>)

// MSC-DAG: @"\01??$VarTmplRedecl2@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z14VarTmplRedecl2I21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) int VarTmplRedecl2;
template<typename T> __declspec(dllimport) int VarTmplRedecl2;
USEVAR(VarTmplRedecl2<ImplicitInst_Imported>)

// MSC-DAG: @"\01??$VarTmplRedecl3@UImplicitInst_Imported@@@@3HA" = external global i32
// GNU-DAG: @_Z14VarTmplRedecl3I21ImplicitInst_ImportedE          = external global i32
template<typename T> __declspec(dllimport) extern int VarTmplRedecl3;
template<typename T>                       extern int VarTmplRedecl3; // dllimport ignored
USEVAR(VarTmplRedecl3<ImplicitInst_Imported>)


// MSC-DAG: @"\01??$ExternalVarTmpl@UImplicitInst_Imported@@@ns@@3HA" = external dllimport global i32
// GNU-DAG: @_ZN2ns15ExternalVarTmplI21ImplicitInst_ImportedEE        = external dllimport global i32
namespace ns { template<typename T> __declspec(dllimport) int ExternalVarTmpl; }
USEVAR(ns::ExternalVarTmpl<ImplicitInst_Imported>)


template<typename T> int VarTmpl;
template<typename T> __declspec(dllimport) int ImportedVarTmpl;

// Import implicit instantiation of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ImplicitInst_ImportedE          = external dllimport global i32
USEVAR(ImportedVarTmpl<ImplicitInst_Imported>)

// Import explicit instantiation declaration of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitDecl_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ExplicitDecl_ImportedE          = external dllimport global i32
extern template int ImportedVarTmpl<ExplicitDecl_Imported>;
USEVAR(ImportedVarTmpl<ExplicitDecl_Imported>)

// An explicit instantiation definition of an imported variable template cannot
// be imported because the template must be defined which is illegal.

// Import specialization of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitSpec_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ExplicitSpec_ImportedE          = external dllimport global i32
template<> __declspec(dllimport) int ImportedVarTmpl<ExplicitSpec_Imported>;
USEVAR(ImportedVarTmpl<ExplicitSpec_Imported>)

// Not importing specialization of an imported variable template without
// explicit dllimport.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitSpec_NotImported@@@@3HA" = global i32 0, align 4
// GNU-DAG: @_Z15ImportedVarTmplI24ExplicitSpec_NotImportedE          = global i32 0, align 4
template<> int ImportedVarTmpl<ExplicitSpec_NotImported>;
USEVAR(ImportedVarTmpl<ExplicitSpec_NotImported>)

// Import explicit instantiation declaration of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitDecl_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitDecl_ImportedE           = external dllimport global i32
extern template __declspec(dllimport) int VarTmpl<ExplicitDecl_Imported>;
USEVAR(VarTmpl<ExplicitDecl_Imported>)

// Import explicit instantiation definition of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitInst_ImportedE           = external dllimport global i32
template __declspec(dllimport) int VarTmpl<ExplicitInst_Imported>;
USEVAR(VarTmpl<ExplicitInst_Imported>)

// Import specialization of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitSpec_ImportedE           = external dllimport global i32
template<> __declspec(dllimport) int VarTmpl<ExplicitSpec_Imported>;
USEVAR(VarTmpl<ExplicitSpec_Imported>)



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration.
// MSC-DAG: declare dllimport void @"\01?decl@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z4declv()
__declspec(dllimport) void decl();
USE(decl)

// extern "C"
// MSC-DAG: declare dllimport void @externC()
// GNU-DAG: declare dllimport void @externC()
extern "C" __declspec(dllimport) void externC();
USE(externC)

// Import inline function.
// MSC-DAG: declare dllimport void @"\01?inlineFunc@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z10inlineFuncv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineFunc@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z10inlineFuncv()
__declspec(dllimport) inline void inlineFunc() {}
USE(inlineFunc)

// MSC-DAG: declare dllimport void @"\01?inlineDecl@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z10inlineDeclv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineDecl@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z10inlineDeclv()
__declspec(dllimport) inline void inlineDecl();
                             void inlineDecl() {}
USE(inlineDecl)

// MSC-DAG: declare dllimport void @"\01?inlineDef@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z9inlineDefv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineDef@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z9inlineDefv()
__declspec(dllimport) void inlineDef();
               inline void inlineDef() {}
USE(inlineDef)

// inline attributes
// MSC-DAG: declare dllimport void @"\01?noinline@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8noinlinev()
__declspec(dllimport) __attribute__((noinline)) inline void noinline() {}
USE(noinline)

// MSC-NOT: @"\01?alwaysInline@@YAXXZ"()
// GNU-NOT: @_Z12alwaysInlinev()
__declspec(dllimport) __attribute__((always_inline)) inline void alwaysInline() {}
USE(alwaysInline)

// Redeclarations
// MSC-DAG: declare dllimport void @"\01?redecl1@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z7redecl1v()
__declspec(dllimport) void redecl1();
__declspec(dllimport) void redecl1();
USE(redecl1)

// NB: MSC issues a warning and makes redecl2/redecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// MSC-DAG: declare void @"\01?redecl2@@YAXXZ"()
// GNU-DAG: declare void @_Z7redecl2v()
__declspec(dllimport) void redecl2();
                      void redecl2();
USE(redecl2)

// MSC-DAG: define void @"\01?redecl3@@YAXXZ"()
// GNU-DAG: define void @_Z7redecl3v()
__declspec(dllimport) void redecl3();
                      void redecl3() {} // dllimport ignored
USE(redecl3)


// Friend functions
// MSC-DAG: declare dllimport void @"\01?friend1@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z7friend1v()
// MSC-DAG: declare           void @"\01?friend2@@YAXXZ"()
// GNU-DAG: declare           void @_Z7friend2v()
// MSC-DAG: define            void @"\01?friend3@@YAXXZ"()
// GNU-DAG: define            void @_Z7friend3v()
struct FuncFriend {
  friend __declspec(dllimport) void friend1();
  friend __declspec(dllimport) void friend2();
  friend __declspec(dllimport) void friend3();
};
__declspec(dllimport) void friend1();
                      void friend2(); // dllimport ignored
                      void friend3() {} // dllimport ignored
USE(friend1)
USE(friend2)
USE(friend3)

// Implicit declarations can be redeclared with dllimport.
// MSC-DAG: declare dllimport noalias i8* @"\01??2@{{YAPAXI|YAPEAX_K}}@Z"(
// GNU-DAG: declare dllimport noalias i8* @_Znw{{[yj]}}(
__declspec(dllimport) void* operator new(__SIZE_TYPE__ n);
void UNIQ(use)() { ::operator new(42); }

// MSC-DAG: declare dllimport void @"\01?externalFunc@ns@@YAXXZ"()
// GNU-DAG: declare dllimport void @_ZN2ns12externalFuncEv()
namespace ns { __declspec(dllimport) void externalFunc(); }
USE(ns::externalFunc)



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Import function template declaration.
// MSC-DAG: declare dllimport void @"\01??$funcTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z12funcTmplDeclI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplDecl();
USE(funcTmplDecl<ImplicitInst_Imported>)

// Function template definitions cannot be imported.

// Import inline function template.
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z15inlineFuncTmpl1I21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl1@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z15inlineFuncTmpl1I21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) inline void inlineFuncTmpl1() {}
USE(inlineFuncTmpl1<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl2@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z15inlineFuncTmpl2I21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl2@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z15inlineFuncTmpl2I21ImplicitInst_ImportedEvv()
template<typename T> inline void __attribute__((dllimport)) inlineFuncTmpl2() {}
USE(inlineFuncTmpl2<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z18inlineFuncTmplDeclI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z18inlineFuncTmplDeclI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) inline void inlineFuncTmplDecl();
template<typename T>                              void inlineFuncTmplDecl() {}
USE(inlineFuncTmplDecl<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmplDef@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z17inlineFuncTmplDefI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmplDef@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z17inlineFuncTmplDefI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void inlineFuncTmplDef();
template<typename T>                inline void inlineFuncTmplDef() {}
USE(inlineFuncTmplDef<ImplicitInst_Imported>)


// Redeclarations
// MSC-DAG: declare dllimport void @"\01??$funcTmplRedecl1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z15funcTmplRedecl1I21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl1();
template<typename T> __declspec(dllimport) void funcTmplRedecl1();
USE(funcTmplRedecl1<ImplicitInst_Imported>)

// MSC-DAG: declare void @"\01??$funcTmplRedecl2@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: declare void @_Z15funcTmplRedecl2I24ImplicitInst_NotImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl2();
template<typename T>                       void funcTmplRedecl2(); // dllimport ignored
USE(funcTmplRedecl2<ImplicitInst_NotImported>)

// MSC-DAG: define linkonce_odr void @"\01??$funcTmplRedecl3@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15funcTmplRedecl3I24ImplicitInst_NotImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl3();
template<typename T>                       void funcTmplRedecl3() {} // dllimport ignored
USE(funcTmplRedecl3<ImplicitInst_NotImported>)


// Function template friends
// MSC-DAG: declare dllimport   void @"\01??$funcTmplFriend1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport   void @_Z15funcTmplFriend1I21ImplicitInst_ImportedEvv()
// MSC-DAG: declare             void @"\01??$funcTmplFriend2@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: declare             void @_Z15funcTmplFriend2I24ImplicitInst_NotImportedEvv()
// MSC-DAG: define linkonce_odr void @"\01??$funcTmplFriend3@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15funcTmplFriend3I24ImplicitInst_NotImportedEvv()
// MSC-DAG: declare dllimport   void @"\01??$funcTmplFriend4@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport   void @_Z15funcTmplFriend4I21ImplicitInst_ImportedEvv()
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllimport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllimport) void funcTmplFriend2();
  template<typename T> friend __declspec(dllimport) void funcTmplFriend3();
  template<typename T> friend __declspec(dllimport) inline void funcTmplFriend4();
};
template<typename T> __declspec(dllimport) void funcTmplFriend1();
template<typename T>                       void funcTmplFriend2(); // dllimport ignored
template<typename T>                       void funcTmplFriend3() {} // dllimport ignored
template<typename T>                       inline void funcTmplFriend4() {}
USE(funcTmplFriend1<ImplicitInst_Imported>)
USE(funcTmplFriend2<ImplicitInst_NotImported>)
USE(funcTmplFriend3<ImplicitInst_NotImported>)
USE(funcTmplFriend4<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$externalFuncTmpl@UImplicitInst_Imported@@@ns@@YAXXZ"()
// GNU-DAG: declare dllimport void @_ZN2ns16externalFuncTmplI21ImplicitInst_ImportedEEvv()
namespace ns { template<typename T> __declspec(dllimport) void externalFuncTmpl(); }
USE(ns::externalFuncTmpl<ImplicitInst_Imported>)


template<typename T> void funcTmpl() {}
template<typename T> inline void inlineFuncTmpl() {}
template<typename T> __declspec(dllimport) void importedFuncTmplDecl();
template<typename T> __declspec(dllimport) inline void importedFuncTmpl() {}

// Import implicit instantiation of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z20importedFuncTmplDeclI21ImplicitInst_ImportedEvv()
USE(importedFuncTmplDecl<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z16importedFuncTmplI21ImplicitInst_ImportedEvv()
USE(importedFuncTmpl<ImplicitInst_Imported>)

// Import explicit instantiation declaration of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI21ExplicitDecl_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z16importedFuncTmplI21ExplicitDecl_ImportedEvv()
extern template void importedFuncTmpl<ExplicitDecl_Imported>();
USE(importedFuncTmpl<ExplicitDecl_Imported>)

// Import explicit instantiation definition of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI21ExplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z16importedFuncTmplI21ExplicitInst_ImportedEvv()
template void importedFuncTmpl<ExplicitInst_Imported>();
USE(importedFuncTmpl<ExplicitInst_Imported>)


// Import specialization of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z20importedFuncTmplDeclI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void importedFuncTmplDecl<ExplicitSpec_Imported>();
USE(importedFuncTmplDecl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void importedFuncTmplDecl<ExplicitSpec_Def_Imported>() {}
//USE(importedFuncTmplDecl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z20importedFuncTmplDeclI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z20importedFuncTmplDeclI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void importedFuncTmplDecl<ExplicitSpec_InlineDef_Imported>() {}
USE(importedFuncTmplDecl<ExplicitSpec_InlineDef_Imported>)


// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Imported>();
USE(importedFuncTmpl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Def_Imported>() {}
//USE(importedFuncTmpl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z16importedFuncTmplI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void importedFuncTmpl<ExplicitSpec_InlineDef_Imported>() {}
USE(importedFuncTmpl<ExplicitSpec_InlineDef_Imported>)


// Not importing specialization of an imported function template without
// explicit dllimport.
// MSC-DAG: define void @"\01??$importedFuncTmpl@UExplicitSpec_NotImported@@@@YAXXZ"()
// GNU-DAG: define void @_Z16importedFuncTmplI24ExplicitSpec_NotImportedEvv()
template<> void importedFuncTmpl<ExplicitSpec_NotImported>() {}
USE(importedFuncTmpl<ExplicitSpec_NotImported>)


// Import explicit instantiation declaration of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitDecl_ImportedEvv()
// GNU-DAG: declare dllimport void @_Z14inlineFuncTmplI21ExplicitDecl_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z14inlineFuncTmplI21ExplicitDecl_ImportedEvv()
extern template __declspec(dllimport) void funcTmpl<ExplicitDecl_Imported>();
extern template __declspec(dllimport) void inlineFuncTmpl<ExplicitDecl_Imported>();
USE(funcTmpl<ExplicitDecl_Imported>)
USE(inlineFuncTmpl<ExplicitDecl_Imported>)


// Import explicit instantiation definition of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitInst_ImportedEvv()
// GNU-DAG: declare dllimport void @_Z14inlineFuncTmplI21ExplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$funcTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z8funcTmplI21ExplicitInst_ImportedEvv()
// GO1-DAG: define available_externally dllimport void @_Z14inlineFuncTmplI21ExplicitInst_ImportedEvv()
template __declspec(dllimport) void funcTmpl<ExplicitInst_Imported>();
template __declspec(dllimport) void inlineFuncTmpl<ExplicitInst_Imported>();
USE(funcTmpl<ExplicitInst_Imported>)
USE(inlineFuncTmpl<ExplicitInst_Imported>)


// Import specialization of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Imported>();
USE(funcTmpl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$funcTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Def_Imported>() {}
//USE(funcTmpl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z8funcTmplI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void funcTmpl<ExplicitSpec_InlineDef_Imported>() {}
USE(funcTmpl<ExplicitSpec_InlineDef_Imported>)
