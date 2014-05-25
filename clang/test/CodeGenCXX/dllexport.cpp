// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=MSC --check-prefix=M32 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=MSC --check-prefix=M64 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=GNU --check-prefix=G64 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Exported {};
struct ExplicitDecl_Exported {};
struct ExplicitInst_Exported {};
struct ExplicitSpec_Exported {};
struct ExplicitSpec_Def_Exported {};
struct ExplicitSpec_InlineDef_Exported {};
struct ExplicitSpec_NotExported {};
struct External { int v; };

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEVAR(var) int UNIQ(use)() { return var; }
#define USE(func) void UNIQ(use)() { func(); }
#define INSTVAR(var) template int var;
#define INST(func) template void func();


//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Declarations are not exported.
// MSC-NOT: @"\01?ExternGlobalDecl@@3HA"
// GNU-NOT: @ExternGlobalDecl
__declspec(dllexport) extern int ExternGlobalDecl;

// dllexport implies a definition.
// MSC-DAG: @"\01?GlobalDef@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalDef            = dllexport global i32 0, align 4
__declspec(dllexport) int GlobalDef;

// Export definition.
// MSC-DAG: @"\01?GlobalInit1@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalInit1            = dllexport global i32 1, align 4
__declspec(dllexport) int GlobalInit1 = 1;

// MSC-DAG: @"\01?GlobalInit2@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalInit2            = dllexport global i32 1, align 4
int __declspec(dllexport) GlobalInit2 = 1;

// Declare, then export definition.
// MSC-DAG: @"\01?GlobalDeclInit@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalDeclInit            = dllexport global i32 1, align 4
__declspec(dllexport) extern int GlobalDeclInit;
int GlobalDeclInit = 1;

// Redeclarations
// MSC-DAG: @"\01?GlobalRedecl1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalRedecl1            = dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl1;
__declspec(dllexport)        int GlobalRedecl1;

// MSC-DAG: @"\01?GlobalRedecl2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalRedecl2            = dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl2;
                             int GlobalRedecl2;

// MSC-DAG: @"\01?ExternalGlobal@ns@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_ZN2ns14ExternalGlobalE      = dllexport global i32 0, align 4
namespace ns { __declspec(dllexport) int ExternalGlobal; }

// MSC-DAG: @"\01?ExternalAutoTypeGlobal@@3UExternal@@A" = dllexport global %struct.External zeroinitializer, align 4
// GNU-DAG: @ExternalAutoTypeGlobal                      = dllexport global %struct.External zeroinitializer, align 4
__declspec(dllexport) auto ExternalAutoTypeGlobal = External();



//===----------------------------------------------------------------------===//
// Variable templates
//===----------------------------------------------------------------------===//

// Declarations are not exported.

// dllexport implies a definition.
// MSC-NOT: @"\01??$VarTmplDef@UExplicitInst_Exported@@@@3HA"
// GNU-NOT: @_Z10VarTmplDefI21ExplicitInst_ExportedE
template<typename T> __declspec(dllexport) int VarTmplDef;
INSTVAR(VarTmplDef<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplImplicitDef@UImplicitInst_Exported@@@@3HA" = external dllexport global
// GNU-DAG: @_Z18VarTmplImplicitDefI21ImplicitInst_ExportedE          = external dllexport global
template<typename T> __declspec(dllexport) int VarTmplImplicitDef;
USEVAR(VarTmplImplicitDef<ImplicitInst_Exported>)

// Export definition.
// MSC-DAG: @"\01??$VarTmplInit1@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z12VarTmplInit1I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template<typename T> __declspec(dllexport) int VarTmplInit1 = 1;
INSTVAR(VarTmplInit1<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplInit2@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z12VarTmplInit2I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template<typename T> int __declspec(dllexport) VarTmplInit2 = 1;
INSTVAR(VarTmplInit2<ExplicitInst_Exported>)

// Declare, then export definition.
// MSC-DAG: @"\01??$VarTmplDeclInit@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z15VarTmplDeclInitI21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template<typename T> __declspec(dllexport) extern int VarTmplDeclInit;
template<typename T>                              int VarTmplDeclInit = 1;
INSTVAR(VarTmplDeclInit<ExplicitInst_Exported>)

// Redeclarations
// MSC-DAG: @"\01??$VarTmplRedecl1@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z14VarTmplRedecl1I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template<typename T> __declspec(dllexport) extern int VarTmplRedecl1;
template<typename T> __declspec(dllexport)        int VarTmplRedecl1 = 1;
INSTVAR(VarTmplRedecl1<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplRedecl2@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z14VarTmplRedecl2I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template<typename T> __declspec(dllexport) extern int VarTmplRedecl2;
template<typename T>                              int VarTmplRedecl2 = 1;
INSTVAR(VarTmplRedecl2<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$ExternalVarTmpl@UExplicitInst_Exported@@@ns@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_ZN2ns15ExternalVarTmplI21ExplicitInst_ExportedEE        = weak_odr dllexport global i32 1, align 4
namespace ns { template<typename T> __declspec(dllexport) int ExternalVarTmpl = 1; }
INSTVAR(ns::ExternalVarTmpl<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$ExternalAutoTypeVarTmpl@UExplicitInst_Exported@@@@3UExternal@@A" = weak_odr dllexport global %struct.External zeroinitializer, align 4
// GNU-DAG: @_Z23ExternalAutoTypeVarTmplI21ExplicitInst_ExportedE                    = weak_odr dllexport global %struct.External zeroinitializer, align 4
template<typename T> __declspec(dllexport) auto ExternalAutoTypeVarTmpl = External();
template External ExternalAutoTypeVarTmpl<ExplicitInst_Exported>;


template<typename T> int VarTmpl = 1;
template<typename T> __declspec(dllexport) int ExportedVarTmpl = 1;

// Export implicit instantiation of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ImplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
USEVAR(ExportedVarTmpl<ImplicitInst_Exported>)

// Export explicit instantiation declaration of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitDecl_ExportedE          = weak_odr dllexport global i32 1, align 4
extern template int ExportedVarTmpl<ExplicitDecl_Exported>;
       template int ExportedVarTmpl<ExplicitDecl_Exported>;

// Export explicit instantiation definition of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, align 4
template __declspec(dllexport) int ExportedVarTmpl<ExplicitInst_Exported>;

// Export specialization of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_Exported@@@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitSpec_ExportedE          = dllexport global i32 0, align 4
template<> __declspec(dllexport) int ExportedVarTmpl<ExplicitSpec_Exported>;

// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_Def_Exported@@@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @_Z15ExportedVarTmplI25ExplicitSpec_Def_ExportedE          = dllexport global i32 1, align 4
template<> __declspec(dllexport) int ExportedVarTmpl<ExplicitSpec_Def_Exported> = 1;

// Not exporting specialization of an exported variable template without
// explicit dllexport.
// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_NotExported@@@@3HA" = global i32 0, align 4
// GNU-DAG: @_Z15ExportedVarTmplI24ExplicitSpec_NotExportedE          = global i32 0, align 4
template<> int ExportedVarTmpl<ExplicitSpec_NotExported>;


// Export explicit instantiation declaration of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitDecl_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitDecl_ExportedE           = weak_odr dllexport global i32 1, align 4
extern template __declspec(dllexport) int VarTmpl<ExplicitDecl_Exported>;
       template __declspec(dllexport) int VarTmpl<ExplicitDecl_Exported>;

// Export explicit instantiation definition of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitInst_ExportedE           = weak_odr dllexport global i32 1, align 4
template __declspec(dllexport) int VarTmpl<ExplicitInst_Exported>;

// Export specialization of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Exported@@@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitSpec_ExportedE           = dllexport global i32 0, align 4
template<> __declspec(dllexport) int VarTmpl<ExplicitSpec_Exported>;

// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Def_Exported@@@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @_Z7VarTmplI25ExplicitSpec_Def_ExportedE           = dllexport global i32 1, align 4
template<> __declspec(dllexport) int VarTmpl<ExplicitSpec_Def_Exported> = 1;



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Declarations are not exported.

// Export function definition.
// MSC-DAG: define dllexport void @"\01?def@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z3defv()
__declspec(dllexport) void def() {}

// extern "C"
// MSC-DAG: define dllexport void @externC()
// GNU-DAG: define dllexport void @externC()
extern "C" __declspec(dllexport) void externC() {}

// Export inline function.
// MSC-DAG: define weak_odr dllexport void @"\01?inlineFunc@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z10inlineFuncv()
__declspec(dllexport) inline void inlineFunc() {}

// MSC-DAG: define weak_odr dllexport void @"\01?inlineDecl@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z10inlineDeclv()
__declspec(dllexport) inline void inlineDecl();
                             void inlineDecl() {}

// MSC-DAG: define weak_odr dllexport void @"\01?inlineDef@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z9inlineDefv()
__declspec(dllexport) void inlineDef();
               inline void inlineDef() {}

// Redeclarations
// MSC-DAG: define dllexport void @"\01?redecl1@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7redecl1v()
__declspec(dllexport) void redecl1();
__declspec(dllexport) void redecl1() {}

// MSC-DAG: define dllexport void @"\01?redecl2@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7redecl2v()
__declspec(dllexport) void redecl2();
                      void redecl2() {}

// Friend functions
// MSC-DAG: define dllexport void @"\01?friend1@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7friend1v()
// MSC-DAG: define dllexport void @"\01?friend2@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7friend2v()
struct FuncFriend {
  friend __declspec(dllexport) void friend1();
  friend __declspec(dllexport) void friend2();
};
__declspec(dllexport) void friend1() {}
                      void friend2() {}

// Implicit declarations can be redeclared with dllexport.
// MSC-DAG: define dllexport noalias i8* @"\01??2@{{YAPAXI|YAPEAX_K}}@Z"(
// GNU-DAG: define dllexport noalias i8* @_Znw{{[yj]}}(
void* alloc(__SIZE_TYPE__ n);
__declspec(dllexport) void* operator new(__SIZE_TYPE__ n) { return alloc(n); }

// MSC-DAG: define dllexport void @"\01?externalFunc@ns@@YAXXZ"()
// GNU-DAG: define dllexport void @_ZN2ns12externalFuncEv()
namespace ns { __declspec(dllexport) void externalFunc() {} }



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Export function template definition.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplDef@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z11funcTmplDefI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplDef() {}
INST(funcTmplDef<ExplicitInst_Exported>)

// Export inline function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmpl1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15inlineFuncTmpl1I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) inline void inlineFuncTmpl1() {}
INST(inlineFuncTmpl1<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmpl2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15inlineFuncTmpl2I21ExplicitInst_ExportedEvv()
template<typename T> inline void __attribute__((dllexport)) inlineFuncTmpl2() {}
INST(inlineFuncTmpl2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmplDecl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z18inlineFuncTmplDeclI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) inline void inlineFuncTmplDecl();
template<typename T>                              void inlineFuncTmplDecl() {}
INST(inlineFuncTmplDecl<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmplDef@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z17inlineFuncTmplDefI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void inlineFuncTmplDef();
template<typename T>                inline void inlineFuncTmplDef() {}
INST(inlineFuncTmplDef<ExplicitInst_Exported>)


// Redeclarations
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl1I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl1();
template<typename T> __declspec(dllexport) void funcTmplRedecl1() {}
INST(funcTmplRedecl1<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl2I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl2();
template<typename T>                       void funcTmplRedecl2() {}
INST(funcTmplRedecl2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl3@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl3I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl3();
template<typename T>                       void funcTmplRedecl3() {}
INST(funcTmplRedecl3<ExplicitInst_Exported>)


// Function template friends
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplFriend1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplFriend1I21ExplicitInst_ExportedEvv()
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplFriend2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplFriend2I21ExplicitInst_ExportedEvv()
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllexport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllexport) void funcTmplFriend2();
};
template<typename T> __declspec(dllexport) void funcTmplFriend1() {}
template<typename T>                       void funcTmplFriend2() {}
INST(funcTmplFriend1<ExplicitInst_Exported>)
INST(funcTmplFriend2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$externalFuncTmpl@UExplicitInst_Exported@@@ns@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_ZN2ns16externalFuncTmplI21ExplicitInst_ExportedEEvv()
namespace ns { template<typename T> __declspec(dllexport) void externalFuncTmpl() {} }
INST(ns::externalFuncTmpl<ExplicitInst_Exported>)


template<typename T> void funcTmpl() {}
template<typename T> __declspec(dllexport) void exportedFuncTmpl() {}

// Export implicit instantiation of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UImplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ImplicitInst_ExportedEvv()
USE(exportedFuncTmpl<ImplicitInst_Exported>)

// Export explicit instantiation declaration of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitDecl_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ExplicitDecl_ExportedEvv()
extern template void exportedFuncTmpl<ExplicitDecl_Exported>();
       template void exportedFuncTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ExplicitInst_ExportedEvv()
template void exportedFuncTmpl<ExplicitInst_Exported>();

// Export specialization of an exported function template.
// MSC-DAG: define dllexport void @"\01??$exportedFuncTmpl@UExplicitSpec_Def_Exported@@@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z16exportedFuncTmplI25ExplicitSpec_Def_ExportedEvv()
template<> __declspec(dllexport) void exportedFuncTmpl<ExplicitSpec_Def_Exported>() {}

// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitSpec_InlineDef_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI31ExplicitSpec_InlineDef_ExportedEvv()
template<> __declspec(dllexport) inline void exportedFuncTmpl<ExplicitSpec_InlineDef_Exported>() {}

// Not exporting specialization of an exported function template without
// explicit dllexport.
// MSC-DAG: define void @"\01??$exportedFuncTmpl@UExplicitSpec_NotExported@@@@YAXXZ"()
// GNU-DAG: define void @_Z16exportedFuncTmplI24ExplicitSpec_NotExportedEvv()
template<> void exportedFuncTmpl<ExplicitSpec_NotExported>() {}


// Export explicit instantiation declaration of a non-exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitDecl_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI21ExplicitDecl_ExportedEvv()
extern template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();
       template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of a non-exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI21ExplicitInst_ExportedEvv()
template __declspec(dllexport) void funcTmpl<ExplicitInst_Exported>();

// Export specialization of a non-exported function template.
// MSC-DAG: define dllexport void @"\01??$funcTmpl@UExplicitSpec_Def_Exported@@@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z8funcTmplI25ExplicitSpec_Def_ExportedEvv()
template<> __declspec(dllexport) void funcTmpl<ExplicitSpec_Def_Exported>() {}

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI31ExplicitSpec_InlineDef_ExportedEvv()
template<> __declspec(dllexport) inline void funcTmpl<ExplicitSpec_InlineDef_Exported>() {}



//===----------------------------------------------------------------------===//
// Precedence
//===----------------------------------------------------------------------===//

// dllexport takes precedence over the dllimport if both are specified.
// MSC-DAG: @"\01?PrecedenceGlobal1A@@3HA" = dllexport global i32 0, align 4
// MSC-DAG: @"\01?PrecedenceGlobal1B@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal1A            = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal1B            = dllexport global i32 0, align 4
__attribute__((dllimport, dllexport))       int PrecedenceGlobal1A; // dllimport ignored
__declspec(dllimport) __declspec(dllexport) int PrecedenceGlobal1B; // dllimport ignored

// MSC-DAG: @"\01?PrecedenceGlobal2A@@3HA" = dllexport global i32 0, align 4
// MSC-DAG: @"\01?PrecedenceGlobal2B@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal2A            = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal2B            = dllexport global i32 0, align 4
__attribute__((dllexport, dllimport))       int PrecedenceGlobal2A; // dllimport ignored
__declspec(dllexport) __declspec(dllimport) int PrecedenceGlobal2B; // dllimport ignored

// MSC-DAG: @"\01?PrecedenceGlobalRedecl1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalRedecl1            = dllexport global i32 0, align 4
__declspec(dllexport) extern int PrecedenceGlobalRedecl1;
__declspec(dllimport)        int PrecedenceGlobalRedecl1 = 0;

// MSC-DAG: @"\01?PrecedenceGlobalRedecl2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalRedecl2            = dllexport global i32 0, align 4
__declspec(dllimport) extern int PrecedenceGlobalRedecl2;
__declspec(dllexport)        int PrecedenceGlobalRedecl2;

// MSC-DAG: @"\01?PrecedenceGlobalMixed1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalMixed1            = dllexport global i32 0, align 4
__attribute__((dllexport)) extern int PrecedenceGlobalMixed1;
__declspec(dllimport)             int PrecedenceGlobalMixed1 = 0;

// MSC-DAG: @"\01?PrecedenceGlobalMixed2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalMixed2            = dllexport global i32 0, align 4
__attribute__((dllimport)) extern int PrecedenceGlobalMixed2;
__declspec(dllexport)             int PrecedenceGlobalMixed2;

// MSC-DAG: define dllexport void @"\01?precedence1A@@YAXXZ"
// MSC-DAG: define dllexport void @"\01?precedence1B@@YAXXZ"
// GNU-DAG: define dllexport void @_Z12precedence1Av()
// GNU-DAG: define dllexport void @_Z12precedence1Bv()
void __attribute__((dllimport, dllexport))       precedence1A() {}
void __declspec(dllimport) __declspec(dllexport) precedence1B() {}

// MSC-DAG: define dllexport void @"\01?precedence2A@@YAXXZ"
// MSC-DAG: define dllexport void @"\01?precedence2B@@YAXXZ"
// GNU-DAG: define dllexport void @_Z12precedence2Av()
// GNU-DAG: define dllexport void @_Z12precedence2Bv()
void __attribute__((dllexport, dllimport))       precedence2A() {}
void __declspec(dllexport) __declspec(dllimport) precedence2B() {}

// MSC-DAG: define dllexport void @"\01?precedenceRedecl1@@YAXXZ"
// GNU-DAG: define dllexport void @_Z17precedenceRedecl1v()
void __declspec(dllimport) precedenceRedecl1();
void __declspec(dllexport) precedenceRedecl1() {}

// MSC-DAG: define dllexport void @"\01?precedenceRedecl2@@YAXXZ"
// GNU-DAG: define dllexport void @_Z17precedenceRedecl2v()
void __declspec(dllexport) precedenceRedecl2();
void __declspec(dllimport) precedenceRedecl2() {}
