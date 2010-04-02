//===--- CodeGenModule.h - Per-Module state for LLVM CodeGen ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENMODULE_H
#define CLANG_CODEGEN_CODEGENMODULE_H

#include "clang/Basic/LangOptions.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "CGBlocks.h"
#include "CGCall.h"
#include "CGCXX.h"
#include "CGVtable.h"
#include "CodeGenTypes.h"
#include "GlobalDecl.h"
#include "Mangle.h"
#include "llvm/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
  class Module;
  class Constant;
  class Function;
  class GlobalValue;
  class TargetData;
  class FunctionType;
  class LLVMContext;
}

namespace clang {
  class TargetCodeGenInfo;
  class ASTContext;
  class FunctionDecl;
  class IdentifierInfo;
  class ObjCMethodDecl;
  class ObjCImplementationDecl;
  class ObjCCategoryImplDecl;
  class ObjCProtocolDecl;
  class ObjCEncodeExpr;
  class BlockExpr;
  class CharUnits;
  class Decl;
  class Expr;
  class Stmt;
  class StringLiteral;
  class NamedDecl;
  class ValueDecl;
  class VarDecl;
  class LangOptions;
  class CodeGenOptions;
  class Diagnostic;
  class AnnotateAttr;
  class CXXDestructorDecl;

namespace CodeGen {

  class CodeGenFunction;
  class CGDebugInfo;
  class CGObjCRuntime;
  class MangleBuffer;
  
/// CodeGenModule - This class organizes the cross-function state that is used
/// while generating LLVM code.
class CodeGenModule : public BlockModule {
  CodeGenModule(const CodeGenModule&);  // DO NOT IMPLEMENT
  void operator=(const CodeGenModule&); // DO NOT IMPLEMENT

  typedef std::vector<std::pair<llvm::Constant*, int> > CtorList;

  ASTContext &Context;
  const LangOptions &Features;
  const CodeGenOptions &CodeGenOpts;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  mutable const TargetCodeGenInfo *TheTargetCodeGenInfo;
  Diagnostic &Diags;
  CodeGenTypes Types;
  MangleContext MangleCtx;

  /// VTables - Holds information about C++ vtables.
  CodeGenVTables VTables;
  
  CGObjCRuntime* Runtime;
  CGDebugInfo* DebugInfo;
  
  llvm::Function *MemCpyFn;
  llvm::Function *MemMoveFn;
  llvm::Function *MemSetFn;

  // WeakRefReferences - A set of references that have only been seen via
  // a weakref so far. This is used to remove the weak of the reference if we ever
  // see a direct reference or a definition.
  llvm::SmallPtrSet<llvm::GlobalValue*, 10> WeakRefReferences;

  /// DeferredDecls - This contains all the decls which have definitions but
  /// which are deferred for emission and therefore should only be output if
  /// they are actually used.  If a decl is in this, then it is known to have
  /// not been referenced yet.
  llvm::StringMap<GlobalDecl> DeferredDecls;

  /// DeferredDeclsToEmit - This is a list of deferred decls which we have seen
  /// that *are* actually referenced.  These get code generated when the module
  /// is done.
  std::vector<GlobalDecl> DeferredDeclsToEmit;

  /// LLVMUsed - List of global values which are required to be
  /// present in the object file; bitcast to i8*. This is used for
  /// forcing visibility of symbols which may otherwise be optimized
  /// out.
  std::vector<llvm::WeakVH> LLVMUsed;

  /// GlobalCtors - Store the list of global constructors and their respective
  /// priorities to be emitted when the translation unit is complete.
  CtorList GlobalCtors;

  /// GlobalDtors - Store the list of global destructors and their respective
  /// priorities to be emitted when the translation unit is complete.
  CtorList GlobalDtors;

  std::vector<llvm::Constant*> Annotations;

  llvm::StringMap<llvm::Constant*> CFConstantStringMap;
  llvm::StringMap<llvm::Constant*> ConstantStringMap;

  /// CXXGlobalInits - Global variables with initializers that need to run
  /// before main.
  std::vector<llvm::Constant*> CXXGlobalInits;

  /// CXXGlobalDtors - Global destructor functions and arguments that need to
  /// run on termination.
  std::vector<std::pair<llvm::Constant*,llvm::Constant*> > CXXGlobalDtors;

  /// CFConstantStringClassRef - Cached reference to the class for constant
  /// strings. This value has type int * but is actually an Obj-C class pointer.
  llvm::Constant *CFConstantStringClassRef;

  /// Lazily create the Objective-C runtime
  void createObjCRuntime();

  llvm::LLVMContext &VMContext;
public:
  CodeGenModule(ASTContext &C, const CodeGenOptions &CodeGenOpts,
                llvm::Module &M, const llvm::TargetData &TD, Diagnostic &Diags);

  ~CodeGenModule();

  /// Release - Finalize LLVM code generation.
  void Release();

  /// getObjCRuntime() - Return a reference to the configured
  /// Objective-C runtime.
  CGObjCRuntime &getObjCRuntime() {
    if (!Runtime) createObjCRuntime();
    return *Runtime;
  }

  /// hasObjCRuntime() - Return true iff an Objective-C runtime has
  /// been configured.
  bool hasObjCRuntime() { return !!Runtime; }

  CGDebugInfo *getDebugInfo() { return DebugInfo; }
  ASTContext &getContext() const { return Context; }
  const CodeGenOptions &getCodeGenOpts() const { return CodeGenOpts; }
  const LangOptions &getLangOptions() const { return Features; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  MangleContext &getMangleContext() { return MangleCtx; }
  CodeGenVTables &getVTables() { return VTables; }
  Diagnostic &getDiags() const { return Diags; }
  const llvm::TargetData &getTargetData() const { return TheTargetData; }
  llvm::LLVMContext &getLLVMContext() { return VMContext; }
  const TargetCodeGenInfo &getTargetCodeGenInfo() const;
  bool isTargetDarwin() const;

  /// getDeclVisibilityMode - Compute the visibility of the decl \arg D.
  LangOptions::VisibilityMode getDeclVisibilityMode(const Decl *D) const;

  /// setGlobalVisibility - Set the visibility for the given LLVM
  /// GlobalValue.
  void setGlobalVisibility(llvm::GlobalValue *GV, const Decl *D) const;

  llvm::Constant *GetAddrOfGlobal(GlobalDecl GD) {
    if (isa<CXXConstructorDecl>(GD.getDecl()))
      return GetAddrOfCXXConstructor(cast<CXXConstructorDecl>(GD.getDecl()),
                                     GD.getCtorType());
    else if (isa<CXXDestructorDecl>(GD.getDecl()))
      return GetAddrOfCXXDestructor(cast<CXXDestructorDecl>(GD.getDecl()),
                                     GD.getDtorType());
    else if (isa<FunctionDecl>(GD.getDecl()))
      return GetAddrOfFunction(GD);
    else
      return GetAddrOfGlobalVar(cast<VarDecl>(GD.getDecl()));
  }

  /// GetAddrOfGlobalVar - Return the llvm::Constant for the address of the
  /// given global variable.  If Ty is non-null and if the global doesn't exist,
  /// then it will be greated with the specified type instead of whatever the
  /// normal requested type would be.
  llvm::Constant *GetAddrOfGlobalVar(const VarDecl *D,
                                     const llvm::Type *Ty = 0);

  /// GetAddrOfFunction - Return the address of the given function.  If Ty is
  /// non-null, then this function will use the specified type if it has to
  /// create it.
  llvm::Constant *GetAddrOfFunction(GlobalDecl GD,
                                    const llvm::Type *Ty = 0);

  /// GetAddrOfRTTIDescriptor - Get the address of the RTTI descriptor 
  /// for the given type.
  llvm::Constant *GetAddrOfRTTIDescriptor(QualType Ty);

  /// GetAddrOfThunk - Get the address of the thunk for the given global decl.
  llvm::Constant *GetAddrOfThunk(GlobalDecl GD, const ThunkInfo &Thunk);

  /// GetWeakRefReference - Get a reference to the target of VD.
  llvm::Constant *GetWeakRefReference(const ValueDecl *VD);

  /// GetNonVirtualBaseClassOffset - Returns the offset from a derived class to 
  /// its base class. Returns null if the offset is 0. 
  llvm::Constant *
  GetNonVirtualBaseClassOffset(const CXXRecordDecl *ClassDecl,
                               const CXXRecordDecl *BaseClassDecl);

  /// GetStringForStringLiteral - Return the appropriate bytes for a string
  /// literal, properly padded to match the literal type. If only the address of
  /// a constant is needed consider using GetAddrOfConstantStringLiteral.
  std::string GetStringForStringLiteral(const StringLiteral *E);

  /// GetAddrOfConstantCFString - Return a pointer to a constant CFString object
  /// for the given string.
  llvm::Constant *GetAddrOfConstantCFString(const StringLiteral *Literal);

  /// GetAddrOfConstantStringFromLiteral - Return a pointer to a constant array
  /// for the given string literal.
  llvm::Constant *GetAddrOfConstantStringFromLiteral(const StringLiteral *S);

  /// GetAddrOfConstantStringFromObjCEncode - Return a pointer to a constant
  /// array for the given ObjCEncodeExpr node.
  llvm::Constant *GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *);

  /// GetAddrOfConstantString - Returns a pointer to a character array
  /// containing the literal. This contents are exactly that of the given
  /// string, i.e. it will not be null terminated automatically; see
  /// GetAddrOfConstantCString. Note that whether the result is actually a
  /// pointer to an LLVM constant depends on Feature.WriteableStrings.
  ///
  /// The result has pointer to array type.
  ///
  /// \param GlobalName If provided, the name to use for the global
  /// (if one is created).
  llvm::Constant *GetAddrOfConstantString(const std::string& str,
                                          const char *GlobalName=0);

  /// GetAddrOfConstantCString - Returns a pointer to a character array
  /// containing the literal and a terminating '\0' character. The result has
  /// pointer to array type.
  ///
  /// \param GlobalName If provided, the name to use for the global (if one is
  /// created).
  llvm::Constant *GetAddrOfConstantCString(const std::string &str,
                                           const char *GlobalName=0);

  /// GetAddrOfCXXConstructor - Return the address of the constructor of the
  /// given type.
  llvm::GlobalValue *GetAddrOfCXXConstructor(const CXXConstructorDecl *D,
                                             CXXCtorType Type);

  /// GetAddrOfCXXDestructor - Return the address of the constructor of the
  /// given type.
  llvm::GlobalValue *GetAddrOfCXXDestructor(const CXXDestructorDecl *D,
                                            CXXDtorType Type);

  /// getBuiltinLibFunction - Given a builtin id for a function like
  /// "__builtin_fabsf", return a Function* for "fabsf".
  llvm::Value *getBuiltinLibFunction(const FunctionDecl *FD,
                                     unsigned BuiltinID);

  llvm::Function *getMemCpyFn();
  llvm::Function *getMemMoveFn();
  llvm::Function *getMemSetFn();
  llvm::Function *getIntrinsic(unsigned IID, const llvm::Type **Tys = 0,
                               unsigned NumTys = 0);

  /// EmitTopLevelDecl - Emit code for a single top level declaration.
  void EmitTopLevelDecl(Decl *D);

  /// AddUsedGlobal - Add a global which should be forced to be
  /// present in the object file; these are emitted to the llvm.used
  /// metadata global.
  void AddUsedGlobal(llvm::GlobalValue *GV);

  void AddAnnotation(llvm::Constant *C) { Annotations.push_back(C); }

  /// AddCXXDtorEntry - Add a destructor and object to add to the C++ global
  /// destructor function.
  void AddCXXDtorEntry(llvm::Constant *DtorFn, llvm::Constant *Object);

  /// CreateRuntimeFunction - Create a new runtime function with the specified
  /// type and name.
  llvm::Constant *CreateRuntimeFunction(const llvm::FunctionType *Ty,
                                        llvm::StringRef Name);
  /// CreateRuntimeVariable - Create a new runtime global variable with the
  /// specified type and name.
  llvm::Constant *CreateRuntimeVariable(const llvm::Type *Ty,
                                        llvm::StringRef Name);

  void UpdateCompletedType(const TagDecl *TD) {
    // Make sure that this type is translated.
    Types.UpdateCompletedType(TD);
  }

  /// EmitConstantExpr - Try to emit the given expression as a
  /// constant; returns 0 if the expression cannot be emitted as a
  /// constant.
  llvm::Constant *EmitConstantExpr(const Expr *E, QualType DestType,
                                   CodeGenFunction *CGF = 0);

  /// EmitNullConstant - Return the result of value-initializing the given
  /// type, i.e. a null expression of the given type.  This is usually,
  /// but not always, an LLVM null constant.
  llvm::Constant *EmitNullConstant(QualType T);

  llvm::Constant *EmitAnnotateAttr(llvm::GlobalValue *GV,
                                   const AnnotateAttr *AA, unsigned LineNo);

  llvm::Constant *EmitPointerToDataMember(const FieldDecl *FD);

  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified stmt yet.
  /// \param OmitOnError - If true, then this error should only be emitted if no
  /// other errors have been reported.
  void ErrorUnsupported(const Stmt *S, const char *Type,
                        bool OmitOnError=false);

  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified decl yet.
  /// \param OmitOnError - If true, then this error should only be emitted if no
  /// other errors have been reported.
  void ErrorUnsupported(const Decl *D, const char *Type,
                        bool OmitOnError=false);

  /// SetInternalFunctionAttributes - Set the attributes on the LLVM
  /// function for the given decl and function info. This applies
  /// attributes necessary for handling the ABI as well as user
  /// specified attributes like section.
  void SetInternalFunctionAttributes(const Decl *D, llvm::Function *F,
                                     const CGFunctionInfo &FI);

  /// SetLLVMFunctionAttributes - Set the LLVM function attributes
  /// (sext, zext, etc).
  void SetLLVMFunctionAttributes(const Decl *D,
                                 const CGFunctionInfo &Info,
                                 llvm::Function *F);

  /// SetLLVMFunctionAttributesForDefinition - Set the LLVM function attributes
  /// which only apply to a function definintion.
  void SetLLVMFunctionAttributesForDefinition(const Decl *D, llvm::Function *F);

  /// ReturnTypeUsesSret - Return true iff the given type uses 'sret' when used
  /// as a return type.
  bool ReturnTypeUsesSret(const CGFunctionInfo &FI);

  /// ConstructAttributeList - Get the LLVM attributes and calling convention to
  /// use for a particular function type.
  ///
  /// \param Info - The function type information.
  /// \param TargetDecl - The decl these attributes are being constructed
  /// for. If supplied the attributes applied to this decl may contribute to the
  /// function attributes and calling convention.
  /// \param PAL [out] - On return, the attribute list to use.
  /// \param CallingConv [out] - On return, the LLVM calling convention to use.
  void ConstructAttributeList(const CGFunctionInfo &Info,
                              const Decl *TargetDecl,
                              AttributeListType &PAL,
                              unsigned &CallingConv);

  void getMangledName(MangleBuffer &Buffer, GlobalDecl D);
  void getMangledName(MangleBuffer &Buffer, const NamedDecl *ND);
  void getMangledCXXCtorName(MangleBuffer &Buffer,
                             const CXXConstructorDecl *D,
                             CXXCtorType Type);
  void getMangledCXXDtorName(MangleBuffer &Buffer,
                             const CXXDestructorDecl *D,
                             CXXDtorType Type);

  void EmitTentativeDefinition(const VarDecl *D);

  enum GVALinkage {
    GVA_Internal,
    GVA_C99Inline,
    GVA_CXXInline,
    GVA_StrongExternal,
    GVA_TemplateInstantiation,
    GVA_ExplicitTemplateInstantiation
  };

  llvm::GlobalVariable::LinkageTypes
  getFunctionLinkage(const FunctionDecl *FD);

  /// getVtableLinkage - Return the appropriate linkage for the vtable, VTT,
  /// and type information of the given class.
  static llvm::GlobalVariable::LinkageTypes 
  getVtableLinkage(const CXXRecordDecl *RD);

  /// GetTargetTypeStoreSize - Return the store size, in character units, of
  /// the given LLVM type.
  CharUnits GetTargetTypeStoreSize(const llvm::Type *Ty) const;

  std::vector<const CXXRecordDecl*> DeferredVtables;

private:
  llvm::GlobalValue *GetGlobalValue(llvm::StringRef Ref);

  llvm::Constant *GetOrCreateLLVMFunction(llvm::StringRef MangledName,
                                          const llvm::Type *Ty,
                                          GlobalDecl D);
  llvm::Constant *GetOrCreateLLVMGlobal(llvm::StringRef MangledName,
                                        const llvm::PointerType *PTy,
                                        const VarDecl *D);

  /// SetCommonAttributes - Set attributes which are common to any
  /// form of a global definition (alias, Objective-C method,
  /// function, global variable).
  ///
  /// NOTE: This should only be called for definitions.
  void SetCommonAttributes(const Decl *D, llvm::GlobalValue *GV);

  /// SetFunctionDefinitionAttributes - Set attributes for a global definition.
  void SetFunctionDefinitionAttributes(const FunctionDecl *D,
                                       llvm::GlobalValue *GV);

  /// SetFunctionAttributes - Set function attributes for a function
  /// declaration.
  void SetFunctionAttributes(GlobalDecl GD,
                             llvm::Function *F,
                             bool IsIncompleteFunction);

  /// EmitGlobal - Emit code for a singal global function or var decl. Forward
  /// declarations are emitted lazily.
  void EmitGlobal(GlobalDecl D);

  void EmitGlobalDefinition(GlobalDecl D);

  void EmitGlobalFunctionDefinition(GlobalDecl GD);
  void EmitGlobalVarDefinition(const VarDecl *D);
  void EmitAliasDefinition(GlobalDecl GD);
  void EmitObjCPropertyImplementations(const ObjCImplementationDecl *D);

  // C++ related functions.

  bool TryEmitDefinitionAsAlias(GlobalDecl Alias, GlobalDecl Target);
  bool TryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D);

  void EmitNamespace(const NamespaceDecl *D);
  void EmitLinkageSpec(const LinkageSpecDecl *D);

  /// EmitCXXConstructors - Emit constructors (base, complete) from a
  /// C++ constructor Decl.
  void EmitCXXConstructors(const CXXConstructorDecl *D);

  /// EmitCXXConstructor - Emit a single constructor with the given type from
  /// a C++ constructor Decl.
  void EmitCXXConstructor(const CXXConstructorDecl *D, CXXCtorType Type);

  /// EmitCXXDestructors - Emit destructors (base, complete) from a
  /// C++ destructor Decl.
  void EmitCXXDestructors(const CXXDestructorDecl *D);

  /// EmitCXXDestructor - Emit a single destructor with the given type from
  /// a C++ destructor Decl.
  void EmitCXXDestructor(const CXXDestructorDecl *D, CXXDtorType Type);

  /// EmitCXXGlobalInitFunc - Emit the function that initializes C++ globals.
  void EmitCXXGlobalInitFunc();

  /// EmitCXXGlobalDtorFunc - Emit the function that destroys C++ globals.
  void EmitCXXGlobalDtorFunc();

  void EmitCXXGlobalVarDeclInitFunc(const VarDecl *D);

  // FIXME: Hardcoding priority here is gross.
  void AddGlobalCtor(llvm::Function *Ctor, int Priority=65535);
  void AddGlobalDtor(llvm::Function *Dtor, int Priority=65535);

  /// EmitCtorList - Generates a global array of functions and priorities using
  /// the given list and name. This array will have appending linkage and is
  /// suitable for use as a LLVM constructor or destructor array.
  void EmitCtorList(const CtorList &Fns, const char *GlobalName);

  void EmitAnnotations(void);

  /// EmitFundamentalRTTIDescriptor - Emit the RTTI descriptors for the
  /// given type.
  void EmitFundamentalRTTIDescriptor(QualType Type);

  /// EmitFundamentalRTTIDescriptors - Emit the RTTI descriptors for the
  /// builtin types.
  void EmitFundamentalRTTIDescriptors();

  /// EmitDeferred - Emit any needed decls for which code generation
  /// was deferred.
  void EmitDeferred(void);

  /// EmitLLVMUsed - Emit the llvm.used metadata used to force
  /// references to global which may otherwise be optimized out.
  void EmitLLVMUsed(void);

  /// MayDeferGeneration - Determine if the given decl can be emitted
  /// lazily; this is only relevant for definitions. The given decl
  /// must be either a function or var decl.
  bool MayDeferGeneration(const ValueDecl *D);
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
