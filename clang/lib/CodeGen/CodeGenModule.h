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
#include "CGBlocks.h"
#include "CGCall.h"
#include "CGCXX.h"
#include "CodeGenTypes.h"
#include "llvm/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ValueHandle.h"
#include <list>

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
  class ASTContext;
  class FunctionDecl;
  class IdentifierInfo;
  class ObjCMethodDecl;
  class ObjCImplementationDecl;
  class ObjCCategoryImplDecl;
  class ObjCProtocolDecl;
  class ObjCEncodeExpr;
  class BlockExpr;
  class Decl;
  class Expr;
  class Stmt;
  class StringLiteral;
  class NamedDecl;
  class ValueDecl;
  class VarDecl;
  class LangOptions;
  class CompileOptions;
  class Diagnostic;
  class AnnotateAttr;
  class CXXDestructorDecl;

namespace CodeGen {

  class CodeGenFunction;
  class CGDebugInfo;
  class CGObjCRuntime;

/// GlobalDecl - represents a global declaration. This can either be a
/// CXXConstructorDecl and the constructor type (Base, Complete).
/// a CXXDestructorDecl and the destructor type (Base, Complete) or
// a regular VarDecl or a FunctionDecl.
class GlobalDecl {
  llvm::PointerIntPair<const ValueDecl*, 2> Value;
  
public:
  GlobalDecl() {}
  
  explicit GlobalDecl(const ValueDecl *VD) : Value(VD, 0) {
    assert(!isa<CXXConstructorDecl>(VD) && "Use other ctor with ctor decls!");
    assert(!isa<CXXDestructorDecl>(VD) && "Use other ctor with dtor decls!");
  }
  GlobalDecl(const CXXConstructorDecl *D, CXXCtorType Type) 
  : Value(D, Type) {}
  GlobalDecl(const CXXDestructorDecl *D, CXXDtorType Type)
  : Value(D, Type) {}
  
  const ValueDecl *getDecl() const { return Value.getPointer(); }
  
  CXXCtorType getCtorType() const {
    assert(isa<CXXConstructorDecl>(getDecl()) && "Decl is not a ctor!");
    return static_cast<CXXCtorType>(Value.getInt());
  }
  
  CXXDtorType getDtorType() const {
    assert(isa<CXXDestructorDecl>(getDecl()) && "Decl is not a dtor!");
    return static_cast<CXXDtorType>(Value.getInt());
  }
};
  
/// CodeGenModule - This class organizes the cross-function state that is used
/// while generating LLVM code.
class CodeGenModule : public BlockModule {
  CodeGenModule(const CodeGenModule&);  // DO NOT IMPLEMENT
  void operator=(const CodeGenModule&); // DO NOT IMPLEMENT

  typedef std::vector< std::pair<llvm::Constant*, int> > CtorList;

  ASTContext &Context;
  const LangOptions &Features;
  const CompileOptions &CompileOpts;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  Diagnostic &Diags;
  CodeGenTypes Types;
  CGObjCRuntime* Runtime;
  CGDebugInfo* DebugInfo;

  llvm::Function *MemCpyFn;
  llvm::Function *MemMoveFn;
  llvm::Function *MemSetFn;

  /// GlobalDeclMap - Mapping of decl names (represented as unique
  /// character pointers from either the identifier table or the set
  /// of mangled names) to global variables we have already
  /// emitted. Note that the entries in this map are the actual
  /// globals and therefore may not be of the same type as the decl,
  /// they should be bitcasted on retrieval. Also note that the
  /// globals are keyed on their source mangled name, not the global name
  /// (which may change with attributes such as asm-labels).  The key
  /// to this map should be generated using getMangledName().
  ///
  /// Note that this map always lines up exactly with the contents of the LLVM
  /// IR symbol table, but this is quicker to query since it is doing uniqued
  /// pointer lookups instead of full string lookups.
  llvm::DenseMap<const char*, llvm::GlobalValue*> GlobalDeclMap;

  /// \brief Contains the strings used for mangled names.
  ///
  /// FIXME: Eventually, this should map from the semantic/canonical
  /// declaration for each global entity to its mangled name (if it
  /// has one).
  llvm::StringSet<> MangledNames;

  /// DeferredDecls - This contains all the decls which have definitions but
  /// which are deferred for emission and therefore should only be output if
  /// they are actually used.  If a decl is in this, then it is known to have
  /// not been referenced yet.  The key to this map is a uniqued mangled name.
  llvm::DenseMap<const char*, GlobalDecl> DeferredDecls;

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

  /// CXXGlobalInits - Variables with global initializers that need to run
  /// before main.
  std::vector<const VarDecl*> CXXGlobalInits;
  
  /// CFConstantStringClassRef - Cached reference to the class for constant
  /// strings. This value has type int * but is actually an Obj-C class pointer.
  llvm::Constant *CFConstantStringClassRef;
  
  llvm::LLVMContext &VMContext;
public:
  CodeGenModule(ASTContext &C, const CompileOptions &CompileOpts,
                llvm::Module &M, const llvm::TargetData &TD, Diagnostic &Diags);

  ~CodeGenModule();

  /// Release - Finalize LLVM code generation.
  void Release();

  /// getObjCRuntime() - Return a reference to the configured
  /// Objective-C runtime.
  CGObjCRuntime &getObjCRuntime() {
    assert(Runtime && "No Objective-C runtime has been configured.");
    return *Runtime;
  }

  /// hasObjCRuntime() - Return true iff an Objective-C runtime has
  /// been configured.
  bool hasObjCRuntime() { return !!Runtime; }

  CGDebugInfo *getDebugInfo() { return DebugInfo; }
  ASTContext &getContext() const { return Context; }
  const CompileOptions &getCompileOpts() const { return CompileOpts; }
  const LangOptions &getLangOptions() const { return Features; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  Diagnostic &getDiags() const { return Diags; }
  const llvm::TargetData &getTargetData() const { return TheTargetData; }
  llvm::LLVMContext &getLLVMContext() { return VMContext; }

  /// getDeclVisibilityMode - Compute the visibility of the decl \arg D.
  LangOptions::VisibilityMode getDeclVisibilityMode(const Decl *D) const;

  /// setGlobalVisibility - Set the visibility for the given LLVM
  /// GlobalValue.
  void setGlobalVisibility(llvm::GlobalValue *GV, const Decl *D) const;

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

  /// GenerateRtti - Generate the rtti information for the given type.
  llvm::Constant *GenerateRtti(const CXXRecordDecl *RD);

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
  llvm::Function *GetAddrOfCXXConstructor(const CXXConstructorDecl *D, 
                                          CXXCtorType Type);

  /// GetAddrOfCXXDestructor - Return the address of the constructor of the
  /// given type.
  llvm::Function *GetAddrOfCXXDestructor(const CXXDestructorDecl *D, 
                                         CXXDtorType Type);
  
  /// getBuiltinLibFunction - Given a builtin id for a function like
  /// "__builtin_fabsf", return a Function* for "fabsf".
  llvm::Value *getBuiltinLibFunction(unsigned BuiltinID);

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

  /// CreateRuntimeFunction - Create a new runtime function with the specified
  /// type and name.
  llvm::Constant *CreateRuntimeFunction(const llvm::FunctionType *Ty,
                                        const char *Name);
  /// CreateRuntimeVariable - Create a new runtime global variable with the
  /// specified type and name.
  llvm::Constant *CreateRuntimeVariable(const llvm::Type *Ty,
                                        const char *Name);

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

  void ConstructAttributeList(const CGFunctionInfo &Info,
                              const Decl *TargetDecl,
                              AttributeListType &PAL);

  const char *getMangledName(const GlobalDecl &D);

  const char *getMangledName(const NamedDecl *ND);
  const char *getMangledCXXCtorName(const CXXConstructorDecl *D, 
                                    CXXCtorType Type);
  const char *getMangledCXXDtorName(const CXXDestructorDecl *D, 
                                    CXXDtorType Type);

  void EmitTentativeDefinition(const VarDecl *D);

  enum GVALinkage {
    GVA_Internal,
    GVA_C99Inline,
    GVA_CXXInline,
    GVA_StrongExternal,
    GVA_TemplateInstantiation
  };
  
private:
  /// UniqueMangledName - Unique a name by (if necessary) inserting it into the
  /// MangledNames string map.
  const char *UniqueMangledName(const char *NameStart, const char *NameEnd);
  
  llvm::Constant *GetOrCreateLLVMFunction(const char *MangledName,
                                          const llvm::Type *Ty,
                                          GlobalDecl D);
  llvm::Constant *GetOrCreateLLVMGlobal(const char *MangledName,
                                        const llvm::PointerType *PTy,
                                        const VarDecl *D);
  void DeferredCopyConstructorToEmit(GlobalDecl D);
  void DeferredCopyAssignmentToEmit(GlobalDecl D);
  void DeferredDestructorToEmit(GlobalDecl D);
  
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
  void SetFunctionAttributes(const FunctionDecl *FD,
                             llvm::Function *F,
                             bool IsIncompleteFunction);

  /// EmitGlobal - Emit code for a singal global function or var decl. Forward
  /// declarations are emitted lazily.
  void EmitGlobal(GlobalDecl D);

  void EmitGlobalDefinition(GlobalDecl D);

  void EmitGlobalFunctionDefinition(GlobalDecl GD);
  void EmitGlobalVarDefinition(const VarDecl *D);
  void EmitAliasDefinition(const ValueDecl *D);
  void EmitObjCPropertyImplementations(const ObjCImplementationDecl *D);

  // C++ related functions.
  
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
  
  /// EmitCXXGlobalInitFunc - Emit a function that initializes C++ globals.
  void EmitCXXGlobalInitFunc();
  
  // FIXME: Hardcoding priority here is gross.
  void AddGlobalCtor(llvm::Function *Ctor, int Priority=65535);
  void AddGlobalDtor(llvm::Function *Dtor, int Priority=65535);

  /// EmitCtorList - Generates a global array of functions and priorities using
  /// the given list and name. This array will have appending linkage and is
  /// suitable for use as a LLVM constructor or destructor array.
  void EmitCtorList(const CtorList &Fns, const char *GlobalName);

  void EmitAnnotations(void);

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
