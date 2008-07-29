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

#include "CodeGenTypes.h"
#include "CGObjCRuntime.h"
#include "clang/AST/Attr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
  class Module;
  class Constant;
  class Function;
  class GlobalValue;
  class TargetData;
  class FunctionType;
}

namespace clang {
  class ASTContext;
  class FunctionDecl;
  class ObjCMethodDecl;
  class ObjCImplementationDecl;
  class ObjCCategoryImplDecl;
  class ObjCProtocolDecl;
  class Decl;
  class Expr;
  class Stmt;
  class NamedDecl;
  class ValueDecl;
  class VarDecl;
  struct LangOptions;
  class Diagnostic;
  class AnnotateAttr;
    
namespace CodeGen {

  class CodeGenFunction;
  class CGDebugInfo;
  
/// CodeGenModule - This class organizes the cross-module state that is used
/// while generating LLVM code.
class CodeGenModule {
  ASTContext &Context;
  const LangOptions &Features;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  Diagnostic &Diags;
  CodeGenTypes Types;
  CGObjCRuntime *Runtime;
  CGDebugInfo *DebugInfo;

  llvm::Function *MemCpyFn;
  llvm::Function *MemMoveFn;
  llvm::Function *MemSetFn;
  llvm::DenseMap<const Decl*, llvm::Constant*> GlobalDeclMap;

  /// List of static global for which code generation is delayed. When
  /// the translation unit has been fully processed we will lazily
  /// emit definitions for only the decls that were actually used.
  /// This should contain only Function and Var decls, and only those
  /// which actually define something.
  std::vector<const ValueDecl*> StaticDecls;
  
  std::vector<llvm::Constant*> GlobalCtors;
  std::vector<llvm::Constant*> Annotations;
    
  llvm::StringMap<llvm::Constant*> CFConstantStringMap;
  llvm::StringMap<llvm::Constant*> ConstantStringMap;
  llvm::Constant *CFConstantStringClassRef;
  
  std::vector<llvm::Function *> BuiltinFunctions;
public:
  CodeGenModule(ASTContext &C, const LangOptions &Features, llvm::Module &M, 
                const llvm::TargetData &TD, Diagnostic &Diags,
                bool GenerateDebugInfo);
  ~CodeGenModule();
  
  CGObjCRuntime *getObjCRuntime() { return Runtime; }
  CGDebugInfo *getDebugInfo() { return DebugInfo; }
  ASTContext &getContext() const { return Context; }
  const LangOptions &getLangOptions() const { return Features; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  Diagnostic &getDiags() const { return Diags; }
  const llvm::TargetData &getTargetData() const { return TheTargetData; }

  /// GetAddrOfGlobalVar - Return the llvm::Constant for the address
  /// of the given global variable.
  llvm::Constant *GetAddrOfGlobalVar(const VarDecl *D);

  /// GetAddrOfFunction - Return the llvm::Constant for the address
  /// of the given function.
  llvm::Constant *GetAddrOfFunction(const FunctionDecl *D);  
  
  /// getBuiltinLibFunction - Given a builtin id for a function like
  /// "__builtin_fabsf", return a Function* for "fabsf".
  ///
  llvm::Function *getBuiltinLibFunction(unsigned BuiltinID);
  llvm::Constant *GetAddrOfConstantCFString(const std::string& str);

  /// GetAddrOfConstantString -- returns a pointer to the character
  /// array containing the literal.  The result is pointer to array type.
  llvm::Constant *GetAddrOfConstantString(const std::string& str);
  llvm::Function *getMemCpyFn();
  llvm::Function *getMemMoveFn();
  llvm::Function *getMemSetFn();
  llvm::Function *getIntrinsic(unsigned IID, const llvm::Type **Tys = 0, 
                               unsigned NumTys = 0);
  
  void EmitObjCMethod(const ObjCMethodDecl *OMD);
  void EmitObjCCategoryImpl(const ObjCCategoryImplDecl *OCD);
  void EmitObjCClassImplementation(const ObjCImplementationDecl *OID);
  void EmitObjCProtocolImplementation(const ObjCProtocolDecl *PD);

  /// EmitGlobal - Emit code for a singal global function or var
  /// decl. Forward declarations are emitted lazily.
  void EmitGlobal(const ValueDecl *D);

  void AddAnnotation(llvm::Constant *C) { Annotations.push_back(C); }

  void UpdateCompletedType(const TagDecl *D);
  llvm::Constant *EmitConstantExpr(const Expr *E, CodeGenFunction *CGF = 0);
  llvm::Constant *EmitAnnotateAttr(llvm::GlobalValue *GV,
                                   const AnnotateAttr *AA, unsigned LineNo);
    
  /// WarnUnsupported - Print out a warning that codegen doesn't support the
  /// specified stmt yet.
    
  void WarnUnsupported(const Stmt *S, const char *Type);
  
  /// WarnUnsupported - Print out a warning that codegen doesn't support the
  /// specified decl yet.
  void WarnUnsupported(const Decl *D, const char *Type);
  
  /// setVisibility - Set the visibility for the given LLVM GlobalValue
  /// according to the given clang AST visibility value.
  static void setVisibility(llvm::GlobalValue *GV,
                            VisibilityAttr::VisibilityTypes);

private:
  /// ReplaceMapValuesWith - This is a really slow and bad function that
  /// searches for any entries in GlobalDeclMap that point to OldVal, changing
  /// them to point to NewVal.  This is badbadbad, FIXME!
  void ReplaceMapValuesWith(llvm::Constant *OldVal, llvm::Constant *NewVal);

  void SetFunctionAttributes(const FunctionDecl *FD,
                             llvm::Function *F,
                             const llvm::FunctionType *FTy);

  void SetGlobalValueAttributes(const FunctionDecl *FD,
                                llvm::GlobalValue *GV);
  
  void EmitGlobalDefinition(const ValueDecl *D);
  llvm::GlobalValue *EmitForwardFunctionDefinition(const FunctionDecl *D);
  void EmitGlobalFunctionDefinition(const FunctionDecl *D);
  void EmitGlobalVarDefinition(const VarDecl *D);

  void AddGlobalCtor(llvm::Function * Ctor);
  void EmitGlobalCtors(void);
  void EmitAnnotations(void);
  void EmitStatics(void);

};
}  // end namespace CodeGen
}  // end namespace clang

#endif
