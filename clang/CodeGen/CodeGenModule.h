//===--- CodeGenModule.h - Per-Module state for LLVM CodeGen --------------===//
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

#ifndef CODEGEN_CODEGENMODULE_H
#define CODEGEN_CODEGENMODULE_H

#include "CodeGenTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
  class Module;
  class Constant;
  class Function;
  class GlobalVariable;
  class TargetData;
}

namespace clang {
  class ASTContext;
  class FunctionDecl;
  class Decl;
  class Expr;
  class Stmt;
  class ValueDecl;
  class VarDecl;
  class FileVarDecl;
  struct LangOptions;
  class Diagnostic;
    
namespace CodeGen {

/// CodeGenModule - This class organizes the cross-module state that is used
/// while generating LLVM code.
class CodeGenModule {
  ASTContext &Context;
  const LangOptions &Features;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  Diagnostic &Diags;
  CodeGenTypes Types;

  llvm::Function *MemCpyFn;
  llvm::DenseMap<const Decl*, llvm::Constant*> GlobalDeclMap;
    
  llvm::StringMap<llvm::Constant*> CFConstantStringMap;
  llvm::StringMap<llvm::Constant*> ConstantStringMap;
  llvm::Constant *CFConstantStringClassRef;
  
  std::vector<llvm::Function *> BuiltinFunctions;
public:
  CodeGenModule(ASTContext &C, const LangOptions &Features, llvm::Module &M, 
                const llvm::TargetData &TD, Diagnostic &Diags);
  
  ASTContext &getContext() const { return Context; }
  const LangOptions &getLangOptions() const { return Features; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  Diagnostic &getDiags() const { return Diags; }
  
  llvm::Constant *GetAddrOfFunctionDecl(const FunctionDecl *D,
                                        bool isDefinition);
  llvm::Constant *GetAddrOfGlobalVar(const VarDecl *D, bool isDefinition);
  
  
  /// getBuiltinLibFunction - Given a builtin id for a function like
  /// "__builtin_fabsf", return a Function* for "fabsf".
  ///
  llvm::Function *getBuiltinLibFunction(unsigned BuiltinID);
  llvm::Constant *GetAddrOfConstantCFString(const std::string& str);
  llvm::Constant *GetAddrOfConstantString(const std::string& str);
  llvm::Function *getMemCpyFn();
  llvm::Function *getIntrinsic(unsigned IID, const llvm::Type **Tys = 0, 
                               unsigned NumTys = 0);
  
  void EmitFunction(const FunctionDecl *FD);
  void EmitGlobalVar(const FileVarDecl *D);
  void EmitGlobalVarDeclarator(const FileVarDecl *D);
  llvm::Constant *EmitGlobalInit(const Expr *Expression);
  
  void PrintStats() {}
  
  /// WarnUnsupported - Print out a warning that codegen doesn't support the
  /// specified stmt yet.
  void WarnUnsupported(const Stmt *S, const char *Type);
  
private:
  /// ReplaceMapValuesWith - This is a really slow and bad function that
  /// searches for any entries in GlobalDeclMap that point to OldVal, changing
  /// them to point to NewVal.  This is badbadbad, FIXME!
  void ReplaceMapValuesWith(llvm::Constant *OldVal, llvm::Constant *NewVal);
  
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
