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
  class StringLiteral;
  class NamedDecl;
  class ValueDecl;
  class VarDecl;
  struct LangOptions;
  class Diagnostic;
  class AnnotateAttr;
    
namespace CodeGen {

  class CodeGenFunction;
  class CGDebugInfo;
  class CGObjCRuntime;
  
/// CodeGenModule - This class organizes the cross-function state that
/// is used while generating LLVM code.
class CodeGenModule {
  typedef std::vector< std::pair<llvm::Constant*, int> > CtorList;

  ASTContext &Context;
  const LangOptions &Features;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  Diagnostic &Diags;
  CodeGenTypes Types;
  CGObjCRuntime* Runtime;
  CGDebugInfo* DebugInfo;

  llvm::Function *MemCpyFn;
  llvm::Function *MemMoveFn;
  llvm::Function *MemSetFn;

  /// GlobalDeclMap - Mapping of decl names global variables we have
  /// already emitted. Note that the entries in this map are the
  /// actual globals and therefore may not be of the same type as the
  /// decl, they should be bitcasted on retrieval. Also note that the
  /// globals are keyed on their source name, not the global name
  /// (which may change with attributes such as asm-labels).
  llvm::StringMap<llvm::GlobalValue*> GlobalDeclMap;

  /// List of static global for which code generation is delayed. When
  /// the translation unit has been fully processed we will lazily
  /// emit definitions for only the decls that were actually used.
  /// This should contain only Function and Var decls, and only those
  /// which actually define something.
  std::vector<const ValueDecl*> StaticDecls;
  
  /// GlobalCtors - Store the list of global constructors and their
  /// respective priorities to be emitted when the translation unit is
  /// complete.
  CtorList GlobalCtors;

  /// GlobalDtors - Store the list of global destructors and their
  /// respective priorities to be emitted when the translation unit is
  /// complete.
  CtorList GlobalDtors;

  std::vector<llvm::Constant*> Annotations;
    
  llvm::StringMap<llvm::Constant*> CFConstantStringMap;
  llvm::StringMap<llvm::Constant*> ConstantStringMap;

  /// CFConstantStringClassRef - Cached reference to the class for
  /// constant strings. This value has type int * but is actually an
  /// Obj-C class pointer.
  llvm::Constant *CFConstantStringClassRef;
  
  std::vector<llvm::Function *> BuiltinFunctions;
public:
  CodeGenModule(ASTContext &C, const LangOptions &Features, llvm::Module &M, 
                const llvm::TargetData &TD, Diagnostic &Diags,
                bool GenerateDebugInfo);

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

  /// GetStringForStringLiteral - Return the appropriate bytes for a
  /// string literal, properly padded to match the literal type. If
  /// only the address of a constant is needed consider using
  /// GetAddrOfConstantStringLiteral.
  std::string GetStringForStringLiteral(const StringLiteral *E);

  /// GetAddrOfConstantCFString - Return a pointer to a
  /// constant CFString object for the given string.
  llvm::Constant *GetAddrOfConstantCFString(const std::string& str);

  /// GetAddrOfConstantStringFromLiteral - Return a pointer to a
  /// constant array for the given string literal.
  llvm::Constant *GetAddrOfConstantStringFromLiteral(const StringLiteral *S);

  /// GetAddrOfConstantString - Returns a pointer to a character array
  /// containing the literal. This contents are exactly that of the
  /// given string, i.e. it will not be null terminated automatically;
  /// see GetAddrOfConstantCString. Note that whether the result is
  /// actually a pointer to an LLVM constant depends on
  /// Feature.WriteableStrings.
  ///
  /// The result has pointer to array type.
  llvm::Constant *GetAddrOfConstantString(const std::string& str);

  /// GetAddrOfConstantCString - Returns a pointer to a character
  /// array containing the literal and a terminating '\-'
  /// character. The result has pointer to array type.
  llvm::Constant *GetAddrOfConstantCString(const std::string &str);
  
  /// getBuiltinLibFunction - Given a builtin id for a function like
  /// "__builtin_fabsf", return a Function* for "fabsf".
  llvm::Function *getBuiltinLibFunction(unsigned BuiltinID);

  llvm::Function *getMemCpyFn();
  llvm::Function *getMemMoveFn();
  llvm::Function *getMemSetFn();
  llvm::Function *getIntrinsic(unsigned IID, const llvm::Type **Tys = 0, 
                               unsigned NumTys = 0);

  /// EmitTopLevelDecl - Emit code for a single top level declaration.
  void EmitTopLevelDecl(Decl *D);

  void AddAnnotation(llvm::Constant *C) { Annotations.push_back(C); }

  void UpdateCompletedType(const TagDecl *D);
  llvm::Constant *EmitConstantExpr(const Expr *E, CodeGenFunction *CGF = 0);
  llvm::Constant *EmitAnnotateAttr(llvm::GlobalValue *GV,
                                   const AnnotateAttr *AA, unsigned LineNo);
    
  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified stmt yet.    
  void ErrorUnsupported(const Stmt *S, const char *Type);
  
  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified decl yet.
  void ErrorUnsupported(const Decl *D, const char *Type);

private:
  void SetFunctionAttributes(const FunctionDecl *FD,
                             llvm::Function *F,
                             const llvm::FunctionType *FTy);

  void SetGlobalValueAttributes(const FunctionDecl *FD,
                                llvm::GlobalValue *GV);
  
  /// EmitGlobal - Emit code for a singal global function or var
  /// decl. Forward declarations are emitted lazily.
  void EmitGlobal(const ValueDecl *D);

  void EmitGlobalDefinition(const ValueDecl *D);
  llvm::GlobalValue *EmitForwardFunctionDefinition(const FunctionDecl *D);
  void EmitGlobalFunctionDefinition(const FunctionDecl *D);
  void EmitGlobalVarDefinition(const VarDecl *D);
  
  // FIXME: Hardcoding priority here is gross.
  void AddGlobalCtor(llvm::Function * Ctor, int Priority=65535);
  void AddGlobalDtor(llvm::Function * Dtor, int Priority=65535);

  /// EmitCtorList - Generates a global array of functions and
  /// priorities using the given list and name. This array will have
  /// appending linkage and is suitable for use as a LLVM constructor
  /// or destructor array.
  void EmitCtorList(const CtorList &Fns, const char *GlobalName);

  void EmitAnnotations(void);
  void EmitStatics(void);

};
}  // end namespace CodeGen
}  // end namespace clang

#endif
