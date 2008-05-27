//===--- ModuleBuilder.cpp - Emit LLVM Code from ASTs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to LLVM Code.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleBuilder.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// LLVM Emitter

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

namespace {
  class CodeGenerator : public ASTConsumer {
    Diagnostic &Diags;
    const llvm::TargetData *TD;
    ASTContext *Ctx;
    const LangOptions &Features;
    bool GenerateDebugInfo;
  protected:
    llvm::Module *&M;
    CodeGen::CodeGenModule *Builder;
  public:
    CodeGenerator(Diagnostic &diags, const LangOptions &LO,
                  llvm::Module *&DestModule, bool DebugInfoFlag)
    : Diags(diags), Features(LO), GenerateDebugInfo(DebugInfoFlag),
    M(DestModule) {}
    
    ~CodeGenerator() {
      delete Builder;
      delete TD;
    }
    
    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;
      
      M->setTargetTriple(Ctx->Target.getTargetTriple());
      M->setDataLayout(Ctx->Target.getTargetDescription());
      TD = new llvm::TargetData(Ctx->Target.getTargetDescription());
      Builder = new CodeGen::CodeGenModule(Context, Features, *M, *TD, Diags,
                                           GenerateDebugInfo);
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      // If an error occurred, stop code generation, but continue parsing and
      // semantic analysis (to ensure all warnings and errors are emitted).
      if (Diags.hasErrorOccurred())
        return;
      
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        Builder->EmitFunction(FD);
      } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
        if (VD->isFileVarDecl())
          Builder->EmitGlobalVarDeclarator(VD);
      } else if (isa<ObjCClassDecl>(D) || isa<ObjCCategoryDecl>(D)) {
        // Forward declaration.  Only used for type checking.
      } else if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(D)){
        Builder->EmitObjCMethod(OMD);
      } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
        if (LSD->getLanguage() == LinkageSpecDecl::lang_cxx)
          Builder->WarnUnsupported(LSD, "linkage spec");
        // FIXME: implement C++ linkage, C linkage works mostly by C
        // language reuse already.
      } else if (FileScopeAsmDecl *AD = dyn_cast<FileScopeAsmDecl>(D)) {
        std::string AsmString(AD->getAsmString()->getStrData(),
                              AD->getAsmString()->getByteLength());
        
        const std::string &S = Builder->getModule().getModuleInlineAsm();
        if (S.empty())
          Builder->getModule().setModuleInlineAsm(AsmString);
        else
          Builder->getModule().setModuleInlineAsm(S + '\n' + AsmString);
      } else {
        assert(isa<TypeDecl>(D) && "Unknown top level decl");
        // TODO: handle debug info?
      }
    }
    
    /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
    /// (e.g. struct, union, enum, class) is completed.  This allows the client to
    /// hack on the type, which can occur at any point in the file (because these
    /// can be defined in declspecs).
    virtual void HandleTagDeclDefinition(TagDecl *D) {
      Builder->UpdateCompletedType(D);
    }
    
  };
}

ASTConsumer *clang::CreateLLVMCodeGen(Diagnostic &Diags, 
                                      const LangOptions &Features,
                                      llvm::Module *&DestModule,
                                      bool GenerateDebugInfo) {
  return new CodeGenerator(Diags, Features, DestModule, GenerateDebugInfo);
}

