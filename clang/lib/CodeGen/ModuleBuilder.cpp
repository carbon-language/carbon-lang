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
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// LLVM Emitter

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/OwningPtr.h"


namespace {
  class VISIBILITY_HIDDEN CodeGeneratorImpl : public CodeGenerator {
    Diagnostic &Diags;
    llvm::OwningPtr<const llvm::TargetData> TD;
    ASTContext *Ctx;
    const LangOptions &Features;
    bool GenerateDebugInfo;
  protected:
    llvm::OwningPtr<llvm::Module> M;
    llvm::OwningPtr<CodeGen::CodeGenModule> Builder;
  public:
    CodeGeneratorImpl(Diagnostic &diags, const LangOptions &LO,
                      const std::string& ModuleName,
                      bool DebugInfoFlag)
    : Diags(diags), Features(LO), GenerateDebugInfo(DebugInfoFlag),
      M(new llvm::Module(ModuleName)) {}
    
    virtual ~CodeGeneratorImpl() {}
    
    virtual llvm::Module* ReleaseModule() {
      return M.take();
    }
    
    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;
      
      M->setTargetTriple(Ctx->Target.getTargetTriple());
      M->setDataLayout(Ctx->Target.getTargetDescription());
      TD.reset(new llvm::TargetData(Ctx->Target.getTargetDescription()));
      Builder.reset(new CodeGen::CodeGenModule(Context, Features, *M, *TD,
                                               Diags, GenerateDebugInfo));
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      // Make sure to emit all elements of a ScopedDecl.
      if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        for (; SD; SD = SD->getNextDeclarator())
          Builder->EmitTopLevelDecl(SD);
      } else {
        Builder->EmitTopLevelDecl(D);
      }
    }

    /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
    /// (e.g. struct, union, enum, class) is completed. This allows the client to
    /// hack on the type, which can occur at any point in the file (because these
    /// can be defined in declspecs).
    virtual void HandleTagDeclDefinition(TagDecl *D) {
      Builder->UpdateCompletedType(D);
    }

    virtual void HandleTranslationUnit(TranslationUnit& TU) {
      if (Diags.hasErrorOccurred()) {
        M.reset();
        return;
      }

      if (Builder)
        Builder->Release();
    };
  };
}

CodeGenerator *clang::CreateLLVMCodeGen(Diagnostic &Diags, 
                                        const LangOptions &Features,
                                        const std::string& ModuleName,
                                        bool GenerateDebugInfo) {
  return new CodeGeneratorImpl(Diags, Features, ModuleName, GenerateDebugInfo);
}
