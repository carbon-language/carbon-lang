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
#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
using namespace clang;

namespace {
  class CodeGeneratorImpl : public CodeGenerator {
    DiagnosticsEngine &Diags;
    std::unique_ptr<const llvm::DataLayout> TD;
    ASTContext *Ctx;
    const CodeGenOptions CodeGenOpts;  // Intentionally copied in.
  protected:
    std::unique_ptr<llvm::Module> M;
    std::unique_ptr<CodeGen::CodeGenModule> Builder;

  public:
    CodeGeneratorImpl(DiagnosticsEngine &diags, const std::string& ModuleName,
                      const CodeGenOptions &CGO, llvm::LLVMContext& C)
      : Diags(diags), CodeGenOpts(CGO),
        M(new llvm::Module(ModuleName, C)) {}

    virtual ~CodeGeneratorImpl() {}

    llvm::Module* GetModule() override {
      return M.get();
    }

    const Decl *GetDeclForMangledName(StringRef MangledName) override {
      GlobalDecl Result;
      if (!Builder->lookupRepresentativeDecl(MangledName, Result))
        return nullptr;
      const Decl *D = Result.getCanonicalDecl().getDecl();
      if (auto FD = dyn_cast<FunctionDecl>(D)) {
        if (FD->hasBody(FD))
          return FD;
      } else if (auto TD = dyn_cast<TagDecl>(D)) {
        if (auto Def = TD->getDefinition())
          return Def;
      }
      return D;
    }

    llvm::Module *ReleaseModule() override { return M.release(); }

    void Initialize(ASTContext &Context) override {
      Ctx = &Context;

      M->setTargetTriple(Ctx->getTargetInfo().getTriple().getTriple());
      M->setDataLayout(Ctx->getTargetInfo().getTargetDescription());
      TD.reset(new llvm::DataLayout(Ctx->getTargetInfo().getTargetDescription()));
      Builder.reset(new CodeGen::CodeGenModule(Context, CodeGenOpts, *M, *TD,
                                               Diags));

      for (size_t i = 0, e = CodeGenOpts.DependentLibraries.size(); i < e; ++i)
        HandleDependentLibrary(CodeGenOpts.DependentLibraries[i]);
    }

    void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override {
      if (Diags.hasErrorOccurred())
        return;

      Builder->HandleCXXStaticMemberVarInstantiation(VD);
    }

    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      if (Diags.hasErrorOccurred())
        return true;

      // Make sure to emit all elements of a Decl.
      for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
        Builder->EmitTopLevelDecl(*I);

      // Emit any deferred inline method definitions.
      for (CXXMethodDecl *MD : DeferredInlineMethodDefinitions)
        Builder->EmitTopLevelDecl(MD);
      DeferredInlineMethodDefinitions.clear();

      return true;
    }

    void HandleInlineMethodDefinition(CXXMethodDecl *D) override {
      if (Diags.hasErrorOccurred())
        return;

      assert(D->doesThisDeclarationHaveABody());

      // We may want to emit this definition. However, that decision might be
      // based on computing the linkage, and we have to defer that in case we
      // are inside of something that will change the method's final linkage,
      // e.g.
      //   typedef struct {
      //     void bar();
      //     void foo() { bar(); }
      //   } A;
      DeferredInlineMethodDefinitions.push_back(D);
    }

    /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
    /// to (e.g. struct, union, enum, class) is completed. This allows the
    /// client hack on the type, which can occur at any point in the file
    /// (because these can be defined in declspecs).
    void HandleTagDeclDefinition(TagDecl *D) override {
      if (Diags.hasErrorOccurred())
        return;

      Builder->UpdateCompletedType(D);
    }

    void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
      if (Diags.hasErrorOccurred())
        return;

      if (CodeGen::CGDebugInfo *DI = Builder->getModuleDebugInfo())
        if (const RecordDecl *RD = dyn_cast<RecordDecl>(D))
          DI->completeRequiredType(RD);
    }

    void HandleTranslationUnit(ASTContext &Ctx) override {
      if (Diags.hasErrorOccurred()) {
        if (Builder)
          Builder->clear();
        M.reset();
        return;
      }

      if (Builder)
        Builder->Release();
    }

    void CompleteTentativeDefinition(VarDecl *D) override {
      if (Diags.hasErrorOccurred())
        return;

      Builder->EmitTentativeDefinition(D);
    }

    void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) override {
      if (Diags.hasErrorOccurred())
        return;

      Builder->EmitVTable(RD, DefinitionRequired);
    }

    void HandleLinkerOptionPragma(llvm::StringRef Opts) override {
      Builder->AppendLinkerOptions(Opts);
    }

    void HandleDetectMismatch(llvm::StringRef Name,
                              llvm::StringRef Value) override {
      Builder->AddDetectMismatch(Name, Value);
    }

    void HandleDependentLibrary(llvm::StringRef Lib) override {
      Builder->AddDependentLib(Lib);
    }

  private:
    std::vector<CXXMethodDecl *> DeferredInlineMethodDefinitions;
  };
}

void CodeGenerator::anchor() { }

CodeGenerator *clang::CreateLLVMCodeGen(DiagnosticsEngine &Diags,
                                        const std::string& ModuleName,
                                        const CodeGenOptions &CGO,
                                        const TargetOptions &/*TO*/,
                                        llvm::LLVMContext& C) {
  return new CodeGeneratorImpl(Diags, ModuleName, CGO, C);
}
