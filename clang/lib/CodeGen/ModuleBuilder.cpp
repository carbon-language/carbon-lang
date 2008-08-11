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
                                               Diags, GenerateDebugInfo,
                                               false));
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      // If an error occurred, stop code generation, but continue parsing and
      // semantic analysis (to ensure all warnings and errors are emitted).
      if (Diags.hasErrorOccurred())
        return;

      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        Builder->EmitGlobal(FD);
      } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
        Builder->EmitGlobal(VD);
      } else if (isa<ObjCClassDecl>(D)){
        //Forward declaration.  Only used for type checking.
      } else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D)){
        // Generate Protocol object.
        Builder->EmitObjCProtocolImplementation(PD);
      } else if (isa<ObjCCategoryDecl>(D)){
        //Only used for typechecking.
      } else if (ObjCCategoryImplDecl *OCD = dyn_cast<ObjCCategoryImplDecl>(D)){
        // Generate methods, attach to category structure
        Builder->EmitObjCCategoryImpl(OCD);
      } else if (ObjCImplementationDecl * OID = 
          dyn_cast<ObjCImplementationDecl>(D)){
        // Generate methods, attach to class structure
        Builder->EmitObjCClassImplementation(OID);
      } else if (isa<ObjCInterfaceDecl>(D)){
        // Ignore - generated when the implementation decl is CodeGen'd
      } else if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(D)){
        Builder->EmitObjCMethod(OMD);
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

      if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        SD = SD->getNextDeclarator();
        if (SD)
          HandleTopLevelDecl(SD);
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
