//===-- ClangResultSynthesizer.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangResultSynthesizer_h_
#define liblldb_ClangResultSynthesizer_h_

#include "clang/Sema/SemaConsumer.h"

namespace clang {
    class Action;
}

namespace lldb_private {

class ClangResultSynthesizer : public clang::SemaConsumer
{
public:
    ClangResultSynthesizer(clang::ASTConsumer *passthrough);
    ~ClangResultSynthesizer();
    
    void Initialize(clang::ASTContext &Context);
    void HandleTopLevelDecl(clang::DeclGroupRef D);
    void HandleTranslationUnit(clang::ASTContext &Ctx);
    void HandleTagDeclDefinition(clang::TagDecl *D);
    void CompleteTentativeDefinition(clang::VarDecl *D);
    void HandleVTable(clang::CXXRecordDecl *RD, bool DefinitionRequired);
    void PrintStats();
    
    void InitializeSema(clang::Sema &S);
    void ForgetSema();
private:
    void TransformTopLevelDecl(clang::Decl *D);
    bool SynthesizeResult(clang::ASTContext &Ctx,
                          clang::FunctionDecl *FunDecl);
    
    clang::ASTContext *m_ast_context;
    clang::ASTConsumer *m_passthrough;
    clang::SemaConsumer *m_passthrough_sema;
    clang::Sema *m_sema;
    clang::Action *m_action;
};

}

#endif