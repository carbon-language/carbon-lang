//===-- ClangASTImporter.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"

using namespace lldb_private;
using namespace clang;

clang::QualType 
ClangASTImporter::CopyType (clang::ASTContext *src_ast,
                            clang::QualType type)
{
    MinionSP minion = GetMinion(src_ast, false);
    
    return minion->Import(type);
}

clang::Decl *
ClangASTImporter::CopyDecl (clang::ASTContext *src_ast,
                            clang::Decl *decl)
{
    MinionSP minion;
    
    if (isa<clang::NamespaceDecl>(decl)) 
        minion = GetMinion(src_ast, true);
    else
        minion = GetMinion(src_ast, false);
    
    return minion->Import(decl);
}

const clang::DeclContext *
ClangASTImporter::CompleteDeclContext (const clang::DeclContext *decl_context)
{
    const Decl *context_decl = dyn_cast<Decl>(decl_context);
    
    if (!context_decl)
        return NULL;
    
    DeclOrigin context_decl_origin = GetDeclOrigin(context_decl);
    
    if (!context_decl_origin.Valid())
        return NULL;
    
    if (!ClangASTContext::GetCompleteDecl(context_decl_origin.ctx, context_decl_origin.decl))
        return NULL;
    
    MinionSP minion = GetMinion(context_decl_origin.ctx, false);
    
    minion->ImportDefinition(context_decl_origin.decl);
    
    return decl_context;
}
