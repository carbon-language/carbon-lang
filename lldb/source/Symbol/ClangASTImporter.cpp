//===-- ClangASTImporter.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "lldb/Core/Log.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"

using namespace lldb_private;
using namespace clang;

clang::QualType 
ClangASTImporter::CopyType (clang::ASTContext *src_ast,
                            clang::QualType type)
{
    MinionSP minion_sp (GetMinion(src_ast, false));
    
    if (minion_sp)
        return minion_sp->Import(type);
    
    return QualType();
}

clang::Decl *
ClangASTImporter::CopyDecl (clang::ASTContext *src_ast,
                            clang::Decl *decl)
{
    MinionSP minion_sp;
    
    if (isa<clang::NamespaceDecl>(decl)) 
        minion_sp = GetMinion(src_ast, true);
    else
        minion_sp = GetMinion(src_ast, false);
    
    if (minion_sp)
        return minion_sp->Import(decl);
    
    return NULL;
}

void
ClangASTImporter::CompleteTagDecl (clang::TagDecl *decl)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
        log->Printf("Completing a TagDecl named %s", decl->getName().str().c_str());
    
    DeclOrigin decl_origin = GetDeclOrigin(decl);
    
    if (!decl_origin.Valid())
        return;
    
    if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
        return;
    
    MinionSP minion_sp (GetMinion(decl_origin.ctx, false));
    
    if (minion_sp)
        minion_sp->ImportDefinition(decl_origin.decl);
    
    return;
}

void
ClangASTImporter::CompleteObjCInterfaceDecl (clang::ObjCInterfaceDecl *interface_decl)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("Completing an ObjCInterfaceDecl named %s", interface_decl->getName().str().c_str());
    
    DeclOrigin decl_origin = GetDeclOrigin(interface_decl);
    
    if (!decl_origin.Valid())
        return;
    
    if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
        return;
    
    MinionSP minion_sp (GetMinion(decl_origin.ctx, false));
    
    if (minion_sp)
        minion_sp->ImportDefinition(decl_origin.decl);
    
    return;
}

clang::Decl 
*ClangASTImporter::Minion::Imported (clang::Decl *from, clang::Decl *to)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    m_master.m_origins[to] = DeclOrigin (m_source_ctx, from);
 
    if (TagDecl *from_tag_decl = dyn_cast<TagDecl>(from))
    {
        TagDecl *to_tag_decl = dyn_cast<TagDecl>(to);
        
        to_tag_decl->setHasExternalLexicalStorage();
                
        if (log)
            log->Printf("Imported a TagDecl named %s%s%s",
                        from_tag_decl->getName().str().c_str(),
                        (to_tag_decl->hasExternalLexicalStorage() ? " Lexical" : ""),
                        (to_tag_decl->hasExternalVisibleStorage() ? " Visible" : ""));
    }
    
    if (isa<ObjCInterfaceDecl>(from))
    {
        ObjCInterfaceDecl *to_interface_decl = dyn_cast<ObjCInterfaceDecl>(to);
        
        if (!to_interface_decl->isForwardDecl())
            to_interface_decl->setExternallyCompleted();
    }
    
    return clang::ASTImporter::Imported(from, to);
}
