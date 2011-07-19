//===-- ClangASTImporter.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangASTImporter_h_
#define liblldb_ClangASTImporter_h_

#include <map>

#include "lldb/lldb-types.h"

#include "clang/AST/ASTImporter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"

namespace lldb_private {

class ClangASTImporter 
{
public:
    ClangASTImporter (clang::ASTContext *target_ctx) :
        m_file_manager(clang::FileSystemOptions()),
        m_target_ctx(target_ctx)
    {
    }
    
    clang::ASTContext *
    TargetASTContext ()
    {
        return m_target_ctx;
    }
    
    clang::QualType
    CopyType (clang::ASTContext *src_ctx,
              clang::QualType type);
    
    clang::Decl *
    CopyDecl (clang::ASTContext *src_ctx,
              clang::Decl *decl);
    
    const clang::DeclContext *
    CompleteDeclContext (const clang::DeclContext *decl_context);
    
    bool
    ResolveDeclOrigin (const clang::Decl *decl, clang::Decl **original_decl, clang::ASTContext **original_ctx)
    {
        DeclOrigin origin = GetDeclOrigin(decl);
        
        if (original_decl)
            *original_decl = origin.decl;
        
        if (original_ctx)
            *original_ctx = origin.ctx;
        
        return origin.Valid();
    }
    
private:
    
    struct DeclOrigin 
    {
        DeclOrigin () :
            ctx(NULL),
            decl(NULL)
        {
        }
        
        DeclOrigin (clang::ASTContext *_ctx,
                    clang::Decl *_decl) :
            ctx(_ctx),
            decl(_decl)
        {
        }
        
        DeclOrigin (const DeclOrigin &rhs)
        {
            ctx = rhs.ctx;
            decl = rhs.decl;
        }
        
        bool 
        Valid ()
        {
            return (ctx != NULL || decl != NULL);
        }
        
        clang::ASTContext  *ctx;
        clang::Decl        *decl;
    };
    
    typedef std::map<const clang::Decl *, DeclOrigin>   OriginMap;
    
    class Minion : public clang::ASTImporter
    {
    public:
        Minion (ClangASTImporter &master,
                clang::ASTContext *source_ctx,
                bool minimal) :
            clang::ASTImporter(*master.m_target_ctx,
                               master.m_file_manager,
                               *source_ctx,
                               master.m_file_manager,
                               minimal),
            m_master(master),
            m_source_ctx(source_ctx)
        {
        }
        
        clang::Decl *Imported (clang::Decl *from, clang::Decl *to)
        {
            m_master.m_origins[to] = DeclOrigin (m_source_ctx, from);
            
            return clang::ASTImporter::Imported(from, to);
        }
        
        ClangASTImporter   &m_master;
        clang::ASTContext  *m_source_ctx;
    };
    
    typedef lldb::SharedPtr<Minion>::Type               MinionSP;
    typedef std::map<clang::ASTContext *, MinionSP>     MinionMap;
    
    MinionSP
    GetMinion (clang::ASTContext *source_ctx, bool minimal)
    {
        MinionMap *minions;
        
        if (minimal)
            minions = &m_minimal_minions;
        else
            minions = &m_minions;
        
        if (minions->find(source_ctx) == minions->end())
            (*minions)[source_ctx] = MinionSP(new Minion(*this, source_ctx, minimal));
        
        return (*minions)[source_ctx];
    }
    
    DeclOrigin
    GetDeclOrigin (const clang::Decl *decl)
    {
        OriginMap::iterator iter = m_origins.find(decl);
        
        if (iter != m_origins.end())
            return iter->second;
        else
            return DeclOrigin();
    }
    
    clang::FileManager m_file_manager;
    clang::ASTContext  *m_target_ctx;
    MinionMap           m_minions;
    MinionMap           m_minimal_minions;
    OriginMap           m_origins;
};
    
}

#endif
