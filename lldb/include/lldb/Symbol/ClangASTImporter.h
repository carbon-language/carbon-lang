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
#if defined(__GNUC__) && !defined(__clang__)
// Gcc complains about ClangNamespaceDecl being an incomplete type
// without this.
#include "lldb/Symbol/ClangNamespaceDecl.h"
#endif

namespace lldb_private {

class ClangASTImporter 
{
public:
    ClangASTImporter () :
        m_file_manager(clang::FileSystemOptions())
    {
    }
    
    clang::QualType
    CopyType (clang::ASTContext *dst_ctx,
              clang::ASTContext *src_ctx,
              clang::QualType type);
    
    clang::Decl *
    CopyDecl (clang::ASTContext *dst_ctx,
              clang::ASTContext *src_ctx,
              clang::Decl *decl);
        
    void
    CompleteTagDecl (clang::TagDecl *decl);
    
    void
    CompleteObjCInterfaceDecl (clang::ObjCInterfaceDecl *interface_decl);
    
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
    
    typedef std::vector < std::pair<lldb::ModuleSP, ClangNamespaceDecl> > NamespaceMap;
    typedef lldb::SharedPtr<NamespaceMap>::Type NamespaceMapSP;
    
    void RegisterNamespaceMap (const clang::NamespaceDecl *decl, 
                               NamespaceMapSP &namespace_map);
    
    class NamespaceMapCompleter 
    {
    public:
        virtual ~NamespaceMapCompleter ();
        
        virtual void CompleteNamespaceMap (NamespaceMapSP &namespace_map,
                                           const ConstString &name,
                                           NamespaceMapSP &parent_map) const = 0;
    };
    
    void InstallMapCompleter (NamespaceMapCompleter &completer)
    {
        m_map_completer = &completer;
    }
                           
    NamespaceMapSP GetNamespaceMap (const clang::NamespaceDecl *decl);
    
    void BuildNamespaceMap (const clang::NamespaceDecl *decl);
    
    void PurgeMaps (clang::ASTContext *dest_ast_ctx);
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
                clang::ASTContext *target_ctx,
                clang::ASTContext *source_ctx) :
            clang::ASTImporter(*target_ctx,
                               master.m_file_manager,
                               *source_ctx,
                               master.m_file_manager,
                               true /*minimal*/),
            m_master(master),
            m_source_ctx(source_ctx)
        {
        }
        
        clang::Decl *Imported (clang::Decl *from, clang::Decl *to);
        
        ClangASTImporter   &m_master;
        clang::ASTContext  *m_source_ctx;
    };
    
    typedef lldb::SharedPtr<Minion>::Type               MinionSP;
    
    struct MinionSpec
    {
        clang::ASTContext *dst;
        clang::ASTContext *src;
        
        MinionSpec (clang::ASTContext *_dst,
                    clang::ASTContext *_src) :
            dst(_dst),
            src(_src)
        {
        }
        
        bool operator<(const MinionSpec &rhs) const
        {
            if (dst < rhs.dst)
                return true;
            if (dst == rhs.dst && src < rhs.src)
                return true;
            return false;
        }
    };
    
    typedef std::map<MinionSpec, MinionSP>     MinionMap;
    
    MinionSP
    GetMinion (clang::ASTContext *target_ctx, clang::ASTContext *source_ctx)
    {
        MinionSpec spec(target_ctx, source_ctx);
        
        if (m_minions.find(spec) == m_minions.end())
            m_minions[spec] = MinionSP(new Minion(*this, target_ctx, source_ctx));
        
        return m_minions[spec];
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
    
    typedef std::map <const clang::NamespaceDecl *, NamespaceMapSP> NamespaceMetaMap;
    
    NamespaceMetaMap        m_namespace_maps;
    NamespaceMapCompleter  *m_map_completer;
    clang::FileManager      m_file_manager;
    MinionMap               m_minions;
    OriginMap               m_origins;
};
    
}

#endif
