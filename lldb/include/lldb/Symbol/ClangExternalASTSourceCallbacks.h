//===-- ClangExternalASTSourceCallbacks.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCallbacks_h_
#define liblldb_ClangExternalASTSourceCallbacks_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>
#include <memory>
#include <stdint.h>

// Other libraries and framework includes

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// releated features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#ifndef NDEBUG
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ExternalASTSource.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"

namespace lldb_private {

class ClangExternalASTSourceCallbacks : public clang::ExternalASTSource 
{
public:

    typedef void (*CompleteTagDeclCallback)(void *baton, clang::TagDecl *);
    typedef void (*CompleteObjCInterfaceDeclCallback)(void *baton, clang::ObjCInterfaceDecl *);
    typedef void (*FindExternalVisibleDeclsByNameCallback)(void *baton, const clang::DeclContext *DC, clang::DeclarationName Name, llvm::SmallVectorImpl <clang::NamedDecl *> *results);

    ClangExternalASTSourceCallbacks (CompleteTagDeclCallback tag_decl_callback,
                                     CompleteObjCInterfaceDeclCallback objc_decl_callback,
                                     FindExternalVisibleDeclsByNameCallback find_by_name_callback,
                                     void *callback_baton) :
        m_callback_tag_decl (tag_decl_callback),
        m_callback_objc_decl (objc_decl_callback),
        m_callback_find_by_name (find_by_name_callback),
        m_callback_baton (callback_baton)
    {
    }
    
    //------------------------------------------------------------------
    // clang::ExternalASTSource
    //------------------------------------------------------------------

    virtual clang::Decl *
    GetExternalDecl (uint32_t ID)
    {
        // This method only needs to be implemented if the AST source ever
        // passes back decl sets as VisibleDeclaration objects.
        return 0; 
    }
    
    virtual clang::Stmt *
    GetExternalDeclStmt (uint64_t Offset)
    {
        // This operation is meant to be used via a LazyOffsetPtr.  It only
        // needs to be implemented if the AST source uses methods like
        // FunctionDecl::setLazyBody when building decls.
        return 0; 
    }
	
    virtual clang::Selector 
    GetExternalSelector (uint32_t ID)
    {
        // This operation only needs to be implemented if the AST source
        // returns non-zero for GetNumKnownSelectors().
        return clang::Selector();
    }

	virtual uint32_t
    GetNumExternalSelectors()
    {
        return 0;
    }
    
    virtual clang::CXXBaseSpecifier *
    GetExternalCXXBaseSpecifiers(uint64_t Offset)
    {
        return NULL; 
    }
	
    virtual void 
    MaterializeVisibleDecls (const clang::DeclContext *decl_ctx)
    {
        return;
    }
	
	virtual bool 
    FindExternalLexicalDecls (const clang::DeclContext *decl_ctx,
                              bool (*isKindWeWant)(clang::Decl::Kind),
                              llvm::SmallVectorImpl<clang::Decl*> &decls)
    {
        // This is used to support iterating through an entire lexical context,
        // which isn't something the debugger should ever need to do.
        // true is for error, that's good enough for me
        return true;
    }
    
    virtual clang::DeclContextLookupResult 
    FindExternalVisibleDeclsByName (const clang::DeclContext *decl_ctx,
                                    clang::DeclarationName decl_name);
    
    virtual void
    CompleteType (clang::TagDecl *tag_decl);
    
    virtual void
    CompleteType (clang::ObjCInterfaceDecl *objc_decl);

    void
    SetExternalSourceCallbacks (CompleteTagDeclCallback tag_decl_callback,
                                CompleteObjCInterfaceDeclCallback objc_decl_callback,
                                FindExternalVisibleDeclsByNameCallback find_by_name_callback,
                                void *callback_baton)
    {
        m_callback_tag_decl = tag_decl_callback;
        m_callback_objc_decl = objc_decl_callback;
        m_callback_find_by_name = find_by_name_callback;
        m_callback_baton = callback_baton;    
    }

    void
    RemoveExternalSourceCallbacks (void *callback_baton)
    {
        if (callback_baton == m_callback_baton)
        {
            m_callback_tag_decl = NULL;
            m_callback_objc_decl = NULL;
            m_callback_find_by_name = NULL;
        }
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangExternalASTSourceCallbacks can see and modify these
    //------------------------------------------------------------------
    CompleteTagDeclCallback                 m_callback_tag_decl;
    CompleteObjCInterfaceDeclCallback       m_callback_objc_decl;
    FindExternalVisibleDeclsByNameCallback  m_callback_find_by_name;
    void *                                  m_callback_baton;
};

} // namespace lldb_private

#endif  // liblldb_ClangExternalASTSourceCallbacks_h_
