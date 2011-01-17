//===-- ClangASTSource.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangASTSource_h_
#define liblldb_ClangASTSource_h_

#include <set>

#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/ExternalASTSource.h"

namespace lldb_private {
    
class ClangExpressionDeclMap;

//----------------------------------------------------------------------
/// @class ClangASTSource ClangASTSource.h "lldb/Expression/ClangASTSource.h"
/// @brief Provider for named objects defined in the debug info for Clang
///
/// As Clang parses an expression, it may encounter names that are not
/// defined inside the expression, including variables, functions, and
/// types.  Clang knows the name it is looking for, but nothing else.
/// The ExternalSemaSource class provides Decls (VarDecl, FunDecl, TypeDecl)
/// to Clang for these names, consulting the ClangExpressionDeclMap to do
/// the actual lookups.
//----------------------------------------------------------------------
class ClangASTSource : public clang::ExternalASTSource 
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// Initializes class variabes.
    ///
    /// @param[in] context
    ///     A reference to the AST context provided by the parser.
    ///
    /// @param[in] declMap
    ///     A reference to the LLDB object that handles entity lookup.
    //------------------------------------------------------------------
	ClangASTSource (clang::ASTContext &context,
                    ClangExpressionDeclMap &decl_map) : 
        m_ast_context (context),
        m_decl_map (decl_map),
        m_active_lookups ()
    {
    }
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
	~ClangASTSource();
	
    //------------------------------------------------------------------
    /// Interface stub that returns NULL.
    //------------------------------------------------------------------
    virtual clang::Decl *
    GetExternalDecl(uint32_t)
    {
        // These are only required for AST source that want to lazily load
        // the declarations (or parts thereof) that they return.
        return NULL;
    }
    
    //------------------------------------------------------------------
    /// Interface stub that returns NULL.
    //------------------------------------------------------------------
    virtual clang::Stmt *
    GetExternalDeclStmt(uint64_t)
    {
        // These are only required for AST source that want to lazily load
        // the declarations (or parts thereof) that they return.
        return NULL;
    }
	
    //------------------------------------------------------------------
    /// Interface stub that returns an undifferentiated Selector.
    //------------------------------------------------------------------
    virtual clang::Selector 
    GetExternalSelector(uint32_t)
    {
        // These are also optional, although it might help with ObjC
        // debugging if we have respectable signatures.  But a more
        // efficient interface (that didn't require scanning all files
        // for method signatures!) might help.
        return clang::Selector();
    }
    
    //------------------------------------------------------------------
    /// Interface stub that returns 0.
    //------------------------------------------------------------------
	virtual uint32_t
    GetNumExternalSelectors()
    {
        // These are also optional, although it might help with ObjC
        // debugging if we have respectable signatures.  But a more
        // efficient interface (that didn't require scanning all files
        // for method signatures!) might help.
        return 0;
    }
    
    //------------------------------------------------------------------
    /// Interface stub that returns NULL.
    //------------------------------------------------------------------
    virtual clang::CXXBaseSpecifier *
    GetExternalCXXBaseSpecifiers(uint64_t Offset)
    {
        return NULL;
    }
	
    //------------------------------------------------------------------
    /// Look up all Decls that match a particular name.  Only handles
    /// Identifiers.  Passes the request on to DeclMap, and calls
    /// SetExternalVisibleDeclsForName with the result. 
    ///
    /// @param[in] DC
    ///     The DeclContext to register the found Decls in.
    ///
    /// @param[in] Name
    ///     The name to find entries for.
    ///
    /// @return
    ///     Whatever SetExternalVisibleDeclsForName returns.
    //------------------------------------------------------------------
    virtual clang::DeclContextLookupResult 
    FindExternalVisibleDeclsByName (const clang::DeclContext *DC,
                                    clang::DeclarationName Name);
    
    //------------------------------------------------------------------
    /// Interface stub.
    //------------------------------------------------------------------
    virtual void 
    MaterializeVisibleDecls (const clang::DeclContext *DC);
	
    //------------------------------------------------------------------
    /// Interface stub that returns true.
    //------------------------------------------------------------------
	virtual bool 
    FindExternalLexicalDecls (const clang::DeclContext *DC,
                              bool (*isKindWeWant)(clang::Decl::Kind),
                              llvm::SmallVectorImpl<clang::Decl*> &Decls);
    
    
    virtual void
    CompleteType (clang::TagDecl *Tag);
    
    virtual void 
    CompleteType (clang::ObjCInterfaceDecl *Class);

    //------------------------------------------------------------------
    /// Called on entering a translation unit.  Tells Clang by calling
    /// setHasExternalVisibleStorage() and setHasExternalLexicalStorage()
    /// that this object has something to say about undefined names.
    ///
    /// @param[in] ASTConsumer
    ///     Unused.
    //------------------------------------------------------------------
    void StartTranslationUnit (clang::ASTConsumer *Consumer);

protected:
    friend struct NameSearchContext;

	clang::ASTContext &m_ast_context;   ///< The parser's AST context, for copying types into
	ClangExpressionDeclMap &m_decl_map; ///< The object that looks up named entities in LLDB
    std::set<const char *> m_active_lookups;
};

//----------------------------------------------------------------------
/// @class NameSearchContext ClangASTSource.h "lldb/Expression/ClangASTSource.h"
/// @brief Container for all objects relevant to a single name lookup
///     
/// LLDB needs to create Decls for entities it finds.  This class communicates
/// what name is being searched for and provides helper functions to construct
/// Decls given appropriate type information.
//----------------------------------------------------------------------
struct NameSearchContext {
    ClangASTSource &m_ast_source;                       ///< The AST source making the request
    llvm::SmallVectorImpl<clang::NamedDecl*> &m_decls;  ///< The list of declarations already constructed
    const clang::DeclarationName &m_decl_name;          ///< The name being looked for
    const clang::DeclContext *m_decl_context;           ///< The DeclContext to put declarations into
    
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// Initializes class variables.
    ///
    /// @param[in] astSource
    ///     A reference to the AST source making a request.
    ///
    /// @param[in] decls
    ///     A reference to a list into which new Decls will be placed.  This
    ///     list is typically empty when the function is called.
    ///
    /// @param[in] name
    ///     The name being searched for (always an Identifier).
    ///
    /// @param[in] dc
    ///     The DeclContext to register Decls in.
    //------------------------------------------------------------------
    NameSearchContext (ClangASTSource &astSource,
                       llvm::SmallVectorImpl<clang::NamedDecl*> &decls,
                       clang::DeclarationName &name,
                       const clang::DeclContext *dc) :
        m_ast_source(astSource),
        m_decls(decls),
        m_decl_name(name),
        m_decl_context(dc) {}
    
    //------------------------------------------------------------------
    /// Return the AST context for the current search.  Useful when copying
    /// types.
    //------------------------------------------------------------------
    clang::ASTContext *GetASTContext();
    
    //------------------------------------------------------------------
    /// Create a VarDecl with the name being searched for and the provided
    /// type and register it in the right places.
    ///
    /// @param[in] type
    ///     The opaque QualType for the VarDecl being registered.
    //------------------------------------------------------------------
    clang::NamedDecl *AddVarDecl(void *type);
    
    //------------------------------------------------------------------
    /// Create a FunDecl with the name being searched for and the provided
    /// type and register it in the right places.
    ///
    /// @param[in] type
    ///     The opaque QualType for the FunDecl being registered.
    //------------------------------------------------------------------
    clang::NamedDecl *AddFunDecl(void *type);
    
    //------------------------------------------------------------------
    /// Create a FunDecl with the name being searched for and generic
    /// type (i.e. intptr_t NAME_GOES_HERE(...)) and register it in the
    /// right places.
    //------------------------------------------------------------------
    clang::NamedDecl *AddGenericFunDecl();
    
    //------------------------------------------------------------------
    /// Create a TypeDecl with the name being searched for and the provided
    /// type and register it in the right places.
    ///
    /// @param[in] type
    ///     The opaque QualType for the TypeDecl being registered.
    //------------------------------------------------------------------
    clang::NamedDecl *AddTypeDecl(void *type);
};

}

#endif