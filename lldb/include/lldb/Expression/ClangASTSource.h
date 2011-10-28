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
#include "lldb/Symbol/ClangASTImporter.h"

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
    /// @param[in] declMap
    ///     A reference to the LLDB object that handles entity lookup.
    //------------------------------------------------------------------
	ClangASTSource () :
        m_ast_context (NULL),
        m_active_lookups (),
        m_import_in_progress (false),
        m_lookups_enabled (false)
    {
    }
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
	~ClangASTSource();
	
    //------------------------------------------------------------------
    /// Interface stubs.
    //------------------------------------------------------------------
    clang::Decl *GetExternalDecl (uint32_t)         {   return NULL;                }
    clang::Stmt *GetExternalDeclStmt (uint64_t)     {   return NULL;                }
	clang::Selector GetExternalSelector (uint32_t)  {   return clang::Selector();   }
    uint32_t GetNumExternalSelectors ()             {   return 0;                   }
    clang::CXXBaseSpecifier *GetExternalCXXBaseSpecifiers (uint64_t Offset)
                                                    {   return NULL;                }
    void MaterializeVisibleDecls (const clang::DeclContext *DC)
                                                    {   return;                     }
	
    void InstallASTContext (clang::ASTContext *ast_context)
    {
        m_ast_context = ast_context;
    }
    
    //
    // APIs for ExternalASTSource
    //

    //------------------------------------------------------------------
    /// Look up all Decls that match a particular name.  Only handles
    /// Identifiers and DeclContexts that are either NamespaceDecls or
    /// TranslationUnitDecls.  Calls SetExternalVisibleDeclsForName with
    /// the result.
    ///
    /// The work for this function is done by
    /// void FindExternalVisibleDecls (NameSearchContext &);
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
    clang::DeclContextLookupResult 
    FindExternalVisibleDeclsByName (const clang::DeclContext *DC,
                                    clang::DeclarationName Name);
    
    //------------------------------------------------------------------
    /// Enumerate all Decls in a given lexical context.
    ///
    /// @param[in] DC
    ///     The DeclContext being searched.
    ///
    /// @param[in] isKindWeWant
    ///     If non-NULL, a callback function that returns true given the
    ///     DeclKinds of desired Decls, and false otherwise.
    ///
    /// @param[in] Decls
    ///     A vector that is filled in with matching Decls.
    //------------------------------------------------------------------
    virtual clang::ExternalLoadResult 
    FindExternalLexicalDecls (const clang::DeclContext *DC,
                              bool (*isKindWeWant)(clang::Decl::Kind),
                              llvm::SmallVectorImpl<clang::Decl*> &Decls);
    
    //------------------------------------------------------------------
    /// Complete a TagDecl.
    ///
    /// @param[in] Tag
    ///     The Decl to be completed in place.
    //------------------------------------------------------------------
    virtual void
    CompleteType (clang::TagDecl *Tag);
    
    //------------------------------------------------------------------
    /// Complete an ObjCInterfaceDecl.
    ///
    /// @param[in] Class
    ///     The Decl to be completed in place.
    //------------------------------------------------------------------
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
    
    //
    // Helper APIs
    //
    
    //------------------------------------------------------------------
    /// The worker function for FindExternalVisibleDeclsByName.
    ///
    /// @param[in] context
    ///     The NameSearchContext to use when filing results.
    //------------------------------------------------------------------
    virtual void FindExternalVisibleDecls (NameSearchContext &context);
    
    void SetImportInProgress (bool import_in_progress) { m_import_in_progress = import_in_progress; }
    bool GetImportInProgress () { return m_import_in_progress; }
    
    void SetLookupsEnabled (bool lookups_enabled) { m_lookups_enabled = lookups_enabled; }
    bool GetLookupsEnabled () { return m_lookups_enabled; }
    
    //----------------------------------------------------------------------
    /// @class ClangASTSourceProxy ClangASTSource.h "lldb/Expression/ClangASTSource.h"
    /// @brief Proxy for ClangASTSource
    ///
    /// Clang AST contexts like to own their AST sources, so this is a
    /// state-free proxy object.
    //----------------------------------------------------------------------
    class ClangASTSourceProxy : public clang::ExternalASTSource
    {
    public:
        ClangASTSourceProxy (ClangASTSource &original) :
            m_original(original)
        {
        }
        
        clang::DeclContextLookupResult 
        FindExternalVisibleDeclsByName (const clang::DeclContext *DC,
                                        clang::DeclarationName Name)
        {
            return m_original.FindExternalVisibleDeclsByName(DC, Name);
        }
        
        virtual clang::ExternalLoadResult 
        FindExternalLexicalDecls (const clang::DeclContext *DC,
                                  bool (*isKindWeWant)(clang::Decl::Kind),
                                  llvm::SmallVectorImpl<clang::Decl*> &Decls)
        {
            return m_original.FindExternalLexicalDecls(DC, isKindWeWant, Decls);
        }
        
        virtual void
        CompleteType (clang::TagDecl *Tag)
        {
            return m_original.CompleteType(Tag);
        }
        
        virtual void 
        CompleteType (clang::ObjCInterfaceDecl *Class)
        {
            return m_original.CompleteType(Class);
        }

        void StartTranslationUnit (clang::ASTConsumer *Consumer)
        {
            return m_original.StartTranslationUnit(Consumer);
        }
    private:
        ClangASTSource &m_original;
    };
    
    clang::ExternalASTSource *CreateProxy()
    {
        return new ClangASTSourceProxy(*this);
    }
    
protected:
    friend struct NameSearchContext;
    
    bool                    m_import_in_progress;
    bool                    m_lookups_enabled;

	clang::ASTContext      *m_ast_context;     ///< The parser's AST context, for copying types into
    std::set<const char *>  m_active_lookups;
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
    ClangASTImporter::NamespaceMapSP m_namespace_map;   ///< The mapping of all namespaces found for this request back to their modules
    const clang::DeclarationName &m_decl_name;          ///< The name being looked for
    const clang::DeclContext *m_decl_context;           ///< The DeclContext to put declarations into
    
    struct {
        bool variable                   : 1;
        bool function_with_type_info    : 1;
        bool function                   : 1;
    } m_found;
    
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
    
    
    //------------------------------------------------------------------
    /// Add Decls from the provided DeclContextLookupResult to the list
    /// of results.
    ///
    /// @param[in] result
    ///     The DeclContextLookupResult, usually returned as the result
    ///     of querying a DeclContext.
    //------------------------------------------------------------------
    void AddLookupResult (clang::DeclContextLookupConstResult result);
    
    //------------------------------------------------------------------
    /// Add a NamedDecl to the list of results.
    ///
    /// @param[in] decl
    ///     The NamedDecl, usually returned as the result
    ///     of querying a DeclContext.
    //------------------------------------------------------------------
    void AddNamedDecl (clang::NamedDecl *decl);
};

}

#endif
