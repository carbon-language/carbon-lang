//===---- CodeCompleteConsumer.h - Code Completion Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeCompleteConsumer class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
#define LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace clang {
  
class Decl;
class DeclContext;
class NamedDecl;
class Scope;
class Sema;
  
/// \brief Abstract interface for a consumer of code-completion 
/// information.
class CodeCompleteConsumer {
  /// \brief The semantic-analysis object to which this code-completion
  /// consumer is attached.
  Sema &SemaRef;
  
public:
  /// \brief Captures a result of code completion.
  struct Result {
    /// \brief Describes the kind of result generated.
    enum ResultKind {
      RK_Declaration = 0, //< Refers to a declaration
      RK_Keyword          //< Refers to a keyword or symbol.
    };
    
    /// \brief The kind of result stored here.
    ResultKind Kind;
    
    union {
      /// \brief When Kind == RK_Declaration, the declaration we are referring
      /// to.
      NamedDecl *Declaration;
      
      /// \brief When Kind == RK_Keyword, the string representing the keyword 
      /// or symbol's spelling.
      const char *Keyword;
    };
    
    /// \brief Describes how good this result is, with zero being the best
    /// result and progressively higher numbers representing poorer results.
    unsigned Rank;
    
    /// \brief Whether this result is hidden by another name.
    bool Hidden : 1;
    
    /// \brief Build a result that refers to a declaration.
    Result(NamedDecl *Declaration, unsigned Rank)
      : Kind(RK_Declaration), Declaration(Declaration), Rank(Rank), 
        Hidden(false) { }
    
    /// \brief Build a result that refers to a keyword or symbol.
    Result(const char *Keyword, unsigned Rank)
      : Kind(RK_Keyword), Keyword(Keyword), Rank(Rank), Hidden(false) { }
  };
    
  /// \brief A container of code-completion results.
  class ResultSet {
  public:
    /// \brief The type of a name-lookup filter, which can be provided to the
    /// name-lookup routines to specify which declarations should be included in
    /// the result set (when it returns true) and which declarations should be
    /// filtered out (returns false).
    typedef bool (CodeCompleteConsumer::*LookupFilter)(NamedDecl *) const;
    
  private:
    /// \brief The actual results we have found.
    std::vector<Result> Results;

    /// \brief A record of all of the declarations we have found and placed
    /// into the result set, used to ensure that no declaration ever gets into
    /// the result set twice.
    llvm::SmallPtrSet<Decl*, 16> AllDeclsFound;
    
    /// \brief A mapping from declaration names to the declarations that have
    /// this name within a particular scope and their index within the list of
    /// results.
    typedef std::multimap<DeclarationName, 
                          std::pair<NamedDecl *, unsigned> > ShadowMap;
    
    /// \brief The code-completion consumer that is producing these results.
    CodeCompleteConsumer &Completer;
    
    /// \brief If non-NULL, a filter function used to remove any code-completion
    /// results that are not desirable.
    LookupFilter Filter;
    
    /// \brief A list of shadow maps, which is used to model name hiding at
    /// different levels of, e.g., the inheritance hierarchy.
    std::list<ShadowMap> ShadowMaps;
        
  public:
    explicit ResultSet(CodeCompleteConsumer &Completer,
                       LookupFilter Filter = 0)
      : Completer(Completer), Filter(Filter) { }
    
    /// \brief Set the filter used for code-completion results.
    void setFilter(LookupFilter Filter) {
      this->Filter = Filter;
    }
    
    typedef std::vector<Result>::iterator iterator;
    iterator begin() { return Results.begin(); }
    iterator end() { return Results.end(); }
    
    Result *data() { return Results.empty()? 0 : &Results.front(); }
    unsigned size() const { return Results.size(); }
    bool empty() const { return Results.empty(); }
        
    /// \brief Add a new result to this result set (if it isn't already in one
    /// of the shadow maps), or replace an existing result (for, e.g., a 
    /// redeclaration).
    void MaybeAddResult(Result R);
    
    /// \brief Enter into a new scope.
    void EnterNewScope();
    
    /// \brief Exit from the current scope.
    void ExitScope();
  };
  
  /// \brief Create a new code-completion consumer and registers it with
  /// the given semantic-analysis object.
  explicit CodeCompleteConsumer(Sema &S);
  
  /// \brief Deregisters and destroys this code-completion consumer.
  virtual ~CodeCompleteConsumer();
  
  /// \brief Retrieve the semantic-analysis object to which this code-completion
  /// consumer is attached.
  Sema &getSema() const { return SemaRef; }
  
  /// \name Code-completion callbacks
  //@{
  
  /// \brief Process the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Result *Results, 
                                          unsigned NumResults) { }
  
  /// \brief Code completion for a member access expression, e.g., "x->" or
  /// "x.".
  ///
  /// \param S is the scope in which code-completion occurs.
  ///
  /// \param BaseType is the type whose members are being accessed.
  ///
  /// \param IsArrow whether this member referenced was written with an
  /// arrow ("->") or a period (".").
  virtual void CodeCompleteMemberReferenceExpr(Scope *S, QualType BaseType,
                                               bool IsArrow);
  
  /// \brief Code completion for a tag name following an enum, class, struct,
  /// or union keyword.
  virtual void CodeCompleteTag(Scope *S, ElaboratedType::TagKind TK);
  
  /// \brief Code completion for a qualified-id, e.g., "std::"
  ///
  /// \param S the scope in which the nested-name-specifier occurs.
  ///
  /// \param NNS the nested-name-specifier before the code-completion location.
  ///
  /// \param EnteringContext whether the parser will be entering the scope of
  /// the qualified-id.
  virtual void CodeCompleteQualifiedId(Scope *S, NestedNameSpecifier *NNS,
                                       bool EnteringContext);
  //@}
  
  /// \name Name lookup functions
  ///
  /// The name lookup functions in this group collect code-completion results
  /// by performing a form of name looking into a scope or declaration context.
  //@{
  unsigned CollectLookupResults(Scope *S, unsigned InitialRank,
                                ResultSet &Results);
  unsigned CollectMemberLookupResults(DeclContext *Ctx, unsigned InitialRank, 
                                      ResultSet &Results);
  unsigned CollectMemberLookupResults(DeclContext *Ctx, unsigned InitialRank, 
                               llvm::SmallPtrSet<DeclContext *, 16> &Visited,
                                      ResultSet &Results);
  //@}
  
  /// \name Name lookup predicates
  ///
  /// These predicates can be passed to the name lookup functions to filter the
  /// results of name lookup. All of the predicates have the same type, so that
  /// 
  //@{
  bool IsNestedNameSpecifier(NamedDecl *ND) const;
  bool IsEnum(NamedDecl *ND) const;
  bool IsClassOrStruct(NamedDecl *ND) const;
  bool IsUnion(NamedDecl *ND) const;
  //@}
  
  /// \name Utility functions
  ///
  //@{
  
  bool canHiddenResultBeFound(NamedDecl *Hidden, NamedDecl *Visible);
  
  //@}
};
  
/// \brief A simple code-completion consumer that prints the results it 
/// receives in a simple format.
class PrintingCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The raw output stream.
  llvm::raw_ostream &OS;
  
public:
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  PrintingCodeCompleteConsumer(Sema &S, llvm::raw_ostream &OS)
    : CodeCompleteConsumer(S), OS(OS) { }
  
  /// \brief Prints the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Result *Results, 
                                          unsigned NumResults);
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
