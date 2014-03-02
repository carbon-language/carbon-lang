//===--- ASTMatchFinder.cpp - Structural query framework ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements an algorithm to efficiently search for matches on AST nodes.
//  Uses memoization to support recursive matches like HasDescendant.
//
//  The general idea is to visit all AST nodes with a RecursiveASTVisitor,
//  calling the Matches(...) method of each matcher we are running on each
//  AST node. The matcher can recurse via the ASTMatchFinder interface.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <deque>
#include <set>

namespace clang {
namespace ast_matchers {
namespace internal {
namespace {

typedef MatchFinder::MatchCallback MatchCallback;

// The maximum number of memoization entries to store.
// 10k has been experimentally found to give a good trade-off
// of performance vs. memory consumption by running matcher
// that match on every statement over a very large codebase.
//
// FIXME: Do some performance optimization in general and
// revisit this number; also, put up micro-benchmarks that we can
// optimize this on.
static const unsigned MaxMemoizationEntries = 10000;

// We use memoization to avoid running the same matcher on the same
// AST node twice.  This struct is the key for looking up match
// result.  It consists of an ID of the MatcherInterface (for
// identifying the matcher), a pointer to the AST node and the
// bound nodes before the matcher was executed.
//
// We currently only memoize on nodes whose pointers identify the
// nodes (\c Stmt and \c Decl, but not \c QualType or \c TypeLoc).
// For \c QualType and \c TypeLoc it is possible to implement
// generation of keys for each type.
// FIXME: Benchmark whether memoization of non-pointer typed nodes
// provides enough benefit for the additional amount of code.
struct MatchKey {
  uint64_t MatcherID;
  ast_type_traits::DynTypedNode Node;
  BoundNodesTreeBuilder BoundNodes;

  bool operator<(const MatchKey &Other) const {
    if (MatcherID != Other.MatcherID)
      return MatcherID < Other.MatcherID;
    if (Node != Other.Node)
      return Node < Other.Node;
    return BoundNodes < Other.BoundNodes;
  }
};

// Used to store the result of a match and possibly bound nodes.
struct MemoizedMatchResult {
  bool ResultOfMatch;
  BoundNodesTreeBuilder Nodes;
};

// A RecursiveASTVisitor that traverses all children or all descendants of
// a node.
class MatchChildASTVisitor
    : public RecursiveASTVisitor<MatchChildASTVisitor> {
public:
  typedef RecursiveASTVisitor<MatchChildASTVisitor> VisitorBase;

  // Creates an AST visitor that matches 'matcher' on all children or
  // descendants of a traversed node. max_depth is the maximum depth
  // to traverse: use 1 for matching the children and INT_MAX for
  // matching the descendants.
  MatchChildASTVisitor(const DynTypedMatcher *Matcher,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder,
                       int MaxDepth,
                       ASTMatchFinder::TraversalKind Traversal,
                       ASTMatchFinder::BindKind Bind)
      : Matcher(Matcher),
        Finder(Finder),
        Builder(Builder),
        CurrentDepth(0),
        MaxDepth(MaxDepth),
        Traversal(Traversal),
        Bind(Bind),
        Matches(false) {}

  // Returns true if a match is found in the subtree rooted at the
  // given AST node. This is done via a set of mutually recursive
  // functions. Here's how the recursion is done (the  *wildcard can
  // actually be Decl, Stmt, or Type):
  //
  //   - Traverse(node) calls BaseTraverse(node) when it needs
  //     to visit the descendants of node.
  //   - BaseTraverse(node) then calls (via VisitorBase::Traverse*(node))
  //     Traverse*(c) for each child c of 'node'.
  //   - Traverse*(c) in turn calls Traverse(c), completing the
  //     recursion.
  bool findMatch(const ast_type_traits::DynTypedNode &DynNode) {
    reset();
    if (const Decl *D = DynNode.get<Decl>())
      traverse(*D);
    else if (const Stmt *S = DynNode.get<Stmt>())
      traverse(*S);
    else if (const NestedNameSpecifier *NNS =
             DynNode.get<NestedNameSpecifier>())
      traverse(*NNS);
    else if (const NestedNameSpecifierLoc *NNSLoc =
             DynNode.get<NestedNameSpecifierLoc>())
      traverse(*NNSLoc);
    else if (const QualType *Q = DynNode.get<QualType>())
      traverse(*Q);
    else if (const TypeLoc *T = DynNode.get<TypeLoc>())
      traverse(*T);
    // FIXME: Add other base types after adding tests.

    // It's OK to always overwrite the bound nodes, as if there was
    // no match in this recursive branch, the result set is empty
    // anyway.
    *Builder = ResultBindings;

    return Matches;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.
  bool TraverseDecl(Decl *DeclNode) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return (DeclNode == NULL) || traverse(*DeclNode);
  }
  bool TraverseStmt(Stmt *StmtNode) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    const Stmt *StmtToTraverse = StmtNode;
    if (Traversal ==
        ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses) {
      const Expr *ExprNode = dyn_cast_or_null<Expr>(StmtNode);
      if (ExprNode != NULL) {
        StmtToTraverse = ExprNode->IgnoreParenImpCasts();
      }
    }
    return (StmtToTraverse == NULL) || traverse(*StmtToTraverse);
  }
  // We assume that the QualType and the contained type are on the same
  // hierarchy level. Thus, we try to match either of them.
  bool TraverseType(QualType TypeNode) {
    if (TypeNode.isNull())
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    // Match the Type.
    if (!match(*TypeNode))
      return false;
    // The QualType is matched inside traverse.
    return traverse(TypeNode);
  }
  // We assume that the TypeLoc, contained QualType and contained Type all are
  // on the same hierarchy level. Thus, we try to match all of them.
  bool TraverseTypeLoc(TypeLoc TypeLocNode) {
    if (TypeLocNode.isNull())
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    // Match the Type.
    if (!match(*TypeLocNode.getType()))
      return false;
    // Match the QualType.
    if (!match(TypeLocNode.getType()))
      return false;
    // The TypeLoc is matched inside traverse.
    return traverse(TypeLocNode);
  }
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return (NNS == NULL) || traverse(*NNS);
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    if (!NNS)
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    if (!match(*NNS.getNestedNameSpecifier()))
      return false;
    return traverse(NNS);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  // Disables data recursion. We intercept Traverse* methods in the RAV, which
  // are not triggered during data recursion.
  bool shouldUseDataRecursionFor(clang::Stmt *S) const { return false; }

private:
  // Used for updating the depth during traversal.
  struct ScopedIncrement {
    explicit ScopedIncrement(int *Depth) : Depth(Depth) { ++(*Depth); }
    ~ScopedIncrement() { --(*Depth); }

   private:
    int *Depth;
  };

  // Resets the state of this object.
  void reset() {
    Matches = false;
    CurrentDepth = 0;
  }

  // Forwards the call to the corresponding Traverse*() method in the
  // base visitor class.
  bool baseTraverse(const Decl &DeclNode) {
    return VisitorBase::TraverseDecl(const_cast<Decl*>(&DeclNode));
  }
  bool baseTraverse(const Stmt &StmtNode) {
    return VisitorBase::TraverseStmt(const_cast<Stmt*>(&StmtNode));
  }
  bool baseTraverse(QualType TypeNode) {
    return VisitorBase::TraverseType(TypeNode);
  }
  bool baseTraverse(TypeLoc TypeLocNode) {
    return VisitorBase::TraverseTypeLoc(TypeLocNode);
  }
  bool baseTraverse(const NestedNameSpecifier &NNS) {
    return VisitorBase::TraverseNestedNameSpecifier(
        const_cast<NestedNameSpecifier*>(&NNS));
  }
  bool baseTraverse(NestedNameSpecifierLoc NNS) {
    return VisitorBase::TraverseNestedNameSpecifierLoc(NNS);
  }

  // Sets 'Matched' to true if 'Matcher' matches 'Node' and:
  //   0 < CurrentDepth <= MaxDepth.
  //
  // Returns 'true' if traversal should continue after this function
  // returns, i.e. if no match is found or 'Bind' is 'BK_All'.
  template <typename T>
  bool match(const T &Node) {
    if (CurrentDepth == 0 || CurrentDepth > MaxDepth) {
      return true;
    }
    if (Bind != ASTMatchFinder::BK_All) {
      BoundNodesTreeBuilder RecursiveBuilder(*Builder);
      if (Matcher->matches(ast_type_traits::DynTypedNode::create(Node), Finder,
                           &RecursiveBuilder)) {
        Matches = true;
        ResultBindings.addMatch(RecursiveBuilder);
        return false; // Abort as soon as a match is found.
      }
    } else {
      BoundNodesTreeBuilder RecursiveBuilder(*Builder);
      if (Matcher->matches(ast_type_traits::DynTypedNode::create(Node), Finder,
                           &RecursiveBuilder)) {
        // After the first match the matcher succeeds.
        Matches = true;
        ResultBindings.addMatch(RecursiveBuilder);
      }
    }
    return true;
  }

  // Traverses the subtree rooted at 'Node'; returns true if the
  // traversal should continue after this function returns.
  template <typename T>
  bool traverse(const T &Node) {
    static_assert(IsBaseType<T>::value,
                  "traverse can only be instantiated with base type");
    if (!match(Node))
      return false;
    return baseTraverse(Node);
  }

  const DynTypedMatcher *const Matcher;
  ASTMatchFinder *const Finder;
  BoundNodesTreeBuilder *const Builder;
  BoundNodesTreeBuilder ResultBindings;
  int CurrentDepth;
  const int MaxDepth;
  const ASTMatchFinder::TraversalKind Traversal;
  const ASTMatchFinder::BindKind Bind;
  bool Matches;
};

// Controls the outermost traversal of the AST and allows to match multiple
// matchers.
class MatchASTVisitor : public RecursiveASTVisitor<MatchASTVisitor>,
                        public ASTMatchFinder {
public:
  MatchASTVisitor(
      std::vector<std::pair<internal::DynTypedMatcher, MatchCallback *> > *
          MatcherCallbackPairs)
      : MatcherCallbackPairs(MatcherCallbackPairs), ActiveASTContext(NULL) {}

  void onStartOfTranslationUnit() {
    for (std::vector<std::pair<internal::DynTypedMatcher,
                               MatchCallback *> >::const_iterator
             I = MatcherCallbackPairs->begin(),
             E = MatcherCallbackPairs->end();
         I != E; ++I) {
      I->second->onStartOfTranslationUnit();
    }
  }

  void onEndOfTranslationUnit() {
    for (std::vector<std::pair<internal::DynTypedMatcher,
                               MatchCallback *> >::const_iterator
             I = MatcherCallbackPairs->begin(),
             E = MatcherCallbackPairs->end();
         I != E; ++I) {
      I->second->onEndOfTranslationUnit();
    }
  }

  void set_active_ast_context(ASTContext *NewActiveASTContext) {
    ActiveASTContext = NewActiveASTContext;
  }

  // The following Visit*() and Traverse*() functions "override"
  // methods in RecursiveASTVisitor.

  bool VisitTypedefNameDecl(TypedefNameDecl *DeclNode) {
    // When we see 'typedef A B', we add name 'B' to the set of names
    // A's canonical type maps to.  This is necessary for implementing
    // isDerivedFrom(x) properly, where x can be the name of the base
    // class or any of its aliases.
    //
    // In general, the is-alias-of (as defined by typedefs) relation
    // is tree-shaped, as you can typedef a type more than once.  For
    // example,
    //
    //   typedef A B;
    //   typedef A C;
    //   typedef C D;
    //   typedef C E;
    //
    // gives you
    //
    //   A
    //   |- B
    //   `- C
    //      |- D
    //      `- E
    //
    // It is wrong to assume that the relation is a chain.  A correct
    // implementation of isDerivedFrom() needs to recognize that B and
    // E are aliases, even though neither is a typedef of the other.
    // Therefore, we cannot simply walk through one typedef chain to
    // find out whether the type name matches.
    const Type *TypeNode = DeclNode->getUnderlyingType().getTypePtr();
    const Type *CanonicalType =  // root of the typedef tree
        ActiveASTContext->getCanonicalType(TypeNode);
    TypeAliases[CanonicalType].insert(DeclNode);
    return true;
  }

  bool TraverseDecl(Decl *DeclNode);
  bool TraverseStmt(Stmt *StmtNode);
  bool TraverseType(QualType TypeNode);
  bool TraverseTypeLoc(TypeLoc TypeNode);
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool memoizedMatchesRecursively(const ast_type_traits::DynTypedNode &Node,
                                  const DynTypedMatcher &Matcher,
                                  BoundNodesTreeBuilder *Builder, int MaxDepth,
                                  TraversalKind Traversal, BindKind Bind) {
    // For AST-nodes that don't have an identity, we can't memoize.
    if (!Node.getMemoizationData())
      return matchesRecursively(Node, Matcher, Builder, MaxDepth, Traversal,
                                Bind);

    MatchKey Key;
    Key.MatcherID = Matcher.getID();
    Key.Node = Node;
    // Note that we key on the bindings *before* the match.
    Key.BoundNodes = *Builder;

    MemoizationMap::iterator I = ResultCache.find(Key);
    if (I != ResultCache.end()) {
      *Builder = I->second.Nodes;
      return I->second.ResultOfMatch;
    }

    MemoizedMatchResult Result;
    Result.Nodes = *Builder;
    Result.ResultOfMatch = matchesRecursively(Node, Matcher, &Result.Nodes,
                                              MaxDepth, Traversal, Bind);
    ResultCache[Key] = Result;
    *Builder = Result.Nodes;
    return Result.ResultOfMatch;
  }

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool matchesRecursively(const ast_type_traits::DynTypedNode &Node,
                          const DynTypedMatcher &Matcher,
                          BoundNodesTreeBuilder *Builder, int MaxDepth,
                          TraversalKind Traversal, BindKind Bind) {
    MatchChildASTVisitor Visitor(
      &Matcher, this, Builder, MaxDepth, Traversal, Bind);
    return Visitor.findMatch(Node);
  }

  virtual bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                  const Matcher<NamedDecl> &Base,
                                  BoundNodesTreeBuilder *Builder);

  // Implements ASTMatchFinder::matchesChildOf.
  virtual bool matchesChildOf(const ast_type_traits::DynTypedNode &Node,
                              const DynTypedMatcher &Matcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traversal,
                              BindKind Bind) {
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesRecursively(Node, Matcher, Builder, 1, Traversal,
                                      Bind);
  }
  // Implements ASTMatchFinder::matchesDescendantOf.
  virtual bool matchesDescendantOf(const ast_type_traits::DynTypedNode &Node,
                                   const DynTypedMatcher &Matcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) {
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesRecursively(Node, Matcher, Builder, INT_MAX,
                                      TK_AsIs, Bind);
  }
  // Implements ASTMatchFinder::matchesAncestorOf.
  virtual bool matchesAncestorOf(const ast_type_traits::DynTypedNode &Node,
                                 const DynTypedMatcher &Matcher,
                                 BoundNodesTreeBuilder *Builder,
                                 AncestorMatchMode MatchMode) {
    // Reset the cache outside of the recursive call to make sure we
    // don't invalidate any iterators.
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesAncestorOfRecursively(Node, Matcher, Builder,
                                                MatchMode);
  }

  // Matches all registered matchers on the given node and calls the
  // result callback for every node that matches.
  void match(const ast_type_traits::DynTypedNode& Node) {
    for (std::vector<std::pair<internal::DynTypedMatcher,
                               MatchCallback *> >::const_iterator
             I = MatcherCallbackPairs->begin(),
             E = MatcherCallbackPairs->end();
         I != E; ++I) {
      BoundNodesTreeBuilder Builder;
      if (I->first.matches(Node, this, &Builder)) {
        MatchVisitor Visitor(ActiveASTContext, I->second);
        Builder.visitMatches(&Visitor);
      }
    }
  }

  template <typename T> void match(const T &Node) {
    match(ast_type_traits::DynTypedNode::create(Node));
  }

  // Implements ASTMatchFinder::getASTContext.
  virtual ASTContext &getASTContext() const { return *ActiveASTContext; }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  // Disables data recursion. We intercept Traverse* methods in the RAV, which
  // are not triggered during data recursion.
  bool shouldUseDataRecursionFor(clang::Stmt *S) const { return false; }

private:
  // Returns whether an ancestor of \p Node matches \p Matcher.
  //
  // The order of matching ((which can lead to different nodes being bound in
  // case there are multiple matches) is breadth first search.
  //
  // To allow memoization in the very common case of having deeply nested
  // expressions inside a template function, we first walk up the AST, memoizing
  // the result of the match along the way, as long as there is only a single
  // parent.
  //
  // Once there are multiple parents, the breadth first search order does not
  // allow simple memoization on the ancestors. Thus, we only memoize as long
  // as there is a single parent.
  bool memoizedMatchesAncestorOfRecursively(
      const ast_type_traits::DynTypedNode &Node, const DynTypedMatcher &Matcher,
      BoundNodesTreeBuilder *Builder, AncestorMatchMode MatchMode) {
    if (Node.get<TranslationUnitDecl>() ==
        ActiveASTContext->getTranslationUnitDecl())
      return false;
    assert(Node.getMemoizationData() &&
           "Invariant broken: only nodes that support memoization may be "
           "used in the parent map.");
    ASTContext::ParentVector Parents = ActiveASTContext->getParents(Node);
    if (Parents.empty()) {
      assert(false && "Found node that is not in the parent map.");
      return false;
    }
    MatchKey Key;
    Key.MatcherID = Matcher.getID();
    Key.Node = Node;
    Key.BoundNodes = *Builder;

    // Note that we cannot use insert and reuse the iterator, as recursive
    // calls to match might invalidate the result cache iterators.
    MemoizationMap::iterator I = ResultCache.find(Key);
    if (I != ResultCache.end()) {
      *Builder = I->second.Nodes;
      return I->second.ResultOfMatch;
    }
    MemoizedMatchResult Result;
    Result.ResultOfMatch = false;
    Result.Nodes = *Builder;
    if (Parents.size() == 1) {
      // Only one parent - do recursive memoization.
      const ast_type_traits::DynTypedNode Parent = Parents[0];
      if (Matcher.matches(Parent, this, &Result.Nodes)) {
        Result.ResultOfMatch = true;
      } else if (MatchMode != ASTMatchFinder::AMM_ParentOnly) {
        // Reset the results to not include the bound nodes from the failed
        // match above.
        Result.Nodes = *Builder;
        Result.ResultOfMatch = memoizedMatchesAncestorOfRecursively(
            Parent, Matcher, &Result.Nodes, MatchMode);
        // Once we get back from the recursive call, the result will be the
        // same as the parent's result.
      }
    } else {
      // Multiple parents - BFS over the rest of the nodes.
      llvm::DenseSet<const void *> Visited;
      std::deque<ast_type_traits::DynTypedNode> Queue(Parents.begin(),
                                                      Parents.end());
      while (!Queue.empty()) {
        Result.Nodes = *Builder;
        if (Matcher.matches(Queue.front(), this, &Result.Nodes)) {
          Result.ResultOfMatch = true;
          break;
        }
        if (MatchMode != ASTMatchFinder::AMM_ParentOnly) {
          ASTContext::ParentVector Ancestors =
              ActiveASTContext->getParents(Queue.front());
          for (ASTContext::ParentVector::const_iterator I = Ancestors.begin(),
                                                        E = Ancestors.end();
               I != E; ++I) {
            // Make sure we do not visit the same node twice.
            // Otherwise, we'll visit the common ancestors as often as there
            // are splits on the way down.
            if (Visited.insert(I->getMemoizationData()).second)
              Queue.push_back(*I);
          }
        }
        Queue.pop_front();
      }
    }
    ResultCache[Key] = Result;

    *Builder = Result.Nodes;
    return Result.ResultOfMatch;
  }

  // Implements a BoundNodesTree::Visitor that calls a MatchCallback with
  // the aggregated bound nodes for each match.
  class MatchVisitor : public BoundNodesTreeBuilder::Visitor {
  public:
    MatchVisitor(ASTContext* Context,
                 MatchFinder::MatchCallback* Callback)
      : Context(Context),
        Callback(Callback) {}

    virtual void visitMatch(const BoundNodes& BoundNodesView) {
      Callback->run(MatchFinder::MatchResult(BoundNodesView, Context));
    }

  private:
    ASTContext* Context;
    MatchFinder::MatchCallback* Callback;
  };

  // Returns true if 'TypeNode' has an alias that matches the given matcher.
  bool typeHasMatchingAlias(const Type *TypeNode,
                            const Matcher<NamedDecl> Matcher,
                            BoundNodesTreeBuilder *Builder) {
    const Type *const CanonicalType =
      ActiveASTContext->getCanonicalType(TypeNode);
    const std::set<const TypedefNameDecl *> &Aliases =
        TypeAliases[CanonicalType];
    for (std::set<const TypedefNameDecl*>::const_iterator
           It = Aliases.begin(), End = Aliases.end();
         It != End; ++It) {
      BoundNodesTreeBuilder Result(*Builder);
      if (Matcher.matches(**It, this, &Result)) {
        *Builder = Result;
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<internal::DynTypedMatcher, MatchCallback *> > *const
  MatcherCallbackPairs;
  ASTContext *ActiveASTContext;

  // Maps a canonical type to its TypedefDecls.
  llvm::DenseMap<const Type*, std::set<const TypedefNameDecl*> > TypeAliases;

  // Maps (matcher, node) -> the match result for memoization.
  typedef std::map<MatchKey, MemoizedMatchResult> MemoizationMap;
  MemoizationMap ResultCache;
};

static CXXRecordDecl *getAsCXXRecordDecl(const Type *TypeNode) {
  // Type::getAs<...>() drills through typedefs.
  if (TypeNode->getAs<DependentNameType>() != NULL ||
      TypeNode->getAs<DependentTemplateSpecializationType>() != NULL ||
      TypeNode->getAs<TemplateTypeParmType>() != NULL)
    // Dependent names and template TypeNode parameters will be matched when
    // the template is instantiated.
    return NULL;
  TemplateSpecializationType const *TemplateType =
      TypeNode->getAs<TemplateSpecializationType>();
  if (TemplateType == NULL) {
    return TypeNode->getAsCXXRecordDecl();
  }
  if (TemplateType->getTemplateName().isDependent())
    // Dependent template specializations will be matched when the
    // template is instantiated.
    return NULL;

  // For template specialization types which are specializing a template
  // declaration which is an explicit or partial specialization of another
  // template declaration, getAsCXXRecordDecl() returns the corresponding
  // ClassTemplateSpecializationDecl.
  //
  // For template specialization types which are specializing a template
  // declaration which is neither an explicit nor partial specialization of
  // another template declaration, getAsCXXRecordDecl() returns NULL and
  // we get the CXXRecordDecl of the templated declaration.
  CXXRecordDecl *SpecializationDecl = TemplateType->getAsCXXRecordDecl();
  if (SpecializationDecl != NULL) {
    return SpecializationDecl;
  }
  NamedDecl *Templated =
      TemplateType->getTemplateName().getAsTemplateDecl()->getTemplatedDecl();
  if (CXXRecordDecl *TemplatedRecord = dyn_cast<CXXRecordDecl>(Templated)) {
    return TemplatedRecord;
  }
  // Now it can still be that we have an alias template.
  TypeAliasDecl *AliasDecl = dyn_cast<TypeAliasDecl>(Templated);
  assert(AliasDecl);
  return getAsCXXRecordDecl(AliasDecl->getUnderlyingType().getTypePtr());
}

// Returns true if the given class is directly or indirectly derived
// from a base type with the given name.  A class is not considered to be
// derived from itself.
bool MatchASTVisitor::classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                         const Matcher<NamedDecl> &Base,
                                         BoundNodesTreeBuilder *Builder) {
  if (!Declaration->hasDefinition())
    return false;
  typedef CXXRecordDecl::base_class_const_iterator BaseIterator;
  for (BaseIterator It = Declaration->bases_begin(),
                    End = Declaration->bases_end();
       It != End; ++It) {
    const Type *TypeNode = It->getType().getTypePtr();

    if (typeHasMatchingAlias(TypeNode, Base, Builder))
      return true;

    CXXRecordDecl *ClassDecl = getAsCXXRecordDecl(TypeNode);
    if (ClassDecl == NULL)
      continue;
    if (ClassDecl == Declaration) {
      // This can happen for recursive template definitions; if the
      // current declaration did not match, we can safely return false.
      return false;
    }
    BoundNodesTreeBuilder Result(*Builder);
    if (Base.matches(*ClassDecl, this, &Result)) {
      *Builder = Result;
      return true;
    }
    if (classIsDerivedFrom(ClassDecl, Base, Builder))
      return true;
  }
  return false;
}

bool MatchASTVisitor::TraverseDecl(Decl *DeclNode) {
  if (DeclNode == NULL) {
    return true;
  }
  match(*DeclNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseDecl(DeclNode);
}

bool MatchASTVisitor::TraverseStmt(Stmt *StmtNode) {
  if (StmtNode == NULL) {
    return true;
  }
  match(*StmtNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseStmt(StmtNode);
}

bool MatchASTVisitor::TraverseType(QualType TypeNode) {
  match(TypeNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseType(TypeNode);
}

bool MatchASTVisitor::TraverseTypeLoc(TypeLoc TypeLocNode) {
  // The RecursiveASTVisitor only visits types if they're not within TypeLocs.
  // We still want to find those types via matchers, so we match them here. Note
  // that the TypeLocs are structurally a shadow-hierarchy to the expressed
  // type, so we visit all involved parts of a compound type when matching on
  // each TypeLoc.
  match(TypeLocNode);
  match(TypeLocNode.getType());
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseTypeLoc(TypeLocNode);
}

bool MatchASTVisitor::TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) {
  match(*NNS);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseNestedNameSpecifier(NNS);
}

bool MatchASTVisitor::TraverseNestedNameSpecifierLoc(
    NestedNameSpecifierLoc NNS) {
  match(NNS);
  // We only match the nested name specifier here (as opposed to traversing it)
  // because the traversal is already done in the parallel "Loc"-hierarchy.
  match(*NNS.getNestedNameSpecifier());
  return
      RecursiveASTVisitor<MatchASTVisitor>::TraverseNestedNameSpecifierLoc(NNS);
}

class MatchASTConsumer : public ASTConsumer {
public:
  MatchASTConsumer(MatchFinder *Finder,
                   MatchFinder::ParsingDoneTestCallback *ParsingDone)
      : Finder(Finder), ParsingDone(ParsingDone) {}

private:
  virtual void HandleTranslationUnit(ASTContext &Context) {
    if (ParsingDone != NULL) {
      ParsingDone->run();
    }
    Finder->matchAST(Context);
  }

  MatchFinder *Finder;
  MatchFinder::ParsingDoneTestCallback *ParsingDone;
};

} // end namespace
} // end namespace internal

MatchFinder::MatchResult::MatchResult(const BoundNodes &Nodes,
                                      ASTContext *Context)
  : Nodes(Nodes), Context(Context),
    SourceManager(&Context->getSourceManager()) {}

MatchFinder::MatchCallback::~MatchCallback() {}
MatchFinder::ParsingDoneTestCallback::~ParsingDoneTestCallback() {}

MatchFinder::MatchFinder() : ParsingDone(NULL) {}

MatchFinder::~MatchFinder() {}

void MatchFinder::addMatcher(const DeclarationMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

void MatchFinder::addMatcher(const TypeMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

void MatchFinder::addMatcher(const StatementMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

void MatchFinder::addMatcher(const NestedNameSpecifierMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

void MatchFinder::addMatcher(const NestedNameSpecifierLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

void MatchFinder::addMatcher(const TypeLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(NodeMatch, Action));
}

bool MatchFinder::addDynamicMatcher(const internal::DynTypedMatcher &NodeMatch,
                                    MatchCallback *Action) {
  if (NodeMatch.canConvertTo<Decl>()) {
    addMatcher(NodeMatch.convertTo<Decl>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<QualType>()) {
    addMatcher(NodeMatch.convertTo<QualType>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<Stmt>()) {
    addMatcher(NodeMatch.convertTo<Stmt>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<NestedNameSpecifier>()) {
    addMatcher(NodeMatch.convertTo<NestedNameSpecifier>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<NestedNameSpecifierLoc>()) {
    addMatcher(NodeMatch.convertTo<NestedNameSpecifierLoc>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<TypeLoc>()) {
    addMatcher(NodeMatch.convertTo<TypeLoc>(), Action);
    return true;
  }
  return false;
}

ASTConsumer *MatchFinder::newASTConsumer() {
  return new internal::MatchASTConsumer(this, ParsingDone);
}

void MatchFinder::match(const clang::ast_type_traits::DynTypedNode &Node,
                        ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&MatcherCallbackPairs);
  Visitor.set_active_ast_context(&Context);
  Visitor.match(Node);
}

void MatchFinder::matchAST(ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&MatcherCallbackPairs);
  Visitor.set_active_ast_context(&Context);
  Visitor.onStartOfTranslationUnit();
  Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  Visitor.onEndOfTranslationUnit();
}

void MatchFinder::registerTestCallbackAfterParsing(
    MatchFinder::ParsingDoneTestCallback *NewParsingDone) {
  ParsingDone = NewParsingDone;
}

} // end namespace ast_matchers
} // end namespace clang
