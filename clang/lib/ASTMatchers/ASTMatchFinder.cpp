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
#include <set>

namespace clang {
namespace ast_matchers {
namespace internal {
namespace {

// We use memoization to avoid running the same matcher on the same
// AST node twice.  This pair is the key for looking up match
// result.  It consists of an ID of the MatcherInterface (for
// identifying the matcher) and a pointer to the AST node.
typedef std::pair<uint64_t, const void*> UntypedMatchInput;

// Used to store the result of a match and possibly bound nodes.
struct MemoizedMatchResult {
  bool ResultOfMatch;
  BoundNodesTree Nodes;
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
  MatchChildASTVisitor(const UntypedBaseMatcher *BaseMatcher,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder,
                       int MaxDepth,
                       ASTMatchFinder::TraversalKind Traversal,
                       ASTMatchFinder::BindKind Bind)
      : BaseMatcher(BaseMatcher),
        Finder(Finder),
        Builder(Builder),
        CurrentDepth(-1),
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
  template <typename T>
  bool findMatch(const T &Node) {
    reset();
    traverse(Node);
    return Matches;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.
  bool TraverseDecl(Decl *DeclNode) {
    return (DeclNode == NULL) || traverse(*DeclNode);
  }
  bool TraverseStmt(Stmt *StmtNode) {
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
  bool TraverseType(QualType TypeNode) {
    return traverse(TypeNode);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

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
    CurrentDepth = -1;
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

  // Traverses the subtree rooted at 'node'; returns true if the
  // traversal should continue after this function returns; also sets
  // matched_ to true if a match is found during the traversal.
  template <typename T>
  bool traverse(const T &Node) {
    TOOLING_COMPILE_ASSERT(IsBaseType<T>::value,
                           traverse_can_only_be_instantiated_with_base_type);
    ScopedIncrement ScopedDepth(&CurrentDepth);
    if (CurrentDepth == 0) {
      // We don't want to match the root node, so just recurse.
      return baseTraverse(Node);
    }
    if (Bind != ASTMatchFinder::BK_All) {
      if (BaseMatcher->matches(Node, Finder, Builder)) {
        Matches = true;
        return false;  // Abort as soon as a match is found.
      }
      if (CurrentDepth < MaxDepth) {
        // The current node doesn't match, and we haven't reached the
        // maximum depth yet, so recurse.
        return baseTraverse(Node);
      }
      // The current node doesn't match, and we have reached the
      // maximum depth, so don't recurse (but continue the traversal
      // such that other nodes at the current level can be visited).
      return true;
    } else {
      BoundNodesTreeBuilder RecursiveBuilder;
      if (BaseMatcher->matches(Node, Finder, &RecursiveBuilder)) {
        // After the first match the matcher succeeds.
        Matches = true;
        Builder->addMatch(RecursiveBuilder.build());
      }
      if (CurrentDepth < MaxDepth) {
        baseTraverse(Node);
      }
      // In kBindAll mode we always search for more matches.
      return true;
    }
  }

  const UntypedBaseMatcher *const BaseMatcher;
  ASTMatchFinder *const Finder;
  BoundNodesTreeBuilder *const Builder;
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
  MatchASTVisitor(std::vector< std::pair<const UntypedBaseMatcher*,
                               MatchFinder::MatchCallback*> > *Triggers)
     : Triggers(Triggers),
       ActiveASTContext(NULL) {
  }

  void set_active_ast_context(ASTContext *NewActiveASTContext) {
    ActiveASTContext = NewActiveASTContext;
  }

  // The following Visit*() and Traverse*() functions "override"
  // methods in RecursiveASTVisitor.

  bool VisitTypedefDecl(TypedefDecl *DeclNode) {
    // When we see 'typedef A B', we add name 'B' to the set of names
    // A's canonical type maps to.  This is necessary for implementing
    // IsDerivedFrom(x) properly, where x can be the name of the base
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
    // implementation of IsDerivedFrom() needs to recognize that B and
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

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  template <typename T>
  bool memoizedMatchesRecursively(const T &Node,
                                  const UntypedBaseMatcher &BaseMatcher,
                                  BoundNodesTreeBuilder *Builder, int MaxDepth,
                                  TraversalKind Traversal, BindKind Bind) {
    TOOLING_COMPILE_ASSERT((llvm::is_same<T, Decl>::value) ||
                           (llvm::is_same<T, Stmt>::value),
                           type_does_not_support_memoization);
    const UntypedMatchInput input(BaseMatcher.getID(), &Node);
    std::pair<MemoizationMap::iterator, bool> InsertResult
      = ResultCache.insert(std::make_pair(input, MemoizedMatchResult()));
    if (InsertResult.second) {
      BoundNodesTreeBuilder DescendantBoundNodesBuilder;
      InsertResult.first->second.ResultOfMatch =
        matchesRecursively(Node, BaseMatcher, &DescendantBoundNodesBuilder,
                           MaxDepth, Traversal, Bind);
      InsertResult.first->second.Nodes =
        DescendantBoundNodesBuilder.build();
    }
    InsertResult.first->second.Nodes.copyTo(Builder);
    return InsertResult.first->second.ResultOfMatch;
  }

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  template <typename T>
  bool matchesRecursively(const T &Node, const UntypedBaseMatcher &BaseMatcher,
                          BoundNodesTreeBuilder *Builder, int MaxDepth,
                          TraversalKind Traversal, BindKind Bind) {
    MatchChildASTVisitor Visitor(
      &BaseMatcher, this, Builder, MaxDepth, Traversal, Bind);
    return Visitor.findMatch(Node);
  }

  virtual bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                  const Matcher<NamedDecl> &Base,
                                  BoundNodesTreeBuilder *Builder);

  // Implements ASTMatchFinder::MatchesChildOf.
  virtual bool matchesChildOf(const Decl &DeclNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traversal,
                              BindKind Bind) {
    return matchesRecursively(DeclNode, BaseMatcher, Builder, 1, Traversal,
                              Bind);
  }
  virtual bool matchesChildOf(const Stmt &StmtNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traversal,
                              BindKind Bind) {
    return matchesRecursively(StmtNode, BaseMatcher, Builder, 1, Traversal,
                              Bind);
  }

  // Implements ASTMatchFinder::MatchesDescendantOf.
  virtual bool matchesDescendantOf(const Decl &DeclNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) {
    return memoizedMatchesRecursively(DeclNode, BaseMatcher, Builder, INT_MAX,
                                      TK_AsIs, Bind);
  }
  virtual bool matchesDescendantOf(const Stmt &StmtNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) {
    return memoizedMatchesRecursively(StmtNode, BaseMatcher, Builder, INT_MAX,
                                      TK_AsIs, Bind);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

private:
  // Implements a BoundNodesTree::Visitor that calls a MatchCallback with
  // the aggregated bound nodes for each match.
  class MatchVisitor : public BoundNodesTree::Visitor {
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
    const std::set<const TypedefDecl*> &Aliases = TypeAliases[CanonicalType];
    for (std::set<const TypedefDecl*>::const_iterator
           It = Aliases.begin(), End = Aliases.end();
         It != End; ++It) {
      if (Matcher.matches(**It, this, Builder))
        return true;
    }
    return false;
  }

  // Matches all registered matchers on the given node and calls the
  // result callback for every node that matches.
  template <typename T>
  void match(const T &node) {
    for (std::vector< std::pair<const UntypedBaseMatcher*,
                      MatchFinder::MatchCallback*> >::const_iterator
             It = Triggers->begin(), End = Triggers->end();
         It != End; ++It) {
      BoundNodesTreeBuilder Builder;
      if (It->first->matches(node, this, &Builder)) {
        BoundNodesTree BoundNodes = Builder.build();
        MatchVisitor Visitor(ActiveASTContext, It->second);
        BoundNodes.visitMatches(&Visitor);
      }
    }
  }

  std::vector< std::pair<const UntypedBaseMatcher*,
               MatchFinder::MatchCallback*> > *const Triggers;
  ASTContext *ActiveASTContext;

  // Maps a canonical type to its TypedefDecls.
  llvm::DenseMap<const Type*, std::set<const TypedefDecl*> > TypeAliases;

  // Maps (matcher, node) -> the match result for memoization.
  typedef llvm::DenseMap<UntypedMatchInput, MemoizedMatchResult> MemoizationMap;
  MemoizationMap ResultCache;
};

// Returns true if the given class is directly or indirectly derived
// from a base type with the given name.  A class is considered to be
// also derived from itself.
bool MatchASTVisitor::classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                         const Matcher<NamedDecl> &Base,
                                         BoundNodesTreeBuilder *Builder) {
  if (Base.matches(*Declaration, this, Builder))
    return true;
  if (!Declaration->hasDefinition())
    return false;
  typedef CXXRecordDecl::base_class_const_iterator BaseIterator;
  for (BaseIterator It = Declaration->bases_begin(),
                    End = Declaration->bases_end(); It != End; ++It) {
    const Type *TypeNode = It->getType().getTypePtr();

    if (typeHasMatchingAlias(TypeNode, Base, Builder))
      return true;

    // Type::getAs<...>() drills through typedefs.
    if (TypeNode->getAs<DependentNameType>() != NULL ||
        TypeNode->getAs<TemplateTypeParmType>() != NULL)
      // Dependent names and template TypeNode parameters will be matched when
      // the template is instantiated.
      continue;
    CXXRecordDecl *ClassDecl = NULL;
    TemplateSpecializationType const *TemplateType =
      TypeNode->getAs<TemplateSpecializationType>();
    if (TemplateType != NULL) {
      if (TemplateType->getTemplateName().isDependent())
        // Dependent template specializations will be matched when the
        // template is instantiated.
        continue;

      // For template specialization types which are specializing a template
      // declaration which is an explicit or partial specialization of another
      // template declaration, getAsCXXRecordDecl() returns the corresponding
      // ClassTemplateSpecializationDecl.
      //
      // For template specialization types which are specializing a template
      // declaration which is neither an explicit nor partial specialization of
      // another template declaration, getAsCXXRecordDecl() returns NULL and
      // we get the CXXRecordDecl of the templated declaration.
      CXXRecordDecl *SpecializationDecl =
        TemplateType->getAsCXXRecordDecl();
      if (SpecializationDecl != NULL) {
        ClassDecl = SpecializationDecl;
      } else {
        ClassDecl = llvm::dyn_cast<CXXRecordDecl>(
            TemplateType->getTemplateName()
                .getAsTemplateDecl()->getTemplatedDecl());
      }
    } else {
      ClassDecl = TypeNode->getAsCXXRecordDecl();
    }
    assert(ClassDecl != NULL);
    assert(ClassDecl != Declaration);
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

bool MatchASTVisitor::TraverseTypeLoc(TypeLoc TypeLoc) {
  return RecursiveASTVisitor<MatchASTVisitor>::
      TraverseType(TypeLoc.getType());
}

class MatchASTConsumer : public ASTConsumer {
public:
  MatchASTConsumer(std::vector< std::pair<const UntypedBaseMatcher*,
                                MatchFinder::MatchCallback*> > *Triggers,
                   MatchFinder::ParsingDoneTestCallback *ParsingDone)
      : Visitor(Triggers),
        ParsingDone(ParsingDone) {}

private:
  virtual void HandleTranslationUnit(ASTContext &Context) {
    if (ParsingDone != NULL) {
      ParsingDone->run();
    }
    Visitor.set_active_ast_context(&Context);
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    Visitor.set_active_ast_context(NULL);
  }

  MatchASTVisitor Visitor;
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

MatchFinder::~MatchFinder() {
  for (std::vector< std::pair<const internal::UntypedBaseMatcher*,
                    MatchFinder::MatchCallback*> >::const_iterator
           It = Triggers.begin(), End = Triggers.end();
       It != End; ++It) {
    delete It->first;
  }
}

void MatchFinder::addMatcher(const DeclarationMatcher &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
    new internal::TypedBaseMatcher<Decl>(NodeMatch), Action));
}

void MatchFinder::addMatcher(const TypeMatcher &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
    new internal::TypedBaseMatcher<QualType>(NodeMatch), Action));
}

void MatchFinder::addMatcher(const StatementMatcher &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
    new internal::TypedBaseMatcher<Stmt>(NodeMatch), Action));
}

ASTConsumer *MatchFinder::newASTConsumer() {
  return new internal::MatchASTConsumer(&Triggers, ParsingDone);
}

void MatchFinder::registerTestCallbackAfterParsing(
    MatchFinder::ParsingDoneTestCallback *NewParsingDone) {
  ParsingDone = NewParsingDone;
}

} // end namespace ast_matchers
} // end namespace clang
