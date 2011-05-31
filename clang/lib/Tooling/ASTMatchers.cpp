//===--- ASTMatchers.cpp - Structural query framework ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a framework of AST matchers that can be used to express
//  structural queries on C++ code.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include <assert.h>
#include <stddef.h>
#include <set>
#include <utility>

namespace clang {
namespace tooling {

// Returns the value that 'a_map' maps 'key' to, or NULL if 'key' is
// not in 'a_map'.
template <typename Map>
static const typename Map::mapped_type *Find(
    const Map &AMap, const typename Map::key_type &Key) {
  typename Map::const_iterator It = AMap.find(Key);
  return It == AMap.end() ? NULL : &It->second;
}

// We use memoization to avoid running the same matcher on the same
// AST node twice.  This pair is the key for looking up match
// result.  It consists of an ID of the MatcherInterface (for
// identifying the matcher) and a pointer to the AST node.
typedef std::pair<uint64_t, const void*> UntypedMatchInput;

// Used to store the result of a match and possibly bound nodes.
struct MemoizedMatchResult {
  bool ResultOfMatch;
  BoundNodes Nodes;
};

// A RecursiveASTVisitor that traverses all children or all descendants of
// a node.
class MatchChildASTVisitor
    : public clang::RecursiveASTVisitor<MatchChildASTVisitor> {
 public:
  typedef clang::RecursiveASTVisitor<MatchChildASTVisitor> VisitorBase;

  // Creates an AST visitor that matches 'matcher' on all children or
  // descendants of a traversed node. max_depth is the maximum depth
  // to traverse: use 1 for matching the children and INT_MAX for
  // matching the descendants.
  MatchChildASTVisitor(const UntypedBaseMatcher *BaseMatcher,
                       ASTMatchFinder *Finder,
                       BoundNodesBuilder *Builder,
                       int MaxDepth,
                       ASTMatchFinder::TraversalMethod Traversal)
      : BaseMatcher(BaseMatcher),
        Finder(Finder),
        Builder(Builder),
        CurrentDepth(-1),
        MaxDepth(MaxDepth),
        Traversal(Traversal),
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
  bool FindMatch(const T &Node) {
    Reset();
    Traverse(Node);
    return Matches;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.
  bool TraverseDecl(clang::Decl *DeclNode) {
    return (DeclNode == NULL) || Traverse(*DeclNode);
  }
  bool TraverseStmt(clang::Stmt *StmtNode) {
    const clang::Stmt *StmtToTraverse = StmtNode;
    if (Traversal ==
        ASTMatchFinder::kIgnoreImplicitCastsAndParentheses) {
      const clang::Expr *ExprNode = dyn_cast_or_null<clang::Expr>(StmtNode);
      if (ExprNode != NULL) {
        StmtToTraverse = ExprNode->IgnoreParenImpCasts();
      }
    }
    return (StmtToTraverse == NULL) || Traverse(*StmtToTraverse);
  }
  bool TraverseType(clang::QualType TypeNode) {
    return Traverse(TypeNode);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }

 private:
  // Resets the state of this object.
  void Reset() {
    Matches = false;
    CurrentDepth = -1;
  }

  // Forwards the call to the corresponding Traverse*() method in the
  // base visitor class.
  bool BaseTraverse(const clang::Decl &DeclNode) {
    return VisitorBase::TraverseDecl(const_cast<clang::Decl*>(&DeclNode));
  }
  bool BaseTraverse(const clang::Stmt &StmtNode) {
    return VisitorBase::TraverseStmt(const_cast<clang::Stmt*>(&StmtNode));
  }
  bool BaseTraverse(clang::QualType TypeNode) {
    return VisitorBase::TraverseType(TypeNode);
  }

  // Traverses the subtree rooted at 'node'; returns true if the
  // traversal should continue after this function returns; also sets
  // matched_ to true if a match is found during the traversal.
  template <typename T>
  bool Traverse(const T &Node) {
    COMPILE_ASSERT(IsBaseType<T>::value,
                   traverse_can_only_be_instantiated_with_base_type);
    ++CurrentDepth;
    bool ShouldContinue;
    if (CurrentDepth == 0) {
      // We don't want to match the root node, so just recurse.
      ShouldContinue = BaseTraverse(Node);
    } else if (BaseMatcher->Matches(Node, Finder, Builder)) {
      Matches = true;
      ShouldContinue = false;  // Abort as soon as a match is found.
    } else if (CurrentDepth < MaxDepth) {
      // The current node doesn't match, and we haven't reached the
      // maximum depth yet, so recurse.
      ShouldContinue = BaseTraverse(Node);
    } else {
      // The current node doesn't match, and we have reached the
      // maximum depth, so don't recurse (but continue the traversal
      // such that other nodes at the current level can be visited).
      ShouldContinue = true;
    }
    --CurrentDepth;
    return ShouldContinue;
  }

  const UntypedBaseMatcher *const BaseMatcher;
  ASTMatchFinder *const Finder;
  BoundNodesBuilder *const Builder;
  int CurrentDepth;
  const int MaxDepth;
  const ASTMatchFinder::TraversalMethod Traversal;
  bool Matches;
};

// Controls the outermost traversal of the AST and allows to match multiple
// matchers.
class MatchASTVisitor : public clang::RecursiveASTVisitor<MatchASTVisitor>,
                        public ASTMatchFinder {
 public:
  MatchASTVisitor(std::vector< std::pair<const UntypedBaseMatcher*,
                               MatchFinder::MatchCallback*> > *Triggers,
                  clang::SourceManager *VisitorSourceManager,
                  clang::LangOptions *LanguageOptions)
     : Triggers(Triggers),
       VisitorSourceManager(VisitorSourceManager),
       LanguageOptions(LanguageOptions),
       ActiveASTContext(NULL) {
    assert(VisitorSourceManager != NULL);
    assert(LanguageOptions != NULL);
    // FIXME: add rewriter_(*source_manager, *language_options)
  }

  void set_active_ast_context(clang::ASTContext *NewActiveASTContext) {
    ActiveASTContext = NewActiveASTContext;
  }

  // The following Visit*() and Traverse*() functions "override"
  // methods in RecursiveASTVisitor.

  bool VisitTypedefDecl(clang::TypedefDecl *DeclNode) {
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
    const clang::Type *TypeNode = DeclNode->getUnderlyingType().getTypePtr();
    const clang::Type *CanonicalType =  // root of the typedef tree
        ActiveASTContext->getCanonicalType(TypeNode);
    TypeToUnqualifiedAliases[CanonicalType].insert(
        DeclNode->getName().str());
    return true;
  }

  bool TraverseDecl(clang::Decl *DeclNode);
  bool TraverseStmt(clang::Stmt *StmtNode);
  bool TraverseType(clang::QualType TypeNode);
  bool TraverseTypeLoc(clang::TypeLoc TypeNode);

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  template <typename T>
  bool MemoizedMatchesRecursively(
      const T &Node, const UntypedBaseMatcher &BaseMatcher,
      BoundNodesBuilder *Builder, int MaxDepth,
      TraversalMethod Traversal) {
    COMPILE_ASSERT((llvm::is_same<T, clang::Decl>::value) ||
                   (llvm::is_same<T, clang::Stmt>::value),
                   type_does_not_support_memoization);
    const UntypedMatchInput input(BaseMatcher.GetID(), &Node);
    std::pair <MemoizationMap::iterator, bool>
        InsertResult = ResultCache.insert(
            std::make_pair(input, MemoizedMatchResult()));
    if (InsertResult.second) {
      BoundNodesBuilder DescendantBoundNodesBuilder;
      InsertResult.first->second.ResultOfMatch =
          MatchesRecursively(Node, BaseMatcher, &DescendantBoundNodesBuilder,
                             MaxDepth, Traversal);
      InsertResult.first->second.Nodes =
          DescendantBoundNodesBuilder.Build();
    }
    InsertResult.first->second.Nodes.CopyTo(Builder);
    return InsertResult.first->second.ResultOfMatch;
  }

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  template <typename T>
  bool MatchesRecursively(
      const T &Node, const UntypedBaseMatcher &BaseMatcher,
      BoundNodesBuilder *Builder, int MaxDepth,
      TraversalMethod Traversal) {
    MatchChildASTVisitor Visitor(
        &BaseMatcher, this, Builder, MaxDepth, Traversal);
    return Visitor.FindMatch(Node);
  }

  virtual bool ClassIsDerivedFrom(const clang::CXXRecordDecl *Declaration,
                                  const std::string &BaseName) const;

  // Implements ASTMatchFinder::MatchesChildOf.
  virtual bool MatchesChildOf(const clang::Decl &DeclNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesBuilder *Builder,
                              TraversalMethod Traversal) {
    return MatchesRecursively(
        DeclNode, BaseMatcher, Builder, 1, Traversal);
  }
  virtual bool MatchesChildOf(const clang::Stmt &StmtNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesBuilder *Builder,
                              TraversalMethod Traversal) {
    return MatchesRecursively(
        StmtNode, BaseMatcher, Builder, 1, Traversal);
  }

  // Implements ASTMatchFinder::MatchesDescendantOf.
  virtual bool MatchesDescendantOf(const clang::Decl &DeclNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesBuilder *Builder) {
    return MemoizedMatchesRecursively(
        DeclNode, BaseMatcher, Builder, INT_MAX, kAsIs);
  }
  virtual bool MatchesDescendantOf(const clang::Stmt &StmtNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesBuilder *Builder) {
    return MemoizedMatchesRecursively(
        StmtNode, BaseMatcher, Builder, INT_MAX, kAsIs);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }

 private:
  // Returns true if 'TypeNode' is also known by the name 'Name'.  In other
  // words, there is a type (including typedef) with the name 'Name'
  // that is equal to 'TypeNode'.
  bool TypeHasAlias(
      const clang::Type *TypeNode, const std::string &Name) const {
    const clang::Type *const CanonicalType =
        ActiveASTContext->getCanonicalType(TypeNode);
    const std::set<std::string> *UnqualifiedAlias =
        Find(TypeToUnqualifiedAliases, CanonicalType);
    return UnqualifiedAlias != NULL && UnqualifiedAlias->count(Name) > 0;
  }

  // Matches all registered matchers on the given node and calls the
  // result callback for every node that matches.
  template <typename T>
  void Match(const T &node) {
    for (std::vector< std::pair<const UntypedBaseMatcher*,
                      MatchFinder::MatchCallback*> >::const_iterator
             It = Triggers->begin(), End = Triggers->end();
         It != End; ++It) {
      BoundNodesBuilder Builder;
      if (It->first->Matches(node, this, &Builder)) {
        MatchFinder::MatchResult Result;
        Result.Nodes = Builder.Build();
        Result.Context = ActiveASTContext;
        Result.SourceManager = VisitorSourceManager;
        It->second->Run(Result);
      }
    }
  }

  std::vector< std::pair<const UntypedBaseMatcher*,
               MatchFinder::MatchCallback*> > *const Triggers;
  clang::SourceManager *const VisitorSourceManager;
  clang::LangOptions *const LanguageOptions;
  clang::ASTContext *ActiveASTContext;

  // Maps a canonical type to the names of its typedefs.
  llvm::DenseMap<const clang::Type*, std::set<std::string> >
      TypeToUnqualifiedAliases;

  // Maps (matcher, node) -> the match result for memoization.
  typedef llvm::DenseMap<UntypedMatchInput, MemoizedMatchResult> MemoizationMap;
  MemoizationMap ResultCache;
};

// Returns true if the given class is directly or indirectly derived
// from a base type with the given name.  A class is considered to be
// also derived from itself.
bool MatchASTVisitor::ClassIsDerivedFrom(
    const clang::CXXRecordDecl *Declaration,
    const std::string &BaseName) const {
  if (std::string(Declaration->getName()) == BaseName) {
    return true;
  }
  if (!Declaration->hasDefinition()) {
    return false;
  }
  typedef clang::CXXRecordDecl::base_class_const_iterator BaseIterator;
  for (BaseIterator It = Declaration->bases_begin(),
           End = Declaration->bases_end(); It != End; ++It) {
    const clang::Type *TypeNode = It->getType().getTypePtr();

    if (TypeHasAlias(TypeNode, BaseName))
      return true;

    // clang::Type::getAs<...>() drills through typedefs.
    if (TypeNode->getAs<clang::DependentNameType>() != NULL ||
        TypeNode->getAs<clang::TemplateTypeParmType>() != NULL) {
      // Dependent names and template TypeNode parameters will be matched when
      // the template is instantiated.
      continue;
    }
    clang::CXXRecordDecl *ClassDecl = NULL;
    clang::TemplateSpecializationType const *TemplateType =
      TypeNode->getAs<clang::TemplateSpecializationType>();
    if (TemplateType != NULL) {
      if (TemplateType->getTemplateName().isDependent()) {
        // Dependent template specializations will be matched when the
        // template is instantiated.
        continue;
      }
      // For template specialization types which are specializing a template
      // declaration which is an explicit or partial specialization of another
      // template declaration, getAsCXXRecordDecl() returns the corresponding
      // ClassTemplateSpecializationDecl.
      //
      // For template specialization types which are specializing a template
      // declaration which is neither an explicit nor partial specialization of
      // another template declaration, getAsCXXRecordDecl() returns NULL and
      // we get the CXXRecordDecl of the templated declaration.
      clang::CXXRecordDecl *SpecializationDecl =
          TemplateType->getAsCXXRecordDecl();
      if (SpecializationDecl != NULL) {
        ClassDecl = SpecializationDecl;
      } else {
        ClassDecl = llvm::dyn_cast<clang::CXXRecordDecl>(
            TemplateType->getTemplateName()
                .getAsTemplateDecl()->getTemplatedDecl());
      }
    } else {
      ClassDecl = TypeNode->getAsCXXRecordDecl();
    }
    assert(ClassDecl != NULL);
    assert(ClassDecl != Declaration);
    if (ClassIsDerivedFrom(ClassDecl, BaseName)) {
      return true;
    }
  }
  return false;
}

bool MatchASTVisitor::TraverseDecl(clang::Decl *DeclNode) {
  if (DeclNode == NULL) {
    return true;
  }
  Match(*DeclNode);
  return clang::RecursiveASTVisitor<MatchASTVisitor>::TraverseDecl(DeclNode);
}

bool MatchASTVisitor::TraverseStmt(clang::Stmt *StmtNode) {
  if (StmtNode == NULL) {
    return true;
  }
  Match(*StmtNode);
  return clang::RecursiveASTVisitor<MatchASTVisitor>::TraverseStmt(StmtNode);
}

bool MatchASTVisitor::TraverseType(clang::QualType TypeNode) {
  Match(TypeNode);
  return clang::RecursiveASTVisitor<MatchASTVisitor>::TraverseType(TypeNode);
}

bool MatchASTVisitor::TraverseTypeLoc(clang::TypeLoc TypeLoc) {
  return clang::RecursiveASTVisitor<MatchASTVisitor>::
      TraverseType(TypeLoc.getType());
}

class MatchASTConsumer : public clang::ASTConsumer {
 public:
  MatchASTConsumer(std::vector< std::pair<const UntypedBaseMatcher*,
                                MatchFinder::MatchCallback*> > *Triggers,
                   MatchFinder::ParsingDoneTestCallback *ParsingDone,
                   clang::SourceManager *ConsumerSourceManager,
                   clang::LangOptions *LanaguageOptions)
      : Visitor(Triggers, ConsumerSourceManager, LanaguageOptions),
        ParsingDone(ParsingDone) {}

 private:
  virtual void HandleTranslationUnit(
      clang::ASTContext &Context) {  // NOLINT: external API uses refs
    if (ParsingDone != NULL) {
      ParsingDone->Run();
    }
    Visitor.set_active_ast_context(&Context);
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    Visitor.set_active_ast_context(NULL);
  }

  MatchASTVisitor Visitor;
  MatchFinder::ParsingDoneTestCallback *ParsingDone;
};

class MatchASTAction : public clang::ASTFrontendAction {
 public:
  explicit MatchASTAction(
      std::vector< std::pair<const UntypedBaseMatcher*,
                   MatchFinder::MatchCallback*> > *Triggers,
      MatchFinder::ParsingDoneTestCallback *ParsingDone)
      : Triggers(Triggers),
        ParsingDone(ParsingDone) {}

 private:
  clang::ASTConsumer *CreateASTConsumer(
      clang::CompilerInstance &Compiler,
      llvm::StringRef) {
    return new MatchASTConsumer(Triggers,
                                ParsingDone,
                                &Compiler.getSourceManager(),
                                &Compiler.getLangOpts());
  }

  std::vector< std::pair<const UntypedBaseMatcher*,
               MatchFinder::MatchCallback*> > *const Triggers;
  MatchFinder::ParsingDoneTestCallback *ParsingDone;
};

MatchFinder::MatchCallback::~MatchCallback() {}
MatchFinder::ParsingDoneTestCallback::~ParsingDoneTestCallback() {}

MatchFinder::MatchFinder() : ParsingDone(NULL) {}

MatchFinder::~MatchFinder() {
  for (std::vector< std::pair<const UntypedBaseMatcher*,
                    MatchFinder::MatchCallback*> >::const_iterator
           It = Triggers.begin(), End = Triggers.end();
       It != End; ++It) {
    delete It->first;
    delete It->second;
  }
}

void MatchFinder::AddMatcher(const Matcher<clang::Decl> &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
      new TypedBaseMatcher<clang::Decl>(NodeMatch), Action));
}

void MatchFinder::AddMatcher(const Matcher<clang::QualType> &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
      new TypedBaseMatcher<clang::QualType>(NodeMatch), Action));
}

void MatchFinder::AddMatcher(const Matcher<clang::Stmt> &NodeMatch,
                             MatchCallback *Action) {
  Triggers.push_back(std::make_pair(
      new TypedBaseMatcher<clang::Stmt>(NodeMatch), Action));
}

bool MatchFinder::FindAll(const std::string &Code) {
  return RunSyntaxOnlyToolOnCode(
      new MatchASTAction(&Triggers, ParsingDone), Code);
}

clang::FrontendAction *MatchFinder::NewVisitorAction() {
  return new MatchASTAction(&Triggers, ParsingDone);
}

class MatchFinderFrontendActionFactory : public FrontendActionFactory {
 public:
  explicit MatchFinderFrontendActionFactory(MatchFinder *Finder)
      : Finder(Finder) {}

  virtual clang::FrontendAction *New() {
    return Finder->NewVisitorAction();
  }

 private:
  MatchFinder *const Finder;
};

FrontendActionFactory *MatchFinder::NewFrontendActionFactory() {
  return new MatchFinderFrontendActionFactory(this);
}

void MatchFinder::RegisterTestCallbackAfterParsing(
    MatchFinder::ParsingDoneTestCallback *NewParsingDone) {
  ParsingDone = NewParsingDone;
}

} // end namespace tooling
} // end namespace clang
