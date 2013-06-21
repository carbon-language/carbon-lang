//===--- ASTMatchersInternal.h - Structural query framework -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//  Matchers are methods that return a Matcher<T> which provides a method
//  Matches(...) which is a predicate on an AST node. The Matches method's
//  parameters define the context of the match, which allows matchers to recurse
//  or store the current node as bound to a specific string, so that it can be
//  retrieved later.
//
//  In general, matchers have two parts:
//  1. A function Matcher<T> MatcherName(<arguments>) which returns a Matcher<T>
//     based on the arguments and optionally on template type deduction based
//     on the arguments. Matcher<T>s form an implicit reverse hierarchy
//     to clang's AST class hierarchy, meaning that you can use a Matcher<Base>
//     everywhere a Matcher<Derived> is required.
//  2. An implementation of a class derived from MatcherInterface<T>.
//
//  The matcher functions are defined in ASTMatchers.h. To make it possible
//  to implement both the matcher function and the implementation of the matcher
//  interface in one place, ASTMatcherMacros.h defines macros that allow
//  implementing a matcher in a single place.
//
//  This file contains the base classes needed to construct the actual matchers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H
#define LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/VariadicFunction.h"
#include "llvm/Support/type_traits.h"
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace ast_matchers {

/// FIXME: Move into the llvm support library.
template <bool> struct CompileAssert {};
#define TOOLING_COMPILE_ASSERT(Expr, Msg) \
  typedef CompileAssert<(bool(Expr))> Msg[bool(Expr) ? 1 : -1]

class BoundNodes;

namespace internal {

/// \brief Internal version of BoundNodes. Holds all the bound nodes.
class BoundNodesMap {
public:
  /// \brief Adds \c Node to the map with key \c ID.
  ///
  /// The node's base type should be in NodeBaseType or it will be unaccessible.
  template <typename T>
  void addNode(StringRef ID, const T* Node) {
    NodeMap[ID] = ast_type_traits::DynTypedNode::create(*Node);
  }

  /// \brief Returns the AST node bound to \c ID.
  ///
  /// Returns NULL if there was no node bound to \c ID or if there is a node but
  /// it cannot be converted to the specified type.
  template <typename T>
  const T *getNodeAs(StringRef ID) const {
    IDToNodeMap::const_iterator It = NodeMap.find(ID);
    if (It == NodeMap.end()) {
      return NULL;
    }
    return It->second.get<T>();
  }

  ast_type_traits::DynTypedNode getNode(StringRef ID) const {
    IDToNodeMap::const_iterator It = NodeMap.find(ID);
    if (It == NodeMap.end()) {
      return ast_type_traits::DynTypedNode();
    }
    return It->second;
  }

  /// \brief Imposes an order on BoundNodesMaps.
  bool operator<(const BoundNodesMap &Other) const {
    return NodeMap < Other.NodeMap;
  }

private:
  /// \brief A map from IDs to the bound nodes.
  ///
  /// Note that we're using std::map here, as for memoization:
  /// - we need a comparison operator
  /// - we need an assignment operator
  typedef std::map<std::string, ast_type_traits::DynTypedNode> IDToNodeMap;

  IDToNodeMap NodeMap;
};

/// \brief Creates BoundNodesTree objects.
///
/// The tree builder is used during the matching process to insert the bound
/// nodes from the Id matcher.
class BoundNodesTreeBuilder {
public:
  /// \brief A visitor interface to visit all BoundNodes results for a
  /// BoundNodesTree.
  class Visitor {
  public:
    virtual ~Visitor() {}

    /// \brief Called multiple times during a single call to VisitMatches(...).
    ///
    /// 'BoundNodesView' contains the bound nodes for a single match.
    virtual void visitMatch(const BoundNodes& BoundNodesView) = 0;
  };

  /// \brief Add a binding from an id to a node.
  template <typename T> void setBinding(const std::string &Id, const T *Node) {
    if (Bindings.empty())
      Bindings.push_back(BoundNodesMap());
    for (unsigned i = 0, e = Bindings.size(); i != e; ++i)
      Bindings[i].addNode(Id, Node);
  }

  /// \brief Adds a branch in the tree.
  void addMatch(const BoundNodesTreeBuilder &Bindings);

  /// \brief Visits all matches that this BoundNodesTree represents.
  ///
  /// The ownership of 'ResultVisitor' remains at the caller.
  void visitMatches(Visitor* ResultVisitor);

  template <typename ExcludePredicate>
  bool removeBindings(const ExcludePredicate &Predicate) {
    Bindings.erase(std::remove_if(Bindings.begin(), Bindings.end(), Predicate),
                   Bindings.end());
    return !Bindings.empty();
  }

  /// \brief Imposes an order on BoundNodesTreeBuilders.
  bool operator<(const BoundNodesTreeBuilder &Other) const {
    return Bindings < Other.Bindings;
  }

private:
  SmallVector<BoundNodesMap, 16> Bindings;
};

class ASTMatchFinder;

/// \brief Generic interface for matchers on an AST node of type T.
///
/// Implement this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public RefCountedBaseVPTR {
public:
  virtual ~MatcherInterface() {}

  /// \brief Returns true if 'Node' can be matched.
  ///
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the AST via 'Finder'.
  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const = 0;
};

/// \brief Interface for matchers that only evaluate properties on a single
/// node.
template <typename T>
class SingleNodeMatcherInterface : public MatcherInterface<T> {
public:
  /// \brief Returns true if the matcher matches the provided node.
  ///
  /// A subclass must implement this instead of Matches().
  virtual bool matchesNode(const T &Node) const = 0;

private:
  /// Implements MatcherInterface::Matches.
  virtual bool matches(const T &Node,
                       ASTMatchFinder * /* Finder */,
                       BoundNodesTreeBuilder * /*  Builder */) const {
    return matchesNode(Node);
  }
};

/// \brief Base class for all matchers that works on a \c DynTypedNode.
///
/// Matcher implementations will check whether the \c DynTypedNode is
/// convertible into the respecitve types and then do the actual match
/// on the actual node, or return false if it is not convertible.
class DynTypedMatcher {
public:
  virtual ~DynTypedMatcher();

  /// \brief Returns true if the matcher matches the given \c DynNode.
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const = 0;

  /// \brief Makes a copy of this matcher object.
  virtual DynTypedMatcher *clone() const = 0;

  /// \brief Returns a unique ID for the matcher.
  virtual uint64_t getID() const = 0;

  /// \brief Bind the specified \p ID to the matcher.
  /// \return A new matcher with the \p ID bound to it if this matcher supports
  ///   binding. Otherwise, returns NULL. Returns NULL by default.
  virtual DynTypedMatcher* tryBind(StringRef ID) const;

  /// \brief Returns the type this matcher works on.
  ///
  /// \c matches() will always return false unless the node passed is of this
  /// or a derived type.
  virtual ast_type_traits::ASTNodeKind getSupportedKind() const = 0;
};

/// \brief Wrapper of a MatcherInterface<T> *that allows copying.
///
/// A Matcher<Base> can be used anywhere a Matcher<Derived> is
/// required. This establishes an is-a relationship which is reverse
/// to the AST hierarchy. In other words, Matcher<T> is contravariant
/// with respect to T. The relationship is built via a type conversion
/// operator rather than a type hierarchy to be able to templatize the
/// type hierarchy instead of spelling it out.
template <typename T>
class Matcher : public DynTypedMatcher {
public:
  /// \brief Takes ownership of the provided implementation pointer.
  explicit Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}

  /// \brief Implicitly converts \c Other to a Matcher<T>.
  ///
  /// Requires \c T to be derived from \c From.
  template <typename From>
  Matcher(const Matcher<From> &Other,
          typename llvm::enable_if_c<
            llvm::is_base_of<From, T>::value &&
            !llvm::is_same<From, T>::value >::type* = 0)
      : Implementation(new ImplicitCastMatcher<From>(Other)) {}

  /// \brief Implicitly converts \c Matcher<Type> to \c Matcher<QualType>.
  ///
  /// The resulting matcher is not strict, i.e. ignores qualifiers.
  template <typename TypeT>
  Matcher(const Matcher<TypeT> &Other,
          typename llvm::enable_if_c<
            llvm::is_same<T, QualType>::value &&
            llvm::is_same<TypeT, Type>::value >::type* = 0)
      : Implementation(new TypeToQualType<TypeT>(Other)) {}

  /// \brief Returns \c true if the passed DynTypedMatcher can be converted
  ///   to a \c Matcher<T>.
  ///
  /// This method verifies that the underlying matcher in \c Other can process
  /// nodes of types T.
  static bool canConstructFrom(const DynTypedMatcher &Other) {
    return Other.getSupportedKind()
        .isBaseOf(ast_type_traits::ASTNodeKind::getFromNodeKind<T>());
  }

  /// \brief Construct a Matcher<T> interface around the dynamic matcher
  ///   \c Other.
  ///
  /// This method asserts that canConstructFrom(Other) is \c true. Callers
  /// should call canConstructFrom(Other) first to make sure that Other is
  /// compatible with T.
  static Matcher<T> constructFrom(const DynTypedMatcher &Other) {
    assert(canConstructFrom(Other));
    return Matcher<T>(new WrappedMatcher(Other));
  }

  /// \brief Forwards the call to the underlying MatcherInterface<T> pointer.
  bool matches(const T &Node,
               ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const {
    if (Implementation->matches(Node, Finder, Builder))
      return true;
    // Delete all bindings when a matcher does not match.
    // This prevents unexpected exposure of bound nodes in unmatches
    // branches of the match tree.
    *Builder = BoundNodesTreeBuilder();
    return false;
  }

  /// \brief Returns an ID that uniquely identifies the matcher.
  uint64_t getID() const {
    /// FIXME: Document the requirements this imposes on matcher
    /// implementations (no new() implementation_ during a Matches()).
    return reinterpret_cast<uint64_t>(Implementation.getPtr());
  }

  /// \brief Returns the type this matcher works on.
  ast_type_traits::ASTNodeKind getSupportedKind() const {
    return ast_type_traits::ASTNodeKind::getFromNodeKind<T>();
  }

  /// \brief Returns whether the matcher matches on the given \c DynNode.
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    const T *Node = DynNode.get<T>();
    if (!Node) return false;
    return matches(*Node, Finder, Builder);
  }

  /// \brief Makes a copy of this matcher object.
  virtual Matcher<T> *clone() const { return new Matcher<T>(*this); }

  /// \brief Allows the conversion of a \c Matcher<Type> to a \c
  /// Matcher<QualType>.
  ///
  /// Depending on the constructor argument, the matcher is either strict, i.e.
  /// does only matches in the absence of qualifiers, or not, i.e. simply
  /// ignores any qualifiers.
  template <typename TypeT>
  class TypeToQualType : public MatcherInterface<QualType> {
   public:
    TypeToQualType(const Matcher<TypeT> &InnerMatcher)
        : InnerMatcher(InnerMatcher) {}

    virtual bool matches(const QualType &Node,
                         ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) const {
      if (Node.isNull())
        return false;
      return InnerMatcher.matches(*Node, Finder, Builder);
    }
   private:
    const Matcher<TypeT> InnerMatcher;
  };

private:
  /// \brief Allows conversion from Matcher<Base> to Matcher<T> if T
  /// is derived from Base.
  template <typename Base>
  class ImplicitCastMatcher : public MatcherInterface<T> {
  public:
    explicit ImplicitCastMatcher(const Matcher<Base> &From)
        : From(From) {}

    virtual bool matches(const T &Node,
                         ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) const {
      return From.matches(Node, Finder, Builder);
    }

  private:
    const Matcher<Base> From;
  };

  /// \brief Simple MatcherInterface<T> wrapper around a DynTypedMatcher.
  class WrappedMatcher : public MatcherInterface<T> {
  public:
    explicit WrappedMatcher(const DynTypedMatcher &Matcher)
        : Inner(Matcher.clone()) {}
    virtual ~WrappedMatcher() {}

    bool matches(const T &Node, ASTMatchFinder *Finder,
                 BoundNodesTreeBuilder *Builder) const {
      return Inner->matches(ast_type_traits::DynTypedNode::create(Node), Finder,
                            Builder);
    }

  private:
    const OwningPtr<DynTypedMatcher> Inner;
  };

  IntrusiveRefCntPtr< MatcherInterface<T> > Implementation;
};  // class Matcher

/// \brief A convenient helper for creating a Matcher<T> without specifying
/// the template type argument.
template <typename T>
inline Matcher<T> makeMatcher(MatcherInterface<T> *Implementation) {
  return Matcher<T>(Implementation);
}

/// \brief Specialization of the conversion functions for QualType.
///
/// These specializations provide the Matcher<Type>->Matcher<QualType>
/// conversion that the static API does.
template <>
inline bool
Matcher<QualType>::canConstructFrom(const DynTypedMatcher &Other) {
  ast_type_traits::ASTNodeKind SourceKind = Other.getSupportedKind();
  // We support implicit conversion from Matcher<Type> to Matcher<QualType>
  return SourceKind.isSame(
             ast_type_traits::ASTNodeKind::getFromNodeKind<Type>()) ||
         SourceKind.isSame(
             ast_type_traits::ASTNodeKind::getFromNodeKind<QualType>());
}

template <>
inline Matcher<QualType>
Matcher<QualType>::constructFrom(const DynTypedMatcher &Other) {
  assert(canConstructFrom(Other));
  ast_type_traits::ASTNodeKind SourceKind = Other.getSupportedKind();
  if (SourceKind.isSame(
          ast_type_traits::ASTNodeKind::getFromNodeKind<Type>())) {
    // We support implicit conversion from Matcher<Type> to Matcher<QualType>
    return Matcher<Type>::constructFrom(Other);
  }
  return makeMatcher(new WrappedMatcher(Other));
}

/// \brief Finds the first node in a range that matches the given matcher.
template <typename MatcherT, typename IteratorT>
bool matchesFirstInRange(const MatcherT &Matcher, IteratorT Start,
                         IteratorT End, ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) {
  for (IteratorT I = Start; I != End; ++I) {
    BoundNodesTreeBuilder Result(*Builder);
    if (Matcher.matches(*I, Finder, &Result)) {
      *Builder = Result;
      return true;
    }
  }
  return false;
}

/// \brief Finds the first node in a pointer range that matches the given
/// matcher.
template <typename MatcherT, typename IteratorT>
bool matchesFirstInPointerRange(const MatcherT &Matcher, IteratorT Start,
                                IteratorT End, ASTMatchFinder *Finder,
                                BoundNodesTreeBuilder *Builder) {
  for (IteratorT I = Start; I != End; ++I) {
    BoundNodesTreeBuilder Result(*Builder);
    if (Matcher.matches(**I, Finder, &Result)) {
      *Builder = Result;
      return true;
    }
  }
  return false;
}

/// \brief Metafunction to determine if type T has a member called getDecl.
template <typename T> struct has_getDecl {
  struct Default { int getDecl; };
  struct Derived : T, Default { };

  template<typename C, C> struct CheckT;

  // If T::getDecl exists, an ambiguity arises and CheckT will
  // not be instantiable. This makes f(...) the only available
  // overload.
  template<typename C>
  static char (&f(CheckT<int Default::*, &C::getDecl>*))[1];
  template<typename C> static char (&f(...))[2];

  static bool const value = sizeof(f<Derived>(0)) == 2;
};

/// \brief Matches overloaded operators with a specific name.
///
/// The type argument ArgT is not used by this matcher but is used by
/// PolymorphicMatcherWithParam1 and should be StringRef.
template <typename T, typename ArgT>
class HasOverloadedOperatorNameMatcher : public SingleNodeMatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_same<T, CXXOperatorCallExpr>::value ||
                          llvm::is_same<T, CXXMethodDecl>::value),
                         unsupported_class_for_matcher);
  TOOLING_COMPILE_ASSERT((llvm::is_same<ArgT, StringRef>::value),
                         argument_type_must_be_StringRef);
public:
  explicit HasOverloadedOperatorNameMatcher(const StringRef Name)
      : SingleNodeMatcherInterface<T>(), Name(Name) {}

  virtual bool matchesNode(const T &Node) const LLVM_OVERRIDE {
    return matchesSpecialized(Node);
  }

private:

  /// \brief CXXOperatorCallExpr exist only for calls to overloaded operators
  /// so this function returns true if the call is to an operator of the given
  /// name.
  bool matchesSpecialized(const CXXOperatorCallExpr &Node) const {
    return getOperatorSpelling(Node.getOperator()) == Name;
  }

  /// \brief Returns true only if CXXMethodDecl represents an overloaded
  /// operator and has the given operator name.
  bool matchesSpecialized(const CXXMethodDecl &Node) const {
    return Node.isOverloadedOperator() &&
           getOperatorSpelling(Node.getOverloadedOperator()) == Name;
  }

  std::string Name;
};

/// \brief Matches declarations for QualType and CallExpr.
///
/// Type argument DeclMatcherT is required by PolymorphicMatcherWithParam1 but
/// not actually used.
template <typename T, typename DeclMatcherT>
class HasDeclarationMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_same< DeclMatcherT,
                                         Matcher<Decl> >::value),
                          instantiated_with_wrong_types);
public:
  explicit HasDeclarationMatcher(const Matcher<Decl> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return matchesSpecialized(Node, Finder, Builder);
  }

private:
  /// \brief If getDecl exists as a member of U, returns whether the inner
  /// matcher matches Node.getDecl().
  template <typename U>
  bool matchesSpecialized(
      const U &Node, ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder,
      typename llvm::enable_if<has_getDecl<U>, int>::type = 0) const {
    return matchesDecl(Node.getDecl(), Finder, Builder);
  }

  /// \brief Extracts the CXXRecordDecl or EnumDecl of a QualType and returns
  /// whether the inner matcher matches on it.
  bool matchesSpecialized(const QualType &Node, ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    /// FIXME: Add other ways to convert...
    if (Node.isNull())
      return false;
    if (const EnumType *AsEnum = dyn_cast<EnumType>(Node.getTypePtr()))
      return matchesDecl(AsEnum->getDecl(), Finder, Builder);
    return matchesDecl(Node->getAsCXXRecordDecl(), Finder, Builder);
  }

  /// \brief Gets the TemplateDecl from a TemplateSpecializationType
  /// and returns whether the inner matches on it.
  bool matchesSpecialized(const TemplateSpecializationType &Node,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getTemplateName().getAsTemplateDecl(),
                       Finder, Builder);
  }

  /// \brief Extracts the Decl of the callee of a CallExpr and returns whether
  /// the inner matcher matches on it.
  bool matchesSpecialized(const CallExpr &Node, ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getCalleeDecl(), Finder, Builder);
  }

  /// \brief Extracts the Decl of the constructor call and returns whether the
  /// inner matcher matches on it.
  bool matchesSpecialized(const CXXConstructExpr &Node,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getConstructor(), Finder, Builder);
  }

  /// \brief Extracts the \c ValueDecl a \c MemberExpr refers to and returns
  /// whether the inner matcher matches on it.
  bool matchesSpecialized(const MemberExpr &Node,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getMemberDecl(), Finder, Builder);
  }

  /// \brief Returns whether the inner matcher \c Node. Returns false if \c Node
  /// is \c NULL.
  bool matchesDecl(const Decl *Node,
                   ASTMatchFinder *Finder,
                   BoundNodesTreeBuilder *Builder) const {
    return Node != NULL && InnerMatcher.matches(*Node, Finder, Builder);
  }

  const Matcher<Decl> InnerMatcher;
};

/// \brief IsBaseType<T>::value is true if T is a "base" type in the AST
/// node class hierarchies.
template <typename T>
struct IsBaseType {
  static const bool value =
      (llvm::is_same<T, Decl>::value ||
       llvm::is_same<T, Stmt>::value ||
       llvm::is_same<T, QualType>::value ||
       llvm::is_same<T, Type>::value ||
       llvm::is_same<T, TypeLoc>::value ||
       llvm::is_same<T, NestedNameSpecifier>::value ||
       llvm::is_same<T, NestedNameSpecifierLoc>::value ||
       llvm::is_same<T, CXXCtorInitializer>::value);
};
template <typename T>
const bool IsBaseType<T>::value;

/// \brief Interface that allows matchers to traverse the AST.
/// FIXME: Find a better name.
///
/// This provides three entry methods for each base node type in the AST:
/// - \c matchesChildOf:
///   Matches a matcher on every child node of the given node. Returns true
///   if at least one child node could be matched.
/// - \c matchesDescendantOf:
///   Matches a matcher on all descendant nodes of the given node. Returns true
///   if at least one descendant matched.
/// - \c matchesAncestorOf:
///   Matches a matcher on all ancestors of the given node. Returns true if
///   at least one ancestor matched.
///
/// FIXME: Currently we only allow Stmt and Decl nodes to start a traversal.
/// In the future, we wan to implement this for all nodes for which it makes
/// sense. In the case of matchesAncestorOf, we'll want to implement it for
/// all nodes, as all nodes have ancestors.
class ASTMatchFinder {
public:
  /// \brief Defines how we descend a level in the AST when we pass
  /// through expressions.
  enum TraversalKind {
    /// Will traverse any child nodes.
    TK_AsIs,
    /// Will not traverse implicit casts and parentheses.
    TK_IgnoreImplicitCastsAndParentheses
  };

  /// \brief Defines how bindings are processed on recursive matches.
  enum BindKind {
    /// Stop at the first match and only bind the first match.
    BK_First,
    /// Create results for all combinations of bindings that match.
    BK_All
  };

  /// \brief Defines which ancestors are considered for a match.
  enum AncestorMatchMode {
    /// All ancestors.
    AMM_All,
    /// Direct parent only.
    AMM_ParentOnly
  };

  virtual ~ASTMatchFinder() {}

  /// \brief Returns true if the given class is directly or indirectly derived
  /// from a base type matching \c base.
  ///
  /// A class is considered to be also derived from itself.
  virtual bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                  const Matcher<NamedDecl> &Base,
                                  BoundNodesTreeBuilder *Builder) = 0;

  template <typename T>
  bool matchesChildOf(const T &Node,
                      const DynTypedMatcher &Matcher,
                      BoundNodesTreeBuilder *Builder,
                      TraversalKind Traverse,
                      BindKind Bind) {
    TOOLING_COMPILE_ASSERT(
        (llvm::is_base_of<Decl, T>::value ||
         llvm::is_base_of<Stmt, T>::value ||
         llvm::is_base_of<NestedNameSpecifier, T>::value ||
         llvm::is_base_of<NestedNameSpecifierLoc, T>::value ||
         llvm::is_base_of<TypeLoc, T>::value ||
         llvm::is_base_of<QualType, T>::value),
        unsupported_type_for_recursive_matching);
   return matchesChildOf(ast_type_traits::DynTypedNode::create(Node),
                          Matcher, Builder, Traverse, Bind);
  }

  template <typename T>
  bool matchesDescendantOf(const T &Node,
                           const DynTypedMatcher &Matcher,
                           BoundNodesTreeBuilder *Builder,
                           BindKind Bind) {
    TOOLING_COMPILE_ASSERT(
        (llvm::is_base_of<Decl, T>::value ||
         llvm::is_base_of<Stmt, T>::value ||
         llvm::is_base_of<NestedNameSpecifier, T>::value ||
         llvm::is_base_of<NestedNameSpecifierLoc, T>::value ||
         llvm::is_base_of<TypeLoc, T>::value ||
         llvm::is_base_of<QualType, T>::value),
        unsupported_type_for_recursive_matching);
    return matchesDescendantOf(ast_type_traits::DynTypedNode::create(Node),
                               Matcher, Builder, Bind);
  }

  // FIXME: Implement support for BindKind.
  template <typename T>
  bool matchesAncestorOf(const T &Node,
                         const DynTypedMatcher &Matcher,
                         BoundNodesTreeBuilder *Builder,
                         AncestorMatchMode MatchMode) {
    TOOLING_COMPILE_ASSERT((llvm::is_base_of<Decl, T>::value ||
                            llvm::is_base_of<Stmt, T>::value),
                           only_Decl_or_Stmt_allowed_for_recursive_matching);
    return matchesAncestorOf(ast_type_traits::DynTypedNode::create(Node),
                             Matcher, Builder, MatchMode);
  }

  virtual ASTContext &getASTContext() const = 0;

protected:
  virtual bool matchesChildOf(const ast_type_traits::DynTypedNode &Node,
                              const DynTypedMatcher &Matcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traverse,
                              BindKind Bind) = 0;

  virtual bool matchesDescendantOf(const ast_type_traits::DynTypedNode &Node,
                                   const DynTypedMatcher &Matcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) = 0;

  virtual bool matchesAncestorOf(const ast_type_traits::DynTypedNode &Node,
                                 const DynTypedMatcher &Matcher,
                                 BoundNodesTreeBuilder *Builder,
                                 AncestorMatchMode MatchMode) = 0;
};

/// \brief Converts a \c Matcher<T> to a matcher of desired type \c To by
/// "adapting" a \c To into a \c T.
///
/// The \c ArgumentAdapterT argument specifies how the adaptation is done.
///
/// For example:
///   \c ArgumentAdaptingMatcher<HasMatcher, T>(InnerMatcher);
/// Given that \c InnerMatcher is of type \c Matcher<T>, this returns a matcher
/// that is convertible into any matcher of type \c To by constructing
/// \c HasMatcher<To, T>(InnerMatcher).
///
/// If a matcher does not need knowledge about the inner type, prefer to use
/// PolymorphicMatcherWithParam1.
template <template <typename ToArg, typename FromArg> class ArgumentAdapterT,
          typename T>
class ArgumentAdaptingMatcher {
public:
  explicit ArgumentAdaptingMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  template <typename To>
  operator Matcher<To>() const {
    return Matcher<To>(new ArgumentAdapterT<To, T>(InnerMatcher));
  }

private:
  const Matcher<T> InnerMatcher;
};

/// \brief A simple type-list implementation.
///
/// It is implemented as a flat struct with a maximum number of arguments to
/// simplify compiler error messages.
/// However, it is used as a "linked list" of types.
///
/// Note: If you need to extend for more types, add them as template arguments
///       and to the "typedef TypeList<...> tail" below. Nothing else is needed.
template <typename T1 = void, typename T2 = void, typename T3 = void,
          typename T4 = void, typename T5 = void, typename T6 = void,
          typename T7 = void, typename T8 = void>
struct TypeList {
  /// \brief The first type on the list.
  typedef T1 head;

  /// \brief A sub list with the tail. ie everything but the head.
  ///
  /// This type is used to do recursion. TypeList<>/EmptyTypeList indicates the
  /// end of the list.
  typedef TypeList<T2, T3, T4, T5, T6, T7, T8> tail;

  /// \brief Helper meta-function to determine if some type \c T is present or
  ///   a parent type in the list.
  template <typename T> struct ContainsSuperOf {
    static const bool value = llvm::is_base_of<head, T>::value ||
                              tail::template ContainsSuperOf<T>::value;
  };
};

/// \brief Specialization of ContainsSuperOf for the empty list.
template <> template <typename T> struct TypeList<>::ContainsSuperOf {
  static const bool value = false;
};

/// \brief The empty type list.
typedef TypeList<> EmptyTypeList;

/// \brief A "type list" that contains all types.
///
/// Useful for matchers like \c anything and \c unless.
typedef TypeList<Decl, Stmt, NestedNameSpecifier, NestedNameSpecifierLoc,
        QualType, Type, TypeLoc, CXXCtorInitializer> AllNodeBaseTypes;

/// \brief Helper meta-function to extract the argument out of a function of
///   type void(Arg).
///
/// See AST_POLYMORPHIC_SUPPORTED_TYPES_* for details.
template <class T> struct ExtractFunctionArgMeta;
template <class T> struct ExtractFunctionArgMeta<void(T)> {
  typedef T type;
};

/// \brief A PolymorphicMatcherWithParamN<MatcherT, P1, ..., PN> object can be
/// created from N parameters p1, ..., pN (of type P1, ..., PN) and
/// used as a Matcher<T> where a MatcherT<T, P1, ..., PN>(p1, ..., pN)
/// can be constructed.
///
/// For example:
/// - PolymorphicMatcherWithParam0<IsDefinitionMatcher>()
///   creates an object that can be used as a Matcher<T> for any type T
///   where an IsDefinitionMatcher<T>() can be constructed.
/// - PolymorphicMatcherWithParam1<ValueEqualsMatcher, int>(42)
///   creates an object that can be used as a Matcher<T> for any type T
///   where a ValueEqualsMatcher<T, int>(42) can be constructed.
template <template <typename T> class MatcherT,
          typename ReturnTypesF = void(AllNodeBaseTypes)>
class PolymorphicMatcherWithParam0 {
public:
  typedef typename ExtractFunctionArgMeta<ReturnTypesF>::type ReturnTypes;
  template <typename T>
  operator Matcher<T>() const {
    TOOLING_COMPILE_ASSERT(ReturnTypes::template ContainsSuperOf<T>::value,
                           right_polymorphic_conversion);
    return Matcher<T>(new MatcherT<T>());
  }
};

template <template <typename T, typename P1> class MatcherT,
          typename P1,
          typename ReturnTypesF = void(AllNodeBaseTypes)>
class PolymorphicMatcherWithParam1 {
public:
  explicit PolymorphicMatcherWithParam1(const P1 &Param1)
      : Param1(Param1) {}

  typedef typename ExtractFunctionArgMeta<ReturnTypesF>::type ReturnTypes;

  template <typename T>
  operator Matcher<T>() const {
    TOOLING_COMPILE_ASSERT(ReturnTypes::template ContainsSuperOf<T>::value,
                           right_polymorphic_conversion);
    return Matcher<T>(new MatcherT<T, P1>(Param1));
  }

private:
  const P1 Param1;
};

template <template <typename T, typename P1, typename P2> class MatcherT,
          typename P1, typename P2,
          typename ReturnTypesF = void(AllNodeBaseTypes)>
class PolymorphicMatcherWithParam2 {
public:
  PolymorphicMatcherWithParam2(const P1 &Param1, const P2 &Param2)
      : Param1(Param1), Param2(Param2) {}

  typedef typename ExtractFunctionArgMeta<ReturnTypesF>::type ReturnTypes;

  template <typename T>
  operator Matcher<T>() const {
    TOOLING_COMPILE_ASSERT(ReturnTypes::template ContainsSuperOf<T>::value,
                           right_polymorphic_conversion);
    return Matcher<T>(new MatcherT<T, P1, P2>(Param1, Param2));
  }

private:
  const P1 Param1;
  const P2 Param2;
};

/// \brief Matches any instance of the given NodeType.
///
/// This is useful when a matcher syntactically requires a child matcher,
/// but the context doesn't care. See for example: anything().
///
/// FIXME: Alternatively we could also create a IsAMatcher or something
/// that checks that a dyn_cast is possible. This is purely needed for the
/// difference between calling for example:
///   record()
/// and
///   record(SomeMatcher)
/// In the second case we need the correct type we were dyn_cast'ed to in order
/// to get the right type for the inner matcher. In the first case we don't need
/// that, but we use the type conversion anyway and insert a TrueMatcher.
template <typename T>
class TrueMatcher : public SingleNodeMatcherInterface<T>  {
public:
  virtual bool matchesNode(const T &Node) const {
    return true;
  }
};

/// \brief Provides a MatcherInterface<T> for a Matcher<To> that matches if T is
/// dyn_cast'able into To and the given Matcher<To> matches on the dyn_cast'ed
/// node.
template <typename T, typename To>
class DynCastMatcher : public MatcherInterface<T> {
public:
  explicit DynCastMatcher(const Matcher<To> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    const To *InnerMatchValue = dyn_cast<To>(&Node);
    return InnerMatchValue != NULL &&
      InnerMatcher.matches(*InnerMatchValue, Finder, Builder);
  }

private:
  const Matcher<To> InnerMatcher;
};

/// \brief Matcher<T> that wraps an inner Matcher<T> and binds the matched node
/// to an ID if the inner matcher matches on the node.
template <typename T>
class IdMatcher : public MatcherInterface<T> {
public:
  /// \brief Creates an IdMatcher that binds to 'ID' if 'InnerMatcher' matches
  /// the node.
  IdMatcher(StringRef ID, const Matcher<T> &InnerMatcher)
      : ID(ID), InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    bool Result = InnerMatcher.matches(Node, Finder, Builder);
    if (Result) {
      Builder->setBinding(ID, &Node);
    }
    return Result;
  }

private:
  const std::string ID;
  const Matcher<T> InnerMatcher;
};

/// \brief A Matcher that allows binding the node it matches to an id.
///
/// BindableMatcher provides a \a bind() method that allows binding the
/// matched node to an id if the match was successful.
template <typename T>
class BindableMatcher : public Matcher<T> {
public:
  BindableMatcher(MatcherInterface<T> *Implementation)
    : Matcher<T>(Implementation) {}

  /// \brief Returns a matcher that will bind the matched node on a match.
  ///
  /// The returned matcher is equivalent to this matcher, but will
  /// bind the matched node on a match.
  Matcher<T> bind(StringRef ID) const {
    return Matcher<T>(new IdMatcher<T>(ID, *this));
  }

  /// \brief Makes a copy of this matcher object.
  virtual BindableMatcher<T>* clone() const {
    return new BindableMatcher<T>(*this);
  }

  /// \brief Bind the specified \c ID to the matcher.
  virtual Matcher<T>* tryBind(StringRef ID) const {
    return new Matcher<T>(bind(ID));
  }
};

/// \brief Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches.
///
/// ChildT must be an AST base type.
template <typename T, typename ChildT>
class HasMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ChildT>::value,
                         has_only_accepts_base_type_matcher);
public:
  explicit HasMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesChildOf(
        Node, ChildMatcher, Builder,
        ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
        ASTMatchFinder::BK_First);
  }

 private:
  const Matcher<ChildT> ChildMatcher;
};

/// \brief Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches. ChildT must be an AST base
/// type.
/// As opposed to the HasMatcher, the ForEachMatcher will produce a match
/// for each child that matches.
template <typename T, typename ChildT>
class ForEachMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ChildT>::value,
                         for_each_only_accepts_base_type_matcher);
 public:
  explicit ForEachMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return Finder->matchesChildOf(
      Node, ChildMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_All);
  }

private:
  const Matcher<ChildT> ChildMatcher;
};

/// \brief Matches nodes of type T if the given Matcher<T> does not match.
///
/// Type argument MatcherT is required by PolymorphicMatcherWithParam1
/// but not actually used. It will always be instantiated with a type
/// convertible to Matcher<T>.
template <typename T, typename MatcherT>
class NotMatcher : public MatcherInterface<T> {
public:
  explicit NotMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    // The 'unless' matcher will always discard the result:
    // If the inner matcher doesn't match, unless returns true,
    // but the inner matcher cannot have bound anything.
    // If the inner matcher matches, the result is false, and
    // any possible binding will be discarded.
    // We still need to hand in all the bound nodes up to this
    // point so the inner matcher can depend on bound nodes,
    // and we need to actively discard the bound nodes, otherwise
    // the inner matcher will reset the bound nodes if it doesn't
    // match, but this would be inversed by 'unless'.
    BoundNodesTreeBuilder Discard(*Builder);
    return !InnerMatcher.matches(Node, Finder, &Discard);
  }

private:
  const Matcher<T> InnerMatcher;
};

/// \brief Matches nodes of type T for which both provided matchers match.
///
/// Type arguments MatcherT1 and MatcherT2 are required by
/// PolymorphicMatcherWithParam2 but not actually used. They will
/// always be instantiated with types convertible to Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AllOfMatcher : public MatcherInterface<T> {
public:
  AllOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnerMatcher2(InnerMatcher2) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    // allOf leads to one matcher for each alternative in the first
    // matcher combined with each alternative in the second matcher.
    // Thus, we can reuse the same Builder.
    return InnerMatcher1.matches(Node, Finder, Builder) &&
           InnerMatcher2.matches(Node, Finder, Builder);
  }

private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnerMatcher2;
};

/// \brief Matches nodes of type T for which at least one of the two provided
/// matchers matches.
///
/// Type arguments MatcherT1 and MatcherT2 are
/// required by PolymorphicMatcherWithParam2 but not actually
/// used. They will always be instantiated with types convertible to
/// Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class EachOfMatcher : public MatcherInterface<T> {
public:
  EachOfMatcher(const Matcher<T> &InnerMatcher1,
                const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnerMatcher2(InnerMatcher2) {
  }

  virtual bool matches(const T &Node, ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    BoundNodesTreeBuilder Result;
    BoundNodesTreeBuilder Builder1(*Builder);
    bool Matched1 = InnerMatcher1.matches(Node, Finder, &Builder1);
    if (Matched1)
      Result.addMatch(Builder1);

    BoundNodesTreeBuilder Builder2(*Builder);
    bool Matched2 = InnerMatcher2.matches(Node, Finder, &Builder2);
    if (Matched2)
      Result.addMatch(Builder2);

    *Builder = Result;
    return Matched1 || Matched2;
  }

private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnerMatcher2;
};

/// \brief Matches nodes of type T for which at least one of the two provided
/// matchers matches.
///
/// Type arguments MatcherT1 and MatcherT2 are
/// required by PolymorphicMatcherWithParam2 but not actually
/// used. They will always be instantiated with types convertible to
/// Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AnyOfMatcher : public MatcherInterface<T> {
public:
  AnyOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnerMatcher2(InnerMatcher2) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    BoundNodesTreeBuilder Result = *Builder;
    if (InnerMatcher1.matches(Node, Finder, &Result)) {
      *Builder = Result;
      return true;
    }
    Result = *Builder;
    if (InnerMatcher2.matches(Node, Finder, &Result)) {
      *Builder = Result;
      return true;
    }
    return false;
  }

private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnerMatcher2;
};

/// \brief Creates a Matcher<T> that matches if all inner matchers match.
template<typename T>
BindableMatcher<T> makeAllOfComposite(
    ArrayRef<const Matcher<T> *> InnerMatchers) {
  if (InnerMatchers.empty())
    return BindableMatcher<T>(new TrueMatcher<T>);
  MatcherInterface<T> *InnerMatcher = new TrueMatcher<T>;
  for (int i = InnerMatchers.size() - 1; i >= 0; --i) {
    InnerMatcher = new AllOfMatcher<T, Matcher<T>, Matcher<T> >(
      *InnerMatchers[i], makeMatcher(InnerMatcher));
  }
  return BindableMatcher<T>(InnerMatcher);
}

/// \brief Creates a Matcher<T> that matches if
/// T is dyn_cast'able into InnerT and all inner matchers match.
///
/// Returns BindableMatcher, as matchers that use dyn_cast have
/// the same object both to match on and to run submatchers on,
/// so there is no ambiguity with what gets bound.
template<typename T, typename InnerT>
BindableMatcher<T> makeDynCastAllOfComposite(
    ArrayRef<const Matcher<InnerT> *> InnerMatchers) {
  return BindableMatcher<T>(new DynCastMatcher<T, InnerT>(
    makeAllOfComposite(InnerMatchers)));
}

/// \brief Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
///
/// DescendantT must be an AST base type.
template <typename T, typename DescendantT>
class HasDescendantMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<DescendantT>::value,
                         has_descendant_only_accepts_base_type_matcher);
public:
  explicit HasDescendantMatcher(const Matcher<DescendantT> &DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesDescendantOf(
        Node, DescendantMatcher, Builder, ASTMatchFinder::BK_First);
  }

 private:
  const Matcher<DescendantT> DescendantMatcher;
};

/// \brief Matches nodes of type \c T that have a parent node of type \c ParentT
/// for which the given inner matcher matches.
///
/// \c ParentT must be an AST base type.
template <typename T, typename ParentT>
class HasParentMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ParentT>::value,
                         has_parent_only_accepts_base_type_matcher);
public:
  explicit HasParentMatcher(const Matcher<ParentT> &ParentMatcher)
      : ParentMatcher(ParentMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesAncestorOf(
        Node, ParentMatcher, Builder, ASTMatchFinder::AMM_ParentOnly);
  }

 private:
  const Matcher<ParentT> ParentMatcher;
};

/// \brief Matches nodes of type \c T that have at least one ancestor node of
/// type \c AncestorT for which the given inner matcher matches.
///
/// \c AncestorT must be an AST base type.
template <typename T, typename AncestorT>
class HasAncestorMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<AncestorT>::value,
                         has_ancestor_only_accepts_base_type_matcher);
public:
  explicit HasAncestorMatcher(const Matcher<AncestorT> &AncestorMatcher)
      : AncestorMatcher(AncestorMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesAncestorOf(
        Node, AncestorMatcher, Builder, ASTMatchFinder::AMM_All);
  }

 private:
  const Matcher<AncestorT> AncestorMatcher;
};

/// \brief Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
///
/// DescendantT must be an AST base type.
/// As opposed to HasDescendantMatcher, ForEachDescendantMatcher will match
/// for each descendant node that matches instead of only for the first.
template <typename T, typename DescendantT>
class ForEachDescendantMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<DescendantT>::value,
                         for_each_descendant_only_accepts_base_type_matcher);
 public:
  explicit ForEachDescendantMatcher(
      const Matcher<DescendantT>& DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return Finder->matchesDescendantOf(Node, DescendantMatcher, Builder,
                                       ASTMatchFinder::BK_All);
  }

private:
  const Matcher<DescendantT> DescendantMatcher;
};

/// \brief Matches on nodes that have a getValue() method if getValue() equals
/// the value the ValueEqualsMatcher was constructed with.
template <typename T, typename ValueT>
class ValueEqualsMatcher : public SingleNodeMatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_base_of<CharacterLiteral, T>::value ||
                         llvm::is_base_of<CXXBoolLiteralExpr,
                                          T>::value ||
                         llvm::is_base_of<FloatingLiteral, T>::value ||
                         llvm::is_base_of<IntegerLiteral, T>::value),
                         the_node_must_have_a_getValue_method);
public:
  explicit ValueEqualsMatcher(const ValueT &ExpectedValue)
      : ExpectedValue(ExpectedValue) {}

  virtual bool matchesNode(const T &Node) const {
    return Node.getValue() == ExpectedValue;
  }

private:
  const ValueT ExpectedValue;
};

/// \brief A VariadicDynCastAllOfMatcher<SourceT, TargetT> object is a
/// variadic functor that takes a number of Matcher<TargetT> and returns a
/// Matcher<SourceT> that matches TargetT nodes that are matched by all of the
/// given matchers, if SourceT can be dynamically casted into TargetT.
///
/// For example:
///   const VariadicDynCastAllOfMatcher<
///       Decl, CXXRecordDecl> record;
/// Creates a functor record(...) that creates a Matcher<Decl> given
/// a variable number of arguments of type Matcher<CXXRecordDecl>.
/// The returned matcher matches if the given Decl can by dynamically
/// casted to CXXRecordDecl and all given matchers match.
template <typename SourceT, typename TargetT>
class VariadicDynCastAllOfMatcher
    : public llvm::VariadicFunction<
        BindableMatcher<SourceT>, Matcher<TargetT>,
        makeDynCastAllOfComposite<SourceT, TargetT> > {
public:
  VariadicDynCastAllOfMatcher() {}
};

/// \brief A \c VariadicAllOfMatcher<T> object is a variadic functor that takes
/// a number of \c Matcher<T> and returns a \c Matcher<T> that matches \c T
/// nodes that are matched by all of the given matchers.
///
/// For example:
///   const VariadicAllOfMatcher<NestedNameSpecifier> nestedNameSpecifier;
/// Creates a functor nestedNameSpecifier(...) that creates a
/// \c Matcher<NestedNameSpecifier> given a variable number of arguments of type
/// \c Matcher<NestedNameSpecifier>.
/// The returned matcher matches if all given matchers match.
template <typename T>
class VariadicAllOfMatcher : public llvm::VariadicFunction<
                               BindableMatcher<T>, Matcher<T>,
                               makeAllOfComposite<T> > {
public:
  VariadicAllOfMatcher() {}
};

/// \brief Matches nodes of type \c TLoc for which the inner
/// \c Matcher<T> matches.
template <typename TLoc, typename T>
class LocMatcher : public MatcherInterface<TLoc> {
public:
  explicit LocMatcher(const Matcher<T> &InnerMatcher)
    : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const TLoc &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    if (!Node)
      return false;
    return InnerMatcher.matches(*extract(Node), Finder, Builder);
  }

private:
  const NestedNameSpecifier *extract(const NestedNameSpecifierLoc &Loc) const {
    return Loc.getNestedNameSpecifier();
  }

  const Matcher<T> InnerMatcher;
};

/// \brief Matches \c TypeLocs based on an inner matcher matching a certain
/// \c QualType.
///
/// Used to implement the \c loc() matcher.
class TypeLocTypeMatcher : public MatcherInterface<TypeLoc> {
public:
  explicit TypeLocTypeMatcher(const Matcher<QualType> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const TypeLoc &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    if (!Node)
      return false;
    return InnerMatcher.matches(Node.getType(), Finder, Builder);
  }

private:
  const Matcher<QualType> InnerMatcher;
};

/// \brief Matches nodes of type \c T for which the inner matcher matches on a
/// another node of type \c T that can be reached using a given traverse
/// function.
template <typename T>
class TypeTraverseMatcher : public MatcherInterface<T> {
public:
  explicit TypeTraverseMatcher(const Matcher<QualType> &InnerMatcher,
                               QualType (T::*TraverseFunction)() const)
      : InnerMatcher(InnerMatcher), TraverseFunction(TraverseFunction) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    QualType NextNode = (Node.*TraverseFunction)();
    if (NextNode.isNull())
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<QualType> InnerMatcher;
  QualType (T::*TraverseFunction)() const;
};

/// \brief Matches nodes of type \c T in a ..Loc hierarchy, for which the inner
/// matcher matches on a another node of type \c T that can be reached using a
/// given traverse function.
template <typename T>
class TypeLocTraverseMatcher : public MatcherInterface<T> {
public:
  explicit TypeLocTraverseMatcher(const Matcher<TypeLoc> &InnerMatcher,
                                  TypeLoc (T::*TraverseFunction)() const)
      : InnerMatcher(InnerMatcher), TraverseFunction(TraverseFunction) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    TypeLoc NextNode = (Node.*TraverseFunction)();
    if (!NextNode)
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<TypeLoc> InnerMatcher;
  TypeLoc (T::*TraverseFunction)() const;
};

template <typename T, typename InnerT>
T makeTypeAllOfComposite(ArrayRef<const Matcher<InnerT> *> InnerMatchers) {
  return T(makeAllOfComposite<InnerT>(InnerMatchers));
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang

#endif // LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H
