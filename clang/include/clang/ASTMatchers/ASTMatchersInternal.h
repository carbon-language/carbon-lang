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

#ifndef LLVM_CLANG_ASTMATCHERS_ASTMATCHERSINTERNAL_H
#define LLVM_CLANG_ASTMATCHERS_ASTMATCHERSINTERNAL_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/VariadicFunction.h"
#include "llvm/Support/ManagedStatic.h"
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace ast_matchers {

class BoundNodes;

namespace internal {

/// \brief Internal version of BoundNodes. Holds all the bound nodes.
class BoundNodesMap {
public:
  /// \brief Adds \c Node to the map with key \c ID.
  ///
  /// The node's base type should be in NodeBaseType or it will be unaccessible.
  void addNode(StringRef ID, const ast_type_traits::DynTypedNode& DynNode) {
    NodeMap[ID] = DynNode;
  }

  /// \brief Returns the AST node bound to \c ID.
  ///
  /// Returns NULL if there was no node bound to \c ID or if there is a node but
  /// it cannot be converted to the specified type.
  template <typename T>
  const T *getNodeAs(StringRef ID) const {
    IDToNodeMap::const_iterator It = NodeMap.find(ID);
    if (It == NodeMap.end()) {
      return nullptr;
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

  /// \brief A map from IDs to the bound nodes.
  ///
  /// Note that we're using std::map here, as for memoization:
  /// - we need a comparison operator
  /// - we need an assignment operator
  typedef std::map<std::string, ast_type_traits::DynTypedNode> IDToNodeMap;

  const IDToNodeMap &getMap() const {
    return NodeMap;
  }

  /// \brief Returns \c true if this \c BoundNodesMap can be compared, i.e. all
  /// stored nodes have memoization data.
  bool isComparable() const {
    for (const auto &IDAndNode : NodeMap) {
      if (!IDAndNode.second.getMemoizationData())
        return false;
    }
    return true;
  }

private:
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
  void setBinding(const std::string &Id,
                  const ast_type_traits::DynTypedNode &DynNode) {
    if (Bindings.empty())
      Bindings.push_back(BoundNodesMap());
    for (BoundNodesMap &Binding : Bindings)
      Binding.addNode(Id, DynNode);
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

  /// \brief Returns \c true if this \c BoundNodesTreeBuilder can be compared,
  /// i.e. all stored node maps have memoization data.
  bool isComparable() const {
    for (const BoundNodesMap &NodesMap : Bindings) {
      if (!NodesMap.isComparable())
        return false;
    }
    return true;
  }

private:
  SmallVector<BoundNodesMap, 16> Bindings;
};

class ASTMatchFinder;

/// \brief Generic interface for all matchers.
///
/// Used by the implementation of Matcher<T> and DynTypedMatcher.
/// In general, implement MatcherInterface<T> or SingleNodeMatcherInterface<T>
/// instead.
class DynMatcherInterface : public RefCountedBaseVPTR {
public:
  /// \brief Returns true if \p DynNode can be matched.
  ///
  /// May bind \p DynNode to an ID via \p Builder, or recurse into
  /// the AST via \p Finder.
  virtual bool dynMatches(const ast_type_traits::DynTypedNode &DynNode,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const = 0;
};

/// \brief Generic interface for matchers on an AST node of type T.
///
/// Implement this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public DynMatcherInterface {
public:
  virtual ~MatcherInterface() {}

  /// \brief Returns true if 'Node' can be matched.
  ///
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the AST via 'Finder'.
  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const = 0;

  bool dynMatches(const ast_type_traits::DynTypedNode &DynNode,
                  ASTMatchFinder *Finder,
                  BoundNodesTreeBuilder *Builder) const override {
    return matches(DynNode.getUnchecked<T>(), Finder, Builder);
  }
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
  bool matches(const T &Node,
               ASTMatchFinder * /* Finder */,
               BoundNodesTreeBuilder * /*  Builder */) const override {
    return matchesNode(Node);
  }
};

template <typename> class Matcher;

/// \brief Matcher that works on a \c DynTypedNode.
///
/// It is constructed from a \c Matcher<T> object and redirects most calls to
/// underlying matcher.
/// It checks whether the \c DynTypedNode is convertible into the type of the
/// underlying matcher and then do the actual match on the actual node, or
/// return false if it is not convertible.
class DynTypedMatcher {
public:
  /// \brief Takes ownership of the provided implementation pointer.
  template <typename T>
  DynTypedMatcher(MatcherInterface<T> *Implementation)
      : AllowBind(false),
        SupportedKind(ast_type_traits::ASTNodeKind::getFromNodeKind<T>()),
        RestrictKind(SupportedKind), Implementation(Implementation) {}

  /// \brief Construct from a variadic function.
  enum VariadicOperator {
    /// \brief Matches nodes for which all provided matchers match.
    VO_AllOf,
    /// \brief Matches nodes for which at least one of the provided matchers
    /// matches.
    VO_AnyOf,
    /// \brief Matches nodes for which at least one of the provided matchers
    /// matches, but doesn't stop at the first match.
    VO_EachOf,
    /// \brief Matches nodes that do not match the provided matcher.
    ///
    /// Uses the variadic matcher interface, but fails if
    /// InnerMatchers.size() != 1.
    VO_UnaryNot
  };
  static DynTypedMatcher
  constructVariadic(VariadicOperator Op,
                    std::vector<DynTypedMatcher> InnerMatchers);

  /// \brief Get a "true" matcher for \p NodeKind.
  ///
  /// It only checks that the node is of the right kind.
  static DynTypedMatcher trueMatcher(ast_type_traits::ASTNodeKind NodeKind);

  void setAllowBind(bool AB) { AllowBind = AB; }

  /// \brief Check whether this matcher could ever match a node of kind \p Kind.
  /// \return \c false if this matcher will never match such a node. Otherwise,
  /// return \c true.
  bool canMatchNodesOfKind(ast_type_traits::ASTNodeKind Kind) const;

  /// \brief Return a matcher that points to the same implementation, but
  ///   restricts the node types for \p Kind.
  DynTypedMatcher dynCastTo(const ast_type_traits::ASTNodeKind Kind) const;

  /// \brief Returns true if the matcher matches the given \c DynNode.
  bool matches(const ast_type_traits::DynTypedNode &DynNode,
               ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder) const;

  /// \brief Same as matches(), but skips the kind check.
  ///
  /// It is faster, but the caller must ensure the node is valid for the
  /// kind of this matcher.
  bool matchesNoKindCheck(const ast_type_traits::DynTypedNode &DynNode,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const;

  /// \brief Bind the specified \p ID to the matcher.
  /// \return A new matcher with the \p ID bound to it if this matcher supports
  ///   binding. Otherwise, returns an empty \c Optional<>.
  llvm::Optional<DynTypedMatcher> tryBind(StringRef ID) const;

  /// \brief Returns a unique \p ID for the matcher.
  ///
  /// Casting a Matcher<T> to Matcher<U> creates a matcher that has the
  /// same \c Implementation pointer, but different \c RestrictKind. We need to
  /// include both in the ID to make it unique.
  ///
  /// \c MatcherIDType supports operator< and provides strict weak ordering.
  typedef std::pair<ast_type_traits::ASTNodeKind, uint64_t> MatcherIDType;
  MatcherIDType getID() const {
    /// FIXME: Document the requirements this imposes on matcher
    /// implementations (no new() implementation_ during a Matches()).
    return std::make_pair(RestrictKind,
                          reinterpret_cast<uint64_t>(Implementation.get()));
  }

  /// \brief Returns the type this matcher works on.
  ///
  /// \c matches() will always return false unless the node passed is of this
  /// or a derived type.
  ast_type_traits::ASTNodeKind getSupportedKind() const {
    return SupportedKind;
  }

  /// \brief Returns \c true if the passed \c DynTypedMatcher can be converted
  ///   to a \c Matcher<T>.
  ///
  /// This method verifies that the underlying matcher in \c Other can process
  /// nodes of types T.
  template <typename T> bool canConvertTo() const {
    return canConvertTo(ast_type_traits::ASTNodeKind::getFromNodeKind<T>());
  }
  bool canConvertTo(ast_type_traits::ASTNodeKind To) const;

  /// \brief Construct a \c Matcher<T> interface around the dynamic matcher.
  ///
  /// This method asserts that \c canConvertTo() is \c true. Callers
  /// should call \c canConvertTo() first to make sure that \c this is
  /// compatible with T.
  template <typename T> Matcher<T> convertTo() const {
    assert(canConvertTo<T>());
    return unconditionalConvertTo<T>();
  }

  /// \brief Same as \c convertTo(), but does not check that the underlying
  ///   matcher can handle a value of T.
  ///
  /// If it is not compatible, then this matcher will never match anything.
  template <typename T> Matcher<T> unconditionalConvertTo() const;

private:
 DynTypedMatcher(ast_type_traits::ASTNodeKind SupportedKind,
                 ast_type_traits::ASTNodeKind RestrictKind,
                 IntrusiveRefCntPtr<DynMatcherInterface> Implementation)
     : AllowBind(false),
       SupportedKind(SupportedKind),
       RestrictKind(RestrictKind),
       Implementation(std::move(Implementation)) {}

  bool AllowBind;
  ast_type_traits::ASTNodeKind SupportedKind;
  /// \brief A potentially stricter node kind.
  ///
  /// It allows to perform implicit and dynamic cast of matchers without
  /// needing to change \c Implementation.
  ast_type_traits::ASTNodeKind RestrictKind;
  IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
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
class Matcher {
public:
  /// \brief Takes ownership of the provided implementation pointer.
  explicit Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}

  /// \brief Implicitly converts \c Other to a Matcher<T>.
  ///
  /// Requires \c T to be derived from \c From.
  template <typename From>
  Matcher(const Matcher<From> &Other,
          typename std::enable_if<std::is_base_of<From, T>::value &&
                                  !std::is_same<From, T>::value>::type * = 0)
      : Implementation(restrictMatcher(Other.Implementation)) {
    assert(Implementation.getSupportedKind().isSame(
        ast_type_traits::ASTNodeKind::getFromNodeKind<T>()));
  }

  /// \brief Implicitly converts \c Matcher<Type> to \c Matcher<QualType>.
  ///
  /// The resulting matcher is not strict, i.e. ignores qualifiers.
  template <typename TypeT>
  Matcher(const Matcher<TypeT> &Other,
          typename std::enable_if<
            std::is_same<T, QualType>::value &&
            std::is_same<TypeT, Type>::value>::type* = 0)
      : Implementation(new TypeToQualType<TypeT>(Other)) {}

  /// \brief Convert \c this into a \c Matcher<T> by applying dyn_cast<> to the
  /// argument.
  /// \c To must be a base class of \c T.
  template <typename To>
  Matcher<To> dynCastTo() const {
    static_assert(std::is_base_of<To, T>::value, "Invalid dynCast call.");
    return Matcher<To>(Implementation);
  }

  /// \brief Forwards the call to the underlying MatcherInterface<T> pointer.
  bool matches(const T &Node,
               ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const {
    return Implementation.matches(ast_type_traits::DynTypedNode::create(Node),
                                  Finder, Builder);
  }

  /// \brief Returns an ID that uniquely identifies the matcher.
  DynTypedMatcher::MatcherIDType getID() const {
    return Implementation.getID();
  }

  /// \brief Extract the dynamic matcher.
  ///
  /// The returned matcher keeps the same restrictions as \c this and remembers
  /// that it is meant to support nodes of type \c T.
  operator DynTypedMatcher() const { return Implementation; }

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

    bool matches(const QualType &Node, ASTMatchFinder *Finder,
                 BoundNodesTreeBuilder *Builder) const override {
      if (Node.isNull())
        return false;
      return InnerMatcher.matches(*Node, Finder, Builder);
    }
   private:
    const Matcher<TypeT> InnerMatcher;
  };

private:
  // For Matcher<T> <=> Matcher<U> conversions.
  template <typename U> friend class Matcher;
  // For DynTypedMatcher::unconditionalConvertTo<T>.
  friend class DynTypedMatcher;

  static DynTypedMatcher restrictMatcher(const DynTypedMatcher &Other) {
    return Other.dynCastTo(ast_type_traits::ASTNodeKind::getFromNodeKind<T>());
  }

  explicit Matcher(const DynTypedMatcher &Implementation)
      : Implementation(restrictMatcher(Implementation)) {
    assert(this->Implementation.getSupportedKind()
               .isSame(ast_type_traits::ASTNodeKind::getFromNodeKind<T>()));
  }

  DynTypedMatcher Implementation;
};  // class Matcher

/// \brief A convenient helper for creating a Matcher<T> without specifying
/// the template type argument.
template <typename T>
inline Matcher<T> makeMatcher(MatcherInterface<T> *Implementation) {
  return Matcher<T>(Implementation);
}

/// \brief Specialization of the conversion functions for QualType.
///
/// This specialization provides the Matcher<Type>->Matcher<QualType>
/// conversion that the static API does.
template <>
inline Matcher<QualType> DynTypedMatcher::convertTo<QualType>() const {
  assert(canConvertTo<QualType>());
  const ast_type_traits::ASTNodeKind SourceKind = getSupportedKind();
  if (SourceKind.isSame(
          ast_type_traits::ASTNodeKind::getFromNodeKind<Type>())) {
    // We support implicit conversion from Matcher<Type> to Matcher<QualType>
    return unconditionalConvertTo<Type>();
  }
  return unconditionalConvertTo<QualType>();
}

/// \brief Finds the first node in a range that matches the given matcher.
template <typename MatcherT, typename IteratorT>
bool matchesFirstInRange(const MatcherT &Matcher, IteratorT Start,
                         IteratorT End, ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) {
  for (IteratorT I = Start; I != End; ++I) {
    BoundNodesTreeBuilder Result(*Builder);
    if (Matcher.matches(*I, Finder, &Result)) {
      *Builder = std::move(Result);
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
      *Builder = std::move(Result);
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

  static bool const value = sizeof(f<Derived>(nullptr)) == 2;
};

/// \brief Matches overloaded operators with a specific name.
///
/// The type argument ArgT is not used by this matcher but is used by
/// PolymorphicMatcherWithParam1 and should be StringRef.
template <typename T, typename ArgT>
class HasOverloadedOperatorNameMatcher : public SingleNodeMatcherInterface<T> {
  static_assert(std::is_same<T, CXXOperatorCallExpr>::value ||
                std::is_base_of<FunctionDecl, T>::value,
                "unsupported class for matcher");
  static_assert(std::is_same<ArgT, StringRef>::value,
                "argument type must be StringRef");

public:
  explicit HasOverloadedOperatorNameMatcher(const StringRef Name)
      : SingleNodeMatcherInterface<T>(), Name(Name) {}

  bool matchesNode(const T &Node) const override {
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
  bool matchesSpecialized(const FunctionDecl &Node) const {
    return Node.isOverloadedOperator() &&
           getOperatorSpelling(Node.getOverloadedOperator()) == Name;
  }

  std::string Name;
};

/// \brief Matches named declarations with a specific name.
///
/// See \c hasName() in ASTMatchers.h for details.
class HasNameMatcher : public SingleNodeMatcherInterface<NamedDecl> {
 public:
  explicit HasNameMatcher(StringRef Name);

  bool matchesNode(const NamedDecl &Node) const override;

 private:
  /// \brief Unqualified match routine.
  ///
  /// It is much faster than the full match, but it only works for unqualified
  /// matches.
  bool matchesNodeUnqualified(const NamedDecl &Node) const;

  /// \brief Full match routine
  ///
  /// It generates the fully qualified name of the declaration (which is
  /// expensive) before trying to match.
  /// It is slower but simple and works on all cases.
  bool matchesNodeFull(const NamedDecl &Node) const;

  const bool UseUnqualifiedMatch;
  const std::string Name;
};

/// \brief Matches declarations for QualType and CallExpr.
///
/// Type argument DeclMatcherT is required by PolymorphicMatcherWithParam1 but
/// not actually used.
template <typename T, typename DeclMatcherT>
class HasDeclarationMatcher : public MatcherInterface<T> {
  static_assert(std::is_same<DeclMatcherT, Matcher<Decl>>::value,
                "instantiated with wrong types");

public:
  explicit HasDeclarationMatcher(const Matcher<Decl> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
    return matchesSpecialized(Node, Finder, Builder);
  }

private:
  /// \brief If getDecl exists as a member of U, returns whether the inner
  /// matcher matches Node.getDecl().
  template <typename U>
  bool matchesSpecialized(
      const U &Node, ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder,
      typename std::enable_if<has_getDecl<U>::value, int>::type = 0) const {
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
    return Node != nullptr && InnerMatcher.matches(*Node, Finder, Builder);
  }

  const Matcher<Decl> InnerMatcher;
};

/// \brief IsBaseType<T>::value is true if T is a "base" type in the AST
/// node class hierarchies.
template <typename T>
struct IsBaseType {
  static const bool value =
      std::is_same<T, Decl>::value ||
      std::is_same<T, Stmt>::value ||
      std::is_same<T, QualType>::value ||
      std::is_same<T, Type>::value ||
      std::is_same<T, TypeLoc>::value ||
      std::is_same<T, NestedNameSpecifier>::value ||
      std::is_same<T, NestedNameSpecifierLoc>::value ||
      std::is_same<T, CXXCtorInitializer>::value;
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
    static_assert(std::is_base_of<Decl, T>::value ||
                  std::is_base_of<Stmt, T>::value ||
                  std::is_base_of<NestedNameSpecifier, T>::value ||
                  std::is_base_of<NestedNameSpecifierLoc, T>::value ||
                  std::is_base_of<TypeLoc, T>::value ||
                  std::is_base_of<QualType, T>::value,
                  "unsupported type for recursive matching");
   return matchesChildOf(ast_type_traits::DynTypedNode::create(Node),
                          Matcher, Builder, Traverse, Bind);
  }

  template <typename T>
  bool matchesDescendantOf(const T &Node,
                           const DynTypedMatcher &Matcher,
                           BoundNodesTreeBuilder *Builder,
                           BindKind Bind) {
    static_assert(std::is_base_of<Decl, T>::value ||
                  std::is_base_of<Stmt, T>::value ||
                  std::is_base_of<NestedNameSpecifier, T>::value ||
                  std::is_base_of<NestedNameSpecifierLoc, T>::value ||
                  std::is_base_of<TypeLoc, T>::value ||
                  std::is_base_of<QualType, T>::value,
                  "unsupported type for recursive matching");
    return matchesDescendantOf(ast_type_traits::DynTypedNode::create(Node),
                               Matcher, Builder, Bind);
  }

  // FIXME: Implement support for BindKind.
  template <typename T>
  bool matchesAncestorOf(const T &Node,
                         const DynTypedMatcher &Matcher,
                         BoundNodesTreeBuilder *Builder,
                         AncestorMatchMode MatchMode) {
    static_assert(std::is_base_of<Decl, T>::value ||
                  std::is_base_of<Stmt, T>::value,
                  "only Decl or Stmt allowed for recursive matching");
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

/// \brief A type-list implementation.
///
/// A list is declared as a tree of type list nodes, where the leafs are the
/// types.
/// However, it is used as a "linked list" of types, by using the ::head and
/// ::tail typedefs.
/// Each node supports up to 4 children (instead of just 2) to reduce the
/// nesting required by large lists.
template <typename T1 = void, typename T2 = void, typename T3 = void,
          typename T4 = void>
struct TypeList {
  /// \brief Implementation detail. Combined with the specializations below,
  ///   this typedef allows for flattening of nested structures.
  typedef TypeList<T1, T2, T3, T4> self;

  /// \brief The first type on the list.
  typedef T1 head;

  /// \brief A sublist with the tail. ie everything but the head.
  ///
  /// This type is used to do recursion. TypeList<>/EmptyTypeList indicates the
  /// end of the list.
  typedef typename TypeList<T2, T3, T4>::self tail;
};

/// \brief Template specialization to allow nested lists.
///
/// First element is a typelist. Pop its first element.
template <typename Sub1, typename Sub2, typename Sub3, typename Sub4,
          typename T2, typename T3, typename T4>
struct TypeList<TypeList<Sub1, Sub2, Sub3, Sub4>, T2, T3,
                T4> : public TypeList<Sub1,
                                      typename TypeList<Sub2, Sub3, Sub4>::self,
                                      typename TypeList<T2, T3, T4>::self> {};

/// \brief Template specialization to allow nested lists.
///
/// First element is an empty typelist. Skip it.
template <typename T2, typename T3, typename T4>
struct TypeList<TypeList<>, T2, T3, T4> : public TypeList<T2, T3, T4> {
};

/// \brief The empty type list.
typedef TypeList<> EmptyTypeList;

/// \brief Helper meta-function to determine if some type \c T is present or
///   a parent type in the list.
template <typename AnyTypeList, typename T>
struct TypeListContainsSuperOf {
  static const bool value =
      std::is_base_of<typename AnyTypeList::head, T>::value ||
      TypeListContainsSuperOf<typename AnyTypeList::tail, T>::value;
};
template <typename T>
struct TypeListContainsSuperOf<EmptyTypeList, T> {
  static const bool value = false;
};

/// \brief A "type list" that contains all types.
///
/// Useful for matchers like \c anything and \c unless.
typedef TypeList<
    TypeList<Decl, Stmt, NestedNameSpecifier, NestedNameSpecifierLoc>,
    TypeList<QualType, Type, TypeLoc, CXXCtorInitializer> > AllNodeBaseTypes;

/// \brief Helper meta-function to extract the argument out of a function of
///   type void(Arg).
///
/// See AST_POLYMORPHIC_SUPPORTED_TYPES_* for details.
template <class T> struct ExtractFunctionArgMeta;
template <class T> struct ExtractFunctionArgMeta<void(T)> {
  typedef T type;
};

/// \brief Default type lists for ArgumentAdaptingMatcher matchers.
typedef AllNodeBaseTypes AdaptativeDefaultFromTypes;
typedef TypeList<TypeList<Decl, Stmt, NestedNameSpecifier>,
                 TypeList<NestedNameSpecifierLoc, TypeLoc, QualType> >
AdaptativeDefaultToTypes;

/// \brief All types that are supported by HasDeclarationMatcher above.
typedef TypeList<TypeList<CallExpr, CXXConstructExpr, DeclRefExpr, EnumType>,
                 TypeList<InjectedClassNameType, LabelStmt, MemberExpr>,
                 TypeList<QualType, RecordType, TagType>,
                 TypeList<TemplateSpecializationType, TemplateTypeParmType,
                          TypedefType, UnresolvedUsingType> >
HasDeclarationSupportedTypes;

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
          typename FromTypes = AdaptativeDefaultFromTypes,
          typename ToTypes = AdaptativeDefaultToTypes>
struct ArgumentAdaptingMatcherFunc {
  template <typename T> class Adaptor {
  public:
    explicit Adaptor(const Matcher<T> &InnerMatcher)
        : InnerMatcher(InnerMatcher) {}

    typedef ToTypes ReturnTypes;

    template <typename To> operator Matcher<To>() const {
      return Matcher<To>(new ArgumentAdapterT<To, T>(InnerMatcher));
    }

  private:
    const Matcher<T> InnerMatcher;
  };

  template <typename T>
  static Adaptor<T> create(const Matcher<T> &InnerMatcher) {
    return Adaptor<T>(InnerMatcher);
  }

  template <typename T>
  Adaptor<T> operator()(const Matcher<T> &InnerMatcher) const {
    return create(InnerMatcher);
  }
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
    static_assert(TypeListContainsSuperOf<ReturnTypes, T>::value,
                  "right polymorphic conversion");
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
    static_assert(TypeListContainsSuperOf<ReturnTypes, T>::value,
                  "right polymorphic conversion");
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
    static_assert(TypeListContainsSuperOf<ReturnTypes, T>::value,
                  "right polymorphic conversion");
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
class TrueMatcher {
 public:
  typedef AllNodeBaseTypes ReturnTypes;

  template <typename T>
  operator Matcher<T>() const {
    return DynTypedMatcher::trueMatcher(
               ast_type_traits::ASTNodeKind::getFromNodeKind<T>())
        .template unconditionalConvertTo<T>();
  }
};

/// \brief A Matcher that allows binding the node it matches to an id.
///
/// BindableMatcher provides a \a bind() method that allows binding the
/// matched node to an id if the match was successful.
template <typename T>
class BindableMatcher : public Matcher<T> {
public:
  explicit BindableMatcher(const Matcher<T> &M) : Matcher<T>(M) {}
  explicit BindableMatcher(MatcherInterface<T> *Implementation)
    : Matcher<T>(Implementation) {}

  /// \brief Returns a matcher that will bind the matched node on a match.
  ///
  /// The returned matcher is equivalent to this matcher, but will
  /// bind the matched node on a match.
  Matcher<T> bind(StringRef ID) const {
    return DynTypedMatcher(*this)
        .tryBind(ID)
        ->template unconditionalConvertTo<T>();
  }

  /// \brief Same as Matcher<T>'s conversion operator, but enables binding on
  /// the returned matcher.
  operator DynTypedMatcher() const {
    DynTypedMatcher Result = static_cast<const Matcher<T>&>(*this);
    Result.setAllowBind(true);
    return Result;
  }
};

/// \brief Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches.
///
/// ChildT must be an AST base type.
template <typename T, typename ChildT>
class HasMatcher : public MatcherInterface<T> {
  static_assert(IsBaseType<ChildT>::value,
                "has only accepts base type matcher");

public:
  explicit HasMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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
  static_assert(IsBaseType<ChildT>::value,
                "for each only accepts base type matcher");

 public:
  explicit ForEachMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  bool matches(const T& Node, ASTMatchFinder* Finder,
               BoundNodesTreeBuilder* Builder) const override {
    return Finder->matchesChildOf(
      Node, ChildMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_All);
  }

private:
  const Matcher<ChildT> ChildMatcher;
};

/// \brief VariadicOperatorMatcher related types.
/// @{

/// \brief "No argument" placeholder to use as template paratemers.
struct VariadicOperatorNoArg {};

/// \brief Polymorphic matcher object that uses a \c
/// DynTypedMatcher::VariadicOperator operator.
///
/// Input matchers can have any type (including other polymorphic matcher
/// types), and the actual Matcher<T> is generated on demand with an implicit
/// coversion operator.
template <typename P1, typename P2 = VariadicOperatorNoArg,
          typename P3 = VariadicOperatorNoArg,
          typename P4 = VariadicOperatorNoArg,
          typename P5 = VariadicOperatorNoArg,
          typename P6 = VariadicOperatorNoArg,
          typename P7 = VariadicOperatorNoArg,
          typename P8 = VariadicOperatorNoArg,
          typename P9 = VariadicOperatorNoArg>
class VariadicOperatorMatcher {
public:
  VariadicOperatorMatcher(DynTypedMatcher::VariadicOperator Op,
                          const P1 &Param1,
                          const P2 &Param2 = VariadicOperatorNoArg(),
                          const P3 &Param3 = VariadicOperatorNoArg(),
                          const P4 &Param4 = VariadicOperatorNoArg(),
                          const P5 &Param5 = VariadicOperatorNoArg(),
                          const P6 &Param6 = VariadicOperatorNoArg(),
                          const P7 &Param7 = VariadicOperatorNoArg(),
                          const P8 &Param8 = VariadicOperatorNoArg(),
                          const P9 &Param9 = VariadicOperatorNoArg())
      : Op(Op), Param1(Param1), Param2(Param2), Param3(Param3),
        Param4(Param4), Param5(Param5), Param6(Param6), Param7(Param7),
        Param8(Param8), Param9(Param9) {}

  template <typename T> operator Matcher<T>() const {
    std::vector<DynTypedMatcher> Matchers;
    addMatcher<T>(Param1, Matchers);
    addMatcher<T>(Param2, Matchers);
    addMatcher<T>(Param3, Matchers);
    addMatcher<T>(Param4, Matchers);
    addMatcher<T>(Param5, Matchers);
    addMatcher<T>(Param6, Matchers);
    addMatcher<T>(Param7, Matchers);
    addMatcher<T>(Param8, Matchers);
    addMatcher<T>(Param9, Matchers);
    return DynTypedMatcher::constructVariadic(Op, std::move(Matchers))
        .template unconditionalConvertTo<T>();
  }

private:
  template <typename T>
  static void addMatcher(const Matcher<T> &M,
                         std::vector<DynTypedMatcher> &Matchers) {
    Matchers.push_back(M);
  }

  /// \brief Overload to ignore \c VariadicOperatorNoArg arguments.
  template <typename T>
  static void addMatcher(VariadicOperatorNoArg,
                         std::vector<DynTypedMatcher> &Matchers) {}

  const DynTypedMatcher::VariadicOperator Op;
  const P1 Param1;
  const P2 Param2;
  const P3 Param3;
  const P4 Param4;
  const P5 Param5;
  const P6 Param6;
  const P7 Param7;
  const P8 Param8;
  const P9 Param9;
};

/// \brief Overloaded function object to generate VariadicOperatorMatcher
///   objects from arbitrary matchers.
///
/// It supports 1-9 argument overloaded operator(). More can be added if needed.
template <unsigned MinCount, unsigned MaxCount>
struct VariadicOperatorMatcherFunc {
  DynTypedMatcher::VariadicOperator Op;

  template <unsigned Count, typename T>
  struct EnableIfValidArity
      : public std::enable_if<MinCount <= Count && Count <= MaxCount, T> {};

  template <typename M1>
  typename EnableIfValidArity<1, VariadicOperatorMatcher<M1> >::type
  operator()(const M1 &P1) const {
    return VariadicOperatorMatcher<M1>(Op, P1);
  }
  template <typename M1, typename M2>
  typename EnableIfValidArity<2, VariadicOperatorMatcher<M1, M2> >::type
  operator()(const M1 &P1, const M2 &P2) const {
    return VariadicOperatorMatcher<M1, M2>(Op, P1, P2);
  }
  template <typename M1, typename M2, typename M3>
  typename EnableIfValidArity<3, VariadicOperatorMatcher<M1, M2, M3> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3) const {
    return VariadicOperatorMatcher<M1, M2, M3>(Op, P1, P2, P3);
  }
  template <typename M1, typename M2, typename M3, typename M4>
  typename EnableIfValidArity<4, VariadicOperatorMatcher<M1, M2, M3, M4> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4>(Op, P1, P2, P3, P4);
  }
  template <typename M1, typename M2, typename M3, typename M4, typename M5>
  typename EnableIfValidArity<
      5, VariadicOperatorMatcher<M1, M2, M3, M4, M5> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4,
             const M5 &P5) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4, M5>(Op, P1, P2, P3, P4, P5);
  }
  template <typename M1, typename M2, typename M3, typename M4, typename M5,
            typename M6>
  typename EnableIfValidArity<
      6, VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4,
             const M5 &P5, const M6 &P6) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6>(
        Op, P1, P2, P3, P4, P5, P6);
  }
  template <typename M1, typename M2, typename M3, typename M4, typename M5,
            typename M6, typename M7>
  typename EnableIfValidArity<
      7, VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4,
             const M5 &P5, const M6 &P6, const M7 &P7) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7>(
        Op, P1, P2, P3, P4, P5, P6, P7);
  }
  template <typename M1, typename M2, typename M3, typename M4, typename M5,
            typename M6, typename M7, typename M8>
  typename EnableIfValidArity<
      8, VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7, M8> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4,
             const M5 &P5, const M6 &P6, const M7 &P7, const M8 &P8) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7, M8>(
        Op, P1, P2, P3, P4, P5, P6, P7, P8);
  }
  template <typename M1, typename M2, typename M3, typename M4, typename M5,
            typename M6, typename M7, typename M8, typename M9>
  typename EnableIfValidArity<
      9, VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7, M8, M9> >::type
  operator()(const M1 &P1, const M2 &P2, const M3 &P3, const M4 &P4,
             const M5 &P5, const M6 &P6, const M7 &P7, const M8 &P8,
             const M9 &P9) const {
    return VariadicOperatorMatcher<M1, M2, M3, M4, M5, M6, M7, M8, M9>(
        Op, P1, P2, P3, P4, P5, P6, P7, P8, P9);
  }
};

/// @}

template <typename T>
inline Matcher<T> DynTypedMatcher::unconditionalConvertTo() const {
  return Matcher<T>(*this);
}

/// \brief Creates a Matcher<T> that matches if all inner matchers match.
template<typename T>
BindableMatcher<T> makeAllOfComposite(
    ArrayRef<const Matcher<T> *> InnerMatchers) {
  // For the size() == 0 case, we return a "true" matcher.
  if (InnerMatchers.size() == 0) {
    return BindableMatcher<T>(TrueMatcher());
  }
  // For the size() == 1 case, we simply return that one matcher.
  // No need to wrap it in a variadic operation.
  if (InnerMatchers.size() == 1) {
    return BindableMatcher<T>(*InnerMatchers[0]);
  }

  std::vector<DynTypedMatcher> DynMatchers;
  DynMatchers.reserve(InnerMatchers.size());
  for (const auto *InnerMatcher : InnerMatchers) {
    DynMatchers.push_back(*InnerMatcher);
  }
  return BindableMatcher<T>(
      DynTypedMatcher::constructVariadic(DynTypedMatcher::VO_AllOf,
                                         std::move(DynMatchers))
          .template unconditionalConvertTo<T>());
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
  return BindableMatcher<T>(
      makeAllOfComposite(InnerMatchers).template dynCastTo<T>());
}

/// \brief Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
///
/// DescendantT must be an AST base type.
template <typename T, typename DescendantT>
class HasDescendantMatcher : public MatcherInterface<T> {
  static_assert(IsBaseType<DescendantT>::value,
                "has descendant only accepts base type matcher");

public:
  explicit HasDescendantMatcher(const Matcher<DescendantT> &DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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
  static_assert(IsBaseType<ParentT>::value,
                "has parent only accepts base type matcher");

public:
  explicit HasParentMatcher(const Matcher<ParentT> &ParentMatcher)
      : ParentMatcher(ParentMatcher) {}

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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
  static_assert(IsBaseType<AncestorT>::value,
                "has ancestor only accepts base type matcher");

public:
  explicit HasAncestorMatcher(const Matcher<AncestorT> &AncestorMatcher)
      : AncestorMatcher(AncestorMatcher) {}

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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
  static_assert(IsBaseType<DescendantT>::value,
                "for each descendant only accepts base type matcher");

 public:
  explicit ForEachDescendantMatcher(
      const Matcher<DescendantT>& DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  bool matches(const T& Node, ASTMatchFinder* Finder,
               BoundNodesTreeBuilder* Builder) const override {
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
  static_assert(std::is_base_of<CharacterLiteral, T>::value ||
                std::is_base_of<CXXBoolLiteralExpr, T>::value ||
                std::is_base_of<FloatingLiteral, T>::value ||
                std::is_base_of<IntegerLiteral, T>::value,
                "the node must have a getValue method");

public:
  explicit ValueEqualsMatcher(const ValueT &ExpectedValue)
      : ExpectedValue(ExpectedValue) {}

  bool matchesNode(const T &Node) const override {
    return Node.getValue() == ExpectedValue;
  }

private:
  const ValueT ExpectedValue;
};

/// \brief Template specializations to easily write matchers for floating point
/// literals.
template <>
inline bool ValueEqualsMatcher<FloatingLiteral, double>::matchesNode(
    const FloatingLiteral &Node) const {
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEsingle)
    return Node.getValue().convertToFloat() == ExpectedValue;
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEdouble)
    return Node.getValue().convertToDouble() == ExpectedValue;
  return false;
}
template <>
inline bool ValueEqualsMatcher<FloatingLiteral, float>::matchesNode(
    const FloatingLiteral &Node) const {
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEsingle)
    return Node.getValue().convertToFloat() == ExpectedValue;
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEdouble)
    return Node.getValue().convertToDouble() == ExpectedValue;
  return false;
}
template <>
inline bool ValueEqualsMatcher<FloatingLiteral, llvm::APFloat>::matchesNode(
    const FloatingLiteral &Node) const {
  return ExpectedValue.compare(Node.getValue()) == llvm::APFloat::cmpEqual;
}

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

  bool matches(const TLoc &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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

  bool matches(const TypeLoc &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
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

  bool matches(const T &Node, ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const override {
    TypeLoc NextNode = (Node.*TraverseFunction)();
    if (!NextNode)
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<TypeLoc> InnerMatcher;
  TypeLoc (T::*TraverseFunction)() const;
};

/// \brief Converts a \c Matcher<InnerT> to a \c Matcher<OuterT>, where
/// \c OuterT is any type that is supported by \c Getter.
///
/// \code Getter<OuterT>::value() \endcode returns a
/// \code InnerTBase (OuterT::*)() \endcode, which is used to adapt a \c OuterT
/// object into a \c InnerT
template <typename InnerTBase,
          template <typename OuterT> class Getter,
          template <typename OuterT> class MatcherImpl,
          typename ReturnTypesF>
class TypeTraversePolymorphicMatcher {
private:
  typedef TypeTraversePolymorphicMatcher<InnerTBase, Getter, MatcherImpl,
                                         ReturnTypesF> Self;
  static Self create(ArrayRef<const Matcher<InnerTBase> *> InnerMatchers);

public:
  typedef typename ExtractFunctionArgMeta<ReturnTypesF>::type ReturnTypes;

  explicit TypeTraversePolymorphicMatcher(
      ArrayRef<const Matcher<InnerTBase> *> InnerMatchers)
      : InnerMatcher(makeAllOfComposite(InnerMatchers)) {}

  template <typename OuterT> operator Matcher<OuterT>() const {
    return Matcher<OuterT>(
        new MatcherImpl<OuterT>(InnerMatcher, Getter<OuterT>::value()));
  }

  struct Func : public llvm::VariadicFunction<Self, Matcher<InnerTBase>,
                                              &Self::create> {
    Func() {}
  };

private:
  const Matcher<InnerTBase> InnerMatcher;
};

/// \brief A simple memoizer of T(*)() functions.
///
/// It will call the passed 'Func' template parameter at most once.
/// Used to support AST_MATCHER_FUNCTION() macro.
template <typename Matcher, Matcher (*Func)()> class MemoizedMatcher {
  struct Wrapper {
    Wrapper() : M(Func()) {}
    Matcher M;
  };

public:
  static const Matcher &getInstance() {
    static llvm::ManagedStatic<Wrapper> Instance;
    return Instance->M;
  }
};

// Define the create() method out of line to silence a GCC warning about
// the struct "Func" having greater visibility than its base, which comes from
// using the flag -fvisibility-inlines-hidden.
template <typename InnerTBase, template <typename OuterT> class Getter,
          template <typename OuterT> class MatcherImpl, typename ReturnTypesF>
TypeTraversePolymorphicMatcher<InnerTBase, Getter, MatcherImpl, ReturnTypesF>
TypeTraversePolymorphicMatcher<
    InnerTBase, Getter, MatcherImpl,
    ReturnTypesF>::create(ArrayRef<const Matcher<InnerTBase> *> InnerMatchers) {
  return Self(InnerMatchers);
}

// FIXME: unify ClassTemplateSpecializationDecl and TemplateSpecializationType's
// APIs for accessing the template argument list.
inline ArrayRef<TemplateArgument>
getTemplateSpecializationArgs(const ClassTemplateSpecializationDecl &D) {
  return D.getTemplateArgs().asArray();
}

inline ArrayRef<TemplateArgument>
getTemplateSpecializationArgs(const TemplateSpecializationType &T) {
  return llvm::makeArrayRef(T.getArgs(), T.getNumArgs());
}

struct NotEqualsBoundNodePredicate {
  bool operator()(const internal::BoundNodesMap &Nodes) const {
    return Nodes.getNode(ID) != Node;
  }
  std::string ID;
  ast_type_traits::DynTypedNode Node;
};

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang

#endif
