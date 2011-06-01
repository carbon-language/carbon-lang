//===--- ASTMatchers.h - Structural query framework -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a framework of AST matchers that can be used to express
//  structural queries on the AST representing C++ code.
//
//  The general idea is to construct a matcher expression that describes a
//  subtree match on the AST. Next, a callback that is executed every time the
//  expression matches is registered, and the matcher is run over the AST of
//  some code. Matched subexpressions can be bound to string IDs and easily
//  be accessed from the registered callback. The callback can than use the
//  AST nodes that the subexpressions matched on to output information about
//  the match or construct changes that can be applied to the code.
//
//  Example:
//  class HandleMatch : public clang::tooling::MatchFinder::MatchCallback {
//   public:
//    virtual void Run(const clang::tooling::MatchFinder::MatchResult &Result) {
//      const clang::CXXRecordDecl *Class =
//          Result.Nodes.GetDeclAs<clang::CXXRecordDecl>("id");
//      ...
//    }
//  };
//
//  int main(int argc, char **argv) {
//    ClangTool Tool(argc, argv);
//    MatchFinder finder;
//    finder.AddMatcher(Id("id", Class(HasName("::a_namespace::AClass"))),
//                      new HandleMatch);
//    return Tool.Run(finder.NewFrontendActionFactory());
//  }
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_AST_MATCHERS_H
#define LLVM_CLANG_TOOLING_AST_MATCHERS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Tooling/VariadicFunction.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"
#include <assert.h>
#include <stdint.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

/// FIXME: Move into the llvm support library.
template <bool> struct CompileAssert {};
#define COMPILE_ASSERT(Expr, Msg) \
  typedef CompileAssert<(bool(Expr))> Msg[bool(Expr) ? 1 : -1]

namespace clang {

class FrontendAction;
class SourceManager;

namespace tooling {

class FrontendActionFactory;
class BoundNodesBuilder;

/// Contains a mapping from IDs to nodes bound to those IDs and provides
/// convenient access to those nodes.
class BoundNodes {
 public:
  BoundNodes() {}

  /// Create BoundNodes from a pre-filled map of bindings.
  BoundNodes(const std::map<std::string, const clang::Decl*> &DeclBindings,
             const std::map<std::string, const clang::Stmt*> &StmtBindings)
      : DeclBindings(DeclBindings), StmtBindings(StmtBindings) {}

  /// Returns the node bound to the specified id if the id was bound to a node
  /// and that node can be converted into the specified type. Returns NULL
  /// otherwise.
  /// FIXME: We'll need one of those for every base type.
  template <typename T>
  const T *GetDeclAs(const std::string &ID) const {
    return GetNodeAs<T>(DeclBindings, ID);
  }
  template <typename T>
  const T *GetStmtAs(const std::string &ID) const {
    return GetNodeAs<T>(StmtBindings, ID);
  }

  /// Adds all bound nodes to bound_nodes_builder.
  void CopyTo(BoundNodesBuilder *CopyToBuilder) const;

 private:
  template <typename T, typename MapT>
  const T *GetNodeAs(const MapT &Bindings, const std::string &ID) const {
    typename MapT::const_iterator It = Bindings.find(ID);
    if (It == Bindings.end()) {
      return NULL;
    }
    return llvm::dyn_cast<T>(It->second);
  }

  std::map<std::string, const clang::Decl*> DeclBindings;
  std::map<std::string, const clang::Stmt*> StmtBindings;
};  // class BoundNodes

/// Creates BoundNodes objects.
class BoundNodesBuilder {
 public:
  BoundNodesBuilder() {}

  /// Add a binding from 'ID' to 'Node'.
  /// FIXME: Add overloads for all AST base types.
  void SetBinding(const std::string &ID, const clang::Decl *Node) {
    DeclBindings[ID] = Node;
  }
  void SetBinding(const std::string &ID, const clang::Stmt *Node) {
    StmtBindings[ID] = Node;
  }

  /// Returns a BoundNodes object containing all current bindings.
  BoundNodes Build() const {
    return BoundNodes(DeclBindings, StmtBindings);
  }

 private:
  BoundNodesBuilder(const BoundNodesBuilder&);  // DO NOT IMPLEMENT
  void operator=(const BoundNodesBuilder&);  // DO NOT IMPLEMENT

  std::map<std::string, const clang::Decl*> DeclBindings;
  std::map<std::string, const clang::Stmt*> StmtBindings;
};

inline void BoundNodes::CopyTo(BoundNodesBuilder *CopyToBuilder) const {
  for (std::map<std::string, const clang::Decl*>::const_iterator
           It = DeclBindings.begin(), End = DeclBindings.end();
       It != End; ++It) {
    CopyToBuilder->SetBinding(It->first, It->second);
  }
  /// FIXME: Pull out method.
  for (std::map<std::string, const clang::Stmt*>::const_iterator
           It = StmtBindings.begin(), End = StmtBindings.end();
       It != End; ++It) {
    CopyToBuilder->SetBinding(It->first, It->second);
  }
}

class ASTMatchFinder;

/// Generic interface for matchers on an AST node of type T. Implement
/// this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public llvm::RefCountedBaseVPTR {
 public:
  virtual ~MatcherInterface() {}

  /// Returns true if 'Node' can be matched.
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the AST via 'Finder'.
  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const = 0;
};

/// Interface for matchers that only evaluate properties on a single node.
template <typename T>
class SingleNodeMatcherInterface : public MatcherInterface<T> {
 public:
  /// Returns true if the matcher matches the provided node. A subclass
  /// must implement this instead of Matches().
  virtual bool MatchesNode(const T &Node) const = 0;

 private:
  /// Implements MatcherInterface::Matches.
  virtual bool Matches(const T &Node,
                       ASTMatchFinder * /* Finder */,
                       BoundNodesBuilder * /*  Builder */) const {
    return MatchesNode(Node);
  }
};

/// Wrapper of a MatcherInterface<T> *that allows copying.
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
  /// Takes ownership of the provided implementation pointer.
  explicit Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}

  /// Forwards the call to the underlying MatcherInterface<T> pointer.
  bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return Implementation->Matches(Node, Finder, Builder);
  }

  /// Implicitly converts this object to a Matcher<Derived>; requires
  /// Derived to be derived from T.
  template <typename Derived>
  operator Matcher<Derived>() const {
    return Matcher<Derived>(new ImplicitCastMatcher<Derived>(*this));
  }

  /// Returns an ID that uniquely identifies the matcher.
  uint64_t GetID() const {
    /// FIXME: Document the requirements this imposes on matcher
    /// implementations (no new() implementation_ during a Matches()).
    return reinterpret_cast<uint64_t>(Implementation.getPtr());
  }

 private:
  /// Allows conversion from Matcher<T> to Matcher<Derived> if Derived
  /// is derived from T.
  template <typename Derived>
  class ImplicitCastMatcher : public MatcherInterface<Derived> {
   public:
    explicit ImplicitCastMatcher(const Matcher<T> &From)
        : From(From) {}

    virtual bool Matches(
        const Derived &Node,
        ASTMatchFinder *Finder,
        BoundNodesBuilder *Builder) const {
      return From.Matches(Node, Finder, Builder);
    }

   private:
    const Matcher<T> From;
  };

  llvm::IntrusiveRefCntPtr< MatcherInterface<T> > Implementation;
};  // class Matcher

/// A convenient helper for creating a Matcher<T> without specifying
/// the template type argument.
template <typename T>
inline Matcher<T> MakeMatcher(MatcherInterface<T> *Implementation) {
  return Matcher<T>(Implementation);
}

/// Matches declarations for QualType and CallExpr. Type argument
/// DeclMatcherT is required by PolymorphicMatcherWithParam1 but not
/// actually used.
template <typename T, typename DeclMatcherT>
class HasDeclarationMatcher : public MatcherInterface<T> {
  COMPILE_ASSERT((llvm::is_same< DeclMatcherT, Matcher<clang::Decl> >::value),
                 instantiated_with_wrong_types);
 public:
  explicit HasDeclarationMatcher(const Matcher<clang::Decl> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return MatchesSpecialized(Node, Finder, Builder);
  }

 private:
  /// Extracts the CXXRecordDecl of a QualType and returns whether the inner
  /// matcher matches on it.
  bool MatchesSpecialized(
      const clang::QualType &Node, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    /// FIXME: Add other ways to convert...
    clang::CXXRecordDecl *NodeAsRecordDecl = Node->getAsCXXRecordDecl();
    return NodeAsRecordDecl != NULL &&
        InnerMatcher.Matches(*NodeAsRecordDecl, Finder, Builder);
  }

  /// Extracts the Decl of the callee of a CallExpr and returns whether the
  /// inner matcher matches on it.
  bool MatchesSpecialized(
      const clang::CallExpr &Node, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    const clang::Decl *NodeAsDecl = Node.getCalleeDecl();
    return NodeAsDecl != NULL &&
        InnerMatcher.Matches(*NodeAsDecl, Finder, Builder);
  }

  /// Extracts the Decl of the constructor call and returns whether the inner
  /// matcher matches on it.
  bool MatchesSpecialized(
      const clang::CXXConstructExpr &Node, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    const clang::Decl *NodeAsDecl = Node.getConstructor();
    return NodeAsDecl != NULL &&
        InnerMatcher.Matches(*NodeAsDecl, Finder, Builder);
  }

  const Matcher<clang::Decl> InnerMatcher;
};

/// IsBaseType<T>::value is true if T is a "base" type in the AST
/// node class hierarchies (i.e. if T is Decl, Stmt, or QualType).
template <typename T>
struct IsBaseType {
  static const bool value = (llvm::is_same<T, clang::Decl>::value ||
                             llvm::is_same<T, clang::Stmt>::value ||
                             llvm::is_same<T, clang::QualType>::value);
};
template <typename T>
const bool IsBaseType<T>::value;

/// Interface that can match any AST base node type and contains default
/// implementations returning false.
class UntypedBaseMatcher {
 public:
  virtual ~UntypedBaseMatcher() {}

  virtual bool Matches(
      const clang::Decl &DeclNode, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return false;
  }
  virtual bool Matches(
      const clang::QualType &TypeNode, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return false;
  }
  virtual bool Matches(
      const clang::Stmt &StmtNode, ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return false;
  }

  /// Returns a unique ID for the matcher.
  virtual uint64_t GetID() const = 0;
};

/// An UntypedBaseMatcher that overwrites the Matches(...) method for node
/// type T. T must be an AST base type.
template <typename T>
class TypedBaseMatcher : public UntypedBaseMatcher {
  COMPILE_ASSERT(IsBaseType<T>::value,
                 typed_base_matcher_can_only_be_used_with_base_type);
 public:
  explicit TypedBaseMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  using UntypedBaseMatcher::Matches;
  /// Implements UntypedBaseMatcher::Matches. Since T is guaranteed to
  /// be a "base" AST node type, this method is guaranteed to override
  /// one of the Matches() methods from UntypedBaseMatcher.
  virtual bool Matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesBuilder *Builder) const {
    return InnerMatcher.Matches(Node, Finder, Builder);
  }

  /// Implements UntypedBaseMatcher::GetID.
  virtual uint64_t GetID() const {
    return InnerMatcher.GetID();
  }

 private:
  Matcher<T> InnerMatcher;
};

/// Interface that allows matchers to traverse the AST.
/// This provides two entry methods for each base node type in the AST:
/// - MatchesChildOf:
///   Matches a matcher on every child node of the given node. Returns true
///   if at least one child node could be matched.
/// - MatchesDescendantOf:
///   Matches a matcher on all descendant nodes of the given node. Returns true
///   if at least one descendant matched.
class ASTMatchFinder {
 public:
  /// Defines how we descend a level in the AST when we pass
  /// through expressions.
  enum TraversalMethod {
    /// Will traverse any child nodes.
    kAsIs,
    /// Will not traverse implicit casts and parentheses.
    kIgnoreImplicitCastsAndParentheses
  };

  virtual ~ASTMatchFinder() {}

  /// Returns true if the given class is directly or indirectly derived
  /// from a base type with the given name.  A class is considered to
  /// be also derived from itself.
  virtual bool ClassIsDerivedFrom(const clang::CXXRecordDecl *Declaration,
                                  const std::string &BaseName) const = 0;

  // FIXME: Implement for other base nodes.
  virtual bool MatchesChildOf(const clang::Decl &DeclNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesBuilder *Builder,
                              TraversalMethod Traverse) = 0;
  virtual bool MatchesChildOf(const clang::Stmt &StmtNode,
                              const UntypedBaseMatcher &BaseMatcher,
                              BoundNodesBuilder *Builder,
                              TraversalMethod Traverse) = 0;

  virtual bool MatchesDescendantOf(const clang::Decl &DeclNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesBuilder *Builder) = 0;
  virtual bool MatchesDescendantOf(const clang::Stmt &StmtNode,
                                   const UntypedBaseMatcher &BaseMatcher,
                                   BoundNodesBuilder *Builder) = 0;
};

/// Converts a Matcher<T> to a matcher of desired type To by "adapting"
/// a To into a T. The ArgumentAdapterT argument specifies how the
/// adaptation is done. For example:
///
///   ArgumentAdaptingMatcher<DynCastMatcher, T>(InnerMatcher);
/// returns a matcher that can be used where a Matcher<To> is required, if
/// To and T are in the same type hierarchy, and thus dyn_cast can be
/// called to convert a To to a T.
///
/// FIXME: Make sure all our applications of this class actually require
/// knowledge about the inner type. DynCastMatcher obviously does, but the
/// Has *matchers require the inner type solely for COMPILE_ASSERT purposes.
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

/// A PolymorphicMatcherWithParamN<MatcherT, P1, ..., PN> object can be
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
template <template <typename T> class MatcherT>
class PolymorphicMatcherWithParam0 {
 public:
  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T>());
  }
};

template <template <typename T, typename P1> class MatcherT,
          typename P1>
class PolymorphicMatcherWithParam1 {
 public:
  explicit PolymorphicMatcherWithParam1(const P1 &Param1)
      : Param1(Param1) {}

  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T, P1>(Param1));
  }

 private:
  const P1 Param1;
};

template <template <typename T, typename P1, typename P2> class MatcherT,
          typename P1, typename P2>
class PolymorphicMatcherWithParam2 {
 public:
  PolymorphicMatcherWithParam2(const P1 &Param1, const P2 &Param2)
      : Param1(Param1), Param2(Param2) {}

  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T, P1, P2>(Param1, Param2));
  }

 private:
  const P1 Param1;
  const P2 Param2;
};

// FIXME: Alternatively we could also create a IsAMatcher or something
// that checks that a dyn_cast is possible. This is purely needed for the
// difference between calling for example:
//   Class()
// and
//   Class(SomeMatcher)
// In the second case we need the correct type we were dyn_cast'ed to in order
// to get the right type for the inner matcher. In the first case we don't need
// that, but we use the type conversion anyway and insert a TrueMatcher.
template <typename T>
class TrueMatcher : public SingleNodeMatcherInterface<T>  {
 public:
  virtual bool MatchesNode(const T &Node) const {
    return true;
  }
};

/// Provides a MatcherInterface<T> for a Matcher<To> that matches if T is
/// dyn_cast'able into To and the given Matcher<To> matches on the dyn_cast'ed
/// node.
template <typename T, typename To>
class DynCastMatcher : public MatcherInterface<T> {
 public:
  explicit DynCastMatcher(const Matcher<To> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool Matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesBuilder *Builder) const {
    const To *InnerMatchValue = llvm::dyn_cast<To>(&Node);
    return InnerMatchValue != NULL &&
        InnerMatcher.Matches(*InnerMatchValue, Finder, Builder);
  }

 private:
  const Matcher<To> InnerMatcher;
};

/// Enables the user to pass a Matcher<clang::CXXMemberCallExpr> to Call().
/// FIXME: Alternatives are using more specific methods than Call, like
/// MemberCall, or not using VariadicFunction for Call and overloading it.
template <>
template <>
inline Matcher<clang::CXXMemberCallExpr>::
operator Matcher<clang::CallExpr>() const {
  return MakeMatcher(
      new DynCastMatcher<clang::CallExpr, clang::CXXMemberCallExpr>(*this));
}

/// Matcher<T> that wraps an inner Matcher<T> and binds the matched node to
/// an ID if the inner matcher matches on the node.
template <typename T>
class IdMatcher : public MatcherInterface<T> {
 public:
  /// Creates an IdMatcher that binds to 'ID' if 'InnerMatcher' matches the node.
  IdMatcher(const std::string &ID, const Matcher<T> &InnerMatcher)
      : ID(ID), InnerMatcher(InnerMatcher) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    bool Result = InnerMatcher.Matches(Node, Finder, Builder);
    if (Result) {
      Builder->SetBinding(ID, &Node);
    }
    return Result;
  }

 private:
  const std::string ID;
  const Matcher<T> InnerMatcher;
};

/// Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches. ChildT must be an AST base
/// type.
template <typename T, typename ChildT>
class HasMatcher : public MatcherInterface<T> {
  COMPILE_ASSERT(IsBaseType<ChildT>::value, has_only_accepts_base_type_matcher);
 public:
  explicit HasMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  virtual bool Matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesBuilder *Builder) const {
    return Finder->MatchesChildOf(
        Node, ChildMatcher, Builder,
        ASTMatchFinder::kIgnoreImplicitCastsAndParentheses);
  }

 private:
  const TypedBaseMatcher<ChildT> ChildMatcher;
};

/// Matches nodes of type T if the given Matcher<T> does not match.
/// Type argument MatcherT is required by PolymorphicMatcherWithParam1
/// but not actually used. It will always be instantiated with a type
/// convertible to Matcher<T>.
template <typename T, typename MatcherT>
class NotMatcher : public MatcherInterface<T> {
 public:
  explicit NotMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return !InnerMatcher.Matches(Node, Finder, Builder);
  }

 private:
  const Matcher<T> InnerMatcher;
};

/// Matches nodes of type T for which both provided matchers
/// match. Type arguments MatcherT1 and MatcherT2 are required by
/// PolymorphicMatcherWithParam2 but not actually used. They will
/// always be instantiated with types convertible to Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AllOfMatcher : public MatcherInterface<T> {
 public:
  AllOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnerMatcher2(InnerMatcher2) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return InnerMatcher1.Matches(Node, Finder, Builder) &&
           InnerMatcher2.Matches(Node, Finder, Builder);
  }

 private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnerMatcher2;
};

/// Matches nodes of type T for which at least one of the two provided
/// matchers matches. Type arguments MatcherT1 and MatcherT2 are
/// required by PolymorphicMatcherWithParam2 but not actually
/// used. They will always be instantiated with types convertible to
/// Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AnyOfMatcher : public MatcherInterface<T> {
 public:
  AnyOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnertMatcher2(InnerMatcher2) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return InnerMatcher1.Matches(Node, Finder, Builder) ||
           InnertMatcher2.Matches(Node, Finder, Builder);
  }

 private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnertMatcher2;
};

/// Creates a Matcher<T> that matches if
/// T is dyn_cast'able into InnerT and all inner matchers match.
template<typename T, typename InnerT>
Matcher<T> MakeDynCastAllOfComposite(
    const Matcher<InnerT> *const InnerMatchers[], int Count) {
  if (Count == 0) {
    return ArgumentAdaptingMatcher<DynCastMatcher, InnerT>(
        MakeMatcher(new TrueMatcher<InnerT>));
  }
  Matcher<InnerT> InnerMatcher = *InnerMatchers[Count-1];
  for (int I = Count-2; I >= 0; --I) {
    InnerMatcher = MakeMatcher(
        new AllOfMatcher<InnerT, Matcher<InnerT>, Matcher<InnerT> >(
            *InnerMatchers[I], InnerMatcher));
  }
  return ArgumentAdaptingMatcher<DynCastMatcher, InnerT>(InnerMatcher);
}

/// Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
/// DescendantT must be an AST base type.
template <typename T, typename DescendantT>
class HasDescendantMatcher : public MatcherInterface<T> {
  COMPILE_ASSERT(IsBaseType<DescendantT>::value,
                 has_descendant_only_accepts_base_type_matcher);
 public:
  explicit HasDescendantMatcher(const Matcher<DescendantT> &DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  virtual bool Matches(
      const T &Node,
      ASTMatchFinder *Finder,
      BoundNodesBuilder *Builder) const {
    return Finder->MatchesDescendantOf(
        Node, DescendantMatcher, Builder);
  }

 private:
  const TypedBaseMatcher<DescendantT> DescendantMatcher;
};

/// Matches on nodes that have a getValue() method if getValue() equals the
/// value the ValueEqualsMatcher was constructed with.
template <typename T, typename ValueT>
class ValueEqualsMatcher : public SingleNodeMatcherInterface<T> {
  COMPILE_ASSERT((llvm::is_base_of<clang::CharacterLiteral, T>::value) ||
                 (llvm::is_base_of<clang::CXXBoolLiteralExpr, T>::value) ||
                 (llvm::is_base_of<clang::FloatingLiteral, T>::value) ||
                 (llvm::is_base_of<clang::IntegerLiteral, T>::value),
                 the_node_must_have_a_getValue_method);
 public:
  explicit ValueEqualsMatcher(const ValueT &ExpectedValue)
      : ExpectedValue(ExpectedValue) {}

  virtual bool MatchesNode(const T &Node) const {
    return Node.getValue() == ExpectedValue;
  }

 private:
  const ValueT ExpectedValue;
};

template <typename T>
class IsDefinitionMatcher : public SingleNodeMatcherInterface<T> {
  COMPILE_ASSERT((llvm::is_base_of<clang::TagDecl, T>::value) ||
                 (llvm::is_base_of<clang::VarDecl, T>::value) ||
                 (llvm::is_base_of<clang::FunctionDecl, T>::value),
                 is_definition_requires_isThisDeclarationADefinition_method);
 public:
  virtual bool MatchesNode(const T &Node) const {
    return Node.isThisDeclarationADefinition();
  }
};

class IsArrowMatcher : public SingleNodeMatcherInterface<clang::MemberExpr> {
 public:
  virtual bool MatchesNode(const clang::MemberExpr &Node) const {
    return Node.isArrow();
  }
};

/// A VariadicDynCastAllOfMatcher<SourceT, TargetT> object is a
/// variadic functor that takes a list of Matcher<TargetT> and returns
/// a Matcher<SourceT> that dyn_casts its argument to TargetT.
template <typename SourceT, typename TargetT>
class VariadicDynCastAllOfMatcher
    : public internal::VariadicFunction<
        Matcher<SourceT>, Matcher<TargetT>,
        MakeDynCastAllOfComposite<SourceT, TargetT> > {
 public:
  VariadicDynCastAllOfMatcher() {}
};

/// AST_MATCHER_P(Type, DefineMatcher, ParamType, Param) { ... }
///
/// defines a single-parameter function named DefineMatcher() that returns a
/// Matcher<Type> object. The code between the curly braces has access
/// to the following variables:
///
///   Node:                  the AST node being matched; its type is Type.
///   Param:                 the parameter passed to the function; its type
///                          is ParamType.
///   Finder:                an ASTMatchFinder*.
///   Builder:               a BoundNodesBuilder*.
///
/// The code should return true if 'Node' matches.
#define AST_MATCHER_P(Type, DefineMatcher, ParamType, Param)                  \
  class matcher_internal_##DefineMatcher##Matcher                             \
      : public MatcherInterface<Type> {                                       \
   public:                                                                    \
    explicit matcher_internal_##DefineMatcher##Matcher(                       \
        const ParamType &A##Param) : Param(A##Param) {}                       \
    virtual bool Matches(                                                     \
        const Type &Node, ASTMatchFinder *Finder,                             \
        BoundNodesBuilder *Builder) const;                                    \
   private:                                                                   \
    const ParamType Param;                                                    \
  };                                                                          \
  inline Matcher<Type> DefineMatcher(const ParamType &Param) {                \
    return MakeMatcher(new matcher_internal_##DefineMatcher##Matcher(Param)); \
  }                                                                           \
  inline bool matcher_internal_##DefineMatcher##Matcher::Matches(             \
      const Type &Node, ASTMatchFinder *Finder,                               \
      BoundNodesBuilder *Builder) const

/// AST_MATCHER_P2(Type, DefineMatcher, ParamType1, Param1, ParamType2, Param2)
/// { ... }
///
/// defines a two-parameter function named DefineMatcher() that returns a
/// Matcher<Type> object. The code between the curly braces has access
/// to the following variables:
///
///   Node:                  the AST node being matched; its type is Type.
///   Param1, Param2:        the parameters passed to the function; their types
///                          are ParamType1 and ParamType2.
///   Finder:                an ASTMatchFinder*.
///   Builder:               a BoundNodesBuilder*.
///
/// The code should return true if 'Node' matches.
#define AST_MATCHER_P2(                                                        \
    Type, DefineMatcher, ParamType1, Param1, ParamType2, Param2)               \
  class matcher_internal_##DefineMatcher##Matcher                              \
      : public MatcherInterface<Type> {                                        \
   public:                                                                     \
    matcher_internal_##DefineMatcher##Matcher(                                 \
        const ParamType1 &A##Param1, const ParamType2 &A##Param2)              \
        : Param1(A##Param1), Param2(A##Param2) {}                              \
    virtual bool Matches(                                                      \
        const Type &Node, ASTMatchFinder *Finder,                              \
        BoundNodesBuilder *Builder) const;                                     \
   private:                                                                    \
    const ParamType1 Param1;                                                   \
    const ParamType2 Param2;                                                   \
  };                                                                           \
  inline Matcher<Type> DefineMatcher(                                          \
      const ParamType1 &Param1, const ParamType2 &Param2) {                    \
    return MakeMatcher(new matcher_internal_##DefineMatcher##Matcher(          \
        Param1, Param2));                                                      \
  }                                                                            \
  inline bool matcher_internal_##DefineMatcher##Matcher::Matches(              \
      const Type &Node, ASTMatchFinder *Finder,                                \
      BoundNodesBuilder *Builder) const

/// AST_POLYMORPHIC_MATCHER_P(DefineMatcher, ParamType, Param) { ... }
///
/// defines a single-parameter function named DefineMatcher() that is
/// polymorphic in the return type. The variables are the same as for
/// AST_MATCHER_P, with the addition of NodeType, which specifies the node type
/// of the matcher Matcher<NodeType> returned by the function matcher().
///
/// FIXME: Pull out common code with above macro?
#define AST_POLYMORPHIC_MATCHER_P(DefineMatcher, ParamType, Param)             \
  template <typename NodeType, typename ParamT>                                \
  class matcher_internal_##DefineMatcher##Matcher                              \
      : public MatcherInterface<NodeType> {                                    \
   public:                                                                     \
    explicit matcher_internal_##DefineMatcher##Matcher(                        \
        const ParamType &A##Param) : Param(A##Param) {}                        \
    virtual bool Matches(                                                      \
        const NodeType &Node, ASTMatchFinder *Finder,                          \
        BoundNodesBuilder *Builder) const;                                     \
   private:                                                                    \
    const ParamType Param;                                                     \
  };                                                                           \
  inline PolymorphicMatcherWithParam1<                                         \
      matcher_internal_##DefineMatcher##Matcher,                               \
      ParamType >                                                              \
    DefineMatcher(const ParamType &Param) {                                    \
    return PolymorphicMatcherWithParam1<                                       \
        matcher_internal_##DefineMatcher##Matcher,                             \
        ParamType >(Param);                                                    \
  }                                                                            \
  template <typename NodeType, typename ParamT>                                \
  bool matcher_internal_##DefineMatcher##Matcher<NodeType, ParamT>::Matches(   \
      const NodeType &Node, ASTMatchFinder *Finder,                            \
      BoundNodesBuilder *Builder) const

/// AST_POLYMORPHIC_MATCHER_P2(
///     DefineMatcher, ParamType1, Param1, ParamType2, Param2) { ... }
///
/// defines a two-parameter function named matcher() that is polymorphic in
/// the return type. The variables are the same as for AST_MATCHER_P2, with the
/// addition of NodeType, which specifies the node type of the matcher
/// Matcher<NodeType> returned by the function DefineMatcher().
#define AST_POLYMORPHIC_MATCHER_P2(                                            \
      DefineMatcher, ParamType1, Param1, ParamType2, Param2)                   \
  template <typename NodeType, typename ParamT1, typename ParamT2>             \
  class matcher_internal_##DefineMatcher##Matcher                              \
      : public MatcherInterface<NodeType> {                                    \
   public:                                                                     \
    matcher_internal_##DefineMatcher##Matcher(                                 \
        const ParamType1 &A##Param1, const ParamType2 &A##Param2)              \
        : Param1(A##Param1), Param2(A##Param2) {}                              \
    virtual bool Matches(                                                      \
        const NodeType &Node, ASTMatchFinder *Finder,                          \
        BoundNodesBuilder *Builder) const;                                     \
   private:                                                                    \
    const ParamType1 Param1;                                                   \
    const ParamType2 Param2;                                                   \
  };                                                                           \
  inline PolymorphicMatcherWithParam2<                                         \
      matcher_internal_##DefineMatcher##Matcher,                               \
      ParamType1, ParamType2 >                                                 \
    DefineMatcher(const ParamType1 &Param1, const ParamType2 &Param2) {        \
    return PolymorphicMatcherWithParam2<                                       \
        matcher_internal_##DefineMatcher##Matcher,                             \
        ParamType1, ParamType2 >(                                              \
        Param1, Param2);                                                       \
  }                                                                            \
  template <typename NodeType, typename ParamT1, typename ParamT2>             \
  bool matcher_internal_##DefineMatcher##Matcher<                              \
      NodeType, ParamT1, ParamT2>::Matches(                                    \
      const NodeType &Node, ASTMatchFinder *Finder,                            \
      BoundNodesBuilder *Builder) const

namespace match {

typedef Matcher<clang::Decl> DeclarationMatcher;
typedef Matcher<clang::QualType> TypeMatcher;
typedef Matcher<clang::Stmt> StatementMatcher;

/// Matches C++ class declarations.
///
/// Example matches X, Z
///   class X;
///   template<class T> class Z {};
const VariadicDynCastAllOfMatcher<clang::Decl, clang::CXXRecordDecl> Class;

/// Matches method declarations.
///
/// Example matches y
///   class X { void y() };
const VariadicDynCastAllOfMatcher<clang::Decl, clang::CXXMethodDecl> Method;

/// Matches variable declarations.
///
/// Example matches a
///   int a;
const VariadicDynCastAllOfMatcher<clang::Decl, clang::VarDecl> Variable;

/// Matches member expressions.
///
/// Given
///   class Y {
///     void x() { this->x(); x(); Y y; y.x(); a; this->b; Y::b; }
///     int a; static int b;
///   };
/// MemberExpression()
///   matches this->x, x, y.x, a, this->b
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::MemberExpr>
MemberExpression;

/// Matches call expressions.
///
/// Example matches x.y()
///   X x;
///   x.y();
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::CallExpr> Call;

/// Matches constructor call expressions (including implicit ones).
///
/// Example matches string(ptr, n) and ptr within arguments of f
///     (matcher = ConstructorCall())
///   void f(const string &a, const string &b);
///   char *ptr;
///   int n;
///   f(string(ptr, n), ptr);
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::CXXConstructExpr>
ConstructorCall;

/// Matches the value of a default argument at the call site.
///
/// Example matches the CXXDefaultArgExpr placeholder inserted for the
///     default value of the second parameter in the call expression f(42)
///     (matcher = DefaultArgument())
///   void f(int x, int y = 0);
///   f(42);
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::CXXDefaultArgExpr>
DefaultArgument;

/// Matches overloaded operator calls.
/// Note that if an operator isn't overloaded, it won't match.  Instead, use
/// BinaryOperator matcher.
/// Currently it does not match operators such as new delete.
/// FIXME: figure out why these do not match?
///
/// Example matches both operator<<((o << b), c) and operator<<(o, b)
///     (matcher = OverloadedOperatorCall())
///   ostream &operator<< (ostream &out, int i) { };
///   ostream &o; int b = 1, c = 1;
///   o << b << c;
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::CXXOperatorCallExpr>
OverloadedOperatorCall;

/// Matches expressions.
///
/// Example matches x()
///   void f() { x(); }
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::Expr> Expression;

/// Matches expressions that refer to declarations.
///
/// Example matches x in if (x)
///   bool x;
///   if (x) {}
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::DeclRefExpr>
DeclarationReference;

/// Matches if statements.
///
/// Example matches 'if (x) {}'
///   if (x) {}
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::IfStmt> If;

/// Matches for statements.
///
/// Example matches 'for (;;) {}'
///   for (;;) {}
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::ForStmt> For;

/// Matches compound statements.
///
/// Example matches '{}' and '{{}}'in 'for (;;) {{}}'
///   for (;;) {{}}
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::CompoundStmt>
CompoundStatement;

/// Matches bool literals.
///
/// Example matches true
///   true
const VariadicDynCastAllOfMatcher<clang::Expr, clang::CXXBoolLiteralExpr>
BoolLiteral;

/// Matches string literals (also matches wide string literals).
///
/// Example matches "abcd", L"abcd"
///   char *s = "abcd"; wchar_t *ws = L"abcd"
const VariadicDynCastAllOfMatcher<clang::Expr, clang::StringLiteral>
StringLiteral;

/// Matches character literals (also matches wchar_t).
/// Not matching Hex-encoded chars (e.g. 0x1234, which is a IntegerLiteral),
/// though.
///
/// Example matches 'a', L'a'
///   char ch = 'a'; wchar_t chw = L'a';
const VariadicDynCastAllOfMatcher<clang::Expr, clang::CharacterLiteral>
CharacterLiteral;

/// Matches integer literals of all sizes / encodings.
/// Not matching character-encoded integers such as L'a'.
///
/// Example matches 1, 1L, 0x1, 1U
const VariadicDynCastAllOfMatcher<clang::Expr, clang::IntegerLiteral>
IntegerLiteral;

/// Matches binary operator expressions.
///
/// Example matches a || b
///   !(a || b)
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::BinaryOperator>
BinaryOperator;

/// Matches unary operator expressions.
///
/// Example matches !a
///   !a || b
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::UnaryOperator>
UnaryOperator;

/// Matches conditional operator expressions.
///
/// Example matches a ? b : c
///   (a ? b : c) + 42
const VariadicDynCastAllOfMatcher<clang::Stmt, clang::ConditionalOperator>
ConditionalOperator;

template<typename C1, typename C2>
PolymorphicMatcherWithParam2<AnyOfMatcher, C1, C2>
AnyOf(const C1 &P1, const C2 &P2) {
  return PolymorphicMatcherWithParam2< AnyOfMatcher, C1, C2 >(P1, P2);
}

template<typename C1, typename C2, typename C3>
PolymorphicMatcherWithParam2<AnyOfMatcher, C1,
    PolymorphicMatcherWithParam2<AnyOfMatcher, C2, C3> >
AnyOf(const C1 &P1, const C2 &P2, const C3 &P3) {
  return AnyOf(P1, AnyOf(P2, P3));
}

template<typename C1, typename C2, typename C3, typename C4>
PolymorphicMatcherWithParam2<AnyOfMatcher, C1,
    PolymorphicMatcherWithParam2<AnyOfMatcher, C2,
        PolymorphicMatcherWithParam2<AnyOfMatcher, C3, C4> > >
AnyOf(const C1 &P1, const C2 &P2, const C3 &P3, const C4 &P4) {
  return AnyOf(P1, AnyOf(P2, AnyOf(P3, P4)));
}

template<typename C1, typename C2>
PolymorphicMatcherWithParam2<AllOfMatcher, C1, C2>
AllOf(const C1 &P1, const C2 &P2) {
  return PolymorphicMatcherWithParam2<AllOfMatcher, C1, C2>(P1, P2);
}

/// Matches NamedDecl nodes that have the specified name. Supports specifying
/// enclosing namespaces or classes by prefixing the name with '<enclosing>::'.
/// Does not match typedefs of an underlying type with the given name.
///
/// Example matches X (name == "X")
///   class X;
///
/// Example matches X (name is one of "::a::b::X", "a::b::X", "b::X", "X")
/// namespace a { namespace b { class X; } }
AST_MATCHER_P(clang::NamedDecl, HasName, std::string, Name) {
  assert(!Name.empty());
  const std::string FullNameString = "::" + Node.getQualifiedNameAsString();
  const llvm::StringRef FullName = FullNameString;
  const llvm::StringRef Pattern = Name;
  if (Pattern.startswith("::")) {
    return FullName == Pattern;
  } else {
    return FullName.endswith(("::" + Pattern).str());
  }
}

/// Matches overloaded operator name given in strings without the "operator"
/// prefix, such as "<<", for OverloadedOperatorCall's.
///
/// Example matches a << b
///     (matcher == OverloadedOperatorCall(HasOverloadedOperatorName("<<")))
///   a << b;
///   c && d;  // assuming both operator<<
///            // and operator&& are overloaded somewhere.
AST_MATCHER_P(clang::CXXOperatorCallExpr,
              HasOverloadedOperatorName, std::string, Name) {
  return clang::getOperatorSpelling(Node.getOperator()) == Name;
}

/// Matches C++ classes that are directly or indirectly derived from
/// the given base class. Note that a class is considered to be also
/// derived from itself.  The parameter specified the name of the base
/// type (either a class or a typedef), and does not allow structural
/// matches for namespaces or template type parameters.
///
/// Example matches X, Y, Z, C (base == "X")
///   class X;                // A class is considered to be derived from itself.
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
///
/// In the following example, Bar matches IsDerivedFrom("X"):
///   class Foo;
///   typedef Foo X;
///   class Bar : public Foo {};  // derived from a type that X is a typedef of
AST_MATCHER_P(clang::CXXRecordDecl, IsDerivedFrom, std::string, Base) {
  assert(!Base.empty());
  return Finder->ClassIsDerivedFrom(&Node, Base);
}

/// Matches AST nodes that have child AST nodes that match the provided matcher.
///
/// Example matches X, Y (matcher = Class(Has(Class(HasName("X")))
///   class X {};  // Matches X, because X::X is a class of name X inside X.
///   class Y { class X {}; };
///   class Z { class Y { class X {}; }; };  // Does not match Z.
///
/// ChildT must be an AST base type.
template <typename ChildT>
ArgumentAdaptingMatcher<HasMatcher, ChildT> Has(
    const Matcher<ChildT> &ChildMatcher) {
  return ArgumentAdaptingMatcher<HasMatcher, ChildT>(ChildMatcher);
}

/// Matches AST nodes that have descendant AST nodes that match the provided
/// matcher.
///
/// Example matches X, Y, Z (matcher = Class(HasDescendant(Class(HasName("X")))))
///   class X {};  // Matches X, because X::X is a class of name X inside X.
///   class Y { class X {}; };
///   class Z { class Y { class X {}; }; };
///
/// DescendantT must be an AST base type.
template <typename DescendantT>
ArgumentAdaptingMatcher<HasDescendantMatcher, DescendantT> HasDescendant(
    const Matcher<DescendantT> &DescendantMatcher) {
  return ArgumentAdaptingMatcher<HasDescendantMatcher, DescendantT>(
      DescendantMatcher);
}

/// Matches if the provided matcher does not match.
///
/// Example matches Y (matcher = Class(Not(HasName("X"))))
///   class X {};
///   class Y {};
template <typename M>
PolymorphicMatcherWithParam1<NotMatcher, M> Not(const M &InnerMatcher) {
  return PolymorphicMatcherWithParam1<NotMatcher, M>(InnerMatcher);
}

/// If the provided matcher matches a node, binds the node to 'id'.
/// FIXME: Add example for accessing it.
template <typename T>
Matcher<T> Id(const std::string &ID, const Matcher<T> &InnerMatcher) {
  return Matcher<T>(new IdMatcher<T>(ID, InnerMatcher));
}

/// Matches a type if the declaration of the type matches the given matcher.
inline PolymorphicMatcherWithParam1< HasDeclarationMatcher,
                                     Matcher<clang::Decl> >
    HasDeclaration(const Matcher<clang::Decl> &InnerMatcher) {
  return PolymorphicMatcherWithParam1< HasDeclarationMatcher,
                                       Matcher<clang::Decl> >(InnerMatcher);
}

/// Matches on the implicit object argument of a member call expression.
///
/// Example matches y.x() (matcher = Call(On(HasType(Class(HasName("Y"))))))
///   class Y { public: void x(); };
///   void z() { Y y; y.x(); }",
///
/// FIXME: Overload to allow directly matching types?
AST_MATCHER_P(
    clang::CXXMemberCallExpr, On, Matcher<clang::Expr>, InnerMatcher) {
  const clang::Expr *ExprNode = const_cast<clang::CXXMemberCallExpr&>(Node)
      .getImplicitObjectArgument()
      ->IgnoreParenImpCasts();
  return (ExprNode != NULL &&
          InnerMatcher.Matches(*ExprNode, Finder, Builder));
}

/// Matches if the call expression's callee expression matches.
///
/// Given
///   class Y { void x() { this->x(); x(); Y y; y.x(); } };
///   void f() { f(); }
/// Call(Callee(Expression()))
///   matches this->x(), x(), y.x(), f()
/// with Callee(...)
///   matching this->x, x, y.x, f respectively
///
/// Note: Callee cannot take the more general Matcher<clang::Expr> because
/// this introduces ambiguous overloads with calls to Callee taking a
/// Matcher<clang::Decl>, as the matcher hierarchy is purely implemented in
/// terms of implicit casts.
AST_MATCHER_P(clang::CallExpr, Callee, Matcher<clang::Stmt>, InnerMatcher) {
  const clang::Expr *ExprNode = Node.getCallee();
  return (ExprNode != NULL &&
          InnerMatcher.Matches(*ExprNode, Finder, Builder));
}

/// Matches if the call expression's callee's declaration matches the given
/// matcher.
///
/// Example matches y.x() (matcher = Call(Callee(Method(HasName("x")))))
///   class Y { public: void x(); };
///   void z() { Y y; y.x();
inline Matcher<clang::CallExpr> Callee(
    const Matcher<clang::Decl> &InnerMatcher) {
  return Matcher<clang::CallExpr>(HasDeclaration(InnerMatcher));
}

/// Matches if the expression's or declaration's type matches a type matcher.
///
/// Example matches x (matcher = Expression(HasType(
///                        HasDeclaration(Class(HasName("X"))))))
///             and z (matcher = Variable(HasType(
///                        HasDeclaration(Class(HasName("X"))))))
///  class X {};
///  void y(X &x) { x; X z; }
AST_POLYMORPHIC_MATCHER_P(HasType, Matcher<clang::QualType>, InnerMatcher) {
  COMPILE_ASSERT((llvm::is_base_of<clang::Expr, NodeType>::value ||
                  llvm::is_base_of<clang::ValueDecl, NodeType>::value),
                 instantiated_with_wrong_types);
  return InnerMatcher.Matches(Node.getType(), Finder, Builder);
}

/// Overloaded to match the declaration of the expression's or value
/// declaration's type.
/// In case of a value declaration (for example a variable declaration),
/// this resolves one layer of indirection. For example, in the value declaration
/// "X x;", Class(HasName("X")) matches the declaration of X, while
/// Variable(HasType(Class(HasName("X")))) matches the declaration of x."
///
/// Example matches x (matcher = Expression(HasType(Class(HasName("X")))))
///             and z (matcher = Variable(HasType(Class(HasName("X")))))
///  class X {};
///  void y(X &x) { x; X z; }
inline PolymorphicMatcherWithParam1<matcher_internal_HasTypeMatcher,
                                    Matcher<clang::QualType> >
HasType(const Matcher<clang::Decl> &InnerMatcher) {
  return HasType(Matcher<clang::QualType>(HasDeclaration(InnerMatcher)));
}

/// Matches if the matched type is a pointer type and the pointee type matches
/// the specified matcher.
///
/// Example matches y->x()
///     (matcher = Call(On(HasType(PointsTo(Class(HasName("Y")))))))
///   class Y { public: void x(); };
///   void z() { Y *y; y->x(); }
AST_MATCHER_P(
    clang::QualType, PointsTo, Matcher<clang::QualType>, InnerMatcher) {
  return (Node->isPointerType() &&
          InnerMatcher.Matches(Node->getPointeeType(), Finder, Builder));
}

/// Overloaded to match the pointee type's declaration.
inline Matcher<clang::QualType> PointsTo(
    const Matcher<clang::Decl> &InnerMatcher) {
  return PointsTo(Matcher<clang::QualType>(HasDeclaration(InnerMatcher)));
}

/// Matches if the matched type is a reference type and the referenced type
/// matches the specified matcher.
///
/// Example matches X &x and const X &y
///     (matcher = Variable(HasType(References(Class(HasName("X"))))))
///   class X {
///     void a(X b) {
///       X &x = b;
///       const X &y = b;
///   };
AST_MATCHER_P(
    clang::QualType, References, Matcher<clang::QualType>, InnerMatcher) {
  return (Node->isReferenceType() &&
          InnerMatcher.Matches(Node->getPointeeType(), Finder, Builder));
}

/// Overloaded to match the referenced type's declaration.
inline Matcher<clang::QualType> References(
    const Matcher<clang::Decl> &InnerMatcher) {
  return References(Matcher<clang::QualType>(HasDeclaration(InnerMatcher)));
}

AST_MATCHER_P(clang::CXXMemberCallExpr, OnImplicitObjectArgument,
              Matcher<clang::Expr>, InnerMatcher) {
  const clang::Expr *ExprNode =
      const_cast<clang::CXXMemberCallExpr&>(Node).getImplicitObjectArgument();
  return (ExprNode != NULL &&
          InnerMatcher.Matches(*ExprNode, Finder, Builder));
}

/// Matches if the expression's type either matches the specified matcher, or
/// is a pointer to a type that matches the InnerMatcher.
inline Matcher<clang::CallExpr> ThisPointerType(
    const Matcher<clang::QualType> &InnerMatcher) {
  return OnImplicitObjectArgument(
      AnyOf(HasType(InnerMatcher), HasType(PointsTo(InnerMatcher))));
}

/// Overloaded to match the type's declaration.
inline Matcher<clang::CallExpr> ThisPointerType(
    const Matcher<clang::Decl> &InnerMatcher) {
  return OnImplicitObjectArgument(
      AnyOf(HasType(InnerMatcher), HasType(PointsTo(InnerMatcher))));
}

/// Matches a DeclRefExpr that refers to a declaration that matches the specified
/// matcher.
///
/// Example matches x in if(x)
///     (matcher = DeclarationReference(To(Variable(HasName("x")))))
///   bool x;
///   if (x) {}
AST_MATCHER_P(clang::DeclRefExpr, To, Matcher<clang::Decl>, InnerMatcher) {
  const clang::Decl *DeclNode = Node.getDecl();
  return (DeclNode != NULL &&
          InnerMatcher.Matches(*DeclNode, Finder, Builder));
}

/// Matches a variable declaration that has an initializer expression that
/// matches the given matcher.
///
/// Example matches x (matcher = Variable(HasInitializer(Call())))
///   bool y() { return true; }
///   bool x = y();
AST_MATCHER_P(
    clang::VarDecl, HasInitializer, Matcher<clang::Expr>, InnerMatcher) {
  const clang::Expr *Initializer = Node.getAnyInitializer();
  return (Initializer != NULL &&
          InnerMatcher.Matches(*Initializer, Finder, Builder));
}

/// Checks that a call expression or a constructor call expression has
/// a specific number of arguments (including absent default arguments).
///
/// Example matches f(0, 0) (matcher = Call(ArgumentCountIs(2)))
///   void f(int x, int y);
///   f(0, 0);
AST_POLYMORPHIC_MATCHER_P(ArgumentCountIs, unsigned, N) {
  COMPILE_ASSERT((llvm::is_base_of<clang::CallExpr, NodeType>::value ||
                  llvm::is_base_of<clang::CXXConstructExpr, NodeType>::value),
                 instantiated_with_wrong_types);
  return Node.getNumArgs() == N;
}

/// Matches the n'th argument of a call expression or a constructor
/// call expression.
///
/// Example matches y in x(y)
///     (matcher = Call(HasArgument(0, DeclarationReference())))
///   void x(int) { int y; x(y); }
AST_POLYMORPHIC_MATCHER_P2(
    HasArgument, unsigned, N, Matcher<clang::Expr>, InnerMatcher) {
  COMPILE_ASSERT((llvm::is_base_of<clang::CallExpr, NodeType>::value ||
                  llvm::is_base_of<clang::CXXConstructExpr, NodeType>::value),
                 instantiated_with_wrong_types);
  assert(N >= 0);
  return (N < Node.getNumArgs() &&
          InnerMatcher.Matches(
              *Node.getArg(N)->IgnoreParenImpCasts(), Finder, Builder));
}

/// Matches any argument of a call expression or a constructor call expression.
///
/// Given
///   void x(int, int, int) { int y; x(1, y, 42); }
/// Call(HasAnyArgument(DeclarationReference()))
///   matches x(1, y, 42)
/// with HasAnyArgument(...)
///   matching y
AST_POLYMORPHIC_MATCHER_P(HasAnyArgument, Matcher<clang::Expr>, InnerMatcher) {
  COMPILE_ASSERT((llvm::is_base_of<clang::CallExpr, NodeType>::value ||
                  llvm::is_base_of<clang::CXXConstructExpr, NodeType>::value),
                  instantiated_with_wrong_types);
  for (unsigned I = 0; I < Node.getNumArgs(); ++I) {
    if (InnerMatcher.Matches(*Node.getArg(I)->IgnoreParenImpCasts(),
                             Finder, Builder)) {
      return true;
    }
  }
  return false;
}

/// Matches the n'th parameter of a function declaration.
///
/// Given
///   class X { void f(int x) {} };
/// Method(HasParameter(0, HasType(Variable())))
///   matches f(int x) {}
/// with HasParameter(...)
///   matching int x
AST_MATCHER_P2(clang::FunctionDecl, HasParameter,
               unsigned, N, Matcher<clang::ParmVarDecl>, InnerMatcher) {
  assert(N >= 0);
  return (N < Node.getNumParams() &&
          InnerMatcher.Matches(
              *Node.getParamDecl(N), Finder, Builder));
}

/// Matches any parameter of a function declaration.
/// Does not match the 'this' parameter of a method.
///
/// Given
///   class X { void f(int x, int y, int z) {} };
/// Method(HasAnyParameter(HasName("y")))
///   matches f(int x, int y, int z) {}
/// with HasAnyParameter(...)
///   matching int y
AST_MATCHER_P(clang::FunctionDecl, HasAnyParameter,
              Matcher<clang::ParmVarDecl>, InnerMatcher) {
  for (unsigned I = 0; I < Node.getNumParams(); ++I) {
    if (InnerMatcher.Matches(*Node.getParamDecl(I), Finder, Builder)) {
      return true;
    }
  }
  return false;
}

/// Matches the condition expression of an if statement or conditional operator.
///
/// Example matches true (matcher = HasCondition(BoolLiteral(Equals(true))))
///   if (true) {}
AST_POLYMORPHIC_MATCHER_P(HasCondition, Matcher<clang::Expr>, InnerMatcher) {
  COMPILE_ASSERT((llvm::is_base_of<clang::IfStmt, NodeType>::value) ||
                 (llvm::is_base_of<clang::ConditionalOperator,
                                   NodeType>::value),
                 has_condition_requires_if_statement_or_conditional_operator);
  const clang::Expr *const Condition = Node.getCond();
  return (Condition != NULL &&
          InnerMatcher.Matches(*Condition, Finder, Builder));
}

/// Matches a 'for' statement that has a given body.
///
/// Given
///   for (;;) {}
/// HasBody(CompoundStatement())
///   matches 'for (;;) {}'
/// with CompoundStatement()
///   matching '{}'
AST_MATCHER_P(clang::ForStmt, HasBody, Matcher<clang::Stmt>, InnerMatcher) {
  const clang::Stmt *const Statement = Node.getBody();
  return (Statement != NULL &&
          InnerMatcher.Matches(*Statement, Finder, Builder));
}

/// Matches compound statements where at least one substatement matches a
/// given matcher.
///
/// Given
///   { {}; 1+2; }
/// HasAnySubstatement(CompoundStatement())
///   matches '{ {}; 1+2; }'
/// with CompoundStatement()
///   matching '{}'
AST_MATCHER_P(clang::CompoundStmt, HasAnySubstatement,
              Matcher<clang::Stmt>, InnerMatcher) {
  for (clang::CompoundStmt::const_body_iterator It = Node.body_begin();
       It != Node.body_end();
       ++It) {
    if (InnerMatcher.Matches(**It, Finder, Builder)) return true;
  }
  return false;
}

/// Checks that a compound statement contains a specific number of child
/// statements.
///
/// Example: Given
///   { for (;;) {} }
/// CompoundStatement(StatementCountIs(0)))
///   matches '{}'
///   but does not match the outer compound statement.
AST_MATCHER_P(clang::CompoundStmt, StatementCountIs, unsigned, N) {
  return Node.size() == N;
}

/// Matches literals that are equal to the given value.
///
/// Example matches true (matcher = BoolLiteral(Equals(true)))
///   true
template <typename ValueT>
PolymorphicMatcherWithParam1<ValueEqualsMatcher, ValueT> Equals(
    const ValueT &Value) {
  return PolymorphicMatcherWithParam1<ValueEqualsMatcher, ValueT>(Value);
}

/// Matches the operator Name of operator expressions (binary or unary).
///
/// Example matches a || b (matcher = BinaryOperator(HasOperatorName("||")))
///   !(a || b)
AST_POLYMORPHIC_MATCHER_P(HasOperatorName, std::string, Name) {
  COMPILE_ASSERT(
      (llvm::is_base_of<clang::BinaryOperator, NodeType>::value) ||
      (llvm::is_base_of<clang::UnaryOperator, NodeType>::value),
      has_condition_requires_if_statement_or_conditional_operator);
  return Name == Node.getOpcodeStr(Node.getOpcode());
}

/// Matches the left hand side of binary operator expressions.
///
/// Example matches a (matcher = BinaryOperator(HasLHS()))
///   a || b
AST_MATCHER_P(clang::BinaryOperator, HasLHS,
              Matcher<clang::Expr>, InnerMatcher) {
  clang::Expr *LeftHandSide = Node.getLHS();
  return (LeftHandSide != NULL &&
          InnerMatcher.Matches(*LeftHandSide, Finder, Builder));
}

/// Matches the right hand side of binary operator expressions.
///
/// Example matches b (matcher = BinaryOperator(HasRHS()))
///   a || b
AST_MATCHER_P(clang::BinaryOperator, HasRHS,
              Matcher<clang::Expr>, InnerMatcher) {
  clang::Expr *RightHandSide = Node.getRHS();
  return (RightHandSide != NULL &&
          InnerMatcher.Matches(*RightHandSide, Finder, Builder));
}

/// Matches if either the left hand side or the right hand side of a binary
/// operator matches.
inline Matcher<clang::BinaryOperator> HasEitherOperand(
    const Matcher<clang::Expr> &InnerMatcher) {
  return AnyOf(HasLHS(InnerMatcher), HasRHS(InnerMatcher));
}

/// Matches if the operand of a unary operator matches.
///
/// Example matches true (matcher = HasOperand(BoolLiteral(Equals(true))))
///   !true
AST_MATCHER_P(clang::UnaryOperator, HasUnaryOperand,
              Matcher<clang::Expr>, InnerMatcher) {
  clang::Expr *Operand = Node.getSubExpr();
  return (Operand != NULL &&
          InnerMatcher.Matches(*Operand, Finder, Builder));
}

/// Matches the true branch expression of a conditional operator.
///
/// Example matches a
///   condition ? a : b
AST_MATCHER_P(clang::ConditionalOperator, HasTrueExpression,
              Matcher<clang::Expr>, InnerMatcher) {
  clang::Expr *Expression = Node.getTrueExpr();
  return (Expression != NULL &&
          InnerMatcher.Matches(*Expression, Finder, Builder));
}

/// Matches the false branch expression of a conditional operator.
///
/// Example matches b
///   condition ? a : b
AST_MATCHER_P(clang::ConditionalOperator, HasFalseExpression,
              Matcher<clang::Expr>, InnerMatcher) {
  clang::Expr *Expression = Node.getFalseExpr();
  return (Expression != NULL &&
          InnerMatcher.Matches(*Expression, Finder, Builder));
}

/// Matches if a declaration has a body attached.
///
/// Example matches A, va, fa
///   class A {};
///   class B;  // Doesn't match, as it has no body.
///   int va;
///   extern int vb;  // Doesn't match, as it doesn't define the variable.
///   void fa() {}
///   void fb();  // Doesn't match, as it has no body.
inline PolymorphicMatcherWithParam0<IsDefinitionMatcher> IsDefinition() {
  return PolymorphicMatcherWithParam0<IsDefinitionMatcher>();
}

/// Matches the class declaration that the given method declaration belongs to.
/// TODO(qrczak): Generalize this for other kinds of declarations.
/// FIXME: What other kind of declarations would we need to generalize
/// this to?
///
/// Example matches A() in the last line
///     (matcher = ConstructorCall(HasDeclaration(Method(
///         OfClass(HasName("A"))))))
///   class A {
///    public:
///     A();
///   };
///   A a = A();
AST_MATCHER_P(clang::CXXMethodDecl, OfClass,
              Matcher<clang::CXXRecordDecl>, InnerMatcher) {
  const clang::CXXRecordDecl *Parent = Node.getParent();
  return (Parent != NULL &&
          InnerMatcher.Matches(*Parent, Finder, Builder));
}

/// Matches member expressions that are called with '->' as opposed to '.'.
/// Member calls on the implicit this pointer match as called with '->'.
///
/// Given
///   class Y {
///     void x() { this->x(); x(); Y y; y.x(); a; this->b; Y::b; }
///     int a;
///     static int b;
///   };
/// MemberExpression(IsArrow())
///   matches this->x, x, y.x, a, this->b
inline Matcher<clang::MemberExpr> IsArrow() {
  return MakeMatcher(new IsArrowMatcher());
}

}  // namespace match

/// Runs over an AST and finds matches.
/// FIXME: Define exactly what "one match" is.
///
/// Not intended to be subclassed.
class MatchFinder {
 public:
  struct MatchResult {
    BoundNodes Nodes;

    ///@{
    /// Utilities for interpreting the matched AST structures.
    clang::ASTContext *Context;
    clang::SourceManager *SourceManager;
    ///@}
  };

  /// Called when the Match registered for it was successfully found in the AST.
  class MatchCallback {
   public:
    virtual ~MatchCallback();
    virtual void Run(const MatchResult &Result) = 0;
  };

  /// Called when parsing is finished. Intended for testing only.
  class ParsingDoneTestCallback {
   public:
    virtual ~ParsingDoneTestCallback();
    virtual void Run() = 0;
  };

  MatchFinder();
  ~MatchFinder();

  /// Adds a NodeMatcher to match when running over the AST.
  /// Calls action with the BoundNodes on every match.
  /// Adding more than one InnerMatcher allows finding different matches in a
  /// single pass over the AST.
  void AddMatcher(const Matcher<clang::Decl> &NodeMatch,
                  MatchCallback *Action);
  /// Adds a NodeMatcher to match when running over the AST.
  /// Calls action with the BoundNodes on every match.
  /// Adding more than one InnerMatcher allows finding different matches in a
  /// single pass over the AST.
  void AddMatcher(const Matcher<clang::QualType> &NodeMatch,
                  MatchCallback *Action);
  /// Adds a NodeMatcher to match when running over the AST.
  /// Calls action with the BoundNodes on every match.
  /// Adding more than one InnerMatcher allows finding different matches in a
  /// single pass over the AST.
  void AddMatcher(const Matcher<clang::Stmt> &NodeMatch,
                  MatchCallback *Action);

  /// Finds all matches in the given code and runs the corresponding triggers.
  /// Returns true if the code parsed correctly.
  bool FindAll(const std::string &Code);

  /// Creates a clang FrontendAction factory that finds all matches.
  FrontendActionFactory *NewFrontendActionFactory();

  /// The provided closure is called after parsing is done, before the AST is
  /// traversed. Useful for benchmarking.
  /// Each call to FindAll(...) will call the closure once.
  void RegisterTestCallbackAfterParsing(ParsingDoneTestCallback *ParsingDone);

 private:
  clang::FrontendAction *NewVisitorAction();

  /// The MatchCallback*'s will be called every time the UntypedBaseMatcher
  /// matches on the AST.
  std::vector< std::pair<const UntypedBaseMatcher*, MatchCallback*> > Triggers;

  /// Called when parsing is done.
  ParsingDoneTestCallback *ParsingDone;

  friend class MatchFinderFrontendActionFactory;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_AST_MATCHERS_H
