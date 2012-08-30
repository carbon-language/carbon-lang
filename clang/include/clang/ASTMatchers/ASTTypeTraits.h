//===--- ASTMatchersTypeTraits.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides a dynamically typed node container that can be used to store
//  an AST base node at runtime in the same storage in a type safe way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H
#define LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

namespace clang {
namespace ast_type_traits {

/// \brief A dynamically typed AST node container.
///
/// Stores an AST node in a type safe way.
/// Use \c create(Node) to create a \c DynTypedNode from an AST node,
/// and \c get<T>() to retrieve the node as type T if the types match.
class DynTypedNode {
public:
  /// \brief Creates a NULL-node, which is needed to be able to use
  /// \c DynTypedNodes in STL data structures.
  DynTypedNode() : Tag(), Node(NULL) {}

  /// \brief Creates a \c DynTypedNode from \c Node.
  template <typename T>
  static DynTypedNode create(T Node) {
    return BaseConverter<T>::create(Node);
  }

  /// \brief Retrieve the stored node as type \c T.
  ///
  /// Returns NULL if the stored node does not have a type that is
  /// convertible to \c T.
  template <typename T>
  T get() const {
    return llvm::dyn_cast<typename llvm::remove_pointer<T>::type>(
      BaseConverter<T>::get(Tag, Node));
  }

private:
  /// \brief Takes care of converting from and to \c T.
  template <typename T, typename EnablerT = void> struct BaseConverter;

  /// \brief Supported base node types.
  enum NodeTypeTag {
    NT_Decl,
    NT_Stmt
  } Tag;

  /// \brief Stores the data of the node.
  // FIXME: We really want to store a union, as we want to support
  // storing TypeLoc nodes by-value.
  // FIXME: Add QualType storage: we'll want to use QualType::getAsOpaquePtr()
  // and getFromOpaquePtr(...) to convert to and from void*, but return the
  // QualType objects by value.
  void *Node;

  DynTypedNode(NodeTypeTag Tag, const void *Node)
    : Tag(Tag), Node(const_cast<void*>(Node)) {}
};
template<typename T> struct DynTypedNode::BaseConverter<T,
    typename llvm::enable_if<llvm::is_base_of<
      Decl, typename llvm::remove_pointer<T>::type > >::type > {
  static Decl *get(NodeTypeTag Tag, void *Node) {
    if (Tag == NT_Decl) return static_cast<Decl*>(Node);
    return NULL;
  }
  static DynTypedNode create(const Decl *Node) {
    return DynTypedNode(NT_Decl, Node);
  }
};
template<typename T> struct DynTypedNode::BaseConverter<T,
    typename llvm::enable_if<llvm::is_base_of<
      Stmt, typename llvm::remove_pointer<T>::type > >::type > {
  static Stmt *get(NodeTypeTag Tag, void *Node) {
    if (Tag == NT_Stmt) return static_cast<Stmt*>(Node);
    return NULL;
  }
  static DynTypedNode create(const Stmt *Node) {
    return DynTypedNode(NT_Stmt, Node);
  }
};

} // end namespace ast_type_traits
} // end namespace clang

#endif // LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H

