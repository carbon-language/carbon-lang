//===-- DeclFriend.h - Classes for C++ friend declarations -*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the section of the AST representing C++ friend
// declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLFRIEND_H
#define LLVM_CLANG_AST_DECLFRIEND_H

#include "clang/AST/DeclCXX.h"

namespace clang {

/// FriendDecl - Represents the declaration of a friend entity,
/// which can be a function, a type, or a templated function or type.
//  For example:
///
/// @code
/// template <typename T> class A {
///   friend int foo(T);
///   friend class B;
///   friend T; // only in C++0x
///   template <typename U> friend class C;
///   template <typename U> friend A& operator+=(A&, const U&) { ... }
/// };
/// @endcode
///
/// The semantic context of a friend decl is its declaring class.
class FriendDecl : public Decl {
public:
  typedef llvm::PointerUnion<NamedDecl*,Type*> FriendUnion;

private:
  // The declaration that's a friend of this class.
  FriendUnion Friend;

  // Location of the 'friend' specifier.
  SourceLocation FriendLoc;

  // FIXME: Hack to keep track of whether this was a friend function
  // template specialization.
  bool WasSpecialization;

  FriendDecl(DeclContext *DC, SourceLocation L, FriendUnion Friend,
             SourceLocation FriendL)
    : Decl(Decl::Friend, DC, L),
      Friend(Friend),
      FriendLoc(FriendL),
      WasSpecialization(false) {
  }

public:
  static FriendDecl *Create(ASTContext &C, DeclContext *DC,
                            SourceLocation L, FriendUnion Friend_,
                            SourceLocation FriendL);

  /// If this friend declaration names an (untemplated but
  /// possibly dependent) type, return the type;  otherwise
  /// return null.  This is used only for C++0x's unelaborated
  /// friend type declarations.
  Type *getFriendType() const {
    return Friend.dyn_cast<Type*>();
  }

  /// If this friend declaration doesn't name an unelaborated
  /// type, return the inner declaration.
  NamedDecl *getFriendDecl() const {
    return Friend.dyn_cast<NamedDecl*>();
  }

  /// Retrieves the location of the 'friend' keyword.
  SourceLocation getFriendLoc() const {
    return FriendLoc;
  }

  bool wasSpecialization() const { return WasSpecialization; }
  void setSpecialization(bool WS) { WasSpecialization = WS; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const FriendDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Decl::Friend; }
};
  
}

#endif
