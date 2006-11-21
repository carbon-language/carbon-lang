//===--- Decl.h - Classes for representing declarations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECL_H
#define LLVM_CLANG_AST_DECL_H

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Type.h"

namespace llvm {
namespace clang {
class IdentifierInfo;
class Stmt;
class FunctionDecl;
  
/// Decl - This represents one declaration (or definition), e.g. a variable, 
/// typedef, function, struct, etc.  
///
class Decl {
public:
  enum Kind {
    Typedef, Function, Variable
  };
private:
  /// DeclKind - This indicates which class this is.
  Kind DeclKind;
  
  /// Identifier - The identifier for this declaration (e.g. the name for the
  /// variable, the tag for a struct).
  IdentifierInfo *Identifier;
  
  /// Type.
  TypeRef DeclType;
  
  /// When this decl is in scope while parsing, the Next field contains a
  /// pointer to the shadowed decl of the same name.  When the scope is popped,
  /// Decls are relinked onto a containing decl object.
  ///
  Decl *Next;
  
public:
  Decl(Kind DK, IdentifierInfo *Id, TypeRef T, Decl *next)
    : DeclKind(DK), Identifier(Id), DeclType(T), Next(next) {}
  virtual ~Decl();
  
  const IdentifierInfo *getIdentifier() const { return Identifier; }
  const char *getName() const;
  
  TypeRef getType() const { return DeclType; }
  Kind getKind() const { return DeclKind; }
  Decl *getNext() const { return Next; }
  void setNext(Decl *N) { Next = N; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *) { return true; }
};

/// TypeDecl - Common base-class for all type name decls, which as Typedefs and
/// Objective-C classes.
class TypeDecl : public Decl {
public:
  TypeDecl(Kind DK, IdentifierInfo *Id, TypeRef T, Decl *Next)
    : Decl(DK, Id, T, Next) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypeDecl *D) { return true; }
};

class TypedefDecl : public TypeDecl {
public:
  // FIXME: Remove Declarator argument.
  TypedefDecl(IdentifierInfo *Id, TypeRef T, Decl *Next)
    : TypeDecl(Typedef, Id, T, Next) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};

/// ObjectDecl - ObjectDecl - Represents a declaration of a value.
class ObjectDecl : public Decl {
protected:
  ObjectDecl(Kind DK, IdentifierInfo *Id, TypeRef T, Decl *Next)
    : Decl(DK, Id, T, Next) {}
public:
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Variable || D->getKind() == Function;
  }
  static bool classof(const ObjectDecl *D) { return true; }
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public ObjectDecl {
  // TODO: Initializer.
public:
  VarDecl(IdentifierInfo *Id, TypeRef T, Decl *Next)
    : ObjectDecl(Variable, Id, T, Next) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Variable; }
  static bool classof(const VarDecl *D) { return true; }
};

/// FunctionDecl - An instance of this class is created to represent a function
/// declaration or definition.
class FunctionDecl : public ObjectDecl {
  // Args etc.
  Stmt *Body;  // Null if a prototype.
  
  /// DeclChain - Linked list of declarations that are defined inside this
  /// function.
  Decl *DeclChain;
public:
  FunctionDecl(IdentifierInfo *Id, TypeRef T, Decl *Next)
    : ObjectDecl(Function, Id, T, Next), Body(0), DeclChain(0) {}
  
  Stmt *getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }
  
  Decl *getDeclChain() const { return DeclChain; }
  void setDeclChain(Decl *D) { DeclChain = D; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Function; }
  static bool classof(const FunctionDecl *D) { return true; }
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
