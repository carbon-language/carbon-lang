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
  /// Identifier - The identifier for this declaration (e.g. the name for the
  /// variable, the tag for a struct).
  IdentifierInfo *Identifier;
  
  /// Type.
  
  /// DeclKind - This indicates which class this is.
  Kind DeclKind;
  
  /// Scope stack info when parsing, otherwise decl list when scope is popped.
  ///
  Decl *Next;
  
public:
  Decl(IdentifierInfo *Id, Kind DK, Decl *next)
    : Identifier(Id), DeclKind(DK), Next(next) {}
  virtual ~Decl();
  
  const IdentifierInfo *getIdentifier() const { return Identifier; }

  Kind getKind() const { return DeclKind; }
  
  Decl *getNext() const { return Next; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *) { return true; }
};

/// TypeDecl - Common base-class for all type name decls, which as Typedefs and
/// Objective-C classes.
class TypeDecl : public Decl {
public:
  TypeDecl(IdentifierInfo *Id, Kind DK, Decl *Next)
    : Decl(Id, DK, Next) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypeDecl *D) { return true; }
};

class TypedefDecl : public TypeDecl {
public:
  // FIXME: Remove Declarator argument.
  TypedefDecl(IdentifierInfo *Id, Decl *Next)
    : TypeDecl(Id, Typedef, Next) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};

/// FunctionDecl - An instance of this class is created to represent a function
/// declaration or definition.
class FunctionDecl : public Decl {
  // Args etc.
  Stmt *Body;  // Null if a prototype.
public:
  FunctionDecl(IdentifierInfo *Id, Decl *Next)
    : Decl(Id, Function, Next), Body(0) {}
  
  Stmt *getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }
  
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Function; }
  static bool classof(const FunctionDecl *D) { return true; }
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public Decl {
  // Initializer.
public:
  VarDecl(IdentifierInfo *Id, Decl *Next)
    : Decl(Id, Variable, Next) {}
  
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Variable; }
  static bool classof(const VarDecl *D) { return true; }
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
