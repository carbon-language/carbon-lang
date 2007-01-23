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
    Typedef, Function, Variable,
    Struct, Union, Class, Enum
  };

  /// IdentifierNamespace - According to C99 6.2.3, there are four namespaces,
  /// labels, tags, members and ordinary identifiers.
  enum IdentifierNamespace {
    IDNS_Label,
    IDNS_Tag,
    IDNS_Member,
    IDNS_Ordinary
  };
private:
  /// DeclKind - This indicates which class this is.
  Kind DeclKind;
  
  /// Loc - The location that this decl.
  SourceLocation Loc;
  
  /// Identifier - The identifier for this declaration (e.g. the name for the
  /// variable, the tag for a struct).
  IdentifierInfo *Identifier;
  
  /// When this decl is in scope while parsing, the Next field contains a
  /// pointer to the shadowed decl of the same name.  When the scope is popped,
  /// Decls are relinked onto a containing decl object.
  ///
  Decl *Next;
  
public:
  Decl(Kind DK, SourceLocation L, IdentifierInfo *Id)
    : DeclKind(DK), Loc(L), Identifier(Id), Next(0) {}
  virtual ~Decl();
  
  IdentifierInfo *getIdentifier() const { return Identifier; }
  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }
  const char *getName() const;
  
  Kind getKind() const { return DeclKind; }
  Decl *getNext() const { return Next; }
  void setNext(Decl *N) { Next = N; }
  
  IdentifierNamespace getIdentifierNamespace() const {
    switch (DeclKind) {
    default: assert(0 && "Unknown decl kind!");
    case Typedef:
    case Function:
    case Variable:
      return IDNS_Ordinary;
    case Struct:
    case Union:
    case Class:
    case Enum:
      return IDNS_Tag;
    }
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *) { return true; }
};

/// TypeDecl - Common base-class for all type name decls, which as Typedefs and
/// Objective-C classes.
class TypeDecl : public Decl {
  /// Type.  FIXME: This isn't a wonderful place for this.
  TypeRef DeclType;
public:
  TypeDecl(Kind DK, SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : Decl(DK, L, Id), DeclType(T) {}

  TypeRef getUnderlyingType() const { return DeclType; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypeDecl *D) { return true; }
};

class TypedefDecl : public TypeDecl {
public:
  // FIXME: Remove Declarator argument.
  TypedefDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : TypeDecl(Typedef, L, Id, T) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};

/// ObjectDecl - ObjectDecl - Represents a declaration of a value.
class ObjectDecl : public Decl {
  TypeRef DeclType;
protected:
  ObjectDecl(Kind DK, SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : Decl(DK, L, Id), DeclType(T) {}
public:

  TypeRef getType() const { return DeclType; }

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
  VarDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : ObjectDecl(Variable, L, Id, T) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Variable; }
  static bool classof(const VarDecl *D) { return true; }
};

/// FunctionDecl - An instance of this class is created to represent a function
/// declaration or definition.
class FunctionDecl : public ObjectDecl {
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.  TODO: we could allocate this space immediately after the
  /// FunctionDecl object to save an allocation like FunctionType does.
  VarDecl **ParamInfo;
  
  Stmt *Body;  // Null if a prototype.
  
  /// DeclChain - Linked list of declarations that are defined inside this
  /// function.
  Decl *DeclChain;
public:
  FunctionDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : ObjectDecl(Function, L, Id, T), ParamInfo(0), Body(0), DeclChain(0) {}
  virtual ~FunctionDecl();
  
  Stmt *getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }
  
  Decl *getDeclChain() const { return DeclChain; }
  void setDeclChain(Decl *D) { DeclChain = D; }

  unsigned getNumParams() const;
  VarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(VarDecl **NewParamInfo, unsigned NumParams);
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Function; }
  static bool classof(const FunctionDecl *D) { return true; }
};

/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl : public Decl {
  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition;
protected:
  TagDecl(Kind DK, SourceLocation L, IdentifierInfo *Id) : Decl(DK, L, Id) {
    IsDefinition = false;
  }
public:
  
  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }
  void setDefinition(bool V) { IsDefinition = V; }
  
  const char *getKindName() const {
    switch (getKind()) {
    default: assert(0 && "Unknown TagDecl!");
    case Struct: return "struct";
    case Union:  return "union";
    case Class:  return "class";
    case Enum:   return "enum";
    }
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Struct || D->getKind() == Union ||
           D->getKind() == Class || D->getKind() == Enum;
  }
  static bool classof(const ObjectDecl *D) { return true; }
};

/// RecordDecl - Represents a struct/union/class.
class RecordDecl : public TagDecl {
public:
  RecordDecl(Kind DK, SourceLocation L, IdentifierInfo *Id) :TagDecl(DK, L, Id){
    assert(classof(static_cast<Decl*>(this)) && "Invalid Kind!");
  }
  
  static bool classof(const Decl *D) {
    return D->getKind() == Struct || D->getKind() == Union ||
           D->getKind() == Class;
  }
  static bool classof(const RecordDecl *D) { return true; }
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
