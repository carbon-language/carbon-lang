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
class Expr;
class Stmt;
class FunctionDecl;
  
/// Decl - This represents one declaration (or definition), e.g. a variable, 
/// typedef, function, struct, etc.  
///
class Decl {
public:
  enum Kind {
    Typedef, Function, Variable, Field, EnumConstant,
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


/// FunctionDecl - An instance of this class is created to represent a function
/// declaration or definition.
class FieldDecl : public ObjectDecl {
public:
  FieldDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : ObjectDecl(Field, L, Id, T) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Field;
  }
  static bool classof(const FieldDecl *D) { return true; }
};

/// EnumConstantDecl - An instance of this object exists for each enum constant
/// that is defined.  For example, in "enum X {a,b}", each of a/b are
/// EnumConstantDecl's, X is an instance of EnumDecl, and the type of a/b is a
/// TaggedType for the X EnumDecl.
class EnumConstantDecl : public ObjectDecl {
public:
  // FIXME: Capture value info.
  EnumConstantDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : ObjectDecl(EnumConstant, L, Id, T) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == EnumConstant;
  }
  static bool classof(const EnumConstantDecl *D) { return true; }
  
};



/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl : public Decl {
  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;
protected:
  TagDecl(Kind DK, SourceLocation L, IdentifierInfo *Id) : Decl(DK, L, Id) {
    IsDefinition = false;
  }
public:
  
  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }
  
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
protected:
  void setDefinition(bool V) { IsDefinition = V; }
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl {
  /// Elements/NumElements - This is a new[]'d array of pointers to
  /// EnumConstantDecls.
  EnumConstantDecl **Elements;   // Null if not defined.
  int NumElements;   // -1 if not defined.
public:
  EnumDecl(SourceLocation L, IdentifierInfo *Id) : TagDecl(Enum, L, Id) {
    Elements = 0;
    NumElements = -1;
  }
  
  /// defineElements - When created, EnumDecl correspond to a forward declared
  /// enum.  This method is used to mark the decl as being defined, with the
  /// specified contents.
  void defineElements(EnumConstantDecl **Elements, unsigned NumElements);
  
  static bool classof(const Decl *D) {
    return D->getKind() == Enum;
  }
  static bool classof(const EnumDecl *D) { return true; }
};


/// RecordDecl - Represents a struct/union/class.
class RecordDecl : public TagDecl {
  /// HasFlexibleArrayMember - This is true if this struct ends with a flexible
  /// array member (e.g. int X[]) or if this union contains a struct that does.
  /// If so, this cannot be contained in arrays or other structs as a member.
  bool HasFlexibleArrayMember : 1;

  /// Members/NumMembers - This is a new[]'d array of pointers to Decls.
  Decl **Members;   // Null if not defined.
  int NumMembers;   // -1 if not defined.
public:
  RecordDecl(Kind DK, SourceLocation L, IdentifierInfo *Id) :TagDecl(DK, L, Id){
    HasFlexibleArrayMember = false;
    assert(classof(static_cast<Decl*>(this)) && "Invalid Kind!");
    Members = 0;
    NumMembers = -1;
  }
  
  bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
  void setHasFlexibleArrayMember(bool V) { HasFlexibleArrayMember = V; }

  /// defineBody - When created, RecordDecl's correspond to a forward declared
  /// record.  This method is used to mark the decl as being defined, with the
  /// specified contents.
  void defineBody(Decl **Members, unsigned numMembers);
  
  static bool classof(const Decl *D) {
    return D->getKind() == Struct || D->getKind() == Union ||
           D->getKind() == Class;
  }
  static bool classof(const RecordDecl *D) { return true; }
};

}  // end namespace clang
}  // end namespace llvm

#endif
