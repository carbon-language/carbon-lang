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
    // Concrete sub-classes of ValueDecl
    Function, BlockVariable, FileVariable, ParmVariable, EnumConstant,
    // Concrete sub-classes of TypeDecl
    Typedef, Struct, Union, Class, Enum, 
    // Concrete sub-class of Decl
    Field
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
  
protected:
  Decl(Kind DK, SourceLocation L, IdentifierInfo *Id)
    : DeclKind(DK), Loc(L), Identifier(Id), Next(0) {}
  virtual ~Decl();
  
public:
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
    case BlockVariable:
    case FileVariable:
    case ParmVariable:
    case EnumConstant:
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

/// ValueDecl - Represent the declaration of a variable (in which case it is 
/// an lvalue) a function (in which case it is a function designator) or
/// an enum constant. 
class ValueDecl : public Decl {
  TypeRef DeclType;
protected:
  ValueDecl(Kind DK, SourceLocation L, IdentifierInfo *Id, TypeRef T): 
             Decl(DK, L, Id), DeclType(T) {}
public:
  // FIXME: should rename to getTypeRef/getCanonicalTypeRef to distinguish
  // TypeRef's from Type's...
  TypeRef getType() const { return DeclType; }
  TypeRef getCanonicalType() const { return DeclType.getCanonicalType(); }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= Function && D->getKind() <= EnumConstant;
  }
  static bool classof(const ValueDecl *D) { return true; }
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public ValueDecl {
public:
  enum StorageClass {
    None, Extern, Static, Auto, Register
  };
  StorageClass getStorageClass() const { return SClass; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() >= BlockVariable && D->getKind() <= ParmVariable; 
  }
  static bool classof(const VarDecl *D) { return true; }
protected:
  VarDecl(Kind DK, SourceLocation L, IdentifierInfo *Id, TypeRef T,
          StorageClass SC)
    : ValueDecl(DK, L, Id, T) {}
private:
  StorageClass SClass;
  // TODO: Initializer.
};

/// BlockVarDecl - Represent a local variable declaration.
class BlockVarDecl : public VarDecl {
public:
  BlockVarDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T, StorageClass S)
    : VarDecl(BlockVariable, L, Id, T, S) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == BlockVariable; }
  static bool classof(const BlockVarDecl *D) { return true; }
};

/// FileVarDecl - Represent a file scoped variable declaration. This
/// will allow us to reason about external variable declarations and tentative 
/// definitions (C99 6.9.2p2) using our type system (without storing a
/// pointer to the decl's scope, which is transient).
class FileVarDecl : public VarDecl {
public:
  FileVarDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T, StorageClass S)
    : VarDecl(FileVariable, L, Id, T, S) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == FileVariable; }
  static bool classof(const FileVarDecl *D) { return true; }
};

/// ParmVarDecl - Represent a parameter to a function.
class ParmVarDecl : public VarDecl {
public:
  ParmVarDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T, StorageClass S)
    : VarDecl(ParmVariable, L, Id, T, S) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ParmVariable; }
  static bool classof(const ParmVarDecl *D) { return true; }
};

/// FunctionDecl - An instance of this class is created to represent a function
/// declaration or definition.
class FunctionDecl : public ValueDecl {
public:
  enum StorageClass {
    None, Extern, Static
  };
  FunctionDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T, StorageClass S=None)
    : ValueDecl(Function, L, Id, T), 
      ParamInfo(0), Body(0), DeclChain(0), SClass(S) {}
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

  StorageClass getStorageClass() const { return SClass; }
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Function; }
  static bool classof(const FunctionDecl *D) { return true; }
private:
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.  TODO: we could allocate this space immediately after the
  /// FunctionDecl object to save an allocation like FunctionType does.
  VarDecl **ParamInfo;
  
  Stmt *Body;  // Null if a prototype.
  
  /// DeclChain - Linked list of declarations that are defined inside this
  /// function.
  Decl *DeclChain;

  StorageClass SClass;
};


/// FieldDecl - An instance of this class is created by Sema::ParseField to 
/// represent a member of a struct/union/class.
class FieldDecl : public Decl {
  TypeRef DeclType;
public:
  FieldDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : Decl(Field, L, Id), DeclType(T) {}

  TypeRef getType() const { return DeclType; }
  TypeRef getCanonicalType() const { return DeclType.getCanonicalType(); }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Field;
  }
  static bool classof(const FieldDecl *D) { return true; }
};

/// EnumConstantDecl - An instance of this object exists for each enum constant
/// that is defined.  For example, in "enum X {a,b}", each of a/b are
/// EnumConstantDecl's, X is an instance of EnumDecl, and the type of a/b is a
/// TagType for the X EnumDecl.
class EnumConstantDecl : public ValueDecl {
public:
  // FIXME: Capture value info.
  EnumConstantDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : ValueDecl(EnumConstant, L, Id, T) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == EnumConstant;
  }
  static bool classof(const EnumConstantDecl *D) { return true; }
};


/// TypeDecl - Represents a declaration of a type.
///
class TypeDecl : public Decl {
  /// TypeForDecl - This indicates the Type object that represents this
  /// TypeDecl.  It is a cache maintained by ASTContext::getTypedefType and
  /// ASTContext::getTagDeclType.
  Type *TypeForDecl;
  friend class ASTContext;
protected:
  TypeDecl(Kind DK, SourceLocation L, IdentifierInfo *Id)
    : Decl(DK, L, Id), TypeForDecl(0) {}
public:
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= Typedef && D->getKind() <= Enum;
  }
  static bool classof(const TypeDecl *D) { return true; }
};


class TypedefDecl : public TypeDecl {
  /// UnderlyingType - This is the type the typedef is set to.
  TypeRef UnderlyingType;
public:
    // FIXME: Remove Declarator argument.
    TypedefDecl(SourceLocation L, IdentifierInfo *Id, TypeRef T)
    : TypeDecl(Typedef, L, Id), UnderlyingType(T) {}
  
  TypeRef getUnderlyingType() const { return UnderlyingType; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};


/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl : public TypeDecl {
  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;
protected:
  TagDecl(Kind DK, SourceLocation L, IdentifierInfo *Id) : TypeDecl(DK, L, Id) {
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
  static bool classof(const TagDecl *D) { return true; }
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
  FieldDecl **Members;   // Null if not defined.
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
  void defineBody(FieldDecl **Members, unsigned numMembers);

  /// getMember - If the member doesn't exist, or there are no members, this 
  /// function will return 0;
  FieldDecl *getMember(IdentifierInfo *name);

  static bool classof(const Decl *D) {
    return D->getKind() == Struct || D->getKind() == Union ||
           D->getKind() == Class;
  }
  static bool classof(const RecordDecl *D) { return true; }
};

}  // end namespace clang
}  // end namespace llvm

#endif
