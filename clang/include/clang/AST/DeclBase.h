//===-- DeclBase.h - Base Classes for representing declarations *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl and ContextDecl interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLBASE_H
#define LLVM_CLANG_AST_DECLBASE_H

#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class FunctionDecl;
class ObjCMethodDecl;
class TagDecl;
class ObjCInterfaceDecl;

/// Decl - This represents one declaration (or definition), e.g. a variable, 
/// typedef, function, struct, etc.  
///
class Decl {
public:
  enum Kind {
    // This lists the concrete classes of Decl in order of the inheritance
    // hierarchy.  This allows us to do efficient classof tests based on the
    // enums below.   The commented out names are abstract class names.
    
    // Decl
    //   NamedDecl
           Field,
             ObjCIvar,
           ObjCCategory,
           ObjCCategoryImpl,
           ObjCImplementation,
           ObjCProtocol,
           PropertyDecl,
    //     ScopedDecl
    //       TypeDecl
               Typedef,
    //         TagDecl
                 Enum,
    //           RecordDecl
                   Struct,
                   Union,
                   Class,
    //       ValueDecl
               EnumConstant,
               Function,
    //         VarDecl
                 BlockVar,
                 FileVar,
                 ParmVar,
         ObjCInterface,
         ObjCCompatibleAlias,
         ObjCMethod,
         ObjCClass,
         ObjCForwardProtocol,
         LinkageSpec,
   FileScopeAsm,
  
    // For each non-leaf class, we now define a mapping to the first/last member
    // of the class, to allow efficient classof.
    NamedFirst  = Field,         NamedLast  = ParmVar,
    FieldFirst  = Field,         FieldLast  = ObjCIvar,
    ScopedFirst = Typedef,       ScopedLast = ParmVar,
    TypeFirst   = Typedef,       TypeLast   = Class,
    TagFirst    = Enum         , TagLast    = Class,
    RecordFirst = Struct       , RecordLast = Class,
    ValueFirst  = EnumConstant , ValueLast  = ParmVar,
    VarFirst    = BlockVar     , VarLast    = ParmVar
  };

  /// IdentifierNamespace - According to C99 6.2.3, there are four namespaces,
  /// labels, tags, members and ordinary identifiers.
  enum IdentifierNamespace {
    IDNS_Label,
    IDNS_Tag,
    IDNS_Member,
    IDNS_Ordinary
  };
  
  /// ObjCDeclQualifier - Qualifier used on types in method declarations
  /// for remote messaging. They are meant for the arguments though and
  /// applied to the Decls (ObjCMethodDecl and ParmVarDecl).
  enum ObjCDeclQualifier {
    OBJC_TQ_None = 0x0,
    OBJC_TQ_In = 0x1,
    OBJC_TQ_Inout = 0x2,
    OBJC_TQ_Out = 0x4,
    OBJC_TQ_Bycopy = 0x8,
    OBJC_TQ_Byref = 0x10,
    OBJC_TQ_Oneway = 0x20
  };
    
private:
  /// Loc - The location that this decl.
  SourceLocation Loc;
  
  /// DeclKind - This indicates which class this is.
  Kind DeclKind   :  8;
  
  /// InvalidDecl - This indicates a semantic error occurred.
  unsigned int InvalidDecl :  1;
  
  /// HasAttrs - This indicates whether the decl has attributes or not.
  unsigned int HasAttrs : 1;
protected:
  Decl(Kind DK, SourceLocation L) : Loc(L), DeclKind(DK), InvalidDecl(0),
    HasAttrs(false) {
    if (Decl::CollectingStats()) addDeclKind(DK);
  }
  
public:
  // TODO: This should probably be made protected once derived classes have
  // destructors.
  virtual ~Decl();
  
  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  Kind getKind() const { return DeclKind; }
  const char *getDeclKindName() const;
  
  void addAttr(Attr *attr);
  const Attr *getAttrs() const;

  template<typename T> const T *getAttr() const {
    for (const Attr *attr = getAttrs(); attr; attr = attr->getNext())
      if (const T *V = dyn_cast<T>(attr))
        return V;

    return 0;
  }
    
  /// setInvalidDecl - Indicates the Decl had a semantic error. This
  /// allows for graceful error recovery.
  void setInvalidDecl() { InvalidDecl = 1; }
  bool isInvalidDecl() const { return (bool) InvalidDecl; }
  
  IdentifierNamespace getIdentifierNamespace() const {
    switch (DeclKind) {
    default: assert(0 && "Unknown decl kind!");
    case Typedef:
    case Function:
    case BlockVar:
    case FileVar:
    case ParmVar:
    case EnumConstant:
    case ObjCInterface:
    case ObjCCompatibleAlias:
      return IDNS_Ordinary;
    case Struct:
    case Union:
    case Class:
    case Enum:
      return IDNS_Tag;
    }
  }
  // global temp stats (until we have a per-module visitor)
  static void addDeclKind(Kind k);
  static bool CollectingStats(bool Enable = false);
  static void PrintStats();
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *) { return true; }
  
  /// Emit - Serialize this Decl to Bitcode.
  void Emit(llvm::Serializer& S) const;
    
  /// Create - Deserialize a Decl from Bitcode.
  static Decl* Create(llvm::Deserializer& D);

protected:
  /// EmitImpl - Provides the subclass-specific serialization logic for
  ///   serializing out a decl.
  virtual void EmitImpl(llvm::Serializer& S) const {
    // FIXME: This will eventually be a pure virtual function.
    assert (false && "Not implemented.");
  }
  
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D);
};

/// ContextDecl - This is used only as base class of specific decl types that
/// can act as declaration contexts. These decls are:
///
///   FunctionDecl
///   ObjCMethodDecl
///   TagDecl
///   ObjCInterfaceDecl
///
class ContextDecl {
  /// DeclKind - This indicates which class this is.
  Decl::Kind DeclKind   :  8;

  // Used in the CastTo template to get the DeclKind
  // from a Decl or a ContextDecl. ContextDecl doesn't have a getKind() method
  // to avoid 'ambiguous access' compiler errors.
  template<typename T> struct KindTrait {
    static Decl::Kind getKind(const T *D) { return D->getKind(); }
  };

  // Used only by the ToDecl and FromDecl methods
  template<typename To, typename From>
  static To *CastTo(const From *D) {
    Decl::Kind DK = KindTrait<From>::getKind(D);
    switch(DK) {
      case Decl::Function:
        return static_cast<FunctionDecl*>(const_cast<From*>(D));
      case Decl::ObjCMethod:
        return static_cast<ObjCMethodDecl*>(const_cast<From*>(D));
      case Decl::ObjCInterface:
        return static_cast<ObjCInterfaceDecl*>(const_cast<From*>(D));
      default:
        // check for TagDecl
        if (DK >= Decl::TagFirst && DK <= Decl::TagLast)
          return static_cast<TagDecl*>(const_cast<From*>(D));
 
        assert(false && "a decl that inherits ContextDecl isn't handled");
        return 0;
    }
  }

protected:
  ContextDecl(Decl::Kind K) : DeclKind(K) {}

public:
  /// getParent - Returns the containing ContextDecl if this is a ScopedDecl,
  /// else returns NULL.
  ContextDecl *getParent() const;

  bool isFunctionOrMethod() const {
    switch (DeclKind) {
      case Decl::Function:
      case Decl::ObjCMethod:
        return true;
      default:
        return false;
    }
  }

  /// ToDecl and FromDecl make Decl <-> ContextDecl castings.
  /// They are intended to be used by the simplify_type and cast_convert_val
  /// templates.
  static Decl        *ToDecl   (const ContextDecl *D);
  static ContextDecl *FromDecl (const Decl *D);

  static bool classof(const Decl *D) {
    switch (D->getKind()) {
      case Decl::Function:
      case Decl::ObjCMethod:
      case Decl::ObjCInterface:
        return true;
      default:
        // check for TagDecl
        return D->getKind() >= Decl::TagFirst &&
               D->getKind() <= Decl::TagLast;
    }
  }
  static bool classof(const ContextDecl *D) { return true; }
  static bool classof(const FunctionDecl *D) { return true; }
  static bool classof(const ObjCMethodDecl *D) { return true; }
  static bool classof(const TagDecl *D) { return true; }
  static bool classof(const ObjCInterfaceDecl *D) { return true; }
};

template<> struct ContextDecl::KindTrait<ContextDecl> {
  static Decl::Kind getKind(const ContextDecl *D) { return D->DeclKind; }
};

} // end clang.

namespace llvm {
/// Implement simplify_type for ContextDecl, so that we can dyn_cast from 
/// ContextDecl to a specific Decl class.
  template<> struct simplify_type<const ::clang::ContextDecl*> {
  typedef ::clang::Decl* SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::ContextDecl *Val) {
    return ::clang::ContextDecl::ToDecl(Val);
  }
};
template<> struct simplify_type< ::clang::ContextDecl*>
  : public simplify_type<const ::clang::ContextDecl*> {};

template<> struct simplify_type<const ::clang::ContextDecl> {
  typedef ::clang::Decl SimpleType;
  static SimpleType &getSimplifiedValue(const ::clang::ContextDecl &Val) {
    return *::clang::ContextDecl::ToDecl(&Val);
  }
};
template<> struct simplify_type< ::clang::ContextDecl>
  : public simplify_type<const ::clang::ContextDecl> {};

/// Implement cast_convert_val for ContextDecl, so that we can dyn_cast from 
/// a Decl class to ContextDecl.
template<class FromTy>
struct cast_convert_val< ::clang::ContextDecl,const FromTy,const FromTy> {
  static ::clang::ContextDecl &doit(const FromTy &Val) {
    return *::clang::ContextDecl::FromDecl(&Val);
  }
};
template<class FromTy>
struct cast_convert_val< ::clang::ContextDecl,FromTy,FromTy>
  : public cast_convert_val< ::clang::ContextDecl,const FromTy,const FromTy>
    {};

template<class FromTy>
struct cast_convert_val< ::clang::ContextDecl,const FromTy*,const FromTy*> {
  static ::clang::ContextDecl *doit(const FromTy *Val) {
    return ::clang::ContextDecl::FromDecl(Val);
  }
};
template<class FromTy>
struct cast_convert_val< ::clang::ContextDecl,FromTy*,FromTy*> 
  : public cast_convert_val< ::clang::ContextDecl,const FromTy*,const FromTy*>
    {};

} // end namespace llvm

#endif
