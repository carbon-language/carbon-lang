//===-- DeclBase.h - Base Classes for representing declarations *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl and DeclContext interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLBASE_H
#define LLVM_CLANG_AST_DECLBASE_H

#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class DeclContext;
class TranslationUnitDecl;
class NamespaceDecl;
class ScopedDecl;
class FunctionDecl;
class CXXRecordDecl;
class EnumDecl;
class ObjCMethodDecl;
class ObjCInterfaceDecl;
class BlockDecl;

/// Decl - This represents one declaration (or definition), e.g. a variable, 
/// typedef, function, struct, etc.  
///
class Decl {
public:
  enum Kind {
    // This lists the concrete classes of Decl in order of the inheritance
    // hierarchy.  This allows us to do efficient classof tests based on the
    // enums below.   The commented out names are abstract class names.
    // [DeclContext] indicates that the class also inherits from DeclContext.
    
    // Decl
         TranslationUnit,  // [DeclContext]
    //   NamedDecl
           Field,
             CXXField,
             ObjCIvar,
             ObjCAtDefsField,
           OverloadedFunction,
           ObjCCategory,
           ObjCCategoryImpl,
           ObjCImplementation,
           ObjCProtocol,
           ObjCProperty,
    //     ScopedDecl
             Namespace,  // [DeclContext]
    //       TypeDecl
               Typedef,
    //         TagDecl
                 Enum,  // [DeclContext]
                 Record,
                   CXXRecord,  // [DeclContext]
    //       ValueDecl
               EnumConstant,
               Function,  // [DeclContext]
                 CXXMethod,
                   CXXConstructor,
                   CXXDestructor,
                   CXXConversion,
               Var,
                 ImplicitParam,
                 CXXClassVar,
                 ParmVar,
           ObjCInterface,  // [DeclContext]
           ObjCCompatibleAlias,
           ObjCMethod,  // [DeclContext]
           ObjCClass,
           ObjCForwardProtocol,
           ObjCPropertyImpl,
         LinkageSpec,
         FileScopeAsm,
	     Block, // [DeclContext]
  
    // For each non-leaf class, we now define a mapping to the first/last member
    // of the class, to allow efficient classof.
    NamedFirst     = Field        , NamedLast     = ParmVar,
    FieldFirst     = Field        , FieldLast     = ObjCAtDefsField,
    ScopedFirst    = Namespace    , ScopedLast    = ParmVar,
    TypeFirst      = Typedef      , TypeLast      = CXXRecord,
    TagFirst       = Enum         , TagLast       = CXXRecord,
    RecordFirst    = Record       , RecordLast    = CXXRecord,
    ValueFirst     = EnumConstant , ValueLast     = ParmVar,
    FunctionFirst  = Function     , FunctionLast  = CXXConversion,
    VarFirst       = Var          , VarLast       = ParmVar
  };

  /// IdentifierNamespace - According to C99 6.2.3, there are four namespaces,
  /// labels, tags, members and ordinary identifiers. These are meant
  /// as bitmasks, so that searches in C++ can look into the "tag" namespace
  /// during ordinary lookup.
  enum IdentifierNamespace {
    IDNS_Label = 0x1,
    IDNS_Tag = 0x2,
    IDNS_Member = 0x4,
    IDNS_Ordinary = 0x8
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
  /// Access - Used by C++ decls for the access specifier.
  // NOTE: VC++ treats enums as signed, avoid using the AccessSpecifier enum
  unsigned Access : 2;
  friend class CXXClassMemberWrapper;

  Decl(Kind DK, SourceLocation L) : Loc(L), DeclKind(DK), InvalidDecl(0),
    HasAttrs(false) {
    if (Decl::CollectingStats()) addDeclKind(DK);
  }

  virtual ~Decl();

public:
  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  Kind getKind() const { return DeclKind; }
  const char *getDeclKindName() const;
  
  void addAttr(Attr *attr);
  const Attr *getAttrs() const;
  void swapAttrs(Decl *D);
  void invalidateAttrs();

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
    case ImplicitParam:
    case Typedef:
    case Function:
    case Var:
    case ParmVar:
    case EnumConstant:
    case ObjCInterface:
    case ObjCCompatibleAlias:
    case OverloadedFunction:
    case CXXField:
    case CXXMethod:
    case CXXClassVar:
      return IDNS_Ordinary;
    case Record:
    case CXXRecord:
    case Enum:
      return IDNS_Tag;
    case Namespace:
      return IdentifierNamespace(IDNS_Tag | IDNS_Ordinary);
    }
  }
  
  // getBody - If this Decl represents a declaration for a body of code,
  //  such as a function or method definition, this method returns the top-level
  //  Stmt* of that body.  Otherwise this method returns null.  
  virtual Stmt* getBody() const { return 0; }
  
  // global temp stats (until we have a per-module visitor)
  static void addDeclKind(Kind k);
  static bool CollectingStats(bool Enable = false);
  static void PrintStats();
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *) { return true; }
  static DeclContext *castToDeclContext(const Decl *);
  static Decl *castFromDeclContext(const DeclContext *);
  
  /// Emit - Serialize this Decl to Bitcode.
  void Emit(llvm::Serializer& S) const;
    
  /// Create - Deserialize a Decl from Bitcode.
  static Decl* Create(llvm::Deserializer& D, ASTContext& C);

  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

protected:
  /// EmitImpl - Provides the subclass-specific serialization logic for
  ///   serializing out a decl.
  virtual void EmitImpl(llvm::Serializer& S) const {
    // FIXME: This will eventually be a pure virtual function.
    assert (false && "Not implemented.");
  }
  
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
};

/// DeclContext - This is used only as base class of specific decl types that
/// can act as declaration contexts. These decls are:
///
///   TranslationUnitDecl
///   NamespaceDecl
///   FunctionDecl
///   CXXRecordDecl
///   EnumDecl
///   ObjCMethodDecl
///   ObjCInterfaceDecl
///   BlockDecl
///
class DeclContext {
  /// DeclKind - This indicates which class this is.
  Decl::Kind DeclKind   :  8;

  /// DeclChain - Linked list of declarations that are defined inside this
  /// declaration context.
  ScopedDecl *DeclChain;

  // Used in the CastTo template to get the DeclKind
  // from a Decl or a DeclContext. DeclContext doesn't have a getKind() method
  // to avoid 'ambiguous access' compiler errors.
  template<typename T> struct KindTrait {
    static Decl::Kind getKind(const T *D) { return D->getKind(); }
  };

  // Used only by the ToDecl and FromDecl methods
  template<typename To, typename From>
  static To *CastTo(const From *D) {
    Decl::Kind DK = KindTrait<From>::getKind(D);
    switch(DK) {
      case Decl::Block:
        return static_cast<BlockDecl*>(const_cast<From*>(D));
      case Decl::TranslationUnit:
        return static_cast<TranslationUnitDecl*>(const_cast<From*>(D));
      case Decl::Namespace:
        return static_cast<NamespaceDecl*>(const_cast<From*>(D));
      case Decl::Enum:
        return static_cast<EnumDecl*>(const_cast<From*>(D));
      case Decl::CXXRecord:
        return static_cast<CXXRecordDecl*>(const_cast<From*>(D));
      case Decl::ObjCMethod:
        return static_cast<ObjCMethodDecl*>(const_cast<From*>(D));
      case Decl::ObjCInterface:
        return static_cast<ObjCInterfaceDecl*>(const_cast<From*>(D));
      default:
        if (DK >= Decl::FunctionFirst && DK <= Decl::FunctionLast)
          return static_cast<FunctionDecl*>(const_cast<From*>(D));

        assert(false && "a decl that inherits DeclContext isn't handled");
        return 0;
    }
  }

protected:
  DeclContext(Decl::Kind K) : DeclKind(K), DeclChain(0) {}

public:
  /// getParent - Returns the containing DeclContext if this is a ScopedDecl,
  /// else returns NULL.
  DeclContext *getParent();
  const DeclContext *getParent() const {
    return const_cast<DeclContext*>(this)->getParent();
  }

  bool isFunctionOrMethod() const {
    switch (DeclKind) {
      case Decl::Block:
      case Decl::Function:
      case Decl::CXXMethod:
      case Decl::ObjCMethod:
        return true;
      default:
        return false;
    }
  }

  const ScopedDecl *getDeclChain() const { return DeclChain; }
  ScopedDecl *getDeclChain() { return DeclChain; }
  void setDeclChain(ScopedDecl *D) { DeclChain = D; }

  static bool classof(const Decl *D) {
    switch (D->getKind()) {
      case Decl::TranslationUnit:
      case Decl::Namespace:
      case Decl::Enum:
      case Decl::CXXRecord:
      case Decl::ObjCMethod:
      case Decl::ObjCInterface:
      case Decl::Block:
        return true;
      default:
        if (D->getKind() >= Decl::FunctionFirst &&
            D->getKind() <= Decl::FunctionLast)
          return true;
        return false;
    }
  }
  static bool classof(const DeclContext *D) { return true; }
  static bool classof(const TranslationUnitDecl *D) { return true; }
  static bool classof(const NamespaceDecl *D) { return true; }
  static bool classof(const FunctionDecl *D) { return true; }
  static bool classof(const CXXRecordDecl *D) { return true; }
  static bool classof(const EnumDecl *D) { return true; }
  static bool classof(const ObjCMethodDecl *D) { return true; }
  static bool classof(const ObjCInterfaceDecl *D) { return true; }
  static bool classof(const BlockDecl *D) { return true; }

private:
  void EmitOutRec(llvm::Serializer& S) const;
  void ReadOutRec(llvm::Deserializer& D, ASTContext& C);

  friend class Decl;
};

template<> struct DeclContext::KindTrait<DeclContext> {
  static Decl::Kind getKind(const DeclContext *D) { return D->DeclKind; }
};

} // end clang.

namespace llvm {

/// Implement a isa_impl_wrap specialization to check whether a DeclContext is
/// a specific Decl.
template<class ToTy>
struct isa_impl_wrap<ToTy,
                     const ::clang::DeclContext,const ::clang::DeclContext> {
  static bool doit(const ::clang::DeclContext &Val) {
    return ToTy::classof(::clang::Decl::castFromDeclContext(&Val));
  }
};
template<class ToTy>
struct isa_impl_wrap<ToTy, ::clang::DeclContext, ::clang::DeclContext>
  : public isa_impl_wrap<ToTy,
                      const ::clang::DeclContext,const ::clang::DeclContext> {};

/// Implement cast_convert_val for Decl -> DeclContext conversions.
template<class FromTy>
struct cast_convert_val< ::clang::DeclContext, FromTy, FromTy> {
  static ::clang::DeclContext &doit(const FromTy &Val) {
    return *FromTy::castToDeclContext(&Val);
  }
};

template<class FromTy>
struct cast_convert_val< ::clang::DeclContext, FromTy*, FromTy*> {
  static ::clang::DeclContext *doit(const FromTy *Val) {
    return FromTy::castToDeclContext(Val);
  }
};

/// Implement cast_convert_val for DeclContext -> Decl conversions.
template<class ToTy>
struct cast_convert_val<ToTy,
                        const ::clang::DeclContext,const ::clang::DeclContext> {
  static ToTy &doit(const ::clang::DeclContext &Val) {
    return *reinterpret_cast<ToTy*>(ToTy::castFromDeclContext(&Val));
  }
};
template<class ToTy>
struct cast_convert_val<ToTy, ::clang::DeclContext, ::clang::DeclContext>
  : public cast_convert_val<ToTy,
                      const ::clang::DeclContext,const ::clang::DeclContext> {};

template<class ToTy>
struct cast_convert_val<ToTy,
                     const ::clang::DeclContext*, const ::clang::DeclContext*> {
  static ToTy *doit(const ::clang::DeclContext *Val) {
    return reinterpret_cast<ToTy*>(ToTy::castFromDeclContext(Val));
  }
};
template<class ToTy>
struct cast_convert_val<ToTy, ::clang::DeclContext*, ::clang::DeclContext*>
  : public cast_convert_val<ToTy,
                    const ::clang::DeclContext*,const ::clang::DeclContext*> {};

} // end namespace llvm

#endif
