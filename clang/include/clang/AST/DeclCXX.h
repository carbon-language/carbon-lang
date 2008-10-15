//===-- DeclCXX.h - Classes for representing C++ declarations *- C++ -*-======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ Decl subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLCXX_H
#define LLVM_CLANG_AST_DECLCXX_H

#include "clang/AST/Decl.h"

namespace clang {
class CXXRecordDecl;

/// CXXFieldDecl - Represents an instance field of a C++ struct/union/class.
class CXXFieldDecl : public FieldDecl {
  CXXRecordDecl *Parent;

  CXXFieldDecl(CXXRecordDecl *RD, SourceLocation L, IdentifierInfo *Id,
               QualType T, Expr *BW = NULL)
    : FieldDecl(CXXField, L, Id, T, BW), Parent(RD) {}
public:
  static CXXFieldDecl *Create(ASTContext &C, CXXRecordDecl *RD,SourceLocation L,
                              IdentifierInfo *Id, QualType T, Expr *BW = NULL);

  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }
  CXXRecordDecl *getParent() const { return Parent; }
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == CXXField; }
  static bool classof(const CXXFieldDecl *D) { return true; }
};

/// CXXRecordDecl - Represents a C++ struct/union/class.
/// The only difference with RecordDecl is that CXXRecordDecl is a DeclContext.
class CXXRecordDecl : public RecordDecl, public DeclContext {
  CXXRecordDecl(TagKind TK, DeclContext *DC,
                SourceLocation L, IdentifierInfo *Id) 
    : RecordDecl(CXXRecord, TK, DC, L, Id), DeclContext(CXXRecord) {}
public:
  static CXXRecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               CXXRecordDecl* PrevDecl=0);
  
  const CXXFieldDecl *getMember(unsigned i) const {
    return cast<const CXXFieldDecl>(RecordDecl::getMember(i));
  }
  CXXFieldDecl *getMember(unsigned i) {
    return cast<CXXFieldDecl>(RecordDecl::getMember(i));
  }

  /// getMember - If the member doesn't exist, or there are no members, this 
  /// function will return 0;
  CXXFieldDecl *getMember(IdentifierInfo *name) {
    return cast_or_null<CXXFieldDecl>(RecordDecl::getMember(name));
  }

  static bool classof(const Decl *D) { return D->getKind() == CXXRecord; }
  static bool classof(const CXXRecordDecl *D) { return true; }
  static DeclContext *castToDeclContext(const CXXRecordDecl *D) {
    return static_cast<DeclContext *>(const_cast<CXXRecordDecl*>(D));
  }
  static CXXRecordDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<CXXRecordDecl *>(const_cast<DeclContext*>(DC));
  }

protected:
  /// EmitImpl - Serialize this CXXRecordDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXRecordDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXRecordDecl* CreateImpl(Kind DK, llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// CXXMethodDecl - Represents a static or instance method of a
/// struct/union/class.
class CXXMethodDecl : public FunctionDecl {

  CXXMethodDecl(CXXRecordDecl *RD, SourceLocation L,
               IdentifierInfo *Id, QualType T,
               bool isStatic, bool isInline, ScopedDecl *PrevDecl)
    : FunctionDecl(CXXMethod, RD, L, Id, T, (isStatic ? Static : None),
                   isInline, PrevDecl) {}
public:
  static CXXMethodDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                              SourceLocation L, IdentifierInfo *Id,
                              QualType T, bool isStatic = false,
                              bool isInline = false,  ScopedDecl *PrevDecl = 0);
  
  bool isStatic() const { return getStorageClass() == Static; }
  bool isInstance() const { return !isStatic(); }

  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

  /// getThisType - Returns the type of 'this' pointer.
  /// Should only be called for instance methods.
  QualType getThisType(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == CXXMethod; }
  static bool classof(const CXXMethodDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this CXXMethodDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXMethodDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXMethodDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// CXXClassVarDecl - Represents a static data member of a struct/union/class.
class CXXClassVarDecl : public VarDecl {

  CXXClassVarDecl(CXXRecordDecl *RD, SourceLocation L,
              IdentifierInfo *Id, QualType T, ScopedDecl *PrevDecl)
    : VarDecl(CXXClassVar, RD, L, Id, T, None, PrevDecl) {}
public:
  static CXXClassVarDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, ScopedDecl *PrevDecl);
  
  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == CXXClassVar; }
  static bool classof(const CXXClassVarDecl *D) { return true; }
  
protected:
  /// EmitImpl - Serialize this CXXClassVarDecl. Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXClassVarDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXClassVarDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};


/// CXXClassMemberWrapper - A wrapper class for C++ class member decls.
/// Common functions like set/getAccess are included here to avoid bloating
/// the interface of non-C++ specific decl classes, like NamedDecl.
class CXXClassMemberWrapper {
  Decl *MD;

public:
  CXXClassMemberWrapper(Decl *D) : MD(D) {
    assert(isMember(D) && "Not a C++ class member!");
  }

  AccessSpecifier getAccess() const {
    return AccessSpecifier(MD->Access);
  }

  void setAccess(AccessSpecifier AS) {
    assert(AS != AS_none && "Access must be specified.");
    MD->Access = AS;
  }

  CXXRecordDecl *getParent() const {
    if (ScopedDecl *SD = dyn_cast<ScopedDecl>(MD)) {
      return cast<CXXRecordDecl>(SD->getDeclContext());
    }
    return cast<CXXFieldDecl>(MD)->getParent();
  }

  static bool isMember(Decl *D) {
    if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
      return isa<CXXRecordDecl>(SD->getDeclContext());
    }
    return isa<CXXFieldDecl>(D);
  }
};

} // end namespace clang

#endif
