//===--- DelayedDiagnostic.h - Delayed declarator diagnostics ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DelayedDiagnostic class, which is used to
// record diagnostics that are being conditionally produced during
// declarator parsing.  Certain kinds of diagnostics --- notably
// deprecation and access control --- are suppressed based on
// semantic properties of the parsed declaration that aren't known
// until it is fully parsed.
//
// This file also defines AccessedEntity.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_DELAYED_DIAGNOSTIC_H
#define LLVM_CLANG_SEMA_DELAYED_DIAGNOSTIC_H

#include "clang/AST/DeclCXX.h"

namespace clang {
namespace sema {

/// A declaration being accessed, together with information about how
/// it was accessed.
class AccessedEntity {
public:
  /// A member declaration found through lookup.  The target is the
  /// member.
  enum MemberNonce { Member };

  /// A hierarchy (base-to-derived or derived-to-base) conversion.
  /// The target is the base class.
  enum BaseNonce { Base };

  bool isMemberAccess() const { return IsMember; }

  AccessedEntity(ASTContext &Context,
                 MemberNonce _,
                 CXXRecordDecl *NamingClass,
                 DeclAccessPair FoundDecl,
                 QualType BaseObjectType)
    : Access(FoundDecl.getAccess()), IsMember(true),
      Target(FoundDecl.getDecl()), NamingClass(NamingClass),
      BaseObjectType(BaseObjectType), Diag(0, Context.getDiagAllocator()) {
  }

  AccessedEntity(ASTContext &Context,
                 BaseNonce _,
                 CXXRecordDecl *BaseClass,
                 CXXRecordDecl *DerivedClass,
                 AccessSpecifier Access)
    : Access(Access), IsMember(false),
      Target(BaseClass),
      NamingClass(DerivedClass),
      Diag(0, Context.getDiagAllocator()) {
  }

  bool isQuiet() const { return Diag.getDiagID() == 0; }

  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

  // These apply to member decls...
  NamedDecl *getTargetDecl() const { return Target; }
  CXXRecordDecl *getNamingClass() const { return NamingClass; }

  // ...and these apply to hierarchy conversions.
  CXXRecordDecl *getBaseClass() const {
    assert(!IsMember); return cast<CXXRecordDecl>(Target);
  }
  CXXRecordDecl *getDerivedClass() const { return NamingClass; }

  /// Retrieves the base object type, important when accessing
  /// an instance member.
  QualType getBaseObjectType() const { return BaseObjectType; }

  /// Sets a diagnostic to be performed.  The diagnostic is given
  /// four (additional) arguments:
  ///   %0 - 0 if the entity was private, 1 if protected
  ///   %1 - the DeclarationName of the entity
  ///   %2 - the TypeDecl type of the naming class
  ///   %3 - the TypeDecl type of the declaring class
  void setDiag(const PartialDiagnostic &PDiag) {
    assert(isQuiet() && "partial diagnostic already defined");
    Diag = PDiag;
  }
  PartialDiagnostic &setDiag(unsigned DiagID) {
    assert(isQuiet() && "partial diagnostic already defined");
    assert(DiagID && "creating null diagnostic");
    Diag.Reset(DiagID);
    return Diag;
  }
  const PartialDiagnostic &getDiag() const {
    return Diag;
  }

private:
  unsigned Access : 2;
  unsigned IsMember : 1;
  NamedDecl *Target;
  CXXRecordDecl *NamingClass;
  QualType BaseObjectType;
  PartialDiagnostic Diag;
};

/// A diagnostic message which has been conditionally emitted pending
/// the complete parsing of the current declaration.
class DelayedDiagnostic {
public:
  enum DDKind { Deprecation, Access };

  unsigned char Kind; // actually a DDKind
  bool Triggered;

  SourceLocation Loc;

  void destroy() {
    switch (Kind) {
    case Access: getAccessData().~AccessedEntity(); break;
    case Deprecation: break;
    }
  }

  static DelayedDiagnostic makeDeprecation(SourceLocation Loc,
                                           const NamedDecl *D,
                                           llvm::StringRef Msg) {
    DelayedDiagnostic DD;
    DD.Kind = Deprecation;
    DD.Triggered = false;
    DD.Loc = Loc;
    DD.DeprecationData.Decl = D;
    DD.DeprecationData.Message = Msg.data();
    DD.DeprecationData.MessageLen = Msg.size();
    return DD;
  }

  static DelayedDiagnostic makeAccess(SourceLocation Loc,
                                      const AccessedEntity &Entity) {
    DelayedDiagnostic DD;
    DD.Kind = Access;
    DD.Triggered = false;
    DD.Loc = Loc;
    new (&DD.getAccessData()) AccessedEntity(Entity);
    return DD;
  }

  AccessedEntity &getAccessData() {
    assert(Kind == Access && "Not an access diagnostic.");
    return *reinterpret_cast<AccessedEntity*>(AccessData);
  }
  const AccessedEntity &getAccessData() const {
    assert(Kind == Access && "Not an access diagnostic.");
    return *reinterpret_cast<const AccessedEntity*>(AccessData);
  }

  const NamedDecl *getDeprecationDecl() const {
    assert(Kind == Deprecation && "Not a deprecation diagnostic.");
    return DeprecationData.Decl;
  }

  llvm::StringRef getDeprecationMessage() const {
    assert(Kind == Deprecation && "Not a deprecation diagnostic.");
    return llvm::StringRef(DeprecationData.Message,
                           DeprecationData.MessageLen);
  }

private:
  union {
    /// Deprecation.
    struct {
      const NamedDecl *Decl;
      const char *Message;
      size_t MessageLen;
    } DeprecationData;

    /// Access control.
    char AccessData[sizeof(AccessedEntity)];
  };
};

}
}

#endif
