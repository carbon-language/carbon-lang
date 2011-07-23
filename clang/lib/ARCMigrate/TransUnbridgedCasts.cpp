//===--- TransUnbridgedCasts.cpp - Tranformations to ARC mode -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// rewriteUnbridgedCasts:
//
// A cast of non-objc pointer to an objc one is checked. If the non-objc pointer
// is from a file-level variable, __bridge cast is used to convert it.
// For the result of a function call that we know is +1/+0,
// __bridge/__bridge_transfer is used.
//
//  NSString *str = (NSString *)kUTTypePlainText;
//  str = b ? kUTTypeRTF : kUTTypePlainText;
//  NSString *_uuidString = (NSString *)CFUUIDCreateString(kCFAllocatorDefault,
//                                                         _uuid);
// ---->
//  NSString *str = (__bridge NSString *)kUTTypePlainText;
//  str = (__bridge NSString *)(b ? kUTTypeRTF : kUTTypePlainText);
// NSString *_uuidString = (__bridge_transfer NSString *)
//                               CFUUIDCreateString(kCFAllocatorDefault, _uuid);
//
// For a C pointer to ObjC, for casting 'self', __bridge is used.
//
//  CFStringRef str = (CFStringRef)self;
// ---->
//  CFStringRef str = (__bridge CFStringRef)self;
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

class UnbridgedCastRewriter : public RecursiveASTVisitor<UnbridgedCastRewriter>{
  MigrationPass &Pass;
  IdentifierInfo *SelfII;
public:
  UnbridgedCastRewriter(MigrationPass &pass) : Pass(pass) {
    SelfII = &Pass.Ctx.Idents.get("self");
  }

  bool VisitCastExpr(CastExpr *E) {
    if (E->getCastKind() != CK_AnyPointerToObjCPointerCast
        && E->getCastKind() != CK_BitCast)
      return true;

    QualType castType = E->getType();
    Expr *castExpr = E->getSubExpr();
    QualType castExprType = castExpr->getType();

    if (castType->isObjCObjectPointerType() &&
        castExprType->isObjCObjectPointerType())
      return true;
    if (!castType->isObjCObjectPointerType() &&
        !castExprType->isObjCObjectPointerType())
      return true;
    
    bool exprRetainable = castExprType->isObjCIndirectLifetimeType();
    bool castRetainable = castType->isObjCIndirectLifetimeType();
    if (exprRetainable == castRetainable) return true;

    if (castExpr->isNullPointerConstant(Pass.Ctx,
                                        Expr::NPC_ValueDependentIsNull))
      return true;

    SourceLocation loc = castExpr->getExprLoc();
    if (loc.isValid() && Pass.Ctx.getSourceManager().isInSystemHeader(loc))
      return true;

    if (castType->isObjCObjectPointerType())
      transformNonObjCToObjCCast(E);
    else
      transformObjCToNonObjCCast(E);

    return true;
  }

private:
  void transformNonObjCToObjCCast(CastExpr *E) {
    if (!E) return;

    // Global vars are assumed that are cast as unretained.
    if (isGlobalVar(E))
      if (E->getSubExpr()->getType()->isPointerType()) {
        castToObjCObject(E, /*retained=*/false);
        return;
      }

    // If the cast is directly over the result of a Core Foundation function
    // try to figure out whether it should be cast as retained or unretained.
    Expr *inner = E->IgnoreParenCasts();
    if (CallExpr *callE = dyn_cast<CallExpr>(inner)) {
      if (FunctionDecl *FD = callE->getDirectCallee()) {
        if (FD->getAttr<CFReturnsRetainedAttr>()) {
          castToObjCObject(E, /*retained=*/true);
          return;
        }
        if (FD->getAttr<CFReturnsNotRetainedAttr>()) {
          castToObjCObject(E, /*retained=*/false);
          return;
        }
        if (FD->isGlobal() &&
            FD->getIdentifier() &&
            ento::cocoa::isRefType(E->getSubExpr()->getType(), "CF",
                                   FD->getIdentifier()->getName())) {
          StringRef fname = FD->getIdentifier()->getName();
          if (fname.endswith("Retain") ||
              fname.find("Create") != StringRef::npos ||
              fname.find("Copy") != StringRef::npos) {
            castToObjCObject(E, /*retained=*/true);
            return;
          }

          if (fname.find("Get") != StringRef::npos) {
            castToObjCObject(E, /*retained=*/false);
            return;
          }
        }
      }
    }
  }

  void castToObjCObject(CastExpr *E, bool retained) {
    rewriteToBridgedCast(E, retained ? OBC_BridgeTransfer : OBC_Bridge);
  }

  void rewriteToBridgedCast(CastExpr *E, ObjCBridgeCastKind Kind) {
    TransformActions &TA = Pass.TA;

    // We will remove the compiler diagnostic.
    if (!TA.hasDiagnostic(diag::err_arc_mismatched_cast,
                          diag::err_arc_cast_requires_bridge,
                          E->getLocStart()))
      return;

    StringRef bridge;
    switch(Kind) {
    case OBC_Bridge:
      bridge = "__bridge "; break;
    case OBC_BridgeTransfer:
      bridge = "__bridge_transfer "; break;
    case OBC_BridgeRetained:
      bridge = "__bridge_retained "; break;
    }

    Transaction Trans(TA);
    TA.clearDiagnostic(diag::err_arc_mismatched_cast,
                       diag::err_arc_cast_requires_bridge,
                       E->getLocStart());
    if (CStyleCastExpr *CCE = dyn_cast<CStyleCastExpr>(E)) {
      TA.insertAfterToken(CCE->getLParenLoc(), bridge);
    } else {
      SourceLocation insertLoc = E->getSubExpr()->getLocStart();
      llvm::SmallString<128> newCast;
      newCast += '(';
      newCast += bridge;
      newCast += E->getType().getAsString(Pass.Ctx.PrintingPolicy);
      newCast += ')';

      if (isa<ParenExpr>(E->getSubExpr())) {
        TA.insert(insertLoc, newCast.str());
      } else {
        newCast += '(';
        TA.insert(insertLoc, newCast.str());
        TA.insertAfterToken(E->getLocEnd(), ")");
      }
    }
  }

  void transformObjCToNonObjCCast(CastExpr *E) {
    if (isSelf(E->getSubExpr()))
      return rewriteToBridgedCast(E, OBC_Bridge);
  }

  bool isSelf(Expr *E) {
    E = E->IgnoreParenLValueCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
      if (DRE->getDecl()->getIdentifier() == SelfII)
        return true;
    return false;
  }
};

} // end anonymous namespace

void trans::rewriteUnbridgedCasts(MigrationPass &pass) {
  UnbridgedCastRewriter trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
