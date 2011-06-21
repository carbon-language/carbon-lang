//===--- TransZeroOutPropsInDealloc.cpp - Tranformations to ARC mode ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// removeZeroOutPropsInDealloc:
//
// Removes zero'ing out "strong" @synthesized properties in a -dealloc method.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class ZeroOutInDeallocRemover :
                           public RecursiveASTVisitor<ZeroOutInDeallocRemover> {
  typedef RecursiveASTVisitor<ZeroOutInDeallocRemover> base;

  MigrationPass &Pass;

  llvm::DenseMap<ObjCPropertyDecl*, ObjCPropertyImplDecl*> SynthesizedProperties;
  ImplicitParamDecl *SelfD;
  ExprSet Removables;

public:
  ZeroOutInDeallocRemover(MigrationPass &pass) : Pass(pass), SelfD(0) { }

  bool VisitObjCMessageExpr(ObjCMessageExpr *ME) {
    ASTContext &Ctx = Pass.Ctx;
    TransformActions &TA = Pass.TA;

    if (ME->getReceiverKind() != ObjCMessageExpr::Instance)
      return true;
    Expr *receiver = ME->getInstanceReceiver();
    if (!receiver)
      return true;

    DeclRefExpr *refE = dyn_cast<DeclRefExpr>(receiver->IgnoreParenCasts());
    if (!refE || refE->getDecl() != SelfD)
      return true;

    bool BackedBySynthesizeSetter = false;
    for (llvm::DenseMap<ObjCPropertyDecl*, ObjCPropertyImplDecl*>::iterator
         P = SynthesizedProperties.begin(), 
         E = SynthesizedProperties.end(); P != E; ++P) {
      ObjCPropertyDecl *PropDecl = P->first;
      if (PropDecl->getSetterName() == ME->getSelector()) {
        BackedBySynthesizeSetter = true;
        break;
      }
    }
    if (!BackedBySynthesizeSetter)
      return true;
    
    // Remove the setter message if RHS is null
    Transaction Trans(TA);
    Expr *RHS = ME->getArg(0);
    bool RHSIsNull = 
      RHS->isNullPointerConstant(Ctx,
                                 Expr::NPC_ValueDependentIsNull);
    if (RHSIsNull && isRemovable(ME))
      TA.removeStmt(ME);

    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *BOE) {
    if (isZeroingPropIvar(BOE) && isRemovable(BOE)) {
      Transaction Trans(Pass.TA);
      Pass.TA.removeStmt(BOE);
    }

    return true;
  }

  bool TraverseObjCMethodDecl(ObjCMethodDecl *D) {
    if (D->getMethodFamily() != OMF_dealloc)
      return true;
    if (!D->hasBody())
      return true;

    ObjCImplDecl *IMD = dyn_cast<ObjCImplDecl>(D->getDeclContext());
    if (!IMD)
      return true;

    SelfD = D->getSelfDecl();
    collectRemovables(D->getBody(), Removables);

    // For a 'dealloc' method use, find all property implementations in
    // this class implementation.
    for (ObjCImplDecl::propimpl_iterator
           I = IMD->propimpl_begin(), EI = IMD->propimpl_end(); I != EI; ++I) {
        ObjCPropertyImplDecl *PID = *I;
        if (PID->getPropertyImplementation() ==
            ObjCPropertyImplDecl::Synthesize) {
          ObjCPropertyDecl *PD = PID->getPropertyDecl();
          ObjCMethodDecl *setterM = PD->getSetterMethodDecl();
          if (!(setterM && setterM->isDefined())) {
            ObjCPropertyDecl::PropertyAttributeKind AttrKind = 
              PD->getPropertyAttributes();
              if (AttrKind & 
                  (ObjCPropertyDecl::OBJC_PR_retain | 
                   ObjCPropertyDecl::OBJC_PR_copy   |
                   ObjCPropertyDecl::OBJC_PR_strong))
                SynthesizedProperties[PD] = PID;
          }
        }
    }

    // Now, remove all zeroing of ivars etc.
    base::TraverseObjCMethodDecl(D);

    // clear out for next method.
    SynthesizedProperties.clear();
    SelfD = 0;
    Removables.clear();
    return true;
  }

  bool TraverseFunctionDecl(FunctionDecl *D) { return true; }
  bool TraverseBlockDecl(BlockDecl *block) { return true; }
  bool TraverseBlockExpr(BlockExpr *block) { return true; }

private:
  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }

  bool isZeroingPropIvar(Expr *E) {
    BinaryOperator *BOE = dyn_cast_or_null<BinaryOperator>(E);
    if (!BOE) return false;

    if (BOE->getOpcode() == BO_Comma)
      return isZeroingPropIvar(BOE->getLHS()) &&
             isZeroingPropIvar(BOE->getRHS());

    if (BOE->getOpcode() != BO_Assign)
        return false;

    ASTContext &Ctx = Pass.Ctx;

    Expr *LHS = BOE->getLHS();
    if (ObjCIvarRefExpr *IV = dyn_cast<ObjCIvarRefExpr>(LHS)) {
      ObjCIvarDecl *IVDecl = IV->getDecl();
      if (!IVDecl->getType()->isObjCObjectPointerType())
        return false;
      bool IvarBacksPropertySynthesis = false;
      for (llvm::DenseMap<ObjCPropertyDecl*, ObjCPropertyImplDecl*>::iterator
           P = SynthesizedProperties.begin(), 
           E = SynthesizedProperties.end(); P != E; ++P) {
        ObjCPropertyImplDecl *PropImpDecl = P->second;
        if (PropImpDecl && PropImpDecl->getPropertyIvarDecl() == IVDecl) {
          IvarBacksPropertySynthesis = true;
          break;
        }
      }
      if (!IvarBacksPropertySynthesis)
        return false;
    }
    else if (ObjCPropertyRefExpr *PropRefExp = dyn_cast<ObjCPropertyRefExpr>(LHS)) {
      // TODO: Using implicit property decl.
      if (PropRefExp->isImplicitProperty())
        return false;
      if (ObjCPropertyDecl *PDecl = PropRefExp->getExplicitProperty()) {
        if (!SynthesizedProperties.count(PDecl))
          return false;
      }
    }
    else
        return false;

    Expr *RHS = BOE->getRHS();
    bool RHSIsNull = RHS->isNullPointerConstant(Ctx,
                                                Expr::NPC_ValueDependentIsNull);
    if (RHSIsNull)
      return true;

    return isZeroingPropIvar(RHS);
  }
};

} // anonymous namespace

void trans::removeZeroOutPropsInDealloc(MigrationPass &pass) {
  ZeroOutInDeallocRemover trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
