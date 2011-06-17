//===--- Tranforms.cpp - Tranformations to ARC mode -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Transformations:
//===----------------------------------------------------------------------===//
//
// castNonObjCToObjC:
//
// A cast of non-objc pointer to an objc one is checked. If the non-objc pointer
// is from a file-level variable, objc_unretainedObject function is used to
// convert it.
//
//  NSString *str = (NSString *)kUTTypePlainText;
//  str = b ? kUTTypeRTF : kUTTypePlainText;
// ---->
//  NSString *str = objc_unretainedObject(kUTTypePlainText);
//  str = objc_unretainedObject(b ? kUTTypeRTF : kUTTypePlainText);
//
// For a C pointer to ObjC, objc_unretainedPointer is used.
//
//  void *vp = str; // NSString*
// ---->
//  void *vp = (void*)objc_unretainedPointer(str);
//
//===----------------------------------------------------------------------===//
//
// rewriteAllocCopyWithZone:
//
// Calls to +allocWithZone/-copyWithZone/-mutableCopyWithZone are changed to
// +alloc/-copy/-mutableCopy if we can safely remove the given parameter.
//
//  Foo *foo1 = [[Foo allocWithZone:[self zone]] init];
// ---->
//  Foo *foo1 = [[Foo alloc] init];
//
//===----------------------------------------------------------------------===//
//
// rewriteAutoreleasePool:
//
// Calls to NSAutoreleasePools will be rewritten as an @autorelease scope.
//
//  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
//  ...
//  [pool release];
// ---->
//  @autorelease {
//  ...
//  }
//
// An NSAutoreleasePool will not be touched if:
// - There is not a corresponding -release/-drain in the same scope
// - Not all references of the NSAutoreleasePool variable can be removed
// - There is a variable that is declared inside the intended @autorelease scope
//   which is also used outside it.
//
//===----------------------------------------------------------------------===//
//
// makeAssignARCSafe:
//
// Add '__strong' where appropriate.
//
//  for (id x in collection) {
//    x = 0;
//  }
// ---->
//  for (__strong id x in collection) {
//    x = 0;
//  }
//
//===----------------------------------------------------------------------===//
//
// removeRetainReleaseDealloc:
//
// Removes retain/release/autorelease/dealloc messages.
//
//  return [[foo retain] autorelease];
// ---->
//  return foo;
//
//===----------------------------------------------------------------------===//
//
// removeEmptyStatements:
//
// Removes empty statements that are leftovers from previous transformations.
// e.g for
//
//  [x retain];
//
// removeRetainReleaseDealloc will leave an empty ";" that removeEmptyStatements
// will remove.
//
//===----------------------------------------------------------------------===//
//
// changeIvarsOfAssignProperties:
//
// If a property is synthesized with 'assign' attribute and the user didn't
// set a lifetime attribute, change the property to 'weak' or add
// __unsafe_unretained if the ARC runtime is not available.
//
//  @interface Foo : NSObject {
//      NSObject *x;
//  }
//  @property (assign) id x;
//  @end
// ---->
//  @interface Foo : NSObject {
//      NSObject *__weak x;
//  }
//  @property (weak) id x;
//  @end
//
//===----------------------------------------------------------------------===//
//
// rewriteUnusedDelegateInit:
//
// Rewrites an unused result of calling a delegate initialization, to assigning
// the result to self.
// e.g
//  [self init];
// ---->
//  self = [self init];
//
//===----------------------------------------------------------------------===//
//
// rewriteBlockObjCVariable:
//
// Adding __block to an obj-c variable could be either because the the variable
// is used for output storage or the user wanted to break a retain cycle.
// This transformation checks whether a reference of the variable for the block
// is actually needed (it is assigned to or its address is taken) or not.
// If the reference is not needed it will assume __block was added to break a
// cycle so it will remove '__block' and add __weak/__unsafe_unretained.
// e.g
//
//   __block Foo *x;
//   bar(^ { [x cake]; });
// ---->
//   __weak Foo *x;
//   bar(^ { [x cake]; });
//
//===----------------------------------------------------------------------===//
//
// removeZeroOutIvarsInDealloc:
//
// Removes zero'ing out "strong" @synthesized properties in a -dealloc method.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/DenseSet.h"
#include <map>

using namespace clang;
using namespace arcmt;
using llvm::StringRef;

//===----------------------------------------------------------------------===//
// Transformations.
//===----------------------------------------------------------------------===//

namespace {

class RemovablesCollector : public RecursiveASTVisitor<RemovablesCollector> {
  llvm::DenseSet<Expr *> &Removables;

public:
  RemovablesCollector(llvm::DenseSet<Expr *> &removables)
  : Removables(removables) { }
  
  bool shouldWalkTypesOfTypeLocs() const { return false; }
  
  bool TraverseStmtExpr(StmtExpr *E) {
    CompoundStmt *S = E->getSubStmt();
    for (CompoundStmt::body_iterator
        I = S->body_begin(), E = S->body_end(); I != E; ++I) {
      if (I != E - 1)
        mark(*I);
      TraverseStmt(*I);
    }
    return true;
  }
  
  bool VisitCompoundStmt(CompoundStmt *S) {
    for (CompoundStmt::body_iterator
        I = S->body_begin(), E = S->body_end(); I != E; ++I)
      mark(*I);
    return true;
  }
  
  bool VisitIfStmt(IfStmt *S) {
    mark(S->getThen());
    mark(S->getElse());
    return true;
  }
  
  bool VisitWhileStmt(WhileStmt *S) {
    mark(S->getBody());
    return true;
  }
  
  bool VisitDoStmt(DoStmt *S) {
    mark(S->getBody());
    return true;
  }
  
  bool VisitForStmt(ForStmt *S) {
    mark(S->getInit());
    mark(S->getInc());
    mark(S->getBody());
    return true;
  }
  
private:
  void mark(Stmt *S) {
    if (!S) return;
    
    if (LabelStmt *Label = dyn_cast<LabelStmt>(S))
      return mark(Label->getSubStmt());
    if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(S))
      return mark(CE->getSubExpr());
    if (ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(S))
      return mark(EWC->getSubExpr());
    if (Expr *E = dyn_cast<Expr>(S))
      Removables.insert(E);
  }
};

} // end anonymous namespace.

static bool HasSideEffects(Expr *E, ASTContext &Ctx) {
  if (!E || !E->HasSideEffects(Ctx))
    return false;

  E = E->IgnoreParenCasts();
  ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E);
  if (!ME)
    return true;
  switch (ME->getMethodFamily()) {
  case OMF_autorelease:
  case OMF_dealloc:
  case OMF_release:
  case OMF_retain:
    switch (ME->getReceiverKind()) {
    case ObjCMessageExpr::SuperInstance:
      return false;
    case ObjCMessageExpr::Instance:
      return HasSideEffects(ME->getInstanceReceiver(), Ctx);
    default:
      break;
    }
    break;
  default:
    break;
  }

  return true;
}

static void removeDeallocMethod(MigrationPass &pass) {
    ASTContext &Ctx = pass.Ctx;
    TransformActions &TA = pass.TA;
    DeclContext *DC = Ctx.getTranslationUnitDecl();
    ObjCMethodDecl *DeallocMethodDecl = 0;
    IdentifierInfo *II = &Ctx.Idents.get("dealloc");
    
    for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
         I != E; ++I) {
        Decl *D = *I;
        if (ObjCImplementationDecl *IMD = 
            dyn_cast<ObjCImplementationDecl>(D)) {
            DeallocMethodDecl = 0;
            for (ObjCImplementationDecl::instmeth_iterator I = 
                 IMD->instmeth_begin(), E = IMD->instmeth_end();
                 I != E; ++I) {
                ObjCMethodDecl *OMD = *I;
                if (OMD->isInstanceMethod() &&
                    OMD->getSelector() == Ctx.Selectors.getSelector(0, &II)) {
                    DeallocMethodDecl = OMD;
                    break;
                }
            }
            if (DeallocMethodDecl && 
                DeallocMethodDecl->getCompoundBody()->body_empty()) {
              Transaction Trans(TA);
              TA.remove(DeallocMethodDecl->getSourceRange());
            }
        }
    }
}

namespace {

class ReferenceClear : public RecursiveASTVisitor<ReferenceClear> {
  llvm::DenseSet<Expr *> &Refs;
public:
  ReferenceClear(llvm::DenseSet<Expr *> &refs) : Refs(refs) { }
  bool VisitDeclRefExpr(DeclRefExpr *E) { Refs.erase(E); return true; }
  bool VisitBlockDeclRefExpr(BlockDeclRefExpr *E) { Refs.erase(E); return true; }
  void clearRefsIn(Stmt *S) { TraverseStmt(S); }
  template <typename iterator>
  void clearRefsIn(iterator begin, iterator end) {
    for (; begin != end; ++begin)
      TraverseStmt(*begin);
  }
};

class ReferenceCollector : public RecursiveASTVisitor<ReferenceCollector> {
  ValueDecl *Dcl;
  llvm::DenseSet<Expr *> &Refs;

public:
  ReferenceCollector(llvm::DenseSet<Expr *> &refs)
    : Dcl(0), Refs(refs) { }

  void lookFor(ValueDecl *D, Stmt *S) {
    Dcl = D;
    TraverseStmt(S);
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->getDecl() == Dcl)
      Refs.insert(E);
    return true;
  }

  bool VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
    if (E->getDecl() == Dcl)
      Refs.insert(E);
    return true;
  }
};

class ReleaseCollector : public RecursiveASTVisitor<ReleaseCollector> {
  Decl *Dcl;
  llvm::SmallVectorImpl<ObjCMessageExpr *> &Releases;

public:
  ReleaseCollector(Decl *D, llvm::SmallVectorImpl<ObjCMessageExpr *> &releases)
    : Dcl(D), Releases(releases) { }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (!E->isInstanceMessage())
      return true;
    if (E->getMethodFamily() != OMF_release)
      return true;
    Expr *instance = E->getInstanceReceiver()->IgnoreParenCasts();
    if (DeclRefExpr *DE = dyn_cast<DeclRefExpr>(instance)) {
      if (DE->getDecl() == Dcl)
        Releases.push_back(E);
    }
    return true;
  }
};

template <typename BODY_TRANS>
class BodyTransform : public RecursiveASTVisitor<BodyTransform<BODY_TRANS> > {
  MigrationPass &Pass;

public:
  BodyTransform(MigrationPass &pass) : Pass(pass) { }

  void handleBody(Decl *D) {
    Stmt *body = D->getBody();
    if (body) {
      BODY_TRANS(D, Pass).transformBody(body);
    }
  }

  bool TraverseBlockDecl(BlockDecl *D) {
    handleBody(D);
    return true;
  }
  bool TraverseObjCMethodDecl(ObjCMethodDecl *D) {
    if (D->isThisDeclarationADefinition())
      handleBody(D);
    return true;
  }
  bool TraverseFunctionDecl(FunctionDecl *D) {
    if (D->isThisDeclarationADefinition())
      handleBody(D);
    return true;
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// makeAssignARCSafe
//===----------------------------------------------------------------------===//

namespace {

class ARCAssignChecker : public RecursiveASTVisitor<ARCAssignChecker> {
  MigrationPass &Pass;
  llvm::DenseSet<VarDecl *> ModifiedVars;

public:
  ARCAssignChecker(MigrationPass &pass) : Pass(pass) { }

  bool VisitBinaryOperator(BinaryOperator *Exp) {
    Expr *E = Exp->getLHS();
    SourceLocation OrigLoc = E->getExprLoc();
    SourceLocation Loc = OrigLoc;
    DeclRefExpr *declRef = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts());
    if (declRef && isa<VarDecl>(declRef->getDecl())) {
      ASTContext &Ctx = Pass.Ctx;
      Expr::isModifiableLvalueResult IsLV = E->isModifiableLvalue(Ctx, &Loc);
      if (IsLV != Expr::MLV_ConstQualified)
        return true;
      VarDecl *var = cast<VarDecl>(declRef->getDecl());
      if (var->isARCPseudoStrong()) {
        Transaction Trans(Pass.TA);
        if (Pass.TA.clearDiagnostic(diag::err_typecheck_arr_assign_enumeration,
                                    Exp->getOperatorLoc())) {
          if (!ModifiedVars.count(var)) {
            TypeLoc TLoc = var->getTypeSourceInfo()->getTypeLoc();
            Pass.TA.insert(TLoc.getBeginLoc(), "__strong ");
            ModifiedVars.insert(var);
          }
        }
      }
    }
    
    return true;
  }
};

} // anonymous namespace

static void makeAssignARCSafe(MigrationPass &pass) {
  ARCAssignChecker assignCheck(pass);
  assignCheck.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// castNonObjCToObjC
//===----------------------------------------------------------------------===//

namespace {

class NonObjCToObjCCaster : public RecursiveASTVisitor<NonObjCToObjCCaster> {
  MigrationPass &Pass;
public:
  NonObjCToObjCCaster(MigrationPass &pass) : Pass(pass) { }

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
    TransformActions &TA = Pass.TA;

    // We will remove the compiler diagnostic.
    if (!TA.hasDiagnostic(diag::err_arc_mismatched_cast,
                          diag::err_arc_cast_requires_bridge,
                          E->getLocStart()))
      return;

    Transaction Trans(TA);
    TA.clearDiagnostic(diag::err_arc_mismatched_cast,
                       diag::err_arc_cast_requires_bridge,
                       E->getLocStart());
    if (CStyleCastExpr *CCE = dyn_cast<CStyleCastExpr>(E)) {
      TA.insertAfterToken(CCE->getLParenLoc(), retained ? "__bridge_transfer "
                                                        : "__bridge ");
    } else {
      SourceLocation insertLoc = E->getSubExpr()->getLocStart();
      llvm::SmallString<128> newCast;
      newCast += '(';
      newCast +=  retained ? "__bridge_transfer " : "__bridge ";
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
    // FIXME: Handle these casts.
    return;
#if 0
    TransformActions &TA = Pass.TA;

    // We will remove the compiler diagnostic.
    if (!TA.hasDiagnostic(diag::err_arc_mismatched_cast,
                          diag::err_arc_cast_requires_bridge,
                          E->getLocStart()))
      return;

    Transaction Trans(TA);
    TA.clearDiagnostic(diag::err_arc_mismatched_cast,
                              diag::err_arc_cast_requires_bridge,
                              E->getLocStart());

    assert(!E->getType()->isObjCObjectPointerType());

    bool shouldCast = !isa<CStyleCastExpr>(E) &&
                      !E->getType()->getPointeeType().isConstQualified();
    SourceLocation loc = E->getSubExpr()->getLocStart();
    if (isa<ParenExpr>(E->getSubExpr())) {
      TA.insert(loc, shouldCast ? "(void*)objc_unretainedPointer"
                                : "objc_unretainedPointer");
    } else {
      TA.insert(loc, shouldCast ? "(void*)objc_unretainedPointer("
                                : "objc_unretainedPointer(");
      TA.insertAfterToken(E->getLocEnd(), ")");
    }
#endif
  }

  static bool isGlobalVar(Expr *E) {
    E = E->IgnoreParenCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
      return DRE->getDecl()->getDeclContext()->isFileContext();
    if (ConditionalOperator *condOp = dyn_cast<ConditionalOperator>(E))
      return isGlobalVar(condOp->getTrueExpr()) &&
             isGlobalVar(condOp->getFalseExpr());

    return false;  
  }
};

} // end anonymous namespace

static void castNonObjCToObjC(MigrationPass &pass) {
  NonObjCToObjCCaster trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// rewriteAllocCopyWithZone
//===----------------------------------------------------------------------===//

namespace {

class AllocCopyWithZoneRewriter :
                         public RecursiveASTVisitor<AllocCopyWithZoneRewriter> {
  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  Selector allocWithZoneSel;
  Selector copyWithZoneSel;
  Selector mutableCopyWithZoneSel;
  Selector zoneSel;
  IdentifierInfo *NSZoneII;

  std::vector<DeclStmt *> NSZoneVars;
  std::vector<Expr *> Removals;

public:
  AllocCopyWithZoneRewriter(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) {
    SelectorTable &sels = pass.Ctx.Selectors;
    IdentifierTable &ids = pass.Ctx.Idents; 
    allocWithZoneSel = sels.getUnarySelector(&ids.get("allocWithZone"));
    copyWithZoneSel = sels.getUnarySelector(&ids.get("copyWithZone"));
    mutableCopyWithZoneSel = sels.getUnarySelector(
                                               &ids.get("mutableCopyWithZone"));
    zoneSel = sels.getNullarySelector(&ids.get("zone"));
    NSZoneII = &ids.get("_NSZone");
  }

  void transformBody(Stmt *body) {
    Body = body;
    // Don't change allocWithZone/copyWithZone messages inside
    // custom implementations of such methods, it can lead to infinite loops.
    if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(Dcl)) {
      Selector sel = MD->getSelector();
      if (sel == allocWithZoneSel ||
          sel == copyWithZoneSel ||
          sel == mutableCopyWithZoneSel ||
          sel == zoneSel)
        return;
    }

    TraverseStmt(body);
  }

  ~AllocCopyWithZoneRewriter() {
    for (std::vector<DeclStmt *>::reverse_iterator
           I = NSZoneVars.rbegin(), E = NSZoneVars.rend(); I != E; ++I) {
      DeclStmt *DS = *I;
      DeclGroupRef group = DS->getDeclGroup();
      std::vector<Expr *> varRemovals = Removals;

      bool areAllVarsUnused = true;
      for (std::reverse_iterator<DeclGroupRef::iterator>
             DI(group.end()), DE(group.begin()); DI != DE; ++DI) {
        VarDecl *VD = cast<VarDecl>(*DI);
        if (isNSZoneVarUsed(VD, varRemovals)) {
          areAllVarsUnused = false;
          break;
        }
        varRemovals.push_back(VD->getInit());
      }

      if (areAllVarsUnused) {
        Transaction Trans(Pass.TA);
        clearUnavailableDiags(DS);
        Pass.TA.removeStmt(DS);
        Removals.swap(varRemovals);
      }
    }
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (!isAllocCopyWithZoneCall(E))
      return true;
    Expr *arg = E->getArg(0);
    if (paramToAllocWithZoneHasSideEffects(arg))
      return true;

    Pass.TA.startTransaction();

    clearUnavailableDiags(arg);
    Pass.TA.clearDiagnostic(diag::err_unavailable_message,
                            E->getReceiverRange().getBegin());

    Pass.TA.remove(SourceRange(E->getSelectorLoc(), arg->getLocEnd()));
    StringRef rewrite;
    if (E->getSelector() == allocWithZoneSel)
      rewrite = "alloc";
    else if (E->getSelector() == copyWithZoneSel)
      rewrite = "copy";
    else {
      assert(E->getSelector() == mutableCopyWithZoneSel);
      rewrite = "mutableCopy";
    }
    Pass.TA.insert(E->getSelectorLoc(), rewrite);

    bool failed = Pass.TA.commitTransaction();
    if (!failed)
      Removals.push_back(arg);

    return true;
  }

  bool VisitDeclStmt(DeclStmt *DS) {
    DeclGroupRef group = DS->getDeclGroup();
    if (group.begin() == group.end())
      return true;
    for (DeclGroupRef::iterator
           DI = group.begin(), DE = group.end(); DI != DE; ++DI)
      if (!isRemovableNSZoneVar(*DI))
        return true;

    NSZoneVars.push_back(DS);
    return true;
  }

private:
  bool isRemovableNSZoneVar(Decl *D) {
    if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (isNSZone(VD->getType()))
        return !paramToAllocWithZoneHasSideEffects(VD->getInit());
    }
    return false;
  }

  bool isNSZone(RecordDecl *RD) {
    return RD && RD->getIdentifier() == NSZoneII;
  }

  bool isNSZone(QualType Ty) {
    QualType pointee = Ty->getPointeeType();
    if (pointee.isNull())
      return false;
    if (const RecordType *recT = pointee->getAsStructureType())
      return isNSZone(recT->getDecl());
    return false;
  }

  bool isNSZoneVarUsed(VarDecl *D, std::vector<Expr *> &removals) {
    llvm::DenseSet<Expr *> refs;

    ReferenceCollector refColl(refs);
    refColl.lookFor(D, Body);

    ReferenceClear refClear(refs);
    refClear.clearRefsIn(removals.begin(), removals.end());

    return !refs.empty();
  }

  bool isAllocCopyWithZoneCall(ObjCMessageExpr *E) {
    if (E->getNumArgs() == 1 &&
        E->getSelector() == allocWithZoneSel &&
        (E->isClassMessage() ||
         Pass.TA.hasDiagnostic(diag::err_unavailable_message,
                               E->getReceiverRange().getBegin())))
      return true;

    return E->isInstanceMessage() &&
           E->getNumArgs() == 1   &&
           (E->getSelector() == copyWithZoneSel ||
            E->getSelector() == mutableCopyWithZoneSel);
  }

  bool isZoneCall(ObjCMessageExpr *E) {
    return E->isInstanceMessage() &&
           E->getNumArgs() == 0   &&
           E->getSelector() == zoneSel;
  }

  bool paramToAllocWithZoneHasSideEffects(Expr *E) {
    if (!HasSideEffects(E, Pass.Ctx))
      return false;
    E = E->IgnoreParenCasts();
    ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E);
    if (!ME)
      return true;
    if (!isZoneCall(ME))
      return true;
    return HasSideEffects(ME->getInstanceReceiver(), Pass.Ctx);
  }

  void clearUnavailableDiags(Stmt *S) {
    if (S)
      Pass.TA.clearDiagnostic(diag::err_unavailable,
                              diag::err_unavailable_message,
                              S->getSourceRange());
  }
};

} // end anonymous namespace

static void rewriteAllocCopyWithZone(MigrationPass &pass) {
  BodyTransform<AllocCopyWithZoneRewriter> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// rewriteAutoreleasePool
//===----------------------------------------------------------------------===//

/// \brief 'Loc' is the end of a statement range. This returns the location
/// immediately after the semicolon following the statement.
/// If no semicolon is found or the location is inside a macro, the returned
/// source location will be invalid.
static SourceLocation findLocationAfterSemi(ASTContext &Ctx,
                                            SourceLocation loc) {
  SourceManager &SM = Ctx.getSourceManager();
  if (loc.isMacroID()) {
    if (!SM.isAtEndOfMacroInstantiation(loc))
      return SourceLocation();
    loc = SM.getInstantiationRange(loc).second;
  }
  loc = Lexer::getLocForEndOfToken(loc, /*Offset=*/0, SM, Ctx.getLangOptions());

  // Break down the source location.
  std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(loc);

  // Try to load the file buffer.
  bool invalidTemp = false;
  llvm::StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
  if (invalidTemp)
    return SourceLocation();

  const char *tokenBegin = file.data() + locInfo.second;

  // Lex from the start of the given location.
  Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
              Ctx.getLangOptions(),
              file.begin(), tokenBegin, file.end());
  Token tok;
  lexer.LexFromRawLexer(tok);
  if (tok.isNot(tok::semi))
    return SourceLocation();

  return tok.getLocation().getFileLocWithOffset(1);
}

namespace {

class AutoreleasePoolRewriter
                         : public RecursiveASTVisitor<AutoreleasePoolRewriter> {
public:
  AutoreleasePoolRewriter(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) {
    PoolII = &pass.Ctx.Idents.get("NSAutoreleasePool");
    DrainSel = pass.Ctx.Selectors.getNullarySelector(
                                                 &pass.Ctx.Idents.get("drain"));
  }

  void transformBody(Stmt *body) {
    Body = body;
    TraverseStmt(body);
  }
  
  ~AutoreleasePoolRewriter() {
    llvm::SmallVector<VarDecl *, 8> VarsToHandle;

    for (std::map<VarDecl *, PoolVarInfo>::iterator
           I = PoolVars.begin(), E = PoolVars.end(); I != E; ++I) {
      VarDecl *var = I->first;
      PoolVarInfo &info = I->second;

      // Check that we can handle/rewrite all references of the pool.

      ReferenceClear refClear(info.Refs);
      refClear.clearRefsIn(info.Dcl);
      for (llvm::SmallVectorImpl<PoolScope>::iterator
             scpI = info.Scopes.begin(),
             scpE = info.Scopes.end(); scpI != scpE; ++scpI) {
        PoolScope &scope = *scpI;
        refClear.clearRefsIn(*scope.Begin);
        refClear.clearRefsIn(*scope.End);
        refClear.clearRefsIn(scope.Releases.begin(), scope.Releases.end());
      }

      // Even if one reference is not handled we will not do anything about that
      // pool variable.
      if (info.Refs.empty())
        VarsToHandle.push_back(var);
    }

    for (unsigned i = 0, e = VarsToHandle.size(); i != e; ++i) {
      PoolVarInfo &info = PoolVars[VarsToHandle[i]];

      Transaction Trans(Pass.TA);

      clearUnavailableDiags(info.Dcl);
      Pass.TA.removeStmt(info.Dcl);

      // Add "@autoreleasepool { }"
      for (llvm::SmallVectorImpl<PoolScope>::iterator
             scpI = info.Scopes.begin(),
             scpE = info.Scopes.end(); scpI != scpE; ++scpI) {
        PoolScope &scope = *scpI;
        clearUnavailableDiags(*scope.Begin);
        clearUnavailableDiags(*scope.End);
        if (scope.IsFollowedBySimpleReturnStmt) {
          // Include the return in the scope.
          Pass.TA.replaceStmt(*scope.Begin, "@autoreleasepool {");
          Pass.TA.removeStmt(*scope.End);
          Stmt::child_iterator retI = scope.End;
          ++retI;
          SourceLocation afterSemi = findLocationAfterSemi(Pass.Ctx,
                                                          (*retI)->getLocEnd());
          assert(afterSemi.isValid() &&
                 "Didn't we check before setting IsFollowedBySimpleReturnStmt "
                 "to true?");
          Pass.TA.insertAfterToken(afterSemi, "\n}");
          Pass.TA.increaseIndentation(
                                SourceRange(scope.getIndentedRange().getBegin(),
                                            (*retI)->getLocEnd()),
                                      scope.CompoundParent->getLocStart());
        } else {
          Pass.TA.replaceStmt(*scope.Begin, "@autoreleasepool {");
          Pass.TA.replaceStmt(*scope.End, "}");
          Pass.TA.increaseIndentation(scope.getIndentedRange(),
                                      scope.CompoundParent->getLocStart());
        }
      }

      // Remove rest of pool var references.
      for (llvm::SmallVectorImpl<PoolScope>::iterator
             scpI = info.Scopes.begin(),
             scpE = info.Scopes.end(); scpI != scpE; ++scpI) {
        PoolScope &scope = *scpI;
        for (llvm::SmallVectorImpl<ObjCMessageExpr *>::iterator
               relI = scope.Releases.begin(),
               relE = scope.Releases.end(); relI != relE; ++relI) {
          clearUnavailableDiags(*relI);
          Pass.TA.removeStmt(*relI);
        }
      }
    }
  }

  bool VisitCompoundStmt(CompoundStmt *S) {
    llvm::SmallVector<PoolScope, 4> Scopes;

    for (Stmt::child_iterator
           I = S->body_begin(), E = S->body_end(); I != E; ++I) {
      Stmt *child = getEssential(*I);
      if (DeclStmt *DclS = dyn_cast<DeclStmt>(child)) {
        if (DclS->isSingleDecl()) {
          if (VarDecl *VD = dyn_cast<VarDecl>(DclS->getSingleDecl())) {
            if (isNSAutoreleasePool(VD->getType())) {
              PoolVarInfo &info = PoolVars[VD];
              info.Dcl = DclS;
              ReferenceCollector refColl(info.Refs);
              refColl.lookFor(VD, S);
              // Does this statement follow the pattern:  
              // NSAutoreleasePool * pool = [NSAutoreleasePool  new];
              if (isPoolCreation(VD->getInit())) {
                Scopes.push_back(PoolScope());
                Scopes.back().PoolVar = VD;
                Scopes.back().CompoundParent = S;
                Scopes.back().Begin = I;
              }
            }
          }
        }
      } else if (BinaryOperator *bop = dyn_cast<BinaryOperator>(child)) {
        if (DeclRefExpr *dref = dyn_cast<DeclRefExpr>(bop->getLHS())) {
          if (VarDecl *VD = dyn_cast<VarDecl>(dref->getDecl())) {
            // Does this statement follow the pattern:  
            // pool = [NSAutoreleasePool  new];
            if (isNSAutoreleasePool(VD->getType()) &&
                isPoolCreation(bop->getRHS())) {
              Scopes.push_back(PoolScope());
              Scopes.back().PoolVar = VD;
              Scopes.back().CompoundParent = S;
              Scopes.back().Begin = I;
            }
          }
        }
      }

      if (Scopes.empty())
        continue;

      if (isPoolDrain(Scopes.back().PoolVar, child)) {
        PoolScope &scope = Scopes.back();
        scope.End = I;
        handlePoolScope(scope, S);
        Scopes.pop_back();
      }
    }
    return true;
  }

private:
  void clearUnavailableDiags(Stmt *S) {
    if (S)
      Pass.TA.clearDiagnostic(diag::err_unavailable,
                              diag::err_unavailable_message,
                              S->getSourceRange());
  }

  struct PoolScope {
    VarDecl *PoolVar;
    CompoundStmt *CompoundParent;
    Stmt::child_iterator Begin;
    Stmt::child_iterator End;
    bool IsFollowedBySimpleReturnStmt;
    llvm::SmallVector<ObjCMessageExpr *, 4> Releases;

    PoolScope() : PoolVar(0), CompoundParent(0), Begin(), End(),
                  IsFollowedBySimpleReturnStmt(false) { }

    SourceRange getIndentedRange() const {
      Stmt::child_iterator rangeS = Begin;
      ++rangeS;
      if (rangeS == End)
        return SourceRange();
      Stmt::child_iterator rangeE = Begin;
      for (Stmt::child_iterator I = rangeS; I != End; ++I)
        ++rangeE;
      return SourceRange((*rangeS)->getLocStart(), (*rangeE)->getLocEnd());
    }
  };

  class NameReferenceChecker : public RecursiveASTVisitor<NameReferenceChecker>{
    ASTContext &Ctx;
    SourceRange ScopeRange;
    SourceLocation &referenceLoc, &declarationLoc;

  public:
    NameReferenceChecker(ASTContext &ctx, PoolScope &scope,
                         SourceLocation &referenceLoc,
                         SourceLocation &declarationLoc)
      : Ctx(ctx), referenceLoc(referenceLoc),
        declarationLoc(declarationLoc) {
      ScopeRange = SourceRange((*scope.Begin)->getLocStart(),
                               (*scope.End)->getLocStart());
    }

    bool VisitDeclRefExpr(DeclRefExpr *E) {
      return checkRef(E->getLocation(), E->getDecl()->getLocation());
    }

    bool VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
      return checkRef(E->getLocation(), E->getDecl()->getLocation());
    }

    bool VisitTypedefTypeLoc(TypedefTypeLoc TL) {
      return checkRef(TL.getBeginLoc(), TL.getTypedefNameDecl()->getLocation());
    }

    bool VisitTagTypeLoc(TagTypeLoc TL) {
      return checkRef(TL.getBeginLoc(), TL.getDecl()->getLocation());
    }

  private:
    bool checkRef(SourceLocation refLoc, SourceLocation declLoc) {
      if (isInScope(declLoc)) {
        referenceLoc = refLoc;
        declarationLoc = declLoc;
        return false;
      }
      return true;
    }

    bool isInScope(SourceLocation loc) {
      SourceManager &SM = Ctx.getSourceManager();
      if (SM.isBeforeInTranslationUnit(loc, ScopeRange.getBegin()))
        return false;
      return SM.isBeforeInTranslationUnit(loc, ScopeRange.getEnd());
    }
  };

  void handlePoolScope(PoolScope &scope, CompoundStmt *compoundS) {
    // Check that all names declared inside the scope are not used
    // outside the scope.
    {
      bool nameUsedOutsideScope = false;
      SourceLocation referenceLoc, declarationLoc;
      Stmt::child_iterator SI = scope.End, SE = compoundS->body_end();
      ++SI;
      // Check if the autoreleasepool scope is followed by a simple return
      // statement, in which case we will include the return in the scope.
      if (SI != SE)
        if (ReturnStmt *retS = dyn_cast<ReturnStmt>(*SI))
          if ((retS->getRetValue() == 0 ||
               isa<DeclRefExpr>(retS->getRetValue()->IgnoreParenCasts())) &&
              findLocationAfterSemi(Pass.Ctx, retS->getLocEnd()).isValid()) {
            scope.IsFollowedBySimpleReturnStmt = true;
            ++SI; // the return will be included in scope, don't check it.
          }
      
      for (; SI != SE; ++SI) {
        nameUsedOutsideScope = !NameReferenceChecker(Pass.Ctx, scope,
                                                     referenceLoc,
                                              declarationLoc).TraverseStmt(*SI);
        if (nameUsedOutsideScope)
          break;
      }

      // If not all references were cleared it means some variables/typenames/etc
      // declared inside the pool scope are used outside of it.
      // We won't try to rewrite the pool.
      if (nameUsedOutsideScope) {
        Pass.TA.reportError("a name is referenced outside the "
            "NSAutoreleasePool scope that it was declared in", referenceLoc);
        Pass.TA.reportNote("name declared here", declarationLoc);
        Pass.TA.reportNote("intended @autoreleasepool scope begins here",
                           (*scope.Begin)->getLocStart());
        Pass.TA.reportNote("intended @autoreleasepool scope ends here",
                           (*scope.End)->getLocStart());
        return;
      }
    }

    // Collect all releases of the pool; they will be removed.
    {
      ReleaseCollector releaseColl(scope.PoolVar, scope.Releases);
      Stmt::child_iterator I = scope.Begin;
      ++I;
      for (; I != scope.End; ++I)
        releaseColl.TraverseStmt(*I);
    }

    PoolVars[scope.PoolVar].Scopes.push_back(scope);
  }

  bool isPoolCreation(Expr *E) {
    if (!E) return false;
    E = getEssential(E);
    ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E);
    if (!ME) return false;
    if (ME->getMethodFamily() == OMF_new &&
        ME->getReceiverKind() == ObjCMessageExpr::Class &&
        isNSAutoreleasePool(ME->getReceiverInterface()))
      return true;
    if (ME->getReceiverKind() == ObjCMessageExpr::Instance &&
        ME->getMethodFamily() == OMF_init) {
      Expr *rec = getEssential(ME->getInstanceReceiver());
      if (ObjCMessageExpr *recME = dyn_cast_or_null<ObjCMessageExpr>(rec)) {
        if (recME->getMethodFamily() == OMF_alloc &&
            recME->getReceiverKind() == ObjCMessageExpr::Class &&
            isNSAutoreleasePool(recME->getReceiverInterface()))
          return true;
      }
    }

    return false;
  }

  bool isPoolDrain(VarDecl *poolVar, Stmt *S) {
    if (!S) return false;
    S = getEssential(S);
    ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S);
    if (!ME) return false;
    if (ME->getReceiverKind() == ObjCMessageExpr::Instance) {
      Expr *rec = getEssential(ME->getInstanceReceiver());
      if (DeclRefExpr *dref = dyn_cast<DeclRefExpr>(rec))
        if (dref->getDecl() == poolVar)
          return ME->getMethodFamily() == OMF_release ||
                 ME->getSelector() == DrainSel;
    }

    return false;
  }

  bool isNSAutoreleasePool(ObjCInterfaceDecl *IDecl) {
    return IDecl && IDecl->getIdentifier() == PoolII;
  }

  bool isNSAutoreleasePool(QualType Ty) {
    QualType pointee = Ty->getPointeeType();
    if (pointee.isNull())
      return false;
    if (const ObjCInterfaceType *interT = pointee->getAs<ObjCInterfaceType>())
      return isNSAutoreleasePool(interT->getDecl());
    return false;
  }

  static Expr *getEssential(Expr *E) {
    return cast<Expr>(getEssential((Stmt*)E));
  }
  static Stmt *getEssential(Stmt *S) {
    if (ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(S))
      S = EWC->getSubExpr();
    if (Expr *E = dyn_cast<Expr>(S))
      S = E->IgnoreParenCasts();
    return S;
  }

  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  IdentifierInfo *PoolII;
  Selector DrainSel;
  
  struct PoolVarInfo {
    DeclStmt *Dcl;
    llvm::DenseSet<Expr *> Refs;
    llvm::SmallVector<PoolScope, 2> Scopes;

    PoolVarInfo() : Dcl(0) { }
  };

  std::map<VarDecl *, PoolVarInfo> PoolVars;
};

} // anonymous namespace

static void rewriteAutoreleasePool(MigrationPass &pass) {
  BodyTransform<AutoreleasePoolRewriter> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// removeRetainReleaseDealloc
//===----------------------------------------------------------------------===//

namespace {

class RetainReleaseDeallocRemover :
                       public RecursiveASTVisitor<RetainReleaseDeallocRemover> {
  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  llvm::DenseSet<Expr *> Removables;
  llvm::OwningPtr<ParentMap> StmtMap;

public:
  RetainReleaseDeallocRemover(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) { }

  void transformBody(Stmt *body) {
    Body = body;
    RemovablesCollector(Removables).TraverseStmt(body);
    StmtMap.reset(new ParentMap(body));
    TraverseStmt(body);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    switch (E->getMethodFamily()) {
    default:
      return true;
    case OMF_retain:
    case OMF_release:
    case OMF_autorelease:
      if (E->getReceiverKind() == ObjCMessageExpr::Instance)
        if (Expr *rec = E->getInstanceReceiver()) {
          rec = rec->IgnoreParenImpCasts();
          if (rec->getType().getObjCLifetime() == Qualifiers::OCL_ExplicitNone){
            std::string err = "It is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "an __unsafe_unretained type";
            Pass.TA.reportError(err, rec->getLocStart());
            return true;
          }
        }
    case OMF_dealloc:
      break;
    }

    switch (E->getReceiverKind()) {
    default:
      return true;
    case ObjCMessageExpr::SuperInstance: {
      Transaction Trans(Pass.TA);
      Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                              diag::err_unavailable,
                              diag::err_unavailable_message,
                              E->getSuperLoc());
      if (tryRemoving(E))
        return true;
      Pass.TA.replace(E->getSourceRange(), "self");
      return true;
    }
    case ObjCMessageExpr::Instance:
      break;
    }

    Expr *rec = E->getInstanceReceiver();
    if (!rec) return true;

    Transaction Trans(Pass.TA);
    Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                            diag::err_unavailable,
                            diag::err_unavailable_message,
                            rec->getExprLoc());
    if (!HasSideEffects(E, Pass.Ctx)) {
      if (tryRemoving(E))
        return true;
    }
    Pass.TA.replace(E->getSourceRange(), rec->getSourceRange());

    return true;
  }

private:
  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
  
  bool tryRemoving(Expr *E) const {
    if (isRemovable(E)) {
      Pass.TA.removeStmt(E);
      return true;
    }

    if (ParenExpr *parenE = dyn_cast_or_null<ParenExpr>(StmtMap->getParent(E)))
      return tryRemoving(parenE);

    if (BinaryOperator *
          bopE = dyn_cast_or_null<BinaryOperator>(StmtMap->getParent(E))) {
      if (bopE->getOpcode() == BO_Comma && bopE->getLHS() == E &&
          isRemovable(bopE)) {
        Pass.TA.replace(bopE->getSourceRange(), bopE->getRHS()->getSourceRange());
        return true;
      }
    }

    return false;
  }

};

} // anonymous namespace

static void removeRetainReleaseDealloc(MigrationPass &pass) {
  BodyTransform<RetainReleaseDeallocRemover> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// removeEmptyStatements
//===----------------------------------------------------------------------===//

namespace {

class EmptyStatementsRemover :
                            public RecursiveASTVisitor<EmptyStatementsRemover> {
  MigrationPass &Pass;
  llvm::DenseSet<unsigned> MacroLocs;

public:
  EmptyStatementsRemover(MigrationPass &pass) : Pass(pass) {
    for (unsigned i = 0, e = Pass.ARCMTMacroLocs.size(); i != e; ++i)
      MacroLocs.insert(Pass.ARCMTMacroLocs[i].getRawEncoding());
  }

  bool TraverseStmtExpr(StmtExpr *E) {
    CompoundStmt *S = E->getSubStmt();
    for (CompoundStmt::body_iterator
           I = S->body_begin(), E = S->body_end(); I != E; ++I) {
      if (I != E - 1)
        check(*I);
      TraverseStmt(*I);
    }
    return true;
  }

  bool VisitCompoundStmt(CompoundStmt *S) {
    for (CompoundStmt::body_iterator
           I = S->body_begin(), E = S->body_end(); I != E; ++I)
      check(*I);
    return true;
  }

  bool isMacroLoc(SourceLocation loc) {
    if (loc.isInvalid()) return false;
    return MacroLocs.count(loc.getRawEncoding());
  }

  ASTContext &getContext() { return Pass.Ctx; }

private:
  /// \brief Returns true if the statement became empty due to previous
  /// transformations.
  class EmptyChecker : public StmtVisitor<EmptyChecker, bool> {
    EmptyStatementsRemover &Trans;

  public:
    EmptyChecker(EmptyStatementsRemover &trans) : Trans(trans) { }

    bool VisitNullStmt(NullStmt *S) {
      return Trans.isMacroLoc(S->getLeadingEmptyMacroLoc());
    }
    bool VisitCompoundStmt(CompoundStmt *S) {
      if (S->body_empty())
        return false; // was already empty, not because of transformations.
      for (CompoundStmt::body_iterator
             I = S->body_begin(), E = S->body_end(); I != E; ++I)
        if (!Visit(*I))
          return false;
      return true;
    }
    bool VisitIfStmt(IfStmt *S) {
      if (S->getConditionVariable())
        return false;
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (HasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getThen() || !Visit(S->getThen()))
        return false;
      if (S->getElse() && !Visit(S->getElse()))
        return false;
      return true;
    }
    bool VisitWhileStmt(WhileStmt *S) {
      if (S->getConditionVariable())
        return false;
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (HasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitDoStmt(DoStmt *S) {
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (HasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
      Expr *Exp = S->getCollection();
      if (!Exp)
        return false;
      if (HasSideEffects(Exp, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
      if (!S->getSubStmt())
        return false;
      return Visit(S->getSubStmt());
    }
  };

  void check(Stmt *S) {
    if (!S) return;
    if (EmptyChecker(*this).Visit(S)) {
      Transaction Trans(Pass.TA);
      Pass.TA.removeStmt(S);
    }
  }
};

} // anonymous namespace

static void removeEmptyStatements(MigrationPass &pass) {
  EmptyStatementsRemover(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());

  for (unsigned i = 0, e = pass.ARCMTMacroLocs.size(); i != e; ++i) {
    Transaction Trans(pass.TA);
    pass.TA.remove(pass.ARCMTMacroLocs[i]);
  }
}

//===----------------------------------------------------------------------===//
// changeIvarsOfAssignProperties.
//===----------------------------------------------------------------------===//

namespace {

class AssignPropertiesTrans {
  MigrationPass &Pass;
  struct PropData {
    ObjCPropertyDecl *PropD;
    ObjCIvarDecl *IvarD;
    bool ShouldChangeToWeak;
    SourceLocation ArcPropAssignErrorLoc;
  };

  typedef llvm::SmallVector<PropData, 2> PropsTy; 
  typedef llvm::DenseMap<unsigned, PropsTy> PropsMapTy;
  PropsMapTy PropsMap;

public:
  AssignPropertiesTrans(MigrationPass &pass) : Pass(pass) { }

  void doTransform(ObjCImplementationDecl *D) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    ObjCInterfaceDecl *IFace = D->getClassInterface();
    for (ObjCInterfaceDecl::prop_iterator
           I = IFace->prop_begin(), E = IFace->prop_end(); I != E; ++I) {
      ObjCPropertyDecl *propD = *I;
      unsigned loc = SM.getInstantiationLoc(propD->getAtLoc()).getRawEncoding();
      PropsTy &props = PropsMap[loc];
      props.push_back(PropData());
      props.back().PropD = propD;
      props.back().IvarD = 0;
      props.back().ShouldChangeToWeak = false;
    }

    typedef DeclContext::specific_decl_iterator<ObjCPropertyImplDecl>
        prop_impl_iterator;
    for (prop_impl_iterator
           I = prop_impl_iterator(D->decls_begin()),
           E = prop_impl_iterator(D->decls_end()); I != E; ++I) {
      VisitObjCPropertyImplDecl(*I);
    }

    for (PropsMapTy::iterator
           I = PropsMap.begin(), E = PropsMap.end(); I != E; ++I) {
      SourceLocation atLoc = SourceLocation::getFromRawEncoding(I->first);
      PropsTy &props = I->second;
      if (shouldApplyWeakToAllProp(props)) {
        if (changeAssignToWeak(atLoc)) {
          // Couldn't add the 'weak' property attribute,
          // try adding __unsafe_unretained.
          applyUnsafeUnretained(props);
        } else {
          for (PropsTy::iterator
                 PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
            applyWeak(*PI);
          }
        }
      } else {
        // We should not add 'weak' attribute since not all properties need it.
        // So just add __unsafe_unretained to the ivars.
        applyUnsafeUnretained(props);
      }
    }
  }

  bool shouldApplyWeakToAllProp(PropsTy &props) {
    for (PropsTy::iterator
           PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
      if (!PI->ShouldChangeToWeak)
        return false;
    }
    return true;
  }

  void applyWeak(PropData &prop) {
    assert(!Pass.Ctx.getLangOptions().ObjCNoAutoRefCountRuntime);

    Transaction Trans(Pass.TA);
    Pass.TA.insert(prop.IvarD->getLocation(), "__weak "); 
    Pass.TA.clearDiagnostic(diag::err_arc_assign_property_lifetime,
                            prop.ArcPropAssignErrorLoc);
  }

  void applyUnsafeUnretained(PropsTy &props) {
    for (PropsTy::iterator
           PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
      if (PI->ShouldChangeToWeak) {
        Transaction Trans(Pass.TA);
        Pass.TA.insert(PI->IvarD->getLocation(), "__unsafe_unretained ");
        Pass.TA.clearDiagnostic(diag::err_arc_assign_property_lifetime,
                                PI->ArcPropAssignErrorLoc);
      }
    }
  }

  bool VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    if (D->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      return true;
    ObjCPropertyDecl *propD = D->getPropertyDecl();
    if (!propD || propD->isInvalidDecl())
      return true;
    ObjCIvarDecl *ivarD = D->getPropertyIvarDecl();
    if (!ivarD || ivarD->isInvalidDecl())
      return true;
    if (!(propD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_assign))
      return true;
    if (isa<AttributedType>(ivarD->getType().getTypePtr()))
      return true;
    if (ivarD->getType().getLocalQualifiers().getObjCLifetime()
          != Qualifiers::OCL_Strong)
      return true;
    if (!Pass.TA.hasDiagnostic(
                      diag::err_arc_assign_property_lifetime, D->getLocation()))
      return true;

    // There is a "error: existing ivar for assign property must be
    // __unsafe_unretained"; fix it.

    if (Pass.Ctx.getLangOptions().ObjCNoAutoRefCountRuntime) {
      // We will just add __unsafe_unretained to the ivar.
      Transaction Trans(Pass.TA);
      Pass.TA.insert(ivarD->getLocation(), "__unsafe_unretained ");
      Pass.TA.clearDiagnostic(
                      diag::err_arc_assign_property_lifetime, D->getLocation());
    } else {
      // Mark that we want the ivar to become weak.
      unsigned loc = SM.getInstantiationLoc(propD->getAtLoc()).getRawEncoding();
      PropsTy &props = PropsMap[loc];
      for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
        if (I->PropD == propD) {
          I->IvarD = ivarD;
          I->ShouldChangeToWeak = true;
          I->ArcPropAssignErrorLoc = D->getLocation();
        }
      }
    }

    return true;
  }

private:
  bool changeAssignToWeak(SourceLocation atLoc) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    llvm::StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
    if (invalidTemp)
      return true;

    const char *tokenBegin = file.data() + locInfo.second;

    // Lex from the start of the given location.
    Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
                Pass.Ctx.getLangOptions(),
                file.begin(), tokenBegin, file.end());
    Token tok;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::at)) return true;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::raw_identifier)) return true;
    if (llvm::StringRef(tok.getRawIdentifierData(), tok.getLength())
          != "property")
      return true;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::l_paren)) return true;
    
    SourceLocation LParen = tok.getLocation();
    SourceLocation assignLoc;
    bool isEmpty = false;

    lexer.LexFromRawLexer(tok);
    if (tok.is(tok::r_paren)) {
      isEmpty = true;
    } else {
      while (1) {
        if (tok.isNot(tok::raw_identifier)) return true;
        llvm::StringRef ident(tok.getRawIdentifierData(), tok.getLength());
        if (ident == "assign")
          assignLoc = tok.getLocation();
  
        do {
          lexer.LexFromRawLexer(tok);
        } while (tok.isNot(tok::comma) && tok.isNot(tok::r_paren));
        if (tok.is(tok::r_paren))
          break;
        lexer.LexFromRawLexer(tok);
      }
    }

    Transaction Trans(Pass.TA);
    if (assignLoc.isValid())
      Pass.TA.replaceText(assignLoc, "assign", "weak");
    else 
      Pass.TA.insertAfterToken(LParen, isEmpty ? "weak" : "weak, ");
    return false;
  }
};

class PropertiesChecker : public RecursiveASTVisitor<PropertiesChecker> {
  MigrationPass &Pass;

public:
  PropertiesChecker(MigrationPass &pass) : Pass(pass) { }

  bool TraverseObjCImplementationDecl(ObjCImplementationDecl *D) {
    AssignPropertiesTrans(Pass).doTransform(D);
    return true;
  }
};

} // anonymous namespace

static void changeIvarsOfAssignProperties(MigrationPass &pass) {
  PropertiesChecker(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// rewriteUnusedDelegateInit
//===----------------------------------------------------------------------===//

namespace {

class UnusedInitRewriter : public RecursiveASTVisitor<UnusedInitRewriter> {
  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  llvm::DenseSet<Expr *> Removables;

public:
  UnusedInitRewriter(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) { }

  void transformBody(Stmt *body) {
    Body = body;
    RemovablesCollector(Removables).TraverseStmt(body);
    TraverseStmt(body);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *ME) {
    if (ME->isDelegateInitCall() &&
        isRemovable(ME) &&
        Pass.TA.hasDiagnostic(diag::err_arc_unused_init_message,
                              ME->getExprLoc())) {
      Transaction Trans(Pass.TA);
      Pass.TA.clearDiagnostic(diag::err_arc_unused_init_message,
                              ME->getExprLoc());
      Pass.TA.insert(ME->getExprLoc(), "self = ");
    }
    return true;
  }

private:
  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
};

} // anonymous namespace

static void rewriteUnusedDelegateInit(MigrationPass &pass) {
  BodyTransform<UnusedInitRewriter> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// rewriteBlockObjCVariable
//===----------------------------------------------------------------------===//

namespace {

class RootBlockObjCVarRewriter :
                          public RecursiveASTVisitor<RootBlockObjCVarRewriter> {
  MigrationPass &Pass;
  llvm::DenseSet<VarDecl *> CheckedVars;

  class BlockVarChecker : public RecursiveASTVisitor<BlockVarChecker> {
    VarDecl *Var;
  
    typedef RecursiveASTVisitor<BlockVarChecker> base;
  public:
    BlockVarChecker(VarDecl *var) : Var(var) { }
  
    bool TraverseImplicitCastExpr(ImplicitCastExpr *castE) {
      if (BlockDeclRefExpr *
            ref = dyn_cast<BlockDeclRefExpr>(castE->getSubExpr())) {
        if (ref->getDecl() == Var) {
          if (castE->getCastKind() == CK_LValueToRValue)
            return true; // Using the value of the variable.
          if (castE->getCastKind() == CK_NoOp && castE->isLValue() &&
              Var->getASTContext().getLangOptions().CPlusPlus)
            return true; // Binding to const C++ reference.
        }
      }

      return base::TraverseImplicitCastExpr(castE);
    }

    bool VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
      if (E->getDecl() == Var)
        return false; // The reference of the variable, and not just its value,
                      //  is needed.
      return true;
    }
  };

public:
  RootBlockObjCVarRewriter(MigrationPass &pass) : Pass(pass) { }

  bool VisitBlockDecl(BlockDecl *block) {
    llvm::SmallVector<VarDecl *, 4> BlockVars;
    
    for (BlockDecl::capture_iterator
           I = block->capture_begin(), E = block->capture_end(); I != E; ++I) {
      VarDecl *var = I->getVariable();
      if (I->isByRef() &&
          !isAlreadyChecked(var) &&
          var->getType()->isObjCObjectPointerType() &&
          isImplicitStrong(var->getType())) {
        BlockVars.push_back(var);
      }
    }

    for (unsigned i = 0, e = BlockVars.size(); i != e; ++i) {
      VarDecl *var = BlockVars[i];
      CheckedVars.insert(var);

      BlockVarChecker checker(var);
      bool onlyValueOfVarIsNeeded = checker.TraverseStmt(block->getBody());
      if (onlyValueOfVarIsNeeded) {
        BlocksAttr *attr = var->getAttr<BlocksAttr>();
        if(!attr)
          continue;
        bool hasARCRuntime = !Pass.Ctx.getLangOptions().ObjCNoAutoRefCountRuntime;
        SourceManager &SM = Pass.Ctx.getSourceManager();
        Transaction Trans(Pass.TA);
        Pass.TA.replaceText(SM.getInstantiationLoc(attr->getLocation()),
                            "__block",
                            hasARCRuntime ? "__weak" : "__unsafe_unretained");
      }

    }

    return true;
  }

private:
  bool isAlreadyChecked(VarDecl *VD) {
    return CheckedVars.count(VD);
  }

  bool isImplicitStrong(QualType ty) {
    if (isa<AttributedType>(ty.getTypePtr()))
      return false;
    return ty.getLocalQualifiers().getObjCLifetime() == Qualifiers::OCL_Strong;
  }
};

class BlockObjCVarRewriter : public RecursiveASTVisitor<BlockObjCVarRewriter> {
  MigrationPass &Pass;

public:
  BlockObjCVarRewriter(MigrationPass &pass) : Pass(pass) { }

  bool TraverseBlockDecl(BlockDecl *block) {
    RootBlockObjCVarRewriter(Pass).TraverseDecl(block);
    return true;
  }
};

} // anonymous namespace

static void rewriteBlockObjCVariable(MigrationPass &pass) {
  BlockObjCVarRewriter trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// removeZeroOutIvarsInDealloc
//===----------------------------------------------------------------------===//

namespace {

class ZeroOutInDeallocRemover :
                           public RecursiveASTVisitor<ZeroOutInDeallocRemover> {
  typedef RecursiveASTVisitor<ZeroOutInDeallocRemover> base;

  MigrationPass &Pass;

  llvm::DenseMap<ObjCPropertyDecl*, ObjCPropertyImplDecl*> SynthesizedProperties;
  ImplicitParamDecl *SelfD;
  llvm::DenseSet<Expr *> Removables;

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
    RemovablesCollector(Removables).TraverseStmt(D->getBody());

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

static void removeZeroOutIvarsInDealloc(MigrationPass &pass) {
  ZeroOutInDeallocRemover trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}

//===----------------------------------------------------------------------===//
// getAllTransformations.
//===----------------------------------------------------------------------===//

static void independentTransforms(MigrationPass &pass) {
  rewriteAutoreleasePool(pass);
  changeIvarsOfAssignProperties(pass);
  removeRetainReleaseDealloc(pass);
  rewriteUnusedDelegateInit(pass);
  removeZeroOutIvarsInDealloc(pass);
  makeAssignARCSafe(pass);
  castNonObjCToObjC(pass);
  rewriteBlockObjCVariable(pass);
  rewriteAllocCopyWithZone(pass);
}

std::vector<TransformFn> arcmt::getAllTransformations() {
  std::vector<TransformFn> transforms;

  // This must come first since rewriteAutoreleasePool depends on -release
  // calls being present to determine the @autorelease ending scope.
  transforms.push_back(independentTransforms);

  transforms.push_back(removeEmptyStatements);
  transforms.push_back(removeDeallocMethod);

  return transforms;
}
