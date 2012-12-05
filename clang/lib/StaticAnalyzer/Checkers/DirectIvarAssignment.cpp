//=- DirectIvarAssignment.cpp - Check rules on ObjC properties -*- C++ ----*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Check that Objective C properties follow the following rules:
//    - The property should be set with the setter, not though a direct
//      assignment.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/ADT/DenseMap.h"

using namespace clang;
using namespace ento;

namespace {

/// The default method filter, which is used to filter out the methods on which
/// the check should not be performed.
///
/// Checks for the init, dealloc, and any other functions that might be allowed
/// to perform direct instance variable assignment based on their name.
struct MethodFilter {
  virtual bool operator()(ObjCMethodDecl *M) {
    if (M->getMethodFamily() == OMF_init ||
        M->getMethodFamily() == OMF_dealloc ||
        M->getMethodFamily() == OMF_copy ||
        M->getMethodFamily() == OMF_mutableCopy ||
        M->getSelector().getNameForSlot(0).find("init") != StringRef::npos ||
        M->getSelector().getNameForSlot(0).find("Init") != StringRef::npos)
      return true;
    return false;
  }
};

static MethodFilter DefaultMethodFilter;

class DirectIvarAssignment :
  public Checker<check::ASTDecl<ObjCImplementationDecl> > {

  typedef llvm::DenseMap<const ObjCIvarDecl*,
                         const ObjCPropertyDecl*> IvarToPropertyMapTy;

  /// A helper class, which walks the AST and locates all assignments to ivars
  /// in the given function.
  class MethodCrawler : public ConstStmtVisitor<MethodCrawler> {
    const IvarToPropertyMapTy &IvarToPropMap;
    const ObjCMethodDecl *MD;
    const ObjCInterfaceDecl *InterfD;
    BugReporter &BR;
    LocationOrAnalysisDeclContext DCtx;

  public:
    MethodCrawler(const IvarToPropertyMapTy &InMap, const ObjCMethodDecl *InMD,
        const ObjCInterfaceDecl *InID,
        BugReporter &InBR, AnalysisDeclContext *InDCtx)
    : IvarToPropMap(InMap), MD(InMD), InterfD(InID), BR(InBR), DCtx(InDCtx) {}

    void VisitStmt(const Stmt *S) { VisitChildren(S); }

    void VisitBinaryOperator(const BinaryOperator *BO);

    void VisitChildren(const Stmt *S) {
      for (Stmt::const_child_range I = S->children(); I; ++I)
        if (*I)
         this->Visit(*I);
    }
  };

public:
  MethodFilter *ShouldSkipMethod;

  DirectIvarAssignment() : ShouldSkipMethod(&DefaultMethodFilter) {}

  void checkASTDecl(const ObjCImplementationDecl *D, AnalysisManager& Mgr,
                    BugReporter &BR) const;
};

static const ObjCIvarDecl *findPropertyBackingIvar(const ObjCPropertyDecl *PD,
                                               const ObjCInterfaceDecl *InterD,
                                               ASTContext &Ctx) {
  // Check for synthesized ivars.
  ObjCIvarDecl *ID = PD->getPropertyIvarDecl();
  if (ID)
    return ID;

  ObjCInterfaceDecl *NonConstInterD = const_cast<ObjCInterfaceDecl*>(InterD);

  // Check for existing "_PropName".
  ID = NonConstInterD->lookupInstanceVariable(PD->getDefaultSynthIvarName(Ctx));
  if (ID)
    return ID;

  // Check for existing "PropName".
  IdentifierInfo *PropIdent = PD->getIdentifier();
  ID = NonConstInterD->lookupInstanceVariable(PropIdent);

  return ID;
}

void DirectIvarAssignment::checkASTDecl(const ObjCImplementationDecl *D,
                                       AnalysisManager& Mgr,
                                       BugReporter &BR) const {
  const ObjCInterfaceDecl *InterD = D->getClassInterface();


  IvarToPropertyMapTy IvarToPropMap;

  // Find all properties for this class.
  for (ObjCInterfaceDecl::prop_iterator I = InterD->prop_begin(),
      E = InterD->prop_end(); I != E; ++I) {
    ObjCPropertyDecl *PD = *I;

    // Find the corresponding IVar.
    const ObjCIvarDecl *ID = findPropertyBackingIvar(PD, InterD,
                                                     Mgr.getASTContext());

    if (!ID)
      continue;

    // Store the IVar to property mapping.
    IvarToPropMap[ID] = PD;
  }

  if (IvarToPropMap.empty())
    return;

  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(),
      E = D->instmeth_end(); I != E; ++I) {

    ObjCMethodDecl *M = *I;
    AnalysisDeclContext *DCtx = Mgr.getAnalysisDeclContext(M);

    if ((*ShouldSkipMethod)(M))
      continue;

    const Stmt *Body = M->getBody();
    assert(Body);

    MethodCrawler MC(IvarToPropMap, M->getCanonicalDecl(), InterD, BR, DCtx);
    MC.VisitStmt(Body);
  }
}

void DirectIvarAssignment::MethodCrawler::VisitBinaryOperator(
                                                    const BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;

  const ObjCIvarRefExpr *IvarRef =
          dyn_cast<ObjCIvarRefExpr>(BO->getLHS()->IgnoreParenCasts());

  if (!IvarRef)
    return;

  if (const ObjCIvarDecl *D = IvarRef->getDecl()) {
    IvarToPropertyMapTy::const_iterator I = IvarToPropMap.find(D);
    if (I != IvarToPropMap.end()) {
      const ObjCPropertyDecl *PD = I->second;

      ObjCMethodDecl *GetterMethod =
          InterfD->getInstanceMethod(PD->getGetterName());
      ObjCMethodDecl *SetterMethod =
          InterfD->getInstanceMethod(PD->getSetterName());

      if (SetterMethod && SetterMethod->getCanonicalDecl() == MD)
        return;

      if (GetterMethod && GetterMethod->getCanonicalDecl() == MD)
        return;

      BR.EmitBasicReport(MD,
          "Property access",
          categories::CoreFoundationObjectiveC,
          "Direct assignment to an instance variable backing a property; "
          "use the setter instead", PathDiagnosticLocation(IvarRef,
                                                          BR.getSourceManager(),
                                                          DCtx));
    }
  }
}
}

// Register the checker that checks for direct accesses in all functions,
// except for the initialization and copy routines.
void ento::registerDirectIvarAssignment(CheckerManager &mgr) {
  mgr.registerChecker<DirectIvarAssignment>();
}

// Register the checker that checks for direct accesses in functions annotated
// with __attribute__((annotate("objc_no_direct_instance_variable_assignmemt"))).
namespace {
struct InvalidatorMethodFilter : MethodFilter {
  virtual bool operator()(ObjCMethodDecl *M) {
    for (specific_attr_iterator<AnnotateAttr>
         AI = M->specific_attr_begin<AnnotateAttr>(),
         AE = M->specific_attr_end<AnnotateAttr>(); AI != AE; ++AI) {
      const AnnotateAttr *Ann = *AI;
      if (Ann->getAnnotation() == "objc_no_direct_instance_variable_assignmemt")
        return false;
    }
    return true;
  }
};

InvalidatorMethodFilter AttrFilter;
}

void ento::registerDirectIvarAssignmentForAnnotatedFunctions(
    CheckerManager &mgr) {
  mgr.registerChecker<DirectIvarAssignment>()->ShouldSkipMethod = &AttrFilter;
}
