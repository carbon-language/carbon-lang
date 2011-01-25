//== ObjCSelfInitChecker.cpp - Checker for 'self' initialization -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines ObjCSelfInitChecker, a builtin check that checks for uses of
// 'self' before proper initialization.
//
//===----------------------------------------------------------------------===//

// This checks initialization methods to verify that they assign 'self' to the
// result of an initialization call (e.g. [super init], or [self initWith..])
// before using 'self' or any instance variable.
//
// To perform the required checking, values are tagged wih flags that indicate
// 1) if the object is the one pointed to by 'self', and 2) if the object
// is the result of an initializer (e.g. [super init]).
//
// Uses of an object that is true for 1) but not 2) trigger a diagnostic.
// The uses that are currently checked are:
//  - Using instance variables.
//  - Returning the object.
//
// Note that we don't check for an invalid 'self' that is the receiver of an
// obj-c message expression to cut down false positives where logging functions
// get information from self (like its class) or doing "invalidation" on self
// when the initialization fails.
//
// Because the object that 'self' points to gets invalidated when a call
// receives a reference to 'self', the checker keeps track and passes the flags
// for 1) and 2) to the new object that 'self' points to after the call.
//
// FIXME (rdar://7937506): In the case of:
//   [super init];
//   return self;
// Have an extra PathDiagnosticPiece in the path that says "called [super init],
// but didn't assign the result to self."

//===----------------------------------------------------------------------===//

// FIXME: Somehow stick the link to Apple's documentation about initializing
// objects in the diagnostics.
// http://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/ObjectiveC/Articles/ocAllocInit.html

#include "ExprEngineInternalChecks.h"
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/PathSensitive/GRStateTrait.h"
#include "clang/StaticAnalyzer/BugReporter/BugType.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/AST/ParentMap.h"

using namespace clang;
using namespace ento;

static bool shouldRunOnFunctionOrMethod(const NamedDecl *ND);
static bool isInitializationMethod(const ObjCMethodDecl *MD);
static bool isInitMessage(const ObjCMessage &msg);
static bool isSelfVar(SVal location, CheckerContext &C);

namespace {
enum SelfFlagEnum {
  /// \brief No flag set.
  SelfFlag_None = 0x0,
  /// \brief Value came from 'self'.
  SelfFlag_Self    = 0x1,
  /// \brief Value came from the result of an initializer (e.g. [super init]).
  SelfFlag_InitRes = 0x2
};
}

namespace {
class ObjCSelfInitChecker : public CheckerVisitor<ObjCSelfInitChecker> {
  /// \brief A call receiving a reference to 'self' invalidates the object that
  /// 'self' contains. This field keeps the "self flags" assigned to the 'self'
  /// object before the call and assign them to the new object that 'self'
  /// points to after the call.
  SelfFlagEnum preCallSelfFlags;

public:
  static void *getTag() { static int tag = 0; return &tag; }
  void postVisitObjCMessage(CheckerContext &C, ObjCMessage msg);
  void PostVisitObjCIvarRefExpr(CheckerContext &C, const ObjCIvarRefExpr *E);
  void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *S);
  void PreVisitGenericCall(CheckerContext &C, const CallExpr *CE);
  void PostVisitGenericCall(CheckerContext &C, const CallExpr *CE);
  virtual void visitLocation(CheckerContext &C, const Stmt *S, SVal location,
                             bool isLoad);
};
} // end anonymous namespace

void ento::registerObjCSelfInitChecker(ExprEngine &Eng) {
  if (Eng.getContext().getLangOptions().ObjC1)
    Eng.registerCheck(new ObjCSelfInitChecker());
}

namespace {

class InitSelfBug : public BugType {
  const std::string desc;
public:
  InitSelfBug() : BugType("missing \"self = [{initializer}]\"",
                          "missing \"self = [{initializer}]\"") {}
};

} // end anonymous namespace

typedef llvm::ImmutableMap<SymbolRef, unsigned> SelfFlag;

namespace clang {
namespace ento {
  template<>
  struct GRStateTrait<SelfFlag> : public GRStatePartialTrait<SelfFlag> {
    static void* GDMIndex() {
      static int index = 0;
      return &index;
    }
  };
}
}

static SelfFlagEnum getSelfFlags(SVal val, const GRState *state) {
  if (SymbolRef sym = val.getAsSymbol())
    if (const unsigned *attachedFlags = state->get<SelfFlag>(sym))
      return (SelfFlagEnum)*attachedFlags;
  return SelfFlag_None;
}

static SelfFlagEnum getSelfFlags(SVal val, CheckerContext &C) {
  return getSelfFlags(val, C.getState());
}

static void addSelfFlag(SVal val, SelfFlagEnum flag, CheckerContext &C) {
  const GRState *state = C.getState();
  // FIXME: We tag the symbol that the SVal wraps but this is conceptually
  // wrong, we should tag the SVal; the fact that there is a symbol behind the
  // SVal is irrelevant.
  if (SymbolRef sym = val.getAsSymbol())
    C.addTransition(state->set<SelfFlag>(sym, getSelfFlags(val, C) | flag));
}

static bool hasSelfFlag(SVal val, SelfFlagEnum flag, CheckerContext &C) {
  return getSelfFlags(val, C) & flag;
}

/// \brief Returns true of the value of the expression is the object that 'self'
/// points to and is an object that did not come from the result of calling
/// an initializer.
static bool isInvalidSelf(const Expr *E, CheckerContext &C) {
  SVal exprVal = C.getState()->getSVal(E);
  if (!hasSelfFlag(exprVal, SelfFlag_Self, C))
    return false; // value did not come from 'self'.
  if (hasSelfFlag(exprVal, SelfFlag_InitRes, C))
    return false; // 'self' is properly initialized.

  return true;
}

static void checkForInvalidSelf(const Expr *E, CheckerContext &C,
                                const char *errorStr) {
  if (!E)
    return;
  if (!isInvalidSelf(E, C))
    return;

  // Generate an error node.
  ExplodedNode *N = C.generateSink();
  if (!N)
    return;

  EnhancedBugReport *report =
    new EnhancedBugReport(*new InitSelfBug(), errorStr, N);
  C.EmitReport(report);
}

void ObjCSelfInitChecker::postVisitObjCMessage(CheckerContext &C,
                                               ObjCMessage msg) {
  // When encountering a message that does initialization (init rule),
  // tag the return value so that we know later on that if self has this value
  // then it is properly initialized.

  // FIXME: A callback should disable checkers at the start of functions.
  if (!shouldRunOnFunctionOrMethod(dyn_cast<NamedDecl>(
                                     C.getCurrentAnalysisContext()->getDecl())))
    return;

  if (isInitMessage(msg)) {
    // Tag the return value as the result of an initializer.
    const GRState *state = C.getState();
    SVal V = state->getSVal(msg.getOriginExpr());
    addSelfFlag(V, SelfFlag_InitRes, C);
    return;
  }

  // We don't check for an invalid 'self' in an obj-c message expression to cut
  // down false positives where logging functions get information from self
  // (like its class) or doing "invalidation" on self when the initialization
  // fails.
}

void ObjCSelfInitChecker::PostVisitObjCIvarRefExpr(CheckerContext &C,
                                                   const ObjCIvarRefExpr *E) {
  // FIXME: A callback should disable checkers at the start of functions.
  if (!shouldRunOnFunctionOrMethod(dyn_cast<NamedDecl>(
                                     C.getCurrentAnalysisContext()->getDecl())))
    return;

  checkForInvalidSelf(E->getBase(), C,
         "Using an ivar before setting 'self' to the result of an initializer");
}

void ObjCSelfInitChecker::PreVisitReturnStmt(CheckerContext &C,
                                             const ReturnStmt *S) {
  // FIXME: A callback should disable checkers at the start of functions.
  if (!shouldRunOnFunctionOrMethod(dyn_cast<NamedDecl>(
                                     C.getCurrentAnalysisContext()->getDecl())))
    return;

  checkForInvalidSelf(S->getRetValue(), C,
          "Returning 'self' before setting it to the result of an initializer");
}

// When a call receives a reference to 'self', [Pre/Post]VisitGenericCall pass
// the SelfFlags from the object 'self' point to before the call, to the new
// object after the call.

void ObjCSelfInitChecker::PreVisitGenericCall(CheckerContext &C,
                                              const CallExpr *CE) {
  const GRState *state = C.getState();
  for (CallExpr::const_arg_iterator
         I = CE->arg_begin(), E = CE->arg_end(); I != E; ++I) {
    SVal argV = state->getSVal(*I);
    if (isSelfVar(argV, C)) {
      preCallSelfFlags = getSelfFlags(state->getSVal(cast<Loc>(argV)), C);
      return;
    }
  }
}

void ObjCSelfInitChecker::PostVisitGenericCall(CheckerContext &C,
                                               const CallExpr *CE) {
  const GRState *state = C.getState();
  for (CallExpr::const_arg_iterator
         I = CE->arg_begin(), E = CE->arg_end(); I != E; ++I) {
    SVal argV = state->getSVal(*I);
    if (isSelfVar(argV, C)) {
      addSelfFlag(state->getSVal(cast<Loc>(argV)), preCallSelfFlags, C);
      return;
    }
  }
}

void ObjCSelfInitChecker::visitLocation(CheckerContext &C, const Stmt *S,
                                        SVal location, bool isLoad) {
  // Tag the result of a load from 'self' so that we can easily know that the
  // value is the object that 'self' points to.
  const GRState *state = C.getState();
  if (isSelfVar(location, C))
    addSelfFlag(state->getSVal(cast<Loc>(location)), SelfFlag_Self, C);
}

// FIXME: A callback should disable checkers at the start of functions.
static bool shouldRunOnFunctionOrMethod(const NamedDecl *ND) {
  if (!ND)
    return false;

  const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(ND);
  if (!MD)
    return false;
  if (!isInitializationMethod(MD))
    return false;

  // self = [super init] applies only to NSObject subclasses.
  // For instance, NSProxy doesn't implement -init.
  ASTContext& Ctx = MD->getASTContext();
  IdentifierInfo* NSObjectII = &Ctx.Idents.get("NSObject");
  ObjCInterfaceDecl* ID = MD->getClassInterface()->getSuperClass();
  for ( ; ID ; ID = ID->getSuperClass()) {
    IdentifierInfo *II = ID->getIdentifier();

    if (II == NSObjectII)
      break;
  }
  if (!ID)
    return false;

  return true;
}

/// \brief Returns true if the location is 'self'.
static bool isSelfVar(SVal location, CheckerContext &C) {
  AnalysisContext *analCtx = C.getCurrentAnalysisContext(); 
  if (!analCtx->getSelfDecl())
    return false;
  if (!isa<loc::MemRegionVal>(location))
    return false;

  loc::MemRegionVal MRV = cast<loc::MemRegionVal>(location);
  if (const DeclRegion *DR = dyn_cast<DeclRegion>(MRV.getRegion()))
    return (DR->getDecl() == analCtx->getSelfDecl());

  return false;
}

static bool isInitializationMethod(const ObjCMethodDecl *MD) {
  // Init methods with prefix like '-(id)_init' are private and the requirements
  // are less strict so we don't check those.
  return MD->isInstanceMethod() &&
      cocoa::deriveNamingConvention(MD->getSelector(),
                                    /*ignorePrefix=*/false) == cocoa::InitRule;
}

static bool isInitMessage(const ObjCMessage &msg) {
  return cocoa::deriveNamingConvention(msg.getSelector()) == cocoa::InitRule;
}
