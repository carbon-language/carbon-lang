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
// To perform the required checking, values are tagged with flags that indicate
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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/AST/ParentMap.h"

using namespace clang;
using namespace ento;

static bool shouldRunOnFunctionOrMethod(const NamedDecl *ND);
static bool isInitializationMethod(const ObjCMethodDecl *MD);
static bool isInitMessage(const ObjCMessage &msg);
static bool isSelfVar(SVal location, CheckerContext &C);

namespace {
class ObjCSelfInitChecker : public Checker<
                                             check::PostObjCMessage,
                                             check::PostStmt<ObjCIvarRefExpr>,
                                             check::PreStmt<ReturnStmt>,
                                             check::PreStmt<CallExpr>,
                                             check::PostStmt<CallExpr>,
                                             check::Location > {
public:
  void checkPostObjCMessage(ObjCMessage msg, CheckerContext &C) const;
  void checkPostStmt(const ObjCIvarRefExpr *E, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkLocation(SVal location, bool isLoad, CheckerContext &C) const;
};
} // end anonymous namespace

namespace {

class InitSelfBug : public BugType {
  const std::string desc;
public:
  InitSelfBug() : BugType("missing \"self = [(super or self) init...]\"",
                          "missing \"self = [(super or self) init...]\"") {}
};

} // end anonymous namespace

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

typedef llvm::ImmutableMap<SymbolRef, unsigned> SelfFlag;
namespace { struct CalledInit {}; }
namespace { struct PreCallSelfFlags {}; }

namespace clang {
namespace ento {
  template<>
  struct GRStateTrait<SelfFlag> : public GRStatePartialTrait<SelfFlag> {
    static void *GDMIndex() { static int index = 0; return &index; }
  };
  template <>
  struct GRStateTrait<CalledInit> : public GRStatePartialTrait<bool> {
    static void *GDMIndex() { static int index = 0; return &index; }
  };

  /// \brief A call receiving a reference to 'self' invalidates the object that
  /// 'self' contains. This keeps the "self flags" assigned to the 'self'
  /// object before the call so we can assign them to the new object that 'self'
  /// points to after the call.
  template <>
  struct GRStateTrait<PreCallSelfFlags> : public GRStatePartialTrait<unsigned> {
    static void *GDMIndex() { static int index = 0; return &index; }
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

static void addSelfFlag(const GRState *state, SVal val,
                        SelfFlagEnum flag, CheckerContext &C) {
  // We tag the symbol that the SVal wraps.
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
  
  if (!C.getState()->get<CalledInit>())
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

void ObjCSelfInitChecker::checkPostObjCMessage(ObjCMessage msg,
                                               CheckerContext &C) const {
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
    
    // FIXME this really should be context sensitive, where we record
    // the current stack frame (for IPA).  Also, we need to clean this
    // value out when we return from this method.
    state = state->set<CalledInit>(true);
    
    SVal V = state->getSVal(msg.getOriginExpr());
    addSelfFlag(state, V, SelfFlag_InitRes, C);
    return;
  }

  // We don't check for an invalid 'self' in an obj-c message expression to cut
  // down false positives where logging functions get information from self
  // (like its class) or doing "invalidation" on self when the initialization
  // fails.
}

void ObjCSelfInitChecker::checkPostStmt(const ObjCIvarRefExpr *E,
                                        CheckerContext &C) const {
  // FIXME: A callback should disable checkers at the start of functions.
  if (!shouldRunOnFunctionOrMethod(dyn_cast<NamedDecl>(
                                     C.getCurrentAnalysisContext()->getDecl())))
    return;

  checkForInvalidSelf(E->getBase(), C,
    "Instance variable used while 'self' is not set to the result of "
                                                 "'[(super or self) init...]'");
}

void ObjCSelfInitChecker::checkPreStmt(const ReturnStmt *S,
                                       CheckerContext &C) const {
  // FIXME: A callback should disable checkers at the start of functions.
  if (!shouldRunOnFunctionOrMethod(dyn_cast<NamedDecl>(
                                     C.getCurrentAnalysisContext()->getDecl())))
    return;

  checkForInvalidSelf(S->getRetValue(), C,
    "Returning 'self' while it is not set to the result of "
                                                 "'[(super or self) init...]'");
}

// When a call receives a reference to 'self', [Pre/Post]VisitGenericCall pass
// the SelfFlags from the object 'self' point to before the call, to the new
// object after the call. This is to avoid invalidation of 'self' by logging
// functions.
// Another common pattern in classes with multiple initializers is to put the
// subclass's common initialization bits into a static function that receives
// the value of 'self', e.g:
// @code
//   if (!(self = [super init]))
//     return nil;
//   if (!(self = _commonInit(self)))
//     return nil;
// @endcode
// Until we can use inter-procedural analysis, in such a call, transfer the
// SelfFlags to the result of the call.

void ObjCSelfInitChecker::checkPreStmt(const CallExpr *CE,
                                       CheckerContext &C) const {
  const GRState *state = C.getState();
  for (CallExpr::const_arg_iterator
         I = CE->arg_begin(), E = CE->arg_end(); I != E; ++I) {
    SVal argV = state->getSVal(*I);
    if (isSelfVar(argV, C)) {
      unsigned selfFlags = getSelfFlags(state->getSVal(cast<Loc>(argV)), C);
      C.addTransition(state->set<PreCallSelfFlags>(selfFlags));
      return;
    } else if (hasSelfFlag(argV, SelfFlag_Self, C)) {
      unsigned selfFlags = getSelfFlags(argV, C);
      C.addTransition(state->set<PreCallSelfFlags>(selfFlags));
      return;
    }
  }
}

void ObjCSelfInitChecker::checkPostStmt(const CallExpr *CE,
                                        CheckerContext &C) const {
  const GRState *state = C.getState();
  for (CallExpr::const_arg_iterator
         I = CE->arg_begin(), E = CE->arg_end(); I != E; ++I) {
    SVal argV = state->getSVal(*I);
    if (isSelfVar(argV, C)) {
      SelfFlagEnum prevFlags = (SelfFlagEnum)state->get<PreCallSelfFlags>();
      state = state->remove<PreCallSelfFlags>();
      addSelfFlag(state, state->getSVal(cast<Loc>(argV)), prevFlags, C);
      return;
    } else if (hasSelfFlag(argV, SelfFlag_Self, C)) {
      SelfFlagEnum prevFlags = (SelfFlagEnum)state->get<PreCallSelfFlags>();
      state = state->remove<PreCallSelfFlags>();
      addSelfFlag(state, state->getSVal(CE), prevFlags, C);
      return;
    }
  }
}

void ObjCSelfInitChecker::checkLocation(SVal location, bool isLoad,
                                        CheckerContext &C) const {
  // Tag the result of a load from 'self' so that we can easily know that the
  // value is the object that 'self' points to.
  const GRState *state = C.getState();
  if (isSelfVar(location, C))
    addSelfFlag(state, state->getSVal(cast<Loc>(location)), SelfFlag_Self, C);
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
  ASTContext &Ctx = MD->getASTContext();
  IdentifierInfo* NSObjectII = &Ctx.Idents.get("NSObject");
  ObjCInterfaceDecl *ID = MD->getClassInterface()->getSuperClass();
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
  return MD->getMethodFamily() == OMF_init;
}

static bool isInitMessage(const ObjCMessage &msg) {
  return msg.getMethodFamily() == OMF_init;
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void ento::registerObjCSelfInitChecker(CheckerManager &mgr) {
  mgr.registerChecker<ObjCSelfInitChecker>();
}
