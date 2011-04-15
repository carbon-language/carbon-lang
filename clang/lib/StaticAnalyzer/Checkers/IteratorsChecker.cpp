//=== IteratorsChecker.cpp - Check for Invalidated Iterators ------*- C++ -*----
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines IteratorsChecker, a number of small checks for conditions
// leading to invalid iterators being used.
// FIXME: Currently only supports 'vector' and 'deque'
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"


using namespace clang;
using namespace ento;

// This is the state associated with each iterator which includes both the
// kind of state and the instance used to initialize it.
// FIXME: add location where invalidated for better error reporting.
namespace {
class RefState {
  enum Kind { BeginValid, EndValid, Invalid, Undefined, Unknown } K;
  const void *VR;

public:
  RefState(Kind k, const void *vr) : K(k), VR(vr) {}

  bool isValid() const { return K == BeginValid || K == EndValid; }
  bool isInvalid() const { return K == Invalid; }
  bool isUndefined() const { return K == Undefined; }
  bool isUnknown() const { return K == Unknown; }
  const MemRegion *getMemRegion() const {
    if (K == BeginValid || K == EndValid)
      return(const MemRegion *)VR;
    return 0;
  }
  const MemberExpr *getMemberExpr() const {
    if (K == Invalid)
      return(const MemberExpr *)VR;
    return 0;
  }

  bool operator==(const RefState &X) const {
    return K == X.K && VR == X.VR;
  }

  static RefState getBeginValid(const MemRegion *vr) {
    assert(vr);
    return RefState(BeginValid, vr);
  }
  static RefState getEndValid(const MemRegion *vr) {
    assert(vr);
    return RefState(EndValid, vr);
  }
  static RefState getInvalid( const MemberExpr *ME ) {
    return RefState(Invalid, ME);
  }
  static RefState getUndefined( void ) {
    return RefState(Undefined, 0);
  }
  static RefState getUnknown( void ) {
    return RefState(Unknown, 0);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
    ID.AddPointer(VR);
  }
};

enum RefKind { NoKind, VectorKind, VectorIteratorKind };

class IteratorsChecker : 
    public Checker<check::PreStmt<CXXOperatorCallExpr>,
                   check::PreStmt<DeclStmt>,
                   check::PreStmt<CXXMemberCallExpr>,
                   check::PreStmt<CallExpr> >
  {
  // Used when parsing iterators and vectors and deques.
  BuiltinBug *BT_Invalid, *BT_Undefined, *BT_Incompatible;

public:
  IteratorsChecker() :
    BT_Invalid(0), BT_Undefined(0), BT_Incompatible(0)
  {}
  static void *getTag() { static int tag; return &tag; }
    
  // Checker entry points.
  void checkPreStmt(const CXXOperatorCallExpr *OCE,
                    CheckerContext &C) const;

  void checkPreStmt(const DeclStmt *DS,
                    CheckerContext &C) const;

  void checkPreStmt(const CXXMemberCallExpr *MCE,
                    CheckerContext &C) const;

  void checkPreStmt(const CallExpr *CE,
                    CheckerContext &C) const;

private:
  const GRState *handleAssign(const GRState *state, const Expr *lexp,
      const Expr *rexp, const LocationContext *LC) const;
  const GRState *handleAssign(const GRState *state, const MemRegion *MR,
      const Expr *rexp, const LocationContext *LC) const;
  const GRState *invalidateIterators(const GRState *state, const MemRegion *MR,
      const MemberExpr *ME) const;
  void checkExpr(CheckerContext &C, const Expr *E) const;
  void checkArgs(CheckerContext &C, const CallExpr *CE) const;
  const MemRegion *getRegion(const GRState *state, const Expr *E,
      const LocationContext *LC) const;
  const DeclRefExpr *getDeclRefExpr(const Expr *E) const;
};

class IteratorState {
public:
  typedef llvm::ImmutableMap<const MemRegion *, RefState> EntryMap;
};
} //end anonymous namespace

namespace clang {
  namespace ento {
    template <>
    struct GRStateTrait<IteratorState> 
      : public GRStatePartialTrait<IteratorState::EntryMap> {
      static void *GDMIndex() { return IteratorsChecker::getTag(); }
    };
  }
}

void ento::registerIteratorsChecker(CheckerManager &mgr) {
  mgr.registerChecker<IteratorsChecker>();
}

// ===============================================
// Utility functions used by visitor functions
// ===============================================

// check a templated type for std::vector or std::deque
static RefKind getTemplateKind(const NamedDecl *td) {
  const DeclContext *dc = td->getDeclContext();
  const NamespaceDecl *nameSpace = dyn_cast<NamespaceDecl>(dc);
  if (!nameSpace || !isa<TranslationUnitDecl>(nameSpace->getDeclContext())
      || nameSpace->getName() != "std")
    return NoKind;
  
  llvm::StringRef name = td->getName();
  return llvm::StringSwitch<RefKind>(name)
    .Cases("vector", "deque", VectorKind)
    .Default(NoKind);
}

static RefKind getTemplateKind(const DeclContext *dc) {
  if (const ClassTemplateSpecializationDecl *td =
      dyn_cast<ClassTemplateSpecializationDecl>(dc))
    return getTemplateKind(cast<NamedDecl>(td));
  return NoKind;
}

static RefKind getTemplateKind(const TypedefType *tdt) {
  const TypedefNameDecl *td = tdt->getDecl();
  RefKind parentKind = getTemplateKind(td->getDeclContext());
  if (parentKind == VectorKind) {
    return llvm::StringSwitch<RefKind>(td->getName())
    .Cases("iterator",
           "const_iterator",
           "reverse_iterator", VectorIteratorKind)
    .Default(NoKind);
  }
  return NoKind;
}

static RefKind getTemplateKind(const TemplateSpecializationType *tsp) {
  const TemplateName &tname = tsp->getTemplateName();
  TemplateDecl *td = tname.getAsTemplateDecl();
  if (!td)
    return NoKind;
  return getTemplateKind(td);
}

static RefKind getTemplateKind(QualType T) {
  if (const TemplateSpecializationType *tsp = 
      T->getAs<TemplateSpecializationType>()) {
    return getTemplateKind(tsp);      
  }
  if (const ElaboratedType *ET = dyn_cast<ElaboratedType>(T)) {
    QualType namedType = ET->getNamedType();
    if (const TypedefType *tdt = namedType->getAs<TypedefType>()) 
      return getTemplateKind(tdt);
    if (const TemplateSpecializationType *tsp = 
        namedType->getAs<TemplateSpecializationType>()) {
      return getTemplateKind(tsp);      
    }
  }
  return NoKind;  
}

// Iterate through our map and invalidate any iterators that were
// initialized fromt the specified instance MemRegion.
const GRState *IteratorsChecker::invalidateIterators(const GRState *state,
                          const MemRegion *MR, const MemberExpr *ME) const {
  IteratorState::EntryMap Map = state->get<IteratorState>();
  if (Map.isEmpty())
    return state;

  // Loop over the entries in the current state.
  // The key doesn't change, so the map iterators won't change.
  for (IteratorState::EntryMap::iterator I = Map.begin(), E = Map.end();
                                                            I != E; ++I) {
    RefState RS = I.getData();
    if (RS.getMemRegion() == MR)
      state = state->set<IteratorState>(I.getKey(), RefState::getInvalid(ME));
  }

  return state;
}

// Handle assigning to an iterator where we don't have the LValue MemRegion.
const GRState *IteratorsChecker::handleAssign(const GRState *state,
    const Expr *lexp, const Expr *rexp, const LocationContext *LC) const {
  // Skip the cast if present.
  if (isa<ImplicitCastExpr>(lexp))
    lexp = dyn_cast<ImplicitCastExpr>(lexp)->getSubExpr();
  SVal sv = state->getSVal(lexp);
  const MemRegion *MR = sv.getAsRegion();
  if (!MR)
    return state;
  RefKind kind = getTemplateKind(lexp->getType());

  // If assigning to a vector, invalidate any iterators currently associated.
  if (kind == VectorKind)
    return invalidateIterators(state, MR, 0);

  // Make sure that we are assigning to an iterator.
  if (getTemplateKind(lexp->getType()) != VectorIteratorKind)
    return state;
  return handleAssign(state, MR, rexp, LC);
}

// handle assigning to an iterator
const GRState *IteratorsChecker::handleAssign(const GRState *state,
    const MemRegion *MR, const Expr *rexp, const LocationContext *LC) const {
  // Assume unknown until we find something definite.
  state = state->set<IteratorState>(MR, RefState::getUnknown());
  if (isa<ImplicitCastExpr>(rexp))
    rexp = dyn_cast<ImplicitCastExpr>(rexp)->getSubExpr();
  // Need to handle three cases: MemberCall, copy, copy with addition.
  if (const CallExpr *CE = dyn_cast<CallExpr>(rexp)) {
    // Handle MemberCall.
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(CE->getCallee())) {
      const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ME->getBase());
      if (!DRE)
        return state;
      // Verify that the type is std::vector<T>.
      if (getTemplateKind(DRE->getType()) != VectorKind)
          return state;
      // Now get the MemRegion associated with the instance.
      const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
      if (!VD)
        return state;
      const MemRegion *IMR = state->getRegion(VD, LC);
      if (!IMR)
        return state;
      // Finally, see if it is one of the calls that will create
      // a valid iterator and mark it if so, else mark as Unknown.
      llvm::StringRef mName = ME->getMemberDecl()->getName();
      
      if (llvm::StringSwitch<bool>(mName)        
          .Cases("begin", "insert", "erase", true).Default(false)) {
        return state->set<IteratorState>(MR, RefState::getBeginValid(IMR));
      }
      if (mName == "end")
        return state->set<IteratorState>(MR, RefState::getEndValid(IMR));

      return state->set<IteratorState>(MR, RefState::getUnknown());
    }
  }
  // Handle straight copy from another iterator.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(rexp)) {
    if (getTemplateKind(DRE->getType()) != VectorIteratorKind)
      return state;
    // Now get the MemRegion associated with the instance.
    const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD)
      return state;
    const MemRegion *IMR = state->getRegion(VD, LC);
    if (!IMR)
      return state;
    // Get the RefState of the iterator being copied.
    const RefState *RS = state->get<IteratorState>(IMR);
    if (!RS)
      return state;
    // Use it to set the state of the LValue.
    return state->set<IteratorState>(MR, *RS);
  }
  // If we have operator+ or operator- ...
  if (const CXXOperatorCallExpr *OCE = dyn_cast<CXXOperatorCallExpr>(rexp)) {
    OverloadedOperatorKind Kind = OCE->getOperator();
    if (Kind == OO_Plus || Kind == OO_Minus) {
      // Check left side of tree for a valid value.
      state = handleAssign( state, MR, OCE->getArg(0), LC);
      const RefState *RS = state->get<IteratorState>(MR);
      // If found, return it.
      if (!RS->isUnknown())
        return state;
      // Otherwise return what we find in the right side.
      return handleAssign(state, MR, OCE->getArg(1), LC);
    }
  }
  // Fall through if nothing matched.
  return state;
}

// Iterate through the arguments looking for an Invalid or Undefined iterator.
void IteratorsChecker::checkArgs(CheckerContext &C, const CallExpr *CE) const {
  for (CallExpr::const_arg_iterator I = CE->arg_begin(), E = CE->arg_end();
       I != E; ++I) {
    checkExpr(C, *I);
  }
}

// Get the DeclRefExpr associated with the expression.
const DeclRefExpr *IteratorsChecker::getDeclRefExpr(const Expr *E) const {
  // If it is a CXXConstructExpr, need to get the subexpression.
  if (const CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(E)) {
    if (CE->getNumArgs()== 1) {
      CXXConstructorDecl *CD = CE->getConstructor();
      if (CD->isTrivial())
        E = CE->getArg(0);
    }
  }
  if (isa<ImplicitCastExpr>(E))
    E = dyn_cast<ImplicitCastExpr>(E)->getSubExpr();
  // If it isn't one of our types, don't do anything.
  if (getTemplateKind(E->getType()) != VectorIteratorKind)
    return NULL;
  return dyn_cast<DeclRefExpr>(E);
}

// Get the MemRegion associated with the expresssion.
const MemRegion *IteratorsChecker::getRegion(const GRState *state,
    const Expr *E, const LocationContext *LC) const {
  const DeclRefExpr *DRE = getDeclRefExpr(E);
  if (!DRE)
    return NULL;
  const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return NULL;
  // return the MemRegion associated with the iterator
  return state->getRegion(VD, LC);
}

// Check the expression and if it is an iterator, generate a diagnostic
// if the iterator is not valid.
// FIXME: this method can generate new nodes, and subsequent logic should
// use those nodes.  We also cannot create multiple nodes at one ProgramPoint
// with the same tag.
void IteratorsChecker::checkExpr(CheckerContext &C, const Expr *E) const {
  const GRState *state = C.getState();
  const MemRegion *MR = getRegion(state, E,
                   C.getPredecessor()->getLocationContext());
  if (!MR)
    return;

  // Get the state associated with the iterator.
  const RefState *RS = state->get<IteratorState>(MR);
  if (!RS)
    return;
  if (RS->isInvalid()) {
    if (ExplodedNode *N = C.generateNode()) {
      if (!BT_Invalid)
        // FIXME: We are eluding constness here.
        const_cast<IteratorsChecker*>(this)->BT_Invalid = new BuiltinBug("");

      std::string msg;
      const MemberExpr *ME = RS->getMemberExpr();
      if (ME) {
        std::string name = ME->getMemberNameInfo().getAsString();
        msg = "Attempt to use an iterator made invalid by call to '" +
                                                                  name + "'";
      }
      else {
        msg = "Attempt to use an iterator made invalid by copying another "
                    "container to its container";
      }

      EnhancedBugReport *R = new EnhancedBugReport(*BT_Invalid, msg, N);
      R->addRange(getDeclRefExpr(E)->getSourceRange());
      C.EmitReport(R);
    }
  }
  else if (RS->isUndefined()) {
    if (ExplodedNode *N = C.generateNode()) {
      if (!BT_Undefined)
        // FIXME: We are eluding constness here.
        const_cast<IteratorsChecker*>(this)->BT_Undefined =
          new BuiltinBug("Use of iterator that is not defined");

      EnhancedBugReport *R = new EnhancedBugReport(*BT_Undefined,
                                           BT_Undefined->getDescription(), N);
      R->addRange(getDeclRefExpr(E)->getSourceRange());
      C.EmitReport(R);
    }
  }
}

// ===============================================
// Path analysis visitor functions
// ===============================================

// For a generic Call, just check the args for bad iterators.
void IteratorsChecker::checkPreStmt(const CallExpr *CE,
                                    CheckerContext &C) const{
  
  // FIXME: These checks are to currently work around a bug
  // in CheckerManager.
  if (isa<CXXOperatorCallExpr>(CE))
    return;
  if (isa<CXXMemberCallExpr>(CE))
    return;

  checkArgs(C, CE);
}

// Handle operator calls. First, if it is operator=, check the argument,
// and handle assigning and set target state appropriately. Otherwise, for
// other operators, check the args for bad iterators and handle comparisons.
void IteratorsChecker::checkPreStmt(const CXXOperatorCallExpr *OCE,
                                    CheckerContext &C) const
{
  const LocationContext *LC = C.getPredecessor()->getLocationContext();
  const GRState *state = C.getState();
  OverloadedOperatorKind Kind = OCE->getOperator();
  if (Kind == OO_Equal) {
    checkExpr(C, OCE->getArg(1));
    state = handleAssign(state, OCE->getArg(0), OCE->getArg(1), LC);
    C.addTransition(state);
    return;
  }
  else {
    checkArgs(C, OCE);
    // If it is a compare and both are iterators, ensure that they are for
    // the same container.
    if (Kind == OO_EqualEqual || Kind == OO_ExclaimEqual ||
        Kind == OO_Less || Kind == OO_LessEqual ||
        Kind == OO_Greater || Kind == OO_GreaterEqual) {
      const MemRegion *MR0, *MR1;
      MR0 = getRegion(state, OCE->getArg(0), LC);
      if (!MR0)
        return;
      MR1 = getRegion(state, OCE->getArg(1), LC);
      if (!MR1)
        return;
      const RefState *RS0, *RS1;
      RS0 = state->get<IteratorState>(MR0);
      if (!RS0)
        return;
      RS1 = state->get<IteratorState>(MR1);
      if (!RS1)
        return;
      if (RS0->getMemRegion() != RS1->getMemRegion()) {
      if (ExplodedNode *N = C.generateNode()) {
          if (!BT_Incompatible)
            const_cast<IteratorsChecker*>(this)->BT_Incompatible =
              new BuiltinBug(
                      "Cannot compare iterators from different containers");

          EnhancedBugReport *R = new EnhancedBugReport(*BT_Incompatible,
                                         BT_Incompatible->getDescription(), N);
          R->addRange(OCE->getSourceRange());
          C.EmitReport(R);
        }
      }
    }
  }
}

// Need to handle DeclStmts to pick up initializing of iterators and to mark
// uninitialized ones as Undefined.
void IteratorsChecker::checkPreStmt(const DeclStmt *DS,
                                    CheckerContext &C) const {
  const Decl* D = *DS->decl_begin();
  const VarDecl* VD = dyn_cast<VarDecl>(D);
  // Only care about iterators.
  if (getTemplateKind(VD->getType()) != VectorIteratorKind)
    return;

  // Get the MemRegion associated with the iterator and mark it as Undefined.
  const GRState *state = C.getState();
  Loc VarLoc = state->getLValue(VD, C.getPredecessor()->getLocationContext());
  const MemRegion *MR = VarLoc.getAsRegion();
  if (!MR)
    return;
  state = state->set<IteratorState>(MR, RefState::getUndefined());

  // if there is an initializer, handle marking Valid if a proper initializer
  const Expr* InitEx = VD->getInit();
  if (InitEx) {
    // FIXME: This is too syntactic.  Since 'InitEx' will be analyzed first
    // it should resolve to an SVal that we can check for validity
    // *semantically* instead of walking through the AST.
    if (const CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(InitEx)) {
      if (CE->getNumArgs() == 1) {
        const Expr *E = CE->getArg(0);
        if (isa<ImplicitCastExpr>(E))
          InitEx = dyn_cast<ImplicitCastExpr>(E)->getSubExpr();
        state = handleAssign(state, MR, InitEx,
                                  C.getPredecessor()->getLocationContext());
      }
    }
  }
  C.addTransition(state);
}


namespace { struct CalledReserved {}; }
namespace clang { namespace ento {
template<> struct GRStateTrait<CalledReserved> 
    :  public GRStatePartialTrait<llvm::ImmutableSet<const MemRegion*> > {
  static void *GDMIndex() { static int index = 0; return &index; }
};
}}

// on a member call, first check the args for any bad iterators
// then, check to see if it is a call to a function that will invalidate
// the iterators
void IteratorsChecker::checkPreStmt(const CXXMemberCallExpr *MCE,
                                    CheckerContext &C) const {
  // Check the arguments.
  checkArgs(C, MCE);
  const MemberExpr *ME = dyn_cast<MemberExpr>(MCE->getCallee());
  if (!ME)
    return;
  // Make sure we have the right kind of container.
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ME->getBase());
  if (!DRE || getTemplateKind(DRE->getType()) != VectorKind)
    return;
  SVal tsv = C.getState()->getSVal(DRE);
  // Get the MemRegion associated with the container instance.
  const MemRegion *MR = tsv.getAsRegion();
  if (!MR)
    return;
  // If we are calling a function that invalidates iterators, mark them
  // appropriately by finding matching instances.
  const GRState *state = C.getState();
  llvm::StringRef mName = ME->getMemberDecl()->getName();
  if (llvm::StringSwitch<bool>(mName)
      .Cases("insert", "reserve", "push_back", true)
      .Cases("erase", "pop_back", "clear", "resize", true)
      .Default(false)) {
    // If there was a 'reserve' call, assume iterators are good.
    if (!state->contains<CalledReserved>(MR))
      state = invalidateIterators(state, MR, ME);
  }
  // Keep track of instances that have called 'reserve'
  // note: do this after we invalidate any iterators by calling 
  // 'reserve' itself.
  if (mName == "reserve")
    state = state->add<CalledReserved>(MR);
  
  if (state != C.getState())
    C.addTransition(state);
}

