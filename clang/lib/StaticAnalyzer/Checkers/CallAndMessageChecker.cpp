//===--- CallAndMessageChecker.cpp ------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines CallAndMessageChecker, a builtin checker that checks for various
// errors of call and objc message expressions.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace ento;

namespace {
class CallAndMessageChecker
  : public Checker< check::PreStmt<CallExpr>, check::PreObjCMessage > {
  mutable OwningPtr<BugType> BT_call_null;
  mutable OwningPtr<BugType> BT_call_undef;
  mutable OwningPtr<BugType> BT_call_arg;
  mutable OwningPtr<BugType> BT_msg_undef;
  mutable OwningPtr<BugType> BT_objc_prop_undef;
  mutable OwningPtr<BugType> BT_msg_arg;
  mutable OwningPtr<BugType> BT_msg_ret;
public:

  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPreObjCMessage(ObjCMessage msg, CheckerContext &C) const;

private:
  static void PreVisitProcessArgs(CheckerContext &C,CallOrObjCMessage callOrMsg,
                             const char *BT_desc, OwningPtr<BugType> &BT);
  static bool PreVisitProcessArg(CheckerContext &C, SVal V,SourceRange argRange,
                                 const Expr *argEx,
                                 const bool checkUninitFields,
                                 const char *BT_desc,
                                 OwningPtr<BugType> &BT);

  static void EmitBadCall(BugType *BT, CheckerContext &C, const CallExpr *CE);
  void emitNilReceiverBug(CheckerContext &C, const ObjCMessage &msg,
                          ExplodedNode *N) const;

  void HandleNilReceiver(CheckerContext &C,
                         ProgramStateRef state,
                         ObjCMessage msg) const;

  static void LazyInit_BT(const char *desc, OwningPtr<BugType> &BT) {
    if (!BT)
      BT.reset(new BuiltinBug(desc));
  }
};
} // end anonymous namespace

void CallAndMessageChecker::EmitBadCall(BugType *BT, CheckerContext &C,
                                        const CallExpr *CE) {
  ExplodedNode *N = C.generateSink();
  if (!N)
    return;

  BugReport *R = new BugReport(*BT, BT->getName(), N);
  R->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N,
                               bugreporter::GetCalleeExpr(N), R));
  C.EmitReport(R);
}

void CallAndMessageChecker::PreVisitProcessArgs(CheckerContext &C,
                                                CallOrObjCMessage callOrMsg,
                                                const char *BT_desc,
                                                OwningPtr<BugType> &BT) {
  // Don't check for uninitialized field values in arguments if the
  // caller has a body that is available and we have the chance to inline it.
  // This is a hack, but is a reasonable compromise betweens sometimes warning
  // and sometimes not depending on if we decide to inline a function.
  const Decl *D = callOrMsg.getDecl();
  const bool checkUninitFields =
    !(C.getAnalysisManager().shouldInlineCall() &&
      (D && D->getBody()));
  
  for (unsigned i = 0, e = callOrMsg.getNumArgs(); i != e; ++i)
    if (PreVisitProcessArg(C, callOrMsg.getArgSVal(i),
                           callOrMsg.getArgSourceRange(i), callOrMsg.getArg(i),
                           checkUninitFields,
                           BT_desc, BT))
      return;
}

bool CallAndMessageChecker::PreVisitProcessArg(CheckerContext &C,
                                               SVal V, SourceRange argRange,
                                               const Expr *argEx,
                                               const bool checkUninitFields,
                                               const char *BT_desc,
                                               OwningPtr<BugType> &BT) {
  if (V.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      LazyInit_BT(BT_desc, BT);

      // Generate a report for this bug.
      BugReport *R = new BugReport(*BT, BT->getName(), N);
      R->addRange(argRange);
      if (argEx)
        R->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N, argEx,
                                                                   R));
      C.EmitReport(R);
    }
    return true;
  }

  if (!checkUninitFields)
    return false;
  
  if (const nonloc::LazyCompoundVal *LV =
        dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    class FindUninitializedField {
    public:
      SmallVector<const FieldDecl *, 10> FieldChain;
    private:
      ASTContext &C;
      StoreManager &StoreMgr;
      MemRegionManager &MrMgr;
      Store store;
    public:
      FindUninitializedField(ASTContext &c, StoreManager &storeMgr,
                             MemRegionManager &mrMgr, Store s)
      : C(c), StoreMgr(storeMgr), MrMgr(mrMgr), store(s) {}

      bool Find(const TypedValueRegion *R) {
        QualType T = R->getValueType();
        if (const RecordType *RT = T->getAsStructureType()) {
          const RecordDecl *RD = RT->getDecl()->getDefinition();
          assert(RD && "Referred record has no definition");
          for (RecordDecl::field_iterator I =
               RD->field_begin(), E = RD->field_end(); I!=E; ++I) {
            const FieldRegion *FR = MrMgr.getFieldRegion(&*I, R);
            FieldChain.push_back(&*I);
            T = I->getType();
            if (T->getAsStructureType()) {
              if (Find(FR))
                return true;
            }
            else {
              const SVal &V = StoreMgr.getBinding(store, loc::MemRegionVal(FR));
              if (V.isUndef())
                return true;
            }
            FieldChain.pop_back();
          }
        }

        return false;
      }
    };

    const LazyCompoundValData *D = LV->getCVData();
    FindUninitializedField F(C.getASTContext(),
                             C.getState()->getStateManager().getStoreManager(),
                             C.getSValBuilder().getRegionManager(),
                             D->getStore());

    if (F.Find(D->getRegion())) {
      if (ExplodedNode *N = C.generateSink()) {
        LazyInit_BT(BT_desc, BT);
        SmallString<512> Str;
        llvm::raw_svector_ostream os(Str);
        os << "Passed-by-value struct argument contains uninitialized data";

        if (F.FieldChain.size() == 1)
          os << " (e.g., field: '" << *F.FieldChain[0] << "')";
        else {
          os << " (e.g., via the field chain: '";
          bool first = true;
          for (SmallVectorImpl<const FieldDecl *>::iterator
               DI = F.FieldChain.begin(), DE = F.FieldChain.end(); DI!=DE;++DI){
            if (first)
              first = false;
            else
              os << '.';
            os << **DI;
          }
          os << "')";
        }

        // Generate a report for this bug.
        BugReport *R = new BugReport(*BT, os.str(), N);
        R->addRange(argRange);

        // FIXME: enhance track back for uninitialized value for arbitrary
        // memregions
        C.EmitReport(R);
      }
      return true;
    }
  }

  return false;
}

void CallAndMessageChecker::checkPreStmt(const CallExpr *CE,
                                         CheckerContext &C) const{

  const Expr *Callee = CE->getCallee()->IgnoreParens();
  const LocationContext *LCtx = C.getLocationContext();
  SVal L = C.getState()->getSVal(Callee, LCtx);

  if (L.isUndef()) {
    if (!BT_call_undef)
      BT_call_undef.reset(new BuiltinBug("Called function pointer is an "
                                         "uninitalized pointer value"));
    EmitBadCall(BT_call_undef.get(), C, CE);
    return;
  }

  if (isa<loc::ConcreteInt>(L)) {
    if (!BT_call_null)
      BT_call_null.reset(
        new BuiltinBug("Called function pointer is null (null dereference)"));
    EmitBadCall(BT_call_null.get(), C, CE);
  }

  PreVisitProcessArgs(C, CallOrObjCMessage(CE, C.getState(), LCtx),
                      "Function call argument is an uninitialized value",
                      BT_call_arg);
}

void CallAndMessageChecker::checkPreObjCMessage(ObjCMessage msg,
                                                CheckerContext &C) const {

  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();

  // FIXME: Handle 'super'?
  if (const Expr *receiver = msg.getInstanceReceiver()) {
    SVal recVal = state->getSVal(receiver, LCtx);
    if (recVal.isUndef()) {
      if (ExplodedNode *N = C.generateSink()) {
        BugType *BT = 0;
        if (msg.isPureMessageExpr()) {
          if (!BT_msg_undef)
            BT_msg_undef.reset(new BuiltinBug("Receiver in message expression "
                                              "is an uninitialized value"));
          BT = BT_msg_undef.get();
        }
        else {
          if (!BT_objc_prop_undef)
            BT_objc_prop_undef.reset(new BuiltinBug("Property access on an "
                                              "uninitialized object pointer"));
          BT = BT_objc_prop_undef.get();
        }
        BugReport *R =
          new BugReport(*BT, BT->getName(), N);
        R->addRange(receiver->getSourceRange());
        R->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N,
                                                                   receiver,
                                                                   R));
        C.EmitReport(R);
      }
      return;
    } else {
      // Bifurcate the state into nil and non-nil ones.
      DefinedOrUnknownSVal receiverVal = cast<DefinedOrUnknownSVal>(recVal);
  
      ProgramStateRef notNilState, nilState;
      llvm::tie(notNilState, nilState) = state->assume(receiverVal);
  
      // Handle receiver must be nil.
      if (nilState && !notNilState) {
        HandleNilReceiver(C, state, msg);
        return;
      }
    }
  }

  const char *bugDesc = msg.isPropertySetter() ?
                     "Argument for property setter is an uninitialized value"
                   : "Argument in message expression is an uninitialized value";
  // Check for any arguments that are uninitialized/undefined.
  PreVisitProcessArgs(C, CallOrObjCMessage(msg, state, LCtx),
                      bugDesc, BT_msg_arg);
}

void CallAndMessageChecker::emitNilReceiverBug(CheckerContext &C,
                                               const ObjCMessage &msg,
                                               ExplodedNode *N) const {

  if (!BT_msg_ret)
    BT_msg_ret.reset(
      new BuiltinBug("Receiver in message expression is "
                     "'nil' and returns a garbage value"));

  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);
  os << "The receiver of message '" << msg.getSelector().getAsString()
     << "' is nil and returns a value of type '"
     << msg.getType(C.getASTContext()).getAsString() << "' that will be garbage";

  BugReport *report = new BugReport(*BT_msg_ret, os.str(), N);
  if (const Expr *receiver = msg.getInstanceReceiver()) {
    report->addRange(receiver->getSourceRange());
    report->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N,
                                                                    receiver,
                                                                    report));
  }
  C.EmitReport(report);
}

static bool supportsNilWithFloatRet(const llvm::Triple &triple) {
  return (triple.getVendor() == llvm::Triple::Apple &&
          (triple.getOS() == llvm::Triple::IOS ||
           !triple.isMacOSXVersionLT(10,5)));
}

void CallAndMessageChecker::HandleNilReceiver(CheckerContext &C,
                                              ProgramStateRef state,
                                              ObjCMessage msg) const {
  ASTContext &Ctx = C.getASTContext();

  // Check the return type of the message expression.  A message to nil will
  // return different values depending on the return type and the architecture.
  QualType RetTy = msg.getType(Ctx);
  CanQualType CanRetTy = Ctx.getCanonicalType(RetTy);
  const LocationContext *LCtx = C.getLocationContext();

  if (CanRetTy->isStructureOrClassType()) {
    // Structure returns are safe since the compiler zeroes them out.
    SVal V = C.getSValBuilder().makeZeroVal(msg.getType(Ctx));
    C.addTransition(state->BindExpr(msg.getMessageExpr(), LCtx, V));
    return;
  }

  // Other cases: check if sizeof(return type) > sizeof(void*)
  if (CanRetTy != Ctx.VoidTy && C.getLocationContext()->getParentMap()
                                  .isConsumedExpr(msg.getMessageExpr())) {
    // Compute: sizeof(void *) and sizeof(return type)
    const uint64_t voidPtrSize = Ctx.getTypeSize(Ctx.VoidPtrTy);
    const uint64_t returnTypeSize = Ctx.getTypeSize(CanRetTy);

    if (voidPtrSize < returnTypeSize &&
        !(supportsNilWithFloatRet(Ctx.getTargetInfo().getTriple()) &&
          (Ctx.FloatTy == CanRetTy ||
           Ctx.DoubleTy == CanRetTy ||
           Ctx.LongDoubleTy == CanRetTy ||
           Ctx.LongLongTy == CanRetTy ||
           Ctx.UnsignedLongLongTy == CanRetTy))) {
      if (ExplodedNode *N = C.generateSink(state))
        emitNilReceiverBug(C, msg, N);
      return;
    }

    // Handle the safe cases where the return value is 0 if the
    // receiver is nil.
    //
    // FIXME: For now take the conservative approach that we only
    // return null values if we *know* that the receiver is nil.
    // This is because we can have surprises like:
    //
    //   ... = [[NSScreens screens] objectAtIndex:0];
    //
    // What can happen is that [... screens] could return nil, but
    // it most likely isn't nil.  We should assume the semantics
    // of this case unless we have *a lot* more knowledge.
    //
    SVal V = C.getSValBuilder().makeZeroVal(msg.getType(Ctx));
    C.addTransition(state->BindExpr(msg.getMessageExpr(), LCtx, V));
    return;
  }

  C.addTransition(state);
}

void ento::registerCallAndMessageChecker(CheckerManager &mgr) {
  mgr.registerChecker<CallAndMessageChecker>();
}
