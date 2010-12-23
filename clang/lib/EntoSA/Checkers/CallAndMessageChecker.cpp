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

#include "ExprEngineInternalChecks.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/EntoSA/BugReporter/BugType.h"
#include "clang/EntoSA/PathSensitive/CheckerVisitor.h"

using namespace clang;
using namespace ento;

namespace {
class CallAndMessageChecker
  : public CheckerVisitor<CallAndMessageChecker> {
  BugType *BT_call_null;
  BugType *BT_call_undef;
  BugType *BT_call_arg;
  BugType *BT_msg_undef;
  BugType *BT_msg_arg;
  BugType *BT_msg_ret;
public:
  CallAndMessageChecker() :
    BT_call_null(0), BT_call_undef(0), BT_call_arg(0),
    BT_msg_undef(0), BT_msg_arg(0), BT_msg_ret(0) {}

  static void *getTag() {
    static int x = 0;
    return &x;
  }

  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
  void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);
  bool evalNilReceiver(CheckerContext &C, const ObjCMessageExpr *ME);

private:
  bool PreVisitProcessArg(CheckerContext &C, const Expr *Ex,
                          const char *BT_desc, BugType *&BT);

  void EmitBadCall(BugType *BT, CheckerContext &C, const CallExpr *CE);
  void emitNilReceiverBug(CheckerContext &C, const ObjCMessageExpr *ME,
                          ExplodedNode *N);

  void HandleNilReceiver(CheckerContext &C, const GRState *state,
                         const ObjCMessageExpr *ME);

  void LazyInit_BT(const char *desc, BugType *&BT) {
    if (!BT)
      BT = new BuiltinBug(desc);
  }
};
} // end anonymous namespace

void ento::RegisterCallAndMessageChecker(ExprEngine &Eng) {
  Eng.registerCheck(new CallAndMessageChecker());
}

void CallAndMessageChecker::EmitBadCall(BugType *BT, CheckerContext &C,
                                        const CallExpr *CE) {
  ExplodedNode *N = C.generateSink();
  if (!N)
    return;

  EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName(), N);
  R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                       bugreporter::GetCalleeExpr(N));
  C.EmitReport(R);
}

bool CallAndMessageChecker::PreVisitProcessArg(CheckerContext &C,
                                               const Expr *Ex,
                                               const char *BT_desc,
                                               BugType *&BT) {

  const SVal &V = C.getState()->getSVal(Ex);

  if (V.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      LazyInit_BT(BT_desc, BT);

      // Generate a report for this bug.
      EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName(), N);
      R->addRange(Ex->getSourceRange());
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Ex);
      C.EmitReport(R);
    }
    return true;
  }

  if (const nonloc::LazyCompoundVal *LV =
        dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    class FindUninitializedField {
    public:
      llvm::SmallVector<const FieldDecl *, 10> FieldChain;
    private:
      ASTContext &C;
      StoreManager &StoreMgr;
      MemRegionManager &MrMgr;
      Store store;
    public:
      FindUninitializedField(ASTContext &c, StoreManager &storeMgr,
                             MemRegionManager &mrMgr, Store s)
      : C(c), StoreMgr(storeMgr), MrMgr(mrMgr), store(s) {}

      bool Find(const TypedRegion *R) {
        QualType T = R->getValueType();
        if (const RecordType *RT = T->getAsStructureType()) {
          const RecordDecl *RD = RT->getDecl()->getDefinition();
          assert(RD && "Referred record has no definition");
          for (RecordDecl::field_iterator I =
               RD->field_begin(), E = RD->field_end(); I!=E; ++I) {
            const FieldRegion *FR = MrMgr.getFieldRegion(*I, R);
            FieldChain.push_back(*I);
            T = (*I)->getType();
            if (T->getAsStructureType()) {
              if (Find(FR))
                return true;
            }
            else {
              const SVal &V = StoreMgr.Retrieve(store, loc::MemRegionVal(FR));
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
        llvm::SmallString<512> Str;
        llvm::raw_svector_ostream os(Str);
        os << "Passed-by-value struct argument contains uninitialized data";

        if (F.FieldChain.size() == 1)
          os << " (e.g., field: '" << F.FieldChain[0] << "')";
        else {
          os << " (e.g., via the field chain: '";
          bool first = true;
          for (llvm::SmallVectorImpl<const FieldDecl *>::iterator
               DI = F.FieldChain.begin(), DE = F.FieldChain.end(); DI!=DE;++DI){
            if (first)
              first = false;
            else
              os << '.';
            os << *DI;
          }
          os << "')";
        }

        // Generate a report for this bug.
        EnhancedBugReport *R = new EnhancedBugReport(*BT, os.str(), N);
        R->addRange(Ex->getSourceRange());

        // FIXME: enhance track back for uninitialized value for arbitrary
        // memregions
        C.EmitReport(R);
      }
      return true;
    }
  }

  return false;
}

void CallAndMessageChecker::PreVisitCallExpr(CheckerContext &C,
                                             const CallExpr *CE){

  const Expr *Callee = CE->getCallee()->IgnoreParens();
  SVal L = C.getState()->getSVal(Callee);

  if (L.isUndef()) {
    if (!BT_call_undef)
      BT_call_undef =
        new BuiltinBug("Called function pointer is an uninitalized pointer value");
    EmitBadCall(BT_call_undef, C, CE);
    return;
  }

  if (isa<loc::ConcreteInt>(L)) {
    if (!BT_call_null)
      BT_call_null =
        new BuiltinBug("Called function pointer is null (null dereference)");
    EmitBadCall(BT_call_null, C, CE);
  }

  for (CallExpr::const_arg_iterator I = CE->arg_begin(), E = CE->arg_end();
       I != E; ++I)
    if (PreVisitProcessArg(C, *I,
                           "Function call argument is an uninitialized value",
                           BT_call_arg))
      return;
}

void CallAndMessageChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                                    const ObjCMessageExpr *ME) {

  const GRState *state = C.getState();

  // FIXME: Handle 'super'?
  if (const Expr *receiver = ME->getInstanceReceiver())
    if (state->getSVal(receiver).isUndef()) {
      if (ExplodedNode *N = C.generateSink()) {
        if (!BT_msg_undef)
          BT_msg_undef =
            new BuiltinBug("Receiver in message expression is an uninitialized value");
        EnhancedBugReport *R =
          new EnhancedBugReport(*BT_msg_undef, BT_msg_undef->getName(), N);
        R->addRange(receiver->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                             receiver);
        C.EmitReport(R);
      }
      return;
    }

  // Check for any arguments that are uninitialized/undefined.
  for (ObjCMessageExpr::const_arg_iterator I = ME->arg_begin(),
         E = ME->arg_end(); I != E; ++I)
    if (PreVisitProcessArg(C, *I,
                           "Argument in message expression "
                           "is an uninitialized value", BT_msg_arg))
        return;
}

bool CallAndMessageChecker::evalNilReceiver(CheckerContext &C,
                                            const ObjCMessageExpr *ME) {
  HandleNilReceiver(C, C.getState(), ME);
  return true; // Nil receiver is not handled elsewhere.
}

void CallAndMessageChecker::emitNilReceiverBug(CheckerContext &C,
                                               const ObjCMessageExpr *ME,
                                               ExplodedNode *N) {

  if (!BT_msg_ret)
    BT_msg_ret =
      new BuiltinBug("Receiver in message expression is "
                     "'nil' and returns a garbage value");

  llvm::SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);
  os << "The receiver of message '" << ME->getSelector().getAsString()
     << "' is nil and returns a value of type '"
     << ME->getType().getAsString() << "' that will be garbage";

  EnhancedBugReport *report = new EnhancedBugReport(*BT_msg_ret, os.str(), N);
  if (const Expr *receiver = ME->getInstanceReceiver()) {
    report->addRange(receiver->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                              receiver);
  }
  C.EmitReport(report);
}

static bool supportsNilWithFloatRet(const llvm::Triple &triple) {
  return triple.getVendor() == llvm::Triple::Apple &&
         (triple.getDarwinMajorNumber() >= 9 || 
          triple.getArch() == llvm::Triple::arm || 
          triple.getArch() == llvm::Triple::thumb);
}

void CallAndMessageChecker::HandleNilReceiver(CheckerContext &C,
                                              const GRState *state,
                                              const ObjCMessageExpr *ME) {

  // Check the return type of the message expression.  A message to nil will
  // return different values depending on the return type and the architecture.
  QualType RetTy = ME->getType();

  ASTContext &Ctx = C.getASTContext();
  CanQualType CanRetTy = Ctx.getCanonicalType(RetTy);

  if (CanRetTy->isStructureOrClassType()) {
    // FIXME: At some point we shouldn't rely on isConsumedExpr(), but instead
    // have the "use of undefined value" be smarter about where the
    // undefined value came from.
    if (C.getPredecessor()->getParentMap().isConsumedExpr(ME)) {
      if (ExplodedNode* N = C.generateSink(state))
        emitNilReceiverBug(C, ME, N);
      return;
    }

    // The result is not consumed by a surrounding expression.  Just propagate
    // the current state.
    C.addTransition(state);
    return;
  }

  // Other cases: check if the return type is smaller than void*.
  if (CanRetTy != Ctx.VoidTy &&
      C.getPredecessor()->getParentMap().isConsumedExpr(ME)) {
    // Compute: sizeof(void *) and sizeof(return type)
    const uint64_t voidPtrSize = Ctx.getTypeSize(Ctx.VoidPtrTy);
    const uint64_t returnTypeSize = Ctx.getTypeSize(CanRetTy);

    if (voidPtrSize < returnTypeSize &&
        !(supportsNilWithFloatRet(Ctx.Target.getTriple()) &&
          (Ctx.FloatTy == CanRetTy ||
           Ctx.DoubleTy == CanRetTy ||
           Ctx.LongDoubleTy == CanRetTy ||
           Ctx.LongLongTy == CanRetTy ||
           Ctx.UnsignedLongLongTy == CanRetTy))) {
      if (ExplodedNode* N = C.generateSink(state))
        emitNilReceiverBug(C, ME, N);
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
    SVal V = C.getSValBuilder().makeZeroVal(ME->getType());
    C.generateNode(state->BindExpr(ME, V));
    return;
  }

  C.addTransition(state);
}
