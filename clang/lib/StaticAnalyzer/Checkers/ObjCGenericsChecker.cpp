//=== ObjCGenericsChecker.cpp - Path sensitive checker for Generics *- C++ -*=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker tries to find type errors that the compiler is not able to catch
// due to the implicit conversions that were introduced for backward
// compatibility.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

// ProgramState trait - a map from symbol to its specialized type.
REGISTER_MAP_WITH_PROGRAMSTATE(TypeParamMap, SymbolRef,
                               const ObjCObjectPointerType *)

namespace {
class ObjCGenericsChecker
    : public Checker<check::DeadSymbols, check::PreObjCMessage,
                     check::PostObjCMessage, check::PostStmt<CastExpr>> {
public:
  ProgramStateRef checkPointerEscape(ProgramStateRef State,
                                     const InvalidatedSymbols &Escaped,
                                     const CallEvent *Call,
                                     PointerEscapeKind Kind) const;

  void checkPreObjCMessage(const ObjCMethodCall &M, CheckerContext &C) const;
  void checkPostObjCMessage(const ObjCMethodCall &M, CheckerContext &C) const;
  void checkPostStmt(const CastExpr *CE, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;

private:
  mutable std::unique_ptr<BugType> BT;
  void initBugType() const {
    if (!BT)
      BT.reset(
          new BugType(this, "Generics", categories::CoreFoundationObjectiveC));
  }

  class GenericsBugVisitor : public BugReporterVisitorImpl<GenericsBugVisitor> {
  public:
    GenericsBugVisitor(SymbolRef S) : Sym(S) {}
    ~GenericsBugVisitor() override {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Sym);
    }

    PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                   const ExplodedNode *PrevN,
                                   BugReporterContext &BRC,
                                   BugReport &BR) override;

  private:
    // The tracked symbol.
    SymbolRef Sym;
  };

  void reportBug(const ObjCObjectPointerType *From,
                 const ObjCObjectPointerType *To, ExplodedNode *N,
                 SymbolRef Sym, CheckerContext &C,
                 const Stmt *ReportedNode = nullptr) const {
    initBugType();
    SmallString<64> Buf;
    llvm::raw_svector_ostream OS(Buf);
    OS << "Incompatible pointer types assigning to '";
    QualType::print(To, Qualifiers(), OS, C.getLangOpts(), llvm::Twine());
    OS << "' from '";
    QualType::print(From, Qualifiers(), OS, C.getLangOpts(), llvm::Twine());
    OS << "'";
    std::unique_ptr<BugReport> R(new BugReport(*BT, OS.str(), N));
    R->markInteresting(Sym);
    R->addVisitor(llvm::make_unique<GenericsBugVisitor>(Sym));
    if (ReportedNode)
      R->addRange(ReportedNode->getSourceRange());
    C.emitReport(std::move(R));
  }
};
} // end anonymous namespace

PathDiagnosticPiece *ObjCGenericsChecker::GenericsBugVisitor::VisitNode(
    const ExplodedNode *N, const ExplodedNode *PrevN, BugReporterContext &BRC,
    BugReport &BR) {
  ProgramStateRef state = N->getState();
  ProgramStateRef statePrev = PrevN->getState();

  const ObjCObjectPointerType *const *TrackedType =
      state->get<TypeParamMap>(Sym);
  const ObjCObjectPointerType *const *TrackedTypePrev =
      statePrev->get<TypeParamMap>(Sym);
  if (!TrackedType)
    return nullptr;

  if (TrackedTypePrev && *TrackedTypePrev == *TrackedType)
    return nullptr;

  // Retrieve the associated statement.
  const Stmt *S = nullptr;
  ProgramPoint ProgLoc = N->getLocation();
  if (Optional<StmtPoint> SP = ProgLoc.getAs<StmtPoint>()) {
    S = SP->getStmt();
  }

  if (!S)
    return nullptr;

  const LangOptions &LangOpts = BRC.getASTContext().getLangOpts();

  SmallString<64> Buf;
  llvm::raw_svector_ostream OS(Buf);
  OS << "Type '";
  QualType::print(*TrackedType, Qualifiers(), OS, LangOpts, llvm::Twine());
  OS << "' is inferred from ";

  if (const auto *ExplicitCast = dyn_cast<ExplicitCastExpr>(S)) {
    OS << "explicit cast (from '";
    QualType::print(ExplicitCast->getSubExpr()->getType().getTypePtr(),
                    Qualifiers(), OS, LangOpts, llvm::Twine());
    OS << "' to '";
    QualType::print(ExplicitCast->getType().getTypePtr(), Qualifiers(), OS,
                    LangOpts, llvm::Twine());
    OS << "')";
  } else if (const auto *ImplicitCast = dyn_cast<ImplicitCastExpr>(S)) {
    OS << "implicit cast (from '";
    QualType::print(ImplicitCast->getSubExpr()->getType().getTypePtr(),
                    Qualifiers(), OS, LangOpts, llvm::Twine());
    OS << "' to '";
    QualType::print(ImplicitCast->getType().getTypePtr(), Qualifiers(), OS,
                    LangOpts, llvm::Twine());
    OS << "')";
  } else {
    OS << "this context";
  }

  // Generate the extra diagnostic.
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return new PathDiagnosticEventPiece(Pos, OS.str(), true, nullptr);
}

void ObjCGenericsChecker::checkDeadSymbols(SymbolReaper &SR,
                                           CheckerContext &C) const {
  if (!SR.hasDeadSymbols())
    return;

  ProgramStateRef State = C.getState();
  TypeParamMapTy TyParMap = State->get<TypeParamMap>();
  for (TypeParamMapTy::iterator I = TyParMap.begin(), E = TyParMap.end();
       I != E; ++I) {
    if (SR.isDead(I->first)) {
      State = State->remove<TypeParamMap>(I->first);
    }
  }
}

static const ObjCObjectPointerType *getMostInformativeDerivedClassImpl(
    const ObjCObjectPointerType *From, const ObjCObjectPointerType *To,
    const ObjCObjectPointerType *MostInformativeCandidate, ASTContext &C) {
  // Checking if from and to are the same classes modulo specialization.
  if (From->getInterfaceDecl()->getCanonicalDecl() ==
      To->getInterfaceDecl()->getCanonicalDecl()) {
    if (To->isSpecialized()) {
      assert(MostInformativeCandidate->isSpecialized());
      return MostInformativeCandidate;
    }
    return From;
  }
  const auto *SuperOfTo =
      To->getObjectType()->getSuperClassType()->getAs<ObjCObjectType>();
  assert(SuperOfTo);
  QualType SuperPtrOfToQual =
      C.getObjCObjectPointerType(QualType(SuperOfTo, 0));
  const auto *SuperPtrOfTo = SuperPtrOfToQual->getAs<ObjCObjectPointerType>();
  if (To->isUnspecialized())
    return getMostInformativeDerivedClassImpl(From, SuperPtrOfTo, SuperPtrOfTo,
                                              C);
  else
    return getMostInformativeDerivedClassImpl(From, SuperPtrOfTo,
                                              MostInformativeCandidate, C);
}

/// Get the most derived class if From that do not loose information about type
/// parameters. To has to be a subclass of From. From has to be specialized.
static const ObjCObjectPointerType *
getMostInformativeDerivedClass(const ObjCObjectPointerType *From,
                               const ObjCObjectPointerType *To, ASTContext &C) {
  return getMostInformativeDerivedClassImpl(From, To, To, C);
}

static bool storeWhenMoreInformative(ProgramStateRef &State, SymbolRef Sym,
                                     const ObjCObjectPointerType *const *Old,
                                     const ObjCObjectPointerType *New,
                                     ASTContext &C) {
  if (!Old || C.canAssignObjCInterfaces(*Old, New)) {
    State = State->set<TypeParamMap>(Sym, New);
    return true;
  }
  return false;
}

void ObjCGenericsChecker::checkPostStmt(const CastExpr *CE,
                                        CheckerContext &C) const {
  if (CE->getCastKind() != CK_BitCast)
    return;

  QualType OriginType = CE->getSubExpr()->getType();
  QualType DestType = CE->getType();

  const auto *OrigObjectPtrType = OriginType->getAs<ObjCObjectPointerType>();
  const auto *DestObjectPtrType = DestType->getAs<ObjCObjectPointerType>();

  if (!OrigObjectPtrType || !DestObjectPtrType)
    return;

  ASTContext &ASTCtxt = C.getASTContext();

  // This checker detects the subtyping relationships using the assignment
  // rules. In order to be able to do this the kindofness must be stripped
  // first. The checker treats every type as kindof type anyways: when the
  // tracked type is the subtype of the static type it tries to look up the
  // methods in the tracked type first.
  OrigObjectPtrType = OrigObjectPtrType->stripObjCKindOfTypeAndQuals(ASTCtxt);
  DestObjectPtrType = DestObjectPtrType->stripObjCKindOfTypeAndQuals(ASTCtxt);

  const ObjCObjectType *OrigObjectType = OrigObjectPtrType->getObjectType();
  const ObjCObjectType *DestObjectType = DestObjectPtrType->getObjectType();

  if (OrigObjectType->isUnspecialized() && DestObjectType->isUnspecialized())
    return;

  ProgramStateRef State = C.getState();
  SymbolRef Sym = State->getSVal(CE, C.getLocationContext()).getAsSymbol();
  if (!Sym)
    return;

  // Check which assignments are legal.
  bool OrigToDest =
      ASTCtxt.canAssignObjCInterfaces(DestObjectPtrType, OrigObjectPtrType);
  bool DestToOrig =
      ASTCtxt.canAssignObjCInterfaces(OrigObjectPtrType, DestObjectPtrType);
  const ObjCObjectPointerType *const *TrackedType =
      State->get<TypeParamMap>(Sym);

  // If OrigObjectType could convert to DestObjectType, this could be an
  // implicit cast. Do not treat that cast as explicit in that case.
  if (isa<ExplicitCastExpr>(CE) && !OrigToDest) {
    if (DestToOrig) {
      // Trust explicit downcasts.
      // However a downcast may also lose information. E. g.:
      //   MutableMap<T, U> : Map
      // The downcast to MutableMap loses the information about the types of the
      // Map (due to the type parameters are not being forwarded to Map), and in
      // general there is no way to recover that information from the
      // declaration. In order to have to most information, lets find the most
      // derived type that has all the type parameters forwarded.
      const ObjCObjectPointerType *WithMostInfo =
          getMostInformativeDerivedClass(OrigObjectPtrType, DestObjectPtrType,
                                         C.getASTContext());
      if (storeWhenMoreInformative(State, Sym, TrackedType, WithMostInfo,
                                   ASTCtxt))
        C.addTransition(State);
      return;
    }
    // Mismatched types. If the DestType specialized, store it. Forget the
    // tracked type otherwise.
    if (DestObjectPtrType->isSpecialized()) {
      State = State->set<TypeParamMap>(Sym, DestObjectPtrType);
      C.addTransition(State);
    } else if (TrackedType) {
      State = State->remove<TypeParamMap>(Sym);
      C.addTransition(State);
    }
    return;
  }

  // Handle implicit casts and explicit upcasts.

  if (DestObjectType->isUnspecialized()) {
    assert(OrigObjectType->isSpecialized());
    // In case we already have some type information for this symbol from a
    // Specialized -> Specialized conversion, do not record the OrigType,
    // because it might contain less type information than the tracked type.
    if (!TrackedType) {
      State = State->set<TypeParamMap>(Sym, OrigObjectPtrType);
      C.addTransition(State);
    }
    return;
  }

  // The destination type is specialized.

  // The tracked type should be the sub or super class of the static destination
  // type. When an (implicit) upcast or a downcast happens according to static
  // types, and there is no subtyping relationship between the tracked and the
  // static destination types, it indicates an error.
  if (TrackedType &&
      !ASTCtxt.canAssignObjCInterfaces(DestObjectPtrType, *TrackedType) &&
      !ASTCtxt.canAssignObjCInterfaces(*TrackedType, DestObjectPtrType)) {
    static CheckerProgramPointTag IllegalConv(this, "IllegalConversion");
    ExplodedNode *N = C.addTransition(State, C.getPredecessor(), &IllegalConv);
    reportBug(*TrackedType, DestObjectPtrType, N, Sym, C);
    return;
  }

  if (OrigToDest && !DestToOrig) {
    // When upcast happens, store the type with the most information about the
    // type parameters.
    const ObjCObjectPointerType *WithMostInfo = getMostInformativeDerivedClass(
        DestObjectPtrType, OrigObjectPtrType, ASTCtxt);
    if (storeWhenMoreInformative(State, Sym, TrackedType, WithMostInfo,
                                 ASTCtxt))
      C.addTransition(State);
    return;
  }

  // Downcast happens.

  // Trust tracked type on unspecialized value -> specialized implicit
  // downcasts.
  if (storeWhenMoreInformative(State, Sym, TrackedType, DestObjectPtrType,
                               ASTCtxt)) {
    C.addTransition(State);
  }
}

static const Expr *stripCastsAndSugar(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  if (const PseudoObjectExpr *POE = dyn_cast<PseudoObjectExpr>(E))
    E = POE->getSyntacticForm()->IgnoreParenImpCasts();
  if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(E))
    E = OVE->getSourceExpr()->IgnoreParenImpCasts();
  return E;
}

// This callback is used to infer the types for Class variables. This info is
// used later to validate messages that sent to classes. Class variables are
// initialized with by invoking the 'class' method on a class.
void ObjCGenericsChecker::checkPostObjCMessage(const ObjCMethodCall &M,
                                               CheckerContext &C) const {
  const ObjCMessageExpr *MessageExpr = M.getOriginExpr();

  SymbolRef Sym = M.getReturnValue().getAsSymbol();
  if (!Sym)
    return;

  Selector Sel = MessageExpr->getSelector();
  // We are only interested in cases where the class method is invoked on a
  // class. This method is provided by the runtime and available on all classes.
  if (MessageExpr->getReceiverKind() != ObjCMessageExpr::Class ||
      Sel.getAsString() != "class")
    return;

  QualType ReceiverType = MessageExpr->getClassReceiver();
  const auto *ReceiverClassType = ReceiverType->getAs<ObjCObjectType>();
  QualType ReceiverClassPointerType =
      C.getASTContext().getObjCObjectPointerType(
          QualType(ReceiverClassType, 0));

  if (!ReceiverClassType->isSpecialized())
    return;
  const auto *InferredType =
      ReceiverClassPointerType->getAs<ObjCObjectPointerType>();
  assert(InferredType);

  ProgramStateRef State = C.getState();
  State = State->set<TypeParamMap>(Sym, InferredType);
  C.addTransition(State);
}

static bool isObjCTypeParamDependent(QualType Type) {
  // It is illegal to typedef parameterized types inside an interface. Therfore
  // an
  // Objective-C type can only be dependent on a type parameter when the type
  // parameter structurally present in the type itself.
  class IsObjCTypeParamDependentTypeVisitor
      : public RecursiveASTVisitor<IsObjCTypeParamDependentTypeVisitor> {
  public:
    IsObjCTypeParamDependentTypeVisitor() : Result(false) {}
    bool VisitTypedefType(const TypedefType *Type) {
      if (isa<ObjCTypeParamDecl>(Type->getDecl())) {
        Result = true;
        return false;
      }
      return true;
    }
    bool getResult() { return Result; }

  private:
    bool Result;
  };

  IsObjCTypeParamDependentTypeVisitor Visitor;
  Visitor.TraverseType(Type);
  return Visitor.getResult();
}

// A method might not be available in the interface indicated by the static
// type. However it might be available in the tracked type. In order to properly
// substitute the type parameters we need the declaration context of the method.
// The more specialized the enclosing class of the method is, the more likely
// that the parameter substitution will be successful.
static const ObjCMethodDecl *
findMethodDecl(const ObjCMessageExpr *MessageExpr,
               const ObjCObjectPointerType *TrackedType, ASTContext &ASTCtxt) {
  const ObjCMethodDecl *Method = nullptr;

  QualType ReceiverType = MessageExpr->getReceiverType();
  const auto *ReceiverObjectPtrType =
      ReceiverType->getAs<ObjCObjectPointerType>();

  // Do this "devirtualization" on instance and class methods only. Trust the
  // static type on super and super class calls.
  if (MessageExpr->getReceiverKind() == ObjCMessageExpr::Instance ||
      MessageExpr->getReceiverKind() == ObjCMessageExpr::Class) {
    // When the receiver type is id, Class, or some super class of the tracked
    // type, look up the method in the tracked type, not in the receiver type.
    // This way we preserve more information.
    if (ReceiverType->isObjCIdType() || ReceiverType->isObjCClassType() ||
        ASTCtxt.canAssignObjCInterfaces(ReceiverObjectPtrType, TrackedType)) {
      const ObjCInterfaceDecl *InterfaceDecl = TrackedType->getInterfaceDecl();
      // The method might not be found.
      Selector Sel = MessageExpr->getSelector();
      Method = InterfaceDecl->lookupInstanceMethod(Sel);
      if (!Method)
        Method = InterfaceDecl->lookupClassMethod(Sel);
    }
  }

  // Fallback to statick method lookup when the one based on the tracked type
  // failed.
  return Method ? Method : MessageExpr->getMethodDecl();
}

// When the receiver has a tracked type, use that type to validate the
// argumments of the message expression and the return value.
void ObjCGenericsChecker::checkPreObjCMessage(const ObjCMethodCall &M,
                                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef Sym = M.getReceiverSVal().getAsSymbol();
  if (!Sym)
    return;

  const ObjCObjectPointerType *const *TrackedType =
      State->get<TypeParamMap>(Sym);
  if (!TrackedType)
    return;

  // Get the type arguments from tracked type and substitute type arguments
  // before do the semantic check.

  ASTContext &ASTCtxt = C.getASTContext();
  const ObjCMessageExpr *MessageExpr = M.getOriginExpr();
  const ObjCMethodDecl *Method =
      findMethodDecl(MessageExpr, *TrackedType, ASTCtxt);

  // It is possible to call non-existent methods in Obj-C.
  if (!Method)
    return;

  Optional<ArrayRef<QualType>> TypeArgs =
      (*TrackedType)->getObjCSubstitutions(Method->getDeclContext());
  // This case might happen when there is an unspecialized override of a
  // specialized method.
  if (!TypeArgs)
    return;

  for (unsigned i = 0; i < Method->param_size(); i++) {
    const Expr *Arg = MessageExpr->getArg(i);
    const ParmVarDecl *Param = Method->parameters()[i];

    QualType OrigParamType = Param->getType();
    if (!isObjCTypeParamDependent(OrigParamType))
      continue;

    QualType ParamType = OrigParamType.substObjCTypeArgs(
        ASTCtxt, *TypeArgs, ObjCSubstitutionContext::Parameter);
    // Check if it can be assigned
    const auto *ParamObjectPtrType = ParamType->getAs<ObjCObjectPointerType>();
    const auto *ArgObjectPtrType =
        stripCastsAndSugar(Arg)->getType()->getAs<ObjCObjectPointerType>();
    if (!ParamObjectPtrType || !ArgObjectPtrType)
      continue;

    // Check if we have more concrete tracked type that is not a super type of
    // the static argument type.
    SVal ArgSVal = M.getArgSVal(i);
    SymbolRef ArgSym = ArgSVal.getAsSymbol();
    if (ArgSym) {
      const ObjCObjectPointerType *const *TrackedArgType =
          State->get<TypeParamMap>(ArgSym);
      if (TrackedArgType &&
          ASTCtxt.canAssignObjCInterfaces(ArgObjectPtrType, *TrackedArgType)) {
        ArgObjectPtrType = *TrackedArgType;
      }
    }

    // Warn when argument is incompatible with the parameter.
    if (!ASTCtxt.canAssignObjCInterfaces(ParamObjectPtrType,
                                         ArgObjectPtrType)) {
      static CheckerProgramPointTag Tag(this, "ArgTypeMismatch");
      ExplodedNode *N = C.addTransition(State, C.getPredecessor(), &Tag);
      reportBug(ArgObjectPtrType, ParamObjectPtrType, N, Sym, C, Arg);
      return;
    }
  }
  QualType StaticResultType = Method->getReturnType();
  // Check whether the result type was a type parameter.
  bool IsDeclaredAsInstanceType =
      StaticResultType == ASTCtxt.getObjCInstanceType();
  if (!isObjCTypeParamDependent(StaticResultType) && !IsDeclaredAsInstanceType)
    return;

  QualType ResultType = Method->getReturnType().substObjCTypeArgs(
      ASTCtxt, *TypeArgs, ObjCSubstitutionContext::Result);
  if (IsDeclaredAsInstanceType)
    ResultType = QualType(*TrackedType, 0);

  const Stmt *Parent =
      C.getCurrentAnalysisDeclContext()->getParentMap().getParent(MessageExpr);
  if (M.getMessageKind() != OCM_Message) {
    // Properties and subscripts are not direct parents.
    Parent =
        C.getCurrentAnalysisDeclContext()->getParentMap().getParent(Parent);
  }

  const auto *ImplicitCast = dyn_cast_or_null<ImplicitCastExpr>(Parent);
  if (!ImplicitCast || ImplicitCast->getCastKind() != CK_BitCast)
    return;

  const auto *ExprTypeAboveCast =
      ImplicitCast->getType()->getAs<ObjCObjectPointerType>();
  const auto *ResultPtrType = ResultType->getAs<ObjCObjectPointerType>();

  if (!ExprTypeAboveCast || !ResultPtrType)
    return;

  // Only warn on unrelated types to avoid too many false positives on
  // downcasts.
  if (!ASTCtxt.canAssignObjCInterfaces(ExprTypeAboveCast, ResultPtrType) &&
      !ASTCtxt.canAssignObjCInterfaces(ResultPtrType, ExprTypeAboveCast)) {
    static CheckerProgramPointTag Tag(this, "ReturnTypeMismatch");
    ExplodedNode *N = C.addTransition(State, C.getPredecessor(), &Tag);
    reportBug(ResultPtrType, ExprTypeAboveCast, N, Sym, C);
    return;
  }
}

/// Register checker.
void ento::registerObjCGenericsChecker(CheckerManager &mgr) {
  mgr.registerChecker<ObjCGenericsChecker>();
}
