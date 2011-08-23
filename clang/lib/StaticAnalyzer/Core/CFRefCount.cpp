// CFRefCount.cpp - Transfer functions for tracking simple values -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the methods for CFRefCount, which implements
//  a reference count checker for Core Foundation (Mac OS X).
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngineBuilders.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/TransferFuncs.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdarg>

using namespace clang;
using namespace ento;
using llvm::StrInStrNoCase;

namespace {
class InstanceReceiver {
  ObjCMessage Msg;
  const LocationContext *LC;
public:
  InstanceReceiver() : LC(0) { }
  InstanceReceiver(const ObjCMessage &msg,
                   const LocationContext *lc = 0) : Msg(msg), LC(lc) {}

  bool isValid() const {
    return Msg.isValid() && Msg.isInstanceMessage();
  }
  operator bool() const {
    return isValid();
  }

  SVal getSValAsScalarOrLoc(const ProgramState *state) {
    assert(isValid());
    // We have an expression for the receiver?  Fetch the value
    // of that expression.
    if (const Expr *Ex = Msg.getInstanceReceiver())
      return state->getSValAsScalarOrLoc(Ex);

    // Otherwise we are sending a message to super.  In this case the
    // object reference is the same as 'self'.
    if (const ImplicitParamDecl *SelfDecl = LC->getSelfDecl())
      return state->getSVal(state->getRegion(SelfDecl, LC));

    return UnknownVal();
  }

  SourceRange getSourceRange() const {
    assert(isValid());
    if (const Expr *Ex = Msg.getInstanceReceiver())
      return Ex->getSourceRange();

    // Otherwise we are sending a message to super.
    SourceLocation L = Msg.getSuperLoc();
    assert(L.isValid());
    return SourceRange(L, L);
  }
};
}

namespace {
class GenericNodeBuilderRefCount {
  StmtNodeBuilder *SNB;
  const Stmt *S;
  const ProgramPointTag *tag;
  EndOfFunctionNodeBuilder *ENB;
public:
  GenericNodeBuilderRefCount(StmtNodeBuilder &snb, const Stmt *s,
                     const ProgramPointTag *t)
  : SNB(&snb), S(s), tag(t), ENB(0) {}

  GenericNodeBuilderRefCount(EndOfFunctionNodeBuilder &enb)
  : SNB(0), S(0), tag(0), ENB(&enb) {}

  ExplodedNode *MakeNode(const ProgramState *state, ExplodedNode *Pred) {
    if (SNB)
      return SNB->generateNode(PostStmt(S, Pred->getLocationContext(), tag),
                               state, Pred);

    assert(ENB);
    return ENB->generateNode(state, Pred);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Primitives used for constructing summaries for function/method calls.
//===----------------------------------------------------------------------===//

/// ArgEffect is used to summarize a function/method call's effect on a
/// particular argument.
enum ArgEffect { Autorelease, Dealloc, DecRef, DecRefMsg, DoNothing,
                 DecRefBridgedTransfered,
                 DoNothingByRef, IncRefMsg, IncRef, MakeCollectable, MayEscape,
                 NewAutoreleasePool, SelfOwn, StopTracking };

namespace llvm {
template <> struct FoldingSetTrait<ArgEffect> {
static inline void Profile(const ArgEffect X, FoldingSetNodeID& ID) {
  ID.AddInteger((unsigned) X);
}
};
} // end llvm namespace

/// ArgEffects summarizes the effects of a function/method call on all of
/// its arguments.
typedef llvm::ImmutableMap<unsigned,ArgEffect> ArgEffects;

namespace {

///  RetEffect is used to summarize a function/method call's behavior with
///  respect to its return value.
class RetEffect {
public:
  enum Kind { NoRet, OwnedSymbol, OwnedAllocatedSymbol,
              NotOwnedSymbol, GCNotOwnedSymbol, ARCNotOwnedSymbol,
              OwnedWhenTrackedReceiver };

  enum ObjKind { CF, ObjC, AnyObj };

private:
  Kind K;
  ObjKind O;

  RetEffect(Kind k, ObjKind o = AnyObj) : K(k), O(o) {}

public:
  Kind getKind() const { return K; }

  ObjKind getObjKind() const { return O; }

  bool isOwned() const {
    return K == OwnedSymbol || K == OwnedAllocatedSymbol ||
           K == OwnedWhenTrackedReceiver;
  }

  bool operator==(const RetEffect &Other) const {
    return K == Other.K && O == Other.O;
  }

  static RetEffect MakeOwnedWhenTrackedReceiver() {
    return RetEffect(OwnedWhenTrackedReceiver, ObjC);
  }

  static RetEffect MakeOwned(ObjKind o, bool isAllocated = false) {
    return RetEffect(isAllocated ? OwnedAllocatedSymbol : OwnedSymbol, o);
  }
  static RetEffect MakeNotOwned(ObjKind o) {
    return RetEffect(NotOwnedSymbol, o);
  }
  static RetEffect MakeGCNotOwned() {
    return RetEffect(GCNotOwnedSymbol, ObjC);
  }
  static RetEffect MakeARCNotOwned() {
    return RetEffect(ARCNotOwnedSymbol, ObjC);
  }
  static RetEffect MakeNoRet() {
    return RetEffect(NoRet);
  }
};

//===----------------------------------------------------------------------===//
// Reference-counting logic (typestate + counts).
//===----------------------------------------------------------------------===//

class RefVal {
public:
  enum Kind {
    Owned = 0, // Owning reference.
    NotOwned,  // Reference is not owned by still valid (not freed).
    Released,  // Object has been released.
    ReturnedOwned, // Returned object passes ownership to caller.
    ReturnedNotOwned, // Return object does not pass ownership to caller.
    ERROR_START,
    ErrorDeallocNotOwned, // -dealloc called on non-owned object.
    ErrorDeallocGC, // Calling -dealloc with GC enabled.
    ErrorUseAfterRelease, // Object used after released.
    ErrorReleaseNotOwned, // Release of an object that was not owned.
    ERROR_LEAK_START,
    ErrorLeak,  // A memory leak due to excessive reference counts.
    ErrorLeakReturned, // A memory leak due to the returning method not having
                       // the correct naming conventions.
    ErrorGCLeakReturned,
    ErrorOverAutorelease,
    ErrorReturnedNotOwned
  };

private:
  Kind kind;
  RetEffect::ObjKind okind;
  unsigned Cnt;
  unsigned ACnt;
  QualType T;

  RefVal(Kind k, RetEffect::ObjKind o, unsigned cnt, unsigned acnt, QualType t)
  : kind(k), okind(o), Cnt(cnt), ACnt(acnt), T(t) {}

public:
  Kind getKind() const { return kind; }

  RetEffect::ObjKind getObjKind() const { return okind; }

  unsigned getCount() const { return Cnt; }
  unsigned getAutoreleaseCount() const { return ACnt; }
  unsigned getCombinedCounts() const { return Cnt + ACnt; }
  void clearCounts() { Cnt = 0; ACnt = 0; }
  void setCount(unsigned i) { Cnt = i; }
  void setAutoreleaseCount(unsigned i) { ACnt = i; }

  QualType getType() const { return T; }

  bool isOwned() const {
    return getKind() == Owned;
  }

  bool isNotOwned() const {
    return getKind() == NotOwned;
  }

  bool isReturnedOwned() const {
    return getKind() == ReturnedOwned;
  }

  bool isReturnedNotOwned() const {
    return getKind() == ReturnedNotOwned;
  }

  static RefVal makeOwned(RetEffect::ObjKind o, QualType t,
                          unsigned Count = 1) {
    return RefVal(Owned, o, Count, 0, t);
  }

  static RefVal makeNotOwned(RetEffect::ObjKind o, QualType t,
                             unsigned Count = 0) {
    return RefVal(NotOwned, o, Count, 0, t);
  }

  // Comparison, profiling, and pretty-printing.

  bool operator==(const RefVal& X) const {
    return kind == X.kind && Cnt == X.Cnt && T == X.T && ACnt == X.ACnt;
  }

  RefVal operator-(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() - i,
                  getAutoreleaseCount(), getType());
  }

  RefVal operator+(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() + i,
                  getAutoreleaseCount(), getType());
  }

  RefVal operator^(Kind k) const {
    return RefVal(k, getObjKind(), getCount(), getAutoreleaseCount(),
                  getType());
  }

  RefVal autorelease() const {
    return RefVal(getKind(), getObjKind(), getCount(), getAutoreleaseCount()+1,
                  getType());
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) kind);
    ID.AddInteger(Cnt);
    ID.AddInteger(ACnt);
    ID.Add(T);
  }

  void print(raw_ostream &Out) const;
};

void RefVal::print(raw_ostream &Out) const {
  if (!T.isNull())
    Out << "Tracked Type:" << T.getAsString() << '\n';

  switch (getKind()) {
    default: assert(false);
    case Owned: {
      Out << "Owned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case NotOwned: {
      Out << "NotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case ReturnedOwned: {
      Out << "ReturnedOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case ReturnedNotOwned: {
      Out << "ReturnedNotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case Released:
      Out << "Released";
      break;

    case ErrorDeallocGC:
      Out << "-dealloc (GC)";
      break;

    case ErrorDeallocNotOwned:
      Out << "-dealloc (not-owned)";
      break;

    case ErrorLeak:
      Out << "Leaked";
      break;

    case ErrorLeakReturned:
      Out << "Leaked (Bad naming)";
      break;

    case ErrorGCLeakReturned:
      Out << "Leaked (GC-ed at return)";
      break;

    case ErrorUseAfterRelease:
      Out << "Use-After-Release [ERROR]";
      break;

    case ErrorReleaseNotOwned:
      Out << "Release of Not-Owned [ERROR]";
      break;

    case RefVal::ErrorOverAutorelease:
      Out << "Over autoreleased";
      break;

    case RefVal::ErrorReturnedNotOwned:
      Out << "Non-owned object returned instead of owned";
      break;
  }

  if (ACnt) {
    Out << " [ARC +" << ACnt << ']';
  }
}
} //end anonymous namespace

//===----------------------------------------------------------------------===//
// RefBindings - State used to track object reference counts.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<SymbolRef, RefVal> RefBindings;

namespace clang {
namespace ento {
template<>
struct ProgramStateTrait<RefBindings>
  : public ProgramStatePartialTrait<RefBindings> {
  static void *GDMIndex() {
    static int RefBIndex = 0;
    return &RefBIndex;
  }
};
}
}

//===----------------------------------------------------------------------===//
// Summaries
//===----------------------------------------------------------------------===//

namespace {
class RetainSummary {
  /// Args - an ordered vector of (index, ArgEffect) pairs, where index
  ///  specifies the argument (starting from 0).  This can be sparsely
  ///  populated; arguments with no entry in Args use 'DefaultArgEffect'.
  ArgEffects Args;

  /// DefaultArgEffect - The default ArgEffect to apply to arguments that
  ///  do not have an entry in Args.
  ArgEffect   DefaultArgEffect;

  /// Receiver - If this summary applies to an Objective-C message expression,
  ///  this is the effect applied to the state of the receiver.
  ArgEffect   Receiver;

  /// Ret - The effect on the return value.  Used to indicate if the
  ///  function/method call returns a new tracked symbol.
  RetEffect   Ret;

public:
  RetainSummary(ArgEffects A, RetEffect R, ArgEffect defaultEff,
                ArgEffect ReceiverEff)
    : Args(A), DefaultArgEffect(defaultEff), Receiver(ReceiverEff), Ret(R) {}

  /// getArg - Return the argument effect on the argument specified by
  ///  idx (starting from 0).
  ArgEffect getArg(unsigned idx) const {
    if (const ArgEffect *AE = Args.lookup(idx))
      return *AE;

    return DefaultArgEffect;
  }
  
  void addArg(ArgEffects::Factory &af, unsigned idx, ArgEffect e) {
    Args = af.add(Args, idx, e);
  }

  /// setDefaultArgEffect - Set the default argument effect.
  void setDefaultArgEffect(ArgEffect E) {
    DefaultArgEffect = E;
  }

  /// getRetEffect - Returns the effect on the return value of the call.
  RetEffect getRetEffect() const { return Ret; }

  /// setRetEffect - Set the effect of the return value of the call.
  void setRetEffect(RetEffect E) { Ret = E; }

  
  /// Sets the effect on the receiver of the message.
  void setReceiverEffect(ArgEffect e) { Receiver = e; }
  
  /// getReceiverEffect - Returns the effect on the receiver of the call.
  ///  This is only meaningful if the summary applies to an ObjCMessageExpr*.
  ArgEffect getReceiverEffect() const { return Receiver; }

  /// Test if two retain summaries are identical. Note that merely equivalent
  /// summaries are not necessarily identical (for example, if an explicit 
  /// argument effect matches the default effect).
  bool operator==(const RetainSummary &Other) const {
    return Args == Other.Args && DefaultArgEffect == Other.DefaultArgEffect &&
           Receiver == Other.Receiver && Ret == Other.Ret;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Data structures for constructing summaries.
//===----------------------------------------------------------------------===//

namespace {
class ObjCSummaryKey {
  IdentifierInfo* II;
  Selector S;
public:
  ObjCSummaryKey(IdentifierInfo* ii, Selector s)
    : II(ii), S(s) {}

  ObjCSummaryKey(const ObjCInterfaceDecl *d, Selector s)
    : II(d ? d->getIdentifier() : 0), S(s) {}

  ObjCSummaryKey(const ObjCInterfaceDecl *d, IdentifierInfo *ii, Selector s)
    : II(d ? d->getIdentifier() : ii), S(s) {}

  ObjCSummaryKey(Selector s)
    : II(0), S(s) {}

  IdentifierInfo* getIdentifier() const { return II; }
  Selector getSelector() const { return S; }
};
}

namespace llvm {
template <> struct DenseMapInfo<ObjCSummaryKey> {
  static inline ObjCSummaryKey getEmptyKey() {
    return ObjCSummaryKey(DenseMapInfo<IdentifierInfo*>::getEmptyKey(),
                          DenseMapInfo<Selector>::getEmptyKey());
  }

  static inline ObjCSummaryKey getTombstoneKey() {
    return ObjCSummaryKey(DenseMapInfo<IdentifierInfo*>::getTombstoneKey(),
                          DenseMapInfo<Selector>::getTombstoneKey());
  }

  static unsigned getHashValue(const ObjCSummaryKey &V) {
    return (DenseMapInfo<IdentifierInfo*>::getHashValue(V.getIdentifier())
            & 0x88888888)
        | (DenseMapInfo<Selector>::getHashValue(V.getSelector())
            & 0x55555555);
  }

  static bool isEqual(const ObjCSummaryKey& LHS, const ObjCSummaryKey& RHS) {
    return DenseMapInfo<IdentifierInfo*>::isEqual(LHS.getIdentifier(),
                                                  RHS.getIdentifier()) &&
           DenseMapInfo<Selector>::isEqual(LHS.getSelector(),
                                           RHS.getSelector());
  }

};
template <>
struct isPodLike<ObjCSummaryKey> { static const bool value = true; };
} // end llvm namespace

namespace {
class ObjCSummaryCache {
  typedef llvm::DenseMap<ObjCSummaryKey, RetainSummary*> MapTy;
  MapTy M;
public:
  ObjCSummaryCache() {}

  RetainSummary* find(const ObjCInterfaceDecl *D, IdentifierInfo *ClsName,
                Selector S) {
    // Lookup the method using the decl for the class @interface.  If we
    // have no decl, lookup using the class name.
    return D ? find(D, S) : find(ClsName, S);
  }

  RetainSummary* find(const ObjCInterfaceDecl *D, Selector S) {
    // Do a lookup with the (D,S) pair.  If we find a match return
    // the iterator.
    ObjCSummaryKey K(D, S);
    MapTy::iterator I = M.find(K);

    if (I != M.end() || !D)
      return I->second;

    // Walk the super chain.  If we find a hit with a parent, we'll end
    // up returning that summary.  We actually allow that key (null,S), as
    // we cache summaries for the null ObjCInterfaceDecl* to allow us to
    // generate initial summaries without having to worry about NSObject
    // being declared.
    // FIXME: We may change this at some point.
    for (ObjCInterfaceDecl *C=D->getSuperClass() ;; C=C->getSuperClass()) {
      if ((I = M.find(ObjCSummaryKey(C, S))) != M.end())
        break;

      if (!C)
        return NULL;
    }

    // Cache the summary with original key to make the next lookup faster
    // and return the iterator.
    RetainSummary *Summ = I->second;
    M[K] = Summ;
    return Summ;
  }

  RetainSummary* find(IdentifierInfo* II, Selector S) {
    // FIXME: Class method lookup.  Right now we dont' have a good way
    // of going between IdentifierInfo* and the class hierarchy.
    MapTy::iterator I = M.find(ObjCSummaryKey(II, S));

    if (I == M.end())
      I = M.find(ObjCSummaryKey(S));

    return I == M.end() ? NULL : I->second;
  }

  RetainSummary*& operator[](ObjCSummaryKey K) {
    return M[K];
  }

  RetainSummary*& operator[](Selector S) {
    return M[ ObjCSummaryKey(S) ];
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Data structures for managing collections of summaries.
//===----------------------------------------------------------------------===//

namespace {
class RetainSummaryManager {

  //==-----------------------------------------------------------------==//
  //  Typedefs.
  //==-----------------------------------------------------------------==//

  typedef llvm::DenseMap<const FunctionDecl*, RetainSummary*>
          FuncSummariesTy;

  typedef ObjCSummaryCache ObjCMethodSummariesTy;

  //==-----------------------------------------------------------------==//
  //  Data.
  //==-----------------------------------------------------------------==//

  /// Ctx - The ASTContext object for the analyzed ASTs.
  ASTContext &Ctx;

  /// CFDictionaryCreateII - An IdentifierInfo* representing the indentifier
  ///  "CFDictionaryCreate".
  IdentifierInfo* CFDictionaryCreateII;

  /// GCEnabled - Records whether or not the analyzed code runs in GC mode.
  const bool GCEnabled;

  /// Records whether or not the analyzed code runs in ARC mode.
  const bool ARCEnabled;

  /// FuncSummaries - A map from FunctionDecls to summaries.
  FuncSummariesTy FuncSummaries;

  /// ObjCClassMethodSummaries - A map from selectors (for instance methods)
  ///  to summaries.
  ObjCMethodSummariesTy ObjCClassMethodSummaries;

  /// ObjCMethodSummaries - A map from selectors to summaries.
  ObjCMethodSummariesTy ObjCMethodSummaries;

  /// BPAlloc - A BumpPtrAllocator used for allocating summaries, ArgEffects,
  ///  and all other data used by the checker.
  llvm::BumpPtrAllocator BPAlloc;

  /// AF - A factory for ArgEffects objects.
  ArgEffects::Factory AF;

  /// ScratchArgs - A holding buffer for construct ArgEffects.
  ArgEffects ScratchArgs;

  /// ObjCAllocRetE - Default return effect for methods returning Objective-C
  ///  objects.
  RetEffect ObjCAllocRetE;

  /// ObjCInitRetE - Default return effect for init methods returning
  ///   Objective-C objects.
  RetEffect ObjCInitRetE;

  RetainSummary DefaultSummary;
  RetainSummary* StopSummary;

  //==-----------------------------------------------------------------==//
  //  Methods.
  //==-----------------------------------------------------------------==//

  /// getArgEffects - Returns a persistent ArgEffects object based on the
  ///  data in ScratchArgs.
  ArgEffects getArgEffects();

  enum UnaryFuncKind { cfretain, cfrelease, cfmakecollectable };

public:
  RetEffect getObjAllocRetEffect() const { return ObjCAllocRetE; }

  RetainSummary* getUnarySummary(const FunctionType* FT, UnaryFuncKind func);

  RetainSummary* getCFSummaryCreateRule(const FunctionDecl *FD);
  RetainSummary* getCFSummaryGetRule(const FunctionDecl *FD);
  RetainSummary* getCFCreateGetRuleSummary(const FunctionDecl *FD, 
                                           StringRef FName);

  RetainSummary* getPersistentSummary(ArgEffects AE, RetEffect RetEff,
                                      ArgEffect ReceiverEff = DoNothing,
                                      ArgEffect DefaultEff = MayEscape);

  RetainSummary* getPersistentSummary(RetEffect RE,
                                      ArgEffect ReceiverEff = DoNothing,
                                      ArgEffect DefaultEff = MayEscape) {
    return getPersistentSummary(getArgEffects(), RE, ReceiverEff, DefaultEff);
  }

  RetainSummary *getPersistentStopSummary() {
    if (StopSummary)
      return StopSummary;

    StopSummary = getPersistentSummary(RetEffect::MakeNoRet(),
                                       StopTracking, StopTracking);

    return StopSummary;
  }

  RetainSummary *getInitMethodSummary(QualType RetTy);

  void InitializeClassMethodSummaries();
  void InitializeMethodSummaries();
private:
  void addNSObjectClsMethSummary(Selector S, RetainSummary *Summ) {
    ObjCClassMethodSummaries[S] = Summ;
  }

  void addNSObjectMethSummary(Selector S, RetainSummary *Summ) {
    ObjCMethodSummaries[S] = Summ;
  }

  void addClassMethSummary(const char* Cls, const char* nullaryName,
                           RetainSummary *Summ) {
    IdentifierInfo* ClsII = &Ctx.Idents.get(Cls);
    Selector S = GetNullarySelector(nullaryName, Ctx);
    ObjCClassMethodSummaries[ObjCSummaryKey(ClsII, S)]  = Summ;
  }

  void addInstMethSummary(const char* Cls, const char* nullaryName,
                          RetainSummary *Summ) {
    IdentifierInfo* ClsII = &Ctx.Idents.get(Cls);
    Selector S = GetNullarySelector(nullaryName, Ctx);
    ObjCMethodSummaries[ObjCSummaryKey(ClsII, S)]  = Summ;
  }

  Selector generateSelector(va_list argp) {
    SmallVector<IdentifierInfo*, 10> II;

    while (const char* s = va_arg(argp, const char*))
      II.push_back(&Ctx.Idents.get(s));

    return Ctx.Selectors.getSelector(II.size(), &II[0]);
  }

  void addMethodSummary(IdentifierInfo *ClsII, ObjCMethodSummariesTy& Summaries,
                        RetainSummary* Summ, va_list argp) {
    Selector S = generateSelector(argp);
    Summaries[ObjCSummaryKey(ClsII, S)] = Summ;
  }

  void addInstMethSummary(const char* Cls, RetainSummary* Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(&Ctx.Idents.get(Cls), ObjCMethodSummaries, Summ, argp);
    va_end(argp);
  }

  void addClsMethSummary(const char* Cls, RetainSummary* Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(&Ctx.Idents.get(Cls),ObjCClassMethodSummaries, Summ, argp);
    va_end(argp);
  }

  void addClsMethSummary(IdentifierInfo *II, RetainSummary* Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(II, ObjCClassMethodSummaries, Summ, argp);
    va_end(argp);
  }

public:

  RetainSummaryManager(ASTContext &ctx, bool gcenabled, bool usesARC)
   : Ctx(ctx),
     CFDictionaryCreateII(&ctx.Idents.get("CFDictionaryCreate")),
     GCEnabled(gcenabled),
     ARCEnabled(usesARC),
     AF(BPAlloc), ScratchArgs(AF.getEmptyMap()),
     ObjCAllocRetE(gcenabled
                    ? RetEffect::MakeGCNotOwned()
                    : (usesARC ? RetEffect::MakeARCNotOwned()
                               : RetEffect::MakeOwned(RetEffect::ObjC, true))),
     ObjCInitRetE(gcenabled 
                    ? RetEffect::MakeGCNotOwned()
                    : (usesARC ? RetEffect::MakeARCNotOwned()
                               : RetEffect::MakeOwnedWhenTrackedReceiver())),
     DefaultSummary(AF.getEmptyMap() /* per-argument effects (none) */,
                    RetEffect::MakeNoRet() /* return effect */,
                    MayEscape, /* default argument effect */
                    DoNothing /* receiver effect */),
     StopSummary(0) {

    InitializeClassMethodSummaries();
    InitializeMethodSummaries();
  }

  ~RetainSummaryManager();

  RetainSummary* getSummary(const FunctionDecl *FD);

  RetainSummary *getInstanceMethodSummary(const ObjCMessage &msg,
                                          const ProgramState *state,
                                          const LocationContext *LC);

  RetainSummary* getInstanceMethodSummary(const ObjCMessage &msg,
                                          const ObjCInterfaceDecl *ID) {
    return getInstanceMethodSummary(msg.getSelector(), 0,
                            ID, msg.getMethodDecl(), msg.getType(Ctx));
  }

  RetainSummary* getInstanceMethodSummary(Selector S, IdentifierInfo *ClsName,
                                          const ObjCInterfaceDecl *ID,
                                          const ObjCMethodDecl *MD,
                                          QualType RetTy);

  RetainSummary *getClassMethodSummary(Selector S, IdentifierInfo *ClsName,
                                       const ObjCInterfaceDecl *ID,
                                       const ObjCMethodDecl *MD,
                                       QualType RetTy);

  RetainSummary *getClassMethodSummary(const ObjCMessage &msg) {
    const ObjCInterfaceDecl *Class = 0;
    if (!msg.isInstanceMessage())
      Class = msg.getReceiverInterface();

    return getClassMethodSummary(msg.getSelector(),
                                 Class? Class->getIdentifier() : 0,
                                 Class,
                                 msg.getMethodDecl(), msg.getType(Ctx));
  }

  /// getMethodSummary - This version of getMethodSummary is used to query
  ///  the summary for the current method being analyzed.
  RetainSummary *getMethodSummary(const ObjCMethodDecl *MD) {
    // FIXME: Eventually this should be unneeded.
    const ObjCInterfaceDecl *ID = MD->getClassInterface();
    Selector S = MD->getSelector();
    IdentifierInfo *ClsName = ID->getIdentifier();
    QualType ResultTy = MD->getResultType();

    if (MD->isInstanceMethod())
      return getInstanceMethodSummary(S, ClsName, ID, MD, ResultTy);
    else
      return getClassMethodSummary(S, ClsName, ID, MD, ResultTy);
  }

  RetainSummary* getCommonMethodSummary(const ObjCMethodDecl *MD,
                                        Selector S, QualType RetTy);

  void updateSummaryFromAnnotations(RetainSummary *&Summ,
                                    const ObjCMethodDecl *MD);

  void updateSummaryFromAnnotations(RetainSummary *&Summ,
                                    const FunctionDecl *FD);

  bool isGCEnabled() const { return GCEnabled; }

  bool isARCEnabled() const { return ARCEnabled; }
  
  bool isARCorGCEnabled() const { return GCEnabled || ARCEnabled; }

  RetainSummary *copySummary(RetainSummary *OldSumm) {
    RetainSummary *Summ = (RetainSummary*) BPAlloc.Allocate<RetainSummary>();
    new (Summ) RetainSummary(*OldSumm);
    return Summ;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of checker data structures.
//===----------------------------------------------------------------------===//

RetainSummaryManager::~RetainSummaryManager() {}

ArgEffects RetainSummaryManager::getArgEffects() {
  ArgEffects AE = ScratchArgs;
  ScratchArgs = AF.getEmptyMap();
  return AE;
}

RetainSummary*
RetainSummaryManager::getPersistentSummary(ArgEffects AE, RetEffect RetEff,
                                           ArgEffect ReceiverEff,
                                           ArgEffect DefaultEff) {
  // Create the summary and return it.
  RetainSummary *Summ = (RetainSummary*) BPAlloc.Allocate<RetainSummary>();
  new (Summ) RetainSummary(AE, RetEff, DefaultEff, ReceiverEff);
  return Summ;
}

//===----------------------------------------------------------------------===//
// Summary creation for functions (largely uses of Core Foundation).
//===----------------------------------------------------------------------===//

static bool isRetain(const FunctionDecl *FD, StringRef FName) {
  return FName.endswith("Retain");
}

static bool isRelease(const FunctionDecl *FD, StringRef FName) {
  return FName.endswith("Release");
}

static bool isMakeCollectable(const FunctionDecl *FD, StringRef FName) {
  // FIXME: Remove FunctionDecl parameter.
  // FIXME: Is it really okay if MakeCollectable isn't a suffix?
  return FName.find("MakeCollectable") != StringRef::npos;
}

RetainSummary* RetainSummaryManager::getSummary(const FunctionDecl *FD) {
  // Look up a summary in our cache of FunctionDecls -> Summaries.
  FuncSummariesTy::iterator I = FuncSummaries.find(FD);
  if (I != FuncSummaries.end())
    return I->second;

  // No summary?  Generate one.
  RetainSummary *S = 0;

  do {
    // We generate "stop" summaries for implicitly defined functions.
    if (FD->isImplicit()) {
      S = getPersistentStopSummary();
      break;
    }
    // For C++ methods, generate an implicit "stop" summary as well.  We
    // can relax this once we have a clear policy for C++ methods and
    // ownership attributes.
    if (isa<CXXMethodDecl>(FD)) {
      S = getPersistentStopSummary();
      break;
    }

    // [PR 3337] Use 'getAs<FunctionType>' to strip away any typedefs on the
    // function's type.
    const FunctionType* FT = FD->getType()->getAs<FunctionType>();
    const IdentifierInfo *II = FD->getIdentifier();
    if (!II)
      break;

    StringRef FName = II->getName();

    // Strip away preceding '_'.  Doing this here will effect all the checks
    // down below.
    FName = FName.substr(FName.find_first_not_of('_'));

    // Inspect the result type.
    QualType RetTy = FT->getResultType();

    // FIXME: This should all be refactored into a chain of "summary lookup"
    //  filters.
    assert(ScratchArgs.isEmpty());

    if (FName == "pthread_create") {
      // Part of: <rdar://problem/7299394>.  This will be addressed
      // better with IPA.
      S = getPersistentStopSummary();
    } else if (FName == "NSMakeCollectable") {
      // Handle: id NSMakeCollectable(CFTypeRef)
      S = (RetTy->isObjCIdType())
          ? getUnarySummary(FT, cfmakecollectable)
          : getPersistentStopSummary();
    } else if (FName == "IOBSDNameMatching" ||
               FName == "IOServiceMatching" ||
               FName == "IOServiceNameMatching" ||
               FName == "IORegistryEntryIDMatching" ||
               FName == "IOOpenFirmwarePathMatching") {
      // Part of <rdar://problem/6961230>. (IOKit)
      // This should be addressed using a API table.
      S = getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true),
                               DoNothing, DoNothing);
    } else if (FName == "IOServiceGetMatchingService" ||
               FName == "IOServiceGetMatchingServices") {
      // FIXES: <rdar://problem/6326900>
      // This should be addressed using a API table.  This strcmp is also
      // a little gross, but there is no need to super optimize here.
      ScratchArgs = AF.add(ScratchArgs, 1, DecRef);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "IOServiceAddNotification" ||
               FName == "IOServiceAddMatchingNotification") {
      // Part of <rdar://problem/6961230>. (IOKit)
      // This should be addressed using a API table.
      ScratchArgs = AF.add(ScratchArgs, 2, DecRef);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "CVPixelBufferCreateWithBytes") {
      // FIXES: <rdar://problem/7283567>
      // Eventually this can be improved by recognizing that the pixel
      // buffer passed to CVPixelBufferCreateWithBytes is released via
      // a callback and doing full IPA to make sure this is done correctly.
      // FIXME: This function has an out parameter that returns an
      // allocated object.
      ScratchArgs = AF.add(ScratchArgs, 7, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "CGBitmapContextCreateWithData") {
      // FIXES: <rdar://problem/7358899>
      // Eventually this can be improved by recognizing that 'releaseInfo'
      // passed to CGBitmapContextCreateWithData is released via
      // a callback and doing full IPA to make sure this is done correctly.
      ScratchArgs = AF.add(ScratchArgs, 8, StopTracking);
      S = getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true),
                               DoNothing, DoNothing);
    } else if (FName == "CVPixelBufferCreateWithPlanarBytes") {
      // FIXES: <rdar://problem/7283567>
      // Eventually this can be improved by recognizing that the pixel
      // buffer passed to CVPixelBufferCreateWithPlanarBytes is released
      // via a callback and doing full IPA to make sure this is done
      // correctly.
      ScratchArgs = AF.add(ScratchArgs, 12, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    }

    // Did we get a summary?
    if (S)
      break;

    // Enable this code once the semantics of NSDeallocateObject are resolved
    // for GC.  <rdar://problem/6619988>
#if 0
    // Handle: NSDeallocateObject(id anObject);
    // This method does allow 'nil' (although we don't check it now).
    if (strcmp(FName, "NSDeallocateObject") == 0) {
      return RetTy == Ctx.VoidTy
        ? getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, Dealloc)
        : getPersistentStopSummary();
    }
#endif

    if (RetTy->isPointerType()) {
      // For CoreFoundation ('CF') types.
      if (cocoa::isRefType(RetTy, "CF", FName)) {
        if (isRetain(FD, FName))
          S = getUnarySummary(FT, cfretain);
        else if (isMakeCollectable(FD, FName))
          S = getUnarySummary(FT, cfmakecollectable);
        else
          S = getCFCreateGetRuleSummary(FD, FName);

        break;
      }

      // For CoreGraphics ('CG') types.
      if (cocoa::isRefType(RetTy, "CG", FName)) {
        if (isRetain(FD, FName))
          S = getUnarySummary(FT, cfretain);
        else
          S = getCFCreateGetRuleSummary(FD, FName);

        break;
      }

      // For the Disk Arbitration API (DiskArbitration/DADisk.h)
      if (cocoa::isRefType(RetTy, "DADisk") ||
          cocoa::isRefType(RetTy, "DADissenter") ||
          cocoa::isRefType(RetTy, "DASessionRef")) {
        S = getCFCreateGetRuleSummary(FD, FName);
        break;
      }

      break;
    }

    // Check for release functions, the only kind of functions that we care
    // about that don't return a pointer type.
    if (FName[0] == 'C' && (FName[1] == 'F' || FName[1] == 'G')) {
      // Test for 'CGCF'.
      FName = FName.substr(FName.startswith("CGCF") ? 4 : 2);

      if (isRelease(FD, FName))
        S = getUnarySummary(FT, cfrelease);
      else {
        assert (ScratchArgs.isEmpty());
        // Remaining CoreFoundation and CoreGraphics functions.
        // We use to assume that they all strictly followed the ownership idiom
        // and that ownership cannot be transferred.  While this is technically
        // correct, many methods allow a tracked object to escape.  For example:
        //
        //   CFMutableDictionaryRef x = CFDictionaryCreateMutable(...);
        //   CFDictionaryAddValue(y, key, x);
        //   CFRelease(x);
        //   ... it is okay to use 'x' since 'y' has a reference to it
        //
        // We handle this and similar cases with the follow heuristic.  If the
        // function name contains "InsertValue", "SetValue", "AddValue",
        // "AppendValue", or "SetAttribute", then we assume that arguments may
        // "escape."  This means that something else holds on to the object,
        // allowing it be used even after its local retain count drops to 0.
        ArgEffect E = (StrInStrNoCase(FName, "InsertValue") != StringRef::npos||
                       StrInStrNoCase(FName, "AddValue") != StringRef::npos ||
                       StrInStrNoCase(FName, "SetValue") != StringRef::npos ||
                       StrInStrNoCase(FName, "AppendValue") != StringRef::npos||
                       StrInStrNoCase(FName, "SetAttribute") != StringRef::npos)
                      ? MayEscape : DoNothing;

        S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, E);
      }
    }
  }
  while (0);

  // Annotations override defaults.
  updateSummaryFromAnnotations(S, FD);

  FuncSummaries[FD] = S;
  return S;
}

RetainSummary*
RetainSummaryManager::getCFCreateGetRuleSummary(const FunctionDecl *FD,
                                                StringRef FName) {
  if (coreFoundation::followsCreateRule(FName))
    return getCFSummaryCreateRule(FD);

  return getCFSummaryGetRule(FD);
}

RetainSummary*
RetainSummaryManager::getUnarySummary(const FunctionType* FT,
                                      UnaryFuncKind func) {

  // Sanity check that this is *really* a unary function.  This can
  // happen if people do weird things.
  const FunctionProtoType* FTP = dyn_cast<FunctionProtoType>(FT);
  if (!FTP || FTP->getNumArgs() != 1)
    return getPersistentStopSummary();

  assert (ScratchArgs.isEmpty());

  ArgEffect Effect;
  switch (func) {
    case cfretain: Effect = IncRef; break;
    case cfrelease: Effect = DecRef; break;
    case cfmakecollectable: Effect = MakeCollectable; break;
    default: llvm_unreachable("Not a supported unary function.");
  }

  ScratchArgs = AF.add(ScratchArgs, 0, Effect);
  return getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
}

RetainSummary* 
RetainSummaryManager::getCFSummaryCreateRule(const FunctionDecl *FD) {
  assert (ScratchArgs.isEmpty());

  if (FD->getIdentifier() == CFDictionaryCreateII) {
    ScratchArgs = AF.add(ScratchArgs, 1, DoNothingByRef);
    ScratchArgs = AF.add(ScratchArgs, 2, DoNothingByRef);
  }

  return getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true));
}

RetainSummary* 
RetainSummaryManager::getCFSummaryGetRule(const FunctionDecl *FD) {
  assert (ScratchArgs.isEmpty());
  return getPersistentSummary(RetEffect::MakeNotOwned(RetEffect::CF),
                              DoNothing, DoNothing);
}

//===----------------------------------------------------------------------===//
// Summary creation for Selectors.
//===----------------------------------------------------------------------===//

RetainSummary*
RetainSummaryManager::getInitMethodSummary(QualType RetTy) {
  assert(ScratchArgs.isEmpty());
  // 'init' methods conceptually return a newly allocated object and claim
  // the receiver.
  if (cocoa::isCocoaObjectRef(RetTy) ||
      coreFoundation::isCFObjectRef(RetTy))
    return getPersistentSummary(ObjCInitRetE, DecRefMsg);

  return 0;
}

void
RetainSummaryManager::updateSummaryFromAnnotations(RetainSummary *&Summ,
                                                   const FunctionDecl *FD) {
  if (!FD)
    return;

  RetainSummary BasedOnDefault(DefaultSummary);
  if (!Summ)
    Summ = &BasedOnDefault;

  // Effects on the parameters.
  unsigned parm_idx = 0;
  for (FunctionDecl::param_const_iterator pi = FD->param_begin(), 
         pe = FD->param_end(); pi != pe; ++pi, ++parm_idx) {
    const ParmVarDecl *pd = *pi;
    if (pd->getAttr<NSConsumedAttr>()) {
      if (!GCEnabled) {
        Summ->addArg(AF, parm_idx, DecRef);      
      }
    } else if (pd->getAttr<CFConsumedAttr>()) {
      Summ->addArg(AF, parm_idx, DecRef);      
    }   
  }
  
  QualType RetTy = FD->getResultType();

  // Determine if there is a special return effect for this method.
  if (cocoa::isCocoaObjectRef(RetTy)) {
    if (FD->getAttr<NSReturnsRetainedAttr>()) {
      Summ->setRetEffect(ObjCAllocRetE);
    }
    else if (FD->getAttr<CFReturnsRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    }
    else if (FD->getAttr<NSReturnsNotRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeNotOwned(RetEffect::ObjC));
    }
    else if (FD->getAttr<CFReturnsNotRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
    }
  } else if (RetTy->getAs<PointerType>()) {
    if (FD->getAttr<CFReturnsRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    }
    else if (FD->getAttr<CFReturnsNotRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
    }
  }

  if (Summ == &BasedOnDefault) {
    if (!(*Summ == DefaultSummary))
      Summ = copySummary(Summ);
    else
      Summ = 0;
  }
}

void
RetainSummaryManager::updateSummaryFromAnnotations(RetainSummary *&Summ,
                                                  const ObjCMethodDecl *MD) {
  if (!MD)
    return;

  RetainSummary BasedOnDefault = DefaultSummary;
  if (!Summ)
    Summ = &BasedOnDefault;

  bool isTrackedLoc = false;

  // Effects on the receiver.
  if (MD->getAttr<NSConsumesSelfAttr>()) {
    if (!GCEnabled)
      Summ->setReceiverEffect(DecRefMsg);      
  }
  
  // Effects on the parameters.
  unsigned parm_idx = 0;
  for (ObjCMethodDecl::param_iterator pi=MD->param_begin(), pe=MD->param_end();
       pi != pe; ++pi, ++parm_idx) {
    const ParmVarDecl *pd = *pi;
    if (pd->getAttr<NSConsumedAttr>()) {
      if (!GCEnabled)
        Summ->addArg(AF, parm_idx, DecRef);      
    }
    else if(pd->getAttr<CFConsumedAttr>()) {
      Summ->addArg(AF, parm_idx, DecRef);      
    }   
  }
  
  // Determine if there is a special return effect for this method.
  if (cocoa::isCocoaObjectRef(MD->getResultType())) {
    if (MD->getAttr<NSReturnsRetainedAttr>()) {
      Summ->setRetEffect(ObjCAllocRetE);
      return;
    }
    if (MD->getAttr<NSReturnsNotRetainedAttr>()) {
      Summ->setRetEffect(RetEffect::MakeNotOwned(RetEffect::ObjC));
      return;
    }

    isTrackedLoc = true;
  }

  if (!isTrackedLoc)
    isTrackedLoc = MD->getResultType()->getAs<PointerType>() != NULL;

  if (isTrackedLoc) {
    if (MD->getAttr<CFReturnsRetainedAttr>())
      Summ->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    else if (MD->getAttr<CFReturnsNotRetainedAttr>())
      Summ->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
  }

  if (Summ == &BasedOnDefault) {
    if (!(*Summ == DefaultSummary))
      Summ = copySummary(Summ);
    else
      Summ = 0;
  }
}

RetainSummary*
RetainSummaryManager::getCommonMethodSummary(const ObjCMethodDecl *MD,
                                             Selector S, QualType RetTy) {

  if (MD) {
    // Scan the method decl for 'void*' arguments.  These should be treated
    // as 'StopTracking' because they are often used with delegates.
    // Delegates are a frequent form of false positives with the retain
    // count checker.
    unsigned i = 0;
    for (ObjCMethodDecl::param_iterator I = MD->param_begin(),
         E = MD->param_end(); I != E; ++I, ++i)
      if (ParmVarDecl *PD = *I) {
        QualType Ty = Ctx.getCanonicalType(PD->getType());
        if (Ty.getLocalUnqualifiedType() == Ctx.VoidPtrTy)
          ScratchArgs = AF.add(ScratchArgs, i, StopTracking);
      }
  }

  // Any special effect for the receiver?
  ArgEffect ReceiverEff = DoNothing;

  // If one of the arguments in the selector has the keyword 'delegate' we
  // should stop tracking the reference count for the receiver.  This is
  // because the reference count is quite possibly handled by a delegate
  // method.
  if (S.isKeywordSelector()) {
    const std::string &str = S.getAsString();
    assert(!str.empty());
    if (StrInStrNoCase(str, "delegate:") != StringRef::npos)
      ReceiverEff = StopTracking;
  }

  // Look for methods that return an owned object.
  if (cocoa::isCocoaObjectRef(RetTy)) {
    // EXPERIMENTAL: assume the Cocoa conventions for all objects returned
    //  by instance methods.
    RetEffect E = cocoa::followsFundamentalRule(S, MD)
                  ? ObjCAllocRetE : RetEffect::MakeNotOwned(RetEffect::ObjC);

    return getPersistentSummary(E, ReceiverEff, MayEscape);
  }

  // Look for methods that return an owned core foundation object.
  if (coreFoundation::isCFObjectRef(RetTy)) {
    RetEffect E = cocoa::followsFundamentalRule(S, MD)
      ? RetEffect::MakeOwned(RetEffect::CF, true)
      : RetEffect::MakeNotOwned(RetEffect::CF);

    return getPersistentSummary(E, ReceiverEff, MayEscape);
  }

  if (ScratchArgs.isEmpty() && ReceiverEff == DoNothing)
    return 0;

  return getPersistentSummary(RetEffect::MakeNoRet(), ReceiverEff, MayEscape);
}

RetainSummary*
RetainSummaryManager::getInstanceMethodSummary(const ObjCMessage &msg,
                                               const ProgramState *state,
                                               const LocationContext *LC) {

  // We need the type-information of the tracked receiver object
  // Retrieve it from the state.
  const Expr *Receiver = msg.getInstanceReceiver();
  const ObjCInterfaceDecl *ID = 0;

  // FIXME: Is this really working as expected?  There are cases where
  //  we just use the 'ID' from the message expression.
  SVal receiverV;

  if (Receiver) {
    receiverV = state->getSValAsScalarOrLoc(Receiver);

    // FIXME: Eventually replace the use of state->get<RefBindings> with
    // a generic API for reasoning about the Objective-C types of symbolic
    // objects.
    if (SymbolRef Sym = receiverV.getAsLocSymbol())
      if (const RefVal *T = state->get<RefBindings>(Sym))
        if (const ObjCObjectPointerType* PT =
            T->getType()->getAs<ObjCObjectPointerType>())
          ID = PT->getInterfaceDecl();

    // FIXME: this is a hack.  This may or may not be the actual method
    //  that is called.
    if (!ID) {
      if (const ObjCObjectPointerType *PT =
          Receiver->getType()->getAs<ObjCObjectPointerType>())
        ID = PT->getInterfaceDecl();
    }
  } else {
    // FIXME: Hack for 'super'.
    ID = msg.getReceiverInterface();
  }

  // FIXME: The receiver could be a reference to a class, meaning that
  //  we should use the class method.
  return getInstanceMethodSummary(msg, ID);
}

RetainSummary*
RetainSummaryManager::getInstanceMethodSummary(Selector S,
                                               IdentifierInfo *ClsName,
                                               const ObjCInterfaceDecl *ID,
                                               const ObjCMethodDecl *MD,
                                               QualType RetTy) {

  // Look up a summary in our summary cache.
  RetainSummary *Summ = ObjCMethodSummaries.find(ID, ClsName, S);

  if (!Summ) {
    assert(ScratchArgs.isEmpty());

    // "initXXX": pass-through for receiver.
    if (cocoa::deriveNamingConvention(S, MD) == cocoa::InitRule)
      Summ = getInitMethodSummary(RetTy);
    else
      Summ = getCommonMethodSummary(MD, S, RetTy);

    // Annotations override defaults.
    updateSummaryFromAnnotations(Summ, MD);

    // Memoize the summary.
    ObjCMethodSummaries[ObjCSummaryKey(ID, ClsName, S)] = Summ;
  }

  return Summ;
}

RetainSummary*
RetainSummaryManager::getClassMethodSummary(Selector S, IdentifierInfo *ClsName,
                                            const ObjCInterfaceDecl *ID,
                                            const ObjCMethodDecl *MD,
                                            QualType RetTy) {

  assert(ClsName && "Class name must be specified.");
  RetainSummary *Summ = ObjCClassMethodSummaries.find(ID, ClsName, S);

  if (!Summ) {
    Summ = getCommonMethodSummary(MD, S, RetTy);

    // Annotations override defaults.
    updateSummaryFromAnnotations(Summ, MD);

    // Memoize the summary.
    ObjCClassMethodSummaries[ObjCSummaryKey(ID, ClsName, S)] = Summ;
  }

  return Summ;
}

void RetainSummaryManager::InitializeClassMethodSummaries() {
  assert(ScratchArgs.isEmpty());
  // Create the [NSAssertionHandler currentHander] summary.
  addClassMethSummary("NSAssertionHandler", "currentHandler",
                getPersistentSummary(RetEffect::MakeNotOwned(RetEffect::ObjC)));

  // Create the [NSAutoreleasePool addObject:] summary.
  ScratchArgs = AF.add(ScratchArgs, 0, Autorelease);
  addClassMethSummary("NSAutoreleasePool", "addObject",
                      getPersistentSummary(RetEffect::MakeNoRet(),
                                           DoNothing, Autorelease));

  // Create the summaries for [NSObject performSelector...].  We treat
  // these as 'stop tracking' for the arguments because they are often
  // used for delegates that can release the object.  When we have better
  // inter-procedural analysis we can potentially do something better.  This
  // workaround is to remove false positives.
  RetainSummary *Summ =
    getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, StopTracking);
  IdentifierInfo *NSObjectII = &Ctx.Idents.get("NSObject");
  addClsMethSummary(NSObjectII, Summ, "performSelector", "withObject",
                    "afterDelay", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelector", "withObject",
                    "afterDelay", "inModes", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelectorOnMainThread",
                    "withObject", "waitUntilDone", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelectorOnMainThread",
                    "withObject", "waitUntilDone", "modes", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelector", "onThread",
                    "withObject", "waitUntilDone", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelector", "onThread",
                    "withObject", "waitUntilDone", "modes", NULL);
  addClsMethSummary(NSObjectII, Summ, "performSelectorInBackground",
                    "withObject", NULL);
}

void RetainSummaryManager::InitializeMethodSummaries() {

  assert (ScratchArgs.isEmpty());

  // Create the "init" selector.  It just acts as a pass-through for the
  // receiver.
  RetainSummary *InitSumm = getPersistentSummary(ObjCInitRetE, DecRefMsg);
  addNSObjectMethSummary(GetNullarySelector("init", Ctx), InitSumm);

  // awakeAfterUsingCoder: behaves basically like an 'init' method.  It
  // claims the receiver and returns a retained object.
  addNSObjectMethSummary(GetUnarySelector("awakeAfterUsingCoder", Ctx),
                         InitSumm);

  // The next methods are allocators.
  RetainSummary *AllocSumm = getPersistentSummary(ObjCAllocRetE);
  RetainSummary *CFAllocSumm =
    getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true));

  // Create the "retain" selector.
  RetEffect NoRet = RetEffect::MakeNoRet();
  RetainSummary *Summ = getPersistentSummary(NoRet, IncRefMsg);
  addNSObjectMethSummary(GetNullarySelector("retain", Ctx), Summ);

  // Create the "release" selector.
  Summ = getPersistentSummary(NoRet, DecRefMsg);
  addNSObjectMethSummary(GetNullarySelector("release", Ctx), Summ);

  // Create the "drain" selector.
  Summ = getPersistentSummary(NoRet, isGCEnabled() ? DoNothing : DecRef);
  addNSObjectMethSummary(GetNullarySelector("drain", Ctx), Summ);

  // Create the -dealloc summary.
  Summ = getPersistentSummary(NoRet, Dealloc);
  addNSObjectMethSummary(GetNullarySelector("dealloc", Ctx), Summ);

  // Create the "autorelease" selector.
  Summ = getPersistentSummary(NoRet, Autorelease);
  addNSObjectMethSummary(GetNullarySelector("autorelease", Ctx), Summ);

  // Specially handle NSAutoreleasePool.
  addInstMethSummary("NSAutoreleasePool", "init",
                     getPersistentSummary(NoRet, NewAutoreleasePool));

  // For NSWindow, allocated objects are (initially) self-owned.
  // FIXME: For now we opt for false negatives with NSWindow, as these objects
  //  self-own themselves.  However, they only do this once they are displayed.
  //  Thus, we need to track an NSWindow's display status.
  //  This is tracked in <rdar://problem/6062711>.
  //  See also http://llvm.org/bugs/show_bug.cgi?id=3714.
  RetainSummary *NoTrackYet = getPersistentSummary(RetEffect::MakeNoRet(),
                                                   StopTracking,
                                                   StopTracking);

  addClassMethSummary("NSWindow", "alloc", NoTrackYet);

#if 0
  addInstMethSummary("NSWindow", NoTrackYet, "initWithContentRect",
                     "styleMask", "backing", "defer", NULL);

  addInstMethSummary("NSWindow", NoTrackYet, "initWithContentRect",
                     "styleMask", "backing", "defer", "screen", NULL);
#endif

  // For NSPanel (which subclasses NSWindow), allocated objects are not
  //  self-owned.
  // FIXME: For now we don't track NSPanels. object for the same reason
  //   as for NSWindow objects.
  addClassMethSummary("NSPanel", "alloc", NoTrackYet);

#if 0
  addInstMethSummary("NSPanel", NoTrackYet, "initWithContentRect",
                     "styleMask", "backing", "defer", NULL);

  addInstMethSummary("NSPanel", NoTrackYet, "initWithContentRect",
                     "styleMask", "backing", "defer", "screen", NULL);
#endif

  // Don't track allocated autorelease pools yet, as it is okay to prematurely
  // exit a method.
  addClassMethSummary("NSAutoreleasePool", "alloc", NoTrackYet);

  // Create summaries QCRenderer/QCView -createSnapShotImageOfType:
  addInstMethSummary("QCRenderer", AllocSumm,
                     "createSnapshotImageOfType", NULL);
  addInstMethSummary("QCView", AllocSumm,
                     "createSnapshotImageOfType", NULL);

  // Create summaries for CIContext, 'createCGImage' and
  // 'createCGLayerWithSize'.  These objects are CF objects, and are not
  // automatically garbage collected.
  addInstMethSummary("CIContext", CFAllocSumm,
                     "createCGImage", "fromRect", NULL);
  addInstMethSummary("CIContext", CFAllocSumm,
                     "createCGImage", "fromRect", "format", "colorSpace", NULL);
  addInstMethSummary("CIContext", CFAllocSumm, "createCGLayerWithSize",
           "info", NULL);
}

//===----------------------------------------------------------------------===//
// AutoreleaseBindings - State used to track objects in autorelease pools.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<SymbolRef, unsigned> ARCounts;
typedef llvm::ImmutableMap<SymbolRef, ARCounts> ARPoolContents;
typedef llvm::ImmutableList<SymbolRef> ARStack;

static int AutoRCIndex = 0;
static int AutoRBIndex = 0;

namespace { class AutoreleasePoolContents {}; }
namespace { class AutoreleaseStack {}; }

namespace clang {
namespace ento {
template<> struct ProgramStateTrait<AutoreleaseStack>
  : public ProgramStatePartialTrait<ARStack> {
  static inline void *GDMIndex() { return &AutoRBIndex; }
};

template<> struct ProgramStateTrait<AutoreleasePoolContents>
  : public ProgramStatePartialTrait<ARPoolContents> {
  static inline void *GDMIndex() { return &AutoRCIndex; }
};
} // end GR namespace
} // end clang namespace

static SymbolRef GetCurrentAutoreleasePool(const ProgramState *state) {
  ARStack stack = state->get<AutoreleaseStack>();
  return stack.isEmpty() ? SymbolRef() : stack.getHead();
}

static const ProgramState *
SendAutorelease(const ProgramState *state,
                ARCounts::Factory &F,
                SymbolRef sym) {
  SymbolRef pool = GetCurrentAutoreleasePool(state);
  const ARCounts *cnts = state->get<AutoreleasePoolContents>(pool);
  ARCounts newCnts(0);

  if (cnts) {
    const unsigned *cnt = (*cnts).lookup(sym);
    newCnts = F.add(*cnts, sym, cnt ? *cnt  + 1 : 1);
  }
  else
    newCnts = F.add(F.getEmptyMap(), sym, 1);

  return state->set<AutoreleasePoolContents>(pool, newCnts);
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

namespace {

class CFRefCount : public TransferFuncs {
public:
  class BindingsPrinter : public ProgramState::Printer {
  public:
    virtual void Print(raw_ostream &Out,
                       const ProgramState *state,
                       const char* nl,
                       const char* sep);
  };

  typedef llvm::DenseMap<const ExplodedNode*, const RetainSummary*>
    SummaryLogTy;

  RetainSummaryManager Summaries;
  SummaryLogTy SummaryLog;
  const LangOptions&   LOpts;
  ARCounts::Factory    ARCountFactory;

  BugType *useAfterRelease, *releaseNotOwned;
  BugType *deallocGC, *deallocNotOwned;
  BugType *leakWithinFunction, *leakAtReturn;
  BugType *overAutorelease;
  BugType *returnNotOwnedForOwned;
  BugReporter *BR;

  const ProgramState *Update(const ProgramState * state,
                             SymbolRef sym,
                             RefVal V,
                             ArgEffect E,
                             RefVal::Kind& hasErr);

public:
  CFRefCount(ASTContext &Ctx, bool gcenabled, const LangOptions& lopts)
    : Summaries(Ctx, gcenabled, (bool)lopts.ObjCAutoRefCount),
      LOpts(lopts), useAfterRelease(0), releaseNotOwned(0),
      deallocGC(0), deallocNotOwned(0),
      leakWithinFunction(0), leakAtReturn(0), overAutorelease(0),
      returnNotOwnedForOwned(0), BR(0) {}

  void RegisterChecks(ExprEngine &Eng);

  virtual void RegisterPrinters(std::vector<ProgramState::Printer*>& Printers) {
    Printers.push_back(new BindingsPrinter());
  }

  bool isGCEnabled() const { return Summaries.isGCEnabled(); }
  bool isARCorGCEnabled() const { return Summaries.isARCorGCEnabled(); }
  
  const LangOptions& getLangOptions() const { return LOpts; }

  const RetainSummary *getSummaryOfNode(const ExplodedNode *N) const {
    SummaryLogTy::const_iterator I = SummaryLog.find(N);
    return I == SummaryLog.end() ? 0 : I->second;
  }

  // Calls.

  void evalCallOrMessage(ExplodedNodeSet &Dst, ExprEngine &Eng,
                         StmtNodeBuilder &Builder,
                         const CallOrObjCMessage &callOrMsg,
                         InstanceReceiver Receiver, const MemRegion *Callee,
                         ExplodedNode *Pred, const ProgramState *state);

  virtual void evalCall(ExplodedNodeSet &Dst,
                        ExprEngine& Eng,
                        StmtNodeBuilder& Builder,
                        const CallExpr *CE, SVal L,
                        ExplodedNode *Pred);


  virtual void evalObjCMessage(ExplodedNodeSet &Dst,
                               ExprEngine& Engine,
                               StmtNodeBuilder& Builder,
                               ObjCMessage msg,
                               ExplodedNode *Pred,
                               const ProgramState *state);

  // End-of-path.

  std::pair<ExplodedNode*, const ProgramState *>
  HandleAutoreleaseCounts(const ProgramState * state,
                          GenericNodeBuilderRefCount Bd,
                          ExplodedNode *Pred,
                          ExprEngine &Eng,
                          SymbolRef Sym,
                          RefVal V,
                          bool &stop);
};

} // end anonymous namespace

static void PrintPool(raw_ostream &Out,
                      SymbolRef Sym,
                      const ProgramState *state) {
  Out << ' ';
  if (Sym)
    Out << Sym->getSymbolID();
  else
    Out << "<pool>";
  Out << ":{";

  // Get the contents of the pool.
  if (const ARCounts *cnts = state->get<AutoreleasePoolContents>(Sym))
    for (ARCounts::iterator J=cnts->begin(), EJ=cnts->end(); J != EJ; ++J)
      Out << '(' << J.getKey() << ',' << J.getData() << ')';

  Out << '}';
}

void CFRefCount::BindingsPrinter::Print(raw_ostream &Out,
                                        const ProgramState *state,
                                        const char* nl, const char* sep) {

  RefBindings B = state->get<RefBindings>();

  if (!B.isEmpty())
    Out << sep << nl;

  for (RefBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {
    Out << (*I).first << " : ";
    (*I).second.print(Out);
    Out << nl;
  }

  // Print the autorelease stack.
  Out << sep << nl << "AR pool stack:";
  ARStack stack = state->get<AutoreleaseStack>();

  PrintPool(Out, SymbolRef(), state);  // Print the caller's pool.
  for (ARStack::iterator I=stack.begin(), E=stack.end(); I!=E; ++I)
    PrintPool(Out, *I, state);

  Out << nl;
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//
static void addExtraTextToCFReport(BugReport &R);

namespace {
  //===-------------===//
  // Bug Descriptions. //
  //===-------------===//

  class CFRefBug : public BugType {
  protected:
    CFRefCount& TF;

    CFRefBug(CFRefCount* tf, StringRef name)
    : BugType(name, "Memory (Core Foundation/Objective-C)"), TF(*tf) {}
  public:

    CFRefCount& getTF() { return TF; }

    // FIXME: Eventually remove.
    virtual const char* getDescription() const = 0;

    virtual bool isLeak() const { return false; }
  };

  class UseAfterRelease : public CFRefBug {
  public:
    UseAfterRelease(CFRefCount* tf)
    : CFRefBug(tf, "Use-after-release") {}

    const char* getDescription() const {
      return "Reference-counted object is used after it is released";
    }
  };

  class BadRelease : public CFRefBug {
  public:
    BadRelease(CFRefCount* tf) : CFRefBug(tf, "Bad release") {}

    const char* getDescription() const {
      return "Incorrect decrement of the reference count of an object that is "
             "not owned at this point by the caller";
    }
  };

  class DeallocGC : public CFRefBug {
  public:
    DeallocGC(CFRefCount *tf)
      : CFRefBug(tf, "-dealloc called while using garbage collection") {}

    const char *getDescription() const {
      return "-dealloc called while using garbage collection";
    }
  };

  class DeallocNotOwned : public CFRefBug {
  public:
    DeallocNotOwned(CFRefCount *tf)
      : CFRefBug(tf, "-dealloc sent to non-exclusively owned object") {}

    const char *getDescription() const {
      return "-dealloc sent to object that may be referenced elsewhere";
    }
  };

  class OverAutorelease : public CFRefBug {
  public:
    OverAutorelease(CFRefCount *tf) :
      CFRefBug(tf, "Object sent -autorelease too many times") {}

    const char *getDescription() const {
      return "Object sent -autorelease too many times";
    }
  };

  class ReturnedNotOwnedForOwned : public CFRefBug {
  public:
    ReturnedNotOwnedForOwned(CFRefCount *tf) :
      CFRefBug(tf, "Method should return an owned object") {}

    const char *getDescription() const {
      return "Object with a +0 retain count returned to caller where a +1 "
             "(owning) retain count is expected";
    }
  };

  class Leak : public CFRefBug {
    const bool isReturn;
  protected:
    Leak(CFRefCount* tf, StringRef name, bool isRet)
    : CFRefBug(tf, name), isReturn(isRet) {}
  public:

    const char* getDescription() const { return ""; }

    bool isLeak() const { return true; }
  };

  class LeakAtReturn : public Leak {
  public:
    LeakAtReturn(CFRefCount* tf, StringRef name)
    : Leak(tf, name, true) {}
  };

  class LeakWithinFunction : public Leak {
  public:
    LeakWithinFunction(CFRefCount* tf, StringRef name)
    : Leak(tf, name, false) {}
  };

  //===---------===//
  // Bug Reports.  //
  //===---------===//

  class CFRefReportVisitor : public BugReporterVisitor {
  protected:
    SymbolRef Sym;
    const CFRefCount &TF;
    
  public:
    CFRefReportVisitor(SymbolRef sym, const CFRefCount &tf)
       : Sym(sym), TF(tf) {}

    virtual void Profile(llvm::FoldingSetNodeID &ID) const {
      static int x = 0;
      ID.AddPointer(&x);
      ID.AddPointer(Sym);
    }

    virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                           const ExplodedNode *PrevN,
                                           BugReporterContext &BRC,
                                           BugReport &BR);

    virtual PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                            const ExplodedNode *N,
                                            BugReport &BR);
  };

  class CFRefLeakReportVisitor : public CFRefReportVisitor {
  public:
    CFRefLeakReportVisitor(SymbolRef sym, const CFRefCount &tf)
       : CFRefReportVisitor(sym, tf) {}

    PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                    const ExplodedNode *N,
                                    BugReport &BR);
  };

  class CFRefReport : public BugReport {
  public:
    CFRefReport(CFRefBug& D, const CFRefCount &tf,
                ExplodedNode *n, SymbolRef sym, bool registerVisitor = true)
      : BugReport(D, D.getDescription(), n) {
      if (registerVisitor)
        addVisitor(new CFRefReportVisitor(sym, tf));
      addExtraTextToCFReport(*this);
    }

    CFRefReport(CFRefBug& D, const CFRefCount &tf,
                ExplodedNode *n, SymbolRef sym, StringRef endText)
      : BugReport(D, D.getDescription(), endText, n) {
      addVisitor(new CFRefReportVisitor(sym, tf));
      addExtraTextToCFReport(*this);
    }

    virtual ~CFRefReport() {}

    virtual std::pair<ranges_iterator, ranges_iterator> getRanges() {
      const CFRefBug& BugTy = static_cast<CFRefBug&>(getBugType());
      if (!BugTy.isLeak())
        return BugReport::getRanges();
      else
        return std::make_pair(ranges_iterator(), ranges_iterator());
    }
  };

  class CFRefLeakReport : public CFRefReport {
    SourceLocation AllocSite;
    const MemRegion* AllocBinding;

  public:
    CFRefLeakReport(CFRefBug& D, const CFRefCount &tf,
                    ExplodedNode *n, SymbolRef sym,
                    ExprEngine& Eng);

    SourceLocation getLocation() const { return AllocSite; }
  };
} // end anonymous namespace

static const char* Msgs[] = {
  // GC only
  "Code is compiled to only use garbage collection",
  // No GC.
  "Code is compiled to use reference counts",
  // Hybrid, with GC.
  "Code is compiled to use either garbage collection (GC) or reference counts"
  " (non-GC).  The bug occurs with GC enabled",
  // Hybrid, without GC
  "Code is compiled to use either garbage collection (GC) or reference counts"
  " (non-GC).  The bug occurs in non-GC mode"
};

// Add the metadata text.
static void addExtraTextToCFReport(BugReport &R) {
  CFRefCount& TF = static_cast<CFRefBug&>(R.getBugType()).getTF();

  switch (TF.getLangOptions().getGCMode()) {
  default:
    assert(false);

  case LangOptions::GCOnly:
    assert (TF.isGCEnabled());
    R.addExtraText(Msgs[0]);
    return;

  case LangOptions::NonGC:
    assert (!TF.isGCEnabled());
    R.addExtraText(Msgs[1]);
    return;

  case LangOptions::HybridGC:
    if (TF.isGCEnabled()) {
      R.addExtraText(Msgs[2]);
      return;
    }
    else {
      R.addExtraText(Msgs[3]);
      return;
    }
  }
}

static inline bool contains(const SmallVectorImpl<ArgEffect>& V,
                            ArgEffect X) {
  for (SmallVectorImpl<ArgEffect>::const_iterator I=V.begin(), E=V.end();
       I!=E; ++I)
    if (*I == X) return true;

  return false;
}

PathDiagnosticPiece *CFRefReportVisitor::VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *PrevN,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) {

  if (!isa<StmtPoint>(N->getLocation()))
    return NULL;

  // Check if the type state has changed.
  const ProgramState *PrevSt = PrevN->getState();
  const ProgramState *CurrSt = N->getState();

  const RefVal* CurrT = CurrSt->get<RefBindings>(Sym);
  if (!CurrT) return NULL;

  const RefVal &CurrV = *CurrT;
  const RefVal *PrevT = PrevSt->get<RefBindings>(Sym);

  // Create a string buffer to constain all the useful things we want
  // to tell the user.
  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  // This is the allocation site since the previous node had no bindings
  // for this symbol.
  if (!PrevT) {
    const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();

    if (const CallExpr *CE = dyn_cast<CallExpr>(S)) {
      // Get the name of the callee (if it is available).
      SVal X = CurrSt->getSValAsScalarOrLoc(CE->getCallee());
      if (const FunctionDecl *FD = X.getAsFunctionDecl())
        os << "Call to function '" << FD << '\'';
      else
        os << "function call";
    }
    else if (isa<ObjCMessageExpr>(S)) {
      os << "Method";
    } else {
      os << "Property";
    }

    if (CurrV.getObjKind() == RetEffect::CF) {
      os << " returns a Core Foundation object with a ";
    }
    else {
      assert (CurrV.getObjKind() == RetEffect::ObjC);
      os << " returns an Objective-C object with a ";
    }

    if (CurrV.isOwned()) {
      os << "+1 retain count";

      if (TF.isGCEnabled()) {
        assert(CurrV.getObjKind() == RetEffect::CF);
        os << ".  "
        "Core Foundation objects are not automatically garbage collected.";
      }
    }
    else {
      assert (CurrV.isNotOwned());
      os << "+0 retain count";
    }

    PathDiagnosticLocation Pos(S, BRC.getSourceManager());
    return new PathDiagnosticEventPiece(Pos, os.str());
  }

  // Gather up the effects that were performed on the object at this
  // program point
  SmallVector<ArgEffect, 2> AEffects;

  if (const RetainSummary *Summ =
        TF.getSummaryOfNode(BRC.getNodeResolver().getOriginalNode(N))) {
    // We only have summaries attached to nodes after evaluating CallExpr and
    // ObjCMessageExprs.
    const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();

    if (const CallExpr *CE = dyn_cast<CallExpr>(S)) {
      // Iterate through the parameter expressions and see if the symbol
      // was ever passed as an argument.
      unsigned i = 0;

      for (CallExpr::const_arg_iterator AI=CE->arg_begin(), AE=CE->arg_end();
           AI!=AE; ++AI, ++i) {

        // Retrieve the value of the argument.  Is it the symbol
        // we are interested in?
        if (CurrSt->getSValAsScalarOrLoc(*AI).getAsLocSymbol() != Sym)
          continue;

        // We have an argument.  Get the effect!
        AEffects.push_back(Summ->getArg(i));
      }
    }
    else if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S)) {
      if (const Expr *receiver = ME->getInstanceReceiver())
        if (CurrSt->getSValAsScalarOrLoc(receiver).getAsLocSymbol() == Sym) {
          // The symbol we are tracking is the receiver.
          AEffects.push_back(Summ->getReceiverEffect());
        }
    }
  }

  do {
    // Get the previous type state.
    RefVal PrevV = *PrevT;

    // Specially handle -dealloc.
    if (!TF.isGCEnabled() && contains(AEffects, Dealloc)) {
      // Determine if the object's reference count was pushed to zero.
      assert(!(PrevV == CurrV) && "The typestate *must* have changed.");
      // We may not have transitioned to 'release' if we hit an error.
      // This case is handled elsewhere.
      if (CurrV.getKind() == RefVal::Released) {
        assert(CurrV.getCombinedCounts() == 0);
        os << "Object released by directly sending the '-dealloc' message";
        break;
      }
    }

    // Specially handle CFMakeCollectable and friends.
    if (contains(AEffects, MakeCollectable)) {
      // Get the name of the function.
      const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();
      SVal X = CurrSt->getSValAsScalarOrLoc(cast<CallExpr>(S)->getCallee());
      const FunctionDecl *FD = X.getAsFunctionDecl();

      if (TF.isGCEnabled()) {
        // Determine if the object's reference count was pushed to zero.
        assert(!(PrevV == CurrV) && "The typestate *must* have changed.");

        os << "In GC mode a call to '" << FD
        <<  "' decrements an object's retain count and registers the "
        "object with the garbage collector. ";

        if (CurrV.getKind() == RefVal::Released) {
          assert(CurrV.getCount() == 0);
          os << "Since it now has a 0 retain count the object can be "
          "automatically collected by the garbage collector.";
        }
        else
          os << "An object must have a 0 retain count to be garbage collected. "
          "After this call its retain count is +" << CurrV.getCount()
          << '.';
      }
      else
        os << "When GC is not enabled a call to '" << FD
        << "' has no effect on its argument.";

      // Nothing more to say.
      break;
    }

    // Determine if the typestate has changed.
    if (!(PrevV == CurrV))
      switch (CurrV.getKind()) {
        case RefVal::Owned:
        case RefVal::NotOwned:

          if (PrevV.getCount() == CurrV.getCount()) {
            // Did an autorelease message get sent?
            if (PrevV.getAutoreleaseCount() == CurrV.getAutoreleaseCount())
              return 0;

            assert(PrevV.getAutoreleaseCount() < CurrV.getAutoreleaseCount());
            os << "Object sent -autorelease message";
            break;
          }

          if (PrevV.getCount() > CurrV.getCount())
            os << "Reference count decremented.";
          else
            os << "Reference count incremented.";

          if (unsigned Count = CurrV.getCount())
            os << " The object now has a +" << Count << " retain count.";

          if (PrevV.getKind() == RefVal::Released) {
            assert(TF.isGCEnabled() && CurrV.getCount() > 0);
            os << " The object is not eligible for garbage collection until the "
            "retain count reaches 0 again.";
          }

          break;

        case RefVal::Released:
          os << "Object released.";
          break;

        case RefVal::ReturnedOwned:
          os << "Object returned to caller as an owning reference (single retain "
          "count transferred to caller)";
          break;

        case RefVal::ReturnedNotOwned:
          os << "Object returned to caller with a +0 retain count";
          break;

        default:
          return NULL;
      }

    // Emit any remaining diagnostics for the argument effects (if any).
    for (SmallVectorImpl<ArgEffect>::iterator I=AEffects.begin(),
         E=AEffects.end(); I != E; ++I) {

      // A bunch of things have alternate behavior under GC.
      if (TF.isGCEnabled())
        switch (*I) {
          default: break;
          case Autorelease:
            os << "In GC mode an 'autorelease' has no effect.";
            continue;
          case IncRefMsg:
            os << "In GC mode the 'retain' message has no effect.";
            continue;
          case DecRefMsg:
            os << "In GC mode the 'release' message has no effect.";
            continue;
        }
    }
  } while (0);

  if (os.str().empty())
    return 0; // We have nothing to say!

  const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();
  PathDiagnosticLocation Pos(S, BRC.getSourceManager());
  PathDiagnosticPiece *P = new PathDiagnosticEventPiece(Pos, os.str());

  // Add the range by scanning the children of the statement for any bindings
  // to Sym.
  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I!=E; ++I)
    if (const Expr *Exp = dyn_cast_or_null<Expr>(*I))
      if (CurrSt->getSValAsScalarOrLoc(Exp).getAsLocSymbol() == Sym) {
        P->addRange(Exp->getSourceRange());
        break;
      }

  return P;
}

namespace {
  class FindUniqueBinding :
  public StoreManager::BindingsHandler {
    SymbolRef Sym;
    const MemRegion* Binding;
    bool First;

  public:
    FindUniqueBinding(SymbolRef sym) : Sym(sym), Binding(0), First(true) {}

    bool HandleBinding(StoreManager& SMgr, Store store, const MemRegion* R,
                       SVal val) {

      SymbolRef SymV = val.getAsSymbol();
      if (!SymV || SymV != Sym)
        return true;

      if (Binding) {
        First = false;
        return false;
      }
      else
        Binding = R;

      return true;
    }

    operator bool() { return First && Binding; }
    const MemRegion* getRegion() { return Binding; }
  };
}

static std::pair<const ExplodedNode*,const MemRegion*>
GetAllocationSite(ProgramStateManager& StateMgr, const ExplodedNode *N,
                  SymbolRef Sym) {

  // Find both first node that referred to the tracked symbol and the
  // memory location that value was store to.
  const ExplodedNode *Last = N;
  const MemRegion* FirstBinding = 0;

  while (N) {
    const ProgramState *St = N->getState();
    RefBindings B = St->get<RefBindings>();

    if (!B.lookup(Sym))
      break;

    FindUniqueBinding FB(Sym);
    StateMgr.iterBindings(St, FB);
    if (FB) FirstBinding = FB.getRegion();

    Last = N;
    N = N->pred_empty() ? NULL : *(N->pred_begin());
  }

  return std::make_pair(Last, FirstBinding);
}

PathDiagnosticPiece*
CFRefReportVisitor::getEndPath(BugReporterContext &BRC,
                               const ExplodedNode *EndN,
                               BugReport &BR) {
  // Tell the BugReporterContext to report cases when the tracked symbol is
  // assigned to different variables, etc.
  BRC.addNotableSymbol(Sym);
  return BugReporterVisitor::getDefaultEndPath(BRC, EndN, BR);
}

PathDiagnosticPiece*
CFRefLeakReportVisitor::getEndPath(BugReporterContext &BRC,
                                   const ExplodedNode *EndN,
                                   BugReport &BR) {

  // Tell the BugReporterContext to report cases when the tracked symbol is
  // assigned to different variables, etc.
  BRC.addNotableSymbol(Sym);

  // We are reporting a leak.  Walk up the graph to get to the first node where
  // the symbol appeared, and also get the first VarDecl that tracked object
  // is stored to.
  const ExplodedNode *AllocNode = 0;
  const MemRegion* FirstBinding = 0;

  llvm::tie(AllocNode, FirstBinding) =
    GetAllocationSite(BRC.getStateManager(), EndN, Sym);

  SourceManager& SMgr = BRC.getSourceManager();

  // Compute an actual location for the leak.  Sometimes a leak doesn't
  // occur at an actual statement (e.g., transition between blocks; end
  // of function) so we need to walk the graph and compute a real location.
  const ExplodedNode *LeakN = EndN;
  PathDiagnosticLocation L;

  while (LeakN) {
    ProgramPoint P = LeakN->getLocation();

    if (const StmtPoint *PS = dyn_cast<StmtPoint>(&P)) {
      L = PathDiagnosticLocation(PS->getStmt()->getLocStart(), SMgr);
      break;
    }
    else if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
      if (const Stmt *Term = BE->getSrc()->getTerminator()) {
        L = PathDiagnosticLocation(Term->getLocStart(), SMgr);
        break;
      }
    }

    LeakN = LeakN->succ_empty() ? 0 : *(LeakN->succ_begin());
  }

  if (!L.isValid()) {
    const Decl &D = EndN->getCodeDecl();
    L = PathDiagnosticLocation(D.getBodyRBrace(), SMgr);
  }

  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  os << "Object leaked: ";

  if (FirstBinding) {
    os << "object allocated and stored into '"
       << FirstBinding->getString() << '\'';
  }
  else
    os << "allocated object";

  // Get the retain count.
  const RefVal* RV = EndN->getState()->get<RefBindings>(Sym);

  if (RV->getKind() == RefVal::ErrorLeakReturned) {
    // FIXME: Per comments in rdar://6320065, "create" only applies to CF
    // objects.  Only "copy", "alloc", "retain" and "new" transfer ownership
    // to the caller for NS objects.
    const Decl *D = &EndN->getCodeDecl();
    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
      os << " is returned from a method whose name ('"
         << MD->getSelector().getAsString()
         << "') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'."
            "  This violates the naming convention rules"
            " given in the Memory Management Guide for Cocoa";
    }
    else {
      const FunctionDecl *FD = cast<FunctionDecl>(D);
      os << " is return from a function whose name ('"
         << FD->getNameAsString()
         << "') does not contain 'Copy' or 'Create'.  This violates the naming"
            " convention rules given the Memory Management Guide for Core"
            " Foundation";
    }    
  }
  else if (RV->getKind() == RefVal::ErrorGCLeakReturned) {
    ObjCMethodDecl &MD = cast<ObjCMethodDecl>(EndN->getCodeDecl());
    os << " and returned from method '" << MD.getSelector().getAsString()
       << "' is potentially leaked when using garbage collection.  Callers "
          "of this method do not expect a returned object with a +1 retain "
          "count since they expect the object to be managed by the garbage "
          "collector";
  }
  else
    os << " is not referenced later in this execution path and has a retain "
          "count of +" << RV->getCount();

  return new PathDiagnosticEventPiece(L, os.str());
}

CFRefLeakReport::CFRefLeakReport(CFRefBug& D, const CFRefCount &tf,
                                 ExplodedNode *n,
                                 SymbolRef sym, ExprEngine& Eng)
: CFRefReport(D, tf, n, sym, false) {

  // Most bug reports are cached at the location where they occurred.
  // With leaks, we want to unique them by the location where they were
  // allocated, and only report a single path.  To do this, we need to find
  // the allocation site of a piece of tracked memory, which we do via a
  // call to GetAllocationSite.  This will walk the ExplodedGraph backwards.
  // Note that this is *not* the trimmed graph; we are guaranteed, however,
  // that all ancestor nodes that represent the allocation site have the
  // same SourceLocation.
  const ExplodedNode *AllocNode = 0;

  llvm::tie(AllocNode, AllocBinding) =  // Set AllocBinding.
    GetAllocationSite(Eng.getStateManager(), getErrorNode(), sym);

  // Get the SourceLocation for the allocation site.
  ProgramPoint P = AllocNode->getLocation();
  AllocSite = cast<PostStmt>(P).getStmt()->getLocStart();

  // Fill in the description of the bug.
  Description.clear();
  llvm::raw_string_ostream os(Description);
  SourceManager& SMgr = Eng.getContext().getSourceManager();
  unsigned AllocLine = SMgr.getExpansionLineNumber(AllocSite);
  os << "Potential leak ";
  if (tf.isGCEnabled()) {
    os << "(when using garbage collection) ";
  }
  os << "of an object allocated on line " << AllocLine;

  // FIXME: AllocBinding doesn't get populated for RegionStore yet.
  if (AllocBinding)
    os << " and stored into '" << AllocBinding->getString() << '\'';

  addVisitor(new CFRefLeakReportVisitor(sym, tf));
}

//===----------------------------------------------------------------------===//
// Main checker logic.
//===----------------------------------------------------------------------===//

/// GetReturnType - Used to get the return type of a message expression or
///  function call with the intention of affixing that type to a tracked symbol.
///  While the the return type can be queried directly from RetEx, when
///  invoking class methods we augment to the return type to be that of
///  a pointer to the class (as opposed it just being id).
static QualType GetReturnType(const Expr *RetE, ASTContext &Ctx) {
  QualType RetTy = RetE->getType();
  // If RetE is not a message expression just return its type.
  // If RetE is a message expression, return its types if it is something
  /// more specific than id.
  if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(RetE))
    if (const ObjCObjectPointerType *PT = RetTy->getAs<ObjCObjectPointerType>())
      if (PT->isObjCQualifiedIdType() || PT->isObjCIdType() ||
          PT->isObjCClassType()) {
        // At this point we know the return type of the message expression is
        // id, id<...>, or Class. If we have an ObjCInterfaceDecl, we know this
        // is a call to a class method whose type we can resolve.  In such
        // cases, promote the return type to XXX* (where XXX is the class).
        const ObjCInterfaceDecl *D = ME->getReceiverInterface();
        return !D ? RetTy :
                    Ctx.getObjCObjectPointerType(Ctx.getObjCInterfaceType(D));
      }

  return RetTy;
}


// HACK: Symbols that have ref-count state that are referenced directly
//  (not as structure or array elements, or via bindings) by an argument
//  should not have their ref-count state stripped after we have
//  done an invalidation pass.
//
// FIXME: This is a global to currently share between CFRefCount and
// RetainReleaseChecker.  Eventually all functionality in CFRefCount should
// be migrated to RetainReleaseChecker, and we can make this a non-global.
llvm::DenseSet<SymbolRef> WhitelistedSymbols;
namespace {
struct ResetWhiteList {
  ResetWhiteList() {}
  ~ResetWhiteList() { WhitelistedSymbols.clear(); } 
};
}

void CFRefCount::evalCallOrMessage(ExplodedNodeSet &Dst, ExprEngine &Eng,
                                   StmtNodeBuilder &Builder,
                                   const CallOrObjCMessage &callOrMsg,
                                   InstanceReceiver Receiver,
                                   const MemRegion *Callee,
                                   ExplodedNode *Pred,
                                   const ProgramState *state) {

  SmallVector<const MemRegion*, 10> RegionsToInvalidate;
  
  // Use RAII to make sure the whitelist is properly cleared.
  ResetWhiteList resetWhiteList;

  // Invalidate all instance variables of the receiver of a message.
  // FIXME: We should be able to do better with inter-procedural analysis.
  if (Receiver) {
    SVal V = Receiver.getSValAsScalarOrLoc(state);
    if (SymbolRef Sym = V.getAsLocSymbol()) {
      if (state->get<RefBindings>(Sym))
        WhitelistedSymbols.insert(Sym);
    }
    if (const MemRegion *region = V.getAsRegion())
      RegionsToInvalidate.push_back(region);
  }
  
  // Invalidate all instance variables for the callee of a C++ method call.
  // FIXME: We should be able to do better with inter-procedural analysis.
  // FIXME: we can probably do better for const versus non-const methods.
  if (callOrMsg.isCXXCall()) {
    if (const MemRegion *callee = callOrMsg.getCXXCallee().getAsRegion())
      RegionsToInvalidate.push_back(callee);
  }
  
  for (unsigned idx = 0, e = callOrMsg.getNumArgs(); idx != e; ++idx) {
    SVal V = callOrMsg.getArgSValAsScalarOrLoc(idx);

    // If we are passing a location wrapped as an integer, unwrap it and
    // invalidate the values referred by the location.
    if (nonloc::LocAsInteger *Wrapped = dyn_cast<nonloc::LocAsInteger>(&V))
      V = Wrapped->getLoc();
    else if (!isa<Loc>(V))
      continue;

    if (SymbolRef Sym = V.getAsLocSymbol())
      if (state->get<RefBindings>(Sym))
        WhitelistedSymbols.insert(Sym);

    if (const MemRegion *R = V.getAsRegion()) {
      // Invalidate the value of the variable passed by reference.

      // Are we dealing with an ElementRegion?  If the element type is
      // a basic integer type (e.g., char, int) and the underying region
      // is a variable region then strip off the ElementRegion.
      // FIXME: We really need to think about this for the general case
      //   as sometimes we are reasoning about arrays and other times
      //   about (char*), etc., is just a form of passing raw bytes.
      //   e.g., void *p = alloca(); foo((char*)p);
      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Checking for 'integral type' is probably too promiscuous, but
        // we'll leave it in for now until we have a systematic way of
        // handling all of these cases.  Eventually we need to come up
        // with an interface to StoreManager so that this logic can be
        // approriately delegated to the respective StoreManagers while
        // still allowing us to do checker-specific logic (e.g.,
        // invalidating reference counts), probably via callbacks.
        if (ER->getElementType()->isIntegralOrEnumerationType()) {
          const MemRegion *superReg = ER->getSuperRegion();
          if (isa<VarRegion>(superReg) || isa<FieldRegion>(superReg) ||
              isa<ObjCIvarRegion>(superReg))
            R = cast<TypedRegion>(superReg);
        }
        // FIXME: What about layers of ElementRegions?
      }

      // Mark this region for invalidation.  We batch invalidate regions
      // below for efficiency.
      RegionsToInvalidate.push_back(R);
    } else {
      // Nuke all other arguments passed by reference.
      // FIXME: is this necessary or correct? This handles the non-Region
      //  cases.  Is it ever valid to store to these?
      state = state->unbindLoc(cast<Loc>(V));
    }
  }

  // Block calls result in all captured values passed-via-reference to be
  // invalidated.
  if (const BlockDataRegion *BR = dyn_cast_or_null<BlockDataRegion>(Callee))
    RegionsToInvalidate.push_back(BR);

  // Invalidate designated regions using the batch invalidation API.

  // FIXME: We can have collisions on the conjured symbol if the
  //  expression *I also creates conjured symbols.  We probably want
  //  to identify conjured symbols by an expression pair: the enclosing
  //  expression (the context) and the expression itself.  This should
  //  disambiguate conjured symbols.
  unsigned Count = Builder.getCurrentBlockCount();
  StoreManager::InvalidatedSymbols IS;

  const Expr *Ex = callOrMsg.getOriginExpr();

  // NOTE: Even if RegionsToInvalidate is empty, we must still invalidate
  //  global variables.
  // NOTE: RetainReleaseChecker handles the actual invalidation of symbols.
  state =
    state->invalidateRegions(RegionsToInvalidate.data(),
                             RegionsToInvalidate.data() +
                             RegionsToInvalidate.size(),
                             Ex, Count, &IS,
                             /* invalidateGlobals = */
                             Eng.doesInvalidateGlobals(callOrMsg));

  Builder.MakeNode(Dst, Ex, Pred, state);
}


void CFRefCount::evalCall(ExplodedNodeSet &Dst,
                          ExprEngine& Eng,
                          StmtNodeBuilder& Builder,
                          const CallExpr *CE, SVal L,
                          ExplodedNode *Pred) {

  evalCallOrMessage(Dst, Eng, Builder, CallOrObjCMessage(CE, Pred->getState()),
                    InstanceReceiver(), L.getAsRegion(), Pred, 
                    Pred->getState());
}

void CFRefCount::evalObjCMessage(ExplodedNodeSet &Dst,
                                 ExprEngine& Eng,
                                 StmtNodeBuilder& Builder,
                                 ObjCMessage msg,
                                 ExplodedNode *Pred,
                                 const ProgramState *state) {

  evalCallOrMessage(Dst, Eng, Builder, CallOrObjCMessage(msg, Pred->getState()),
                    InstanceReceiver(msg, Pred->getLocationContext()),
                    /* Callee = */ 0, Pred, state);
}

const ProgramState * CFRefCount::Update(const ProgramState * state,
                                        SymbolRef sym,
                                        RefVal V,
                                        ArgEffect E,
                                        RefVal::Kind& hasErr) {

  // In GC mode [... release] and [... retain] do nothing.
  switch (E) {
    default: break;
    case IncRefMsg: E = isARCorGCEnabled() ? DoNothing : IncRef; break;
    case DecRefMsg: E = isARCorGCEnabled() ? DoNothing : DecRef; break;
    case MakeCollectable: E = isGCEnabled() ? DecRef : DoNothing; break;
    case NewAutoreleasePool: E = isGCEnabled() ? DoNothing :
                                                 NewAutoreleasePool; break;
  }

  // Handle all use-after-releases.
  if (!isGCEnabled() && V.getKind() == RefVal::Released) {
    V = V ^ RefVal::ErrorUseAfterRelease;
    hasErr = V.getKind();
    return state->set<RefBindings>(sym, V);
  }

  switch (E) {
    case DecRefMsg:
    case IncRefMsg:
    case MakeCollectable:
      assert(false &&
             "DecRefMsg/IncRefMsg/MakeCollectable already transformed");
      return state;
      
    case Dealloc:
      // Any use of -dealloc in GC is *bad*.
      if (isGCEnabled()) {
        V = V ^ RefVal::ErrorDeallocGC;
        hasErr = V.getKind();
        break;
      }

      switch (V.getKind()) {
        default:
          assert(false && "Invalid case.");
        case RefVal::Owned:
          // The object immediately transitions to the released state.
          V = V ^ RefVal::Released;
          V.clearCounts();
          return state->set<RefBindings>(sym, V);
        case RefVal::NotOwned:
          V = V ^ RefVal::ErrorDeallocNotOwned;
          hasErr = V.getKind();
          break;
      }
      break;

    case NewAutoreleasePool:
      assert(!isGCEnabled());
      return state->add<AutoreleaseStack>(sym);

    case MayEscape:
      if (V.getKind() == RefVal::Owned) {
        V = V ^ RefVal::NotOwned;
        break;
      }

      // Fall-through.

    case DoNothingByRef:
    case DoNothing:
      return state;

    case Autorelease:
      if (isGCEnabled())
        return state;

      // Update the autorelease counts.
      state = SendAutorelease(state, ARCountFactory, sym);
      V = V.autorelease();
      break;

    case StopTracking:
      return state->remove<RefBindings>(sym);

    case IncRef:
      switch (V.getKind()) {
        default:
          assert(false);

        case RefVal::Owned:
        case RefVal::NotOwned:
          V = V + 1;
          break;
        case RefVal::Released:
          // Non-GC cases are handled above.
          assert(isGCEnabled());
          V = (V ^ RefVal::Owned) + 1;
          break;
      }
      break;

    case SelfOwn:
      V = V ^ RefVal::NotOwned;
      // Fall-through.
    case DecRef:
    case DecRefBridgedTransfered:
      switch (V.getKind()) {
        default:
          // case 'RefVal::Released' handled above.
          assert (false);

        case RefVal::Owned:
          assert(V.getCount() > 0);
          if (V.getCount() == 1)
            V = V ^ (E == DecRefBridgedTransfered ? 
                      RefVal::NotOwned : RefVal::Released);
          V = V - 1;
          break;

        case RefVal::NotOwned:
          if (V.getCount() > 0)
            V = V - 1;
          else {
            V = V ^ RefVal::ErrorReleaseNotOwned;
            hasErr = V.getKind();
          }
          break;

        case RefVal::Released:
          // Non-GC cases are handled above.
          assert(isGCEnabled());
          V = V ^ RefVal::ErrorUseAfterRelease;
          hasErr = V.getKind();
          break;
      }
      break;
  }
  return state->set<RefBindings>(sym, V);
}

//===----------------------------------------------------------------------===//
// Handle dead symbols and end-of-path.
//===----------------------------------------------------------------------===//

std::pair<ExplodedNode*, const ProgramState *>
CFRefCount::HandleAutoreleaseCounts(const ProgramState *state, 
                                    GenericNodeBuilderRefCount Bd,
                                    ExplodedNode *Pred,
                                    ExprEngine &Eng,
                                    SymbolRef Sym,
                                    RefVal V,
                                    bool &stop) {

  unsigned ACnt = V.getAutoreleaseCount();
  stop = false;

  // No autorelease counts?  Nothing to be done.
  if (!ACnt)
    return std::make_pair(Pred, state);

  assert(!isGCEnabled() && "Autorelease counts in GC mode?");
  unsigned Cnt = V.getCount();

  // FIXME: Handle sending 'autorelease' to already released object.

  if (V.getKind() == RefVal::ReturnedOwned)
    ++Cnt;

  if (ACnt <= Cnt) {
    if (ACnt == Cnt) {
      V.clearCounts();
      if (V.getKind() == RefVal::ReturnedOwned)
        V = V ^ RefVal::ReturnedNotOwned;
      else
        V = V ^ RefVal::NotOwned;
    }
    else {
      V.setCount(Cnt - ACnt);
      V.setAutoreleaseCount(0);
    }
    state = state->set<RefBindings>(Sym, V);
    ExplodedNode *N = Bd.MakeNode(state, Pred);
    stop = (N == 0);
    return std::make_pair(N, state);
  }

  // Woah!  More autorelease counts then retain counts left.
  // Emit hard error.
  stop = true;
  V = V ^ RefVal::ErrorOverAutorelease;
  state = state->set<RefBindings>(Sym, V);

  if (ExplodedNode *N = Bd.MakeNode(state, Pred)) {
    N->markAsSink();

    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    os << "Object over-autoreleased: object was sent -autorelease ";
    if (V.getAutoreleaseCount() > 1)
      os << V.getAutoreleaseCount() << " times ";
    os << "but the object has a +" << V.getCount() << " retain count";

    CFRefReport *report =
      new CFRefReport(*static_cast<CFRefBug*>(overAutorelease),
                      *this, N, Sym, os.str());
    BR->EmitReport(report);
  }

  return std::make_pair((ExplodedNode*)0, state);
}

//===----------------------------------------------------------------------===//
// Pieces of the retain/release checker implemented using a CheckerVisitor.
// More pieces of the retain/release checker will be migrated to this interface
// (ideally, all of it some day).
//===----------------------------------------------------------------------===//

namespace {
class RetainReleaseChecker
  : public Checker< check::Bind,
                    check::DeadSymbols,
                    check::EndPath,
                    check::PostStmt<BlockExpr>,
                    check::PostStmt<CastExpr>,
                    check::PostStmt<CallExpr>,
                    check::PostObjCMessage,
                    check::PreStmt<ReturnStmt>,
                    check::RegionChanges,
                    eval::Assume,
                    eval::Call > {

  typedef llvm::DenseMap<SymbolRef, const SimpleProgramPointTag *> SymbolTagMap;

  // This map is only used to ensure proper deletion of any allocated tags.
  mutable SymbolTagMap DeadSymbolTags;

public:  

  virtual ~RetainReleaseChecker() {
    DeleteContainerSeconds(DeadSymbolTags);
  }

  void checkBind(SVal loc, SVal val, CheckerContext &C) const;
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
  void checkPostStmt(const CastExpr *CE, CheckerContext &C) const;

  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostObjCMessage(const ObjCMessage &Msg, CheckerContext &C) const;
  void checkSummary(const RetainSummary &Summ, const CallOrObjCMessage &Call,
                    InstanceReceiver Receiver, CheckerContext &C) const;

  bool evalCall(const CallExpr *CE, CheckerContext &C) const;

  const ProgramState *evalAssume(const ProgramState *state, SVal Cond,
                                 bool Assumption) const;

  const ProgramState *checkRegionChanges(const ProgramState *state,
                          const StoreManager::InvalidatedSymbols *invalidated,
                                         const MemRegion * const *begin,
                                         const MemRegion * const *end) const;
                                        
  bool wantsRegionChangeUpdate(const ProgramState *state) const {
    return true;
  }

  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  void checkReturnWithRetEffect(const ReturnStmt *S, CheckerContext &C,
                                ExplodedNode *Pred, RetEffect RE, RefVal X,
                                SymbolRef Sym, const ProgramState *state) const;
                                              
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkEndPath(EndOfFunctionNodeBuilder &Builder, ExprEngine &Eng) const;

  void processNonLeakError(const ProgramState *St, SourceRange ErrorRange,
                           RefVal::Kind ErrorKind, SymbolRef Sym,
                           CheckerContext &C) const;

  const ProgramPointTag *getDeadSymbolTag(SymbolRef sym) const;

  const ProgramState *handleSymbolDeath(const ProgramState *state,
                                        SymbolRef sid, RefVal V,
                                      SmallVectorImpl<SymbolRef> &Leaked) const;

  ExplodedNode *processLeaks(const ProgramState *state,
                             SmallVectorImpl<SymbolRef> &Leaked,
                             GenericNodeBuilderRefCount &Builder,
                             ExprEngine &Eng,
                             ExplodedNode *Pred = 0) const;
};
} // end anonymous namespace

namespace {
class StopTrackingCallback : public SymbolVisitor {
  const ProgramState *state;
public:
  StopTrackingCallback(const ProgramState *st) : state(st) {}
  const ProgramState *getState() const { return state; }

  bool VisitSymbol(SymbolRef sym) {
    state = state->remove<RefBindings>(sym);
    return true;
  }
};
} // end anonymous namespace


void RetainReleaseChecker::checkBind(SVal loc, SVal val,
                                     CheckerContext &C) const {
  // Are we storing to something that causes the value to "escape"?
  bool escapes = true;

  // A value escapes in three possible cases (this may change):
  //
  // (1) we are binding to something that is not a memory region.
  // (2) we are binding to a memregion that does not have stack storage
  // (3) we are binding to a memregion with stack storage that the store
  //     does not understand.
  const ProgramState *state = C.getState();

  if (loc::MemRegionVal *regionLoc = dyn_cast<loc::MemRegionVal>(&loc)) {
    escapes = !regionLoc->getRegion()->hasStackStorage();

    if (!escapes) {
      // To test (3), generate a new state with the binding added.  If it is
      // the same state, then it escapes (since the store cannot represent
      // the binding).
      escapes = (state == (state->bindLoc(*regionLoc, val)));
    }
  }

  // If our store can represent the binding and we aren't storing to something
  // that doesn't have local storage then just return and have the simulation
  // state continue as is.
  if (!escapes)
      return;

  // Otherwise, find all symbols referenced by 'val' that we are tracking
  // and stop tracking them.
  state = state->scanReachableSymbols<StopTrackingCallback>(val).getState();
  C.addTransition(state);
}

// Assumptions.

const ProgramState *RetainReleaseChecker::evalAssume(const ProgramState *state,
                                                     SVal Cond,
                                                     bool Assumption) const {

  // FIXME: We may add to the interface of evalAssume the list of symbols
  //  whose assumptions have changed.  For now we just iterate through the
  //  bindings and check if any of the tracked symbols are NULL.  This isn't
  //  too bad since the number of symbols we will track in practice are
  //  probably small and evalAssume is only called at branches and a few
  //  other places.
  RefBindings B = state->get<RefBindings>();

  if (B.isEmpty())
    return state;

  bool changed = false;
  RefBindings::Factory &RefBFactory = state->get_context<RefBindings>();

  for (RefBindings::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    // Check if the symbol is null (or equal to any constant).
    // If this is the case, stop tracking the symbol.
    if (state->getSymVal(I.getKey())) {
      changed = true;
      B = RefBFactory.remove(B, I.getKey());
    }
  }

  if (changed)
    state = state->set<RefBindings>(B);

  return state;
}

const ProgramState *
RetainReleaseChecker::checkRegionChanges(const ProgramState *state,
                            const StoreManager::InvalidatedSymbols *invalidated,
                                         const MemRegion * const *begin,
                                         const MemRegion * const *end) const {
  if (!invalidated)
    return state;

  for (StoreManager::InvalidatedSymbols::const_iterator I=invalidated->begin(),
       E = invalidated->end(); I!=E; ++I) {
    SymbolRef sym = *I;
    if (WhitelistedSymbols.count(sym))
      continue;
    // Remove any existing reference-count binding.
    state = state->remove<RefBindings>(sym);
  }
  return state;
}

void RetainReleaseChecker::checkPostStmt(const BlockExpr *BE,
                                         CheckerContext &C) const {

  // Scan the BlockDecRefExprs for any object the retain/release checker
  // may be tracking.
  if (!BE->getBlockDecl()->hasCaptures())
    return;

  const ProgramState *state = C.getState();
  const BlockDataRegion *R =
    cast<BlockDataRegion>(state->getSVal(BE).getAsRegion());

  BlockDataRegion::referenced_vars_iterator I = R->referenced_vars_begin(),
                                            E = R->referenced_vars_end();

  if (I == E)
    return;

  // FIXME: For now we invalidate the tracking of all symbols passed to blocks
  // via captured variables, even though captured variables result in a copy
  // and in implicit increment/decrement of a retain count.
  SmallVector<const MemRegion*, 10> Regions;
  const LocationContext *LC = C.getPredecessor()->getLocationContext();
  MemRegionManager &MemMgr = C.getSValBuilder().getRegionManager();

  for ( ; I != E; ++I) {
    const VarRegion *VR = *I;
    if (VR->getSuperRegion() == R) {
      VR = MemMgr.getVarRegion(VR->getDecl(), LC);
    }
    Regions.push_back(VR);
  }

  state =
    state->scanReachableSymbols<StopTrackingCallback>(Regions.data(),
                                    Regions.data() + Regions.size()).getState();
  C.addTransition(state);
}

void RetainReleaseChecker::checkPostStmt(const CastExpr *CE,
                                         CheckerContext &C) const {
  const ObjCBridgedCastExpr *BE = dyn_cast<ObjCBridgedCastExpr>(CE);
  if (!BE)
    return;
  
  ArgEffect AE = IncRef;
  
  switch (BE->getBridgeKind()) {
    case clang::OBC_Bridge:
      // Do nothing.
      return;
    case clang::OBC_BridgeRetained:
      AE = IncRef;
      break;      
    case clang::OBC_BridgeTransfer:
      AE = DecRefBridgedTransfered;
      break;
  }
  
  const ProgramState *state = C.getState();
  SymbolRef Sym = state->getSVal(CE).getAsLocSymbol();
  if (!Sym)
    return;
  const RefVal* T = state->get<RefBindings>(Sym);
  if (!T)
    return;

  // This is gross.  Once the checker and CFRefCount are unified,
  // this will go away.
  CFRefCount &cf = static_cast<CFRefCount&>(C.getEngine().getTF());
  RefVal::Kind hasErr = (RefVal::Kind) 0;
  state = cf.Update(state, Sym, *T, AE, hasErr);
  
  if (hasErr) {
    
    return;
  }

  C.generateNode(state);
}

void RetainReleaseChecker::checkPostStmt(const CallExpr *CE,
                                         CheckerContext &C) const {
  // FIXME: This goes away once the RetainSummaryManager moves to the checker.
  CFRefCount &TF = static_cast<CFRefCount&>(C.getEngine().getTF());
  RetainSummaryManager &Summaries = TF.Summaries;

  // Get the callee.
  const ProgramState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);

  RetainSummary *Summ = 0;

  // FIXME: Better support for blocks.  For now we stop tracking anything
  // that is passed to blocks.
  // FIXME: Need to handle variables that are "captured" by the block.
  if (dyn_cast_or_null<BlockDataRegion>(L.getAsRegion())) {
    Summ = Summaries.getPersistentStopSummary();
  } else if (const FunctionDecl *FD = L.getAsFunctionDecl()) {
    Summ = Summaries.getSummary(FD);
  } else if (const CXXMemberCallExpr *me = dyn_cast<CXXMemberCallExpr>(CE)) {
    if (const CXXMethodDecl *MD = me->getMethodDecl())
      Summ = Summaries.getSummary(MD);
  }

  // If we didn't get a summary, this function doesn't affect retain counts.
  if (!Summ)
    return;

  checkSummary(*Summ, CallOrObjCMessage(CE, state), InstanceReceiver(), C);
}

void RetainReleaseChecker::checkPostObjCMessage(const ObjCMessage &Msg, 
                                                CheckerContext &C) const {
  // FIXME: This goes away once the RetainSummaryManager moves to the checker.
  CFRefCount &TF = static_cast<CFRefCount&>(C.getEngine().getTF());
  RetainSummaryManager &Summaries = TF.Summaries;

  const ProgramState *state = C.getState();
  ExplodedNode *Pred = C.getPredecessor();

  RetainSummary *Summ;
  if (Msg.isInstanceMessage()) {
    const LocationContext *LC = Pred->getLocationContext();
    Summ = Summaries.getInstanceMethodSummary(Msg, state, LC);
  } else {
    Summ = Summaries.getClassMethodSummary(Msg);    
  }

  // If we didn't get a summary, this message doesn't affect retain counts.
  if (!Summ)
    return;

  checkSummary(*Summ, CallOrObjCMessage(Msg, state),
               InstanceReceiver(Msg, Pred->getLocationContext()), C);
}

void RetainReleaseChecker::checkSummary(const RetainSummary &Summ,
                                        const CallOrObjCMessage &CallOrMsg,
                                        InstanceReceiver Receiver,
                                        CheckerContext &C) const {
  // FIXME: This goes away once the Update() method moves to the checker.
  CFRefCount &TF = static_cast<CFRefCount&>(C.getEngine().getTF());

  const ProgramState *state = C.getState();

  // Evaluate the effect of the arguments.
  RefVal::Kind hasErr = (RefVal::Kind) 0;
  SourceRange ErrorRange;
  SymbolRef ErrorSym = 0;

  for (unsigned idx = 0, e = CallOrMsg.getNumArgs(); idx != e; ++idx) {
    SVal V = CallOrMsg.getArgSValAsScalarOrLoc(idx);

    if (SymbolRef Sym = V.getAsLocSymbol()) {
      if (RefBindings::data_type *T = state->get<RefBindings>(Sym)) {
        state = TF.Update(state, Sym, *T, Summ.getArg(idx), hasErr);
        if (hasErr) {
          ErrorRange = CallOrMsg.getArgSourceRange(idx);
          ErrorSym = Sym;
          break;
        }
      }
    }
  }

  // Evaluate the effect on the message receiver.
  bool ReceiverIsTracked = false;
  if (!hasErr && Receiver) {
    if (SymbolRef Sym = Receiver.getSValAsScalarOrLoc(state).getAsLocSymbol()) {
      if (const RefVal *T = state->get<RefBindings>(Sym)) {
        ReceiverIsTracked = true;
        state = TF.Update(state, Sym, *T, Summ.getReceiverEffect(), hasErr);
        if (hasErr) {
          ErrorRange = Receiver.getSourceRange();
          ErrorSym = Sym;
        }
      }
    }
  }

  // Process any errors.
  if (hasErr) {
    processNonLeakError(state, ErrorRange, hasErr, ErrorSym, C);
    return;
  }

  // Consult the summary for the return value.
  RetEffect RE = Summ.getRetEffect();

  if (RE.getKind() == RetEffect::OwnedWhenTrackedReceiver) {
    if (ReceiverIsTracked)
      RE = TF.Summaries.getObjAllocRetEffect();
    else
      RE = RetEffect::MakeNoRet();
  }

  switch (RE.getKind()) {
    default:
      llvm_unreachable("Unhandled RetEffect."); break;

    case RetEffect::NoRet:
      // No work necessary.
      break;

    case RetEffect::OwnedAllocatedSymbol:
    case RetEffect::OwnedSymbol: {
      SymbolRef Sym = state->getSVal(CallOrMsg.getOriginExpr()).getAsSymbol();
      if (!Sym)
        break;

      // Use the result type from callOrMsg as it automatically adjusts
      // for methods/functions that return references.
      QualType ResultTy = CallOrMsg.getResultType(C.getASTContext());
      state = state->set<RefBindings>(Sym, RefVal::makeOwned(RE.getObjKind(),
                                                             ResultTy));

      // FIXME: Add a flag to the checker where allocations are assumed to
      // *not* fail. (The code below is out-of-date, though.)
#if 0
      if (RE.getKind() == RetEffect::OwnedAllocatedSymbol) {
        bool isFeasible;
        state = state.assume(loc::SymbolVal(Sym), true, isFeasible);
        assert(isFeasible && "Cannot assume fresh symbol is non-null.");
      }
#endif

      break;
    }

    case RetEffect::GCNotOwnedSymbol:
    case RetEffect::ARCNotOwnedSymbol:
    case RetEffect::NotOwnedSymbol: {
      const Expr *Ex = CallOrMsg.getOriginExpr();
      SymbolRef Sym = state->getSVal(Ex).getAsSymbol();
      if (!Sym)
        break;

      // Use GetReturnType in order to give [NSFoo alloc] the type NSFoo *.
      QualType ResultTy = GetReturnType(Ex, C.getASTContext());
      state = state->set<RefBindings>(Sym, RefVal::makeNotOwned(RE.getObjKind(),
                                                                ResultTy));
      break;
    }
  }

  // This check is actually necessary; otherwise the statement builder thinks
  // we've hit a previously-found path.
  // Normally addTransition takes care of this, but we want the node pointer.
  ExplodedNode *NewNode;
  if (state == C.getState()) {
    NewNode = C.getPredecessor();
  } else {
    NewNode = C.generateNode(state);
  }

  // Annotate the edge with summary we used.
  // FIXME: The summary log should live on RetainReleaseChecker.
  if (NewNode) TF.SummaryLog[NewNode] = &Summ;
}

void RetainReleaseChecker::processNonLeakError(const ProgramState *St,
                                               SourceRange ErrorRange,
                                               RefVal::Kind ErrorKind,
                                               SymbolRef Sym,
                                               CheckerContext &C) const {
  ExplodedNode *N = C.generateSink(St);
  if (!N)
    return;

  // FIXME: This goes away once the these bug types move to the checker,
  // and CFRefReport no longer depends on CFRefCount.
  CFRefCount &TF = static_cast<CFRefCount&>(C.getEngine().getTF());

  CFRefBug *BT;
  switch (ErrorKind) {
    default:
      llvm_unreachable("Unhandled error.");
      return;
    case RefVal::ErrorUseAfterRelease:
      BT = static_cast<CFRefBug*>(TF.useAfterRelease);
      break;
    case RefVal::ErrorReleaseNotOwned:
      BT = static_cast<CFRefBug*>(TF.releaseNotOwned);
      break;
    case RefVal::ErrorDeallocGC:
      BT = static_cast<CFRefBug*>(TF.deallocGC);
      break;
    case RefVal::ErrorDeallocNotOwned:
      BT = static_cast<CFRefBug*>(TF.deallocNotOwned);
      break;
  }

  CFRefReport *report = new CFRefReport(*BT, TF, N, Sym);
  report->addRange(ErrorRange);
  C.EmitReport(report);
}

bool RetainReleaseChecker::evalCall(const CallExpr *CE,
                                    CheckerContext &C) const {
  // Get the callee. We're only interested in simple C functions.
  const ProgramState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);

  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return false;

  // For now, we're only handling the functions that return aliases of their
  // arguments: CFRetain and CFMakeCollectable (and their families).
  // Eventually we should add other functions we can model entirely,
  // such as CFRelease, which don't invalidate their arguments or globals.
  if (CE->getNumArgs() != 1)
    return false;

  // Get the name of the function.
  StringRef FName = II->getName();
  FName = FName.substr(FName.find_first_not_of('_'));

  // See if it's one of the specific functions we know how to eval.
  bool canEval = false;

  QualType ResultTy = FD->getResultType();
  if (ResultTy->isObjCIdType()) {
    // Handle: id NSMakeCollectable(CFTypeRef)
    canEval = II->isStr("NSMakeCollectable");
  } else if (ResultTy->isPointerType()) {
    // Handle: (CF|CG)Retain
    //         CFMakeCollectable
    // It's okay to be a little sloppy here (CGMakeCollectable doesn't exist).
    if (cocoa::isRefType(ResultTy, "CF", FName) ||
        cocoa::isRefType(ResultTy, "CG", FName)) {
      canEval = isRetain(FD, FName) || isMakeCollectable(FD, FName);
    }
  }
        
  if (!canEval)
    return false;

  // Bind the return value.
  SVal RetVal = state->getSVal(CE->getArg(0));
  if (RetVal.isUnknown()) {
    // If the receiver is unknown, conjure a return value.
    SValBuilder &SVB = C.getSValBuilder();
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    SVal RetVal = SVB.getConjuredSymbolVal(0, CE, ResultTy, Count);
  }
  state = state->BindExpr(CE, RetVal, false);

  // FIXME: This should not be necessary, but otherwise the argument seems to be
  // considered alive during the next statement.
  if (const MemRegion *ArgRegion = RetVal.getAsRegion()) {
    // Save the refcount status of the argument.
    SymbolRef Sym = RetVal.getAsLocSymbol();
    RefBindings::data_type *Binding = 0;
    if (Sym)
      Binding = state->get<RefBindings>(Sym);

    // Invalidate the argument region.
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    state = state->invalidateRegion(ArgRegion, CE, Count);

    // Restore the refcount status of the argument.
    if (Binding)
      state = state->set<RefBindings>(Sym, *Binding);
  }

  C.addTransition(state);
  return true;
}

 // Return statements.

void RetainReleaseChecker::checkPreStmt(const ReturnStmt *S,
                                        CheckerContext &C) const {
  const Expr *RetE = S->getRetValue();
  if (!RetE)
    return;

  const ProgramState *state = C.getState();
  SymbolRef Sym = state->getSValAsScalarOrLoc(RetE).getAsLocSymbol();
  if (!Sym)
    return;

  // Get the reference count binding (if any).
  const RefVal *T = state->get<RefBindings>(Sym);
  if (!T)
    return;

  // Change the reference count.
  RefVal X = *T;

  switch (X.getKind()) {
    case RefVal::Owned: {
      unsigned cnt = X.getCount();
      assert(cnt > 0);
      X.setCount(cnt - 1);
      X = X ^ RefVal::ReturnedOwned;
      break;
    }

    case RefVal::NotOwned: {
      unsigned cnt = X.getCount();
      if (cnt) {
        X.setCount(cnt - 1);
        X = X ^ RefVal::ReturnedOwned;
      }
      else {
        X = X ^ RefVal::ReturnedNotOwned;
      }
      break;
    }

    default:
      return;
  }

  // Update the binding.
  state = state->set<RefBindings>(Sym, X);
  ExplodedNode *Pred = C.generateNode(state);

  // At this point we have updated the state properly.
  // Everything after this is merely checking to see if the return value has
  // been over- or under-retained.

  // Did we cache out?
  if (!Pred)
    return;

  ExprEngine &Eng = C.getEngine();
  // FIXME: This goes away once HandleAutoreleaseCount() and the
  // RetainSummariesManager move to RetainReleaseChecker.
  CFRefCount &TF = static_cast<CFRefCount&>(Eng.getTF());

  // Update the autorelease counts.
  static SimpleProgramPointTag
         AutoreleaseTag("RetainReleaseChecker : Autorelease");
  GenericNodeBuilderRefCount Bd(C.getNodeBuilder(), S, &AutoreleaseTag);
  bool stop = false;
  llvm::tie(Pred, state) = TF.HandleAutoreleaseCounts(state, Bd, Pred, Eng, Sym,
                                                      X, stop);

  // Did we cache out?
  if (!Pred || stop)
    return;

  // Get the updated binding.
  T = state->get<RefBindings>(Sym);
  assert(T);
  X = *T;

  // Consult the summary of the enclosing method.
  const Decl *CD = &Pred->getCodeDecl();

  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(CD)) {
    // Unlike regular functions, /all/ ObjC methods are assumed to always
    // follow Cocoa retain-count conventions, not just those with special
    // names or attributes.
    const RetainSummary *Summ = TF.Summaries.getMethodSummary(MD);
    RetEffect RE = Summ ? Summ->getRetEffect() : RetEffect::MakeNoRet();
    checkReturnWithRetEffect(S, C, Pred, RE, X, Sym, state);
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CD)) {
    if (!isa<CXXMethodDecl>(FD))
      if (const RetainSummary *Summ = TF.Summaries.getSummary(FD))
        checkReturnWithRetEffect(S, C, Pred, Summ->getRetEffect(), X,
                                 Sym, state);
  }
}

void RetainReleaseChecker::checkReturnWithRetEffect(const ReturnStmt *S,
                                                    CheckerContext &C,
                                                    ExplodedNode *Pred,
                                                    RetEffect RE, RefVal X,
                                                    SymbolRef Sym,
                                              const ProgramState *state) const {
  // FIXME: This goes away once the these bug types move to the checker,
  // and CFRefReport no longer depends on CFRefCount.
  CFRefCount &TF = static_cast<CFRefCount&>(C.getEngine().getTF());

  // Any leaks or other errors?
  if (X.isReturnedOwned() && X.getCount() == 0) {
    if (RE.getKind() != RetEffect::NoRet) {
      bool hasError = false;
      if (TF.isGCEnabled() && RE.getObjKind() == RetEffect::ObjC) {
        // Things are more complicated with garbage collection.  If the
        // returned object is suppose to be an Objective-C object, we have
        // a leak (as the caller expects a GC'ed object) because no
        // method should return ownership unless it returns a CF object.
        hasError = true;
        X = X ^ RefVal::ErrorGCLeakReturned;
      }
      else if (!RE.isOwned()) {
        // Either we are using GC and the returned object is a CF type
        // or we aren't using GC.  In either case, we expect that the
        // enclosing method is expected to return ownership.
        hasError = true;
        X = X ^ RefVal::ErrorLeakReturned;
      }

      if (hasError) {
        // Generate an error node.
        state = state->set<RefBindings>(Sym, X);
        StmtNodeBuilder &Builder = C.getNodeBuilder();

        static SimpleProgramPointTag
               ReturnOwnLeakTag("RetainReleaseChecker : ReturnsOwnLeak");
        ExplodedNode *N = Builder.generateNode(S, state, Pred,
                                               &ReturnOwnLeakTag);
        if (N) {
          CFRefReport *report =
            new CFRefLeakReport(*static_cast<CFRefBug*>(TF.leakAtReturn), TF,
                                N, Sym, C.getEngine());
          C.EmitReport(report);
        }
      }
    }
  } else if (X.isReturnedNotOwned()) {
    if (RE.isOwned()) {
      // Trying to return a not owned object to a caller expecting an
      // owned object.
      state = state->set<RefBindings>(Sym, X ^ RefVal::ErrorReturnedNotOwned);
      StmtNodeBuilder &Builder = C.getNodeBuilder();

      static SimpleProgramPointTag
             ReturnNotOwnedTag("RetainReleaseChecker : ReturnNotOwnedForOwned");
      // FIXME: This PostStmt is a lie. But currently CFRefReport expects all
      // interesting things to happen in PostStmt nodes.
      ExplodedNode *N = Builder.generateNode(S, state, Pred, 
                                             &ReturnNotOwnedTag);
      if (N) {
        CFRefReport *report =
            new CFRefReport(*static_cast<CFRefBug*>(TF.returnNotOwnedForOwned),
                            TF, N, Sym);
        C.EmitReport(report);
      }
    }
  }
}

// Handle dead symbols (potential leaks).

const ProgramState *
RetainReleaseChecker::handleSymbolDeath(const ProgramState *state,
                                        SymbolRef sid, RefVal V,
                                    SmallVectorImpl<SymbolRef> &Leaked) const {
  bool hasLeak;
  if (V.isOwned())
    hasLeak = true;
  else if (V.isNotOwned() || V.isReturnedOwned())
    hasLeak = (V.getCount() > 0);

  if (!hasLeak)
    return state->remove<RefBindings>(sid);

  Leaked.push_back(sid);
  return state->set<RefBindings>(sid, V ^ RefVal::ErrorLeak);
}

ExplodedNode *
RetainReleaseChecker::processLeaks(const ProgramState *state,
                                   SmallVectorImpl<SymbolRef> &Leaked,
                                   GenericNodeBuilderRefCount &Builder,
                                   ExprEngine &Eng,
                                   ExplodedNode *Pred) const {

  if (Leaked.empty())
    return Pred;

  // Generate an intermediate node representing the leak point.
  ExplodedNode *N = Builder.MakeNode(state, Pred);

  if (N) {
    for (SmallVectorImpl<SymbolRef>::iterator
         I = Leaked.begin(), E = Leaked.end(); I != E; ++I) {

      // FIXME: This goes away once the these bug types move to the checker,
      // and CFRefReport no longer depends on CFRefCount.
      CFRefCount &TF = static_cast<CFRefCount&>(Eng.getTF());
      CFRefBug *BT = static_cast<CFRefBug*>(Pred ? TF.leakWithinFunction
                                                 : TF.leakAtReturn);
      assert(BT && "BugType not initialized.");
      CFRefLeakReport *report = new CFRefLeakReport(*BT, TF, N, *I, Eng);
      Eng.getBugReporter().EmitReport(report);
    }
  }

  return N;
}

void RetainReleaseChecker::checkEndPath(EndOfFunctionNodeBuilder &Builder,
                                        ExprEngine &Eng) const {
  // FIXME: This goes away once HandleAutoreleaseCounts moves to the checker.
  CFRefCount &TF = static_cast<CFRefCount&>(Eng.getTF());

  const ProgramState *state = Builder.getState();
  GenericNodeBuilderRefCount Bd(Builder);
  RefBindings B = state->get<RefBindings>();
  ExplodedNode *Pred = 0;

  for (RefBindings::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    bool stop = false;
    llvm::tie(Pred, state) = TF.HandleAutoreleaseCounts(state, Bd, Pred, Eng,
                                                        (*I).first,
                                                        (*I).second, stop);

    if (stop)
      return;
  }

  B = state->get<RefBindings>();
  SmallVector<SymbolRef, 10> Leaked;

  for (RefBindings::iterator I = B.begin(), E = B.end(); I != E; ++I)
    state = handleSymbolDeath(state, (*I).first, (*I).second, Leaked);

  processLeaks(state, Leaked, Bd, Eng, Pred);
}

const ProgramPointTag *
RetainReleaseChecker::getDeadSymbolTag(SymbolRef sym) const {
  const SimpleProgramPointTag *&tag = DeadSymbolTags[sym];
  if (!tag) {
    llvm::SmallString<64> buf;
    llvm::raw_svector_ostream out(buf);
    out << "RetainReleaseChecker : Dead Symbol : " << sym->getSymbolID();
    tag = new SimpleProgramPointTag(out.str());
  }
  return tag;  
}

void RetainReleaseChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                            CheckerContext &C) const {
  StmtNodeBuilder &Builder = C.getNodeBuilder();
  ExprEngine &Eng = C.getEngine();
  const Stmt *S = C.getStmt();
  ExplodedNode *Pred = C.getPredecessor();

  // FIXME: This goes away once HandleAutoreleaseCounts moves to the checker.
  CFRefCount &TF = static_cast<CFRefCount&>(Eng.getTF());

  const ProgramState *state = C.getState();
  RefBindings B = state->get<RefBindings>();

  // Update counts from autorelease pools
  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    if (const RefVal *T = B.lookup(Sym)){
      // Use the symbol as the tag.
      // FIXME: This might not be as unique as we would like.
      GenericNodeBuilderRefCount Bd(Builder, S, getDeadSymbolTag(Sym));
      bool stop = false;
      llvm::tie(Pred, state) = TF.HandleAutoreleaseCounts(state, Bd, Pred, Eng,
                                                          Sym, *T, stop);
      if (stop)
        return;
    }
  }

  B = state->get<RefBindings>();
  SmallVector<SymbolRef, 10> Leaked;

  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I) {
    if (const RefVal *T = B.lookup(*I))
      state = handleSymbolDeath(state, *I, *T, Leaked);
  }

  {
    GenericNodeBuilderRefCount Bd(Builder, S, this);
    Pred = processLeaks(state, Leaked, Bd, Eng, Pred);
  }

  // Did we cache out?
  if (!Pred)
    return;

  // Now generate a new node that nukes the old bindings.
  RefBindings::Factory &F = state->get_context<RefBindings>();

  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I)
    B = F.remove(B, *I);

  state = state->set<RefBindings>(B);
  C.generateNode(state, Pred);
}

//===----------------------------------------------------------------------===//
// Transfer function creation for external clients.
//===----------------------------------------------------------------------===//

void CFRefCount::RegisterChecks(ExprEngine& Eng) {
  BugReporter &BR = Eng.getBugReporter();

  useAfterRelease = new UseAfterRelease(this);
  BR.Register(useAfterRelease);

  releaseNotOwned = new BadRelease(this);
  BR.Register(releaseNotOwned);

  deallocGC = new DeallocGC(this);
  BR.Register(deallocGC);

  deallocNotOwned = new DeallocNotOwned(this);
  BR.Register(deallocNotOwned);

  overAutorelease = new OverAutorelease(this);
  BR.Register(overAutorelease);

  returnNotOwnedForOwned = new ReturnedNotOwnedForOwned(this);
  BR.Register(returnNotOwnedForOwned);

  // First register "return" leaks.
  const char* name = 0;

  if (isGCEnabled())
    name = "Leak of returned object when using garbage collection";
  else if (getLangOptions().getGCMode() == LangOptions::HybridGC)
    name = "Leak of returned object when not using garbage collection (GC) in "
    "dual GC/non-GC code";
  else {
    assert(getLangOptions().getGCMode() == LangOptions::NonGC);
    name = "Leak of returned object";
  }

  // Leaks should not be reported if they are post-dominated by a sink.
  leakAtReturn = new LeakAtReturn(this, name);
  leakAtReturn->setSuppressOnSink(true);
  BR.Register(leakAtReturn);

  // Second, register leaks within a function/method.
  if (isGCEnabled())
    name = "Leak of object when using garbage collection";
  else if (getLangOptions().getGCMode() == LangOptions::HybridGC)
    name = "Leak of object when not using garbage collection (GC) in "
    "dual GC/non-GC code";
  else {
    assert(getLangOptions().getGCMode() == LangOptions::NonGC);
    name = "Leak";
  }

  // Leaks should not be reported if they are post-dominated by sinks.
  leakWithinFunction = new LeakWithinFunction(this, name);
  leakWithinFunction->setSuppressOnSink(true);
  BR.Register(leakWithinFunction);

  // Save the reference to the BugReporter.
  this->BR = &BR;

  // Register the RetainReleaseChecker with the ExprEngine object.
  // Functionality in CFRefCount will be migrated to RetainReleaseChecker
  // over time.
  // FIXME: HACK! Remove TransferFuncs and turn all of CFRefCount into fully
  // using the checker mechanism.
  Eng.getCheckerManager().registerChecker<RetainReleaseChecker>();
}

TransferFuncs* ento::MakeCFRefCountTF(ASTContext &Ctx, bool GCEnabled,
                                         const LangOptions& lopts) {
  return new CFRefCount(Ctx, GCEnabled, lopts);
}
