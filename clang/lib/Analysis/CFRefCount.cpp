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

#include "GRSimpleVals.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/STLExtras.h"
#include <ostream>
#include <sstream>

using namespace clang;
using llvm::CStrInCStrNoCase;

//===----------------------------------------------------------------------===//
// Selector creation functions.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}

static inline Selector GetUnarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(1, &II);
}

//===----------------------------------------------------------------------===//
// Type querying functions.
//===----------------------------------------------------------------------===//

static bool isCFRefType(QualType T) {
  
  if (!T->isPointerType())
    return false;
  
  // Check the typedef for the name "CF" and the substring "Ref".  
  TypedefType* TD = dyn_cast<TypedefType>(T.getTypePtr());
  
  if (!TD)
    return false;
  
  const char* TDName = TD->getDecl()->getIdentifier()->getName();
  assert (TDName);
  
  if (TDName[0] != 'C' || TDName[1] != 'F')
    return false;
  
  if (strstr(TDName, "Ref") == 0)
    return false;
  
  return true;
}

static bool isCGRefType(QualType T) {
  
  if (!T->isPointerType())
    return false;
  
  // Check the typedef for the name "CG" and the substring "Ref".  
  TypedefType* TD = dyn_cast<TypedefType>(T.getTypePtr());
  
  if (!TD)
    return false;
  
  const char* TDName = TD->getDecl()->getIdentifier()->getName();
  assert (TDName);
  
  if (TDName[0] != 'C' || TDName[1] != 'G')
    return false;
  
  if (strstr(TDName, "Ref") == 0)
    return false;
  
  return true;
}

static bool isNSType(QualType T) {
  
  if (!T->isPointerType())
    return false;
  
  ObjCInterfaceType* OT = dyn_cast<ObjCInterfaceType>(T.getTypePtr());
  
  if (!OT)
    return false;
  
  const char* ClsName = OT->getDecl()->getIdentifier()->getName();
  assert (ClsName);
  
  if (ClsName[0] != 'N' || ClsName[1] != 'S')
    return false;
  
  return true;
}

//===----------------------------------------------------------------------===//
// Primitives used for constructing summaries for function/method calls.
//===----------------------------------------------------------------------===//

namespace {
/// ArgEffect is used to summarize a function/method call's effect on a
/// particular argument.
enum ArgEffect { IncRef, DecRef, DoNothing, DoNothingByRef,
                 StopTracking, MayEscape, SelfOwn, Autorelease };

/// ArgEffects summarizes the effects of a function/method call on all of
/// its arguments.
typedef std::vector<std::pair<unsigned,ArgEffect> > ArgEffects;
}

namespace llvm {
template <> struct FoldingSetTrait<ArgEffects> {
  static void Profile(const ArgEffects& X, FoldingSetNodeID& ID) {
    for (ArgEffects::const_iterator I = X.begin(), E = X.end(); I!= E; ++I) {
      ID.AddInteger(I->first);
      ID.AddInteger((unsigned) I->second);
    }
  }    
};
} // end llvm namespace

namespace {

///  RetEffect is used to summarize a function/method call's behavior with
///  respect to its return value.  
class VISIBILITY_HIDDEN RetEffect {
public:
  enum Kind { NoRet, Alias, OwnedSymbol, OwnedAllocatedSymbol,
              NotOwnedSymbol, ReceiverAlias };
  
private:
  unsigned Data;
  RetEffect(Kind k, unsigned D = 0) { Data = (D << 3) | (unsigned) k; }
  
public:
  
  Kind getKind() const { return (Kind) (Data & 0x7); }
  
  unsigned getIndex() const { 
    assert(getKind() == Alias);
    return Data >> 3;
  }
  
  static RetEffect MakeAlias(unsigned Idx) {
    return RetEffect(Alias, Idx);
  }
  static RetEffect MakeReceiverAlias() {
    return RetEffect(ReceiverAlias);
  }  
  static RetEffect MakeOwned(bool isAllocated = false) {
    return RetEffect(isAllocated ? OwnedAllocatedSymbol : OwnedSymbol);
  }  
  static RetEffect MakeNotOwned() {
    return RetEffect(NotOwnedSymbol);
  }  
  static RetEffect MakeNoRet() {
    return RetEffect(NoRet);
  }
  
  operator Kind() const {
    return getKind();
  }  
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger(Data);
  }
};
  
  
class VISIBILITY_HIDDEN RetainSummary : public llvm::FoldingSetNode {
  /// Args - an ordered vector of (index, ArgEffect) pairs, where index
  ///  specifies the argument (starting from 0).  This can be sparsely
  ///  populated; arguments with no entry in Args use 'DefaultArgEffect'.
  ArgEffects* Args;
  
  /// DefaultArgEffect - The default ArgEffect to apply to arguments that
  ///  do not have an entry in Args.
  ArgEffect   DefaultArgEffect;
  
  /// Receiver - If this summary applies to an Objective-C message expression,
  ///  this is the effect applied to the state of the receiver.
  ArgEffect   Receiver;
  
  /// Ret - The effect on the return value.  Used to indicate if the
  ///  function/method call returns a new tracked symbol, returns an
  ///  alias of one of the arguments in the call, and so on.
  RetEffect   Ret;
  
  /// EndPath - Indicates that execution of this method/function should
  ///  terminate the simulation of a path.
  bool EndPath;
  
public:
  
  RetainSummary(ArgEffects* A, RetEffect R, ArgEffect defaultEff,
                ArgEffect ReceiverEff, bool endpath = false)
    : Args(A), DefaultArgEffect(defaultEff), Receiver(ReceiverEff), Ret(R),
      EndPath(endpath) {}  
  
  /// getArg - Return the argument effect on the argument specified by
  ///  idx (starting from 0).
  ArgEffect getArg(unsigned idx) const {

    if (!Args)
      return DefaultArgEffect;
    
    // If Args is present, it is likely to contain only 1 element.
    // Just do a linear search.  Do it from the back because functions with
    // large numbers of arguments will be tail heavy with respect to which
    // argument they actually modify with respect to the reference count.    
    for (ArgEffects::reverse_iterator I=Args->rbegin(), E=Args->rend();
           I!=E; ++I) {
      
      if (idx > I->first)
        return DefaultArgEffect;
      
      if (idx == I->first)
        return I->second;
    }
    
    return DefaultArgEffect;
  }
  
  /// getRetEffect - Returns the effect on the return value of the call.
  RetEffect getRetEffect() const {
    return Ret;
  }
  
  /// isEndPath - Returns true if executing the given method/function should
  ///  terminate the path.
  bool isEndPath() const { return EndPath; }
  
  /// getReceiverEffect - Returns the effect on the receiver of the call.
  ///  This is only meaningful if the summary applies to an ObjCMessageExpr*.
  ArgEffect getReceiverEffect() const {
    return Receiver;
  }
  
  typedef ArgEffects::const_iterator ExprIterator;
  
  ExprIterator begin_args() const { return Args->begin(); }
  ExprIterator end_args()   const { return Args->end(); }
  
  static void Profile(llvm::FoldingSetNodeID& ID, ArgEffects* A,
                      RetEffect RetEff, ArgEffect DefaultEff,
                      ArgEffect ReceiverEff, bool EndPath) {
    ID.AddPointer(A);
    ID.Add(RetEff);
    ID.AddInteger((unsigned) DefaultEff);
    ID.AddInteger((unsigned) ReceiverEff);
    ID.AddInteger((unsigned) EndPath);
  }
      
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, Args, Ret, DefaultArgEffect, Receiver, EndPath);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Data structures for constructing summaries.
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN ObjCSummaryKey {
  IdentifierInfo* II;
  Selector S;
public:    
  ObjCSummaryKey(IdentifierInfo* ii, Selector s)
    : II(ii), S(s) {}

  ObjCSummaryKey(ObjCInterfaceDecl* d, Selector s)
    : II(d ? d->getIdentifier() : 0), S(s) {}
  
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
  
  static bool isPod() {
    return DenseMapInfo<ObjCInterfaceDecl*>::isPod() &&
           DenseMapInfo<Selector>::isPod();
  }
};
} // end llvm namespace
  
namespace {
class VISIBILITY_HIDDEN ObjCSummaryCache {
  typedef llvm::DenseMap<ObjCSummaryKey, RetainSummary*> MapTy;
  MapTy M;
public:
  ObjCSummaryCache() {}
  
  typedef MapTy::iterator iterator;
  
  iterator find(ObjCInterfaceDecl* D, Selector S) {
    
    // Do a lookup with the (D,S) pair.  If we find a match return
    // the iterator.
    ObjCSummaryKey K(D, S);
    MapTy::iterator I = M.find(K);
    
    if (I != M.end() || !D)
      return I;
    
    // Walk the super chain.  If we find a hit with a parent, we'll end
    // up returning that summary.  We actually allow that key (null,S), as
    // we cache summaries for the null ObjCInterfaceDecl* to allow us to
    // generate initial summaries without having to worry about NSObject
    // being declared.
    // FIXME: We may change this at some point.
    for (ObjCInterfaceDecl* C=D->getSuperClass() ;; C=C->getSuperClass()) {
      if ((I = M.find(ObjCSummaryKey(C, S))) != M.end())
        break;
      
      if (!C)
        return I;
    }
    
    // Cache the summary with original key to make the next lookup faster 
    // and return the iterator.
    M[K] = I->second;
    return I;
  }
  
  
  iterator find(Expr* Receiver, Selector S) {
    return find(getReceiverDecl(Receiver), S);
  }
  
  iterator find(IdentifierInfo* II, Selector S) {
    // FIXME: Class method lookup.  Right now we dont' have a good way
    // of going between IdentifierInfo* and the class hierarchy.
    iterator I = M.find(ObjCSummaryKey(II, S));
    return I == M.end() ? M.find(ObjCSummaryKey(S)) : I;
  }
  
  ObjCInterfaceDecl* getReceiverDecl(Expr* E) {
    
    const PointerType* PT = E->getType()->getAsPointerType();
    if (!PT) return 0;
    
    ObjCInterfaceType* OI = dyn_cast<ObjCInterfaceType>(PT->getPointeeType());
    if (!OI) return 0;
    
    return OI ? OI->getDecl() : 0;
  }
  
  iterator end() { return M.end(); }
  
  RetainSummary*& operator[](ObjCMessageExpr* ME) {
    
    Selector S = ME->getSelector();
    
    if (Expr* Receiver = ME->getReceiver()) {
      ObjCInterfaceDecl* OD = getReceiverDecl(Receiver);
      return OD ? M[ObjCSummaryKey(OD->getIdentifier(), S)] : M[S];
    }
    
    return M[ObjCSummaryKey(ME->getClassName(), S)];
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
class VISIBILITY_HIDDEN RetainSummaryManager {

  //==-----------------------------------------------------------------==//
  //  Typedefs.
  //==-----------------------------------------------------------------==//
  
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<ArgEffects> >
          ArgEffectsSetTy;
  
  typedef llvm::FoldingSet<RetainSummary>
          SummarySetTy;
  
  typedef llvm::DenseMap<FunctionDecl*, RetainSummary*>
          FuncSummariesTy;
  
  typedef ObjCSummaryCache ObjCMethodSummariesTy;
    
  //==-----------------------------------------------------------------==//
  //  Data.
  //==-----------------------------------------------------------------==//
  
  /// Ctx - The ASTContext object for the analyzed ASTs.
  ASTContext& Ctx;
  
  /// NSWindowII - An IdentifierInfo* representing the identifier "NSWindow."
  IdentifierInfo* NSWindowII;

  /// NSPanelII - An IdentifierInfo* representing the identifier "NSPanel."
  IdentifierInfo* NSPanelII;
  
  /// NSAssertionHandlerII - An IdentifierInfo* representing the identifier
  //  "NSAssertionHandler".
  IdentifierInfo* NSAssertionHandlerII;
  
  /// CFDictionaryCreateII - An IdentifierInfo* representing the indentifier
  ///  "CFDictionaryCreate".
  IdentifierInfo* CFDictionaryCreateII;
  
  /// GCEnabled - Records whether or not the analyzed code runs in GC mode.
  const bool GCEnabled;
  
  /// SummarySet - A FoldingSet of uniqued summaries.
  SummarySetTy SummarySet;
  
  /// FuncSummaries - A map from FunctionDecls to summaries.
  FuncSummariesTy FuncSummaries; 
  
  /// ObjCClassMethodSummaries - A map from selectors (for instance methods)
  ///  to summaries.
  ObjCMethodSummariesTy ObjCClassMethodSummaries;

  /// ObjCMethodSummaries - A map from selectors to summaries.
  ObjCMethodSummariesTy ObjCMethodSummaries;

  /// ArgEffectsSet - A FoldingSet of uniqued ArgEffects.
  ArgEffectsSetTy ArgEffectsSet;
  
  /// BPAlloc - A BumpPtrAllocator used for allocating summaries, ArgEffects,
  ///  and all other data used by the checker.
  llvm::BumpPtrAllocator BPAlloc;
  
  /// ScratchArgs - A holding buffer for construct ArgEffects.
  ArgEffects ScratchArgs;
  
  RetainSummary* StopSummary;
  
  //==-----------------------------------------------------------------==//
  //  Methods.
  //==-----------------------------------------------------------------==//
  
  /// getArgEffects - Returns a persistent ArgEffects object based on the
  ///  data in ScratchArgs.
  ArgEffects*   getArgEffects();

  enum UnaryFuncKind { cfretain, cfrelease, cfmakecollectable };  
  RetainSummary* getUnarySummary(FunctionDecl* FD, UnaryFuncKind func);
  
  RetainSummary* getNSSummary(FunctionDecl* FD, const char* FName);
  RetainSummary* getCFSummary(FunctionDecl* FD, const char* FName);
  RetainSummary* getCGSummary(FunctionDecl* FD, const char* FName);
  
  RetainSummary* getCFSummaryCreateRule(FunctionDecl* FD);
  RetainSummary* getCFSummaryGetRule(FunctionDecl* FD);  
  RetainSummary* getCFCreateGetRuleSummary(FunctionDecl* FD, const char* FName);  
  
  RetainSummary* getPersistentSummary(ArgEffects* AE, RetEffect RetEff,
                                      ArgEffect ReceiverEff = DoNothing,
                                      ArgEffect DefaultEff = MayEscape,
                                      bool isEndPath = false);
                 

  RetainSummary* getPersistentSummary(RetEffect RE,
                                      ArgEffect ReceiverEff = DoNothing,
                                      ArgEffect DefaultEff = MayEscape) {
    return getPersistentSummary(getArgEffects(), RE, ReceiverEff, DefaultEff);
  }
  
  
  RetainSummary* getPersistentStopSummary() {
    if (StopSummary)
      return StopSummary;
    
    StopSummary = getPersistentSummary(RetEffect::MakeNoRet(),
                                       StopTracking, StopTracking);
    
    return StopSummary;
  }  

  RetainSummary* getInitMethodSummary(ObjCMessageExpr* ME);

  void InitializeClassMethodSummaries();
  void InitializeMethodSummaries();
      
  void addClsMethSummary(IdentifierInfo* ClsII, Selector S,
                         RetainSummary* Summ) {
    ObjCClassMethodSummaries[ObjCSummaryKey(ClsII, S)] = Summ;
  }
  
  void addNSObjectClsMethSummary(Selector S, RetainSummary *Summ) {
    ObjCClassMethodSummaries[S] = Summ;
  }
    
  void addNSObjectMethSummary(Selector S, RetainSummary *Summ) {
    ObjCMethodSummaries[S] = Summ;
  }
  
  void addNSWindowMethSummary(Selector S, RetainSummary *Summ) {
    ObjCMethodSummaries[ObjCSummaryKey(NSWindowII, S)] = Summ;
  }
  
  void addNSPanelMethSummary(Selector S, RetainSummary *Summ) {
    ObjCMethodSummaries[ObjCSummaryKey(NSPanelII, S)] = Summ;
  }
  
  void addPanicSummary(IdentifierInfo* ClsII, Selector S) {
    RetainSummary* Summ = getPersistentSummary(0, RetEffect::MakeNoRet(),
                                               DoNothing,  DoNothing, true);
    
    ObjCMethodSummaries[ObjCSummaryKey(ClsII, S)] = Summ;
  }
  
public:
  
  RetainSummaryManager(ASTContext& ctx, bool gcenabled)
   : Ctx(ctx),
     NSWindowII(&ctx.Idents.get("NSWindow")),
     NSPanelII(&ctx.Idents.get("NSPanel")),
     NSAssertionHandlerII(&ctx.Idents.get("NSAssertionHandler")),
     CFDictionaryCreateII(&ctx.Idents.get("CFDictionaryCreate")),
     GCEnabled(gcenabled), StopSummary(0) {

    InitializeClassMethodSummaries();
    InitializeMethodSummaries();
  }
  
  ~RetainSummaryManager();
  
  RetainSummary* getSummary(FunctionDecl* FD);  
  RetainSummary* getMethodSummary(ObjCMessageExpr* ME, ObjCInterfaceDecl* ID);
  RetainSummary* getClassMethodSummary(IdentifierInfo* ClsName, Selector S);
  
  bool isGCEnabled() const { return GCEnabled; }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of checker data structures.
//===----------------------------------------------------------------------===//

RetainSummaryManager::~RetainSummaryManager() {
  
  // FIXME: The ArgEffects could eventually be allocated from BPAlloc, 
  //   mitigating the need to do explicit cleanup of the
  //   Argument-Effect summaries.
  
  for (ArgEffectsSetTy::iterator I = ArgEffectsSet.begin(), 
                                 E = ArgEffectsSet.end(); I!=E; ++I)
    I->getValue().~ArgEffects();
}

ArgEffects* RetainSummaryManager::getArgEffects() {

  if (ScratchArgs.empty())
    return NULL;
  
  // Compute a profile for a non-empty ScratchArgs.
  llvm::FoldingSetNodeID profile;
  profile.Add(ScratchArgs);
  void* InsertPos;
  
  // Look up the uniqued copy, or create a new one.
  llvm::FoldingSetNodeWrapper<ArgEffects>* E =
    ArgEffectsSet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (E) {
    ScratchArgs.clear();
    return &E->getValue();
  }
  
  E = (llvm::FoldingSetNodeWrapper<ArgEffects>*)
        BPAlloc.Allocate<llvm::FoldingSetNodeWrapper<ArgEffects> >();
                       
  new (E) llvm::FoldingSetNodeWrapper<ArgEffects>(ScratchArgs);
  ArgEffectsSet.InsertNode(E, InsertPos);

  ScratchArgs.clear();
  return &E->getValue();
}

RetainSummary*
RetainSummaryManager::getPersistentSummary(ArgEffects* AE, RetEffect RetEff,
                                           ArgEffect ReceiverEff,
                                           ArgEffect DefaultEff,
                                           bool isEndPath) {
  
  // Generate a profile for the summary.
  llvm::FoldingSetNodeID profile;
  RetainSummary::Profile(profile, AE, RetEff, DefaultEff, ReceiverEff,
                         isEndPath);
  
  // Look up the uniqued summary, or create one if it doesn't exist.
  void* InsertPos;  
  RetainSummary* Summ = SummarySet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (Summ)
    return Summ;
  
  // Create the summary and return it.
  Summ = (RetainSummary*) BPAlloc.Allocate<RetainSummary>();
  new (Summ) RetainSummary(AE, RetEff, DefaultEff, ReceiverEff, isEndPath);
  SummarySet.InsertNode(Summ, InsertPos);
  
  return Summ;
}

//===----------------------------------------------------------------------===//
// Summary creation for functions (largely uses of Core Foundation).
//===----------------------------------------------------------------------===//

RetainSummary* RetainSummaryManager::getSummary(FunctionDecl* FD) {

  SourceLocation Loc = FD->getLocation();
  
  if (!Loc.isFileID())
    return NULL;
  
  // Look up a summary in our cache of FunctionDecls -> Summaries.
  FuncSummariesTy::iterator I = FuncSummaries.find(FD);

  if (I != FuncSummaries.end())
    return I->second;

  // No summary.  Generate one.
  const char* FName = FD->getIdentifier()->getName();
    
  RetainSummary *S = 0;
  
  FunctionType* FT = dyn_cast<FunctionType>(FD->getType());

  do {
    if (FT) {
      
      QualType T = FT->getResultType();
      
      if (isCFRefType(T)) {
        S = getCFSummary(FD, FName);
        break;
      }
      
      if (isCGRefType(T)) {
        S = getCGSummary(FD, FName );
        break;
      }
    }

    if (FName[0] == 'C' && FName[1] == 'F')
      S = getCFSummary(FD, FName);  
    else if (FName[0] == 'N' && FName[1] == 'S')
      S = getNSSummary(FD, FName);
  }
  while (0);

  FuncSummaries[FD] = S;
  return S;  
}

RetainSummary* RetainSummaryManager::getNSSummary(FunctionDecl* FD,
                                                  const char* FName) {
  FName += 2;
  
  if (strcmp(FName, "MakeCollectable") == 0)
    return getUnarySummary(FD, cfmakecollectable);

  return 0;  
}

static bool isRetain(FunctionDecl* FD, const char* FName) {
  const char* loc = strstr(FName, "Retain");
  return loc && loc[sizeof("Retain")-1] == '\0';
}

static bool isRelease(FunctionDecl* FD, const char* FName) {
  const char* loc = strstr(FName, "Release");
  return loc && loc[sizeof("Release")-1] == '\0';
}

RetainSummary* RetainSummaryManager::getCFSummary(FunctionDecl* FD,
                                                  const char* FName) {

  if (FName[0] == 'C' && FName[1] == 'F')
    FName += 2;

  if (isRetain(FD, FName))
    return getUnarySummary(FD, cfretain);
  
  if (isRelease(FD, FName))
    return getUnarySummary(FD, cfrelease);
  
  if (strcmp(FName, "MakeCollectable") == 0)
    return getUnarySummary(FD, cfmakecollectable);

  return getCFCreateGetRuleSummary(FD, FName);
}

RetainSummary* RetainSummaryManager::getCGSummary(FunctionDecl* FD,
                                                  const char* FName) {
  
  if (FName[0] == 'C' && FName[1] == 'G')
    FName += 2;
  
  if (isRelease(FD, FName))
    return getUnarySummary(FD, cfrelease);
  
  if (isRetain(FD, FName))
    return getUnarySummary(FD, cfretain);
  
  return getCFCreateGetRuleSummary(FD, FName);
}

RetainSummary*
RetainSummaryManager::getCFCreateGetRuleSummary(FunctionDecl* FD,
                                                const char* FName) {
  
  if (strstr(FName, "Create") || strstr(FName, "Copy"))
    return getCFSummaryCreateRule(FD);
  
  if (strstr(FName, "Get"))
    return getCFSummaryGetRule(FD);
  
  return 0;
}

RetainSummary*
RetainSummaryManager::getUnarySummary(FunctionDecl* FD, UnaryFuncKind func) {
  
  FunctionTypeProto* FT =
    dyn_cast<FunctionTypeProto>(FD->getType().getTypePtr());
  
  if (FT) {
    
    if (FT->getNumArgs() != 1)
      return 0;
  
    TypedefType* ArgT = dyn_cast<TypedefType>(FT->getArgType(0).getTypePtr());
  
    if (!ArgT)
      return 0;

    if (!ArgT->isPointerType())
      return NULL;
  }
  
  assert (ScratchArgs.empty());
  
  switch (func) {
    case cfretain: {
      ScratchArgs.push_back(std::make_pair(0, IncRef));
      return getPersistentSummary(RetEffect::MakeAlias(0),
                                  DoNothing, DoNothing);
    }
      
    case cfrelease: {
      ScratchArgs.push_back(std::make_pair(0, DecRef));
      return getPersistentSummary(RetEffect::MakeNoRet(),
                                  DoNothing, DoNothing);
    }
      
    case cfmakecollectable: {
      if (GCEnabled)
        ScratchArgs.push_back(std::make_pair(0, DecRef));
      
      return getPersistentSummary(RetEffect::MakeAlias(0),
                                  DoNothing, DoNothing);    
    }
      
    default:
      assert (false && "Not a supported unary function.");
  }
}

RetainSummary* RetainSummaryManager::getCFSummaryCreateRule(FunctionDecl* FD) {
 
  FunctionType* FT =
    dyn_cast<FunctionType>(FD->getType().getTypePtr());
  
  if (FT && !isCFRefType(FT->getResultType()))
    return getPersistentSummary(RetEffect::MakeNoRet());

  assert (ScratchArgs.empty());
  
  if (FD->getIdentifier() == CFDictionaryCreateII) {
    ScratchArgs.push_back(std::make_pair(1, DoNothingByRef));
    ScratchArgs.push_back(std::make_pair(2, DoNothingByRef));
  }
  
  return getPersistentSummary(RetEffect::MakeOwned(true));
}

RetainSummary* RetainSummaryManager::getCFSummaryGetRule(FunctionDecl* FD) {
  
  FunctionType* FT =
    dyn_cast<FunctionType>(FD->getType().getTypePtr());
  
  if (FT) {
    QualType RetTy = FT->getResultType();
  
    // FIXME: For now we assume that all pointer types returned are referenced
    // counted.  Since this is the "Get" rule, we assume non-ownership, which
    // works fine for things that are not reference counted.  We do this because
    // some generic data structures return "void*".  We need something better
    // in the future.
  
    if (!isCFRefType(RetTy) && !RetTy->isPointerType())
      return getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
  }
  
  // FIXME: Add special-cases for functions that retain/release.  For now
  //  just handle the default case.
  
  assert (ScratchArgs.empty());  
  return getPersistentSummary(RetEffect::MakeNotOwned(), DoNothing, DoNothing);
}

//===----------------------------------------------------------------------===//
// Summary creation for Selectors.
//===----------------------------------------------------------------------===//

RetainSummary*
RetainSummaryManager::getInitMethodSummary(ObjCMessageExpr* ME) {
  assert(ScratchArgs.empty());
    
  RetainSummary* Summ =
    getPersistentSummary(RetEffect::MakeReceiverAlias());
  
  ObjCMethodSummaries[ME] = Summ;
  return Summ;
}


RetainSummary*
RetainSummaryManager::getMethodSummary(ObjCMessageExpr* ME,
                                       ObjCInterfaceDecl* ID) {

  Selector S = ME->getSelector();
  
  // Look up a summary in our summary cache.  
  ObjCMethodSummariesTy::iterator I = ObjCMethodSummaries.find(ID, S);
  
  if (I != ObjCMethodSummaries.end())
    return I->second;
    
  if (!ME->getType()->isPointerType())
    return 0;
  
  // "initXXX": pass-through for receiver.

  const char* s = S.getIdentifierInfoForSlot(0)->getName();
  assert (ScratchArgs.empty());
  
  if (strncmp(s, "init", 4) == 0 || strncmp(s, "_init", 5) == 0)
    return getInitMethodSummary(ME);  
  
  // "copyXXX", "createXXX", "newXXX": allocators.  

  if (!isNSType(ME->getReceiver()->getType()))
    return 0;
  
  if (CStrInCStrNoCase(s, "create") || CStrInCStrNoCase(s, "copy")  || 
      CStrInCStrNoCase(s, "new")) {
    
    RetEffect E = isGCEnabled() ? RetEffect::MakeNoRet()
                                : RetEffect::MakeOwned(true);  

    RetainSummary* Summ = getPersistentSummary(E);
    ObjCMethodSummaries[ME] = Summ;
    return Summ;
  }
  
  return 0;
}

RetainSummary*
RetainSummaryManager::getClassMethodSummary(IdentifierInfo* ClsName,
                                            Selector S) {
  
  // FIXME: Eventually we should properly do class method summaries, but
  // it requires us being able to walk the type hierarchy.  Unfortunately,
  // we cannot do this with just an IdentifierInfo* for the class name.
  
  // Look up a summary in our cache of Selectors -> Summaries.
  ObjCMethodSummariesTy::iterator I = ObjCClassMethodSummaries.find(ClsName, S);
  
  if (I != ObjCClassMethodSummaries.end())
    return I->second;
  
  return 0;
}

void RetainSummaryManager::InitializeClassMethodSummaries() {
  
  assert (ScratchArgs.empty());
  
  RetEffect E = isGCEnabled() ? RetEffect::MakeNoRet()
                              : RetEffect::MakeOwned(true);  
  
  RetainSummary* Summ = getPersistentSummary(E);
  
  // Create the summaries for "alloc", "new", and "allocWithZone:" for
  // NSObject and its derivatives.
  addNSObjectClsMethSummary(GetNullarySelector("alloc", Ctx), Summ);
  addNSObjectClsMethSummary(GetNullarySelector("new", Ctx), Summ);
  addNSObjectClsMethSummary(GetUnarySelector("allocWithZone", Ctx), Summ);
  
  // Create the [NSAssertionHandler currentHander] summary.  
  addClsMethSummary(NSAssertionHandlerII,
                    GetNullarySelector("currentHandler", Ctx),
                    getPersistentSummary(RetEffect::MakeNotOwned()));  
}

void RetainSummaryManager::InitializeMethodSummaries() {
  
  assert (ScratchArgs.empty());  
  
  // Create the "init" selector.  It just acts as a pass-through for the
  // receiver.
  RetainSummary* InitSumm = getPersistentSummary(RetEffect::MakeReceiverAlias());
  addNSObjectMethSummary(GetNullarySelector("init", Ctx), InitSumm);
  
  // The next methods are allocators.
  RetEffect E = isGCEnabled() ? RetEffect::MakeNoRet()
                              : RetEffect::MakeOwned(true);
  
  RetainSummary* Summ = getPersistentSummary(E);  
  
  // Create the "copy" selector.  
  addNSObjectMethSummary(GetNullarySelector("copy", Ctx), Summ);
  
  // Create the "mutableCopy" selector.
  addNSObjectMethSummary(GetNullarySelector("mutableCopy", Ctx), Summ);

  // Create the "retain" selector.
  E = RetEffect::MakeReceiverAlias();
  Summ = getPersistentSummary(E, isGCEnabled() ? DoNothing : IncRef);
  addNSObjectMethSummary(GetNullarySelector("retain", Ctx), Summ);
  
  // Create the "release" selector.
  Summ = getPersistentSummary(E, isGCEnabled() ? DoNothing : DecRef);
  addNSObjectMethSummary(GetNullarySelector("release", Ctx), Summ);
  
  // Create the "drain" selector.
  Summ = getPersistentSummary(E, isGCEnabled() ? DoNothing : DecRef);
  addNSObjectMethSummary(GetNullarySelector("drain", Ctx), Summ);

  // Create the "autorelease" selector.
  Summ = getPersistentSummary(E, isGCEnabled() ? DoNothing : Autorelease);
  addNSObjectMethSummary(GetNullarySelector("autorelease", Ctx), Summ);

  // For NSWindow, allocated objects are (initially) self-owned.
  // For NSPanel (which subclasses NSWindow), allocated objects are not
  //  self-owned.
  
  RetainSummary *NSWindowSumm =
    getPersistentSummary(RetEffect::MakeReceiverAlias(), SelfOwn);

  // Create the "initWithContentRect:styleMask:backing:defer:" selector.
  llvm::SmallVector<IdentifierInfo*, 10> II;
  II.push_back(&Ctx.Idents.get("initWithContentRect"));
  II.push_back(&Ctx.Idents.get("styleMask"));
  II.push_back(&Ctx.Idents.get("backing"));
  II.push_back(&Ctx.Idents.get("defer"));  
  Selector S = Ctx.Selectors.getSelector(II.size(), &II[0]);      
  addNSWindowMethSummary(S, NSWindowSumm);
  addNSPanelMethSummary(S, InitSumm);
  
  // Create the "initWithContentRect:styleMask:backing:defer:screen:" selector.
  II.push_back(&Ctx.Idents.get("screen"));
  S = Ctx.Selectors.getSelector(II.size(), &II[0]);
  addNSWindowMethSummary(S, NSWindowSumm);
  addNSPanelMethSummary(S, InitSumm);
  
  // Create NSAssertionHandler summaries.
  II.clear();
  II.push_back(&Ctx.Idents.get("handleFailureInFunction"));
  II.push_back(&Ctx.Idents.get("file"));
  II.push_back(&Ctx.Idents.get("lineNumber"));
  II.push_back(&Ctx.Idents.get("description"));
  S = Ctx.Selectors.getSelector(II.size(), &II[0]);
  addPanicSummary(NSAssertionHandlerII, S);
  
  II.clear();
  II.push_back(&Ctx.Idents.get("handleFailureInMethod"));
  II.push_back(&Ctx.Idents.get("object"));
  II.push_back(&Ctx.Idents.get("file"));
  II.push_back(&Ctx.Idents.get("lineNumber"));
  II.push_back(&Ctx.Idents.get("description"));
  S = Ctx.Selectors.getSelector(II.size(), &II[0]);
  addPanicSummary(NSAssertionHandlerII, S);
}

//===----------------------------------------------------------------------===//
// Reference-counting logic (typestate + counts).
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN RefVal {
public:  
  
  enum Kind {
    Owned = 0, // Owning reference.    
    NotOwned,  // Reference is not owned by still valid (not freed).    
    Released,  // Object has been released.
    ReturnedOwned, // Returned object passes ownership to caller.
    ReturnedNotOwned, // Return object does not pass ownership to caller.
    ErrorUseAfterRelease, // Object used after released.    
    ErrorReleaseNotOwned, // Release of an object that was not owned.
    ErrorLeak  // A memory leak due to excessive reference counts.
  };
  
private:
  
  Kind kind;
  unsigned Cnt;
  QualType T;

  RefVal(Kind k, unsigned cnt, QualType t) : kind(k), Cnt(cnt), T(t) {}
  RefVal(Kind k, unsigned cnt = 0) : kind(k), Cnt(cnt) {}

public:  
  
  Kind getKind() const { return kind; }

  unsigned getCount() const { return Cnt; }  
  QualType getType() const { return T; }
  
  // Useful predicates.
  
  static bool isError(Kind k) { return k >= ErrorUseAfterRelease; }
  
  static bool isLeak(Kind k) { return k == ErrorLeak; }
  
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
  
  bool isNonLeakError() const {
    Kind k = getKind();
    return isError(k) && !isLeak(k);
  }
  
  // State creation: normal state.
  
  static RefVal makeOwned(QualType t, unsigned Count = 1) {
    return RefVal(Owned, Count, t);
  }
  
  static RefVal makeNotOwned(QualType t, unsigned Count = 0) {
    return RefVal(NotOwned, Count, t);
  }

  static RefVal makeReturnedOwned(unsigned Count) {
    return RefVal(ReturnedOwned, Count);
  }
  
  static RefVal makeReturnedNotOwned() {
    return RefVal(ReturnedNotOwned);
  }
  
  // State creation: errors.

#if 0
  static RefVal makeLeak(unsigned Count) { return RefVal(ErrorLeak, Count); }  
  static RefVal makeReleased() { return RefVal(Released); }
  static RefVal makeUseAfterRelease() { return RefVal(ErrorUseAfterRelease); }
  static RefVal makeReleaseNotOwned() { return RefVal(ErrorReleaseNotOwned); }
#endif
  
  // Comparison, profiling, and pretty-printing.
  
  bool operator==(const RefVal& X) const {
    return kind == X.kind && Cnt == X.Cnt && T == X.T;
  }
  
  RefVal operator-(size_t i) const {
    return RefVal(getKind(), getCount() - i, getType());
  }
  
  RefVal operator+(size_t i) const {
    return RefVal(getKind(), getCount() + i, getType());
  }
  
  RefVal operator^(Kind k) const {
    return RefVal(k, getCount(), getType());
  }
    
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) kind);
    ID.AddInteger(Cnt);
    ID.Add(T);
  }

  void print(std::ostream& Out) const;
};
  
void RefVal::print(std::ostream& Out) const {
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
      
    case ErrorLeak:
      Out << "Leaked";
      break;            
      
    case ErrorUseAfterRelease:
      Out << "Use-After-Release [ERROR]";
      break;
      
    case ErrorReleaseNotOwned:
      Out << "Release of Not-Owned [ERROR]";
      break;
  }
}
  
//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN CFRefCount : public GRSimpleVals {
public:
  // Type definitions.  
  typedef llvm::ImmutableMap<SymbolID, RefVal> RefBindings;
  
  typedef RefBindings::Factory RefBFactoryTy;
  
  typedef llvm::DenseMap<GRExprEngine::NodeTy*,std::pair<Expr*, SymbolID> >
          ReleasesNotOwnedTy;
  
  typedef ReleasesNotOwnedTy UseAfterReleasesTy;
    
  typedef llvm::DenseMap<GRExprEngine::NodeTy*, std::vector<SymbolID>*>
          LeaksTy;

  class BindingsPrinter : public ValueState::CheckerStatePrinter {
  public:
    virtual void PrintCheckerState(std::ostream& Out, void* State,
                                   const char* nl, const char* sep);
  };

private:
  // Instance variables.
  
  RetainSummaryManager Summaries;  
  const LangOptions&   LOpts;
  RefBFactoryTy        RefBFactory;
     
  UseAfterReleasesTy UseAfterReleases;
  ReleasesNotOwnedTy ReleasesNotOwned;
  LeaksTy            Leaks;
  
  BindingsPrinter Printer;
  
  Selector RetainSelector;
  Selector ReleaseSelector;
  Selector AutoreleaseSelector;

public:
  
  static RefBindings GetRefBindings(const ValueState& StImpl) {
    return RefBindings((const RefBindings::TreeTy*) StImpl.CheckerState);
  }

private:
  
  static void SetRefBindings(ValueState& StImpl, RefBindings B) {
    StImpl.CheckerState = B.getRoot();
  }

  RefBindings Remove(RefBindings B, SymbolID sym) {
    return RefBFactory.Remove(B, sym);
  }
  
  RefBindings Update(RefBindings B, SymbolID sym, RefVal V, ArgEffect E,
                     RefVal::Kind& hasErr);
  
  void ProcessNonLeakError(ExplodedNodeSet<ValueState>& Dst,
                           GRStmtNodeBuilder<ValueState>& Builder,
                           Expr* NodeExpr, Expr* ErrorExpr,                        
                           ExplodedNode<ValueState>* Pred,
                           const ValueState* St,
                           RefVal::Kind hasErr, SymbolID Sym);
  
  const ValueState* HandleSymbolDeath(ValueStateManager& VMgr,
                                      const ValueState* St,
                                      SymbolID sid, RefVal V, bool& hasLeak);
  
  const ValueState* NukeBinding(ValueStateManager& VMgr, const ValueState* St,
                                SymbolID sid);
  
public:
  
  CFRefCount(ASTContext& Ctx, bool gcenabled, const LangOptions& lopts)
    : Summaries(Ctx, gcenabled),
      LOpts(lopts),
      RetainSelector(GetNullarySelector("retain", Ctx)),
      ReleaseSelector(GetNullarySelector("release", Ctx)),
      AutoreleaseSelector(GetNullarySelector("autorelease", Ctx)) {}
  
  virtual ~CFRefCount() {
    for (LeaksTy::iterator I = Leaks.begin(), E = Leaks.end(); I!=E; ++I)
      delete I->second;
  }
  
  virtual void RegisterChecks(GRExprEngine& Eng);
 
  virtual ValueState::CheckerStatePrinter* getCheckerStatePrinter() {
    return &Printer;
  }
  
  bool isGCEnabled() const { return Summaries.isGCEnabled(); }
  const LangOptions& getLangOptions() const { return LOpts; }
  
  // Calls.

  void EvalSummary(ExplodedNodeSet<ValueState>& Dst,
                   GRExprEngine& Eng,
                   GRStmtNodeBuilder<ValueState>& Builder,
                   Expr* Ex,
                   Expr* Receiver,
                   RetainSummary* Summ,
                   ExprIterator arg_beg, ExprIterator arg_end,                             
                   ExplodedNode<ValueState>* Pred);
    
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Eng,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, RVal L,
                        ExplodedNode<ValueState>* Pred);  
  
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<ValueState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<ValueState>* Pred);
  
  bool EvalObjCMessageExprAux(ExplodedNodeSet<ValueState>& Dst,
                              GRExprEngine& Engine,
                              GRStmtNodeBuilder<ValueState>& Builder,
                              ObjCMessageExpr* ME,
                              ExplodedNode<ValueState>* Pred);

  // Stores.
  
  virtual void EvalStore(ExplodedNodeSet<ValueState>& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder<ValueState>& Builder,
                         Expr* E, ExplodedNode<ValueState>* Pred,
                         const ValueState* St, RVal TargetLV, RVal Val);
  // End-of-path.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<ValueState>& Builder);
  
  virtual void EvalDeadSymbols(ExplodedNodeSet<ValueState>& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder<ValueState>& Builder,
                               ExplodedNode<ValueState>* Pred,
                               Stmt* S,
                               const ValueState* St,
                               const ValueStateManager::DeadSymbolsTy& Dead);
  // Return statements.
  
  virtual void EvalReturn(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          ReturnStmt* S,
                          ExplodedNode<ValueState>* Pred);

  // Assumptions.

  virtual const ValueState* EvalAssume(ValueStateManager& VMgr,
                                       const ValueState* St, RVal Cond,
                                       bool Assumption, bool& isFeasible);

  // Error iterators.

  typedef UseAfterReleasesTy::iterator use_after_iterator;  
  typedef ReleasesNotOwnedTy::iterator bad_release_iterator;
  typedef LeaksTy::iterator            leaks_iterator;
  
  use_after_iterator use_after_begin() { return UseAfterReleases.begin(); }
  use_after_iterator use_after_end() { return UseAfterReleases.end(); }
  
  bad_release_iterator bad_release_begin() { return ReleasesNotOwned.begin(); }
  bad_release_iterator bad_release_end() { return ReleasesNotOwned.end(); }
  
  leaks_iterator leaks_begin() { return Leaks.begin(); }
  leaks_iterator leaks_end() { return Leaks.end(); }
};

} // end anonymous namespace




void CFRefCount::BindingsPrinter::PrintCheckerState(std::ostream& Out,
                                                    void* State, const char* nl,
                                                    const char* sep) {
  RefBindings B((RefBindings::TreeTy*) State);
  
  if (State)
    Out << sep << nl;
  
  for (RefBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {
    Out << (*I).first << " : ";
    (*I).second.print(Out);
    Out << nl;
  }
}

static inline ArgEffect GetArgE(RetainSummary* Summ, unsigned idx) {
  return Summ ? Summ->getArg(idx) : MayEscape;
}

static inline RetEffect GetRetEffect(RetainSummary* Summ) {
  return Summ ? Summ->getRetEffect() : RetEffect::MakeNoRet();
}

static inline ArgEffect GetReceiverE(RetainSummary* Summ) {
  return Summ ? Summ->getReceiverEffect() : DoNothing;
}

static inline bool IsEndPath(RetainSummary* Summ) {
  return Summ ? Summ->isEndPath() : false;
}

void CFRefCount::ProcessNonLeakError(ExplodedNodeSet<ValueState>& Dst,
                                     GRStmtNodeBuilder<ValueState>& Builder,
                                     Expr* NodeExpr, Expr* ErrorExpr,                        
                                     ExplodedNode<ValueState>* Pred,
                                     const ValueState* St,
                                     RefVal::Kind hasErr, SymbolID Sym) {
  Builder.BuildSinks = true;
  GRExprEngine::NodeTy* N  = Builder.MakeNode(Dst, NodeExpr, Pred, St);

  if (!N) return;
    
  switch (hasErr) {
    default: assert(false);
    case RefVal::ErrorUseAfterRelease:
      UseAfterReleases[N] = std::make_pair(ErrorExpr, Sym);
      break;
      
    case RefVal::ErrorReleaseNotOwned:
      ReleasesNotOwned[N] = std::make_pair(ErrorExpr, Sym);
      break;
  }
}

/// GetReturnType - Used to get the return type of a message expression or
///  function call with the intention of affixing that type to a tracked symbol.
///  While the the return type can be queried directly from RetEx, when
///  invoking class methods we augment to the return type to be that of
///  a pointer to the class (as opposed it just being id).
static QualType GetReturnType(Expr* RetE, ASTContext& Ctx) {

  QualType RetTy = RetE->getType();

  // FIXME: We aren't handling id<...>.
  const PointerType* PT = RetTy.getCanonicalType()->getAsPointerType();
  
  if (!PT)
    return RetTy;
    
  // If RetEx is not a message expression just return its type.
  // If RetEx is a message expression, return its types if it is something
  /// more specific than id.
  
  ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(RetE);
  
  if (!ME || !Ctx.isObjCIdType(PT->getPointeeType()))
    return RetTy;
  
  ObjCInterfaceDecl* D = ME->getClassInfo().first;  

  // At this point we know the return type of the message expression is id.
  // If we have an ObjCInterceDecl, we know this is a call to a class method
  // whose type we can resolve.  In such cases, promote the return type to
  // Class*.  
  return !D ? RetTy : Ctx.getPointerType(Ctx.getObjCInterfaceType(D));
}


void CFRefCount::EvalSummary(ExplodedNodeSet<ValueState>& Dst,
                             GRExprEngine& Eng,
                             GRStmtNodeBuilder<ValueState>& Builder,
                             Expr* Ex,
                             Expr* Receiver,
                             RetainSummary* Summ,
                             ExprIterator arg_beg, ExprIterator arg_end,                             
                             ExplodedNode<ValueState>* Pred) {
  
  // Get the state.
  ValueStateManager& StateMgr = Eng.getStateManager();  
  const ValueState* St = Builder.GetState(Pred);

  // Evaluate the effect of the arguments.
  ValueState StVals = *St;
  RefVal::Kind hasErr = (RefVal::Kind) 0;
  unsigned idx = 0;
  Expr* ErrorExpr = NULL;
  SymbolID ErrorSym = 0;                                        
  
  for (ExprIterator I = arg_beg; I != arg_end; ++I, ++idx) {
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<lval::SymbolVal>(V)) {
      SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
      RefBindings B = GetRefBindings(StVals);      
      
      if (RefBindings::data_type* T = B.lookup(Sym)) {
        B = Update(B, Sym, *T, GetArgE(Summ, idx), hasErr);
        SetRefBindings(StVals, B);
        
        if (hasErr) {
          ErrorExpr = *I;
          ErrorSym = Sym;
          break;
        }
      }
    }  
    else if (isa<LVal>(V)) {
#if 0
      // Nuke all arguments passed by reference.
      StateMgr.Unbind(StVals, cast<LVal>(V));
#else
      if (lval::DeclVal* DV = dyn_cast<lval::DeclVal>(&V)) {

        if (GetArgE(Summ, idx) == DoNothingByRef)
          continue;
        
        // Invalidate the value of the variable passed by reference.
        
        // FIXME: Either this logic should also be replicated in GRSimpleVals
        //  or should be pulled into a separate "constraint engine."
        
        // FIXME: We can have collisions on the conjured symbol if the
        //  expression *I also creates conjured symbols.  We probably want
        //  to identify conjured symbols by an expression pair: the enclosing
        //  expression (the context) and the expression itself.  This should
        //  disambiguate conjured symbols. 

        // Is the invalidated variable something that we were tracking?
        RVal X = StateMgr.GetRVal(&StVals, *DV);
        
        if (isa<lval::SymbolVal>(X)) {
          SymbolID Sym = cast<lval::SymbolVal>(X).getSymbol();
          SetRefBindings(StVals,RefBFactory.Remove(GetRefBindings(StVals),Sym));
        }

        // Set the value of the variable to be a conjured symbol.
        unsigned Count = Builder.getCurrentBlockCount();
        SymbolID NewSym = Eng.getSymbolManager().getConjuredSymbol(*I, Count);
      
        StateMgr.SetRVal(StVals, *DV,
                         LVal::IsLValType(DV->getDecl()->getType())
                         ? cast<RVal>(lval::SymbolVal(NewSym))
                         : cast<RVal>(nonlval::SymbolVal(NewSym)));
      }
      else {
        // Nuke all other arguments passed by reference.
        StateMgr.Unbind(StVals, cast<LVal>(V));
      }
#endif
    }
    else if (isa<nonlval::LValAsInteger>(V))
      StateMgr.Unbind(StVals, cast<nonlval::LValAsInteger>(V).getLVal());
  } 
  
  // Evaluate the effect on the message receiver.  
  if (!ErrorExpr && Receiver) {
    RVal V = StateMgr.GetRVal(St, Receiver);

    if (isa<lval::SymbolVal>(V)) {
      SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
      RefBindings B = GetRefBindings(StVals);      
      
      if (const RefVal* T = B.lookup(Sym)) {
        B = Update(B, Sym, *T, GetReceiverE(Summ), hasErr);
        SetRefBindings(StVals, B);
        
        if (hasErr) {
          ErrorExpr = Receiver;
          ErrorSym = Sym;
        }
      }
    }
  }

  // Get the persistent state.  
  St = StateMgr.getPersistentState(StVals);
  
  // Process any errors.  
  if (hasErr) {
    ProcessNonLeakError(Dst, Builder, Ex, ErrorExpr, Pred, St,
                        hasErr, ErrorSym);
    return;
  }
  
  // Consult the summary for the return value.  
  RetEffect RE = GetRetEffect(Summ);
  
  switch (RE.getKind()) {
    default:
      assert (false && "Unhandled RetEffect."); break;
      
    case RetEffect::NoRet:
      
      // Make up a symbol for the return value (not reference counted).
      // FIXME: This is basically copy-and-paste from GRSimpleVals.  We 
      //  should compose behavior, not copy it.
      
      if (Ex->getType() != Eng.getContext().VoidTy) {    
        unsigned Count = Builder.getCurrentBlockCount();
        SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(Ex, Count);
        
        RVal X = LVal::IsLValType(Ex->getType())
               ? cast<RVal>(lval::SymbolVal(Sym)) 
               : cast<RVal>(nonlval::SymbolVal(Sym));
        
        St = StateMgr.SetRVal(St, Ex, X, Eng.getCFG().isBlkExpr(Ex), false);
      }      
      
      break;
      
    case RetEffect::Alias: {
      unsigned idx = RE.getIndex();
      assert (arg_end >= arg_beg);
      assert (idx < (unsigned) (arg_end - arg_beg));
      RVal V = StateMgr.GetRVal(St, *(arg_beg+idx));
      St = StateMgr.SetRVal(St, Ex, V, Eng.getCFG().isBlkExpr(Ex), false);
      break;
    }
      
    case RetEffect::ReceiverAlias: {
      assert (Receiver);
      RVal V = StateMgr.GetRVal(St, Receiver);
      St = StateMgr.SetRVal(St, Ex, V, Eng.getCFG().isBlkExpr(Ex), false);
      break;
    }
      
    case RetEffect::OwnedAllocatedSymbol:
    case RetEffect::OwnedSymbol: {
      unsigned Count = Builder.getCurrentBlockCount();
      SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(Ex, Count);
      QualType RetT = GetReturnType(Ex, Eng.getContext());
      
      ValueState StImpl = *St;
      RefBindings B = GetRefBindings(StImpl);
      SetRefBindings(StImpl, RefBFactory.Add(B, Sym, RefVal::makeOwned(RetT)));
      
      St = StateMgr.SetRVal(StateMgr.getPersistentState(StImpl),
                            Ex, lval::SymbolVal(Sym),
                            Eng.getCFG().isBlkExpr(Ex), false);
      
      // FIXME: Add a flag to the checker where allocations are allowed to fail.      
      if (RE.getKind() == RetEffect::OwnedAllocatedSymbol)
        St = StateMgr.AddNE(St, Sym, Eng.getBasicVals().getZeroWithPtrWidth());
      
      break;
    }
      
    case RetEffect::NotOwnedSymbol: {
      unsigned Count = Builder.getCurrentBlockCount();
      SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(Ex, Count);
      QualType RetT = GetReturnType(Ex, Eng.getContext());
      
      ValueState StImpl = *St;
      RefBindings B = GetRefBindings(StImpl);
      SetRefBindings(StImpl, RefBFactory.Add(B, Sym,
                                             RefVal::makeNotOwned(RetT)));
      
      St = StateMgr.SetRVal(StateMgr.getPersistentState(StImpl),
                            Ex, lval::SymbolVal(Sym),
                            Eng.getCFG().isBlkExpr(Ex), false);
      
      break;
    }
  }
  
  // Is this a sink?
  if (IsEndPath(Summ))
    Builder.MakeSinkNode(Dst, Ex, Pred, St);
  else
    Builder.MakeNode(Dst, Ex, Pred, St);
}


void CFRefCount::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Eng,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          CallExpr* CE, RVal L,
                          ExplodedNode<ValueState>* Pred) {
  
  
  RetainSummary* Summ = NULL;
  
  // Get the summary.

  if (isa<lval::FuncVal>(L)) {  
    lval::FuncVal FV = cast<lval::FuncVal>(L);
    FunctionDecl* FD = FV.getDecl();
    Summ = Summaries.getSummary(FD);
  }
  
  EvalSummary(Dst, Eng, Builder, CE, 0, Summ,
              CE->arg_begin(), CE->arg_end(), Pred);
}


void CFRefCount::EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                     GRExprEngine& Eng,
                                     GRStmtNodeBuilder<ValueState>& Builder,
                                     ObjCMessageExpr* ME,
                                     ExplodedNode<ValueState>* Pred) {
  
  RetainSummary* Summ;
  
  if (Expr* Receiver = ME->getReceiver()) {
    // We need the type-information of the tracked receiver object
    // Retrieve it from the state.
    ObjCInterfaceDecl* ID = 0;

    // FIXME: Wouldn't it be great if this code could be reduced?  It's just
    // a chain of lookups.
    const ValueState* St = Builder.GetState(Pred);
    RVal V = Eng.getStateManager().GetRVal(St, Receiver );

    if (isa<lval::SymbolVal>(V)) {
      SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
      
      if (const RefVal* T  = GetRefBindings(*St).lookup(Sym)) {
        QualType Ty = T->getType();
        
        if (const PointerType* PT = Ty->getAsPointerType()) {
          QualType PointeeTy = PT->getPointeeType();
          
          if (ObjCInterfaceType* IT = dyn_cast<ObjCInterfaceType>(PointeeTy))
            ID = IT->getDecl();
        }
      }
    }
    
    Summ = Summaries.getMethodSummary(ME, ID);
  }
  else
    Summ = Summaries.getClassMethodSummary(ME->getClassName(),
                                           ME->getSelector());

  EvalSummary(Dst, Eng, Builder, ME, ME->getReceiver(), Summ,
              ME->arg_begin(), ME->arg_end(), Pred);
}
  
// Stores.

void CFRefCount::EvalStore(ExplodedNodeSet<ValueState>& Dst,
                           GRExprEngine& Eng,
                           GRStmtNodeBuilder<ValueState>& Builder,
                           Expr* E, ExplodedNode<ValueState>* Pred,
                           const ValueState* St, RVal TargetLV, RVal Val) {
  
  // Check if we have a binding for "Val" and if we are storing it to something
  // we don't understand or otherwise the value "escapes" the function.
  
  if (!isa<lval::SymbolVal>(Val))
    return;
  
  // Are we storing to something that causes the value to "escape"?
  
  bool escapes = false;
  
  if (!isa<lval::DeclVal>(TargetLV))
    escapes = true;
  else
    escapes = cast<lval::DeclVal>(TargetLV).getDecl()->hasGlobalStorage();
  
  if (!escapes)
    return;
  
  SymbolID Sym = cast<lval::SymbolVal>(Val).getSymbol();
  
  if (!GetRefBindings(*St).lookup(Sym))
    return;
  
  // Nuke the binding.  
  St = NukeBinding(Eng.getStateManager(), St, Sym);
  
  // Hand of the remaining logic to the parent implementation.
  GRSimpleVals::EvalStore(Dst, Eng, Builder, E, Pred, St, TargetLV, Val);
}


const ValueState* CFRefCount::NukeBinding(ValueStateManager& VMgr,
                                          const ValueState* St,
                                          SymbolID sid) {
  ValueState StImpl = *St;
  RefBindings B = GetRefBindings(StImpl);
  StImpl.CheckerState = RefBFactory.Remove(B, sid).getRoot();
  return VMgr.getPersistentState(StImpl);
}

// End-of-path.

const ValueState* CFRefCount::HandleSymbolDeath(ValueStateManager& VMgr,
                                          const ValueState* St, SymbolID sid,
                                          RefVal V, bool& hasLeak) {
    
  hasLeak = V.isOwned() || 
            ((V.isNotOwned() || V.isReturnedOwned()) && V.getCount() > 0);

  if (!hasLeak)
    return NukeBinding(VMgr, St, sid);
  
  RefBindings B = GetRefBindings(*St);
  ValueState StImpl = *St;
  StImpl.CheckerState = RefBFactory.Add(B, sid, V^RefVal::ErrorLeak).getRoot();
  
  return VMgr.getPersistentState(StImpl);
}

void CFRefCount::EvalEndPath(GRExprEngine& Eng,
                             GREndPathNodeBuilder<ValueState>& Builder) {
  
  const ValueState* St = Builder.getState();
  RefBindings B = GetRefBindings(*St);
  
  llvm::SmallVector<SymbolID, 10> Leaked;
  
  for (RefBindings::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    bool hasLeak = false;
    
    St = HandleSymbolDeath(Eng.getStateManager(), St,
                           (*I).first, (*I).second, hasLeak);
    
    if (hasLeak) Leaked.push_back((*I).first);
  }

  if (Leaked.empty())
    return;
  
  ExplodedNode<ValueState>* N = Builder.MakeNode(St);  
  
  if (!N)
    return;
    
  std::vector<SymbolID>*& LeaksAtNode = Leaks[N];
  assert (!LeaksAtNode);
  LeaksAtNode = new std::vector<SymbolID>();
  
  for (llvm::SmallVector<SymbolID, 10>::iterator I=Leaked.begin(),
       E = Leaked.end(); I != E; ++I)
    (*LeaksAtNode).push_back(*I);
}

// Dead symbols.

void CFRefCount::EvalDeadSymbols(ExplodedNodeSet<ValueState>& Dst,
                                 GRExprEngine& Eng,
                                 GRStmtNodeBuilder<ValueState>& Builder,
                                 ExplodedNode<ValueState>* Pred,
                                 Stmt* S,
                                 const ValueState* St,
                                 const ValueStateManager::DeadSymbolsTy& Dead) {
    
  // FIXME: a lot of copy-and-paste from EvalEndPath.  Refactor.
  
  RefBindings B = GetRefBindings(*St);
  llvm::SmallVector<SymbolID, 10> Leaked;
  
  for (ValueStateManager::DeadSymbolsTy::const_iterator
       I=Dead.begin(), E=Dead.end(); I!=E; ++I) {
    
    const RefVal* T = B.lookup(*I);

    if (!T)
      continue;
    
    bool hasLeak = false;
    
    St = HandleSymbolDeath(Eng.getStateManager(), St, *I, *T, hasLeak);
    
    if (hasLeak)
      Leaked.push_back(*I);    
  }
  
  if (Leaked.empty())
    return;    
  
  ExplodedNode<ValueState>* N = Builder.MakeNode(Dst, S, Pred, St);  
  
  if (!N)
    return;
  
  std::vector<SymbolID>*& LeaksAtNode = Leaks[N];
  assert (!LeaksAtNode);
  LeaksAtNode = new std::vector<SymbolID>();
  
  for (llvm::SmallVector<SymbolID, 10>::iterator I=Leaked.begin(),
       E = Leaked.end(); I != E; ++I)
    (*LeaksAtNode).push_back(*I);    
}

 // Return statements.

void CFRefCount::EvalReturn(ExplodedNodeSet<ValueState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            ReturnStmt* S,
                            ExplodedNode<ValueState>* Pred) {
  
  Expr* RetE = S->getRetValue();
  if (!RetE) return;
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  const ValueState* St = Builder.GetState(Pred);
  RVal V = StateMgr.GetRVal(St, RetE);
  
  if (!isa<lval::SymbolVal>(V))
    return;
  
  // Get the reference count binding (if any).
  SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
  RefBindings B = GetRefBindings(*St);
  const RefVal* T = B.lookup(Sym);
  
  if (!T)
    return;
  
  // Change the reference count.
  
  RefVal X = *T;  
  
  switch (X.getKind()) {
      
    case RefVal::Owned: { 
      unsigned cnt = X.getCount();
      assert (cnt > 0);
      X = RefVal::makeReturnedOwned(cnt - 1);
      break;
    }
      
    case RefVal::NotOwned: {
      unsigned cnt = X.getCount();
      X = cnt ? RefVal::makeReturnedOwned(cnt - 1)
              : RefVal::makeReturnedNotOwned();
      break;
    }
      
    default: 
      return;
  }
  
  // Update the binding.

  ValueState StImpl = *St;
  StImpl.CheckerState = RefBFactory.Add(B, Sym, X).getRoot();        
  Builder.MakeNode(Dst, S, Pred, StateMgr.getPersistentState(StImpl));
}

// Assumptions.

const ValueState* CFRefCount::EvalAssume(ValueStateManager& VMgr,
                                         const ValueState* St,
                                         RVal Cond, bool Assumption,
                                         bool& isFeasible) {

  // FIXME: We may add to the interface of EvalAssume the list of symbols
  //  whose assumptions have changed.  For now we just iterate through the
  //  bindings and check if any of the tracked symbols are NULL.  This isn't
  //  too bad since the number of symbols we will track in practice are 
  //  probably small and EvalAssume is only called at branches and a few
  //  other places.
    
  RefBindings B = GetRefBindings(*St);
  
  if (B.isEmpty())
    return St;
  
  bool changed = false;

  for (RefBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {    

    // Check if the symbol is null (or equal to any constant).
    // If this is the case, stop tracking the symbol.
  
    if (St->getSymVal(I.getKey())) {
      changed = true;
      B = RefBFactory.Remove(B, I.getKey());
    }
  }
  
  if (!changed)
    return St;
  
  ValueState StImpl = *St;
  StImpl.CheckerState = B.getRoot();
  return VMgr.getPersistentState(StImpl);
}

CFRefCount::RefBindings CFRefCount::Update(RefBindings B, SymbolID sym,
                                           RefVal V, ArgEffect E,
                                           RefVal::Kind& hasErr) {
  
  // FIXME: This dispatch can potentially be sped up by unifiying it into
  //  a single switch statement.  Opt for simplicity for now.
  
  switch (E) {
    default:
      assert (false && "Unhandled CFRef transition.");

    case MayEscape:
      if (V.getKind() == RefVal::Owned) {
        V = V ^ RefVal::NotOwned;
        break;
      }

      // Fall-through.
      
    case DoNothingByRef:
    case DoNothing:
      if (!isGCEnabled() && V.getKind() == RefVal::Released) {
        V = V ^ RefVal::ErrorUseAfterRelease;
        hasErr = V.getKind();
        break;
      }
      
      return B;

    case Autorelease:          
    case StopTracking:
      return RefBFactory.Remove(B, sym);
      
    case IncRef:      
      switch (V.getKind()) {
        default:
          assert(false);

        case RefVal::Owned:
        case RefVal::NotOwned:
          V = V + 1;
          break;
          
        case RefVal::Released:
          if (isGCEnabled())
            V = V ^ RefVal::Owned;
          else {          
            V = V ^ RefVal::ErrorUseAfterRelease;
            hasErr = V.getKind();
          }
          
          break;
      }
      
      break;
      
    case SelfOwn:
      V = V ^ RefVal::NotOwned;
      
    case DecRef:
      switch (V.getKind()) {
        default:
          assert (false);
          
        case RefVal::Owned:
          V = V.getCount() > 1 ? V - 1 : V ^ RefVal::Released;
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
          V = V ^ RefVal::ErrorUseAfterRelease;
          hasErr = V.getKind();
          break;          
      }
      
      break;
  }

  return RefBFactory.Add(B, sym, V);
}


//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

namespace {
  
  //===-------------===//
  // Bug Descriptions. //
  //===-------------===//  
  
  class VISIBILITY_HIDDEN CFRefBug : public BugTypeCacheLocation {
  protected:
    CFRefCount& TF;
    
  public:
    CFRefBug(CFRefCount& tf) : TF(tf) {}
    
    CFRefCount& getTF() { return TF; }
    const CFRefCount& getTF() const { return TF; }

    virtual bool isLeak() const { return false; }
  };
  
  class VISIBILITY_HIDDEN UseAfterRelease : public CFRefBug {
  public:
    UseAfterRelease(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      return "Use-After-Release";
    }
    virtual const char* getDescription() const {
      return "Reference-counted object is used"
             " after it is released.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
  };
  
  class VISIBILITY_HIDDEN BadRelease : public CFRefBug {
  public:
    BadRelease(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      return "Bad Release";
    }
    virtual const char* getDescription() const {
      return "Incorrect decrement of the reference count of a "
      "CoreFoundation object: "
      "The object is not owned at this point by the caller.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
  };
  
  class VISIBILITY_HIDDEN Leak : public CFRefBug {
  public:
    Leak(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      
      if (getTF().isGCEnabled())
        return "Memory Leak (GC)";
      
      if (getTF().getLangOptions().getGCMode() == LangOptions::HybridGC)
        return "Memory Leak (Hybrid MM, non-GC)";
      
      assert (getTF().getLangOptions().getGCMode() == LangOptions::NonGC);
      return "Memory Leak";
    }
    
    virtual const char* getDescription() const {
      return "Object leaked.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
    virtual void GetErrorNodes(std::vector<ExplodedNode<ValueState>*>& Nodes);
    virtual bool isLeak() const { return true; }
    virtual bool isCached(BugReport& R);
  };
  
  //===---------===//
  // Bug Reports.  //
  //===---------===//
  
  class VISIBILITY_HIDDEN CFRefReport : public RangedBugReport {
    SymbolID Sym;
  public:
    CFRefReport(CFRefBug& D, ExplodedNode<ValueState> *n, SymbolID sym)
      : RangedBugReport(D, n), Sym(sym) {}
        
    virtual ~CFRefReport() {}
    
    CFRefBug& getBugType() {
      return (CFRefBug&) RangedBugReport::getBugType();
    }
    const CFRefBug& getBugType() const {
      return (const CFRefBug&) RangedBugReport::getBugType();
    }
    
    virtual void getRanges(BugReporter& BR, const SourceRange*& beg,           
                           const SourceRange*& end) {
      
      if (!getBugType().isLeak())
        RangedBugReport::getRanges(BR, beg, end);
      else {
        beg = 0;
        end = 0;
      }
    }
    
    SymbolID getSymbol() const { return Sym; }
    
    virtual PathDiagnosticPiece* getEndPath(BugReporter& BR,
                                            ExplodedNode<ValueState>* N);
    
    virtual std::pair<const char**,const char**> getExtraDescriptiveText();
    
    virtual PathDiagnosticPiece* VisitNode(ExplodedNode<ValueState>* N,
                                           ExplodedNode<ValueState>* PrevN,
                                           ExplodedGraph<ValueState>& G,
                                           BugReporter& BR);
  };
  
  
} // end anonymous namespace

void CFRefCount::RegisterChecks(GRExprEngine& Eng) {
  Eng.Register(new UseAfterRelease(*this));
  Eng.Register(new BadRelease(*this));
  Eng.Register(new Leak(*this));
}


static const char* Msgs[] = {
  "Code is compiled in garbage collection only mode"  // GC only
  "  (the bug occurs with garbage collection enabled).",
  
  "Code is compiled without garbage collection.", // No GC.
  
  "Code is compiled for use with and without garbage collection (GC)."
  "  The bug occurs with GC enabled.", // Hybrid, with GC.
  
  "Code is compiled for use with and without garbage collection (GC)."
  "  The bug occurs in non-GC mode."  // Hyrbird, without GC/
};

std::pair<const char**,const char**> CFRefReport::getExtraDescriptiveText() {
  CFRefCount& TF = static_cast<CFRefBug&>(getBugType()).getTF();

  switch (TF.getLangOptions().getGCMode()) {
    default:
      assert(false);
          
    case LangOptions::GCOnly:
      assert (TF.isGCEnabled());
      return std::make_pair(&Msgs[0], &Msgs[0]+1);
      
    case LangOptions::NonGC:
      assert (!TF.isGCEnabled());
      return std::make_pair(&Msgs[1], &Msgs[1]+1);
    
    case LangOptions::HybridGC:
      if (TF.isGCEnabled())
        return std::make_pair(&Msgs[2], &Msgs[2]+1);
      else
        return std::make_pair(&Msgs[3], &Msgs[3]+1);
  }
}

PathDiagnosticPiece* CFRefReport::VisitNode(ExplodedNode<ValueState>* N,
                                            ExplodedNode<ValueState>* PrevN,
                                            ExplodedGraph<ValueState>& G,
                                            BugReporter& BR) {

  // Check if the type state has changed.
  
  const ValueState* PrevSt = PrevN->getState();
  const ValueState* CurrSt = N->getState();
  
  CFRefCount::RefBindings PrevB = CFRefCount::GetRefBindings(*PrevSt);
  CFRefCount::RefBindings CurrB = CFRefCount::GetRefBindings(*CurrSt);
  
  const RefVal* PrevT = PrevB.lookup(Sym);
  const RefVal* CurrT = CurrB.lookup(Sym);
  
  if (!CurrT)
    return NULL;  
  
  const char* Msg = NULL;  
  const RefVal& CurrV = *CurrB.lookup(Sym);

  if (!PrevT) {
    
    Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();

    if (CurrV.isOwned()) {

      if (isa<CallExpr>(S))
        Msg = "Function call returns an object with a +1 retain count"
              " (owning reference).";
      else {
        assert (isa<ObjCMessageExpr>(S));
        Msg = "Method returns an object with a +1 retain count"
              " (owning reference).";
      }
    }
    else {
      assert (CurrV.isNotOwned());
      
      if (isa<CallExpr>(S))
        Msg = "Function call returns an object with a +0 retain count"
              " (non-owning reference).";
      else {
        assert (isa<ObjCMessageExpr>(S));
        Msg = "Method returns an object with a +0 retain count"
              " (non-owning reference).";
      }      
    }
    
    FullSourceLoc Pos(S->getLocStart(), BR.getContext().getSourceManager());
    PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos, Msg);
    
    if (Expr* Exp = dyn_cast<Expr>(S))
      P->addRange(Exp->getSourceRange());
    
    return P;    
  }
  
  // Determine if the typestate has changed.  
  RefVal PrevV = *PrevB.lookup(Sym);
  
  if (PrevV == CurrV)
    return NULL;
  
  // The typestate has changed.
  
  std::ostringstream os;
  
  switch (CurrV.getKind()) {
    case RefVal::Owned:
    case RefVal::NotOwned:

      if (PrevV.getCount() == CurrV.getCount())
        return 0;
      
      if (PrevV.getCount() > CurrV.getCount())
        os << "Reference count decremented.";
      else
        os << "Reference count incremented.";
      
      if (unsigned Count = CurrV.getCount()) {

        os << " Object has +" << Count;
        
        if (Count > 1)
          os << " retain counts.";
        else
          os << " retain count.";
      }
      
      Msg = os.str().c_str();
      
      break;
      
    case RefVal::Released:
      Msg = "Object released.";
      break;
      
    case RefVal::ReturnedOwned:
      Msg = "Object returned to caller as owning reference (single retain count"
            " transferred to caller).";
      break;
      
    case RefVal::ReturnedNotOwned:
      Msg = "Object returned to caller with a +0 (non-owning) retain count.";
      break;

    default:
      return NULL;
  }
  
  Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();    
  FullSourceLoc Pos(S->getLocStart(), BR.getContext().getSourceManager());
  PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos, Msg);
  
  // Add the range by scanning the children of the statement for any bindings
  // to Sym.
  
  ValueStateManager& VSM = cast<GRBugReporter>(BR).getStateManager();
  
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Expr* Exp = dyn_cast_or_null<Expr>(*I)) {
      RVal X = VSM.GetRVal(CurrSt, Exp);
      
      if (lval::SymbolVal* SV = dyn_cast<lval::SymbolVal>(&X))
        if (SV->getSymbol() == Sym) {
          P->addRange(Exp->getSourceRange()); break;
        }
    }
  
  return P;
}

static std::pair<ExplodedNode<ValueState>*,VarDecl*>
GetAllocationSite(ExplodedNode<ValueState>* N, SymbolID Sym) {

  typedef CFRefCount::RefBindings RefBindings;
  ExplodedNode<ValueState>* Last = N;
  
  // Find the first node that referred to the tracked symbol.  We also
  // try and find the first VarDecl the value was stored to.
  
  VarDecl* FirstDecl = 0;
  
  while (N) {
    const ValueState* St = N->getState();
    RefBindings B = RefBindings((RefBindings::TreeTy*) St->CheckerState);
    
    if (!B.lookup(Sym))
      break;
    
    VarDecl* VD = 0;
    
    // Determine if there is an LVal binding to the symbol.
    for (ValueState::vb_iterator I=St->vb_begin(), E=St->vb_end(); I!=E; ++I) {
      if (!isa<lval::SymbolVal>(I->second)  // Is the value a symbol?
          || cast<lval::SymbolVal>(I->second).getSymbol() != Sym)
        continue;
      
      if (VD) {  // Multiple decls map to this symbol.
        VD = 0;
        break;
      }
      
      VD = I->first;
    }
    
    if (VD) FirstDecl = VD;
    
    Last = N;
    N = N->pred_empty() ? NULL : *(N->pred_begin());    
  }
  
  return std::make_pair(Last, FirstDecl);
}

PathDiagnosticPiece* CFRefReport::getEndPath(BugReporter& BR,
                                             ExplodedNode<ValueState>* EndN) {

  // Tell the BugReporter to report cases when the tracked symbol is
  // assigned to different variables, etc.
  cast<GRBugReporter>(BR).addNotableSymbol(Sym);
  
  if (!getBugType().isLeak())
    return RangedBugReport::getEndPath(BR, EndN);

  typedef CFRefCount::RefBindings RefBindings;

  // Get the retain count.

  unsigned long RetCount = 
    CFRefCount::GetRefBindings(*EndN->getState()).lookup(Sym)->getCount();
  
  // We are a leak.  Walk up the graph to get to the first node where the
  // symbol appeared, and also get the first VarDecl that tracked object
  // is stored to.

  ExplodedNode<ValueState>* AllocNode = 0;
  VarDecl* FirstDecl = 0;
  llvm::tie(AllocNode, FirstDecl) = GetAllocationSite(EndN, Sym);
  
  // Get the allocate site.  
  assert (AllocNode);
  Stmt* FirstStmt = cast<PostStmt>(AllocNode->getLocation()).getStmt();

  SourceManager& SMgr = BR.getContext().getSourceManager();
  unsigned AllocLine = SMgr.getLogicalLineNumber(FirstStmt->getLocStart());

  // Get the leak site.  We may have multiple ExplodedNodes (one with the
  // leak) that occur on the same line number; if the node with the leak
  // has any immediate predecessor nodes with the same line number, find
  // any transitive-successors that have a different statement and use that
  // line number instead.  This avoids emiting a diagnostic like:
  //
  //    // 'y' is leaked.
  //  int x = foo(y);
  //
  //  instead we want:
  //
  //  int x = foo(y);
  //   // 'y' is leaked.
  
  Stmt* S = getStmt(BR);  // This is the statement where the leak occured.
  assert (S);
  unsigned EndLine = SMgr.getLogicalLineNumber(S->getLocStart());

  // Look in the *trimmed* graph at the immediate predecessor of EndN.  Does
  // it occur on the same line?

  PathDiagnosticPiece::DisplayHint Hint = PathDiagnosticPiece::Above;
  
  assert (!EndN->pred_empty()); // Not possible to have 0 predecessors.
  ExplodedNode<ValueState> *Pred = *(EndN->pred_begin());
  ProgramPoint PredPos = Pred->getLocation();
  
  if (PostStmt* PredPS = dyn_cast<PostStmt>(&PredPos)) {

    Stmt* SPred = PredPS->getStmt();
    
    // Predecessor at same line?
    if (SMgr.getLogicalLineNumber(SPred->getLocStart()) != EndLine) {
      Hint = PathDiagnosticPiece::Below;
      S = SPred;
    }
  }
  
  // Generate the diagnostic.
  FullSourceLoc L( S->getLocStart(), SMgr);
  std::ostringstream os;
  
  os << "Object allocated on line " << AllocLine;
  
  if (FirstDecl)
    os << " and stored into '" << FirstDecl->getName() << '\'';
    
  os << " is no longer referenced after this point and has a retain count of +"
     << RetCount << " (object leaked).";
  
  return new PathDiagnosticPiece(L, os.str(), Hint);
}

void UseAfterRelease::EmitWarnings(BugReporter& BR) {

  for (CFRefCount::use_after_iterator I = TF.use_after_begin(),
        E = TF.use_after_end(); I != E; ++I) {
    
    CFRefReport report(*this, I->first, I->second.second);
    report.addRange(I->second.first->getSourceRange());    
    BR.EmitWarning(report);    
  }
}

void BadRelease::EmitWarnings(BugReporter& BR) {
  
  for (CFRefCount::bad_release_iterator I = TF.bad_release_begin(),
       E = TF.bad_release_end(); I != E; ++I) {
    
    CFRefReport report(*this, I->first, I->second.second);
    report.addRange(I->second.first->getSourceRange());    
    BR.EmitWarning(report);    
  }  
}

void Leak::EmitWarnings(BugReporter& BR) {
  
  for (CFRefCount::leaks_iterator I = TF.leaks_begin(),
       E = TF.leaks_end(); I != E; ++I) {
    
    std::vector<SymbolID>& SymV = *(I->second);
    unsigned n = SymV.size();
    
    for (unsigned i = 0; i < n; ++i) {
      CFRefReport report(*this, I->first, SymV[i]);
      BR.EmitWarning(report);
    }
  }  
}

void Leak::GetErrorNodes(std::vector<ExplodedNode<ValueState>*>& Nodes) {
  for (CFRefCount::leaks_iterator I=TF.leaks_begin(), E=TF.leaks_end();
       I!=E; ++I)
    Nodes.push_back(I->first);
}

bool Leak::isCached(BugReport& R) {
  
  // Most bug reports are cached at the location where they occured.
  // With leaks, we want to unique them by the location where they were
  // allocated, and only report only a single path.
  
  SymbolID Sym = static_cast<CFRefReport&>(R).getSymbol();

  ExplodedNode<ValueState>* AllocNode =
    GetAllocationSite(R.getEndNode(), Sym).first;
  
  if (!AllocNode)
    return false;
  
  return BugTypeCacheLocation::isCached(AllocNode->getLocation());
}

//===----------------------------------------------------------------------===//
// Transfer function creation for external clients.
//===----------------------------------------------------------------------===//

GRTransferFuncs* clang::MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                         const LangOptions& lopts) {
  return new CFRefCount(Ctx, GCEnabled, lopts);
}  
