//===- ThreadSafety.cpp ----------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for thread safety (e.g. deadlocks and race
// conditions), based off of an annotation system.
//
// See http://clang.llvm.org/docs/LanguageExtensions.html#threadsafety for more
// information.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <vector>

using namespace clang;
using namespace thread_safety;

// Key method definition
ThreadSafetyHandler::~ThreadSafetyHandler() {}

// Helper function
static Expr *getParent(Expr *Exp) {
  if (MemberExpr *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getBase();
  if (CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(Exp))
    return CE->getImplicitObjectArgument();
  return 0;
}

namespace {
/// \brief Implements a set of CFGBlocks using a BitVector.
///
/// This class contains a minimal interface, primarily dictated by the SetType
/// template parameter of the llvm::po_iterator template, as used with external
/// storage. We also use this set to keep track of which CFGBlocks we visit
/// during the analysis.
class CFGBlockSet {
  llvm::BitVector VisitedBlockIDs;

public:
  // po_iterator requires this iterator, but the only interface needed is the
  // value_type typedef.
  struct iterator {
    typedef const CFGBlock *value_type;
  };

  CFGBlockSet() {}
  CFGBlockSet(const CFG *G) : VisitedBlockIDs(G->getNumBlockIDs(), false) {}

  /// \brief Set the bit associated with a particular CFGBlock.
  /// This is the important method for the SetType template parameter.
  bool insert(const CFGBlock *Block) {
    // Note that insert() is called by po_iterator, which doesn't check to make
    // sure that Block is non-null.  Moreover, the CFGBlock iterator will
    // occasionally hand out null pointers for pruned edges, so we catch those
    // here.
    if (Block == 0)
      return false;  // if an edge is trivially false.
    if (VisitedBlockIDs.test(Block->getBlockID()))
      return false;
    VisitedBlockIDs.set(Block->getBlockID());
    return true;
  }

  /// \brief Check if the bit for a CFGBlock has been already set.
  /// This method is for tracking visited blocks in the main threadsafety loop.
  /// Block must not be null.
  bool alreadySet(const CFGBlock *Block) {
    return VisitedBlockIDs.test(Block->getBlockID());
  }
};

/// \brief We create a helper class which we use to iterate through CFGBlocks in
/// the topological order.
class TopologicallySortedCFG {
  typedef llvm::po_iterator<const CFG*, CFGBlockSet, true>  po_iterator;

  std::vector<const CFGBlock*> Blocks;

public:
  typedef std::vector<const CFGBlock*>::reverse_iterator iterator;

  TopologicallySortedCFG(const CFG *CFGraph) {
    Blocks.reserve(CFGraph->getNumBlockIDs());
    CFGBlockSet BSet(CFGraph);

    for (po_iterator I = po_iterator::begin(CFGraph, BSet),
         E = po_iterator::end(CFGraph, BSet); I != E; ++I) {
      Blocks.push_back(*I);
    }
  }

  iterator begin() {
    return Blocks.rbegin();
  }

  iterator end() {
    return Blocks.rend();
  }

  bool empty() {
    return begin() == end();
  }
};

/// \brief A MutexID object uniquely identifies a particular mutex, and
/// is built from an Expr* (i.e. calling a lock function).
///
/// Thread-safety analysis works by comparing lock expressions.  Within the
/// body of a function, an expression such as "x->foo->bar.mu" will resolve to
/// a particular mutex object at run-time.  Subsequent occurrences of the same
/// expression (where "same" means syntactic equality) will refer to the same
/// run-time object if three conditions hold:
/// (1) Local variables in the expression, such as "x" have not changed.
/// (2) Values on the heap that affect the expression have not changed.
/// (3) The expression involves only pure function calls.
/// The current implementation assumes, but does not verify, that multiple uses
/// of the same lock expression satisfies these criteria.
///
/// Clang introduces an additional wrinkle, which is that it is difficult to
/// derive canonical expressions, or compare expressions directly for equality.
/// Thus, we identify a mutex not by an Expr, but by the set of named
/// declarations that are referenced by the Expr.  In other words,
/// x->foo->bar.mu will be a four element vector with the Decls for
/// mu, bar, and foo, and x.  The vector will uniquely identify the expression
/// for all practical purposes.
///
/// Note we will need to perform substitution on "this" and function parameter
/// names when constructing a lock expression.
///
/// For example:
/// class C { Mutex Mu;  void lock() EXCLUSIVE_LOCK_FUNCTION(this->Mu); };
/// void myFunc(C *X) { ... X->lock() ... }
/// The original expression for the mutex acquired by myFunc is "this->Mu", but
/// "X" is substituted for "this" so we get X->Mu();
///
/// For another example:
/// foo(MyList *L) EXCLUSIVE_LOCKS_REQUIRED(L->Mu) { ... }
/// MyList *MyL;
/// foo(MyL);  // requires lock MyL->Mu to be held
class MutexID {
  SmallVector<NamedDecl*, 2> DeclSeq;

  /// Build a Decl sequence representing the lock from the given expression.
  /// Recursive function that bottoms out when the final DeclRefExpr is reached.
  // FIXME: Lock expressions that involve array indices or function calls.
  // FIXME: Deal with LockReturned attribute.
  void buildMutexID(Expr *Exp, Expr *Parent) {
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp)) {
      NamedDecl *ND = cast<NamedDecl>(DRE->getDecl()->getCanonicalDecl());
      DeclSeq.push_back(ND);
    } else if (MemberExpr *ME = dyn_cast<MemberExpr>(Exp)) {
      NamedDecl *ND = ME->getMemberDecl();
      DeclSeq.push_back(ND);
      buildMutexID(ME->getBase(), Parent);
    } else if (isa<CXXThisExpr>(Exp)) {
      if (Parent)
        buildMutexID(Parent, 0);
      else
        return; // mutexID is still valid in this case
    } else if (CastExpr *CE = dyn_cast<CastExpr>(Exp))
      buildMutexID(CE->getSubExpr(), Parent);
    else
      DeclSeq.clear(); // invalid lock expression
  }

public:
  MutexID(Expr *LExpr, Expr *ParentExpr) {
    buildMutexID(LExpr, ParentExpr);
  }

  /// If we encounter part of a lock expression we cannot parse
  bool isValid() const {
    return !DeclSeq.empty();
  }

  bool operator==(const MutexID &other) const {
    return DeclSeq == other.DeclSeq;
  }

  bool operator!=(const MutexID &other) const {
    return !(*this == other);
  }

  // SmallVector overloads Operator< to do lexicographic ordering. Note that
  // we use pointer equality (and <) to compare NamedDecls. This means the order
  // of MutexIDs in a lockset is nondeterministic. In order to output
  // diagnostics in a deterministic ordering, we must order all diagnostics to
  // output by SourceLocation when iterating through this lockset.
  bool operator<(const MutexID &other) const {
    return DeclSeq < other.DeclSeq;
  }

  /// \brief Returns the name of the first Decl in the list for a given MutexID;
  /// e.g. the lock expression foo.bar() has name "bar".
  /// The caret will point unambiguously to the lock expression, so using this
  /// name in diagnostics is a way to get simple, and consistent, mutex names.
  /// We do not want to output the entire expression text for security reasons.
  StringRef getName() const {
    assert(isValid());
    return DeclSeq.front()->getName();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (SmallVectorImpl<NamedDecl*>::const_iterator I = DeclSeq.begin(),
         E = DeclSeq.end(); I != E; ++I) {
      ID.AddPointer(*I);
    }
  }
};

/// \brief This is a helper class that stores info about the most recent
/// accquire of a Lock.
///
/// The main body of the analysis maps MutexIDs to LockDatas.
struct LockData {
  SourceLocation AcquireLoc;

  /// \brief LKind stores whether a lock is held shared or exclusively.
  /// Note that this analysis does not currently support either re-entrant
  /// locking or lock "upgrading" and "downgrading" between exclusive and
  /// shared.
  ///
  /// FIXME: add support for re-entrant locking and lock up/downgrading
  LockKind LKind;

  LockData(SourceLocation AcquireLoc, LockKind LKind)
    : AcquireLoc(AcquireLoc), LKind(LKind) {}

  bool operator==(const LockData &other) const {
    return AcquireLoc == other.AcquireLoc && LKind == other.LKind;
  }

  bool operator!=(const LockData &other) const {
    return !(*this == other);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(AcquireLoc.getRawEncoding());
      ID.AddInteger(LKind);
    }
};

/// A Lockset maps each MutexID (defined above) to information about how it has
/// been locked.
typedef llvm::ImmutableMap<MutexID, LockData> Lockset;

/// \brief We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public StmtVisitor<BuildLockset> {
  ThreadSafetyHandler &Handler;
  Lockset LSet;
  Lockset::Factory &LocksetFactory;

  // Helper functions
  void removeLock(SourceLocation UnlockLoc, Expr *LockExp, Expr *Parent);
  void addLock(SourceLocation LockLoc, Expr *LockExp, Expr *Parent,
               LockKind LK);
  const ValueDecl *getValueDecl(Expr *Exp);
  void warnIfMutexNotHeld (const NamedDecl *D, Expr *Exp, AccessKind AK,
                           Expr *MutexExp, ProtectedOperationKind POK);
  void checkAccess(Expr *Exp, AccessKind AK);
  void checkDereference(Expr *Exp, AccessKind AK);

  template <class AttrType>
  void addLocksToSet(LockKind LK, Attr *Attr, CXXMemberCallExpr *Exp);

  /// \brief Returns true if the lockset contains a lock, regardless of whether
  /// the lock is held exclusively or shared.
  bool locksetContains(MutexID Lock) const {
    return LSet.lookup(Lock);
  }

  /// \brief Returns true if the lockset contains a lock with the passed in
  /// locktype.
  bool locksetContains(MutexID Lock, LockKind KindRequested) const {
    const LockData *LockHeld = LSet.lookup(Lock);
    return (LockHeld && KindRequested == LockHeld->LKind);
  }

  /// \brief Returns true if the lockset contains a lock with at least the
  /// passed in locktype. So for example, if we pass in LK_Shared, this function
  /// returns true if the lock is held LK_Shared or LK_Exclusive. If we pass in
  /// LK_Exclusive, this function returns true if the lock is held LK_Exclusive.
  bool locksetContainsAtLeast(MutexID Lock, LockKind KindRequested) const {
    switch (KindRequested) {
      case LK_Shared:
        return locksetContains(Lock);
      case LK_Exclusive:
        return locksetContains(Lock, KindRequested);
    }
    llvm_unreachable("Unknown LockKind");
  }

public:
  BuildLockset(ThreadSafetyHandler &Handler, Lockset LS, Lockset::Factory &F)
    : StmtVisitor<BuildLockset>(), Handler(Handler), LSet(LS),
      LocksetFactory(F) {}

  Lockset getLockset() {
    return LSet;
  }

  void VisitUnaryOperator(UnaryOperator *UO);
  void VisitBinaryOperator(BinaryOperator *BO);
  void VisitCastExpr(CastExpr *CE);
  void VisitCXXMemberCallExpr(CXXMemberCallExpr *Exp);
};

/// \brief Add a new lock to the lockset, warning if the lock is already there.
/// \param LockLoc The source location of the acquire
/// \param LockExp The lock expression corresponding to the lock to be added
void BuildLockset::addLock(SourceLocation LockLoc, Expr *LockExp, Expr *Parent,
                           LockKind LK) {
  // FIXME: deal with acquired before/after annotations. We can write a first
  // pass that does the transitive lookup lazily, and refine afterwards.
  MutexID Mutex(LockExp, Parent);
  if (!Mutex.isValid()) {
    Handler.handleInvalidLockExp(LockExp->getExprLoc());
    return;
  }

  LockData NewLock(LockLoc, LK);

  // FIXME: Don't always warn when we have support for reentrant locks.
  if (locksetContains(Mutex))
    Handler.handleDoubleLock(Mutex.getName(), LockLoc);
  LSet = LocksetFactory.add(LSet, Mutex, NewLock);
}

/// \brief Remove a lock from the lockset, warning if the lock is not there.
/// \param LockExp The lock expression corresponding to the lock to be removed
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void BuildLockset::removeLock(SourceLocation UnlockLoc, Expr *LockExp,
                              Expr *Parent) {
  MutexID Mutex(LockExp, Parent);
  if (!Mutex.isValid()) {
    Handler.handleInvalidLockExp(LockExp->getExprLoc());
    return;
  }

  Lockset NewLSet = LocksetFactory.remove(LSet, Mutex);
  if(NewLSet == LSet)
    Handler.handleUnmatchedUnlock(Mutex.getName(), UnlockLoc);

  LSet = NewLSet;
}

/// \brief Gets the value decl pointer from DeclRefExprs or MemberExprs
const ValueDecl *BuildLockset::getValueDecl(Expr *Exp) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return 0;
}

/// \brief Warn if the LSet does not contain a lock sufficient to protect access
/// of at least the passed in AccessType.
void BuildLockset::warnIfMutexNotHeld(const NamedDecl *D, Expr *Exp,
                                      AccessKind AK, Expr *MutexExp,
                                      ProtectedOperationKind POK) {
  LockKind LK = getLockKindFromAccessKind(AK);
  Expr *Parent = getParent(Exp);
  MutexID Mutex(MutexExp, Parent);
  if (!Mutex.isValid())
    Handler.handleInvalidLockExp(MutexExp->getExprLoc());
  else if (!locksetContainsAtLeast(Mutex, LK))
    Handler.handleMutexNotHeld(D, POK, Mutex.getName(), LK, Exp->getExprLoc());
}


/// \brief This method identifies variable dereferences and checks pt_guarded_by
/// and pt_guarded_var annotations. Note that we only check these annotations
/// at the time a pointer is dereferenced.
/// FIXME: We need to check for other types of pointer dereferences
/// (e.g. [], ->) and deal with them here.
/// \param Exp An expression that has been read or written.
void BuildLockset::checkDereference(Expr *Exp, AccessKind AK) {
  UnaryOperator *UO = dyn_cast<UnaryOperator>(Exp);
  if (!UO || UO->getOpcode() != clang::UO_Deref)
    return;
  Exp = UO->getSubExpr()->IgnoreParenCasts();

  const ValueDecl *D = getValueDecl(Exp);
  if(!D || !D->hasAttrs())
    return;

  if (D->getAttr<PtGuardedVarAttr>() && LSet.isEmpty())
    Handler.handleNoMutexHeld(D, POK_VarDereference, AK, Exp->getExprLoc());

  const AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0, Size = ArgAttrs.size(); i < Size; ++i)
    if (PtGuardedByAttr *PGBAttr = dyn_cast<PtGuardedByAttr>(ArgAttrs[i]))
      warnIfMutexNotHeld(D, Exp, AK, PGBAttr->getArg(), POK_VarDereference);
}

/// \brief Checks guarded_by and guarded_var attributes.
/// Whenever we identify an access (read or write) of a DeclRefExpr or
/// MemberExpr, we need to check whether there are any guarded_by or
/// guarded_var attributes, and make sure we hold the appropriate mutexes.
void BuildLockset::checkAccess(Expr *Exp, AccessKind AK) {
  const ValueDecl *D = getValueDecl(Exp);
  if(!D || !D->hasAttrs())
    return;

  if (D->getAttr<GuardedVarAttr>() && LSet.isEmpty())
    Handler.handleNoMutexHeld(D, POK_VarAccess, AK, Exp->getExprLoc());

  const AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0, Size = ArgAttrs.size(); i < Size; ++i)
    if (GuardedByAttr *GBAttr = dyn_cast<GuardedByAttr>(ArgAttrs[i]))
      warnIfMutexNotHeld(D, Exp, AK, GBAttr->getArg(), POK_VarAccess);
}

/// \brief For unary operations which read and write a variable, we need to
/// check whether we hold any required mutexes. Reads are checked in
/// VisitCastExpr.
void BuildLockset::VisitUnaryOperator(UnaryOperator *UO) {
  switch (UO->getOpcode()) {
    case clang::UO_PostDec:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PreInc: {
      Expr *SubExp = UO->getSubExpr()->IgnoreParenCasts();
      checkAccess(SubExp, AK_Written);
      checkDereference(SubExp, AK_Written);
      break;
    }
    default:
      break;
  }
}

/// For binary operations which assign to a variable (writes), we need to check
/// whether we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitBinaryOperator(BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;
  Expr *LHSExp = BO->getLHS()->IgnoreParenCasts();
  checkAccess(LHSExp, AK_Written);
  checkDereference(LHSExp, AK_Written);
}

/// Whenever we do an LValue to Rvalue cast, we are reading a variable and
/// need to ensure we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitCastExpr(CastExpr *CE) {
  if (CE->getCastKind() != CK_LValueToRValue)
    return;
  Expr *SubExp = CE->getSubExpr()->IgnoreParenCasts();
  checkAccess(SubExp, AK_Read);
  checkDereference(SubExp, AK_Read);
}

/// \brief This function, parameterized by an attribute type, is used to add a
/// set of locks specified as attribute arguments to the lockset.
template <typename AttrType>
void BuildLockset::addLocksToSet(LockKind LK, Attr *Attr,
                                 CXXMemberCallExpr *Exp) {
  typedef typename AttrType::args_iterator iterator_type;
  SourceLocation ExpLocation = Exp->getExprLoc();
  Expr *Parent = Exp->getImplicitObjectArgument();
  AttrType *SpecificAttr = cast<AttrType>(Attr);

  if (SpecificAttr->args_size() == 0) {
    // The mutex held is the "this" object.
    addLock(ExpLocation, Parent, 0, LK);
    return;
  }

  for (iterator_type I = SpecificAttr->args_begin(),
       E = SpecificAttr->args_end(); I != E; ++I)
    addLock(ExpLocation, *I, Parent, LK);
}

/// \brief When visiting CXXMemberCallExprs we need to examine the attributes on
/// the method that is being called and add, remove or check locks in the
/// lockset accordingly.
///
/// FIXME: For classes annotated with one of the guarded annotations, we need
/// to treat const method calls as reads and non-const method calls as writes,
/// and check that the appropriate locks are held. Non-const method calls with
/// the same signature as const method calls can be also treated as reads.
///
/// FIXME: We need to also visit CallExprs to catch/check global functions.
///
/// FIXME: Do not flag an error for member variables accessed in constructors/
/// destructors
void BuildLockset::VisitCXXMemberCallExpr(CXXMemberCallExpr *Exp) {
  NamedDecl *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());

  SourceLocation ExpLocation = Exp->getExprLoc();
  Expr *Parent = Exp->getImplicitObjectArgument();

  if(!D || !D->hasAttrs())
    return;

  AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0; i < ArgAttrs.size(); ++i) {
    Attr *Attr = ArgAttrs[i];
    switch (Attr->getKind()) {
      // When we encounter an exclusive lock function, we need to add the lock
      // to our lockset with kind exclusive.
      case attr::ExclusiveLockFunction:
        addLocksToSet<ExclusiveLockFunctionAttr>(LK_Exclusive, Attr, Exp);
        break;

      // When we encounter a shared lock function, we need to add the lock
      // to our lockset with kind shared.
      case attr::SharedLockFunction:
        addLocksToSet<SharedLockFunctionAttr>(LK_Shared, Attr, Exp);
        break;

      // When we encounter an unlock function, we need to remove unlocked
      // mutexes from the lockset, and flag a warning if they are not there.
      case attr::UnlockFunction: {
        UnlockFunctionAttr *UFAttr = cast<UnlockFunctionAttr>(Attr);

        if (UFAttr->args_size() == 0) { // The lock held is the "this" object.
          removeLock(ExpLocation, Parent, 0);
          break;
        }

        for (UnlockFunctionAttr::args_iterator I = UFAttr->args_begin(),
             E = UFAttr->args_end(); I != E; ++I)
          removeLock(ExpLocation, *I, Parent);
        break;
      }

      case attr::ExclusiveLocksRequired: {
        ExclusiveLocksRequiredAttr *ELRAttr =
            cast<ExclusiveLocksRequiredAttr>(Attr);

        for (ExclusiveLocksRequiredAttr::args_iterator
             I = ELRAttr->args_begin(), E = ELRAttr->args_end(); I != E; ++I)
          warnIfMutexNotHeld(D, Exp, AK_Written, *I, POK_FunctionCall);
        break;
      }

      case attr::SharedLocksRequired: {
        SharedLocksRequiredAttr *SLRAttr = cast<SharedLocksRequiredAttr>(Attr);

        for (SharedLocksRequiredAttr::args_iterator I = SLRAttr->args_begin(),
             E = SLRAttr->args_end(); I != E; ++I)
          warnIfMutexNotHeld(D, Exp, AK_Read, *I, POK_FunctionCall);
        break;
      }

      case attr::LocksExcluded: {
        LocksExcludedAttr *LEAttr = cast<LocksExcludedAttr>(Attr);
        for (LocksExcludedAttr::args_iterator I = LEAttr->args_begin(),
            E = LEAttr->args_end(); I != E; ++I) {
          MutexID Mutex(*I, Parent);
          if (!Mutex.isValid())
            Handler.handleInvalidLockExp((*I)->getExprLoc());
          else if (locksetContains(Mutex))
            Handler.handleFunExcludesLock(D->getName(), Mutex.getName(),
                                          ExpLocation);
        }
        break;
      }

      // Ignore other (non thread-safety) attributes
      default:
        break;
    }
  }
}

} // end anonymous namespace

/// \brief Compute the intersection of two locksets and issue warnings for any
/// locks in the symmetric difference.
///
/// This function is used at a merge point in the CFG when comparing the lockset
/// of each branch being merged. For example, given the following sequence:
/// A; if () then B; else C; D; we need to check that the lockset after B and C
/// are the same. In the event of a difference, we use the intersection of these
/// two locksets at the start of D.
static Lockset intersectAndWarn(ThreadSafetyHandler &Handler,
                                const Lockset LSet1, const Lockset LSet2,
                                Lockset::Factory &Fact, LockErrorKind LEK) {
  Lockset Intersection = LSet1;
  for (Lockset::iterator I = LSet2.begin(), E = LSet2.end(); I != E; ++I) {
    const MutexID &LSet2Mutex = I.getKey();
    const LockData &LSet2LockData = I.getData();
    if (const LockData *LD = LSet1.lookup(LSet2Mutex)) {
      if (LD->LKind != LSet2LockData.LKind) {
        Handler.handleExclusiveAndShared(LSet2Mutex.getName(),
                                         LSet2LockData.AcquireLoc,
                                         LD->AcquireLoc);
        if (LD->LKind != LK_Exclusive)
          Intersection = Fact.add(Intersection, LSet2Mutex, LSet2LockData);
      }
    } else {
      Handler.handleMutexHeldEndOfScope(LSet2Mutex.getName(),
                                        LSet2LockData.AcquireLoc, LEK);
    }
  }

  for (Lockset::iterator I = LSet1.begin(), E = LSet1.end(); I != E; ++I) {
    if (!LSet2.contains(I.getKey())) {
      const MutexID &Mutex = I.getKey();
      const LockData &MissingLock = I.getData();
      Handler.handleMutexHeldEndOfScope(Mutex.getName(),
                                        MissingLock.AcquireLoc, LEK);
      Intersection = Fact.remove(Intersection, Mutex);
    }
  }
  return Intersection;
}

static Lockset addLock(ThreadSafetyHandler &Handler,
                       Lockset::Factory &LocksetFactory,
                       Lockset &LSet, Expr *LockExp, LockKind LK,
                       SourceLocation Loc) {
  MutexID Mutex(LockExp, 0);
  if (!Mutex.isValid()) {
    Handler.handleInvalidLockExp(LockExp->getExprLoc());
    return LSet;
  }
  LockData NewLock(Loc, LK);
  return LocksetFactory.add(LSet, Mutex, NewLock);
}

namespace clang {
namespace thread_safety {
/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void runThreadSafetyAnalysis(AnalysisContext &AC,
                             ThreadSafetyHandler &Handler) {
  CFG *CFGraph = AC.getCFG();
  if (!CFGraph) return;
  const Decl *D = AC.getDecl();
  if (D && D->getAttr<NoThreadSafetyAnalysisAttr>()) return;

  Lockset::Factory LocksetFactory;

  // FIXME: Swith to SmallVector? Otherwise improve performance impact?
  std::vector<Lockset> EntryLocksets(CFGraph->getNumBlockIDs(),
                                     LocksetFactory.getEmptyMap());
  std::vector<Lockset> ExitLocksets(CFGraph->getNumBlockIDs(),
                                    LocksetFactory.getEmptyMap());

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  TopologicallySortedCFG SortedGraph(CFGraph);
  CFGBlockSet VisitedBlocks(CFGraph);

  if (!SortedGraph.empty() && D->hasAttrs()) {
    const CFGBlock *FirstBlock = *SortedGraph.begin();
    Lockset &InitialLockset = EntryLocksets[FirstBlock->getBlockID()];
    const AttrVec &ArgAttrs = D->getAttrs();
    for(unsigned i = 0; i < ArgAttrs.size(); ++i) {
      Attr *Attr = ArgAttrs[i];
      SourceLocation AttrLoc = Attr->getLocation();
      if (SharedLocksRequiredAttr *SLRAttr
            = dyn_cast<SharedLocksRequiredAttr>(Attr)) {
        for (SharedLocksRequiredAttr::args_iterator
            SLRIter = SLRAttr->args_begin(),
            SLREnd = SLRAttr->args_end(); SLRIter != SLREnd; ++SLRIter)
          InitialLockset = addLock(Handler, LocksetFactory, InitialLockset,
                                   *SLRIter, LK_Shared,
                                   AttrLoc);
      } else if (ExclusiveLocksRequiredAttr *ELRAttr
                   = dyn_cast<ExclusiveLocksRequiredAttr>(Attr)) {
        for (ExclusiveLocksRequiredAttr::args_iterator
            ELRIter = ELRAttr->args_begin(),
            ELREnd = ELRAttr->args_end(); ELRIter != ELREnd; ++ELRIter)
          InitialLockset = addLock(Handler, LocksetFactory, InitialLockset,
                                   *ELRIter, LK_Exclusive,
                                   AttrLoc);
      }
    }
  }

  for (TopologicallySortedCFG::iterator I = SortedGraph.begin(),
       E = SortedGraph.end(); I!= E; ++I) {
    const CFGBlock *CurrBlock = *I;
    int CurrBlockID = CurrBlock->getBlockID();

    VisitedBlocks.insert(CurrBlock);

    // Use the default initial lockset in case there are no predecessors.
    Lockset &Entryset = EntryLocksets[CurrBlockID];
    Lockset &Exitset = ExitLocksets[CurrBlockID];

    // Iterate through the predecessor blocks and warn if the lockset for all
    // predecessors is not the same. We take the entry lockset of the current
    // block to be the intersection of all previous locksets.
    // FIXME: By keeping the intersection, we may output more errors in future
    // for a lock which is not in the intersection, but was in the union. We
    // may want to also keep the union in future. As an example, let's say
    // the intersection contains Mutex L, and the union contains L and M.
    // Later we unlock M. At this point, we would output an error because we
    // never locked M; although the real error is probably that we forgot to
    // lock M on all code paths. Conversely, let's say that later we lock M.
    // In this case, we should compare against the intersection instead of the
    // union because the real error is probably that we forgot to unlock M on
    // all code paths.
    bool LocksetInitialized = false;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {

      // if *PI -> CurrBlock is a back edge
      if (*PI == 0 || !VisitedBlocks.alreadySet(*PI))
        continue;

      int PrevBlockID = (*PI)->getBlockID();
      if (!LocksetInitialized) {
        Entryset = ExitLocksets[PrevBlockID];
        LocksetInitialized = true;
      } else {
        Entryset = intersectAndWarn(Handler, Entryset,
                                    ExitLocksets[PrevBlockID], LocksetFactory,
                                    LEK_LockedSomePredecessors);
      }
    }

    BuildLockset LocksetBuilder(Handler, Entryset, LocksetFactory);
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      if (const CFGStmt *CfgStmt = dyn_cast<CFGStmt>(&*BI))
        LocksetBuilder.Visit(const_cast<Stmt*>(CfgStmt->getStmt()));
    }
    Exitset = LocksetBuilder.getLockset();

    // For every back edge from CurrBlock (the end of the loop) to another block
    // (FirstLoopBlock) we need to check that the Lockset of Block is equal to
    // the one held at the beginning of FirstLoopBlock. We can look up the
    // Lockset held at the beginning of FirstLoopBlock in the EntryLockSets map.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {

      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == 0 || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      Lockset PreLoop = EntryLocksets[FirstLoopBlock->getBlockID()];
      Lockset LoopEnd = ExitLocksets[CurrBlockID];
      intersectAndWarn(Handler, LoopEnd, PreLoop, LocksetFactory,
                       LEK_LockedSomeLoopIterations);
    }
  }

  Lockset InitialLockset = EntryLocksets[CFGraph->getEntry().getBlockID()];
  Lockset FinalLockset = ExitLocksets[CFGraph->getExit().getBlockID()];

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(Handler, InitialLockset, FinalLockset, LocksetFactory,
                   LEK_LockedAtEndOfFunction);
}

/// \brief Helper function that returns a LockKind required for the given level
/// of access.
LockKind getLockKindFromAccessKind(AccessKind AK) {
  switch (AK) {
    case AK_Read :
      return LK_Shared;
    case AK_Written :
      return LK_Exclusive;
  }
  llvm_unreachable("Unknown AccessKind");
}
}} // end namespace clang::thread_safety
