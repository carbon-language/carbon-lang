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
// See http://clang.llvm.org/docs/ThreadSafetyAnalysis.html
// for more information.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/Analysis/Analyses/ThreadSafetyLogical.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/Analyses/ThreadSafetyTraverse.h"
#include "clang/Analysis/Analyses/ThreadSafetyCommon.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>


namespace clang {
namespace threadSafety {

// Key method definition
ThreadSafetyHandler::~ThreadSafetyHandler() {}

class TILPrinter :
  public til::PrettyPrinter<TILPrinter, llvm::raw_ostream> {};


/// Issue a warning about an invalid lock expression
static void warnInvalidLock(ThreadSafetyHandler &Handler,
                            const Expr *MutexExp, const NamedDecl *D,
                            const Expr *DeclExp, StringRef Kind) {
  SourceLocation Loc;
  if (DeclExp)
    Loc = DeclExp->getExprLoc();

  // FIXME: add a note about the attribute location in MutexExp or D
  if (Loc.isValid())
    Handler.handleInvalidLockExp(Kind, Loc);
}


// Various helper functions on til::SExpr
namespace sx {

bool isUniversal(const til::SExpr *E) {
  return isa<til::Wildcard>(E);
}

bool equals(const til::SExpr *E1, const til::SExpr *E2) {
  return til::EqualsComparator::compareExprs(E1, E2);
}

const til::SExpr* ignorePtrCasts(const til::SExpr *E) {
  if (auto *CE = dyn_cast<til::Cast>(E)) {
    if (CE->castOpcode() == til::CAST_objToPtr)
      return CE->expr();
  }
  return E;
}

bool matches(const til::SExpr *E1, const til::SExpr *E2) {
  // We treat a top-level wildcard as the "univsersal" lock.
  // It matches everything for the purpose of checking locks, but not
  // for unlocking them.
  if (isa<til::Wildcard>(E1))
    return isa<til::Wildcard>(E2);
  if (isa<til::Wildcard>(E2))
    return isa<til::Wildcard>(E1);

  return til::MatchComparator::compareExprs(E1, E2);
}

bool partiallyMatches(const til::SExpr *E1, const til::SExpr *E2) {
  auto *PE1 = dyn_cast_or_null<til::Project>(E1);
  if (!PE1)
    return false;
  auto *PE2 = dyn_cast_or_null<til::Project>(E2);
  if (!PE2)
    return false;
  return PE1->clangDecl() == PE2->clangDecl();
}

std::string toString(const til::SExpr *E) {
  std::stringstream ss;
  til::StdPrinter::print(E, ss);
  return ss.str();
}

bool shouldIgnore(const til::SExpr *E) {
  if (!E)
    return true;
  // Trap mutex expressions like nullptr, or 0.
  // Any literal value is nonsense.
  if (isa<til::Literal>(E))
    return true;
  return false;
}

}  // end namespace sx



/// \brief A short list of SExprs
class MutexIDList : public SmallVector<const til::SExpr*, 3> {
public:
  /// \brief Push M onto list, but discard duplicates.
  void push_back_nodup(const til::SExpr *E) {
    iterator It = std::find_if(begin(), end(), [=](const til::SExpr *E2) {
      return sx::equals(E, E2);
    });
    if (It == end())
      push_back(E);
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
  bool     Asserted;           // for asserted locks
  bool     Managed;            // for ScopedLockable objects
  const til::SExpr* UnderlyingMutex;    // for ScopedLockable objects

  LockData(SourceLocation AcquireLoc, LockKind LKind, bool M=false,
           bool Asrt=false)
    : AcquireLoc(AcquireLoc), LKind(LKind), Asserted(Asrt), Managed(M),
      UnderlyingMutex(nullptr)
  {}

  LockData(SourceLocation AcquireLoc, LockKind LKind, const til::SExpr *Mu)
    : AcquireLoc(AcquireLoc), LKind(LKind), Asserted(false), Managed(false),
      UnderlyingMutex(Mu)
  {}

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

  bool isAtLeast(LockKind LK) {
    return (LK == LK_Shared) || (LKind == LK_Exclusive);
  }
};


/// \brief A FactEntry stores a single fact that is known at a particular point
/// in the program execution.  Currently, this is information regarding a lock
/// that is held at that point.
struct FactEntry {
  const til::SExpr *MutID;
  LockData LDat;

  FactEntry(const til::SExpr* M, const LockData& L)
    : MutID(M), LDat(L)
  { }
};


typedef unsigned short FactID;

/// \brief FactManager manages the memory for all facts that are created during
/// the analysis of a single routine.
class FactManager {
private:
  std::vector<FactEntry> Facts;

public:
  FactID newLock(const til::SExpr *M, const LockData& L) {
    Facts.push_back(FactEntry(M, L));
    return static_cast<unsigned short>(Facts.size() - 1);
  }

  const FactEntry& operator[](FactID F) const { return Facts[F]; }
  FactEntry&       operator[](FactID F)       { return Facts[F]; }
};


/// \brief A FactSet is the set of facts that are known to be true at a
/// particular program point.  FactSets must be small, because they are
/// frequently copied, and are thus implemented as a set of indices into a
/// table maintained by a FactManager.  A typical FactSet only holds 1 or 2
/// locks, so we can get away with doing a linear search for lookup.  Note
/// that a hashtable or map is inappropriate in this case, because lookups
/// may involve partial pattern matches, rather than exact matches.
class FactSet {
private:
  typedef SmallVector<FactID, 4> FactVec;

  FactVec FactIDs;

public:
  typedef FactVec::iterator       iterator;
  typedef FactVec::const_iterator const_iterator;

  iterator       begin()       { return FactIDs.begin(); }
  const_iterator begin() const { return FactIDs.begin(); }

  iterator       end()       { return FactIDs.end(); }
  const_iterator end() const { return FactIDs.end(); }

  bool isEmpty() const { return FactIDs.size() == 0; }

  FactID addLock(FactManager& FM, const til::SExpr *M, const LockData& L) {
    FactID F = FM.newLock(M, L);
    FactIDs.push_back(F);
    return F;
  }

  bool removeLock(FactManager& FM, const til::SExpr *M) {
    unsigned n = FactIDs.size();
    if (n == 0)
      return false;

    for (unsigned i = 0; i < n-1; ++i) {
      if (sx::matches(FM[FactIDs[i]].MutID, M)) {
        FactIDs[i] = FactIDs[n-1];
        FactIDs.pop_back();
        return true;
      }
    }
    if (sx::matches(FM[FactIDs[n-1]].MutID, M)) {
      FactIDs.pop_back();
      return true;
    }
    return false;
  }

  iterator findLockIter(FactManager &FM, const til::SExpr *M) {
    return std::find_if(begin(), end(), [&](FactID ID) {
      return sx::matches(FM[ID].MutID, M);
    });
  }

  LockData *findLock(FactManager &FM, const til::SExpr *M) const {
    auto I = std::find_if(begin(), end(), [&](FactID ID) {
      return sx::matches(FM[ID].MutID, M);
    });

    return I != end() ? &FM[*I].LDat : nullptr;
  }

  LockData *findLockUniv(FactManager &FM, const til::SExpr *M) const {
    auto I = std::find_if(begin(), end(), [&](FactID ID) -> bool {
      const til::SExpr *E = FM[ID].MutID;
      return sx::isUniversal(E) || sx::matches(E, M);
    });

    return I != end() ? &FM[*I].LDat : nullptr;
  }

  FactEntry *findPartialMatch(FactManager &FM, const til::SExpr *M) const {
    auto I = std::find_if(begin(), end(), [&](FactID ID) {
      return sx::partiallyMatches(FM[ID].MutID, M);
    });

    return I != end() ? &FM[*I] : nullptr;
  }
};


/// A Lockset maps each SExpr (defined above) to information about how it has
/// been locked.
typedef llvm::ImmutableMap<til::SExpr*, LockData> Lockset;
typedef llvm::ImmutableMap<const NamedDecl*, unsigned> LocalVarContext;

class LocalVariableMap;

/// A side (entry or exit) of a CFG node.
enum CFGBlockSide { CBS_Entry, CBS_Exit };

/// CFGBlockInfo is a struct which contains all the information that is
/// maintained for each block in the CFG.  See LocalVariableMap for more
/// information about the contexts.
struct CFGBlockInfo {
  FactSet EntrySet;             // Lockset held at entry to block
  FactSet ExitSet;              // Lockset held at exit from block
  LocalVarContext EntryContext; // Context held at entry to block
  LocalVarContext ExitContext;  // Context held at exit from block
  SourceLocation EntryLoc;      // Location of first statement in block
  SourceLocation ExitLoc;       // Location of last statement in block.
  unsigned EntryIndex;          // Used to replay contexts later
  bool Reachable;               // Is this block reachable?

  const FactSet &getSet(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntrySet : ExitSet;
  }
  SourceLocation getLocation(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntryLoc : ExitLoc;
  }

private:
  CFGBlockInfo(LocalVarContext EmptyCtx)
    : EntryContext(EmptyCtx), ExitContext(EmptyCtx), Reachable(false)
  { }

public:
  static CFGBlockInfo getEmptyBlockInfo(LocalVariableMap &M);
};



// A LocalVariableMap maintains a map from local variables to their currently
// valid definitions.  It provides SSA-like functionality when traversing the
// CFG.  Like SSA, each definition or assignment to a variable is assigned a
// unique name (an integer), which acts as the SSA name for that definition.
// The total set of names is shared among all CFG basic blocks.
// Unlike SSA, we do not rewrite expressions to replace local variables declrefs
// with their SSA-names.  Instead, we compute a Context for each point in the
// code, which maps local variables to the appropriate SSA-name.  This map
// changes with each assignment.
//
// The map is computed in a single pass over the CFG.  Subsequent analyses can
// then query the map to find the appropriate Context for a statement, and use
// that Context to look up the definitions of variables.
class LocalVariableMap {
public:
  typedef LocalVarContext Context;

  /// A VarDefinition consists of an expression, representing the value of the
  /// variable, along with the context in which that expression should be
  /// interpreted.  A reference VarDefinition does not itself contain this
  /// information, but instead contains a pointer to a previous VarDefinition.
  struct VarDefinition {
  public:
    friend class LocalVariableMap;

    const NamedDecl *Dec;  // The original declaration for this variable.
    const Expr *Exp;       // The expression for this variable, OR
    unsigned Ref;          // Reference to another VarDefinition
    Context Ctx;           // The map with which Exp should be interpreted.

    bool isReference() { return !Exp; }

  private:
    // Create ordinary variable definition
    VarDefinition(const NamedDecl *D, const Expr *E, Context C)
      : Dec(D), Exp(E), Ref(0), Ctx(C)
    { }

    // Create reference to previous definition
    VarDefinition(const NamedDecl *D, unsigned R, Context C)
      : Dec(D), Exp(nullptr), Ref(R), Ctx(C)
    { }
  };

private:
  Context::Factory ContextFactory;
  std::vector<VarDefinition> VarDefinitions;
  std::vector<unsigned> CtxIndices;
  std::vector<std::pair<Stmt*, Context> > SavedContexts;

public:
  LocalVariableMap() {
    // index 0 is a placeholder for undefined variables (aka phi-nodes).
    VarDefinitions.push_back(VarDefinition(nullptr, 0u, getEmptyContext()));
  }

  /// Look up a definition, within the given context.
  const VarDefinition* lookup(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return nullptr;
    assert(*i < VarDefinitions.size());
    return &VarDefinitions[*i];
  }

  /// Look up the definition for D within the given context.  Returns
  /// NULL if the expression is not statically known.  If successful, also
  /// modifies Ctx to hold the context of the return Expr.
  const Expr* lookupExpr(const NamedDecl *D, Context &Ctx) {
    const unsigned *P = Ctx.lookup(D);
    if (!P)
      return nullptr;

    unsigned i = *P;
    while (i > 0) {
      if (VarDefinitions[i].Exp) {
        Ctx = VarDefinitions[i].Ctx;
        return VarDefinitions[i].Exp;
      }
      i = VarDefinitions[i].Ref;
    }
    return nullptr;
  }

  Context getEmptyContext() { return ContextFactory.getEmptyMap(); }

  /// Return the next context after processing S.  This function is used by
  /// clients of the class to get the appropriate context when traversing the
  /// CFG.  It must be called for every assignment or DeclStmt.
  Context getNextContext(unsigned &CtxIndex, Stmt *S, Context C) {
    if (SavedContexts[CtxIndex+1].first == S) {
      CtxIndex++;
      Context Result = SavedContexts[CtxIndex].second;
      return Result;
    }
    return C;
  }

  void dumpVarDefinitionName(unsigned i) {
    if (i == 0) {
      llvm::errs() << "Undefined";
      return;
    }
    const NamedDecl *Dec = VarDefinitions[i].Dec;
    if (!Dec) {
      llvm::errs() << "<<NULL>>";
      return;
    }
    Dec->printName(llvm::errs());
    llvm::errs() << "." << i << " " << ((const void*) Dec);
  }

  /// Dumps an ASCII representation of the variable map to llvm::errs()
  void dump() {
    for (unsigned i = 1, e = VarDefinitions.size(); i < e; ++i) {
      const Expr *Exp = VarDefinitions[i].Exp;
      unsigned Ref = VarDefinitions[i].Ref;

      dumpVarDefinitionName(i);
      llvm::errs() << " = ";
      if (Exp) Exp->dump();
      else {
        dumpVarDefinitionName(Ref);
        llvm::errs() << "\n";
      }
    }
  }

  /// Dumps an ASCII representation of a Context to llvm::errs()
  void dumpContext(Context C) {
    for (Context::iterator I = C.begin(), E = C.end(); I != E; ++I) {
      const NamedDecl *D = I.getKey();
      D->printName(llvm::errs());
      const unsigned *i = C.lookup(D);
      llvm::errs() << " -> ";
      dumpVarDefinitionName(*i);
      llvm::errs() << "\n";
    }
  }

  /// Builds the variable map.
  void traverseCFG(CFG *CFGraph, const PostOrderCFGView *SortedGraph,
                   std::vector<CFGBlockInfo> &BlockInfo);

protected:
  // Get the current context index
  unsigned getContextIndex() { return SavedContexts.size()-1; }

  // Save the current context for later replay
  void saveContext(Stmt *S, Context C) {
    SavedContexts.push_back(std::make_pair(S,C));
  }

  // Adds a new definition to the given context, and returns a new context.
  // This method should be called when declaring a new variable.
  Context addDefinition(const NamedDecl *D, const Expr *Exp, Context Ctx) {
    assert(!Ctx.contains(D));
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
    return NewCtx;
  }

  // Add a new reference to an existing definition.
  Context addReference(const NamedDecl *D, unsigned i, Context Ctx) {
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(VarDefinition(D, i, Ctx));
    return NewCtx;
  }

  // Updates a definition only if that definition is already in the map.
  // This method should be called when assigning to an existing variable.
  Context updateDefinition(const NamedDecl *D, Expr *Exp, Context Ctx) {
    if (Ctx.contains(D)) {
      unsigned newID = VarDefinitions.size();
      Context NewCtx = ContextFactory.remove(Ctx, D);
      NewCtx = ContextFactory.add(NewCtx, D, newID);
      VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
      return NewCtx;
    }
    return Ctx;
  }

  // Removes a definition from the context, but keeps the variable name
  // as a valid variable.  The index 0 is a placeholder for cleared definitions.
  Context clearDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
      NewCtx = ContextFactory.add(NewCtx, D, 0);
    }
    return NewCtx;
  }

  // Remove a definition entirely frmo the context.
  Context removeDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
    }
    return NewCtx;
  }

  Context intersectContexts(Context C1, Context C2);
  Context createReferenceContext(Context C);
  void intersectBackEdge(Context C1, Context C2);

  friend class VarMapBuilder;
};


// This has to be defined after LocalVariableMap.
CFGBlockInfo CFGBlockInfo::getEmptyBlockInfo(LocalVariableMap &M) {
  return CFGBlockInfo(M.getEmptyContext());
}


/// Visitor which builds a LocalVariableMap
class VarMapBuilder : public StmtVisitor<VarMapBuilder> {
public:
  LocalVariableMap* VMap;
  LocalVariableMap::Context Ctx;

  VarMapBuilder(LocalVariableMap *VM, LocalVariableMap::Context C)
    : VMap(VM), Ctx(C) {}

  void VisitDeclStmt(DeclStmt *S);
  void VisitBinaryOperator(BinaryOperator *BO);
};


// Add new local variables to the variable map
void VarMapBuilder::VisitDeclStmt(DeclStmt *S) {
  bool modifiedCtx = false;
  DeclGroupRef DGrp = S->getDeclGroup();
  for (const auto *D : DGrp) {
    if (const auto *VD = dyn_cast_or_null<VarDecl>(D)) {
      const Expr *E = VD->getInit();

      // Add local variables with trivial type to the variable map
      QualType T = VD->getType();
      if (T.isTrivialType(VD->getASTContext())) {
        Ctx = VMap->addDefinition(VD, E, Ctx);
        modifiedCtx = true;
      }
    }
  }
  if (modifiedCtx)
    VMap->saveContext(S, Ctx);
}

// Update local variable definitions in variable map
void VarMapBuilder::VisitBinaryOperator(BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;

  Expr *LHSExp = BO->getLHS()->IgnoreParenCasts();

  // Update the variable map and current context.
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(LHSExp)) {
    ValueDecl *VDec = DRE->getDecl();
    if (Ctx.lookup(VDec)) {
      if (BO->getOpcode() == BO_Assign)
        Ctx = VMap->updateDefinition(VDec, BO->getRHS(), Ctx);
      else
        // FIXME -- handle compound assignment operators
        Ctx = VMap->clearDefinition(VDec, Ctx);
      VMap->saveContext(BO, Ctx);
    }
  }
}


// Computes the intersection of two contexts.  The intersection is the
// set of variables which have the same definition in both contexts;
// variables with different definitions are discarded.
LocalVariableMap::Context
LocalVariableMap::intersectContexts(Context C1, Context C2) {
  Context Result = C1;
  for (const auto &P : C1) {
    const NamedDecl *Dec = P.first;
    const unsigned *i2 = C2.lookup(Dec);
    if (!i2)             // variable doesn't exist on second path
      Result = removeDefinition(Dec, Result);
    else if (*i2 != P.second)  // variable exists, but has different definition
      Result = clearDefinition(Dec, Result);
  }
  return Result;
}

// For every variable in C, create a new variable that refers to the
// definition in C.  Return a new context that contains these new variables.
// (We use this for a naive implementation of SSA on loop back-edges.)
LocalVariableMap::Context LocalVariableMap::createReferenceContext(Context C) {
  Context Result = getEmptyContext();
  for (const auto &P : C)
    Result = addReference(P.first, P.second, Result);
  return Result;
}

// This routine also takes the intersection of C1 and C2, but it does so by
// altering the VarDefinitions.  C1 must be the result of an earlier call to
// createReferenceContext.
void LocalVariableMap::intersectBackEdge(Context C1, Context C2) {
  for (const auto &P : C1) {
    unsigned i1 = P.second;
    VarDefinition *VDef = &VarDefinitions[i1];
    assert(VDef->isReference());

    const unsigned *i2 = C2.lookup(P.first);
    if (!i2 || (*i2 != i1))
      VDef->Ref = 0;    // Mark this variable as undefined
  }
}


// Traverse the CFG in topological order, so all predecessors of a block
// (excluding back-edges) are visited before the block itself.  At
// each point in the code, we calculate a Context, which holds the set of
// variable definitions which are visible at that point in execution.
// Visible variables are mapped to their definitions using an array that
// contains all definitions.
//
// At join points in the CFG, the set is computed as the intersection of
// the incoming sets along each edge, E.g.
//
//                       { Context                 | VarDefinitions }
//   int x = 0;          { x -> x1                 | x1 = 0 }
//   int y = 0;          { x -> x1, y -> y1        | y1 = 0, x1 = 0 }
//   if (b) x = 1;       { x -> x2, y -> y1        | x2 = 1, y1 = 0, ... }
//   else   x = 2;       { x -> x3, y -> y1        | x3 = 2, x2 = 1, ... }
//   ...                 { y -> y1  (x is unknown) | x3 = 2, x2 = 1, ... }
//
// This is essentially a simpler and more naive version of the standard SSA
// algorithm.  Those definitions that remain in the intersection are from blocks
// that strictly dominate the current block.  We do not bother to insert proper
// phi nodes, because they are not used in our analysis; instead, wherever
// a phi node would be required, we simply remove that definition from the
// context (E.g. x above).
//
// The initial traversal does not capture back-edges, so those need to be
// handled on a separate pass.  Whenever the first pass encounters an
// incoming back edge, it duplicates the context, creating new definitions
// that refer back to the originals.  (These correspond to places where SSA
// might have to insert a phi node.)  On the second pass, these definitions are
// set to NULL if the variable has changed on the back-edge (i.e. a phi
// node was actually required.)  E.g.
//
//                       { Context           | VarDefinitions }
//   int x = 0, y = 0;   { x -> x1, y -> y1  | y1 = 0, x1 = 0 }
//   while (b)           { x -> x2, y -> y1  | [1st:] x2=x1; [2nd:] x2=NULL; }
//     x = x+1;          { x -> x3, y -> y1  | x3 = x2 + 1, ... }
//   ...                 { y -> y1           | x3 = 2, x2 = 1, ... }
//
void LocalVariableMap::traverseCFG(CFG *CFGraph,
                                   const PostOrderCFGView *SortedGraph,
                                   std::vector<CFGBlockInfo> &BlockInfo) {
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  CtxIndices.resize(CFGraph->getNumBlockIDs());

  for (const auto *CurrBlock : *SortedGraph) {
    int CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    VisitedBlocks.insert(CurrBlock);

    // Calculate the entry context for the current block
    bool HasBackEdges = false;
    bool CtxInit = true;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge, so skip it
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI)) {
        HasBackEdges = true;
        continue;
      }

      int PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      if (CtxInit) {
        CurrBlockInfo->EntryContext = PrevBlockInfo->ExitContext;
        CtxInit = false;
      }
      else {
        CurrBlockInfo->EntryContext =
          intersectContexts(CurrBlockInfo->EntryContext,
                            PrevBlockInfo->ExitContext);
      }
    }

    // Duplicate the context if we have back-edges, so we can call
    // intersectBackEdges later.
    if (HasBackEdges)
      CurrBlockInfo->EntryContext =
        createReferenceContext(CurrBlockInfo->EntryContext);

    // Create a starting context index for the current block
    saveContext(nullptr, CurrBlockInfo->EntryContext);
    CurrBlockInfo->EntryIndex = getContextIndex();

    // Visit all the statements in the basic block.
    VarMapBuilder VMapBuilder(this, CurrBlockInfo->EntryContext);
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      switch (BI->getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI->castAs<CFGStmt>();
          VMapBuilder.Visit(const_cast<Stmt*>(CS.getStmt()));
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitContext = VMapBuilder.Ctx;

    // Mark variables on back edges as "unknown" if they've been changed.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {
      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      Context LoopBegin = BlockInfo[FirstLoopBlock->getBlockID()].EntryContext;
      Context LoopEnd   = CurrBlockInfo->ExitContext;
      intersectBackEdge(LoopBegin, LoopEnd);
    }
  }

  // Put an extra entry at the end of the indexed context array
  unsigned exitID = CFGraph->getExit().getBlockID();
  saveContext(nullptr, BlockInfo[exitID].ExitContext);
}

/// Find the appropriate source locations to use when producing diagnostics for
/// each block in the CFG.
static void findBlockLocations(CFG *CFGraph,
                               const PostOrderCFGView *SortedGraph,
                               std::vector<CFGBlockInfo> &BlockInfo) {
  for (const auto *CurrBlock : *SortedGraph) {
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlock->getBlockID()];

    // Find the source location of the last statement in the block, if the
    // block is not empty.
    if (const Stmt *S = CurrBlock->getTerminator()) {
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc = S->getLocStart();
    } else {
      for (CFGBlock::const_reverse_iterator BI = CurrBlock->rbegin(),
           BE = CurrBlock->rend(); BI != BE; ++BI) {
        // FIXME: Handle other CFGElement kinds.
        if (Optional<CFGStmt> CS = BI->getAs<CFGStmt>()) {
          CurrBlockInfo->ExitLoc = CS->getStmt()->getLocStart();
          break;
        }
      }
    }

    if (!CurrBlockInfo->ExitLoc.isInvalid()) {
      // This block contains at least one statement. Find the source location
      // of the first statement in the block.
      for (CFGBlock::const_iterator BI = CurrBlock->begin(),
           BE = CurrBlock->end(); BI != BE; ++BI) {
        // FIXME: Handle other CFGElement kinds.
        if (Optional<CFGStmt> CS = BI->getAs<CFGStmt>()) {
          CurrBlockInfo->EntryLoc = CS->getStmt()->getLocStart();
          break;
        }
      }
    } else if (CurrBlock->pred_size() == 1 && *CurrBlock->pred_begin() &&
               CurrBlock != &CFGraph->getExit()) {
      // The block is empty, and has a single predecessor. Use its exit
      // location.
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc =
          BlockInfo[(*CurrBlock->pred_begin())->getBlockID()].ExitLoc;
    }
  }
}

/// \brief Class which implements the core thread safety analysis routines.
class ThreadSafetyAnalyzer {
  friend class BuildLockset;

  llvm::BumpPtrAllocator Bpa;
  threadSafety::til::MemRegionRef Arena;
  threadSafety::SExprBuilder SxBuilder;

  ThreadSafetyHandler       &Handler;
  LocalVariableMap          LocalVarMap;
  FactManager               FactMan;
  std::vector<CFGBlockInfo> BlockInfo;

public:
  ThreadSafetyAnalyzer(ThreadSafetyHandler &H)
     : Arena(&Bpa), SxBuilder(Arena), Handler(H) {}

  void addLock(FactSet &FSet, const til::SExpr *Mutex, const LockData &LDat,
               StringRef DiagKind);
  void removeLock(FactSet &FSet, const til::SExpr *Mutex,
                  SourceLocation UnlockLoc, bool FullyRemove, LockKind Kind,
                  StringRef DiagKind);

  template <typename AttrType>
  void getMutexIDs(MutexIDList &Mtxs, AttrType *Attr, Expr *Exp,
                   const NamedDecl *D, VarDecl *SelfDecl = nullptr);

  template <class AttrType>
  void getMutexIDs(MutexIDList &Mtxs, AttrType *Attr, Expr *Exp,
                   const NamedDecl *D,
                   const CFGBlock *PredBlock, const CFGBlock *CurrBlock,
                   Expr *BrE, bool Neg);

  const CallExpr* getTrylockCallExpr(const Stmt *Cond, LocalVarContext C,
                                     bool &Negate);

  void getEdgeLockset(FactSet &Result, const FactSet &ExitSet,
                      const CFGBlock* PredBlock,
                      const CFGBlock *CurrBlock);

  void intersectAndWarn(FactSet &FSet1, const FactSet &FSet2,
                        SourceLocation JoinLoc,
                        LockErrorKind LEK1, LockErrorKind LEK2,
                        bool Modify=true);

  void intersectAndWarn(FactSet &FSet1, const FactSet &FSet2,
                        SourceLocation JoinLoc, LockErrorKind LEK1,
                        bool Modify=true) {
    intersectAndWarn(FSet1, FSet2, JoinLoc, LEK1, LEK1, Modify);
  }

  void runAnalysis(AnalysisDeclContext &AC);
};

/// \brief Gets the value decl pointer from DeclRefExprs or MemberExprs.
static const ValueDecl *getValueDecl(const Expr *Exp) {
  if (const auto *CE = dyn_cast<ImplicitCastExpr>(Exp))
    return getValueDecl(CE->getSubExpr());

  if (const auto *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const auto *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return nullptr;
}

template <typename Ty>
class has_arg_iterator_range {
  typedef char yes[1];
  typedef char no[2];

  template <typename Inner>
  static yes& test(Inner *I, decltype(I->args()) * = nullptr);

  template <typename>
  static no& test(...);

public:
  static const bool value = sizeof(test<Ty>(nullptr)) == sizeof(yes);
};

static StringRef ClassifyDiagnostic(const CapabilityAttr *A) {
  return A->getName();
}

static StringRef ClassifyDiagnostic(QualType VDT) {
  // We need to look at the declaration of the type of the value to determine
  // which it is. The type should either be a record or a typedef, or a pointer
  // or reference thereof.
  if (const auto *RT = VDT->getAs<RecordType>()) {
    if (const auto *RD = RT->getDecl())
      if (const auto *CA = RD->getAttr<CapabilityAttr>())
        return ClassifyDiagnostic(CA);
  } else if (const auto *TT = VDT->getAs<TypedefType>()) {
    if (const auto *TD = TT->getDecl())
      if (const auto *CA = TD->getAttr<CapabilityAttr>())
        return ClassifyDiagnostic(CA);
  } else if (VDT->isPointerType() || VDT->isReferenceType())
    return ClassifyDiagnostic(VDT->getPointeeType());

  return "mutex";
}

static StringRef ClassifyDiagnostic(const ValueDecl *VD) {
  assert(VD && "No ValueDecl passed");

  // The ValueDecl is the declaration of a mutex or role (hopefully).
  return ClassifyDiagnostic(VD->getType());
}

template <typename AttrTy>
static typename std::enable_if<!has_arg_iterator_range<AttrTy>::value,
                               StringRef>::type
ClassifyDiagnostic(const AttrTy *A) {
  if (const ValueDecl *VD = getValueDecl(A->getArg()))
    return ClassifyDiagnostic(VD);
  return "mutex";
}

template <typename AttrTy>
static typename std::enable_if<has_arg_iterator_range<AttrTy>::value,
                               StringRef>::type
ClassifyDiagnostic(const AttrTy *A) {
  for (const auto *Arg : A->args()) {
    if (const ValueDecl *VD = getValueDecl(Arg))
      return ClassifyDiagnostic(VD);
  }
  return "mutex";
}

/// \brief Add a new lock to the lockset, warning if the lock is already there.
/// \param Mutex -- the Mutex expression for the lock
/// \param LDat  -- the LockData for the lock
void ThreadSafetyAnalyzer::addLock(FactSet &FSet, const til::SExpr *Mutex,
                                   const LockData &LDat, StringRef DiagKind) {
  // FIXME: deal with acquired before/after annotations.
  // FIXME: Don't always warn when we have support for reentrant locks.
  if (sx::shouldIgnore(Mutex))
    return;

  if (FSet.findLock(FactMan, Mutex)) {
    if (!LDat.Asserted)
      Handler.handleDoubleLock(DiagKind, sx::toString(Mutex), LDat.AcquireLoc);
  } else {
    FSet.addLock(FactMan, Mutex, LDat);
  }
}


/// \brief Remove a lock from the lockset, warning if the lock is not there.
/// \param Mutex The lock expression corresponding to the lock to be removed
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void ThreadSafetyAnalyzer::removeLock(FactSet &FSet, const til::SExpr *Mutex,
                                      SourceLocation UnlockLoc,
                                      bool FullyRemove, LockKind ReceivedKind,
                                      StringRef DiagKind) {
  if (sx::shouldIgnore(Mutex))
    return;

  const LockData *LDat = FSet.findLock(FactMan, Mutex);
  if (!LDat) {
    Handler.handleUnmatchedUnlock(DiagKind, sx::toString(Mutex), UnlockLoc);
    return;
  }

  // Generic lock removal doesn't care about lock kind mismatches, but
  // otherwise diagnose when the lock kinds are mismatched.
  if (ReceivedKind != LK_Generic && LDat->LKind != ReceivedKind) {
    Handler.handleIncorrectUnlockKind(DiagKind, sx::toString(Mutex),
                                      LDat->LKind, ReceivedKind, UnlockLoc);
    return;
  }

  if (LDat->UnderlyingMutex) {
    // This is scoped lockable object, which manages the real mutex.
    if (FullyRemove) {
      // We're destroying the managing object.
      // Remove the underlying mutex if it exists; but don't warn.
      if (FSet.findLock(FactMan, LDat->UnderlyingMutex))
        FSet.removeLock(FactMan, LDat->UnderlyingMutex);
    } else {
      // We're releasing the underlying mutex, but not destroying the
      // managing object.  Warn on dual release.
      if (!FSet.findLock(FactMan, LDat->UnderlyingMutex)) {
        Handler.handleUnmatchedUnlock(
            DiagKind, sx::toString(LDat->UnderlyingMutex), UnlockLoc);
      }
      FSet.removeLock(FactMan, LDat->UnderlyingMutex);
      return;
    }
  }
  FSet.removeLock(FactMan, Mutex);
}


/// \brief Extract the list of mutexIDs from the attribute on an expression,
/// and push them onto Mtxs, discarding any duplicates.
template <typename AttrType>
void ThreadSafetyAnalyzer::getMutexIDs(MutexIDList &Mtxs, AttrType *Attr,
                                       Expr *Exp, const NamedDecl *D,
                                       VarDecl *SelfDecl) {
  if (Attr->args_size() == 0) {
    // The mutex held is the "this" object.
    til::SExpr *Mu = SxBuilder.translateAttrExpr(nullptr, D, Exp, SelfDecl);
    if (Mu && isa<til::Undefined>(Mu)) {
       warnInvalidLock(Handler, nullptr, D, Exp, ClassifyDiagnostic(Attr));
       return;
    }
    //else
    if (Mu)
      Mtxs.push_back_nodup(Mu);
    return;
  }

  for (const auto *Arg : Attr->args()) {
    til::SExpr *Mu = SxBuilder.translateAttrExpr(Arg, D, Exp, SelfDecl);
    if (Mu && isa<til::Undefined>(Mu)) {
       warnInvalidLock(Handler, nullptr, D, Exp, ClassifyDiagnostic(Attr));
       return;
    }
    //else
    if (Mu)
      Mtxs.push_back_nodup(Mu);
  }
}


/// \brief Extract the list of mutexIDs from a trylock attribute.  If the
/// trylock applies to the given edge, then push them onto Mtxs, discarding
/// any duplicates.
template <class AttrType>
void ThreadSafetyAnalyzer::getMutexIDs(MutexIDList &Mtxs, AttrType *Attr,
                                       Expr *Exp, const NamedDecl *D,
                                       const CFGBlock *PredBlock,
                                       const CFGBlock *CurrBlock,
                                       Expr *BrE, bool Neg) {
  // Find out which branch has the lock
  bool branch = false;
  if (CXXBoolLiteralExpr *BLE = dyn_cast_or_null<CXXBoolLiteralExpr>(BrE))
    branch = BLE->getValue();
  else if (IntegerLiteral *ILE = dyn_cast_or_null<IntegerLiteral>(BrE))
    branch = ILE->getValue().getBoolValue();

  int branchnum = branch ? 0 : 1;
  if (Neg)
    branchnum = !branchnum;

  // If we've taken the trylock branch, then add the lock
  int i = 0;
  for (CFGBlock::const_succ_iterator SI = PredBlock->succ_begin(),
       SE = PredBlock->succ_end(); SI != SE && i < 2; ++SI, ++i) {
    if (*SI == CurrBlock && i == branchnum)
      getMutexIDs(Mtxs, Attr, Exp, D);
  }
}


bool getStaticBooleanValue(Expr* E, bool& TCond) {
  if (isa<CXXNullPtrLiteralExpr>(E) || isa<GNUNullExpr>(E)) {
    TCond = false;
    return true;
  } else if (CXXBoolLiteralExpr *BLE = dyn_cast<CXXBoolLiteralExpr>(E)) {
    TCond = BLE->getValue();
    return true;
  } else if (IntegerLiteral *ILE = dyn_cast<IntegerLiteral>(E)) {
    TCond = ILE->getValue().getBoolValue();
    return true;
  } else if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E)) {
    return getStaticBooleanValue(CE->getSubExpr(), TCond);
  }
  return false;
}


// If Cond can be traced back to a function call, return the call expression.
// The negate variable should be called with false, and will be set to true
// if the function call is negated, e.g. if (!mu.tryLock(...))
const CallExpr* ThreadSafetyAnalyzer::getTrylockCallExpr(const Stmt *Cond,
                                                         LocalVarContext C,
                                                         bool &Negate) {
  if (!Cond)
    return nullptr;

  if (const CallExpr *CallExp = dyn_cast<CallExpr>(Cond)) {
    return CallExp;
  }
  else if (const ParenExpr *PE = dyn_cast<ParenExpr>(Cond)) {
    return getTrylockCallExpr(PE->getSubExpr(), C, Negate);
  }
  else if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(Cond)) {
    return getTrylockCallExpr(CE->getSubExpr(), C, Negate);
  }
  else if (const ExprWithCleanups* EWC = dyn_cast<ExprWithCleanups>(Cond)) {
    return getTrylockCallExpr(EWC->getSubExpr(), C, Negate);
  }
  else if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    const Expr *E = LocalVarMap.lookupExpr(DRE->getDecl(), C);
    return getTrylockCallExpr(E, C, Negate);
  }
  else if (const UnaryOperator *UOP = dyn_cast<UnaryOperator>(Cond)) {
    if (UOP->getOpcode() == UO_LNot) {
      Negate = !Negate;
      return getTrylockCallExpr(UOP->getSubExpr(), C, Negate);
    }
    return nullptr;
  }
  else if (const BinaryOperator *BOP = dyn_cast<BinaryOperator>(Cond)) {
    if (BOP->getOpcode() == BO_EQ || BOP->getOpcode() == BO_NE) {
      if (BOP->getOpcode() == BO_NE)
        Negate = !Negate;

      bool TCond = false;
      if (getStaticBooleanValue(BOP->getRHS(), TCond)) {
        if (!TCond) Negate = !Negate;
        return getTrylockCallExpr(BOP->getLHS(), C, Negate);
      }
      TCond = false;
      if (getStaticBooleanValue(BOP->getLHS(), TCond)) {
        if (!TCond) Negate = !Negate;
        return getTrylockCallExpr(BOP->getRHS(), C, Negate);
      }
      return nullptr;
    }
    if (BOP->getOpcode() == BO_LAnd) {
      // LHS must have been evaluated in a different block.
      return getTrylockCallExpr(BOP->getRHS(), C, Negate);
    }
    if (BOP->getOpcode() == BO_LOr) {
      return getTrylockCallExpr(BOP->getRHS(), C, Negate);
    }
    return nullptr;
  }
  return nullptr;
}


/// \brief Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
void ThreadSafetyAnalyzer::getEdgeLockset(FactSet& Result,
                                          const FactSet &ExitSet,
                                          const CFGBlock *PredBlock,
                                          const CFGBlock *CurrBlock) {
  Result = ExitSet;

  const Stmt *Cond = PredBlock->getTerminatorCondition();
  if (!Cond)
    return;

  bool Negate = false;
  const CFGBlockInfo *PredBlockInfo = &BlockInfo[PredBlock->getBlockID()];
  const LocalVarContext &LVarCtx = PredBlockInfo->ExitContext;
  StringRef CapDiagKind = "mutex";

  CallExpr *Exp =
    const_cast<CallExpr*>(getTrylockCallExpr(Cond, LVarCtx, Negate));
  if (!Exp)
    return;

  NamedDecl *FunDecl = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());
  if(!FunDecl || !FunDecl->hasAttrs())
    return;

  MutexIDList ExclusiveLocksToAdd;
  MutexIDList SharedLocksToAdd;

  // If the condition is a call to a Trylock function, then grab the attributes
  for (auto *Attr : FunDecl->getAttrs()) {
    switch (Attr->getKind()) {
      case attr::ExclusiveTrylockFunction: {
        ExclusiveTrylockFunctionAttr *A =
          cast<ExclusiveTrylockFunctionAttr>(Attr);
        getMutexIDs(ExclusiveLocksToAdd, A, Exp, FunDecl,
                    PredBlock, CurrBlock, A->getSuccessValue(), Negate);
        CapDiagKind = ClassifyDiagnostic(A);
        break;
      }
      case attr::SharedTrylockFunction: {
        SharedTrylockFunctionAttr *A =
          cast<SharedTrylockFunctionAttr>(Attr);
        getMutexIDs(SharedLocksToAdd, A, Exp, FunDecl,
                    PredBlock, CurrBlock, A->getSuccessValue(), Negate);
        CapDiagKind = ClassifyDiagnostic(A);
        break;
      }
      default:
        break;
    }
  }

  // Add and remove locks.
  SourceLocation Loc = Exp->getExprLoc();
  for (const auto &ExclusiveLockToAdd : ExclusiveLocksToAdd)
    addLock(Result, ExclusiveLockToAdd, LockData(Loc, LK_Exclusive),
            CapDiagKind);
  for (const auto &SharedLockToAdd : SharedLocksToAdd)
    addLock(Result, SharedLockToAdd, LockData(Loc, LK_Shared), CapDiagKind);
}

/// \brief We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public StmtVisitor<BuildLockset> {
  friend class ThreadSafetyAnalyzer;

  ThreadSafetyAnalyzer *Analyzer;
  FactSet FSet;
  LocalVariableMap::Context LVarCtx;
  unsigned CtxIndex;

  // Helper functions

  void warnIfMutexNotHeld(const NamedDecl *D, const Expr *Exp, AccessKind AK,
                          Expr *MutexExp, ProtectedOperationKind POK,
                          StringRef DiagKind);
  void warnIfMutexHeld(const NamedDecl *D, const Expr *Exp, Expr *MutexExp,
                       StringRef DiagKind);

  void checkAccess(const Expr *Exp, AccessKind AK);
  void checkPtAccess(const Expr *Exp, AccessKind AK);

  void handleCall(Expr *Exp, const NamedDecl *D, VarDecl *VD = nullptr);

public:
  BuildLockset(ThreadSafetyAnalyzer *Anlzr, CFGBlockInfo &Info)
    : StmtVisitor<BuildLockset>(),
      Analyzer(Anlzr),
      FSet(Info.EntrySet),
      LVarCtx(Info.EntryContext),
      CtxIndex(Info.EntryIndex)
  {}

  void VisitUnaryOperator(UnaryOperator *UO);
  void VisitBinaryOperator(BinaryOperator *BO);
  void VisitCastExpr(CastExpr *CE);
  void VisitCallExpr(CallExpr *Exp);
  void VisitCXXConstructExpr(CXXConstructExpr *Exp);
  void VisitDeclStmt(DeclStmt *S);
};

/// \brief Warn if the LSet does not contain a lock sufficient to protect access
/// of at least the passed in AccessKind.
void BuildLockset::warnIfMutexNotHeld(const NamedDecl *D, const Expr *Exp,
                                      AccessKind AK, Expr *MutexExp,
                                      ProtectedOperationKind POK,
                                      StringRef DiagKind) {
  LockKind LK = getLockKindFromAccessKind(AK);

  til::SExpr *Mutex = Analyzer->SxBuilder.translateAttrExpr(MutexExp, D, Exp);
  if (!Mutex) {
    // TODO: invalid locks?
    // warnInvalidLock(Analyzer->Handler, MutexExp, D, Exp, DiagKind);
    return;
  } else if (sx::shouldIgnore(Mutex)) {
    return;
  }

  LockData* LDat = FSet.findLockUniv(Analyzer->FactMan, Mutex);
  bool NoError = true;
  if (!LDat) {
    // No exact match found.  Look for a partial match.
    FactEntry* FEntry = FSet.findPartialMatch(Analyzer->FactMan, Mutex);
    if (FEntry) {
      // Warn that there's no precise match.
      LDat = &FEntry->LDat;
      std::string PartMatchStr = sx::toString(FEntry->MutID);
      StringRef   PartMatchName(PartMatchStr);
      Analyzer->Handler.handleMutexNotHeld(DiagKind, D, POK,
                                           sx::toString(Mutex),
                                           LK, Exp->getExprLoc(),
                                           &PartMatchName);
    } else {
      // Warn that there's no match at all.
      Analyzer->Handler.handleMutexNotHeld(DiagKind, D, POK,
                                           sx::toString(Mutex),
                                           LK, Exp->getExprLoc());
    }
    NoError = false;
  }
  // Make sure the mutex we found is the right kind.
  if (NoError && LDat && !LDat->isAtLeast(LK))
    Analyzer->Handler.handleMutexNotHeld(DiagKind, D, POK,
                                         sx::toString(Mutex),
                                         LK, Exp->getExprLoc());
}

/// \brief Warn if the LSet contains the given lock.
void BuildLockset::warnIfMutexHeld(const NamedDecl *D, const Expr *Exp,
                                   Expr *MutexExp,
                                   StringRef DiagKind) {
  til::SExpr *Mutex = Analyzer->SxBuilder.translateAttrExpr(MutexExp, D, Exp);
  if (!Mutex) {
    // TODO: invalid locks?
    // warnInvalidLock(Analyzer->Handler, MutexExp, D, Exp, DiagKind);
    return;
  }

  LockData* LDat = FSet.findLock(Analyzer->FactMan, Mutex);
  if (LDat)
    Analyzer->Handler.handleFunExcludesLock(
        DiagKind, D->getNameAsString(), sx::toString(Mutex), Exp->getExprLoc());
}

/// \brief Checks guarded_by and pt_guarded_by attributes.
/// Whenever we identify an access (read or write) to a DeclRefExpr that is
/// marked with guarded_by, we must ensure the appropriate mutexes are held.
/// Similarly, we check if the access is to an expression that dereferences
/// a pointer marked with pt_guarded_by.
void BuildLockset::checkAccess(const Expr *Exp, AccessKind AK) {
  Exp = Exp->IgnoreParenCasts();

  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(Exp)) {
    // For dereferences
    if (UO->getOpcode() == clang::UO_Deref)
      checkPtAccess(UO->getSubExpr(), AK);
    return;
  }

  if (const ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(Exp)) {
    checkPtAccess(AE->getLHS(), AK);
    return;
  }

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(Exp)) {
    if (ME->isArrow())
      checkPtAccess(ME->getBase(), AK);
    else
      checkAccess(ME->getBase(), AK);
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<GuardedVarAttr>() && FSet.isEmpty())
    Analyzer->Handler.handleNoMutexHeld("mutex", D, POK_VarAccess, AK,
                                        Exp->getExprLoc());

  for (const auto *I : D->specific_attrs<GuardedByAttr>())
    warnIfMutexNotHeld(D, Exp, AK, I->getArg(), POK_VarAccess,
                       ClassifyDiagnostic(I));
}

/// \brief Checks pt_guarded_by and pt_guarded_var attributes.
void BuildLockset::checkPtAccess(const Expr *Exp, AccessKind AK) {
  while (true) {
    if (const ParenExpr *PE = dyn_cast<ParenExpr>(Exp)) {
      Exp = PE->getSubExpr();
      continue;
    }
    if (const CastExpr *CE = dyn_cast<CastExpr>(Exp)) {
      if (CE->getCastKind() == CK_ArrayToPointerDecay) {
        // If it's an actual array, and not a pointer, then it's elements
        // are protected by GUARDED_BY, not PT_GUARDED_BY;
        checkAccess(CE->getSubExpr(), AK);
        return;
      }
      Exp = CE->getSubExpr();
      continue;
    }
    break;
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<PtGuardedVarAttr>() && FSet.isEmpty())
    Analyzer->Handler.handleNoMutexHeld("mutex", D, POK_VarDereference, AK,
                                        Exp->getExprLoc());

  for (auto const *I : D->specific_attrs<PtGuardedByAttr>())
    warnIfMutexNotHeld(D, Exp, AK, I->getArg(), POK_VarDereference,
                       ClassifyDiagnostic(I));
}

/// \brief Process a function call, method call, constructor call,
/// or destructor call.  This involves looking at the attributes on the
/// corresponding function/method/constructor/destructor, issuing warnings,
/// and updating the locksets accordingly.
///
/// FIXME: For classes annotated with one of the guarded annotations, we need
/// to treat const method calls as reads and non-const method calls as writes,
/// and check that the appropriate locks are held. Non-const method calls with
/// the same signature as const method calls can be also treated as reads.
///
void BuildLockset::handleCall(Expr *Exp, const NamedDecl *D, VarDecl *VD) {
  SourceLocation Loc = Exp->getExprLoc();
  const AttrVec &ArgAttrs = D->getAttrs();
  MutexIDList ExclusiveLocksToAdd, SharedLocksToAdd;
  MutexIDList ExclusiveLocksToRemove, SharedLocksToRemove, GenericLocksToRemove;
  StringRef CapDiagKind = "mutex";

  for(unsigned i = 0; i < ArgAttrs.size(); ++i) {
    Attr *At = const_cast<Attr*>(ArgAttrs[i]);
    switch (At->getKind()) {
      // When we encounter a lock function, we need to add the lock to our
      // lockset.
      case attr::AcquireCapability: {
        auto *A = cast<AcquireCapabilityAttr>(At);
        Analyzer->getMutexIDs(A->isShared() ? SharedLocksToAdd
                                            : ExclusiveLocksToAdd,
                              A, Exp, D, VD);

        CapDiagKind = ClassifyDiagnostic(A);
        break;
      }

      // An assert will add a lock to the lockset, but will not generate
      // a warning if it is already there, and will not generate a warning
      // if it is not removed.
      case attr::AssertExclusiveLock: {
        AssertExclusiveLockAttr *A = cast<AssertExclusiveLockAttr>(At);

        MutexIDList AssertLocks;
        Analyzer->getMutexIDs(AssertLocks, A, Exp, D, VD);
        for (const auto &AssertLock : AssertLocks)
          Analyzer->addLock(FSet, AssertLock,
                            LockData(Loc, LK_Exclusive, false, true),
                            ClassifyDiagnostic(A));
        break;
      }
      case attr::AssertSharedLock: {
        AssertSharedLockAttr *A = cast<AssertSharedLockAttr>(At);

        MutexIDList AssertLocks;
        Analyzer->getMutexIDs(AssertLocks, A, Exp, D, VD);
        for (const auto &AssertLock : AssertLocks)
          Analyzer->addLock(FSet, AssertLock,
                            LockData(Loc, LK_Shared, false, true),
                            ClassifyDiagnostic(A));
        break;
      }

      // When we encounter an unlock function, we need to remove unlocked
      // mutexes from the lockset, and flag a warning if they are not there.
      case attr::ReleaseCapability: {
        auto *A = cast<ReleaseCapabilityAttr>(At);
        if (A->isGeneric())
          Analyzer->getMutexIDs(GenericLocksToRemove, A, Exp, D, VD);
        else if (A->isShared())
          Analyzer->getMutexIDs(SharedLocksToRemove, A, Exp, D, VD);
        else
          Analyzer->getMutexIDs(ExclusiveLocksToRemove, A, Exp, D, VD);

        CapDiagKind = ClassifyDiagnostic(A);
        break;
      }

      case attr::RequiresCapability: {
        RequiresCapabilityAttr *A = cast<RequiresCapabilityAttr>(At);
        for (auto *Arg : A->args())
          warnIfMutexNotHeld(D, Exp, A->isShared() ? AK_Read : AK_Written, Arg,
                             POK_FunctionCall, ClassifyDiagnostic(A));
        break;
      }

      case attr::LocksExcluded: {
        LocksExcludedAttr *A = cast<LocksExcludedAttr>(At);
        for (auto *Arg : A->args())
          warnIfMutexHeld(D, Exp, Arg, ClassifyDiagnostic(A));
        break;
      }

      // Ignore attributes unrelated to thread-safety
      default:
        break;
    }
  }

  // Figure out if we're calling the constructor of scoped lockable class
  bool isScopedVar = false;
  if (VD) {
    if (const CXXConstructorDecl *CD = dyn_cast<const CXXConstructorDecl>(D)) {
      const CXXRecordDecl* PD = CD->getParent();
      if (PD && PD->hasAttr<ScopedLockableAttr>())
        isScopedVar = true;
    }
  }

  // Add locks.
  for (const auto &M : ExclusiveLocksToAdd)
    Analyzer->addLock(FSet, M, LockData(Loc, LK_Exclusive, isScopedVar),
                      CapDiagKind);
  for (const auto &M : SharedLocksToAdd)
    Analyzer->addLock(FSet, M, LockData(Loc, LK_Shared, isScopedVar),
                      CapDiagKind);

  // Add the managing object as a dummy mutex, mapped to the underlying mutex.
  // FIXME -- this doesn't work if we acquire multiple locks.
  if (isScopedVar) {
    SourceLocation MLoc = VD->getLocation();
    DeclRefExpr DRE(VD, false, VD->getType(), VK_LValue, VD->getLocation());
    til::SExpr *SMutex = Analyzer->SxBuilder.translateAttrExpr(&DRE, nullptr);

    for (const auto &M : ExclusiveLocksToAdd)
      Analyzer->addLock(FSet, SMutex, LockData(MLoc, LK_Exclusive, M),
                        CapDiagKind);
    for (const auto &M : SharedLocksToAdd)
      Analyzer->addLock(FSet, SMutex, LockData(MLoc, LK_Shared, M),
                        CapDiagKind);

    // handle corner case where the underlying mutex is invalid
    if (ExclusiveLocksToAdd.size() == 0 && SharedLocksToAdd.size() == 0) {
      Analyzer->addLock(FSet, SMutex, LockData(MLoc, LK_Exclusive),
                        CapDiagKind);
    }
  }

  // Remove locks.
  // FIXME -- should only fully remove if the attribute refers to 'this'.
  bool Dtor = isa<CXXDestructorDecl>(D);
  for (const auto &M : ExclusiveLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Exclusive, CapDiagKind);
  for (const auto &M : SharedLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Shared, CapDiagKind);
  for (const auto &M : GenericLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Generic, CapDiagKind);
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
      checkAccess(UO->getSubExpr(), AK_Written);
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

  // adjust the context
  LVarCtx = Analyzer->LocalVarMap.getNextContext(CtxIndex, BO, LVarCtx);

  checkAccess(BO->getLHS(), AK_Written);
}


/// Whenever we do an LValue to Rvalue cast, we are reading a variable and
/// need to ensure we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitCastExpr(CastExpr *CE) {
  if (CE->getCastKind() != CK_LValueToRValue)
    return;
  checkAccess(CE->getSubExpr(), AK_Read);
}


void BuildLockset::VisitCallExpr(CallExpr *Exp) {
  if (CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(Exp)) {
    MemberExpr *ME = dyn_cast<MemberExpr>(CE->getCallee());
    // ME can be null when calling a method pointer
    CXXMethodDecl *MD = CE->getMethodDecl();

    if (ME && MD) {
      if (ME->isArrow()) {
        if (MD->isConst()) {
          checkPtAccess(CE->getImplicitObjectArgument(), AK_Read);
        } else {  // FIXME -- should be AK_Written
          checkPtAccess(CE->getImplicitObjectArgument(), AK_Read);
        }
      } else {
        if (MD->isConst())
          checkAccess(CE->getImplicitObjectArgument(), AK_Read);
        else     // FIXME -- should be AK_Written
          checkAccess(CE->getImplicitObjectArgument(), AK_Read);
      }
    }
  } else if (CXXOperatorCallExpr *OE = dyn_cast<CXXOperatorCallExpr>(Exp)) {
    switch (OE->getOperator()) {
      case OO_Equal: {
        const Expr *Target = OE->getArg(0);
        const Expr *Source = OE->getArg(1);
        checkAccess(Target, AK_Written);
        checkAccess(Source, AK_Read);
        break;
      }
      case OO_Star:
      case OO_Arrow:
      case OO_Subscript: {
        const Expr *Obj = OE->getArg(0);
        checkAccess(Obj, AK_Read);
        checkPtAccess(Obj, AK_Read);
        break;
      }
      default: {
        const Expr *Obj = OE->getArg(0);
        checkAccess(Obj, AK_Read);
        break;
      }
    }
  }
  NamedDecl *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());
  if(!D || !D->hasAttrs())
    return;
  handleCall(Exp, D);
}

void BuildLockset::VisitCXXConstructExpr(CXXConstructExpr *Exp) {
  const CXXConstructorDecl *D = Exp->getConstructor();
  if (D && D->isCopyConstructor()) {
    const Expr* Source = Exp->getArg(0);
    checkAccess(Source, AK_Read);
  }
  // FIXME -- only handles constructors in DeclStmt below.
}

void BuildLockset::VisitDeclStmt(DeclStmt *S) {
  // adjust the context
  LVarCtx = Analyzer->LocalVarMap.getNextContext(CtxIndex, S, LVarCtx);

  for (auto *D : S->getDeclGroup()) {
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(D)) {
      Expr *E = VD->getInit();
      // handle constructors that involve temporaries
      if (ExprWithCleanups *EWC = dyn_cast_or_null<ExprWithCleanups>(E))
        E = EWC->getSubExpr();

      if (CXXConstructExpr *CE = dyn_cast_or_null<CXXConstructExpr>(E)) {
        NamedDecl *CtorD = dyn_cast_or_null<NamedDecl>(CE->getConstructor());
        if (!CtorD || !CtorD->hasAttrs())
          return;
        handleCall(CE, CtorD, VD);
      }
    }
  }
}



/// \brief Compute the intersection of two locksets and issue warnings for any
/// locks in the symmetric difference.
///
/// This function is used at a merge point in the CFG when comparing the lockset
/// of each branch being merged. For example, given the following sequence:
/// A; if () then B; else C; D; we need to check that the lockset after B and C
/// are the same. In the event of a difference, we use the intersection of these
/// two locksets at the start of D.
///
/// \param FSet1 The first lockset.
/// \param FSet2 The second lockset.
/// \param JoinLoc The location of the join point for error reporting
/// \param LEK1 The error message to report if a mutex is missing from LSet1
/// \param LEK2 The error message to report if a mutex is missing from Lset2
void ThreadSafetyAnalyzer::intersectAndWarn(FactSet &FSet1,
                                            const FactSet &FSet2,
                                            SourceLocation JoinLoc,
                                            LockErrorKind LEK1,
                                            LockErrorKind LEK2,
                                            bool Modify) {
  FactSet FSet1Orig = FSet1;

  // Find locks in FSet2 that conflict or are not in FSet1, and warn.
  for (const auto &Fact : FSet2) {
    const til::SExpr *FSet2Mutex = FactMan[Fact].MutID;
    const LockData &LDat2 = FactMan[Fact].LDat;
    FactSet::iterator I1 = FSet1.findLockIter(FactMan, FSet2Mutex);

    if (I1 != FSet1.end()) {
      const LockData* LDat1 = &FactMan[*I1].LDat;
      if (LDat1->LKind != LDat2.LKind) {
        Handler.handleExclusiveAndShared("mutex", sx::toString(FSet2Mutex),
                                         LDat2.AcquireLoc, LDat1->AcquireLoc);
        if (Modify && LDat1->LKind != LK_Exclusive) {
          // Take the exclusive lock, which is the one in FSet2.
          *I1 = Fact;
        }
      }
      else if (LDat1->Asserted && !LDat2.Asserted) {
        // The non-asserted lock in FSet2 is the one we want to track.
        *I1 = Fact;
      }
    } else {
      if (LDat2.UnderlyingMutex) {
        if (FSet2.findLock(FactMan, LDat2.UnderlyingMutex)) {
          // If this is a scoped lock that manages another mutex, and if the
          // underlying mutex is still held, then warn about the underlying
          // mutex.
          Handler.handleMutexHeldEndOfScope("mutex",
                                            sx::toString(LDat2.UnderlyingMutex),
                                            LDat2.AcquireLoc, JoinLoc, LEK1);
        }
      }
      else if (!LDat2.Managed && !sx::isUniversal(FSet2Mutex) &&
               !LDat2.Asserted)
        Handler.handleMutexHeldEndOfScope("mutex", sx::toString(FSet2Mutex),
                                          LDat2.AcquireLoc, JoinLoc, LEK1);
    }
  }

  // Find locks in FSet1 that are not in FSet2, and remove them.
  for (const auto &Fact : FSet1Orig) {
    const til::SExpr *FSet1Mutex = FactMan[Fact].MutID;
    const LockData &LDat1 = FactMan[Fact].LDat;

    if (!FSet2.findLock(FactMan, FSet1Mutex)) {
      if (LDat1.UnderlyingMutex) {
        if (FSet1Orig.findLock(FactMan, LDat1.UnderlyingMutex)) {
          // If this is a scoped lock that manages another mutex, and if the
          // underlying mutex is still held, then warn about the underlying
          // mutex.
          Handler.handleMutexHeldEndOfScope("mutex",
                                            sx::toString(LDat1.UnderlyingMutex),
                                            LDat1.AcquireLoc, JoinLoc, LEK1);
        }
      }
      else if (!LDat1.Managed && !sx::isUniversal(FSet1Mutex) &&
               !LDat1.Asserted)
        Handler.handleMutexHeldEndOfScope("mutex", sx::toString(FSet1Mutex),
                                          LDat1.AcquireLoc, JoinLoc, LEK2);
      if (Modify)
        FSet1.removeLock(FactMan, FSet1Mutex);
    }
  }
}


// Return true if block B never continues to its successors.
inline bool neverReturns(const CFGBlock* B) {
  if (B->hasNoReturnElement())
    return true;
  if (B->empty())
    return false;

  CFGElement Last = B->back();
  if (Optional<CFGStmt> S = Last.getAs<CFGStmt>()) {
    if (isa<CXXThrowExpr>(S->getStmt()))
      return true;
  }
  return false;
}


/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void ThreadSafetyAnalyzer::runAnalysis(AnalysisDeclContext &AC) {
  // TODO: this whole function needs be rewritten as a visitor for CFGWalker.
  // For now, we just use the walker to set things up.
  threadSafety::CFGWalker walker;
  if (!walker.init(AC))
    return;

  // AC.dumpCFG(true);
  // threadSafety::printSCFG(walker);

  CFG *CFGraph = walker.getGraph();
  const NamedDecl *D = walker.getDecl();

  if (D->hasAttr<NoThreadSafetyAnalysisAttr>())
    return;

  // FIXME: Do something a bit more intelligent inside constructor and
  // destructor code.  Constructors and destructors must assume unique access
  // to 'this', so checks on member variable access is disabled, but we should
  // still enable checks on other objects.
  if (isa<CXXConstructorDecl>(D))
    return;  // Don't check inside constructors.
  if (isa<CXXDestructorDecl>(D))
    return;  // Don't check inside destructors.

  BlockInfo.resize(CFGraph->getNumBlockIDs(),
    CFGBlockInfo::getEmptyBlockInfo(LocalVarMap));

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  const PostOrderCFGView *SortedGraph = walker.getSortedGraph();
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  // Mark entry block as reachable
  BlockInfo[CFGraph->getEntry().getBlockID()].Reachable = true;

  // Compute SSA names for local variables
  LocalVarMap.traverseCFG(CFGraph, SortedGraph, BlockInfo);

  // Fill in source locations for all CFGBlocks.
  findBlockLocations(CFGraph, SortedGraph, BlockInfo);

  MutexIDList ExclusiveLocksAcquired;
  MutexIDList SharedLocksAcquired;
  MutexIDList LocksReleased;

  // Add locks from exclusive_locks_required and shared_locks_required
  // to initial lockset. Also turn off checking for lock and unlock functions.
  // FIXME: is there a more intelligent way to check lock/unlock functions?
  if (!SortedGraph->empty() && D->hasAttrs()) {
    const CFGBlock *FirstBlock = *SortedGraph->begin();
    FactSet &InitialLockset = BlockInfo[FirstBlock->getBlockID()].EntrySet;
    const AttrVec &ArgAttrs = D->getAttrs();

    MutexIDList ExclusiveLocksToAdd;
    MutexIDList SharedLocksToAdd;
    StringRef CapDiagKind = "mutex";

    SourceLocation Loc = D->getLocation();
    for (const auto *Attr : ArgAttrs) {
      Loc = Attr->getLocation();
      if (const auto *A = dyn_cast<RequiresCapabilityAttr>(Attr)) {
        getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                    nullptr, D);
        CapDiagKind = ClassifyDiagnostic(A);
      } else if (const auto *A = dyn_cast<ReleaseCapabilityAttr>(Attr)) {
        // UNLOCK_FUNCTION() is used to hide the underlying lock implementation.
        // We must ignore such methods.
        if (A->args_size() == 0)
          return;
        // FIXME -- deal with exclusive vs. shared unlock functions?
        getMutexIDs(ExclusiveLocksToAdd, A, nullptr, D);
        getMutexIDs(LocksReleased, A, nullptr, D);
        CapDiagKind = ClassifyDiagnostic(A);
      } else if (const auto *A = dyn_cast<AcquireCapabilityAttr>(Attr)) {
        if (A->args_size() == 0)
          return;
        getMutexIDs(A->isShared() ? SharedLocksAcquired
                                  : ExclusiveLocksAcquired,
                    A, nullptr, D);
        CapDiagKind = ClassifyDiagnostic(A);
      } else if (isa<ExclusiveTrylockFunctionAttr>(Attr)) {
        // Don't try to check trylock functions for now
        return;
      } else if (isa<SharedTrylockFunctionAttr>(Attr)) {
        // Don't try to check trylock functions for now
        return;
      }
    }

    // FIXME -- Loc can be wrong here.
    for (const auto &ExclusiveLockToAdd : ExclusiveLocksToAdd)
      addLock(InitialLockset, ExclusiveLockToAdd, LockData(Loc, LK_Exclusive),
              CapDiagKind);
    for (const auto &SharedLockToAdd : SharedLocksToAdd)
      addLock(InitialLockset, SharedLockToAdd, LockData(Loc, LK_Shared),
              CapDiagKind);
  }

  for (const auto *CurrBlock : *SortedGraph) {
    int CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    // Use the default initial lockset in case there are no predecessors.
    VisitedBlocks.insert(CurrBlock);

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
    SmallVector<CFGBlock *, 8> SpecialBlocks;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {

      // if *PI -> CurrBlock is a back edge
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI))
        continue;

      int PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      // Ignore edges from blocks that can't return.
      if (neverReturns(*PI) || !PrevBlockInfo->Reachable)
        continue;

      // Okay, we can reach this block from the entry.
      CurrBlockInfo->Reachable = true;

      // If the previous block ended in a 'continue' or 'break' statement, then
      // a difference in locksets is probably due to a bug in that block, rather
      // than in some other predecessor. In that case, keep the other
      // predecessor's lockset.
      if (const Stmt *Terminator = (*PI)->getTerminator()) {
        if (isa<ContinueStmt>(Terminator) || isa<BreakStmt>(Terminator)) {
          SpecialBlocks.push_back(*PI);
          continue;
        }
      }

      FactSet PrevLockset;
      getEdgeLockset(PrevLockset, PrevBlockInfo->ExitSet, *PI, CurrBlock);

      if (!LocksetInitialized) {
        CurrBlockInfo->EntrySet = PrevLockset;
        LocksetInitialized = true;
      } else {
        intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                         CurrBlockInfo->EntryLoc,
                         LEK_LockedSomePredecessors);
      }
    }

    // Skip rest of block if it's not reachable.
    if (!CurrBlockInfo->Reachable)
      continue;

    // Process continue and break blocks. Assume that the lockset for the
    // resulting block is unaffected by any discrepancies in them.
    for (const auto *PrevBlock : SpecialBlocks) {
      int PrevBlockID = PrevBlock->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      if (!LocksetInitialized) {
        CurrBlockInfo->EntrySet = PrevBlockInfo->ExitSet;
        LocksetInitialized = true;
      } else {
        // Determine whether this edge is a loop terminator for diagnostic
        // purposes. FIXME: A 'break' statement might be a loop terminator, but
        // it might also be part of a switch. Also, a subsequent destructor
        // might add to the lockset, in which case the real issue might be a
        // double lock on the other path.
        const Stmt *Terminator = PrevBlock->getTerminator();
        bool IsLoop = Terminator && isa<ContinueStmt>(Terminator);

        FactSet PrevLockset;
        getEdgeLockset(PrevLockset, PrevBlockInfo->ExitSet,
                       PrevBlock, CurrBlock);

        // Do not update EntrySet.
        intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                         PrevBlockInfo->ExitLoc,
                         IsLoop ? LEK_LockedSomeLoopIterations
                                : LEK_LockedSomePredecessors,
                         false);
      }
    }

    BuildLockset LocksetBuilder(this, *CurrBlockInfo);

    // Visit all the statements in the basic block.
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      switch (BI->getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI->castAs<CFGStmt>();
          LocksetBuilder.Visit(const_cast<Stmt*>(CS.getStmt()));
          break;
        }
        // Ignore BaseDtor, MemberDtor, and TemporaryDtor for now.
        case CFGElement::AutomaticObjectDtor: {
          CFGAutomaticObjDtor AD = BI->castAs<CFGAutomaticObjDtor>();
          CXXDestructorDecl *DD = const_cast<CXXDestructorDecl *>(
              AD.getDestructorDecl(AC.getASTContext()));
          if (!DD->hasAttrs())
            break;

          // Create a dummy expression,
          VarDecl *VD = const_cast<VarDecl*>(AD.getVarDecl());
          DeclRefExpr DRE(VD, false, VD->getType(), VK_LValue,
                          AD.getTriggerStmt()->getLocEnd());
          LocksetBuilder.handleCall(&DRE, DD);
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitSet = LocksetBuilder.FSet;

    // For every back edge from CurrBlock (the end of the loop) to another block
    // (FirstLoopBlock) we need to check that the Lockset of Block is equal to
    // the one held at the beginning of FirstLoopBlock. We can look up the
    // Lockset held at the beginning of FirstLoopBlock in the EntryLockSets map.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {

      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      CFGBlockInfo *PreLoop = &BlockInfo[FirstLoopBlock->getBlockID()];
      CFGBlockInfo *LoopEnd = &BlockInfo[CurrBlockID];
      intersectAndWarn(LoopEnd->ExitSet, PreLoop->EntrySet,
                       PreLoop->EntryLoc,
                       LEK_LockedSomeLoopIterations,
                       false);
    }
  }

  CFGBlockInfo *Initial = &BlockInfo[CFGraph->getEntry().getBlockID()];
  CFGBlockInfo *Final   = &BlockInfo[CFGraph->getExit().getBlockID()];

  // Skip the final check if the exit block is unreachable.
  if (!Final->Reachable)
    return;

  // By default, we expect all locks held on entry to be held on exit.
  FactSet ExpectedExitSet = Initial->EntrySet;

  // Adjust the expected exit set by adding or removing locks, as declared
  // by *-LOCK_FUNCTION and UNLOCK_FUNCTION.  The intersect below will then
  // issue the appropriate warning.
  // FIXME: the location here is not quite right.
  for (const auto &Lock : ExclusiveLocksAcquired)
    ExpectedExitSet.addLock(FactMan, Lock,
                            LockData(D->getLocation(), LK_Exclusive));
  for (const auto &Lock : SharedLocksAcquired)
    ExpectedExitSet.addLock(FactMan, Lock,
                            LockData(D->getLocation(), LK_Shared));
  for (const auto &Lock : LocksReleased)
    ExpectedExitSet.removeLock(FactMan, Lock);

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(ExpectedExitSet, Final->ExitSet,
                   Final->ExitLoc,
                   LEK_LockedAtEndOfFunction,
                   LEK_NotLockedAtEndOfFunction,
                   false);
}


/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void runThreadSafetyAnalysis(AnalysisDeclContext &AC,
                             ThreadSafetyHandler &Handler) {
  ThreadSafetyAnalyzer Analyzer(Handler);
  Analyzer.runAnalysis(AC);
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

}} // end namespace clang::threadSafety
