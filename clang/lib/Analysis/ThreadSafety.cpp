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
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>
#include <vector>

using namespace clang;
using namespace thread_safety;

// Key method definition
ThreadSafetyHandler::~ThreadSafetyHandler() {}

namespace {

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
///
/// The current implementation assumes, but does not verify, that multiple uses
/// of the same lock expression satisfies these criteria.
///
/// Clang introduces an additional wrinkle, which is that it is difficult to
/// derive canonical expressions, or compare expressions directly for equality.
/// Thus, we identify a mutex not by an Expr, but by the list of named
/// declarations that are referenced by the Expr.  In other words,
/// x->foo->bar.mu will be a four element vector with the Decls for
/// mu, bar, and foo, and x.  The vector will uniquely identify the expression
/// for all practical purposes.  Null is used to denote 'this'.
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

  /// \brief Encapsulates the lexical context of a function call.  The lexical
  /// context includes the arguments to the call, including the implicit object
  /// argument.  When an attribute containing a mutex expression is attached to
  /// a method, the expression may refer to formal parameters of the method.
  /// Actual arguments must be substituted for formal parameters to derive
  /// the appropriate mutex expression in the lexical context where the function
  /// is called.  PrevCtx holds the context in which the arguments themselves
  /// should be evaluated; multiple calling contexts can be chained together
  /// by the lock_returned attribute.
  struct CallingContext {
    const NamedDecl* AttrDecl;  // The decl to which the attribute is attached.
    Expr*            SelfArg;   // Implicit object argument -- e.g. 'this'
    unsigned         NumArgs;   // Number of funArgs
    Expr**           FunArgs;   // Function arguments
    CallingContext*  PrevCtx;   // The previous context; or 0 if none.

    CallingContext(const NamedDecl* D = 0, Expr* S = 0,
                   unsigned N = 0, Expr** A = 0, CallingContext* P = 0)
      : AttrDecl(D), SelfArg(S), NumArgs(N), FunArgs(A), PrevCtx(P)
    { }
  };

  /// Build a Decl sequence representing the lock from the given expression.
  /// Recursive function that terminates on DeclRefExpr.
  /// Note: this function merely creates a MutexID; it does not check to
  /// ensure that the original expression is a valid mutex expression.
  void buildMutexID(Expr *Exp, CallingContext* CallCtx) {
    if (!Exp) {
      DeclSeq.clear();
      return;
    }

    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp)) {
      NamedDecl *ND = cast<NamedDecl>(DRE->getDecl()->getCanonicalDecl());
      ParmVarDecl *PV = dyn_cast_or_null<ParmVarDecl>(ND);
      if (PV) {
        FunctionDecl *FD =
          cast<FunctionDecl>(PV->getDeclContext())->getCanonicalDecl();
        unsigned i = PV->getFunctionScopeIndex();

        if (CallCtx && CallCtx->FunArgs &&
            FD == CallCtx->AttrDecl->getCanonicalDecl()) {
          // Substitute call arguments for references to function parameters
          assert(i < CallCtx->NumArgs);
          buildMutexID(CallCtx->FunArgs[i], CallCtx->PrevCtx);
          return;
        }
        // Map the param back to the param of the original function declaration.
        DeclSeq.push_back(FD->getParamDecl(i));
        return;
      }
      // Not a function parameter -- just store the reference.
      DeclSeq.push_back(ND);
    } else if (isa<CXXThisExpr>(Exp)) {
      // Substitute parent for 'this'
      if (CallCtx && CallCtx->SelfArg)
        buildMutexID(CallCtx->SelfArg, CallCtx->PrevCtx);
      else {
        DeclSeq.push_back(0);  // Use 0 to represent 'this'.
        return;  // mutexID is still valid in this case
      }
    } else if (MemberExpr *ME = dyn_cast<MemberExpr>(Exp)) {
      NamedDecl *ND = ME->getMemberDecl();
      DeclSeq.push_back(ND);
      buildMutexID(ME->getBase(), CallCtx);
    } else if (CXXMemberCallExpr *CMCE = dyn_cast<CXXMemberCallExpr>(Exp)) {
      // When calling a function with a lock_returned attribute, replace
      // the function call with the expression in lock_returned.
      if (LockReturnedAttr* At =
            CMCE->getMethodDecl()->getAttr<LockReturnedAttr>()) {
        CallingContext LRCallCtx(CMCE->getMethodDecl());
        LRCallCtx.SelfArg = CMCE->getImplicitObjectArgument();
        LRCallCtx.NumArgs = CMCE->getNumArgs();
        LRCallCtx.FunArgs = CMCE->getArgs();
        LRCallCtx.PrevCtx = CallCtx;
        buildMutexID(At->getArg(), &LRCallCtx);
        return;
      }
      DeclSeq.push_back(CMCE->getMethodDecl()->getCanonicalDecl());
      buildMutexID(CMCE->getImplicitObjectArgument(), CallCtx);
      unsigned NumCallArgs = CMCE->getNumArgs();
      Expr** CallArgs = CMCE->getArgs();
      for (unsigned i = 0; i < NumCallArgs; ++i) {
        buildMutexID(CallArgs[i], CallCtx);
      }
    } else if (CallExpr *CE = dyn_cast<CallExpr>(Exp)) {
      if (LockReturnedAttr* At =
            CE->getDirectCallee()->getAttr<LockReturnedAttr>()) {
        CallingContext LRCallCtx(CE->getDirectCallee());
        LRCallCtx.NumArgs = CE->getNumArgs();
        LRCallCtx.FunArgs = CE->getArgs();
        LRCallCtx.PrevCtx = CallCtx;
        buildMutexID(At->getArg(), &LRCallCtx);
        return;
      }
      buildMutexID(CE->getCallee(), CallCtx);
      unsigned NumCallArgs = CE->getNumArgs();
      Expr** CallArgs = CE->getArgs();
      for (unsigned i = 0; i < NumCallArgs; ++i) {
        buildMutexID(CallArgs[i], CallCtx);
      }
    } else if (BinaryOperator *BOE = dyn_cast<BinaryOperator>(Exp)) {
      buildMutexID(BOE->getLHS(), CallCtx);
      buildMutexID(BOE->getRHS(), CallCtx);
    } else if (UnaryOperator *UOE = dyn_cast<UnaryOperator>(Exp)) {
      buildMutexID(UOE->getSubExpr(), CallCtx);
    } else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(Exp)) {
      buildMutexID(ASE->getBase(), CallCtx);
      buildMutexID(ASE->getIdx(), CallCtx);
    } else if (AbstractConditionalOperator *CE =
                 dyn_cast<AbstractConditionalOperator>(Exp)) {
      buildMutexID(CE->getCond(), CallCtx);
      buildMutexID(CE->getTrueExpr(), CallCtx);
      buildMutexID(CE->getFalseExpr(), CallCtx);
    } else if (ChooseExpr *CE = dyn_cast<ChooseExpr>(Exp)) {
      buildMutexID(CE->getCond(), CallCtx);
      buildMutexID(CE->getLHS(), CallCtx);
      buildMutexID(CE->getRHS(), CallCtx);
    } else if (CastExpr *CE = dyn_cast<CastExpr>(Exp)) {
      buildMutexID(CE->getSubExpr(), CallCtx);
    } else if (ParenExpr *PE = dyn_cast<ParenExpr>(Exp)) {
      buildMutexID(PE->getSubExpr(), CallCtx);
    } else if (isa<CharacterLiteral>(Exp) ||
             isa<CXXNullPtrLiteralExpr>(Exp) ||
             isa<GNUNullExpr>(Exp) ||
             isa<CXXBoolLiteralExpr>(Exp) ||
             isa<FloatingLiteral>(Exp) ||
             isa<ImaginaryLiteral>(Exp) ||
             isa<IntegerLiteral>(Exp) ||
             isa<StringLiteral>(Exp) ||
             isa<ObjCStringLiteral>(Exp)) {
      return;  // FIXME: Ignore literals for now
    } else {
      // Ignore.  FIXME: mark as invalid expression?
    }
  }

  /// \brief Construct a MutexID from an expression.
  /// \param MutexExp The original mutex expression within an attribute
  /// \param DeclExp An expression involving the Decl on which the attribute
  ///        occurs.
  /// \param D  The declaration to which the lock/unlock attribute is attached.
  void buildMutexIDFromExp(Expr *MutexExp, Expr *DeclExp, const NamedDecl *D) {
    CallingContext CallCtx(D);

    // If we are processing a raw attribute expression, with no substitutions.
    if (DeclExp == 0) {
      buildMutexID(MutexExp, 0);
      return;
    }

    // Examine DeclExp to find SelfArg and FunArgs, which are used to substitute
    // for formal parameters when we call buildMutexID later.
    if (MemberExpr *ME = dyn_cast<MemberExpr>(DeclExp)) {
      CallCtx.SelfArg = ME->getBase();
    } else if (CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(DeclExp)) {
      CallCtx.SelfArg = CE->getImplicitObjectArgument();
      CallCtx.NumArgs = CE->getNumArgs();
      CallCtx.FunArgs = CE->getArgs();
    } else if (CallExpr *CE = dyn_cast<CallExpr>(DeclExp)) {
      CallCtx.NumArgs = CE->getNumArgs();
      CallCtx.FunArgs = CE->getArgs();
    } else if (CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(DeclExp)) {
      CallCtx.SelfArg = 0;  // FIXME -- get the parent from DeclStmt
      CallCtx.NumArgs = CE->getNumArgs();
      CallCtx.FunArgs = CE->getArgs();
    } else if (D && isa<CXXDestructorDecl>(D)) {
      // There's no such thing as a "destructor call" in the AST.
      CallCtx.SelfArg = DeclExp;
    }

    // If the attribute has no arguments, then assume the argument is "this".
    if (MutexExp == 0) {
      buildMutexID(CallCtx.SelfArg, 0);
      return;
    }

    // For most attributes.
    buildMutexID(MutexExp, &CallCtx);
  }

public:
  explicit MutexID(clang::Decl::EmptyShell e) {
    DeclSeq.clear();
  }

  /// \param MutexExp The original mutex expression within an attribute
  /// \param DeclExp An expression involving the Decl on which the attribute
  ///        occurs.
  /// \param D  The declaration to which the lock/unlock attribute is attached.
  /// Caller must check isValid() after construction.
  MutexID(Expr* MutexExp, Expr *DeclExp, const NamedDecl* D) {
    buildMutexIDFromExp(MutexExp, DeclExp, D);
  }

  /// Return true if this is a valid decl sequence.
  /// Caller must call this by hand after construction to handle errors.
  bool isValid() const {
    return !DeclSeq.empty();
  }

  /// Issue a warning about an invalid lock expression
  static void warnInvalidLock(ThreadSafetyHandler &Handler, Expr* MutexExp,
                              Expr *DeclExp, const NamedDecl* D) {
    SourceLocation Loc;
    if (DeclExp)
      Loc = DeclExp->getExprLoc();

    // FIXME: add a note about the attribute location in MutexExp or D
    if (Loc.isValid())
      Handler.handleInvalidLockExp(Loc);
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
  std::string getName() const {
    assert(isValid());
    if (!DeclSeq.front())
      return "this";  // Use 0 to represent 'this'.
    return DeclSeq.front()->getNameAsString();
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
  MutexID UnderlyingMutex;  // for ScopedLockable objects

  LockData(SourceLocation AcquireLoc, LockKind LKind)
    : AcquireLoc(AcquireLoc), LKind(LKind), UnderlyingMutex(Decl::EmptyShell())
  {}

  LockData(SourceLocation AcquireLoc, LockKind LKind, const MutexID &Mu)
    : AcquireLoc(AcquireLoc), LKind(LKind), UnderlyingMutex(Mu) {}

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
typedef llvm::ImmutableMap<const NamedDecl*, unsigned> LocalVarContext;

class LocalVariableMap;

/// A side (entry or exit) of a CFG node.
enum CFGBlockSide { CBS_Entry, CBS_Exit };

/// CFGBlockInfo is a struct which contains all the information that is
/// maintained for each block in the CFG.  See LocalVariableMap for more
/// information about the contexts.
struct CFGBlockInfo {
  Lockset EntrySet;             // Lockset held at entry to block
  Lockset ExitSet;              // Lockset held at exit from block
  LocalVarContext EntryContext; // Context held at entry to block
  LocalVarContext ExitContext;  // Context held at exit from block
  SourceLocation EntryLoc;      // Location of first statement in block
  SourceLocation ExitLoc;       // Location of last statement in block.
  unsigned EntryIndex;          // Used to replay contexts later

  const Lockset &getSet(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntrySet : ExitSet;
  }
  SourceLocation getLocation(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntryLoc : ExitLoc;
  }

private:
  CFGBlockInfo(Lockset EmptySet, LocalVarContext EmptyCtx)
    : EntrySet(EmptySet), ExitSet(EmptySet),
      EntryContext(EmptyCtx), ExitContext(EmptyCtx)
  { }

public:
  static CFGBlockInfo getEmptyBlockInfo(Lockset::Factory &F,
                                        LocalVariableMap &M);
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
      : Dec(D), Exp(0), Ref(R), Ctx(C)
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
    VarDefinitions.push_back(VarDefinition(0, 0u, getEmptyContext()));
  }

  /// Look up a definition, within the given context.
  const VarDefinition* lookup(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return 0;
    assert(*i < VarDefinitions.size());
    return &VarDefinitions[*i];
  }

  /// Look up the definition for D within the given context.  Returns
  /// NULL if the expression is not statically known.  If successful, also
  /// modifies Ctx to hold the context of the return Expr.
  const Expr* lookupExpr(const NamedDecl *D, Context &Ctx) {
    const unsigned *P = Ctx.lookup(D);
    if (!P)
      return 0;

    unsigned i = *P;
    while (i > 0) {
      if (VarDefinitions[i].Exp) {
        Ctx = VarDefinitions[i].Ctx;
        return VarDefinitions[i].Exp;
      }
      i = VarDefinitions[i].Ref;
    }
    return 0;
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
    llvm::errs() << "." << i << " " << ((void*) Dec);
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
  void traverseCFG(CFG *CFGraph, PostOrderCFGView *SortedGraph,
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
  Context addDefinition(const NamedDecl *D, Expr *Exp, Context Ctx) {
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
CFGBlockInfo CFGBlockInfo::getEmptyBlockInfo(Lockset::Factory &F,
                                             LocalVariableMap &M) {
  return CFGBlockInfo(F.getEmptyMap(), M.getEmptyContext());
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
  for (DeclGroupRef::iterator I = DGrp.begin(), E = DGrp.end(); I != E; ++I) {
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(*I)) {
      Expr *E = VD->getInit();

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
  for (Context::iterator I = C1.begin(), E = C1.end(); I != E; ++I) {
    const NamedDecl *Dec = I.getKey();
    unsigned i1 = I.getData();
    const unsigned *i2 = C2.lookup(Dec);
    if (!i2)             // variable doesn't exist on second path
      Result = removeDefinition(Dec, Result);
    else if (*i2 != i1)  // variable exists, but has different definition
      Result = clearDefinition(Dec, Result);
  }
  return Result;
}

// For every variable in C, create a new variable that refers to the
// definition in C.  Return a new context that contains these new variables.
// (We use this for a naive implementation of SSA on loop back-edges.)
LocalVariableMap::Context LocalVariableMap::createReferenceContext(Context C) {
  Context Result = getEmptyContext();
  for (Context::iterator I = C.begin(), E = C.end(); I != E; ++I) {
    const NamedDecl *Dec = I.getKey();
    unsigned i = I.getData();
    Result = addReference(Dec, i, Result);
  }
  return Result;
}

// This routine also takes the intersection of C1 and C2, but it does so by
// altering the VarDefinitions.  C1 must be the result of an earlier call to
// createReferenceContext.
void LocalVariableMap::intersectBackEdge(Context C1, Context C2) {
  for (Context::iterator I = C1.begin(), E = C1.end(); I != E; ++I) {
    const NamedDecl *Dec = I.getKey();
    unsigned i1 = I.getData();
    VarDefinition *VDef = &VarDefinitions[i1];
    assert(VDef->isReference());

    const unsigned *i2 = C2.lookup(Dec);
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
// set to NULL if the the variable has changed on the back-edge (i.e. a phi
// node was actually required.)  E.g.
//
//                       { Context           | VarDefinitions }
//   int x = 0, y = 0;   { x -> x1, y -> y1  | y1 = 0, x1 = 0 }
//   while (b)           { x -> x2, y -> y1  | [1st:] x2=x1; [2nd:] x2=NULL; }
//     x = x+1;          { x -> x3, y -> y1  | x3 = x2 + 1, ... }
//   ...                 { y -> y1           | x3 = 2, x2 = 1, ... }
//
void LocalVariableMap::traverseCFG(CFG *CFGraph,
                                   PostOrderCFGView *SortedGraph,
                                   std::vector<CFGBlockInfo> &BlockInfo) {
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  CtxIndices.resize(CFGraph->getNumBlockIDs());

  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I!= E; ++I) {
    const CFGBlock *CurrBlock = *I;
    int CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    VisitedBlocks.insert(CurrBlock);

    // Calculate the entry context for the current block
    bool HasBackEdges = false;
    bool CtxInit = true;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge, so skip it
      if (*PI == 0 || !VisitedBlocks.alreadySet(*PI)) {
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
    saveContext(0, CurrBlockInfo->EntryContext);
    CurrBlockInfo->EntryIndex = getContextIndex();

    // Visit all the statements in the basic block.
    VarMapBuilder VMapBuilder(this, CurrBlockInfo->EntryContext);
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      switch (BI->getKind()) {
        case CFGElement::Statement: {
          const CFGStmt *CS = cast<CFGStmt>(&*BI);
          VMapBuilder.Visit(const_cast<Stmt*>(CS->getStmt()));
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
      if (*SI == 0 || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      Context LoopBegin = BlockInfo[FirstLoopBlock->getBlockID()].EntryContext;
      Context LoopEnd   = CurrBlockInfo->ExitContext;
      intersectBackEdge(LoopBegin, LoopEnd);
    }
  }

  // Put an extra entry at the end of the indexed context array
  unsigned exitID = CFGraph->getExit().getBlockID();
  saveContext(0, BlockInfo[exitID].ExitContext);
}

/// Find the appropriate source locations to use when producing diagnostics for
/// each block in the CFG.
static void findBlockLocations(CFG *CFGraph,
                               PostOrderCFGView *SortedGraph,
                               std::vector<CFGBlockInfo> &BlockInfo) {
  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I!= E; ++I) {
    const CFGBlock *CurrBlock = *I;
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlock->getBlockID()];

    // Find the source location of the last statement in the block, if the
    // block is not empty.
    if (const Stmt *S = CurrBlock->getTerminator()) {
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc = S->getLocStart();
    } else {
      for (CFGBlock::const_reverse_iterator BI = CurrBlock->rbegin(),
           BE = CurrBlock->rend(); BI != BE; ++BI) {
        // FIXME: Handle other CFGElement kinds.
        if (const CFGStmt *CS = dyn_cast<CFGStmt>(&*BI)) {
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
        if (const CFGStmt *CS = dyn_cast<CFGStmt>(&*BI)) {
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

  ThreadSafetyHandler       &Handler;
  Lockset::Factory          LocksetFactory;
  LocalVariableMap          LocalVarMap;
  std::vector<CFGBlockInfo> BlockInfo;

public:
  ThreadSafetyAnalyzer(ThreadSafetyHandler &H) : Handler(H) {}

  Lockset addLock(const Lockset &LSet, const MutexID &Mutex,
                  const LockData &LDat);
  Lockset addLock(const Lockset &LSet, Expr *MutexExp, const NamedDecl *D,
                  const LockData &LDat);
  Lockset removeLock(const Lockset &LSet, const MutexID &Mutex,
                     SourceLocation UnlockLoc);

  template <class AttrType>
  Lockset addLocksToSet(const Lockset &LSet, LockKind LK, AttrType *Attr,
                        Expr *Exp, NamedDecl *D, VarDecl *VD = 0);
  Lockset removeLocksFromSet(const Lockset &LSet,
                             UnlockFunctionAttr *Attr,
                             Expr *Exp, NamedDecl* FunDecl);

  template <class AttrType>
  Lockset addTrylock(const Lockset &LSet,
                     LockKind LK, AttrType *Attr, Expr *Exp, NamedDecl *FunDecl,
                     const CFGBlock* PredBlock, const CFGBlock *CurrBlock,
                     Expr *BrE, bool Neg);
  const CallExpr* getTrylockCallExpr(const Stmt *Cond, LocalVarContext C,
                                     bool &Negate);

  Lockset getEdgeLockset(const Lockset &ExitSet,
                         const CFGBlock* PredBlock,
                         const CFGBlock *CurrBlock);

  Lockset intersectAndWarn(const Lockset &LSet1, const Lockset &LSet2,
                           SourceLocation JoinLoc, LockErrorKind LEK);

  void runAnalysis(AnalysisDeclContext &AC);
};


/// \brief Add a new lock to the lockset, warning if the lock is already there.
/// \param Mutex -- the Mutex expression for the lock
/// \param LDat  -- the LockData for the lock
Lockset ThreadSafetyAnalyzer::addLock(const Lockset &LSet,
                                      const MutexID &Mutex,
                                      const LockData &LDat) {
  // FIXME: deal with acquired before/after annotations.
  // FIXME: Don't always warn when we have support for reentrant locks.
  if (LSet.lookup(Mutex)) {
    Handler.handleDoubleLock(Mutex.getName(), LDat.AcquireLoc);
    return LSet;
  } else {
    return LocksetFactory.add(LSet, Mutex, LDat);
  }
}

/// \brief Construct a new mutex and add it to the lockset.
Lockset ThreadSafetyAnalyzer::addLock(const Lockset &LSet,
                                      Expr *MutexExp, const NamedDecl *D,
                                      const LockData &LDat) {
  MutexID Mutex(MutexExp, 0, D);
  if (!Mutex.isValid()) {
    MutexID::warnInvalidLock(Handler, MutexExp, 0, D);
    return LSet;
  }
  return addLock(LSet, Mutex, LDat);
}


/// \brief Remove a lock from the lockset, warning if the lock is not there.
/// \param LockExp The lock expression corresponding to the lock to be removed
/// \param UnlockLoc The source location of the unlock (only used in error msg)
Lockset ThreadSafetyAnalyzer::removeLock(const Lockset &LSet,
                                         const MutexID &Mutex,
                                         SourceLocation UnlockLoc) {
  const LockData *LDat = LSet.lookup(Mutex);
  if (!LDat) {
    Handler.handleUnmatchedUnlock(Mutex.getName(), UnlockLoc);
    return LSet;
  }
  else {
    Lockset Result = LSet;
    // For scoped-lockable vars, remove the mutex associated with this var.
    if (LDat->UnderlyingMutex.isValid())
      Result = removeLock(Result, LDat->UnderlyingMutex, UnlockLoc);
    return LocksetFactory.remove(Result, Mutex);
  }
}

/// \brief This function, parameterized by an attribute type, is used to add a
/// set of locks specified as attribute arguments to the lockset.
template <typename AttrType>
Lockset ThreadSafetyAnalyzer::addLocksToSet(const Lockset &LSet,
                                            LockKind LK, AttrType *Attr,
                                            Expr *Exp, NamedDecl* FunDecl,
                                            VarDecl *VD) {
  typedef typename AttrType::args_iterator iterator_type;

  SourceLocation ExpLocation = Exp->getExprLoc();

  // Figure out if we're calling the constructor of scoped lockable class
  bool isScopedVar = false;
  if (VD) {
    if (CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FunDecl)) {
      CXXRecordDecl* PD = CD->getParent();
      if (PD && PD->getAttr<ScopedLockableAttr>())
        isScopedVar = true;
    }
  }

  if (Attr->args_size() == 0) {
    // The mutex held is the "this" object.
    MutexID Mutex(0, Exp, FunDecl);
    if (!Mutex.isValid()) {
      MutexID::warnInvalidLock(Handler, 0, Exp, FunDecl);
      return LSet;
    }
    else {
      return addLock(LSet, Mutex, LockData(ExpLocation, LK));
    }
  }

  Lockset Result = LSet;
  for (iterator_type I=Attr->args_begin(), E=Attr->args_end(); I != E; ++I) {
    MutexID Mutex(*I, Exp, FunDecl);
    if (!Mutex.isValid())
      MutexID::warnInvalidLock(Handler, *I, Exp, FunDecl);
    else {
      Result = addLock(Result, Mutex, LockData(ExpLocation, LK));
      if (isScopedVar) {
        // For scoped lockable vars, map this var to its underlying mutex.
        DeclRefExpr DRE(VD, false, VD->getType(), VK_LValue, VD->getLocation());
        MutexID SMutex(&DRE, 0, 0);
        Result = addLock(Result, SMutex,
                         LockData(VD->getLocation(), LK, Mutex));
      }
    }
  }
  return Result;
}

/// \brief This function removes a set of locks specified as attribute
/// arguments from the lockset.
Lockset ThreadSafetyAnalyzer::removeLocksFromSet(const Lockset &LSet,
                                                 UnlockFunctionAttr *Attr,
                                                 Expr *Exp, NamedDecl* FunDecl) {
  SourceLocation ExpLocation;
  if (Exp) ExpLocation = Exp->getExprLoc();

  if (Attr->args_size() == 0) {
    // The mutex held is the "this" object.
    MutexID Mu(0, Exp, FunDecl);
    if (!Mu.isValid()) {
      MutexID::warnInvalidLock(Handler, 0, Exp, FunDecl);
      return LSet;
    } else {
      return removeLock(LSet, Mu, ExpLocation);
    }
  }

  Lockset Result = LSet;
  for (UnlockFunctionAttr::args_iterator I = Attr->args_begin(),
       E = Attr->args_end(); I != E; ++I) {
    MutexID Mutex(*I, Exp, FunDecl);
    if (!Mutex.isValid())
      MutexID::warnInvalidLock(Handler, *I, Exp, FunDecl);
    else
      Result = removeLock(Result, Mutex, ExpLocation);
  }
  return Result;
}


/// \brief Add lock to set, if the current block is in the taken branch of a
/// trylock.
template <class AttrType>
Lockset ThreadSafetyAnalyzer::addTrylock(const Lockset &LSet,
                                         LockKind LK, AttrType *Attr,
                                         Expr *Exp, NamedDecl *FunDecl,
                                         const CFGBlock *PredBlock,
                                         const CFGBlock *CurrBlock,
                                         Expr *BrE, bool Neg) {
  // Find out which branch has the lock
  bool branch = 0;
  if (CXXBoolLiteralExpr *BLE = dyn_cast_or_null<CXXBoolLiteralExpr>(BrE)) {
    branch = BLE->getValue();
  }
  else if (IntegerLiteral *ILE = dyn_cast_or_null<IntegerLiteral>(BrE)) {
    branch = ILE->getValue().getBoolValue();
  }
  int branchnum = branch ? 0 : 1;
  if (Neg) branchnum = !branchnum;

  Lockset Result = LSet;
  // If we've taken the trylock branch, then add the lock
  int i = 0;
  for (CFGBlock::const_succ_iterator SI = PredBlock->succ_begin(),
       SE = PredBlock->succ_end(); SI != SE && i < 2; ++SI, ++i) {
    if (*SI == CurrBlock && i == branchnum) {
      Result = addLocksToSet(Result, LK, Attr, Exp, FunDecl, 0);
    }
  }
  return Result;
}


// If Cond can be traced back to a function call, return the call expression.
// The negate variable should be called with false, and will be set to true
// if the function call is negated, e.g. if (!mu.tryLock(...))
const CallExpr* ThreadSafetyAnalyzer::getTrylockCallExpr(const Stmt *Cond,
                                                         LocalVarContext C,
                                                         bool &Negate) {
  if (!Cond)
    return 0;

  if (const CallExpr *CallExp = dyn_cast<CallExpr>(Cond)) {
    return CallExp;
  }
  else if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(Cond)) {
    return getTrylockCallExpr(CE->getSubExpr(), C, Negate);
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
  }
  // FIXME -- handle && and || as well.
  return NULL;
}


/// \brief Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
Lockset ThreadSafetyAnalyzer::getEdgeLockset(const Lockset &ExitSet,
                                             const CFGBlock *PredBlock,
                                             const CFGBlock *CurrBlock) {
  if (!PredBlock->getTerminatorCondition())
    return ExitSet;

  bool Negate = false;
  const Stmt *Cond = PredBlock->getTerminatorCondition();
  const CFGBlockInfo *PredBlockInfo = &BlockInfo[PredBlock->getBlockID()];
  const LocalVarContext &LVarCtx = PredBlockInfo->ExitContext;

  CallExpr *Exp = const_cast<CallExpr*>(
    getTrylockCallExpr(Cond, LVarCtx, Negate));
  if (!Exp)
    return ExitSet;

  NamedDecl *FunDecl = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());
  if(!FunDecl || !FunDecl->hasAttrs())
    return ExitSet;

  Lockset Result = ExitSet;

  // If the condition is a call to a Trylock function, then grab the attributes
  AttrVec &ArgAttrs = FunDecl->getAttrs();
  for (unsigned i = 0; i < ArgAttrs.size(); ++i) {
    Attr *Attr = ArgAttrs[i];
    switch (Attr->getKind()) {
      case attr::ExclusiveTrylockFunction: {
        ExclusiveTrylockFunctionAttr *A =
          cast<ExclusiveTrylockFunctionAttr>(Attr);
        Result = addTrylock(Result, LK_Exclusive, A, Exp, FunDecl,
                            PredBlock, CurrBlock,
                            A->getSuccessValue(), Negate);
        break;
      }
      case attr::SharedTrylockFunction: {
        SharedTrylockFunctionAttr *A =
          cast<SharedTrylockFunctionAttr>(Attr);
        Result = addTrylock(Result, LK_Shared, A, Exp, FunDecl,
                            PredBlock, CurrBlock,
                            A->getSuccessValue(), Negate);
        break;
      }
      default:
        break;
    }
  }
  return Result;
}


/// \brief We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public StmtVisitor<BuildLockset> {
  friend class ThreadSafetyAnalyzer;

  ThreadSafetyAnalyzer *Analyzer;
  Lockset LSet;
  LocalVariableMap::Context LVarCtx;
  unsigned CtxIndex;

  // Helper functions
  const ValueDecl *getValueDecl(Expr *Exp);

  void warnIfMutexNotHeld(const NamedDecl *D, Expr *Exp, AccessKind AK,
                          Expr *MutexExp, ProtectedOperationKind POK);

  void checkAccess(Expr *Exp, AccessKind AK);
  void checkDereference(Expr *Exp, AccessKind AK);
  void handleCall(Expr *Exp, NamedDecl *D, VarDecl *VD = 0);

  /// \brief Returns true if the lockset contains a lock, regardless of whether
  /// the lock is held exclusively or shared.
  bool locksetContains(const MutexID &Lock) const {
    return LSet.lookup(Lock);
  }

  /// \brief Returns true if the lockset contains a lock with the passed in
  /// locktype.
  bool locksetContains(const MutexID &Lock, LockKind KindRequested) const {
    const LockData *LockHeld = LSet.lookup(Lock);
    return (LockHeld && KindRequested == LockHeld->LKind);
  }

  /// \brief Returns true if the lockset contains a lock with at least the
  /// passed in locktype. So for example, if we pass in LK_Shared, this function
  /// returns true if the lock is held LK_Shared or LK_Exclusive. If we pass in
  /// LK_Exclusive, this function returns true if the lock is held LK_Exclusive.
  bool locksetContainsAtLeast(const MutexID &Lock,
                              LockKind KindRequested) const {
    switch (KindRequested) {
      case LK_Shared:
        return locksetContains(Lock);
      case LK_Exclusive:
        return locksetContains(Lock, KindRequested);
    }
    llvm_unreachable("Unknown LockKind");
  }

public:
  BuildLockset(ThreadSafetyAnalyzer *Anlzr, CFGBlockInfo &Info)
    : StmtVisitor<BuildLockset>(),
      Analyzer(Anlzr),
      LSet(Info.EntrySet),
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


/// \brief Gets the value decl pointer from DeclRefExprs or MemberExprs
const ValueDecl *BuildLockset::getValueDecl(Expr *Exp) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return 0;
}

/// \brief Warn if the LSet does not contain a lock sufficient to protect access
/// of at least the passed in AccessKind.
void BuildLockset::warnIfMutexNotHeld(const NamedDecl *D, Expr *Exp,
                                      AccessKind AK, Expr *MutexExp,
                                      ProtectedOperationKind POK) {
  LockKind LK = getLockKindFromAccessKind(AK);

  MutexID Mutex(MutexExp, Exp, D);
  if (!Mutex.isValid())
    MutexID::warnInvalidLock(Analyzer->Handler, MutexExp, Exp, D);
  else if (!locksetContainsAtLeast(Mutex, LK))
    Analyzer->Handler.handleMutexNotHeld(D, POK, Mutex.getName(), LK,
                                         Exp->getExprLoc());
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
    Analyzer->Handler.handleNoMutexHeld(D, POK_VarDereference, AK,
                                        Exp->getExprLoc());

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
    Analyzer->Handler.handleNoMutexHeld(D, POK_VarAccess, AK,
                                        Exp->getExprLoc());

  const AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0, Size = ArgAttrs.size(); i < Size; ++i)
    if (GuardedByAttr *GBAttr = dyn_cast<GuardedByAttr>(ArgAttrs[i]))
      warnIfMutexNotHeld(D, Exp, AK, GBAttr->getArg(), POK_VarAccess);
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
/// FIXME: We need to also visit CallExprs to catch/check global functions.
///
/// FIXME: Do not flag an error for member variables accessed in constructors/
/// destructors
void BuildLockset::handleCall(Expr *Exp, NamedDecl *D, VarDecl *VD) {
  AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0; i < ArgAttrs.size(); ++i) {
    Attr *Attr = ArgAttrs[i];
    switch (Attr->getKind()) {
      // When we encounter an exclusive lock function, we need to add the lock
      // to our lockset with kind exclusive.
      case attr::ExclusiveLockFunction: {
        ExclusiveLockFunctionAttr *A = cast<ExclusiveLockFunctionAttr>(Attr);
        LSet = Analyzer->addLocksToSet(LSet, LK_Exclusive, A, Exp, D, VD);
        break;
      }

      // When we encounter a shared lock function, we need to add the lock
      // to our lockset with kind shared.
      case attr::SharedLockFunction: {
        SharedLockFunctionAttr *A = cast<SharedLockFunctionAttr>(Attr);
        LSet = Analyzer->addLocksToSet(LSet, LK_Shared, A, Exp, D, VD);
        break;
      }

      // When we encounter an unlock function, we need to remove unlocked
      // mutexes from the lockset, and flag a warning if they are not there.
      case attr::UnlockFunction: {
        UnlockFunctionAttr *UFAttr = cast<UnlockFunctionAttr>(Attr);
        LSet = Analyzer->removeLocksFromSet(LSet, UFAttr, Exp, D);
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
          MutexID Mutex(*I, Exp, D);
          if (!Mutex.isValid())
            MutexID::warnInvalidLock(Analyzer->Handler, *I, Exp, D);
          else if (locksetContains(Mutex))
            Analyzer->Handler.handleFunExcludesLock(D->getName(),
                                                    Mutex.getName(),
                                                    Exp->getExprLoc());
        }
        break;
      }

      // Ignore other (non thread-safety) attributes
      default:
        break;
    }
  }
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

  // adjust the context
  LVarCtx = Analyzer->LocalVarMap.getNextContext(CtxIndex, BO, LVarCtx);

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


void BuildLockset::VisitCallExpr(CallExpr *Exp) {
  NamedDecl *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());
  if(!D || !D->hasAttrs())
    return;
  handleCall(Exp, D);
}

void BuildLockset::VisitCXXConstructExpr(CXXConstructExpr *Exp) {
  // FIXME -- only handles constructors in DeclStmt below.
}

void BuildLockset::VisitDeclStmt(DeclStmt *S) {
  // adjust the context
  LVarCtx = Analyzer->LocalVarMap.getNextContext(CtxIndex, S, LVarCtx);

  DeclGroupRef DGrp = S->getDeclGroup();
  for (DeclGroupRef::iterator I = DGrp.begin(), E = DGrp.end(); I != E; ++I) {
    Decl *D = *I;
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(D)) {
      Expr *E = VD->getInit();
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
/// \param LSet1 The first lockset.
/// \param LSet2 The second lockset.
/// \param JoinLoc The location of the join point for error reporting
/// \param LEK The error message to report.
Lockset ThreadSafetyAnalyzer::intersectAndWarn(const Lockset &LSet1,
                                               const Lockset &LSet2,
                                               SourceLocation JoinLoc,
                                               LockErrorKind LEK) {
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
          Intersection = LocksetFactory.add(Intersection, LSet2Mutex,
                                            LSet2LockData);
      }
    } else {
      Handler.handleMutexHeldEndOfScope(LSet2Mutex.getName(),
                                        LSet2LockData.AcquireLoc,
                                        JoinLoc, LEK);
    }
  }

  for (Lockset::iterator I = LSet1.begin(), E = LSet1.end(); I != E; ++I) {
    if (!LSet2.contains(I.getKey())) {
      const MutexID &Mutex = I.getKey();
      const LockData &MissingLock = I.getData();
      Handler.handleMutexHeldEndOfScope(Mutex.getName(),
                                        MissingLock.AcquireLoc,
                                        JoinLoc, LEK);
      Intersection = LocksetFactory.remove(Intersection, Mutex);
    }
  }
  return Intersection;
}


/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void ThreadSafetyAnalyzer::runAnalysis(AnalysisDeclContext &AC) {
  CFG *CFGraph = AC.getCFG();
  if (!CFGraph) return;
  const NamedDecl *D = dyn_cast_or_null<NamedDecl>(AC.getDecl());

  // AC.dumpCFG(true);

  if (!D)
    return;  // Ignore anonymous functions for now.
  if (D->getAttr<NoThreadSafetyAnalysisAttr>())
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
    CFGBlockInfo::getEmptyBlockInfo(LocksetFactory, LocalVarMap));

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  PostOrderCFGView *SortedGraph = AC.getAnalysis<PostOrderCFGView>();
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  // Compute SSA names for local variables
  LocalVarMap.traverseCFG(CFGraph, SortedGraph, BlockInfo);

  // Fill in source locations for all CFGBlocks.
  findBlockLocations(CFGraph, SortedGraph, BlockInfo);

  // Add locks from exclusive_locks_required and shared_locks_required
  // to initial lockset. Also turn off checking for lock and unlock functions.
  // FIXME: is there a more intelligent way to check lock/unlock functions?
  if (!SortedGraph->empty() && D->hasAttrs()) {
    const CFGBlock *FirstBlock = *SortedGraph->begin();
    Lockset &InitialLockset = BlockInfo[FirstBlock->getBlockID()].EntrySet;
    const AttrVec &ArgAttrs = D->getAttrs();
    for (unsigned i = 0; i < ArgAttrs.size(); ++i) {
      Attr *Attr = ArgAttrs[i];
      SourceLocation AttrLoc = Attr->getLocation();
      if (SharedLocksRequiredAttr *SLRAttr
            = dyn_cast<SharedLocksRequiredAttr>(Attr)) {
        for (SharedLocksRequiredAttr::args_iterator
             SLRIter = SLRAttr->args_begin(),
             SLREnd = SLRAttr->args_end(); SLRIter != SLREnd; ++SLRIter)
          InitialLockset = addLock(InitialLockset, *SLRIter, D,
                                   LockData(AttrLoc, LK_Shared));
      } else if (ExclusiveLocksRequiredAttr *ELRAttr
                   = dyn_cast<ExclusiveLocksRequiredAttr>(Attr)) {
        for (ExclusiveLocksRequiredAttr::args_iterator
             ELRIter = ELRAttr->args_begin(),
             ELREnd = ELRAttr->args_end(); ELRIter != ELREnd; ++ELRIter)
          InitialLockset = addLock(InitialLockset, *ELRIter, D,
                                   LockData(AttrLoc, LK_Exclusive));
      } else if (isa<UnlockFunctionAttr>(Attr)) {
        // Don't try to check unlock functions for now
        return;
      } else if (isa<ExclusiveLockFunctionAttr>(Attr)) {
        // Don't try to check lock functions for now
        return;
      } else if (isa<SharedLockFunctionAttr>(Attr)) {
        // Don't try to check lock functions for now
        return;
      }
    }
  }

  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I!= E; ++I) {
    const CFGBlock *CurrBlock = *I;
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
    llvm::SmallVector<CFGBlock*, 8> SpecialBlocks;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {

      // if *PI -> CurrBlock is a back edge
      if (*PI == 0 || !VisitedBlocks.alreadySet(*PI))
        continue;

      // Ignore edges from blocks that can't return.
      if ((*PI)->hasNoReturnElement())
        continue;

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

      int PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];
      Lockset PrevLockset =
        getEdgeLockset(PrevBlockInfo->ExitSet, *PI, CurrBlock);

      if (!LocksetInitialized) {
        CurrBlockInfo->EntrySet = PrevLockset;
        LocksetInitialized = true;
      } else {
        CurrBlockInfo->EntrySet =
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                           CurrBlockInfo->EntryLoc,
                           LEK_LockedSomePredecessors);
      }
    }

    // Process continue and break blocks. Assume that the lockset for the
    // resulting block is unaffected by any discrepancies in them.
    for (unsigned SpecialI = 0, SpecialN = SpecialBlocks.size();
         SpecialI < SpecialN; ++SpecialI) {
      CFGBlock *PrevBlock = SpecialBlocks[SpecialI];
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

        Lockset PrevLockset =
          getEdgeLockset(PrevBlockInfo->ExitSet, PrevBlock, CurrBlock);

        // Do not update EntrySet.
        intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                         PrevBlockInfo->ExitLoc,
                         IsLoop ? LEK_LockedSomeLoopIterations
                                : LEK_LockedSomePredecessors);
      }
    }

    BuildLockset LocksetBuilder(this, *CurrBlockInfo);

    // Visit all the statements in the basic block.
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      switch (BI->getKind()) {
        case CFGElement::Statement: {
          const CFGStmt *CS = cast<CFGStmt>(&*BI);
          LocksetBuilder.Visit(const_cast<Stmt*>(CS->getStmt()));
          break;
        }
        // Ignore BaseDtor, MemberDtor, and TemporaryDtor for now.
        case CFGElement::AutomaticObjectDtor: {
          const CFGAutomaticObjDtor *AD = cast<CFGAutomaticObjDtor>(&*BI);
          CXXDestructorDecl *DD = const_cast<CXXDestructorDecl*>(
            AD->getDestructorDecl(AC.getASTContext()));
          if (!DD->hasAttrs())
            break;

          // Create a dummy expression,
          VarDecl *VD = const_cast<VarDecl*>(AD->getVarDecl());
          DeclRefExpr DRE(VD, false, VD->getType(), VK_LValue,
                          AD->getTriggerStmt()->getLocEnd());
          LocksetBuilder.handleCall(&DRE, DD);
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitSet = LocksetBuilder.LSet;

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
      CFGBlockInfo *PreLoop = &BlockInfo[FirstLoopBlock->getBlockID()];
      CFGBlockInfo *LoopEnd = &BlockInfo[CurrBlockID];
      intersectAndWarn(LoopEnd->ExitSet, PreLoop->EntrySet,
                       PreLoop->EntryLoc,
                       LEK_LockedSomeLoopIterations);
    }
  }

  CFGBlockInfo *Initial = &BlockInfo[CFGraph->getEntry().getBlockID()];
  CFGBlockInfo *Final   = &BlockInfo[CFGraph->getExit().getBlockID()];

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(Initial->EntrySet, Final->ExitSet,
                   Final->ExitLoc,
                   LEK_LockedAtEndOfFunction);
}

} // end anonymous namespace


namespace clang {
namespace thread_safety {

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

}} // end namespace clang::thread_safety
