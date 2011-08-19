//===---  BugReporterVisitor.h - Generate PathDiagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares BugReporterVisitors, which are used to generate enhanced
//  diagnostic traces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_BUGREPORTERVISITOR
#define LLVM_CLANG_GR_BUGREPORTERVISITOR

#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {

namespace ento {

class BugReport;
class BugReporterContext;
class ExplodedNode;
class MemRegion;
class PathDiagnosticPiece;

class BugReporterVisitor : public llvm::FoldingSetNode {
public:
  virtual ~BugReporterVisitor();

  /// \brief Return a diagnostic piece which should be associated with the
  /// given node.
  ///
  /// The last parameter can be used to register a new visitor with the given
  /// BugReport while processing a node.
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *PrevN,
                                         BugReporterContext &BRC,
                                         BugReport &BR) = 0;

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;
};

class FindLastStoreBRVisitor : public BugReporterVisitor {
  const MemRegion *R;
  SVal V;
  bool satisfied;
  const ExplodedNode *StoreSite;

public:
  /// \brief Convenience method to create a visitor given only the MemRegion.
  /// Returns NULL if the visitor cannot be created. For example, when the
  /// corresponding value is unknown.
  static BugReporterVisitor *createVisitorObject(const ExplodedNode *N,
                                                 const MemRegion *R);

  /// Creates a visitor for every VarDecl inside a Stmt and registers it with
  /// the BugReport.
  static void registerStatementVarDecls(BugReport &BR, const Stmt *S);

  FindLastStoreBRVisitor(SVal v, const MemRegion *r)
  : R(r), V(v), satisfied(false), StoreSite(0) {
    assert (!V.isUnknown() && "Cannot track unknown value.");

    // TODO: Does it make sense to allow undef values here?
    // (If not, also see UndefCapturedBlockVarChecker)?
  }

  void Profile(llvm::FoldingSetNodeID &ID) const;

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

class TrackConstraintBRVisitor : public BugReporterVisitor {
  DefinedSVal Constraint;
  const bool Assumption;
  bool isSatisfied;

public:
  TrackConstraintBRVisitor(DefinedSVal constraint, bool assumption)
  : Constraint(constraint), Assumption(assumption), isSatisfied(false) {}

  void Profile(llvm::FoldingSetNodeID &ID) const;

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

class NilReceiverBRVisitor : public BugReporterVisitor {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR);
};

/// Visitor that tries to report interesting diagnostics from conditions.
class ConditionBRVisitor : public BugReporterVisitor {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *Prev,
                                         BugReporterContext &BRC,
                                         BugReport &BR);

  PathDiagnosticPiece *VisitTerminator(const Stmt *Term,
                                       const ProgramState *CurrentState,
                                       const ProgramState *PrevState,
                                       const CFGBlock *srcBlk,
                                       const CFGBlock *dstBlk,
                                       BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     bool tookTrue,
                                     BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const DeclRefExpr *DR,
                                     const bool tookTrue,
                                     BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const BinaryOperator *BExpr,
                                     const bool tookTrue,
                                     BugReporterContext &BRC);

  bool patternMatch(const Expr *Ex,
                    llvm::raw_ostream &Out,
                    BugReporterContext &BRC);
};

namespace bugreporter {

BugReporterVisitor *getTrackNullOrUndefValueVisitor(const ExplodedNode *N,
                                                    const Stmt *S);

const Stmt *GetDerefExpr(const ExplodedNode *N);
const Stmt *GetDenomExpr(const ExplodedNode *N);
const Stmt *GetCalleeExpr(const ExplodedNode *N);
const Stmt *GetRetValExpr(const ExplodedNode *N);

} // end namespace clang
} // end namespace ento
} // end namespace bugreporter


#endif //LLVM_CLANG_GR__BUGREPORTERVISITOR
