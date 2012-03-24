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

/// \brief BugReporterVisitors are used to add custom diagnostics along a path.
///
/// Custom visitors should subclass the BugReporterVisitorImpl class for a
/// default implementation of the clone() method.
/// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
/// default implementation of clone() will NOT do the right thing, and you
/// will have to provide your own implementation.)
class BugReporterVisitor : public llvm::FoldingSetNode {
public:
  virtual ~BugReporterVisitor();

  /// \brief Returns a copy of this BugReporter.
  ///
  /// Custom BugReporterVisitors should not override this method directly.
  /// Instead, they should inherit from BugReporterVisitorImpl and provide
  /// a protected or public copy constructor.
  ///
  /// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
  /// default implementation of clone() will NOT do the right thing, and you
  /// will have to provide your own implementation.)
  virtual BugReporterVisitor *clone() const = 0;

  /// \brief Return a diagnostic piece which should be associated with the
  /// given node.
  ///
  /// The last parameter can be used to register a new visitor with the given
  /// BugReport while processing a node.
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *PrevN,
                                         BugReporterContext &BRC,
                                         BugReport &BR) = 0;

  /// \brief Provide custom definition for the final diagnostic piece on the
  /// path - the piece, which is displayed before the path is expanded.
  ///
  /// If returns NULL the default implementation will be used.
  /// Also note that at most one visitor of a BugReport should generate a
  /// non-NULL end of path diagnostic piece.
  virtual PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                          const ExplodedNode *N,
                                          BugReport &BR);

  virtual void Profile(llvm::FoldingSetNodeID &ID) const = 0;

  /// \brief Generates the default final diagnostic piece.
  static PathDiagnosticPiece *getDefaultEndPath(BugReporterContext &BRC,
                                                const ExplodedNode *N,
                                                BugReport &BR);

};

/// This class provides a convenience implementation for clone() using the
/// Curiously-Recurring Template Pattern. If you are implementing a custom
/// BugReporterVisitor, subclass BugReporterVisitorImpl and provide a public
/// or protected copy constructor.
///
/// (Warning: if you have a deep subclass of BugReporterVisitorImpl, the
/// default implementation of clone() will NOT do the right thing, and you
/// will have to provide your own implementation.)
template <class DERIVED>
class BugReporterVisitorImpl : public BugReporterVisitor {
  virtual BugReporterVisitor *clone() const {
    return new DERIVED(*static_cast<const DERIVED *>(this));
  }
};

class FindLastStoreBRVisitor
  : public BugReporterVisitorImpl<FindLastStoreBRVisitor>
{
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

class TrackConstraintBRVisitor
  : public BugReporterVisitorImpl<TrackConstraintBRVisitor>
{
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

class NilReceiverBRVisitor
  : public BugReporterVisitorImpl<NilReceiverBRVisitor>
{
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
class ConditionBRVisitor : public BugReporterVisitorImpl<ConditionBRVisitor> {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  
  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *Prev,
                                         BugReporterContext &BRC,
                                         BugReport &BR);

  PathDiagnosticPiece *VisitNodeImpl(const ExplodedNode *N,
                                     const ExplodedNode *Prev,
                                     BugReporterContext &BRC,
                                     BugReport &BR);
  
  PathDiagnosticPiece *VisitTerminator(const Stmt *Term,
                                       const ExplodedNode *N,
                                       const CFGBlock *srcBlk,
                                       const CFGBlock *dstBlk,
                                       BugReport &R,
                                       BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const DeclRefExpr *DR,
                                     const bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const BinaryOperator *BExpr,
                                     const bool tookTrue,
                                     BugReporterContext &BRC,
                                     BugReport &R,
                                     const ExplodedNode *N);
  
  PathDiagnosticPiece *VisitConditionVariable(StringRef LhsString,
                                              const Expr *CondVarExpr,
                                              const bool tookTrue,
                                              BugReporterContext &BRC,
                                              BugReport &R,
                                              const ExplodedNode *N);

  bool patternMatch(const Expr *Ex,
                    llvm::raw_ostream &Out,
                    BugReporterContext &BRC,
                    BugReport &R,
                    const ExplodedNode *N,
                    llvm::Optional<bool> &prunable);
};
  
namespace bugreporter {

BugReporterVisitor *getTrackNullOrUndefValueVisitor(const ExplodedNode *N,
                                                    const Stmt *S,
                                                    BugReport *R);

const Stmt *GetDerefExpr(const ExplodedNode *N);
const Stmt *GetDenomExpr(const ExplodedNode *N);
const Stmt *GetCalleeExpr(const ExplodedNode *N);
const Stmt *GetRetValExpr(const ExplodedNode *N);

} // end namespace clang
} // end namespace ento
} // end namespace bugreporter


#endif //LLVM_CLANG_GR__BUGREPORTERVISITOR
