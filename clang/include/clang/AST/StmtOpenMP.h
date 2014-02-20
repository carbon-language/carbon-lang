//===- StmtOpenMP.h - Classes for OpenMP directives  ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file defines OpenMP AST classes for executable directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTOPENMP_H
#define LLVM_CLANG_AST_STMTOPENMP_H

#include "clang/AST/Expr.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for directives.
//===----------------------------------------------------------------------===//

/// \brief This is a basic class for representing single OpenMP executable
/// directive.
///
class OMPExecutableDirective : public Stmt {
  friend class ASTStmtReader;
  /// \brief Kind of the directive.
  OpenMPDirectiveKind Kind;
  /// \brief Starting location of the directive (directive keyword).
  SourceLocation StartLoc;
  /// \brief Ending location of the directive.
  SourceLocation EndLoc;
  /// \brief Pointer to the list of clauses.
  llvm::MutableArrayRef<OMPClause *> Clauses;
  /// \brief Associated statement (if any) and expressions.
  llvm::MutableArrayRef<Stmt *> StmtAndExpressions;
protected:
  /// \brief Build instance of directive of class \a K.
  ///
  /// \param SC Statement class.
  /// \param K Kind of OpenMP directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  ///
  template <typename T>
  OMPExecutableDirective(const T *, StmtClass SC, OpenMPDirectiveKind K,
                         SourceLocation StartLoc, SourceLocation EndLoc,
                         unsigned NumClauses, unsigned NumberOfExpressions)
    : Stmt(SC), Kind(K), StartLoc(StartLoc), EndLoc(EndLoc),
      Clauses(reinterpret_cast<OMPClause **>(static_cast<T *>(this) + 1),
              NumClauses),
      StmtAndExpressions(reinterpret_cast<Stmt **>(Clauses.end()),
                         NumberOfExpressions) { }

  /// \brief Sets the list of variables for this clause.
  ///
  /// \param Clauses The list of clauses for the directive.
  ///
  void setClauses(ArrayRef<OMPClause *> Clauses);

  /// \brief Set the associated statement for the directive.
  ///
  /// /param S Associated statement.
  ///
  void setAssociatedStmt(Stmt *S) {
    StmtAndExpressions[0] = S;
  }

public:
  /// \brief Returns starting location of directive kind.
  SourceLocation getLocStart() const { return StartLoc; }
  /// \brief Returns ending location of directive.
  SourceLocation getLocEnd() const { return EndLoc; }

  /// \brief Set starting location of directive kind.
  ///
  /// \param Loc New starting location of directive.
  ///
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// \brief Set ending location of directive.
  ///
  /// \param Loc New ending location of directive.
  ///
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// \brief Get number of clauses.
  unsigned getNumClauses() const { return Clauses.size(); }

  /// \brief Returns specified clause.
  ///
  /// \param i Number of clause.
  ///
  OMPClause *getClause(unsigned i) const {
    assert(i < Clauses.size() && "index out of bound!");
    return Clauses[i];
  }

  /// \brief Returns statement associated with the directive.
  Stmt *getAssociatedStmt() const {
    return StmtAndExpressions[0];
  }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOMPExecutableDirectiveConstant &&
           S->getStmtClass() <= lastOMPExecutableDirectiveConstant;
  }

  child_range children() {
    return child_range(StmtAndExpressions.begin(), StmtAndExpressions.end());
  }

  ArrayRef<OMPClause *> clauses() { return Clauses; }

  ArrayRef<OMPClause *> clauses() const { return Clauses; }
};

/// \brief This represents '#pragma omp parallel' directive.
///
/// \code
/// #pragma omp parallel private(a,b) reduction(+: c,d)
/// \endcode
/// In this example directive '#pragma omp parallel' has clauses 'private'
/// with the variables 'a' and 'b' and 'reduction' with operator '+' and
/// variables 'c' and 'd'.
///
class OMPParallelDirective : public OMPExecutableDirective {
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending Location of the directive.
  ///
  OMPParallelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned N)
    : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                             StartLoc, EndLoc, N, 1) { }

  /// \brief Build an empty directive.
  ///
  /// \param N Number of clauses.
  ///
  explicit OMPParallelDirective(unsigned N)
    : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                             SourceLocation(), SourceLocation(), N, 1) { }
public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement associated with the directive.
  ///
  static OMPParallelDirective *Create(const ASTContext &C,
                                      SourceLocation StartLoc,
                                      SourceLocation EndLoc,
                                      ArrayRef<OMPClause *> Clauses,
                                      Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param N The number of clauses.
  ///
  static OMPParallelDirective *CreateEmpty(const ASTContext &C, unsigned N,
                                           EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelDirectiveClass;
  }
};

}  // end namespace clang

#endif
