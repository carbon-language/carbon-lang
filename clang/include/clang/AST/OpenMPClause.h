//===- OpenMPClause.h - Classes for OpenMP clauses --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file defines OpenMP AST classes for clauses.
/// There are clauses for executable directives, clauses for declarative
/// directives and clauses which can be used in both kinds of directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENMPCLAUSE_H
#define LLVM_CLANG_AST_OPENMPCLAUSE_H

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for clauses.
//===----------------------------------------------------------------------===//

/// \brief This is a basic class for representing single OpenMP clause.
///
class OMPClause {
  /// \brief Starting location of the clause (the clause keyword).
  SourceLocation StartLoc;
  /// \brief Ending location of the clause.
  SourceLocation EndLoc;
  /// \brief Kind of the clause.
  OpenMPClauseKind Kind;

protected:
  OMPClause(OpenMPClauseKind K, SourceLocation StartLoc, SourceLocation EndLoc)
      : StartLoc(StartLoc), EndLoc(EndLoc), Kind(K) {}

public:
  /// \brief Returns the starting location of the clause.
  SourceLocation getLocStart() const { return StartLoc; }
  /// \brief Returns the ending location of the clause.
  SourceLocation getLocEnd() const { return EndLoc; }

  /// \brief Sets the starting location of the clause.
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// \brief Sets the ending location of the clause.
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// \brief Returns kind of OpenMP clause (private, shared, reduction, etc.).
  OpenMPClauseKind getClauseKind() const { return Kind; }

  bool isImplicit() const { return StartLoc.isInvalid(); }

  StmtRange children();
  ConstStmtRange children() const {
    return const_cast<OMPClause *>(this)->children();
  }
  static bool classof(const OMPClause *T) { return true; }
};

/// \brief This represents clauses with the list of variables like 'private',
/// 'firstprivate', 'copyin', 'shared', or 'reduction' clauses in the
/// '#pragma omp ...' directives.
template <class T> class OMPVarList {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of variables in the list.
  unsigned NumVars;

protected:
  /// \brief Fetches list of variables associated with this clause.
  llvm::MutableArrayRef<Expr *> getVarRefs() {
    return llvm::MutableArrayRef<Expr *>(
        reinterpret_cast<Expr **>(static_cast<T *>(this) + 1), NumVars);
  }

  /// \brief Sets the list of variables for this clause.
  void setVarRefs(ArrayRef<Expr *> VL) {
    assert(VL.size() == NumVars &&
           "Number of variables is not the same as the preallocated buffer");
    std::copy(VL.begin(), VL.end(),
              reinterpret_cast<Expr **>(static_cast<T *>(this) + 1));
  }

  /// \brief Build clause with number of variables \a N.
  ///
  /// \param N Number of the variables in the clause.
  ///
  OMPVarList(SourceLocation LParenLoc, unsigned N)
      : LParenLoc(LParenLoc), NumVars(N) {}

public:
  typedef llvm::MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }
  varlist_iterator varlist_begin() { return getVarRefs().begin(); }
  varlist_iterator varlist_end() { return getVarRefs().end(); }
  varlist_const_iterator varlist_begin() const { return getVarRefs().begin(); }
  varlist_const_iterator varlist_end() const { return getVarRefs().end(); }

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Fetches list of all variables in the clause.
  ArrayRef<const Expr *> getVarRefs() const {
    return ArrayRef<const Expr *>(
        reinterpret_cast<const Expr *const *>(static_cast<const T *>(this) + 1),
        NumVars);
  }
};

/// \brief This represents 'if' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp parallel if(a > 5)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'if'
/// clause with condition 'a > 5'.
///
class OMPIfClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Condition of the 'if' clause.
  Stmt *Condition;

  /// \brief Set condition.
  ///
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// \brief Build 'if' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPIfClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
              SourceLocation EndLoc)
      : OMPClause(OMPC_if, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// \brief Build an empty clause.
  ///
  OMPIfClause()
      : OMPClause(OMPC_if, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Condition(0) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_if;
  }

  StmtRange children() { return StmtRange(&Condition, &Condition + 1); }
};

/// \brief This represents 'default' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp parallel default(shared)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'default'
/// clause with kind 'shared'.
///
class OMPDefaultClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'default' clause.
  OpenMPDefaultClauseKind Kind;
  /// \brief Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// \brief Set kind of the clauses.
  ///
  /// \param K Argument of clause.
  ///
  void setDefaultKind(OpenMPDefaultClauseKind K) { Kind = K; }

  /// \brief Set argument location.
  ///
  /// \param KLoc Argument location.
  ///
  void setDefaultKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// \brief Build 'default' clause with argument \a A ('none' or 'shared').
  ///
  /// \param A Argument of the clause ('none' or 'shared').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPDefaultClause(OpenMPDefaultClauseKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_default, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// \brief Build an empty clause.
  ///
  OMPDefaultClause()
      : OMPClause(OMPC_default, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Kind(OMPC_DEFAULT_unknown),
        KindKwLoc(SourceLocation()) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns kind of the clause.
  OpenMPDefaultClauseKind getDefaultKind() const { return Kind; }

  /// \brief Returns location of clause kind.
  SourceLocation getDefaultKindKwLoc() const { return KindKwLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_default;
  }

  StmtRange children() { return StmtRange(); }
};

/// \brief This represents clause 'private' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel private(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'private'
/// with the variables 'a' and 'b'.
///
class OMPPrivateClause : public OMPClause, public OMPVarList<OMPPrivateClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPPrivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc, unsigned N)
      : OMPClause(OMPC_private, StartLoc, EndLoc),
        OMPVarList<OMPPrivateClause>(LParenLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPPrivateClause(unsigned N)
      : OMPClause(OMPC_private, SourceLocation(), SourceLocation()),
        OMPVarList<OMPPrivateClause>(SourceLocation(), N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPPrivateClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation EndLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPPrivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_private;
  }
};

/// \brief This represents clause 'firstprivate' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp parallel firstprivate(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'firstprivate'
/// with the variables 'a' and 'b'.
///
class OMPFirstprivateClause : public OMPClause,
                              public OMPVarList<OMPFirstprivateClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPFirstprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                        SourceLocation EndLoc, unsigned N)
      : OMPClause(OMPC_firstprivate, StartLoc, EndLoc),
        OMPVarList<OMPFirstprivateClause>(LParenLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPFirstprivateClause(unsigned N)
      : OMPClause(OMPC_firstprivate, SourceLocation(), SourceLocation()),
        OMPVarList<OMPFirstprivateClause>(SourceLocation(), N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPFirstprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation EndLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPFirstprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_firstprivate;
  }
};

/// \brief This represents clause 'shared' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel shared(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'shared'
/// with the variables 'a' and 'b'.
///
class OMPSharedClause : public OMPClause, public OMPVarList<OMPSharedClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPSharedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPClause(OMPC_shared, StartLoc, EndLoc),
        OMPVarList<OMPSharedClause>(LParenLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPSharedClause(unsigned N)
      : OMPClause(OMPC_shared, SourceLocation(), SourceLocation()),
        OMPVarList<OMPSharedClause>(SourceLocation(), N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPSharedClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPSharedClause *CreateEmpty(const ASTContext &C, unsigned N);

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_shared;
  }
};

} // end namespace clang

#endif
