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
template <class T> class OMPVarListClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of variables in the list.
  unsigned NumVars;

protected:
  /// \brief Fetches list of variables associated with this clause.
  llvm::MutableArrayRef<Expr *> getVarRefs() {
    return llvm::MutableArrayRef<Expr *>(
        reinterpret_cast<Expr **>(
            reinterpret_cast<char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<Expr *>())),
        NumVars);
  }

  /// \brief Sets the list of variables for this clause.
  void setVarRefs(ArrayRef<Expr *> VL) {
    assert(VL.size() == NumVars &&
           "Number of variables is not the same as the preallocated buffer");
    std::copy(
        VL.begin(), VL.end(),
        reinterpret_cast<Expr **>(
            reinterpret_cast<char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<Expr *>())));
  }

  /// \brief Build a clause with \a N variables
  ///
  /// \param K Kind of the clause.
  /// \param StartLoc Starting location of the clause (the clause keyword).
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPVarListClause(OpenMPClauseKind K, SourceLocation StartLoc,
                   SourceLocation LParenLoc, SourceLocation EndLoc, unsigned N)
      : OMPClause(K, StartLoc, EndLoc), LParenLoc(LParenLoc), NumVars(N) {}

public:
  typedef llvm::MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
  typedef llvm::iterator_range<varlist_iterator> varlist_range;
  typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }

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
        reinterpret_cast<const Expr *const *>(
            reinterpret_cast<const char *>(this) +
            llvm::RoundUpToAlignment(sizeof(T), llvm::alignOf<Expr *>())),
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
        LParenLoc(SourceLocation()), Condition(nullptr) {}

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

/// \brief This represents 'num_threads' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp parallel num_threads(6)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'num_threads'
/// clause with number of threads '6'.
///
class OMPNumThreadsClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Condition of the 'num_threads' clause.
  Stmt *NumThreads;

  /// \brief Set condition.
  ///
  void setNumThreads(Expr *NThreads) { NumThreads = NThreads; }

public:
  /// \brief Build 'num_threads' clause with condition \a NumThreads.
  ///
  /// \param NumThreads Number of threads for the construct.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPNumThreadsClause(Expr *NumThreads, SourceLocation StartLoc,
                      SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_num_threads, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumThreads(NumThreads) {}

  /// \brief Build an empty clause.
  ///
  OMPNumThreadsClause()
      : OMPClause(OMPC_num_threads, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumThreads(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns number of threads.
  Expr *getNumThreads() const { return cast_or_null<Expr>(NumThreads); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_num_threads;
  }

  StmtRange children() { return StmtRange(&NumThreads, &NumThreads + 1); }
};

/// \brief This represents 'safelen' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp simd safelen(4)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'safelen'
/// with single expression '4'.
/// If the safelen clause is used then no two iterations executed
/// concurrently with SIMD instructions can have a greater distance
/// in the logical iteration space than its value. The parameter of
/// the safelen clause must be a constant positive integer expression.
///
class OMPSafelenClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Safe iteration space distance.
  Stmt *Safelen;

  /// \brief Set safelen.
  void setSafelen(Expr *Len) { Safelen = Len; }

public:
  /// \brief Build 'safelen' clause.
  ///
  /// \param Len Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPSafelenClause(Expr *Len, SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_safelen, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Safelen(Len) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPSafelenClause()
      : OMPClause(OMPC_safelen, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Safelen(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return safe iteration space distance.
  Expr *getSafelen() const { return cast_or_null<Expr>(Safelen); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_safelen;
  }

  StmtRange children() { return StmtRange(&Safelen, &Safelen + 1); }
};

/// \brief This represents 'collapse' clause in the '#pragma omp ...'
/// directive.
///
/// \code
/// #pragma omp simd collapse(3)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'collapse'
/// with single expression '3'.
/// The parameter must be a constant positive integer expression, it specifies
/// the number of nested loops that should be collapsed into a single iteration
/// space.
///
class OMPCollapseClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief Number of for-loops.
  Stmt *NumForLoops;

  /// \brief Set the number of associated for-loops.
  void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
  /// \brief Build 'collapse' clause.
  ///
  /// \param Num Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPCollapseClause(Expr *Num, SourceLocation StartLoc,
                    SourceLocation LParenLoc, SourceLocation EndLoc)
      : OMPClause(OMPC_collapse, StartLoc, EndLoc), LParenLoc(LParenLoc),
        NumForLoops(Num) {}

  /// \brief Build an empty clause.
  ///
  explicit OMPCollapseClause()
      : OMPClause(OMPC_collapse, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), NumForLoops(nullptr) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Return the number of associated for-loops.
  Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_collapse;
  }

  StmtRange children() { return StmtRange(&NumForLoops, &NumForLoops + 1); }
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

/// \brief This represents 'proc_bind' clause in the '#pragma omp ...' directive.
///
/// \code
/// #pragma omp parallel proc_bind(master)
/// \endcode
/// In this example directive '#pragma omp parallel' has simple 'proc_bind'
/// clause with kind 'master'.
///
class OMPProcBindClause : public OMPClause {
  friend class OMPClauseReader;
  /// \brief Location of '('.
  SourceLocation LParenLoc;
  /// \brief A kind of the 'proc_bind' clause.
  OpenMPProcBindClauseKind Kind;
  /// \brief Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// \brief Set kind of the clause.
  ///
  /// \param K Kind of clause.
  ///
  void setProcBindKind(OpenMPProcBindClauseKind K) { Kind = K; }

  /// \brief Set clause kind location.
  ///
  /// \param KLoc Kind location.
  ///
  void setProcBindKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// \brief Build 'proc_bind' clause with argument \a A ('master', 'close' or
  ///        'spread').
  ///
  /// \param A Argument of the clause ('master', 'close' or 'spread').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  ///
  OMPProcBindClause(OpenMPProcBindClauseKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OMPClause(OMPC_proc_bind, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// \brief Build an empty clause.
  ///
  OMPProcBindClause()
      : OMPClause(OMPC_proc_bind, SourceLocation(), SourceLocation()),
        LParenLoc(SourceLocation()), Kind(OMPC_PROC_BIND_unknown),
        KindKwLoc(SourceLocation()) {}

  /// \brief Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// \brief Returns kind of the clause.
  OpenMPProcBindClauseKind getProcBindKind() const { return Kind; }

  /// \brief Returns location of clause kind.
  SourceLocation getProcBindKindKwLoc() const { return KindKwLoc; }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_proc_bind;
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
class OMPPrivateClause : public OMPVarListClause<OMPPrivateClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPPrivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPPrivateClause>(OMPC_private, StartLoc, LParenLoc,
                                           EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPPrivateClause(unsigned N)
      : OMPVarListClause<OMPPrivateClause>(OMPC_private, SourceLocation(),
                                           SourceLocation(), SourceLocation(),
                                           N) {}

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
class OMPFirstprivateClause : public OMPVarListClause<OMPFirstprivateClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPFirstprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                        SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPFirstprivateClause>(OMPC_firstprivate, StartLoc,
                                                LParenLoc, EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPFirstprivateClause(unsigned N)
      : OMPVarListClause<OMPFirstprivateClause>(
            OMPC_firstprivate, SourceLocation(), SourceLocation(),
            SourceLocation(), N) {}

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
class OMPSharedClause : public OMPVarListClause<OMPSharedClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPSharedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPSharedClause>(OMPC_shared, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPSharedClause(unsigned N)
      : OMPVarListClause<OMPSharedClause>(OMPC_shared, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

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

/// \brief This represents clause 'linear' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp simd linear(a,b : 2)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'linear'
/// with variables 'a', 'b' and linear step '2'.
///
class OMPLinearClause : public OMPVarListClause<OMPLinearClause> {
  friend class OMPClauseReader;
  /// \brief Location of ':'.
  SourceLocation ColonLoc;

  /// \brief Sets the linear step for clause.
  void setStep(Expr *Step) { *varlist_end() = Step; }

  /// \brief Build 'linear' clause with given number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of variables.
  ///
  OMPLinearClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation ColonLoc, SourceLocation EndLoc,
                  unsigned NumVars)
      : OMPVarListClause<OMPLinearClause>(OMPC_linear, StartLoc, LParenLoc,
                                          EndLoc, NumVars),
        ColonLoc(ColonLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of variables.
  ///
  explicit OMPLinearClause(unsigned NumVars)
      : OMPVarListClause<OMPLinearClause>(OMPC_linear, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          NumVars),
        ColonLoc(SourceLocation()) {}

public:
  /// \brief Creates clause with a list of variables \a VL and a linear step
  /// \a Step.
  ///
  /// \param C AST Context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param Step Linear step.
  static OMPLinearClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation ColonLoc, SourceLocation EndLoc,
                                 ArrayRef<Expr *> VL, Expr *Step);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of variables.
  ///
  static OMPLinearClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

  /// \brief Sets the location of ':'.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }
  /// \brief Returns the location of '('.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// \brief Returns linear step.
  Expr *getStep() { return *varlist_end(); }
  /// \brief Returns linear step.
  const Expr *getStep() const { return *varlist_end(); }

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end() + 1));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_linear;
  }
};

/// \brief This represents clause 'aligned' in the '#pragma omp ...'
/// directives.
///
/// \code
/// #pragma omp simd aligned(a,b : 8)
/// \endcode
/// In this example directive '#pragma omp simd' has clause 'aligned'
/// with variables 'a', 'b' and alignment '8'.
///
class OMPAlignedClause : public OMPVarListClause<OMPAlignedClause> {
  friend class OMPClauseReader;
  /// \brief Location of ':'.
  SourceLocation ColonLoc;

  /// \brief Sets the alignment for clause.
  void setAlignment(Expr *A) { *varlist_end() = A; }

  /// \brief Build 'aligned' clause with given number of variables \a NumVars.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param NumVars Number of variables.
  ///
  OMPAlignedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation ColonLoc, SourceLocation EndLoc,
                   unsigned NumVars)
      : OMPVarListClause<OMPAlignedClause>(OMPC_aligned, StartLoc, LParenLoc,
                                           EndLoc, NumVars),
        ColonLoc(ColonLoc) {}

  /// \brief Build an empty clause.
  ///
  /// \param NumVars Number of variables.
  ///
  explicit OMPAlignedClause(unsigned NumVars)
      : OMPVarListClause<OMPAlignedClause>(OMPC_aligned, SourceLocation(),
                                           SourceLocation(), SourceLocation(),
                                           NumVars),
        ColonLoc(SourceLocation()) {}

public:
  /// \brief Creates clause with a list of variables \a VL and alignment \a A.
  ///
  /// \param C AST Context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param A Alignment.
  static OMPAlignedClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation ColonLoc,
                                  SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                  Expr *A);

  /// \brief Creates an empty clause with the place for \a NumVars variables.
  ///
  /// \param C AST context.
  /// \param NumVars Number of variables.
  ///
  static OMPAlignedClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

  /// \brief Sets the location of ':'.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }
  /// \brief Returns the location of ':'.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// \brief Returns alignment.
  Expr *getAlignment() { return *varlist_end(); }
  /// \brief Returns alignment.
  const Expr *getAlignment() const { return *varlist_end(); }

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end() + 1));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_aligned;
  }
};

/// \brief This represents clause 'copyin' in the '#pragma omp ...' directives.
///
/// \code
/// #pragma omp parallel copyin(a,b)
/// \endcode
/// In this example directive '#pragma omp parallel' has clause 'copyin'
/// with the variables 'a' and 'b'.
///
class OMPCopyinClause : public OMPVarListClause<OMPCopyinClause> {
  /// \brief Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  ///
  OMPCopyinClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OMPVarListClause<OMPCopyinClause>(OMPC_copyin, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// \brief Build an empty clause.
  ///
  /// \param N Number of variables.
  ///
  explicit OMPCopyinClause(unsigned N)
      : OMPVarListClause<OMPCopyinClause>(OMPC_copyin, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

public:
  /// \brief Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  ///
  static OMPCopyinClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);
  /// \brief Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  ///
  static OMPCopyinClause *CreateEmpty(const ASTContext &C, unsigned N);

  StmtRange children() {
    return StmtRange(reinterpret_cast<Stmt **>(varlist_begin()),
                     reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OMPClause *T) {
    return T->getClauseKind() == OMPC_copyin;
  }
};

} // end namespace clang

#endif
