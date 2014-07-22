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
  /// \brief Numbers of clauses.
  const unsigned NumClauses;
  /// \brief Number of child expressions/stmts.
  const unsigned NumChildren;
  /// \brief Offset from this to the start of clauses.
  /// There are NumClauses pointers to clauses, they are followed by
  /// NumChildren pointers to child stmts/exprs (if the directive type
  /// requires an associated stmt, then it has to be the first of them).
  const unsigned ClausesOffset;

  /// \brief Get the clauses storage.
  MutableArrayRef<OMPClause *> getClauses() {
    OMPClause **ClauseStorage = reinterpret_cast<OMPClause **>(
        reinterpret_cast<char *>(this) + ClausesOffset);
    return MutableArrayRef<OMPClause *>(ClauseStorage, NumClauses);
  }

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
                         unsigned NumClauses, unsigned NumChildren)
      : Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
        EndLoc(std::move(EndLoc)), NumClauses(NumClauses),
        NumChildren(NumChildren),
        ClausesOffset(llvm::RoundUpToAlignment(sizeof(T),
                                               llvm::alignOf<OMPClause *>())) {}

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
    assert(hasAssociatedStmt() && "no associated statement.");
    *child_begin() = S;
  }

public:
  /// \brief Iterates over a filtered subrange of clauses applied to a
  /// directive.
  ///
  /// This iterator visits only those declarations that meet some run-time
  /// criteria.
  template <class FilterPredicate> class filtered_clause_iterator {
    ArrayRef<OMPClause *>::const_iterator Current;
    ArrayRef<OMPClause *>::const_iterator End;
    FilterPredicate Pred;
    void SkipToNextClause() {
      while (Current != End && !Pred(*Current))
        ++Current;
    }

  public:
    typedef const OMPClause *value_type;
    filtered_clause_iterator() : Current(), End() {}
    filtered_clause_iterator(ArrayRef<OMPClause *> Arr, FilterPredicate Pred)
        : Current(Arr.begin()), End(Arr.end()), Pred(Pred) {
      SkipToNextClause();
    }
    value_type operator*() const { return *Current; }
    value_type operator->() const { return *Current; }
    filtered_clause_iterator &operator++() {
      ++Current;
      SkipToNextClause();
      return *this;
    }

    filtered_clause_iterator operator++(int) {
      filtered_clause_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    bool operator!() { return Current == End; }
    operator bool() { return Current != End; }
  };

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
  unsigned getNumClauses() const { return NumClauses; }

  /// \brief Returns specified clause.
  ///
  /// \param i Number of clause.
  ///
  OMPClause *getClause(unsigned i) const { return clauses()[i]; }

  /// \brief Returns true if directive has associated statement.
  bool hasAssociatedStmt() const { return NumChildren > 0; }

  /// \brief Returns statement associated with the directive.
  Stmt *getAssociatedStmt() const {
    assert(hasAssociatedStmt() && "no associated statement.");
    return const_cast<Stmt *>(*child_begin());
  }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOMPExecutableDirectiveConstant &&
           S->getStmtClass() <= lastOMPExecutableDirectiveConstant;
  }

  child_range children() {
    if (!hasAssociatedStmt())
      return child_range();
    Stmt **ChildStorage = reinterpret_cast<Stmt **>(getClauses().end());
    return child_range(ChildStorage, ChildStorage + NumChildren);
  }

  ArrayRef<OMPClause *> clauses() { return getClauses(); }

  ArrayRef<OMPClause *> clauses() const {
    return const_cast<OMPExecutableDirective *>(this)->getClauses();
  }
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
                       unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement associated with the directive.
  ///
  static OMPParallelDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a N clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelDirectiveClass;
  }
};

/// \brief This represents '#pragma omp simd' directive.
///
/// \code
/// #pragma omp simd private(a,b) linear(i,j:s) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp simd' has clauses 'private'
/// with the variables 'a' and 'b', 'linear' with variables 'i', 'j' and
/// linear step 's', 'reduction' with operator '+' and variables 'c' and 'd'.
///
class OMPSimdDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Number of collapsed loops as specified by 'collapse' clause.
  unsigned CollapsedNum;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSimdDirectiveClass, OMPD_simd, StartLoc,
                               EndLoc, NumClauses, 1),
        CollapsedNum(CollapsedNum) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSimdDirectiveClass, OMPD_simd,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        CollapsedNum(CollapsedNum) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPSimdDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc, unsigned CollapsedNum,
                                  ArrayRef<OMPClause *> Clauses,
                                  Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSimdDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                       unsigned CollapsedNum, EmptyShell);

  unsigned getCollapsedNumber() const { return CollapsedNum; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSimdDirectiveClass;
  }
};

/// \brief This represents '#pragma omp for' directive.
///
/// \code
/// #pragma omp for private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp for' has clauses 'private' with the
/// variables 'a' and 'b' and 'reduction' with operator '+' and variables 'c'
/// and 'd'.
///
class OMPForDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Number of collapsed loops as specified by 'collapse' clause.
  unsigned CollapsedNum;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                  unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPForDirectiveClass, OMPD_for, StartLoc,
                               EndLoc, NumClauses, 1),
        CollapsedNum(CollapsedNum) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPForDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPForDirectiveClass, OMPD_for,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1),
        CollapsedNum(CollapsedNum) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPForDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation EndLoc, unsigned CollapsedNum,
                                 ArrayRef<OMPClause *> Clauses,
                                 Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPForDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                      unsigned CollapsedNum, EmptyShell);

  unsigned getCollapsedNumber() const { return CollapsedNum; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPForDirectiveClass;
  }
};

/// \brief This represents '#pragma omp sections' directive.
///
/// \code
/// #pragma omp sections private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp sections' has clauses 'private' with
/// the variables 'a' and 'b' and 'reduction' with operator '+' and variables
/// 'c' and 'd'.
///
class OMPSectionsDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSectionsDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPSectionsDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSectionsDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSectionsDirectiveClass;
  }
};

/// \brief This represents '#pragma omp section' directive.
///
/// \code
/// #pragma omp section
/// \endcode
///
class OMPSectionDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPSectionDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
                               StartLoc, EndLoc, 0, 1) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPSectionDirective()
      : OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
                               SourceLocation(), SourceLocation(), 0, 1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPSectionDirective *Create(const ASTContext &C,
                                     SourceLocation StartLoc,
                                     SourceLocation EndLoc,
                                     Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPSectionDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSectionDirectiveClass;
  }
};

/// \brief This represents '#pragma omp single' directive.
///
/// \code
/// #pragma omp single private(a,b) copyprivate(c,d)
/// \endcode
/// In this example directive '#pragma omp single' has clauses 'private' with
/// the variables 'a' and 'b' and 'copyprivate' with variables 'c' and 'd'.
///
class OMPSingleDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPSingleDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPSingleDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPSingleDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPSingleDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPSingleDirectiveClass;
  }
};

/// \brief This represents '#pragma omp master' directive.
///
/// \code
/// #pragma omp master
/// \endcode
///
class OMPMasterDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPMasterDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
                               StartLoc, EndLoc, 0, 1) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPMasterDirective()
      : OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
                               SourceLocation(), SourceLocation(), 0, 1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPMasterDirective *Create(const ASTContext &C,
                                    SourceLocation StartLoc,
                                    SourceLocation EndLoc,
                                    Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPMasterDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPMasterDirectiveClass;
  }
};

/// \brief This represents '#pragma omp critical' directive.
///
/// \code
/// #pragma omp critical
/// \endcode
///
class OMPCriticalDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Name of the directive.
  DeclarationNameInfo DirName;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPCriticalDirective(const DeclarationNameInfo &Name, SourceLocation StartLoc,
                       SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
                               StartLoc, EndLoc, 0, 1),
        DirName(Name) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPCriticalDirective()
      : OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
                               SourceLocation(), SourceLocation(), 0, 1),
        DirName() {}

  /// \brief Set name of the directive.
  ///
  /// \param Name Name of the directive.
  ///
  void setDirectiveName(const DeclarationNameInfo &Name) { DirName = Name; }

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPCriticalDirective *
  Create(const ASTContext &C, const DeclarationNameInfo &Name,
         SourceLocation StartLoc, SourceLocation EndLoc, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPCriticalDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  /// \brief Return name of the directive.
  ///
  DeclarationNameInfo getDirectiveName() const { return DirName; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPCriticalDirectiveClass;
  }
};

/// \brief This represents '#pragma omp parallel for' directive.
///
/// \code
/// #pragma omp parallel for private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp parallel for' has clauses 'private'
/// with the variables 'a' and 'b' and 'reduction' with operator '+' and
/// variables 'c' and 'd'.
///
class OMPParallelForDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Number of collapsed loops as specified by 'collapse' clause.
  unsigned CollapsedNum;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  OMPParallelForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                          unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelForDirectiveClass,
                               OMPD_parallel_for, StartLoc, EndLoc, NumClauses,
                               1),
        CollapsedNum(CollapsedNum) {}

  /// \brief Build an empty directive.
  ///
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelForDirective(unsigned CollapsedNum, unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelForDirectiveClass,
                               OMPD_parallel_for, SourceLocation(),
                               SourceLocation(), NumClauses, 1),
        CollapsedNum(CollapsedNum) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param CollapsedNum Number of collapsed loops.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPParallelForDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
         Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place
  /// for \a NumClauses clauses.
  ///
  /// \param C AST context.
  /// \param CollapsedNum Number of collapsed nested loops.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelForDirective *CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses,
                                              unsigned CollapsedNum,
                                              EmptyShell);

  unsigned getCollapsedNumber() const { return CollapsedNum; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelForDirectiveClass;
  }
};

/// \brief This represents '#pragma omp parallel sections' directive.
///
/// \code
/// #pragma omp parallel sections private(a,b) reduction(+:c,d)
/// \endcode
/// In this example directive '#pragma omp parallel sections' has clauses
/// 'private' with the variables 'a' and 'b' and 'reduction' with operator '+'
/// and variables 'c' and 'd'.
///
class OMPParallelSectionsDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPParallelSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                               unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
                               OMPD_parallel_sections, StartLoc, EndLoc,
                               NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPParallelSectionsDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
                               OMPD_parallel_sections, SourceLocation(),
                               SourceLocation(), NumClauses, 1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPParallelSectionsDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPParallelSectionsDirective *
  CreateEmpty(const ASTContext &C, unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPParallelSectionsDirectiveClass;
  }
};

/// \brief This represents '#pragma omp task' directive.
///
/// \code
/// #pragma omp task private(a,b) final(d)
/// \endcode
/// In this example directive '#pragma omp task' has clauses 'private' with the
/// variables 'a' and 'b' and 'final' with condition 'd'.
///
class OMPTaskDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPTaskDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task, StartLoc,
                               EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPTaskDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPTaskDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OMPClause *> Clauses,
                                  Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPTaskDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                       EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskyield' directive.
///
/// \code
/// #pragma omp taskyield
/// \endcode
///
class OMPTaskyieldDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPTaskyieldDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPTaskyieldDirective()
      : OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPTaskyieldDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPTaskyieldDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskyieldDirectiveClass;
  }
};

/// \brief This represents '#pragma omp barrier' directive.
///
/// \code
/// #pragma omp barrier
/// \endcode
///
class OMPBarrierDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPBarrierDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPBarrierDirective()
      : OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPBarrierDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPBarrierDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPBarrierDirectiveClass;
  }
};

/// \brief This represents '#pragma omp taskwait' directive.
///
/// \code
/// #pragma omp taskwait
/// \endcode
///
class OMPTaskwaitDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
                               StartLoc, EndLoc, 0, 0) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPTaskwaitDirective()
      : OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
                               SourceLocation(), SourceLocation(), 0, 0) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OMPTaskwaitDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPTaskwaitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPTaskwaitDirectiveClass;
  }
};

/// \brief This represents '#pragma omp flush' directive.
///
/// \code
/// #pragma omp flush(a,b)
/// \endcode
/// In this example directive '#pragma omp flush' has 2 arguments- variables 'a'
/// and 'b'.
/// 'omp flush' directive does not have clauses but have an optional list of
/// variables to flush. This list of variables is stored within some fake clause
/// FlushClause.
class OMPFlushDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPFlushDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                    unsigned NumClauses)
      : OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
                               StartLoc, EndLoc, NumClauses, 0) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPFlushDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
                               SourceLocation(), SourceLocation(), NumClauses,
                               0) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses (only single OMPFlushClause clause is
  /// allowed).
  ///
  static OMPFlushDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation EndLoc,
                                   ArrayRef<OMPClause *> Clauses);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPFlushDirective *CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPFlushDirectiveClass;
  }
};

/// \brief This represents '#pragma omp ordered' directive.
///
/// \code
/// #pragma omp ordered
/// \endcode
///
class OMPOrderedDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OMPOrderedDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
                               StartLoc, EndLoc, 0, 1) {}

  /// \brief Build an empty directive.
  ///
  explicit OMPOrderedDirective()
      : OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
                               SourceLocation(), SourceLocation(), 0, 1) {}

public:
  /// \brief Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPOrderedDirective *Create(const ASTContext &C,
                                     SourceLocation StartLoc,
                                     SourceLocation EndLoc,
                                     Stmt *AssociatedStmt);

  /// \brief Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OMPOrderedDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPOrderedDirectiveClass;
  }
};

/// \brief This represents '#pragma omp atomic' directive.
///
/// \code
/// #pragma omp atomic capture
/// \endcode
/// In this example directive '#pragma omp atomic' has clause 'capture'.
///
class OMPAtomicDirective : public OMPExecutableDirective {
  friend class ASTStmtReader;
  /// \brief Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumClauses Number of clauses.
  ///
  OMPAtomicDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                     unsigned NumClauses)
      : OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
                               StartLoc, EndLoc, NumClauses, 1) {}

  /// \brief Build an empty directive.
  ///
  /// \param NumClauses Number of clauses.
  ///
  explicit OMPAtomicDirective(unsigned NumClauses)
      : OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
                               SourceLocation(), SourceLocation(), NumClauses,
                               1) {}

public:
  /// \brief Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OMPAtomicDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

  /// \brief Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OMPAtomicDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPAtomicDirectiveClass;
  }
};

} // end namespace clang

#endif
