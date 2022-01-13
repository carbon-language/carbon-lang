//===- ExprConcepts.h - C++2a Concepts expressions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines Expressions and AST nodes for C++2a concepts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPRCONCEPTS_H
#define LLVM_CLANG_AST_EXPRCONCEPTS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/TrailingObjects.h"
#include <utility>
#include <string>

namespace clang {
class ASTStmtReader;
class ASTStmtWriter;

/// \brief Represents the specialization of a concept - evaluates to a prvalue
/// of type bool.
///
/// According to C++2a [expr.prim.id]p3 an id-expression that denotes the
/// specialization of a concept results in a prvalue of type bool.
class ConceptSpecializationExpr final : public Expr, public ConceptReference,
      private llvm::TrailingObjects<ConceptSpecializationExpr,
                                    TemplateArgument> {
  friend class ASTStmtReader;
  friend TrailingObjects;
public:
  using SubstitutionDiagnostic = std::pair<SourceLocation, std::string>;

protected:
  /// \brief The number of template arguments in the tail-allocated list of
  /// converted template arguments.
  unsigned NumTemplateArgs;

  /// \brief Information about the satisfaction of the named concept with the
  /// given arguments. If this expression is value dependent, this is to be
  /// ignored.
  ASTConstraintSatisfaction *Satisfaction;

  ConceptSpecializationExpr(const ASTContext &C, NestedNameSpecifierLoc NNS,
                            SourceLocation TemplateKWLoc,
                            DeclarationNameInfo ConceptNameInfo,
                            NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
                            const ASTTemplateArgumentListInfo *ArgsAsWritten,
                            ArrayRef<TemplateArgument> ConvertedArgs,
                            const ConstraintSatisfaction *Satisfaction);

  ConceptSpecializationExpr(const ASTContext &C, ConceptDecl *NamedConcept,
                            ArrayRef<TemplateArgument> ConvertedArgs,
                            const ConstraintSatisfaction *Satisfaction,
                            bool Dependent,
                            bool ContainsUnexpandedParameterPack);

  ConceptSpecializationExpr(EmptyShell Empty, unsigned NumTemplateArgs);

public:

  static ConceptSpecializationExpr *
  Create(const ASTContext &C, NestedNameSpecifierLoc NNS,
         SourceLocation TemplateKWLoc, DeclarationNameInfo ConceptNameInfo,
         NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
         const ASTTemplateArgumentListInfo *ArgsAsWritten,
         ArrayRef<TemplateArgument> ConvertedArgs,
         const ConstraintSatisfaction *Satisfaction);

  static ConceptSpecializationExpr *
  Create(const ASTContext &C, ConceptDecl *NamedConcept,
         ArrayRef<TemplateArgument> ConvertedArgs,
         const ConstraintSatisfaction *Satisfaction,
         bool Dependent,
         bool ContainsUnexpandedParameterPack);

  static ConceptSpecializationExpr *
  Create(ASTContext &C, EmptyShell Empty, unsigned NumTemplateArgs);

  ArrayRef<TemplateArgument> getTemplateArguments() const {
    return ArrayRef<TemplateArgument>(getTrailingObjects<TemplateArgument>(),
                                      NumTemplateArgs);
  }

  /// \brief Set new template arguments for this concept specialization.
  void setTemplateArguments(ArrayRef<TemplateArgument> Converted);

  /// \brief Whether or not the concept with the given arguments was satisfied
  /// when the expression was created.
  /// The expression must not be dependent.
  bool isSatisfied() const {
    assert(!isValueDependent()
           && "isSatisfied called on a dependent ConceptSpecializationExpr");
    return Satisfaction->IsSatisfied;
  }

  /// \brief Get elaborated satisfaction info about the template arguments'
  /// satisfaction of the named concept.
  /// The expression must not be dependent.
  const ASTConstraintSatisfaction &getSatisfaction() const {
    assert(!isValueDependent()
           && "getSatisfaction called on dependent ConceptSpecializationExpr");
    return *Satisfaction;
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ConceptSpecializationExprClass;
  }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return ConceptName.getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {
    // If the ConceptSpecializationExpr is the ImmediatelyDeclaredConstraint
    // of a TypeConstraint written syntactically as a constrained-parameter,
    // there may not be a template argument list.
    return ArgsAsWritten->RAngleLoc.isValid() ? ArgsAsWritten->RAngleLoc
                                              : ConceptName.getEndLoc();
  }

  // Iterators
  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

namespace concepts {

/// \brief A static requirement that can be used in a requires-expression to
/// check properties of types and expression.
class Requirement {
public:
  // Note - simple and compound requirements are both represented by the same
  // class (ExprRequirement).
  enum RequirementKind { RK_Type, RK_Simple, RK_Compound, RK_Nested };
private:
  const RequirementKind Kind;
  // FIXME: use RequirementDependence to model dependence?
  bool Dependent : 1;
  bool ContainsUnexpandedParameterPack : 1;
  bool Satisfied : 1;
public:
  struct SubstitutionDiagnostic {
    StringRef SubstitutedEntity;
    // FIXME: Store diagnostics semantically and not as prerendered strings.
    //  Fixing this probably requires serialization of PartialDiagnostic
    //  objects.
    SourceLocation DiagLoc;
    StringRef DiagMessage;
  };

  Requirement(RequirementKind Kind, bool IsDependent,
              bool ContainsUnexpandedParameterPack, bool IsSatisfied = true) :
      Kind(Kind), Dependent(IsDependent),
      ContainsUnexpandedParameterPack(ContainsUnexpandedParameterPack),
      Satisfied(IsSatisfied) {}

  RequirementKind getKind() const { return Kind; }

  bool isSatisfied() const {
    assert(!Dependent &&
           "isSatisfied can only be called on non-dependent requirements.");
    return Satisfied;
  }

  void setSatisfied(bool IsSatisfied) {
    assert(!Dependent &&
           "setSatisfied can only be called on non-dependent requirements.");
    Satisfied = IsSatisfied;
  }

  void setDependent(bool IsDependent) { Dependent = IsDependent; }
  bool isDependent() const { return Dependent; }

  void setContainsUnexpandedParameterPack(bool Contains) {
    ContainsUnexpandedParameterPack = Contains;
  }
  bool containsUnexpandedParameterPack() const {
    return ContainsUnexpandedParameterPack;
  }
};

/// \brief A requires-expression requirement which queries the existence of a
/// type name or type template specialization ('type' requirements).
class TypeRequirement : public Requirement {
public:
  enum SatisfactionStatus {
      SS_Dependent,
      SS_SubstitutionFailure,
      SS_Satisfied
  };
private:
  llvm::PointerUnion<SubstitutionDiagnostic *, TypeSourceInfo *> Value;
  SatisfactionStatus Status;
public:
  friend ASTStmtReader;
  friend ASTStmtWriter;

  /// \brief Construct a type requirement from a type. If the given type is not
  /// dependent, this indicates that the type exists and the requirement will be
  /// satisfied. Otherwise, the SubstitutionDiagnostic constructor is to be
  /// used.
  TypeRequirement(TypeSourceInfo *T);

  /// \brief Construct a type requirement when the nested name specifier is
  /// invalid due to a bad substitution. The requirement is unsatisfied.
  TypeRequirement(SubstitutionDiagnostic *Diagnostic) :
      Requirement(RK_Type, false, false, false), Value(Diagnostic),
      Status(SS_SubstitutionFailure) {}

  SatisfactionStatus getSatisfactionStatus() const { return Status; }
  void setSatisfactionStatus(SatisfactionStatus Status) {
    this->Status = Status;
  }

  bool isSubstitutionFailure() const {
    return Status == SS_SubstitutionFailure;
  }

  SubstitutionDiagnostic *getSubstitutionDiagnostic() const {
    assert(Status == SS_SubstitutionFailure &&
           "Attempted to get substitution diagnostic when there has been no "
           "substitution failure.");
    return Value.get<SubstitutionDiagnostic *>();
  }

  TypeSourceInfo *getType() const {
    assert(!isSubstitutionFailure() &&
           "Attempted to get type when there has been a substitution failure.");
    return Value.get<TypeSourceInfo *>();
  }

  static bool classof(const Requirement *R) {
    return R->getKind() == RK_Type;
  }
};

/// \brief A requires-expression requirement which queries the validity and
/// properties of an expression ('simple' and 'compound' requirements).
class ExprRequirement : public Requirement {
public:
  enum SatisfactionStatus {
      SS_Dependent,
      SS_ExprSubstitutionFailure,
      SS_NoexceptNotMet,
      SS_TypeRequirementSubstitutionFailure,
      SS_ConstraintsNotSatisfied,
      SS_Satisfied
  };
  class ReturnTypeRequirement {
      llvm::PointerIntPair<
          llvm::PointerUnion<TemplateParameterList *, SubstitutionDiagnostic *>,
          1, bool>
          TypeConstraintInfo;
  public:
      friend ASTStmtReader;
      friend ASTStmtWriter;

      /// \brief No return type requirement was specified.
      ReturnTypeRequirement() : TypeConstraintInfo(nullptr, false) {}

      /// \brief A return type requirement was specified but it was a
      /// substitution failure.
      ReturnTypeRequirement(SubstitutionDiagnostic *SubstDiag) :
          TypeConstraintInfo(SubstDiag, false) {}

      /// \brief A 'type constraint' style return type requirement.
      /// \param TPL an invented template parameter list containing a single
      /// type parameter with a type-constraint.
      // TODO: Can we maybe not save the whole template parameter list and just
      //  the type constraint? Saving the whole TPL makes it easier to handle in
      //  serialization but is less elegant.
      ReturnTypeRequirement(TemplateParameterList *TPL);

      bool isDependent() const {
        return TypeConstraintInfo.getInt();
      }

      bool containsUnexpandedParameterPack() const {
        if (!isTypeConstraint())
          return false;
        return getTypeConstraintTemplateParameterList()
                ->containsUnexpandedParameterPack();
      }

      bool isEmpty() const {
        return TypeConstraintInfo.getPointer().isNull();
      }

      bool isSubstitutionFailure() const {
        return !isEmpty() &&
            TypeConstraintInfo.getPointer().is<SubstitutionDiagnostic *>();
      }

      bool isTypeConstraint() const {
        return !isEmpty() &&
            TypeConstraintInfo.getPointer().is<TemplateParameterList *>();
      }

      SubstitutionDiagnostic *getSubstitutionDiagnostic() const {
        assert(isSubstitutionFailure());
        return TypeConstraintInfo.getPointer().get<SubstitutionDiagnostic *>();
      }

      const TypeConstraint *getTypeConstraint() const;

      TemplateParameterList *getTypeConstraintTemplateParameterList() const {
        assert(isTypeConstraint());
        return TypeConstraintInfo.getPointer().get<TemplateParameterList *>();
      }
  };
private:
  llvm::PointerUnion<Expr *, SubstitutionDiagnostic *> Value;
  SourceLocation NoexceptLoc; // May be empty if noexcept wasn't specified.
  ReturnTypeRequirement TypeReq;
  ConceptSpecializationExpr *SubstitutedConstraintExpr;
  SatisfactionStatus Status;
public:
  friend ASTStmtReader;
  friend ASTStmtWriter;

  /// \brief Construct a compound requirement.
  /// \param E the expression which is checked by this requirement.
  /// \param IsSimple whether this was a simple requirement in source.
  /// \param NoexceptLoc the location of the noexcept keyword, if it was
  /// specified, otherwise an empty location.
  /// \param Req the requirement for the type of the checked expression.
  /// \param Status the satisfaction status of this requirement.
  ExprRequirement(
      Expr *E, bool IsSimple, SourceLocation NoexceptLoc,
      ReturnTypeRequirement Req, SatisfactionStatus Status,
      ConceptSpecializationExpr *SubstitutedConstraintExpr = nullptr);

  /// \brief Construct a compound requirement whose expression was a
  /// substitution failure. The requirement is not satisfied.
  /// \param E the diagnostic emitted while instantiating the original
  /// expression.
  /// \param IsSimple whether this was a simple requirement in source.
  /// \param NoexceptLoc the location of the noexcept keyword, if it was
  /// specified, otherwise an empty location.
  /// \param Req the requirement for the type of the checked expression (omit
  /// if no requirement was specified).
  ExprRequirement(SubstitutionDiagnostic *E, bool IsSimple,
                  SourceLocation NoexceptLoc, ReturnTypeRequirement Req = {});

  bool isSimple() const { return getKind() == RK_Simple; }
  bool isCompound() const { return getKind() == RK_Compound; }

  bool hasNoexceptRequirement() const { return NoexceptLoc.isValid(); }
  SourceLocation getNoexceptLoc() const { return NoexceptLoc; }

  SatisfactionStatus getSatisfactionStatus() const { return Status; }

  bool isExprSubstitutionFailure() const {
    return Status == SS_ExprSubstitutionFailure;
  }

  const ReturnTypeRequirement &getReturnTypeRequirement() const {
    return TypeReq;
  }

  ConceptSpecializationExpr *
  getReturnTypeRequirementSubstitutedConstraintExpr() const {
    assert(Status >= SS_TypeRequirementSubstitutionFailure);
    return SubstitutedConstraintExpr;
  }

  SubstitutionDiagnostic *getExprSubstitutionDiagnostic() const {
    assert(isExprSubstitutionFailure() &&
           "Attempted to get expression substitution diagnostic when there has "
           "been no expression substitution failure");
    return Value.get<SubstitutionDiagnostic *>();
  }

  Expr *getExpr() const {
    assert(!isExprSubstitutionFailure() &&
           "ExprRequirement has no expression because there has been a "
           "substitution failure.");
    return Value.get<Expr *>();
  }

  static bool classof(const Requirement *R) {
    return R->getKind() == RK_Compound || R->getKind() == RK_Simple;
  }
};

/// \brief A requires-expression requirement which is satisfied when a general
/// constraint expression is satisfied ('nested' requirements).
class NestedRequirement : public Requirement {
  llvm::PointerUnion<Expr *, SubstitutionDiagnostic *> Value;
  const ASTConstraintSatisfaction *Satisfaction = nullptr;

public:
  friend ASTStmtReader;
  friend ASTStmtWriter;

  NestedRequirement(SubstitutionDiagnostic *SubstDiag) :
      Requirement(RK_Nested, /*IsDependent=*/false,
                  /*ContainsUnexpandedParameterPack*/false,
                  /*IsSatisfied=*/false), Value(SubstDiag) {}

  NestedRequirement(Expr *Constraint) :
      Requirement(RK_Nested, /*IsDependent=*/true,
                  Constraint->containsUnexpandedParameterPack()),
      Value(Constraint) {
    assert(Constraint->isInstantiationDependent() &&
           "Nested requirement with non-dependent constraint must be "
           "constructed with a ConstraintSatisfaction object");
  }

  NestedRequirement(ASTContext &C, Expr *Constraint,
                    const ConstraintSatisfaction &Satisfaction) :
      Requirement(RK_Nested, Constraint->isInstantiationDependent(),
                  Constraint->containsUnexpandedParameterPack(),
                  Satisfaction.IsSatisfied),
      Value(Constraint),
      Satisfaction(ASTConstraintSatisfaction::Create(C, Satisfaction)) {}

  bool isSubstitutionFailure() const {
    return Value.is<SubstitutionDiagnostic *>();
  }

  SubstitutionDiagnostic *getSubstitutionDiagnostic() const {
    assert(isSubstitutionFailure() &&
           "getSubstitutionDiagnostic() may not be called when there was no "
           "substitution failure.");
    return Value.get<SubstitutionDiagnostic *>();
  }

  Expr *getConstraintExpr() const {
    assert(!isSubstitutionFailure() && "getConstraintExpr() may not be called "
                                       "on nested requirements with "
                                       "substitution failures.");
    return Value.get<Expr *>();
  }

  const ASTConstraintSatisfaction &getConstraintSatisfaction() const {
    assert(!isSubstitutionFailure() && "getConstraintSatisfaction() may not be "
                                       "called on nested requirements with "
                                       "substitution failures.");
    return *Satisfaction;
  }

  static bool classof(const Requirement *R) {
    return R->getKind() == RK_Nested;
  }
};

} // namespace concepts

/// C++2a [expr.prim.req]:
///     A requires-expression provides a concise way to express requirements on
///     template arguments. A requirement is one that can be checked by name
///     lookup (6.4) or by checking properties of types and expressions.
///     [...]
///     A requires-expression is a prvalue of type bool [...]
class RequiresExpr final : public Expr,
    llvm::TrailingObjects<RequiresExpr, ParmVarDecl *,
                          concepts::Requirement *> {
  friend TrailingObjects;
  friend class ASTStmtReader;

  unsigned NumLocalParameters;
  unsigned NumRequirements;
  RequiresExprBodyDecl *Body;
  SourceLocation RBraceLoc;

  unsigned numTrailingObjects(OverloadToken<ParmVarDecl *>) const {
    return NumLocalParameters;
  }

  unsigned numTrailingObjects(OverloadToken<concepts::Requirement *>) const {
    return NumRequirements;
  }

  RequiresExpr(ASTContext &C, SourceLocation RequiresKWLoc,
               RequiresExprBodyDecl *Body,
               ArrayRef<ParmVarDecl *> LocalParameters,
               ArrayRef<concepts::Requirement *> Requirements,
               SourceLocation RBraceLoc);
  RequiresExpr(ASTContext &C, EmptyShell Empty, unsigned NumLocalParameters,
               unsigned NumRequirements);

public:
  static RequiresExpr *
  Create(ASTContext &C, SourceLocation RequiresKWLoc,
         RequiresExprBodyDecl *Body, ArrayRef<ParmVarDecl *> LocalParameters,
         ArrayRef<concepts::Requirement *> Requirements,
         SourceLocation RBraceLoc);
  static RequiresExpr *
  Create(ASTContext &C, EmptyShell Empty, unsigned NumLocalParameters,
         unsigned NumRequirements);

  ArrayRef<ParmVarDecl *> getLocalParameters() const {
    return {getTrailingObjects<ParmVarDecl *>(), NumLocalParameters};
  }

  RequiresExprBodyDecl *getBody() const { return Body; }

  ArrayRef<concepts::Requirement *> getRequirements() const {
    return {getTrailingObjects<concepts::Requirement *>(), NumRequirements};
  }

  /// \brief Whether or not the requires clause is satisfied.
  /// The expression must not be dependent.
  bool isSatisfied() const {
    assert(!isValueDependent()
           && "isSatisfied called on a dependent RequiresExpr");
    return RequiresExprBits.IsSatisfied;
  }

  SourceLocation getRequiresKWLoc() const {
    return RequiresExprBits.RequiresKWLoc;
  }

  SourceLocation getRBraceLoc() const { return RBraceLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == RequiresExprClass;
  }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return RequiresExprBits.RequiresKWLoc;
  }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return RBraceLoc;
  }

  // Iterators
  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

} // namespace clang

#endif // LLVM_CLANG_AST_EXPRCONCEPTS_H
