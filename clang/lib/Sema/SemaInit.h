//===--- SemaInit.h - Semantic Analysis for Initializers --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides supporting data types for initialization of objects.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_INIT_H
#define LLVM_CLANG_SEMA_INIT_H

#include "SemaOverload.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Parse/Action.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace clang {
  
class CXXBaseSpecifier;
class DeclaratorDecl;
class DeclaratorInfo;
class FieldDecl;
class FunctionDecl;
class ParmVarDecl;
class Sema;
class TypeLoc;
class VarDecl;
  
/// \brief Describes an entity that is being initialized.
class InitializedEntity {
public:
  /// \brief Specifies the kind of entity being initialized.
  enum EntityKind {
    /// \brief The entity being initialized is a variable.
    EK_Variable,
    /// \brief The entity being initialized is a function parameter.
    EK_Parameter,
    /// \brief The entity being initialized is the result of a function call.
    EK_Result,
    /// \brief The entity being initialized is an exception object that
    /// is being thrown.
    EK_Exception,
    /// \brief The entity being initialized is a temporary object.
    EK_Temporary,
    /// \brief The entity being initialized is a base member subobject.
    EK_Base,
    /// \brief The entity being initialized is a non-static data member 
    /// subobject.
    EK_Member
  };
  
private:
  /// \brief The kind of entity being initialized.
  EntityKind Kind;

  /// \brief The type of the object or reference being initialized along with 
  /// its location information.
  TypeLoc TL;
  
  union {
    /// \brief When Kind == EK_Variable, EK_Parameter, or EK_Member, 
    /// the VarDecl, ParmVarDecl, or FieldDecl, respectively.
    DeclaratorDecl *VariableOrMember;
    
    /// \brief When Kind == EK_Result or EK_Exception, the location of the 
    /// 'return' or 'throw' keyword, respectively. When Kind == EK_Temporary,
    /// the location where the temporary is being created.
    unsigned Location;
    
    /// \brief When Kind == EK_Base, the base specifier that provides the 
    /// base class.
    CXXBaseSpecifier *Base;
  };

  InitializedEntity() { }

  /// \brief Create the initialization entity for a variable.
  InitializedEntity(VarDecl *Var)
    : Kind(EK_Variable), 
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Var)) 
  {
    InitDeclLoc();
  }
  
  /// \brief Create the initialization entity for a parameter.
  InitializedEntity(ParmVarDecl *Parm)
    : Kind(EK_Parameter), 
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Parm)) 
  { 
    InitDeclLoc();
  }
  
  /// \brief Create the initialization entity for the result of a function,
  /// throwing an object, or performing an explicit cast.
  InitializedEntity(EntityKind Kind, SourceLocation Loc, TypeLoc TL)
    : Kind(Kind), TL(TL), Location(Loc.getRawEncoding()) { }
  
  /// \brief Create the initialization entity for a member subobject.
  InitializedEntity(FieldDecl *Member) 
    : Kind(EK_Member), 
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Member))
  { 
    InitDeclLoc();
  }
  
  /// \brief Initialize type-location information from a declaration.
  void InitDeclLoc();
  
public:
  /// \brief Create the initialization entity for a variable.
  static InitializedEntity InitializeVariable(VarDecl *Var) {
    return InitializedEntity(Var);
  }
  
  /// \brief Create the initialization entity for a parameter.
  static InitializedEntity InitializeParameter(ParmVarDecl *Parm) {
    return InitializedEntity(Parm);
  }

  /// \brief Create the initialization entity for the result of a function.
  static InitializedEntity InitializeResult(SourceLocation ReturnLoc,
                                            TypeLoc TL) {
    return InitializedEntity(EK_Result, ReturnLoc, TL);
  }

  /// \brief Create the initialization entity for an exception object.
  static InitializedEntity InitializeException(SourceLocation ThrowLoc,
                                               TypeLoc TL) {
    return InitializedEntity(EK_Exception, ThrowLoc, TL);
  }
  
  /// \brief Create the initialization entity for a temporary.
  static InitializedEntity InitializeTemporary(EntityKind Kind, TypeLoc TL) {
    return InitializedEntity(Kind, SourceLocation(), TL);
  }
  
  /// \brief Create the initialization entity for a base class subobject.
  static InitializedEntity InitializeBase(ASTContext &Context,
                                          CXXBaseSpecifier *Base);
  
  /// \brief Create the initialize entity for a member subobject.
  static InitializedEntity InitializeMember(FieldDecl *Member) {
    return InitializedEntity(Member);
  }
  
  /// \brief Determine the kind of initialization.
  EntityKind getKind() const { return Kind; }
  
  /// \brief Retrieve type being initialized.
  TypeLoc getType() const { return TL; }
  
  /// \brief Determine the location of the 'return' keyword when initializing
  /// the result of a function call.
  SourceLocation getReturnLoc() const {
    assert(getKind() == EK_Result && "No 'return' location!");
    return SourceLocation::getFromRawEncoding(Location);
  }
  
  /// \brief Determine the location of the 'throw' keyword when initializing
  /// an exception object.
  SourceLocation getThrowLoc() const {
    assert(getKind() == EK_Exception && "No 'throw' location!");
    return SourceLocation::getFromRawEncoding(Location);
  }
};
  
/// \brief Describes the kind of initialization being performed, along with 
/// location information for tokens related to the initialization (equal sign,
/// parentheses).
class InitializationKind {
public:
  /// \brief The kind of initialization being performed.
  enum InitKind {
    IK_Direct,  ///< Direct initialization
    IK_Copy,    ///< Copy initialization
    IK_Default, ///< Default initialization
    IK_Value    ///< Value initialization
  };
  
private:
  /// \brief The kind of initialization that we're storing.
  enum StoredInitKind {
    SIK_Direct = IK_Direct,   ///< Direct initialization
    SIK_Copy = IK_Copy,       ///< Copy initialization
    SIK_Default = IK_Default, ///< Default initialization
    SIK_Value = IK_Value,     ///< Value initialization
    SIK_DirectCast,  ///< Direct initialization due to a cast
    /// \brief Direct initialization due to a C-style or functional cast.
    SIK_DirectCStyleOrFunctionalCast
  };
  
  /// \brief The kind of initialization being performed.
  StoredInitKind Kind;
  
  /// \brief The source locations involved in the initialization.
  SourceLocation Locations[3];
  
  InitializationKind(StoredInitKind Kind, SourceLocation Loc1, 
                     SourceLocation Loc2, SourceLocation Loc3)
    : Kind(Kind) 
  {
    Locations[0] = Loc1;
    Locations[1] = Loc2;
    Locations[2] = Loc3;
  }
  
public:
  /// \brief Create a direct initialization.
  static InitializationKind CreateDirect(SourceLocation InitLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation RParenLoc) {
    return InitializationKind(SIK_Direct, InitLoc, LParenLoc, RParenLoc);
  }

  /// \brief Create a direct initialization due to a cast.
  static InitializationKind CreateCast(SourceRange TypeRange,
                                       bool IsCStyleCast) {
    return InitializationKind(IsCStyleCast? SIK_DirectCStyleOrFunctionalCast
                                          : SIK_DirectCast,
                              TypeRange.getBegin(), TypeRange.getBegin(), 
                              TypeRange.getEnd());
  }
  
  /// \brief Create a copy initialization.
  static InitializationKind CreateCopy(SourceLocation InitLoc,
                                       SourceLocation EqualLoc) {
    return InitializationKind(SIK_Copy, InitLoc, EqualLoc, EqualLoc);
  }
  
  /// \brief Create a default initialization.
  static InitializationKind CreateDefault(SourceLocation InitLoc) {
    return InitializationKind(SIK_Default, InitLoc, InitLoc, InitLoc);
  }
  
  /// \brief Create a value initialization.
  static InitializationKind CreateValue(SourceLocation InitLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation RParenLoc) {
    return InitializationKind(SIK_Value, InitLoc, LParenLoc, RParenLoc);
  }
  
  /// \brief Determine the initialization kind.
  InitKind getKind() const {
    if (Kind > SIK_Value)
      return IK_Direct;
    
    return (InitKind)Kind;
  }
  
  /// \brief Determine whether this initialization is an explicit cast.
  bool isExplicitCast() const {
    return Kind == SIK_DirectCast || Kind == SIK_DirectCStyleOrFunctionalCast;
  }
  
  /// \brief Determine whether this initialization is a C-style cast.
  bool isCStyleOrFunctionalCast() const { 
    return Kind == SIK_DirectCStyleOrFunctionalCast; 
  }
  
  /// \brief Retrieve the location at which initialization is occurring.
  SourceLocation getLocation() const { return Locations[0]; }
  
  /// \brief Retrieve the source range that covers the initialization.
  SourceRange getRange() const { 
    return SourceRange(Locations[0], Locations[1]);
  }
  
  /// \brief Retrieve the location of the equal sign for copy initialization
  /// (if present).
  SourceLocation getEqualLoc() const {
    assert(Kind == SIK_Copy && "Only copy initialization has an '='");
    return Locations[1];
  }
  
  /// \brief Retrieve the source range containing the locations of the open
  /// and closing parentheses for value and direct initializations.
  SourceRange getParenRange() const {
    assert((getKind() == IK_Direct || Kind == SIK_Value) &&
           "Only direct- and value-initialization have parentheses");
    return SourceRange(Locations[1], Locations[2]);
  }
};

/// \brief Describes the sequence of initializations required to initialize
/// a given object or reference with a set of arguments.
class InitializationSequence {
public:
  /// \brief Describes the kind of initialization sequence computed.
  enum SequenceKind {
    /// \brief A failed initialization sequence. The failure kind tells what
    /// happened.
    FailedSequence = 0,
    
    /// \brief A dependent initialization, which could not be
    /// type-checked due to the presence of dependent types or
    /// dependently-type expressions.
    DependentSequence,

    /// \brief A user-defined conversion sequence.
    UserDefinedConversion,
    
    /// \brief A reference binding.
    ReferenceBinding,

    /// \brief List initialization
    ListInitialization
  };
  
  /// \brief Describes the kind of a particular step in an initialization
  /// sequence.
  enum StepKind {
    /// \brief Resolve the address of an overloaded function to a specific
    /// function declaration.
    SK_ResolveAddressOfOverloadedFunction,
    /// \brief Perform a derived-to-base cast, producing an rvalue.
    SK_CastDerivedToBaseRValue,
    /// \brief Perform a derived-to-base cast, producing an lvalue.
    SK_CastDerivedToBaseLValue,
    /// \brief Reference binding to an lvalue.
    SK_BindReference,
    /// \brief Reference binding to a temporary.
    SK_BindReferenceToTemporary,
    /// \brief Perform a user-defined conversion, either via a conversion
    /// function or via a constructor.
    SK_UserConversion,
    /// \brief Perform a qualification conversion, producing an rvalue.
    SK_QualificationConversionRValue,
    /// \brief Perform a qualification conversion, producing an lvalue.
    SK_QualificationConversionLValue,
    /// \brief Perform an implicit conversion sequence.
    SK_ConversionSequence,
    /// \brief Perform list-initialization
    SK_ListInitialization
  };
  
  /// \brief A single step in the initialization sequence.
  class Step {
  public:
    /// \brief The kind of conversion or initialization step we are taking.
    StepKind Kind;
    
    // \brief The type that results from this initialization.
    QualType Type;
    
    union {
      /// \brief When Kind == SK_ResolvedOverloadedFunction or Kind ==
      /// SK_UserConversion, the function that the expression should be 
      /// resolved to or the conversion function to call, respectively.
      FunctionDecl *Function;
      
      /// \brief When Kind = SK_ConversionSequence, the implicit conversion
      /// sequence 
      ImplicitConversionSequence *ICS;
    };
    
    void Destroy();
  };
  
private:
  /// \brief The kind of initialization sequence computed.
  enum SequenceKind SequenceKind;
  
  /// \brief Steps taken by this initialization.
  llvm::SmallVector<Step, 4> Steps;
  
public:
  /// \brief Describes why initialization failed.
  enum FailureKind {
    /// \brief Too many initializers provided for a reference.
    FK_TooManyInitsForReference,
    /// \brief Array must be initialized with an initializer list.
    FK_ArrayNeedsInitList,
    /// \brief Array must be initialized with an initializer list or a 
    /// string literal.
    FK_ArrayNeedsInitListOrStringLiteral,
    /// \brief Cannot resolve the address of an overloaded function.
    FK_AddressOfOverloadFailed,
    /// \brief Overloading due to reference initialization failed.
    FK_ReferenceInitOverloadFailed,
    /// \brief Non-const lvalue reference binding to a temporary.
    FK_NonConstLValueReferenceBindingToTemporary,
    /// \brief Non-const lvalue reference binding to an lvalue of unrelated
    /// type.
    FK_NonConstLValueReferenceBindingToUnrelated,
    /// \brief Rvalue reference binding to an lvalue.
    FK_RValueReferenceBindingToLValue,
    /// \brief Reference binding drops qualifiers.
    FK_ReferenceInitDropsQualifiers,
    /// \brief Reference binding failed.
    FK_ReferenceInitFailed,
    /// \brief Implicit conversion failed.
    FK_ConversionFailed,
    /// \brief Too many initializers for scalar
    FK_TooManyInitsForScalar,
    /// \brief Reference initialization from an initializer list
    FK_ReferenceBindingToInitList,
    /// \brief Initialization of some unused destination type with an
    /// initializer list.
    FK_InitListBadDestinationType,
    /// \brief Overloading for a user-defined conversion failed.
    FK_UserConversionOverloadFailed
  };
  
private:
  /// \brief The reason why initialization failued.
  FailureKind Failure;

  /// \brief The failed result of overload resolution.
  OverloadingResult FailedOverloadResult;
  
  /// \brief The candidate set created when initialization failed.
  OverloadCandidateSet FailedCandidateSet;
  
public:
  /// \brief Try to perform initialization of the given entity, creating a 
  /// record of the steps required to perform the initialization.
  ///
  /// The generated initialization sequence will either contain enough
  /// information to diagnose 
  ///
  /// \param S the semantic analysis object.
  ///
  /// \param Entity the entity being initialized.
  ///
  /// \param Kind the kind of initialization being performed.
  ///
  /// \param Args the argument(s) provided for initialization.
  ///
  /// \param NumArgs the number of arguments provided for initialization.
  InitializationSequence(Sema &S, 
                         const InitializedEntity &Entity,
                         const InitializationKind &Kind,
                         Expr **Args,
                         unsigned NumArgs);
  
  ~InitializationSequence();
  
  /// \brief Perform the actual initialization of the given entity based on
  /// the computed initialization sequence.
  ///
  /// \param S the semantic analysis object.
  ///
  /// \param Entity the entity being initialized.
  ///
  /// \param Kind the kind of initialization being performed.
  ///
  /// \param Args the argument(s) provided for initialization, ownership of
  /// which is transfered into the routine.
  ///
  /// \param ResultType if non-NULL, will be set to the type of the
  /// initialized object, which is the type of the declaration in most
  /// cases. However, when the initialized object is a variable of
  /// incomplete array type and the initializer is an initializer
  /// list, this type will be set to the completed array type.
  ///
  /// \returns an expression that performs the actual object initialization, if
  /// the initialization is well-formed. Otherwise, emits diagnostics
  /// and returns an invalid expression.
  Action::OwningExprResult Perform(Sema &S,
                                   const InitializedEntity &Entity,
                                   const InitializationKind &Kind,
                                   Action::MultiExprArg Args,
                                   QualType *ResultType = 0);
  
  /// \brief Diagnose an potentially-invalid initialization sequence.
  ///
  /// \returns true if the initialization sequence was ill-formed, 
  /// false otherwise.
  bool Diagnose(Sema &S, 
                const InitializedEntity &Entity,
                const InitializationKind &Kind,
                Expr **Args, unsigned NumArgs);
  
  /// \brief Determine the kind of initialization sequence computed.
  enum SequenceKind getKind() const { return SequenceKind; }
  
  /// \brief Set the kind of sequence computed.
  void setSequenceKind(enum SequenceKind SK) { SequenceKind = SK; }
  
  /// \brief Determine whether the initialization sequence is valid.
  operator bool() const { return SequenceKind != FailedSequence; }
  
  typedef llvm::SmallVector<Step, 4>::const_iterator step_iterator;
  step_iterator step_begin() const { return Steps.begin(); }
  step_iterator step_end()   const { return Steps.end(); }

  /// \brief Add a new step in the initialization that resolves the address
  /// of an overloaded function to a specific function declaration.
  ///
  /// \param Function the function to which the overloaded function reference
  /// resolves.
  void AddAddressOverloadResolutionStep(FunctionDecl *Function);
  
  /// \brief Add a new step in the initialization that performs a derived-to-
  /// base cast.
  ///
  /// \param BaseType the base type to which we will be casting.
  ///
  /// \param IsLValue true if the result of this cast will be treated as 
  /// an lvalue.
  void AddDerivedToBaseCastStep(QualType BaseType, bool IsLValue);
     
  /// \brief Add a new step binding a reference to an object.
  ///
  /// \param BindingTemporary true if we are binding a reference to a temporary
  /// object (thereby extending its lifetime); false if we are binding to an
  /// lvalue or an lvalue treated as an rvalue.
  void AddReferenceBindingStep(QualType T, bool BindingTemporary);
  
  /// \brief Add a new step invoking a conversion function, which is either
  /// a constructor or a conversion function.
  void AddUserConversionStep(FunctionDecl *Function, QualType T);
  
  /// \brief Add a new step that performs a qualification conversion to the
  /// given type.
  void AddQualificationConversionStep(QualType Ty, bool IsLValue);
  
  /// \brief Add a new step that applies an implicit conversion sequence.
  void AddConversionSequenceStep(const ImplicitConversionSequence &ICS,
                                 QualType T);

  /// \brief Add a list-initialiation step  
  void AddListInitializationStep(QualType T);

  /// \brief Note that this initialization sequence failed.
  void SetFailed(FailureKind Failure) {
    SequenceKind = FailedSequence;
    this->Failure = Failure;
  }
  
  /// \brief Note that this initialization sequence failed due to failed
  /// overload resolution.
  void SetOverloadFailure(FailureKind Failure, OverloadingResult Result);
  
  /// \brief Retrieve a reference to the candidate set when overload
  /// resolution fails.
  OverloadCandidateSet &getFailedCandidateSet() {
    return FailedCandidateSet;
  }

  /// \brief Determine why initialization failed.
  FailureKind getFailureKind() const {
    assert(getKind() == FailedSequence && "Not an initialization failure!");
    return Failure;
  }
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_INIT_H
