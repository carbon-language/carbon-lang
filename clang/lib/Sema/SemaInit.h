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
#include "clang/AST/Type.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Parse/Action.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace llvm {
  class raw_ostream;
}

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
    /// \brief The entity being initialized is a non-static data member 
    /// subobject.
    EK_Member,
    /// \brief The entity being initialized is an element of an array.
    EK_ArrayElement,
    /// \brief The entity being initialized is an object (or array of
    /// objects) allocated via new.
    EK_New,
    /// \brief The entity being initialized is a temporary object.
    EK_Temporary,
    /// \brief The entity being initialized is a base member subobject.
    EK_Base,
    /// \brief The entity being initialized is an element of a vector.
    /// or vector.
    EK_VectorElement,
    /// \brief The entity being initialized is a field of block descriptor for
    /// the copied-in c++ object.
    EK_BlockElement
  };
  
private:
  /// \brief The kind of entity being initialized.
  EntityKind Kind;

  /// \brief If non-NULL, the parent entity in which this
  /// initialization occurs.
  const InitializedEntity *Parent;

  /// \brief The type of the object or reference being initialized.
  QualType Type;
  
  union {
    /// \brief When Kind == EK_Variable, EK_Parameter, or EK_Member, 
    /// the VarDecl, ParmVarDecl, or FieldDecl, respectively.
    DeclaratorDecl *VariableOrMember;
    
    struct {
      /// \brief When Kind == EK_Result, EK_Exception, or EK_New, the
      /// location of the 'return', 'throw', or 'new' keyword,
      /// respectively. When Kind == EK_Temporary, the location where
      /// the temporary is being created.
      unsigned Location;
      
      /// \brief Whether the 
      bool NRVO;
    } LocAndNRVO;
    
    /// \brief When Kind == EK_Base, the base specifier that provides the 
    /// base class. The lower bit specifies whether the base is an inherited
    /// virtual base.
    uintptr_t Base;

    /// \brief When Kind == EK_ArrayElement or EK_VectorElement, the
    /// index of the array or vector element being initialized. 
    unsigned Index;
  };

  InitializedEntity() { }

  /// \brief Create the initialization entity for a variable.
  InitializedEntity(VarDecl *Var)
    : Kind(EK_Variable), Parent(0), Type(Var->getType()),
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Var)) { }
  
  /// \brief Create the initialization entity for a parameter.
  InitializedEntity(ParmVarDecl *Parm)
    : Kind(EK_Parameter), Parent(0), Type(Parm->getType().getUnqualifiedType()),
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Parm)) { }
  
  /// \brief Create the initialization entity for the result of a
  /// function, throwing an object, performing an explicit cast, or
  /// initializing a parameter for which there is no declaration.
  InitializedEntity(EntityKind Kind, SourceLocation Loc, QualType Type,
                    bool NRVO = false)
    : Kind(Kind), Parent(0), Type(Type)
  {
    LocAndNRVO.Location = Loc.getRawEncoding();
    LocAndNRVO.NRVO = NRVO;
  }
  
  /// \brief Create the initialization entity for a member subobject.
  InitializedEntity(FieldDecl *Member, const InitializedEntity *Parent) 
    : Kind(EK_Member), Parent(Parent), Type(Member->getType()),
      VariableOrMember(reinterpret_cast<DeclaratorDecl*>(Member)) { }
  
  /// \brief Create the initialization entity for an array element.
  InitializedEntity(ASTContext &Context, unsigned Index, 
                    const InitializedEntity &Parent);

public:
  /// \brief Create the initialization entity for a variable.
  static InitializedEntity InitializeVariable(VarDecl *Var) {
    return InitializedEntity(Var);
  }
  
  /// \brief Create the initialization entity for a parameter.
  static InitializedEntity InitializeParameter(ParmVarDecl *Parm) {
    return InitializedEntity(Parm);
  }

  /// \brief Create the initialization entity for a parameter that is
  /// only known by its type.
  static InitializedEntity InitializeParameter(QualType Type) {
    InitializedEntity Entity;
    Entity.Kind = EK_Parameter;
    Entity.Type = Type;
    Entity.Parent = 0;
    Entity.VariableOrMember = 0;
    return Entity;
  }

  /// \brief Create the initialization entity for the result of a function.
  static InitializedEntity InitializeResult(SourceLocation ReturnLoc,
                                            QualType Type, bool NRVO) {
    return InitializedEntity(EK_Result, ReturnLoc, Type, NRVO);
  }

  static InitializedEntity InitializeBlock(SourceLocation BlockVarLoc,
                                           QualType Type, bool NRVO) {
    return InitializedEntity(EK_BlockElement, BlockVarLoc, Type, NRVO);
  }
  
  /// \brief Create the initialization entity for an exception object.
  static InitializedEntity InitializeException(SourceLocation ThrowLoc,
                                               QualType Type, bool NRVO) {
    return InitializedEntity(EK_Exception, ThrowLoc, Type, NRVO);
  }

  /// \brief Create the initialization entity for an object allocated via new.
  static InitializedEntity InitializeNew(SourceLocation NewLoc, QualType Type) {
    return InitializedEntity(EK_New, NewLoc, Type);
  }
  
  /// \brief Create the initialization entity for a temporary.
  static InitializedEntity InitializeTemporary(QualType Type) {
    return InitializedEntity(EK_Temporary, SourceLocation(), Type);
  }
  
  /// \brief Create the initialization entity for a base class subobject.
  static InitializedEntity InitializeBase(ASTContext &Context,
                                          CXXBaseSpecifier *Base,
                                          bool IsInheritedVirtualBase);
  
  /// \brief Create the initialization entity for a member subobject.
  static InitializedEntity InitializeMember(FieldDecl *Member,
                                          const InitializedEntity *Parent = 0) {
    return InitializedEntity(Member, Parent);
  }
  
  /// \brief Create the initialization entity for an array element.
  static InitializedEntity InitializeElement(ASTContext &Context, 
                                             unsigned Index, 
                                             const InitializedEntity &Parent) {
    return InitializedEntity(Context, Index, Parent);
  }

  /// \brief Determine the kind of initialization.
  EntityKind getKind() const { return Kind; }
  
  /// \brief Retrieve the parent of the entity being initialized, when
  /// the initialization itself is occuring within the context of a
  /// larger initialization.
  const InitializedEntity *getParent() const { return Parent; }

  /// \brief Retrieve type being initialized.
  QualType getType() const { return Type; }
  
  /// \brief Retrieve the name of the entity being initialized.
  DeclarationName getName() const;

  /// \brief Retrieve the variable, parameter, or field being
  /// initialized.
  DeclaratorDecl *getDecl() const;

  /// \brief Determine whether this initialization allows the named return 
  /// value optimization, which also applies to thrown objects.
  bool allowsNRVO() const;
                                  
  /// \brief Retrieve the base specifier.
  CXXBaseSpecifier *getBaseSpecifier() const {
    assert(getKind() == EK_Base && "Not a base specifier");
    return reinterpret_cast<CXXBaseSpecifier *>(Base & ~0x1);
  }

  /// \brief Return whether the base is an inherited virtual base.
  bool isInheritedVirtualBase() const {
    assert(getKind() == EK_Base && "Not a base specifier");
    return Base & 0x1;
  }

  /// \brief Determine the location of the 'return' keyword when initializing
  /// the result of a function call.
  SourceLocation getReturnLoc() const {
    assert(getKind() == EK_Result && "No 'return' location!");
    return SourceLocation::getFromRawEncoding(LocAndNRVO.Location);
  }
  
  /// \brief Determine the location of the 'throw' keyword when initializing
  /// an exception object.
  SourceLocation getThrowLoc() const {
    assert(getKind() == EK_Exception && "No 'throw' location!");
    return SourceLocation::getFromRawEncoding(LocAndNRVO.Location);
  }

  /// \brief If this is already the initializer for an array or vector
  /// element, sets the element index.
  void setElementIndex(unsigned Index) {
    assert(getKind() == EK_ArrayElement || getKind() == EK_VectorElement);
    this->Index = Index;
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
    SIK_ImplicitValue,        ///< Implicit value initialization
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
                                        SourceLocation RParenLoc,
                                        bool isImplicit = false) {
    return InitializationKind(isImplicit? SIK_ImplicitValue : SIK_Value, 
                              InitLoc, LParenLoc, RParenLoc);
  }
  
  /// \brief Determine the initialization kind.
  InitKind getKind() const {
    if (Kind > SIK_ImplicitValue)
      return IK_Direct;
    if (Kind == SIK_ImplicitValue)
      return IK_Value;

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

  /// \brief Determine whether this initialization is an implicit
  /// value-initialization, e.g., as occurs during aggregate
  /// initialization.
  bool isImplicitValueInit() const { return Kind == SIK_ImplicitValue; }

  /// \brief Retrieve the location at which initialization is occurring.
  SourceLocation getLocation() const { return Locations[0]; }
  
  /// \brief Retrieve the source range that covers the initialization.
  SourceRange getRange() const { 
    return SourceRange(Locations[0], Locations[2]);
  }
  
  /// \brief Retrieve the location of the equal sign for copy initialization
  /// (if present).
  SourceLocation getEqualLoc() const {
    assert(Kind == SIK_Copy && "Only copy initialization has an '='");
    return Locations[1];
  }
  
  bool isCopyInit() const { return Kind == SIK_Copy; }
  
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
  ///
  /// FIXME: Much of this information is in the initialization steps... why is
  /// it duplicated here?
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
    
    /// \brief A constructor call.
    ConstructorInitialization,
    
    /// \brief A reference binding.
    ReferenceBinding,

    /// \brief List initialization
    ListInitialization,
    
    /// \brief Zero-initialization.
    ZeroInitialization,
    
    /// \brief No initialization required.
    NoInitialization,
    
    /// \brief Standard conversion sequence.
    StandardConversion,

    /// \brief C conversion sequence.
    CAssignment,

    /// \brief String initialization
    StringInit
  };
  
  /// \brief Describes the kind of a particular step in an initialization
  /// sequence.
  enum StepKind {
    /// \brief Resolve the address of an overloaded function to a specific
    /// function declaration.
    SK_ResolveAddressOfOverloadedFunction,
    /// \brief Perform a derived-to-base cast, producing an rvalue.
    SK_CastDerivedToBaseRValue,
    /// \brief Perform a derived-to-base cast, producing an xvalue.
    SK_CastDerivedToBaseXValue,
    /// \brief Perform a derived-to-base cast, producing an lvalue.
    SK_CastDerivedToBaseLValue,
    /// \brief Reference binding to an lvalue.
    SK_BindReference,
    /// \brief Reference binding to a temporary.
    SK_BindReferenceToTemporary,
    /// \brief An optional copy of a temporary object to another
    /// temporary object, which is permitted (but not required) by
    /// C++98/03 but not C++0x.
    SK_ExtraneousCopyToTemporary,
    /// \brief Perform a user-defined conversion, either via a conversion
    /// function or via a constructor.
    SK_UserConversion,
    /// \brief Perform a qualification conversion, producing an rvalue.
    SK_QualificationConversionRValue,
    /// \brief Perform a qualification conversion, producing an xvalue.
    SK_QualificationConversionXValue,
    /// \brief Perform a qualification conversion, producing an lvalue.
    SK_QualificationConversionLValue,
    /// \brief Perform an implicit conversion sequence.
    SK_ConversionSequence,
    /// \brief Perform list-initialization
    SK_ListInitialization,
    /// \brief Perform initialization via a constructor.
    SK_ConstructorInitialization,
    /// \brief Zero-initialize the object
    SK_ZeroInitialization,
    /// \brief C assignment
    SK_CAssignment,
    /// \brief Initialization by string
    SK_StringInit,
    /// \brief An initialization that "converts" an Objective-C object
    /// (not a point to an object) to another Objective-C object type.
    SK_ObjCObjectConversion
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
      ///
      /// Always a FunctionDecl.
      /// For conversion decls, the naming class is the source type.
      /// For construct decls, the naming class is the target type.
      struct {
        FunctionDecl *Function;
        DeclAccessPair FoundDecl;
      } Function;
      
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
    FK_UserConversionOverloadFailed,
    /// \brief Overloaded for initialization by constructor failed.
    FK_ConstructorOverloadFailed,
    /// \brief Default-initialization of a 'const' object.
    FK_DefaultInitOfConst,
    /// \brief Initialization of an incomplete type.
    FK_Incomplete
  };
  
private:
  /// \brief The reason why initialization failued.
  FailureKind Failure;

  /// \brief The failed result of overload resolution.
  OverloadingResult FailedOverloadResult;
  
  /// \brief The candidate set created when initialization failed.
  OverloadCandidateSet FailedCandidateSet;

  /// \brief Prints a follow-up note that highlights the location of
  /// the initialized entity, if it's remote.
  void PrintInitLocationNote(Sema &S, const InitializedEntity &Entity);

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

  /// \brief Determine whether this initialization is a direct reference 
  /// binding (C++ [dcl.init.ref]).
  bool isDirectReferenceBinding() const;
  
  /// \brief Determine whether this initialization failed due to an ambiguity.
  bool isAmbiguous() const;
  
  /// \brief Determine whether this initialization is direct call to a 
  /// constructor.
  bool isConstructorInitialization() const;
  
  /// \brief Add a new step in the initialization that resolves the address
  /// of an overloaded function to a specific function declaration.
  ///
  /// \param Function the function to which the overloaded function reference
  /// resolves.
  void AddAddressOverloadResolutionStep(FunctionDecl *Function,
                                        DeclAccessPair Found);
  
  /// \brief Add a new step in the initialization that performs a derived-to-
  /// base cast.
  ///
  /// \param BaseType the base type to which we will be casting.
  ///
  /// \param IsLValue true if the result of this cast will be treated as 
  /// an lvalue.
  void AddDerivedToBaseCastStep(QualType BaseType,
                                ImplicitCastExpr::ResultCategory Category);
     
  /// \brief Add a new step binding a reference to an object.
  ///
  /// \param BindingTemporary True if we are binding a reference to a temporary
  /// object (thereby extending its lifetime); false if we are binding to an
  /// lvalue or an lvalue treated as an rvalue.
  ///
  /// \param UnnecessaryCopy True if we should check for a copy
  /// constructor for a completely unnecessary but
  void AddReferenceBindingStep(QualType T, bool BindingTemporary);

  /// \brief Add a new step that makes an extraneous copy of the input
  /// to a temporary of the same class type.
  ///
  /// This extraneous copy only occurs during reference binding in
  /// C++98/03, where we are permitted (but not required) to introduce
  /// an extra copy. At a bare minimum, we must check that we could
  /// call the copy constructor, and produce a diagnostic if the copy
  /// constructor is inaccessible or no copy constructor matches.
  //
  /// \param T The type of the temporary being created.
  void AddExtraneousCopyToTemporary(QualType T);

  /// \brief Add a new step invoking a conversion function, which is either
  /// a constructor or a conversion function.
  void AddUserConversionStep(FunctionDecl *Function,
                             DeclAccessPair FoundDecl,
                             QualType T);
  
  /// \brief Add a new step that performs a qualification conversion to the
  /// given type.
  void AddQualificationConversionStep(QualType Ty,
                                     ImplicitCastExpr::ResultCategory Category);
  
  /// \brief Add a new step that applies an implicit conversion sequence.
  void AddConversionSequenceStep(const ImplicitConversionSequence &ICS,
                                 QualType T);

  /// \brief Add a list-initialiation step  
  void AddListInitializationStep(QualType T);

  /// \brief Add a constructor-initialization step.
  void AddConstructorInitializationStep(CXXConstructorDecl *Constructor,
                                        AccessSpecifier Access,
                                        QualType T);

  /// \brief Add a zero-initialization step.
  void AddZeroInitializationStep(QualType T);
  
  /// \brief Add a C assignment step.
  //
  // FIXME: It isn't clear whether this should ever be needed;
  // ideally, we would handle everything needed in C in the common
  // path. However, that isn't the case yet.
  void AddCAssignmentStep(QualType T);

  /// \brief Add a string init step.
  void AddStringInitStep(QualType T);

  /// \brief Add an Objective-C object conversion step, which is
  /// always a no-op.
  void AddObjCObjectConversionStep(QualType T);

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
  
  /// \brief Dump a representation of this initialization sequence to 
  /// the given stream, for debugging purposes.
  void dump(llvm::raw_ostream &OS) const;
  
  /// \brief Dump a representation of this initialization sequence to 
  /// standard error, for debugging purposes.
  void dump() const;
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_INIT_H
