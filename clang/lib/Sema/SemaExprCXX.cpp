//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ expressions.
//
//===----------------------------------------------------------------------===//

#include "SemaInherit.h"
#include "Sema.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

/// ActOnCXXConversionFunctionExpr - Parse a C++ conversion function
/// name (e.g., operator void const *) as an expression. This is
/// very similar to ActOnIdentifierExpr, except that instead of
/// providing an identifier the parser provides the type of the
/// conversion function.
Sema::OwningExprResult
Sema::ActOnCXXConversionFunctionExpr(Scope *S, SourceLocation OperatorLoc,
                                     TypeTy *Ty, bool HasTrailingLParen,
                                     const CXXScopeSpec &SS,
                                     bool isAddressOfOperand) {
  //FIXME: Preserve type source info.
  QualType ConvType = GetTypeFromParser(Ty);
  CanQualType ConvTypeCanon = Context.getCanonicalType(ConvType);
  DeclarationName ConvName 
    = Context.DeclarationNames.getCXXConversionFunctionName(ConvTypeCanon);
  return ActOnDeclarationNameExpr(S, OperatorLoc, ConvName, HasTrailingLParen,
                                  &SS, isAddressOfOperand);
}

/// ActOnCXXOperatorFunctionIdExpr - Parse a C++ overloaded operator
/// name (e.g., @c operator+ ) as an expression. This is very
/// similar to ActOnIdentifierExpr, except that instead of providing
/// an identifier the parser provides the kind of overloaded
/// operator that was parsed.
Sema::OwningExprResult
Sema::ActOnCXXOperatorFunctionIdExpr(Scope *S, SourceLocation OperatorLoc,
                                     OverloadedOperatorKind Op,
                                     bool HasTrailingLParen,
                                     const CXXScopeSpec &SS,
                                     bool isAddressOfOperand) {
  DeclarationName Name = Context.DeclarationNames.getCXXOperatorName(Op);
  return ActOnDeclarationNameExpr(S, OperatorLoc, Name, HasTrailingLParen, &SS,
                                  isAddressOfOperand);
}

/// ActOnCXXTypeidOfType - Parse typeid( type-id ).
Action::OwningExprResult
Sema::ActOnCXXTypeid(SourceLocation OpLoc, SourceLocation LParenLoc,
                     bool isType, void *TyOrExpr, SourceLocation RParenLoc) {
  NamespaceDecl *StdNs = GetStdNamespace();
  if (!StdNs)
    return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));

  if (isType)
    // FIXME: Preserve type source info.
    TyOrExpr = GetTypeFromParser(TyOrExpr).getAsOpaquePtr();

  IdentifierInfo *TypeInfoII = &PP.getIdentifierTable().get("type_info");
  Decl *TypeInfoDecl = LookupQualifiedName(StdNs, TypeInfoII, LookupTagName);
  RecordDecl *TypeInfoRecordDecl = dyn_cast_or_null<RecordDecl>(TypeInfoDecl);
  if (!TypeInfoRecordDecl)
    return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));

  QualType TypeInfoType = Context.getTypeDeclType(TypeInfoRecordDecl);

  if (!isType) {
    // C++0x [expr.typeid]p3:
    //   When typeid is applied to an expression other than an lvalue of a 
    //   polymorphic class type [...] [the] expression is an unevaluated 
    //   operand.
    
    // FIXME: if the type of the expression is a class type, the class
    // shall be completely defined.
    bool isUnevaluatedOperand = true;
    Expr *E = static_cast<Expr *>(TyOrExpr);
    if (E && !E->isTypeDependent() && E->isLvalue(Context) == Expr::LV_Valid) {
      QualType T = E->getType();
      if (const RecordType *RecordT = T->getAs<RecordType>()) {
        CXXRecordDecl *RecordD = cast<CXXRecordDecl>(RecordT->getDecl());
        if (RecordD->isPolymorphic())
          isUnevaluatedOperand = false;
      }
    }
    
    // If this is an unevaluated operand, clear out the set of declaration
    // references we have been computing.
    if (isUnevaluatedOperand)
      PotentiallyReferencedDeclStack.back().clear();
  }
  
  return Owned(new (Context) CXXTypeidExpr(isType, TyOrExpr,
                                           TypeInfoType.withConst(),
                                           SourceRange(OpLoc, RParenLoc)));
}

/// ActOnCXXBoolLiteral - Parse {true,false} literals.
Action::OwningExprResult
Sema::ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind == tok::kw_true || Kind == tok::kw_false) &&
         "Unknown C++ Boolean value!");
  return Owned(new (Context) CXXBoolLiteralExpr(Kind == tok::kw_true,
                                                Context.BoolTy, OpLoc));
}

/// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
Action::OwningExprResult
Sema::ActOnCXXNullPtrLiteral(SourceLocation Loc) {
  return Owned(new (Context) CXXNullPtrLiteralExpr(Context.NullPtrTy, Loc));
}

/// ActOnCXXThrow - Parse throw expressions.
Action::OwningExprResult
Sema::ActOnCXXThrow(SourceLocation OpLoc, ExprArg E) {
  Expr *Ex = E.takeAs<Expr>();
  if (Ex && !Ex->isTypeDependent() && CheckCXXThrowOperand(OpLoc, Ex))
    return ExprError();
  return Owned(new (Context) CXXThrowExpr(Ex, Context.VoidTy, OpLoc));
}

/// CheckCXXThrowOperand - Validate the operand of a throw.
bool Sema::CheckCXXThrowOperand(SourceLocation ThrowLoc, Expr *&E) {
  // C++ [except.throw]p3:
  //   [...] adjusting the type from "array of T" or "function returning T"
  //   to "pointer to T" or "pointer to function returning T", [...]
  DefaultFunctionArrayConversion(E);

  //   If the type of the exception would be an incomplete type or a pointer
  //   to an incomplete type other than (cv) void the program is ill-formed.
  QualType Ty = E->getType();
  int isPointer = 0;
  if (const PointerType* Ptr = Ty->getAs<PointerType>()) {
    Ty = Ptr->getPointeeType();
    isPointer = 1;
  }
  if (!isPointer || !Ty->isVoidType()) {
    if (RequireCompleteType(ThrowLoc, Ty,
                            isPointer ? diag::err_throw_incomplete_ptr
                                      : diag::err_throw_incomplete,
                            E->getSourceRange(), SourceRange(), QualType()))
      return true;
  }

  // FIXME: Construct a temporary here.
  return false;
}

Action::OwningExprResult Sema::ActOnCXXThis(SourceLocation ThisLoc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  if (!isa<FunctionDecl>(CurContext))
    return ExprError(Diag(ThisLoc, diag::err_invalid_this_use));

  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext))
    if (MD->isInstance())
      return Owned(new (Context) CXXThisExpr(ThisLoc,
                                             MD->getThisType(Context)));

  return ExprError(Diag(ThisLoc, diag::err_invalid_this_use));
}

/// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
Action::OwningExprResult
Sema::ActOnCXXTypeConstructExpr(SourceRange TypeRange, TypeTy *TypeRep,
                                SourceLocation LParenLoc,
                                MultiExprArg exprs,
                                SourceLocation *CommaLocs,
                                SourceLocation RParenLoc) {
  assert(TypeRep && "Missing type!");
  // FIXME: Preserve type source info.
  QualType Ty = GetTypeFromParser(TypeRep);
  unsigned NumExprs = exprs.size();
  Expr **Exprs = (Expr**)exprs.get();
  SourceLocation TyBeginLoc = TypeRange.getBegin();
  SourceRange FullRange = SourceRange(TyBeginLoc, RParenLoc);

  if (Ty->isDependentType() ||
      CallExpr::hasAnyTypeDependentArguments(Exprs, NumExprs)) {
    exprs.release();
    
    return Owned(CXXUnresolvedConstructExpr::Create(Context, 
                                                    TypeRange.getBegin(), Ty, 
                                                    LParenLoc,
                                                    Exprs, NumExprs,
                                                    RParenLoc));
  }


  // C++ [expr.type.conv]p1:
  // If the expression list is a single expression, the type conversion
  // expression is equivalent (in definedness, and if defined in meaning) to the
  // corresponding cast expression.
  //
  if (NumExprs == 1) {
    CastExpr::CastKind Kind = CastExpr::CK_Unknown;
    if (CheckCastTypes(TypeRange, Ty, Exprs[0], Kind, /*functional-style*/true))
      return ExprError();
    exprs.release();
    return Owned(new (Context) CXXFunctionalCastExpr(Ty.getNonReferenceType(),
                                                     Ty, TyBeginLoc, Kind,
                                                     Exprs[0], RParenLoc));
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    CXXRecordDecl *Record = cast<CXXRecordDecl>(RT->getDecl());

    // FIXME: We should always create a CXXTemporaryObjectExpr here unless
    // both the ctor and dtor are trivial.
    if (NumExprs > 1 || Record->hasUserDeclaredConstructor()) {
      CXXConstructorDecl *Constructor
        = PerformInitializationByConstructor(Ty, Exprs, NumExprs,
                                             TypeRange.getBegin(),
                                             SourceRange(TypeRange.getBegin(),
                                                         RParenLoc),
                                             DeclarationName(),
                                             IK_Direct);

      if (!Constructor)
        return ExprError();

      exprs.release();
      Expr *E = new (Context) CXXTemporaryObjectExpr(Context, Constructor, 
                                                     Ty, TyBeginLoc, Exprs,
                                                     NumExprs, RParenLoc);
      return MaybeBindToTemporary(E);
    }

    // Fall through to value-initialize an object of class type that
    // doesn't have a user-declared default constructor.
  }

  // C++ [expr.type.conv]p1:
  // If the expression list specifies more than a single value, the type shall
  // be a class with a suitably declared constructor.
  //
  if (NumExprs > 1)
    return ExprError(Diag(CommaLocs[0],
                          diag::err_builtin_func_cast_more_than_one_arg)
      << FullRange);

  assert(NumExprs == 0 && "Expected 0 expressions");

  // C++ [expr.type.conv]p2:
  // The expression T(), where T is a simple-type-specifier for a non-array
  // complete object type or the (possibly cv-qualified) void type, creates an
  // rvalue of the specified type, which is value-initialized.
  //
  if (Ty->isArrayType())
    return ExprError(Diag(TyBeginLoc,
                          diag::err_value_init_for_array_type) << FullRange);
  if (!Ty->isDependentType() && !Ty->isVoidType() &&
      RequireCompleteType(TyBeginLoc, Ty,
                          diag::err_invalid_incomplete_type_use, FullRange))
    return ExprError();

  if (RequireNonAbstractType(TyBeginLoc, Ty,
                             diag::err_allocation_of_abstract_type))
    return ExprError();
  
  exprs.release();
  return Owned(new (Context) CXXZeroInitValueExpr(Ty, TyBeginLoc, RParenLoc));
}


/// ActOnCXXNew - Parsed a C++ 'new' expression (C++ 5.3.4), as in e.g.:
/// @code new (memory) int[size][4] @endcode
/// or
/// @code ::new Foo(23, "hello") @endcode
/// For the interpretation of this heap of arguments, consult the base version.
Action::OwningExprResult
Sema::ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                  SourceLocation PlacementLParen, MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen, bool ParenTypeId,
                  Declarator &D, SourceLocation ConstructorLParen,
                  MultiExprArg ConstructorArgs,
                  SourceLocation ConstructorRParen)
{
  Expr *ArraySize = 0;
  unsigned Skip = 0;
  // If the specified type is an array, unwrap it and save the expression.
  if (D.getNumTypeObjects() > 0 &&
      D.getTypeObject(0).Kind == DeclaratorChunk::Array) {
    DeclaratorChunk &Chunk = D.getTypeObject(0);
    if (Chunk.Arr.hasStatic)
      return ExprError(Diag(Chunk.Loc, diag::err_static_illegal_in_new)
        << D.getSourceRange());
    if (!Chunk.Arr.NumElts)
      return ExprError(Diag(Chunk.Loc, diag::err_array_new_needs_size)
        << D.getSourceRange());
    ArraySize = static_cast<Expr*>(Chunk.Arr.NumElts);
    Skip = 1;
  }

  //FIXME: Store DeclaratorInfo in CXXNew expression.
  DeclaratorInfo *DInfo = 0;
  QualType AllocType = GetTypeForDeclarator(D, /*Scope=*/0, &DInfo, Skip);
  if (D.isInvalidType())
    return ExprError();

  // Every dimension shall be of constant size.
  unsigned i = 1;
  QualType ElementType = AllocType;
  while (const ArrayType *Array = Context.getAsArrayType(ElementType)) {
    if (!Array->isConstantArrayType()) {
      Diag(D.getTypeObject(i).Loc, diag::err_new_array_nonconst)
        << static_cast<Expr*>(D.getTypeObject(i).Arr.NumElts)->getSourceRange();
      return ExprError();
    }
    ElementType = Array->getElementType();
    ++i;
  }

  return BuildCXXNew(StartLoc, UseGlobal, 
                     PlacementLParen,
                     move(PlacementArgs), 
                     PlacementRParen,
                     ParenTypeId,
                     AllocType, 
                     D.getSourceRange().getBegin(),
                     D.getSourceRange(),
                     Owned(ArraySize),
                     ConstructorLParen,
                     move(ConstructorArgs),
                     ConstructorRParen);
}

Sema::OwningExprResult 
Sema::BuildCXXNew(SourceLocation StartLoc, bool UseGlobal,
                  SourceLocation PlacementLParen,
                  MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen,
                  bool ParenTypeId, 
                  QualType AllocType,
                  SourceLocation TypeLoc,
                  SourceRange TypeRange,
                  ExprArg ArraySizeE,
                  SourceLocation ConstructorLParen,
                  MultiExprArg ConstructorArgs,
                  SourceLocation ConstructorRParen) {
  if (CheckAllocatedType(AllocType, TypeLoc, TypeRange))
    return ExprError();

  QualType ResultType = Context.getPointerType(AllocType);

  // That every array dimension except the first is constant was already
  // checked by the type check above.

  // C++ 5.3.4p6: "The expression in a direct-new-declarator shall have integral
  //   or enumeration type with a non-negative value."
  Expr *ArraySize = (Expr *)ArraySizeE.get();
  if (ArraySize && !ArraySize->isTypeDependent()) {
    QualType SizeType = ArraySize->getType();
    if (!SizeType->isIntegralType() && !SizeType->isEnumeralType())
      return ExprError(Diag(ArraySize->getSourceRange().getBegin(),
                            diag::err_array_size_not_integral)
        << SizeType << ArraySize->getSourceRange());
    // Let's see if this is a constant < 0. If so, we reject it out of hand.
    // We don't care about special rules, so we tell the machinery it's not
    // evaluated - it gives us a result in more cases.
    if (!ArraySize->isValueDependent()) {
      llvm::APSInt Value;
      if (ArraySize->isIntegerConstantExpr(Value, Context, 0, false)) {
        if (Value < llvm::APSInt(
                        llvm::APInt::getNullValue(Value.getBitWidth()), false))
          return ExprError(Diag(ArraySize->getSourceRange().getBegin(),
                           diag::err_typecheck_negative_array_size)
            << ArraySize->getSourceRange());
      }
    }
  }

  FunctionDecl *OperatorNew = 0;
  FunctionDecl *OperatorDelete = 0;
  Expr **PlaceArgs = (Expr**)PlacementArgs.get();
  unsigned NumPlaceArgs = PlacementArgs.size();
  if (!AllocType->isDependentType() &&
      !Expr::hasAnyTypeDependentArguments(PlaceArgs, NumPlaceArgs) &&
      FindAllocationFunctions(StartLoc,
                              SourceRange(PlacementLParen, PlacementRParen),
                              UseGlobal, AllocType, ArraySize, PlaceArgs,
                              NumPlaceArgs, OperatorNew, OperatorDelete))
    return ExprError();

  bool Init = ConstructorLParen.isValid();
  // --- Choosing a constructor ---
  // C++ 5.3.4p15
  // 1) If T is a POD and there's no initializer (ConstructorLParen is invalid)
  //   the object is not initialized. If the object, or any part of it, is
  //   const-qualified, it's an error.
  // 2) If T is a POD and there's an empty initializer, the object is value-
  //   initialized.
  // 3) If T is a POD and there's one initializer argument, the object is copy-
  //   constructed.
  // 4) If T is a POD and there's more initializer arguments, it's an error.
  // 5) If T is not a POD, the initializer arguments are used as constructor
  //   arguments.
  //
  // Or by the C++0x formulation:
  // 1) If there's no initializer, the object is default-initialized according
  //    to C++0x rules.
  // 2) Otherwise, the object is direct-initialized.
  CXXConstructorDecl *Constructor = 0;
  Expr **ConsArgs = (Expr**)ConstructorArgs.get();
  const RecordType *RT;
  unsigned NumConsArgs = ConstructorArgs.size();
  if (AllocType->isDependentType()) {
    // Skip all the checks.
  } else if ((RT = AllocType->getAs<RecordType>()) &&
             !AllocType->isAggregateType()) {
    Constructor = PerformInitializationByConstructor(
                      AllocType, ConsArgs, NumConsArgs,
                      TypeLoc,
                      SourceRange(TypeLoc, ConstructorRParen),
                      RT->getDecl()->getDeclName(),
                      NumConsArgs != 0 ? IK_Direct : IK_Default);
    if (!Constructor)
      return ExprError();
  } else {
    if (!Init) {
      // FIXME: Check that no subpart is const.
      if (AllocType.isConstQualified())
        return ExprError(Diag(StartLoc, diag::err_new_uninitialized_const)
                           << TypeRange);
    } else if (NumConsArgs == 0) {
      // Object is value-initialized. Do nothing.
    } else if (NumConsArgs == 1) {
      // Object is direct-initialized.
      // FIXME: What DeclarationName do we pass in here?
      if (CheckInitializerTypes(ConsArgs[0], AllocType, StartLoc,
                                DeclarationName() /*AllocType.getAsString()*/,
                                /*DirectInit=*/true))
        return ExprError();
    } else {
      return ExprError(Diag(StartLoc,
                            diag::err_builtin_direct_init_more_than_one_arg)
        << SourceRange(ConstructorLParen, ConstructorRParen));
    }
  }

  // FIXME: Also check that the destructor is accessible. (C++ 5.3.4p16)

  PlacementArgs.release();
  ConstructorArgs.release();
  ArraySizeE.release();
  return Owned(new (Context) CXXNewExpr(UseGlobal, OperatorNew, PlaceArgs,
                        NumPlaceArgs, ParenTypeId, ArraySize, Constructor, Init,
                        ConsArgs, NumConsArgs, OperatorDelete, ResultType,
                        StartLoc, Init ? ConstructorRParen : SourceLocation()));  
}

/// CheckAllocatedType - Checks that a type is suitable as the allocated type
/// in a new-expression.
/// dimension off and stores the size expression in ArraySize.
bool Sema::CheckAllocatedType(QualType AllocType, SourceLocation Loc,
                              SourceRange R)
{
  // C++ 5.3.4p1: "[The] type shall be a complete object type, but not an
  //   abstract class type or array thereof.
  if (AllocType->isFunctionType())
    return Diag(Loc, diag::err_bad_new_type)
      << AllocType << 0 << R;
  else if (AllocType->isReferenceType())
    return Diag(Loc, diag::err_bad_new_type)
      << AllocType << 1 << R;
  else if (!AllocType->isDependentType() &&
           RequireCompleteType(Loc, AllocType,
                               diag::err_new_incomplete_type,
                               R))
    return true;
  else if (RequireNonAbstractType(Loc, AllocType,
                                  diag::err_allocation_of_abstract_type))
    return true;

  return false;
}

/// FindAllocationFunctions - Finds the overloads of operator new and delete
/// that are appropriate for the allocation.
bool Sema::FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
                                   bool UseGlobal, QualType AllocType,
                                   bool IsArray, Expr **PlaceArgs,
                                   unsigned NumPlaceArgs,
                                   FunctionDecl *&OperatorNew,
                                   FunctionDecl *&OperatorDelete)
{
  // --- Choosing an allocation function ---
  // C++ 5.3.4p8 - 14 & 18
  // 1) If UseGlobal is true, only look in the global scope. Else, also look
  //   in the scope of the allocated class.
  // 2) If an array size is given, look for operator new[], else look for
  //   operator new.
  // 3) The first argument is always size_t. Append the arguments from the
  //   placement form.
  // FIXME: Also find the appropriate delete operator.

  llvm::SmallVector<Expr*, 8> AllocArgs(1 + NumPlaceArgs);
  // We don't care about the actual value of this argument.
  // FIXME: Should the Sema create the expression and embed it in the syntax
  // tree? Or should the consumer just recalculate the value?
  IntegerLiteral Size(llvm::APInt::getNullValue(
                      Context.Target.getPointerWidth(0)),
                      Context.getSizeType(),
                      SourceLocation());
  AllocArgs[0] = &Size;
  std::copy(PlaceArgs, PlaceArgs + NumPlaceArgs, AllocArgs.begin() + 1);

  DeclarationName NewName = Context.DeclarationNames.getCXXOperatorName(
                                        IsArray ? OO_Array_New : OO_New);
  if (AllocType->isRecordType() && !UseGlobal) {
    CXXRecordDecl *Record 
      = cast<CXXRecordDecl>(AllocType->getAs<RecordType>()->getDecl());
    // FIXME: We fail to find inherited overloads.
    if (FindAllocationOverload(StartLoc, Range, NewName, &AllocArgs[0],
                          AllocArgs.size(), Record, /*AllowMissing=*/true,
                          OperatorNew))
      return true;
  }
  if (!OperatorNew) {
    // Didn't find a member overload. Look for a global one.
    DeclareGlobalNewDelete();
    DeclContext *TUDecl = Context.getTranslationUnitDecl();
    if (FindAllocationOverload(StartLoc, Range, NewName, &AllocArgs[0],
                          AllocArgs.size(), TUDecl, /*AllowMissing=*/false,
                          OperatorNew))
      return true;
  }

  // FindAllocationOverload can change the passed in arguments, so we need to
  // copy them back.
  if (NumPlaceArgs > 0)
    std::copy(&AllocArgs[1], AllocArgs.end(), PlaceArgs);
  
  return false;
}

/// FindAllocationOverload - Find an fitting overload for the allocation
/// function in the specified scope.
bool Sema::FindAllocationOverload(SourceLocation StartLoc, SourceRange Range,
                                  DeclarationName Name, Expr** Args,
                                  unsigned NumArgs, DeclContext *Ctx,
                                  bool AllowMissing, FunctionDecl *&Operator)
{
  DeclContext::lookup_iterator Alloc, AllocEnd;
  llvm::tie(Alloc, AllocEnd) = Ctx->lookup(Name);
  if (Alloc == AllocEnd) {
    if (AllowMissing)
      return false;
    return Diag(StartLoc, diag::err_ovl_no_viable_function_in_call)
      << Name << Range;
  }

  OverloadCandidateSet Candidates;
  for (; Alloc != AllocEnd; ++Alloc) {
    // Even member operator new/delete are implicitly treated as
    // static, so don't use AddMemberCandidate.
    if (FunctionDecl *Fn = dyn_cast<FunctionDecl>(*Alloc))
      AddOverloadCandidate(Fn, Args, NumArgs, Candidates,
                           /*SuppressUserConversions=*/false);
  }

  // Do the resolution.
  OverloadCandidateSet::iterator Best;
  switch(BestViableFunction(Candidates, StartLoc, Best)) {
  case OR_Success: {
    // Got one!
    FunctionDecl *FnDecl = Best->Function;
    // The first argument is size_t, and the first parameter must be size_t,
    // too. This is checked on declaration and can be assumed. (It can't be
    // asserted on, though, since invalid decls are left in there.)
    for (unsigned i = 1; i < NumArgs; ++i) {
      // FIXME: Passing word to diagnostic.
      if (PerformCopyInitialization(Args[i],
                                    FnDecl->getParamDecl(i)->getType(),
                                    "passing"))
        return true;
    }
    Operator = FnDecl;
    return false;
  }

  case OR_No_Viable_Function:
    Diag(StartLoc, diag::err_ovl_no_viable_function_in_call)
      << Name << Range;
    PrintOverloadCandidates(Candidates, /*OnlyViable=*/false);
    return true;

  case OR_Ambiguous:
    Diag(StartLoc, diag::err_ovl_ambiguous_call)
      << Name << Range;
    PrintOverloadCandidates(Candidates, /*OnlyViable=*/true);
    return true;

  case OR_Deleted:
    Diag(StartLoc, diag::err_ovl_deleted_call)
      << Best->Function->isDeleted()
      << Name << Range;
    PrintOverloadCandidates(Candidates, /*OnlyViable=*/true);
    return true;
  }
  assert(false && "Unreachable, bad result from BestViableFunction");
  return true;
}


/// DeclareGlobalNewDelete - Declare the global forms of operator new and
/// delete. These are:
/// @code
///   void* operator new(std::size_t) throw(std::bad_alloc);
///   void* operator new[](std::size_t) throw(std::bad_alloc);
///   void operator delete(void *) throw();
///   void operator delete[](void *) throw();
/// @endcode
/// Note that the placement and nothrow forms of new are *not* implicitly
/// declared. Their use requires including \<new\>.
void Sema::DeclareGlobalNewDelete()
{
  if (GlobalNewDeleteDeclared)
    return;
  GlobalNewDeleteDeclared = true;

  QualType VoidPtr = Context.getPointerType(Context.VoidTy);
  QualType SizeT = Context.getSizeType();

  // FIXME: Exception specifications are not added.
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_New),
      VoidPtr, SizeT);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Array_New),
      VoidPtr, SizeT);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Delete),
      Context.VoidTy, VoidPtr);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Array_Delete),
      Context.VoidTy, VoidPtr);
}

/// DeclareGlobalAllocationFunction - Declares a single implicit global
/// allocation function if it doesn't already exist.
void Sema::DeclareGlobalAllocationFunction(DeclarationName Name,
                                           QualType Return, QualType Argument)
{
  DeclContext *GlobalCtx = Context.getTranslationUnitDecl();

  // Check if this function is already declared.
  {
    DeclContext::lookup_iterator Alloc, AllocEnd;
    for (llvm::tie(Alloc, AllocEnd) = GlobalCtx->lookup(Name);
         Alloc != AllocEnd; ++Alloc) {
      // FIXME: Do we need to check for default arguments here?
      FunctionDecl *Func = cast<FunctionDecl>(*Alloc);
      if (Func->getNumParams() == 1 &&
          Context.getCanonicalType(Func->getParamDecl(0)->getType())==Argument)
        return;
    }
  }

  QualType FnType = Context.getFunctionType(Return, &Argument, 1, false, 0);
  FunctionDecl *Alloc =
    FunctionDecl::Create(Context, GlobalCtx, SourceLocation(), Name,
                         FnType, /*DInfo=*/0, FunctionDecl::None, false, true,
                         SourceLocation());
  Alloc->setImplicit();
  ParmVarDecl *Param = ParmVarDecl::Create(Context, Alloc, SourceLocation(),
                                           0, Argument, /*DInfo=*/0,
                                           VarDecl::None, 0);
  Alloc->setParams(Context, &Param, 1);

  // FIXME: Also add this declaration to the IdentifierResolver, but
  // make sure it is at the end of the chain to coincide with the
  // global scope.
  ((DeclContext *)TUScope->getEntity())->addDecl(Alloc);
}

/// ActOnCXXDelete - Parsed a C++ 'delete' expression (C++ 5.3.5), as in:
/// @code ::delete ptr; @endcode
/// or
/// @code delete [] ptr; @endcode
Action::OwningExprResult
Sema::ActOnCXXDelete(SourceLocation StartLoc, bool UseGlobal,
                     bool ArrayForm, ExprArg Operand)
{
  // C++ 5.3.5p1: "The operand shall have a pointer type, or a class type
  //   having a single conversion function to a pointer type. The result has
  //   type void."
  // DR599 amends "pointer type" to "pointer to object type" in both cases.

  FunctionDecl *OperatorDelete = 0;
  
  Expr *Ex = (Expr *)Operand.get();
  if (!Ex->isTypeDependent()) {
    QualType Type = Ex->getType();

    if (Type->isRecordType()) {
      // FIXME: Find that one conversion function and amend the type.
    }

    if (!Type->isPointerType())
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex->getSourceRange());

    QualType Pointee = Type->getAs<PointerType>()->getPointeeType();
    if (Pointee->isFunctionType() || Pointee->isVoidType())
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex->getSourceRange());
    else if (!Pointee->isDependentType() &&
             RequireCompleteType(StartLoc, Pointee, 
                                 diag::warn_delete_incomplete,
                                 Ex->getSourceRange()))
      return ExprError();

    // FIXME: This should be shared with the code for finding the delete 
    // operator in ActOnCXXNew.
    IntegerLiteral Size(llvm::APInt::getNullValue(
                        Context.Target.getPointerWidth(0)),
                        Context.getSizeType(),
                        SourceLocation());
    ImplicitCastExpr Cast(Context.getPointerType(Context.VoidTy),
                          CastExpr::CK_Unknown, &Size, false);
    Expr *DeleteArg = &Cast;
    
    DeclarationName DeleteName = Context.DeclarationNames.getCXXOperatorName(
                                      ArrayForm ? OO_Array_Delete : OO_Delete);

    if (Pointee->isRecordType() && !UseGlobal) {
      CXXRecordDecl *Record 
        = cast<CXXRecordDecl>(Pointee->getAs<RecordType>()->getDecl());
      // FIXME: We fail to find inherited overloads.
      if (FindAllocationOverload(StartLoc, SourceRange(), DeleteName,
                                 &DeleteArg, 1, Record, /*AllowMissing=*/true,
                                 OperatorDelete))
        return ExprError();
    }
    
    if (!OperatorDelete) {
      // Didn't find a member overload. Look for a global one.
      DeclareGlobalNewDelete();
      DeclContext *TUDecl = Context.getTranslationUnitDecl();
      if (FindAllocationOverload(StartLoc, SourceRange(), DeleteName, 
                                 &DeleteArg, 1, TUDecl, /*AllowMissing=*/false,
                                 OperatorDelete))
        return ExprError();
    }
    
    // FIXME: Check access and ambiguity of operator delete and destructor.
  }

  Operand.release();
  return Owned(new (Context) CXXDeleteExpr(Context.VoidTy, UseGlobal, ArrayForm,
                                           OperatorDelete, Ex, StartLoc));
}


/// ActOnCXXConditionDeclarationExpr - Parsed a condition declaration of a
/// C++ if/switch/while/for statement.
/// e.g: "if (int x = f()) {...}"
Action::OwningExprResult
Sema::ActOnCXXConditionDeclarationExpr(Scope *S, SourceLocation StartLoc,
                                       Declarator &D,
                                       SourceLocation EqualLoc,
                                       ExprArg AssignExprVal) {
  assert(AssignExprVal.get() && "Null assignment expression");

  // C++ 6.4p2:
  // The declarator shall not specify a function or an array.
  // The type-specifier-seq shall not contain typedef and shall not declare a
  // new class or enumeration.

  assert(D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
         "Parser allowed 'typedef' as storage class of condition decl.");

  // FIXME: Store DeclaratorInfo in the expression.
  DeclaratorInfo *DInfo = 0;
  TagDecl *OwnedTag = 0;
  QualType Ty = GetTypeForDeclarator(D, S, &DInfo, /*Skip=*/0, &OwnedTag);
  
  if (Ty->isFunctionType()) { // The declarator shall not specify a function...
    // We exit without creating a CXXConditionDeclExpr because a FunctionDecl
    // would be created and CXXConditionDeclExpr wants a VarDecl.
    return ExprError(Diag(StartLoc, diag::err_invalid_use_of_function_type)
      << SourceRange(StartLoc, EqualLoc));
  } else if (Ty->isArrayType()) { // ...or an array.
    Diag(StartLoc, diag::err_invalid_use_of_array_type)
      << SourceRange(StartLoc, EqualLoc);
  } else if (OwnedTag && OwnedTag->isDefinition()) {
    // The type-specifier-seq shall not declare a new class or enumeration.
    Diag(OwnedTag->getLocation(), diag::err_type_defined_in_condition);
  }

  DeclPtrTy Dcl = ActOnDeclarator(S, D);
  if (!Dcl)
    return ExprError();
  AddInitializerToDecl(Dcl, move(AssignExprVal), /*DirectInit=*/false);

  // Mark this variable as one that is declared within a conditional.
  // We know that the decl had to be a VarDecl because that is the only type of
  // decl that can be assigned and the grammar requires an '='.
  VarDecl *VD = cast<VarDecl>(Dcl.getAs<Decl>());
  VD->setDeclaredInCondition(true);
  return Owned(new (Context) CXXConditionDeclExpr(StartLoc, EqualLoc, VD));
}

/// CheckCXXBooleanCondition - Returns true if a conversion to bool is invalid.
bool Sema::CheckCXXBooleanCondition(Expr *&CondExpr) {
  // C++ 6.4p4:
  // The value of a condition that is an initialized declaration in a statement
  // other than a switch statement is the value of the declared variable
  // implicitly converted to type bool. If that conversion is ill-formed, the
  // program is ill-formed.
  // The value of a condition that is an expression is the value of the
  // expression, implicitly converted to bool.
  //
  return PerformContextuallyConvertToBool(CondExpr);
}

/// Helper function to determine whether this is the (deprecated) C++
/// conversion from a string literal to a pointer to non-const char or
/// non-const wchar_t (for narrow and wide string literals,
/// respectively).
bool 
Sema::IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType) {
  // Look inside the implicit cast, if it exists.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(From))
    From = Cast->getSubExpr();

  // A string literal (2.13.4) that is not a wide string literal can
  // be converted to an rvalue of type "pointer to char"; a wide
  // string literal can be converted to an rvalue of type "pointer
  // to wchar_t" (C++ 4.2p2).
  if (StringLiteral *StrLit = dyn_cast<StringLiteral>(From))
    if (const PointerType *ToPtrType = ToType->getAs<PointerType>())
      if (const BuiltinType *ToPointeeType 
          = ToPtrType->getPointeeType()->getAsBuiltinType()) {
        // This conversion is considered only when there is an
        // explicit appropriate pointer target type (C++ 4.2p2).
        if (ToPtrType->getPointeeType().getCVRQualifiers() == 0 &&
            ((StrLit->isWide() && ToPointeeType->isWideCharType()) ||
             (!StrLit->isWide() &&
              (ToPointeeType->getKind() == BuiltinType::Char_U ||
               ToPointeeType->getKind() == BuiltinType::Char_S))))
          return true;
      }

  return false;
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType. Returns true if there was an
/// error, false otherwise. The expression From is replaced with the
/// converted expression. Flavor is the kind of conversion we're
/// performing, used in the error message. If @p AllowExplicit,
/// explicit user-defined conversions are permitted. @p Elidable should be true
/// when called for copies which may be elided (C++ 12.8p15). C++0x overload
/// resolution works differently in that case.
bool
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const char *Flavor, bool AllowExplicit,
                                bool Elidable)
{
  ImplicitConversionSequence ICS;
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
  if (Elidable && getLangOptions().CPlusPlus0x) {
    ICS = TryImplicitConversion(From, ToType, /*SuppressUserConversions*/false,
                                AllowExplicit, /*ForceRValue*/true);
  }
  if (ICS.ConversionKind == ImplicitConversionSequence::BadConversion) {
    ICS = TryImplicitConversion(From, ToType, false, AllowExplicit);
  }
  return PerformImplicitConversion(From, ToType, ICS, Flavor);
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType using the pre-computed implicit
/// conversion sequence ICS. Returns true if there was an error, false
/// otherwise. The expression From is replaced with the converted
/// expression. Flavor is the kind of conversion we're performing,
/// used in the error message.
bool
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const ImplicitConversionSequence &ICS,
                                const char* Flavor) {
  switch (ICS.ConversionKind) {
  case ImplicitConversionSequence::StandardConversion:
    if (PerformImplicitConversion(From, ToType, ICS.Standard, Flavor))
      return true;
    break;

  case ImplicitConversionSequence::UserDefinedConversion:
    // FIXME: This is, of course, wrong. We'll need to actually call the
    // constructor or conversion operator, and then cope with the standard
    // conversions.
    ImpCastExprToType(From, ToType.getNonReferenceType(), 
                      CastExpr::CK_Unknown,
                      ToType->isLValueReferenceType());
    return false;

  case ImplicitConversionSequence::EllipsisConversion:
    assert(false && "Cannot perform an ellipsis conversion");
    return false;

  case ImplicitConversionSequence::BadConversion:
    return true;
  }

  // Everything went well.
  return false;
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType by following the standard
/// conversion sequence SCS. Returns true if there was an error, false
/// otherwise. The expression From is replaced with the converted
/// expression. Flavor is the context in which we're performing this
/// conversion, for use in error messages.
bool 
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const StandardConversionSequence& SCS,
                                const char *Flavor) {
  // Overall FIXME: we are recomputing too many types here and doing far too
  // much extra work. What this means is that we need to keep track of more
  // information that is computed when we try the implicit conversion initially,
  // so that we don't need to recompute anything here.
  QualType FromType = From->getType();

  if (SCS.CopyConstructor) {
    // FIXME: When can ToType be a reference type?
    assert(!ToType->isReferenceType());
    
    From = BuildCXXConstructExpr(ToType, SCS.CopyConstructor, &From, 1);
    return false;
  }

  // Perform the first implicit conversion.
  switch (SCS.First) {
  case ICK_Identity:
  case ICK_Lvalue_To_Rvalue:
    // Nothing to do.
    break;

  case ICK_Array_To_Pointer:
    FromType = Context.getArrayDecayedType(FromType);
    ImpCastExprToType(From, FromType, CastExpr::CK_ArrayToPointerDecay);
    break;

  case ICK_Function_To_Pointer:
    if (Context.getCanonicalType(FromType) == Context.OverloadTy) {
      FunctionDecl *Fn = ResolveAddressOfOverloadedFunction(From, ToType, true);
      if (!Fn)
        return true;

      if (DiagnoseUseOfDecl(Fn, From->getSourceRange().getBegin()))
        return true;

      FixOverloadedFunctionReference(From, Fn);
      FromType = From->getType();
    }
    FromType = Context.getPointerType(FromType);
    ImpCastExprToType(From, FromType);
    break;

  default:
    assert(false && "Improper first standard conversion");
    break;
  }

  // Perform the second implicit conversion
  switch (SCS.Second) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Integral_Promotion:
  case ICK_Floating_Promotion:
  case ICK_Complex_Promotion:
  case ICK_Integral_Conversion:
  case ICK_Floating_Conversion:
  case ICK_Complex_Conversion:
  case ICK_Floating_Integral:
  case ICK_Complex_Real:
  case ICK_Compatible_Conversion:
      // FIXME: Go deeper to get the unqualified type!
    FromType = ToType.getUnqualifiedType();
    ImpCastExprToType(From, FromType);
    break;

  case ICK_Pointer_Conversion:
    if (SCS.IncompatibleObjC) {
      // Diagnose incompatible Objective-C conversions
      Diag(From->getSourceRange().getBegin(), 
           diag::ext_typecheck_convert_incompatible_pointer)
        << From->getType() << ToType << Flavor
        << From->getSourceRange();
    }

    if (CheckPointerConversion(From, ToType))
      return true;
    ImpCastExprToType(From, ToType);
    break;

  case ICK_Pointer_Member:
    if (CheckMemberPointerConversion(From, ToType))
      return true;
    ImpCastExprToType(From, ToType);
    break;

  case ICK_Boolean_Conversion:
    FromType = Context.BoolTy;
    ImpCastExprToType(From, FromType);
    break;

  default:
    assert(false && "Improper second standard conversion");
    break;
  }

  switch (SCS.Third) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Qualification:
    // FIXME: Not sure about lvalue vs rvalue here in the presence of rvalue
    // references.
    ImpCastExprToType(From, ToType.getNonReferenceType(), 
                      CastExpr::CK_Unknown,
                      ToType->isLValueReferenceType());
    break;

  default:
    assert(false && "Improper second standard conversion");
    break;
  }

  return false;
}

Sema::OwningExprResult Sema::ActOnUnaryTypeTrait(UnaryTypeTrait OTT,
                                                 SourceLocation KWLoc,
                                                 SourceLocation LParen,
                                                 TypeTy *Ty,
                                                 SourceLocation RParen) {
  QualType T = GetTypeFromParser(Ty);
  
  // According to http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html
  // all traits except __is_class, __is_enum and __is_union require a the type
  // to be complete.
  if (OTT != UTT_IsClass && OTT != UTT_IsEnum && OTT != UTT_IsUnion) {
    if (RequireCompleteType(KWLoc, T, 
                            diag::err_incomplete_type_used_in_type_trait_expr,
                            SourceRange(), SourceRange(), T))
      return ExprError();
  }

  // There is no point in eagerly computing the value. The traits are designed
  // to be used from type trait templates, so Ty will be a template parameter
  // 99% of the time.
  return Owned(new (Context) UnaryTypeTraitExpr(KWLoc, OTT, T,
                                                RParen, Context.BoolTy));
}

QualType Sema::CheckPointerToMemberOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isIndirect)
{
  const char *OpSpelling = isIndirect ? "->*" : ".*";
  // C++ 5.5p2
  //   The binary operator .* [p3: ->*] binds its second operand, which shall
  //   be of type "pointer to member of T" (where T is a completely-defined
  //   class type) [...]
  QualType RType = rex->getType();
  const MemberPointerType *MemPtr = RType->getAs<MemberPointerType>();
  if (!MemPtr) {
    Diag(Loc, diag::err_bad_memptr_rhs)
      << OpSpelling << RType << rex->getSourceRange();
    return QualType();
  } 

  QualType Class(MemPtr->getClass(), 0);

  // C++ 5.5p2
  //   [...] to its first operand, which shall be of class T or of a class of
  //   which T is an unambiguous and accessible base class. [p3: a pointer to
  //   such a class]
  QualType LType = lex->getType();
  if (isIndirect) {
    if (const PointerType *Ptr = LType->getAs<PointerType>())
      LType = Ptr->getPointeeType().getNonReferenceType();
    else {
      Diag(Loc, diag::err_bad_memptr_lhs)
        << OpSpelling << 1 << LType << lex->getSourceRange();
      return QualType();
    }
  }

  if (Context.getCanonicalType(Class).getUnqualifiedType() !=
      Context.getCanonicalType(LType).getUnqualifiedType()) {
    BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                    /*DetectVirtual=*/false);
    // FIXME: Would it be useful to print full ambiguity paths, or is that
    // overkill?
    if (!IsDerivedFrom(LType, Class, Paths) ||
        Paths.isAmbiguous(Context.getCanonicalType(Class))) {
      Diag(Loc, diag::err_bad_memptr_lhs) << OpSpelling
        << (int)isIndirect << lex->getType() << lex->getSourceRange();
      return QualType();
    }
  }

  // C++ 5.5p2
  //   The result is an object or a function of the type specified by the
  //   second operand.
  // The cv qualifiers are the union of those in the pointer and the left side,
  // in accordance with 5.5p5 and 5.2.5.
  // FIXME: This returns a dereferenced member function pointer as a normal
  // function type. However, the only operation valid on such functions is
  // calling them. There's also a GCC extension to get a function pointer to the
  // thing, which is another complication, because this type - unlike the type
  // that is the result of this expression - takes the class as the first
  // argument.
  // We probably need a "MemberFunctionClosureType" or something like that.
  QualType Result = MemPtr->getPointeeType();
  if (LType.isConstQualified())
    Result.addConst();
  if (LType.isVolatileQualified())
    Result.addVolatile();
  return Result;
}

/// \brief Get the target type of a standard or user-defined conversion.
static QualType TargetType(const ImplicitConversionSequence &ICS) {
  assert((ICS.ConversionKind ==
              ImplicitConversionSequence::StandardConversion ||
          ICS.ConversionKind ==
              ImplicitConversionSequence::UserDefinedConversion) &&
         "function only valid for standard or user-defined conversions");
  if (ICS.ConversionKind == ImplicitConversionSequence::StandardConversion)
    return QualType::getFromOpaquePtr(ICS.Standard.ToTypePtr);
  return QualType::getFromOpaquePtr(ICS.UserDefined.After.ToTypePtr);
}

/// \brief Try to convert a type to another according to C++0x 5.16p3.
///
/// This is part of the parameter validation for the ? operator. If either
/// value operand is a class type, the two operands are attempted to be
/// converted to each other. This function does the conversion in one direction.
/// It emits a diagnostic and returns true only if it finds an ambiguous
/// conversion.
static bool TryClassUnification(Sema &Self, Expr *From, Expr *To,
                                SourceLocation QuestionLoc,
                                ImplicitConversionSequence &ICS)
{
  // C++0x 5.16p3
  //   The process for determining whether an operand expression E1 of type T1
  //   can be converted to match an operand expression E2 of type T2 is defined
  //   as follows:
  //   -- If E2 is an lvalue:
  if (To->isLvalue(Self.Context) == Expr::LV_Valid) {
    //   E1 can be converted to match E2 if E1 can be implicitly converted to
    //   type "lvalue reference to T2", subject to the constraint that in the
    //   conversion the reference must bind directly to E1.
    if (!Self.CheckReferenceInit(From,
                            Self.Context.getLValueReferenceType(To->getType()),
                            &ICS))
    {
      assert((ICS.ConversionKind ==
                  ImplicitConversionSequence::StandardConversion ||
              ICS.ConversionKind ==
                  ImplicitConversionSequence::UserDefinedConversion) &&
             "expected a definite conversion");
      bool DirectBinding =
        ICS.ConversionKind == ImplicitConversionSequence::StandardConversion ?
        ICS.Standard.DirectBinding : ICS.UserDefined.After.DirectBinding;
      if (DirectBinding)
        return false;
    }
  }
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
  //   -- If E2 is an rvalue, or if the conversion above cannot be done:
  //      -- if E1 and E2 have class type, and the underlying class types are
  //         the same or one is a base class of the other:
  QualType FTy = From->getType();
  QualType TTy = To->getType();
  const RecordType *FRec = FTy->getAs<RecordType>();
  const RecordType *TRec = TTy->getAs<RecordType>();
  bool FDerivedFromT = FRec && TRec && Self.IsDerivedFrom(FTy, TTy);
  if (FRec && TRec && (FRec == TRec ||
        FDerivedFromT || Self.IsDerivedFrom(TTy, FTy))) {
    //         E1 can be converted to match E2 if the class of T2 is the
    //         same type as, or a base class of, the class of T1, and
    //         [cv2 > cv1].
    if ((FRec == TRec || FDerivedFromT) && TTy.isAtLeastAsQualifiedAs(FTy)) {
      // Could still fail if there's no copy constructor.
      // FIXME: Is this a hard error then, or just a conversion failure? The
      // standard doesn't say.
      ICS = Self.TryCopyInitialization(From, TTy);
    }
  } else {
    //     -- Otherwise: E1 can be converted to match E2 if E1 can be
    //        implicitly converted to the type that expression E2 would have
    //        if E2 were converted to an rvalue.
    // First find the decayed type.
    if (TTy->isFunctionType())
      TTy = Self.Context.getPointerType(TTy);
    else if(TTy->isArrayType())
      TTy = Self.Context.getArrayDecayedType(TTy);

    // Now try the implicit conversion.
    // FIXME: This doesn't detect ambiguities.
    ICS = Self.TryImplicitConversion(From, TTy);
  }
  return false;
}

/// \brief Try to find a common type for two according to C++0x 5.16p5.
///
/// This is part of the parameter validation for the ? operator. If either
/// value operand is a class type, overload resolution is used to find a
/// conversion to a common type.
static bool FindConditionalOverload(Sema &Self, Expr *&LHS, Expr *&RHS,
                                    SourceLocation Loc) {
  Expr *Args[2] = { LHS, RHS };
  OverloadCandidateSet CandidateSet;
  Self.AddBuiltinOperatorCandidates(OO_Conditional, Args, 2, CandidateSet);

  OverloadCandidateSet::iterator Best;
  switch (Self.BestViableFunction(CandidateSet, Loc, Best)) {
    case Sema::OR_Success:
      // We found a match. Perform the conversions on the arguments and move on.
      if (Self.PerformImplicitConversion(LHS, Best->BuiltinTypes.ParamTypes[0],
                                         Best->Conversions[0], "converting") ||
          Self.PerformImplicitConversion(RHS, Best->BuiltinTypes.ParamTypes[1],
                                         Best->Conversions[1], "converting"))
        break;
      return false;

    case Sema::OR_No_Viable_Function:
      Self.Diag(Loc, diag::err_typecheck_cond_incompatible_operands)
        << LHS->getType() << RHS->getType()
        << LHS->getSourceRange() << RHS->getSourceRange();
      return true;

    case Sema::OR_Ambiguous:
      Self.Diag(Loc, diag::err_conditional_ambiguous_ovl)
        << LHS->getType() << RHS->getType()
        << LHS->getSourceRange() << RHS->getSourceRange();
      // FIXME: Print the possible common types by printing the return types of
      // the viable candidates.
      break;

    case Sema::OR_Deleted:
      assert(false && "Conditional operator has only built-in overloads");
      break;
  }
  return true;
}

/// \brief Perform an "extended" implicit conversion as returned by
/// TryClassUnification.
///
/// TryClassUnification generates ICSs that include reference bindings.
/// PerformImplicitConversion is not suitable for this; it chokes if the
/// second part of a standard conversion is ICK_DerivedToBase. This function
/// handles the reference binding specially.
static bool ConvertForConditional(Sema &Self, Expr *&E,
                                  const ImplicitConversionSequence &ICS)
{
  if (ICS.ConversionKind == ImplicitConversionSequence::StandardConversion &&
      ICS.Standard.ReferenceBinding) {
    assert(ICS.Standard.DirectBinding &&
           "TryClassUnification should never generate indirect ref bindings");
    // FIXME: CheckReferenceInit should be able to reuse the ICS instead of
    // redoing all the work.
    return Self.CheckReferenceInit(E, Self.Context.getLValueReferenceType(
                                        TargetType(ICS)));
  }
  if (ICS.ConversionKind == ImplicitConversionSequence::UserDefinedConversion &&
      ICS.UserDefined.After.ReferenceBinding) {
    assert(ICS.UserDefined.After.DirectBinding &&
           "TryClassUnification should never generate indirect ref bindings");
    return Self.CheckReferenceInit(E, Self.Context.getLValueReferenceType(
                                        TargetType(ICS)));
  }
  if (Self.PerformImplicitConversion(E, TargetType(ICS), ICS, "converting"))
    return true;
  return false;
}

/// \brief Check the operands of ?: under C++ semantics.
///
/// See C++ [expr.cond]. Note that LHS is never null, even for the GNU x ?: y
/// extension. In this case, LHS == Cond. (But they're not aliases.)
QualType Sema::CXXCheckConditionalOperands(Expr *&Cond, Expr *&LHS, Expr *&RHS,
                                           SourceLocation QuestionLoc) {
  // FIXME: Handle C99's complex types, vector types, block pointers and Obj-C++
  // interface pointers.

  // C++0x 5.16p1
  //   The first expression is contextually converted to bool.
  if (!Cond->isTypeDependent()) {
    if (CheckCXXBooleanCondition(Cond))
      return QualType();
  }

  // Either of the arguments dependent?
  if (LHS->isTypeDependent() || RHS->isTypeDependent())
    return Context.DependentTy;

  // C++0x 5.16p2
  //   If either the second or the third operand has type (cv) void, ...
  QualType LTy = LHS->getType();
  QualType RTy = RHS->getType();
  bool LVoid = LTy->isVoidType();
  bool RVoid = RTy->isVoidType();
  if (LVoid || RVoid) {
    //   ... then the [l2r] conversions are performed on the second and third
    //   operands ...
    DefaultFunctionArrayConversion(LHS);
    DefaultFunctionArrayConversion(RHS);
    LTy = LHS->getType();
    RTy = RHS->getType();

    //   ... and one of the following shall hold:
    //   -- The second or the third operand (but not both) is a throw-
    //      expression; the result is of the type of the other and is an rvalue.
    bool LThrow = isa<CXXThrowExpr>(LHS);
    bool RThrow = isa<CXXThrowExpr>(RHS);
    if (LThrow && !RThrow)
      return RTy;
    if (RThrow && !LThrow)
      return LTy;

    //   -- Both the second and third operands have type void; the result is of
    //      type void and is an rvalue.
    if (LVoid && RVoid)
      return Context.VoidTy;

    // Neither holds, error.
    Diag(QuestionLoc, diag::err_conditional_void_nonvoid)
      << (LVoid ? RTy : LTy) << (LVoid ? 0 : 1)
      << LHS->getSourceRange() << RHS->getSourceRange();
    return QualType();
  }

  // Neither is void.

  // C++0x 5.16p3
  //   Otherwise, if the second and third operand have different types, and
  //   either has (cv) class type, and attempt is made to convert each of those
  //   operands to the other.
  if (Context.getCanonicalType(LTy) != Context.getCanonicalType(RTy) &&
      (LTy->isRecordType() || RTy->isRecordType())) {
    ImplicitConversionSequence ICSLeftToRight, ICSRightToLeft;
    // These return true if a single direction is already ambiguous.
    if (TryClassUnification(*this, LHS, RHS, QuestionLoc, ICSLeftToRight))
      return QualType();
    if (TryClassUnification(*this, RHS, LHS, QuestionLoc, ICSRightToLeft))
      return QualType();

    bool HaveL2R = ICSLeftToRight.ConversionKind !=
      ImplicitConversionSequence::BadConversion;
    bool HaveR2L = ICSRightToLeft.ConversionKind !=
      ImplicitConversionSequence::BadConversion;
    //   If both can be converted, [...] the program is ill-formed.
    if (HaveL2R && HaveR2L) {
      Diag(QuestionLoc, diag::err_conditional_ambiguous)
        << LTy << RTy << LHS->getSourceRange() << RHS->getSourceRange();
      return QualType();
    }

    //   If exactly one conversion is possible, that conversion is applied to
    //   the chosen operand and the converted operands are used in place of the
    //   original operands for the remainder of this section.
    if (HaveL2R) {
      if (ConvertForConditional(*this, LHS, ICSLeftToRight))
        return QualType();
      LTy = LHS->getType();
    } else if (HaveR2L) {
      if (ConvertForConditional(*this, RHS, ICSRightToLeft))
        return QualType();
      RTy = RHS->getType();
    }
  }

  // C++0x 5.16p4
  //   If the second and third operands are lvalues and have the same type,
  //   the result is of that type [...]
  bool Same = Context.getCanonicalType(LTy) == Context.getCanonicalType(RTy);
  if (Same && LHS->isLvalue(Context) == Expr::LV_Valid &&
      RHS->isLvalue(Context) == Expr::LV_Valid)
    return LTy;

  // C++0x 5.16p5
  //   Otherwise, the result is an rvalue. If the second and third operands
  //   do not have the same type, and either has (cv) class type, ...
  if (!Same && (LTy->isRecordType() || RTy->isRecordType())) {
    //   ... overload resolution is used to determine the conversions (if any)
    //   to be applied to the operands. If the overload resolution fails, the
    //   program is ill-formed.
    if (FindConditionalOverload(*this, LHS, RHS, QuestionLoc))
      return QualType();
  }

  // C++0x 5.16p6
  //   LValue-to-rvalue, array-to-pointer, and function-to-pointer standard
  //   conversions are performed on the second and third operands.
  DefaultFunctionArrayConversion(LHS);
  DefaultFunctionArrayConversion(RHS);
  LTy = LHS->getType();
  RTy = RHS->getType();

  //   After those conversions, one of the following shall hold:
  //   -- The second and third operands have the same type; the result
  //      is of that type.
  if (Context.getCanonicalType(LTy) == Context.getCanonicalType(RTy))
    return LTy;

  //   -- The second and third operands have arithmetic or enumeration type;
  //      the usual arithmetic conversions are performed to bring them to a
  //      common type, and the result is of that type.
  if (LTy->isArithmeticType() && RTy->isArithmeticType()) {
    UsualArithmeticConversions(LHS, RHS);
    return LHS->getType();
  }

  //   -- The second and third operands have pointer type, or one has pointer
  //      type and the other is a null pointer constant; pointer conversions
  //      and qualification conversions are performed to bring them to their
  //      composite pointer type. The result is of the composite pointer type.
  QualType Composite = FindCompositePointerType(LHS, RHS);
  if (!Composite.isNull())
    return Composite;

  // Fourth bullet is same for pointers-to-member. However, the possible
  // conversions are far more limited: we have null-to-pointer, upcast of
  // containing class, and second-level cv-ness.
  // cv-ness is not a union, but must match one of the two operands. (Which,
  // frankly, is stupid.)
  const MemberPointerType *LMemPtr = LTy->getAs<MemberPointerType>();
  const MemberPointerType *RMemPtr = RTy->getAs<MemberPointerType>();
  if (LMemPtr && RHS->isNullPointerConstant(Context)) {
    ImpCastExprToType(RHS, LTy);
    return LTy;
  }
  if (RMemPtr && LHS->isNullPointerConstant(Context)) {
    ImpCastExprToType(LHS, RTy);
    return RTy;
  }
  if (LMemPtr && RMemPtr) {
    QualType LPointee = LMemPtr->getPointeeType();
    QualType RPointee = RMemPtr->getPointeeType();
    // First, we check that the unqualified pointee type is the same. If it's
    // not, there's no conversion that will unify the two pointers.
    if (Context.getCanonicalType(LPointee).getUnqualifiedType() ==
        Context.getCanonicalType(RPointee).getUnqualifiedType()) {
      // Second, we take the greater of the two cv qualifications. If neither
      // is greater than the other, the conversion is not possible.
      unsigned Q = LPointee.getCVRQualifiers() | RPointee.getCVRQualifiers();
      if (Q == LPointee.getCVRQualifiers() || Q == RPointee.getCVRQualifiers()){
        // Third, we check if either of the container classes is derived from
        // the other.
        QualType LContainer(LMemPtr->getClass(), 0);
        QualType RContainer(RMemPtr->getClass(), 0);
        QualType MoreDerived;
        if (Context.getCanonicalType(LContainer) ==
            Context.getCanonicalType(RContainer))
          MoreDerived = LContainer;
        else if (IsDerivedFrom(LContainer, RContainer))
          MoreDerived = LContainer;
        else if (IsDerivedFrom(RContainer, LContainer))
          MoreDerived = RContainer;

        if (!MoreDerived.isNull()) {
          // The type 'Q Pointee (MoreDerived::*)' is the common type.
          // We don't use ImpCastExprToType here because this could still fail
          // for ambiguous or inaccessible conversions.
          QualType Common = Context.getMemberPointerType(
            LPointee.getQualifiedType(Q), MoreDerived.getTypePtr());
          if (PerformImplicitConversion(LHS, Common, "converting"))
            return QualType();
          if (PerformImplicitConversion(RHS, Common, "converting"))
            return QualType();
          return Common;
        }
      }
    }
  }

  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHS->getType() << RHS->getType()
    << LHS->getSourceRange() << RHS->getSourceRange();
  return QualType();
}

/// \brief Find a merged pointer type and convert the two expressions to it.
///
/// This finds the composite pointer type for @p E1 and @p E2 according to
/// C++0x 5.9p2. It converts both expressions to this type and returns it.
/// It does not emit diagnostics.
QualType Sema::FindCompositePointerType(Expr *&E1, Expr *&E2) {
  assert(getLangOptions().CPlusPlus && "This function assumes C++");
  QualType T1 = E1->getType(), T2 = E2->getType();
  if(!T1->isAnyPointerType() && !T2->isAnyPointerType())
    return QualType();

  // C++0x 5.9p2
  //   Pointer conversions and qualification conversions are performed on
  //   pointer operands to bring them to their composite pointer type. If
  //   one operand is a null pointer constant, the composite pointer type is
  //   the type of the other operand.
  if (E1->isNullPointerConstant(Context)) {
    ImpCastExprToType(E1, T2);
    return T2;
  }
  if (E2->isNullPointerConstant(Context)) {
    ImpCastExprToType(E2, T1);
    return T1;
  }
  // Now both have to be pointers.
  if(!T1->isPointerType() || !T2->isPointerType())
    return QualType();

  //   Otherwise, of one of the operands has type "pointer to cv1 void," then
  //   the other has type "pointer to cv2 T" and the composite pointer type is
  //   "pointer to cv12 void," where cv12 is the union of cv1 and cv2.
  //   Otherwise, the composite pointer type is a pointer type similar to the
  //   type of one of the operands, with a cv-qualification signature that is
  //   the union of the cv-qualification signatures of the operand types.
  // In practice, the first part here is redundant; it's subsumed by the second.
  // What we do here is, we build the two possible composite types, and try the
  // conversions in both directions. If only one works, or if the two composite
  // types are the same, we have succeeded.
  llvm::SmallVector<unsigned, 4> QualifierUnion;
  QualType Composite1 = T1, Composite2 = T2;
  const PointerType *Ptr1, *Ptr2;
  while ((Ptr1 = Composite1->getAs<PointerType>()) &&
         (Ptr2 = Composite2->getAs<PointerType>())) {
    Composite1 = Ptr1->getPointeeType();
    Composite2 = Ptr2->getPointeeType();
    QualifierUnion.push_back(
      Composite1.getCVRQualifiers() | Composite2.getCVRQualifiers());
  }
  // Rewrap the composites as pointers with the union CVRs.
  for (llvm::SmallVector<unsigned, 4>::iterator I = QualifierUnion.begin(),
       E = QualifierUnion.end(); I != E; ++I) {
    Composite1 = Context.getPointerType(Composite1.getQualifiedType(*I));
    Composite2 = Context.getPointerType(Composite2.getQualifiedType(*I));
  }

  ImplicitConversionSequence E1ToC1 = TryImplicitConversion(E1, Composite1);
  ImplicitConversionSequence E2ToC1 = TryImplicitConversion(E2, Composite1);
  ImplicitConversionSequence E1ToC2, E2ToC2;
  E1ToC2.ConversionKind = ImplicitConversionSequence::BadConversion;
  E2ToC2.ConversionKind = ImplicitConversionSequence::BadConversion;
  if (Context.getCanonicalType(Composite1) !=
      Context.getCanonicalType(Composite2)) {
    E1ToC2 = TryImplicitConversion(E1, Composite2);
    E2ToC2 = TryImplicitConversion(E2, Composite2);
  }

  bool ToC1Viable = E1ToC1.ConversionKind !=
                      ImplicitConversionSequence::BadConversion
                 && E2ToC1.ConversionKind !=
                      ImplicitConversionSequence::BadConversion;
  bool ToC2Viable = E1ToC2.ConversionKind !=
                      ImplicitConversionSequence::BadConversion
                 && E2ToC2.ConversionKind !=
                      ImplicitConversionSequence::BadConversion;
  if (ToC1Viable && !ToC2Viable) {
    if (!PerformImplicitConversion(E1, Composite1, E1ToC1, "converting") &&
        !PerformImplicitConversion(E2, Composite1, E2ToC1, "converting"))
      return Composite1;
  }
  if (ToC2Viable && !ToC1Viable) {
    if (!PerformImplicitConversion(E1, Composite2, E1ToC2, "converting") &&
        !PerformImplicitConversion(E2, Composite2, E2ToC2, "converting"))
      return Composite2;
  }
  return QualType();
}

Sema::OwningExprResult Sema::MaybeBindToTemporary(Expr *E) {
  if (!Context.getLangOptions().CPlusPlus)
    return Owned(E);
  
  const RecordType *RT = E->getType()->getAs<RecordType>();
  if (!RT)
    return Owned(E);
  
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
  if (RD->hasTrivialDestructor())
    return Owned(E);
  
  CXXTemporary *Temp = CXXTemporary::Create(Context, 
                                            RD->getDestructor(Context));
  ExprTemporaries.push_back(Temp);
  if (CXXDestructorDecl *Destructor =
        const_cast<CXXDestructorDecl*>(RD->getDestructor(Context)))
    MarkDeclarationReferenced(E->getExprLoc(), Destructor);
  // FIXME: Add the temporary to the temporaries vector.
  return Owned(CXXBindTemporaryExpr::Create(Context, Temp, E));
}

Expr *Sema::MaybeCreateCXXExprWithTemporaries(Expr *SubExpr, 
                                              bool ShouldDestroyTemps) {
  assert(SubExpr && "sub expression can't be null!");
  
  if (ExprTemporaries.empty())
    return SubExpr;
  
  Expr *E = CXXExprWithTemporaries::Create(Context, SubExpr,
                                           &ExprTemporaries[0], 
                                           ExprTemporaries.size(),
                                           ShouldDestroyTemps);
  ExprTemporaries.clear();
  
  return E;
}

Sema::OwningExprResult Sema::ActOnFinishFullExpr(ExprArg Arg) {
  Expr *FullExpr = Arg.takeAs<Expr>();
  if (FullExpr)
    FullExpr = MaybeCreateCXXExprWithTemporaries(FullExpr, 
                                                 /*ShouldDestroyTemps=*/true);

  return Owned(FullExpr);
}
