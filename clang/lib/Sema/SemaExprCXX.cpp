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

#include "Sema.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
using namespace clang;

/// ActOnCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
Action::ExprResult
Sema::ActOnCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                    SourceLocation LAngleBracketLoc, TypeTy *Ty,
                    SourceLocation RAngleBracketLoc,
                    SourceLocation LParenLoc, ExprTy *E,
                    SourceLocation RParenLoc) {
  CXXCastExpr::Opcode Op;

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!");
  case tok::kw_const_cast:       Op = CXXCastExpr::ConstCast;       break;
  case tok::kw_dynamic_cast:     Op = CXXCastExpr::DynamicCast;     break;
  case tok::kw_reinterpret_cast: Op = CXXCastExpr::ReinterpretCast; break;
  case tok::kw_static_cast:      Op = CXXCastExpr::StaticCast;      break;
  }
  
  return new CXXCastExpr(Op, QualType::getFromOpaquePtr(Ty), (Expr*)E, OpLoc);
}

/// ActOnCXXBoolLiteral - Parse {true,false} literals.
Action::ExprResult
Sema::ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind != tok::kw_true || Kind != tok::kw_false) &&
         "Unknown C++ Boolean value!");
  return new CXXBoolLiteralExpr(Kind == tok::kw_true, Context.BoolTy, OpLoc);
}

/// ActOnCXXThrow - Parse throw expressions.
Action::ExprResult
Sema::ActOnCXXThrow(SourceLocation OpLoc, ExprTy *E) {
  return new CXXThrowExpr((Expr*)E, Context.VoidTy, OpLoc);
}

Action::ExprResult Sema::ActOnCXXThis(SourceLocation ThisLoc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  if (!isa<FunctionDecl>(CurContext)) {
    Diag(ThisLoc, diag::err_invalid_this_use);
    return ExprResult(true);
  }

  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext))
    if (MD->isInstance())
      return new PredefinedExpr(ThisLoc, MD->getThisType(Context),
                                PredefinedExpr::CXXThis);

  return Diag(ThisLoc, diag::err_invalid_this_use);
}

/// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
Action::ExprResult
Sema::ActOnCXXTypeConstructExpr(SourceRange TypeRange, TypeTy *TypeRep,
                                SourceLocation LParenLoc,
                                ExprTy **ExprTys, unsigned NumExprs,
                                SourceLocation *CommaLocs,
                                SourceLocation RParenLoc) {
  assert(TypeRep && "Missing type!");
  QualType Ty = QualType::getFromOpaquePtr(TypeRep);
  Expr **Exprs = (Expr**)ExprTys;
  SourceLocation TyBeginLoc = TypeRange.getBegin();
  SourceRange FullRange = SourceRange(TyBeginLoc, RParenLoc);

  if (const RecordType *RT = Ty->getAsRecordType()) {
    // C++ 5.2.3p1:
    // If the simple-type-specifier specifies a class type, the class type shall
    // be complete.
    //
    if (!RT->getDecl()->isDefinition())
      return Diag(TyBeginLoc, diag::err_invalid_incomplete_type_use,
                  Ty.getAsString(), FullRange);

    unsigned DiagID = PP.getDiagnostics().getCustomDiagID(Diagnostic::Error,
                                    "class constructors are not supported yet");
    return Diag(TyBeginLoc, DiagID);
  }

  // C++ 5.2.3p1:
  // If the expression list is a single expression, the type conversion
  // expression is equivalent (in definedness, and if defined in meaning) to the
  // corresponding cast expression.
  //
  if (NumExprs == 1) {
    if (CheckCastTypes(TypeRange, Ty, Exprs[0]))
      return true;
    return new CXXFunctionalCastExpr(Ty, TyBeginLoc, Exprs[0], RParenLoc);
  }

  // C++ 5.2.3p1:
  // If the expression list specifies more than a single value, the type shall
  // be a class with a suitably declared constructor.
  //
  if (NumExprs > 1)
    return Diag(CommaLocs[0], diag::err_builtin_func_cast_more_than_one_arg,
                FullRange);

  assert(NumExprs == 0 && "Expected 0 expressions");

  // C++ 5.2.3p2:
  // The expression T(), where T is a simple-type-specifier for a non-array
  // complete object type or the (possibly cv-qualified) void type, creates an
  // rvalue of the specified type, which is value-initialized.
  //
  if (Ty->isArrayType())
    return Diag(TyBeginLoc, diag::err_value_init_for_array_type, FullRange);
  if (Ty->isIncompleteType() && !Ty->isVoidType())
    return Diag(TyBeginLoc, diag::err_invalid_incomplete_type_use,
                Ty.getAsString(), FullRange);

  return new CXXZeroInitValueExpr(Ty, TyBeginLoc, RParenLoc);
}


/// ActOnCXXConditionDeclarationExpr - Parsed a condition declaration of a
/// C++ if/switch/while/for statement.
/// e.g: "if (int x = f()) {...}"
Action::ExprResult
Sema::ActOnCXXConditionDeclarationExpr(Scope *S, SourceLocation StartLoc,
                                       Declarator &D,
                                       SourceLocation EqualLoc,
                                       ExprTy *AssignExprVal) {
  assert(AssignExprVal && "Null assignment expression");

  // C++ 6.4p2:
  // The declarator shall not specify a function or an array.
  // The type-specifier-seq shall not contain typedef and shall not declare a
  // new class or enumeration.

  assert(D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
         "Parser allowed 'typedef' as storage class of condition decl.");

  QualType Ty = GetTypeForDeclarator(D, S);
  
  if (Ty->isFunctionType()) { // The declarator shall not specify a function...
    // We exit without creating a CXXConditionDeclExpr because a FunctionDecl
    // would be created and CXXConditionDeclExpr wants a VarDecl.
    return Diag(StartLoc, diag::err_invalid_use_of_function_type,
                SourceRange(StartLoc, EqualLoc));
  } else if (Ty->isArrayType()) { // ...or an array.
    Diag(StartLoc, diag::err_invalid_use_of_array_type,
         SourceRange(StartLoc, EqualLoc));
  } else if (const RecordType *RT = Ty->getAsRecordType()) {
    RecordDecl *RD = RT->getDecl();
    // The type-specifier-seq shall not declare a new class...
    if (RD->isDefinition() && (RD->getIdentifier() == 0 || S->isDeclScope(RD)))
      Diag(RD->getLocation(), diag::err_type_defined_in_condition);
  } else if (const EnumType *ET = Ty->getAsEnumType()) {
    EnumDecl *ED = ET->getDecl();
    // ...or enumeration.
    if (ED->isDefinition() && (ED->getIdentifier() == 0 || S->isDeclScope(ED)))
      Diag(ED->getLocation(), diag::err_type_defined_in_condition);
  }

  DeclTy *Dcl = ActOnDeclarator(S, D, 0);
  if (!Dcl)
    return true;
  AddInitializerToDecl(Dcl, AssignExprVal);

  return new CXXConditionDeclExpr(StartLoc, EqualLoc,
                                       cast<VarDecl>(static_cast<Decl *>(Dcl)));
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
  QualType Ty = CondExpr->getType(); // Save the type.
  AssignConvertType
    ConvTy = CheckSingleAssignmentConstraints(Context.BoolTy, CondExpr);
  if (ConvTy == Incompatible)
    return Diag(CondExpr->getLocStart(), diag::err_typecheck_bool_condition,
                Ty.getAsString(), CondExpr->getSourceRange());
  return false;
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
    if (const PointerType *ToPtrType = ToType->getAsPointerType())
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
/// converted expression.
bool 
Sema::PerformImplicitConversion(Expr *&From, QualType ToType)
{
  ImplicitConversionSequence ICS = TryCopyInitialization(From, ToType);
  switch (ICS.ConversionKind) {
  case ImplicitConversionSequence::StandardConversion:
    if (PerformImplicitConversion(From, ToType, ICS.Standard))
      return true;
    break;

  case ImplicitConversionSequence::UserDefinedConversion:
    // FIXME: This is, of course, wrong. We'll need to actually call
    // the constructor or conversion operator, and then cope with the
    // standard conversions.
    ImpCastExprToType(From, ToType);
    break;

  case ImplicitConversionSequence::EllipsisConversion:
    assert(false && "Cannot perform an ellipsis conversion");
    break;

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
/// expression.
bool 
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const StandardConversionSequence& SCS)
{
  // Overall FIXME: we are recomputing too many types here and doing
  // far too much extra work. What this means is that we need to keep
  // track of more information that is computed when we try the
  // implicit conversion initially, so that we don't need to recompute
  // anything here.
  QualType FromType = From->getType();

  // Perform the first implicit conversion.
  switch (SCS.First) {
  case ICK_Identity:
  case ICK_Lvalue_To_Rvalue:
    // Nothing to do.
    break;

  case ICK_Array_To_Pointer:
    FromType = Context.getArrayDecayedType(FromType);
    ImpCastExprToType(From, FromType);
    break;

  case ICK_Function_To_Pointer:
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
  case ICK_Integral_Conversion:
  case ICK_Floating_Conversion:
  case ICK_Floating_Integral:
    FromType = ToType.getUnqualifiedType();
    ImpCastExprToType(From, FromType);
    break;

  case ICK_Pointer_Conversion:
    if (CheckPointerConversion(From, ToType))
      return true;
    ImpCastExprToType(From, ToType);
    break;

  case ICK_Pointer_Member:
    // FIXME: Implement pointer-to-member conversions.
    assert(false && "Pointer-to-member conversions are unsupported");
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
    ImpCastExprToType(From, ToType);
    break;

  default:
    assert(false && "Improper second standard conversion");
    break;
  }

  return false;
}

