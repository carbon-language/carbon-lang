//===--- SemaExpr.cpp - Semantic Analysis for Expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
using namespace clang;

//===----------------------------------------------------------------------===//
//  Standard Promotions and Conversions
//===----------------------------------------------------------------------===//

/// DefaultFunctionArrayConversion (C99 6.3.2.1p3, C99 6.3.2.1p4).
void Sema::DefaultFunctionArrayConversion(Expr *&E) {
  QualType Ty = E->getType();
  assert(!Ty.isNull() && "DefaultFunctionArrayConversion - missing type");

  if (const ReferenceType *ref = Ty->getAsReferenceType()) {
    ImpCastExprToType(E, ref->getPointeeType()); // C++ [expr]
    Ty = E->getType();
  }
  if (Ty->isFunctionType())
    ImpCastExprToType(E, Context.getPointerType(Ty));
  else if (Ty->isArrayType()) {
    // In C90 mode, arrays only promote to pointers if the array expression is
    // an lvalue.  The relevant legalese is C90 6.2.2.1p3: "an lvalue that has
    // type 'array of type' is converted to an expression that has type 'pointer
    // to type'...".  In C99 this was changed to: C99 6.3.2.1p3: "an expression
    // that has type 'array of type' ...".  The relevant change is "an lvalue"
    // (C90) to "an expression" (C99).
    if (getLangOptions().C99 || E->isLvalue(Context) == Expr::LV_Valid)
      ImpCastExprToType(E, Context.getArrayDecayedType(Ty));
  }
}

/// UsualUnaryConversions - Performs various conversions that are common to most
/// operators (C99 6.3). The conversions of array and function types are 
/// sometimes surpressed. For example, the array->pointer conversion doesn't
/// apply if the array is an argument to the sizeof or address (&) operators.
/// In these instances, this routine should *not* be called.
Expr *Sema::UsualUnaryConversions(Expr *&Expr) {
  QualType Ty = Expr->getType();
  assert(!Ty.isNull() && "UsualUnaryConversions - missing type");
  
  if (const ReferenceType *Ref = Ty->getAsReferenceType()) {
    ImpCastExprToType(Expr, Ref->getPointeeType()); // C++ [expr]
    Ty = Expr->getType();
  }
  if (Ty->isPromotableIntegerType()) // C99 6.3.1.1p2
    ImpCastExprToType(Expr, Context.IntTy);
  else
    DefaultFunctionArrayConversion(Expr);
  
  return Expr;
}

/// DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
/// do not have a prototype. Arguments that have type float are promoted to 
/// double. All other argument types are converted by UsualUnaryConversions().
void Sema::DefaultArgumentPromotion(Expr *&Expr) {
  QualType Ty = Expr->getType();
  assert(!Ty.isNull() && "DefaultArgumentPromotion - missing type");
  
  // If this is a 'float' (CVR qualified or typedef) promote to double.
  if (const BuiltinType *BT = Ty->getAsBuiltinType())
    if (BT->getKind() == BuiltinType::Float)
      return ImpCastExprToType(Expr, Context.DoubleTy);
  
  UsualUnaryConversions(Expr);
}

/// UsualArithmeticConversions - Performs various conversions that are common to
/// binary operators (C99 6.3.1.8). If both operands aren't arithmetic, this
/// routine returns the first non-arithmetic type found. The client is 
/// responsible for emitting appropriate error diagnostics.
/// FIXME: verify the conversion rules for "complex int" are consistent with
/// GCC.
QualType Sema::UsualArithmeticConversions(Expr *&lhsExpr, Expr *&rhsExpr,
                                          bool isCompAssign) {
  if (!isCompAssign) {
    UsualUnaryConversions(lhsExpr);
    UsualUnaryConversions(rhsExpr);
  }
  // For conversion purposes, we ignore any qualifiers. 
  // For example, "const float" and "float" are equivalent.
  QualType lhs =
    Context.getCanonicalType(lhsExpr->getType()).getUnqualifiedType();
  QualType rhs = 
    Context.getCanonicalType(rhsExpr->getType()).getUnqualifiedType();
  
  // If both types are identical, no conversion is needed.
  if (lhs == rhs)
    return lhs;
  
  // If either side is a non-arithmetic type (e.g. a pointer), we are done.
  // The caller can deal with this (e.g. pointer + int).
  if (!lhs->isArithmeticType() || !rhs->isArithmeticType())
    return lhs;
    
  // At this point, we have two different arithmetic types. 
  
  // Handle complex types first (C99 6.3.1.8p1).
  if (lhs->isComplexType() || rhs->isComplexType()) {
    // if we have an integer operand, the result is the complex type.
    if (rhs->isIntegerType() || rhs->isComplexIntegerType()) { 
      // convert the rhs to the lhs complex type.
      if (!isCompAssign) ImpCastExprToType(rhsExpr, lhs);
      return lhs;
    }
    if (lhs->isIntegerType() || lhs->isComplexIntegerType()) { 
      // convert the lhs to the rhs complex type.
      if (!isCompAssign) ImpCastExprToType(lhsExpr, rhs);
      return rhs;
    }
    // This handles complex/complex, complex/float, or float/complex.
    // When both operands are complex, the shorter operand is converted to the 
    // type of the longer, and that is the type of the result. This corresponds 
    // to what is done when combining two real floating-point operands. 
    // The fun begins when size promotion occur across type domains. 
    // From H&S 6.3.4: When one operand is complex and the other is a real
    // floating-point type, the less precise type is converted, within it's 
    // real or complex domain, to the precision of the other type. For example,
    // when combining a "long double" with a "double _Complex", the 
    // "double _Complex" is promoted to "long double _Complex".
    int result = Context.getFloatingTypeOrder(lhs, rhs);
    
    if (result > 0) { // The left side is bigger, convert rhs. 
      rhs = Context.getFloatingTypeOfSizeWithinDomain(lhs, rhs);
      if (!isCompAssign)
        ImpCastExprToType(rhsExpr, rhs);
    } else if (result < 0) { // The right side is bigger, convert lhs. 
      lhs = Context.getFloatingTypeOfSizeWithinDomain(rhs, lhs);
      if (!isCompAssign)
        ImpCastExprToType(lhsExpr, lhs);
    } 
    // At this point, lhs and rhs have the same rank/size. Now, make sure the
    // domains match. This is a requirement for our implementation, C99
    // does not require this promotion.
    if (lhs != rhs) { // Domains don't match, we have complex/float mix.
      if (lhs->isRealFloatingType()) { // handle "double, _Complex double".
        if (!isCompAssign)
          ImpCastExprToType(lhsExpr, rhs);
        return rhs;
      } else { // handle "_Complex double, double".
        if (!isCompAssign)
          ImpCastExprToType(rhsExpr, lhs);
        return lhs;
      }
    }
    return lhs; // The domain/size match exactly.
  }
  // Now handle "real" floating types (i.e. float, double, long double).
  if (lhs->isRealFloatingType() || rhs->isRealFloatingType()) {
    // if we have an integer operand, the result is the real floating type.
    if (rhs->isIntegerType() || rhs->isComplexIntegerType()) { 
      // convert rhs to the lhs floating point type.
      if (!isCompAssign) ImpCastExprToType(rhsExpr, lhs);
      return lhs;
    }
    if (lhs->isIntegerType() || lhs->isComplexIntegerType()) { 
      // convert lhs to the rhs floating point type.
      if (!isCompAssign) ImpCastExprToType(lhsExpr, rhs);
      return rhs;
    }
    // We have two real floating types, float/complex combos were handled above.
    // Convert the smaller operand to the bigger result.
    int result = Context.getFloatingTypeOrder(lhs, rhs);
    
    if (result > 0) { // convert the rhs
      if (!isCompAssign) ImpCastExprToType(rhsExpr, lhs);
      return lhs;
    }
    if (result < 0) { // convert the lhs
      if (!isCompAssign) ImpCastExprToType(lhsExpr, rhs); // convert the lhs
      return rhs;
    }
    assert(0 && "Sema::UsualArithmeticConversions(): illegal float comparison");
  }
  if (lhs->isComplexIntegerType() || rhs->isComplexIntegerType()) {
    // Handle GCC complex int extension.
    const ComplexType *lhsComplexInt = lhs->getAsComplexIntegerType();
    const ComplexType *rhsComplexInt = rhs->getAsComplexIntegerType();

    if (lhsComplexInt && rhsComplexInt) {
      if (Context.getIntegerTypeOrder(lhsComplexInt->getElementType(), 
                                      rhsComplexInt->getElementType()) >= 0) {
        // convert the rhs
        if (!isCompAssign) ImpCastExprToType(rhsExpr, lhs);
        return lhs;
      }
      if (!isCompAssign) 
        ImpCastExprToType(lhsExpr, rhs); // convert the lhs
      return rhs;
    } else if (lhsComplexInt && rhs->isIntegerType()) {
      // convert the rhs to the lhs complex type.
      if (!isCompAssign) ImpCastExprToType(rhsExpr, lhs);
      return lhs;
    } else if (rhsComplexInt && lhs->isIntegerType()) {
      // convert the lhs to the rhs complex type.
      if (!isCompAssign) ImpCastExprToType(lhsExpr, rhs);
      return rhs;
    }
  }
  // Finally, we have two differing integer types.
  // The rules for this case are in C99 6.3.1.8
  int compare = Context.getIntegerTypeOrder(lhs, rhs);
  bool lhsSigned = lhs->isSignedIntegerType(),
       rhsSigned = rhs->isSignedIntegerType();
  QualType destType;
  if (lhsSigned == rhsSigned) {
    // Same signedness; use the higher-ranked type
    destType = compare >= 0 ? lhs : rhs;
  } else if (compare != (lhsSigned ? 1 : -1)) {
    // The unsigned type has greater than or equal rank to the
    // signed type, so use the unsigned type
    destType = lhsSigned ? rhs : lhs;
  } else if (Context.getIntWidth(lhs) != Context.getIntWidth(rhs)) {
    // The two types are different widths; if we are here, that
    // means the signed type is larger than the unsigned type, so
    // use the signed type.
    destType = lhsSigned ? lhs : rhs;
  } else {
    // The signed type is higher-ranked than the unsigned type,
    // but isn't actually any bigger (like unsigned int and long
    // on most 32-bit systems).  Use the unsigned type corresponding
    // to the signed type.
    destType = Context.getCorrespondingUnsignedType(lhsSigned ? lhs : rhs);
  }
  if (!isCompAssign) {
    ImpCastExprToType(lhsExpr, destType);
    ImpCastExprToType(rhsExpr, destType);
  }
  return destType;
}

//===----------------------------------------------------------------------===//
//  Semantic Analysis for various Expression Types
//===----------------------------------------------------------------------===//


/// ActOnStringLiteral - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
/// 
Action::ExprResult
Sema::ActOnStringLiteral(const Token *StringToks, unsigned NumStringToks) {
  assert(NumStringToks && "Must have at least one string!");

  StringLiteralParser Literal(StringToks, NumStringToks, PP, Context.Target);
  if (Literal.hadError)
    return ExprResult(true);

  llvm::SmallVector<SourceLocation, 4> StringTokLocs;
  for (unsigned i = 0; i != NumStringToks; ++i)
    StringTokLocs.push_back(StringToks[i].getLocation());

  // Verify that pascal strings aren't too large.
  if (Literal.Pascal && Literal.GetStringLength() > 256)
    return Diag(StringToks[0].getLocation(), diag::err_pascal_string_too_long,
                SourceRange(StringToks[0].getLocation(),
                            StringToks[NumStringToks-1].getLocation()));
  
  QualType StrTy = Context.CharTy;
  if (Literal.AnyWide) StrTy = Context.getWCharType();
  if (Literal.Pascal) StrTy = Context.UnsignedCharTy;
  
  // Get an array type for the string, according to C99 6.4.5.  This includes
  // the nul terminator character as well as the string length for pascal
  // strings.
  StrTy = Context.getConstantArrayType(StrTy,
                                   llvm::APInt(32, Literal.GetStringLength()+1),
                                       ArrayType::Normal, 0);
  
  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  return new StringLiteral(Literal.GetString(), Literal.GetStringLength(), 
                           Literal.AnyWide, StrTy, 
                           StringToks[0].getLocation(),
                           StringToks[NumStringToks-1].getLocation());
}


/// ActOnIdentifierExpr - The parser read an identifier in expression context,
/// validate it per-C99 6.5.1.  HasTrailingLParen indicates whether this
/// identifier is used in a function call context.
Sema::ExprResult Sema::ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                           IdentifierInfo &II,
                                           bool HasTrailingLParen) {
  // Could be enum-constant, value decl, instance variable, etc.
  Decl *D = LookupDecl(&II, Decl::IDNS_Ordinary, S);
  
  // If this reference is in an Objective-C method, then ivar lookup happens as
  // well.
  if (getCurMethodDecl()) {
    ScopedDecl *SD = dyn_cast_or_null<ScopedDecl>(D);
    // There are two cases to handle here.  1) scoped lookup could have failed,
    // in which case we should look for an ivar.  2) scoped lookup could have
    // found a decl, but that decl is outside the current method (i.e. a global
    // variable).  In these two cases, we do a lookup for an ivar with this
    // name, if the lookup suceeds, we replace it our current decl.
    if (SD == 0 || SD->isDefinedOutsideFunctionOrMethod()) {
      ObjCInterfaceDecl *IFace = getCurMethodDecl()->getClassInterface();
      if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(&II)) {
        // FIXME: This should use a new expr for a direct reference, don't turn
        // this into Self->ivar, just return a BareIVarExpr or something.
        IdentifierInfo &II = Context.Idents.get("self");
        ExprResult SelfExpr = ActOnIdentifierExpr(S, Loc, II, false);
        return new ObjCIvarRefExpr(IV, IV->getType(), Loc, 
                                 static_cast<Expr*>(SelfExpr.Val), true, true);
      }
    }
    // Needed to implement property "super.method" notation.
    if (SD == 0 && &II == SuperID) {
      QualType T = Context.getPointerType(Context.getObjCInterfaceType(
                     getCurMethodDecl()->getClassInterface()));
      return new PredefinedExpr(Loc, T, PredefinedExpr::ObjCSuper);
    }
  }
  
  if (D == 0) {
    // Otherwise, this could be an implicitly declared function reference (legal
    // in C90, extension in C99).
    if (HasTrailingLParen &&
        !getLangOptions().CPlusPlus) // Not in C++.
      D = ImplicitlyDefineFunction(Loc, II, S);
    else {
      // If this name wasn't predeclared and if this is not a function call,
      // diagnose the problem.
      return Diag(Loc, diag::err_undeclared_var_use, II.getName());
    }
  }
  
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    // check if referencing an identifier with __attribute__((deprecated)).
    if (VD->getAttr<DeprecatedAttr>())
      Diag(Loc, diag::warn_deprecated, VD->getName());

    // Only create DeclRefExpr's for valid Decl's.
    if (VD->isInvalidDecl())
      return true;
    return new DeclRefExpr(VD, VD->getType(), Loc);
  }

  if (CXXFieldDecl *FD = dyn_cast<CXXFieldDecl>(D)) {
    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
      if (MD->isStatic())
        // "invalid use of member 'x' in static member function"
        return Diag(Loc, diag::err_invalid_member_use_in_static_method,
                    FD->getName());
      if (cast<CXXRecordDecl>(MD->getParent()) != FD->getParent())
        // "invalid use of nonstatic data member 'x'"
        return Diag(Loc, diag::err_invalid_non_static_member_use,
                    FD->getName());

      if (FD->isInvalidDecl())
        return true;

      // FIXME: Use DeclRefExpr or a new Expr for a direct CXXField reference.
      ExprResult ThisExpr = ActOnCXXThis(SourceLocation());
      return new MemberExpr(static_cast<Expr*>(ThisExpr.Val),
                            true, FD, Loc, FD->getType());
    }

    return Diag(Loc, diag::err_invalid_non_static_member_use, FD->getName());
  }
  
  if (isa<TypedefDecl>(D))
    return Diag(Loc, diag::err_unexpected_typedef, II.getName());
  if (isa<ObjCInterfaceDecl>(D))
    return Diag(Loc, diag::err_unexpected_interface, II.getName());
  if (isa<NamespaceDecl>(D))
    return Diag(Loc, diag::err_unexpected_namespace, II.getName());

  assert(0 && "Invalid decl");
  abort();
}

Sema::ExprResult Sema::ActOnPredefinedExpr(SourceLocation Loc,
                                           tok::TokenKind Kind) {
  PredefinedExpr::IdentType IT;
  
  switch (Kind) {
  default: assert(0 && "Unknown simple primary expr!");
  case tok::kw___func__: IT = PredefinedExpr::Func; break; // [C99 6.4.2.2]
  case tok::kw___FUNCTION__: IT = PredefinedExpr::Function; break;
  case tok::kw___PRETTY_FUNCTION__: IT = PredefinedExpr::PrettyFunction; break;
  }

  // Verify that this is in a function context.
  if (getCurFunctionDecl() == 0 && getCurMethodDecl() == 0)
    return Diag(Loc, diag::err_predef_outside_function);
  
  // Pre-defined identifiers are of type char[x], where x is the length of the
  // string.
  unsigned Length;
  if (getCurFunctionDecl())
    Length = getCurFunctionDecl()->getIdentifier()->getLength();
  else
    Length = getCurMethodDecl()->getSynthesizedMethodSize();
  
  llvm::APInt LengthI(32, Length + 1);
  QualType ResTy = Context.CharTy.getQualifiedType(QualType::Const);
  ResTy = Context.getConstantArrayType(ResTy, LengthI, ArrayType::Normal, 0);
  return new PredefinedExpr(Loc, ResTy, IT);
}

Sema::ExprResult Sema::ActOnCharacterConstant(const Token &Tok) {
  llvm::SmallString<16> CharBuffer;
  CharBuffer.resize(Tok.getLength());
  const char *ThisTokBegin = &CharBuffer[0];
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);
  
  CharLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength,
                            Tok.getLocation(), PP);
  if (Literal.hadError())
    return ExprResult(true);

  QualType type = getLangOptions().CPlusPlus ? Context.CharTy : Context.IntTy;

  return new CharacterLiteral(Literal.getValue(), Literal.isWide(), type,
                              Tok.getLocation());
}

Action::ExprResult Sema::ActOnNumericConstant(const Token &Tok) {
  // fast path for a single digit (which is quite common). A single digit 
  // cannot have a trigraph, escaped newline, radix prefix, or type suffix.
  if (Tok.getLength() == 1) {
    const char *Ty = PP.getSourceManager().getCharacterData(Tok.getLocation());
    
    unsigned IntSize =static_cast<unsigned>(Context.getTypeSize(Context.IntTy));
    return ExprResult(new IntegerLiteral(llvm::APInt(IntSize, *Ty-'0'),
                                         Context.IntTy, 
                                         Tok.getLocation()));
  }
  llvm::SmallString<512> IntegerBuffer;
  IntegerBuffer.resize(Tok.getLength());
  const char *ThisTokBegin = &IntegerBuffer[0];
  
  // Get the spelling of the token, which eliminates trigraphs, etc.
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);
  NumericLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength, 
                               Tok.getLocation(), PP);
  if (Literal.hadError)
    return ExprResult(true);
  
  Expr *Res;
  
  if (Literal.isFloatingLiteral()) {
    QualType Ty;
    if (Literal.isFloat)
      Ty = Context.FloatTy;
    else if (!Literal.isLong)
      Ty = Context.DoubleTy;
    else
      Ty = Context.LongDoubleTy;

    const llvm::fltSemantics &Format = Context.getFloatTypeSemantics(Ty);

    // isExact will be set by GetFloatValue().
    bool isExact = false;
    Res = new FloatingLiteral(Literal.GetFloatValue(Format, &isExact), &isExact,
                              Ty, Tok.getLocation());
    
  } else if (!Literal.isIntegerLiteral()) {
    return ExprResult(true);
  } else {
    QualType Ty;

    // long long is a C99 feature.
    if (!getLangOptions().C99 && !getLangOptions().CPlusPlus0x &&
        Literal.isLongLong)
      Diag(Tok.getLocation(), diag::ext_longlong);

    // Get the value in the widest-possible width.
    llvm::APInt ResultVal(Context.Target.getIntMaxTWidth(), 0);
   
    if (Literal.GetIntegerValue(ResultVal)) {
      // If this value didn't fit into uintmax_t, warn and force to ull.
      Diag(Tok.getLocation(), diag::warn_integer_too_large);
      Ty = Context.UnsignedLongLongTy;
      assert(Context.getTypeSize(Ty) == ResultVal.getBitWidth() &&
             "long long is not intmax_t?");
    } else {
      // If this value fits into a ULL, try to figure out what else it fits into
      // according to the rules of C99 6.4.4.1p5.
      
      // Octal, Hexadecimal, and integers with a U suffix are allowed to
      // be an unsigned int.
      bool AllowUnsigned = Literal.isUnsigned || Literal.getRadix() != 10;

      // Check from smallest to largest, picking the smallest type we can.
      unsigned Width = 0;
      if (!Literal.isLong && !Literal.isLongLong) {
        // Are int/unsigned possibilities?
        unsigned IntSize = Context.Target.getIntWidth();
        
        // Does it fit in a unsigned int?
        if (ResultVal.isIntN(IntSize)) {
          // Does it fit in a signed int?
          if (!Literal.isUnsigned && ResultVal[IntSize-1] == 0)
            Ty = Context.IntTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedIntTy;
          Width = IntSize;
        }
      }
      
      // Are long/unsigned long possibilities?
      if (Ty.isNull() && !Literal.isLongLong) {
        unsigned LongSize = Context.Target.getLongWidth();
     
        // Does it fit in a unsigned long?
        if (ResultVal.isIntN(LongSize)) {
          // Does it fit in a signed long?
          if (!Literal.isUnsigned && ResultVal[LongSize-1] == 0)
            Ty = Context.LongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongTy;
          Width = LongSize;
        }
      }      
      
      // Finally, check long long if needed.
      if (Ty.isNull()) {
        unsigned LongLongSize = Context.Target.getLongLongWidth();
        
        // Does it fit in a unsigned long long?
        if (ResultVal.isIntN(LongLongSize)) {
          // Does it fit in a signed long long?
          if (!Literal.isUnsigned && ResultVal[LongLongSize-1] == 0)
            Ty = Context.LongLongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongLongTy;
          Width = LongLongSize;
        }
      }
      
      // If we still couldn't decide a type, we probably have something that
      // does not fit in a signed long long, but has no U suffix.
      if (Ty.isNull()) {
        Diag(Tok.getLocation(), diag::warn_integer_too_large_for_signed);
        Ty = Context.UnsignedLongLongTy;
        Width = Context.Target.getLongLongWidth();
      }
      
      if (ResultVal.getBitWidth() != Width)
        ResultVal.trunc(Width);
    }

    Res = new IntegerLiteral(ResultVal, Ty, Tok.getLocation());
  }
  
  // If this is an imaginary literal, create the ImaginaryLiteral wrapper.
  if (Literal.isImaginary)
    Res = new ImaginaryLiteral(Res, Context.getComplexType(Res->getType()));
  
  return Res;
}

Action::ExprResult Sema::ActOnParenExpr(SourceLocation L, SourceLocation R,
                                        ExprTy *Val) {
  Expr *E = (Expr *)Val;
  assert((E != 0) && "ActOnParenExpr() missing expr");
  return new ParenExpr(L, R, E);
}

/// The UsualUnaryConversions() function is *not* called by this routine.
/// See C99 6.3.2.1p[2-4] for more details.
QualType Sema::CheckSizeOfAlignOfOperand(QualType exprType, 
                                         SourceLocation OpLoc,
                                         const SourceRange &ExprRange,
                                         bool isSizeof) {
  // C99 6.5.3.4p1:
  if (isa<FunctionType>(exprType) && isSizeof)
    // alignof(function) is allowed.
    Diag(OpLoc, diag::ext_sizeof_function_type, ExprRange);
  else if (exprType->isVoidType())
    Diag(OpLoc, diag::ext_sizeof_void_type, isSizeof ? "sizeof" : "__alignof",
         ExprRange);
  else if (exprType->isIncompleteType()) {
    Diag(OpLoc, isSizeof ? diag::err_sizeof_incomplete_type : 
                           diag::err_alignof_incomplete_type,
         exprType.getAsString(), ExprRange);
    return QualType(); // error
  }
  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return Context.getSizeType();
}

Action::ExprResult Sema::
ActOnSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                           SourceLocation LPLoc, TypeTy *Ty,
                           SourceLocation RPLoc) {
  // If error parsing type, ignore.
  if (Ty == 0) return true;
  
  // Verify that this is a valid expression.
  QualType ArgTy = QualType::getFromOpaquePtr(Ty);
  
  QualType resultType =
    CheckSizeOfAlignOfOperand(ArgTy, OpLoc, SourceRange(LPLoc, RPLoc),isSizeof);

  if (resultType.isNull())
    return true;
  return new SizeOfAlignOfTypeExpr(isSizeof, ArgTy, resultType, OpLoc, RPLoc);
}

QualType Sema::CheckRealImagOperand(Expr *&V, SourceLocation Loc) {
  DefaultFunctionArrayConversion(V);
  
  // These operators return the element type of a complex type.
  if (const ComplexType *CT = V->getType()->getAsComplexType())
    return CT->getElementType();
  
  // Otherwise they pass through real integer and floating point types here.
  if (V->getType()->isArithmeticType())
    return V->getType();
  
  // Reject anything else.
  Diag(Loc, diag::err_realimag_invalid_type, V->getType().getAsString());
  return QualType();
}



Action::ExprResult Sema::ActOnPostfixUnaryOp(SourceLocation OpLoc, 
                                             tok::TokenKind Kind,
                                             ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }
  QualType result = CheckIncrementDecrementOperand((Expr *)Input, OpLoc);
  if (result.isNull())
    return true;
  return new UnaryOperator((Expr *)Input, Opc, result, OpLoc);
}

Action::ExprResult Sema::
ActOnArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                        ExprTy *Idx, SourceLocation RLoc) {
  Expr *LHSExp = static_cast<Expr*>(Base), *RHSExp = static_cast<Expr*>(Idx);

  // Perform default conversions.
  DefaultFunctionArrayConversion(LHSExp);
  DefaultFunctionArrayConversion(RHSExp);
  
  QualType LHSTy = LHSExp->getType(), RHSTy = RHSExp->getType();

  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely equivalent
  // to the expression *((e1)+(e2)). This means the array "Base" may actually be
  // in the subscript position. As a result, we need to derive the array base 
  // and index from the expression types.
  Expr *BaseExpr, *IndexExpr;
  QualType ResultType;
  if (const PointerType *PTy = LHSTy->getAsPointerType()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    // FIXME: need to deal with const...
    ResultType = PTy->getPointeeType();
  } else if (const PointerType *PTy = RHSTy->getAsPointerType()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    // FIXME: need to deal with const...
    ResultType = PTy->getPointeeType();
  } else if (const VectorType *VTy = LHSTy->getAsVectorType()) {
    BaseExpr = LHSExp;    // vectors: V[123]
    IndexExpr = RHSExp;
    
    // Component access limited to variables (reject vec4.rg[1]).
    if (!isa<DeclRefExpr>(BaseExpr) && !isa<ArraySubscriptExpr>(BaseExpr) &&
        !isa<ExtVectorElementExpr>(BaseExpr))
      return Diag(LLoc, diag::err_ext_vector_component_access, 
                  SourceRange(LLoc, RLoc));
    // FIXME: need to deal with const...
    ResultType = VTy->getElementType();
  } else {
    return Diag(LHSExp->getLocStart(), diag::err_typecheck_subscript_value, 
                RHSExp->getSourceRange());
  }              
  // C99 6.5.2.1p1
  if (!IndexExpr->getType()->isIntegerType())
    return Diag(IndexExpr->getLocStart(), diag::err_typecheck_subscript,
                IndexExpr->getSourceRange());

  // C99 6.5.2.1p1: "shall have type "pointer to *object* type".  In practice,
  // the following check catches trying to index a pointer to a function (e.g.
  // void (*)(int)) and pointers to incomplete types.  Functions are not
  // objects in C99.
  if (!ResultType->isObjectType())
    return Diag(BaseExpr->getLocStart(), 
                diag::err_typecheck_subscript_not_object,
                BaseExpr->getType().getAsString(), BaseExpr->getSourceRange());

  return new ArraySubscriptExpr(LHSExp, RHSExp, ResultType, RLoc);
}

QualType Sema::
CheckExtVectorComponent(QualType baseType, SourceLocation OpLoc,
                        IdentifierInfo &CompName, SourceLocation CompLoc) {
  const ExtVectorType *vecType = baseType->getAsExtVectorType();

  // This flag determines whether or not the component is to be treated as a 
  // special name, or a regular GLSL-style component access.
  bool SpecialComponent = false;
  
  // The vector accessor can't exceed the number of elements.
  const char *compStr = CompName.getName();
  if (strlen(compStr) > vecType->getNumElements()) {
    Diag(OpLoc, diag::err_ext_vector_component_exceeds_length, 
                baseType.getAsString(), SourceRange(CompLoc));
    return QualType();
  }

  // Check that we've found one of the special components, or that the component
  // names must come from the same set.
  if (!strcmp(compStr, "hi") || !strcmp(compStr, "lo") || 
      !strcmp(compStr, "e") || !strcmp(compStr, "o")) {
    SpecialComponent = true;
  } else if (vecType->getPointAccessorIdx(*compStr) != -1) {
    do
      compStr++;
    while (*compStr && vecType->getPointAccessorIdx(*compStr) != -1);
  } else if (vecType->getColorAccessorIdx(*compStr) != -1) {
    do
      compStr++;
    while (*compStr && vecType->getColorAccessorIdx(*compStr) != -1);
  } else if (vecType->getTextureAccessorIdx(*compStr) != -1) {
    do 
      compStr++;
    while (*compStr && vecType->getTextureAccessorIdx(*compStr) != -1);
  }
    
  if (!SpecialComponent && *compStr) { 
    // We didn't get to the end of the string. This means the component names
    // didn't come from the same set *or* we encountered an illegal name.
    Diag(OpLoc, diag::err_ext_vector_component_name_illegal, 
         std::string(compStr,compStr+1), SourceRange(CompLoc));
    return QualType();
  }
  // Each component accessor can't exceed the vector type.
  compStr = CompName.getName();
  while (*compStr) {
    if (vecType->isAccessorWithinNumElements(*compStr))
      compStr++;
    else
      break;
  }
  if (!SpecialComponent && *compStr) { 
    // We didn't get to the end of the string. This means a component accessor
    // exceeds the number of elements in the vector.
    Diag(OpLoc, diag::err_ext_vector_component_exceeds_length, 
                baseType.getAsString(), SourceRange(CompLoc));
    return QualType();
  }

  // If we have a special component name, verify that the current vector length
  // is an even number, since all special component names return exactly half
  // the elements.
  if (SpecialComponent && (vecType->getNumElements() & 1U)) {
    return QualType();
  }
  
  // The component accessor looks fine - now we need to compute the actual type.
  // The vector type is implied by the component accessor. For example, 
  // vec4.b is a float, vec4.xy is a vec2, vec4.rgb is a vec3, etc.
  // vec4.hi, vec4.lo, vec4.e, and vec4.o all return vec2.
  unsigned CompSize = SpecialComponent ? vecType->getNumElements() / 2
                                       : strlen(CompName.getName());
  if (CompSize == 1)
    return vecType->getElementType();
    
  QualType VT = Context.getExtVectorType(vecType->getElementType(), CompSize);
  // Now look up the TypeDefDecl from the vector type. Without this, 
  // diagostics look bad. We want extended vector types to appear built-in.
  for (unsigned i = 0, E = ExtVectorDecls.size(); i != E; ++i) {
    if (ExtVectorDecls[i]->getUnderlyingType() == VT)
      return Context.getTypedefType(ExtVectorDecls[i]);
  }
  return VT; // should never get here (a typedef type should always be found).
}

/// constructSetterName - Return the setter name for the given
/// identifier, i.e. "set" + Name where the initial character of Name
/// has been capitalized.
// FIXME: Merge with same routine in Parser. But where should this
// live?
static IdentifierInfo *constructSetterName(IdentifierTable &Idents,
                                           const IdentifierInfo *Name) {
  unsigned N = Name->getLength();
  char *SelectorName = new char[3 + N];
  memcpy(SelectorName, "set", 3);
  memcpy(&SelectorName[3], Name->getName(), N);
  SelectorName[3] = toupper(SelectorName[3]);

  IdentifierInfo *Setter = 
    &Idents.get(SelectorName, &SelectorName[3 + N]);
  delete[] SelectorName;
  return Setter;
}

Action::ExprResult Sema::
ActOnMemberReferenceExpr(ExprTy *Base, SourceLocation OpLoc,
                         tok::TokenKind OpKind, SourceLocation MemberLoc,
                         IdentifierInfo &Member) {
  Expr *BaseExpr = static_cast<Expr *>(Base);
  assert(BaseExpr && "no record expression");

  // Perform default conversions.
  DefaultFunctionArrayConversion(BaseExpr);
  
  QualType BaseType = BaseExpr->getType();
  assert(!BaseType.isNull() && "no type for member expression");
  
  // Get the type being accessed in BaseType.  If this is an arrow, the BaseExpr
  // must have pointer type, and the accessed type is the pointee.
  if (OpKind == tok::arrow) {
    if (const PointerType *PT = BaseType->getAsPointerType())
      BaseType = PT->getPointeeType();
    else
      return Diag(MemberLoc, diag::err_typecheck_member_reference_arrow,
                  BaseType.getAsString(), BaseExpr->getSourceRange());
  }
  
  // Handle field access to simple records.  This also handles access to fields
  // of the ObjC 'id' struct.
  if (const RecordType *RTy = BaseType->getAsRecordType()) {
    RecordDecl *RDecl = RTy->getDecl();
    if (RTy->isIncompleteType())
      return Diag(OpLoc, diag::err_typecheck_incomplete_tag, RDecl->getName(),
                  BaseExpr->getSourceRange());
    // The record definition is complete, now make sure the member is valid.
    FieldDecl *MemberDecl = RDecl->getMember(&Member);
    if (!MemberDecl)
      return Diag(MemberLoc, diag::err_typecheck_no_member, Member.getName(),
                  BaseExpr->getSourceRange());

    // Figure out the type of the member; see C99 6.5.2.3p3
    // FIXME: Handle address space modifiers
    QualType MemberType = MemberDecl->getType();
    unsigned combinedQualifiers =
        MemberType.getCVRQualifiers() | BaseType.getCVRQualifiers();
    MemberType = MemberType.getQualifiedType(combinedQualifiers);

    return new MemberExpr(BaseExpr, OpKind == tok::arrow, MemberDecl,
                          MemberLoc, MemberType);
  }
  
  // Handle access to Objective-C instance variables, such as "Obj->ivar" and
  // (*Obj).ivar.
  if (const ObjCInterfaceType *IFTy = BaseType->getAsObjCInterfaceType()) {
    if (ObjCIvarDecl *IV = IFTy->getDecl()->lookupInstanceVariable(&Member))
      return new ObjCIvarRefExpr(IV, IV->getType(), MemberLoc, BaseExpr, 
                                 OpKind == tok::arrow);
    return Diag(MemberLoc, diag::err_typecheck_member_reference_ivar,
                IFTy->getDecl()->getName(), Member.getName(),
                BaseExpr->getSourceRange());
  }
  
  // Handle Objective-C property access, which is "Obj.property" where Obj is a
  // pointer to a (potentially qualified) interface type.
  const PointerType *PTy;
  const ObjCInterfaceType *IFTy;
  if (OpKind == tok::period && (PTy = BaseType->getAsPointerType()) &&
      (IFTy = PTy->getPointeeType()->getAsObjCInterfaceType())) {
    ObjCInterfaceDecl *IFace = IFTy->getDecl();

    // Search for a declared property first.
    if (ObjCPropertyDecl *PD = IFace->FindPropertyDeclaration(&Member))
      return new ObjCPropertyRefExpr(PD, PD->getType(), MemberLoc, BaseExpr);
    
    // Check protocols on qualified interfaces.
    for (ObjCInterfaceType::qual_iterator I = IFTy->qual_begin(),
         E = IFTy->qual_end(); I != E; ++I)
      if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(&Member))
        return new ObjCPropertyRefExpr(PD, PD->getType(), MemberLoc, BaseExpr);

    // If that failed, look for an "implicit" property by seeing if the nullary
    // selector is implemented.

    // FIXME: The logic for looking up nullary and unary selectors should be
    // shared with the code in ActOnInstanceMessage.

    Selector Sel = PP.getSelectorTable().getNullarySelector(&Member);
    ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);
    
    // If this reference is in an @implementation, check for 'private' methods.
    if (!Getter)
      if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
        if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
          if (ObjCImplementationDecl *ImpDecl = 
              ObjCImplementations[ClassDecl->getIdentifier()])
            Getter = ImpDecl->getInstanceMethod(Sel);

    if (Getter) {
      // If we found a getter then this may be a valid dot-reference, we
      // need to also look for the matching setter.
      IdentifierInfo *SetterName = constructSetterName(PP.getIdentifierTable(),
                                                       &Member);
      Selector SetterSel = PP.getSelectorTable().getUnarySelector(SetterName);
      ObjCMethodDecl *Setter = IFace->lookupInstanceMethod(SetterSel);

      if (!Setter) {
        if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
          if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
            if (ObjCImplementationDecl *ImpDecl = 
                ObjCImplementations[ClassDecl->getIdentifier()])
              Setter = ImpDecl->getInstanceMethod(SetterSel);
      }

      // FIXME: There are some issues here. First, we are not
      // diagnosing accesses to read-only properties because we do not
      // know if this is a getter or setter yet. Second, we are
      // checking that the type of the setter matches the type we
      // expect.
      return new ObjCPropertyRefExpr(Getter, Setter, Getter->getResultType(), 
                                     MemberLoc, BaseExpr);
    }
  }
  
  // Handle 'field access' to vectors, such as 'V.xx'.
  if (BaseType->isExtVectorType() && OpKind == tok::period) {
    // Component access limited to variables (reject vec4.rg.g).
    if (!isa<DeclRefExpr>(BaseExpr) && !isa<ArraySubscriptExpr>(BaseExpr) &&
        !isa<ExtVectorElementExpr>(BaseExpr))
      return Diag(MemberLoc, diag::err_ext_vector_component_access, 
                  BaseExpr->getSourceRange());
    QualType ret = CheckExtVectorComponent(BaseType, OpLoc, Member, MemberLoc);
    if (ret.isNull())
      return true;
    return new ExtVectorElementExpr(ret, BaseExpr, Member, MemberLoc);
  }
  
  return Diag(MemberLoc, diag::err_typecheck_member_reference_struct_union,
              BaseType.getAsString(), BaseExpr->getSourceRange());
}

/// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::ExprResult Sema::
ActOnCallExpr(ExprTy *fn, SourceLocation LParenLoc,
              ExprTy **args, unsigned NumArgs,
              SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  Expr *Fn = static_cast<Expr *>(fn);
  Expr **Args = reinterpret_cast<Expr**>(args);
  assert(Fn && "no function call expression");
  FunctionDecl *FDecl = NULL;

  // Promote the function operand.
  UsualUnaryConversions(Fn);

  // If we're directly calling a function, get the declaration for
  // that function.
  if (ImplicitCastExpr *IcExpr = dyn_cast<ImplicitCastExpr>(Fn))
    if (DeclRefExpr *DRExpr = dyn_cast<DeclRefExpr>(IcExpr->getSubExpr()))
      FDecl = dyn_cast<FunctionDecl>(DRExpr->getDecl());

  // Make the call expr early, before semantic checks.  This guarantees cleanup
  // of arguments and function on error.
  llvm::OwningPtr<CallExpr> TheCall(new CallExpr(Fn, Args, NumArgs,
                                                 Context.BoolTy, RParenLoc));
  
  // C99 6.5.2.2p1 - "The expression that denotes the called function shall have
  // type pointer to function".
  const PointerType *PT = Fn->getType()->getAsPointerType();
  if (PT == 0)
    return Diag(LParenLoc, diag::err_typecheck_call_not_function,
                Fn->getSourceRange());
  const FunctionType *FuncT = PT->getPointeeType()->getAsFunctionType();
  if (FuncT == 0)
    return Diag(LParenLoc, diag::err_typecheck_call_not_function,
                Fn->getSourceRange());
  
  // We know the result type of the call, set it.
  TheCall->setType(FuncT->getResultType());
    
  if (const FunctionTypeProto *Proto = dyn_cast<FunctionTypeProto>(FuncT)) {
    // C99 6.5.2.2p7 - the arguments are implicitly converted, as if by 
    // assignment, to the types of the corresponding parameter, ...
    unsigned NumArgsInProto = Proto->getNumArgs();
    unsigned NumArgsToCheck = NumArgs;
    
    // If too few arguments are available (and we don't have default
    // arguments for the remaining parameters), don't make the call.
    if (NumArgs < NumArgsInProto) {
      if (FDecl && NumArgs >= FDecl->getMinRequiredArguments()) {
        // Use default arguments for missing arguments
        NumArgsToCheck = NumArgsInProto;
        TheCall->setNumArgs(NumArgsInProto);
      } else
        return Diag(RParenLoc, diag::err_typecheck_call_too_few_args,
                    Fn->getSourceRange());
    }

    // If too many are passed and not variadic, error on the extras and drop
    // them.
    if (NumArgs > NumArgsInProto) {
      if (!Proto->isVariadic()) {
        Diag(Args[NumArgsInProto]->getLocStart(), 
             diag::err_typecheck_call_too_many_args, Fn->getSourceRange(),
             SourceRange(Args[NumArgsInProto]->getLocStart(),
                         Args[NumArgs-1]->getLocEnd()));
        // This deletes the extra arguments.
        TheCall->setNumArgs(NumArgsInProto);
      }
      NumArgsToCheck = NumArgsInProto;
    }
    
    // Continue to check argument types (even if we have too few/many args).
    for (unsigned i = 0; i != NumArgsToCheck; i++) {
      QualType ProtoArgType = Proto->getArgType(i);

      Expr *Arg;
      if (i < NumArgs) 
        Arg = Args[i];
      else 
        Arg = new CXXDefaultArgExpr(FDecl->getParamDecl(i));
      QualType ArgType = Arg->getType();

      // Compute implicit casts from the operand to the formal argument type.
      AssignConvertType ConvTy =
        CheckSingleAssignmentConstraints(ProtoArgType, Arg);
      TheCall->setArg(i, Arg);

      if (DiagnoseAssignmentResult(ConvTy, Arg->getLocStart(), ProtoArgType,
                                   ArgType, Arg, "passing"))
        return true;
    }
    
    // If this is a variadic call, handle args passed through "...".
    if (Proto->isVariadic()) {
      // Promote the arguments (C99 6.5.2.2p7).
      for (unsigned i = NumArgsInProto; i != NumArgs; i++) {
        Expr *Arg = Args[i];
        DefaultArgumentPromotion(Arg);
        TheCall->setArg(i, Arg);
      }
    }
  } else {
    assert(isa<FunctionTypeNoProto>(FuncT) && "Unknown FunctionType!");
    
    // Promote the arguments (C99 6.5.2.2p6).
    for (unsigned i = 0; i != NumArgs; i++) {
      Expr *Arg = Args[i];
      DefaultArgumentPromotion(Arg);
      TheCall->setArg(i, Arg);
    }
  }

  // Do special checking on direct calls to functions.
  if (FDecl)
    return CheckFunctionCall(FDecl, TheCall.take());

  return TheCall.take();
}

Action::ExprResult Sema::
ActOnCompoundLiteral(SourceLocation LParenLoc, TypeTy *Ty,
                     SourceLocation RParenLoc, ExprTy *InitExpr) {
  assert((Ty != 0) && "ActOnCompoundLiteral(): missing type");
  QualType literalType = QualType::getFromOpaquePtr(Ty);
  // FIXME: put back this assert when initializers are worked out.
  //assert((InitExpr != 0) && "ActOnCompoundLiteral(): missing expression");
  Expr *literalExpr = static_cast<Expr*>(InitExpr);

  if (literalType->isArrayType()) {
    if (literalType->isVariableArrayType())
      return Diag(LParenLoc,
                  diag::err_variable_object_no_init,
                  SourceRange(LParenLoc,
                              literalExpr->getSourceRange().getEnd()));
  } else if (literalType->isIncompleteType()) {
    return Diag(LParenLoc,
                diag::err_typecheck_decl_incomplete_type,
                literalType.getAsString(),
                SourceRange(LParenLoc,
                            literalExpr->getSourceRange().getEnd()));
  }

  if (CheckInitializerTypes(literalExpr, literalType))
    return true;

  bool isFileScope = !getCurFunctionDecl() && !getCurMethodDecl();
  if (isFileScope) { // 6.5.2.5p3
    if (CheckForConstantInitializer(literalExpr, literalType))
      return true;
  }
  return new CompoundLiteralExpr(LParenLoc, literalType, literalExpr, isFileScope);
}

Action::ExprResult Sema::
ActOnInitList(SourceLocation LBraceLoc, ExprTy **initlist, unsigned NumInit,
              SourceLocation RBraceLoc) {
  Expr **InitList = reinterpret_cast<Expr**>(initlist);

  // Semantic analysis for initializers is done by ActOnDeclarator() and
  // CheckInitializer() - it requires knowledge of the object being intialized. 
  
  InitListExpr *E = new InitListExpr(LBraceLoc, InitList, NumInit, RBraceLoc);
  E->setType(Context.VoidTy); // FIXME: just a place holder for now.
  return E;
}

/// CheckCastTypes - Check type constraints for casting between types.
bool Sema::CheckCastTypes(SourceRange TyR, QualType castType, Expr *&castExpr) {
  UsualUnaryConversions(castExpr);

  // C99 6.5.4p2: the cast type needs to be void or scalar and the expression
  // type needs to be scalar.
  if (castType->isVoidType()) {
    // Cast to void allows any expr type.
  } else if (!castType->isScalarType() && !castType->isVectorType()) {
    // GCC struct/union extension: allow cast to self.
    if (Context.getCanonicalType(castType) !=
        Context.getCanonicalType(castExpr->getType()) ||
        (!castType->isStructureType() && !castType->isUnionType())) {
      // Reject any other conversions to non-scalar types.
      return Diag(TyR.getBegin(), diag::err_typecheck_cond_expect_scalar, 
                  castType.getAsString(), castExpr->getSourceRange());
    }
      
    // accept this, but emit an ext-warn.
    Diag(TyR.getBegin(), diag::ext_typecheck_cast_nonscalar, 
         castType.getAsString(), castExpr->getSourceRange());
  } else if (!castExpr->getType()->isScalarType() && 
             !castExpr->getType()->isVectorType()) {
    return Diag(castExpr->getLocStart(), 
                diag::err_typecheck_expect_scalar_operand, 
                castExpr->getType().getAsString(),castExpr->getSourceRange());
  } else if (castExpr->getType()->isVectorType()) {
    if (CheckVectorCast(TyR, castExpr->getType(), castType))
      return true;
  } else if (castType->isVectorType()) {
    if (CheckVectorCast(TyR, castType, castExpr->getType()))
      return true;
  }
  return false;
}

bool Sema::CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty) {
  assert(VectorTy->isVectorType() && "Not a vector type!");
  
  if (Ty->isVectorType() || Ty->isIntegerType()) {
    if (Context.getTypeSize(VectorTy) != Context.getTypeSize(Ty))
      return Diag(R.getBegin(),
                  Ty->isVectorType() ? 
                  diag::err_invalid_conversion_between_vectors :
                  diag::err_invalid_conversion_between_vector_and_integer,
                  VectorTy.getAsString().c_str(),
                  Ty.getAsString().c_str(), R);
  } else
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar,
                VectorTy.getAsString().c_str(),
                Ty.getAsString().c_str(), R);
  
  return false;
}

Action::ExprResult Sema::
ActOnCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
              SourceLocation RParenLoc, ExprTy *Op) {
  assert((Ty != 0) && (Op != 0) && "ActOnCastExpr(): missing type or expr");

  Expr *castExpr = static_cast<Expr*>(Op);
  QualType castType = QualType::getFromOpaquePtr(Ty);

  if (CheckCastTypes(SourceRange(LParenLoc, RParenLoc), castType, castExpr))
    return true;
  return new ExplicitCastExpr(castType, castExpr, LParenLoc);
}

/// Note that lex is not null here, even if this is the gnu "x ?: y" extension.
/// In that case, lex = cond.
inline QualType Sema::CheckConditionalOperands( // C99 6.5.15
  Expr *&cond, Expr *&lex, Expr *&rex, SourceLocation questionLoc) {
  UsualUnaryConversions(cond);
  UsualUnaryConversions(lex);
  UsualUnaryConversions(rex);
  QualType condT = cond->getType();
  QualType lexT = lex->getType();
  QualType rexT = rex->getType();

  // first, check the condition.
  if (!condT->isScalarType()) { // C99 6.5.15p2
    Diag(cond->getLocStart(), diag::err_typecheck_cond_expect_scalar, 
         condT.getAsString());
    return QualType();
  }
  
  // Now check the two expressions.
  
  // If both operands have arithmetic type, do the usual arithmetic conversions
  // to find a common type: C99 6.5.15p3,5.
  if (lexT->isArithmeticType() && rexT->isArithmeticType()) {
    UsualArithmeticConversions(lex, rex);
    return lex->getType();
  }
  
  // If both operands are the same structure or union type, the result is that
  // type.
  if (const RecordType *LHSRT = lexT->getAsRecordType()) {    // C99 6.5.15p3
    if (const RecordType *RHSRT = rexT->getAsRecordType())
      if (LHSRT->getDecl() == RHSRT->getDecl())
        // "If both the operands have structure or union type, the result has 
        // that type."  This implies that CV qualifiers are dropped.
        return lexT.getUnqualifiedType();
  }
  
  // C99 6.5.15p5: "If both operands have void type, the result has void type."
  // The following || allows only one side to be void (a GCC-ism).
  if (lexT->isVoidType() || rexT->isVoidType()) {
    if (!lexT->isVoidType())
      Diag(rex->getLocStart(), diag::ext_typecheck_cond_one_void, 
           rex->getSourceRange());
    if (!rexT->isVoidType())
      Diag(lex->getLocStart(), diag::ext_typecheck_cond_one_void,
           lex->getSourceRange());
    ImpCastExprToType(lex, Context.VoidTy);
    ImpCastExprToType(rex, Context.VoidTy);
    return Context.VoidTy;
  }
  // C99 6.5.15p6 - "if one operand is a null pointer constant, the result has
  // the type of the other operand."
  if (lexT->isPointerType() && rex->isNullPointerConstant(Context)) {
    ImpCastExprToType(rex, lexT); // promote the null to a pointer.
    return lexT;
  }
  if (rexT->isPointerType() && lex->isNullPointerConstant(Context)) {
    ImpCastExprToType(lex, rexT); // promote the null to a pointer.
    return rexT;
  }
  // Allow any Objective-C types to devolve to id type.
  // FIXME: This seems to match gcc behavior, although that is very
  // arguably incorrect. For example, (xxx ? (id<P>) : (id<P>)) has
  // type id, which seems broken.
  if (Context.isObjCObjectPointerType(lexT) && 
      Context.isObjCObjectPointerType(rexT)) {
    // FIXME: This is not the correct composite type. This only
    // happens to work because id can more or less be used anywhere,
    // however this may change the type of method sends.
    // FIXME: gcc adds some type-checking of the arguments and emits
    // (confusing) incompatible comparison warnings in some
    // cases. Investigate.
    QualType compositeType = Context.getObjCIdType();
    ImpCastExprToType(lex, compositeType);
    ImpCastExprToType(rex, compositeType);
    return compositeType;
  }
  // Handle the case where both operands are pointers before we handle null
  // pointer constants in case both operands are null pointer constants.
  if (const PointerType *LHSPT = lexT->getAsPointerType()) { // C99 6.5.15p3,6
    if (const PointerType *RHSPT = rexT->getAsPointerType()) {
      // get the "pointed to" types
      QualType lhptee = LHSPT->getPointeeType();
      QualType rhptee = RHSPT->getPointeeType();

      // ignore qualifiers on void (C99 6.5.15p3, clause 6)
      if (lhptee->isVoidType() &&
          rhptee->isIncompleteOrObjectType()) {
        // Figure out necessary qualifiers (C99 6.5.15p6)
        QualType destPointee=lhptee.getQualifiedType(rhptee.getCVRQualifiers());
        QualType destType = Context.getPointerType(destPointee);
        ImpCastExprToType(lex, destType); // add qualifiers if necessary
        ImpCastExprToType(rex, destType); // promote to void*
        return destType;
      }
      if (rhptee->isVoidType() && lhptee->isIncompleteOrObjectType()) {
        QualType destPointee=rhptee.getQualifiedType(lhptee.getCVRQualifiers());
        QualType destType = Context.getPointerType(destPointee);
        ImpCastExprToType(lex, destType); // add qualifiers if necessary
        ImpCastExprToType(rex, destType); // promote to void*
        return destType;
      }

      if (!Context.typesAreCompatible(lhptee.getUnqualifiedType(), 
                                      rhptee.getUnqualifiedType())) {
        Diag(questionLoc, diag::warn_typecheck_cond_incompatible_pointers,
             lexT.getAsString(), rexT.getAsString(),
             lex->getSourceRange(), rex->getSourceRange());
        // In this situation, assume a conservative type; in general
        // we assume void* type. No especially good reason, but this
        // is what gcc does, and we do have to pick to get a
        // consistent AST. However, if either type is an Objective-C
        // object type then use id.
        QualType incompatTy;
        if (Context.isObjCObjectPointerType(lexT) || 
            Context.isObjCObjectPointerType(rexT)) {
          incompatTy = Context.getObjCIdType();
        } else {
          incompatTy = Context.getPointerType(Context.VoidTy);
        }
        ImpCastExprToType(lex, incompatTy);
        ImpCastExprToType(rex, incompatTy);
        return incompatTy;
      }
      // The pointer types are compatible.
      // C99 6.5.15p6: If both operands are pointers to compatible types *or* to
      // differently qualified versions of compatible types, the result type is
      // a pointer to an appropriately qualified version of the *composite*
      // type.
      // FIXME: Need to calculate the composite type.
      // FIXME: Need to add qualifiers
      QualType compositeType = lexT;
      ImpCastExprToType(lex, compositeType);
      ImpCastExprToType(rex, compositeType);
      return compositeType;
    }
  }
  // Otherwise, the operands are not compatible.
  Diag(questionLoc, diag::err_typecheck_cond_incompatible_operands,
       lexT.getAsString(), rexT.getAsString(),
       lex->getSourceRange(), rex->getSourceRange());
  return QualType();
}

/// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::ExprResult Sema::ActOnConditionalOp(SourceLocation QuestionLoc, 
                                            SourceLocation ColonLoc,
                                            ExprTy *Cond, ExprTy *LHS,
                                            ExprTy *RHS) {
  Expr *CondExpr = (Expr *) Cond;
  Expr *LHSExpr = (Expr *) LHS, *RHSExpr = (Expr *) RHS;

  // If this is the gnu "x ?: y" extension, analyze the types as though the LHS
  // was the condition.
  bool isLHSNull = LHSExpr == 0;
  if (isLHSNull)
    LHSExpr = CondExpr;
  
  QualType result = CheckConditionalOperands(CondExpr, LHSExpr, 
                                             RHSExpr, QuestionLoc);
  if (result.isNull())
    return true;
  return new ConditionalOperator(CondExpr, isLHSNull ? 0 : LHSExpr,
                                 RHSExpr, result);
}


// CheckPointerTypesForAssignment - This is a very tricky routine (despite
// being closely modeled after the C99 spec:-). The odd characteristic of this 
// routine is it effectively iqnores the qualifiers on the top level pointee.
// This circumvents the usual type rules specified in 6.2.7p1 & 6.7.5.[1-3].
// FIXME: add a couple examples in this comment.
Sema::AssignConvertType 
Sema::CheckPointerTypesForAssignment(QualType lhsType, QualType rhsType) {
  QualType lhptee, rhptee;
  
  // get the "pointed to" type (ignoring qualifiers at the top level)
  lhptee = lhsType->getAsPointerType()->getPointeeType();
  rhptee = rhsType->getAsPointerType()->getPointeeType();
  
  // make sure we operate on the canonical type
  lhptee = Context.getCanonicalType(lhptee);
  rhptee = Context.getCanonicalType(rhptee);

  AssignConvertType ConvTy = Compatible;
  
  // C99 6.5.16.1p1: This following citation is common to constraints 
  // 3 & 4 (below). ...and the type *pointed to* by the left has all the 
  // qualifiers of the type *pointed to* by the right; 
  // FIXME: Handle ASQualType
  if ((lhptee.getCVRQualifiers() & rhptee.getCVRQualifiers()) != 
       rhptee.getCVRQualifiers())
    ConvTy = CompatiblePointerDiscardsQualifiers;

  // C99 6.5.16.1p1 (constraint 4): If one operand is a pointer to an object or 
  // incomplete type and the other is a pointer to a qualified or unqualified 
  // version of void...
  if (lhptee->isVoidType()) {
    if (rhptee->isIncompleteOrObjectType())
      return ConvTy;
    
    // As an extension, we allow cast to/from void* to function pointer.
    assert(rhptee->isFunctionType());
    return FunctionVoidPointer;
  }
  
  if (rhptee->isVoidType()) {
    if (lhptee->isIncompleteOrObjectType())
      return ConvTy;

    // As an extension, we allow cast to/from void* to function pointer.
    assert(lhptee->isFunctionType());
    return FunctionVoidPointer;
  }

  // Check for ObjC interfaces
  const ObjCInterfaceType* LHSIface = lhptee->getAsObjCInterfaceType();
  const ObjCInterfaceType* RHSIface = rhptee->getAsObjCInterfaceType();
  if (LHSIface && RHSIface &&
      Context.canAssignObjCInterfaces(LHSIface, RHSIface))
    return ConvTy;

  // ID acts sort of like void* for ObjC interfaces
  if (LHSIface && Context.isObjCIdType(rhptee))
    return ConvTy;
  if (RHSIface && Context.isObjCIdType(lhptee))
    return ConvTy;

  // C99 6.5.16.1p1 (constraint 3): both operands are pointers to qualified or 
  // unqualified versions of compatible types, ...
  if (!Context.typesAreCompatible(lhptee.getUnqualifiedType(), 
                                  rhptee.getUnqualifiedType()))
    return IncompatiblePointer; // this "trumps" PointerAssignDiscardsQualifiers
  return ConvTy;
}

/// CheckBlockPointerTypesForAssignment - This routine determines whether two
/// block pointer types are compatible or whether a block and normal pointer
/// are compatible. It is more restrict than comparing two function pointer
// types.
Sema::AssignConvertType 
Sema::CheckBlockPointerTypesForAssignment(QualType lhsType, 
                                          QualType rhsType) {
  QualType lhptee, rhptee;
  
  // get the "pointed to" type (ignoring qualifiers at the top level)
  lhptee = lhsType->getAsBlockPointerType()->getPointeeType();
  rhptee = rhsType->getAsBlockPointerType()->getPointeeType(); 
  
  // make sure we operate on the canonical type
  lhptee = Context.getCanonicalType(lhptee);
  rhptee = Context.getCanonicalType(rhptee);
  
  AssignConvertType ConvTy = Compatible;
  
  // For blocks we enforce that qualifiers are identical.
  if (lhptee.getCVRQualifiers() != rhptee.getCVRQualifiers())
    ConvTy = CompatiblePointerDiscardsQualifiers;
    
  if (!Context.typesAreBlockCompatible(lhptee, rhptee))
    return IncompatibleBlockPointer; 
  return ConvTy;
}

/// CheckAssignmentConstraints (C99 6.5.16) - This routine currently 
/// has code to accommodate several GCC extensions when type checking 
/// pointers. Here are some objectionable examples that GCC considers warnings:
///
///  int a, *pint;
///  short *pshort;
///  struct foo *pfoo;
///
///  pint = pshort; // warning: assignment from incompatible pointer type
///  a = pint; // warning: assignment makes integer from pointer without a cast
///  pint = a; // warning: assignment makes pointer from integer without a cast
///  pint = pfoo; // warning: assignment from incompatible pointer type
///
/// As a result, the code for dealing with pointers is more complex than the
/// C99 spec dictates. 
///
Sema::AssignConvertType
Sema::CheckAssignmentConstraints(QualType lhsType, QualType rhsType) {
  // Get canonical types.  We're not formatting these types, just comparing
  // them.
  lhsType = Context.getCanonicalType(lhsType).getUnqualifiedType();
  rhsType = Context.getCanonicalType(rhsType).getUnqualifiedType();

  if (lhsType == rhsType)
    return Compatible; // Common case: fast path an exact match.

  if (lhsType->isReferenceType() || rhsType->isReferenceType()) {
    if (Context.typesAreCompatible(lhsType, rhsType))
      return Compatible;
    return Incompatible;
  }

  if (lhsType->isObjCQualifiedIdType() || rhsType->isObjCQualifiedIdType()) {
    if (ObjCQualifiedIdTypesAreCompatible(lhsType, rhsType, false))
      return Compatible;
    // Relax integer conversions like we do for pointers below.
    if (rhsType->isIntegerType())
      return IntToPointer;
    if (lhsType->isIntegerType())
      return PointerToInt;
    return Incompatible;
  }

  if (lhsType->isVectorType() || rhsType->isVectorType()) {
    // For ExtVector, allow vector splats; float -> <n x float>
    if (const ExtVectorType *LV = lhsType->getAsExtVectorType())
      if (LV->getElementType() == rhsType)
        return Compatible;

    // If we are allowing lax vector conversions, and LHS and RHS are both
    // vectors, the total size only needs to be the same. This is a bitcast; 
    // no bits are changed but the result type is different.
    if (getLangOptions().LaxVectorConversions &&
        lhsType->isVectorType() && rhsType->isVectorType()) {
      if (Context.getTypeSize(lhsType) == Context.getTypeSize(rhsType))
        return Compatible;
    }
    return Incompatible;
  }      

  if (lhsType->isArithmeticType() && rhsType->isArithmeticType())
    return Compatible;

  if (isa<PointerType>(lhsType)) {
    if (rhsType->isIntegerType())
      return IntToPointer;

    if (isa<PointerType>(rhsType))
      return CheckPointerTypesForAssignment(lhsType, rhsType);
      
    if (const BlockPointerType *BPT = rhsType->getAsBlockPointerType())
      if (BPT->getPointeeType()->isVoidType())
        return BlockVoidPointer;
      
    return Incompatible;
  }

  if (isa<BlockPointerType>(lhsType)) {
    if (rhsType->isIntegerType())
      return IntToPointer;
    
    if (rhsType->isBlockPointerType())
      return CheckBlockPointerTypesForAssignment(lhsType, rhsType);
      
    if (const PointerType *RHSPT = rhsType->getAsPointerType()) {
      if (RHSPT->getPointeeType()->isVoidType())
        return BlockVoidPointer;
    }
    return Incompatible;
  }

  if (isa<PointerType>(rhsType)) {
    // C99 6.5.16.1p1: the left operand is _Bool and the right is a pointer.
    if (lhsType == Context.BoolTy)
      return Compatible;

    if (lhsType->isIntegerType())
      return PointerToInt;

    if (isa<PointerType>(lhsType)) 
      return CheckPointerTypesForAssignment(lhsType, rhsType);
      
    if (isa<BlockPointerType>(lhsType) && 
        rhsType->getAsPointerType()->getPointeeType()->isVoidType())
      return BlockVoidPointer;
    return Incompatible;
  }

  if (isa<TagType>(lhsType) && isa<TagType>(rhsType)) {
    if (Context.typesAreCompatible(lhsType, rhsType))
      return Compatible;
  }
  return Incompatible;
}

Sema::AssignConvertType
Sema::CheckSingleAssignmentConstraints(QualType lhsType, Expr *&rExpr) {
  // C99 6.5.16.1p1: the left operand is a pointer and the right is
  // a null pointer constant.
  if ((lhsType->isPointerType() || lhsType->isObjCQualifiedIdType() ||
       lhsType->isBlockPointerType()) 
      && rExpr->isNullPointerConstant(Context)) {
    ImpCastExprToType(rExpr, lhsType);
    return Compatible;
  }
  
  // We don't allow conversion of non-null-pointer constants to integers.
  if (lhsType->isBlockPointerType() && rExpr->getType()->isIntegerType())
    return IntToBlockPointer;

  // This check seems unnatural, however it is necessary to ensure the proper
  // conversion of functions/arrays. If the conversion were done for all
  // DeclExpr's (created by ActOnIdentifierExpr), it would mess up the unary
  // expressions that surpress this implicit conversion (&, sizeof).
  //
  // Suppress this for references: C99 8.5.3p5.  FIXME: revisit when references
  // are better understood.
  if (!lhsType->isReferenceType())
    DefaultFunctionArrayConversion(rExpr);

  Sema::AssignConvertType result =
    CheckAssignmentConstraints(lhsType, rExpr->getType());
  
  // C99 6.5.16.1p2: The value of the right operand is converted to the
  // type of the assignment expression.
  if (rExpr->getType() != lhsType)
    ImpCastExprToType(rExpr, lhsType);
  return result;
}

Sema::AssignConvertType
Sema::CheckCompoundAssignmentConstraints(QualType lhsType, QualType rhsType) {
  return CheckAssignmentConstraints(lhsType, rhsType);
}

QualType Sema::InvalidOperands(SourceLocation loc, Expr *&lex, Expr *&rex) {
  Diag(loc, diag::err_typecheck_invalid_operands, 
       lex->getType().getAsString(), rex->getType().getAsString(),
       lex->getSourceRange(), rex->getSourceRange());
  return QualType();
}

inline QualType Sema::CheckVectorOperands(SourceLocation loc, Expr *&lex, 
                                                              Expr *&rex) {
  // For conversion purposes, we ignore any qualifiers. 
  // For example, "const float" and "float" are equivalent.
  QualType lhsType =
    Context.getCanonicalType(lex->getType()).getUnqualifiedType();
  QualType rhsType =
    Context.getCanonicalType(rex->getType()).getUnqualifiedType();
  
  // If the vector types are identical, return.
  if (lhsType == rhsType)
    return lhsType;

  // Handle the case of a vector & extvector type of the same size and element
  // type.  It would be nice if we only had one vector type someday.
  if (getLangOptions().LaxVectorConversions)
    if (const VectorType *LV = lhsType->getAsVectorType())
      if (const VectorType *RV = rhsType->getAsVectorType())
        if (LV->getElementType() == RV->getElementType() &&
            LV->getNumElements() == RV->getNumElements())
          return lhsType->isExtVectorType() ? lhsType : rhsType;

  // If the lhs is an extended vector and the rhs is a scalar of the same type
  // or a literal, promote the rhs to the vector type.
  if (const ExtVectorType *V = lhsType->getAsExtVectorType()) {
    QualType eltType = V->getElementType();
    
    if ((eltType->getAsBuiltinType() == rhsType->getAsBuiltinType()) || 
        (eltType->isIntegerType() && isa<IntegerLiteral>(rex)) ||
        (eltType->isFloatingType() && isa<FloatingLiteral>(rex))) {
      ImpCastExprToType(rex, lhsType);
      return lhsType;
    }
  }

  // If the rhs is an extended vector and the lhs is a scalar of the same type,
  // promote the lhs to the vector type.
  if (const ExtVectorType *V = rhsType->getAsExtVectorType()) {
    QualType eltType = V->getElementType();

    if ((eltType->getAsBuiltinType() == lhsType->getAsBuiltinType()) || 
        (eltType->isIntegerType() && isa<IntegerLiteral>(lex)) ||
        (eltType->isFloatingType() && isa<FloatingLiteral>(lex))) {
      ImpCastExprToType(lex, rhsType);
      return rhsType;
    }
  }

  // You cannot convert between vector values of different size.
  Diag(loc, diag::err_typecheck_vector_not_convertable, 
       lex->getType().getAsString(), rex->getType().getAsString(),
       lex->getSourceRange(), rex->getSourceRange());
  return QualType();
}    

inline QualType Sema::CheckMultiplyDivideOperands(
  Expr *&lex, Expr *&rex, SourceLocation loc, bool isCompAssign) 
{
  QualType lhsType = lex->getType(), rhsType = rex->getType();

  if (lhsType->isVectorType() || rhsType->isVectorType())
    return CheckVectorOperands(loc, lex, rex);
    
  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);
  
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    return compType;
  return InvalidOperands(loc, lex, rex);
}

inline QualType Sema::CheckRemainderOperands(
  Expr *&lex, Expr *&rex, SourceLocation loc, bool isCompAssign) 
{
  QualType lhsType = lex->getType(), rhsType = rex->getType();

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);
  
  if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
    return compType;
  return InvalidOperands(loc, lex, rex);
}

inline QualType Sema::CheckAdditionOperands( // C99 6.5.6
  Expr *&lex, Expr *&rex, SourceLocation loc, bool isCompAssign) 
{
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(loc, lex, rex);

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  // handle the common case first (both operands are arithmetic).
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    return compType;

  // Put any potential pointer into PExp
  Expr* PExp = lex, *IExp = rex;
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);

  if (const PointerType* PTy = PExp->getType()->getAsPointerType()) {
    if (IExp->getType()->isIntegerType()) {
      // Check for arithmetic on pointers to incomplete types
      if (!PTy->getPointeeType()->isObjectType()) {
        if (PTy->getPointeeType()->isVoidType()) {
          Diag(loc, diag::ext_gnu_void_ptr, 
               lex->getSourceRange(), rex->getSourceRange());
        } else {
          Diag(loc, diag::err_typecheck_arithmetic_incomplete_type,
               lex->getType().getAsString(), lex->getSourceRange());
          return QualType();
        }
      }
      return PExp->getType();
    }
  }

  return InvalidOperands(loc, lex, rex);
}

// C99 6.5.6
QualType Sema::CheckSubtractionOperands(Expr *&lex, Expr *&rex,
                                        SourceLocation loc, bool isCompAssign) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(loc, lex, rex);
    
  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);
  
  // Enforce type constraints: C99 6.5.6p3.
  
  // Handle the common case first (both operands are arithmetic).
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    return compType;
  
  // Either ptr - int   or   ptr - ptr.
  if (const PointerType *LHSPTy = lex->getType()->getAsPointerType()) {
    QualType lpointee = LHSPTy->getPointeeType();
    
    // The LHS must be an object type, not incomplete, function, etc.
    if (!lpointee->isObjectType()) {
      // Handle the GNU void* extension.
      if (lpointee->isVoidType()) {
        Diag(loc, diag::ext_gnu_void_ptr, 
             lex->getSourceRange(), rex->getSourceRange());
      } else {
        Diag(loc, diag::err_typecheck_sub_ptr_object,
             lex->getType().getAsString(), lex->getSourceRange());
        return QualType();
      }
    }

    // The result type of a pointer-int computation is the pointer type.
    if (rex->getType()->isIntegerType())
      return lex->getType();
    
    // Handle pointer-pointer subtractions.
    if (const PointerType *RHSPTy = rex->getType()->getAsPointerType()) {
      QualType rpointee = RHSPTy->getPointeeType();
      
      // RHS must be an object type, unless void (GNU).
      if (!rpointee->isObjectType()) {
        // Handle the GNU void* extension.
        if (rpointee->isVoidType()) {
          if (!lpointee->isVoidType())
            Diag(loc, diag::ext_gnu_void_ptr, 
                 lex->getSourceRange(), rex->getSourceRange());
        } else {
          Diag(loc, diag::err_typecheck_sub_ptr_object,
               rex->getType().getAsString(), rex->getSourceRange());
          return QualType();
        }
      }
      
      // Pointee types must be compatible.
      if (!Context.typesAreCompatible(
              Context.getCanonicalType(lpointee).getUnqualifiedType(), 
              Context.getCanonicalType(rpointee).getUnqualifiedType())) {
        Diag(loc, diag::err_typecheck_sub_ptr_compatible,
             lex->getType().getAsString(), rex->getType().getAsString(),
             lex->getSourceRange(), rex->getSourceRange());
        return QualType();
      }
      
      return Context.getPointerDiffType();
    }
  }
  
  return InvalidOperands(loc, lex, rex);
}

// C99 6.5.7
QualType Sema::CheckShiftOperands(Expr *&lex, Expr *&rex, SourceLocation loc,
                                  bool isCompAssign) {
  // C99 6.5.7p2: Each of the operands shall have integer type.
  if (!lex->getType()->isIntegerType() || !rex->getType()->isIntegerType())
    return InvalidOperands(loc, lex, rex);
  
  // Shifts don't perform usual arithmetic conversions, they just do integer
  // promotions on each operand. C99 6.5.7p3
  if (!isCompAssign)
    UsualUnaryConversions(lex);
  UsualUnaryConversions(rex);
  
  // "The type of the result is that of the promoted left operand."
  return lex->getType();
}

static bool areComparableObjCInterfaces(QualType LHS, QualType RHS,
                                        ASTContext& Context) {
  const ObjCInterfaceType* LHSIface = LHS->getAsObjCInterfaceType();
  const ObjCInterfaceType* RHSIface = RHS->getAsObjCInterfaceType();
  // ID acts sort of like void* for ObjC interfaces
  if (LHSIface && Context.isObjCIdType(RHS))
    return true;
  if (RHSIface && Context.isObjCIdType(LHS))
    return true;
  if (!LHSIface || !RHSIface)
    return false;
  return Context.canAssignObjCInterfaces(LHSIface, RHSIface) ||
         Context.canAssignObjCInterfaces(RHSIface, LHSIface);
}

// C99 6.5.8
QualType Sema::CheckCompareOperands(Expr *&lex, Expr *&rex, SourceLocation loc,
                                    bool isRelational) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorCompareOperands(lex, rex, loc, isRelational);
  
  // C99 6.5.8p3 / C99 6.5.9p4
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    UsualArithmeticConversions(lex, rex);
  else {
    UsualUnaryConversions(lex);
    UsualUnaryConversions(rex);
  }
  QualType lType = lex->getType();
  QualType rType = rex->getType();
  
  // For non-floating point types, check for self-comparisons of the form
  // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
  // often indicate logic errors in the program.
  if (!lType->isFloatingType()) {
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(lex->IgnoreParens()))
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(rex->IgnoreParens()))
        if (DRL->getDecl() == DRR->getDecl())
          Diag(loc, diag::warn_selfcomparison);      
  }
  
  if (isRelational) {
    if (lType->isRealType() && rType->isRealType())
      return Context.IntTy;
  } else {
    // Check for comparisons of floating point operands using != and ==.
    if (lType->isFloatingType()) {
      assert (rType->isFloatingType());
      CheckFloatComparison(loc,lex,rex);
    }
    
    if (lType->isArithmeticType() && rType->isArithmeticType())
      return Context.IntTy;
  }
  
  bool LHSIsNull = lex->isNullPointerConstant(Context);
  bool RHSIsNull = rex->isNullPointerConstant(Context);
  
  // All of the following pointer related warnings are GCC extensions, except
  // when handling null pointer constants. One day, we can consider making them
  // errors (when -pedantic-errors is enabled).
  if (lType->isPointerType() && rType->isPointerType()) { // C99 6.5.8p2
    QualType LCanPointeeTy =
      Context.getCanonicalType(lType->getAsPointerType()->getPointeeType());
    QualType RCanPointeeTy =
      Context.getCanonicalType(rType->getAsPointerType()->getPointeeType());
    
    if (!LHSIsNull && !RHSIsNull &&                       // C99 6.5.9p2
        !LCanPointeeTy->isVoidType() && !RCanPointeeTy->isVoidType() &&
        !Context.typesAreCompatible(LCanPointeeTy.getUnqualifiedType(),
                                    RCanPointeeTy.getUnqualifiedType()) &&
        !areComparableObjCInterfaces(LCanPointeeTy, RCanPointeeTy, Context)) {
      Diag(loc, diag::ext_typecheck_comparison_of_distinct_pointers,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    }
    ImpCastExprToType(rex, lType); // promote the pointer to pointer
    return Context.IntTy;
  }
  // Handle block pointer types.
  if (lType->isBlockPointerType() && rType->isBlockPointerType()) {
    QualType lpointee = lType->getAsBlockPointerType()->getPointeeType();
    QualType rpointee = rType->getAsBlockPointerType()->getPointeeType();
    
    if (!LHSIsNull && !RHSIsNull &&
        !Context.typesAreBlockCompatible(lpointee, rpointee)) {
      Diag(loc, diag::err_typecheck_comparison_of_distinct_blocks,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    }
    ImpCastExprToType(rex, lType); // promote the pointer to pointer
    return Context.IntTy;
  }

  if ((lType->isObjCQualifiedIdType() || rType->isObjCQualifiedIdType())) {
    if (ObjCQualifiedIdTypesAreCompatible(lType, rType, true)) {
      ImpCastExprToType(rex, lType);
      return Context.IntTy;
    }
  }
  if ((lType->isPointerType() || lType->isObjCQualifiedIdType()) && 
       rType->isIntegerType()) {
    if (!RHSIsNull)
      Diag(loc, diag::ext_typecheck_comparison_of_pointer_integer,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    ImpCastExprToType(rex, lType); // promote the integer to pointer
    return Context.IntTy;
  }
  if (lType->isIntegerType() && 
      (rType->isPointerType() || rType->isObjCQualifiedIdType())) {
    if (!LHSIsNull)
      Diag(loc, diag::ext_typecheck_comparison_of_pointer_integer,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    ImpCastExprToType(lex, rType); // promote the integer to pointer
    return Context.IntTy;
  }
  // Handle block pointers.
  if (lType->isBlockPointerType() && rType->isIntegerType()) {
    if (!RHSIsNull)
      Diag(loc, diag::ext_typecheck_comparison_of_pointer_integer,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    ImpCastExprToType(rex, lType); // promote the integer to pointer
    return Context.IntTy;
  }
  if (lType->isIntegerType() && rType->isBlockPointerType()) {
    if (!LHSIsNull)
      Diag(loc, diag::ext_typecheck_comparison_of_pointer_integer,
           lType.getAsString(), rType.getAsString(),
           lex->getSourceRange(), rex->getSourceRange());
    ImpCastExprToType(lex, rType); // promote the integer to pointer
    return Context.IntTy;
  }
  return InvalidOperands(loc, lex, rex);
}

/// CheckVectorCompareOperands - vector comparisons are a clang extension that
/// operates on extended vector types.  Instead of producing an IntTy result, 
/// like a scalar comparison, a vector comparison produces a vector of integer
/// types.
QualType Sema::CheckVectorCompareOperands(Expr *&lex, Expr *&rex,
                                          SourceLocation loc,
                                          bool isRelational) {
  // Check to make sure we're operating on vectors of the same type and width,
  // Allowing one side to be a scalar of element type.
  QualType vType = CheckVectorOperands(loc, lex, rex);
  if (vType.isNull())
    return vType;
  
  QualType lType = lex->getType();
  QualType rType = rex->getType();
  
  // For non-floating point types, check for self-comparisons of the form
  // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
  // often indicate logic errors in the program.
  if (!lType->isFloatingType()) {
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(lex->IgnoreParens()))
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(rex->IgnoreParens()))
        if (DRL->getDecl() == DRR->getDecl())
          Diag(loc, diag::warn_selfcomparison);      
  }
  
  // Check for comparisons of floating point operands using != and ==.
  if (!isRelational && lType->isFloatingType()) {
    assert (rType->isFloatingType());
    CheckFloatComparison(loc,lex,rex);
  }
  
  // Return the type for the comparison, which is the same as vector type for
  // integer vectors, or an integer type of identical size and number of
  // elements for floating point vectors.
  if (lType->isIntegerType())
    return lType;
  
  const VectorType *VTy = lType->getAsVectorType();

  // FIXME: need to deal with non-32b int / non-64b long long
  unsigned TypeSize = Context.getTypeSize(VTy->getElementType());
  if (TypeSize == 32) {
    return Context.getExtVectorType(Context.IntTy, VTy->getNumElements());
  }
  assert(TypeSize == 64 && "Unhandled vector element size in vector compare");
  return Context.getExtVectorType(Context.LongLongTy, VTy->getNumElements());
}

inline QualType Sema::CheckBitwiseOperands(
  Expr *&lex, Expr *&rex, SourceLocation loc, bool isCompAssign) 
{
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(loc, lex, rex);

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);
  
  if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
    return compType;
  return InvalidOperands(loc, lex, rex);
}

inline QualType Sema::CheckLogicalOperands( // C99 6.5.[13,14]
  Expr *&lex, Expr *&rex, SourceLocation loc) 
{
  UsualUnaryConversions(lex);
  UsualUnaryConversions(rex);
  
  if (lex->getType()->isScalarType() && rex->getType()->isScalarType())
    return Context.IntTy;
  return InvalidOperands(loc, lex, rex);
}

inline QualType Sema::CheckAssignmentOperands( // C99 6.5.16.1
  Expr *lex, Expr *&rex, SourceLocation loc, QualType compoundType) 
{
  QualType lhsType = lex->getType();
  QualType rhsType = compoundType.isNull() ? rex->getType() : compoundType;
  Expr::isModifiableLvalueResult mlval = lex->isModifiableLvalue(Context); 

  switch (mlval) { // C99 6.5.16p2
  case Expr::MLV_Valid: 
    break;
  case Expr::MLV_ConstQualified:
    Diag(loc, diag::err_typecheck_assign_const, lex->getSourceRange());
    return QualType();
  case Expr::MLV_ArrayType: 
    Diag(loc, diag::err_typecheck_array_not_modifiable_lvalue,
         lhsType.getAsString(), lex->getSourceRange());
    return QualType(); 
  case Expr::MLV_NotObjectType: 
    Diag(loc, diag::err_typecheck_non_object_not_modifiable_lvalue,
         lhsType.getAsString(), lex->getSourceRange());
    return QualType();
  case Expr::MLV_InvalidExpression:
    Diag(loc, diag::err_typecheck_expression_not_modifiable_lvalue,
         lex->getSourceRange());
    return QualType();
  case Expr::MLV_IncompleteType:
  case Expr::MLV_IncompleteVoidType:
    Diag(loc, diag::err_typecheck_incomplete_type_not_modifiable_lvalue,
         lhsType.getAsString(), lex->getSourceRange());
    return QualType();
  case Expr::MLV_DuplicateVectorComponents:
    Diag(loc, diag::err_typecheck_duplicate_vector_components_not_mlvalue,
         lex->getSourceRange());
    return QualType();
  }

  AssignConvertType ConvTy;
  if (compoundType.isNull()) {
    // Simple assignment "x = y".
    ConvTy = CheckSingleAssignmentConstraints(lhsType, rex);
    
    // If the RHS is a unary plus or minus, check to see if they = and + are
    // right next to each other.  If so, the user may have typo'd "x =+ 4"
    // instead of "x += 4".
    Expr *RHSCheck = rex;
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(RHSCheck))
      RHSCheck = ICE->getSubExpr();
    if (UnaryOperator *UO = dyn_cast<UnaryOperator>(RHSCheck)) {
      if ((UO->getOpcode() == UnaryOperator::Plus ||
           UO->getOpcode() == UnaryOperator::Minus) &&
          loc.isFileID() && UO->getOperatorLoc().isFileID() &&
          // Only if the two operators are exactly adjacent.
          loc.getFileLocWithOffset(1) == UO->getOperatorLoc())
        Diag(loc, diag::warn_not_compound_assign,
             UO->getOpcode() == UnaryOperator::Plus ? "+" : "-",
             SourceRange(UO->getOperatorLoc(), UO->getOperatorLoc()));
    }
  } else {
    // Compound assignment "x += y"
    ConvTy = CheckCompoundAssignmentConstraints(lhsType, rhsType);
  }

  if (DiagnoseAssignmentResult(ConvTy, loc, lhsType, rhsType,
                               rex, "assigning"))
    return QualType();
  
  // C99 6.5.16p3: The type of an assignment expression is the type of the
  // left operand unless the left operand has qualified type, in which case
  // it is the unqualified version of the type of the left operand. 
  // C99 6.5.16.1p2: In simple assignment, the value of the right operand
  // is converted to the type of the assignment expression (above).
  // C++ 5.17p1: the type of the assignment expression is that of its left
  // oprdu.
  return lhsType.getUnqualifiedType();
}

inline QualType Sema::CheckCommaOperands( // C99 6.5.17
  Expr *&lex, Expr *&rex, SourceLocation loc) {
  
  // Comma performs lvalue conversion (C99 6.3.2.1), but not unary conversions.
  DefaultFunctionArrayConversion(rex);
  return rex->getType();
}

/// CheckIncrementDecrementOperand - unlike most "Check" methods, this routine
/// doesn't need to call UsualUnaryConversions or UsualArithmeticConversions.
QualType Sema::CheckIncrementDecrementOperand(Expr *op, SourceLocation OpLoc) {
  QualType resType = op->getType();
  assert(!resType.isNull() && "no type for increment/decrement expression");

  // C99 6.5.2.4p1: We allow complex as a GCC extension.
  if (const PointerType *pt = resType->getAsPointerType()) {
    if (pt->getPointeeType()->isVoidType()) {
      Diag(OpLoc, diag::ext_gnu_void_ptr, op->getSourceRange());
    } else if (!pt->getPointeeType()->isObjectType()) {
      // C99 6.5.2.4p2, 6.5.6p2
      Diag(OpLoc, diag::err_typecheck_arithmetic_incomplete_type,
           resType.getAsString(), op->getSourceRange());
      return QualType();
    }
  } else if (!resType->isRealType()) {
    if (resType->isComplexType()) 
      // C99 does not support ++/-- on complex types.
      Diag(OpLoc, diag::ext_integer_increment_complex,
           resType.getAsString(), op->getSourceRange());
    else {
      Diag(OpLoc, diag::err_typecheck_illegal_increment_decrement,
           resType.getAsString(), op->getSourceRange());
      return QualType();
    }
  }
  // At this point, we know we have a real, complex or pointer type. 
  // Now make sure the operand is a modifiable lvalue.
  Expr::isModifiableLvalueResult mlval = op->isModifiableLvalue(Context);
  if (mlval != Expr::MLV_Valid) {
    // FIXME: emit a more precise diagnostic...
    Diag(OpLoc, diag::err_typecheck_invalid_lvalue_incr_decr,
         op->getSourceRange());
    return QualType();
  }
  return resType;
}

/// getPrimaryDecl - Helper function for CheckAddressOfOperand().
/// This routine allows us to typecheck complex/recursive expressions
/// where the declaration is needed for type checking. We only need to
/// handle cases when the expression references a function designator
/// or is an lvalue. Here are some examples:
///  - &(x) => x
///  - &*****f => f for f a function designator.
///  - &s.xx => s
///  - &s.zz[1].yy -> s, if zz is an array
///  - *(x + 1) -> x, if x is an array
///  - &"123"[2] -> 0
///  - & __real__ x -> x
static ValueDecl *getPrimaryDecl(Expr *E) {
  switch (E->getStmtClass()) {
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(E)->getDecl();
  case Stmt::MemberExprClass:
    // Fields cannot be declared with a 'register' storage class.
    // &X->f is always ok, even if X is declared register.
    if (cast<MemberExpr>(E)->isArrow())
      return 0;
    return getPrimaryDecl(cast<MemberExpr>(E)->getBase());
  case Stmt::ArraySubscriptExprClass: {
    // &X[4] and &4[X] refers to X if X is not a pointer.
  
    ValueDecl *VD = getPrimaryDecl(cast<ArraySubscriptExpr>(E)->getBase());
    if (!VD || VD->getType()->isPointerType())
      return 0;
    else
      return VD;
  }
  case Stmt::UnaryOperatorClass: {
    UnaryOperator *UO = cast<UnaryOperator>(E);
    
    switch(UO->getOpcode()) {
    case UnaryOperator::Deref: {
      // *(X + 1) refers to X if X is not a pointer.
      ValueDecl *VD = getPrimaryDecl(UO->getSubExpr());
      if (!VD || VD->getType()->isPointerType())
        return 0;
      return VD;
    }
    case UnaryOperator::Real:
    case UnaryOperator::Imag:
    case UnaryOperator::Extension:
      return getPrimaryDecl(UO->getSubExpr());
    default:
      return 0;
    }
  }
  case Stmt::BinaryOperatorClass: {
    BinaryOperator *BO = cast<BinaryOperator>(E);

    // Handle cases involving pointer arithmetic. The result of an
    // Assign or AddAssign is not an lvalue so they can be ignored.

    // (x + n) or (n + x) => x
    if (BO->getOpcode() == BinaryOperator::Add) {
      if (BO->getLHS()->getType()->isPointerType()) {
        return getPrimaryDecl(BO->getLHS());
      } else if (BO->getRHS()->getType()->isPointerType()) {
        return getPrimaryDecl(BO->getRHS());
      }
    }

    return 0;
  }
  case Stmt::ParenExprClass:
    return getPrimaryDecl(cast<ParenExpr>(E)->getSubExpr());
  case Stmt::ImplicitCastExprClass:
    // &X[4] when X is an array, has an implicit cast from array to pointer.
    return getPrimaryDecl(cast<ImplicitCastExpr>(E)->getSubExpr());
  default:
    return 0;
  }
}

/// CheckAddressOfOperand - The operand of & must be either a function
/// designator or an lvalue designating an object. If it is an lvalue, the 
/// object cannot be declared with storage class register or be a bit field.
/// Note: The usual conversions are *not* applied to the operand of the & 
/// operator (C99 6.3.2.1p[2-4]), and its result is never an lvalue.
QualType Sema::CheckAddressOfOperand(Expr *op, SourceLocation OpLoc) {
  if (getLangOptions().C99) {
    // Implement C99-only parts of addressof rules.
    if (UnaryOperator* uOp = dyn_cast<UnaryOperator>(op)) {
      if (uOp->getOpcode() == UnaryOperator::Deref)
        // Per C99 6.5.3.2, the address of a deref always returns a valid result
        // (assuming the deref expression is valid).
        return uOp->getSubExpr()->getType();
    }
    // Technically, there should be a check for array subscript
    // expressions here, but the result of one is always an lvalue anyway.
  }
  ValueDecl *dcl = getPrimaryDecl(op);
  Expr::isLvalueResult lval = op->isLvalue(Context);
  
  if (lval != Expr::LV_Valid) { // C99 6.5.3.2p1
    if (!dcl || !isa<FunctionDecl>(dcl)) {// allow function designators
      // FIXME: emit more specific diag...
      Diag(OpLoc, diag::err_typecheck_invalid_lvalue_addrof, 
           op->getSourceRange());
      return QualType();
    }
  } else if (MemberExpr *MemExpr = dyn_cast<MemberExpr>(op)) { // C99 6.5.3.2p1
    if (MemExpr->getMemberDecl()->isBitField()) {
      Diag(OpLoc, diag::err_typecheck_address_of, 
           std::string("bit-field"), op->getSourceRange());
      return QualType();
    }
  // Check for Apple extension for accessing vector components.
  } else if (isa<ArraySubscriptExpr>(op) &&
           cast<ArraySubscriptExpr>(op)->getBase()->getType()->isVectorType()) {
    Diag(OpLoc, diag::err_typecheck_address_of, 
         std::string("vector"), op->getSourceRange());
    return QualType();
  } else if (dcl) { // C99 6.5.3.2p1
    // We have an lvalue with a decl. Make sure the decl is not declared 
    // with the register storage-class specifier.
    if (const VarDecl *vd = dyn_cast<VarDecl>(dcl)) {
      if (vd->getStorageClass() == VarDecl::Register) {
        Diag(OpLoc, diag::err_typecheck_address_of, 
             std::string("register variable"), op->getSourceRange());
        return QualType();
      }
    } else 
      assert(0 && "Unknown/unexpected decl type");
  }
  
  // If the operand has type "type", the result has type "pointer to type".
  return Context.getPointerType(op->getType());
}

QualType Sema::CheckIndirectionOperand(Expr *op, SourceLocation OpLoc) {
  UsualUnaryConversions(op);
  QualType qType = op->getType();
  
  if (const PointerType *PT = qType->getAsPointerType()) {
    // Note that per both C89 and C99, this is always legal, even
    // if ptype is an incomplete type or void.
    // It would be possible to warn about dereferencing a
    // void pointer, but it's completely well-defined,
    // and such a warning is unlikely to catch any mistakes.
    return PT->getPointeeType();
  }
  Diag(OpLoc, diag::err_typecheck_indirection_requires_pointer, 
       qType.getAsString(), op->getSourceRange());
  return QualType();
}

static inline BinaryOperator::Opcode ConvertTokenKindToBinaryOpcode(
  tok::TokenKind Kind) {
  BinaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case tok::star:                 Opc = BinaryOperator::Mul; break;
  case tok::slash:                Opc = BinaryOperator::Div; break;
  case tok::percent:              Opc = BinaryOperator::Rem; break;
  case tok::plus:                 Opc = BinaryOperator::Add; break;
  case tok::minus:                Opc = BinaryOperator::Sub; break;
  case tok::lessless:             Opc = BinaryOperator::Shl; break;
  case tok::greatergreater:       Opc = BinaryOperator::Shr; break;
  case tok::lessequal:            Opc = BinaryOperator::LE; break;
  case tok::less:                 Opc = BinaryOperator::LT; break;
  case tok::greaterequal:         Opc = BinaryOperator::GE; break;
  case tok::greater:              Opc = BinaryOperator::GT; break;
  case tok::exclaimequal:         Opc = BinaryOperator::NE; break;
  case tok::equalequal:           Opc = BinaryOperator::EQ; break;
  case tok::amp:                  Opc = BinaryOperator::And; break;
  case tok::caret:                Opc = BinaryOperator::Xor; break;
  case tok::pipe:                 Opc = BinaryOperator::Or; break;
  case tok::ampamp:               Opc = BinaryOperator::LAnd; break;
  case tok::pipepipe:             Opc = BinaryOperator::LOr; break;
  case tok::equal:                Opc = BinaryOperator::Assign; break;
  case tok::starequal:            Opc = BinaryOperator::MulAssign; break;
  case tok::slashequal:           Opc = BinaryOperator::DivAssign; break;
  case tok::percentequal:         Opc = BinaryOperator::RemAssign; break;
  case tok::plusequal:            Opc = BinaryOperator::AddAssign; break;
  case tok::minusequal:           Opc = BinaryOperator::SubAssign; break;
  case tok::lesslessequal:        Opc = BinaryOperator::ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BinaryOperator::ShrAssign; break;
  case tok::ampequal:             Opc = BinaryOperator::AndAssign; break;
  case tok::caretequal:           Opc = BinaryOperator::XorAssign; break;
  case tok::pipeequal:            Opc = BinaryOperator::OrAssign; break;
  case tok::comma:                Opc = BinaryOperator::Comma; break;
  }
  return Opc;
}

static inline UnaryOperator::Opcode ConvertTokenKindToUnaryOpcode(
  tok::TokenKind Kind) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:     Opc = UnaryOperator::PreInc; break;
  case tok::minusminus:   Opc = UnaryOperator::PreDec; break;
  case tok::amp:          Opc = UnaryOperator::AddrOf; break;
  case tok::star:         Opc = UnaryOperator::Deref; break;
  case tok::plus:         Opc = UnaryOperator::Plus; break;
  case tok::minus:        Opc = UnaryOperator::Minus; break;
  case tok::tilde:        Opc = UnaryOperator::Not; break;
  case tok::exclaim:      Opc = UnaryOperator::LNot; break;
  case tok::kw_sizeof:    Opc = UnaryOperator::SizeOf; break;
  case tok::kw___alignof: Opc = UnaryOperator::AlignOf; break;
  case tok::kw___real:    Opc = UnaryOperator::Real; break;
  case tok::kw___imag:    Opc = UnaryOperator::Imag; break;
  case tok::kw___extension__: Opc = UnaryOperator::Extension; break;
  }
  return Opc;
}

// Binary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ActOnBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                    ExprTy *LHS, ExprTy *RHS) {
  BinaryOperator::Opcode Opc = ConvertTokenKindToBinaryOpcode(Kind);
  Expr *lhs = (Expr *)LHS, *rhs = (Expr*)RHS;

  assert((lhs != 0) && "ActOnBinOp(): missing left expression");
  assert((rhs != 0) && "ActOnBinOp(): missing right expression");

  QualType ResultTy;  // Result type of the binary operator.
  QualType CompTy;    // Computation type for compound assignments (e.g. '+=')
  
  switch (Opc) {
  default:
    assert(0 && "Unknown binary expr!");
  case BinaryOperator::Assign:
    ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, QualType());
    break;
  case BinaryOperator::Mul: 
  case BinaryOperator::Div:
    ResultTy = CheckMultiplyDivideOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::Rem:
    ResultTy = CheckRemainderOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::Add:
    ResultTy = CheckAdditionOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::Sub:
    ResultTy = CheckSubtractionOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::Shl: 
  case BinaryOperator::Shr:
    ResultTy = CheckShiftOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::LE:
  case BinaryOperator::LT:
  case BinaryOperator::GE:
  case BinaryOperator::GT:
    ResultTy = CheckCompareOperands(lhs, rhs, TokLoc, true);
    break;
  case BinaryOperator::EQ:
  case BinaryOperator::NE:
    ResultTy = CheckCompareOperands(lhs, rhs, TokLoc, false);
    break;
  case BinaryOperator::And:
  case BinaryOperator::Xor:
  case BinaryOperator::Or:
    ResultTy = CheckBitwiseOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::LAnd:
  case BinaryOperator::LOr:
    ResultTy = CheckLogicalOperands(lhs, rhs, TokLoc);
    break;
  case BinaryOperator::MulAssign:
  case BinaryOperator::DivAssign:
    CompTy = CheckMultiplyDivideOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::RemAssign:
    CompTy = CheckRemainderOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::AddAssign:
    CompTy = CheckAdditionOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::SubAssign:
    CompTy = CheckSubtractionOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::ShlAssign:
  case BinaryOperator::ShrAssign:
    CompTy = CheckShiftOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::AndAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::OrAssign:
    CompTy = CheckBitwiseOperands(lhs, rhs, TokLoc, true);
    if (!CompTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, TokLoc, CompTy);
    break;
  case BinaryOperator::Comma:
    ResultTy = CheckCommaOperands(lhs, rhs, TokLoc);
    break;
  }
  if (ResultTy.isNull())
    return true;
  if (CompTy.isNull())
    return new BinaryOperator(lhs, rhs, Opc, ResultTy, TokLoc);
  else
    return new CompoundAssignOperator(lhs, rhs, Opc, ResultTy, CompTy, TokLoc);
}

// Unary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ActOnUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                      ExprTy *input) {
  Expr *Input = (Expr*)input;
  UnaryOperator::Opcode Opc = ConvertTokenKindToUnaryOpcode(Op);
  QualType resultType;
  switch (Opc) {
  default:
    assert(0 && "Unimplemented unary expr!");
  case UnaryOperator::PreInc:
  case UnaryOperator::PreDec:
    resultType = CheckIncrementDecrementOperand(Input, OpLoc);
    break;
  case UnaryOperator::AddrOf: 
    resultType = CheckAddressOfOperand(Input, OpLoc);
    break;
  case UnaryOperator::Deref: 
    DefaultFunctionArrayConversion(Input);
    resultType = CheckIndirectionOperand(Input, OpLoc);
    break;
  case UnaryOperator::Plus:
  case UnaryOperator::Minus:
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    if (!resultType->isArithmeticType())  // C99 6.5.3.3p1
      return Diag(OpLoc, diag::err_typecheck_unary_expr, 
                  resultType.getAsString());
    break;
  case UnaryOperator::Not: // bitwise complement
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    // C99 6.5.3.3p1. We allow complex int and float as a GCC extension.
    if (resultType->isComplexType() || resultType->isComplexIntegerType())
      // C99 does not support '~' for complex conjugation.
      Diag(OpLoc, diag::ext_integer_complement_complex,
           resultType.getAsString(), Input->getSourceRange());
    else if (!resultType->isIntegerType())
      return Diag(OpLoc, diag::err_typecheck_unary_expr,
                  resultType.getAsString(), Input->getSourceRange());
    break;
  case UnaryOperator::LNot: // logical negation
    // Unlike +/-/~, integer promotions aren't done here (C99 6.5.3.3p5).
    DefaultFunctionArrayConversion(Input);
    resultType = Input->getType();
    if (!resultType->isScalarType()) // C99 6.5.3.3p1
      return Diag(OpLoc, diag::err_typecheck_unary_expr,
                  resultType.getAsString());
    // LNot always has type int. C99 6.5.3.3p5.
    resultType = Context.IntTy;
    break;
  case UnaryOperator::SizeOf:
    resultType = CheckSizeOfAlignOfOperand(Input->getType(), OpLoc,
                                           Input->getSourceRange(), true);
    break;
  case UnaryOperator::AlignOf:
    resultType = CheckSizeOfAlignOfOperand(Input->getType(), OpLoc,
                                           Input->getSourceRange(), false);
    break;
  case UnaryOperator::Real:
  case UnaryOperator::Imag:
    resultType = CheckRealImagOperand(Input, OpLoc);
    break;
  case UnaryOperator::Extension:
    resultType = Input->getType();
    break;
  }
  if (resultType.isNull())
    return true;
  return new UnaryOperator(Input, Opc, resultType, OpLoc);
}

/// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
Sema::ExprResult Sema::ActOnAddrLabel(SourceLocation OpLoc, 
                                      SourceLocation LabLoc,
                                      IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = LabelMap[LabelII];
  
  // If we haven't seen this label yet, create a forward reference. It
  // will be validated and/or cleaned up in ActOnFinishFunctionBody.
  if (LabelDecl == 0)
    LabelDecl = new LabelStmt(LabLoc, LabelII, 0);
  
  // Create the AST node.  The address of a label always has type 'void*'.
  return new AddrLabelExpr(OpLoc, LabLoc, LabelDecl,
                           Context.getPointerType(Context.VoidTy));
}

Sema::ExprResult Sema::ActOnStmtExpr(SourceLocation LPLoc, StmtTy *substmt,
                                     SourceLocation RPLoc) { // "({..})"
  Stmt *SubStmt = static_cast<Stmt*>(substmt);
  assert(SubStmt && isa<CompoundStmt>(SubStmt) && "Invalid action invocation!");
  CompoundStmt *Compound = cast<CompoundStmt>(SubStmt);

  // FIXME: there are a variety of strange constraints to enforce here, for
  // example, it is not possible to goto into a stmt expression apparently.
  // More semantic analysis is needed.
  
  // FIXME: the last statement in the compount stmt has its value used.  We
  // should not warn about it being unused.

  // If there are sub stmts in the compound stmt, take the type of the last one
  // as the type of the stmtexpr.
  QualType Ty = Context.VoidTy;
  
  if (!Compound->body_empty()) {
    Stmt *LastStmt = Compound->body_back();
    // If LastStmt is a label, skip down through into the body.
    while (LabelStmt *Label = dyn_cast<LabelStmt>(LastStmt))
      LastStmt = Label->getSubStmt();
    
    if (Expr *LastExpr = dyn_cast<Expr>(LastStmt))
      Ty = LastExpr->getType();
  }
  
  return new StmtExpr(Compound, Ty, LPLoc, RPLoc);
}

Sema::ExprResult Sema::ActOnBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                            SourceLocation TypeLoc,
                                            TypeTy *argty,
                                            OffsetOfComponent *CompPtr,
                                            unsigned NumComponents,
                                            SourceLocation RPLoc) {
  QualType ArgTy = QualType::getFromOpaquePtr(argty);
  assert(!ArgTy.isNull() && "Missing type argument!");
  
  // We must have at least one component that refers to the type, and the first
  // one is known to be a field designator.  Verify that the ArgTy represents
  // a struct/union/class.
  if (!ArgTy->isRecordType())
    return Diag(TypeLoc, diag::err_offsetof_record_type,ArgTy.getAsString());
  
  // Otherwise, create a compound literal expression as the base, and
  // iteratively process the offsetof designators.
  Expr *Res = new CompoundLiteralExpr(SourceLocation(), ArgTy, 0, false);
  
  // offsetof with non-identifier designators (e.g. "offsetof(x, a.b[c])") are a
  // GCC extension, diagnose them.
  if (NumComponents != 1)
    Diag(BuiltinLoc, diag::ext_offsetof_extended_field_designator,
         SourceRange(CompPtr[1].LocStart, CompPtr[NumComponents-1].LocEnd));
  
  for (unsigned i = 0; i != NumComponents; ++i) {
    const OffsetOfComponent &OC = CompPtr[i];
    if (OC.isBrackets) {
      // Offset of an array sub-field.  TODO: Should we allow vector elements?
      const ArrayType *AT = Context.getAsArrayType(Res->getType());
      if (!AT) {
        delete Res;
        return Diag(OC.LocEnd, diag::err_offsetof_array_type,
                    Res->getType().getAsString());
      }
      
      // FIXME: C++: Verify that operator[] isn't overloaded.

      // C99 6.5.2.1p1
      Expr *Idx = static_cast<Expr*>(OC.U.E);
      if (!Idx->getType()->isIntegerType())
        return Diag(Idx->getLocStart(), diag::err_typecheck_subscript,
                    Idx->getSourceRange());
      
      Res = new ArraySubscriptExpr(Res, Idx, AT->getElementType(), OC.LocEnd);
      continue;
    }
    
    const RecordType *RC = Res->getType()->getAsRecordType();
    if (!RC) {
      delete Res;
      return Diag(OC.LocEnd, diag::err_offsetof_record_type,
                  Res->getType().getAsString());
    }
      
    // Get the decl corresponding to this.
    RecordDecl *RD = RC->getDecl();
    FieldDecl *MemberDecl = RD->getMember(OC.U.IdentInfo);
    if (!MemberDecl)
      return Diag(BuiltinLoc, diag::err_typecheck_no_member,
                  OC.U.IdentInfo->getName(),
                  SourceRange(OC.LocStart, OC.LocEnd));
    
    // FIXME: C++: Verify that MemberDecl isn't a static field.
    // FIXME: Verify that MemberDecl isn't a bitfield.
    // MemberDecl->getType() doesn't get the right qualifiers, but it doesn't
    // matter here.
    Res = new MemberExpr(Res, false, MemberDecl, OC.LocEnd, MemberDecl->getType());
  }
  
  return new UnaryOperator(Res, UnaryOperator::OffsetOf, Context.getSizeType(),
                           BuiltinLoc);
}


Sema::ExprResult Sema::ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc, 
                                                TypeTy *arg1, TypeTy *arg2,
                                                SourceLocation RPLoc) {
  QualType argT1 = QualType::getFromOpaquePtr(arg1);
  QualType argT2 = QualType::getFromOpaquePtr(arg2);
  
  assert((!argT1.isNull() && !argT2.isNull()) && "Missing type argument(s)");
  
  return new TypesCompatibleExpr(Context.IntTy, BuiltinLoc, argT1, argT2,RPLoc);
}

Sema::ExprResult Sema::ActOnChooseExpr(SourceLocation BuiltinLoc, ExprTy *cond, 
                                       ExprTy *expr1, ExprTy *expr2,
                                       SourceLocation RPLoc) {
  Expr *CondExpr = static_cast<Expr*>(cond);
  Expr *LHSExpr = static_cast<Expr*>(expr1);
  Expr *RHSExpr = static_cast<Expr*>(expr2);
  
  assert((CondExpr && LHSExpr && RHSExpr) && "Missing type argument(s)");

  // The conditional expression is required to be a constant expression.
  llvm::APSInt condEval(32);
  SourceLocation ExpLoc;
  if (!CondExpr->isIntegerConstantExpr(condEval, Context, &ExpLoc))
    return Diag(ExpLoc, diag::err_typecheck_choose_expr_requires_constant,
                 CondExpr->getSourceRange());

  // If the condition is > zero, then the AST type is the same as the LSHExpr.
  QualType resType = condEval.getZExtValue() ? LHSExpr->getType() : 
                                               RHSExpr->getType();
  return new ChooseExpr(BuiltinLoc, CondExpr, LHSExpr, RHSExpr, resType, RPLoc);
}

//===----------------------------------------------------------------------===//
// Clang Extensions.
//===----------------------------------------------------------------------===//

/// ActOnBlockStart - This callback is invoked when a block literal is started.
void Sema::ActOnBlockStart(SourceLocation CaretLoc, Scope *BlockScope,
                           Declarator &ParamInfo) {
  // Analyze block parameters.
  BlockSemaInfo *BSI = new BlockSemaInfo();
  
  // Add BSI to CurBlock.
  BSI->PrevBlockInfo = CurBlock;
  CurBlock = BSI;
  
  BSI->ReturnType = 0;
  BSI->TheScope = BlockScope;
  
  // Analyze arguments to block.
  assert(ParamInfo.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = ParamInfo.getTypeObject(0).Fun;
  
  BSI->hasPrototype = FTI.hasPrototype;
  BSI->isVariadic = true;
  
  // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs function that takes
  // no arguments, not a function that takes a single void argument.
  if (FTI.hasPrototype &&
      FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
      (!((ParmVarDecl *)FTI.ArgInfo[0].Param)->getType().getCVRQualifiers() &&
        ((ParmVarDecl *)FTI.ArgInfo[0].Param)->getType()->isVoidType())) {
    // empty arg list, don't push any params.
    BSI->isVariadic = false;
  } else if (FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
      BSI->Params.push_back((ParmVarDecl *)FTI.ArgInfo[i].Param);
    BSI->isVariadic = FTI.isVariadic;
  }
}

/// ActOnBlockError - If there is an error parsing a block, this callback
/// is invoked to pop the information about the block from the action impl.
void Sema::ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
  // Ensure that CurBlock is deleted.
  llvm::OwningPtr<BlockSemaInfo> CC(CurBlock);
  
  // Pop off CurBlock, handle nested blocks.
  CurBlock = CurBlock->PrevBlockInfo;
  
  // FIXME: Delete the ParmVarDecl objects as well???
  
}

/// ActOnBlockStmtExpr - This is called when the body of a block statement
/// literal was successfully completed.  ^(int x){...}
Sema::ExprResult Sema::ActOnBlockStmtExpr(SourceLocation CaretLoc, StmtTy *body,
                                          Scope *CurScope) {
  // Ensure that CurBlock is deleted.
  llvm::OwningPtr<BlockSemaInfo> BSI(CurBlock);
  llvm::OwningPtr<CompoundStmt> Body(static_cast<CompoundStmt*>(body));

  // Pop off CurBlock, handle nested blocks.
  CurBlock = CurBlock->PrevBlockInfo;
  
  QualType RetTy = Context.VoidTy;
  if (BSI->ReturnType)
    RetTy = QualType(BSI->ReturnType, 0);
  
  llvm::SmallVector<QualType, 8> ArgTypes;
  for (unsigned i = 0, e = BSI->Params.size(); i != e; ++i)
    ArgTypes.push_back(BSI->Params[i]->getType());
  
  QualType BlockTy;
  if (!BSI->hasPrototype)
    BlockTy = Context.getFunctionTypeNoProto(RetTy);
  else
    BlockTy = Context.getFunctionType(RetTy, &ArgTypes[0], ArgTypes.size(),
                                      BSI->isVariadic);
  
  BlockTy = Context.getBlockPointerType(BlockTy);
  return new BlockStmtExpr(CaretLoc, BlockTy, 
                           &BSI->Params[0], BSI->Params.size(), Body.take());
}

/// ActOnBlockExprExpr - This is called when the body of a block
/// expression literal was successfully completed.  ^(int x)[foo bar: x]
Sema::ExprResult Sema::ActOnBlockExprExpr(SourceLocation CaretLoc, ExprTy *body,
                                      Scope *CurScope) {
  // Ensure that CurBlock is deleted.
  llvm::OwningPtr<BlockSemaInfo> BSI(CurBlock);
  llvm::OwningPtr<Expr> Body(static_cast<Expr*>(body));

  // Pop off CurBlock, handle nested blocks.
  CurBlock = CurBlock->PrevBlockInfo;
  
  if (BSI->ReturnType) {
    Diag(CaretLoc, diag::err_return_in_block_expression);
    return true;
  }
  
  QualType RetTy = Body->getType();
  
  llvm::SmallVector<QualType, 8> ArgTypes;
  for (unsigned i = 0, e = BSI->Params.size(); i != e; ++i)
    ArgTypes.push_back(BSI->Params[i]->getType());
  
  QualType BlockTy;
  if (!BSI->hasPrototype)
    BlockTy = Context.getFunctionTypeNoProto(RetTy);
  else
    BlockTy = Context.getFunctionType(RetTy, &ArgTypes[0], ArgTypes.size(),
                                      BSI->isVariadic);
  
  BlockTy = Context.getBlockPointerType(BlockTy);
  return new BlockExprExpr(CaretLoc, BlockTy, 
                           &BSI->Params[0], BSI->Params.size(), Body.take());
}

/// ExprsMatchFnType - return true if the Exprs in array Args have
/// QualTypes that match the QualTypes of the arguments of the FnType.
/// The number of arguments has already been validated to match the number of
/// arguments in FnType.
static bool ExprsMatchFnType(Expr **Args, const FunctionTypeProto *FnType,
                             ASTContext &Context) {
  unsigned NumParams = FnType->getNumArgs();
  for (unsigned i = 0; i != NumParams; ++i) {
    QualType ExprTy = Context.getCanonicalType(Args[i]->getType());
    QualType ParmTy = Context.getCanonicalType(FnType->getArgType(i));

    if (ExprTy.getUnqualifiedType() != ParmTy.getUnqualifiedType())
      return false;
  }
  return true;
}

Sema::ExprResult Sema::ActOnOverloadExpr(ExprTy **args, unsigned NumArgs,
                                         SourceLocation *CommaLocs,
                                         SourceLocation BuiltinLoc,
                                         SourceLocation RParenLoc) {
  // __builtin_overload requires at least 2 arguments
  if (NumArgs < 2)
    return Diag(RParenLoc, diag::err_typecheck_call_too_few_args,
                SourceRange(BuiltinLoc, RParenLoc));

  // The first argument is required to be a constant expression.  It tells us
  // the number of arguments to pass to each of the functions to be overloaded.
  Expr **Args = reinterpret_cast<Expr**>(args);
  Expr *NParamsExpr = Args[0];
  llvm::APSInt constEval(32);
  SourceLocation ExpLoc;
  if (!NParamsExpr->isIntegerConstantExpr(constEval, Context, &ExpLoc))
    return Diag(ExpLoc, diag::err_overload_expr_requires_non_zero_constant,
                NParamsExpr->getSourceRange());
  
  // Verify that the number of parameters is > 0
  unsigned NumParams = constEval.getZExtValue();
  if (NumParams == 0)
    return Diag(ExpLoc, diag::err_overload_expr_requires_non_zero_constant,
                NParamsExpr->getSourceRange());
  // Verify that we have at least 1 + NumParams arguments to the builtin.
  if ((NumParams + 1) > NumArgs)
    return Diag(RParenLoc, diag::err_typecheck_call_too_few_args,
                SourceRange(BuiltinLoc, RParenLoc));

  // Figure out the return type, by matching the args to one of the functions
  // listed after the parameters.
  OverloadExpr *OE = 0;
  for (unsigned i = NumParams + 1; i < NumArgs; ++i) {
    // UsualUnaryConversions will convert the function DeclRefExpr into a 
    // pointer to function.
    Expr *Fn = UsualUnaryConversions(Args[i]);
    const FunctionTypeProto *FnType = 0;
    if (const PointerType *PT = Fn->getType()->getAsPointerType())
      FnType = PT->getPointeeType()->getAsFunctionTypeProto();
 
    // The Expr type must be FunctionTypeProto, since FunctionTypeProto has no
    // parameters, and the number of parameters must match the value passed to
    // the builtin.
    if (!FnType || (FnType->getNumArgs() != NumParams))
      return Diag(Fn->getExprLoc(), diag::err_overload_incorrect_fntype, 
                  Fn->getSourceRange());

    // Scan the parameter list for the FunctionType, checking the QualType of
    // each parameter against the QualTypes of the arguments to the builtin.
    // If they match, return a new OverloadExpr.
    if (ExprsMatchFnType(Args+1, FnType, Context)) {
      if (OE)
        return Diag(Fn->getExprLoc(), diag::err_overload_multiple_match,
                    OE->getFn()->getSourceRange());
      // Remember our match, and continue processing the remaining arguments
      // to catch any errors.
      OE = new OverloadExpr(Args, NumArgs, i, FnType->getResultType(),
                            BuiltinLoc, RParenLoc);
    }
  }
  // Return the newly created OverloadExpr node, if we succeded in matching
  // exactly one of the candidate functions.
  if (OE)
    return OE;

  // If we didn't find a matching function Expr in the __builtin_overload list
  // the return an error.
  std::string typeNames;
  for (unsigned i = 0; i != NumParams; ++i) {
    if (i != 0) typeNames += ", ";
    typeNames += Args[i+1]->getType().getAsString();
  }

  return Diag(BuiltinLoc, diag::err_overload_no_match, typeNames,
              SourceRange(BuiltinLoc, RParenLoc));
}

Sema::ExprResult Sema::ActOnVAArg(SourceLocation BuiltinLoc,
                                  ExprTy *expr, TypeTy *type,
                                  SourceLocation RPLoc) {
  Expr *E = static_cast<Expr*>(expr);
  QualType T = QualType::getFromOpaquePtr(type);

  InitBuiltinVaListType();

  // Get the va_list type
  QualType VaListType = Context.getBuiltinVaListType();
  // Deal with implicit array decay; for example, on x86-64,
  // va_list is an array, but it's supposed to decay to
  // a pointer for va_arg.
  if (VaListType->isArrayType())
    VaListType = Context.getArrayDecayedType(VaListType);
  // Make sure the input expression also decays appropriately.
  UsualUnaryConversions(E);

  if (CheckAssignmentConstraints(VaListType, E->getType()) != Compatible)
    return Diag(E->getLocStart(),
                diag::err_first_argument_to_va_arg_not_of_type_va_list,
                E->getType().getAsString(),
                E->getSourceRange());
  
  // FIXME: Warn if a non-POD type is passed in.
  
  return new VAArgExpr(BuiltinLoc, E, T, RPLoc);
}

bool Sema::DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                    SourceLocation Loc,
                                    QualType DstType, QualType SrcType,
                                    Expr *SrcExpr, const char *Flavor) {
  // Decode the result (notice that AST's are still created for extensions).
  bool isInvalid = false;
  unsigned DiagKind;
  switch (ConvTy) {
  default: assert(0 && "Unknown conversion type");
  case Compatible: return false;
  case PointerToInt:
    DiagKind = diag::ext_typecheck_convert_pointer_int;
    break;
  case IntToPointer:
    DiagKind = diag::ext_typecheck_convert_int_pointer;
    break;
  case IncompatiblePointer:
    DiagKind = diag::ext_typecheck_convert_incompatible_pointer;
    break;
  case FunctionVoidPointer:
    DiagKind = diag::ext_typecheck_convert_pointer_void_func;
    break;
  case CompatiblePointerDiscardsQualifiers:
    DiagKind = diag::ext_typecheck_convert_discards_qualifiers;
    break;
  case IntToBlockPointer:
    DiagKind = diag::err_int_to_block_pointer;
    break;
  case IncompatibleBlockPointer:
    DiagKind = diag::err_typecheck_convert_incompatible_block_pointer;
    break;
  case BlockVoidPointer:
    DiagKind = diag::ext_typecheck_convert_pointer_void_block;
    break;
  case Incompatible:
    DiagKind = diag::err_typecheck_convert_incompatible;
    isInvalid = true;
    break;
  }
  
  Diag(Loc, DiagKind, DstType.getAsString(), SrcType.getAsString(), Flavor,
       SrcExpr->getSourceRange());
  return isInvalid;
}
