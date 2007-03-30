//===--- SemaExpr.cpp - Semantic Analysis for Expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;
using namespace clang;

/// ParseStringLiteral - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
/// 
Action::ExprResult
Sema::ParseStringLiteral(const LexerToken *StringToks, unsigned NumStringToks) {
  assert(NumStringToks && "Must have at least one string!");

  StringLiteralParser Literal(StringToks, NumStringToks, PP, Context.Target);
  if (Literal.hadError)
    return ExprResult(true);

  SmallVector<SourceLocation, 4> StringTokLocs;
  for (unsigned i = 0; i != NumStringToks; ++i)
    StringTokLocs.push_back(StringToks[i].getLocation());
  
  // FIXME: handle wchar_t
  TypeRef t = Context.getPointerType(Context.CharTy);
  
  // FIXME: use factory.
  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  return new StringLiteral(Literal.GetString(), Literal.GetStringLength(), 
                           Literal.AnyWide, t);
}


/// ParseIdentifierExpr - The parser read an identifier in expression context,
/// validate it per-C99 6.5.1.  HasTrailingLParen indicates whether this
/// identifier is used in an function call context.
Sema::ExprResult Sema::ParseIdentifierExpr(Scope *S, SourceLocation Loc,
                                           IdentifierInfo &II,
                                           bool HasTrailingLParen) {
  // Could be enum-constant or decl.
  Decl *D = LookupScopedDecl(&II, Decl::IDNS_Ordinary, Loc, S);
  if (D == 0) {
    // Otherwise, this could be an implicitly declared function reference (legal
    // in C90, extension in C99).
    if (HasTrailingLParen &&
        // Not in C++.
        !getLangOptions().CPlusPlus)
      D = ImplicitlyDefineFunction(Loc, II, S);
    else
      // If this name wasn't predeclared and if this is not a function call,
      // diagnose the problem.
      return Diag(Loc, diag::err_undeclared_var_use, II.getName());
  }
  
  if (ObjectDecl *OD = dyn_cast<ObjectDecl>(D)) {
    return new DeclRefExpr(OD);
  } else if (isa<TypedefDecl>(D))
    return Diag(Loc, diag::err_unexpected_typedef, II.getName());

  assert(0 && "Invalid decl");
}

Sema::ExprResult Sema::ParseSimplePrimaryExpr(SourceLocation Loc,
                                              tok::TokenKind Kind) {
  switch (Kind) {
  default:
    assert(0 && "Unknown simple primary expr!");
  case tok::char_constant:     // constant: character-constant
    // TODO: MOVE this to be some other callback.
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    return 0;
  }
}

Action::ExprResult Sema::ParseNumericConstant(const LexerToken &Tok) {
  // fast path for a single digit (which is quite common). A single digit 
  // cannot have a trigraph, escaped newline, radix prefix, or type suffix.
  if (Tok.getLength() == 1) {
    const char *t = PP.getSourceManager().getCharacterData(Tok.getLocation());
    return ExprResult(new IntegerLiteral(*t-'0', Context.IntTy));
  }
  SmallString<512> IntegerBuffer;
  IntegerBuffer.resize(Tok.getLength());
  const char *ThisTokBegin = &IntegerBuffer[0];
  
  // Get the spelling of the token, which eliminates trigraphs, etc.  Notes:
  // - We know that ThisTokBuf points to a buffer that is big enough for the 
  //   whole token and 'spelled' tokens can only shrink.
  // - In practice, the local buffer is only used when the spelling doesn't
  //   match the original token (which is rare). The common case simply returns
  //   a pointer to a *constant* buffer (avoiding a copy). 
  
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);
  NumericLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength, 
                               Tok.getLocation(), PP);
  if (Literal.hadError)
    return ExprResult(true);

  if (Literal.isIntegerLiteral()) {
    TypeRef t;
    if (Literal.hasSuffix()) {
      if (Literal.isLong) 
        t = Literal.isUnsigned ? Context.UnsignedLongTy : Context.LongTy;
      else if (Literal.isLongLong) 
        t = Literal.isUnsigned ? Context.UnsignedLongLongTy : Context.LongLongTy;
      else 
        t = Context.UnsignedIntTy;
    } else {
      t = Context.IntTy; // implicit type is "int"
    }
    uintmax_t val;
    if (Literal.GetIntegerValue(val)) {
      return new IntegerLiteral(val, t);
    } 
  } else if (Literal.isFloatingLiteral()) {
    // FIXME: fill in the value and compute the real type...
    return new FloatingLiteral(7.7, Context.FloatTy);
  }
  return ExprResult(true);
}

Action::ExprResult Sema::ParseParenExpr(SourceLocation L, SourceLocation R,
                                        ExprTy *Val) {
  return Val;
}


// Unary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ParseUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                      ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Op) {
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
  case tok::ampamp:       Opc = UnaryOperator::AddrLabel; break;
  case tok::kw___extension__: 
    return Input;
    //Opc = UnaryOperator::Extension;
    //break;
  }
  if (Opc == UnaryOperator::PreInc || Opc == UnaryOperator::PreDec)
    return CheckIncrementDecrementOperand((Expr *)Input, OpLoc, Opc);
  
  // when all the check functions are written, this will go away...
  return new UnaryOperator((Expr*)Input, Opc, 0);
}

Action::ExprResult Sema::
ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                           SourceLocation LParenLoc, TypeTy *Ty,
                           SourceLocation RParenLoc) {
  // If error parsing type, ignore.
  if (Ty == 0) return true;
  
  // Verify that this is a valid expression.
  TypeRef ArgTy = TypeRef::getFromOpaquePtr(Ty);
  
  if (isa<FunctionType>(ArgTy) && isSizeof) {
    // alignof(function) is allowed.
    Diag(OpLoc, diag::ext_sizeof_function_type);
    return new IntegerLiteral(1, Context.IntTy);
  } else if (ArgTy->isVoidType()) {
    Diag(OpLoc, diag::ext_sizeof_void_type, isSizeof ? "sizeof" : "__alignof");
  } else if (ArgTy->isIncompleteType()) {
    std::string TypeName;
    ArgTy->getAsString(TypeName);
    Diag(OpLoc, isSizeof ? diag::err_sizeof_incomplete_type : 
         diag::err_alignof_incomplete_type, TypeName);
    return new IntegerLiteral(0, Context.IntTy);
  }
  
  return new SizeOfAlignOfTypeExpr(isSizeof, ArgTy, Context.IntTy);
}


Action::ExprResult Sema::ParsePostfixUnaryOp(SourceLocation OpLoc, 
                                             tok::TokenKind Kind,
                                             ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }
  return CheckIncrementDecrementOperand((Expr *)Input, OpLoc, Opc);
}

Action::ExprResult Sema::
ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                        ExprTy *Idx, SourceLocation RLoc) {
  TypeRef t1 = ((Expr *)Base)->getTypeRef();
  TypeRef t2 = ((Expr *)Idx)->getTypeRef();

  assert(!t1.isNull() && "no type for array base expression");
  assert(!t2.isNull() && "no type for array index expression");

  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely equivalent
  // to the expression *((e1)+(e2)). This means the array "Base" may actually be 
  // in the subscript position. As a result, we need to derive the array base 
  // and index from the expression types.
  
  TypeRef baseType, indexType;
  if (isa<ArrayType>(t1) || isa<PointerType>(t1)) {
    baseType = t1;
    indexType = t2;
  } else if (isa<ArrayType>(t2) || isa<PointerType>(t2)) { // uncommon case
    baseType = t2;
    indexType = t1;
  } else 
    return Diag(LLoc, diag::err_typecheck_subscript_value);

  // C99 6.5.2.1p1
  if (!indexType->isIntegralType())
    return Diag(LLoc, diag::err_typecheck_subscript);

  // FIXME: need to deal with const...
  TypeRef resultType;
  if (ArrayType *ary = dyn_cast<ArrayType>(baseType)) {
    resultType = ary->getElementType();
  } else if (PointerType *ary = dyn_cast<PointerType>(baseType)) {
    resultType = ary->getPointeeType();
    // in practice, the following check catches trying to index a pointer
    // to a function (e.g. void (*)(int)). Functions are not objects in c99.
    if (!resultType->isObjectType())
      return Diag(LLoc, diag::err_typecheck_subscript_not_object, baseType);    
  } 
  return new ArraySubscriptExpr((Expr*)Base, (Expr*)Idx, resultType);
}

Action::ExprResult Sema::
ParseMemberReferenceExpr(ExprTy *Base, SourceLocation OpLoc,
                         tok::TokenKind OpKind, SourceLocation MemberLoc,
                         IdentifierInfo &Member) {
  TypeRef BT = ((Expr *)Base)->getTypeRef();

  assert(!BT.isNull() && "no type for member expression");

  if (OpKind == tok::arrow) {
    if (PointerType *PT = dyn_cast<PointerType>(BT))
      BT = PT->getPointeeType();
    else
      return Diag(OpLoc, diag::err_typecheck_member_reference_arrow);
  }
  if (isa<RecordType>(BT)) { // get the struct/union definition from the type.
    RecordDecl *RD = cast<RecordType>(BT)->getDecl();
    
    if (BT->isIncompleteType())
      return Diag(OpLoc, diag::err_typecheck_incomplete_tag, RD->getName());
    
    if (FieldDecl *MemberDecl = RD->getMember(&Member))
      return new MemberExpr((Expr*)Base, OpKind == tok::arrow, MemberDecl);
    else
      return Diag(OpLoc, diag::err_typecheck_no_member, Member.getName());
  }
  return Diag(OpLoc, diag::err_typecheck_member_reference_structUnion);
}

/// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::ExprResult Sema::
ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
              ExprTy **Args, unsigned NumArgs,
              SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  return new CallExpr((Expr*)Fn, (Expr**)Args, NumArgs);
}

Action::ExprResult Sema::
ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
              SourceLocation RParenLoc, ExprTy *Op) {
  // If error parsing type, ignore.
  if (Ty == 0) return true;
  return new CastExpr(TypeRef::getFromOpaquePtr(Ty), (Expr*)Op);
}



// Binary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ParseBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                    ExprTy *LHS, ExprTy *RHS) {
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

  // perform implicit conversions (C99 6.3)
  Expr *e1 = ImplicitConversion((Expr*)LHS);
  Expr *e2 = ImplicitConversion((Expr*)RHS);
  
  if (BinaryOperator::isMultiplicativeOp(Opc)) 
    CheckMultiplicativeOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isAdditiveOp(Opc))
    CheckAdditiveOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isShiftOp(Opc))
    CheckShiftOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isRelationalOp(Opc))
    CheckRelationalOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isEqualityOp(Opc))
    CheckEqualityOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isBitwiseOp(Opc))
    CheckBitwiseOperands((Expr*)LHS, (Expr*)RHS);
  else if (BinaryOperator::isLogicalOp(Opc))
    CheckLogicalOperands((Expr*)LHS, (Expr*)RHS);
  
  return new BinaryOperator((Expr*)LHS, (Expr*)RHS, Opc);
}

/// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::ExprResult Sema::ParseConditionalOp(SourceLocation QuestionLoc, 
                                            SourceLocation ColonLoc,
                                            ExprTy *Cond, ExprTy *LHS,
                                            ExprTy *RHS) {
  return new ConditionalOperator((Expr*)Cond, (Expr*)LHS, (Expr*)RHS);
}

Expr *Sema::ImplicitConversion(Expr *E) {
#if 0
  TypeRef t = E->getTypeRef();
  if (t != 0) t.dump();
  else printf("no type for expr %s\n", E->getStmtClassName());
#endif
  return E;
}

Action::ExprResult Sema::CheckMultiplicativeOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckAdditiveOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckShiftOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckRelationalOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckEqualityOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckBitwiseOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult Sema::CheckLogicalOperands(Expr *op1, Expr *op2) {
  return false;
}

Action::ExprResult
Sema::CheckIncrementDecrementOperand(Expr *op, SourceLocation OpLoc,
                                               UnaryOperator::Opcode OpCode) {
  TypeRef type = op->getTypeRef();

  assert(!type.isNull() && "no type for increment/decrement expression");
  
  if (const PointerType *pt = dyn_cast<PointerType>(type)) {
    if (!pt->getPointeeType()->isObjectType()) // C99 6.5.6p2
      return Diag(OpLoc, diag::err_typecheck_arithmetic_incomplete_type, type);    
  } else if (!type->isRealType()) // C99 6.5.2.4: isRealType excludes complex.
    // FIXME: Allow Complex as a GCC extension.
    return Diag(OpLoc, diag::err_typecheck_illegal_increment_decrement, type);    

  // At this point, we know we have a real or pointer type. As a result, the
  // following predicate is overkill (i.e. it will check for types we know we
  // don't have in this context). Nevertheless, we model the C99 spec closely.
  if (!type.isModifiableLvalue())
    return Diag(OpLoc, diag::err_typecheck_not_modifiable, type);

  return new UnaryOperator(op, OpCode, type);
}

