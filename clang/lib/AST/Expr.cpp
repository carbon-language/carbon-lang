//===--- Expr.cpp - Expression AST Node Implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// getValueAsApproximateDouble - This returns the value as an inaccurate
/// double.  Note that this may cause loss of precision, but is useful for
/// debugging dumps, etc.
double FloatingLiteral::getValueAsApproximateDouble() const {
  llvm::APFloat V = getValue();
  V.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmNearestTiesToEven);
  return V.convertToDouble();
}


StringLiteral::StringLiteral(const char *strData, unsigned byteLength, 
                             bool Wide, QualType t, SourceLocation firstLoc,
                             SourceLocation lastLoc) : 
  Expr(StringLiteralClass, t) {
  // OPTIMIZE: could allocate this appended to the StringLiteral.
  char *AStrData = new char[byteLength];
  memcpy(AStrData, strData, byteLength);
  StrData = AStrData;
  ByteLength = byteLength;
  IsWide = Wide;
  firstTokLoc = firstLoc;
  lastTokLoc = lastLoc;
}

StringLiteral::~StringLiteral() {
  delete[] StrData;
}

bool UnaryOperator::isPostfix(Opcode Op) {
  switch (Op) {
  case PostInc:
  case PostDec:
    return true;
  default:
    return false;
  }
}

bool UnaryOperator::isPrefix(Opcode Op) {
  switch (Op) {
    case PreInc:
    case PreDec:
      return true;
    default:
      return false;
  }
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "sizeof" or "[pre]++".
const char *UnaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown unary operator");
  case PostInc: return "++";
  case PostDec: return "--";
  case PreInc:  return "++";
  case PreDec:  return "--";
  case AddrOf:  return "&";
  case Deref:   return "*";
  case Plus:    return "+";
  case Minus:   return "-";
  case Not:     return "~";
  case LNot:    return "!";
  case Real:    return "__real";
  case Imag:    return "__imag";
  case SizeOf:  return "sizeof";
  case AlignOf: return "alignof";
  case Extension: return "__extension__";
  case OffsetOf: return "__builtin_offsetof";
  }
}

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//


CallExpr::CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t,
                   SourceLocation rparenloc)
  : Expr(CallExprClass, t), NumArgs(numargs) {
  SubExprs = new Stmt*[numargs+1];
  SubExprs[FN] = fn;
  for (unsigned i = 0; i != numargs; ++i)
    SubExprs[i+ARGS_START] = args[i];
  RParenLoc = rparenloc;
}

/// setNumArgs - This changes the number of arguments present in this call.
/// Any orphaned expressions are deleted by this, and any new operands are set
/// to null.
void CallExpr::setNumArgs(unsigned NumArgs) {
  // No change, just return.
  if (NumArgs == getNumArgs()) return;
  
  // If shrinking # arguments, just delete the extras and forgot them.
  if (NumArgs < getNumArgs()) {
    for (unsigned i = NumArgs, e = getNumArgs(); i != e; ++i)
      delete getArg(i);
    this->NumArgs = NumArgs;
    return;
  }

  // Otherwise, we are growing the # arguments.  New an bigger argument array.
  Stmt **NewSubExprs = new Stmt*[NumArgs+1];
  // Copy over args.
  for (unsigned i = 0; i != getNumArgs()+ARGS_START; ++i)
    NewSubExprs[i] = SubExprs[i];
  // Null out new args.
  for (unsigned i = getNumArgs()+ARGS_START; i != NumArgs+ARGS_START; ++i)
    NewSubExprs[i] = 0;
  
  delete[] SubExprs;
  SubExprs = NewSubExprs;
  this->NumArgs = NumArgs;
}

bool CallExpr::isBuiltinConstantExpr() const {
  // All simple function calls (e.g. func()) are implicitly cast to pointer to
  // function. As a result, we try and obtain the DeclRefExpr from the 
  // ImplicitCastExpr.
  const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(getCallee());
  if (!ICE) // FIXME: deal with more complex calls (e.g. (func)(), (*func)()).
    return false;
    
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
  if (!DRE)
    return false;

  const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FDecl)
    return false;
    
  unsigned builtinID = FDecl->getIdentifier()->getBuiltinID();
  if (!builtinID)
    return false;

  // We have a builtin that is a constant expression
  return builtinID == Builtin::BI__builtin___CFStringMakeConstantString ||
         builtinID == Builtin::BI__builtin_classify_type ||
         builtinID == Builtin::BI__builtin_huge_valf;
}

bool CallExpr::isBuiltinClassifyType(llvm::APSInt &Result) const {
  // The following enum mimics gcc's internal "typeclass.h" file.
  enum gcc_type_class {
    no_type_class = -1,
    void_type_class, integer_type_class, char_type_class,
    enumeral_type_class, boolean_type_class,
    pointer_type_class, reference_type_class, offset_type_class,
    real_type_class, complex_type_class,
    function_type_class, method_type_class,
    record_type_class, union_type_class,
    array_type_class, string_type_class,
    lang_type_class
  };
  Result.setIsSigned(true);
  
  // All simple function calls (e.g. func()) are implicitly cast to pointer to
  // function. As a result, we try and obtain the DeclRefExpr from the 
  // ImplicitCastExpr.
  const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(getCallee());
  if (!ICE) // FIXME: deal with more complex calls (e.g. (func)(), (*func)()).
    return false;
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
  if (!DRE)
    return false;

  // We have a DeclRefExpr.
  if (strcmp(DRE->getDecl()->getName(), "__builtin_classify_type") == 0) {
    // If no argument was supplied, default to "no_type_class". This isn't 
    // ideal, however it's what gcc does.
    Result = static_cast<uint64_t>(no_type_class);
    if (NumArgs >= 1) {
      QualType argType = getArg(0)->getType();
      
      if (argType->isVoidType())
        Result = void_type_class;
      else if (argType->isEnumeralType())
        Result = enumeral_type_class;
      else if (argType->isBooleanType())
        Result = boolean_type_class;
      else if (argType->isCharType())
        Result = string_type_class; // gcc doesn't appear to use char_type_class
      else if (argType->isIntegerType())
        Result = integer_type_class;
      else if (argType->isPointerType())
        Result = pointer_type_class;
      else if (argType->isReferenceType())
        Result = reference_type_class;
      else if (argType->isRealType())
        Result = real_type_class;
      else if (argType->isComplexType())
        Result = complex_type_class;
      else if (argType->isFunctionType())
        Result = function_type_class;
      else if (argType->isStructureType())
        Result = record_type_class;
      else if (argType->isUnionType())
        Result = union_type_class;
      else if (argType->isArrayType())
        Result = array_type_class;
      else if (argType->isUnionType())
        Result = union_type_class;
      else  // FIXME: offset_type_class, method_type_class, & lang_type_class?
        assert(0 && "CallExpr::isBuiltinClassifyType(): unimplemented type");
    }
    return true;
  }
  return false;
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "<<=".
const char *BinaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown binary operator");
  case Mul:       return "*";
  case Div:       return "/";
  case Rem:       return "%";
  case Add:       return "+";
  case Sub:       return "-";
  case Shl:       return "<<";
  case Shr:       return ">>";
  case LT:        return "<";
  case GT:        return ">";
  case LE:        return "<=";
  case GE:        return ">=";
  case EQ:        return "==";
  case NE:        return "!=";
  case And:       return "&";
  case Xor:       return "^";
  case Or:        return "|";
  case LAnd:      return "&&";
  case LOr:       return "||";
  case Assign:    return "=";
  case MulAssign: return "*=";
  case DivAssign: return "/=";
  case RemAssign: return "%=";
  case AddAssign: return "+=";
  case SubAssign: return "-=";
  case ShlAssign: return "<<=";
  case ShrAssign: return ">>=";
  case AndAssign: return "&=";
  case XorAssign: return "^=";
  case OrAssign:  return "|=";
  case Comma:     return ",";
  }
}

InitListExpr::InitListExpr(SourceLocation lbraceloc, 
                           Expr **initexprs, unsigned numinits,
                           SourceLocation rbraceloc)
  : Expr(InitListExprClass, QualType()),
    LBraceLoc(lbraceloc), RBraceLoc(rbraceloc)
{
  for (unsigned i = 0; i != numinits; i++)
    InitExprs.push_back(initexprs[i]);
}

/// getFunctionType - Return the underlying function type for this block.
///
const FunctionType *BlockExpr::getFunctionType() const {
  return getType()->getAsBlockPointerType()->
                    getPointeeType()->getAsFunctionType();
}

//===----------------------------------------------------------------------===//
// Generic Expression Routines
//===----------------------------------------------------------------------===//

/// hasLocalSideEffect - Return true if this immediate expression has side
/// effects, not counting any sub-expressions.
bool Expr::hasLocalSideEffect() const {
  switch (getStmtClass()) {
  default:
    return false;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->hasLocalSideEffect();
  case UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(this);
    
    switch (UO->getOpcode()) {
    default: return false;
    case UnaryOperator::PostInc:
    case UnaryOperator::PostDec:
    case UnaryOperator::PreInc:
    case UnaryOperator::PreDec:
      return true;                     // ++/--

    case UnaryOperator::Deref:
      // Dereferencing a volatile pointer is a side-effect.
      return getType().isVolatileQualified();
    case UnaryOperator::Real:
    case UnaryOperator::Imag:
      // accessing a piece of a volatile complex is a side-effect.
      return UO->getSubExpr()->getType().isVolatileQualified();

    case UnaryOperator::Extension:
      return UO->getSubExpr()->hasLocalSideEffect();
    }
  }
  case BinaryOperatorClass: {
    const BinaryOperator *BinOp = cast<BinaryOperator>(this);
    // Consider comma to have side effects if the LHS and RHS both do.
    if (BinOp->getOpcode() == BinaryOperator::Comma)
      return BinOp->getLHS()->hasLocalSideEffect() &&
             BinOp->getRHS()->hasLocalSideEffect();
      
    return BinOp->isAssignmentOp();
  }
  case CompoundAssignOperatorClass:
    return true;

  case ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(this);
    return Exp->getCond()->hasLocalSideEffect()
           || (Exp->getLHS() && Exp->getLHS()->hasLocalSideEffect())
           || (Exp->getRHS() && Exp->getRHS()->hasLocalSideEffect());
  }

  case MemberExprClass:
  case ArraySubscriptExprClass:
    // If the base pointer or element is to a volatile pointer/field, accessing
    // if is a side effect.
    return getType().isVolatileQualified();

  case CallExprClass:
    // TODO: check attributes for pure/const.   "void foo() { strlen("bar"); }"
    // should warn.
    return true;
  case ObjCMessageExprClass:
    return true;
  case StmtExprClass: {
    // Statement exprs don't logically have side effects themselves, but are
    // sometimes used in macros in ways that give them a type that is unused.
    // For example ({ blah; foo(); }) will end up with a type if foo has a type.
    // however, if the result of the stmt expr is dead, we don't want to emit a
    // warning.
    const CompoundStmt *CS = cast<StmtExpr>(this)->getSubStmt();
    if (!CS->body_empty())
      if (const Expr *E = dyn_cast<Expr>(CS->body_back()))
        return E->hasLocalSideEffect();
    return false;
  }
  case ExplicitCastExprClass:
  case CXXFunctionalCastExprClass:
    // If this is a cast to void, check the operand.  Otherwise, the result of
    // the cast is unused.
    if (getType()->isVoidType())
      return cast<CastExpr>(this)->getSubExpr()->hasLocalSideEffect();
    return false;

  case ImplicitCastExprClass:
    // Check the operand, since implicit casts are inserted by Sema
    return cast<ImplicitCastExpr>(this)->getSubExpr()->hasLocalSideEffect();

  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)->getExpr()->hasLocalSideEffect();
  }     
}

/// isLvalue - C99 6.3.2.1: an lvalue is an expression with an object type or an
/// incomplete type other than void. Nonarray expressions that can be lvalues:
///  - name, where name must be a variable
///  - e[i]
///  - (e), where e must be an lvalue
///  - e.name, where e must be an lvalue
///  - e->name
///  - *e, the type of e cannot be a function type
///  - string-constant
///  - (__real__ e) and (__imag__ e) where e is an lvalue  [GNU extension]
///  - reference type [C++ [expr]]
///
Expr::isLvalueResult Expr::isLvalue(ASTContext &Ctx) const {
  // first, check the type (C99 6.3.2.1)
  if (TR->isFunctionType()) // from isObjectType()
    return LV_NotObjectType;

  // Allow qualified void which is an incomplete type other than void (yuck).
  if (TR->isVoidType() && !Ctx.getCanonicalType(TR).getCVRQualifiers())
    return LV_IncompleteVoidType;

  if (TR->isReferenceType()) // C++ [expr]
    return LV_Valid;

  // the type looks fine, now check the expression
  switch (getStmtClass()) {
  case StringLiteralClass: // C99 6.5.1p4
    return LV_Valid;
  case ArraySubscriptExprClass: // C99 6.5.3p4 (e1[e2] == (*((e1)+(e2))))
    // For vectors, make sure base is an lvalue (i.e. not a function call).
    if (cast<ArraySubscriptExpr>(this)->getBase()->getType()->isVectorType())
      return cast<ArraySubscriptExpr>(this)->getBase()->isLvalue(Ctx);
    return LV_Valid;
  case DeclRefExprClass: { // C99 6.5.1p2
    const Decl *RefdDecl = cast<DeclRefExpr>(this)->getDecl();
    if (isa<VarDecl>(RefdDecl) || isa<ImplicitParamDecl>(RefdDecl))
      return LV_Valid;
    break;
  }
  case BlockDeclRefExprClass: {
    const BlockDeclRefExpr *BDR = cast<BlockDeclRefExpr>(this);
    if (BDR->isByRef() && isa<VarDecl>(BDR->getDecl()))
      return LV_Valid;
    break;
  }
  case MemberExprClass: { // C99 6.5.2.3p4
    const MemberExpr *m = cast<MemberExpr>(this);
    return m->isArrow() ? LV_Valid : m->getBase()->isLvalue(Ctx);
  }
  case UnaryOperatorClass:
    if (cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Deref)
      return LV_Valid; // C99 6.5.3p4

    if (cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Real ||
        cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Imag ||
        cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Extension)
      return cast<UnaryOperator>(this)->getSubExpr()->isLvalue(Ctx);  // GNU.
    break;
  case ParenExprClass: // C99 6.5.1p5
    return cast<ParenExpr>(this)->getSubExpr()->isLvalue(Ctx);
  case CompoundLiteralExprClass: // C99 6.5.2.5p5
    return LV_Valid;
  case ExtVectorElementExprClass:
    if (cast<ExtVectorElementExpr>(this)->containsDuplicateElements())
      return LV_DuplicateVectorComponents;
    return LV_Valid;
  case ObjCIvarRefExprClass: // ObjC instance variables are lvalues.
    return LV_Valid;
  case ObjCPropertyRefExprClass: // FIXME: check if read-only property.
    return LV_Valid;
  case PredefinedExprClass:
    return (cast<PredefinedExpr>(this)->getIdentType()
               == PredefinedExpr::CXXThis
            ? LV_InvalidExpression : LV_Valid);
  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)->getExpr()->isLvalue(Ctx);
  case CXXConditionDeclExprClass:
    return LV_Valid;
  default:
    break;
  }
  return LV_InvalidExpression;
}

/// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
/// does not have an incomplete type, does not have a const-qualified type, and
/// if it is a structure or union, does not have any member (including, 
/// recursively, any member or element of all contained aggregates or unions)
/// with a const-qualified type.
Expr::isModifiableLvalueResult Expr::isModifiableLvalue(ASTContext &Ctx) const {
  isLvalueResult lvalResult = isLvalue(Ctx);
    
  switch (lvalResult) {
  case LV_Valid: break;
  case LV_NotObjectType: return MLV_NotObjectType;
  case LV_IncompleteVoidType: return MLV_IncompleteVoidType;
  case LV_DuplicateVectorComponents: return MLV_DuplicateVectorComponents;
  case LV_InvalidExpression: return MLV_InvalidExpression;
  }
  
  QualType CT = Ctx.getCanonicalType(getType());
  
  if (CT.isConstQualified())
    return MLV_ConstQualified;
  if (CT->isArrayType())
    return MLV_ArrayType;
  if (CT->isIncompleteType())
    return MLV_IncompleteType;
    
  if (const RecordType *r = CT->getAsRecordType()) {
    if (r->hasConstFields()) 
      return MLV_ConstQualified;
  }
  return MLV_Valid;    
}

/// hasGlobalStorage - Return true if this expression has static storage
/// duration.  This means that the address of this expression is a link-time
/// constant.
bool Expr::hasGlobalStorage() const {
  switch (getStmtClass()) {
  default:
    return false;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->hasGlobalStorage();
  case ImplicitCastExprClass:
    return cast<ImplicitCastExpr>(this)->getSubExpr()->hasGlobalStorage();
  case CompoundLiteralExprClass:
    return cast<CompoundLiteralExpr>(this)->isFileScope();
  case DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(this)->getDecl();
    if (const VarDecl *VD = dyn_cast<VarDecl>(D))
      return VD->hasGlobalStorage();
    if (isa<FunctionDecl>(D))
      return true;
    return false;
  }
  case MemberExprClass: {
    const MemberExpr *M = cast<MemberExpr>(this);
    return !M->isArrow() && M->getBase()->hasGlobalStorage();
  }
  case ArraySubscriptExprClass:
    return cast<ArraySubscriptExpr>(this)->getBase()->hasGlobalStorage();
  case PredefinedExprClass:
    return true;
  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)->getExpr()->hasGlobalStorage();
  }
}

Expr* Expr::IgnoreParens() {
  Expr* E = this;
  while (ParenExpr* P = dyn_cast<ParenExpr>(E))
    E = P->getSubExpr();
  
  return E;
}

/// IgnoreParenCasts - Ignore parentheses and casts.  Strip off any ParenExpr
/// or CastExprs or ImplicitCastExprs, returning their operand.
Expr *Expr::IgnoreParenCasts() {
  Expr *E = this;
  while (true) {
    if (ParenExpr *P = dyn_cast<ParenExpr>(E))
      E = P->getSubExpr();
    else if (CastExpr *P = dyn_cast<CastExpr>(E))
      E = P->getSubExpr();
    else
      return E;
  }
}


bool Expr::isConstantExpr(ASTContext &Ctx, SourceLocation *Loc) const {
  switch (getStmtClass()) {
  default:
    if (Loc) *Loc = getLocStart();
    return false;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isConstantExpr(Ctx, Loc);
  case StringLiteralClass:
  case ObjCStringLiteralClass:
  case FloatingLiteralClass:
  case IntegerLiteralClass:
  case CharacterLiteralClass:
  case ImaginaryLiteralClass:
  case TypesCompatibleExprClass:
  case CXXBoolLiteralExprClass:
  case AddrLabelExprClass:
    return true;
  case CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(this);
    if (CE->isBuiltinConstantExpr())
      return true;
    if (Loc) *Loc = getLocStart();
    return false;
  }
  case DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(this)->getDecl();
    // Accept address of function.
    if (isa<EnumConstantDecl>(D) || isa<FunctionDecl>(D))
      return true;
    if (Loc) *Loc = getLocStart();
    if (isa<VarDecl>(D))
      return TR->isArrayType();
    return false;
  }
  case CompoundLiteralExprClass:
    if (Loc) *Loc = getLocStart();
    // Allow "(int []){2,4}", since the array will be converted to a pointer.
    // Allow "(vector type){2,4}" since the elements are all constant.
    return TR->isArrayType() || TR->isVectorType();
  case UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(this);
    
    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::AddrOf) {
      if (!Exp->getSubExpr()->hasGlobalStorage()) {
        if (Loc) *Loc = getLocStart();
        return false;
      }
      return true;
    }

    // Get the operand value.  If this is sizeof/alignof, do not evalute the
    // operand.  This affects C99 6.6p3.
    if (!Exp->isSizeOfAlignOfOp() && 
        Exp->getOpcode() != UnaryOperator::OffsetOf &&
        !Exp->getSubExpr()->isConstantExpr(Ctx, Loc))
      return false;
  
    switch (Exp->getOpcode()) {
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    default:
      if (Loc) *Loc = Exp->getOperatorLoc();
      return false;
    case UnaryOperator::Extension:
      return true;  // FIXME: this is wrong.
    case UnaryOperator::SizeOf:
    case UnaryOperator::AlignOf:
    case UnaryOperator::OffsetOf:
      // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
      if (!Exp->getSubExpr()->getType()->isConstantSizeType()) {
        if (Loc) *Loc = Exp->getOperatorLoc();
        return false;
      }
      return true;
    case UnaryOperator::LNot:
    case UnaryOperator::Plus:
    case UnaryOperator::Minus:
    case UnaryOperator::Not:
      return true;
    }
  }
  case SizeOfAlignOfTypeExprClass: {
    const SizeOfAlignOfTypeExpr *Exp = cast<SizeOfAlignOfTypeExpr>(this);
    // alignof always evaluates to a constant.
    if (Exp->isSizeOf() && !Exp->getArgumentType()->isVoidType() &&
        !Exp->getArgumentType()->isConstantSizeType()) {
      if (Loc) *Loc = Exp->getOperatorLoc();
      return false;
    }
    return true;
  }
  case BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(this);
    
    // The LHS of a constant expr is always evaluated and needed.
    if (!Exp->getLHS()->isConstantExpr(Ctx, Loc))
      return false;

    if (!Exp->getRHS()->isConstantExpr(Ctx, Loc))
      return false;
    return true;
  }
  case ImplicitCastExprClass:
  case ExplicitCastExprClass:
  case CXXFunctionalCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(this)->getSubExpr();
    SourceLocation CastLoc = getLocStart();
    if (!SubExpr->isConstantExpr(Ctx, Loc)) {
      if (Loc) *Loc = SubExpr->getLocStart();
      return false;
    }
    return true;
  }
  case ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(this);
    if (!Exp->getCond()->isConstantExpr(Ctx, Loc) ||
        // Handle the GNU extension for missing LHS.
        !(Exp->getLHS() && Exp->getLHS()->isConstantExpr(Ctx, Loc)) ||
        !Exp->getRHS()->isConstantExpr(Ctx, Loc))
      return false;
    return true;
  }
  case InitListExprClass: {
    const InitListExpr *Exp = cast<InitListExpr>(this);
    unsigned numInits = Exp->getNumInits();
    for (unsigned i = 0; i < numInits; i++) {
      if (!Exp->getInit(i)->isConstantExpr(Ctx, Loc)) {
        if (Loc) *Loc = Exp->getInit(i)->getLocStart();
        return false;
      }
    }
    return true;
  }
  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)->getExpr()->isConstantExpr(Ctx, Loc);
  }
}

/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression. Note: With the introduction of VLA's in
/// C99 the result of the sizeof operator is no longer always a constant
/// expression. The generalization of the wording to include any subexpression
/// that is not evaluated (C99 6.6p3) means that nonconstant subexpressions
/// can appear as operands to other operators (e.g. &&, ||, ?:). For instance,
/// "0 || f()" can be treated as a constant expression. In C90 this expression,
/// occurring in a context requiring a constant, would have been a constraint
/// violation. FIXME: This routine currently implements C90 semantics.
/// To properly implement C99 semantics this routine will need to evaluate
/// expressions involving operators previously mentioned.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: This should ext-warn on overflow during evaluation!  ISO C does not
/// permit this.  This includes things like (int)1e1000
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.
bool Expr::isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  switch (getStmtClass()) {
  default:
    if (Loc) *Loc = getLocStart();
    return false;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->
                     isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated);
  case IntegerLiteralClass:
    Result = cast<IntegerLiteral>(this)->getValue();
    break;
  case CharacterLiteralClass: {
    const CharacterLiteral *CL = cast<CharacterLiteral>(this);
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
    Result = CL->getValue();
    Result.setIsUnsigned(!getType()->isSignedIntegerType());
    break;
  }
  case CXXBoolLiteralExprClass: {
    const CXXBoolLiteralExpr *BL = cast<CXXBoolLiteralExpr>(this);
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
    Result = BL->getValue();
    Result.setIsUnsigned(!getType()->isSignedIntegerType());
    break;
  }
  case CXXZeroInitValueExprClass:
    Result.clear();
    break;
  case TypesCompatibleExprClass: {
    const TypesCompatibleExpr *TCE = cast<TypesCompatibleExpr>(this);
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
    Result = Ctx.typesAreCompatible(TCE->getArgType1(), TCE->getArgType2());
    break;
  }
  case CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(this);
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
    if (CE->isBuiltinClassifyType(Result))
      break;
    if (Loc) *Loc = getLocStart();
    return false;
  }
  case DeclRefExprClass:
    if (const EnumConstantDecl *D = 
          dyn_cast<EnumConstantDecl>(cast<DeclRefExpr>(this)->getDecl())) {
      Result = D->getInitVal();
      break;
    }
    if (Loc) *Loc = getLocStart();
    return false;
  case UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(this);
    
    // Get the operand value.  If this is sizeof/alignof, do not evalute the
    // operand.  This affects C99 6.6p3.
    if (!Exp->isSizeOfAlignOfOp() && !Exp->isOffsetOfOp() &&
        !Exp->getSubExpr()->isIntegerConstantExpr(Result, Ctx, Loc,isEvaluated))
      return false;

    switch (Exp->getOpcode()) {
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    default:
      if (Loc) *Loc = Exp->getOperatorLoc();
      return false;
    case UnaryOperator::Extension:
      return true;  // FIXME: this is wrong.
    case UnaryOperator::SizeOf:
    case UnaryOperator::AlignOf:
      // Return the result in the right width.
      Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
        
      // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
      if (Exp->getSubExpr()->getType()->isVoidType()) {
        Result = 1;
        break;
      }
        
      // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
      if (!Exp->getSubExpr()->getType()->isConstantSizeType()) {
        if (Loc) *Loc = Exp->getOperatorLoc();
        return false;
      }
      
      // Get information about the size or align.
      if (Exp->getSubExpr()->getType()->isFunctionType()) {
        // GCC extension: sizeof(function) = 1.
        Result = Exp->getOpcode() == UnaryOperator::AlignOf ? 4 : 1;
      } else {
        unsigned CharSize = Ctx.Target.getCharWidth();
        if (Exp->getOpcode() == UnaryOperator::AlignOf)
          Result = Ctx.getTypeAlign(Exp->getSubExpr()->getType()) / CharSize;
        else
          Result = Ctx.getTypeSize(Exp->getSubExpr()->getType()) / CharSize;
      }
      break;
    case UnaryOperator::LNot: {
      bool Val = Result == 0;
      Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
      Result = Val;
      break;
    }
    case UnaryOperator::Plus:
      break;
    case UnaryOperator::Minus:
      Result = -Result;
      break;
    case UnaryOperator::Not:
      Result = ~Result;
      break;
    case UnaryOperator::OffsetOf:
      Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
      Result = Exp->evaluateOffsetOf(Ctx);
    }
    break;
  }
  case SizeOfAlignOfTypeExprClass: {
    const SizeOfAlignOfTypeExpr *Exp = cast<SizeOfAlignOfTypeExpr>(this);
    
    // Return the result in the right width.
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(getType())));
    
    // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
    if (Exp->getArgumentType()->isVoidType()) {
      Result = 1;
      break;
    }
    
    // alignof always evaluates to a constant, sizeof does if arg is not VLA.
    if (Exp->isSizeOf() && !Exp->getArgumentType()->isConstantSizeType()) {
      if (Loc) *Loc = Exp->getOperatorLoc();
      return false;
    }

    // Get information about the size or align.
    if (Exp->getArgumentType()->isFunctionType()) {
      // GCC extension: sizeof(function) = 1.
      Result = Exp->isSizeOf() ? 1 : 4;
    } else { 
      unsigned CharSize = Ctx.Target.getCharWidth();
      if (Exp->isSizeOf())
        Result = Ctx.getTypeSize(Exp->getArgumentType()) / CharSize;
      else
        Result = Ctx.getTypeAlign(Exp->getArgumentType()) / CharSize;
    }
    break;
  }
  case BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(this);
    
    // The LHS of a constant expr is always evaluated and needed.
    if (!Exp->getLHS()->isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated))
      return false;
    
    llvm::APSInt RHS(Result);
    
    // The short-circuiting &&/|| operators don't necessarily evaluate their
    // RHS.  Make sure to pass isEvaluated down correctly.
    if (Exp->isLogicalOp()) {
      bool RHSEval;
      if (Exp->getOpcode() == BinaryOperator::LAnd)
        RHSEval = Result != 0;
      else {
        assert(Exp->getOpcode() == BinaryOperator::LOr &&"Unexpected logical");
        RHSEval = Result == 0;
      }
      
      if (!Exp->getRHS()->isIntegerConstantExpr(RHS, Ctx, Loc,
                                                isEvaluated & RHSEval))
        return false;
    } else {
      if (!Exp->getRHS()->isIntegerConstantExpr(RHS, Ctx, Loc, isEvaluated))
        return false;
    }
    
    switch (Exp->getOpcode()) {
    default:
      if (Loc) *Loc = getLocStart();
      return false;
    case BinaryOperator::Mul:
      Result *= RHS;
      break;
    case BinaryOperator::Div:
      if (RHS == 0) {
        if (!isEvaluated) break;
        if (Loc) *Loc = getLocStart();
        return false;
      }
      Result /= RHS;
      break;
    case BinaryOperator::Rem:
      if (RHS == 0) {
        if (!isEvaluated) break;
        if (Loc) *Loc = getLocStart();
        return false;
      }
      Result %= RHS;
      break;
    case BinaryOperator::Add: Result += RHS; break;
    case BinaryOperator::Sub: Result -= RHS; break;
    case BinaryOperator::Shl:
      Result <<= 
        static_cast<uint32_t>(RHS.getLimitedValue(Result.getBitWidth()-1));
      break;
    case BinaryOperator::Shr:
      Result >>= 
        static_cast<uint32_t>(RHS.getLimitedValue(Result.getBitWidth()-1));
      break;
    case BinaryOperator::LT:  Result = Result < RHS; break;
    case BinaryOperator::GT:  Result = Result > RHS; break;
    case BinaryOperator::LE:  Result = Result <= RHS; break;
    case BinaryOperator::GE:  Result = Result >= RHS; break;
    case BinaryOperator::EQ:  Result = Result == RHS; break;
    case BinaryOperator::NE:  Result = Result != RHS; break;
    case BinaryOperator::And: Result &= RHS; break;
    case BinaryOperator::Xor: Result ^= RHS; break;
    case BinaryOperator::Or:  Result |= RHS; break;
    case BinaryOperator::LAnd:
      Result = Result != 0 && RHS != 0;
      break;
    case BinaryOperator::LOr:
      Result = Result != 0 || RHS != 0;
      break;
      
    case BinaryOperator::Comma:
      // C99 6.6p3: "shall not contain assignment, ..., or comma operators,
      // *except* when they are contained within a subexpression that is not
      // evaluated".  Note that Assignment can never happen due to constraints
      // on the LHS subexpr, so we don't need to check it here.
      if (isEvaluated) {
        if (Loc) *Loc = getLocStart();
        return false;
      }
      
      // The result of the constant expr is the RHS.
      Result = RHS;
      return true;
    }
    
    assert(!Exp->isAssignmentOp() && "LHS can't be a constant expr!");
    break;
  }
  case ImplicitCastExprClass:
  case ExplicitCastExprClass:
  case CXXFunctionalCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(this)->getSubExpr();
    SourceLocation CastLoc = getLocStart();
    
    // C99 6.6p6: shall only convert arithmetic types to integer types.
    if (!SubExpr->getType()->isArithmeticType() ||
        !getType()->isIntegerType()) {
      if (Loc) *Loc = SubExpr->getLocStart();
      return false;
    }

    uint32_t DestWidth = static_cast<uint32_t>(Ctx.getTypeSize(getType()));
    
    // Handle simple integer->integer casts.
    if (SubExpr->getType()->isIntegerType()) {
      if (!SubExpr->isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated))
        return false;
      
      // Figure out if this is a truncate, extend or noop cast.
      // If the input is signed, do a sign extend, noop, or truncate.
      if (getType()->isBooleanType()) {
        // Conversion to bool compares against zero.
        Result = Result != 0;
        Result.zextOrTrunc(DestWidth);
      } else if (SubExpr->getType()->isSignedIntegerType())
        Result.sextOrTrunc(DestWidth);
      else  // If the input is unsigned, do a zero extend, noop, or truncate.
        Result.zextOrTrunc(DestWidth);
      break;
    }
    
    // Allow floating constants that are the immediate operands of casts or that
    // are parenthesized.
    const Expr *Operand = SubExpr;
    while (const ParenExpr *PE = dyn_cast<ParenExpr>(Operand))
      Operand = PE->getSubExpr();

    // If this isn't a floating literal, we can't handle it.
    const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(Operand);
    if (!FL) {
      if (Loc) *Loc = Operand->getLocStart();
      return false;
    }

    // If the destination is boolean, compare against zero.
    if (getType()->isBooleanType()) {
      Result = !FL->getValue().isZero();
      Result.zextOrTrunc(DestWidth);
      break;
    }     
    
    // Determine whether we are converting to unsigned or signed.
    bool DestSigned = getType()->isSignedIntegerType();

    // TODO: Warn on overflow, but probably not here: isIntegerConstantExpr can
    // be called multiple times per AST.
    uint64_t Space[4]; 
    (void)FL->getValue().convertToInteger(Space, DestWidth, DestSigned,
                                          llvm::APFloat::rmTowardZero);
    Result = llvm::APInt(DestWidth, 4, Space);
    break;
  }
  case ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(this);
    
    if (!Exp->getCond()->isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated))
      return false;
    
    const Expr *TrueExp  = Exp->getLHS();
    const Expr *FalseExp = Exp->getRHS();
    if (Result == 0) std::swap(TrueExp, FalseExp);
    
    // Evaluate the false one first, discard the result.
    if (FalseExp && !FalseExp->isIntegerConstantExpr(Result, Ctx, Loc, false))
      return false;
    // Evalute the true one, capture the result.
    if (TrueExp && 
        !TrueExp->isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated))
      return false;
    break;
  }
  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)
             ->isIntegerConstantExpr(Result, Ctx, Loc, isEvaluated);
  }

  // Cases that are valid constant exprs fall through to here.
  Result.setIsUnsigned(getType()->isUnsignedIntegerType());
  return true;
}

/// isNullPointerConstant - C99 6.3.2.3p3 -  Return true if this is either an
/// integer constant expression with the value zero, or if this is one that is
/// cast to void*.
bool Expr::isNullPointerConstant(ASTContext &Ctx) const {
  // Strip off a cast to void*, if it exists.
  if (const ExplicitCastExpr *CE = dyn_cast<ExplicitCastExpr>(this)) {
    // Check that it is a cast to void*.
    if (const PointerType *PT = CE->getType()->getAsPointerType()) {
      QualType Pointee = PT->getPointeeType();
      if (Pointee.getCVRQualifiers() == 0 && 
          Pointee->isVoidType() &&                                 // to void*
          CE->getSubExpr()->getType()->isIntegerType())            // from int.
        return CE->getSubExpr()->isNullPointerConstant(Ctx);
    }
  } else if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(this)) {
    // Ignore the ImplicitCastExpr type entirely.
    return ICE->getSubExpr()->isNullPointerConstant(Ctx);
  } else if (const ParenExpr *PE = dyn_cast<ParenExpr>(this)) {
    // Accept ((void*)0) as a null pointer constant, as many other
    // implementations do.
    return PE->getSubExpr()->isNullPointerConstant(Ctx);
  } else if (const CXXDefaultArgExpr *DefaultArg 
               = dyn_cast<CXXDefaultArgExpr>(this)) {
    // See through default argument expressions
    return DefaultArg->getExpr()->isNullPointerConstant(Ctx);
  }
  
  // This expression must be an integer type.
  if (!getType()->isIntegerType())
    return false;
  
  // If we have an integer constant expression, we need to *evaluate* it and
  // test for the value 0.
  llvm::APSInt Val(32);
  return isIntegerConstantExpr(Val, Ctx, 0, true) && Val == 0;
}

unsigned ExtVectorElementExpr::getNumElements() const {
  if (const VectorType *VT = getType()->getAsVectorType())
    return VT->getNumElements();
  return 1;
}

/// containsDuplicateElements - Return true if any element access is repeated.
bool ExtVectorElementExpr::containsDuplicateElements() const {
  const char *compStr = Accessor.getName();
  unsigned length = strlen(compStr);
  
  for (unsigned i = 0; i < length-1; i++) {
    const char *s = compStr+i;
    for (const char c = *s++; *s; s++)
      if (c == *s) 
        return true;
  }
  return false;
}

/// getEncodedElementAccess - We encode the fields as a llvm ConstantArray.
void ExtVectorElementExpr::getEncodedElementAccess(
                                  llvm::SmallVectorImpl<unsigned> &Elts) const {
  const char *compStr = Accessor.getName();
 
  bool isHi =   !strcmp(compStr, "hi");
  bool isLo =   !strcmp(compStr, "lo");
  bool isEven = !strcmp(compStr, "e");
  bool isOdd  = !strcmp(compStr, "o");
    
  for (unsigned i = 0, e = getNumElements(); i != e; ++i) {
    uint64_t Index;
    
    if (isHi)
      Index = e + i;
    else if (isLo)
      Index = i;
    else if (isEven)
      Index = 2 * i;
    else if (isOdd)
      Index = 2 * i + 1;
    else
      Index = ExtVectorType::getAccessorIdx(compStr[i]);

    Elts.push_back(Index);
  }
}

// constructor for instance messages.
ObjCMessageExpr::ObjCMessageExpr(Expr *receiver, Selector selInfo,
                QualType retType, ObjCMethodDecl *mproto,
                SourceLocation LBrac, SourceLocation RBrac,
                Expr **ArgExprs, unsigned nargs)
  : Expr(ObjCMessageExprClass, retType), SelName(selInfo), 
    MethodProto(mproto) {
  NumArgs = nargs;
  SubExprs = new Stmt*[NumArgs+1];
  SubExprs[RECEIVER] = receiver;
  if (NumArgs) {
    for (unsigned i = 0; i != NumArgs; ++i)
      SubExprs[i+ARGS_START] = static_cast<Expr *>(ArgExprs[i]);
  }
  LBracloc = LBrac;
  RBracloc = RBrac;
}

// constructor for class messages. 
// FIXME: clsName should be typed to ObjCInterfaceType
ObjCMessageExpr::ObjCMessageExpr(IdentifierInfo *clsName, Selector selInfo,
                QualType retType, ObjCMethodDecl *mproto,
                SourceLocation LBrac, SourceLocation RBrac,
                Expr **ArgExprs, unsigned nargs)
  : Expr(ObjCMessageExprClass, retType), SelName(selInfo), 
    MethodProto(mproto) {
  NumArgs = nargs;
  SubExprs = new Stmt*[NumArgs+1];
  SubExprs[RECEIVER] = (Expr*) ((uintptr_t) clsName | IsClsMethDeclUnknown);
  if (NumArgs) {
    for (unsigned i = 0; i != NumArgs; ++i)
      SubExprs[i+ARGS_START] = static_cast<Expr *>(ArgExprs[i]);
  }
  LBracloc = LBrac;
  RBracloc = RBrac;
}

// constructor for class messages. 
ObjCMessageExpr::ObjCMessageExpr(ObjCInterfaceDecl *cls, Selector selInfo,
                                 QualType retType, ObjCMethodDecl *mproto,
                                 SourceLocation LBrac, SourceLocation RBrac,
                                 Expr **ArgExprs, unsigned nargs)
: Expr(ObjCMessageExprClass, retType), SelName(selInfo), 
MethodProto(mproto) {
  NumArgs = nargs;
  SubExprs = new Stmt*[NumArgs+1];
  SubExprs[RECEIVER] = (Expr*) ((uintptr_t) cls | IsClsMethDeclKnown);
  if (NumArgs) {
    for (unsigned i = 0; i != NumArgs; ++i)
      SubExprs[i+ARGS_START] = static_cast<Expr *>(ArgExprs[i]);
  }
  LBracloc = LBrac;
  RBracloc = RBrac;
}

ObjCMessageExpr::ClassInfo ObjCMessageExpr::getClassInfo() const {
  uintptr_t x = (uintptr_t) SubExprs[RECEIVER];
  switch (x & Flags) {
    default:
      assert(false && "Invalid ObjCMessageExpr.");
    case IsInstMeth:
      return ClassInfo(0, 0);
    case IsClsMethDeclUnknown:
      return ClassInfo(0, (IdentifierInfo*) (x & ~Flags));
    case IsClsMethDeclKnown: {
      ObjCInterfaceDecl* D = (ObjCInterfaceDecl*) (x & ~Flags);
      return ClassInfo(D, D->getIdentifier());
    }
  }
}

bool ChooseExpr::isConditionTrue(ASTContext &C) const {
  return getCond()->getIntegerConstantExprValue(C) != 0;
}

static int64_t evaluateOffsetOf(ASTContext& C, const Expr *E)
{
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    QualType Ty = ME->getBase()->getType();
    
    RecordDecl *RD = Ty->getAsRecordType()->getDecl();
    const ASTRecordLayout &RL = C.getASTRecordLayout(RD);
    FieldDecl *FD = ME->getMemberDecl();
    
    // FIXME: This is linear time.
    unsigned i = 0, e = 0;
    for (i = 0, e = RD->getNumMembers(); i != e; i++) {
      if (RD->getMember(i) == FD)
        break;
    }
    
    return RL.getFieldOffset(i) + evaluateOffsetOf(C, ME->getBase());
  } else if (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    const Expr *Base = ASE->getBase();
    
    int64_t size = C.getTypeSize(ASE->getType());
    size *= ASE->getIdx()->getIntegerConstantExprValue(C).getSExtValue();
    
    return size + evaluateOffsetOf(C, Base);
  } else if (isa<CompoundLiteralExpr>(E))
    return 0;  

  assert(0 && "Unknown offsetof subexpression!");
  return 0;
}

int64_t UnaryOperator::evaluateOffsetOf(ASTContext& C) const
{
  assert(Opc == OffsetOf && "Unary operator not offsetof!");
  
  unsigned CharSize = C.Target.getCharWidth();
  return ::evaluateOffsetOf(C, cast<Expr>(Val)) / CharSize;
}

void SizeOfAlignOfTypeExpr::Destroy(ASTContext& C) {
  // Override default behavior of traversing children. We do not want
  // to delete the type.
}

//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

// DeclRefExpr
Stmt::child_iterator DeclRefExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator DeclRefExpr::child_end() { return child_iterator(); }

// ObjCIvarRefExpr
Stmt::child_iterator ObjCIvarRefExpr::child_begin() { return &Base; }
Stmt::child_iterator ObjCIvarRefExpr::child_end() { return &Base+1; }

// ObjCPropertyRefExpr
Stmt::child_iterator ObjCPropertyRefExpr::child_begin() { return &Base; }
Stmt::child_iterator ObjCPropertyRefExpr::child_end() { return &Base+1; }

// PredefinedExpr
Stmt::child_iterator PredefinedExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator PredefinedExpr::child_end() { return child_iterator(); }

// IntegerLiteral
Stmt::child_iterator IntegerLiteral::child_begin() { return child_iterator(); }
Stmt::child_iterator IntegerLiteral::child_end() { return child_iterator(); }

// CharacterLiteral
Stmt::child_iterator CharacterLiteral::child_begin() { return child_iterator(); }
Stmt::child_iterator CharacterLiteral::child_end() { return child_iterator(); }

// FloatingLiteral
Stmt::child_iterator FloatingLiteral::child_begin() { return child_iterator(); }
Stmt::child_iterator FloatingLiteral::child_end() { return child_iterator(); }

// ImaginaryLiteral
Stmt::child_iterator ImaginaryLiteral::child_begin() { return &Val; }
Stmt::child_iterator ImaginaryLiteral::child_end() { return &Val+1; }

// StringLiteral
Stmt::child_iterator StringLiteral::child_begin() { return child_iterator(); }
Stmt::child_iterator StringLiteral::child_end() { return child_iterator(); }

// ParenExpr
Stmt::child_iterator ParenExpr::child_begin() { return &Val; }
Stmt::child_iterator ParenExpr::child_end() { return &Val+1; }

// UnaryOperator
Stmt::child_iterator UnaryOperator::child_begin() { return &Val; }
Stmt::child_iterator UnaryOperator::child_end() { return &Val+1; }

// SizeOfAlignOfTypeExpr
Stmt::child_iterator SizeOfAlignOfTypeExpr::child_begin() { 
  // If the type is a VLA type (and not a typedef), the size expression of the
  // VLA needs to be treated as an executable expression.
  if (VariableArrayType* T = dyn_cast<VariableArrayType>(Ty.getTypePtr()))
    return child_iterator(T);
  else
    return child_iterator(); 
}
Stmt::child_iterator SizeOfAlignOfTypeExpr::child_end() {
  return child_iterator(); 
}

// ArraySubscriptExpr
Stmt::child_iterator ArraySubscriptExpr::child_begin() {
  return &SubExprs[0];
}
Stmt::child_iterator ArraySubscriptExpr::child_end() {
  return &SubExprs[0]+END_EXPR;
}

// CallExpr
Stmt::child_iterator CallExpr::child_begin() {
  return &SubExprs[0];
}
Stmt::child_iterator CallExpr::child_end() {
  return &SubExprs[0]+NumArgs+ARGS_START;
}

// MemberExpr
Stmt::child_iterator MemberExpr::child_begin() { return &Base; }
Stmt::child_iterator MemberExpr::child_end() { return &Base+1; }

// ExtVectorElementExpr
Stmt::child_iterator ExtVectorElementExpr::child_begin() { return &Base; }
Stmt::child_iterator ExtVectorElementExpr::child_end() { return &Base+1; }

// CompoundLiteralExpr
Stmt::child_iterator CompoundLiteralExpr::child_begin() { return &Init; }
Stmt::child_iterator CompoundLiteralExpr::child_end() { return &Init+1; }

// CastExpr
Stmt::child_iterator CastExpr::child_begin() { return &Op; }
Stmt::child_iterator CastExpr::child_end() { return &Op+1; }

// BinaryOperator
Stmt::child_iterator BinaryOperator::child_begin() {
  return &SubExprs[0];
}
Stmt::child_iterator BinaryOperator::child_end() {
  return &SubExprs[0]+END_EXPR;
}

// ConditionalOperator
Stmt::child_iterator ConditionalOperator::child_begin() {
  return &SubExprs[0];
}
Stmt::child_iterator ConditionalOperator::child_end() {
  return &SubExprs[0]+END_EXPR;
}

// AddrLabelExpr
Stmt::child_iterator AddrLabelExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator AddrLabelExpr::child_end() { return child_iterator(); }

// StmtExpr
Stmt::child_iterator StmtExpr::child_begin() { return &SubStmt; }
Stmt::child_iterator StmtExpr::child_end() { return &SubStmt+1; }

// TypesCompatibleExpr
Stmt::child_iterator TypesCompatibleExpr::child_begin() {
  return child_iterator();
}

Stmt::child_iterator TypesCompatibleExpr::child_end() {
  return child_iterator();
}

// ChooseExpr
Stmt::child_iterator ChooseExpr::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator ChooseExpr::child_end() { return &SubExprs[0]+END_EXPR; }

// OverloadExpr
Stmt::child_iterator OverloadExpr::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator OverloadExpr::child_end() { return &SubExprs[0]+NumExprs; }

// ShuffleVectorExpr
Stmt::child_iterator ShuffleVectorExpr::child_begin() {
  return &SubExprs[0];
}
Stmt::child_iterator ShuffleVectorExpr::child_end() {
  return &SubExprs[0]+NumExprs;
}

// VAArgExpr
Stmt::child_iterator VAArgExpr::child_begin() { return &Val; }
Stmt::child_iterator VAArgExpr::child_end() { return &Val+1; }

// InitListExpr
Stmt::child_iterator InitListExpr::child_begin() {
  return InitExprs.size() ? &InitExprs[0] : 0;
}
Stmt::child_iterator InitListExpr::child_end() {
  return InitExprs.size() ? &InitExprs[0] + InitExprs.size() : 0;
}

// ObjCStringLiteral
Stmt::child_iterator ObjCStringLiteral::child_begin() { 
  return child_iterator();
}
Stmt::child_iterator ObjCStringLiteral::child_end() {
  return child_iterator();
}

// ObjCEncodeExpr
Stmt::child_iterator ObjCEncodeExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator ObjCEncodeExpr::child_end() { return child_iterator(); }

// ObjCSelectorExpr
Stmt::child_iterator ObjCSelectorExpr::child_begin() { 
  return child_iterator();
}
Stmt::child_iterator ObjCSelectorExpr::child_end() {
  return child_iterator();
}

// ObjCProtocolExpr
Stmt::child_iterator ObjCProtocolExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator ObjCProtocolExpr::child_end() {
  return child_iterator();
}

// ObjCMessageExpr
Stmt::child_iterator ObjCMessageExpr::child_begin() {  
  return getReceiver() ? &SubExprs[0] : &SubExprs[0] + ARGS_START;
}
Stmt::child_iterator ObjCMessageExpr::child_end() {
  return &SubExprs[0]+ARGS_START+getNumArgs();
}

// Blocks
Stmt::child_iterator BlockExpr::child_begin() {
  return reinterpret_cast<Stmt**>(&Body);
}
Stmt::child_iterator BlockExpr::child_end() {
  return reinterpret_cast<Stmt**>(&Body)+1;
}

Stmt::child_iterator BlockDeclRefExpr::child_begin(){return child_iterator();}
Stmt::child_iterator BlockDeclRefExpr::child_end() { return child_iterator();}

