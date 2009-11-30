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
#include "clang/AST/ExprCXX.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace clang;

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

void ExplicitTemplateArgumentList::initializeFrom(
                                      const TemplateArgumentListInfo &Info) {
  LAngleLoc = Info.getLAngleLoc();
  RAngleLoc = Info.getRAngleLoc();
  NumTemplateArgs = Info.size();

  TemplateArgumentLoc *ArgBuffer = getTemplateArgs();
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    new (&ArgBuffer[i]) TemplateArgumentLoc(Info[i]);
}

void ExplicitTemplateArgumentList::copyInto(
                                      TemplateArgumentListInfo &Info) const {
  Info.setLAngleLoc(LAngleLoc);
  Info.setRAngleLoc(RAngleLoc);
  for (unsigned I = 0; I != NumTemplateArgs; ++I)
    Info.addArgument(getTemplateArgs()[I]);
}

std::size_t ExplicitTemplateArgumentList::sizeFor(
                                      const TemplateArgumentListInfo &Info) {
  return sizeof(ExplicitTemplateArgumentList) +
         sizeof(TemplateArgumentLoc) * Info.size();
}

void DeclRefExpr::computeDependence() {
  TypeDependent = false;
  ValueDependent = false;
  
  NamedDecl *D = getDecl();

  // (TD) C++ [temp.dep.expr]p3:
  //   An id-expression is type-dependent if it contains:
  //
  // and 
  //
  // (VD) C++ [temp.dep.constexpr]p2:
  //  An identifier is value-dependent if it is:

  //  (TD)  - an identifier that was declared with dependent type
  //  (VD)  - a name declared with a dependent type,
  if (getType()->isDependentType()) {
    TypeDependent = true;
    ValueDependent = true;
  }
  //  (TD)  - a conversion-function-id that specifies a dependent type
  else if (D->getDeclName().getNameKind() 
                               == DeclarationName::CXXConversionFunctionName &&
           D->getDeclName().getCXXNameType()->isDependentType()) {
    TypeDependent = true;
    ValueDependent = true;
  }
  //  (TD)  - a template-id that is dependent,
  else if (hasExplicitTemplateArgumentList() && 
           TemplateSpecializationType::anyDependentTemplateArguments(
                                                       getTemplateArgs(), 
                                                       getNumTemplateArgs())) {
    TypeDependent = true;
    ValueDependent = true;
  }
  //  (VD)  - the name of a non-type template parameter,
  else if (isa<NonTypeTemplateParmDecl>(D))
    ValueDependent = true;
  //  (VD) - a constant with integral or enumeration type and is
  //         initialized with an expression that is value-dependent.
  else if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    if (Var->getType()->isIntegralType() &&
        Var->getType().getCVRQualifiers() == Qualifiers::Const &&
        Var->getInit() &&
        Var->getInit()->isValueDependent())
    ValueDependent = true;
  }
  //  (TD)  - a nested-name-specifier or a qualified-id that names a
  //          member of an unknown specialization.
  //        (handled by DependentScopeDeclRefExpr)
}

DeclRefExpr::DeclRefExpr(NestedNameSpecifier *Qualifier, 
                         SourceRange QualifierRange,
                         NamedDecl *D, SourceLocation NameLoc,
                         const TemplateArgumentListInfo *TemplateArgs,
                         QualType T)
  : Expr(DeclRefExprClass, T, false, false),
    DecoratedD(D,
               (Qualifier? HasQualifierFlag : 0) |
               (TemplateArgs ? HasExplicitTemplateArgumentListFlag : 0)),
    Loc(NameLoc) {
  assert(!isa<OverloadedFunctionDecl>(D));
  if (Qualifier) {
    NameQualifier *NQ = getNameQualifier();
    NQ->NNS = Qualifier;
    NQ->Range = QualifierRange;
  }
      
  if (TemplateArgs)
    getExplicitTemplateArgumentList()->initializeFrom(*TemplateArgs);

  computeDependence();
}

DeclRefExpr *DeclRefExpr::Create(ASTContext &Context,
                                 NestedNameSpecifier *Qualifier,
                                 SourceRange QualifierRange,
                                 NamedDecl *D,
                                 SourceLocation NameLoc,
                                 QualType T,
                                 const TemplateArgumentListInfo *TemplateArgs) {
  std::size_t Size = sizeof(DeclRefExpr);
  if (Qualifier != 0)
    Size += sizeof(NameQualifier);
  
  if (TemplateArgs)
    Size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);
  
  void *Mem = Context.Allocate(Size, llvm::alignof<DeclRefExpr>());
  return new (Mem) DeclRefExpr(Qualifier, QualifierRange, D, NameLoc,
                               TemplateArgs, T);
}

SourceRange DeclRefExpr::getSourceRange() const {
  // FIXME: Does not handle multi-token names well, e.g., operator[].
  SourceRange R(Loc);
  
  if (hasQualifier())
    R.setBegin(getQualifierRange().getBegin());
  if (hasExplicitTemplateArgumentList())
    R.setEnd(getRAngleLoc());
  return R;
}

// FIXME: Maybe this should use DeclPrinter with a special "print predefined
// expr" policy instead.
std::string PredefinedExpr::ComputeName(ASTContext &Context, IdentType IT,
                                        const Decl *CurrentDecl) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CurrentDecl)) {
    if (IT != PrettyFunction)
      return FD->getNameAsString();

    llvm::SmallString<256> Name;
    llvm::raw_svector_ostream Out(Name);

    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MD->isVirtual())
        Out << "virtual ";
    }

    PrintingPolicy Policy(Context.getLangOptions());
    Policy.SuppressTagKind = true;

    std::string Proto = FD->getQualifiedNameAsString(Policy);

    const FunctionType *AFT = FD->getType()->getAs<FunctionType>();
    const FunctionProtoType *FT = 0;
    if (FD->hasWrittenPrototype())
      FT = dyn_cast<FunctionProtoType>(AFT);

    Proto += "(";
    if (FT) {
      llvm::raw_string_ostream POut(Proto);
      for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
        if (i) POut << ", ";
        std::string Param;
        FD->getParamDecl(i)->getType().getAsStringInternal(Param, Policy);
        POut << Param;
      }

      if (FT->isVariadic()) {
        if (FD->getNumParams()) POut << ", ";
        POut << "...";
      }
    }
    Proto += ")";

    AFT->getResultType().getAsStringInternal(Proto, Policy);

    Out << Proto;

    Out.flush();
    return Name.str().str();
  }
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(CurrentDecl)) {
    llvm::SmallString<256> Name;
    llvm::raw_svector_ostream Out(Name);
    Out << (MD->isInstanceMethod() ? '-' : '+');
    Out << '[';
    Out << MD->getClassInterface()->getNameAsString();
    if (const ObjCCategoryImplDecl *CID =
        dyn_cast<ObjCCategoryImplDecl>(MD->getDeclContext())) {
      Out << '(';
      Out <<  CID->getNameAsString();
      Out <<  ')';
    }
    Out <<  ' ';
    Out << MD->getSelector().getAsString();
    Out <<  ']';

    Out.flush();
    return Name.str().str();
  }
  if (isa<TranslationUnitDecl>(CurrentDecl) && IT == PrettyFunction) {
    // __PRETTY_FUNCTION__ -> "top level", the others produce an empty string.
    return "top level";
  }
  return "";
}

/// getValueAsApproximateDouble - This returns the value as an inaccurate
/// double.  Note that this may cause loss of precision, but is useful for
/// debugging dumps, etc.
double FloatingLiteral::getValueAsApproximateDouble() const {
  llvm::APFloat V = getValue();
  bool ignored;
  V.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmNearestTiesToEven,
            &ignored);
  return V.convertToDouble();
}

StringLiteral *StringLiteral::Create(ASTContext &C, const char *StrData,
                                     unsigned ByteLength, bool Wide,
                                     QualType Ty,
                                     const SourceLocation *Loc,
                                     unsigned NumStrs) {
  // Allocate enough space for the StringLiteral plus an array of locations for
  // any concatenated string tokens.
  void *Mem = C.Allocate(sizeof(StringLiteral)+
                         sizeof(SourceLocation)*(NumStrs-1),
                         llvm::alignof<StringLiteral>());
  StringLiteral *SL = new (Mem) StringLiteral(Ty);

  // OPTIMIZE: could allocate this appended to the StringLiteral.
  char *AStrData = new (C, 1) char[ByteLength];
  memcpy(AStrData, StrData, ByteLength);
  SL->StrData = AStrData;
  SL->ByteLength = ByteLength;
  SL->IsWide = Wide;
  SL->TokLocs[0] = Loc[0];
  SL->NumConcatenated = NumStrs;

  if (NumStrs != 1)
    memcpy(&SL->TokLocs[1], Loc+1, sizeof(SourceLocation)*(NumStrs-1));
  return SL;
}

StringLiteral *StringLiteral::CreateEmpty(ASTContext &C, unsigned NumStrs) {
  void *Mem = C.Allocate(sizeof(StringLiteral)+
                         sizeof(SourceLocation)*(NumStrs-1),
                         llvm::alignof<StringLiteral>());
  StringLiteral *SL = new (Mem) StringLiteral(QualType());
  SL->StrData = 0;
  SL->ByteLength = 0;
  SL->NumConcatenated = NumStrs;
  return SL;
}

void StringLiteral::DoDestroy(ASTContext &C) {
  C.Deallocate(const_cast<char*>(StrData));
  Expr::DoDestroy(C);
}

void StringLiteral::setString(ASTContext &C, llvm::StringRef Str) {
  if (StrData)
    C.Deallocate(const_cast<char*>(StrData));

  char *AStrData = new (C, 1) char[Str.size()];
  memcpy(AStrData, Str.data(), Str.size());
  StrData = AStrData;
  ByteLength = Str.size();
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
  case Extension: return "__extension__";
  case OffsetOf: return "__builtin_offsetof";
  }
}

UnaryOperator::Opcode
UnaryOperator::getOverloadedOpcode(OverloadedOperatorKind OO, bool Postfix) {
  switch (OO) {
  default: assert(false && "No unary operator for overloaded function");
  case OO_PlusPlus:   return Postfix ? PostInc : PreInc;
  case OO_MinusMinus: return Postfix ? PostDec : PreDec;
  case OO_Amp:        return AddrOf;
  case OO_Star:       return Deref;
  case OO_Plus:       return Plus;
  case OO_Minus:      return Minus;
  case OO_Tilde:      return Not;
  case OO_Exclaim:    return LNot;
  }
}

OverloadedOperatorKind UnaryOperator::getOverloadedOperator(Opcode Opc) {
  switch (Opc) {
  case PostInc: case PreInc: return OO_PlusPlus;
  case PostDec: case PreDec: return OO_MinusMinus;
  case AddrOf: return OO_Amp;
  case Deref: return OO_Star;
  case Plus: return OO_Plus;
  case Minus: return OO_Minus;
  case Not: return OO_Tilde;
  case LNot: return OO_Exclaim;
  default: return OO_None;
  }
}


//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

CallExpr::CallExpr(ASTContext& C, StmtClass SC, Expr *fn, Expr **args,
                   unsigned numargs, QualType t, SourceLocation rparenloc)
  : Expr(SC, t,
         fn->isTypeDependent() || hasAnyTypeDependentArguments(args, numargs),
         fn->isValueDependent() || hasAnyValueDependentArguments(args,numargs)),
    NumArgs(numargs) {

  SubExprs = new (C) Stmt*[numargs+1];
  SubExprs[FN] = fn;
  for (unsigned i = 0; i != numargs; ++i)
    SubExprs[i+ARGS_START] = args[i];

  RParenLoc = rparenloc;
}

CallExpr::CallExpr(ASTContext& C, Expr *fn, Expr **args, unsigned numargs,
                   QualType t, SourceLocation rparenloc)
  : Expr(CallExprClass, t,
         fn->isTypeDependent() || hasAnyTypeDependentArguments(args, numargs),
         fn->isValueDependent() || hasAnyValueDependentArguments(args,numargs)),
    NumArgs(numargs) {

  SubExprs = new (C) Stmt*[numargs+1];
  SubExprs[FN] = fn;
  for (unsigned i = 0; i != numargs; ++i)
    SubExprs[i+ARGS_START] = args[i];

  RParenLoc = rparenloc;
}

CallExpr::CallExpr(ASTContext &C, StmtClass SC, EmptyShell Empty)
  : Expr(SC, Empty), SubExprs(0), NumArgs(0) {
  SubExprs = new (C) Stmt*[1];
}

void CallExpr::DoDestroy(ASTContext& C) {
  DestroyChildren(C);
  if (SubExprs) C.Deallocate(SubExprs);
  this->~CallExpr();
  C.Deallocate(this);
}

FunctionDecl *CallExpr::getDirectCallee() {
  Expr *CEE = getCallee()->IgnoreParenCasts();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE))
    return dyn_cast<FunctionDecl>(DRE->getDecl());

  return 0;
}

/// setNumArgs - This changes the number of arguments present in this call.
/// Any orphaned expressions are deleted by this, and any new operands are set
/// to null.
void CallExpr::setNumArgs(ASTContext& C, unsigned NumArgs) {
  // No change, just return.
  if (NumArgs == getNumArgs()) return;

  // If shrinking # arguments, just delete the extras and forgot them.
  if (NumArgs < getNumArgs()) {
    for (unsigned i = NumArgs, e = getNumArgs(); i != e; ++i)
      getArg(i)->Destroy(C);
    this->NumArgs = NumArgs;
    return;
  }

  // Otherwise, we are growing the # arguments.  New an bigger argument array.
  Stmt **NewSubExprs = new (C) Stmt*[NumArgs+1];
  // Copy over args.
  for (unsigned i = 0; i != getNumArgs()+ARGS_START; ++i)
    NewSubExprs[i] = SubExprs[i];
  // Null out new args.
  for (unsigned i = getNumArgs()+ARGS_START; i != NumArgs+ARGS_START; ++i)
    NewSubExprs[i] = 0;

  if (SubExprs) C.Deallocate(SubExprs);
  SubExprs = NewSubExprs;
  this->NumArgs = NumArgs;
}

/// isBuiltinCall - If this is a call to a builtin, return the builtin ID.  If
/// not, return 0.
unsigned CallExpr::isBuiltinCall(ASTContext &Context) const {
  // All simple function calls (e.g. func()) are implicitly cast to pointer to
  // function. As a result, we try and obtain the DeclRefExpr from the
  // ImplicitCastExpr.
  const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(getCallee());
  if (!ICE) // FIXME: deal with more complex calls (e.g. (func)(), (*func)()).
    return 0;

  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
  if (!DRE)
    return 0;

  const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FDecl)
    return 0;

  if (!FDecl->getIdentifier())
    return 0;

  return FDecl->getBuiltinID();
}

QualType CallExpr::getCallReturnType() const {
  QualType CalleeType = getCallee()->getType();
  if (const PointerType *FnTypePtr = CalleeType->getAs<PointerType>())
    CalleeType = FnTypePtr->getPointeeType();
  else if (const BlockPointerType *BPT = CalleeType->getAs<BlockPointerType>())
    CalleeType = BPT->getPointeeType();

  const FunctionType *FnType = CalleeType->getAs<FunctionType>();
  return FnType->getResultType();
}

MemberExpr::MemberExpr(Expr *base, bool isarrow, NestedNameSpecifier *qual,
                       SourceRange qualrange, NamedDecl *memberdecl,
                       SourceLocation l, const TemplateArgumentListInfo *targs,
                       QualType ty)
  : Expr(MemberExprClass, ty,
         base->isTypeDependent() || (qual && qual->isDependent()),
         base->isValueDependent() || (qual && qual->isDependent())),
    Base(base), MemberDecl(memberdecl), MemberLoc(l), IsArrow(isarrow),
    HasQualifier(qual != 0), HasExplicitTemplateArgumentList(targs) {
  // Initialize the qualifier, if any.
  if (HasQualifier) {
    NameQualifier *NQ = getMemberQualifier();
    NQ->NNS = qual;
    NQ->Range = qualrange;
  }

  // Initialize the explicit template argument list, if any.
  if (targs)
    getExplicitTemplateArgumentList()->initializeFrom(*targs);
}

MemberExpr *MemberExpr::Create(ASTContext &C, Expr *base, bool isarrow,
                               NestedNameSpecifier *qual,
                               SourceRange qualrange,
                               NamedDecl *memberdecl,
                               SourceLocation l,
                               const TemplateArgumentListInfo *targs,
                               QualType ty) {
  std::size_t Size = sizeof(MemberExpr);
  if (qual != 0)
    Size += sizeof(NameQualifier);

  if (targs)
    Size += ExplicitTemplateArgumentList::sizeFor(*targs);

  void *Mem = C.Allocate(Size, llvm::alignof<MemberExpr>());
  return new (Mem) MemberExpr(base, isarrow, qual, qualrange, memberdecl, l,
                              targs, ty);
}

const char *CastExpr::getCastKindName() const {
  switch (getCastKind()) {
  case CastExpr::CK_Unknown:
    return "Unknown";
  case CastExpr::CK_BitCast:
    return "BitCast";
  case CastExpr::CK_NoOp:
    return "NoOp";
  case CastExpr::CK_BaseToDerived:
    return "BaseToDerived";
  case CastExpr::CK_DerivedToBase:
    return "DerivedToBase";
  case CastExpr::CK_Dynamic:
    return "Dynamic";
  case CastExpr::CK_ToUnion:
    return "ToUnion";
  case CastExpr::CK_ArrayToPointerDecay:
    return "ArrayToPointerDecay";
  case CastExpr::CK_FunctionToPointerDecay:
    return "FunctionToPointerDecay";
  case CastExpr::CK_NullToMemberPointer:
    return "NullToMemberPointer";
  case CastExpr::CK_BaseToDerivedMemberPointer:
    return "BaseToDerivedMemberPointer";
  case CastExpr::CK_DerivedToBaseMemberPointer:
    return "DerivedToBaseMemberPointer";
  case CastExpr::CK_UserDefinedConversion:
    return "UserDefinedConversion";
  case CastExpr::CK_ConstructorConversion:
    return "ConstructorConversion";
  case CastExpr::CK_IntegralToPointer:
    return "IntegralToPointer";
  case CastExpr::CK_PointerToIntegral:
    return "PointerToIntegral";
  case CastExpr::CK_ToVoid:
    return "ToVoid";
  case CastExpr::CK_VectorSplat:
    return "VectorSplat";
  case CastExpr::CK_IntegralCast:
    return "IntegralCast";
  case CastExpr::CK_IntegralToFloating:
    return "IntegralToFloating";
  case CastExpr::CK_FloatingToIntegral:
    return "FloatingToIntegral";
  case CastExpr::CK_FloatingCast:
    return "FloatingCast";
  case CastExpr::CK_MemberPointerToBoolean:
    return "MemberPointerToBoolean";
  }

  assert(0 && "Unhandled cast kind!");
  return 0;
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "<<=".
const char *BinaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  case PtrMemD:   return ".*";
  case PtrMemI:   return "->*";
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

  return "";
}

BinaryOperator::Opcode
BinaryOperator::getOverloadedOpcode(OverloadedOperatorKind OO) {
  switch (OO) {
  default: assert(false && "Not an overloadable binary operator");
  case OO_Plus: return Add;
  case OO_Minus: return Sub;
  case OO_Star: return Mul;
  case OO_Slash: return Div;
  case OO_Percent: return Rem;
  case OO_Caret: return Xor;
  case OO_Amp: return And;
  case OO_Pipe: return Or;
  case OO_Equal: return Assign;
  case OO_Less: return LT;
  case OO_Greater: return GT;
  case OO_PlusEqual: return AddAssign;
  case OO_MinusEqual: return SubAssign;
  case OO_StarEqual: return MulAssign;
  case OO_SlashEqual: return DivAssign;
  case OO_PercentEqual: return RemAssign;
  case OO_CaretEqual: return XorAssign;
  case OO_AmpEqual: return AndAssign;
  case OO_PipeEqual: return OrAssign;
  case OO_LessLess: return Shl;
  case OO_GreaterGreater: return Shr;
  case OO_LessLessEqual: return ShlAssign;
  case OO_GreaterGreaterEqual: return ShrAssign;
  case OO_EqualEqual: return EQ;
  case OO_ExclaimEqual: return NE;
  case OO_LessEqual: return LE;
  case OO_GreaterEqual: return GE;
  case OO_AmpAmp: return LAnd;
  case OO_PipePipe: return LOr;
  case OO_Comma: return Comma;
  case OO_ArrowStar: return PtrMemI;
  }
}

OverloadedOperatorKind BinaryOperator::getOverloadedOperator(Opcode Opc) {
  static const OverloadedOperatorKind OverOps[] = {
    /* .* Cannot be overloaded */OO_None, OO_ArrowStar,
    OO_Star, OO_Slash, OO_Percent,
    OO_Plus, OO_Minus,
    OO_LessLess, OO_GreaterGreater,
    OO_Less, OO_Greater, OO_LessEqual, OO_GreaterEqual,
    OO_EqualEqual, OO_ExclaimEqual,
    OO_Amp,
    OO_Caret,
    OO_Pipe,
    OO_AmpAmp,
    OO_PipePipe,
    OO_Equal, OO_StarEqual,
    OO_SlashEqual, OO_PercentEqual,
    OO_PlusEqual, OO_MinusEqual,
    OO_LessLessEqual, OO_GreaterGreaterEqual,
    OO_AmpEqual, OO_CaretEqual,
    OO_PipeEqual,
    OO_Comma
  };
  return OverOps[Opc];
}

InitListExpr::InitListExpr(SourceLocation lbraceloc,
                           Expr **initExprs, unsigned numInits,
                           SourceLocation rbraceloc)
  : Expr(InitListExprClass, QualType(), false, false),
    LBraceLoc(lbraceloc), RBraceLoc(rbraceloc), SyntacticForm(0),
    UnionFieldInit(0), HadArrayRangeDesignator(false) 
{      
  for (unsigned I = 0; I != numInits; ++I) {
    if (initExprs[I]->isTypeDependent())
      TypeDependent = true;
    if (initExprs[I]->isValueDependent())
      ValueDependent = true;
  }
      
  InitExprs.insert(InitExprs.end(), initExprs, initExprs+numInits);
}

void InitListExpr::reserveInits(unsigned NumInits) {
  if (NumInits > InitExprs.size())
    InitExprs.reserve(NumInits);
}

void InitListExpr::resizeInits(ASTContext &Context, unsigned NumInits) {
  for (unsigned Idx = NumInits, LastIdx = InitExprs.size();
       Idx < LastIdx; ++Idx)
    InitExprs[Idx]->Destroy(Context);
  InitExprs.resize(NumInits, 0);
}

Expr *InitListExpr::updateInit(unsigned Init, Expr *expr) {
  if (Init >= InitExprs.size()) {
    InitExprs.insert(InitExprs.end(), Init - InitExprs.size() + 1, 0);
    InitExprs.back() = expr;
    return 0;
  }

  Expr *Result = cast_or_null<Expr>(InitExprs[Init]);
  InitExprs[Init] = expr;
  return Result;
}

/// getFunctionType - Return the underlying function type for this block.
///
const FunctionType *BlockExpr::getFunctionType() const {
  return getType()->getAs<BlockPointerType>()->
                    getPointeeType()->getAs<FunctionType>();
}

SourceLocation BlockExpr::getCaretLocation() const {
  return TheBlock->getCaretLocation();
}
const Stmt *BlockExpr::getBody() const {
  return TheBlock->getBody();
}
Stmt *BlockExpr::getBody() {
  return TheBlock->getBody();
}


//===----------------------------------------------------------------------===//
// Generic Expression Routines
//===----------------------------------------------------------------------===//

/// isUnusedResultAWarning - Return true if this immediate expression should
/// be warned about if the result is unused.  If so, fill in Loc and Ranges
/// with location to warn on and the source range[s] to report with the
/// warning.
bool Expr::isUnusedResultAWarning(SourceLocation &Loc, SourceRange &R1,
                                  SourceRange &R2, ASTContext &Ctx) const {
  // Don't warn if the expr is type dependent. The type could end up
  // instantiating to void.
  if (isTypeDependent())
    return false;

  switch (getStmtClass()) {
  default:
    Loc = getExprLoc();
    R1 = getSourceRange();
    return true;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->
      isUnusedResultAWarning(Loc, R1, R2, Ctx);
  case UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(this);

    switch (UO->getOpcode()) {
    default: break;
    case UnaryOperator::PostInc:
    case UnaryOperator::PostDec:
    case UnaryOperator::PreInc:
    case UnaryOperator::PreDec:                 // ++/--
      return false;  // Not a warning.
    case UnaryOperator::Deref:
      // Dereferencing a volatile pointer is a side-effect.
      if (Ctx.getCanonicalType(getType()).isVolatileQualified())
        return false;
      break;
    case UnaryOperator::Real:
    case UnaryOperator::Imag:
      // accessing a piece of a volatile complex is a side-effect.
      if (Ctx.getCanonicalType(UO->getSubExpr()->getType())
          .isVolatileQualified())
        return false;
      break;
    case UnaryOperator::Extension:
      return UO->getSubExpr()->isUnusedResultAWarning(Loc, R1, R2, Ctx);
    }
    Loc = UO->getOperatorLoc();
    R1 = UO->getSubExpr()->getSourceRange();
    return true;
  }
  case BinaryOperatorClass: {
    const BinaryOperator *BO = cast<BinaryOperator>(this);
    // Consider comma to have side effects if the LHS or RHS does.
    if (BO->getOpcode() == BinaryOperator::Comma)
      return (BO->getRHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx) ||
              BO->getLHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx));

    if (BO->isAssignmentOp())
      return false;
    Loc = BO->getOperatorLoc();
    R1 = BO->getLHS()->getSourceRange();
    R2 = BO->getRHS()->getSourceRange();
    return true;
  }
  case CompoundAssignOperatorClass:
    return false;

  case ConditionalOperatorClass: {
    // The condition must be evaluated, but if either the LHS or RHS is a
    // warning, warn about them.
    const ConditionalOperator *Exp = cast<ConditionalOperator>(this);
    if (Exp->getLHS() &&
        Exp->getLHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx))
      return true;
    return Exp->getRHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx);
  }

  case MemberExprClass:
    // If the base pointer or element is to a volatile pointer/field, accessing
    // it is a side effect.
    if (Ctx.getCanonicalType(getType()).isVolatileQualified())
      return false;
    Loc = cast<MemberExpr>(this)->getMemberLoc();
    R1 = SourceRange(Loc, Loc);
    R2 = cast<MemberExpr>(this)->getBase()->getSourceRange();
    return true;

  case ArraySubscriptExprClass:
    // If the base pointer or element is to a volatile pointer/field, accessing
    // it is a side effect.
    if (Ctx.getCanonicalType(getType()).isVolatileQualified())
      return false;
    Loc = cast<ArraySubscriptExpr>(this)->getRBracketLoc();
    R1 = cast<ArraySubscriptExpr>(this)->getLHS()->getSourceRange();
    R2 = cast<ArraySubscriptExpr>(this)->getRHS()->getSourceRange();
    return true;

  case CallExprClass:
  case CXXOperatorCallExprClass:
  case CXXMemberCallExprClass: {
    // If this is a direct call, get the callee.
    const CallExpr *CE = cast<CallExpr>(this);
    if (const FunctionDecl *FD = CE->getDirectCallee()) {
      // If the callee has attribute pure, const, or warn_unused_result, warn
      // about it. void foo() { strlen("bar"); } should warn.
      //
      // Note: If new cases are added here, DiagnoseUnusedExprResult should be
      // updated to match for QoI.
      if (FD->getAttr<WarnUnusedResultAttr>() ||
          FD->getAttr<PureAttr>() || FD->getAttr<ConstAttr>()) {
        Loc = CE->getCallee()->getLocStart();
        R1 = CE->getCallee()->getSourceRange();

        if (unsigned NumArgs = CE->getNumArgs())
          R2 = SourceRange(CE->getArg(0)->getLocStart(),
                           CE->getArg(NumArgs-1)->getLocEnd());
        return true;
      }
    }
    return false;
  }

  case CXXTemporaryObjectExprClass:
  case CXXConstructExprClass:
    return false;

  case ObjCMessageExprClass:
    return false;

  case ObjCImplicitSetterGetterRefExprClass: {   // Dot syntax for message send.
#if 0
    const ObjCImplicitSetterGetterRefExpr *Ref =
      cast<ObjCImplicitSetterGetterRefExpr>(this);
    // FIXME: We really want the location of the '.' here.
    Loc = Ref->getLocation();
    R1 = SourceRange(Ref->getLocation(), Ref->getLocation());
    if (Ref->getBase())
      R2 = Ref->getBase()->getSourceRange();
#else
    Loc = getExprLoc();
    R1 = getSourceRange();
#endif
    return true;
  }
  case StmtExprClass: {
    // Statement exprs don't logically have side effects themselves, but are
    // sometimes used in macros in ways that give them a type that is unused.
    // For example ({ blah; foo(); }) will end up with a type if foo has a type.
    // however, if the result of the stmt expr is dead, we don't want to emit a
    // warning.
    const CompoundStmt *CS = cast<StmtExpr>(this)->getSubStmt();
    if (!CS->body_empty())
      if (const Expr *E = dyn_cast<Expr>(CS->body_back()))
        return E->isUnusedResultAWarning(Loc, R1, R2, Ctx);

    Loc = cast<StmtExpr>(this)->getLParenLoc();
    R1 = getSourceRange();
    return true;
  }
  case CStyleCastExprClass:
    // If this is an explicit cast to void, allow it.  People do this when they
    // think they know what they're doing :).
    if (getType()->isVoidType())
      return false;
    Loc = cast<CStyleCastExpr>(this)->getLParenLoc();
    R1 = cast<CStyleCastExpr>(this)->getSubExpr()->getSourceRange();
    return true;
  case CXXFunctionalCastExprClass: {
    const CastExpr *CE = cast<CastExpr>(this);
    
    // If this is a cast to void or a constructor conversion, check the operand.
    // Otherwise, the result of the cast is unused.
    if (CE->getCastKind() == CastExpr::CK_ToVoid ||
        CE->getCastKind() == CastExpr::CK_ConstructorConversion)
      return (cast<CastExpr>(this)->getSubExpr()
              ->isUnusedResultAWarning(Loc, R1, R2, Ctx));
    Loc = cast<CXXFunctionalCastExpr>(this)->getTypeBeginLoc();
    R1 = cast<CXXFunctionalCastExpr>(this)->getSubExpr()->getSourceRange();
    return true;
  }

  case ImplicitCastExprClass:
    // Check the operand, since implicit casts are inserted by Sema
    return (cast<ImplicitCastExpr>(this)
            ->getSubExpr()->isUnusedResultAWarning(Loc, R1, R2, Ctx));

  case CXXDefaultArgExprClass:
    return (cast<CXXDefaultArgExpr>(this)
            ->getExpr()->isUnusedResultAWarning(Loc, R1, R2, Ctx));

  case CXXNewExprClass:
    // FIXME: In theory, there might be new expressions that don't have side
    // effects (e.g. a placement new with an uninitialized POD).
  case CXXDeleteExprClass:
    return false;
  case CXXBindTemporaryExprClass:
    return (cast<CXXBindTemporaryExpr>(this)
            ->getSubExpr()->isUnusedResultAWarning(Loc, R1, R2, Ctx));
  case CXXExprWithTemporariesClass:
    return (cast<CXXExprWithTemporaries>(this)
            ->getSubExpr()->isUnusedResultAWarning(Loc, R1, R2, Ctx));
  }
}

/// DeclCanBeLvalue - Determine whether the given declaration can be
/// an lvalue. This is a helper routine for isLvalue.
static bool DeclCanBeLvalue(const NamedDecl *Decl, ASTContext &Ctx) {
  // C++ [temp.param]p6:
  //   A non-type non-reference template-parameter is not an lvalue.
  if (const NonTypeTemplateParmDecl *NTTParm
        = dyn_cast<NonTypeTemplateParmDecl>(Decl))
    return NTTParm->getType()->isReferenceType();

  return isa<VarDecl>(Decl) || isa<FieldDecl>(Decl) ||
    // C++ 3.10p2: An lvalue refers to an object or function.
    (Ctx.getLangOptions().CPlusPlus &&
     (isa<FunctionDecl>(Decl) || isa<OverloadedFunctionDecl>(Decl) ||
      isa<FunctionTemplateDecl>(Decl)));
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
  assert(!TR->isReferenceType() && "Expressions can't have reference type.");

  isLvalueResult Res = isLvalueInternal(Ctx);
  if (Res != LV_Valid || Ctx.getLangOptions().CPlusPlus)
    return Res;

  // first, check the type (C99 6.3.2.1). Expressions with function
  // type in C are not lvalues, but they can be lvalues in C++.
  if (TR->isFunctionType() || TR == Ctx.OverloadTy)
    return LV_NotObjectType;

  // Allow qualified void which is an incomplete type other than void (yuck).
  if (TR->isVoidType() && !Ctx.getCanonicalType(TR).hasQualifiers())
    return LV_IncompleteVoidType;

  return LV_Valid;
}

// Check whether the expression can be sanely treated like an l-value
Expr::isLvalueResult Expr::isLvalueInternal(ASTContext &Ctx) const {
  switch (getStmtClass()) {
  case StringLiteralClass:  // C99 6.5.1p4
  case ObjCEncodeExprClass: // @encode behaves like its string in every way.
    return LV_Valid;
  case ArraySubscriptExprClass: // C99 6.5.3p4 (e1[e2] == (*((e1)+(e2))))
    // For vectors, make sure base is an lvalue (i.e. not a function call).
    if (cast<ArraySubscriptExpr>(this)->getBase()->getType()->isVectorType())
      return cast<ArraySubscriptExpr>(this)->getBase()->isLvalue(Ctx);
    return LV_Valid;
  case DeclRefExprClass: { // C99 6.5.1p2
    const NamedDecl *RefdDecl = cast<DeclRefExpr>(this)->getDecl();
    if (DeclCanBeLvalue(RefdDecl, Ctx))
      return LV_Valid;
    break;
  }
  case BlockDeclRefExprClass: {
    const BlockDeclRefExpr *BDR = cast<BlockDeclRefExpr>(this);
    if (isa<VarDecl>(BDR->getDecl()))
      return LV_Valid;
    break;
  }
  case MemberExprClass: {
    const MemberExpr *m = cast<MemberExpr>(this);
    if (Ctx.getLangOptions().CPlusPlus) { // C++ [expr.ref]p4:
      NamedDecl *Member = m->getMemberDecl();
      // C++ [expr.ref]p4:
      //   If E2 is declared to have type "reference to T", then E1.E2
      //   is an lvalue.
      if (ValueDecl *Value = dyn_cast<ValueDecl>(Member))
        if (Value->getType()->isReferenceType())
          return LV_Valid;

      //   -- If E2 is a static data member [...] then E1.E2 is an lvalue.
      if (isa<VarDecl>(Member) && Member->getDeclContext()->isRecord())
        return LV_Valid;

      //   -- If E2 is a non-static data member [...]. If E1 is an
      //      lvalue, then E1.E2 is an lvalue.
      if (isa<FieldDecl>(Member))
        return m->isArrow() ? LV_Valid : m->getBase()->isLvalue(Ctx);

      //   -- If it refers to a static member function [...], then
      //      E1.E2 is an lvalue.
      //   -- Otherwise, if E1.E2 refers to a non-static member
      //      function [...], then E1.E2 is not an lvalue.
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Member))
        return Method->isStatic()? LV_Valid : LV_MemberFunction;

      //   -- If E2 is a member enumerator [...], the expression E1.E2
      //      is not an lvalue.
      if (isa<EnumConstantDecl>(Member))
        return LV_InvalidExpression;

        // Not an lvalue.
      return LV_InvalidExpression;
    }

    // C99 6.5.2.3p4
    return m->isArrow() ? LV_Valid : m->getBase()->isLvalue(Ctx);
  }
  case UnaryOperatorClass:
    if (cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Deref)
      return LV_Valid; // C99 6.5.3p4

    if (cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Real ||
        cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Imag ||
        cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Extension)
      return cast<UnaryOperator>(this)->getSubExpr()->isLvalue(Ctx);  // GNU.

    if (Ctx.getLangOptions().CPlusPlus && // C++ [expr.pre.incr]p1
        (cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::PreInc ||
         cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::PreDec))
      return LV_Valid;
    break;
  case ImplicitCastExprClass:
    return cast<ImplicitCastExpr>(this)->isLvalueCast()? LV_Valid
                                                       : LV_InvalidExpression;
  case ParenExprClass: // C99 6.5.1p5
    return cast<ParenExpr>(this)->getSubExpr()->isLvalue(Ctx);
  case BinaryOperatorClass:
  case CompoundAssignOperatorClass: {
    const BinaryOperator *BinOp = cast<BinaryOperator>(this);

    if (Ctx.getLangOptions().CPlusPlus && // C++ [expr.comma]p1
        BinOp->getOpcode() == BinaryOperator::Comma)
      return BinOp->getRHS()->isLvalue(Ctx);

    // C++ [expr.mptr.oper]p6
    // The result of a .* expression is an lvalue only if its first operand is 
    // an lvalue and its second operand is a pointer to data member. 
    if (BinOp->getOpcode() == BinaryOperator::PtrMemD &&
        !BinOp->getType()->isFunctionType())
      return BinOp->getLHS()->isLvalue(Ctx);

    // The result of an ->* expression is an lvalue only if its second operand 
    // is a pointer to data member.
    if (BinOp->getOpcode() == BinaryOperator::PtrMemI &&
        !BinOp->getType()->isFunctionType()) {
      QualType Ty = BinOp->getRHS()->getType();
      if (Ty->isMemberPointerType() && !Ty->isMemberFunctionPointerType())
        return LV_Valid;
    }
    
    if (!BinOp->isAssignmentOp())
      return LV_InvalidExpression;

    if (Ctx.getLangOptions().CPlusPlus)
      // C++ [expr.ass]p1:
      //   The result of an assignment operation [...] is an lvalue.
      return LV_Valid;


    // C99 6.5.16:
    //   An assignment expression [...] is not an lvalue.
    return LV_InvalidExpression;
  }
  case CallExprClass:
  case CXXOperatorCallExprClass:
  case CXXMemberCallExprClass: {
    // C++0x [expr.call]p10
    //   A function call is an lvalue if and only if the result type
    //   is an lvalue reference.
    QualType ReturnType = cast<CallExpr>(this)->getCallReturnType();
    if (ReturnType->isLValueReferenceType())
      return LV_Valid;

    break;
  }
  case CompoundLiteralExprClass: // C99 6.5.2.5p5
    return LV_Valid;
  case ChooseExprClass:
    // __builtin_choose_expr is an lvalue if the selected operand is.
    return cast<ChooseExpr>(this)->getChosenSubExpr(Ctx)->isLvalue(Ctx);
  case ExtVectorElementExprClass:
    if (cast<ExtVectorElementExpr>(this)->containsDuplicateElements())
      return LV_DuplicateVectorComponents;
    return LV_Valid;
  case ObjCIvarRefExprClass: // ObjC instance variables are lvalues.
    return LV_Valid;
  case ObjCPropertyRefExprClass: // FIXME: check if read-only property.
    return LV_Valid;
  case ObjCImplicitSetterGetterRefExprClass: // FIXME: check if read-only property.
    return LV_Valid;
  case PredefinedExprClass:
    return LV_Valid;
  case UnresolvedLookupExprClass:
    return LV_Valid;
  case CXXDefaultArgExprClass:
    return cast<CXXDefaultArgExpr>(this)->getExpr()->isLvalue(Ctx);
  case CStyleCastExprClass:
  case CXXFunctionalCastExprClass:
  case CXXStaticCastExprClass:
  case CXXDynamicCastExprClass:
  case CXXReinterpretCastExprClass:
  case CXXConstCastExprClass:
    // The result of an explicit cast is an lvalue if the type we are
    // casting to is an lvalue reference type. See C++ [expr.cast]p1,
    // C++ [expr.static.cast]p2, C++ [expr.dynamic.cast]p2,
    // C++ [expr.reinterpret.cast]p1, C++ [expr.const.cast]p1.
    if (cast<ExplicitCastExpr>(this)->getTypeAsWritten()->
          isLValueReferenceType())
      return LV_Valid;
    break;
  case CXXTypeidExprClass:
    // C++ 5.2.8p1: The result of a typeid expression is an lvalue of ...
    return LV_Valid;
  case CXXBindTemporaryExprClass:
    return cast<CXXBindTemporaryExpr>(this)->getSubExpr()->
      isLvalueInternal(Ctx);
  case ConditionalOperatorClass: {
    // Complicated handling is only for C++.
    if (!Ctx.getLangOptions().CPlusPlus)
      return LV_InvalidExpression;

    // Sema should have taken care to ensure that a CXXTemporaryObjectExpr is
    // everywhere there's an object converted to an rvalue. Also, any other
    // casts should be wrapped by ImplicitCastExprs. There's just the special
    // case involving throws to work out.
    const ConditionalOperator *Cond = cast<ConditionalOperator>(this);
    Expr *True = Cond->getTrueExpr();
    Expr *False = Cond->getFalseExpr();
    // C++0x 5.16p2
    //   If either the second or the third operand has type (cv) void, [...]
    //   the result [...] is an rvalue.
    if (True->getType()->isVoidType() || False->getType()->isVoidType())
      return LV_InvalidExpression;

    // Both sides must be lvalues for the result to be an lvalue.
    if (True->isLvalue(Ctx) != LV_Valid || False->isLvalue(Ctx) != LV_Valid)
      return LV_InvalidExpression;

    // That's it.
    return LV_Valid;
  }

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
Expr::isModifiableLvalueResult
Expr::isModifiableLvalue(ASTContext &Ctx, SourceLocation *Loc) const {
  isLvalueResult lvalResult = isLvalue(Ctx);

  switch (lvalResult) {
  case LV_Valid:
    // C++ 3.10p11: Functions cannot be modified, but pointers to
    // functions can be modifiable.
    if (Ctx.getLangOptions().CPlusPlus && TR->isFunctionType())
      return MLV_NotObjectType;
    break;

  case LV_NotObjectType: return MLV_NotObjectType;
  case LV_IncompleteVoidType: return MLV_IncompleteVoidType;
  case LV_DuplicateVectorComponents: return MLV_DuplicateVectorComponents;
  case LV_InvalidExpression:
    // If the top level is a C-style cast, and the subexpression is a valid
    // lvalue, then this is probably a use of the old-school "cast as lvalue"
    // GCC extension.  We don't support it, but we want to produce good
    // diagnostics when it happens so that the user knows why.
    if (const CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(IgnoreParens())) {
      if (CE->getSubExpr()->isLvalue(Ctx) == LV_Valid) {
        if (Loc)
          *Loc = CE->getLParenLoc();
        return MLV_LValueCast;
      }
    }
    return MLV_InvalidExpression;
  case LV_MemberFunction: return MLV_MemberFunction;
  }

  // The following is illegal:
  //   void takeclosure(void (^C)(void));
  //   void func() { int x = 1; takeclosure(^{ x = 7; }); }
  //
  if (const BlockDeclRefExpr *BDR = dyn_cast<BlockDeclRefExpr>(this)) {
    if (!BDR->isByRef() && isa<VarDecl>(BDR->getDecl()))
      return MLV_NotBlockQualified;
  }

  // Assigning to an 'implicit' property?
  if (const ObjCImplicitSetterGetterRefExpr* Expr =
        dyn_cast<ObjCImplicitSetterGetterRefExpr>(this)) {
    if (Expr->getSetterMethod() == 0)
      return MLV_NoSetterProperty;
  }

  QualType CT = Ctx.getCanonicalType(getType());

  if (CT.isConstQualified())
    return MLV_ConstQualified;
  if (CT->isArrayType())
    return MLV_ArrayType;
  if (CT->isIncompleteType())
    return MLV_IncompleteType;

  if (const RecordType *r = CT->getAs<RecordType>()) {
    if (r->hasConstFields())
      return MLV_ConstQualified;
  }

  return MLV_Valid;
}

/// isOBJCGCCandidate - Check if an expression is objc gc'able.
/// returns true, if it is; false otherwise.
bool Expr::isOBJCGCCandidate(ASTContext &Ctx) const {
  switch (getStmtClass()) {
  default:
    return false;
  case ObjCIvarRefExprClass:
    return true;
  case Expr::UnaryOperatorClass:
    return cast<UnaryOperator>(this)->getSubExpr()->isOBJCGCCandidate(Ctx);
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isOBJCGCCandidate(Ctx);
  case ImplicitCastExprClass:
    return cast<ImplicitCastExpr>(this)->getSubExpr()->isOBJCGCCandidate(Ctx);
  case CStyleCastExprClass:
    return cast<CStyleCastExpr>(this)->getSubExpr()->isOBJCGCCandidate(Ctx);
  case DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(this)->getDecl();
    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
      QualType T = VD->getType();
      // dereferencing to a  pointer is always a gc'able candidate,
      // unless it is __weak.
      return T->isPointerType() &&
             (Ctx.getObjCGCAttrKind(T) != Qualifiers::Weak);
    }
    return false;
  }
  case MemberExprClass: {
    const MemberExpr *M = cast<MemberExpr>(this);
    return M->getBase()->isOBJCGCCandidate(Ctx);
  }
  case ArraySubscriptExprClass:
    return cast<ArraySubscriptExpr>(this)->getBase()->isOBJCGCCandidate(Ctx);
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

/// IgnoreParenNoopCasts - Ignore parentheses and casts that do not change the
/// value (including ptr->int casts of the same size).  Strip off any
/// ParenExpr or CastExprs, returning their operand.
Expr *Expr::IgnoreParenNoopCasts(ASTContext &Ctx) {
  Expr *E = this;
  while (true) {
    if (ParenExpr *P = dyn_cast<ParenExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }

    if (CastExpr *P = dyn_cast<CastExpr>(E)) {
      // We ignore integer <-> casts that are of the same width, ptr<->ptr and
      // ptr<->int casts of the same width.  We also ignore all identify casts.
      Expr *SE = P->getSubExpr();

      if (Ctx.hasSameUnqualifiedType(E->getType(), SE->getType())) {
        E = SE;
        continue;
      }

      if ((E->getType()->isPointerType() || E->getType()->isIntegralType()) &&
          (SE->getType()->isPointerType() || SE->getType()->isIntegralType()) &&
          Ctx.getTypeSize(E->getType()) == Ctx.getTypeSize(SE->getType())) {
        E = SE;
        continue;
      }
    }

    return E;
  }
}


/// hasAnyTypeDependentArguments - Determines if any of the expressions
/// in Exprs is type-dependent.
bool Expr::hasAnyTypeDependentArguments(Expr** Exprs, unsigned NumExprs) {
  for (unsigned I = 0; I < NumExprs; ++I)
    if (Exprs[I]->isTypeDependent())
      return true;

  return false;
}

/// hasAnyValueDependentArguments - Determines if any of the expressions
/// in Exprs is value-dependent.
bool Expr::hasAnyValueDependentArguments(Expr** Exprs, unsigned NumExprs) {
  for (unsigned I = 0; I < NumExprs; ++I)
    if (Exprs[I]->isValueDependent())
      return true;

  return false;
}

bool Expr::isConstantInitializer(ASTContext &Ctx) const {
  // This function is attempting whether an expression is an initializer
  // which can be evaluated at compile-time.  isEvaluatable handles most
  // of the cases, but it can't deal with some initializer-specific
  // expressions, and it can't deal with aggregates; we deal with those here,
  // and fall back to isEvaluatable for the other cases.

  // FIXME: This function assumes the variable being assigned to
  // isn't a reference type!

  switch (getStmtClass()) {
  default: break;
  case StringLiteralClass:
  case ObjCStringLiteralClass:
  case ObjCEncodeExprClass:
    return true;
  case CompoundLiteralExprClass: {
    // This handles gcc's extension that allows global initializers like
    // "struct x {int x;} x = (struct x) {};".
    // FIXME: This accepts other cases it shouldn't!
    const Expr *Exp = cast<CompoundLiteralExpr>(this)->getInitializer();
    return Exp->isConstantInitializer(Ctx);
  }
  case InitListExprClass: {
    // FIXME: This doesn't deal with fields with reference types correctly.
    // FIXME: This incorrectly allows pointers cast to integers to be assigned
    // to bitfields.
    const InitListExpr *Exp = cast<InitListExpr>(this);
    unsigned numInits = Exp->getNumInits();
    for (unsigned i = 0; i < numInits; i++) {
      if (!Exp->getInit(i)->isConstantInitializer(Ctx))
        return false;
    }
    return true;
  }
  case ImplicitValueInitExprClass:
    return true;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isConstantInitializer(Ctx);
  case UnaryOperatorClass: {
    const UnaryOperator* Exp = cast<UnaryOperator>(this);
    if (Exp->getOpcode() == UnaryOperator::Extension)
      return Exp->getSubExpr()->isConstantInitializer(Ctx);
    break;
  }
  case BinaryOperatorClass: {
    // Special case &&foo - &&bar.  It would be nice to generalize this somehow
    // but this handles the common case.
    const BinaryOperator *Exp = cast<BinaryOperator>(this);
    if (Exp->getOpcode() == BinaryOperator::Sub &&
        isa<AddrLabelExpr>(Exp->getLHS()->IgnoreParenNoopCasts(Ctx)) &&
        isa<AddrLabelExpr>(Exp->getRHS()->IgnoreParenNoopCasts(Ctx)))
      return true;
    break;
  }
  case ImplicitCastExprClass:
  case CStyleCastExprClass:
    // Handle casts with a destination that's a struct or union; this
    // deals with both the gcc no-op struct cast extension and the
    // cast-to-union extension.
    if (getType()->isRecordType())
      return cast<CastExpr>(this)->getSubExpr()->isConstantInitializer(Ctx);
      
    // Integer->integer casts can be handled here, which is important for
    // things like (int)(&&x-&&y).  Scary but true.
    if (getType()->isIntegerType() &&
        cast<CastExpr>(this)->getSubExpr()->getType()->isIntegerType())
      return cast<CastExpr>(this)->getSubExpr()->isConstantInitializer(Ctx);
      
    break;
  }
  return isEvaluatable(Ctx);
}

/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.

// CheckICE - This function does the fundamental ICE checking: the returned
// ICEDiag contains a Val of 0, 1, or 2, and a possibly null SourceLocation.
// Note that to reduce code duplication, this helper does no evaluation
// itself; the caller checks whether the expression is evaluatable, and
// in the rare cases where CheckICE actually cares about the evaluated
// value, it calls into Evalute.
//
// Meanings of Val:
// 0: This expression is an ICE if it can be evaluated by Evaluate.
// 1: This expression is not an ICE, but if it isn't evaluated, it's
//    a legal subexpression for an ICE. This return value is used to handle
//    the comma operator in C99 mode.
// 2: This expression is not an ICE, and is not a legal subexpression for one.

struct ICEDiag {
  unsigned Val;
  SourceLocation Loc;

  public:
  ICEDiag(unsigned v, SourceLocation l) : Val(v), Loc(l) {}
  ICEDiag() : Val(0) {}
};

ICEDiag NoDiag() { return ICEDiag(); }

static ICEDiag CheckEvalInICE(const Expr* E, ASTContext &Ctx) {
  Expr::EvalResult EVResult;
  if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
      !EVResult.Val.isInt()) {
    return ICEDiag(2, E->getLocStart());
  }
  return NoDiag();
}

static ICEDiag CheckICE(const Expr* E, ASTContext &Ctx) {
  assert(!E->isValueDependent() && "Should not see value dependent exprs!");
  if (!E->getType()->isIntegralType()) {
    return ICEDiag(2, E->getLocStart());
  }

  switch (E->getStmtClass()) {
#define STMT(Node, Base) case Expr::Node##Class:
#define EXPR(Node, Base)
#include "clang/AST/StmtNodes.def"
  case Expr::PredefinedExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::StringLiteralClass:
  case Expr::ArraySubscriptExprClass:
  case Expr::MemberExprClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::InitListExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::VAArgExprClass:
  case Expr::AddrLabelExprClass:
  case Expr::StmtExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXThrowExprClass:
  case Expr::CXXNewExprClass:
  case Expr::CXXDeleteExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::DependentScopeDeclRefExprClass:
  case Expr::CXXConstructExprClass:
  case Expr::CXXBindTemporaryExprClass:
  case Expr::CXXExprWithTemporariesClass:
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::UnresolvedMemberExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCImplicitSetterGetterRefExprClass:
  case Expr::ObjCSuperExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::BlockExprClass:
  case Expr::BlockDeclRefExprClass:
  case Expr::NoStmtClass:
  case Expr::ExprClass:
    return ICEDiag(2, E->getLocStart());

  case Expr::GNUNullExprClass:
    // GCC considers the GNU __null value to be an integral constant expression.
    return NoDiag();

  case Expr::ParenExprClass:
    return CheckICE(cast<ParenExpr>(E)->getSubExpr(), Ctx);
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXZeroInitValueExprClass:
  case Expr::TypesCompatibleExprClass:
  case Expr::UnaryTypeTraitExprClass:
    return NoDiag();
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    if (CE->isBuiltinCall(Ctx))
      return CheckEvalInICE(E, Ctx);
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::DeclRefExprClass:
    if (isa<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return NoDiag();
    if (Ctx.getLangOptions().CPlusPlus &&
        E->getType().getCVRQualifiers() == Qualifiers::Const) {
      // C++ 7.1.5.1p2
      //   A variable of non-volatile const-qualified integral or enumeration
      //   type initialized by an ICE can be used in ICEs.
      if (const VarDecl *Dcl =
              dyn_cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl())) {
        Qualifiers Quals = Ctx.getCanonicalType(Dcl->getType()).getQualifiers();
        if (Quals.hasVolatile() || !Quals.hasConst())
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
        
        // Look for the definition of this variable, which will actually have
        // an initializer.
        const VarDecl *Def = 0;
        const Expr *Init = Dcl->getDefinition(Def);
        if (Init) {
          if (Def->isInitKnownICE()) {
            // We have already checked whether this subexpression is an
            // integral constant expression.
            if (Def->isInitICE())
              return NoDiag();
            else
              return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          // C++ [class.static.data]p4:
          //   If a static data member is of const integral or const 
          //   enumeration type, its declaration in the class definition can
          //   specify a constant-initializer which shall be an integral 
          //   constant expression (5.19). In that case, the member can appear
          //   in integral constant expressions.
          if (Def->isOutOfLine()) {
            Dcl->setInitKnownICE(Ctx, false);
            return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }
          
          ICEDiag Result = CheckICE(Init, Ctx);
          // Cache the result of the ICE test.
          Dcl->setInitKnownICE(Ctx, Result.Val == 0);
          return Result;
        }
      }
    }
    return ICEDiag(2, E->getLocStart());
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);
    switch (Exp->getOpcode()) {
    case UnaryOperator::PostInc:
    case UnaryOperator::PostDec:
    case UnaryOperator::PreInc:
    case UnaryOperator::PreDec:
    case UnaryOperator::AddrOf:
    case UnaryOperator::Deref:
      return ICEDiag(2, E->getLocStart());

    case UnaryOperator::Extension:
    case UnaryOperator::LNot:
    case UnaryOperator::Plus:
    case UnaryOperator::Minus:
    case UnaryOperator::Not:
    case UnaryOperator::Real:
    case UnaryOperator::Imag:
      return CheckICE(Exp->getSubExpr(), Ctx);
    case UnaryOperator::OffsetOf:
      // Note that per C99, offsetof must be an ICE. And AFAIK, using
      // Evaluate matches the proposed gcc behavior for cases like
      // "offsetof(struct s{int x[4];}, x[!.0])".  This doesn't affect
      // compliance: we should warn earlier for offsetof expressions with
      // array subscripts that aren't ICEs, and if the array subscripts
      // are ICEs, the value of the offsetof must be an integer constant.
      return CheckEvalInICE(E, Ctx);
    }
  }
  case Expr::SizeOfAlignOfExprClass: {
    const SizeOfAlignOfExpr *Exp = cast<SizeOfAlignOfExpr>(E);
    if (Exp->isSizeOf() && Exp->getTypeOfArgument()->isVariableArrayType())
      return ICEDiag(2, E->getLocStart());
    return NoDiag();
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);
    switch (Exp->getOpcode()) {
    case BinaryOperator::PtrMemD:
    case BinaryOperator::PtrMemI:
    case BinaryOperator::Assign:
    case BinaryOperator::MulAssign:
    case BinaryOperator::DivAssign:
    case BinaryOperator::RemAssign:
    case BinaryOperator::AddAssign:
    case BinaryOperator::SubAssign:
    case BinaryOperator::ShlAssign:
    case BinaryOperator::ShrAssign:
    case BinaryOperator::AndAssign:
    case BinaryOperator::XorAssign:
    case BinaryOperator::OrAssign:
      return ICEDiag(2, E->getLocStart());

    case BinaryOperator::Mul:
    case BinaryOperator::Div:
    case BinaryOperator::Rem:
    case BinaryOperator::Add:
    case BinaryOperator::Sub:
    case BinaryOperator::Shl:
    case BinaryOperator::Shr:
    case BinaryOperator::LT:
    case BinaryOperator::GT:
    case BinaryOperator::LE:
    case BinaryOperator::GE:
    case BinaryOperator::EQ:
    case BinaryOperator::NE:
    case BinaryOperator::And:
    case BinaryOperator::Xor:
    case BinaryOperator::Or:
    case BinaryOperator::Comma: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (Exp->getOpcode() == BinaryOperator::Div ||
          Exp->getOpcode() == BinaryOperator::Rem) {
        // Evaluate gives an error for undefined Div/Rem, so make sure
        // we don't evaluate one.
        if (LHSResult.Val != 2 && RHSResult.Val != 2) {
          llvm::APSInt REval = Exp->getRHS()->EvaluateAsInt(Ctx);
          if (REval == 0)
            return ICEDiag(1, E->getLocStart());
          if (REval.isSigned() && REval.isAllOnesValue()) {
            llvm::APSInt LEval = Exp->getLHS()->EvaluateAsInt(Ctx);
            if (LEval.isMinSignedValue())
              return ICEDiag(1, E->getLocStart());
          }
        }
      }
      if (Exp->getOpcode() == BinaryOperator::Comma) {
        if (Ctx.getLangOptions().C99) {
          // C99 6.6p3 introduces a strange edge case: comma can be in an ICE
          // if it isn't evaluated.
          if (LHSResult.Val == 0 && RHSResult.Val == 0)
            return ICEDiag(1, E->getLocStart());
        } else {
          // In both C89 and C++, commas in ICEs are illegal.
          return ICEDiag(2, E->getLocStart());
        }
      }
      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    case BinaryOperator::LAnd:
    case BinaryOperator::LOr: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (LHSResult.Val == 0 && RHSResult.Val == 1) {
        // Rare case where the RHS has a comma "side-effect"; we need
        // to actually check the condition to see whether the side
        // with the comma is evaluated.
        if ((Exp->getOpcode() == BinaryOperator::LAnd) !=
            (Exp->getLHS()->EvaluateAsInt(Ctx) == 0))
          return RHSResult;
        return NoDiag();
      }

      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    }
  }
  case Expr::CastExprClass:
  case Expr::ImplicitCastExprClass:
  case Expr::ExplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXNamedCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(E)->getSubExpr();
    if (SubExpr->getType()->isIntegralType())
      return CheckICE(SubExpr, Ctx);
    if (isa<FloatingLiteral>(SubExpr->IgnoreParens()))
      return NoDiag();
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(E);
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // then only the true side is actually considered in an integer constant
    // expression, and it is fully evaluated.  This is an important GNU
    // extension.  See GCC PR38377 for discussion.
    if (const CallExpr *CallCE = dyn_cast<CallExpr>(Exp->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall(Ctx) == Builtin::BI__builtin_constant_p) {
        Expr::EvalResult EVResult;
        if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
            !EVResult.Val.isInt()) {
          return ICEDiag(2, E->getLocStart());
        }
        return NoDiag();
      }
    ICEDiag CondResult = CheckICE(Exp->getCond(), Ctx);
    ICEDiag TrueResult = CheckICE(Exp->getTrueExpr(), Ctx);
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);
    if (CondResult.Val == 2)
      return CondResult;
    if (TrueResult.Val == 2)
      return TrueResult;
    if (FalseResult.Val == 2)
      return FalseResult;
    if (CondResult.Val == 1)
      return CondResult;
    if (TrueResult.Val == 0 && FalseResult.Val == 0)
      return NoDiag();
    // Rare case where the diagnostics depend on which side is evaluated
    // Note that if we get here, CondResult is 0, and at least one of
    // TrueResult and FalseResult is non-zero.
    if (Exp->getCond()->EvaluateAsInt(Ctx) == 0) {
      return FalseResult;
    }
    return TrueResult;
  }
  case Expr::CXXDefaultArgExprClass:
    return CheckICE(cast<CXXDefaultArgExpr>(E)->getExpr(), Ctx);
  case Expr::ChooseExprClass: {
    return CheckICE(cast<ChooseExpr>(E)->getChosenSubExpr(Ctx), Ctx);
  }
  }

  // Silence a GCC warning
  return ICEDiag(2, E->getLocStart());
}

bool Expr::isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  ICEDiag d = CheckICE(this, Ctx);
  if (d.Val != 0) {
    if (Loc) *Loc = d.Loc;
    return false;
  }
  EvalResult EvalResult;
  if (!Evaluate(EvalResult, Ctx))
    llvm::llvm_unreachable("ICE cannot be evaluated!");
  assert(!EvalResult.HasSideEffects && "ICE with side effects!");
  assert(EvalResult.Val.isInt() && "ICE that isn't integer!");
  Result = EvalResult.Val.getInt();
  return true;
}

/// isNullPointerConstant - C99 6.3.2.3p3 -  Return true if this is either an
/// integer constant expression with the value zero, or if this is one that is
/// cast to void*.
bool Expr::isNullPointerConstant(ASTContext &Ctx,
                                 NullPointerConstantValueDependence NPC) const {
  if (isValueDependent()) {
    switch (NPC) {
    case NPC_NeverValueDependent:
      assert(false && "Unexpected value dependent expression!");
      // If the unthinkable happens, fall through to the safest alternative.
        
    case NPC_ValueDependentIsNull:
      return isTypeDependent() || getType()->isIntegralType();
        
    case NPC_ValueDependentIsNotNull:
      return false;
    }
  }

  // Strip off a cast to void*, if it exists. Except in C++.
  if (const ExplicitCastExpr *CE = dyn_cast<ExplicitCastExpr>(this)) {
    if (!Ctx.getLangOptions().CPlusPlus) {
      // Check that it is a cast to void*.
      if (const PointerType *PT = CE->getType()->getAs<PointerType>()) {
        QualType Pointee = PT->getPointeeType();
        if (!Pointee.hasQualifiers() &&
            Pointee->isVoidType() &&                              // to void*
            CE->getSubExpr()->getType()->isIntegerType())         // from int.
          return CE->getSubExpr()->isNullPointerConstant(Ctx, NPC);
      }
    }
  } else if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(this)) {
    // Ignore the ImplicitCastExpr type entirely.
    return ICE->getSubExpr()->isNullPointerConstant(Ctx, NPC);
  } else if (const ParenExpr *PE = dyn_cast<ParenExpr>(this)) {
    // Accept ((void*)0) as a null pointer constant, as many other
    // implementations do.
    return PE->getSubExpr()->isNullPointerConstant(Ctx, NPC);
  } else if (const CXXDefaultArgExpr *DefaultArg
               = dyn_cast<CXXDefaultArgExpr>(this)) {
    // See through default argument expressions
    return DefaultArg->getExpr()->isNullPointerConstant(Ctx, NPC);
  } else if (isa<GNUNullExpr>(this)) {
    // The GNU __null extension is always a null pointer constant.
    return true;
  }

  // C++0x nullptr_t is always a null pointer constant.
  if (getType()->isNullPtrType())
    return true;

  // This expression must be an integer type.
  if (!getType()->isIntegerType() || 
      (Ctx.getLangOptions().CPlusPlus && getType()->isEnumeralType()))
    return false;

  // If we have an integer constant expression, we need to *evaluate* it and
  // test for the value 0.
  llvm::APSInt Result;
  return isIntegerConstantExpr(Result, Ctx) && Result == 0;
}

FieldDecl *Expr::getBitField() {
  Expr *E = this->IgnoreParens();

  if (MemberExpr *MemRef = dyn_cast<MemberExpr>(E))
    if (FieldDecl *Field = dyn_cast<FieldDecl>(MemRef->getMemberDecl()))
      if (Field->isBitField())
        return Field;

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(E))
    if (BinOp->isAssignmentOp() && BinOp->getLHS())
      return BinOp->getLHS()->getBitField();

  return 0;
}

/// isArrow - Return true if the base expression is a pointer to vector,
/// return false if the base expression is a vector.
bool ExtVectorElementExpr::isArrow() const {
  return getBase()->getType()->isPointerType();
}

unsigned ExtVectorElementExpr::getNumElements() const {
  if (const VectorType *VT = getType()->getAs<VectorType>())
    return VT->getNumElements();
  return 1;
}

/// containsDuplicateElements - Return true if any element access is repeated.
bool ExtVectorElementExpr::containsDuplicateElements() const {
  // FIXME: Refactor this code to an accessor on the AST node which returns the
  // "type" of component access, and share with code below and in Sema.
  llvm::StringRef Comp = Accessor->getName();

  // Halving swizzles do not contain duplicate elements.
  if (Comp == "hi" || Comp == "lo" || Comp == "even" || Comp == "odd")
    return false;

  // Advance past s-char prefix on hex swizzles.
  if (Comp[0] == 's' || Comp[0] == 'S')
    Comp = Comp.substr(1);

  for (unsigned i = 0, e = Comp.size(); i != e; ++i)
    if (Comp.substr(i + 1).find(Comp[i]) != llvm::StringRef::npos)
        return true;

  return false;
}

/// getEncodedElementAccess - We encode the fields as a llvm ConstantArray.
void ExtVectorElementExpr::getEncodedElementAccess(
                                  llvm::SmallVectorImpl<unsigned> &Elts) const {
  llvm::StringRef Comp = Accessor->getName();
  if (Comp[0] == 's' || Comp[0] == 'S')
    Comp = Comp.substr(1);

  bool isHi =   Comp == "hi";
  bool isLo =   Comp == "lo";
  bool isEven = Comp == "even";
  bool isOdd  = Comp == "odd";

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
      Index = ExtVectorType::getAccessorIdx(Comp[i]);

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

void ObjCMessageExpr::setClassInfo(const ObjCMessageExpr::ClassInfo &CI) {
  if (CI.first == 0 && CI.second == 0)
    SubExprs[RECEIVER] = (Expr*)((uintptr_t)0 | IsInstMeth);
  else if (CI.first == 0)
    SubExprs[RECEIVER] = (Expr*)((uintptr_t)CI.second | IsClsMethDeclUnknown);
  else
    SubExprs[RECEIVER] = (Expr*)((uintptr_t)CI.first | IsClsMethDeclKnown);
}


bool ChooseExpr::isConditionTrue(ASTContext &C) const {
  return getCond()->EvaluateAsInt(C) != 0;
}

void ShuffleVectorExpr::setExprs(ASTContext &C, Expr ** Exprs,
                                 unsigned NumExprs) {
  if (SubExprs) C.Deallocate(SubExprs);

  SubExprs = new (C) Stmt* [NumExprs];
  this->NumExprs = NumExprs;
  memcpy(SubExprs, Exprs, sizeof(Expr *) * NumExprs);
}

void ShuffleVectorExpr::DoDestroy(ASTContext& C) {
  DestroyChildren(C);
  if (SubExprs) C.Deallocate(SubExprs);
  this->~ShuffleVectorExpr();
  C.Deallocate(this);
}

void SizeOfAlignOfExpr::DoDestroy(ASTContext& C) {
  // Override default behavior of traversing children. If this has a type
  // operand and the type is a variable-length array, the child iteration
  // will iterate over the size expression. However, this expression belongs
  // to the type, not to this, so we don't want to delete it.
  // We still want to delete this expression.
  if (isArgumentType()) {
    this->~SizeOfAlignOfExpr();
    C.Deallocate(this);
  }
  else
    Expr::DoDestroy(C);
}

//===----------------------------------------------------------------------===//
//  DesignatedInitExpr
//===----------------------------------------------------------------------===//

IdentifierInfo *DesignatedInitExpr::Designator::getFieldName() {
  assert(Kind == FieldDesignator && "Only valid on a field designator");
  if (Field.NameOrField & 0x01)
    return reinterpret_cast<IdentifierInfo *>(Field.NameOrField&~0x01);
  else
    return getField()->getIdentifier();
}

DesignatedInitExpr::DesignatedInitExpr(QualType Ty, unsigned NumDesignators,
                                       const Designator *Designators,
                                       SourceLocation EqualOrColonLoc,
                                       bool GNUSyntax,
                                       Expr **IndexExprs,
                                       unsigned NumIndexExprs,
                                       Expr *Init)
  : Expr(DesignatedInitExprClass, Ty,
         Init->isTypeDependent(), Init->isValueDependent()),
    EqualOrColonLoc(EqualOrColonLoc), GNUSyntax(GNUSyntax),
    NumDesignators(NumDesignators), NumSubExprs(NumIndexExprs + 1) {
  this->Designators = new Designator[NumDesignators];

  // Record the initializer itself.
  child_iterator Child = child_begin();
  *Child++ = Init;

  // Copy the designators and their subexpressions, computing
  // value-dependence along the way.
  unsigned IndexIdx = 0;
  for (unsigned I = 0; I != NumDesignators; ++I) {
    this->Designators[I] = Designators[I];

    if (this->Designators[I].isArrayDesignator()) {
      // Compute type- and value-dependence.
      Expr *Index = IndexExprs[IndexIdx];
      ValueDependent = ValueDependent ||
        Index->isTypeDependent() || Index->isValueDependent();

      // Copy the index expressions into permanent storage.
      *Child++ = IndexExprs[IndexIdx++];
    } else if (this->Designators[I].isArrayRangeDesignator()) {
      // Compute type- and value-dependence.
      Expr *Start = IndexExprs[IndexIdx];
      Expr *End = IndexExprs[IndexIdx + 1];
      ValueDependent = ValueDependent ||
        Start->isTypeDependent() || Start->isValueDependent() ||
        End->isTypeDependent() || End->isValueDependent();

      // Copy the start/end expressions into permanent storage.
      *Child++ = IndexExprs[IndexIdx++];
      *Child++ = IndexExprs[IndexIdx++];
    }
  }

  assert(IndexIdx == NumIndexExprs && "Wrong number of index expressions");
}

DesignatedInitExpr *
DesignatedInitExpr::Create(ASTContext &C, Designator *Designators,
                           unsigned NumDesignators,
                           Expr **IndexExprs, unsigned NumIndexExprs,
                           SourceLocation ColonOrEqualLoc,
                           bool UsesColonSyntax, Expr *Init) {
  void *Mem = C.Allocate(sizeof(DesignatedInitExpr) +
                         sizeof(Stmt *) * (NumIndexExprs + 1), 8);
  return new (Mem) DesignatedInitExpr(C.VoidTy, NumDesignators, Designators,
                                      ColonOrEqualLoc, UsesColonSyntax,
                                      IndexExprs, NumIndexExprs, Init);
}

DesignatedInitExpr *DesignatedInitExpr::CreateEmpty(ASTContext &C,
                                                    unsigned NumIndexExprs) {
  void *Mem = C.Allocate(sizeof(DesignatedInitExpr) +
                         sizeof(Stmt *) * (NumIndexExprs + 1), 8);
  return new (Mem) DesignatedInitExpr(NumIndexExprs + 1);
}

void DesignatedInitExpr::setDesignators(const Designator *Desigs,
                                        unsigned NumDesigs) {
  if (Designators)
    delete [] Designators;

  Designators = new Designator[NumDesigs];
  NumDesignators = NumDesigs;
  for (unsigned I = 0; I != NumDesigs; ++I)
    Designators[I] = Desigs[I];
}

SourceRange DesignatedInitExpr::getSourceRange() const {
  SourceLocation StartLoc;
  Designator &First =
    *const_cast<DesignatedInitExpr*>(this)->designators_begin();
  if (First.isFieldDesignator()) {
    if (GNUSyntax)
      StartLoc = SourceLocation::getFromRawEncoding(First.Field.FieldLoc);
    else
      StartLoc = SourceLocation::getFromRawEncoding(First.Field.DotLoc);
  } else
    StartLoc =
      SourceLocation::getFromRawEncoding(First.ArrayOrRange.LBracketLoc);
  return SourceRange(StartLoc, getInit()->getSourceRange().getEnd());
}

Expr *DesignatedInitExpr::getArrayIndex(const Designator& D) {
  assert(D.Kind == Designator::ArrayDesignator && "Requires array designator");
  char* Ptr = static_cast<char*>(static_cast<void *>(this));
  Ptr += sizeof(DesignatedInitExpr);
  Stmt **SubExprs = reinterpret_cast<Stmt**>(reinterpret_cast<void**>(Ptr));
  return cast<Expr>(*(SubExprs + D.ArrayOrRange.Index + 1));
}

Expr *DesignatedInitExpr::getArrayRangeStart(const Designator& D) {
  assert(D.Kind == Designator::ArrayRangeDesignator &&
         "Requires array range designator");
  char* Ptr = static_cast<char*>(static_cast<void *>(this));
  Ptr += sizeof(DesignatedInitExpr);
  Stmt **SubExprs = reinterpret_cast<Stmt**>(reinterpret_cast<void**>(Ptr));
  return cast<Expr>(*(SubExprs + D.ArrayOrRange.Index + 1));
}

Expr *DesignatedInitExpr::getArrayRangeEnd(const Designator& D) {
  assert(D.Kind == Designator::ArrayRangeDesignator &&
         "Requires array range designator");
  char* Ptr = static_cast<char*>(static_cast<void *>(this));
  Ptr += sizeof(DesignatedInitExpr);
  Stmt **SubExprs = reinterpret_cast<Stmt**>(reinterpret_cast<void**>(Ptr));
  return cast<Expr>(*(SubExprs + D.ArrayOrRange.Index + 2));
}

/// \brief Replaces the designator at index @p Idx with the series
/// of designators in [First, Last).
void DesignatedInitExpr::ExpandDesignator(unsigned Idx,
                                          const Designator *First,
                                          const Designator *Last) {
  unsigned NumNewDesignators = Last - First;
  if (NumNewDesignators == 0) {
    std::copy_backward(Designators + Idx + 1,
                       Designators + NumDesignators,
                       Designators + Idx);
    --NumNewDesignators;
    return;
  } else if (NumNewDesignators == 1) {
    Designators[Idx] = *First;
    return;
  }

  Designator *NewDesignators
    = new Designator[NumDesignators - 1 + NumNewDesignators];
  std::copy(Designators, Designators + Idx, NewDesignators);
  std::copy(First, Last, NewDesignators + Idx);
  std::copy(Designators + Idx + 1, Designators + NumDesignators,
            NewDesignators + Idx + NumNewDesignators);
  delete [] Designators;
  Designators = NewDesignators;
  NumDesignators = NumDesignators - 1 + NumNewDesignators;
}

void DesignatedInitExpr::DoDestroy(ASTContext &C) {
  delete [] Designators;
  Expr::DoDestroy(C);
}

ParenListExpr::ParenListExpr(ASTContext& C, SourceLocation lparenloc,
                             Expr **exprs, unsigned nexprs,
                             SourceLocation rparenloc)
: Expr(ParenListExprClass, QualType(),
       hasAnyTypeDependentArguments(exprs, nexprs),
       hasAnyValueDependentArguments(exprs, nexprs)),
  NumExprs(nexprs), LParenLoc(lparenloc), RParenLoc(rparenloc) {

  Exprs = new (C) Stmt*[nexprs];
  for (unsigned i = 0; i != nexprs; ++i)
    Exprs[i] = exprs[i];
}

void ParenListExpr::DoDestroy(ASTContext& C) {
  DestroyChildren(C);
  if (Exprs) C.Deallocate(Exprs);
  this->~ParenListExpr();
  C.Deallocate(this);
}

//===----------------------------------------------------------------------===//
//  ExprIterator.
//===----------------------------------------------------------------------===//

Expr* ExprIterator::operator[](size_t idx) { return cast<Expr>(I[idx]); }
Expr* ExprIterator::operator*() const { return cast<Expr>(*I); }
Expr* ExprIterator::operator->() const { return cast<Expr>(*I); }
const Expr* ConstExprIterator::operator[](size_t idx) const {
  return cast<Expr>(I[idx]);
}
const Expr* ConstExprIterator::operator*() const { return cast<Expr>(*I); }
const Expr* ConstExprIterator::operator->() const { return cast<Expr>(*I); }

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

// ObjCImplicitSetterGetterRefExpr
Stmt::child_iterator ObjCImplicitSetterGetterRefExpr::child_begin() {
  return &Base;
}
Stmt::child_iterator ObjCImplicitSetterGetterRefExpr::child_end() {
  return &Base+1;
}

// ObjCSuperExpr
Stmt::child_iterator ObjCSuperExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator ObjCSuperExpr::child_end() { return child_iterator(); }

// ObjCIsaExpr
Stmt::child_iterator ObjCIsaExpr::child_begin() { return &Base; }
Stmt::child_iterator ObjCIsaExpr::child_end() { return &Base+1; }

// PredefinedExpr
Stmt::child_iterator PredefinedExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator PredefinedExpr::child_end() { return child_iterator(); }

// IntegerLiteral
Stmt::child_iterator IntegerLiteral::child_begin() { return child_iterator(); }
Stmt::child_iterator IntegerLiteral::child_end() { return child_iterator(); }

// CharacterLiteral
Stmt::child_iterator CharacterLiteral::child_begin() { return child_iterator();}
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

// SizeOfAlignOfExpr
Stmt::child_iterator SizeOfAlignOfExpr::child_begin() {
  // If this is of a type and the type is a VLA type (and not a typedef), the
  // size expression of the VLA needs to be treated as an executable expression.
  // Why isn't this weirdness documented better in StmtIterator?
  if (isArgumentType()) {
    if (VariableArrayType* T = dyn_cast<VariableArrayType>(
                                   getArgumentType().getTypePtr()))
      return child_iterator(T);
    return child_iterator();
  }
  return child_iterator(&Argument.Ex);
}
Stmt::child_iterator SizeOfAlignOfExpr::child_end() {
  if (isArgumentType())
    return child_iterator();
  return child_iterator(&Argument.Ex + 1);
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

// GNUNullExpr
Stmt::child_iterator GNUNullExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator GNUNullExpr::child_end() { return child_iterator(); }

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

// DesignatedInitExpr
Stmt::child_iterator DesignatedInitExpr::child_begin() {
  char* Ptr = static_cast<char*>(static_cast<void *>(this));
  Ptr += sizeof(DesignatedInitExpr);
  return reinterpret_cast<Stmt**>(reinterpret_cast<void**>(Ptr));
}
Stmt::child_iterator DesignatedInitExpr::child_end() {
  return child_iterator(&*child_begin() + NumSubExprs);
}

// ImplicitValueInitExpr
Stmt::child_iterator ImplicitValueInitExpr::child_begin() {
  return child_iterator();
}

Stmt::child_iterator ImplicitValueInitExpr::child_end() {
  return child_iterator();
}

// ParenListExpr
Stmt::child_iterator ParenListExpr::child_begin() {
  return &Exprs[0];
}
Stmt::child_iterator ParenListExpr::child_end() {
  return &Exprs[0]+NumExprs;
}

// ObjCStringLiteral
Stmt::child_iterator ObjCStringLiteral::child_begin() {
  return &String;
}
Stmt::child_iterator ObjCStringLiteral::child_end() {
  return &String+1;
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
Stmt::child_iterator BlockExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator BlockExpr::child_end() { return child_iterator(); }

Stmt::child_iterator BlockDeclRefExpr::child_begin() { return child_iterator();}
Stmt::child_iterator BlockDeclRefExpr::child_end() { return child_iterator(); }
