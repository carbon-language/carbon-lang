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

void Expr::ANCHOR() {} // key function for Expr class.

/// isKnownToHaveBooleanValue - Return true if this is an integer expression
/// that is known to return 0 or 1.  This happens for _Bool/bool expressions
/// but also int expressions which are produced by things like comparisons in
/// C.
bool Expr::isKnownToHaveBooleanValue() const {
  // If this value has _Bool type, it is obvious 0/1.
  if (getType()->isBooleanType()) return true;
  // If this is a non-scalar-integer type, we don't care enough to try. 
  if (!getType()->isIntegralOrEnumerationType()) return false;
  
  if (const ParenExpr *PE = dyn_cast<ParenExpr>(this))
    return PE->getSubExpr()->isKnownToHaveBooleanValue();
  
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(this)) {
    switch (UO->getOpcode()) {
    case UnaryOperator::Plus:
    case UnaryOperator::Extension:
      return UO->getSubExpr()->isKnownToHaveBooleanValue();
    default:
      return false;
    }
  }
  
  // Only look through implicit casts.  If the user writes
  // '(int) (a && b)' treat it as an arbitrary int.
  if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(this))
    return CE->getSubExpr()->isKnownToHaveBooleanValue();
  
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(this)) {
    switch (BO->getOpcode()) {
    default: return false;
    case BinaryOperator::LT:   // Relational operators.
    case BinaryOperator::GT:
    case BinaryOperator::LE:
    case BinaryOperator::GE:
    case BinaryOperator::EQ:   // Equality operators.
    case BinaryOperator::NE:
    case BinaryOperator::LAnd: // AND operator.
    case BinaryOperator::LOr:  // Logical OR operator.
      return true;
        
    case BinaryOperator::And:  // Bitwise AND operator.
    case BinaryOperator::Xor:  // Bitwise XOR operator.
    case BinaryOperator::Or:   // Bitwise OR operator.
      // Handle things like (x==2)|(y==12).
      return BO->getLHS()->isKnownToHaveBooleanValue() &&
             BO->getRHS()->isKnownToHaveBooleanValue();
        
    case BinaryOperator::Comma:
    case BinaryOperator::Assign:
      return BO->getRHS()->isKnownToHaveBooleanValue();
    }
  }
  
  if (const ConditionalOperator *CO = dyn_cast<ConditionalOperator>(this))
    return CO->getTrueExpr()->isKnownToHaveBooleanValue() &&
           CO->getFalseExpr()->isKnownToHaveBooleanValue();
  
  return false;
}

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

std::size_t ExplicitTemplateArgumentList::sizeFor(unsigned NumTemplateArgs) {
  return sizeof(ExplicitTemplateArgumentList) +
         sizeof(TemplateArgumentLoc) * NumTemplateArgs;
}

std::size_t ExplicitTemplateArgumentList::sizeFor(
                                      const TemplateArgumentListInfo &Info) {
  return sizeFor(Info.size());
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
    if (Var->getType()->isIntegralOrEnumerationType() &&
        Var->getType().getCVRQualifiers() == Qualifiers::Const) {
      if (const Expr *Init = Var->getAnyInitializer())
        if (Init->isValueDependent())
          ValueDependent = true;
    } 
    // (VD) - FIXME: Missing from the standard: 
    //      -  a member function or a static data member of the current 
    //         instantiation
    else if (Var->isStaticDataMember() && 
             Var->getDeclContext()->isDependentContext())
      ValueDependent = true;
  } 
  // (VD) - FIXME: Missing from the standard: 
  //      -  a member function or a static data member of the current 
  //         instantiation
  else if (isa<CXXMethodDecl>(D) && D->getDeclContext()->isDependentContext())
    ValueDependent = true;
  //  (TD)  - a nested-name-specifier or a qualified-id that names a
  //          member of an unknown specialization.
  //        (handled by DependentScopeDeclRefExpr)
}

DeclRefExpr::DeclRefExpr(NestedNameSpecifier *Qualifier, 
                         SourceRange QualifierRange,
                         ValueDecl *D, SourceLocation NameLoc,
                         const TemplateArgumentListInfo *TemplateArgs,
                         QualType T)
  : Expr(DeclRefExprClass, T, false, false),
    DecoratedD(D,
               (Qualifier? HasQualifierFlag : 0) |
               (TemplateArgs ? HasExplicitTemplateArgumentListFlag : 0)),
    Loc(NameLoc) {
  if (Qualifier) {
    NameQualifier *NQ = getNameQualifier();
    NQ->NNS = Qualifier;
    NQ->Range = QualifierRange;
  }
      
  if (TemplateArgs)
    getExplicitTemplateArgumentList()->initializeFrom(*TemplateArgs);

  computeDependence();
}

DeclRefExpr::DeclRefExpr(NestedNameSpecifier *Qualifier,
                         SourceRange QualifierRange,
                         ValueDecl *D, const DeclarationNameInfo &NameInfo,
                         const TemplateArgumentListInfo *TemplateArgs,
                         QualType T)
  : Expr(DeclRefExprClass, T, false, false),
    DecoratedD(D,
               (Qualifier? HasQualifierFlag : 0) |
               (TemplateArgs ? HasExplicitTemplateArgumentListFlag : 0)),
    Loc(NameInfo.getLoc()), DNLoc(NameInfo.getInfo()) {
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
                                 ValueDecl *D,
                                 SourceLocation NameLoc,
                                 QualType T,
                                 const TemplateArgumentListInfo *TemplateArgs) {
  return Create(Context, Qualifier, QualifierRange, D,
                DeclarationNameInfo(D->getDeclName(), NameLoc),
                T, TemplateArgs);
}

DeclRefExpr *DeclRefExpr::Create(ASTContext &Context,
                                 NestedNameSpecifier *Qualifier,
                                 SourceRange QualifierRange,
                                 ValueDecl *D,
                                 const DeclarationNameInfo &NameInfo,
                                 QualType T,
                                 const TemplateArgumentListInfo *TemplateArgs) {
  std::size_t Size = sizeof(DeclRefExpr);
  if (Qualifier != 0)
    Size += sizeof(NameQualifier);
  
  if (TemplateArgs)
    Size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);
  
  void *Mem = Context.Allocate(Size, llvm::alignof<DeclRefExpr>());
  return new (Mem) DeclRefExpr(Qualifier, QualifierRange, D, NameInfo,
                               TemplateArgs, T);
}

DeclRefExpr *DeclRefExpr::CreateEmpty(ASTContext &Context, bool HasQualifier,
                                      unsigned NumTemplateArgs) {
  std::size_t Size = sizeof(DeclRefExpr);
  if (HasQualifier)
    Size += sizeof(NameQualifier);
  
  if (NumTemplateArgs)
    Size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  
  void *Mem = Context.Allocate(Size, llvm::alignof<DeclRefExpr>());
  return new (Mem) DeclRefExpr(EmptyShell());
}

SourceRange DeclRefExpr::getSourceRange() const {
  SourceRange R = getNameInfo().getSourceRange();
  if (hasQualifier())
    R.setBegin(getQualifierRange().getBegin());
  if (hasExplicitTemplateArgumentList())
    R.setEnd(getRAngleLoc());
  return R;
}

// FIXME: Maybe this should use DeclPrinter with a special "print predefined
// expr" policy instead.
std::string PredefinedExpr::ComputeName(IdentType IT, const Decl *CurrentDecl) {
  ASTContext &Context = CurrentDecl->getASTContext();

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CurrentDecl)) {
    if (IT != PrettyFunction && IT != PrettyFunctionNoVirtual)
      return FD->getNameAsString();

    llvm::SmallString<256> Name;
    llvm::raw_svector_ostream Out(Name);

    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MD->isVirtual() && IT != PrettyFunctionNoVirtual)
        Out << "virtual ";
      if (MD->isStatic())
        Out << "static ";
    }

    PrintingPolicy Policy(Context.getLangOptions());

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

    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      Qualifiers ThisQuals = Qualifiers::fromCVRMask(MD->getTypeQualifiers());
      if (ThisQuals.hasConst())
        Proto += " const";
      if (ThisQuals.hasVolatile())
        Proto += " volatile";
    }

    if (!isa<CXXConstructorDecl>(FD) && !isa<CXXDestructorDecl>(FD))
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

    // For incorrect code, there might not be an ObjCInterfaceDecl.  Do
    // a null check to avoid a crash.
    if (const ObjCInterfaceDecl *ID = MD->getClassInterface())
      Out << ID;

    if (const ObjCCategoryImplDecl *CID =
        dyn_cast<ObjCCategoryImplDecl>(MD->getDeclContext()))
      Out << '(' << CID << ')';

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

void StringLiteral::setString(ASTContext &C, llvm::StringRef Str) {
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

Decl *CallExpr::getCalleeDecl() {
  Expr *CEE = getCallee()->IgnoreParenCasts();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE))
    return DRE->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(CEE))
    return ME->getMemberDecl();

  return 0;
}

FunctionDecl *CallExpr::getDirectCallee() {
  return dyn_cast_or_null<FunctionDecl>(getCalleeDecl());
}

/// setNumArgs - This changes the number of arguments present in this call.
/// Any orphaned expressions are deleted by this, and any new operands are set
/// to null.
void CallExpr::setNumArgs(ASTContext& C, unsigned NumArgs) {
  // No change, just return.
  if (NumArgs == getNumArgs()) return;

  // If shrinking # arguments, just delete the extras and forgot them.
  if (NumArgs < getNumArgs()) {
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
  else if (const MemberPointerType *MPT
                                      = CalleeType->getAs<MemberPointerType>())
    CalleeType = MPT->getPointeeType();
    
  const FunctionType *FnType = CalleeType->getAs<FunctionType>();
  return FnType->getResultType();
}

OffsetOfExpr *OffsetOfExpr::Create(ASTContext &C, QualType type, 
                                   SourceLocation OperatorLoc,
                                   TypeSourceInfo *tsi, 
                                   OffsetOfNode* compsPtr, unsigned numComps, 
                                   Expr** exprsPtr, unsigned numExprs,
                                   SourceLocation RParenLoc) {
  void *Mem = C.Allocate(sizeof(OffsetOfExpr) +
                         sizeof(OffsetOfNode) * numComps + 
                         sizeof(Expr*) * numExprs);

  return new (Mem) OffsetOfExpr(C, type, OperatorLoc, tsi, compsPtr, numComps,
                                exprsPtr, numExprs, RParenLoc);
}

OffsetOfExpr *OffsetOfExpr::CreateEmpty(ASTContext &C,
                                        unsigned numComps, unsigned numExprs) {
  void *Mem = C.Allocate(sizeof(OffsetOfExpr) +
                         sizeof(OffsetOfNode) * numComps +
                         sizeof(Expr*) * numExprs);
  return new (Mem) OffsetOfExpr(numComps, numExprs);
}

OffsetOfExpr::OffsetOfExpr(ASTContext &C, QualType type, 
                           SourceLocation OperatorLoc, TypeSourceInfo *tsi,
                           OffsetOfNode* compsPtr, unsigned numComps, 
                           Expr** exprsPtr, unsigned numExprs,
                           SourceLocation RParenLoc)
  : Expr(OffsetOfExprClass, type, /*TypeDependent=*/false, 
         /*ValueDependent=*/tsi->getType()->isDependentType() ||
         hasAnyTypeDependentArguments(exprsPtr, numExprs) ||
         hasAnyValueDependentArguments(exprsPtr, numExprs)),
    OperatorLoc(OperatorLoc), RParenLoc(RParenLoc), TSInfo(tsi), 
    NumComps(numComps), NumExprs(numExprs) 
{
  for(unsigned i = 0; i < numComps; ++i) {
    setComponent(i, compsPtr[i]);
  }
  
  for(unsigned i = 0; i < numExprs; ++i) {
    setIndexExpr(i, exprsPtr[i]);
  }
}

IdentifierInfo *OffsetOfExpr::OffsetOfNode::getFieldName() const {
  assert(getKind() == Field || getKind() == Identifier);
  if (getKind() == Field)
    return getField()->getIdentifier();
  
  return reinterpret_cast<IdentifierInfo *> (Data & ~(uintptr_t)Mask);
}

MemberExpr *MemberExpr::Create(ASTContext &C, Expr *base, bool isarrow,
                               NestedNameSpecifier *qual,
                               SourceRange qualrange,
                               ValueDecl *memberdecl,
                               DeclAccessPair founddecl,
                               DeclarationNameInfo nameinfo,
                               const TemplateArgumentListInfo *targs,
                               QualType ty) {
  std::size_t Size = sizeof(MemberExpr);

  bool hasQualOrFound = (qual != 0 ||
                         founddecl.getDecl() != memberdecl ||
                         founddecl.getAccess() != memberdecl->getAccess());
  if (hasQualOrFound)
    Size += sizeof(MemberNameQualifier);

  if (targs)
    Size += ExplicitTemplateArgumentList::sizeFor(*targs);

  void *Mem = C.Allocate(Size, llvm::alignof<MemberExpr>());
  MemberExpr *E = new (Mem) MemberExpr(base, isarrow, memberdecl, nameinfo, ty);

  if (hasQualOrFound) {
    if (qual && qual->isDependent()) {
      E->setValueDependent(true);
      E->setTypeDependent(true);
    }
    E->HasQualifierOrFoundDecl = true;

    MemberNameQualifier *NQ = E->getMemberQualifier();
    NQ->NNS = qual;
    NQ->Range = qualrange;
    NQ->FoundDecl = founddecl;
  }

  if (targs) {
    E->HasExplicitTemplateArgumentList = true;
    E->getExplicitTemplateArgumentList()->initializeFrom(*targs);
  }

  return E;
}

const char *CastExpr::getCastKindName() const {
  switch (getCastKind()) {
  case CastExpr::CK_Unknown:
    return "Unknown";
  case CastExpr::CK_BitCast:
    return "BitCast";
  case CastExpr::CK_LValueBitCast:
    return "LValueBitCast";
  case CastExpr::CK_NoOp:
    return "NoOp";
  case CastExpr::CK_BaseToDerived:
    return "BaseToDerived";
  case CastExpr::CK_DerivedToBase:
    return "DerivedToBase";
  case CastExpr::CK_UncheckedDerivedToBase:
    return "UncheckedDerivedToBase";
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
  case CastExpr::CK_AnyPointerToObjCPointerCast:
    return "AnyPointerToObjCPointerCast";
  case CastExpr::CK_AnyPointerToBlockPointerCast:
    return "AnyPointerToBlockPointerCast";
  case CastExpr::CK_ObjCObjectLValueCast:
    return "ObjCObjectLValueCast";
  }

  assert(0 && "Unhandled cast kind!");
  return 0;
}

Expr *CastExpr::getSubExprAsWritten() {
  Expr *SubExpr = 0;
  CastExpr *E = this;
  do {
    SubExpr = E->getSubExpr();
    
    // Skip any temporary bindings; they're implicit.
    if (CXXBindTemporaryExpr *Binder = dyn_cast<CXXBindTemporaryExpr>(SubExpr))
      SubExpr = Binder->getSubExpr();
    
    // Conversions by constructor and conversion functions have a
    // subexpression describing the call; strip it off.
    if (E->getCastKind() == CastExpr::CK_ConstructorConversion)
      SubExpr = cast<CXXConstructExpr>(SubExpr)->getArg(0);
    else if (E->getCastKind() == CastExpr::CK_UserDefinedConversion)
      SubExpr = cast<CXXMemberCallExpr>(SubExpr)->getImplicitObjectArgument();
    
    // If the subexpression we're left with is an implicit cast, look
    // through that, too.
  } while ((E = dyn_cast<ImplicitCastExpr>(SubExpr)));  
  
  return SubExpr;
}

CXXBaseSpecifier **CastExpr::path_buffer() {
  switch (getStmtClass()) {
#define ABSTRACT_STMT(x)
#define CASTEXPR(Type, Base) \
  case Stmt::Type##Class: \
    return reinterpret_cast<CXXBaseSpecifier**>(static_cast<Type*>(this)+1);
#define STMT(Type, Base)
#include "clang/AST/StmtNodes.inc"
  default:
    llvm_unreachable("non-cast expressions not possible here");
    return 0;
  }
}

void CastExpr::setCastPath(const CXXCastPath &Path) {
  assert(Path.size() == path_size());
  memcpy(path_buffer(), Path.data(), Path.size() * sizeof(CXXBaseSpecifier*));
}

ImplicitCastExpr *ImplicitCastExpr::Create(ASTContext &C, QualType T,
                                           CastKind Kind, Expr *Operand,
                                           const CXXCastPath *BasePath,
                                           ResultCategory Cat) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(ImplicitCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  ImplicitCastExpr *E =
    new (Buffer) ImplicitCastExpr(T, Kind, Operand, PathSize, Cat);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

ImplicitCastExpr *ImplicitCastExpr::CreateEmpty(ASTContext &C,
                                                unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(ImplicitCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) ImplicitCastExpr(EmptyShell(), PathSize);
}


CStyleCastExpr *CStyleCastExpr::Create(ASTContext &C, QualType T,
                                       CastKind K, Expr *Op,
                                       const CXXCastPath *BasePath,
                                       TypeSourceInfo *WrittenTy,
                                       SourceLocation L, SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(CStyleCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  CStyleCastExpr *E =
    new (Buffer) CStyleCastExpr(T, K, Op, PathSize, WrittenTy, L, R);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CStyleCastExpr *CStyleCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(CStyleCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CStyleCastExpr(EmptyShell(), PathSize);
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

InitListExpr::InitListExpr(ASTContext &C, SourceLocation lbraceloc,
                           Expr **initExprs, unsigned numInits,
                           SourceLocation rbraceloc)
  : Expr(InitListExprClass, QualType(), false, false),
    InitExprs(C, numInits),
    LBraceLoc(lbraceloc), RBraceLoc(rbraceloc), SyntacticForm(0),
    UnionFieldInit(0), HadArrayRangeDesignator(false) 
{      
  for (unsigned I = 0; I != numInits; ++I) {
    if (initExprs[I]->isTypeDependent())
      TypeDependent = true;
    if (initExprs[I]->isValueDependent())
      ValueDependent = true;
  }
      
  InitExprs.insert(C, InitExprs.end(), initExprs, initExprs+numInits);
}

void InitListExpr::reserveInits(ASTContext &C, unsigned NumInits) {
  if (NumInits > InitExprs.size())
    InitExprs.reserve(C, NumInits);
}

void InitListExpr::resizeInits(ASTContext &C, unsigned NumInits) {
  InitExprs.resize(C, NumInits, 0);
}

Expr *InitListExpr::updateInit(ASTContext &C, unsigned Init, Expr *expr) {
  if (Init >= InitExprs.size()) {
    InitExprs.insert(C, InitExprs.end(), Init - InitExprs.size() + 1, 0);
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
    if (getType()->isVoidType())
      return false;
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
    switch (BO->getOpcode()) {
      default:
        break;
      // Consider the RHS of comma for side effects. LHS was checked by
      // Sema::CheckCommaOperands.
      case BinaryOperator::Comma:
        // ((foo = <blah>), 0) is an idiom for hiding the result (and
        // lvalue-ness) of an assignment written in a macro.
        if (IntegerLiteral *IE =
              dyn_cast<IntegerLiteral>(BO->getRHS()->IgnoreParens()))
          if (IE->getValue() == 0)
            return false;
        return BO->getRHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx);
      // Consider '||', '&&' to have side effects if the LHS or RHS does.
      case BinaryOperator::LAnd:
      case BinaryOperator::LOr:
        if (!BO->getLHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx) ||
            !BO->getRHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx))
          return false;
        break;
    }
    if (BO->isAssignmentOp())
      return false;
    Loc = BO->getOperatorLoc();
    R1 = BO->getLHS()->getSourceRange();
    R2 = BO->getRHS()->getSourceRange();
    return true;
  }
  case CompoundAssignOperatorClass:
  case VAArgExprClass:
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
    if (const Decl *FD = CE->getCalleeDecl()) {
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

  case ObjCMessageExprClass: {
    const ObjCMessageExpr *ME = cast<ObjCMessageExpr>(this);
    const ObjCMethodDecl *MD = ME->getMethodDecl();
    if (MD && MD->getAttr<WarnUnusedResultAttr>()) {
      Loc = getExprLoc();
      return true;
    }
    return false;
  }

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

    if (getType()->isVoidType())
      return false;
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
    if (getType()->isVoidType())
      return false;
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

Expr *Expr::IgnoreParenImpCasts() {
  Expr *E = this;
  while (true) {
    if (ParenExpr *P = dyn_cast<ParenExpr>(E))
      E = P->getSubExpr();
    else if (ImplicitCastExpr *P = dyn_cast<ImplicitCastExpr>(E))
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
      // ptr<->int casts of the same width.  We also ignore all identity casts.
      Expr *SE = P->getSubExpr();

      if (Ctx.hasSameUnqualifiedType(E->getType(), SE->getType())) {
        E = SE;
        continue;
      }

      if ((E->getType()->isPointerType() || 
           E->getType()->isIntegralType(Ctx)) &&
          (SE->getType()->isPointerType() || 
           SE->getType()->isIntegralType(Ctx)) &&
          Ctx.getTypeSize(E->getType()) == Ctx.getTypeSize(SE->getType())) {
        E = SE;
        continue;
      }
    }

    return E;
  }
}

bool Expr::isDefaultArgument() const {
  const Expr *E = this;
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  
  return isa<CXXDefaultArgExpr>(E);
}

/// \brief Skip over any no-op casts and any temporary-binding
/// expressions.
static const Expr *skipTemporaryBindingsAndNoOpCasts(const Expr *E) {
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CastExpr::CK_NoOp)
      E = ICE->getSubExpr();
    else
      break;
  }

  while (const CXXBindTemporaryExpr *BE = dyn_cast<CXXBindTemporaryExpr>(E))
    E = BE->getSubExpr();

  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CastExpr::CK_NoOp)
      E = ICE->getSubExpr();
    else
      break;
  }
  
  return E;
}

const Expr *Expr::getTemporaryObject() const {
  const Expr *E = skipTemporaryBindingsAndNoOpCasts(this);

  // A cast can produce a temporary object. The object's construction
  // is represented as a CXXConstructExpr.
  if (const CastExpr *Cast = dyn_cast<CastExpr>(E)) {
    // Only user-defined and constructor conversions can produce
    // temporary objects.
    if (Cast->getCastKind() != CastExpr::CK_ConstructorConversion &&
        Cast->getCastKind() != CastExpr::CK_UserDefinedConversion)
      return 0;

    // Strip off temporary bindings and no-op casts.
    const Expr *Sub = skipTemporaryBindingsAndNoOpCasts(Cast->getSubExpr());

    // If this is a constructor conversion, see if we have an object
    // construction.
    if (Cast->getCastKind() == CastExpr::CK_ConstructorConversion)
      return dyn_cast<CXXConstructExpr>(Sub);

    // If this is a user-defined conversion, see if we have a call to
    // a function that itself returns a temporary object.
    if (Cast->getCastKind() == CastExpr::CK_UserDefinedConversion)
      if (const CallExpr *CE = dyn_cast<CallExpr>(Sub))
        if (CE->getCallReturnType()->isRecordType())
          return CE;

    return 0;
  }

  // A call returning a class type returns a temporary.
  if (const CallExpr *CE = dyn_cast<CallExpr>(E)) {
    if (CE->getCallReturnType()->isRecordType())
      return CE;

    return 0;
  }

  // Explicit temporary object constructors create temporaries.
  return dyn_cast<CXXTemporaryObjectExpr>(E);
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

bool Expr::isConstantInitializer(ASTContext &Ctx, bool IsForRef) const {
  // This function is attempting whether an expression is an initializer
  // which can be evaluated at compile-time.  isEvaluatable handles most
  // of the cases, but it can't deal with some initializer-specific
  // expressions, and it can't deal with aggregates; we deal with those here,
  // and fall back to isEvaluatable for the other cases.

  // If we ever capture reference-binding directly in the AST, we can
  // kill the second parameter.

  if (IsForRef) {
    EvalResult Result;
    return EvaluateAsLValue(Result, Ctx) && !Result.HasSideEffects;
  }

  switch (getStmtClass()) {
  default: break;
  case StringLiteralClass:
  case ObjCStringLiteralClass:
  case ObjCEncodeExprClass:
    return true;
  case CXXTemporaryObjectExprClass:
  case CXXConstructExprClass: {
    const CXXConstructExpr *CE = cast<CXXConstructExpr>(this);

    // Only if it's
    // 1) an application of the trivial default constructor or
    if (!CE->getConstructor()->isTrivial()) return false;
    if (!CE->getNumArgs()) return true;

    // 2) an elidable trivial copy construction of an operand which is
    //    itself a constant initializer.  Note that we consider the
    //    operand on its own, *not* as a reference binding.
    return CE->isElidable() &&
           CE->getArg(0)->isConstantInitializer(Ctx, false);
  }
  case CompoundLiteralExprClass: {
    // This handles gcc's extension that allows global initializers like
    // "struct x {int x;} x = (struct x) {};".
    // FIXME: This accepts other cases it shouldn't!
    const Expr *Exp = cast<CompoundLiteralExpr>(this)->getInitializer();
    return Exp->isConstantInitializer(Ctx, false);
  }
  case InitListExprClass: {
    // FIXME: This doesn't deal with fields with reference types correctly.
    // FIXME: This incorrectly allows pointers cast to integers to be assigned
    // to bitfields.
    const InitListExpr *Exp = cast<InitListExpr>(this);
    unsigned numInits = Exp->getNumInits();
    for (unsigned i = 0; i < numInits; i++) {
      if (!Exp->getInit(i)->isConstantInitializer(Ctx, false))
        return false;
    }
    return true;
  }
  case ImplicitValueInitExprClass:
    return true;
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()
      ->isConstantInitializer(Ctx, IsForRef);
  case UnaryOperatorClass: {
    const UnaryOperator* Exp = cast<UnaryOperator>(this);
    if (Exp->getOpcode() == UnaryOperator::Extension)
      return Exp->getSubExpr()->isConstantInitializer(Ctx, false);
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
  case CXXFunctionalCastExprClass:
  case CXXStaticCastExprClass:
  case ImplicitCastExprClass:
  case CStyleCastExprClass:
    // Handle casts with a destination that's a struct or union; this
    // deals with both the gcc no-op struct cast extension and the
    // cast-to-union extension.
    if (getType()->isRecordType())
      return cast<CastExpr>(this)->getSubExpr()
        ->isConstantInitializer(Ctx, false);
      
    // Integer->integer casts can be handled here, which is important for
    // things like (int)(&&x-&&y).  Scary but true.
    if (getType()->isIntegerType() &&
        cast<CastExpr>(this)->getSubExpr()->getType()->isIntegerType())
      return cast<CastExpr>(this)->getSubExpr()
        ->isConstantInitializer(Ctx, false);
      
    break;
  }
  return isEvaluatable(Ctx);
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
      return isTypeDependent() || getType()->isIntegralType(Ctx);
        
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

  while (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCategory() != ImplicitCastExpr::RValue &&
        ICE->getCastKind() == CastExpr::CK_NoOp)
      E = ICE->getSubExpr()->IgnoreParens();
    else
      break;
  }

  if (MemberExpr *MemRef = dyn_cast<MemberExpr>(E))
    if (FieldDecl *Field = dyn_cast<FieldDecl>(MemRef->getMemberDecl()))
      if (Field->isBitField())
        return Field;

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(E))
    if (BinOp->isAssignmentOp() && BinOp->getLHS())
      return BinOp->getLHS()->getBitField();

  return 0;
}

bool Expr::refersToVectorElement() const {
  const Expr *E = this->IgnoreParens();
  
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCategory() != ImplicitCastExpr::RValue &&
        ICE->getCastKind() == CastExpr::CK_NoOp)
      E = ICE->getSubExpr()->IgnoreParens();
    else
      break;
  }
  
  if (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E))
    return ASE->getBase()->getType()->isVectorType();

  if (isa<ExtVectorElementExpr>(E))
    return true;

  return false;
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

ObjCMessageExpr::ObjCMessageExpr(QualType T,
                                 SourceLocation LBracLoc,
                                 SourceLocation SuperLoc,
                                 bool IsInstanceSuper,
                                 QualType SuperType,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, /*TypeDependent=*/false,
         /*ValueDependent=*/false),
    NumArgs(NumArgs), Kind(IsInstanceSuper? SuperInstance : SuperClass),
    HasMethod(Method != 0), SuperLoc(SuperLoc),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(SuperType.getAsOpaquePtr());
  if (NumArgs)
    memcpy(getArgs(), Args, NumArgs * sizeof(Expr *));
}

ObjCMessageExpr::ObjCMessageExpr(QualType T,
                                 SourceLocation LBracLoc,
                                 TypeSourceInfo *Receiver,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, T->isDependentType(),
         (T->isDependentType() || 
          hasAnyValueDependentArguments(Args, NumArgs))),
    NumArgs(NumArgs), Kind(Class), HasMethod(Method != 0),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(Receiver);
  if (NumArgs)
    memcpy(getArgs(), Args, NumArgs * sizeof(Expr *));
}

ObjCMessageExpr::ObjCMessageExpr(QualType T,
                                 SourceLocation LBracLoc,
                                 Expr *Receiver,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, Receiver->isTypeDependent(),
         (Receiver->isTypeDependent() || 
          hasAnyValueDependentArguments(Args, NumArgs))),
    NumArgs(NumArgs), Kind(Instance), HasMethod(Method != 0),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(Receiver);
  if (NumArgs)
    memcpy(getArgs(), Args, NumArgs * sizeof(Expr *));
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         SourceLocation LBracLoc,
                                         SourceLocation SuperLoc,
                                         bool IsInstanceSuper,
                                         QualType SuperType,
                                         Selector Sel, 
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, LBracLoc, SuperLoc, IsInstanceSuper,
                                   SuperType, Sel, Method, Args, NumArgs, 
                                   RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         SourceLocation LBracLoc,
                                         TypeSourceInfo *Receiver,
                                         Selector Sel, 
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, LBracLoc, Receiver, Sel, Method, Args, 
                                   NumArgs, RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         SourceLocation LBracLoc,
                                         Expr *Receiver,
                                         Selector Sel, 
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, LBracLoc, Receiver, Sel, Method, Args, 
                                   NumArgs, RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::CreateEmpty(ASTContext &Context, 
                                              unsigned NumArgs) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(EmptyShell(), NumArgs);
}
         
Selector ObjCMessageExpr::getSelector() const {
  if (HasMethod)
    return reinterpret_cast<const ObjCMethodDecl *>(SelectorOrMethod)
                                                               ->getSelector();
  return Selector(SelectorOrMethod); 
}

ObjCInterfaceDecl *ObjCMessageExpr::getReceiverInterface() const {
  switch (getReceiverKind()) {
  case Instance:
    if (const ObjCObjectPointerType *Ptr
          = getInstanceReceiver()->getType()->getAs<ObjCObjectPointerType>())
      return Ptr->getInterfaceDecl();
    break;

  case Class:
    if (const ObjCObjectType *Ty
          = getClassReceiver()->getAs<ObjCObjectType>())
      return Ty->getInterface();
    break;

  case SuperInstance:
    if (const ObjCObjectPointerType *Ptr
          = getSuperType()->getAs<ObjCObjectPointerType>())
      return Ptr->getInterfaceDecl();
    break;

  case SuperClass:
    if (const ObjCObjectPointerType *Iface
                       = getSuperType()->getAs<ObjCObjectPointerType>())
      return Iface->getInterfaceDecl();
    break;
  }

  return 0;
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

DesignatedInitExpr::DesignatedInitExpr(ASTContext &C, QualType Ty, 
                                       unsigned NumDesignators,
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
  this->Designators = new (C) Designator[NumDesignators];

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
  return new (Mem) DesignatedInitExpr(C, C.VoidTy, NumDesignators, Designators,
                                      ColonOrEqualLoc, UsesColonSyntax,
                                      IndexExprs, NumIndexExprs, Init);
}

DesignatedInitExpr *DesignatedInitExpr::CreateEmpty(ASTContext &C,
                                                    unsigned NumIndexExprs) {
  void *Mem = C.Allocate(sizeof(DesignatedInitExpr) +
                         sizeof(Stmt *) * (NumIndexExprs + 1), 8);
  return new (Mem) DesignatedInitExpr(NumIndexExprs + 1);
}

void DesignatedInitExpr::setDesignators(ASTContext &C,
                                        const Designator *Desigs,
                                        unsigned NumDesigs) {
  Designators = new (C) Designator[NumDesigs];
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
void DesignatedInitExpr::ExpandDesignator(ASTContext &C, unsigned Idx,
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
    = new (C) Designator[NumDesignators - 1 + NumNewDesignators];
  std::copy(Designators, Designators + Idx, NewDesignators);
  std::copy(First, Last, NewDesignators + Idx);
  std::copy(Designators + Idx + 1, Designators + NumDesignators,
            NewDesignators + Idx + NumNewDesignators);
  Designators = NewDesignators;
  NumDesignators = NumDesignators - 1 + NumNewDesignators;
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
  // If this is accessing a class member, skip that entry.
  if (Base) return &Base;
  return &Base+1;
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

// OffsetOfExpr
Stmt::child_iterator OffsetOfExpr::child_begin() {
  return reinterpret_cast<Stmt **> (reinterpret_cast<OffsetOfNode *> (this + 1)
                                      + NumComps);
}
Stmt::child_iterator OffsetOfExpr::child_end() {
  return child_iterator(&*child_begin() + NumExprs);
}

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
  if (getReceiverKind() == Instance)
    return reinterpret_cast<Stmt **>(this + 1);
  return getArgs();
}
Stmt::child_iterator ObjCMessageExpr::child_end() {
  return getArgs() + getNumArgs();
}

// Blocks
Stmt::child_iterator BlockExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator BlockExpr::child_end() { return child_iterator(); }

Stmt::child_iterator BlockDeclRefExpr::child_begin() { return child_iterator();}
Stmt::child_iterator BlockDeclRefExpr::child_end() { return child_iterator(); }
