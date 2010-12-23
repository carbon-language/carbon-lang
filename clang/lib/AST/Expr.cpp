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
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
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
    case UO_Plus:
    case UO_Extension:
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
    case BO_LT:   // Relational operators.
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:   // Equality operators.
    case BO_NE:
    case BO_LAnd: // AND operator.
    case BO_LOr:  // Logical OR operator.
      return true;
        
    case BO_And:  // Bitwise AND operator.
    case BO_Xor:  // Bitwise XOR operator.
    case BO_Or:   // Bitwise OR operator.
      // Handle things like (x==2)|(y==12).
      return BO->getLHS()->isKnownToHaveBooleanValue() &&
             BO->getRHS()->isKnownToHaveBooleanValue();
        
    case BO_Comma:
    case BO_Assign:
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

void ExplicitTemplateArgumentList::initializeFrom(
                                   const TemplateArgumentListInfo &Info,
                                   bool &Dependent, 
                                   bool &ContainsUnexpandedParameterPack) {
  LAngleLoc = Info.getLAngleLoc();
  RAngleLoc = Info.getRAngleLoc();
  NumTemplateArgs = Info.size();

  TemplateArgumentLoc *ArgBuffer = getTemplateArgs();
  for (unsigned i = 0; i != NumTemplateArgs; ++i) {
    Dependent = Dependent || Info[i].getArgument().isDependent();
    ContainsUnexpandedParameterPack 
      = ContainsUnexpandedParameterPack || 
        Info[i].getArgument().containsUnexpandedParameterPack();

    new (&ArgBuffer[i]) TemplateArgumentLoc(Info[i]);
  }
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
  ExprBits.TypeDependent = false;
  ExprBits.ValueDependent = false;
  
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
    ExprBits.TypeDependent = true;
    ExprBits.ValueDependent = true;
  }
  //  (TD)  - a conversion-function-id that specifies a dependent type
  else if (D->getDeclName().getNameKind() 
                               == DeclarationName::CXXConversionFunctionName &&
           D->getDeclName().getCXXNameType()->isDependentType()) {
    ExprBits.TypeDependent = true;
    ExprBits.ValueDependent = true;
  }
  //  (TD)  - a template-id that is dependent,
  else if (hasExplicitTemplateArgs() && 
           TemplateSpecializationType::anyDependentTemplateArguments(
                                                       getTemplateArgs(), 
                                                       getNumTemplateArgs())) {
    ExprBits.TypeDependent = true;
    ExprBits.ValueDependent = true;
  }
  //  (VD)  - the name of a non-type template parameter,
  else if (isa<NonTypeTemplateParmDecl>(D))
    ExprBits.ValueDependent = true;
  //  (VD) - a constant with integral or enumeration type and is
  //         initialized with an expression that is value-dependent.
  else if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    if (Var->getType()->isIntegralOrEnumerationType() &&
        Var->getType().getCVRQualifiers() == Qualifiers::Const) {
      if (const Expr *Init = Var->getAnyInitializer())
        if (Init->isValueDependent())
          ExprBits.ValueDependent = true;
    } 
    // (VD) - FIXME: Missing from the standard: 
    //      -  a member function or a static data member of the current 
    //         instantiation
    else if (Var->isStaticDataMember() && 
             Var->getDeclContext()->isDependentContext())
      ExprBits.ValueDependent = true;
  } 
  // (VD) - FIXME: Missing from the standard: 
  //      -  a member function or a static data member of the current 
  //         instantiation
  else if (isa<CXXMethodDecl>(D) && D->getDeclContext()->isDependentContext())
    ExprBits.ValueDependent = true;
  //  (TD)  - a nested-name-specifier or a qualified-id that names a
  //          member of an unknown specialization.
  //        (handled by DependentScopeDeclRefExpr)

  // Determine whether this expression contains any unexpanded parameter
  // packs.
  // Is the declaration a parameter pack?
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    if (NTTP->isParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;
  }
  // FIXME: Variadic templates function parameter packs.
}

DeclRefExpr::DeclRefExpr(NestedNameSpecifier *Qualifier, 
                         SourceRange QualifierRange,
                         ValueDecl *D, SourceLocation NameLoc,
                         const TemplateArgumentListInfo *TemplateArgs,
                         QualType T, ExprValueKind VK)
  : Expr(DeclRefExprClass, T, VK, OK_Ordinary, false, false, false),
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
    getExplicitTemplateArgs().initializeFrom(*TemplateArgs);

  computeDependence();
}

DeclRefExpr::DeclRefExpr(NestedNameSpecifier *Qualifier,
                         SourceRange QualifierRange,
                         ValueDecl *D, const DeclarationNameInfo &NameInfo,
                         const TemplateArgumentListInfo *TemplateArgs,
                         QualType T, ExprValueKind VK)
  : Expr(DeclRefExprClass, T, VK, OK_Ordinary, false, false, false),
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
    getExplicitTemplateArgs().initializeFrom(*TemplateArgs);

  computeDependence();
}

DeclRefExpr *DeclRefExpr::Create(ASTContext &Context,
                                 NestedNameSpecifier *Qualifier,
                                 SourceRange QualifierRange,
                                 ValueDecl *D,
                                 SourceLocation NameLoc,
                                 QualType T,
                                 ExprValueKind VK,
                                 const TemplateArgumentListInfo *TemplateArgs) {
  return Create(Context, Qualifier, QualifierRange, D,
                DeclarationNameInfo(D->getDeclName(), NameLoc),
                T, VK, TemplateArgs);
}

DeclRefExpr *DeclRefExpr::Create(ASTContext &Context,
                                 NestedNameSpecifier *Qualifier,
                                 SourceRange QualifierRange,
                                 ValueDecl *D,
                                 const DeclarationNameInfo &NameInfo,
                                 QualType T,
                                 ExprValueKind VK,
                                 const TemplateArgumentListInfo *TemplateArgs) {
  std::size_t Size = sizeof(DeclRefExpr);
  if (Qualifier != 0)
    Size += sizeof(NameQualifier);
  
  if (TemplateArgs)
    Size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);
  
  void *Mem = Context.Allocate(Size, llvm::alignOf<DeclRefExpr>());
  return new (Mem) DeclRefExpr(Qualifier, QualifierRange, D, NameInfo,
                               TemplateArgs, T, VK);
}

DeclRefExpr *DeclRefExpr::CreateEmpty(ASTContext &Context, bool HasQualifier,
                                      unsigned NumTemplateArgs) {
  std::size_t Size = sizeof(DeclRefExpr);
  if (HasQualifier)
    Size += sizeof(NameQualifier);
  
  if (NumTemplateArgs)
    Size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  
  void *Mem = Context.Allocate(Size, llvm::alignOf<DeclRefExpr>());
  return new (Mem) DeclRefExpr(EmptyShell());
}

SourceRange DeclRefExpr::getSourceRange() const {
  SourceRange R = getNameInfo().getSourceRange();
  if (hasQualifier())
    R.setBegin(getQualifierRange().getBegin());
  if (hasExplicitTemplateArgs())
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

void APNumericStorage::setIntValue(ASTContext &C, const llvm::APInt &Val) {
  if (hasAllocation())
    C.Deallocate(pVal);

  BitWidth = Val.getBitWidth();
  unsigned NumWords = Val.getNumWords();
  const uint64_t* Words = Val.getRawData();
  if (NumWords > 1) {
    pVal = new (C) uint64_t[NumWords];
    std::copy(Words, Words + NumWords, pVal);
  } else if (NumWords == 1)
    VAL = Words[0];
  else
    VAL = 0;
}

IntegerLiteral *
IntegerLiteral::Create(ASTContext &C, const llvm::APInt &V,
                       QualType type, SourceLocation l) {
  return new (C) IntegerLiteral(C, V, type, l);
}

IntegerLiteral *
IntegerLiteral::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) IntegerLiteral(Empty);
}

FloatingLiteral *
FloatingLiteral::Create(ASTContext &C, const llvm::APFloat &V,
                        bool isexact, QualType Type, SourceLocation L) {
  return new (C) FloatingLiteral(C, V, isexact, Type, L);
}

FloatingLiteral *
FloatingLiteral::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) FloatingLiteral(Empty);
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
                         llvm::alignOf<StringLiteral>());
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
                         llvm::alignOf<StringLiteral>());
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

/// getLocationOfByte - Return a source location that points to the specified
/// byte of this string literal.
///
/// Strings are amazingly complex.  They can be formed from multiple tokens and
/// can have escape sequences in them in addition to the usual trigraph and
/// escaped newline business.  This routine handles this complexity.
///
SourceLocation StringLiteral::
getLocationOfByte(unsigned ByteNo, const SourceManager &SM,
                  const LangOptions &Features, const TargetInfo &Target) const {
  assert(!isWide() && "This doesn't work for wide strings yet");
  
  // Loop over all of the tokens in this string until we find the one that
  // contains the byte we're looking for.
  unsigned TokNo = 0;
  while (1) {
    assert(TokNo < getNumConcatenated() && "Invalid byte number!");
    SourceLocation StrTokLoc = getStrTokenLoc(TokNo);
    
    // Get the spelling of the string so that we can get the data that makes up
    // the string literal, not the identifier for the macro it is potentially
    // expanded through.
    SourceLocation StrTokSpellingLoc = SM.getSpellingLoc(StrTokLoc);
    
    // Re-lex the token to get its length and original spelling.
    std::pair<FileID, unsigned> LocInfo =SM.getDecomposedLoc(StrTokSpellingLoc);
    bool Invalid = false;
    llvm::StringRef Buffer = SM.getBufferData(LocInfo.first, &Invalid);
    if (Invalid)
      return StrTokSpellingLoc;
    
    const char *StrData = Buffer.data()+LocInfo.second;
    
    // Create a langops struct and enable trigraphs.  This is sufficient for
    // relexing tokens.
    LangOptions LangOpts;
    LangOpts.Trigraphs = true;
    
    // Create a lexer starting at the beginning of this token.
    Lexer TheLexer(StrTokSpellingLoc, Features, Buffer.begin(), StrData,
                   Buffer.end());
    Token TheTok;
    TheLexer.LexFromRawLexer(TheTok);
    
    // Use the StringLiteralParser to compute the length of the string in bytes.
    StringLiteralParser SLP(&TheTok, 1, SM, Features, Target);
    unsigned TokNumBytes = SLP.GetStringLength();
    
    // If the byte is in this token, return the location of the byte.
    if (ByteNo < TokNumBytes ||
        (ByteNo == TokNumBytes && TokNo == getNumConcatenated())) {
      unsigned Offset = SLP.getOffsetOfStringByte(TheTok, ByteNo); 
      
      // Now that we know the offset of the token in the spelling, use the
      // preprocessor to get the offset in the original source.
      return Lexer::AdvanceToTokenCharacter(StrTokLoc, Offset, SM, Features);
    }
    
    // Move to the next string token.
    ++TokNo;
    ByteNo -= TokNumBytes;
  }
}



/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "sizeof" or "[pre]++".
const char *UnaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown unary operator");
  case UO_PostInc: return "++";
  case UO_PostDec: return "--";
  case UO_PreInc:  return "++";
  case UO_PreDec:  return "--";
  case UO_AddrOf:  return "&";
  case UO_Deref:   return "*";
  case UO_Plus:    return "+";
  case UO_Minus:   return "-";
  case UO_Not:     return "~";
  case UO_LNot:    return "!";
  case UO_Real:    return "__real";
  case UO_Imag:    return "__imag";
  case UO_Extension: return "__extension__";
  }
}

UnaryOperatorKind
UnaryOperator::getOverloadedOpcode(OverloadedOperatorKind OO, bool Postfix) {
  switch (OO) {
  default: assert(false && "No unary operator for overloaded function");
  case OO_PlusPlus:   return Postfix ? UO_PostInc : UO_PreInc;
  case OO_MinusMinus: return Postfix ? UO_PostDec : UO_PreDec;
  case OO_Amp:        return UO_AddrOf;
  case OO_Star:       return UO_Deref;
  case OO_Plus:       return UO_Plus;
  case OO_Minus:      return UO_Minus;
  case OO_Tilde:      return UO_Not;
  case OO_Exclaim:    return UO_LNot;
  }
}

OverloadedOperatorKind UnaryOperator::getOverloadedOperator(Opcode Opc) {
  switch (Opc) {
  case UO_PostInc: case UO_PreInc: return OO_PlusPlus;
  case UO_PostDec: case UO_PreDec: return OO_MinusMinus;
  case UO_AddrOf: return OO_Amp;
  case UO_Deref: return OO_Star;
  case UO_Plus: return OO_Plus;
  case UO_Minus: return OO_Minus;
  case UO_Not: return OO_Tilde;
  case UO_LNot: return OO_Exclaim;
  default: return OO_None;
  }
}


//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

CallExpr::CallExpr(ASTContext& C, StmtClass SC, Expr *fn, Expr **args,
                   unsigned numargs, QualType t, ExprValueKind VK,
                   SourceLocation rparenloc)
  : Expr(SC, t, VK, OK_Ordinary,
         fn->isTypeDependent(),
         fn->isValueDependent(),
         fn->containsUnexpandedParameterPack()),
    NumArgs(numargs) {

  SubExprs = new (C) Stmt*[numargs+1];
  SubExprs[FN] = fn;
  for (unsigned i = 0; i != numargs; ++i) {
    if (args[i]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (args[i]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (args[i]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i+ARGS_START] = args[i];
  }

  RParenLoc = rparenloc;
}

CallExpr::CallExpr(ASTContext& C, Expr *fn, Expr **args, unsigned numargs,
                   QualType t, ExprValueKind VK, SourceLocation rparenloc)
  : Expr(CallExprClass, t, VK, OK_Ordinary,
         fn->isTypeDependent(),
         fn->isValueDependent(),
         fn->containsUnexpandedParameterPack()),
    NumArgs(numargs) {

  SubExprs = new (C) Stmt*[numargs+1];
  SubExprs[FN] = fn;
  for (unsigned i = 0; i != numargs; ++i) {
    if (args[i]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (args[i]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (args[i]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i+ARGS_START] = args[i];
  }

  RParenLoc = rparenloc;
}

CallExpr::CallExpr(ASTContext &C, StmtClass SC, EmptyShell Empty)
  : Expr(SC, Empty), SubExprs(0), NumArgs(0) {
  // FIXME: Why do we allocate this?
  SubExprs = new (C) Stmt*[1];
}

Decl *CallExpr::getCalleeDecl() {
  Expr *CEE = getCallee()->IgnoreParenCasts();
  // If we're calling a dereference, look at the pointer instead.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(CEE)) {
    if (BO->isPtrMemOp())
      CEE = BO->getRHS()->IgnoreParenCasts();
  } else if (UnaryOperator *UO = dyn_cast<UnaryOperator>(CEE)) {
    if (UO->getOpcode() == UO_Deref)
      CEE = UO->getSubExpr()->IgnoreParenCasts();
  }
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
  : Expr(OffsetOfExprClass, type, VK_RValue, OK_Ordinary,
         /*TypeDependent=*/false, 
         /*ValueDependent=*/tsi->getType()->isDependentType(),
         tsi->getType()->containsUnexpandedParameterPack()),
    OperatorLoc(OperatorLoc), RParenLoc(RParenLoc), TSInfo(tsi), 
    NumComps(numComps), NumExprs(numExprs) 
{
  for(unsigned i = 0; i < numComps; ++i) {
    setComponent(i, compsPtr[i]);
  }
  
  for(unsigned i = 0; i < numExprs; ++i) {
    if (exprsPtr[i]->isTypeDependent() || exprsPtr[i]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (exprsPtr[i]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

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
                               QualType ty,
                               ExprValueKind vk,
                               ExprObjectKind ok) {
  std::size_t Size = sizeof(MemberExpr);

  bool hasQualOrFound = (qual != 0 ||
                         founddecl.getDecl() != memberdecl ||
                         founddecl.getAccess() != memberdecl->getAccess());
  if (hasQualOrFound)
    Size += sizeof(MemberNameQualifier);

  if (targs)
    Size += ExplicitTemplateArgumentList::sizeFor(*targs);

  void *Mem = C.Allocate(Size, llvm::alignOf<MemberExpr>());
  MemberExpr *E = new (Mem) MemberExpr(base, isarrow, memberdecl, nameinfo,
                                       ty, vk, ok);

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
    E->getExplicitTemplateArgs().initializeFrom(*targs);
  }

  return E;
}

const char *CastExpr::getCastKindName() const {
  switch (getCastKind()) {
  case CK_Dependent:
    return "Dependent";
  case CK_BitCast:
    return "BitCast";
  case CK_LValueBitCast:
    return "LValueBitCast";
  case CK_LValueToRValue:
    return "LValueToRValue";
  case CK_GetObjCProperty:
    return "GetObjCProperty";
  case CK_NoOp:
    return "NoOp";
  case CK_BaseToDerived:
    return "BaseToDerived";
  case CK_DerivedToBase:
    return "DerivedToBase";
  case CK_UncheckedDerivedToBase:
    return "UncheckedDerivedToBase";
  case CK_Dynamic:
    return "Dynamic";
  case CK_ToUnion:
    return "ToUnion";
  case CK_ArrayToPointerDecay:
    return "ArrayToPointerDecay";
  case CK_FunctionToPointerDecay:
    return "FunctionToPointerDecay";
  case CK_NullToMemberPointer:
    return "NullToMemberPointer";
  case CK_NullToPointer:
    return "NullToPointer";
  case CK_BaseToDerivedMemberPointer:
    return "BaseToDerivedMemberPointer";
  case CK_DerivedToBaseMemberPointer:
    return "DerivedToBaseMemberPointer";
  case CK_UserDefinedConversion:
    return "UserDefinedConversion";
  case CK_ConstructorConversion:
    return "ConstructorConversion";
  case CK_IntegralToPointer:
    return "IntegralToPointer";
  case CK_PointerToIntegral:
    return "PointerToIntegral";
  case CK_PointerToBoolean:
    return "PointerToBoolean";
  case CK_ToVoid:
    return "ToVoid";
  case CK_VectorSplat:
    return "VectorSplat";
  case CK_IntegralCast:
    return "IntegralCast";
  case CK_IntegralToBoolean:
    return "IntegralToBoolean";
  case CK_IntegralToFloating:
    return "IntegralToFloating";
  case CK_FloatingToIntegral:
    return "FloatingToIntegral";
  case CK_FloatingCast:
    return "FloatingCast";
  case CK_FloatingToBoolean:
    return "FloatingToBoolean";
  case CK_MemberPointerToBoolean:
    return "MemberPointerToBoolean";
  case CK_AnyPointerToObjCPointerCast:
    return "AnyPointerToObjCPointerCast";
  case CK_AnyPointerToBlockPointerCast:
    return "AnyPointerToBlockPointerCast";
  case CK_ObjCObjectLValueCast:
    return "ObjCObjectLValueCast";
  case CK_FloatingRealToComplex:
    return "FloatingRealToComplex";
  case CK_FloatingComplexToReal:
    return "FloatingComplexToReal";
  case CK_FloatingComplexToBoolean:
    return "FloatingComplexToBoolean";
  case CK_FloatingComplexCast:
    return "FloatingComplexCast";
  case CK_FloatingComplexToIntegralComplex:
    return "FloatingComplexToIntegralComplex";
  case CK_IntegralRealToComplex:
    return "IntegralRealToComplex";
  case CK_IntegralComplexToReal:
    return "IntegralComplexToReal";
  case CK_IntegralComplexToBoolean:
    return "IntegralComplexToBoolean";
  case CK_IntegralComplexCast:
    return "IntegralComplexCast";
  case CK_IntegralComplexToFloatingComplex:
    return "IntegralComplexToFloatingComplex";
  }

  llvm_unreachable("Unhandled cast kind!");
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
    if (E->getCastKind() == CK_ConstructorConversion)
      SubExpr = cast<CXXConstructExpr>(SubExpr)->getArg(0);
    else if (E->getCastKind() == CK_UserDefinedConversion)
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
                                           ExprValueKind VK) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(ImplicitCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  ImplicitCastExpr *E =
    new (Buffer) ImplicitCastExpr(T, Kind, Operand, PathSize, VK);
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
                                       ExprValueKind VK, CastKind K, Expr *Op,
                                       const CXXCastPath *BasePath,
                                       TypeSourceInfo *WrittenTy,
                                       SourceLocation L, SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(CStyleCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  CStyleCastExpr *E =
    new (Buffer) CStyleCastExpr(T, VK, K, Op, PathSize, WrittenTy, L, R);
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
  case BO_PtrMemD:   return ".*";
  case BO_PtrMemI:   return "->*";
  case BO_Mul:       return "*";
  case BO_Div:       return "/";
  case BO_Rem:       return "%";
  case BO_Add:       return "+";
  case BO_Sub:       return "-";
  case BO_Shl:       return "<<";
  case BO_Shr:       return ">>";
  case BO_LT:        return "<";
  case BO_GT:        return ">";
  case BO_LE:        return "<=";
  case BO_GE:        return ">=";
  case BO_EQ:        return "==";
  case BO_NE:        return "!=";
  case BO_And:       return "&";
  case BO_Xor:       return "^";
  case BO_Or:        return "|";
  case BO_LAnd:      return "&&";
  case BO_LOr:       return "||";
  case BO_Assign:    return "=";
  case BO_MulAssign: return "*=";
  case BO_DivAssign: return "/=";
  case BO_RemAssign: return "%=";
  case BO_AddAssign: return "+=";
  case BO_SubAssign: return "-=";
  case BO_ShlAssign: return "<<=";
  case BO_ShrAssign: return ">>=";
  case BO_AndAssign: return "&=";
  case BO_XorAssign: return "^=";
  case BO_OrAssign:  return "|=";
  case BO_Comma:     return ",";
  }

  return "";
}

BinaryOperatorKind
BinaryOperator::getOverloadedOpcode(OverloadedOperatorKind OO) {
  switch (OO) {
  default: assert(false && "Not an overloadable binary operator");
  case OO_Plus: return BO_Add;
  case OO_Minus: return BO_Sub;
  case OO_Star: return BO_Mul;
  case OO_Slash: return BO_Div;
  case OO_Percent: return BO_Rem;
  case OO_Caret: return BO_Xor;
  case OO_Amp: return BO_And;
  case OO_Pipe: return BO_Or;
  case OO_Equal: return BO_Assign;
  case OO_Less: return BO_LT;
  case OO_Greater: return BO_GT;
  case OO_PlusEqual: return BO_AddAssign;
  case OO_MinusEqual: return BO_SubAssign;
  case OO_StarEqual: return BO_MulAssign;
  case OO_SlashEqual: return BO_DivAssign;
  case OO_PercentEqual: return BO_RemAssign;
  case OO_CaretEqual: return BO_XorAssign;
  case OO_AmpEqual: return BO_AndAssign;
  case OO_PipeEqual: return BO_OrAssign;
  case OO_LessLess: return BO_Shl;
  case OO_GreaterGreater: return BO_Shr;
  case OO_LessLessEqual: return BO_ShlAssign;
  case OO_GreaterGreaterEqual: return BO_ShrAssign;
  case OO_EqualEqual: return BO_EQ;
  case OO_ExclaimEqual: return BO_NE;
  case OO_LessEqual: return BO_LE;
  case OO_GreaterEqual: return BO_GE;
  case OO_AmpAmp: return BO_LAnd;
  case OO_PipePipe: return BO_LOr;
  case OO_Comma: return BO_Comma;
  case OO_ArrowStar: return BO_PtrMemI;
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
  : Expr(InitListExprClass, QualType(), VK_RValue, OK_Ordinary, false, false,
         false),
    InitExprs(C, numInits),
    LBraceLoc(lbraceloc), RBraceLoc(rbraceloc), SyntacticForm(0),
    UnionFieldInit(0), HadArrayRangeDesignator(false) 
{      
  for (unsigned I = 0; I != numInits; ++I) {
    if (initExprs[I]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (initExprs[I]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (initExprs[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;
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

SourceRange InitListExpr::getSourceRange() const {
  if (SyntacticForm)
    return SyntacticForm->getSourceRange();
  SourceLocation Beg = LBraceLoc, End = RBraceLoc;
  if (Beg.isInvalid()) {
    // Find the first non-null initializer.
    for (InitExprsTy::const_iterator I = InitExprs.begin(),
                                     E = InitExprs.end(); 
      I != E; ++I) {
      if (Stmt *S = *I) {
        Beg = S->getLocStart();
        break;
      }  
    }
  }
  if (End.isInvalid()) {
    // Find the first non-null initializer from the end.
    for (InitExprsTy::const_reverse_iterator I = InitExprs.rbegin(),
                                             E = InitExprs.rend();
      I != E; ++I) {
      if (Stmt *S = *I) {
        End = S->getSourceRange().getEnd();
        break;
      }  
    }
  }
  return SourceRange(Beg, End);
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
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:                 // ++/--
      return false;  // Not a warning.
    case UO_Deref:
      // Dereferencing a volatile pointer is a side-effect.
      if (Ctx.getCanonicalType(getType()).isVolatileQualified())
        return false;
      break;
    case UO_Real:
    case UO_Imag:
      // accessing a piece of a volatile complex is a side-effect.
      if (Ctx.getCanonicalType(UO->getSubExpr()->getType())
          .isVolatileQualified())
        return false;
      break;
    case UO_Extension:
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
      case BO_Comma:
        // ((foo = <blah>), 0) is an idiom for hiding the result (and
        // lvalue-ness) of an assignment written in a macro.
        if (IntegerLiteral *IE =
              dyn_cast<IntegerLiteral>(BO->getRHS()->IgnoreParens()))
          if (IE->getValue() == 0)
            return false;
        return BO->getRHS()->isUnusedResultAWarning(Loc, R1, R2, Ctx);
      // Consider '||', '&&' to have side effects if the LHS or RHS does.
      case BO_LAnd:
      case BO_LOr:
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

  case ObjCPropertyRefExprClass:
    Loc = getExprLoc();
    R1 = getSourceRange();
    return true;

  case StmtExprClass: {
    // Statement exprs don't logically have side effects themselves, but are
    // sometimes used in macros in ways that give them a type that is unused.
    // For example ({ blah; foo(); }) will end up with a type if foo has a type.
    // however, if the result of the stmt expr is dead, we don't want to emit a
    // warning.
    const CompoundStmt *CS = cast<StmtExpr>(this)->getSubStmt();
    if (!CS->body_empty()) {
      if (const Expr *E = dyn_cast<Expr>(CS->body_back()))
        return E->isUnusedResultAWarning(Loc, R1, R2, Ctx);
      if (const LabelStmt *Label = dyn_cast<LabelStmt>(CS->body_back()))
        if (const Expr *E = dyn_cast<Expr>(Label->getSubStmt()))
          return E->isUnusedResultAWarning(Loc, R1, R2, Ctx);
    }

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
    if (CE->getCastKind() == CK_ToVoid ||
        CE->getCastKind() == CK_ConstructorConversion)
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
  case ExprWithCleanupsClass:
    return (cast<ExprWithCleanups>(this)
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

bool Expr::isBoundMemberFunction(ASTContext &Ctx) const {
  if (isTypeDependent())
    return false;
  return ClassifyLValue(Ctx) == Expr::LV_MemberFunction;
}

static Expr::CanThrowResult MergeCanThrow(Expr::CanThrowResult CT1,
                                          Expr::CanThrowResult CT2) {
  // CanThrowResult constants are ordered so that the maximum is the correct
  // merge result.
  return CT1 > CT2 ? CT1 : CT2;
}

static Expr::CanThrowResult CanSubExprsThrow(ASTContext &C, const Expr *CE) {
  Expr *E = const_cast<Expr*>(CE);
  Expr::CanThrowResult R = Expr::CT_Cannot;
  for (Expr::child_iterator I = E->child_begin(), IE = E->child_end();
       I != IE && R != Expr::CT_Can; ++I) {
    R = MergeCanThrow(R, cast<Expr>(*I)->CanThrow(C));
  }
  return R;
}

static Expr::CanThrowResult CanCalleeThrow(const Decl *D,
                                           bool NullThrows = true) {
  if (!D)
    return NullThrows ? Expr::CT_Can : Expr::CT_Cannot;

  // See if we can get a function type from the decl somehow.
  const ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD) // If we have no clue what we're calling, assume the worst.
    return Expr::CT_Can;

  // As an extension, we assume that __attribute__((nothrow)) functions don't
  // throw.
  if (isa<FunctionDecl>(D) && D->hasAttr<NoThrowAttr>())
    return Expr::CT_Cannot;

  QualType T = VD->getType();
  const FunctionProtoType *FT;
  if ((FT = T->getAs<FunctionProtoType>())) {
  } else if (const PointerType *PT = T->getAs<PointerType>())
    FT = PT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const ReferenceType *RT = T->getAs<ReferenceType>())
    FT = RT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const MemberPointerType *MT = T->getAs<MemberPointerType>())
    FT = MT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const BlockPointerType *BT = T->getAs<BlockPointerType>())
    FT = BT->getPointeeType()->getAs<FunctionProtoType>();

  if (!FT)
    return Expr::CT_Can;

  return FT->hasEmptyExceptionSpec() ? Expr::CT_Cannot : Expr::CT_Can;
}

static Expr::CanThrowResult CanDynamicCastThrow(const CXXDynamicCastExpr *DC) {
  if (DC->isTypeDependent())
    return Expr::CT_Dependent;

  if (!DC->getTypeAsWritten()->isReferenceType())
    return Expr::CT_Cannot;

  return DC->getCastKind() == clang::CK_Dynamic? Expr::CT_Can : Expr::CT_Cannot;
}

static Expr::CanThrowResult CanTypeidThrow(ASTContext &C,
                                           const CXXTypeidExpr *DC) {
  if (DC->isTypeOperand())
    return Expr::CT_Cannot;

  Expr *Op = DC->getExprOperand();
  if (Op->isTypeDependent())
    return Expr::CT_Dependent;

  const RecordType *RT = Op->getType()->getAs<RecordType>();
  if (!RT)
    return Expr::CT_Cannot;

  if (!cast<CXXRecordDecl>(RT->getDecl())->isPolymorphic())
    return Expr::CT_Cannot;

  if (Op->Classify(C).isPRValue())
    return Expr::CT_Cannot;

  return Expr::CT_Can;
}

Expr::CanThrowResult Expr::CanThrow(ASTContext &C) const {
  // C++ [expr.unary.noexcept]p3:
  //   [Can throw] if in a potentially-evaluated context the expression would
  //   contain:
  switch (getStmtClass()) {
  case CXXThrowExprClass:
    //   - a potentially evaluated throw-expression
    return CT_Can;

  case CXXDynamicCastExprClass: {
    //   - a potentially evaluated dynamic_cast expression dynamic_cast<T>(v),
    //     where T is a reference type, that requires a run-time check
    CanThrowResult CT = CanDynamicCastThrow(cast<CXXDynamicCastExpr>(this));
    if (CT == CT_Can)
      return CT;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

  case CXXTypeidExprClass:
    //   - a potentially evaluated typeid expression applied to a glvalue
    //     expression whose type is a polymorphic class type
    return CanTypeidThrow(C, cast<CXXTypeidExpr>(this));

    //   - a potentially evaluated call to a function, member function, function
    //     pointer, or member function pointer that does not have a non-throwing
    //     exception-specification
  case CallExprClass:
  case CXXOperatorCallExprClass:
  case CXXMemberCallExprClass: {
    CanThrowResult CT = CanCalleeThrow(cast<CallExpr>(this)->getCalleeDecl());
    if (CT == CT_Can)
      return CT;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

  case CXXConstructExprClass:
  case CXXTemporaryObjectExprClass: {
    CanThrowResult CT = CanCalleeThrow(
        cast<CXXConstructExpr>(this)->getConstructor());
    if (CT == CT_Can)
      return CT;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

  case CXXNewExprClass: {
    CanThrowResult CT = MergeCanThrow(
        CanCalleeThrow(cast<CXXNewExpr>(this)->getOperatorNew()),
        CanCalleeThrow(cast<CXXNewExpr>(this)->getConstructor(),
                       /*NullThrows*/false));
    if (CT == CT_Can)
      return CT;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

  case CXXDeleteExprClass: {
    CanThrowResult CT = CanCalleeThrow(
        cast<CXXDeleteExpr>(this)->getOperatorDelete());
    if (CT == CT_Can)
      return CT;
    const Expr *Arg = cast<CXXDeleteExpr>(this)->getArgument();
    // Unwrap exactly one implicit cast, which converts all pointers to void*.
    if (const ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
      Arg = Cast->getSubExpr();
    if (const PointerType *PT = Arg->getType()->getAs<PointerType>()) {
      if (const RecordType *RT = PT->getPointeeType()->getAs<RecordType>()) {
        CanThrowResult CT2 = CanCalleeThrow(
            cast<CXXRecordDecl>(RT->getDecl())->getDestructor());
        if (CT2 == CT_Can)
          return CT2;
        CT = MergeCanThrow(CT, CT2);
      }
    }
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

  case CXXBindTemporaryExprClass: {
    // The bound temporary has to be destroyed again, which might throw.
    CanThrowResult CT = CanCalleeThrow(
      cast<CXXBindTemporaryExpr>(this)->getTemporary()->getDestructor());
    if (CT == CT_Can)
      return CT;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

    // ObjC message sends are like function calls, but never have exception
    // specs.
  case ObjCMessageExprClass:
  case ObjCPropertyRefExprClass:
    return CT_Can;

    // Many other things have subexpressions, so we have to test those.
    // Some are simple:
  case ParenExprClass:
  case MemberExprClass:
  case CXXReinterpretCastExprClass:
  case CXXConstCastExprClass:
  case ConditionalOperatorClass:
  case CompoundLiteralExprClass:
  case ExtVectorElementExprClass:
  case InitListExprClass:
  case DesignatedInitExprClass:
  case ParenListExprClass:
  case VAArgExprClass:
  case CXXDefaultArgExprClass:
  case ExprWithCleanupsClass:
  case ObjCIvarRefExprClass:
  case ObjCIsaExprClass:
  case ShuffleVectorExprClass:
    return CanSubExprsThrow(C, this);

    // Some might be dependent for other reasons.
  case UnaryOperatorClass:
  case ArraySubscriptExprClass:
  case ImplicitCastExprClass:
  case CStyleCastExprClass:
  case CXXStaticCastExprClass:
  case CXXFunctionalCastExprClass:
  case BinaryOperatorClass:
  case CompoundAssignOperatorClass: {
    CanThrowResult CT = isTypeDependent() ? CT_Dependent : CT_Cannot;
    return MergeCanThrow(CT, CanSubExprsThrow(C, this));
  }

    // FIXME: We should handle StmtExpr, but that opens a MASSIVE can of worms.
  case StmtExprClass:
    return CT_Can;

  case ChooseExprClass:
    if (isTypeDependent() || isValueDependent())
      return CT_Dependent;
    return cast<ChooseExpr>(this)->getChosenSubExpr(C)->CanThrow(C);

    // Some expressions are always dependent.
  case DependentScopeDeclRefExprClass:
  case CXXUnresolvedConstructExprClass:
  case CXXDependentScopeMemberExprClass:
    return CT_Dependent;

  default:
    // All other expressions don't have subexpressions, or else they are
    // unevaluated.
    return CT_Cannot;
  }
}

Expr* Expr::IgnoreParens() {
  Expr* E = this;
  while (true) {
    if (ParenExpr* P = dyn_cast<ParenExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }
    if (UnaryOperator* P = dyn_cast<UnaryOperator>(E)) {
      if (P->getOpcode() == UO_Extension) {
        E = P->getSubExpr();
        continue;
      }
    }
    return E;
  }
}

/// IgnoreParenCasts - Ignore parentheses and casts.  Strip off any ParenExpr
/// or CastExprs or ImplicitCastExprs, returning their operand.
Expr *Expr::IgnoreParenCasts() {
  Expr *E = this;
  while (true) {
    if (ParenExpr* P = dyn_cast<ParenExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }
    if (CastExpr *P = dyn_cast<CastExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }
    if (UnaryOperator* P = dyn_cast<UnaryOperator>(E)) {
      if (P->getOpcode() == UO_Extension) {
        E = P->getSubExpr();
        continue;
      }
    }
    return E;
  }
}

/// IgnoreParenLValueCasts - Ignore parentheses and lvalue-to-rvalue
/// casts.  This is intended purely as a temporary workaround for code
/// that hasn't yet been rewritten to do the right thing about those
/// casts, and may disappear along with the last internal use.
Expr *Expr::IgnoreParenLValueCasts() {
  Expr *E = this;
  while (true) {
    if (ParenExpr *P = dyn_cast<ParenExpr>(E)) {
      E = P->getSubExpr();
      continue;
    } else if (CastExpr *P = dyn_cast<CastExpr>(E)) {
      if (P->getCastKind() == CK_LValueToRValue) {
        E = P->getSubExpr();
        continue;
      }
    } else if (UnaryOperator* P = dyn_cast<UnaryOperator>(E)) {
      if (P->getOpcode() == UO_Extension) {
        E = P->getSubExpr();
        continue;
      }
    }
    break;
  }
  return E;
}
  
Expr *Expr::IgnoreParenImpCasts() {
  Expr *E = this;
  while (true) {
    if (ParenExpr *P = dyn_cast<ParenExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }
    if (ImplicitCastExpr *P = dyn_cast<ImplicitCastExpr>(E)) {
      E = P->getSubExpr();
      continue;
    }
    if (UnaryOperator* P = dyn_cast<UnaryOperator>(E)) {
      if (P->getOpcode() == UO_Extension) {
        E = P->getSubExpr();
        continue;
      }
    }
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

    if (UnaryOperator* P = dyn_cast<UnaryOperator>(E)) {
      if (P->getOpcode() == UO_Extension) {
        E = P->getSubExpr();
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
static const Expr *skipTemporaryBindingsNoOpCastsAndParens(const Expr *E) {
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_NoOp)
      E = ICE->getSubExpr();
    else
      break;
  }

  while (const CXXBindTemporaryExpr *BE = dyn_cast<CXXBindTemporaryExpr>(E))
    E = BE->getSubExpr();

  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_NoOp)
      E = ICE->getSubExpr();
    else
      break;
  }

  return E->IgnoreParens();
}

/// isTemporaryObject - Determines if this expression produces a
/// temporary of the given class type.
bool Expr::isTemporaryObject(ASTContext &C, const CXXRecordDecl *TempTy) const {
  if (!C.hasSameUnqualifiedType(getType(), C.getTypeDeclType(TempTy)))
    return false;

  const Expr *E = skipTemporaryBindingsNoOpCastsAndParens(this);

  // Temporaries are by definition pr-values of class type.
  if (!E->Classify(C).isPRValue()) {
    // In this context, property reference is a message call and is pr-value.
    if (!isa<ObjCPropertyRefExpr>(E))
      return false;
  }

  // Black-list a few cases which yield pr-values of class type that don't
  // refer to temporaries of that type:

  // - implicit derived-to-base conversions
  if (isa<ImplicitCastExpr>(E)) {
    switch (cast<ImplicitCastExpr>(E)->getCastKind()) {
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
      return false;
    default:
      break;
    }
  }

  // - member expressions (all)
  if (isa<MemberExpr>(E))
    return false;

  return true;
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
  case ChooseExprClass:
    return cast<ChooseExpr>(this)->getChosenSubExpr(Ctx)
      ->isConstantInitializer(Ctx, IsForRef);
  case UnaryOperatorClass: {
    const UnaryOperator* Exp = cast<UnaryOperator>(this);
    if (Exp->getOpcode() == UO_Extension)
      return Exp->getSubExpr()->isConstantInitializer(Ctx, false);
    break;
  }
  case BinaryOperatorClass: {
    // Special case &&foo - &&bar.  It would be nice to generalize this somehow
    // but this handles the common case.
    const BinaryOperator *Exp = cast<BinaryOperator>(this);
    if (Exp->getOpcode() == BO_Sub &&
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

  if (const RecordType *UT = getType()->getAsUnionType())
    if (UT && UT->getDecl()->hasAttr<TransparentUnionAttr>())
      if (const CompoundLiteralExpr *CLE = dyn_cast<CompoundLiteralExpr>(this)){
        const Expr *InitExpr = CLE->getInitializer();
        if (const InitListExpr *ILE = dyn_cast<InitListExpr>(InitExpr))
          return ILE->getInit(0)->isNullPointerConstant(Ctx, NPC);
      }
  // This expression must be an integer type.
  if (!getType()->isIntegerType() || 
      (Ctx.getLangOptions().CPlusPlus && getType()->isEnumeralType()))
    return false;

  // If we have an integer constant expression, we need to *evaluate* it and
  // test for the value 0.
  llvm::APSInt Result;
  return isIntegerConstantExpr(Result, Ctx) && Result == 0;
}

/// \brief If this expression is an l-value for an Objective C
/// property, find the underlying property reference expression.
const ObjCPropertyRefExpr *Expr::getObjCProperty() const {
  const Expr *E = this;
  while (true) {
    assert((E->getValueKind() == VK_LValue &&
            E->getObjectKind() == OK_ObjCProperty) &&
           "expression is not a property reference");
    E = E->IgnoreParenCasts();
    if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
      if (BO->getOpcode() == BO_Comma) {
        E = BO->getRHS();
        continue;
      }
    }

    break;
  }

  return cast<ObjCPropertyRefExpr>(E);
}

FieldDecl *Expr::getBitField() {
  Expr *E = this->IgnoreParens();

  while (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_LValueToRValue ||
        (ICE->getValueKind() != VK_RValue && ICE->getCastKind() == CK_NoOp))
      E = ICE->getSubExpr()->IgnoreParens();
    else
      break;
  }

  if (MemberExpr *MemRef = dyn_cast<MemberExpr>(E))
    if (FieldDecl *Field = dyn_cast<FieldDecl>(MemRef->getMemberDecl()))
      if (Field->isBitField())
        return Field;

  if (DeclRefExpr *DeclRef = dyn_cast<DeclRefExpr>(E))
    if (FieldDecl *Field = dyn_cast<FieldDecl>(DeclRef->getDecl()))
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
    if (ICE->getValueKind() != VK_RValue &&
        ICE->getCastKind() == CK_NoOp)
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
                                 ExprValueKind VK,
                                 SourceLocation LBracLoc,
                                 SourceLocation SuperLoc,
                                 bool IsInstanceSuper,
                                 QualType SuperType,
                                 Selector Sel, 
                                 SourceLocation SelLoc,
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, VK, OK_Ordinary,
         /*TypeDependent=*/false, /*ValueDependent=*/false,
         /*ContainsUnexpandedParameterPack=*/false),
    NumArgs(NumArgs), Kind(IsInstanceSuper? SuperInstance : SuperClass),
    HasMethod(Method != 0), SuperLoc(SuperLoc),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    SelectorLoc(SelLoc), LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(SuperType.getAsOpaquePtr());
  if (NumArgs)
    memcpy(getArgs(), Args, NumArgs * sizeof(Expr *));
}

ObjCMessageExpr::ObjCMessageExpr(QualType T,
                                 ExprValueKind VK,
                                 SourceLocation LBracLoc,
                                 TypeSourceInfo *Receiver,
                                 Selector Sel,
                                 SourceLocation SelLoc,
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, VK, OK_Ordinary, T->isDependentType(),
         T->isDependentType(), T->containsUnexpandedParameterPack()),
    NumArgs(NumArgs), Kind(Class), HasMethod(Method != 0),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    SelectorLoc(SelLoc), LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(Receiver);
  Stmt **MyArgs = getArgs();
  for (unsigned I = 0; I != NumArgs; ++I) {
    if (Args[I]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (Args[I]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;
  
    MyArgs[I] = Args[I];
  }
}

ObjCMessageExpr::ObjCMessageExpr(QualType T,
                                 ExprValueKind VK,
                                 SourceLocation LBracLoc,
                                 Expr *Receiver,
                                 Selector Sel, 
                                 SourceLocation SelLoc,
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc)
  : Expr(ObjCMessageExprClass, T, VK, OK_Ordinary, Receiver->isTypeDependent(),
         Receiver->isTypeDependent(),
         Receiver->containsUnexpandedParameterPack()),
    NumArgs(NumArgs), Kind(Instance), HasMethod(Method != 0),
    SelectorOrMethod(reinterpret_cast<uintptr_t>(Method? Method
                                                       : Sel.getAsOpaquePtr())),
    SelectorLoc(SelLoc), LBracLoc(LBracLoc), RBracLoc(RBracLoc) 
{
  setReceiverPointer(Receiver);
  Stmt **MyArgs = getArgs();
  for (unsigned I = 0; I != NumArgs; ++I) {
    if (Args[I]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (Args[I]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;
  
    MyArgs[I] = Args[I];
  }
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         ExprValueKind VK,
                                         SourceLocation LBracLoc,
                                         SourceLocation SuperLoc,
                                         bool IsInstanceSuper,
                                         QualType SuperType,
                                         Selector Sel, 
                                         SourceLocation SelLoc,
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, VK, LBracLoc, SuperLoc, IsInstanceSuper,
                                   SuperType, Sel, SelLoc, Method, Args,NumArgs, 
                                   RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         ExprValueKind VK,
                                         SourceLocation LBracLoc,
                                         TypeSourceInfo *Receiver,
                                         Selector Sel, 
                                         SourceLocation SelLoc,
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, VK, LBracLoc, Receiver, Sel, SelLoc,
                                   Method, Args, NumArgs, RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::Create(ASTContext &Context, QualType T,
                                         ExprValueKind VK,
                                         SourceLocation LBracLoc,
                                         Expr *Receiver,
                                         Selector Sel,
                                         SourceLocation SelLoc,
                                         ObjCMethodDecl *Method,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation RBracLoc) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(T, VK, LBracLoc, Receiver, Sel, SelLoc,
                                   Method, Args, NumArgs, RBracLoc);
}

ObjCMessageExpr *ObjCMessageExpr::CreateEmpty(ASTContext &Context, 
                                              unsigned NumArgs) {
  unsigned Size = sizeof(ObjCMessageExpr) + sizeof(void *) + 
    NumArgs * sizeof(Expr *);
  void *Mem = Context.Allocate(Size, llvm::AlignOf<ObjCMessageExpr>::Alignment);
  return new (Mem) ObjCMessageExpr(EmptyShell(), NumArgs);
}

SourceRange ObjCMessageExpr::getReceiverRange() const {
  switch (getReceiverKind()) {
  case Instance:
    return getInstanceReceiver()->getSourceRange();

  case Class:
    return getClassReceiverTypeInfo()->getTypeLoc().getSourceRange();

  case SuperInstance:
  case SuperClass:
    return getSuperLoc();
  }

  return SourceLocation();
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

ShuffleVectorExpr::ShuffleVectorExpr(ASTContext &C, Expr **args, unsigned nexpr,
                                     QualType Type, SourceLocation BLoc,
                                     SourceLocation RP) 
   : Expr(ShuffleVectorExprClass, Type, VK_RValue, OK_Ordinary,
          Type->isDependentType(), Type->isDependentType(),
          Type->containsUnexpandedParameterPack()),
     BuiltinLoc(BLoc), RParenLoc(RP), NumExprs(nexpr) 
{
  SubExprs = new (C) Stmt*[nexpr];
  for (unsigned i = 0; i < nexpr; i++) {
    if (args[i]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (args[i]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (args[i]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i] = args[i];
  }
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
         Init->getValueKind(), Init->getObjectKind(),
         Init->isTypeDependent(), Init->isValueDependent(),
         Init->containsUnexpandedParameterPack()),
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
      if (Index->isTypeDependent() || Index->isValueDependent())
        ExprBits.ValueDependent = true;

      // Propagate unexpanded parameter packs.
      if (Index->containsUnexpandedParameterPack())
        ExprBits.ContainsUnexpandedParameterPack = true;

      // Copy the index expressions into permanent storage.
      *Child++ = IndexExprs[IndexIdx++];
    } else if (this->Designators[I].isArrayRangeDesignator()) {
      // Compute type- and value-dependence.
      Expr *Start = IndexExprs[IndexIdx];
      Expr *End = IndexExprs[IndexIdx + 1];
      if (Start->isTypeDependent() || Start->isValueDependent() ||
          End->isTypeDependent() || End->isValueDependent())
        ExprBits.ValueDependent = true;

      // Propagate unexpanded parameter packs.
      if (Start->containsUnexpandedParameterPack() ||
          End->containsUnexpandedParameterPack())
        ExprBits.ContainsUnexpandedParameterPack = true;

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
  : Expr(ParenListExprClass, QualType(), VK_RValue, OK_Ordinary,
         false, false, false),
    NumExprs(nexprs), LParenLoc(lparenloc), RParenLoc(rparenloc) {

  Exprs = new (C) Stmt*[nexprs];
  for (unsigned i = 0; i != nexprs; ++i) {
    if (exprs[i]->isTypeDependent())
      ExprBits.TypeDependent = true;
    if (exprs[i]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (exprs[i]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    Exprs[i] = exprs[i];
  }
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
Stmt::child_iterator ObjCPropertyRefExpr::child_begin()
{ 
  if (Receiver.is<Stmt*>()) {
    // Hack alert!
    return reinterpret_cast<Stmt**> (&Receiver);
  }
  return child_iterator(); 
}

Stmt::child_iterator ObjCPropertyRefExpr::child_end()
{ return Receiver.is<Stmt*>() ? 
          reinterpret_cast<Stmt**> (&Receiver)+1 : 
          child_iterator(); 
}

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

// OpaqueValueExpr
SourceRange OpaqueValueExpr::getSourceRange() const { return SourceRange(); }
Stmt::child_iterator OpaqueValueExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator OpaqueValueExpr::child_end() { return child_iterator(); }

