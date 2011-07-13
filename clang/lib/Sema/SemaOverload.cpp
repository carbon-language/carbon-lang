//===--- SemaOverload.cpp - C++ Overloading ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ overloading.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace clang {
using namespace sema;

/// A convenience routine for creating a decayed reference to a
/// function.
static ExprResult
CreateFunctionRefExpr(Sema &S, FunctionDecl *Fn,
                      SourceLocation Loc = SourceLocation()) {
  ExprResult E = S.Owned(new (S.Context) DeclRefExpr(Fn, Fn->getType(), VK_LValue, Loc));
  E = S.DefaultFunctionArrayConversion(E.take());
  if (E.isInvalid())
    return ExprError();
  return move(E);
}

static bool IsStandardConversion(Sema &S, Expr* From, QualType ToType,
                                 bool InOverloadResolution,
                                 StandardConversionSequence &SCS,
                                 bool CStyle,
                                 bool AllowObjCWritebackConversion);
  
static bool IsTransparentUnionStandardConversion(Sema &S, Expr* From, 
                                                 QualType &ToType,
                                                 bool InOverloadResolution,
                                                 StandardConversionSequence &SCS,
                                                 bool CStyle);
static OverloadingResult
IsUserDefinedConversion(Sema &S, Expr *From, QualType ToType,
                        UserDefinedConversionSequence& User,
                        OverloadCandidateSet& Conversions,
                        bool AllowExplicit);


static ImplicitConversionSequence::CompareKind
CompareStandardConversionSequences(Sema &S,
                                   const StandardConversionSequence& SCS1,
                                   const StandardConversionSequence& SCS2);

static ImplicitConversionSequence::CompareKind
CompareQualificationConversions(Sema &S,
                                const StandardConversionSequence& SCS1,
                                const StandardConversionSequence& SCS2);

static ImplicitConversionSequence::CompareKind
CompareDerivedToBaseConversions(Sema &S,
                                const StandardConversionSequence& SCS1,
                                const StandardConversionSequence& SCS2);



/// GetConversionCategory - Retrieve the implicit conversion
/// category corresponding to the given implicit conversion kind.
ImplicitConversionCategory
GetConversionCategory(ImplicitConversionKind Kind) {
  static const ImplicitConversionCategory
    Category[(int)ICK_Num_Conversion_Kinds] = {
    ICC_Identity,
    ICC_Lvalue_Transformation,
    ICC_Lvalue_Transformation,
    ICC_Lvalue_Transformation,
    ICC_Identity,
    ICC_Qualification_Adjustment,
    ICC_Promotion,
    ICC_Promotion,
    ICC_Promotion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion
  };
  return Category[(int)Kind];
}

/// GetConversionRank - Retrieve the implicit conversion rank
/// corresponding to the given implicit conversion kind.
ImplicitConversionRank GetConversionRank(ImplicitConversionKind Kind) {
  static const ImplicitConversionRank
    Rank[(int)ICK_Num_Conversion_Kinds] = {
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Promotion,
    ICR_Promotion,
    ICR_Promotion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Complex_Real_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Writeback_Conversion
  };
  return Rank[(int)Kind];
}

/// GetImplicitConversionName - Return the name of this kind of
/// implicit conversion.
const char* GetImplicitConversionName(ImplicitConversionKind Kind) {
  static const char* const Name[(int)ICK_Num_Conversion_Kinds] = {
    "No conversion",
    "Lvalue-to-rvalue",
    "Array-to-pointer",
    "Function-to-pointer",
    "Noreturn adjustment",
    "Qualification",
    "Integral promotion",
    "Floating point promotion",
    "Complex promotion",
    "Integral conversion",
    "Floating conversion",
    "Complex conversion",
    "Floating-integral conversion",
    "Pointer conversion",
    "Pointer-to-member conversion",
    "Boolean conversion",
    "Compatible-types conversion",
    "Derived-to-base conversion",
    "Vector conversion",
    "Vector splat",
    "Complex-real conversion",
    "Block Pointer conversion",
    "Transparent Union Conversion"
    "Writeback conversion"
  };
  return Name[Kind];
}

/// StandardConversionSequence - Set the standard conversion
/// sequence to the identity conversion.
void StandardConversionSequence::setAsIdentityConversion() {
  First = ICK_Identity;
  Second = ICK_Identity;
  Third = ICK_Identity;
  DeprecatedStringLiteralToCharPtr = false;
  QualificationIncludesObjCLifetime = false;
  ReferenceBinding = false;
  DirectBinding = false;
  IsLvalueReference = true;
  BindsToFunctionLvalue = false;
  BindsToRvalue = false;
  BindsImplicitObjectArgumentWithoutRefQualifier = false;
  ObjCLifetimeConversionBinding = false;
  CopyConstructor = 0;
}

/// getRank - Retrieve the rank of this standard conversion sequence
/// (C++ 13.3.3.1.1p3). The rank is the largest rank of each of the
/// implicit conversions.
ImplicitConversionRank StandardConversionSequence::getRank() const {
  ImplicitConversionRank Rank = ICR_Exact_Match;
  if  (GetConversionRank(First) > Rank)
    Rank = GetConversionRank(First);
  if  (GetConversionRank(Second) > Rank)
    Rank = GetConversionRank(Second);
  if  (GetConversionRank(Third) > Rank)
    Rank = GetConversionRank(Third);
  return Rank;
}

/// isPointerConversionToBool - Determines whether this conversion is
/// a conversion of a pointer or pointer-to-member to bool. This is
/// used as part of the ranking of standard conversion sequences
/// (C++ 13.3.3.2p4).
bool StandardConversionSequence::isPointerConversionToBool() const {
  // Note that FromType has not necessarily been transformed by the
  // array-to-pointer or function-to-pointer implicit conversions, so
  // check for their presence as well as checking whether FromType is
  // a pointer.
  if (getToType(1)->isBooleanType() &&
      (getFromType()->isPointerType() ||
       getFromType()->isObjCObjectPointerType() ||
       getFromType()->isBlockPointerType() ||
       getFromType()->isNullPtrType() ||
       First == ICK_Array_To_Pointer || First == ICK_Function_To_Pointer))
    return true;

  return false;
}

/// isPointerConversionToVoidPointer - Determines whether this
/// conversion is a conversion of a pointer to a void pointer. This is
/// used as part of the ranking of standard conversion sequences (C++
/// 13.3.3.2p4).
bool
StandardConversionSequence::
isPointerConversionToVoidPointer(ASTContext& Context) const {
  QualType FromType = getFromType();
  QualType ToType = getToType(1);

  // Note that FromType has not necessarily been transformed by the
  // array-to-pointer implicit conversion, so check for its presence
  // and redo the conversion to get a pointer.
  if (First == ICK_Array_To_Pointer)
    FromType = Context.getArrayDecayedType(FromType);

  if (Second == ICK_Pointer_Conversion && FromType->isAnyPointerType())
    if (const PointerType* ToPtrType = ToType->getAs<PointerType>())
      return ToPtrType->getPointeeType()->isVoidType();

  return false;
}

/// DebugPrint - Print this standard conversion sequence to standard
/// error. Useful for debugging overloading issues.
void StandardConversionSequence::DebugPrint() const {
  llvm::raw_ostream &OS = llvm::errs();
  bool PrintedSomething = false;
  if (First != ICK_Identity) {
    OS << GetImplicitConversionName(First);
    PrintedSomething = true;
  }

  if (Second != ICK_Identity) {
    if (PrintedSomething) {
      OS << " -> ";
    }
    OS << GetImplicitConversionName(Second);

    if (CopyConstructor) {
      OS << " (by copy constructor)";
    } else if (DirectBinding) {
      OS << " (direct reference binding)";
    } else if (ReferenceBinding) {
      OS << " (reference binding)";
    }
    PrintedSomething = true;
  }

  if (Third != ICK_Identity) {
    if (PrintedSomething) {
      OS << " -> ";
    }
    OS << GetImplicitConversionName(Third);
    PrintedSomething = true;
  }

  if (!PrintedSomething) {
    OS << "No conversions required";
  }
}

/// DebugPrint - Print this user-defined conversion sequence to standard
/// error. Useful for debugging overloading issues.
void UserDefinedConversionSequence::DebugPrint() const {
  llvm::raw_ostream &OS = llvm::errs();
  if (Before.First || Before.Second || Before.Third) {
    Before.DebugPrint();
    OS << " -> ";
  }
  OS << '\'' << ConversionFunction << '\'';
  if (After.First || After.Second || After.Third) {
    OS << " -> ";
    After.DebugPrint();
  }
}

/// DebugPrint - Print this implicit conversion sequence to standard
/// error. Useful for debugging overloading issues.
void ImplicitConversionSequence::DebugPrint() const {
  llvm::raw_ostream &OS = llvm::errs();
  switch (ConversionKind) {
  case StandardConversion:
    OS << "Standard conversion: ";
    Standard.DebugPrint();
    break;
  case UserDefinedConversion:
    OS << "User-defined conversion: ";
    UserDefined.DebugPrint();
    break;
  case EllipsisConversion:
    OS << "Ellipsis conversion";
    break;
  case AmbiguousConversion:
    OS << "Ambiguous conversion";
    break;
  case BadConversion:
    OS << "Bad conversion";
    break;
  }

  OS << "\n";
}

void AmbiguousConversionSequence::construct() {
  new (&conversions()) ConversionSet();
}

void AmbiguousConversionSequence::destruct() {
  conversions().~ConversionSet();
}

void
AmbiguousConversionSequence::copyFrom(const AmbiguousConversionSequence &O) {
  FromTypePtr = O.FromTypePtr;
  ToTypePtr = O.ToTypePtr;
  new (&conversions()) ConversionSet(O.conversions());
}

namespace {
  // Structure used by OverloadCandidate::DeductionFailureInfo to store
  // template parameter and template argument information.
  struct DFIParamWithArguments {
    TemplateParameter Param;
    TemplateArgument FirstArg;
    TemplateArgument SecondArg;
  };
}

/// \brief Convert from Sema's representation of template deduction information
/// to the form used in overload-candidate information.
OverloadCandidate::DeductionFailureInfo
static MakeDeductionFailureInfo(ASTContext &Context,
                                Sema::TemplateDeductionResult TDK,
                                TemplateDeductionInfo &Info) {
  OverloadCandidate::DeductionFailureInfo Result;
  Result.Result = static_cast<unsigned>(TDK);
  Result.Data = 0;
  switch (TDK) {
  case Sema::TDK_Success:
  case Sema::TDK_InstantiationDepth:
  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
    break;

  case Sema::TDK_Incomplete:
  case Sema::TDK_InvalidExplicitArguments:
    Result.Data = Info.Param.getOpaqueValue();
    break;

  case Sema::TDK_Inconsistent:
  case Sema::TDK_Underqualified: {
    // FIXME: Should allocate from normal heap so that we can free this later.
    DFIParamWithArguments *Saved = new (Context) DFIParamWithArguments;
    Saved->Param = Info.Param;
    Saved->FirstArg = Info.FirstArg;
    Saved->SecondArg = Info.SecondArg;
    Result.Data = Saved;
    break;
  }

  case Sema::TDK_SubstitutionFailure:
    Result.Data = Info.take();
    break;

  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    break;
  }

  return Result;
}

void OverloadCandidate::DeductionFailureInfo::Destroy() {
  switch (static_cast<Sema::TemplateDeductionResult>(Result)) {
  case Sema::TDK_Success:
  case Sema::TDK_InstantiationDepth:
  case Sema::TDK_Incomplete:
  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
  case Sema::TDK_InvalidExplicitArguments:
    break;

  case Sema::TDK_Inconsistent:
  case Sema::TDK_Underqualified:
    // FIXME: Destroy the data?
    Data = 0;
    break;

  case Sema::TDK_SubstitutionFailure:
    // FIXME: Destroy the template arugment list?
    Data = 0;
    break;

  // Unhandled
  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    break;
  }
}

TemplateParameter
OverloadCandidate::DeductionFailureInfo::getTemplateParameter() {
  switch (static_cast<Sema::TemplateDeductionResult>(Result)) {
  case Sema::TDK_Success:
  case Sema::TDK_InstantiationDepth:
  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
  case Sema::TDK_SubstitutionFailure:
    return TemplateParameter();

  case Sema::TDK_Incomplete:
  case Sema::TDK_InvalidExplicitArguments:
    return TemplateParameter::getFromOpaqueValue(Data);

  case Sema::TDK_Inconsistent:
  case Sema::TDK_Underqualified:
    return static_cast<DFIParamWithArguments*>(Data)->Param;

  // Unhandled
  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    break;
  }

  return TemplateParameter();
}

TemplateArgumentList *
OverloadCandidate::DeductionFailureInfo::getTemplateArgumentList() {
  switch (static_cast<Sema::TemplateDeductionResult>(Result)) {
    case Sema::TDK_Success:
    case Sema::TDK_InstantiationDepth:
    case Sema::TDK_TooManyArguments:
    case Sema::TDK_TooFewArguments:
    case Sema::TDK_Incomplete:
    case Sema::TDK_InvalidExplicitArguments:
    case Sema::TDK_Inconsistent:
    case Sema::TDK_Underqualified:
      return 0;

    case Sema::TDK_SubstitutionFailure:
      return static_cast<TemplateArgumentList*>(Data);

    // Unhandled
    case Sema::TDK_NonDeducedMismatch:
    case Sema::TDK_FailedOverloadResolution:
      break;
  }

  return 0;
}

const TemplateArgument *OverloadCandidate::DeductionFailureInfo::getFirstArg() {
  switch (static_cast<Sema::TemplateDeductionResult>(Result)) {
  case Sema::TDK_Success:
  case Sema::TDK_InstantiationDepth:
  case Sema::TDK_Incomplete:
  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
  case Sema::TDK_InvalidExplicitArguments:
  case Sema::TDK_SubstitutionFailure:
    return 0;

  case Sema::TDK_Inconsistent:
  case Sema::TDK_Underqualified:
    return &static_cast<DFIParamWithArguments*>(Data)->FirstArg;

  // Unhandled
  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    break;
  }

  return 0;
}

const TemplateArgument *
OverloadCandidate::DeductionFailureInfo::getSecondArg() {
  switch (static_cast<Sema::TemplateDeductionResult>(Result)) {
  case Sema::TDK_Success:
  case Sema::TDK_InstantiationDepth:
  case Sema::TDK_Incomplete:
  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
  case Sema::TDK_InvalidExplicitArguments:
  case Sema::TDK_SubstitutionFailure:
    return 0;

  case Sema::TDK_Inconsistent:
  case Sema::TDK_Underqualified:
    return &static_cast<DFIParamWithArguments*>(Data)->SecondArg;

  // Unhandled
  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    break;
  }

  return 0;
}

void OverloadCandidateSet::clear() {
  inherited::clear();
  Functions.clear();
}

// IsOverload - Determine whether the given New declaration is an
// overload of the declarations in Old. This routine returns false if
// New and Old cannot be overloaded, e.g., if New has the same
// signature as some function in Old (C++ 1.3.10) or if the Old
// declarations aren't functions (or function templates) at all. When
// it does return false, MatchedDecl will point to the decl that New
// cannot be overloaded with.  This decl may be a UsingShadowDecl on
// top of the underlying declaration.
//
// Example: Given the following input:
//
//   void f(int, float); // #1
//   void f(int, int); // #2
//   int f(int, int); // #3
//
// When we process #1, there is no previous declaration of "f",
// so IsOverload will not be used.
//
// When we process #2, Old contains only the FunctionDecl for #1.  By
// comparing the parameter types, we see that #1 and #2 are overloaded
// (since they have different signatures), so this routine returns
// false; MatchedDecl is unchanged.
//
// When we process #3, Old is an overload set containing #1 and #2. We
// compare the signatures of #3 to #1 (they're overloaded, so we do
// nothing) and then #3 to #2. Since the signatures of #3 and #2 are
// identical (return types of functions are not part of the
// signature), IsOverload returns false and MatchedDecl will be set to
// point to the FunctionDecl for #2.
//
// 'NewIsUsingShadowDecl' indicates that 'New' is being introduced
// into a class by a using declaration.  The rules for whether to hide
// shadow declarations ignore some properties which otherwise figure
// into a function template's signature.
Sema::OverloadKind
Sema::CheckOverload(Scope *S, FunctionDecl *New, const LookupResult &Old,
                    NamedDecl *&Match, bool NewIsUsingDecl) {
  for (LookupResult::iterator I = Old.begin(), E = Old.end();
         I != E; ++I) {
    NamedDecl *OldD = *I;

    bool OldIsUsingDecl = false;
    if (isa<UsingShadowDecl>(OldD)) {
      OldIsUsingDecl = true;

      // We can always introduce two using declarations into the same
      // context, even if they have identical signatures.
      if (NewIsUsingDecl) continue;

      OldD = cast<UsingShadowDecl>(OldD)->getTargetDecl();
    }

    // If either declaration was introduced by a using declaration,
    // we'll need to use slightly different rules for matching.
    // Essentially, these rules are the normal rules, except that
    // function templates hide function templates with different
    // return types or template parameter lists.
    bool UseMemberUsingDeclRules =
      (OldIsUsingDecl || NewIsUsingDecl) && CurContext->isRecord();

    if (FunctionTemplateDecl *OldT = dyn_cast<FunctionTemplateDecl>(OldD)) {
      if (!IsOverload(New, OldT->getTemplatedDecl(), UseMemberUsingDeclRules)) {
        if (UseMemberUsingDeclRules && OldIsUsingDecl) {
          HideUsingShadowDecl(S, cast<UsingShadowDecl>(*I));
          continue;
        }

        Match = *I;
        return Ovl_Match;
      }
    } else if (FunctionDecl *OldF = dyn_cast<FunctionDecl>(OldD)) {
      if (!IsOverload(New, OldF, UseMemberUsingDeclRules)) {
        if (UseMemberUsingDeclRules && OldIsUsingDecl) {
          HideUsingShadowDecl(S, cast<UsingShadowDecl>(*I));
          continue;
        }

        Match = *I;
        return Ovl_Match;
      }
    } else if (isa<UsingDecl>(OldD)) {
      // We can overload with these, which can show up when doing
      // redeclaration checks for UsingDecls.
      assert(Old.getLookupKind() == LookupUsingDeclName);
    } else if (isa<TagDecl>(OldD)) {
      // We can always overload with tags by hiding them.
    } else if (isa<UnresolvedUsingValueDecl>(OldD)) {
      // Optimistically assume that an unresolved using decl will
      // overload; if it doesn't, we'll have to diagnose during
      // template instantiation.
    } else {
      // (C++ 13p1):
      //   Only function declarations can be overloaded; object and type
      //   declarations cannot be overloaded.
      Match = *I;
      return Ovl_NonFunction;
    }
  }

  return Ovl_Overload;
}

bool Sema::IsOverload(FunctionDecl *New, FunctionDecl *Old,
                      bool UseUsingDeclRules) {
  // If both of the functions are extern "C", then they are not
  // overloads.
  if (Old->isExternC() && New->isExternC())
    return false;

  FunctionTemplateDecl *OldTemplate = Old->getDescribedFunctionTemplate();
  FunctionTemplateDecl *NewTemplate = New->getDescribedFunctionTemplate();

  // C++ [temp.fct]p2:
  //   A function template can be overloaded with other function templates
  //   and with normal (non-template) functions.
  if ((OldTemplate == 0) != (NewTemplate == 0))
    return true;

  // Is the function New an overload of the function Old?
  QualType OldQType = Context.getCanonicalType(Old->getType());
  QualType NewQType = Context.getCanonicalType(New->getType());

  // Compare the signatures (C++ 1.3.10) of the two functions to
  // determine whether they are overloads. If we find any mismatch
  // in the signature, they are overloads.

  // If either of these functions is a K&R-style function (no
  // prototype), then we consider them to have matching signatures.
  if (isa<FunctionNoProtoType>(OldQType.getTypePtr()) ||
      isa<FunctionNoProtoType>(NewQType.getTypePtr()))
    return false;

  const FunctionProtoType* OldType = cast<FunctionProtoType>(OldQType);
  const FunctionProtoType* NewType = cast<FunctionProtoType>(NewQType);

  // The signature of a function includes the types of its
  // parameters (C++ 1.3.10), which includes the presence or absence
  // of the ellipsis; see C++ DR 357).
  if (OldQType != NewQType &&
      (OldType->getNumArgs() != NewType->getNumArgs() ||
       OldType->isVariadic() != NewType->isVariadic() ||
       !FunctionArgTypesAreEqual(OldType, NewType)))
    return true;

  // C++ [temp.over.link]p4:
  //   The signature of a function template consists of its function
  //   signature, its return type and its template parameter list. The names
  //   of the template parameters are significant only for establishing the
  //   relationship between the template parameters and the rest of the
  //   signature.
  //
  // We check the return type and template parameter lists for function
  // templates first; the remaining checks follow.
  //
  // However, we don't consider either of these when deciding whether
  // a member introduced by a shadow declaration is hidden.
  if (!UseUsingDeclRules && NewTemplate &&
      (!TemplateParameterListsAreEqual(NewTemplate->getTemplateParameters(),
                                       OldTemplate->getTemplateParameters(),
                                       false, TPL_TemplateMatch) ||
       OldType->getResultType() != NewType->getResultType()))
    return true;

  // If the function is a class member, its signature includes the
  // cv-qualifiers (if any) and ref-qualifier (if any) on the function itself.
  //
  // As part of this, also check whether one of the member functions
  // is static, in which case they are not overloads (C++
  // 13.1p2). While not part of the definition of the signature,
  // this check is important to determine whether these functions
  // can be overloaded.
  CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
  CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
  if (OldMethod && NewMethod &&
      !OldMethod->isStatic() && !NewMethod->isStatic() &&
      (OldMethod->getTypeQualifiers() != NewMethod->getTypeQualifiers() ||
       OldMethod->getRefQualifier() != NewMethod->getRefQualifier())) {
    if (!UseUsingDeclRules &&
        OldMethod->getRefQualifier() != NewMethod->getRefQualifier() &&
        (OldMethod->getRefQualifier() == RQ_None ||
         NewMethod->getRefQualifier() == RQ_None)) {
      // C++0x [over.load]p2:
      //   - Member function declarations with the same name and the same
      //     parameter-type-list as well as member function template
      //     declarations with the same name, the same parameter-type-list, and
      //     the same template parameter lists cannot be overloaded if any of
      //     them, but not all, have a ref-qualifier (8.3.5).
      Diag(NewMethod->getLocation(), diag::err_ref_qualifier_overload)
        << NewMethod->getRefQualifier() << OldMethod->getRefQualifier();
      Diag(OldMethod->getLocation(), diag::note_previous_declaration);
    }

    return true;
  }

  // The signatures match; this is not an overload.
  return false;
}

/// \brief Checks availability of the function depending on the current
/// function context. Inside an unavailable function, unavailability is ignored.
///
/// \returns true if \arg FD is unavailable and current context is inside
/// an available function, false otherwise.
bool Sema::isFunctionConsideredUnavailable(FunctionDecl *FD) {
  return FD->isUnavailable() && !cast<Decl>(CurContext)->isUnavailable();
}

/// TryImplicitConversion - Attempt to perform an implicit conversion
/// from the given expression (Expr) to the given type (ToType). This
/// function returns an implicit conversion sequence that can be used
/// to perform the initialization. Given
///
///   void f(float f);
///   void g(int i) { f(i); }
///
/// this routine would produce an implicit conversion sequence to
/// describe the initialization of f from i, which will be a standard
/// conversion sequence containing an lvalue-to-rvalue conversion (C++
/// 4.1) followed by a floating-integral conversion (C++ 4.9).
//
/// Note that this routine only determines how the conversion can be
/// performed; it does not actually perform the conversion. As such,
/// it will not produce any diagnostics if no conversion is available,
/// but will instead return an implicit conversion sequence of kind
/// "BadConversion".
///
/// If @p SuppressUserConversions, then user-defined conversions are
/// not permitted.
/// If @p AllowExplicit, then explicit user-defined conversions are
/// permitted.
///
/// \param AllowObjCWritebackConversion Whether we allow the Objective-C
/// writeback conversion, which allows __autoreleasing id* parameters to
/// be initialized with __strong id* or __weak id* arguments.
static ImplicitConversionSequence
TryImplicitConversion(Sema &S, Expr *From, QualType ToType,
                      bool SuppressUserConversions,
                      bool AllowExplicit,
                      bool InOverloadResolution,
                      bool CStyle,
                      bool AllowObjCWritebackConversion) {
  ImplicitConversionSequence ICS;
  if (IsStandardConversion(S, From, ToType, InOverloadResolution,
                           ICS.Standard, CStyle, AllowObjCWritebackConversion)){
    ICS.setStandard();
    return ICS;
  }

  if (!S.getLangOptions().CPlusPlus) {
    ICS.setBad(BadConversionSequence::no_conversion, From, ToType);
    return ICS;
  }

  // C++ [over.ics.user]p4:
  //   A conversion of an expression of class type to the same class
  //   type is given Exact Match rank, and a conversion of an
  //   expression of class type to a base class of that type is
  //   given Conversion rank, in spite of the fact that a copy/move
  //   constructor (i.e., a user-defined conversion function) is
  //   called for those cases.
  QualType FromType = From->getType();
  if (ToType->getAs<RecordType>() && FromType->getAs<RecordType>() &&
      (S.Context.hasSameUnqualifiedType(FromType, ToType) ||
       S.IsDerivedFrom(FromType, ToType))) {
    ICS.setStandard();
    ICS.Standard.setAsIdentityConversion();
    ICS.Standard.setFromType(FromType);
    ICS.Standard.setAllToTypes(ToType);

    // We don't actually check at this point whether there is a valid
    // copy/move constructor, since overloading just assumes that it
    // exists. When we actually perform initialization, we'll find the
    // appropriate constructor to copy the returned object, if needed.
    ICS.Standard.CopyConstructor = 0;

    // Determine whether this is considered a derived-to-base conversion.
    if (!S.Context.hasSameUnqualifiedType(FromType, ToType))
      ICS.Standard.Second = ICK_Derived_To_Base;

    return ICS;
  }

  if (SuppressUserConversions) {
    // We're not in the case above, so there is no conversion that
    // we can perform.
    ICS.setBad(BadConversionSequence::no_conversion, From, ToType);
    return ICS;
  }

  // Attempt user-defined conversion.
  OverloadCandidateSet Conversions(From->getExprLoc());
  OverloadingResult UserDefResult
    = IsUserDefinedConversion(S, From, ToType, ICS.UserDefined, Conversions,
                              AllowExplicit);

  if (UserDefResult == OR_Success) {
    ICS.setUserDefined();
    // C++ [over.ics.user]p4:
    //   A conversion of an expression of class type to the same class
    //   type is given Exact Match rank, and a conversion of an
    //   expression of class type to a base class of that type is
    //   given Conversion rank, in spite of the fact that a copy
    //   constructor (i.e., a user-defined conversion function) is
    //   called for those cases.
    if (CXXConstructorDecl *Constructor
          = dyn_cast<CXXConstructorDecl>(ICS.UserDefined.ConversionFunction)) {
      QualType FromCanon
        = S.Context.getCanonicalType(From->getType().getUnqualifiedType());
      QualType ToCanon
        = S.Context.getCanonicalType(ToType).getUnqualifiedType();
      if (Constructor->isCopyConstructor() &&
          (FromCanon == ToCanon || S.IsDerivedFrom(FromCanon, ToCanon))) {
        // Turn this into a "standard" conversion sequence, so that it
        // gets ranked with standard conversion sequences.
        ICS.setStandard();
        ICS.Standard.setAsIdentityConversion();
        ICS.Standard.setFromType(From->getType());
        ICS.Standard.setAllToTypes(ToType);
        ICS.Standard.CopyConstructor = Constructor;
        if (ToCanon != FromCanon)
          ICS.Standard.Second = ICK_Derived_To_Base;
      }
    }

    // C++ [over.best.ics]p4:
    //   However, when considering the argument of a user-defined
    //   conversion function that is a candidate by 13.3.1.3 when
    //   invoked for the copying of the temporary in the second step
    //   of a class copy-initialization, or by 13.3.1.4, 13.3.1.5, or
    //   13.3.1.6 in all cases, only standard conversion sequences and
    //   ellipsis conversion sequences are allowed.
    if (SuppressUserConversions && ICS.isUserDefined()) {
      ICS.setBad(BadConversionSequence::suppressed_user, From, ToType);
    }
  } else if (UserDefResult == OR_Ambiguous && !SuppressUserConversions) {
    ICS.setAmbiguous();
    ICS.Ambiguous.setFromType(From->getType());
    ICS.Ambiguous.setToType(ToType);
    for (OverloadCandidateSet::iterator Cand = Conversions.begin();
         Cand != Conversions.end(); ++Cand)
      if (Cand->Viable)
        ICS.Ambiguous.addConversion(Cand->Function);
  } else {
    ICS.setBad(BadConversionSequence::no_conversion, From, ToType);
  }

  return ICS;
}

ImplicitConversionSequence
Sema::TryImplicitConversion(Expr *From, QualType ToType,
                            bool SuppressUserConversions,
                            bool AllowExplicit,
                            bool InOverloadResolution,
                            bool CStyle,
                            bool AllowObjCWritebackConversion) {
  return clang::TryImplicitConversion(*this, From, ToType, 
                                      SuppressUserConversions, AllowExplicit,
                                      InOverloadResolution, CStyle, 
                                      AllowObjCWritebackConversion);
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType. Returns the
/// converted expression. Flavor is the kind of conversion we're
/// performing, used in the error message. If @p AllowExplicit,
/// explicit user-defined conversions are permitted.
ExprResult
Sema::PerformImplicitConversion(Expr *From, QualType ToType,
                                AssignmentAction Action, bool AllowExplicit) {
  ImplicitConversionSequence ICS;
  return PerformImplicitConversion(From, ToType, Action, AllowExplicit, ICS);
}

ExprResult
Sema::PerformImplicitConversion(Expr *From, QualType ToType,
                                AssignmentAction Action, bool AllowExplicit,
                                ImplicitConversionSequence& ICS) {
  // Objective-C ARC: Determine whether we will allow the writeback conversion.
  bool AllowObjCWritebackConversion
    = getLangOptions().ObjCAutoRefCount && 
      (Action == AA_Passing || Action == AA_Sending);
  

  ICS = clang::TryImplicitConversion(*this, From, ToType,
                                     /*SuppressUserConversions=*/false,
                                     AllowExplicit,
                                     /*InOverloadResolution=*/false,
                                     /*CStyle=*/false,
                                     AllowObjCWritebackConversion);
  return PerformImplicitConversion(From, ToType, ICS, Action);
}

/// \brief Determine whether the conversion from FromType to ToType is a valid
/// conversion that strips "noreturn" off the nested function type.
bool Sema::IsNoReturnConversion(QualType FromType, QualType ToType,
                                QualType &ResultTy) {
  if (Context.hasSameUnqualifiedType(FromType, ToType))
    return false;

  // Permit the conversion F(t __attribute__((noreturn))) -> F(t)
  // where F adds one of the following at most once:
  //   - a pointer
  //   - a member pointer
  //   - a block pointer
  CanQualType CanTo = Context.getCanonicalType(ToType);
  CanQualType CanFrom = Context.getCanonicalType(FromType);
  Type::TypeClass TyClass = CanTo->getTypeClass();
  if (TyClass != CanFrom->getTypeClass()) return false;
  if (TyClass != Type::FunctionProto && TyClass != Type::FunctionNoProto) {
    if (TyClass == Type::Pointer) {
      CanTo = CanTo.getAs<PointerType>()->getPointeeType();
      CanFrom = CanFrom.getAs<PointerType>()->getPointeeType();
    } else if (TyClass == Type::BlockPointer) {
      CanTo = CanTo.getAs<BlockPointerType>()->getPointeeType();
      CanFrom = CanFrom.getAs<BlockPointerType>()->getPointeeType();
    } else if (TyClass == Type::MemberPointer) {
      CanTo = CanTo.getAs<MemberPointerType>()->getPointeeType();
      CanFrom = CanFrom.getAs<MemberPointerType>()->getPointeeType();
    } else {
      return false;
    }

    TyClass = CanTo->getTypeClass();
    if (TyClass != CanFrom->getTypeClass()) return false;
    if (TyClass != Type::FunctionProto && TyClass != Type::FunctionNoProto)
      return false;
  }

  const FunctionType *FromFn = cast<FunctionType>(CanFrom);
  FunctionType::ExtInfo EInfo = FromFn->getExtInfo();
  if (!EInfo.getNoReturn()) return false;

  FromFn = Context.adjustFunctionType(FromFn, EInfo.withNoReturn(false));
  assert(QualType(FromFn, 0).isCanonical());
  if (QualType(FromFn, 0) != CanTo) return false;

  ResultTy = ToType;
  return true;
}

/// \brief Determine whether the conversion from FromType to ToType is a valid
/// vector conversion.
///
/// \param ICK Will be set to the vector conversion kind, if this is a vector
/// conversion.
static bool IsVectorConversion(ASTContext &Context, QualType FromType,
                               QualType ToType, ImplicitConversionKind &ICK) {
  // We need at least one of these types to be a vector type to have a vector
  // conversion.
  if (!ToType->isVectorType() && !FromType->isVectorType())
    return false;

  // Identical types require no conversions.
  if (Context.hasSameUnqualifiedType(FromType, ToType))
    return false;

  // There are no conversions between extended vector types, only identity.
  if (ToType->isExtVectorType()) {
    // There are no conversions between extended vector types other than the
    // identity conversion.
    if (FromType->isExtVectorType())
      return false;

    // Vector splat from any arithmetic type to a vector.
    if (FromType->isArithmeticType()) {
      ICK = ICK_Vector_Splat;
      return true;
    }
  }

  // We can perform the conversion between vector types in the following cases:
  // 1)vector types are equivalent AltiVec and GCC vector types
  // 2)lax vector conversions are permitted and the vector types are of the
  //   same size
  if (ToType->isVectorType() && FromType->isVectorType()) {
    if (Context.areCompatibleVectorTypes(FromType, ToType) ||
        (Context.getLangOptions().LaxVectorConversions &&
         (Context.getTypeSize(FromType) == Context.getTypeSize(ToType)))) {
      ICK = ICK_Vector_Conversion;
      return true;
    }
  }

  return false;
}

/// IsStandardConversion - Determines whether there is a standard
/// conversion sequence (C++ [conv], C++ [over.ics.scs]) from the
/// expression From to the type ToType. Standard conversion sequences
/// only consider non-class types; for conversions that involve class
/// types, use TryImplicitConversion. If a conversion exists, SCS will
/// contain the standard conversion sequence required to perform this
/// conversion and this routine will return true. Otherwise, this
/// routine will return false and the value of SCS is unspecified.
static bool IsStandardConversion(Sema &S, Expr* From, QualType ToType,
                                 bool InOverloadResolution,
                                 StandardConversionSequence &SCS,
                                 bool CStyle,
                                 bool AllowObjCWritebackConversion) {
  QualType FromType = From->getType();

  // Standard conversions (C++ [conv])
  SCS.setAsIdentityConversion();
  SCS.DeprecatedStringLiteralToCharPtr = false;
  SCS.IncompatibleObjC = false;
  SCS.setFromType(FromType);
  SCS.CopyConstructor = 0;

  // There are no standard conversions for class types in C++, so
  // abort early. When overloading in C, however, we do permit
  if (FromType->isRecordType() || ToType->isRecordType()) {
    if (S.getLangOptions().CPlusPlus)
      return false;

    // When we're overloading in C, we allow, as standard conversions,
  }

  // The first conversion can be an lvalue-to-rvalue conversion,
  // array-to-pointer conversion, or function-to-pointer conversion
  // (C++ 4p1).

  if (FromType == S.Context.OverloadTy) {
    DeclAccessPair AccessPair;
    if (FunctionDecl *Fn
          = S.ResolveAddressOfOverloadedFunction(From, ToType, false,
                                                 AccessPair)) {
      // We were able to resolve the address of the overloaded function,
      // so we can convert to the type of that function.
      FromType = Fn->getType();

      // we can sometimes resolve &foo<int> regardless of ToType, so check
      // if the type matches (identity) or we are converting to bool
      if (!S.Context.hasSameUnqualifiedType(
                      S.ExtractUnqualifiedFunctionType(ToType), FromType)) {
        QualType resultTy;
        // if the function type matches except for [[noreturn]], it's ok
        if (!S.IsNoReturnConversion(FromType,
              S.ExtractUnqualifiedFunctionType(ToType), resultTy))
          // otherwise, only a boolean conversion is standard   
          if (!ToType->isBooleanType()) 
            return false; 
      }

      // Check if the "from" expression is taking the address of an overloaded
      // function and recompute the FromType accordingly. Take advantage of the
      // fact that non-static member functions *must* have such an address-of
      // expression. 
      CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn);
      if (Method && !Method->isStatic()) {
        assert(isa<UnaryOperator>(From->IgnoreParens()) &&
               "Non-unary operator on non-static member address");
        assert(cast<UnaryOperator>(From->IgnoreParens())->getOpcode()
               == UO_AddrOf &&
               "Non-address-of operator on non-static member address");
        const Type *ClassType
          = S.Context.getTypeDeclType(Method->getParent()).getTypePtr();
        FromType = S.Context.getMemberPointerType(FromType, ClassType);
      } else if (isa<UnaryOperator>(From->IgnoreParens())) {
        assert(cast<UnaryOperator>(From->IgnoreParens())->getOpcode() ==
               UO_AddrOf &&
               "Non-address-of operator for overloaded function expression");
        FromType = S.Context.getPointerType(FromType);
      }

      // Check that we've computed the proper type after overload resolution.
      assert(S.Context.hasSameType(
        FromType,
        S.FixOverloadedFunctionReference(From, AccessPair, Fn)->getType()));
    } else {
      return false;
    }
  }
  // Lvalue-to-rvalue conversion (C++ 4.1):
  //   An lvalue (3.10) of a non-function, non-array type T can be
  //   converted to an rvalue.
  bool argIsLValue = From->isLValue();
  if (argIsLValue &&
      !FromType->isFunctionType() && !FromType->isArrayType() &&
      S.Context.getCanonicalType(FromType) != S.Context.OverloadTy) {
    SCS.First = ICK_Lvalue_To_Rvalue;

    // If T is a non-class type, the type of the rvalue is the
    // cv-unqualified version of T. Otherwise, the type of the rvalue
    // is T (C++ 4.1p1). C++ can't get here with class types; in C, we
    // just strip the qualifiers because they don't matter.
    FromType = FromType.getUnqualifiedType();
  } else if (FromType->isArrayType()) {
    // Array-to-pointer conversion (C++ 4.2)
    SCS.First = ICK_Array_To_Pointer;

    // An lvalue or rvalue of type "array of N T" or "array of unknown
    // bound of T" can be converted to an rvalue of type "pointer to
    // T" (C++ 4.2p1).
    FromType = S.Context.getArrayDecayedType(FromType);

    if (S.IsStringLiteralToNonConstPointerConversion(From, ToType)) {
      // This conversion is deprecated. (C++ D.4).
      SCS.DeprecatedStringLiteralToCharPtr = true;

      // For the purpose of ranking in overload resolution
      // (13.3.3.1.1), this conversion is considered an
      // array-to-pointer conversion followed by a qualification
      // conversion (4.4). (C++ 4.2p2)
      SCS.Second = ICK_Identity;
      SCS.Third = ICK_Qualification;
      SCS.QualificationIncludesObjCLifetime = false;
      SCS.setAllToTypes(FromType);
      return true;
    }
  } else if (FromType->isFunctionType() && argIsLValue) {
    // Function-to-pointer conversion (C++ 4.3).
    SCS.First = ICK_Function_To_Pointer;

    // An lvalue of function type T can be converted to an rvalue of
    // type "pointer to T." The result is a pointer to the
    // function. (C++ 4.3p1).
    FromType = S.Context.getPointerType(FromType);
  } else {
    // We don't require any conversions for the first step.
    SCS.First = ICK_Identity;
  }
  SCS.setToType(0, FromType);

  // The second conversion can be an integral promotion, floating
  // point promotion, integral conversion, floating point conversion,
  // floating-integral conversion, pointer conversion,
  // pointer-to-member conversion, or boolean conversion (C++ 4p1).
  // For overloading in C, this can also be a "compatible-type"
  // conversion.
  bool IncompatibleObjC = false;
  ImplicitConversionKind SecondICK = ICK_Identity;
  if (S.Context.hasSameUnqualifiedType(FromType, ToType)) {
    // The unqualified versions of the types are the same: there's no
    // conversion to do.
    SCS.Second = ICK_Identity;
  } else if (S.IsIntegralPromotion(From, FromType, ToType)) {
    // Integral promotion (C++ 4.5).
    SCS.Second = ICK_Integral_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if (S.IsFloatingPointPromotion(FromType, ToType)) {
    // Floating point promotion (C++ 4.6).
    SCS.Second = ICK_Floating_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if (S.IsComplexPromotion(FromType, ToType)) {
    // Complex promotion (Clang extension)
    SCS.Second = ICK_Complex_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if (ToType->isBooleanType() &&
             (FromType->isArithmeticType() ||
              FromType->isAnyPointerType() ||
              FromType->isBlockPointerType() ||
              FromType->isMemberPointerType() ||
              FromType->isNullPtrType())) {
    // Boolean conversions (C++ 4.12).
    SCS.Second = ICK_Boolean_Conversion;
    FromType = S.Context.BoolTy;
  } else if (FromType->isIntegralOrUnscopedEnumerationType() &&
             ToType->isIntegralType(S.Context)) {
    // Integral conversions (C++ 4.7).
    SCS.Second = ICK_Integral_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if (FromType->isAnyComplexType() && ToType->isComplexType()) {
    // Complex conversions (C99 6.3.1.6)
    SCS.Second = ICK_Complex_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if ((FromType->isAnyComplexType() && ToType->isArithmeticType()) ||
             (ToType->isAnyComplexType() && FromType->isArithmeticType())) {
    // Complex-real conversions (C99 6.3.1.7)
    SCS.Second = ICK_Complex_Real;
    FromType = ToType.getUnqualifiedType();
  } else if (FromType->isRealFloatingType() && ToType->isRealFloatingType()) {
    // Floating point conversions (C++ 4.8).
    SCS.Second = ICK_Floating_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if ((FromType->isRealFloatingType() &&
              ToType->isIntegralType(S.Context)) ||
             (FromType->isIntegralOrUnscopedEnumerationType() &&
              ToType->isRealFloatingType())) {
    // Floating-integral conversions (C++ 4.9).
    SCS.Second = ICK_Floating_Integral;
    FromType = ToType.getUnqualifiedType();
  } else if (S.IsBlockPointerConversion(FromType, ToType, FromType)) {
    SCS.Second = ICK_Block_Pointer_Conversion;
  } else if (AllowObjCWritebackConversion &&
             S.isObjCWritebackConversion(FromType, ToType, FromType)) {
    SCS.Second = ICK_Writeback_Conversion;
  } else if (S.IsPointerConversion(From, FromType, ToType, InOverloadResolution,
                                   FromType, IncompatibleObjC)) {
    // Pointer conversions (C++ 4.10).
    SCS.Second = ICK_Pointer_Conversion;
    SCS.IncompatibleObjC = IncompatibleObjC;
    FromType = FromType.getUnqualifiedType();
  } else if (S.IsMemberPointerConversion(From, FromType, ToType,
                                         InOverloadResolution, FromType)) {
    // Pointer to member conversions (4.11).
    SCS.Second = ICK_Pointer_Member;
  } else if (IsVectorConversion(S.Context, FromType, ToType, SecondICK)) {
    SCS.Second = SecondICK;
    FromType = ToType.getUnqualifiedType();
  } else if (!S.getLangOptions().CPlusPlus &&
             S.Context.typesAreCompatible(ToType, FromType)) {
    // Compatible conversions (Clang extension for C function overloading)
    SCS.Second = ICK_Compatible_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if (S.IsNoReturnConversion(FromType, ToType, FromType)) {
    // Treat a conversion that strips "noreturn" as an identity conversion.
    SCS.Second = ICK_NoReturn_Adjustment;
  } else if (IsTransparentUnionStandardConversion(S, From, ToType,
                                             InOverloadResolution,
                                             SCS, CStyle)) {
    SCS.Second = ICK_TransparentUnionConversion;
    FromType = ToType;
  } else {
    // No second conversion required.
    SCS.Second = ICK_Identity;
  }
  SCS.setToType(1, FromType);

  QualType CanonFrom;
  QualType CanonTo;
  // The third conversion can be a qualification conversion (C++ 4p1).
  bool ObjCLifetimeConversion;
  if (S.IsQualificationConversion(FromType, ToType, CStyle, 
                                  ObjCLifetimeConversion)) {
    SCS.Third = ICK_Qualification;
    SCS.QualificationIncludesObjCLifetime = ObjCLifetimeConversion;
    FromType = ToType;
    CanonFrom = S.Context.getCanonicalType(FromType);
    CanonTo = S.Context.getCanonicalType(ToType);
  } else {
    // No conversion required
    SCS.Third = ICK_Identity;

    // C++ [over.best.ics]p6:
    //   [...] Any difference in top-level cv-qualification is
    //   subsumed by the initialization itself and does not constitute
    //   a conversion. [...]
    CanonFrom = S.Context.getCanonicalType(FromType);
    CanonTo = S.Context.getCanonicalType(ToType);
    if (CanonFrom.getLocalUnqualifiedType()
                                       == CanonTo.getLocalUnqualifiedType() &&
        (CanonFrom.getLocalCVRQualifiers() != CanonTo.getLocalCVRQualifiers()
         || CanonFrom.getObjCGCAttr() != CanonTo.getObjCGCAttr()
         || CanonFrom.getObjCLifetime() != CanonTo.getObjCLifetime())) {
      FromType = ToType;
      CanonFrom = CanonTo;
    }
  }
  SCS.setToType(2, FromType);

  // If we have not converted the argument type to the parameter type,
  // this is a bad conversion sequence.
  if (CanonFrom != CanonTo)
    return false;

  return true;
}
  
static bool
IsTransparentUnionStandardConversion(Sema &S, Expr* From, 
                                     QualType &ToType,
                                     bool InOverloadResolution,
                                     StandardConversionSequence &SCS,
                                     bool CStyle) {
    
  const RecordType *UT = ToType->getAsUnionType();
  if (!UT || !UT->getDecl()->hasAttr<TransparentUnionAttr>())
    return false;
  // The field to initialize within the transparent union.
  RecordDecl *UD = UT->getDecl();
  // It's compatible if the expression matches any of the fields.
  for (RecordDecl::field_iterator it = UD->field_begin(),
       itend = UD->field_end();
       it != itend; ++it) {
    if (IsStandardConversion(S, From, it->getType(), InOverloadResolution, SCS,
                             CStyle, /*ObjCWritebackConversion=*/false)) {
      ToType = it->getType();
      return true;
    }
  }
  return false;
}

/// IsIntegralPromotion - Determines whether the conversion from the
/// expression From (whose potentially-adjusted type is FromType) to
/// ToType is an integral promotion (C++ 4.5). If so, returns true and
/// sets PromotedType to the promoted type.
bool Sema::IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType) {
  const BuiltinType *To = ToType->getAs<BuiltinType>();
  // All integers are built-in.
  if (!To) {
    return false;
  }

  // An rvalue of type char, signed char, unsigned char, short int, or
  // unsigned short int can be converted to an rvalue of type int if
  // int can represent all the values of the source type; otherwise,
  // the source rvalue can be converted to an rvalue of type unsigned
  // int (C++ 4.5p1).
  if (FromType->isPromotableIntegerType() && !FromType->isBooleanType() &&
      !FromType->isEnumeralType()) {
    if (// We can promote any signed, promotable integer type to an int
        (FromType->isSignedIntegerType() ||
         // We can promote any unsigned integer type whose size is
         // less than int to an int.
         (!FromType->isSignedIntegerType() &&
          Context.getTypeSize(FromType) < Context.getTypeSize(ToType)))) {
      return To->getKind() == BuiltinType::Int;
    }

    return To->getKind() == BuiltinType::UInt;
  }

  // C++0x [conv.prom]p3:
  //   A prvalue of an unscoped enumeration type whose underlying type is not
  //   fixed (7.2) can be converted to an rvalue a prvalue of the first of the
  //   following types that can represent all the values of the enumeration
  //   (i.e., the values in the range bmin to bmax as described in 7.2): int,
  //   unsigned int, long int, unsigned long int, long long int, or unsigned
  //   long long int. If none of the types in that list can represent all the
  //   values of the enumeration, an rvalue a prvalue of an unscoped enumeration
  //   type can be converted to an rvalue a prvalue of the extended integer type
  //   with lowest integer conversion rank (4.13) greater than the rank of long
  //   long in which all the values of the enumeration can be represented. If
  //   there are two such extended types, the signed one is chosen.
  if (const EnumType *FromEnumType = FromType->getAs<EnumType>()) {
    // C++0x 7.2p9: Note that this implicit enum to int conversion is not
    // provided for a scoped enumeration.
    if (FromEnumType->getDecl()->isScoped())
      return false;

    // We have already pre-calculated the promotion type, so this is trivial.
    if (ToType->isIntegerType() &&
        !RequireCompleteType(From->getLocStart(), FromType, PDiag()))
      return Context.hasSameUnqualifiedType(ToType,
                                FromEnumType->getDecl()->getPromotionType());
  }

  // C++0x [conv.prom]p2:
  //   A prvalue of type char16_t, char32_t, or wchar_t (3.9.1) can be converted
  //   to an rvalue a prvalue of the first of the following types that can
  //   represent all the values of its underlying type: int, unsigned int,
  //   long int, unsigned long int, long long int, or unsigned long long int.
  //   If none of the types in that list can represent all the values of its
  //   underlying type, an rvalue a prvalue of type char16_t, char32_t,
  //   or wchar_t can be converted to an rvalue a prvalue of its underlying
  //   type.
  if (FromType->isAnyCharacterType() && !FromType->isCharType() &&
      ToType->isIntegerType()) {
    // Determine whether the type we're converting from is signed or
    // unsigned.
    bool FromIsSigned;
    uint64_t FromSize = Context.getTypeSize(FromType);

    // FIXME: Is wchar_t signed or unsigned? We assume it's signed for now.
    FromIsSigned = true;

    // The types we'll try to promote to, in the appropriate
    // order. Try each of these types.
    QualType PromoteTypes[6] = {
      Context.IntTy, Context.UnsignedIntTy,
      Context.LongTy, Context.UnsignedLongTy ,
      Context.LongLongTy, Context.UnsignedLongLongTy
    };
    for (int Idx = 0; Idx < 6; ++Idx) {
      uint64_t ToSize = Context.getTypeSize(PromoteTypes[Idx]);
      if (FromSize < ToSize ||
          (FromSize == ToSize &&
           FromIsSigned == PromoteTypes[Idx]->isSignedIntegerType())) {
        // We found the type that we can promote to. If this is the
        // type we wanted, we have a promotion. Otherwise, no
        // promotion.
        return Context.hasSameUnqualifiedType(ToType, PromoteTypes[Idx]);
      }
    }
  }

  // An rvalue for an integral bit-field (9.6) can be converted to an
  // rvalue of type int if int can represent all the values of the
  // bit-field; otherwise, it can be converted to unsigned int if
  // unsigned int can represent all the values of the bit-field. If
  // the bit-field is larger yet, no integral promotion applies to
  // it. If the bit-field has an enumerated type, it is treated as any
  // other value of that type for promotion purposes (C++ 4.5p3).
  // FIXME: We should delay checking of bit-fields until we actually perform the
  // conversion.
  using llvm::APSInt;
  if (From)
    if (FieldDecl *MemberDecl = From->getBitField()) {
      APSInt BitWidth;
      if (FromType->isIntegralType(Context) &&
          MemberDecl->getBitWidth()->isIntegerConstantExpr(BitWidth, Context)) {
        APSInt ToSize(BitWidth.getBitWidth(), BitWidth.isUnsigned());
        ToSize = Context.getTypeSize(ToType);

        // Are we promoting to an int from a bitfield that fits in an int?
        if (BitWidth < ToSize ||
            (FromType->isSignedIntegerType() && BitWidth <= ToSize)) {
          return To->getKind() == BuiltinType::Int;
        }

        // Are we promoting to an unsigned int from an unsigned bitfield
        // that fits into an unsigned int?
        if (FromType->isUnsignedIntegerType() && BitWidth <= ToSize) {
          return To->getKind() == BuiltinType::UInt;
        }

        return false;
      }
    }

  // An rvalue of type bool can be converted to an rvalue of type int,
  // with false becoming zero and true becoming one (C++ 4.5p4).
  if (FromType->isBooleanType() && To->getKind() == BuiltinType::Int) {
    return true;
  }

  return false;
}

/// IsFloatingPointPromotion - Determines whether the conversion from
/// FromType to ToType is a floating point promotion (C++ 4.6). If so,
/// returns true and sets PromotedType to the promoted type.
bool Sema::IsFloatingPointPromotion(QualType FromType, QualType ToType) {
  /// An rvalue of type float can be converted to an rvalue of type
  /// double. (C++ 4.6p1).
  if (const BuiltinType *FromBuiltin = FromType->getAs<BuiltinType>())
    if (const BuiltinType *ToBuiltin = ToType->getAs<BuiltinType>()) {
      if (FromBuiltin->getKind() == BuiltinType::Float &&
          ToBuiltin->getKind() == BuiltinType::Double)
        return true;

      // C99 6.3.1.5p1:
      //   When a float is promoted to double or long double, or a
      //   double is promoted to long double [...].
      if (!getLangOptions().CPlusPlus &&
          (FromBuiltin->getKind() == BuiltinType::Float ||
           FromBuiltin->getKind() == BuiltinType::Double) &&
          (ToBuiltin->getKind() == BuiltinType::LongDouble))
        return true;
    }

  return false;
}

/// \brief Determine if a conversion is a complex promotion.
///
/// A complex promotion is defined as a complex -> complex conversion
/// where the conversion between the underlying real types is a
/// floating-point or integral promotion.
bool Sema::IsComplexPromotion(QualType FromType, QualType ToType) {
  const ComplexType *FromComplex = FromType->getAs<ComplexType>();
  if (!FromComplex)
    return false;

  const ComplexType *ToComplex = ToType->getAs<ComplexType>();
  if (!ToComplex)
    return false;

  return IsFloatingPointPromotion(FromComplex->getElementType(),
                                  ToComplex->getElementType()) ||
    IsIntegralPromotion(0, FromComplex->getElementType(),
                        ToComplex->getElementType());
}

/// BuildSimilarlyQualifiedPointerType - In a pointer conversion from
/// the pointer type FromPtr to a pointer to type ToPointee, with the
/// same type qualifiers as FromPtr has on its pointee type. ToType,
/// if non-empty, will be a pointer to ToType that may or may not have
/// the right set of qualifiers on its pointee.
///
static QualType
BuildSimilarlyQualifiedPointerType(const Type *FromPtr,
                                   QualType ToPointee, QualType ToType,
                                   ASTContext &Context,
                                   bool StripObjCLifetime = false) {
  assert((FromPtr->getTypeClass() == Type::Pointer ||
          FromPtr->getTypeClass() == Type::ObjCObjectPointer) &&
         "Invalid similarly-qualified pointer type");

  /// Conversions to 'id' subsume cv-qualifier conversions.
  if (ToType->isObjCIdType() || ToType->isObjCQualifiedIdType()) 
    return ToType.getUnqualifiedType();

  QualType CanonFromPointee
    = Context.getCanonicalType(FromPtr->getPointeeType());
  QualType CanonToPointee = Context.getCanonicalType(ToPointee);
  Qualifiers Quals = CanonFromPointee.getQualifiers();

  if (StripObjCLifetime)
    Quals.removeObjCLifetime();
  
  // Exact qualifier match -> return the pointer type we're converting to.
  if (CanonToPointee.getLocalQualifiers() == Quals) {
    // ToType is exactly what we need. Return it.
    if (!ToType.isNull())
      return ToType.getUnqualifiedType();

    // Build a pointer to ToPointee. It has the right qualifiers
    // already.
    if (isa<ObjCObjectPointerType>(ToType))
      return Context.getObjCObjectPointerType(ToPointee);
    return Context.getPointerType(ToPointee);
  }

  // Just build a canonical type that has the right qualifiers.
  QualType QualifiedCanonToPointee
    = Context.getQualifiedType(CanonToPointee.getLocalUnqualifiedType(), Quals);

  if (isa<ObjCObjectPointerType>(ToType))
    return Context.getObjCObjectPointerType(QualifiedCanonToPointee);
  return Context.getPointerType(QualifiedCanonToPointee);
}

static bool isNullPointerConstantForConversion(Expr *Expr,
                                               bool InOverloadResolution,
                                               ASTContext &Context) {
  // Handle value-dependent integral null pointer constants correctly.
  // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#903
  if (Expr->isValueDependent() && !Expr->isTypeDependent() &&
      Expr->getType()->isIntegerType() && !Expr->getType()->isEnumeralType())
    return !InOverloadResolution;

  return Expr->isNullPointerConstant(Context,
                    InOverloadResolution? Expr::NPC_ValueDependentIsNotNull
                                        : Expr::NPC_ValueDependentIsNull);
}

/// IsPointerConversion - Determines whether the conversion of the
/// expression From, which has the (possibly adjusted) type FromType,
/// can be converted to the type ToType via a pointer conversion (C++
/// 4.10). If so, returns true and places the converted type (that
/// might differ from ToType in its cv-qualifiers at some level) into
/// ConvertedType.
///
/// This routine also supports conversions to and from block pointers
/// and conversions with Objective-C's 'id', 'id<protocols...>', and
/// pointers to interfaces. FIXME: Once we've determined the
/// appropriate overloading rules for Objective-C, we may want to
/// split the Objective-C checks into a different routine; however,
/// GCC seems to consider all of these conversions to be pointer
/// conversions, so for now they live here. IncompatibleObjC will be
/// set if the conversion is an allowed Objective-C conversion that
/// should result in a warning.
bool Sema::IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                               bool InOverloadResolution,
                               QualType& ConvertedType,
                               bool &IncompatibleObjC) {
  IncompatibleObjC = false;
  if (isObjCPointerConversion(FromType, ToType, ConvertedType,
                              IncompatibleObjC))
    return true;

  // Conversion from a null pointer constant to any Objective-C pointer type.
  if (ToType->isObjCObjectPointerType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // Blocks: Block pointers can be converted to void*.
  if (FromType->isBlockPointerType() && ToType->isPointerType() &&
      ToType->getAs<PointerType>()->getPointeeType()->isVoidType()) {
    ConvertedType = ToType;
    return true;
  }
  // Blocks: A null pointer constant can be converted to a block
  // pointer type.
  if (ToType->isBlockPointerType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // If the left-hand-side is nullptr_t, the right side can be a null
  // pointer constant.
  if (ToType->isNullPtrType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  const PointerType* ToTypePtr = ToType->getAs<PointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a pointer type (C++ 4.10p1).
  if (isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // Beyond this point, both types need to be pointers
  // , including objective-c pointers.
  QualType ToPointeeType = ToTypePtr->getPointeeType();
  if (FromType->isObjCObjectPointerType() && ToPointeeType->isVoidType() &&
      !getLangOptions().ObjCAutoRefCount) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(
                                      FromType->getAs<ObjCObjectPointerType>(),
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }
  const PointerType *FromTypePtr = FromType->getAs<PointerType>();
  if (!FromTypePtr)
    return false;

  QualType FromPointeeType = FromTypePtr->getPointeeType();

  // If the unqualified pointee types are the same, this can't be a
  // pointer conversion, so don't do all of the work below.
  if (Context.hasSameUnqualifiedType(FromPointeeType, ToPointeeType))
    return false;

  // An rvalue of type "pointer to cv T," where T is an object type,
  // can be converted to an rvalue of type "pointer to cv void" (C++
  // 4.10p2).
  if (FromPointeeType->isIncompleteOrObjectType() &&
      ToPointeeType->isVoidType()) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context,
                                                   /*StripObjCLifetime=*/true);
    return true;
  }

  // MSVC allows implicit function to void* type conversion.
  if (getLangOptions().Microsoft && FromPointeeType->isFunctionType() &&
      ToPointeeType->isVoidType()) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  // When we're overloading in C, we allow a special kind of pointer
  // conversion for compatible-but-not-identical pointee types.
  if (!getLangOptions().CPlusPlus &&
      Context.typesAreCompatible(FromPointeeType, ToPointeeType)) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  // C++ [conv.ptr]p3:
  //
  //   An rvalue of type "pointer to cv D," where D is a class type,
  //   can be converted to an rvalue of type "pointer to cv B," where
  //   B is a base class (clause 10) of D. If B is an inaccessible
  //   (clause 11) or ambiguous (10.2) base class of D, a program that
  //   necessitates this conversion is ill-formed. The result of the
  //   conversion is a pointer to the base class sub-object of the
  //   derived class object. The null pointer value is converted to
  //   the null pointer value of the destination type.
  //
  // Note that we do not check for ambiguity or inaccessibility
  // here. That is handled by CheckPointerConversion.
  if (getLangOptions().CPlusPlus &&
      FromPointeeType->isRecordType() && ToPointeeType->isRecordType() &&
      !Context.hasSameUnqualifiedType(FromPointeeType, ToPointeeType) &&
      !RequireCompleteType(From->getLocStart(), FromPointeeType, PDiag()) &&
      IsDerivedFrom(FromPointeeType, ToPointeeType)) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  if (FromPointeeType->isVectorType() && ToPointeeType->isVectorType() &&
      Context.areCompatibleVectorTypes(FromPointeeType, ToPointeeType)) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }
  
  return false;
}
 
/// \brief Adopt the given qualifiers for the given type.
static QualType AdoptQualifiers(ASTContext &Context, QualType T, Qualifiers Qs){
  Qualifiers TQs = T.getQualifiers();
  
  // Check whether qualifiers already match.
  if (TQs == Qs)
    return T;
  
  if (Qs.compatiblyIncludes(TQs))
    return Context.getQualifiedType(T, Qs);
  
  return Context.getQualifiedType(T.getUnqualifiedType(), Qs);
}

/// isObjCPointerConversion - Determines whether this is an
/// Objective-C pointer conversion. Subroutine of IsPointerConversion,
/// with the same arguments and return values.
bool Sema::isObjCPointerConversion(QualType FromType, QualType ToType,
                                   QualType& ConvertedType,
                                   bool &IncompatibleObjC) {
  if (!getLangOptions().ObjC1)
    return false;

  // The set of qualifiers on the type we're converting from.
  Qualifiers FromQualifiers = FromType.getQualifiers();
  
  // First, we handle all conversions on ObjC object pointer types.
  const ObjCObjectPointerType* ToObjCPtr =
    ToType->getAs<ObjCObjectPointerType>();
  const ObjCObjectPointerType *FromObjCPtr =
    FromType->getAs<ObjCObjectPointerType>();

  if (ToObjCPtr && FromObjCPtr) {
    // If the pointee types are the same (ignoring qualifications),
    // then this is not a pointer conversion.
    if (Context.hasSameUnqualifiedType(ToObjCPtr->getPointeeType(),
                                       FromObjCPtr->getPointeeType()))
      return false;

    // Check for compatible 
    // Objective C++: We're able to convert between "id" or "Class" and a
    // pointer to any interface (in both directions).
    if (ToObjCPtr->isObjCBuiltinType() && FromObjCPtr->isObjCBuiltinType()) {
      ConvertedType = AdoptQualifiers(Context, ToType, FromQualifiers);
      return true;
    }
    // Conversions with Objective-C's id<...>.
    if ((FromObjCPtr->isObjCQualifiedIdType() ||
         ToObjCPtr->isObjCQualifiedIdType()) &&
        Context.ObjCQualifiedIdTypesAreCompatible(ToType, FromType,
                                                  /*compare=*/false)) {
      ConvertedType = AdoptQualifiers(Context, ToType, FromQualifiers);
      return true;
    }
    // Objective C++: We're able to convert from a pointer to an
    // interface to a pointer to a different interface.
    if (Context.canAssignObjCInterfaces(ToObjCPtr, FromObjCPtr)) {
      const ObjCInterfaceType* LHS = ToObjCPtr->getInterfaceType();
      const ObjCInterfaceType* RHS = FromObjCPtr->getInterfaceType();
      if (getLangOptions().CPlusPlus && LHS && RHS &&
          !ToObjCPtr->getPointeeType().isAtLeastAsQualifiedAs(
                                                FromObjCPtr->getPointeeType()))
        return false;
      ConvertedType = BuildSimilarlyQualifiedPointerType(FromObjCPtr,
                                                   ToObjCPtr->getPointeeType(),
                                                         ToType, Context);
      ConvertedType = AdoptQualifiers(Context, ConvertedType, FromQualifiers);
      return true;
    }

    if (Context.canAssignObjCInterfaces(FromObjCPtr, ToObjCPtr)) {
      // Okay: this is some kind of implicit downcast of Objective-C
      // interfaces, which is permitted. However, we're going to
      // complain about it.
      IncompatibleObjC = true;
      ConvertedType = BuildSimilarlyQualifiedPointerType(FromObjCPtr,
                                                   ToObjCPtr->getPointeeType(),
                                                         ToType, Context);
      ConvertedType = AdoptQualifiers(Context, ConvertedType, FromQualifiers);
      return true;
    }
  }
  // Beyond this point, both types need to be C pointers or block pointers.
  QualType ToPointeeType;
  if (const PointerType *ToCPtr = ToType->getAs<PointerType>())
    ToPointeeType = ToCPtr->getPointeeType();
  else if (const BlockPointerType *ToBlockPtr =
            ToType->getAs<BlockPointerType>()) {
    // Objective C++: We're able to convert from a pointer to any object
    // to a block pointer type.
    if (FromObjCPtr && FromObjCPtr->isObjCBuiltinType()) {
      ConvertedType = AdoptQualifiers(Context, ToType, FromQualifiers);
      return true;
    }
    ToPointeeType = ToBlockPtr->getPointeeType();
  }
  else if (FromType->getAs<BlockPointerType>() &&
           ToObjCPtr && ToObjCPtr->isObjCBuiltinType()) {
    // Objective C++: We're able to convert from a block pointer type to a
    // pointer to any object.
    ConvertedType = AdoptQualifiers(Context, ToType, FromQualifiers);
    return true;
  }
  else
    return false;

  QualType FromPointeeType;
  if (const PointerType *FromCPtr = FromType->getAs<PointerType>())
    FromPointeeType = FromCPtr->getPointeeType();
  else if (const BlockPointerType *FromBlockPtr =
           FromType->getAs<BlockPointerType>())
    FromPointeeType = FromBlockPtr->getPointeeType();
  else
    return false;

  // If we have pointers to pointers, recursively check whether this
  // is an Objective-C conversion.
  if (FromPointeeType->isPointerType() && ToPointeeType->isPointerType() &&
      isObjCPointerConversion(FromPointeeType, ToPointeeType, ConvertedType,
                              IncompatibleObjC)) {
    // We always complain about this conversion.
    IncompatibleObjC = true;
    ConvertedType = Context.getPointerType(ConvertedType);
    ConvertedType = AdoptQualifiers(Context, ConvertedType, FromQualifiers);
    return true;
  }
  // Allow conversion of pointee being objective-c pointer to another one;
  // as in I* to id.
  if (FromPointeeType->getAs<ObjCObjectPointerType>() &&
      ToPointeeType->getAs<ObjCObjectPointerType>() &&
      isObjCPointerConversion(FromPointeeType, ToPointeeType, ConvertedType,
                              IncompatibleObjC)) {
        
    ConvertedType = Context.getPointerType(ConvertedType);
    ConvertedType = AdoptQualifiers(Context, ConvertedType, FromQualifiers);
    return true;
  }

  // If we have pointers to functions or blocks, check whether the only
  // differences in the argument and result types are in Objective-C
  // pointer conversions. If so, we permit the conversion (but
  // complain about it).
  const FunctionProtoType *FromFunctionType
    = FromPointeeType->getAs<FunctionProtoType>();
  const FunctionProtoType *ToFunctionType
    = ToPointeeType->getAs<FunctionProtoType>();
  if (FromFunctionType && ToFunctionType) {
    // If the function types are exactly the same, this isn't an
    // Objective-C pointer conversion.
    if (Context.getCanonicalType(FromPointeeType)
          == Context.getCanonicalType(ToPointeeType))
      return false;

    // Perform the quick checks that will tell us whether these
    // function types are obviously different.
    if (FromFunctionType->getNumArgs() != ToFunctionType->getNumArgs() ||
        FromFunctionType->isVariadic() != ToFunctionType->isVariadic() ||
        FromFunctionType->getTypeQuals() != ToFunctionType->getTypeQuals())
      return false;

    bool HasObjCConversion = false;
    if (Context.getCanonicalType(FromFunctionType->getResultType())
          == Context.getCanonicalType(ToFunctionType->getResultType())) {
      // Okay, the types match exactly. Nothing to do.
    } else if (isObjCPointerConversion(FromFunctionType->getResultType(),
                                       ToFunctionType->getResultType(),
                                       ConvertedType, IncompatibleObjC)) {
      // Okay, we have an Objective-C pointer conversion.
      HasObjCConversion = true;
    } else {
      // Function types are too different. Abort.
      return false;
    }

    // Check argument types.
    for (unsigned ArgIdx = 0, NumArgs = FromFunctionType->getNumArgs();
         ArgIdx != NumArgs; ++ArgIdx) {
      QualType FromArgType = FromFunctionType->getArgType(ArgIdx);
      QualType ToArgType = ToFunctionType->getArgType(ArgIdx);
      if (Context.getCanonicalType(FromArgType)
            == Context.getCanonicalType(ToArgType)) {
        // Okay, the types match exactly. Nothing to do.
      } else if (isObjCPointerConversion(FromArgType, ToArgType,
                                         ConvertedType, IncompatibleObjC)) {
        // Okay, we have an Objective-C pointer conversion.
        HasObjCConversion = true;
      } else {
        // Argument types are too different. Abort.
        return false;
      }
    }

    if (HasObjCConversion) {
      // We had an Objective-C conversion. Allow this pointer
      // conversion, but complain about it.
      ConvertedType = AdoptQualifiers(Context, ToType, FromQualifiers);
      IncompatibleObjC = true;
      return true;
    }
  }

  return false;
}

/// \brief Determine whether this is an Objective-C writeback conversion,
/// used for parameter passing when performing automatic reference counting.
///
/// \param FromType The type we're converting form.
///
/// \param ToType The type we're converting to.
///
/// \param ConvertedType The type that will be produced after applying
/// this conversion.
bool Sema::isObjCWritebackConversion(QualType FromType, QualType ToType,
                                     QualType &ConvertedType) {
  if (!getLangOptions().ObjCAutoRefCount || 
      Context.hasSameUnqualifiedType(FromType, ToType))
    return false;
  
  // Parameter must be a pointer to __autoreleasing (with no other qualifiers).
  QualType ToPointee;
  if (const PointerType *ToPointer = ToType->getAs<PointerType>())
    ToPointee = ToPointer->getPointeeType();
  else
    return false;
  
  Qualifiers ToQuals = ToPointee.getQualifiers();
  if (!ToPointee->isObjCLifetimeType() || 
      ToQuals.getObjCLifetime() != Qualifiers::OCL_Autoreleasing ||
      !ToQuals.withoutObjCGLifetime().empty())
    return false;
  
  // Argument must be a pointer to __strong to __weak.
  QualType FromPointee;
  if (const PointerType *FromPointer = FromType->getAs<PointerType>())
    FromPointee = FromPointer->getPointeeType();
  else
    return false;
  
  Qualifiers FromQuals = FromPointee.getQualifiers();
  if (!FromPointee->isObjCLifetimeType() ||
      (FromQuals.getObjCLifetime() != Qualifiers::OCL_Strong &&
       FromQuals.getObjCLifetime() != Qualifiers::OCL_Weak))
    return false;
  
  // Make sure that we have compatible qualifiers.
  FromQuals.setObjCLifetime(Qualifiers::OCL_Autoreleasing);
  if (!ToQuals.compatiblyIncludes(FromQuals))
    return false;
  
  // Remove qualifiers from the pointee type we're converting from; they
  // aren't used in the compatibility check belong, and we'll be adding back
  // qualifiers (with __autoreleasing) if the compatibility check succeeds.
  FromPointee = FromPointee.getUnqualifiedType();
  
  // The unqualified form of the pointee types must be compatible.
  ToPointee = ToPointee.getUnqualifiedType();
  bool IncompatibleObjC;
  if (Context.typesAreCompatible(FromPointee, ToPointee))
    FromPointee = ToPointee;
  else if (!isObjCPointerConversion(FromPointee, ToPointee, FromPointee,
                                    IncompatibleObjC))
    return false;
  
  /// \brief Construct the type we're converting to, which is a pointer to
  /// __autoreleasing pointee.
  FromPointee = Context.getQualifiedType(FromPointee, FromQuals);
  ConvertedType = Context.getPointerType(FromPointee);
  return true;
}

bool Sema::IsBlockPointerConversion(QualType FromType, QualType ToType,
                                    QualType& ConvertedType) {
  QualType ToPointeeType;
  if (const BlockPointerType *ToBlockPtr =
        ToType->getAs<BlockPointerType>())
    ToPointeeType = ToBlockPtr->getPointeeType();
  else
    return false;
  
  QualType FromPointeeType;
  if (const BlockPointerType *FromBlockPtr =
      FromType->getAs<BlockPointerType>())
    FromPointeeType = FromBlockPtr->getPointeeType();
  else
    return false;
  // We have pointer to blocks, check whether the only
  // differences in the argument and result types are in Objective-C
  // pointer conversions. If so, we permit the conversion.
  
  const FunctionProtoType *FromFunctionType
    = FromPointeeType->getAs<FunctionProtoType>();
  const FunctionProtoType *ToFunctionType
    = ToPointeeType->getAs<FunctionProtoType>();
  
  if (!FromFunctionType || !ToFunctionType)
    return false;

  if (Context.hasSameType(FromPointeeType, ToPointeeType))
    return true;
    
  // Perform the quick checks that will tell us whether these
  // function types are obviously different.
  if (FromFunctionType->getNumArgs() != ToFunctionType->getNumArgs() ||
      FromFunctionType->isVariadic() != ToFunctionType->isVariadic())
    return false;
    
  FunctionType::ExtInfo FromEInfo = FromFunctionType->getExtInfo();
  FunctionType::ExtInfo ToEInfo = ToFunctionType->getExtInfo();
  if (FromEInfo != ToEInfo)
    return false;

  bool IncompatibleObjC = false;
  if (Context.hasSameType(FromFunctionType->getResultType(), 
                          ToFunctionType->getResultType())) {
    // Okay, the types match exactly. Nothing to do.
  } else {
    QualType RHS = FromFunctionType->getResultType();
    QualType LHS = ToFunctionType->getResultType();
    if ((!getLangOptions().CPlusPlus || !RHS->isRecordType()) &&
        !RHS.hasQualifiers() && LHS.hasQualifiers())
       LHS = LHS.getUnqualifiedType();

     if (Context.hasSameType(RHS,LHS)) {
       // OK exact match.
     } else if (isObjCPointerConversion(RHS, LHS,
                                        ConvertedType, IncompatibleObjC)) {
     if (IncompatibleObjC)
       return false;
     // Okay, we have an Objective-C pointer conversion.
     }
     else
       return false;
   }
    
   // Check argument types.
   for (unsigned ArgIdx = 0, NumArgs = FromFunctionType->getNumArgs();
        ArgIdx != NumArgs; ++ArgIdx) {
     IncompatibleObjC = false;
     QualType FromArgType = FromFunctionType->getArgType(ArgIdx);
     QualType ToArgType = ToFunctionType->getArgType(ArgIdx);
     if (Context.hasSameType(FromArgType, ToArgType)) {
       // Okay, the types match exactly. Nothing to do.
     } else if (isObjCPointerConversion(ToArgType, FromArgType,
                                        ConvertedType, IncompatibleObjC)) {
       if (IncompatibleObjC)
         return false;
       // Okay, we have an Objective-C pointer conversion.
     } else
       // Argument types are too different. Abort.
       return false;
   }
   ConvertedType = ToType;
   return true;
}

/// FunctionArgTypesAreEqual - This routine checks two function proto types
/// for equlity of their argument types. Caller has already checked that
/// they have same number of arguments. This routine assumes that Objective-C
/// pointer types which only differ in their protocol qualifiers are equal.
bool Sema::FunctionArgTypesAreEqual(const FunctionProtoType *OldType,
                                    const FunctionProtoType *NewType) {
  if (!getLangOptions().ObjC1)
    return std::equal(OldType->arg_type_begin(), OldType->arg_type_end(),
                      NewType->arg_type_begin());

  for (FunctionProtoType::arg_type_iterator O = OldType->arg_type_begin(),
       N = NewType->arg_type_begin(),
       E = OldType->arg_type_end(); O && (O != E); ++O, ++N) {
    QualType ToType = (*O);
    QualType FromType = (*N);
    if (ToType != FromType) {
      if (const PointerType *PTTo = ToType->getAs<PointerType>()) {
        if (const PointerType *PTFr = FromType->getAs<PointerType>())
          if ((PTTo->getPointeeType()->isObjCQualifiedIdType() &&
               PTFr->getPointeeType()->isObjCQualifiedIdType()) ||
              (PTTo->getPointeeType()->isObjCQualifiedClassType() &&
               PTFr->getPointeeType()->isObjCQualifiedClassType()))
            continue;
      }
      else if (const ObjCObjectPointerType *PTTo =
                 ToType->getAs<ObjCObjectPointerType>()) {
        if (const ObjCObjectPointerType *PTFr =
              FromType->getAs<ObjCObjectPointerType>())
          if (PTTo->getInterfaceDecl() == PTFr->getInterfaceDecl())
            continue;
      }
      return false;
    }
  }
  return true;
}

/// CheckPointerConversion - Check the pointer conversion from the
/// expression From to the type ToType. This routine checks for
/// ambiguous or inaccessible derived-to-base pointer
/// conversions for which IsPointerConversion has already returned
/// true. It returns true and produces a diagnostic if there was an
/// error, or returns false otherwise.
bool Sema::CheckPointerConversion(Expr *From, QualType ToType,
                                  CastKind &Kind,
                                  CXXCastPath& BasePath,
                                  bool IgnoreBaseAccess) {
  QualType FromType = From->getType();
  bool IsCStyleOrFunctionalCast = IgnoreBaseAccess;

  Kind = CK_BitCast;

  if (!IsCStyleOrFunctionalCast &&
      Context.hasSameUnqualifiedType(From->getType(), Context.BoolTy) &&
      From->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull))
    DiagRuntimeBehavior(From->getExprLoc(), From,
                        PDiag(diag::warn_impcast_bool_to_null_pointer)
                          << ToType << From->getSourceRange());

  if (const PointerType *FromPtrType = FromType->getAs<PointerType>())
    if (const PointerType *ToPtrType = ToType->getAs<PointerType>()) {
      QualType FromPointeeType = FromPtrType->getPointeeType(),
               ToPointeeType   = ToPtrType->getPointeeType();

      if (FromPointeeType->isRecordType() && ToPointeeType->isRecordType() &&
          !Context.hasSameUnqualifiedType(FromPointeeType, ToPointeeType)) {
        // We must have a derived-to-base conversion. Check an
        // ambiguous or inaccessible conversion.
        if (CheckDerivedToBaseConversion(FromPointeeType, ToPointeeType,
                                         From->getExprLoc(),
                                         From->getSourceRange(), &BasePath,
                                         IgnoreBaseAccess))
          return true;

        // The conversion was successful.
        Kind = CK_DerivedToBase;
      }
    }
  if (const ObjCObjectPointerType *FromPtrType =
        FromType->getAs<ObjCObjectPointerType>()) {
    if (const ObjCObjectPointerType *ToPtrType =
          ToType->getAs<ObjCObjectPointerType>()) {
      // Objective-C++ conversions are always okay.
      // FIXME: We should have a different class of conversions for the
      // Objective-C++ implicit conversions.
      if (FromPtrType->isObjCBuiltinType() || ToPtrType->isObjCBuiltinType())
        return false;
    }
  }

  // We shouldn't fall into this case unless it's valid for other
  // reasons.
  if (From->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull))
    Kind = CK_NullToPointer;

  return false;
}

/// IsMemberPointerConversion - Determines whether the conversion of the
/// expression From, which has the (possibly adjusted) type FromType, can be
/// converted to the type ToType via a member pointer conversion (C++ 4.11).
/// If so, returns true and places the converted type (that might differ from
/// ToType in its cv-qualifiers at some level) into ConvertedType.
bool Sema::IsMemberPointerConversion(Expr *From, QualType FromType,
                                     QualType ToType,
                                     bool InOverloadResolution,
                                     QualType &ConvertedType) {
  const MemberPointerType *ToTypePtr = ToType->getAs<MemberPointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a member pointer (C++ 4.11p1)
  if (From->isNullPointerConstant(Context,
                    InOverloadResolution? Expr::NPC_ValueDependentIsNotNull
                                        : Expr::NPC_ValueDependentIsNull)) {
    ConvertedType = ToType;
    return true;
  }

  // Otherwise, both types have to be member pointers.
  const MemberPointerType *FromTypePtr = FromType->getAs<MemberPointerType>();
  if (!FromTypePtr)
    return false;

  // A pointer to member of B can be converted to a pointer to member of D,
  // where D is derived from B (C++ 4.11p2).
  QualType FromClass(FromTypePtr->getClass(), 0);
  QualType ToClass(ToTypePtr->getClass(), 0);

  if (!Context.hasSameUnqualifiedType(FromClass, ToClass) &&
      !RequireCompleteType(From->getLocStart(), ToClass, PDiag()) &&
      IsDerivedFrom(ToClass, FromClass)) {
    ConvertedType = Context.getMemberPointerType(FromTypePtr->getPointeeType(),
                                                 ToClass.getTypePtr());
    return true;
  }

  return false;
}

/// CheckMemberPointerConversion - Check the member pointer conversion from the
/// expression From to the type ToType. This routine checks for ambiguous or
/// virtual or inaccessible base-to-derived member pointer conversions
/// for which IsMemberPointerConversion has already returned true. It returns
/// true and produces a diagnostic if there was an error, or returns false
/// otherwise.
bool Sema::CheckMemberPointerConversion(Expr *From, QualType ToType,
                                        CastKind &Kind,
                                        CXXCastPath &BasePath,
                                        bool IgnoreBaseAccess) {
  QualType FromType = From->getType();
  const MemberPointerType *FromPtrType = FromType->getAs<MemberPointerType>();
  if (!FromPtrType) {
    // This must be a null pointer to member pointer conversion
    assert(From->isNullPointerConstant(Context,
                                       Expr::NPC_ValueDependentIsNull) &&
           "Expr must be null pointer constant!");
    Kind = CK_NullToMemberPointer;
    return false;
  }

  const MemberPointerType *ToPtrType = ToType->getAs<MemberPointerType>();
  assert(ToPtrType && "No member pointer cast has a target type "
                      "that is not a member pointer.");

  QualType FromClass = QualType(FromPtrType->getClass(), 0);
  QualType ToClass   = QualType(ToPtrType->getClass(), 0);

  // FIXME: What about dependent types?
  assert(FromClass->isRecordType() && "Pointer into non-class.");
  assert(ToClass->isRecordType() && "Pointer into non-class.");

  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/true);
  bool DerivationOkay = IsDerivedFrom(ToClass, FromClass, Paths);
  assert(DerivationOkay &&
         "Should not have been called if derivation isn't OK.");
  (void)DerivationOkay;

  if (Paths.isAmbiguous(Context.getCanonicalType(FromClass).
                                  getUnqualifiedType())) {
    std::string PathDisplayStr = getAmbiguousPathsDisplayString(Paths);
    Diag(From->getExprLoc(), diag::err_ambiguous_memptr_conv)
      << 0 << FromClass << ToClass << PathDisplayStr << From->getSourceRange();
    return true;
  }

  if (const RecordType *VBase = Paths.getDetectedVirtual()) {
    Diag(From->getExprLoc(), diag::err_memptr_conv_via_virtual)
      << FromClass << ToClass << QualType(VBase, 0)
      << From->getSourceRange();
    return true;
  }

  if (!IgnoreBaseAccess)
    CheckBaseClassAccess(From->getExprLoc(), FromClass, ToClass,
                         Paths.front(),
                         diag::err_downcast_from_inaccessible_base);

  // Must be a base to derived member conversion.
  BuildBasePathArray(Paths, BasePath);
  Kind = CK_BaseToDerivedMemberPointer;
  return false;
}

/// IsQualificationConversion - Determines whether the conversion from
/// an rvalue of type FromType to ToType is a qualification conversion
/// (C++ 4.4).
///
/// \param ObjCLifetimeConversion Output parameter that will be set to indicate
/// when the qualification conversion involves a change in the Objective-C
/// object lifetime.
bool
Sema::IsQualificationConversion(QualType FromType, QualType ToType,
                                bool CStyle, bool &ObjCLifetimeConversion) {
  FromType = Context.getCanonicalType(FromType);
  ToType = Context.getCanonicalType(ToType);
  ObjCLifetimeConversion = false;
  
  // If FromType and ToType are the same type, this is not a
  // qualification conversion.
  if (FromType.getUnqualifiedType() == ToType.getUnqualifiedType())
    return false;

  // (C++ 4.4p4):
  //   A conversion can add cv-qualifiers at levels other than the first
  //   in multi-level pointers, subject to the following rules: [...]
  bool PreviousToQualsIncludeConst = true;
  bool UnwrappedAnyPointer = false;
  while (Context.UnwrapSimilarPointerTypes(FromType, ToType)) {
    // Within each iteration of the loop, we check the qualifiers to
    // determine if this still looks like a qualification
    // conversion. Then, if all is well, we unwrap one more level of
    // pointers or pointers-to-members and do it all again
    // until there are no more pointers or pointers-to-members left to
    // unwrap.
    UnwrappedAnyPointer = true;

    Qualifiers FromQuals = FromType.getQualifiers();
    Qualifiers ToQuals = ToType.getQualifiers();
    
    // Objective-C ARC:
    //   Check Objective-C lifetime conversions.
    if (FromQuals.getObjCLifetime() != ToQuals.getObjCLifetime() &&
        UnwrappedAnyPointer) {
      if (ToQuals.compatiblyIncludesObjCLifetime(FromQuals)) {
        ObjCLifetimeConversion = true;
        FromQuals.removeObjCLifetime();
        ToQuals.removeObjCLifetime();
      } else {
        // Qualification conversions cannot cast between different
        // Objective-C lifetime qualifiers.
        return false;
      }
    }
    
    // Allow addition/removal of GC attributes but not changing GC attributes.
    if (FromQuals.getObjCGCAttr() != ToQuals.getObjCGCAttr() &&
        (!FromQuals.hasObjCGCAttr() || !ToQuals.hasObjCGCAttr())) {
      FromQuals.removeObjCGCAttr();
      ToQuals.removeObjCGCAttr();
    }
    
    //   -- for every j > 0, if const is in cv 1,j then const is in cv
    //      2,j, and similarly for volatile.
    if (!CStyle && !ToQuals.compatiblyIncludes(FromQuals))
      return false;

    //   -- if the cv 1,j and cv 2,j are different, then const is in
    //      every cv for 0 < k < j.
    if (!CStyle && FromQuals.getCVRQualifiers() != ToQuals.getCVRQualifiers()
        && !PreviousToQualsIncludeConst)
      return false;

    // Keep track of whether all prior cv-qualifiers in the "to" type
    // include const.
    PreviousToQualsIncludeConst
      = PreviousToQualsIncludeConst && ToQuals.hasConst();
  }

  // We are left with FromType and ToType being the pointee types
  // after unwrapping the original FromType and ToType the same number
  // of types. If we unwrapped any pointers, and if FromType and
  // ToType have the same unqualified type (since we checked
  // qualifiers above), then this is a qualification conversion.
  return UnwrappedAnyPointer && Context.hasSameUnqualifiedType(FromType,ToType);
}

/// Determines whether there is a user-defined conversion sequence
/// (C++ [over.ics.user]) that converts expression From to the type
/// ToType. If such a conversion exists, User will contain the
/// user-defined conversion sequence that performs such a conversion
/// and this routine will return true. Otherwise, this routine returns
/// false and User is unspecified.
///
/// \param AllowExplicit  true if the conversion should consider C++0x
/// "explicit" conversion functions as well as non-explicit conversion
/// functions (C++0x [class.conv.fct]p2).
static OverloadingResult
IsUserDefinedConversion(Sema &S, Expr *From, QualType ToType,
                        UserDefinedConversionSequence& User,
                        OverloadCandidateSet& CandidateSet,
                        bool AllowExplicit) {
  // Whether we will only visit constructors.
  bool ConstructorsOnly = false;

  // If the type we are conversion to is a class type, enumerate its
  // constructors.
  if (const RecordType *ToRecordType = ToType->getAs<RecordType>()) {
    // C++ [over.match.ctor]p1:
    //   When objects of class type are direct-initialized (8.5), or
    //   copy-initialized from an expression of the same or a
    //   derived class type (8.5), overload resolution selects the
    //   constructor. [...] For copy-initialization, the candidate
    //   functions are all the converting constructors (12.3.1) of
    //   that class. The argument list is the expression-list within
    //   the parentheses of the initializer.
    if (S.Context.hasSameUnqualifiedType(ToType, From->getType()) ||
        (From->getType()->getAs<RecordType>() &&
         S.IsDerivedFrom(From->getType(), ToType)))
      ConstructorsOnly = true;

    S.RequireCompleteType(From->getLocStart(), ToType, S.PDiag());
    // RequireCompleteType may have returned true due to some invalid decl
    // during template instantiation, but ToType may be complete enough now
    // to try to recover.
    if (ToType->isIncompleteType()) {
      // We're not going to find any constructors.
    } else if (CXXRecordDecl *ToRecordDecl
                 = dyn_cast<CXXRecordDecl>(ToRecordType->getDecl())) {
      DeclContext::lookup_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd) = S.LookupConstructors(ToRecordDecl);
           Con != ConEnd; ++Con) {
        NamedDecl *D = *Con;
        DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());

        // Find the constructor (which may be a template).
        CXXConstructorDecl *Constructor = 0;
        FunctionTemplateDecl *ConstructorTmpl
          = dyn_cast<FunctionTemplateDecl>(D);
        if (ConstructorTmpl)
          Constructor
            = cast<CXXConstructorDecl>(ConstructorTmpl->getTemplatedDecl());
        else
          Constructor = cast<CXXConstructorDecl>(D);

        if (!Constructor->isInvalidDecl() &&
            Constructor->isConvertingConstructor(AllowExplicit)) {
          if (ConstructorTmpl)
            S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                           /*ExplicitArgs*/ 0,
                                           &From, 1, CandidateSet,
                                           /*SuppressUserConversions=*/
                                             !ConstructorsOnly);
          else
            // Allow one user-defined conversion when user specifies a
            // From->ToType conversion via an static cast (c-style, etc).
            S.AddOverloadCandidate(Constructor, FoundDecl,
                                   &From, 1, CandidateSet,
                                   /*SuppressUserConversions=*/
                                     !ConstructorsOnly);
        }
      }
    }
  }

  // Enumerate conversion functions, if we're allowed to.
  if (ConstructorsOnly) {
  } else if (S.RequireCompleteType(From->getLocStart(), From->getType(),
                                   S.PDiag(0) << From->getSourceRange())) {
    // No conversion functions from incomplete types.
  } else if (const RecordType *FromRecordType
                                   = From->getType()->getAs<RecordType>()) {
    if (CXXRecordDecl *FromRecordDecl
         = dyn_cast<CXXRecordDecl>(FromRecordType->getDecl())) {
      // Add all of the conversion functions as candidates.
      const UnresolvedSetImpl *Conversions
        = FromRecordDecl->getVisibleConversionFunctions();
      for (UnresolvedSetImpl::iterator I = Conversions->begin(),
             E = Conversions->end(); I != E; ++I) {
        DeclAccessPair FoundDecl = I.getPair();
        NamedDecl *D = FoundDecl.getDecl();
        CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(D->getDeclContext());
        if (isa<UsingShadowDecl>(D))
          D = cast<UsingShadowDecl>(D)->getTargetDecl();

        CXXConversionDecl *Conv;
        FunctionTemplateDecl *ConvTemplate;
        if ((ConvTemplate = dyn_cast<FunctionTemplateDecl>(D)))
          Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
        else
          Conv = cast<CXXConversionDecl>(D);

        if (AllowExplicit || !Conv->isExplicit()) {
          if (ConvTemplate)
            S.AddTemplateConversionCandidate(ConvTemplate, FoundDecl,
                                             ActingContext, From, ToType,
                                             CandidateSet);
          else
            S.AddConversionCandidate(Conv, FoundDecl, ActingContext,
                                     From, ToType, CandidateSet);
        }
      }
    }
  }

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(S, From->getLocStart(), Best, true)) {
  case OR_Success:
    // Record the standard conversion we used and the conversion function.
    if (CXXConstructorDecl *Constructor
          = dyn_cast<CXXConstructorDecl>(Best->Function)) {
      S.MarkDeclarationReferenced(From->getLocStart(), Constructor);

      // C++ [over.ics.user]p1:
      //   If the user-defined conversion is specified by a
      //   constructor (12.3.1), the initial standard conversion
      //   sequence converts the source type to the type required by
      //   the argument of the constructor.
      //
      QualType ThisType = Constructor->getThisType(S.Context);
      if (Best->Conversions[0].isEllipsis())
        User.EllipsisConversion = true;
      else {
        User.Before = Best->Conversions[0].Standard;
        User.EllipsisConversion = false;
      }
      User.ConversionFunction = Constructor;
      User.FoundConversionFunction = Best->FoundDecl.getDecl();
      User.After.setAsIdentityConversion();
      User.After.setFromType(ThisType->getAs<PointerType>()->getPointeeType());
      User.After.setAllToTypes(ToType);
      return OR_Success;
    } else if (CXXConversionDecl *Conversion
                 = dyn_cast<CXXConversionDecl>(Best->Function)) {
      S.MarkDeclarationReferenced(From->getLocStart(), Conversion);

      // C++ [over.ics.user]p1:
      //
      //   [...] If the user-defined conversion is specified by a
      //   conversion function (12.3.2), the initial standard
      //   conversion sequence converts the source type to the
      //   implicit object parameter of the conversion function.
      User.Before = Best->Conversions[0].Standard;
      User.ConversionFunction = Conversion;
      User.FoundConversionFunction = Best->FoundDecl.getDecl();
      User.EllipsisConversion = false;

      // C++ [over.ics.user]p2:
      //   The second standard conversion sequence converts the
      //   result of the user-defined conversion to the target type
      //   for the sequence. Since an implicit conversion sequence
      //   is an initialization, the special rules for
      //   initialization by user-defined conversion apply when
      //   selecting the best user-defined conversion for a
      //   user-defined conversion sequence (see 13.3.3 and
      //   13.3.3.1).
      User.After = Best->FinalConversion;
      return OR_Success;
    } else {
      llvm_unreachable("Not a constructor or conversion function?");
      return OR_No_Viable_Function;
    }

  case OR_No_Viable_Function:
    return OR_No_Viable_Function;
  case OR_Deleted:
    // No conversion here! We're done.
    return OR_Deleted;

  case OR_Ambiguous:
    return OR_Ambiguous;
  }

  return OR_No_Viable_Function;
}

bool
Sema::DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType) {
  ImplicitConversionSequence ICS;
  OverloadCandidateSet CandidateSet(From->getExprLoc());
  OverloadingResult OvResult =
    IsUserDefinedConversion(*this, From, ToType, ICS.UserDefined,
                            CandidateSet, false);
  if (OvResult == OR_Ambiguous)
    Diag(From->getSourceRange().getBegin(),
         diag::err_typecheck_ambiguous_condition)
          << From->getType() << ToType << From->getSourceRange();
  else if (OvResult == OR_No_Viable_Function && !CandidateSet.empty())
    Diag(From->getSourceRange().getBegin(),
         diag::err_typecheck_nonviable_condition)
    << From->getType() << ToType << From->getSourceRange();
  else
    return false;
  CandidateSet.NoteCandidates(*this, OCD_AllCandidates, &From, 1);
  return true;
}

/// CompareImplicitConversionSequences - Compare two implicit
/// conversion sequences to determine whether one is better than the
/// other or if they are indistinguishable (C++ 13.3.3.2).
static ImplicitConversionSequence::CompareKind
CompareImplicitConversionSequences(Sema &S,
                                   const ImplicitConversionSequence& ICS1,
                                   const ImplicitConversionSequence& ICS2)
{
  // (C++ 13.3.3.2p2): When comparing the basic forms of implicit
  // conversion sequences (as defined in 13.3.3.1)
  //   -- a standard conversion sequence (13.3.3.1.1) is a better
  //      conversion sequence than a user-defined conversion sequence or
  //      an ellipsis conversion sequence, and
  //   -- a user-defined conversion sequence (13.3.3.1.2) is a better
  //      conversion sequence than an ellipsis conversion sequence
  //      (13.3.3.1.3).
  //
  // C++0x [over.best.ics]p10:
  //   For the purpose of ranking implicit conversion sequences as
  //   described in 13.3.3.2, the ambiguous conversion sequence is
  //   treated as a user-defined sequence that is indistinguishable
  //   from any other user-defined conversion sequence.
  if (ICS1.getKindRank() < ICS2.getKindRank())
    return ImplicitConversionSequence::Better;
  else if (ICS2.getKindRank() < ICS1.getKindRank())
    return ImplicitConversionSequence::Worse;

  // The following checks require both conversion sequences to be of
  // the same kind.
  if (ICS1.getKind() != ICS2.getKind())
    return ImplicitConversionSequence::Indistinguishable;

  // Two implicit conversion sequences of the same form are
  // indistinguishable conversion sequences unless one of the
  // following rules apply: (C++ 13.3.3.2p3):
  if (ICS1.isStandard())
    return CompareStandardConversionSequences(S, ICS1.Standard, ICS2.Standard);
  else if (ICS1.isUserDefined()) {
    // User-defined conversion sequence U1 is a better conversion
    // sequence than another user-defined conversion sequence U2 if
    // they contain the same user-defined conversion function or
    // constructor and if the second standard conversion sequence of
    // U1 is better than the second standard conversion sequence of
    // U2 (C++ 13.3.3.2p3).
    if (ICS1.UserDefined.ConversionFunction ==
          ICS2.UserDefined.ConversionFunction)
      return CompareStandardConversionSequences(S,
                                                ICS1.UserDefined.After,
                                                ICS2.UserDefined.After);
  }

  return ImplicitConversionSequence::Indistinguishable;
}

static bool hasSimilarType(ASTContext &Context, QualType T1, QualType T2) {
  while (Context.UnwrapSimilarPointerTypes(T1, T2)) {
    Qualifiers Quals;
    T1 = Context.getUnqualifiedArrayType(T1, Quals);
    T2 = Context.getUnqualifiedArrayType(T2, Quals);
  }

  return Context.hasSameUnqualifiedType(T1, T2);
}

// Per 13.3.3.2p3, compare the given standard conversion sequences to
// determine if one is a proper subset of the other.
static ImplicitConversionSequence::CompareKind
compareStandardConversionSubsets(ASTContext &Context,
                                 const StandardConversionSequence& SCS1,
                                 const StandardConversionSequence& SCS2) {
  ImplicitConversionSequence::CompareKind Result
    = ImplicitConversionSequence::Indistinguishable;

  // the identity conversion sequence is considered to be a subsequence of
  // any non-identity conversion sequence
  if (SCS1.isIdentityConversion() && !SCS2.isIdentityConversion())
    return ImplicitConversionSequence::Better;
  else if (!SCS1.isIdentityConversion() && SCS2.isIdentityConversion())
    return ImplicitConversionSequence::Worse;

  if (SCS1.Second != SCS2.Second) {
    if (SCS1.Second == ICK_Identity)
      Result = ImplicitConversionSequence::Better;
    else if (SCS2.Second == ICK_Identity)
      Result = ImplicitConversionSequence::Worse;
    else
      return ImplicitConversionSequence::Indistinguishable;
  } else if (!hasSimilarType(Context, SCS1.getToType(1), SCS2.getToType(1)))
    return ImplicitConversionSequence::Indistinguishable;

  if (SCS1.Third == SCS2.Third) {
    return Context.hasSameType(SCS1.getToType(2), SCS2.getToType(2))? Result
                             : ImplicitConversionSequence::Indistinguishable;
  }

  if (SCS1.Third == ICK_Identity)
    return Result == ImplicitConversionSequence::Worse
             ? ImplicitConversionSequence::Indistinguishable
             : ImplicitConversionSequence::Better;

  if (SCS2.Third == ICK_Identity)
    return Result == ImplicitConversionSequence::Better
             ? ImplicitConversionSequence::Indistinguishable
             : ImplicitConversionSequence::Worse;

  return ImplicitConversionSequence::Indistinguishable;
}

/// \brief Determine whether one of the given reference bindings is better
/// than the other based on what kind of bindings they are.
static bool isBetterReferenceBindingKind(const StandardConversionSequence &SCS1,
                                       const StandardConversionSequence &SCS2) {
  // C++0x [over.ics.rank]p3b4:
  //   -- S1 and S2 are reference bindings (8.5.3) and neither refers to an
  //      implicit object parameter of a non-static member function declared
  //      without a ref-qualifier, and *either* S1 binds an rvalue reference
  //      to an rvalue and S2 binds an lvalue reference *or S1 binds an
  //      lvalue reference to a function lvalue and S2 binds an rvalue
  //      reference*.
  //
  // FIXME: Rvalue references. We're going rogue with the above edits,
  // because the semantics in the current C++0x working paper (N3225 at the
  // time of this writing) break the standard definition of std::forward
  // and std::reference_wrapper when dealing with references to functions.
  // Proposed wording changes submitted to CWG for consideration.
  if (SCS1.BindsImplicitObjectArgumentWithoutRefQualifier ||
      SCS2.BindsImplicitObjectArgumentWithoutRefQualifier)
    return false;

  return (!SCS1.IsLvalueReference && SCS1.BindsToRvalue &&
          SCS2.IsLvalueReference) ||
         (SCS1.IsLvalueReference && SCS1.BindsToFunctionLvalue &&
          !SCS2.IsLvalueReference);
}

/// CompareStandardConversionSequences - Compare two standard
/// conversion sequences to determine whether one is better than the
/// other or if they are indistinguishable (C++ 13.3.3.2p3).
static ImplicitConversionSequence::CompareKind
CompareStandardConversionSequences(Sema &S,
                                   const StandardConversionSequence& SCS1,
                                   const StandardConversionSequence& SCS2)
{
  // Standard conversion sequence S1 is a better conversion sequence
  // than standard conversion sequence S2 if (C++ 13.3.3.2p3):

  //  -- S1 is a proper subsequence of S2 (comparing the conversion
  //     sequences in the canonical form defined by 13.3.3.1.1,
  //     excluding any Lvalue Transformation; the identity conversion
  //     sequence is considered to be a subsequence of any
  //     non-identity conversion sequence) or, if not that,
  if (ImplicitConversionSequence::CompareKind CK
        = compareStandardConversionSubsets(S.Context, SCS1, SCS2))
    return CK;

  //  -- the rank of S1 is better than the rank of S2 (by the rules
  //     defined below), or, if not that,
  ImplicitConversionRank Rank1 = SCS1.getRank();
  ImplicitConversionRank Rank2 = SCS2.getRank();
  if (Rank1 < Rank2)
    return ImplicitConversionSequence::Better;
  else if (Rank2 < Rank1)
    return ImplicitConversionSequence::Worse;

  // (C++ 13.3.3.2p4): Two conversion sequences with the same rank
  // are indistinguishable unless one of the following rules
  // applies:

  //   A conversion that is not a conversion of a pointer, or
  //   pointer to member, to bool is better than another conversion
  //   that is such a conversion.
  if (SCS1.isPointerConversionToBool() != SCS2.isPointerConversionToBool())
    return SCS2.isPointerConversionToBool()
             ? ImplicitConversionSequence::Better
             : ImplicitConversionSequence::Worse;

  // C++ [over.ics.rank]p4b2:
  //
  //   If class B is derived directly or indirectly from class A,
  //   conversion of B* to A* is better than conversion of B* to
  //   void*, and conversion of A* to void* is better than conversion
  //   of B* to void*.
  bool SCS1ConvertsToVoid
    = SCS1.isPointerConversionToVoidPointer(S.Context);
  bool SCS2ConvertsToVoid
    = SCS2.isPointerConversionToVoidPointer(S.Context);
  if (SCS1ConvertsToVoid != SCS2ConvertsToVoid) {
    // Exactly one of the conversion sequences is a conversion to
    // a void pointer; it's the worse conversion.
    return SCS2ConvertsToVoid ? ImplicitConversionSequence::Better
                              : ImplicitConversionSequence::Worse;
  } else if (!SCS1ConvertsToVoid && !SCS2ConvertsToVoid) {
    // Neither conversion sequence converts to a void pointer; compare
    // their derived-to-base conversions.
    if (ImplicitConversionSequence::CompareKind DerivedCK
          = CompareDerivedToBaseConversions(S, SCS1, SCS2))
      return DerivedCK;
  } else if (SCS1ConvertsToVoid && SCS2ConvertsToVoid &&
             !S.Context.hasSameType(SCS1.getFromType(), SCS2.getFromType())) {
    // Both conversion sequences are conversions to void
    // pointers. Compare the source types to determine if there's an
    // inheritance relationship in their sources.
    QualType FromType1 = SCS1.getFromType();
    QualType FromType2 = SCS2.getFromType();

    // Adjust the types we're converting from via the array-to-pointer
    // conversion, if we need to.
    if (SCS1.First == ICK_Array_To_Pointer)
      FromType1 = S.Context.getArrayDecayedType(FromType1);
    if (SCS2.First == ICK_Array_To_Pointer)
      FromType2 = S.Context.getArrayDecayedType(FromType2);

    QualType FromPointee1 = FromType1->getPointeeType().getUnqualifiedType();
    QualType FromPointee2 = FromType2->getPointeeType().getUnqualifiedType();

    if (S.IsDerivedFrom(FromPointee2, FromPointee1))
      return ImplicitConversionSequence::Better;
    else if (S.IsDerivedFrom(FromPointee1, FromPointee2))
      return ImplicitConversionSequence::Worse;

    // Objective-C++: If one interface is more specific than the
    // other, it is the better one.
    const ObjCObjectPointerType* FromObjCPtr1
      = FromType1->getAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType* FromObjCPtr2
      = FromType2->getAs<ObjCObjectPointerType>();
    if (FromObjCPtr1 && FromObjCPtr2) {
      bool AssignLeft = S.Context.canAssignObjCInterfaces(FromObjCPtr1, 
                                                          FromObjCPtr2);
      bool AssignRight = S.Context.canAssignObjCInterfaces(FromObjCPtr2, 
                                                           FromObjCPtr1);
      if (AssignLeft != AssignRight) {
        return AssignLeft? ImplicitConversionSequence::Better
                         : ImplicitConversionSequence::Worse;
      }
    }
  }

  // Compare based on qualification conversions (C++ 13.3.3.2p3,
  // bullet 3).
  if (ImplicitConversionSequence::CompareKind QualCK
        = CompareQualificationConversions(S, SCS1, SCS2))
    return QualCK;

  if (SCS1.ReferenceBinding && SCS2.ReferenceBinding) {
    // Check for a better reference binding based on the kind of bindings.
    if (isBetterReferenceBindingKind(SCS1, SCS2))
      return ImplicitConversionSequence::Better;
    else if (isBetterReferenceBindingKind(SCS2, SCS1))
      return ImplicitConversionSequence::Worse;

    // C++ [over.ics.rank]p3b4:
    //   -- S1 and S2 are reference bindings (8.5.3), and the types to
    //      which the references refer are the same type except for
    //      top-level cv-qualifiers, and the type to which the reference
    //      initialized by S2 refers is more cv-qualified than the type
    //      to which the reference initialized by S1 refers.
    QualType T1 = SCS1.getToType(2);
    QualType T2 = SCS2.getToType(2);
    T1 = S.Context.getCanonicalType(T1);
    T2 = S.Context.getCanonicalType(T2);
    Qualifiers T1Quals, T2Quals;
    QualType UnqualT1 = S.Context.getUnqualifiedArrayType(T1, T1Quals);
    QualType UnqualT2 = S.Context.getUnqualifiedArrayType(T2, T2Quals);
    if (UnqualT1 == UnqualT2) {
      // Objective-C++ ARC: If the references refer to objects with different
      // lifetimes, prefer bindings that don't change lifetime.
      if (SCS1.ObjCLifetimeConversionBinding != 
                                          SCS2.ObjCLifetimeConversionBinding) {
        return SCS1.ObjCLifetimeConversionBinding
                                           ? ImplicitConversionSequence::Worse
                                           : ImplicitConversionSequence::Better;
      }
      
      // If the type is an array type, promote the element qualifiers to the
      // type for comparison.
      if (isa<ArrayType>(T1) && T1Quals)
        T1 = S.Context.getQualifiedType(UnqualT1, T1Quals);
      if (isa<ArrayType>(T2) && T2Quals)
        T2 = S.Context.getQualifiedType(UnqualT2, T2Quals);
      if (T2.isMoreQualifiedThan(T1))
        return ImplicitConversionSequence::Better;
      else if (T1.isMoreQualifiedThan(T2))
        return ImplicitConversionSequence::Worse;      
    }
  }

  return ImplicitConversionSequence::Indistinguishable;
}

/// CompareQualificationConversions - Compares two standard conversion
/// sequences to determine whether they can be ranked based on their
/// qualification conversions (C++ 13.3.3.2p3 bullet 3).
ImplicitConversionSequence::CompareKind
CompareQualificationConversions(Sema &S,
                                const StandardConversionSequence& SCS1,
                                const StandardConversionSequence& SCS2) {
  // C++ 13.3.3.2p3:
  //  -- S1 and S2 differ only in their qualification conversion and
  //     yield similar types T1 and T2 (C++ 4.4), respectively, and the
  //     cv-qualification signature of type T1 is a proper subset of
  //     the cv-qualification signature of type T2, and S1 is not the
  //     deprecated string literal array-to-pointer conversion (4.2).
  if (SCS1.First != SCS2.First || SCS1.Second != SCS2.Second ||
      SCS1.Third != SCS2.Third || SCS1.Third != ICK_Qualification)
    return ImplicitConversionSequence::Indistinguishable;

  // FIXME: the example in the standard doesn't use a qualification
  // conversion (!)
  QualType T1 = SCS1.getToType(2);
  QualType T2 = SCS2.getToType(2);
  T1 = S.Context.getCanonicalType(T1);
  T2 = S.Context.getCanonicalType(T2);
  Qualifiers T1Quals, T2Quals;
  QualType UnqualT1 = S.Context.getUnqualifiedArrayType(T1, T1Quals);
  QualType UnqualT2 = S.Context.getUnqualifiedArrayType(T2, T2Quals);

  // If the types are the same, we won't learn anything by unwrapped
  // them.
  if (UnqualT1 == UnqualT2)
    return ImplicitConversionSequence::Indistinguishable;

  // If the type is an array type, promote the element qualifiers to the type
  // for comparison.
  if (isa<ArrayType>(T1) && T1Quals)
    T1 = S.Context.getQualifiedType(UnqualT1, T1Quals);
  if (isa<ArrayType>(T2) && T2Quals)
    T2 = S.Context.getQualifiedType(UnqualT2, T2Quals);

  ImplicitConversionSequence::CompareKind Result
    = ImplicitConversionSequence::Indistinguishable;
  
  // Objective-C++ ARC:
  //   Prefer qualification conversions not involving a change in lifetime
  //   to qualification conversions that do not change lifetime.
  if (SCS1.QualificationIncludesObjCLifetime != 
                                      SCS2.QualificationIncludesObjCLifetime) {
    Result = SCS1.QualificationIncludesObjCLifetime
               ? ImplicitConversionSequence::Worse
               : ImplicitConversionSequence::Better;
  }
  
  while (S.Context.UnwrapSimilarPointerTypes(T1, T2)) {
    // Within each iteration of the loop, we check the qualifiers to
    // determine if this still looks like a qualification
    // conversion. Then, if all is well, we unwrap one more level of
    // pointers or pointers-to-members and do it all again
    // until there are no more pointers or pointers-to-members left
    // to unwrap. This essentially mimics what
    // IsQualificationConversion does, but here we're checking for a
    // strict subset of qualifiers.
    if (T1.getCVRQualifiers() == T2.getCVRQualifiers())
      // The qualifiers are the same, so this doesn't tell us anything
      // about how the sequences rank.
      ;
    else if (T2.isMoreQualifiedThan(T1)) {
      // T1 has fewer qualifiers, so it could be the better sequence.
      if (Result == ImplicitConversionSequence::Worse)
        // Neither has qualifiers that are a subset of the other's
        // qualifiers.
        return ImplicitConversionSequence::Indistinguishable;

      Result = ImplicitConversionSequence::Better;
    } else if (T1.isMoreQualifiedThan(T2)) {
      // T2 has fewer qualifiers, so it could be the better sequence.
      if (Result == ImplicitConversionSequence::Better)
        // Neither has qualifiers that are a subset of the other's
        // qualifiers.
        return ImplicitConversionSequence::Indistinguishable;

      Result = ImplicitConversionSequence::Worse;
    } else {
      // Qualifiers are disjoint.
      return ImplicitConversionSequence::Indistinguishable;
    }

    // If the types after this point are equivalent, we're done.
    if (S.Context.hasSameUnqualifiedType(T1, T2))
      break;
  }

  // Check that the winning standard conversion sequence isn't using
  // the deprecated string literal array to pointer conversion.
  switch (Result) {
  case ImplicitConversionSequence::Better:
    if (SCS1.DeprecatedStringLiteralToCharPtr)
      Result = ImplicitConversionSequence::Indistinguishable;
    break;

  case ImplicitConversionSequence::Indistinguishable:
    break;

  case ImplicitConversionSequence::Worse:
    if (SCS2.DeprecatedStringLiteralToCharPtr)
      Result = ImplicitConversionSequence::Indistinguishable;
    break;
  }

  return Result;
}

/// CompareDerivedToBaseConversions - Compares two standard conversion
/// sequences to determine whether they can be ranked based on their
/// various kinds of derived-to-base conversions (C++
/// [over.ics.rank]p4b3).  As part of these checks, we also look at
/// conversions between Objective-C interface types.
ImplicitConversionSequence::CompareKind
CompareDerivedToBaseConversions(Sema &S,
                                const StandardConversionSequence& SCS1,
                                const StandardConversionSequence& SCS2) {
  QualType FromType1 = SCS1.getFromType();
  QualType ToType1 = SCS1.getToType(1);
  QualType FromType2 = SCS2.getFromType();
  QualType ToType2 = SCS2.getToType(1);

  // Adjust the types we're converting from via the array-to-pointer
  // conversion, if we need to.
  if (SCS1.First == ICK_Array_To_Pointer)
    FromType1 = S.Context.getArrayDecayedType(FromType1);
  if (SCS2.First == ICK_Array_To_Pointer)
    FromType2 = S.Context.getArrayDecayedType(FromType2);

  // Canonicalize all of the types.
  FromType1 = S.Context.getCanonicalType(FromType1);
  ToType1 = S.Context.getCanonicalType(ToType1);
  FromType2 = S.Context.getCanonicalType(FromType2);
  ToType2 = S.Context.getCanonicalType(ToType2);

  // C++ [over.ics.rank]p4b3:
  //
  //   If class B is derived directly or indirectly from class A and
  //   class C is derived directly or indirectly from B,
  //
  // Compare based on pointer conversions.
  if (SCS1.Second == ICK_Pointer_Conversion &&
      SCS2.Second == ICK_Pointer_Conversion &&
      /*FIXME: Remove if Objective-C id conversions get their own rank*/
      FromType1->isPointerType() && FromType2->isPointerType() &&
      ToType1->isPointerType() && ToType2->isPointerType()) {
    QualType FromPointee1
      = FromType1->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType ToPointee1
      = ToType1->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType FromPointee2
      = FromType2->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType ToPointee2
      = ToType2->getAs<PointerType>()->getPointeeType().getUnqualifiedType();

    //   -- conversion of C* to B* is better than conversion of C* to A*,
    if (FromPointee1 == FromPointee2 && ToPointee1 != ToPointee2) {
      if (S.IsDerivedFrom(ToPointee1, ToPointee2))
        return ImplicitConversionSequence::Better;
      else if (S.IsDerivedFrom(ToPointee2, ToPointee1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- conversion of B* to A* is better than conversion of C* to A*,
    if (FromPointee1 != FromPointee2 && ToPointee1 == ToPointee2) {
      if (S.IsDerivedFrom(FromPointee2, FromPointee1))
        return ImplicitConversionSequence::Better;
      else if (S.IsDerivedFrom(FromPointee1, FromPointee2))
        return ImplicitConversionSequence::Worse;
    }
  } else if (SCS1.Second == ICK_Pointer_Conversion &&
             SCS2.Second == ICK_Pointer_Conversion) {
    const ObjCObjectPointerType *FromPtr1
      = FromType1->getAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType *FromPtr2
      = FromType2->getAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType *ToPtr1
      = ToType1->getAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType *ToPtr2
      = ToType2->getAs<ObjCObjectPointerType>();
    
    if (FromPtr1 && FromPtr2 && ToPtr1 && ToPtr2) {
      // Apply the same conversion ranking rules for Objective-C pointer types
      // that we do for C++ pointers to class types. However, we employ the
      // Objective-C pseudo-subtyping relationship used for assignment of
      // Objective-C pointer types.
      bool FromAssignLeft
        = S.Context.canAssignObjCInterfaces(FromPtr1, FromPtr2);
      bool FromAssignRight
        = S.Context.canAssignObjCInterfaces(FromPtr2, FromPtr1);
      bool ToAssignLeft
        = S.Context.canAssignObjCInterfaces(ToPtr1, ToPtr2);
      bool ToAssignRight
        = S.Context.canAssignObjCInterfaces(ToPtr2, ToPtr1);
      
      // A conversion to an a non-id object pointer type or qualified 'id' 
      // type is better than a conversion to 'id'.
      if (ToPtr1->isObjCIdType() &&
          (ToPtr2->isObjCQualifiedIdType() || ToPtr2->getInterfaceDecl()))
        return ImplicitConversionSequence::Worse;
      if (ToPtr2->isObjCIdType() &&
          (ToPtr1->isObjCQualifiedIdType() || ToPtr1->getInterfaceDecl()))
        return ImplicitConversionSequence::Better;
      
      // A conversion to a non-id object pointer type is better than a 
      // conversion to a qualified 'id' type 
      if (ToPtr1->isObjCQualifiedIdType() && ToPtr2->getInterfaceDecl())
        return ImplicitConversionSequence::Worse;
      if (ToPtr2->isObjCQualifiedIdType() && ToPtr1->getInterfaceDecl())
        return ImplicitConversionSequence::Better;
  
      // A conversion to an a non-Class object pointer type or qualified 'Class' 
      // type is better than a conversion to 'Class'.
      if (ToPtr1->isObjCClassType() &&
          (ToPtr2->isObjCQualifiedClassType() || ToPtr2->getInterfaceDecl()))
        return ImplicitConversionSequence::Worse;
      if (ToPtr2->isObjCClassType() &&
          (ToPtr1->isObjCQualifiedClassType() || ToPtr1->getInterfaceDecl()))
        return ImplicitConversionSequence::Better;
      
      // A conversion to a non-Class object pointer type is better than a 
      // conversion to a qualified 'Class' type.
      if (ToPtr1->isObjCQualifiedClassType() && ToPtr2->getInterfaceDecl())
        return ImplicitConversionSequence::Worse;
      if (ToPtr2->isObjCQualifiedClassType() && ToPtr1->getInterfaceDecl())
        return ImplicitConversionSequence::Better;

      //   -- "conversion of C* to B* is better than conversion of C* to A*,"
      if (S.Context.hasSameType(FromType1, FromType2) && 
          !FromPtr1->isObjCIdType() && !FromPtr1->isObjCClassType() &&
          (ToAssignLeft != ToAssignRight))
        return ToAssignLeft? ImplicitConversionSequence::Worse
                           : ImplicitConversionSequence::Better;

      //   -- "conversion of B* to A* is better than conversion of C* to A*,"
      if (S.Context.hasSameUnqualifiedType(ToType1, ToType2) &&
          (FromAssignLeft != FromAssignRight))
        return FromAssignLeft? ImplicitConversionSequence::Better
        : ImplicitConversionSequence::Worse;
    }
  }
  
  // Ranking of member-pointer types.
  if (SCS1.Second == ICK_Pointer_Member && SCS2.Second == ICK_Pointer_Member &&
      FromType1->isMemberPointerType() && FromType2->isMemberPointerType() &&
      ToType1->isMemberPointerType() && ToType2->isMemberPointerType()) {
    const MemberPointerType * FromMemPointer1 =
                                        FromType1->getAs<MemberPointerType>();
    const MemberPointerType * ToMemPointer1 =
                                          ToType1->getAs<MemberPointerType>();
    const MemberPointerType * FromMemPointer2 =
                                          FromType2->getAs<MemberPointerType>();
    const MemberPointerType * ToMemPointer2 =
                                          ToType2->getAs<MemberPointerType>();
    const Type *FromPointeeType1 = FromMemPointer1->getClass();
    const Type *ToPointeeType1 = ToMemPointer1->getClass();
    const Type *FromPointeeType2 = FromMemPointer2->getClass();
    const Type *ToPointeeType2 = ToMemPointer2->getClass();
    QualType FromPointee1 = QualType(FromPointeeType1, 0).getUnqualifiedType();
    QualType ToPointee1 = QualType(ToPointeeType1, 0).getUnqualifiedType();
    QualType FromPointee2 = QualType(FromPointeeType2, 0).getUnqualifiedType();
    QualType ToPointee2 = QualType(ToPointeeType2, 0).getUnqualifiedType();
    // conversion of A::* to B::* is better than conversion of A::* to C::*,
    if (FromPointee1 == FromPointee2 && ToPointee1 != ToPointee2) {
      if (S.IsDerivedFrom(ToPointee1, ToPointee2))
        return ImplicitConversionSequence::Worse;
      else if (S.IsDerivedFrom(ToPointee2, ToPointee1))
        return ImplicitConversionSequence::Better;
    }
    // conversion of B::* to C::* is better than conversion of A::* to C::*
    if (ToPointee1 == ToPointee2 && FromPointee1 != FromPointee2) {
      if (S.IsDerivedFrom(FromPointee1, FromPointee2))
        return ImplicitConversionSequence::Better;
      else if (S.IsDerivedFrom(FromPointee2, FromPointee1))
        return ImplicitConversionSequence::Worse;
    }
  }

  if (SCS1.Second == ICK_Derived_To_Base) {
    //   -- conversion of C to B is better than conversion of C to A,
    //   -- binding of an expression of type C to a reference of type
    //      B& is better than binding an expression of type C to a
    //      reference of type A&,
    if (S.Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        !S.Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (S.IsDerivedFrom(ToType1, ToType2))
        return ImplicitConversionSequence::Better;
      else if (S.IsDerivedFrom(ToType2, ToType1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- conversion of B to A is better than conversion of C to A.
    //   -- binding of an expression of type B to a reference of type
    //      A& is better than binding an expression of type C to a
    //      reference of type A&,
    if (!S.Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        S.Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (S.IsDerivedFrom(FromType2, FromType1))
        return ImplicitConversionSequence::Better;
      else if (S.IsDerivedFrom(FromType1, FromType2))
        return ImplicitConversionSequence::Worse;
    }
  }

  return ImplicitConversionSequence::Indistinguishable;
}

/// CompareReferenceRelationship - Compare the two types T1 and T2 to
/// determine whether they are reference-related,
/// reference-compatible, reference-compatible with added
/// qualification, or incompatible, for use in C++ initialization by
/// reference (C++ [dcl.ref.init]p4). Neither type can be a reference
/// type, and the first type (T1) is the pointee type of the reference
/// type being initialized.
Sema::ReferenceCompareResult
Sema::CompareReferenceRelationship(SourceLocation Loc,
                                   QualType OrigT1, QualType OrigT2,
                                   bool &DerivedToBase,
                                   bool &ObjCConversion,
                                   bool &ObjCLifetimeConversion) {
  assert(!OrigT1->isReferenceType() &&
    "T1 must be the pointee type of the reference type");
  assert(!OrigT2->isReferenceType() && "T2 cannot be a reference type");

  QualType T1 = Context.getCanonicalType(OrigT1);
  QualType T2 = Context.getCanonicalType(OrigT2);
  Qualifiers T1Quals, T2Quals;
  QualType UnqualT1 = Context.getUnqualifiedArrayType(T1, T1Quals);
  QualType UnqualT2 = Context.getUnqualifiedArrayType(T2, T2Quals);

  // C++ [dcl.init.ref]p4:
  //   Given types "cv1 T1" and "cv2 T2," "cv1 T1" is
  //   reference-related to "cv2 T2" if T1 is the same type as T2, or
  //   T1 is a base class of T2.
  DerivedToBase = false;
  ObjCConversion = false;
  ObjCLifetimeConversion = false;
  if (UnqualT1 == UnqualT2) {
    // Nothing to do.
  } else if (!RequireCompleteType(Loc, OrigT2, PDiag()) &&
           IsDerivedFrom(UnqualT2, UnqualT1))
    DerivedToBase = true;
  else if (UnqualT1->isObjCObjectOrInterfaceType() &&
           UnqualT2->isObjCObjectOrInterfaceType() &&
           Context.canBindObjCObjectType(UnqualT1, UnqualT2))
    ObjCConversion = true;
  else
    return Ref_Incompatible;

  // At this point, we know that T1 and T2 are reference-related (at
  // least).

  // If the type is an array type, promote the element qualifiers to the type
  // for comparison.
  if (isa<ArrayType>(T1) && T1Quals)
    T1 = Context.getQualifiedType(UnqualT1, T1Quals);
  if (isa<ArrayType>(T2) && T2Quals)
    T2 = Context.getQualifiedType(UnqualT2, T2Quals);

  // C++ [dcl.init.ref]p4:
  //   "cv1 T1" is reference-compatible with "cv2 T2" if T1 is
  //   reference-related to T2 and cv1 is the same cv-qualification
  //   as, or greater cv-qualification than, cv2. For purposes of
  //   overload resolution, cases for which cv1 is greater
  //   cv-qualification than cv2 are identified as
  //   reference-compatible with added qualification (see 13.3.3.2).
  //
  // Note that we also require equivalence of Objective-C GC and address-space
  // qualifiers when performing these computations, so that e.g., an int in
  // address space 1 is not reference-compatible with an int in address
  // space 2.
  if (T1Quals.getObjCLifetime() != T2Quals.getObjCLifetime() &&
      T1Quals.compatiblyIncludesObjCLifetime(T2Quals)) {
    T1Quals.removeObjCLifetime();
    T2Quals.removeObjCLifetime();    
    ObjCLifetimeConversion = true;
  }
    
  if (T1Quals == T2Quals)
    return Ref_Compatible;
  else if (T1Quals.compatiblyIncludes(T2Quals))
    return Ref_Compatible_With_Added_Qualification;
  else
    return Ref_Related;
}

/// \brief Look for a user-defined conversion to an value reference-compatible
///        with DeclType. Return true if something definite is found.
static bool
FindConversionForRefInit(Sema &S, ImplicitConversionSequence &ICS,
                         QualType DeclType, SourceLocation DeclLoc,
                         Expr *Init, QualType T2, bool AllowRvalues,
                         bool AllowExplicit) {
  assert(T2->isRecordType() && "Can only find conversions of record types.");
  CXXRecordDecl *T2RecordDecl
    = dyn_cast<CXXRecordDecl>(T2->getAs<RecordType>()->getDecl());

  OverloadCandidateSet CandidateSet(DeclLoc);
  const UnresolvedSetImpl *Conversions
    = T2RecordDecl->getVisibleConversionFunctions();
  for (UnresolvedSetImpl::iterator I = Conversions->begin(),
         E = Conversions->end(); I != E; ++I) {
    NamedDecl *D = *I;
    CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(D->getDeclContext());
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();

    FunctionTemplateDecl *ConvTemplate
      = dyn_cast<FunctionTemplateDecl>(D);
    CXXConversionDecl *Conv;
    if (ConvTemplate)
      Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
    else
      Conv = cast<CXXConversionDecl>(D);

    // If this is an explicit conversion, and we're not allowed to consider
    // explicit conversions, skip it.
    if (!AllowExplicit && Conv->isExplicit())
      continue;

    if (AllowRvalues) {
      bool DerivedToBase = false;
      bool ObjCConversion = false;
      bool ObjCLifetimeConversion = false;
      if (!ConvTemplate &&
          S.CompareReferenceRelationship(
            DeclLoc,
            Conv->getConversionType().getNonReferenceType()
              .getUnqualifiedType(),
            DeclType.getNonReferenceType().getUnqualifiedType(),
            DerivedToBase, ObjCConversion, ObjCLifetimeConversion) ==
          Sema::Ref_Incompatible)
        continue;
    } else {
      // If the conversion function doesn't return a reference type,
      // it can't be considered for this conversion. An rvalue reference
      // is only acceptable if its referencee is a function type.

      const ReferenceType *RefType =
        Conv->getConversionType()->getAs<ReferenceType>();
      if (!RefType ||
          (!RefType->isLValueReferenceType() &&
           !RefType->getPointeeType()->isFunctionType()))
        continue;
    }

    if (ConvTemplate)
      S.AddTemplateConversionCandidate(ConvTemplate, I.getPair(), ActingDC,
                                       Init, DeclType, CandidateSet);
    else
      S.AddConversionCandidate(Conv, I.getPair(), ActingDC, Init,
                               DeclType, CandidateSet);
  }

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(S, DeclLoc, Best, true)) {
  case OR_Success:
    // C++ [over.ics.ref]p1:
    //
    //   [...] If the parameter binds directly to the result of
    //   applying a conversion function to the argument
    //   expression, the implicit conversion sequence is a
    //   user-defined conversion sequence (13.3.3.1.2), with the
    //   second standard conversion sequence either an identity
    //   conversion or, if the conversion function returns an
    //   entity of a type that is a derived class of the parameter
    //   type, a derived-to-base Conversion.
    if (!Best->FinalConversion.DirectBinding)
      return false;

    if (Best->Function)
      S.MarkDeclarationReferenced(DeclLoc, Best->Function);
    ICS.setUserDefined();
    ICS.UserDefined.Before = Best->Conversions[0].Standard;
    ICS.UserDefined.After = Best->FinalConversion;
    ICS.UserDefined.ConversionFunction = Best->Function;
    ICS.UserDefined.FoundConversionFunction = Best->FoundDecl.getDecl();
    ICS.UserDefined.EllipsisConversion = false;
    assert(ICS.UserDefined.After.ReferenceBinding &&
           ICS.UserDefined.After.DirectBinding &&
           "Expected a direct reference binding!");
    return true;

  case OR_Ambiguous:
    ICS.setAmbiguous();
    for (OverloadCandidateSet::iterator Cand = CandidateSet.begin();
         Cand != CandidateSet.end(); ++Cand)
      if (Cand->Viable)
        ICS.Ambiguous.addConversion(Cand->Function);
    return true;

  case OR_No_Viable_Function:
  case OR_Deleted:
    // There was no suitable conversion, or we found a deleted
    // conversion; continue with other checks.
    return false;
  }

  return false;
}

/// \brief Compute an implicit conversion sequence for reference
/// initialization.
static ImplicitConversionSequence
TryReferenceInit(Sema &S, Expr *&Init, QualType DeclType,
                 SourceLocation DeclLoc,
                 bool SuppressUserConversions,
                 bool AllowExplicit) {
  assert(DeclType->isReferenceType() && "Reference init needs a reference");

  // Most paths end in a failed conversion.
  ImplicitConversionSequence ICS;
  ICS.setBad(BadConversionSequence::no_conversion, Init, DeclType);

  QualType T1 = DeclType->getAs<ReferenceType>()->getPointeeType();
  QualType T2 = Init->getType();

  // If the initializer is the address of an overloaded function, try
  // to resolve the overloaded function. If all goes well, T2 is the
  // type of the resulting function.
  if (S.Context.getCanonicalType(T2) == S.Context.OverloadTy) {
    DeclAccessPair Found;
    if (FunctionDecl *Fn = S.ResolveAddressOfOverloadedFunction(Init, DeclType,
                                                                false, Found))
      T2 = Fn->getType();
  }

  // Compute some basic properties of the types and the initializer.
  bool isRValRef = DeclType->isRValueReferenceType();
  bool DerivedToBase = false;
  bool ObjCConversion = false;
  bool ObjCLifetimeConversion = false;
  Expr::Classification InitCategory = Init->Classify(S.Context);
  Sema::ReferenceCompareResult RefRelationship
    = S.CompareReferenceRelationship(DeclLoc, T1, T2, DerivedToBase,
                                     ObjCConversion, ObjCLifetimeConversion);


  // C++0x [dcl.init.ref]p5:
  //   A reference to type "cv1 T1" is initialized by an expression
  //   of type "cv2 T2" as follows:

  //     -- If reference is an lvalue reference and the initializer expression
  if (!isRValRef) {
    //     -- is an lvalue (but is not a bit-field), and "cv1 T1" is
    //        reference-compatible with "cv2 T2," or
    //
    // Per C++ [over.ics.ref]p4, we don't check the bit-field property here.
    if (InitCategory.isLValue() &&
        RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification) {
      // C++ [over.ics.ref]p1:
      //   When a parameter of reference type binds directly (8.5.3)
      //   to an argument expression, the implicit conversion sequence
      //   is the identity conversion, unless the argument expression
      //   has a type that is a derived class of the parameter type,
      //   in which case the implicit conversion sequence is a
      //   derived-to-base Conversion (13.3.3.1).
      ICS.setStandard();
      ICS.Standard.First = ICK_Identity;
      ICS.Standard.Second = DerivedToBase? ICK_Derived_To_Base
                         : ObjCConversion? ICK_Compatible_Conversion
                         : ICK_Identity;
      ICS.Standard.Third = ICK_Identity;
      ICS.Standard.FromTypePtr = T2.getAsOpaquePtr();
      ICS.Standard.setToType(0, T2);
      ICS.Standard.setToType(1, T1);
      ICS.Standard.setToType(2, T1);
      ICS.Standard.ReferenceBinding = true;
      ICS.Standard.DirectBinding = true;
      ICS.Standard.IsLvalueReference = !isRValRef;
      ICS.Standard.BindsToFunctionLvalue = T2->isFunctionType();
      ICS.Standard.BindsToRvalue = false;
      ICS.Standard.BindsImplicitObjectArgumentWithoutRefQualifier = false;
      ICS.Standard.ObjCLifetimeConversionBinding = ObjCLifetimeConversion;
      ICS.Standard.CopyConstructor = 0;

      // Nothing more to do: the inaccessibility/ambiguity check for
      // derived-to-base conversions is suppressed when we're
      // computing the implicit conversion sequence (C++
      // [over.best.ics]p2).
      return ICS;
    }

    //       -- has a class type (i.e., T2 is a class type), where T1 is
    //          not reference-related to T2, and can be implicitly
    //          converted to an lvalue of type "cv3 T3," where "cv1 T1"
    //          is reference-compatible with "cv3 T3" 92) (this
    //          conversion is selected by enumerating the applicable
    //          conversion functions (13.3.1.6) and choosing the best
    //          one through overload resolution (13.3)),
    if (!SuppressUserConversions && T2->isRecordType() &&
        !S.RequireCompleteType(DeclLoc, T2, 0) &&
        RefRelationship == Sema::Ref_Incompatible) {
      if (FindConversionForRefInit(S, ICS, DeclType, DeclLoc,
                                   Init, T2, /*AllowRvalues=*/false,
                                   AllowExplicit))
        return ICS;
    }
  }

  //     -- Otherwise, the reference shall be an lvalue reference to a
  //        non-volatile const type (i.e., cv1 shall be const), or the reference
  //        shall be an rvalue reference.
  //
  // We actually handle one oddity of C++ [over.ics.ref] at this
  // point, which is that, due to p2 (which short-circuits reference
  // binding by only attempting a simple conversion for non-direct
  // bindings) and p3's strange wording, we allow a const volatile
  // reference to bind to an rvalue. Hence the check for the presence
  // of "const" rather than checking for "const" being the only
  // qualifier.
  // This is also the point where rvalue references and lvalue inits no longer
  // go together.
  if (!isRValRef && !T1.isConstQualified())
    return ICS;

  //       -- If the initializer expression
  //
  //            -- is an xvalue, class prvalue, array prvalue or function
  //               lvalue and "cv1 T1" is reference-compatible with "cv2 T2", or
  if (RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification &&
      (InitCategory.isXValue() ||
      (InitCategory.isPRValue() && (T2->isRecordType() || T2->isArrayType())) ||
      (InitCategory.isLValue() && T2->isFunctionType()))) {
    ICS.setStandard();
    ICS.Standard.First = ICK_Identity;
    ICS.Standard.Second = DerivedToBase? ICK_Derived_To_Base
                      : ObjCConversion? ICK_Compatible_Conversion
                      : ICK_Identity;
    ICS.Standard.Third = ICK_Identity;
    ICS.Standard.FromTypePtr = T2.getAsOpaquePtr();
    ICS.Standard.setToType(0, T2);
    ICS.Standard.setToType(1, T1);
    ICS.Standard.setToType(2, T1);
    ICS.Standard.ReferenceBinding = true;
    // In C++0x, this is always a direct binding. In C++98/03, it's a direct
    // binding unless we're binding to a class prvalue.
    // Note: Although xvalues wouldn't normally show up in C++98/03 code, we
    // allow the use of rvalue references in C++98/03 for the benefit of
    // standard library implementors; therefore, we need the xvalue check here.
    ICS.Standard.DirectBinding =
      S.getLangOptions().CPlusPlus0x ||
      (InitCategory.isPRValue() && !T2->isRecordType());
    ICS.Standard.IsLvalueReference = !isRValRef;
    ICS.Standard.BindsToFunctionLvalue = T2->isFunctionType();
    ICS.Standard.BindsToRvalue = InitCategory.isRValue();
    ICS.Standard.BindsImplicitObjectArgumentWithoutRefQualifier = false;
    ICS.Standard.ObjCLifetimeConversionBinding = ObjCLifetimeConversion;
    ICS.Standard.CopyConstructor = 0;
    return ICS;
  }

  //            -- has a class type (i.e., T2 is a class type), where T1 is not
  //               reference-related to T2, and can be implicitly converted to
  //               an xvalue, class prvalue, or function lvalue of type
  //               "cv3 T3", where "cv1 T1" is reference-compatible with
  //               "cv3 T3",
  //
  //          then the reference is bound to the value of the initializer
  //          expression in the first case and to the result of the conversion
  //          in the second case (or, in either case, to an appropriate base
  //          class subobject).
  if (!SuppressUserConversions && RefRelationship == Sema::Ref_Incompatible &&
      T2->isRecordType() && !S.RequireCompleteType(DeclLoc, T2, 0) &&
      FindConversionForRefInit(S, ICS, DeclType, DeclLoc,
                               Init, T2, /*AllowRvalues=*/true,
                               AllowExplicit)) {
    // In the second case, if the reference is an rvalue reference
    // and the second standard conversion sequence of the
    // user-defined conversion sequence includes an lvalue-to-rvalue
    // conversion, the program is ill-formed.
    if (ICS.isUserDefined() && isRValRef &&
        ICS.UserDefined.After.First == ICK_Lvalue_To_Rvalue)
      ICS.setBad(BadConversionSequence::no_conversion, Init, DeclType);

    return ICS;
  }

  //       -- Otherwise, a temporary of type "cv1 T1" is created and
  //          initialized from the initializer expression using the
  //          rules for a non-reference copy initialization (8.5). The
  //          reference is then bound to the temporary. If T1 is
  //          reference-related to T2, cv1 must be the same
  //          cv-qualification as, or greater cv-qualification than,
  //          cv2; otherwise, the program is ill-formed.
  if (RefRelationship == Sema::Ref_Related) {
    // If cv1 == cv2 or cv1 is a greater cv-qualified than cv2, then
    // we would be reference-compatible or reference-compatible with
    // added qualification. But that wasn't the case, so the reference
    // initialization fails.
    //
    // Note that we only want to check address spaces and cvr-qualifiers here.
    // ObjC GC and lifetime qualifiers aren't important.
    Qualifiers T1Quals = T1.getQualifiers();
    Qualifiers T2Quals = T2.getQualifiers();
    T1Quals.removeObjCGCAttr();
    T1Quals.removeObjCLifetime();
    T2Quals.removeObjCGCAttr();
    T2Quals.removeObjCLifetime();
    if (!T1Quals.compatiblyIncludes(T2Quals))
      return ICS;
  }

  // If at least one of the types is a class type, the types are not
  // related, and we aren't allowed any user conversions, the
  // reference binding fails. This case is important for breaking
  // recursion, since TryImplicitConversion below will attempt to
  // create a temporary through the use of a copy constructor.
  if (SuppressUserConversions && RefRelationship == Sema::Ref_Incompatible &&
      (T1->isRecordType() || T2->isRecordType()))
    return ICS;

  // If T1 is reference-related to T2 and the reference is an rvalue
  // reference, the initializer expression shall not be an lvalue.
  if (RefRelationship >= Sema::Ref_Related &&
      isRValRef && Init->Classify(S.Context).isLValue())
    return ICS;

  // C++ [over.ics.ref]p2:
  //   When a parameter of reference type is not bound directly to
  //   an argument expression, the conversion sequence is the one
  //   required to convert the argument expression to the
  //   underlying type of the reference according to
  //   13.3.3.1. Conceptually, this conversion sequence corresponds
  //   to copy-initializing a temporary of the underlying type with
  //   the argument expression. Any difference in top-level
  //   cv-qualification is subsumed by the initialization itself
  //   and does not constitute a conversion.
  ICS = TryImplicitConversion(S, Init, T1, SuppressUserConversions,
                              /*AllowExplicit=*/false,
                              /*InOverloadResolution=*/false,
                              /*CStyle=*/false,
                              /*AllowObjCWritebackConversion=*/false);

  // Of course, that's still a reference binding.
  if (ICS.isStandard()) {
    ICS.Standard.ReferenceBinding = true;
    ICS.Standard.IsLvalueReference = !isRValRef;
    ICS.Standard.BindsToFunctionLvalue = T2->isFunctionType();
    ICS.Standard.BindsToRvalue = true;
    ICS.Standard.BindsImplicitObjectArgumentWithoutRefQualifier = false;
    ICS.Standard.ObjCLifetimeConversionBinding = false;
  } else if (ICS.isUserDefined()) {
    ICS.UserDefined.After.ReferenceBinding = true;
    ICS.Standard.IsLvalueReference = !isRValRef;
    ICS.Standard.BindsToFunctionLvalue = T2->isFunctionType();
    ICS.Standard.BindsToRvalue = true;
    ICS.Standard.BindsImplicitObjectArgumentWithoutRefQualifier = false;
    ICS.Standard.ObjCLifetimeConversionBinding = false;
  }

  return ICS;
}

/// TryCopyInitialization - Try to copy-initialize a value of type
/// ToType from the expression From. Return the implicit conversion
/// sequence required to pass this argument, which may be a bad
/// conversion sequence (meaning that the argument cannot be passed to
/// a parameter of this type). If @p SuppressUserConversions, then we
/// do not permit any user-defined conversion sequences.
static ImplicitConversionSequence
TryCopyInitialization(Sema &S, Expr *From, QualType ToType,
                      bool SuppressUserConversions,
                      bool InOverloadResolution,
                      bool AllowObjCWritebackConversion) {
  if (ToType->isReferenceType())
    return TryReferenceInit(S, From, ToType,
                            /*FIXME:*/From->getLocStart(),
                            SuppressUserConversions,
                            /*AllowExplicit=*/false);

  return TryImplicitConversion(S, From, ToType,
                               SuppressUserConversions,
                               /*AllowExplicit=*/false,
                               InOverloadResolution,
                               /*CStyle=*/false,
                               AllowObjCWritebackConversion);
}

/// TryObjectArgumentInitialization - Try to initialize the object
/// parameter of the given member function (@c Method) from the
/// expression @p From.
static ImplicitConversionSequence
TryObjectArgumentInitialization(Sema &S, QualType OrigFromType,
                                Expr::Classification FromClassification,
                                CXXMethodDecl *Method,
                                CXXRecordDecl *ActingContext) {
  QualType ClassType = S.Context.getTypeDeclType(ActingContext);
  // [class.dtor]p2: A destructor can be invoked for a const, volatile or
  //                 const volatile object.
  unsigned Quals = isa<CXXDestructorDecl>(Method) ?
    Qualifiers::Const | Qualifiers::Volatile : Method->getTypeQualifiers();
  QualType ImplicitParamType =  S.Context.getCVRQualifiedType(ClassType, Quals);

  // Set up the conversion sequence as a "bad" conversion, to allow us
  // to exit early.
  ImplicitConversionSequence ICS;

  // We need to have an object of class type.
  QualType FromType = OrigFromType;
  if (const PointerType *PT = FromType->getAs<PointerType>()) {
    FromType = PT->getPointeeType();

    // When we had a pointer, it's implicitly dereferenced, so we
    // better have an lvalue.
    assert(FromClassification.isLValue());
  }

  assert(FromType->isRecordType());

  // C++0x [over.match.funcs]p4:
  //   For non-static member functions, the type of the implicit object
  //   parameter is
  //
  //     - "lvalue reference to cv X" for functions declared without a
  //        ref-qualifier or with the & ref-qualifier
  //     - "rvalue reference to cv X" for functions declared with the &&
  //        ref-qualifier
  //
  // where X is the class of which the function is a member and cv is the
  // cv-qualification on the member function declaration.
  //
  // However, when finding an implicit conversion sequence for the argument, we
  // are not allowed to create temporaries or perform user-defined conversions
  // (C++ [over.match.funcs]p5). We perform a simplified version of
  // reference binding here, that allows class rvalues to bind to
  // non-constant references.

  // First check the qualifiers.
  QualType FromTypeCanon = S.Context.getCanonicalType(FromType);
  if (ImplicitParamType.getCVRQualifiers()
                                    != FromTypeCanon.getLocalCVRQualifiers() &&
      !ImplicitParamType.isAtLeastAsQualifiedAs(FromTypeCanon)) {
    ICS.setBad(BadConversionSequence::bad_qualifiers,
               OrigFromType, ImplicitParamType);
    return ICS;
  }

  // Check that we have either the same type or a derived type. It
  // affects the conversion rank.
  QualType ClassTypeCanon = S.Context.getCanonicalType(ClassType);
  ImplicitConversionKind SecondKind;
  if (ClassTypeCanon == FromTypeCanon.getLocalUnqualifiedType()) {
    SecondKind = ICK_Identity;
  } else if (S.IsDerivedFrom(FromType, ClassType))
    SecondKind = ICK_Derived_To_Base;
  else {
    ICS.setBad(BadConversionSequence::unrelated_class,
               FromType, ImplicitParamType);
    return ICS;
  }

  // Check the ref-qualifier.
  switch (Method->getRefQualifier()) {
  case RQ_None:
    // Do nothing; we don't care about lvalueness or rvalueness.
    break;

  case RQ_LValue:
    if (!FromClassification.isLValue() && Quals != Qualifiers::Const) {
      // non-const lvalue reference cannot bind to an rvalue
      ICS.setBad(BadConversionSequence::lvalue_ref_to_rvalue, FromType,
                 ImplicitParamType);
      return ICS;
    }
    break;

  case RQ_RValue:
    if (!FromClassification.isRValue()) {
      // rvalue reference cannot bind to an lvalue
      ICS.setBad(BadConversionSequence::rvalue_ref_to_lvalue, FromType,
                 ImplicitParamType);
      return ICS;
    }
    break;
  }

  // Success. Mark this as a reference binding.
  ICS.setStandard();
  ICS.Standard.setAsIdentityConversion();
  ICS.Standard.Second = SecondKind;
  ICS.Standard.setFromType(FromType);
  ICS.Standard.setAllToTypes(ImplicitParamType);
  ICS.Standard.ReferenceBinding = true;
  ICS.Standard.DirectBinding = true;
  ICS.Standard.IsLvalueReference = Method->getRefQualifier() != RQ_RValue;
  ICS.Standard.BindsToFunctionLvalue = false;
  ICS.Standard.BindsToRvalue = FromClassification.isRValue();
  ICS.Standard.BindsImplicitObjectArgumentWithoutRefQualifier
    = (Method->getRefQualifier() == RQ_None);
  return ICS;
}

/// PerformObjectArgumentInitialization - Perform initialization of
/// the implicit object parameter for the given Method with the given
/// expression.
ExprResult
Sema::PerformObjectArgumentInitialization(Expr *From,
                                          NestedNameSpecifier *Qualifier,
                                          NamedDecl *FoundDecl,
                                          CXXMethodDecl *Method) {
  QualType FromRecordType, DestType;
  QualType ImplicitParamRecordType  =
    Method->getThisType(Context)->getAs<PointerType>()->getPointeeType();

  Expr::Classification FromClassification;
  if (const PointerType *PT = From->getType()->getAs<PointerType>()) {
    FromRecordType = PT->getPointeeType();
    DestType = Method->getThisType(Context);
    FromClassification = Expr::Classification::makeSimpleLValue();
  } else {
    FromRecordType = From->getType();
    DestType = ImplicitParamRecordType;
    FromClassification = From->Classify(Context);
  }

  // Note that we always use the true parent context when performing
  // the actual argument initialization.
  ImplicitConversionSequence ICS
    = TryObjectArgumentInitialization(*this, From->getType(), FromClassification,
                                      Method, Method->getParent());
  if (ICS.isBad()) {
    if (ICS.Bad.Kind == BadConversionSequence::bad_qualifiers) {
      Qualifiers FromQs = FromRecordType.getQualifiers();
      Qualifiers ToQs = DestType.getQualifiers();
      unsigned CVR = FromQs.getCVRQualifiers() & ~ToQs.getCVRQualifiers();
      if (CVR) {
        Diag(From->getSourceRange().getBegin(),
             diag::err_member_function_call_bad_cvr)
          << Method->getDeclName() << FromRecordType << (CVR - 1)
          << From->getSourceRange();
        Diag(Method->getLocation(), diag::note_previous_decl)
          << Method->getDeclName();
        return ExprError();
      }
    }

    return Diag(From->getSourceRange().getBegin(),
                diag::err_implicit_object_parameter_init)
       << ImplicitParamRecordType << FromRecordType << From->getSourceRange();
  }

  if (ICS.Standard.Second == ICK_Derived_To_Base) {
    ExprResult FromRes =
      PerformObjectMemberConversion(From, Qualifier, FoundDecl, Method);
    if (FromRes.isInvalid())
      return ExprError();
    From = FromRes.take();
  }

  if (!Context.hasSameType(From->getType(), DestType))
    From = ImpCastExprToType(From, DestType, CK_NoOp,
                      From->getType()->isPointerType() ? VK_RValue : VK_LValue).take();
  return Owned(From);
}

/// TryContextuallyConvertToBool - Attempt to contextually convert the
/// expression From to bool (C++0x [conv]p3).
static ImplicitConversionSequence
TryContextuallyConvertToBool(Sema &S, Expr *From) {
  // FIXME: This is pretty broken.
  return TryImplicitConversion(S, From, S.Context.BoolTy,
                               // FIXME: Are these flags correct?
                               /*SuppressUserConversions=*/false,
                               /*AllowExplicit=*/true,
                               /*InOverloadResolution=*/false,
                               /*CStyle=*/false,
                               /*AllowObjCWritebackConversion=*/false);
}

/// PerformContextuallyConvertToBool - Perform a contextual conversion
/// of the expression From to bool (C++0x [conv]p3).
ExprResult Sema::PerformContextuallyConvertToBool(Expr *From) {
  ImplicitConversionSequence ICS = TryContextuallyConvertToBool(*this, From);
  if (!ICS.isBad())
    return PerformImplicitConversion(From, Context.BoolTy, ICS, AA_Converting);

  if (!DiagnoseMultipleUserDefinedConversion(From, Context.BoolTy))
    return Diag(From->getSourceRange().getBegin(),
                diag::err_typecheck_bool_condition)
                  << From->getType() << From->getSourceRange();
  return ExprError();
}

/// TryContextuallyConvertToObjCId - Attempt to contextually convert the
/// expression From to 'id'.
static ImplicitConversionSequence
TryContextuallyConvertToObjCId(Sema &S, Expr *From) {
  QualType Ty = S.Context.getObjCIdType();
  return TryImplicitConversion(S, From, Ty,
                               // FIXME: Are these flags correct?
                               /*SuppressUserConversions=*/false,
                               /*AllowExplicit=*/true,
                               /*InOverloadResolution=*/false,
                               /*CStyle=*/false,
                               /*AllowObjCWritebackConversion=*/false);
}

/// PerformContextuallyConvertToObjCId - Perform a contextual conversion
/// of the expression From to 'id'.
ExprResult Sema::PerformContextuallyConvertToObjCId(Expr *From) {
  QualType Ty = Context.getObjCIdType();
  ImplicitConversionSequence ICS = TryContextuallyConvertToObjCId(*this, From);
  if (!ICS.isBad())
    return PerformImplicitConversion(From, Ty, ICS, AA_Converting);
  return ExprError();
}

/// \brief Attempt to convert the given expression to an integral or
/// enumeration type.
///
/// This routine will attempt to convert an expression of class type to an
/// integral or enumeration type, if that class type only has a single
/// conversion to an integral or enumeration type.
///
/// \param Loc The source location of the construct that requires the
/// conversion.
///
/// \param FromE The expression we're converting from.
///
/// \param NotIntDiag The diagnostic to be emitted if the expression does not
/// have integral or enumeration type.
///
/// \param IncompleteDiag The diagnostic to be emitted if the expression has
/// incomplete class type.
///
/// \param ExplicitConvDiag The diagnostic to be emitted if we're calling an
/// explicit conversion function (because no implicit conversion functions
/// were available). This is a recovery mode.
///
/// \param ExplicitConvNote The note to be emitted with \p ExplicitConvDiag,
/// showing which conversion was picked.
///
/// \param AmbigDiag The diagnostic to be emitted if there is more than one
/// conversion function that could convert to integral or enumeration type.
///
/// \param AmbigNote The note to be emitted with \p AmbigDiag for each
/// usable conversion function.
///
/// \param ConvDiag The diagnostic to be emitted if we are calling a conversion
/// function, which may be an extension in this case.
///
/// \returns The expression, converted to an integral or enumeration type if
/// successful.
ExprResult
Sema::ConvertToIntegralOrEnumerationType(SourceLocation Loc, Expr *From,
                                         const PartialDiagnostic &NotIntDiag,
                                       const PartialDiagnostic &IncompleteDiag,
                                     const PartialDiagnostic &ExplicitConvDiag,
                                     const PartialDiagnostic &ExplicitConvNote,
                                         const PartialDiagnostic &AmbigDiag,
                                         const PartialDiagnostic &AmbigNote,
                                         const PartialDiagnostic &ConvDiag) {
  // We can't perform any more checking for type-dependent expressions.
  if (From->isTypeDependent())
    return Owned(From);

  // If the expression already has integral or enumeration type, we're golden.
  QualType T = From->getType();
  if (T->isIntegralOrEnumerationType())
    return Owned(From);

  // FIXME: Check for missing '()' if T is a function type?

  // If we don't have a class type in C++, there's no way we can get an
  // expression of integral or enumeration type.
  const RecordType *RecordTy = T->getAs<RecordType>();
  if (!RecordTy || !getLangOptions().CPlusPlus) {
    Diag(Loc, NotIntDiag)
      << T << From->getSourceRange();
    return Owned(From);
  }

  // We must have a complete class type.
  if (RequireCompleteType(Loc, T, IncompleteDiag))
    return Owned(From);

  // Look for a conversion to an integral or enumeration type.
  UnresolvedSet<4> ViableConversions;
  UnresolvedSet<4> ExplicitConversions;
  const UnresolvedSetImpl *Conversions
    = cast<CXXRecordDecl>(RecordTy->getDecl())->getVisibleConversionFunctions();

  for (UnresolvedSetImpl::iterator I = Conversions->begin(),
                                   E = Conversions->end();
       I != E;
       ++I) {
    if (CXXConversionDecl *Conversion
          = dyn_cast<CXXConversionDecl>((*I)->getUnderlyingDecl()))
      if (Conversion->getConversionType().getNonReferenceType()
            ->isIntegralOrEnumerationType()) {
        if (Conversion->isExplicit())
          ExplicitConversions.addDecl(I.getDecl(), I.getAccess());
        else
          ViableConversions.addDecl(I.getDecl(), I.getAccess());
      }
  }

  switch (ViableConversions.size()) {
  case 0:
    if (ExplicitConversions.size() == 1) {
      DeclAccessPair Found = ExplicitConversions[0];
      CXXConversionDecl *Conversion
        = cast<CXXConversionDecl>(Found->getUnderlyingDecl());

      // The user probably meant to invoke the given explicit
      // conversion; use it.
      QualType ConvTy
        = Conversion->getConversionType().getNonReferenceType();
      std::string TypeStr;
      ConvTy.getAsStringInternal(TypeStr, Context.PrintingPolicy);

      Diag(Loc, ExplicitConvDiag)
        << T << ConvTy
        << FixItHint::CreateInsertion(From->getLocStart(),
                                      "static_cast<" + TypeStr + ">(")
        << FixItHint::CreateInsertion(PP.getLocForEndOfToken(From->getLocEnd()),
                                      ")");
      Diag(Conversion->getLocation(), ExplicitConvNote)
        << ConvTy->isEnumeralType() << ConvTy;

      // If we aren't in a SFINAE context, build a call to the
      // explicit conversion function.
      if (isSFINAEContext())
        return ExprError();

      CheckMemberOperatorAccess(From->getExprLoc(), From, 0, Found);
      ExprResult Result = BuildCXXMemberCallExpr(From, Found, Conversion);
      if (Result.isInvalid())
        return ExprError();

      From = Result.get();
    }

    // We'll complain below about a non-integral condition type.
    break;

  case 1: {
    // Apply this conversion.
    DeclAccessPair Found = ViableConversions[0];
    CheckMemberOperatorAccess(From->getExprLoc(), From, 0, Found);

    CXXConversionDecl *Conversion
      = cast<CXXConversionDecl>(Found->getUnderlyingDecl());
    QualType ConvTy
      = Conversion->getConversionType().getNonReferenceType();
    if (ConvDiag.getDiagID()) {
      if (isSFINAEContext())
        return ExprError();

      Diag(Loc, ConvDiag)
        << T << ConvTy->isEnumeralType() << ConvTy << From->getSourceRange();
    }

    ExprResult Result = BuildCXXMemberCallExpr(From, Found,
                          cast<CXXConversionDecl>(Found->getUnderlyingDecl()));
    if (Result.isInvalid())
      return ExprError();

    From = Result.get();
    break;
  }

  default:
    Diag(Loc, AmbigDiag)
      << T << From->getSourceRange();
    for (unsigned I = 0, N = ViableConversions.size(); I != N; ++I) {
      CXXConversionDecl *Conv
        = cast<CXXConversionDecl>(ViableConversions[I]->getUnderlyingDecl());
      QualType ConvTy = Conv->getConversionType().getNonReferenceType();
      Diag(Conv->getLocation(), AmbigNote)
        << ConvTy->isEnumeralType() << ConvTy;
    }
    return Owned(From);
  }

  if (!From->getType()->isIntegralOrEnumerationType())
    Diag(Loc, NotIntDiag)
      << From->getType() << From->getSourceRange();

  return Owned(From);
}

/// AddOverloadCandidate - Adds the given function to the set of
/// candidate functions, using the given function call arguments.  If
/// @p SuppressUserConversions, then don't allow user-defined
/// conversions via constructors or conversion operators.
///
/// \para PartialOverloading true if we are performing "partial" overloading
/// based on an incomplete set of function arguments. This feature is used by
/// code completion.
void
Sema::AddOverloadCandidate(FunctionDecl *Function,
                           DeclAccessPair FoundDecl,
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool SuppressUserConversions,
                           bool PartialOverloading) {
  const FunctionProtoType* Proto
    = dyn_cast<FunctionProtoType>(Function->getType()->getAs<FunctionType>());
  assert(Proto && "Functions without a prototype cannot be overloaded");
  assert(!Function->getDescribedFunctionTemplate() &&
         "Use AddTemplateOverloadCandidate for function templates");

  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Function)) {
    if (!isa<CXXConstructorDecl>(Method)) {
      // If we get here, it's because we're calling a member function
      // that is named without a member access expression (e.g.,
      // "this->f") that was either written explicitly or created
      // implicitly. This can happen with a qualified call to a member
      // function, e.g., X::f(). We use an empty type for the implied
      // object argument (C++ [over.call.func]p3), and the acting context
      // is irrelevant.
      AddMethodCandidate(Method, FoundDecl, Method->getParent(),
                         QualType(), Expr::Classification::makeSimpleLValue(),
                         Args, NumArgs, CandidateSet,
                         SuppressUserConversions);
      return;
    }
    // We treat a constructor like a non-member function, since its object
    // argument doesn't participate in overload resolution.
  }

  if (!CandidateSet.isNewCandidate(Function))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);

  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Function)){
    // C++ [class.copy]p3:
    //   A member function template is never instantiated to perform the copy
    //   of a class object to an object of its class type.
    QualType ClassType = Context.getTypeDeclType(Constructor->getParent());
    if (NumArgs == 1 &&
        Constructor->isSpecializationCopyingObject() &&
        (Context.hasSameUnqualifiedType(ClassType, Args[0]->getType()) ||
         IsDerivedFrom(Args[0]->getType(), ClassType)))
      return;
  }

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.FoundDecl = FoundDecl;
  Candidate.Function = Function;
  Candidate.Viable = true;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.ExplicitCallArguments = NumArgs;

  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if ((NumArgs + (PartialOverloading && NumArgs)) > NumArgsInProto &&
      !Proto->isVariadic()) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_many_arguments;
    return;
  }

  // (C++ 13.3.2p2): A candidate function having more than m parameters
  // is viable only if the (m+1)st parameter has a default argument
  // (8.3.6). For the purposes of overload resolution, the
  // parameter list is truncated on the right, so that there are
  // exactly m parameters.
  unsigned MinRequiredArgs = Function->getMinRequiredArguments();
  if (NumArgs < MinRequiredArgs && !PartialOverloading) {
    // Not enough arguments.
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_few_arguments;
    return;
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Conversions.resize(NumArgs);
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx]
        = TryCopyInitialization(*this, Args[ArgIdx], ParamType,
                                SuppressUserConversions,
                                /*InOverloadResolution=*/true,
                                /*AllowObjCWritebackConversion=*/
                                  getLangOptions().ObjCAutoRefCount);
      if (Candidate.Conversions[ArgIdx].isBad()) {
        Candidate.Viable = false;
        Candidate.FailureKind = ovl_fail_bad_conversion;
        break;
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx].setEllipsis();
    }
  }
}

/// \brief Add all of the function declarations in the given function set to
/// the overload canddiate set.
void Sema::AddFunctionCandidates(const UnresolvedSetImpl &Fns,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 bool SuppressUserConversions) {
  for (UnresolvedSetIterator F = Fns.begin(), E = Fns.end(); F != E; ++F) {
    NamedDecl *D = F.getDecl()->getUnderlyingDecl();
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      if (isa<CXXMethodDecl>(FD) && !cast<CXXMethodDecl>(FD)->isStatic())
        AddMethodCandidate(cast<CXXMethodDecl>(FD), F.getPair(),
                           cast<CXXMethodDecl>(FD)->getParent(),
                           Args[0]->getType(), Args[0]->Classify(Context),
                           Args + 1, NumArgs - 1,
                           CandidateSet, SuppressUserConversions);
      else
        AddOverloadCandidate(FD, F.getPair(), Args, NumArgs, CandidateSet,
                             SuppressUserConversions);
    } else {
      FunctionTemplateDecl *FunTmpl = cast<FunctionTemplateDecl>(D);
      if (isa<CXXMethodDecl>(FunTmpl->getTemplatedDecl()) &&
          !cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl())->isStatic())
        AddMethodTemplateCandidate(FunTmpl, F.getPair(),
                              cast<CXXRecordDecl>(FunTmpl->getDeclContext()),
                                   /*FIXME: explicit args */ 0,
                                   Args[0]->getType(),
                                   Args[0]->Classify(Context),
                                   Args + 1, NumArgs - 1,
                                   CandidateSet,
                                   SuppressUserConversions);
      else
        AddTemplateOverloadCandidate(FunTmpl, F.getPair(),
                                     /*FIXME: explicit args */ 0,
                                     Args, NumArgs, CandidateSet,
                                     SuppressUserConversions);
    }
  }
}

/// AddMethodCandidate - Adds a named decl (which is some kind of
/// method) as a method candidate to the given overload set.
void Sema::AddMethodCandidate(DeclAccessPair FoundDecl,
                              QualType ObjectType,
                              Expr::Classification ObjectClassification,
                              Expr **Args, unsigned NumArgs,
                              OverloadCandidateSet& CandidateSet,
                              bool SuppressUserConversions) {
  NamedDecl *Decl = FoundDecl.getDecl();
  CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(Decl->getDeclContext());

  if (isa<UsingShadowDecl>(Decl))
    Decl = cast<UsingShadowDecl>(Decl)->getTargetDecl();

  if (FunctionTemplateDecl *TD = dyn_cast<FunctionTemplateDecl>(Decl)) {
    assert(isa<CXXMethodDecl>(TD->getTemplatedDecl()) &&
           "Expected a member function template");
    AddMethodTemplateCandidate(TD, FoundDecl, ActingContext,
                               /*ExplicitArgs*/ 0,
                               ObjectType, ObjectClassification, Args, NumArgs,
                               CandidateSet,
                               SuppressUserConversions);
  } else {
    AddMethodCandidate(cast<CXXMethodDecl>(Decl), FoundDecl, ActingContext,
                       ObjectType, ObjectClassification, Args, NumArgs,
                       CandidateSet, SuppressUserConversions);
  }
}

/// AddMethodCandidate - Adds the given C++ member function to the set
/// of candidate functions, using the given function call arguments
/// and the object argument (@c Object). For example, in a call
/// @c o.f(a1,a2), @c Object will contain @c o and @c Args will contain
/// both @c a1 and @c a2. If @p SuppressUserConversions, then don't
/// allow user-defined conversions via constructors or conversion
/// operators.
void
Sema::AddMethodCandidate(CXXMethodDecl *Method, DeclAccessPair FoundDecl,
                         CXXRecordDecl *ActingContext, QualType ObjectType,
                         Expr::Classification ObjectClassification,
                         Expr **Args, unsigned NumArgs,
                         OverloadCandidateSet& CandidateSet,
                         bool SuppressUserConversions) {
  const FunctionProtoType* Proto
    = dyn_cast<FunctionProtoType>(Method->getType()->getAs<FunctionType>());
  assert(Proto && "Methods without a prototype cannot be overloaded");
  assert(!isa<CXXConstructorDecl>(Method) &&
         "Use AddOverloadCandidate for constructors");

  if (!CandidateSet.isNewCandidate(Method))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.FoundDecl = FoundDecl;
  Candidate.Function = Method;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.ExplicitCallArguments = NumArgs;

  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if (NumArgs > NumArgsInProto && !Proto->isVariadic()) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_many_arguments;
    return;
  }

  // (C++ 13.3.2p2): A candidate function having more than m parameters
  // is viable only if the (m+1)st parameter has a default argument
  // (8.3.6). For the purposes of overload resolution, the
  // parameter list is truncated on the right, so that there are
  // exactly m parameters.
  unsigned MinRequiredArgs = Method->getMinRequiredArguments();
  if (NumArgs < MinRequiredArgs) {
    // Not enough arguments.
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_few_arguments;
    return;
  }

  Candidate.Viable = true;
  Candidate.Conversions.resize(NumArgs + 1);

  if (Method->isStatic() || ObjectType.isNull())
    // The implicit object argument is ignored.
    Candidate.IgnoreObjectArgument = true;
  else {
    // Determine the implicit conversion sequence for the object
    // parameter.
    Candidate.Conversions[0]
      = TryObjectArgumentInitialization(*this, ObjectType, ObjectClassification,
                                        Method, ActingContext);
    if (Candidate.Conversions[0].isBad()) {
      Candidate.Viable = false;
      Candidate.FailureKind = ovl_fail_bad_conversion;
      return;
    }
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx + 1]
        = TryCopyInitialization(*this, Args[ArgIdx], ParamType,
                                SuppressUserConversions,
                                /*InOverloadResolution=*/true,
                                /*AllowObjCWritebackConversion=*/
                                  getLangOptions().ObjCAutoRefCount);
      if (Candidate.Conversions[ArgIdx + 1].isBad()) {
        Candidate.Viable = false;
        Candidate.FailureKind = ovl_fail_bad_conversion;
        break;
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx + 1].setEllipsis();
    }
  }
}

/// \brief Add a C++ member function template as a candidate to the candidate
/// set, using template argument deduction to produce an appropriate member
/// function template specialization.
void
Sema::AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
                                 DeclAccessPair FoundDecl,
                                 CXXRecordDecl *ActingContext,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                 QualType ObjectType,
                                 Expr::Classification ObjectClassification,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 bool SuppressUserConversions) {
  if (!CandidateSet.isNewCandidate(MethodTmpl))
    return;

  // C++ [over.match.funcs]p7:
  //   In each case where a candidate is a function template, candidate
  //   function template specializations are generated using template argument
  //   deduction (14.8.3, 14.8.2). Those candidates are then handled as
  //   candidate functions in the usual way.113) A given name can refer to one
  //   or more function templates and also to a set of overloaded non-template
  //   functions. In such a case, the candidate functions generated from each
  //   function template are combined with the set of non-template candidate
  //   functions.
  TemplateDeductionInfo Info(Context, CandidateSet.getLocation());
  FunctionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
      = DeduceTemplateArguments(MethodTmpl, ExplicitTemplateArgs,
                                Args, NumArgs, Specialization, Info)) {
    CandidateSet.push_back(OverloadCandidate());
    OverloadCandidate &Candidate = CandidateSet.back();
    Candidate.FoundDecl = FoundDecl;
    Candidate.Function = MethodTmpl->getTemplatedDecl();
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_deduction;
    Candidate.IsSurrogate = false;
    Candidate.IgnoreObjectArgument = false;
    Candidate.ExplicitCallArguments = NumArgs;
    Candidate.DeductionFailure = MakeDeductionFailureInfo(Context, Result,
                                                          Info);
    return;
  }

  // Add the function template specialization produced by template argument
  // deduction as a candidate.
  assert(Specialization && "Missing member function template specialization?");
  assert(isa<CXXMethodDecl>(Specialization) &&
         "Specialization is not a member function?");
  AddMethodCandidate(cast<CXXMethodDecl>(Specialization), FoundDecl,
                     ActingContext, ObjectType, ObjectClassification,
                     Args, NumArgs, CandidateSet, SuppressUserConversions);
}

/// \brief Add a C++ function template specialization as a candidate
/// in the candidate set, using template argument deduction to produce
/// an appropriate function template specialization.
void
Sema::AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
                                   DeclAccessPair FoundDecl,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet,
                                   bool SuppressUserConversions) {
  if (!CandidateSet.isNewCandidate(FunctionTemplate))
    return;

  // C++ [over.match.funcs]p7:
  //   In each case where a candidate is a function template, candidate
  //   function template specializations are generated using template argument
  //   deduction (14.8.3, 14.8.2). Those candidates are then handled as
  //   candidate functions in the usual way.113) A given name can refer to one
  //   or more function templates and also to a set of overloaded non-template
  //   functions. In such a case, the candidate functions generated from each
  //   function template are combined with the set of non-template candidate
  //   functions.
  TemplateDeductionInfo Info(Context, CandidateSet.getLocation());
  FunctionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
        = DeduceTemplateArguments(FunctionTemplate, ExplicitTemplateArgs,
                                  Args, NumArgs, Specialization, Info)) {
    CandidateSet.push_back(OverloadCandidate());
    OverloadCandidate &Candidate = CandidateSet.back();
    Candidate.FoundDecl = FoundDecl;
    Candidate.Function = FunctionTemplate->getTemplatedDecl();
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_deduction;
    Candidate.IsSurrogate = false;
    Candidate.IgnoreObjectArgument = false;
    Candidate.ExplicitCallArguments = NumArgs;
    Candidate.DeductionFailure = MakeDeductionFailureInfo(Context, Result,
                                                          Info);
    return;
  }

  // Add the function template specialization produced by template argument
  // deduction as a candidate.
  assert(Specialization && "Missing function template specialization?");
  AddOverloadCandidate(Specialization, FoundDecl, Args, NumArgs, CandidateSet,
                       SuppressUserConversions);
}

/// AddConversionCandidate - Add a C++ conversion function as a
/// candidate in the candidate set (C++ [over.match.conv],
/// C++ [over.match.copy]). From is the expression we're converting from,
/// and ToType is the type that we're eventually trying to convert to
/// (which may or may not be the same type as the type that the
/// conversion function produces).
void
Sema::AddConversionCandidate(CXXConversionDecl *Conversion,
                             DeclAccessPair FoundDecl,
                             CXXRecordDecl *ActingContext,
                             Expr *From, QualType ToType,
                             OverloadCandidateSet& CandidateSet) {
  assert(!Conversion->getDescribedFunctionTemplate() &&
         "Conversion function templates use AddTemplateConversionCandidate");
  QualType ConvType = Conversion->getConversionType().getNonReferenceType();
  if (!CandidateSet.isNewCandidate(Conversion))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.FoundDecl = FoundDecl;
  Candidate.Function = Conversion;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.FinalConversion.setAsIdentityConversion();
  Candidate.FinalConversion.setFromType(ConvType);
  Candidate.FinalConversion.setAllToTypes(ToType);
  Candidate.Viable = true;
  Candidate.Conversions.resize(1);
  Candidate.ExplicitCallArguments = 1;

  // C++ [over.match.funcs]p4:
  //   For conversion functions, the function is considered to be a member of
  //   the class of the implicit implied object argument for the purpose of
  //   defining the type of the implicit object parameter.
  //
  // Determine the implicit conversion sequence for the implicit
  // object parameter.
  QualType ImplicitParamType = From->getType();
  if (const PointerType *FromPtrType = ImplicitParamType->getAs<PointerType>())
    ImplicitParamType = FromPtrType->getPointeeType();
  CXXRecordDecl *ConversionContext
    = cast<CXXRecordDecl>(ImplicitParamType->getAs<RecordType>()->getDecl());

  Candidate.Conversions[0]
    = TryObjectArgumentInitialization(*this, From->getType(),
                                      From->Classify(Context),
                                      Conversion, ConversionContext);

  if (Candidate.Conversions[0].isBad()) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_conversion;
    return;
  }

  // We won't go through a user-define type conversion function to convert a
  // derived to base as such conversions are given Conversion Rank. They only
  // go through a copy constructor. 13.3.3.1.2-p4 [over.ics.user]
  QualType FromCanon
    = Context.getCanonicalType(From->getType().getUnqualifiedType());
  QualType ToCanon = Context.getCanonicalType(ToType).getUnqualifiedType();
  if (FromCanon == ToCanon || IsDerivedFrom(FromCanon, ToCanon)) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_trivial_conversion;
    return;
  }

  // To determine what the conversion from the result of calling the
  // conversion function to the type we're eventually trying to
  // convert to (ToType), we need to synthesize a call to the
  // conversion function and attempt copy initialization from it. This
  // makes sure that we get the right semantics with respect to
  // lvalues/rvalues and the type. Fortunately, we can allocate this
  // call on the stack and we don't need its arguments to be
  // well-formed.
  DeclRefExpr ConversionRef(Conversion, Conversion->getType(),
                            VK_LValue, From->getLocStart());
  ImplicitCastExpr ConversionFn(ImplicitCastExpr::OnStack,
                                Context.getPointerType(Conversion->getType()),
                                CK_FunctionToPointerDecay,
                                &ConversionRef, VK_RValue);

  QualType ConversionType = Conversion->getConversionType();
  if (RequireCompleteType(From->getLocStart(), ConversionType, 0)) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_final_conversion;
    return;
  }

  ExprValueKind VK = Expr::getValueKindForType(ConversionType);

  // Note that it is safe to allocate CallExpr on the stack here because
  // there are 0 arguments (i.e., nothing is allocated using ASTContext's
  // allocator).
  QualType CallResultType = ConversionType.getNonLValueExprType(Context);
  CallExpr Call(Context, &ConversionFn, 0, 0, CallResultType, VK,
                From->getLocStart());
  ImplicitConversionSequence ICS =
    TryCopyInitialization(*this, &Call, ToType,
                          /*SuppressUserConversions=*/true,
                          /*InOverloadResolution=*/false,
                          /*AllowObjCWritebackConversion=*/false);

  switch (ICS.getKind()) {
  case ImplicitConversionSequence::StandardConversion:
    Candidate.FinalConversion = ICS.Standard;

    // C++ [over.ics.user]p3:
    //   If the user-defined conversion is specified by a specialization of a
    //   conversion function template, the second standard conversion sequence
    //   shall have exact match rank.
    if (Conversion->getPrimaryTemplate() &&
        GetConversionRank(ICS.Standard.Second) != ICR_Exact_Match) {
      Candidate.Viable = false;
      Candidate.FailureKind = ovl_fail_final_conversion_not_exact;
    }

    // C++0x [dcl.init.ref]p5:
    //    In the second case, if the reference is an rvalue reference and
    //    the second standard conversion sequence of the user-defined
    //    conversion sequence includes an lvalue-to-rvalue conversion, the
    //    program is ill-formed.
    if (ToType->isRValueReferenceType() &&
        ICS.Standard.First == ICK_Lvalue_To_Rvalue) {
      Candidate.Viable = false;
      Candidate.FailureKind = ovl_fail_bad_final_conversion;
    }
    break;

  case ImplicitConversionSequence::BadConversion:
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_final_conversion;
    break;

  default:
    assert(false &&
           "Can only end up with a standard conversion sequence or failure");
  }
}

/// \brief Adds a conversion function template specialization
/// candidate to the overload set, using template argument deduction
/// to deduce the template arguments of the conversion function
/// template from the type that we are converting to (C++
/// [temp.deduct.conv]).
void
Sema::AddTemplateConversionCandidate(FunctionTemplateDecl *FunctionTemplate,
                                     DeclAccessPair FoundDecl,
                                     CXXRecordDecl *ActingDC,
                                     Expr *From, QualType ToType,
                                     OverloadCandidateSet &CandidateSet) {
  assert(isa<CXXConversionDecl>(FunctionTemplate->getTemplatedDecl()) &&
         "Only conversion function templates permitted here");

  if (!CandidateSet.isNewCandidate(FunctionTemplate))
    return;

  TemplateDeductionInfo Info(Context, CandidateSet.getLocation());
  CXXConversionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
        = DeduceTemplateArguments(FunctionTemplate, ToType,
                                  Specialization, Info)) {
    CandidateSet.push_back(OverloadCandidate());
    OverloadCandidate &Candidate = CandidateSet.back();
    Candidate.FoundDecl = FoundDecl;
    Candidate.Function = FunctionTemplate->getTemplatedDecl();
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_deduction;
    Candidate.IsSurrogate = false;
    Candidate.IgnoreObjectArgument = false;
    Candidate.ExplicitCallArguments = 1;
    Candidate.DeductionFailure = MakeDeductionFailureInfo(Context, Result,
                                                          Info);
    return;
  }

  // Add the conversion function template specialization produced by
  // template argument deduction as a candidate.
  assert(Specialization && "Missing function template specialization?");
  AddConversionCandidate(Specialization, FoundDecl, ActingDC, From, ToType,
                         CandidateSet);
}

/// AddSurrogateCandidate - Adds a "surrogate" candidate function that
/// converts the given @c Object to a function pointer via the
/// conversion function @c Conversion, and then attempts to call it
/// with the given arguments (C++ [over.call.object]p2-4). Proto is
/// the type of function that we'll eventually be calling.
void Sema::AddSurrogateCandidate(CXXConversionDecl *Conversion,
                                 DeclAccessPair FoundDecl,
                                 CXXRecordDecl *ActingContext,
                                 const FunctionProtoType *Proto,
                                 Expr *Object,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet) {
  if (!CandidateSet.isNewCandidate(Conversion))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);

  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.FoundDecl = FoundDecl;
  Candidate.Function = 0;
  Candidate.Surrogate = Conversion;
  Candidate.Viable = true;
  Candidate.IsSurrogate = true;
  Candidate.IgnoreObjectArgument = false;
  Candidate.Conversions.resize(NumArgs + 1);
  Candidate.ExplicitCallArguments = NumArgs;

  // Determine the implicit conversion sequence for the implicit
  // object parameter.
  ImplicitConversionSequence ObjectInit
    = TryObjectArgumentInitialization(*this, Object->getType(),
                                      Object->Classify(Context),
                                      Conversion, ActingContext);
  if (ObjectInit.isBad()) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_bad_conversion;
    Candidate.Conversions[0] = ObjectInit;
    return;
  }

  // The first conversion is actually a user-defined conversion whose
  // first conversion is ObjectInit's standard conversion (which is
  // effectively a reference binding). Record it as such.
  Candidate.Conversions[0].setUserDefined();
  Candidate.Conversions[0].UserDefined.Before = ObjectInit.Standard;
  Candidate.Conversions[0].UserDefined.EllipsisConversion = false;
  Candidate.Conversions[0].UserDefined.ConversionFunction = Conversion;
  Candidate.Conversions[0].UserDefined.FoundConversionFunction
    = FoundDecl.getDecl();
  Candidate.Conversions[0].UserDefined.After
    = Candidate.Conversions[0].UserDefined.Before;
  Candidate.Conversions[0].UserDefined.After.setAsIdentityConversion();

  // Find the
  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if (NumArgs > NumArgsInProto && !Proto->isVariadic()) {
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_many_arguments;
    return;
  }

  // Function types don't have any default arguments, so just check if
  // we have enough arguments.
  if (NumArgs < NumArgsInProto) {
    // Not enough arguments.
    Candidate.Viable = false;
    Candidate.FailureKind = ovl_fail_too_few_arguments;
    return;
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx + 1]
        = TryCopyInitialization(*this, Args[ArgIdx], ParamType,
                                /*SuppressUserConversions=*/false,
                                /*InOverloadResolution=*/false,
                                /*AllowObjCWritebackConversion=*/
                                  getLangOptions().ObjCAutoRefCount);
      if (Candidate.Conversions[ArgIdx + 1].isBad()) {
        Candidate.Viable = false;
        Candidate.FailureKind = ovl_fail_bad_conversion;
        break;
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx + 1].setEllipsis();
    }
  }
}

/// \brief Add overload candidates for overloaded operators that are
/// member functions.
///
/// Add the overloaded operator candidates that are member functions
/// for the operator Op that was used in an operator expression such
/// as "x Op y". , Args/NumArgs provides the operator arguments, and
/// CandidateSet will store the added overload candidates. (C++
/// [over.match.oper]).
void Sema::AddMemberOperatorCandidates(OverloadedOperatorKind Op,
                                       SourceLocation OpLoc,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet& CandidateSet,
                                       SourceRange OpRange) {
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  // C++ [over.match.oper]p3:
  //   For a unary operator @ with an operand of a type whose
  //   cv-unqualified version is T1, and for a binary operator @ with
  //   a left operand of a type whose cv-unqualified version is T1 and
  //   a right operand of a type whose cv-unqualified version is T2,
  //   three sets of candidate functions, designated member
  //   candidates, non-member candidates and built-in candidates, are
  //   constructed as follows:
  QualType T1 = Args[0]->getType();

  //     -- If T1 is a class type, the set of member candidates is the
  //        result of the qualified lookup of T1::operator@
  //        (13.3.1.1.1); otherwise, the set of member candidates is
  //        empty.
  if (const RecordType *T1Rec = T1->getAs<RecordType>()) {
    // Complete the type if it can be completed. Otherwise, we're done.
    if (RequireCompleteType(OpLoc, T1, PDiag()))
      return;

    LookupResult Operators(*this, OpName, OpLoc, LookupOrdinaryName);
    LookupQualifiedName(Operators, T1Rec->getDecl());
    Operators.suppressDiagnostics();

    for (LookupResult::iterator Oper = Operators.begin(),
                             OperEnd = Operators.end();
         Oper != OperEnd;
         ++Oper)
      AddMethodCandidate(Oper.getPair(), Args[0]->getType(),
                         Args[0]->Classify(Context), Args + 1, NumArgs - 1,
                         CandidateSet,
                         /* SuppressUserConversions = */ false);
  }
}

/// AddBuiltinCandidate - Add a candidate for a built-in
/// operator. ResultTy and ParamTys are the result and parameter types
/// of the built-in candidate, respectively. Args and NumArgs are the
/// arguments being passed to the candidate. IsAssignmentOperator
/// should be true when this built-in candidate is an assignment
/// operator. NumContextualBoolArguments is the number of arguments
/// (at the beginning of the argument list) that will be contextually
/// converted to bool.
void Sema::AddBuiltinCandidate(QualType ResultTy, QualType *ParamTys,
                               Expr **Args, unsigned NumArgs,
                               OverloadCandidateSet& CandidateSet,
                               bool IsAssignmentOperator,
                               unsigned NumContextualBoolArguments) {
  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.FoundDecl = DeclAccessPair::make(0, AS_none);
  Candidate.Function = 0;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.BuiltinTypes.ResultTy = ResultTy;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    Candidate.BuiltinTypes.ParamTypes[ArgIdx] = ParamTys[ArgIdx];

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Viable = true;
  Candidate.Conversions.resize(NumArgs);
  Candidate.ExplicitCallArguments = NumArgs;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    // C++ [over.match.oper]p4:
    //   For the built-in assignment operators, conversions of the
    //   left operand are restricted as follows:
    //     -- no temporaries are introduced to hold the left operand, and
    //     -- no user-defined conversions are applied to the left
    //        operand to achieve a type match with the left-most
    //        parameter of a built-in candidate.
    //
    // We block these conversions by turning off user-defined
    // conversions, since that is the only way that initialization of
    // a reference to a non-class type can occur from something that
    // is not of the same type.
    if (ArgIdx < NumContextualBoolArguments) {
      assert(ParamTys[ArgIdx] == Context.BoolTy &&
             "Contextual conversion to bool requires bool type");
      Candidate.Conversions[ArgIdx]
        = TryContextuallyConvertToBool(*this, Args[ArgIdx]);
    } else {
      Candidate.Conversions[ArgIdx]
        = TryCopyInitialization(*this, Args[ArgIdx], ParamTys[ArgIdx],
                                ArgIdx == 0 && IsAssignmentOperator,
                                /*InOverloadResolution=*/false,
                                /*AllowObjCWritebackConversion=*/
                                  getLangOptions().ObjCAutoRefCount);
    }
    if (Candidate.Conversions[ArgIdx].isBad()) {
      Candidate.Viable = false;
      Candidate.FailureKind = ovl_fail_bad_conversion;
      break;
    }
  }
}

/// BuiltinCandidateTypeSet - A set of types that will be used for the
/// candidate operator functions for built-in operators (C++
/// [over.built]). The types are separated into pointer types and
/// enumeration types.
class BuiltinCandidateTypeSet  {
  /// TypeSet - A set of types.
  typedef llvm::SmallPtrSet<QualType, 8> TypeSet;

  /// PointerTypes - The set of pointer types that will be used in the
  /// built-in candidates.
  TypeSet PointerTypes;

  /// MemberPointerTypes - The set of member pointer types that will be
  /// used in the built-in candidates.
  TypeSet MemberPointerTypes;

  /// EnumerationTypes - The set of enumeration types that will be
  /// used in the built-in candidates.
  TypeSet EnumerationTypes;

  /// \brief The set of vector types that will be used in the built-in
  /// candidates.
  TypeSet VectorTypes;

  /// \brief A flag indicating non-record types are viable candidates
  bool HasNonRecordTypes;

  /// \brief A flag indicating whether either arithmetic or enumeration types
  /// were present in the candidate set.
  bool HasArithmeticOrEnumeralTypes;

  /// \brief A flag indicating whether the nullptr type was present in the
  /// candidate set.
  bool HasNullPtrType;
  
  /// Sema - The semantic analysis instance where we are building the
  /// candidate type set.
  Sema &SemaRef;

  /// Context - The AST context in which we will build the type sets.
  ASTContext &Context;

  bool AddPointerWithMoreQualifiedTypeVariants(QualType Ty,
                                               const Qualifiers &VisibleQuals);
  bool AddMemberPointerWithMoreQualifiedTypeVariants(QualType Ty);

public:
  /// iterator - Iterates through the types that are part of the set.
  typedef TypeSet::iterator iterator;

  BuiltinCandidateTypeSet(Sema &SemaRef)
    : HasNonRecordTypes(false),
      HasArithmeticOrEnumeralTypes(false),
      HasNullPtrType(false),
      SemaRef(SemaRef),
      Context(SemaRef.Context) { }

  void AddTypesConvertedFrom(QualType Ty,
                             SourceLocation Loc,
                             bool AllowUserConversions,
                             bool AllowExplicitConversions,
                             const Qualifiers &VisibleTypeConversionsQuals);

  /// pointer_begin - First pointer type found;
  iterator pointer_begin() { return PointerTypes.begin(); }

  /// pointer_end - Past the last pointer type found;
  iterator pointer_end() { return PointerTypes.end(); }

  /// member_pointer_begin - First member pointer type found;
  iterator member_pointer_begin() { return MemberPointerTypes.begin(); }

  /// member_pointer_end - Past the last member pointer type found;
  iterator member_pointer_end() { return MemberPointerTypes.end(); }

  /// enumeration_begin - First enumeration type found;
  iterator enumeration_begin() { return EnumerationTypes.begin(); }

  /// enumeration_end - Past the last enumeration type found;
  iterator enumeration_end() { return EnumerationTypes.end(); }

  iterator vector_begin() { return VectorTypes.begin(); }
  iterator vector_end() { return VectorTypes.end(); }

  bool hasNonRecordTypes() { return HasNonRecordTypes; }
  bool hasArithmeticOrEnumeralTypes() { return HasArithmeticOrEnumeralTypes; }
  bool hasNullPtrType() const { return HasNullPtrType; }
};

/// AddPointerWithMoreQualifiedTypeVariants - Add the pointer type @p Ty to
/// the set of pointer types along with any more-qualified variants of
/// that type. For example, if @p Ty is "int const *", this routine
/// will add "int const *", "int const volatile *", "int const
/// restrict *", and "int const volatile restrict *" to the set of
/// pointer types. Returns true if the add of @p Ty itself succeeded,
/// false otherwise.
///
/// FIXME: what to do about extended qualifiers?
bool
BuiltinCandidateTypeSet::AddPointerWithMoreQualifiedTypeVariants(QualType Ty,
                                             const Qualifiers &VisibleQuals) {

  // Insert this type.
  if (!PointerTypes.insert(Ty))
    return false;

  QualType PointeeTy;
  const PointerType *PointerTy = Ty->getAs<PointerType>();
  bool buildObjCPtr = false;
  if (!PointerTy) {
    if (const ObjCObjectPointerType *PTy = Ty->getAs<ObjCObjectPointerType>()) {
      PointeeTy = PTy->getPointeeType();
      buildObjCPtr = true;
    }
    else
      assert(false && "type was not a pointer type!");
  }
  else
    PointeeTy = PointerTy->getPointeeType();

  // Don't add qualified variants of arrays. For one, they're not allowed
  // (the qualifier would sink to the element type), and for another, the
  // only overload situation where it matters is subscript or pointer +- int,
  // and those shouldn't have qualifier variants anyway.
  if (PointeeTy->isArrayType())
    return true;
  unsigned BaseCVR = PointeeTy.getCVRQualifiers();
  if (const ConstantArrayType *Array =Context.getAsConstantArrayType(PointeeTy))
    BaseCVR = Array->getElementType().getCVRQualifiers();
  bool hasVolatile = VisibleQuals.hasVolatile();
  bool hasRestrict = VisibleQuals.hasRestrict();

  // Iterate through all strict supersets of BaseCVR.
  for (unsigned CVR = BaseCVR+1; CVR <= Qualifiers::CVRMask; ++CVR) {
    if ((CVR | BaseCVR) != CVR) continue;
    // Skip over Volatile/Restrict if no Volatile/Restrict found anywhere
    // in the types.
    if ((CVR & Qualifiers::Volatile) && !hasVolatile) continue;
    if ((CVR & Qualifiers::Restrict) && !hasRestrict) continue;
    QualType QPointeeTy = Context.getCVRQualifiedType(PointeeTy, CVR);
    if (!buildObjCPtr)
      PointerTypes.insert(Context.getPointerType(QPointeeTy));
    else
      PointerTypes.insert(Context.getObjCObjectPointerType(QPointeeTy));
  }

  return true;
}

/// AddMemberPointerWithMoreQualifiedTypeVariants - Add the pointer type @p Ty
/// to the set of pointer types along with any more-qualified variants of
/// that type. For example, if @p Ty is "int const *", this routine
/// will add "int const *", "int const volatile *", "int const
/// restrict *", and "int const volatile restrict *" to the set of
/// pointer types. Returns true if the add of @p Ty itself succeeded,
/// false otherwise.
///
/// FIXME: what to do about extended qualifiers?
bool
BuiltinCandidateTypeSet::AddMemberPointerWithMoreQualifiedTypeVariants(
    QualType Ty) {
  // Insert this type.
  if (!MemberPointerTypes.insert(Ty))
    return false;

  const MemberPointerType *PointerTy = Ty->getAs<MemberPointerType>();
  assert(PointerTy && "type was not a member pointer type!");

  QualType PointeeTy = PointerTy->getPointeeType();
  // Don't add qualified variants of arrays. For one, they're not allowed
  // (the qualifier would sink to the element type), and for another, the
  // only overload situation where it matters is subscript or pointer +- int,
  // and those shouldn't have qualifier variants anyway.
  if (PointeeTy->isArrayType())
    return true;
  const Type *ClassTy = PointerTy->getClass();

  // Iterate through all strict supersets of the pointee type's CVR
  // qualifiers.
  unsigned BaseCVR = PointeeTy.getCVRQualifiers();
  for (unsigned CVR = BaseCVR+1; CVR <= Qualifiers::CVRMask; ++CVR) {
    if ((CVR | BaseCVR) != CVR) continue;

    QualType QPointeeTy = Context.getCVRQualifiedType(PointeeTy, CVR);
    MemberPointerTypes.insert(
      Context.getMemberPointerType(QPointeeTy, ClassTy));
  }

  return true;
}

/// AddTypesConvertedFrom - Add each of the types to which the type @p
/// Ty can be implicit converted to the given set of @p Types. We're
/// primarily interested in pointer types and enumeration types. We also
/// take member pointer types, for the conditional operator.
/// AllowUserConversions is true if we should look at the conversion
/// functions of a class type, and AllowExplicitConversions if we
/// should also include the explicit conversion functions of a class
/// type.
void
BuiltinCandidateTypeSet::AddTypesConvertedFrom(QualType Ty,
                                               SourceLocation Loc,
                                               bool AllowUserConversions,
                                               bool AllowExplicitConversions,
                                               const Qualifiers &VisibleQuals) {
  // Only deal with canonical types.
  Ty = Context.getCanonicalType(Ty);

  // Look through reference types; they aren't part of the type of an
  // expression for the purposes of conversions.
  if (const ReferenceType *RefTy = Ty->getAs<ReferenceType>())
    Ty = RefTy->getPointeeType();

  // If we're dealing with an array type, decay to the pointer.
  if (Ty->isArrayType())
    Ty = SemaRef.Context.getArrayDecayedType(Ty);

  // Otherwise, we don't care about qualifiers on the type.
  Ty = Ty.getLocalUnqualifiedType();

  // Flag if we ever add a non-record type.
  const RecordType *TyRec = Ty->getAs<RecordType>();
  HasNonRecordTypes = HasNonRecordTypes || !TyRec;

  // Flag if we encounter an arithmetic type.
  HasArithmeticOrEnumeralTypes =
    HasArithmeticOrEnumeralTypes || Ty->isArithmeticType();

  if (Ty->isObjCIdType() || Ty->isObjCClassType())
    PointerTypes.insert(Ty);
  else if (Ty->getAs<PointerType>() || Ty->getAs<ObjCObjectPointerType>()) {
    // Insert our type, and its more-qualified variants, into the set
    // of types.
    if (!AddPointerWithMoreQualifiedTypeVariants(Ty, VisibleQuals))
      return;
  } else if (Ty->isMemberPointerType()) {
    // Member pointers are far easier, since the pointee can't be converted.
    if (!AddMemberPointerWithMoreQualifiedTypeVariants(Ty))
      return;
  } else if (Ty->isEnumeralType()) {
    HasArithmeticOrEnumeralTypes = true;
    EnumerationTypes.insert(Ty);
  } else if (Ty->isVectorType()) {
    // We treat vector types as arithmetic types in many contexts as an
    // extension.
    HasArithmeticOrEnumeralTypes = true;
    VectorTypes.insert(Ty);
  } else if (Ty->isNullPtrType()) {
    HasNullPtrType = true;
  } else if (AllowUserConversions && TyRec) {
    // No conversion functions in incomplete types.
    if (SemaRef.RequireCompleteType(Loc, Ty, 0))
      return;

    CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
    const UnresolvedSetImpl *Conversions
      = ClassDecl->getVisibleConversionFunctions();
    for (UnresolvedSetImpl::iterator I = Conversions->begin(),
           E = Conversions->end(); I != E; ++I) {
      NamedDecl *D = I.getDecl();
      if (isa<UsingShadowDecl>(D))
        D = cast<UsingShadowDecl>(D)->getTargetDecl();

      // Skip conversion function templates; they don't tell us anything
      // about which builtin types we can convert to.
      if (isa<FunctionTemplateDecl>(D))
        continue;

      CXXConversionDecl *Conv = cast<CXXConversionDecl>(D);
      if (AllowExplicitConversions || !Conv->isExplicit()) {
        AddTypesConvertedFrom(Conv->getConversionType(), Loc, false, false,
                              VisibleQuals);
      }
    }
  }
}

/// \brief Helper function for AddBuiltinOperatorCandidates() that adds
/// the volatile- and non-volatile-qualified assignment operators for the
/// given type to the candidate set.
static void AddBuiltinAssignmentOperatorCandidates(Sema &S,
                                                   QualType T,
                                                   Expr **Args,
                                                   unsigned NumArgs,
                                    OverloadCandidateSet &CandidateSet) {
  QualType ParamTypes[2];

  // T& operator=(T&, T)
  ParamTypes[0] = S.Context.getLValueReferenceType(T);
  ParamTypes[1] = T;
  S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                        /*IsAssignmentOperator=*/true);

  if (!S.Context.getCanonicalType(T).isVolatileQualified()) {
    // volatile T& operator=(volatile T&, T)
    ParamTypes[0]
      = S.Context.getLValueReferenceType(S.Context.getVolatileType(T));
    ParamTypes[1] = T;
    S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                          /*IsAssignmentOperator=*/true);
  }
}

/// CollectVRQualifiers - This routine returns Volatile/Restrict qualifiers,
/// if any, found in visible type conversion functions found in ArgExpr's type.
static  Qualifiers CollectVRQualifiers(ASTContext &Context, Expr* ArgExpr) {
    Qualifiers VRQuals;
    const RecordType *TyRec;
    if (const MemberPointerType *RHSMPType =
        ArgExpr->getType()->getAs<MemberPointerType>())
      TyRec = RHSMPType->getClass()->getAs<RecordType>();
    else
      TyRec = ArgExpr->getType()->getAs<RecordType>();
    if (!TyRec) {
      // Just to be safe, assume the worst case.
      VRQuals.addVolatile();
      VRQuals.addRestrict();
      return VRQuals;
    }

    CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
    if (!ClassDecl->hasDefinition())
      return VRQuals;

    const UnresolvedSetImpl *Conversions =
      ClassDecl->getVisibleConversionFunctions();

    for (UnresolvedSetImpl::iterator I = Conversions->begin(),
           E = Conversions->end(); I != E; ++I) {
      NamedDecl *D = I.getDecl();
      if (isa<UsingShadowDecl>(D))
        D = cast<UsingShadowDecl>(D)->getTargetDecl();
      if (CXXConversionDecl *Conv = dyn_cast<CXXConversionDecl>(D)) {
        QualType CanTy = Context.getCanonicalType(Conv->getConversionType());
        if (const ReferenceType *ResTypeRef = CanTy->getAs<ReferenceType>())
          CanTy = ResTypeRef->getPointeeType();
        // Need to go down the pointer/mempointer chain and add qualifiers
        // as see them.
        bool done = false;
        while (!done) {
          if (const PointerType *ResTypePtr = CanTy->getAs<PointerType>())
            CanTy = ResTypePtr->getPointeeType();
          else if (const MemberPointerType *ResTypeMPtr =
                CanTy->getAs<MemberPointerType>())
            CanTy = ResTypeMPtr->getPointeeType();
          else
            done = true;
          if (CanTy.isVolatileQualified())
            VRQuals.addVolatile();
          if (CanTy.isRestrictQualified())
            VRQuals.addRestrict();
          if (VRQuals.hasRestrict() && VRQuals.hasVolatile())
            return VRQuals;
        }
      }
    }
    return VRQuals;
}

namespace {

/// \brief Helper class to manage the addition of builtin operator overload
/// candidates. It provides shared state and utility methods used throughout
/// the process, as well as a helper method to add each group of builtin
/// operator overloads from the standard to a candidate set.
class BuiltinOperatorOverloadBuilder {
  // Common instance state available to all overload candidate addition methods.
  Sema &S;
  Expr **Args;
  unsigned NumArgs;
  Qualifiers VisibleTypeConversionsQuals;
  bool HasArithmeticOrEnumeralCandidateType;
  llvm::SmallVectorImpl<BuiltinCandidateTypeSet> &CandidateTypes;
  OverloadCandidateSet &CandidateSet;

  // Define some constants used to index and iterate over the arithemetic types
  // provided via the getArithmeticType() method below.
  // The "promoted arithmetic types" are the arithmetic
  // types are that preserved by promotion (C++ [over.built]p2).
  static const unsigned FirstIntegralType = 3;
  static const unsigned LastIntegralType = 18;
  static const unsigned FirstPromotedIntegralType = 3,
                        LastPromotedIntegralType = 9;
  static const unsigned FirstPromotedArithmeticType = 0,
                        LastPromotedArithmeticType = 9;
  static const unsigned NumArithmeticTypes = 18;

  /// \brief Get the canonical type for a given arithmetic type index.
  CanQualType getArithmeticType(unsigned index) {
    assert(index < NumArithmeticTypes);
    static CanQualType ASTContext::* const
      ArithmeticTypes[NumArithmeticTypes] = {
      // Start of promoted types.
      &ASTContext::FloatTy,
      &ASTContext::DoubleTy,
      &ASTContext::LongDoubleTy,

      // Start of integral types.
      &ASTContext::IntTy,
      &ASTContext::LongTy,
      &ASTContext::LongLongTy,
      &ASTContext::UnsignedIntTy,
      &ASTContext::UnsignedLongTy,
      &ASTContext::UnsignedLongLongTy,
      // End of promoted types.

      &ASTContext::BoolTy,
      &ASTContext::CharTy,
      &ASTContext::WCharTy,
      &ASTContext::Char16Ty,
      &ASTContext::Char32Ty,
      &ASTContext::SignedCharTy,
      &ASTContext::ShortTy,
      &ASTContext::UnsignedCharTy,
      &ASTContext::UnsignedShortTy,
      // End of integral types.
      // FIXME: What about complex?
    };
    return S.Context.*ArithmeticTypes[index];
  }

  /// \brief Gets the canonical type resulting from the usual arithemetic
  /// converions for the given arithmetic types.
  CanQualType getUsualArithmeticConversions(unsigned L, unsigned R) {
    // Accelerator table for performing the usual arithmetic conversions.
    // The rules are basically:
    //   - if either is floating-point, use the wider floating-point
    //   - if same signedness, use the higher rank
    //   - if same size, use unsigned of the higher rank
    //   - use the larger type
    // These rules, together with the axiom that higher ranks are
    // never smaller, are sufficient to precompute all of these results
    // *except* when dealing with signed types of higher rank.
    // (we could precompute SLL x UI for all known platforms, but it's
    // better not to make any assumptions).
    enum PromotedType {
                  Flt,  Dbl, LDbl,   SI,   SL,  SLL,   UI,   UL,  ULL, Dep=-1
    };
    static PromotedType ConversionsTable[LastPromotedArithmeticType]
                                        [LastPromotedArithmeticType] = {
      /* Flt*/ {  Flt,  Dbl, LDbl,  Flt,  Flt,  Flt,  Flt,  Flt,  Flt },
      /* Dbl*/ {  Dbl,  Dbl, LDbl,  Dbl,  Dbl,  Dbl,  Dbl,  Dbl,  Dbl },
      /*LDbl*/ { LDbl, LDbl, LDbl, LDbl, LDbl, LDbl, LDbl, LDbl, LDbl },
      /*  SI*/ {  Flt,  Dbl, LDbl,   SI,   SL,  SLL,   UI,   UL,  ULL },
      /*  SL*/ {  Flt,  Dbl, LDbl,   SL,   SL,  SLL,  Dep,   UL,  ULL },
      /* SLL*/ {  Flt,  Dbl, LDbl,  SLL,  SLL,  SLL,  Dep,  Dep,  ULL },
      /*  UI*/ {  Flt,  Dbl, LDbl,   UI,  Dep,  Dep,   UI,   UL,  ULL },
      /*  UL*/ {  Flt,  Dbl, LDbl,   UL,   UL,  Dep,   UL,   UL,  ULL },
      /* ULL*/ {  Flt,  Dbl, LDbl,  ULL,  ULL,  ULL,  ULL,  ULL,  ULL },
    };

    assert(L < LastPromotedArithmeticType);
    assert(R < LastPromotedArithmeticType);
    int Idx = ConversionsTable[L][R];

    // Fast path: the table gives us a concrete answer.
    if (Idx != Dep) return getArithmeticType(Idx);

    // Slow path: we need to compare widths.
    // An invariant is that the signed type has higher rank.
    CanQualType LT = getArithmeticType(L),
                RT = getArithmeticType(R);
    unsigned LW = S.Context.getIntWidth(LT),
             RW = S.Context.getIntWidth(RT);

    // If they're different widths, use the signed type.
    if (LW > RW) return LT;
    else if (LW < RW) return RT;

    // Otherwise, use the unsigned type of the signed type's rank.
    if (L == SL || R == SL) return S.Context.UnsignedLongTy;
    assert(L == SLL || R == SLL);
    return S.Context.UnsignedLongLongTy;
  }

  /// \brief Helper method to factor out the common pattern of adding overloads
  /// for '++' and '--' builtin operators.
  void addPlusPlusMinusMinusStyleOverloads(QualType CandidateTy,
                                           bool HasVolatile) {
    QualType ParamTypes[2] = {
      S.Context.getLValueReferenceType(CandidateTy),
      S.Context.IntTy
    };

    // Non-volatile version.
    if (NumArgs == 1)
      S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
    else
      S.AddBuiltinCandidate(CandidateTy, ParamTypes, Args, 2, CandidateSet);

    // Use a heuristic to reduce number of builtin candidates in the set:
    // add volatile version only if there are conversions to a volatile type.
    if (HasVolatile) {
      ParamTypes[0] =
        S.Context.getLValueReferenceType(
          S.Context.getVolatileType(CandidateTy));
      if (NumArgs == 1)
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
      else
        S.AddBuiltinCandidate(CandidateTy, ParamTypes, Args, 2, CandidateSet);
    }
  }

public:
  BuiltinOperatorOverloadBuilder(
    Sema &S, Expr **Args, unsigned NumArgs,
    Qualifiers VisibleTypeConversionsQuals,
    bool HasArithmeticOrEnumeralCandidateType,
    llvm::SmallVectorImpl<BuiltinCandidateTypeSet> &CandidateTypes,
    OverloadCandidateSet &CandidateSet)
    : S(S), Args(Args), NumArgs(NumArgs),
      VisibleTypeConversionsQuals(VisibleTypeConversionsQuals),
      HasArithmeticOrEnumeralCandidateType(
        HasArithmeticOrEnumeralCandidateType),
      CandidateTypes(CandidateTypes),
      CandidateSet(CandidateSet) {
    // Validate some of our static helper constants in debug builds.
    assert(getArithmeticType(FirstPromotedIntegralType) == S.Context.IntTy &&
           "Invalid first promoted integral type");
    assert(getArithmeticType(LastPromotedIntegralType - 1)
             == S.Context.UnsignedLongLongTy &&
           "Invalid last promoted integral type");
    assert(getArithmeticType(FirstPromotedArithmeticType)
             == S.Context.FloatTy &&
           "Invalid first promoted arithmetic type");
    assert(getArithmeticType(LastPromotedArithmeticType - 1)
             == S.Context.UnsignedLongLongTy &&
           "Invalid last promoted arithmetic type");
  }

  // C++ [over.built]p3:
  //
  //   For every pair (T, VQ), where T is an arithmetic type, and VQ
  //   is either volatile or empty, there exist candidate operator
  //   functions of the form
  //
  //       VQ T&      operator++(VQ T&);
  //       T          operator++(VQ T&, int);
  //
  // C++ [over.built]p4:
  //
  //   For every pair (T, VQ), where T is an arithmetic type other
  //   than bool, and VQ is either volatile or empty, there exist
  //   candidate operator functions of the form
  //
  //       VQ T&      operator--(VQ T&);
  //       T          operator--(VQ T&, int);
  void addPlusPlusMinusMinusArithmeticOverloads(OverloadedOperatorKind Op) {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Arith = (Op == OO_PlusPlus? 0 : 1);
         Arith < NumArithmeticTypes; ++Arith) {
      addPlusPlusMinusMinusStyleOverloads(
        getArithmeticType(Arith),
        VisibleTypeConversionsQuals.hasVolatile());
    }
  }

  // C++ [over.built]p5:
  //
  //   For every pair (T, VQ), where T is a cv-qualified or
  //   cv-unqualified object type, and VQ is either volatile or
  //   empty, there exist candidate operator functions of the form
  //
  //       T*VQ&      operator++(T*VQ&);
  //       T*VQ&      operator--(T*VQ&);
  //       T*         operator++(T*VQ&, int);
  //       T*         operator--(T*VQ&, int);
  void addPlusPlusMinusMinusPointerOverloads() {
    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      // Skip pointer types that aren't pointers to object types.
      if (!(*Ptr)->getPointeeType()->isObjectType())
        continue;

      addPlusPlusMinusMinusStyleOverloads(*Ptr,
        (!S.Context.getCanonicalType(*Ptr).isVolatileQualified() &&
         VisibleTypeConversionsQuals.hasVolatile()));
    }
  }

  // C++ [over.built]p6:
  //   For every cv-qualified or cv-unqualified object type T, there
  //   exist candidate operator functions of the form
  //
  //       T&         operator*(T*);
  //
  // C++ [over.built]p7:
  //   For every function type T that does not have cv-qualifiers or a
  //   ref-qualifier, there exist candidate operator functions of the form
  //       T&         operator*(T*);
  void addUnaryStarPointerOverloads() {
    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      QualType ParamTy = *Ptr;
      QualType PointeeTy = ParamTy->getPointeeType();
      if (!PointeeTy->isObjectType() && !PointeeTy->isFunctionType())
        continue;

      if (const FunctionProtoType *Proto =PointeeTy->getAs<FunctionProtoType>())
        if (Proto->getTypeQuals() || Proto->getRefQualifier())
          continue;

      S.AddBuiltinCandidate(S.Context.getLValueReferenceType(PointeeTy),
                            &ParamTy, Args, 1, CandidateSet);
    }
  }

  // C++ [over.built]p9:
  //  For every promoted arithmetic type T, there exist candidate
  //  operator functions of the form
  //
  //       T         operator+(T);
  //       T         operator-(T);
  void addUnaryPlusOrMinusArithmeticOverloads() {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Arith = FirstPromotedArithmeticType;
         Arith < LastPromotedArithmeticType; ++Arith) {
      QualType ArithTy = getArithmeticType(Arith);
      S.AddBuiltinCandidate(ArithTy, &ArithTy, Args, 1, CandidateSet);
    }

    // Extension: We also add these operators for vector types.
    for (BuiltinCandidateTypeSet::iterator
              Vec = CandidateTypes[0].vector_begin(),
           VecEnd = CandidateTypes[0].vector_end();
         Vec != VecEnd; ++Vec) {
      QualType VecTy = *Vec;
      S.AddBuiltinCandidate(VecTy, &VecTy, Args, 1, CandidateSet);
    }
  }

  // C++ [over.built]p8:
  //   For every type T, there exist candidate operator functions of
  //   the form
  //
  //       T*         operator+(T*);
  void addUnaryPlusPointerOverloads() {
    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      QualType ParamTy = *Ptr;
      S.AddBuiltinCandidate(ParamTy, &ParamTy, Args, 1, CandidateSet);
    }
  }

  // C++ [over.built]p10:
  //   For every promoted integral type T, there exist candidate
  //   operator functions of the form
  //
  //        T         operator~(T);
  void addUnaryTildePromotedIntegralOverloads() {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Int = FirstPromotedIntegralType;
         Int < LastPromotedIntegralType; ++Int) {
      QualType IntTy = getArithmeticType(Int);
      S.AddBuiltinCandidate(IntTy, &IntTy, Args, 1, CandidateSet);
    }

    // Extension: We also add this operator for vector types.
    for (BuiltinCandidateTypeSet::iterator
              Vec = CandidateTypes[0].vector_begin(),
           VecEnd = CandidateTypes[0].vector_end();
         Vec != VecEnd; ++Vec) {
      QualType VecTy = *Vec;
      S.AddBuiltinCandidate(VecTy, &VecTy, Args, 1, CandidateSet);
    }
  }

  // C++ [over.match.oper]p16:
  //   For every pointer to member type T, there exist candidate operator
  //   functions of the form
  //
  //        bool operator==(T,T);
  //        bool operator!=(T,T);
  void addEqualEqualOrNotEqualMemberPointerOverloads() {
    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
      for (BuiltinCandidateTypeSet::iterator
                MemPtr = CandidateTypes[ArgIdx].member_pointer_begin(),
             MemPtrEnd = CandidateTypes[ArgIdx].member_pointer_end();
           MemPtr != MemPtrEnd;
           ++MemPtr) {
        // Don't add the same builtin candidate twice.
        if (!AddedTypes.insert(S.Context.getCanonicalType(*MemPtr)))
          continue;

        QualType ParamTypes[2] = { *MemPtr, *MemPtr };
        S.AddBuiltinCandidate(S.Context.BoolTy, ParamTypes, Args, 2,
                              CandidateSet);
      }
    }
  }

  // C++ [over.built]p15:
  //
  //   For every T, where T is an enumeration type, a pointer type, or 
  //   std::nullptr_t, there exist candidate operator functions of the form
  //
  //        bool       operator<(T, T);
  //        bool       operator>(T, T);
  //        bool       operator<=(T, T);
  //        bool       operator>=(T, T);
  //        bool       operator==(T, T);
  //        bool       operator!=(T, T);
  void addRelationalPointerOrEnumeralOverloads() {
    // C++ [over.built]p1:
    //   If there is a user-written candidate with the same name and parameter
    //   types as a built-in candidate operator function, the built-in operator
    //   function is hidden and is not included in the set of candidate
    //   functions.
    //
    // The text is actually in a note, but if we don't implement it then we end
    // up with ambiguities when the user provides an overloaded operator for
    // an enumeration type. Note that only enumeration types have this problem,
    // so we track which enumeration types we've seen operators for. Also, the
    // only other overloaded operator with enumeration argumenst, operator=,
    // cannot be overloaded for enumeration types, so this is the only place
    // where we must suppress candidates like this.
    llvm::DenseSet<std::pair<CanQualType, CanQualType> >
      UserDefinedBinaryOperators;

    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
      if (CandidateTypes[ArgIdx].enumeration_begin() !=
          CandidateTypes[ArgIdx].enumeration_end()) {
        for (OverloadCandidateSet::iterator C = CandidateSet.begin(),
                                         CEnd = CandidateSet.end();
             C != CEnd; ++C) {
          if (!C->Viable || !C->Function || C->Function->getNumParams() != 2)
            continue;

          QualType FirstParamType =
            C->Function->getParamDecl(0)->getType().getUnqualifiedType();
          QualType SecondParamType =
            C->Function->getParamDecl(1)->getType().getUnqualifiedType();

          // Skip if either parameter isn't of enumeral type.
          if (!FirstParamType->isEnumeralType() ||
              !SecondParamType->isEnumeralType())
            continue;

          // Add this operator to the set of known user-defined operators.
          UserDefinedBinaryOperators.insert(
            std::make_pair(S.Context.getCanonicalType(FirstParamType),
                           S.Context.getCanonicalType(SecondParamType)));
        }
      }
    }

    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
      for (BuiltinCandidateTypeSet::iterator
                Ptr = CandidateTypes[ArgIdx].pointer_begin(),
             PtrEnd = CandidateTypes[ArgIdx].pointer_end();
           Ptr != PtrEnd; ++Ptr) {
        // Don't add the same builtin candidate twice.
        if (!AddedTypes.insert(S.Context.getCanonicalType(*Ptr)))
          continue;

        QualType ParamTypes[2] = { *Ptr, *Ptr };
        S.AddBuiltinCandidate(S.Context.BoolTy, ParamTypes, Args, 2,
                              CandidateSet);
      }
      for (BuiltinCandidateTypeSet::iterator
                Enum = CandidateTypes[ArgIdx].enumeration_begin(),
             EnumEnd = CandidateTypes[ArgIdx].enumeration_end();
           Enum != EnumEnd; ++Enum) {
        CanQualType CanonType = S.Context.getCanonicalType(*Enum);

        // Don't add the same builtin candidate twice, or if a user defined
        // candidate exists.
        if (!AddedTypes.insert(CanonType) ||
            UserDefinedBinaryOperators.count(std::make_pair(CanonType,
                                                            CanonType)))
          continue;

        QualType ParamTypes[2] = { *Enum, *Enum };
        S.AddBuiltinCandidate(S.Context.BoolTy, ParamTypes, Args, 2,
                              CandidateSet);
      }
      
      if (CandidateTypes[ArgIdx].hasNullPtrType()) {
        CanQualType NullPtrTy = S.Context.getCanonicalType(S.Context.NullPtrTy);
        if (AddedTypes.insert(NullPtrTy) &&
            !UserDefinedBinaryOperators.count(std::make_pair(NullPtrTy, 
                                                             NullPtrTy))) {
          QualType ParamTypes[2] = { NullPtrTy, NullPtrTy };
          S.AddBuiltinCandidate(S.Context.BoolTy, ParamTypes, Args, 2, 
                                CandidateSet);
        }
      }
    }
  }

  // C++ [over.built]p13:
  //
  //   For every cv-qualified or cv-unqualified object type T
  //   there exist candidate operator functions of the form
  //
  //      T*         operator+(T*, ptrdiff_t);
  //      T&         operator[](T*, ptrdiff_t);    [BELOW]
  //      T*         operator-(T*, ptrdiff_t);
  //      T*         operator+(ptrdiff_t, T*);
  //      T&         operator[](ptrdiff_t, T*);    [BELOW]
  //
  // C++ [over.built]p14:
  //
  //   For every T, where T is a pointer to object type, there
  //   exist candidate operator functions of the form
  //
  //      ptrdiff_t  operator-(T, T);
  void addBinaryPlusOrMinusPointerOverloads(OverloadedOperatorKind Op) {
    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (int Arg = 0; Arg < 2; ++Arg) {
      QualType AsymetricParamTypes[2] = {
        S.Context.getPointerDiffType(),
        S.Context.getPointerDiffType(),
      };
      for (BuiltinCandidateTypeSet::iterator
                Ptr = CandidateTypes[Arg].pointer_begin(),
             PtrEnd = CandidateTypes[Arg].pointer_end();
           Ptr != PtrEnd; ++Ptr) {
        QualType PointeeTy = (*Ptr)->getPointeeType();
        if (!PointeeTy->isObjectType())
          continue;

        AsymetricParamTypes[Arg] = *Ptr;
        if (Arg == 0 || Op == OO_Plus) {
          // operator+(T*, ptrdiff_t) or operator-(T*, ptrdiff_t)
          // T* operator+(ptrdiff_t, T*);
          S.AddBuiltinCandidate(*Ptr, AsymetricParamTypes, Args, 2,
                                CandidateSet);
        }
        if (Op == OO_Minus) {
          // ptrdiff_t operator-(T, T);
          if (!AddedTypes.insert(S.Context.getCanonicalType(*Ptr)))
            continue;

          QualType ParamTypes[2] = { *Ptr, *Ptr };
          S.AddBuiltinCandidate(S.Context.getPointerDiffType(), ParamTypes,
                                Args, 2, CandidateSet);
        }
      }
    }
  }

  // C++ [over.built]p12:
  //
  //   For every pair of promoted arithmetic types L and R, there
  //   exist candidate operator functions of the form
  //
  //        LR         operator*(L, R);
  //        LR         operator/(L, R);
  //        LR         operator+(L, R);
  //        LR         operator-(L, R);
  //        bool       operator<(L, R);
  //        bool       operator>(L, R);
  //        bool       operator<=(L, R);
  //        bool       operator>=(L, R);
  //        bool       operator==(L, R);
  //        bool       operator!=(L, R);
  //
  //   where LR is the result of the usual arithmetic conversions
  //   between types L and R.
  //
  // C++ [over.built]p24:
  //
  //   For every pair of promoted arithmetic types L and R, there exist
  //   candidate operator functions of the form
  //
  //        LR       operator?(bool, L, R);
  //
  //   where LR is the result of the usual arithmetic conversions
  //   between types L and R.
  // Our candidates ignore the first parameter.
  void addGenericBinaryArithmeticOverloads(bool isComparison) {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Left = FirstPromotedArithmeticType;
         Left < LastPromotedArithmeticType; ++Left) {
      for (unsigned Right = FirstPromotedArithmeticType;
           Right < LastPromotedArithmeticType; ++Right) {
        QualType LandR[2] = { getArithmeticType(Left),
                              getArithmeticType(Right) };
        QualType Result =
          isComparison ? S.Context.BoolTy
                       : getUsualArithmeticConversions(Left, Right);
        S.AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }

    // Extension: Add the binary operators ==, !=, <, <=, >=, >, *, /, and the
    // conditional operator for vector types.
    for (BuiltinCandidateTypeSet::iterator
              Vec1 = CandidateTypes[0].vector_begin(),
           Vec1End = CandidateTypes[0].vector_end();
         Vec1 != Vec1End; ++Vec1) {
      for (BuiltinCandidateTypeSet::iterator
                Vec2 = CandidateTypes[1].vector_begin(),
             Vec2End = CandidateTypes[1].vector_end();
           Vec2 != Vec2End; ++Vec2) {
        QualType LandR[2] = { *Vec1, *Vec2 };
        QualType Result = S.Context.BoolTy;
        if (!isComparison) {
          if ((*Vec1)->isExtVectorType() || !(*Vec2)->isExtVectorType())
            Result = *Vec1;
          else
            Result = *Vec2;
        }

        S.AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }
  }

  // C++ [over.built]p17:
  //
  //   For every pair of promoted integral types L and R, there
  //   exist candidate operator functions of the form
  //
  //      LR         operator%(L, R);
  //      LR         operator&(L, R);
  //      LR         operator^(L, R);
  //      LR         operator|(L, R);
  //      L          operator<<(L, R);
  //      L          operator>>(L, R);
  //
  //   where LR is the result of the usual arithmetic conversions
  //   between types L and R.
  void addBinaryBitwiseArithmeticOverloads(OverloadedOperatorKind Op) {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Left = FirstPromotedIntegralType;
         Left < LastPromotedIntegralType; ++Left) {
      for (unsigned Right = FirstPromotedIntegralType;
           Right < LastPromotedIntegralType; ++Right) {
        QualType LandR[2] = { getArithmeticType(Left),
                              getArithmeticType(Right) };
        QualType Result = (Op == OO_LessLess || Op == OO_GreaterGreater)
            ? LandR[0]
            : getUsualArithmeticConversions(Left, Right);
        S.AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }
  }

  // C++ [over.built]p20:
  //
  //   For every pair (T, VQ), where T is an enumeration or
  //   pointer to member type and VQ is either volatile or
  //   empty, there exist candidate operator functions of the form
  //
  //        VQ T&      operator=(VQ T&, T);
  void addAssignmentMemberPointerOrEnumeralOverloads() {
    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (unsigned ArgIdx = 0; ArgIdx < 2; ++ArgIdx) {
      for (BuiltinCandidateTypeSet::iterator
                Enum = CandidateTypes[ArgIdx].enumeration_begin(),
             EnumEnd = CandidateTypes[ArgIdx].enumeration_end();
           Enum != EnumEnd; ++Enum) {
        if (!AddedTypes.insert(S.Context.getCanonicalType(*Enum)))
          continue;

        AddBuiltinAssignmentOperatorCandidates(S, *Enum, Args, 2,
                                               CandidateSet);
      }

      for (BuiltinCandidateTypeSet::iterator
                MemPtr = CandidateTypes[ArgIdx].member_pointer_begin(),
             MemPtrEnd = CandidateTypes[ArgIdx].member_pointer_end();
           MemPtr != MemPtrEnd; ++MemPtr) {
        if (!AddedTypes.insert(S.Context.getCanonicalType(*MemPtr)))
          continue;

        AddBuiltinAssignmentOperatorCandidates(S, *MemPtr, Args, 2,
                                               CandidateSet);
      }
    }
  }

  // C++ [over.built]p19:
  //
  //   For every pair (T, VQ), where T is any type and VQ is either
  //   volatile or empty, there exist candidate operator functions
  //   of the form
  //
  //        T*VQ&      operator=(T*VQ&, T*);
  //
  // C++ [over.built]p21:
  //
  //   For every pair (T, VQ), where T is a cv-qualified or
  //   cv-unqualified object type and VQ is either volatile or
  //   empty, there exist candidate operator functions of the form
  //
  //        T*VQ&      operator+=(T*VQ&, ptrdiff_t);
  //        T*VQ&      operator-=(T*VQ&, ptrdiff_t);
  void addAssignmentPointerOverloads(bool isEqualOp) {
    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      // If this is operator=, keep track of the builtin candidates we added.
      if (isEqualOp)
        AddedTypes.insert(S.Context.getCanonicalType(*Ptr));
      else if (!(*Ptr)->getPointeeType()->isObjectType())
        continue;

      // non-volatile version
      QualType ParamTypes[2] = {
        S.Context.getLValueReferenceType(*Ptr),
        isEqualOp ? *Ptr : S.Context.getPointerDiffType(),
      };
      S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                            /*IsAssigmentOperator=*/ isEqualOp);

      if (!S.Context.getCanonicalType(*Ptr).isVolatileQualified() &&
          VisibleTypeConversionsQuals.hasVolatile()) {
        // volatile version
        ParamTypes[0] =
          S.Context.getLValueReferenceType(S.Context.getVolatileType(*Ptr));
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                              /*IsAssigmentOperator=*/isEqualOp);
      }
    }

    if (isEqualOp) {
      for (BuiltinCandidateTypeSet::iterator
                Ptr = CandidateTypes[1].pointer_begin(),
             PtrEnd = CandidateTypes[1].pointer_end();
           Ptr != PtrEnd; ++Ptr) {
        // Make sure we don't add the same candidate twice.
        if (!AddedTypes.insert(S.Context.getCanonicalType(*Ptr)))
          continue;

        QualType ParamTypes[2] = {
          S.Context.getLValueReferenceType(*Ptr),
          *Ptr,
        };

        // non-volatile version
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                              /*IsAssigmentOperator=*/true);

        if (!S.Context.getCanonicalType(*Ptr).isVolatileQualified() &&
            VisibleTypeConversionsQuals.hasVolatile()) {
          // volatile version
          ParamTypes[0] =
            S.Context.getLValueReferenceType(S.Context.getVolatileType(*Ptr));
          S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2,
                                CandidateSet, /*IsAssigmentOperator=*/true);
        }
      }
    }
  }

  // C++ [over.built]p18:
  //
  //   For every triple (L, VQ, R), where L is an arithmetic type,
  //   VQ is either volatile or empty, and R is a promoted
  //   arithmetic type, there exist candidate operator functions of
  //   the form
  //
  //        VQ L&      operator=(VQ L&, R);
  //        VQ L&      operator*=(VQ L&, R);
  //        VQ L&      operator/=(VQ L&, R);
  //        VQ L&      operator+=(VQ L&, R);
  //        VQ L&      operator-=(VQ L&, R);
  void addAssignmentArithmeticOverloads(bool isEqualOp) {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Left = 0; Left < NumArithmeticTypes; ++Left) {
      for (unsigned Right = FirstPromotedArithmeticType;
           Right < LastPromotedArithmeticType; ++Right) {
        QualType ParamTypes[2];
        ParamTypes[1] = getArithmeticType(Right);

        // Add this built-in operator as a candidate (VQ is empty).
        ParamTypes[0] =
          S.Context.getLValueReferenceType(getArithmeticType(Left));
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                              /*IsAssigmentOperator=*/isEqualOp);

        // Add this built-in operator as a candidate (VQ is 'volatile').
        if (VisibleTypeConversionsQuals.hasVolatile()) {
          ParamTypes[0] =
            S.Context.getVolatileType(getArithmeticType(Left));
          ParamTypes[0] = S.Context.getLValueReferenceType(ParamTypes[0]);
          S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2,
                                CandidateSet,
                                /*IsAssigmentOperator=*/isEqualOp);
        }
      }
    }

    // Extension: Add the binary operators =, +=, -=, *=, /= for vector types.
    for (BuiltinCandidateTypeSet::iterator
              Vec1 = CandidateTypes[0].vector_begin(),
           Vec1End = CandidateTypes[0].vector_end();
         Vec1 != Vec1End; ++Vec1) {
      for (BuiltinCandidateTypeSet::iterator
                Vec2 = CandidateTypes[1].vector_begin(),
             Vec2End = CandidateTypes[1].vector_end();
           Vec2 != Vec2End; ++Vec2) {
        QualType ParamTypes[2];
        ParamTypes[1] = *Vec2;
        // Add this built-in operator as a candidate (VQ is empty).
        ParamTypes[0] = S.Context.getLValueReferenceType(*Vec1);
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                              /*IsAssigmentOperator=*/isEqualOp);

        // Add this built-in operator as a candidate (VQ is 'volatile').
        if (VisibleTypeConversionsQuals.hasVolatile()) {
          ParamTypes[0] = S.Context.getVolatileType(*Vec1);
          ParamTypes[0] = S.Context.getLValueReferenceType(ParamTypes[0]);
          S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2,
                                CandidateSet,
                                /*IsAssigmentOperator=*/isEqualOp);
        }
      }
    }
  }

  // C++ [over.built]p22:
  //
  //   For every triple (L, VQ, R), where L is an integral type, VQ
  //   is either volatile or empty, and R is a promoted integral
  //   type, there exist candidate operator functions of the form
  //
  //        VQ L&       operator%=(VQ L&, R);
  //        VQ L&       operator<<=(VQ L&, R);
  //        VQ L&       operator>>=(VQ L&, R);
  //        VQ L&       operator&=(VQ L&, R);
  //        VQ L&       operator^=(VQ L&, R);
  //        VQ L&       operator|=(VQ L&, R);
  void addAssignmentIntegralOverloads() {
    if (!HasArithmeticOrEnumeralCandidateType)
      return;

    for (unsigned Left = FirstIntegralType; Left < LastIntegralType; ++Left) {
      for (unsigned Right = FirstPromotedIntegralType;
           Right < LastPromotedIntegralType; ++Right) {
        QualType ParamTypes[2];
        ParamTypes[1] = getArithmeticType(Right);

        // Add this built-in operator as a candidate (VQ is empty).
        ParamTypes[0] =
          S.Context.getLValueReferenceType(getArithmeticType(Left));
        S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
        if (VisibleTypeConversionsQuals.hasVolatile()) {
          // Add this built-in operator as a candidate (VQ is 'volatile').
          ParamTypes[0] = getArithmeticType(Left);
          ParamTypes[0] = S.Context.getVolatileType(ParamTypes[0]);
          ParamTypes[0] = S.Context.getLValueReferenceType(ParamTypes[0]);
          S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2,
                                CandidateSet);
        }
      }
    }
  }

  // C++ [over.operator]p23:
  //
  //   There also exist candidate operator functions of the form
  //
  //        bool        operator!(bool);
  //        bool        operator&&(bool, bool);
  //        bool        operator||(bool, bool);
  void addExclaimOverload() {
    QualType ParamTy = S.Context.BoolTy;
    S.AddBuiltinCandidate(ParamTy, &ParamTy, Args, 1, CandidateSet,
                          /*IsAssignmentOperator=*/false,
                          /*NumContextualBoolArguments=*/1);
  }
  void addAmpAmpOrPipePipeOverload() {
    QualType ParamTypes[2] = { S.Context.BoolTy, S.Context.BoolTy };
    S.AddBuiltinCandidate(S.Context.BoolTy, ParamTypes, Args, 2, CandidateSet,
                          /*IsAssignmentOperator=*/false,
                          /*NumContextualBoolArguments=*/2);
  }

  // C++ [over.built]p13:
  //
  //   For every cv-qualified or cv-unqualified object type T there
  //   exist candidate operator functions of the form
  //
  //        T*         operator+(T*, ptrdiff_t);     [ABOVE]
  //        T&         operator[](T*, ptrdiff_t);
  //        T*         operator-(T*, ptrdiff_t);     [ABOVE]
  //        T*         operator+(ptrdiff_t, T*);     [ABOVE]
  //        T&         operator[](ptrdiff_t, T*);
  void addSubscriptOverloads() {
    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      QualType ParamTypes[2] = { *Ptr, S.Context.getPointerDiffType() };
      QualType PointeeType = (*Ptr)->getPointeeType();
      if (!PointeeType->isObjectType())
        continue;

      QualType ResultTy = S.Context.getLValueReferenceType(PointeeType);

      // T& operator[](T*, ptrdiff_t)
      S.AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);
    }

    for (BuiltinCandidateTypeSet::iterator
              Ptr = CandidateTypes[1].pointer_begin(),
           PtrEnd = CandidateTypes[1].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      QualType ParamTypes[2] = { S.Context.getPointerDiffType(), *Ptr };
      QualType PointeeType = (*Ptr)->getPointeeType();
      if (!PointeeType->isObjectType())
        continue;

      QualType ResultTy = S.Context.getLValueReferenceType(PointeeType);

      // T& operator[](ptrdiff_t, T*)
      S.AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);
    }
  }

  // C++ [over.built]p11:
  //    For every quintuple (C1, C2, T, CV1, CV2), where C2 is a class type,
  //    C1 is the same type as C2 or is a derived class of C2, T is an object
  //    type or a function type, and CV1 and CV2 are cv-qualifier-seqs,
  //    there exist candidate operator functions of the form
  //
  //      CV12 T& operator->*(CV1 C1*, CV2 T C2::*);
  //
  //    where CV12 is the union of CV1 and CV2.
  void addArrowStarOverloads() {
    for (BuiltinCandidateTypeSet::iterator
             Ptr = CandidateTypes[0].pointer_begin(),
           PtrEnd = CandidateTypes[0].pointer_end();
         Ptr != PtrEnd; ++Ptr) {
      QualType C1Ty = (*Ptr);
      QualType C1;
      QualifierCollector Q1;
      C1 = QualType(Q1.strip(C1Ty->getPointeeType()), 0);
      if (!isa<RecordType>(C1))
        continue;
      // heuristic to reduce number of builtin candidates in the set.
      // Add volatile/restrict version only if there are conversions to a
      // volatile/restrict type.
      if (!VisibleTypeConversionsQuals.hasVolatile() && Q1.hasVolatile())
        continue;
      if (!VisibleTypeConversionsQuals.hasRestrict() && Q1.hasRestrict())
        continue;
      for (BuiltinCandidateTypeSet::iterator
                MemPtr = CandidateTypes[1].member_pointer_begin(),
             MemPtrEnd = CandidateTypes[1].member_pointer_end();
           MemPtr != MemPtrEnd; ++MemPtr) {
        const MemberPointerType *mptr = cast<MemberPointerType>(*MemPtr);
        QualType C2 = QualType(mptr->getClass(), 0);
        C2 = C2.getUnqualifiedType();
        if (C1 != C2 && !S.IsDerivedFrom(C1, C2))
          break;
        QualType ParamTypes[2] = { *Ptr, *MemPtr };
        // build CV12 T&
        QualType T = mptr->getPointeeType();
        if (!VisibleTypeConversionsQuals.hasVolatile() &&
            T.isVolatileQualified())
          continue;
        if (!VisibleTypeConversionsQuals.hasRestrict() &&
            T.isRestrictQualified())
          continue;
        T = Q1.apply(S.Context, T);
        QualType ResultTy = S.Context.getLValueReferenceType(T);
        S.AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);
      }
    }
  }

  // Note that we don't consider the first argument, since it has been
  // contextually converted to bool long ago. The candidates below are
  // therefore added as binary.
  //
  // C++ [over.built]p25:
  //   For every type T, where T is a pointer, pointer-to-member, or scoped
  //   enumeration type, there exist candidate operator functions of the form
  //
  //        T        operator?(bool, T, T);
  //
  void addConditionalOperatorOverloads() {
    /// Set of (canonical) types that we've already handled.
    llvm::SmallPtrSet<QualType, 8> AddedTypes;

    for (unsigned ArgIdx = 0; ArgIdx < 2; ++ArgIdx) {
      for (BuiltinCandidateTypeSet::iterator
                Ptr = CandidateTypes[ArgIdx].pointer_begin(),
             PtrEnd = CandidateTypes[ArgIdx].pointer_end();
           Ptr != PtrEnd; ++Ptr) {
        if (!AddedTypes.insert(S.Context.getCanonicalType(*Ptr)))
          continue;

        QualType ParamTypes[2] = { *Ptr, *Ptr };
        S.AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);
      }

      for (BuiltinCandidateTypeSet::iterator
                MemPtr = CandidateTypes[ArgIdx].member_pointer_begin(),
             MemPtrEnd = CandidateTypes[ArgIdx].member_pointer_end();
           MemPtr != MemPtrEnd; ++MemPtr) {
        if (!AddedTypes.insert(S.Context.getCanonicalType(*MemPtr)))
          continue;

        QualType ParamTypes[2] = { *MemPtr, *MemPtr };
        S.AddBuiltinCandidate(*MemPtr, ParamTypes, Args, 2, CandidateSet);
      }

      if (S.getLangOptions().CPlusPlus0x) {
        for (BuiltinCandidateTypeSet::iterator
                  Enum = CandidateTypes[ArgIdx].enumeration_begin(),
               EnumEnd = CandidateTypes[ArgIdx].enumeration_end();
             Enum != EnumEnd; ++Enum) {
          if (!(*Enum)->getAs<EnumType>()->getDecl()->isScoped())
            continue;

          if (!AddedTypes.insert(S.Context.getCanonicalType(*Enum)))
            continue;

          QualType ParamTypes[2] = { *Enum, *Enum };
          S.AddBuiltinCandidate(*Enum, ParamTypes, Args, 2, CandidateSet);
        }
      }
    }
  }
};

} // end anonymous namespace

/// AddBuiltinOperatorCandidates - Add the appropriate built-in
/// operator overloads to the candidate set (C++ [over.built]), based
/// on the operator @p Op and the arguments given. For example, if the
/// operator is a binary '+', this routine might add "int
/// operator+(int, int)" to cover integer addition.
void
Sema::AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
                                   SourceLocation OpLoc,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet) {
  // Find all of the types that the arguments can convert to, but only
  // if the operator we're looking at has built-in operator candidates
  // that make use of these types. Also record whether we encounter non-record
  // candidate types or either arithmetic or enumeral candidate types.
  Qualifiers VisibleTypeConversionsQuals;
  VisibleTypeConversionsQuals.addConst();
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    VisibleTypeConversionsQuals += CollectVRQualifiers(Context, Args[ArgIdx]);

  bool HasNonRecordCandidateType = false;
  bool HasArithmeticOrEnumeralCandidateType = false;
  llvm::SmallVector<BuiltinCandidateTypeSet, 2> CandidateTypes;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    CandidateTypes.push_back(BuiltinCandidateTypeSet(*this));
    CandidateTypes[ArgIdx].AddTypesConvertedFrom(Args[ArgIdx]->getType(),
                                                 OpLoc,
                                                 true,
                                                 (Op == OO_Exclaim ||
                                                  Op == OO_AmpAmp ||
                                                  Op == OO_PipePipe),
                                                 VisibleTypeConversionsQuals);
    HasNonRecordCandidateType = HasNonRecordCandidateType ||
        CandidateTypes[ArgIdx].hasNonRecordTypes();
    HasArithmeticOrEnumeralCandidateType =
        HasArithmeticOrEnumeralCandidateType ||
        CandidateTypes[ArgIdx].hasArithmeticOrEnumeralTypes();
  }

  // Exit early when no non-record types have been added to the candidate set
  // for any of the arguments to the operator.
  if (!HasNonRecordCandidateType)
    return;

  // Setup an object to manage the common state for building overloads.
  BuiltinOperatorOverloadBuilder OpBuilder(*this, Args, NumArgs,
                                           VisibleTypeConversionsQuals,
                                           HasArithmeticOrEnumeralCandidateType,
                                           CandidateTypes, CandidateSet);

  // Dispatch over the operation to add in only those overloads which apply.
  switch (Op) {
  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    assert(false && "Expected an overloaded operator");
    break;

  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Call:
    assert(false && "Special operators don't use AddBuiltinOperatorCandidates");
    break;

  case OO_Comma:
  case OO_Arrow:
    // C++ [over.match.oper]p3:
    //   -- For the operator ',', the unary operator '&', or the
    //      operator '->', the built-in candidates set is empty.
    break;

  case OO_Plus: // '+' is either unary or binary
    if (NumArgs == 1)
      OpBuilder.addUnaryPlusPointerOverloads();
    // Fall through.

  case OO_Minus: // '-' is either unary or binary
    if (NumArgs == 1) {
      OpBuilder.addUnaryPlusOrMinusArithmeticOverloads();
    } else {
      OpBuilder.addBinaryPlusOrMinusPointerOverloads(Op);
      OpBuilder.addGenericBinaryArithmeticOverloads(/*isComparison=*/false);
    }
    break;

  case OO_Star: // '*' is either unary or binary
    if (NumArgs == 1)
      OpBuilder.addUnaryStarPointerOverloads();
    else
      OpBuilder.addGenericBinaryArithmeticOverloads(/*isComparison=*/false);
    break;

  case OO_Slash:
    OpBuilder.addGenericBinaryArithmeticOverloads(/*isComparison=*/false);
    break;

  case OO_PlusPlus:
  case OO_MinusMinus:
    OpBuilder.addPlusPlusMinusMinusArithmeticOverloads(Op);
    OpBuilder.addPlusPlusMinusMinusPointerOverloads();
    break;

  case OO_EqualEqual:
  case OO_ExclaimEqual:
    OpBuilder.addEqualEqualOrNotEqualMemberPointerOverloads();
    // Fall through.

  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
    OpBuilder.addRelationalPointerOrEnumeralOverloads();
    OpBuilder.addGenericBinaryArithmeticOverloads(/*isComparison=*/true);
    break;

  case OO_Percent:
  case OO_Caret:
  case OO_Pipe:
  case OO_LessLess:
  case OO_GreaterGreater:
    OpBuilder.addBinaryBitwiseArithmeticOverloads(Op);
    break;

  case OO_Amp: // '&' is either unary or binary
    if (NumArgs == 1)
      // C++ [over.match.oper]p3:
      //   -- For the operator ',', the unary operator '&', or the
      //      operator '->', the built-in candidates set is empty.
      break;

    OpBuilder.addBinaryBitwiseArithmeticOverloads(Op);
    break;

  case OO_Tilde:
    OpBuilder.addUnaryTildePromotedIntegralOverloads();
    break;

  case OO_Equal:
    OpBuilder.addAssignmentMemberPointerOrEnumeralOverloads();
    // Fall through.

  case OO_PlusEqual:
  case OO_MinusEqual:
    OpBuilder.addAssignmentPointerOverloads(Op == OO_Equal);
    // Fall through.

  case OO_StarEqual:
  case OO_SlashEqual:
    OpBuilder.addAssignmentArithmeticOverloads(Op == OO_Equal);
    break;

  case OO_PercentEqual:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_AmpEqual:
  case OO_CaretEqual:
  case OO_PipeEqual:
    OpBuilder.addAssignmentIntegralOverloads();
    break;

  case OO_Exclaim:
    OpBuilder.addExclaimOverload();
    break;

  case OO_AmpAmp:
  case OO_PipePipe:
    OpBuilder.addAmpAmpOrPipePipeOverload();
    break;

  case OO_Subscript:
    OpBuilder.addSubscriptOverloads();
    break;

  case OO_ArrowStar:
    OpBuilder.addArrowStarOverloads();
    break;

  case OO_Conditional:
    OpBuilder.addConditionalOperatorOverloads();
    OpBuilder.addGenericBinaryArithmeticOverloads(/*isComparison=*/false);
    break;
  }
}

/// \brief Add function candidates found via argument-dependent lookup
/// to the set of overloading candidates.
///
/// This routine performs argument-dependent name lookup based on the
/// given function name (which may also be an operator name) and adds
/// all of the overload candidates found by ADL to the overload
/// candidate set (C++ [basic.lookup.argdep]).
void
Sema::AddArgumentDependentLookupCandidates(DeclarationName Name,
                                           bool Operator,
                                           Expr **Args, unsigned NumArgs,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                           OverloadCandidateSet& CandidateSet,
                                           bool PartialOverloading,
                                           bool StdNamespaceIsAssociated) {
  ADLResult Fns;

  // FIXME: This approach for uniquing ADL results (and removing
  // redundant candidates from the set) relies on pointer-equality,
  // which means we need to key off the canonical decl.  However,
  // always going back to the canonical decl might not get us the
  // right set of default arguments.  What default arguments are
  // we supposed to consider on ADL candidates, anyway?

  // FIXME: Pass in the explicit template arguments?
  ArgumentDependentLookup(Name, Operator, Args, NumArgs, Fns,
                          StdNamespaceIsAssociated);

  // Erase all of the candidates we already knew about.
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                                   CandEnd = CandidateSet.end();
       Cand != CandEnd; ++Cand)
    if (Cand->Function) {
      Fns.erase(Cand->Function);
      if (FunctionTemplateDecl *FunTmpl = Cand->Function->getPrimaryTemplate())
        Fns.erase(FunTmpl);
    }

  // For each of the ADL candidates we found, add it to the overload
  // set.
  for (ADLResult::iterator I = Fns.begin(), E = Fns.end(); I != E; ++I) {
    DeclAccessPair FoundDecl = DeclAccessPair::make(*I, AS_none);
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*I)) {
      if (ExplicitTemplateArgs)
        continue;

      AddOverloadCandidate(FD, FoundDecl, Args, NumArgs, CandidateSet,
                           false, PartialOverloading);
    } else
      AddTemplateOverloadCandidate(cast<FunctionTemplateDecl>(*I),
                                   FoundDecl, ExplicitTemplateArgs,
                                   Args, NumArgs, CandidateSet);
  }
}

/// isBetterOverloadCandidate - Determines whether the first overload
/// candidate is a better candidate than the second (C++ 13.3.3p1).
bool
isBetterOverloadCandidate(Sema &S,
                          const OverloadCandidate &Cand1,
                          const OverloadCandidate &Cand2,
                          SourceLocation Loc,
                          bool UserDefinedConversion) {
  // Define viable functions to be better candidates than non-viable
  // functions.
  if (!Cand2.Viable)
    return Cand1.Viable;
  else if (!Cand1.Viable)
    return false;

  // C++ [over.match.best]p1:
  //
  //   -- if F is a static member function, ICS1(F) is defined such
  //      that ICS1(F) is neither better nor worse than ICS1(G) for
  //      any function G, and, symmetrically, ICS1(G) is neither
  //      better nor worse than ICS1(F).
  unsigned StartArg = 0;
  if (Cand1.IgnoreObjectArgument || Cand2.IgnoreObjectArgument)
    StartArg = 1;

  // C++ [over.match.best]p1:
  //   A viable function F1 is defined to be a better function than another
  //   viable function F2 if for all arguments i, ICSi(F1) is not a worse
  //   conversion sequence than ICSi(F2), and then...
  unsigned NumArgs = Cand1.Conversions.size();
  assert(Cand2.Conversions.size() == NumArgs && "Overload candidate mismatch");
  bool HasBetterConversion = false;
  for (unsigned ArgIdx = StartArg; ArgIdx < NumArgs; ++ArgIdx) {
    switch (CompareImplicitConversionSequences(S,
                                               Cand1.Conversions[ArgIdx],
                                               Cand2.Conversions[ArgIdx])) {
    case ImplicitConversionSequence::Better:
      // Cand1 has a better conversion sequence.
      HasBetterConversion = true;
      break;

    case ImplicitConversionSequence::Worse:
      // Cand1 can't be better than Cand2.
      return false;

    case ImplicitConversionSequence::Indistinguishable:
      // Do nothing.
      break;
    }
  }

  //    -- for some argument j, ICSj(F1) is a better conversion sequence than
  //       ICSj(F2), or, if not that,
  if (HasBetterConversion)
    return true;

  //     - F1 is a non-template function and F2 is a function template
  //       specialization, or, if not that,
  if ((!Cand1.Function || !Cand1.Function->getPrimaryTemplate()) &&
      Cand2.Function && Cand2.Function->getPrimaryTemplate())
    return true;

  //   -- F1 and F2 are function template specializations, and the function
  //      template for F1 is more specialized than the template for F2
  //      according to the partial ordering rules described in 14.5.5.2, or,
  //      if not that,
  if (Cand1.Function && Cand1.Function->getPrimaryTemplate() &&
      Cand2.Function && Cand2.Function->getPrimaryTemplate()) {
    if (FunctionTemplateDecl *BetterTemplate
          = S.getMoreSpecializedTemplate(Cand1.Function->getPrimaryTemplate(),
                                         Cand2.Function->getPrimaryTemplate(),
                                         Loc,
                       isa<CXXConversionDecl>(Cand1.Function)? TPOC_Conversion
                                                             : TPOC_Call,
                                         Cand1.ExplicitCallArguments))
      return BetterTemplate == Cand1.Function->getPrimaryTemplate();
  }

  //   -- the context is an initialization by user-defined conversion
  //      (see 8.5, 13.3.1.5) and the standard conversion sequence
  //      from the return type of F1 to the destination type (i.e.,
  //      the type of the entity being initialized) is a better
  //      conversion sequence than the standard conversion sequence
  //      from the return type of F2 to the destination type.
  if (UserDefinedConversion && Cand1.Function && Cand2.Function &&
      isa<CXXConversionDecl>(Cand1.Function) &&
      isa<CXXConversionDecl>(Cand2.Function)) {
    switch (CompareStandardConversionSequences(S,
                                               Cand1.FinalConversion,
                                               Cand2.FinalConversion)) {
    case ImplicitConversionSequence::Better:
      // Cand1 has a better conversion sequence.
      return true;

    case ImplicitConversionSequence::Worse:
      // Cand1 can't be better than Cand2.
      return false;

    case ImplicitConversionSequence::Indistinguishable:
      // Do nothing
      break;
    }
  }

  return false;
}

/// \brief Computes the best viable function (C++ 13.3.3)
/// within an overload candidate set.
///
/// \param CandidateSet the set of candidate functions.
///
/// \param Loc the location of the function name (or operator symbol) for
/// which overload resolution occurs.
///
/// \param Best f overload resolution was successful or found a deleted
/// function, Best points to the candidate function found.
///
/// \returns The result of overload resolution.
OverloadingResult
OverloadCandidateSet::BestViableFunction(Sema &S, SourceLocation Loc,
                                         iterator &Best,
                                         bool UserDefinedConversion) {
  // Find the best viable function.
  Best = end();
  for (iterator Cand = begin(); Cand != end(); ++Cand) {
    if (Cand->Viable)
      if (Best == end() || isBetterOverloadCandidate(S, *Cand, *Best, Loc,
                                                     UserDefinedConversion))
        Best = Cand;
  }

  // If we didn't find any viable functions, abort.
  if (Best == end())
    return OR_No_Viable_Function;

  // Make sure that this function is better than every other viable
  // function. If not, we have an ambiguity.
  for (iterator Cand = begin(); Cand != end(); ++Cand) {
    if (Cand->Viable &&
        Cand != Best &&
        !isBetterOverloadCandidate(S, *Best, *Cand, Loc,
                                   UserDefinedConversion)) {
      Best = end();
      return OR_Ambiguous;
    }
  }

  // Best is the best viable function.
  if (Best->Function &&
      (Best->Function->isDeleted() ||
       S.isFunctionConsideredUnavailable(Best->Function)))
    return OR_Deleted;

  return OR_Success;
}

namespace {

enum OverloadCandidateKind {
  oc_function,
  oc_method,
  oc_constructor,
  oc_function_template,
  oc_method_template,
  oc_constructor_template,
  oc_implicit_default_constructor,
  oc_implicit_copy_constructor,
  oc_implicit_move_constructor,
  oc_implicit_copy_assignment,
  oc_implicit_move_assignment,
  oc_implicit_inherited_constructor
};

OverloadCandidateKind ClassifyOverloadCandidate(Sema &S,
                                                FunctionDecl *Fn,
                                                std::string &Description) {
  bool isTemplate = false;

  if (FunctionTemplateDecl *FunTmpl = Fn->getPrimaryTemplate()) {
    isTemplate = true;
    Description = S.getTemplateArgumentBindingsText(
      FunTmpl->getTemplateParameters(), *Fn->getTemplateSpecializationArgs());
  }

  if (CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(Fn)) {
    if (!Ctor->isImplicit())
      return isTemplate ? oc_constructor_template : oc_constructor;

    if (Ctor->getInheritedConstructor())
      return oc_implicit_inherited_constructor;

    if (Ctor->isDefaultConstructor())
      return oc_implicit_default_constructor;

    if (Ctor->isMoveConstructor())
      return oc_implicit_move_constructor;

    assert(Ctor->isCopyConstructor() &&
           "unexpected sort of implicit constructor");
    return oc_implicit_copy_constructor;
  }

  if (CXXMethodDecl *Meth = dyn_cast<CXXMethodDecl>(Fn)) {
    // This actually gets spelled 'candidate function' for now, but
    // it doesn't hurt to split it out.
    if (!Meth->isImplicit())
      return isTemplate ? oc_method_template : oc_method;

    if (Meth->isMoveAssignmentOperator())
      return oc_implicit_move_assignment;

    assert(Meth->isCopyAssignmentOperator()
           && "implicit method is not copy assignment operator?");
    return oc_implicit_copy_assignment;
  }

  return isTemplate ? oc_function_template : oc_function;
}

void MaybeEmitInheritedConstructorNote(Sema &S, FunctionDecl *Fn) {
  const CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(Fn);
  if (!Ctor) return;

  Ctor = Ctor->getInheritedConstructor();
  if (!Ctor) return;

  S.Diag(Ctor->getLocation(), diag::note_ovl_candidate_inherited_constructor);
}

} // end anonymous namespace

// Notes the location of an overload candidate.
void Sema::NoteOverloadCandidate(FunctionDecl *Fn) {
  std::string FnDesc;
  OverloadCandidateKind K = ClassifyOverloadCandidate(*this, Fn, FnDesc);
  Diag(Fn->getLocation(), diag::note_ovl_candidate)
    << (unsigned) K << FnDesc;
  MaybeEmitInheritedConstructorNote(*this, Fn);
}

//Notes the location of all overload candidates designated through 
// OverloadedExpr
void Sema::NoteAllOverloadCandidates(Expr* OverloadedExpr) {
  assert(OverloadedExpr->getType() == Context.OverloadTy);

  OverloadExpr::FindResult Ovl = OverloadExpr::find(OverloadedExpr);
  OverloadExpr *OvlExpr = Ovl.Expression;

  for (UnresolvedSetIterator I = OvlExpr->decls_begin(),
                            IEnd = OvlExpr->decls_end(); 
       I != IEnd; ++I) {
    if (FunctionTemplateDecl *FunTmpl = 
                dyn_cast<FunctionTemplateDecl>((*I)->getUnderlyingDecl()) ) {
      NoteOverloadCandidate(FunTmpl->getTemplatedDecl());   
    } else if (FunctionDecl *Fun 
                      = dyn_cast<FunctionDecl>((*I)->getUnderlyingDecl()) ) {
      NoteOverloadCandidate(Fun);
    }
  }
}

/// Diagnoses an ambiguous conversion.  The partial diagnostic is the
/// "lead" diagnostic; it will be given two arguments, the source and
/// target types of the conversion.
void ImplicitConversionSequence::DiagnoseAmbiguousConversion(
                                 Sema &S,
                                 SourceLocation CaretLoc,
                                 const PartialDiagnostic &PDiag) const {
  S.Diag(CaretLoc, PDiag)
    << Ambiguous.getFromType() << Ambiguous.getToType();
  for (AmbiguousConversionSequence::const_iterator
         I = Ambiguous.begin(), E = Ambiguous.end(); I != E; ++I) {
    S.NoteOverloadCandidate(*I);
  }
}

namespace {

void DiagnoseBadConversion(Sema &S, OverloadCandidate *Cand, unsigned I) {
  const ImplicitConversionSequence &Conv = Cand->Conversions[I];
  assert(Conv.isBad());
  assert(Cand->Function && "for now, candidate must be a function");
  FunctionDecl *Fn = Cand->Function;

  // There's a conversion slot for the object argument if this is a
  // non-constructor method.  Note that 'I' corresponds the
  // conversion-slot index.
  bool isObjectArgument = false;
  if (isa<CXXMethodDecl>(Fn) && !isa<CXXConstructorDecl>(Fn)) {
    if (I == 0)
      isObjectArgument = true;
    else
      I--;
  }

  std::string FnDesc;
  OverloadCandidateKind FnKind = ClassifyOverloadCandidate(S, Fn, FnDesc);

  Expr *FromExpr = Conv.Bad.FromExpr;
  QualType FromTy = Conv.Bad.getFromType();
  QualType ToTy = Conv.Bad.getToType();

  if (FromTy == S.Context.OverloadTy) {
    assert(FromExpr && "overload set argument came from implicit argument?");
    Expr *E = FromExpr->IgnoreParens();
    if (isa<UnaryOperator>(E))
      E = cast<UnaryOperator>(E)->getSubExpr()->IgnoreParens();
    DeclarationName Name = cast<OverloadExpr>(E)->getName();

    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_overload)
      << (unsigned) FnKind << FnDesc
      << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
      << ToTy << Name << I+1;
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // Do some hand-waving analysis to see if the non-viability is due
  // to a qualifier mismatch.
  CanQualType CFromTy = S.Context.getCanonicalType(FromTy);
  CanQualType CToTy = S.Context.getCanonicalType(ToTy);
  if (CanQual<ReferenceType> RT = CToTy->getAs<ReferenceType>())
    CToTy = RT->getPointeeType();
  else {
    // TODO: detect and diagnose the full richness of const mismatches.
    if (CanQual<PointerType> FromPT = CFromTy->getAs<PointerType>())
      if (CanQual<PointerType> ToPT = CToTy->getAs<PointerType>())
        CFromTy = FromPT->getPointeeType(), CToTy = ToPT->getPointeeType();
  }

  if (CToTy.getUnqualifiedType() == CFromTy.getUnqualifiedType() &&
      !CToTy.isAtLeastAsQualifiedAs(CFromTy)) {
    // It is dumb that we have to do this here.
    while (isa<ArrayType>(CFromTy))
      CFromTy = CFromTy->getAs<ArrayType>()->getElementType();
    while (isa<ArrayType>(CToTy))
      CToTy = CFromTy->getAs<ArrayType>()->getElementType();

    Qualifiers FromQs = CFromTy.getQualifiers();
    Qualifiers ToQs = CToTy.getQualifiers();

    if (FromQs.getAddressSpace() != ToQs.getAddressSpace()) {
      S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_addrspace)
        << (unsigned) FnKind << FnDesc
        << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
        << FromTy
        << FromQs.getAddressSpace() << ToQs.getAddressSpace()
        << (unsigned) isObjectArgument << I+1;
      MaybeEmitInheritedConstructorNote(S, Fn);
      return;
    }

    if (FromQs.getObjCLifetime() != ToQs.getObjCLifetime()) {
      S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_ownership)
        << (unsigned) FnKind << FnDesc
        << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
        << FromTy
        << FromQs.getObjCLifetime() << ToQs.getObjCLifetime()
        << (unsigned) isObjectArgument << I+1;
      MaybeEmitInheritedConstructorNote(S, Fn);
      return;
    }

    if (FromQs.getObjCGCAttr() != ToQs.getObjCGCAttr()) {
      S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_gc)
      << (unsigned) FnKind << FnDesc
      << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
      << FromTy
      << FromQs.getObjCGCAttr() << ToQs.getObjCGCAttr()
      << (unsigned) isObjectArgument << I+1;
      MaybeEmitInheritedConstructorNote(S, Fn);
      return;
    }

    unsigned CVR = FromQs.getCVRQualifiers() & ~ToQs.getCVRQualifiers();
    assert(CVR && "unexpected qualifiers mismatch");

    if (isObjectArgument) {
      S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_cvr_this)
        << (unsigned) FnKind << FnDesc
        << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
        << FromTy << (CVR - 1);
    } else {
      S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_cvr)
        << (unsigned) FnKind << FnDesc
        << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
        << FromTy << (CVR - 1) << I+1;
    }
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // Diagnose references or pointers to incomplete types differently,
  // since it's far from impossible that the incompleteness triggered
  // the failure.
  QualType TempFromTy = FromTy.getNonReferenceType();
  if (const PointerType *PTy = TempFromTy->getAs<PointerType>())
    TempFromTy = PTy->getPointeeType();
  if (TempFromTy->isIncompleteType()) {
    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_conv_incomplete)
      << (unsigned) FnKind << FnDesc
      << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
      << FromTy << ToTy << (unsigned) isObjectArgument << I+1;
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // Diagnose base -> derived pointer conversions.
  unsigned BaseToDerivedConversion = 0;
  if (const PointerType *FromPtrTy = FromTy->getAs<PointerType>()) {
    if (const PointerType *ToPtrTy = ToTy->getAs<PointerType>()) {
      if (ToPtrTy->getPointeeType().isAtLeastAsQualifiedAs(
                                               FromPtrTy->getPointeeType()) &&
          !FromPtrTy->getPointeeType()->isIncompleteType() &&
          !ToPtrTy->getPointeeType()->isIncompleteType() &&
          S.IsDerivedFrom(ToPtrTy->getPointeeType(),
                          FromPtrTy->getPointeeType()))
        BaseToDerivedConversion = 1;
    }
  } else if (const ObjCObjectPointerType *FromPtrTy
                                    = FromTy->getAs<ObjCObjectPointerType>()) {
    if (const ObjCObjectPointerType *ToPtrTy
                                        = ToTy->getAs<ObjCObjectPointerType>())
      if (const ObjCInterfaceDecl *FromIface = FromPtrTy->getInterfaceDecl())
        if (const ObjCInterfaceDecl *ToIface = ToPtrTy->getInterfaceDecl())
          if (ToPtrTy->getPointeeType().isAtLeastAsQualifiedAs(
                                                FromPtrTy->getPointeeType()) &&
              FromIface->isSuperClassOf(ToIface))
            BaseToDerivedConversion = 2;
  } else if (const ReferenceType *ToRefTy = ToTy->getAs<ReferenceType>()) {
      if (ToRefTy->getPointeeType().isAtLeastAsQualifiedAs(FromTy) &&
          !FromTy->isIncompleteType() &&
          !ToRefTy->getPointeeType()->isIncompleteType() &&
          S.IsDerivedFrom(ToRefTy->getPointeeType(), FromTy))
        BaseToDerivedConversion = 3;
    }

  if (BaseToDerivedConversion) {
    S.Diag(Fn->getLocation(),
           diag::note_ovl_candidate_bad_base_to_derived_conv)
      << (unsigned) FnKind << FnDesc
      << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
      << (BaseToDerivedConversion - 1)
      << FromTy << ToTy << I+1;
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // TODO: specialize more based on the kind of mismatch
  S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_conv)
    << (unsigned) FnKind << FnDesc
    << (FromExpr ? FromExpr->getSourceRange() : SourceRange())
    << FromTy << ToTy << (unsigned) isObjectArgument << I+1;
  MaybeEmitInheritedConstructorNote(S, Fn);
}

void DiagnoseArityMismatch(Sema &S, OverloadCandidate *Cand,
                           unsigned NumFormalArgs) {
  // TODO: treat calls to a missing default constructor as a special case

  FunctionDecl *Fn = Cand->Function;
  const FunctionProtoType *FnTy = Fn->getType()->getAs<FunctionProtoType>();

  unsigned MinParams = Fn->getMinRequiredArguments();

  // With invalid overloaded operators, it's possible that we think we
  // have an arity mismatch when it fact it looks like we have the
  // right number of arguments, because only overloaded operators have
  // the weird behavior of overloading member and non-member functions.
  // Just don't report anything.
  if (Fn->isInvalidDecl() && 
      Fn->getDeclName().getNameKind() == DeclarationName::CXXOperatorName)
    return;

  // at least / at most / exactly
  unsigned mode, modeCount;
  if (NumFormalArgs < MinParams) {
    assert((Cand->FailureKind == ovl_fail_too_few_arguments) ||
           (Cand->FailureKind == ovl_fail_bad_deduction &&
            Cand->DeductionFailure.Result == Sema::TDK_TooFewArguments));
    if (MinParams != FnTy->getNumArgs() ||
        FnTy->isVariadic() || FnTy->isTemplateVariadic())
      mode = 0; // "at least"
    else
      mode = 2; // "exactly"
    modeCount = MinParams;
  } else {
    assert((Cand->FailureKind == ovl_fail_too_many_arguments) ||
           (Cand->FailureKind == ovl_fail_bad_deduction &&
            Cand->DeductionFailure.Result == Sema::TDK_TooManyArguments));
    if (MinParams != FnTy->getNumArgs())
      mode = 1; // "at most"
    else
      mode = 2; // "exactly"
    modeCount = FnTy->getNumArgs();
  }

  std::string Description;
  OverloadCandidateKind FnKind = ClassifyOverloadCandidate(S, Fn, Description);

  S.Diag(Fn->getLocation(), diag::note_ovl_candidate_arity)
    << (unsigned) FnKind << (Fn->getDescribedFunctionTemplate() != 0) << mode
    << modeCount << NumFormalArgs;
  MaybeEmitInheritedConstructorNote(S, Fn);
}

/// Diagnose a failed template-argument deduction.
void DiagnoseBadDeduction(Sema &S, OverloadCandidate *Cand,
                          Expr **Args, unsigned NumArgs) {
  FunctionDecl *Fn = Cand->Function; // pattern

  TemplateParameter Param = Cand->DeductionFailure.getTemplateParameter();
  NamedDecl *ParamD;
  (ParamD = Param.dyn_cast<TemplateTypeParmDecl*>()) ||
  (ParamD = Param.dyn_cast<NonTypeTemplateParmDecl*>()) ||
  (ParamD = Param.dyn_cast<TemplateTemplateParmDecl*>());
  switch (Cand->DeductionFailure.Result) {
  case Sema::TDK_Success:
    llvm_unreachable("TDK_success while diagnosing bad deduction");

  case Sema::TDK_Incomplete: {
    assert(ParamD && "no parameter found for incomplete deduction result");
    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_incomplete_deduction)
      << ParamD->getDeclName();
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  case Sema::TDK_Underqualified: {
    assert(ParamD && "no parameter found for bad qualifiers deduction result");
    TemplateTypeParmDecl *TParam = cast<TemplateTypeParmDecl>(ParamD);

    QualType Param = Cand->DeductionFailure.getFirstArg()->getAsType();

    // Param will have been canonicalized, but it should just be a
    // qualified version of ParamD, so move the qualifiers to that.
    QualifierCollector Qs;
    Qs.strip(Param);
    QualType NonCanonParam = Qs.apply(S.Context, TParam->getTypeForDecl());
    assert(S.Context.hasSameType(Param, NonCanonParam));

    // Arg has also been canonicalized, but there's nothing we can do
    // about that.  It also doesn't matter as much, because it won't
    // have any template parameters in it (because deduction isn't
    // done on dependent types).
    QualType Arg = Cand->DeductionFailure.getSecondArg()->getAsType();

    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_underqualified)
      << ParamD->getDeclName() << Arg << NonCanonParam;
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  case Sema::TDK_Inconsistent: {
    assert(ParamD && "no parameter found for inconsistent deduction result");
    int which = 0;
    if (isa<TemplateTypeParmDecl>(ParamD))
      which = 0;
    else if (isa<NonTypeTemplateParmDecl>(ParamD))
      which = 1;
    else {
      which = 2;
    }

    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_inconsistent_deduction)
      << which << ParamD->getDeclName()
      << *Cand->DeductionFailure.getFirstArg()
      << *Cand->DeductionFailure.getSecondArg();
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  case Sema::TDK_InvalidExplicitArguments:
    assert(ParamD && "no parameter found for invalid explicit arguments");
    if (ParamD->getDeclName())
      S.Diag(Fn->getLocation(),
             diag::note_ovl_candidate_explicit_arg_mismatch_named)
        << ParamD->getDeclName();
    else {
      int index = 0;
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(ParamD))
        index = TTP->getIndex();
      else if (NonTypeTemplateParmDecl *NTTP
                                  = dyn_cast<NonTypeTemplateParmDecl>(ParamD))
        index = NTTP->getIndex();
      else
        index = cast<TemplateTemplateParmDecl>(ParamD)->getIndex();
      S.Diag(Fn->getLocation(),
             diag::note_ovl_candidate_explicit_arg_mismatch_unnamed)
        << (index + 1);
    }
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;

  case Sema::TDK_TooManyArguments:
  case Sema::TDK_TooFewArguments:
    DiagnoseArityMismatch(S, Cand, NumArgs);
    return;

  case Sema::TDK_InstantiationDepth:
    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_instantiation_depth);
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;

  case Sema::TDK_SubstitutionFailure: {
    std::string ArgString;
    if (TemplateArgumentList *Args
                            = Cand->DeductionFailure.getTemplateArgumentList())
      ArgString = S.getTemplateArgumentBindingsText(
                    Fn->getDescribedFunctionTemplate()->getTemplateParameters(),
                                                    *Args);
    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_substitution_failure)
      << ArgString;
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // TODO: diagnose these individually, then kill off
  // note_ovl_candidate_bad_deduction, which is uselessly vague.
  case Sema::TDK_NonDeducedMismatch:
  case Sema::TDK_FailedOverloadResolution:
    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_bad_deduction);
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }
}

/// Generates a 'note' diagnostic for an overload candidate.  We've
/// already generated a primary error at the call site.
///
/// It really does need to be a single diagnostic with its caret
/// pointed at the candidate declaration.  Yes, this creates some
/// major challenges of technical writing.  Yes, this makes pointing
/// out problems with specific arguments quite awkward.  It's still
/// better than generating twenty screens of text for every failed
/// overload.
///
/// It would be great to be able to express per-candidate problems
/// more richly for those diagnostic clients that cared, but we'd
/// still have to be just as careful with the default diagnostics.
void NoteFunctionCandidate(Sema &S, OverloadCandidate *Cand,
                           Expr **Args, unsigned NumArgs) {
  FunctionDecl *Fn = Cand->Function;

  // Note deleted candidates, but only if they're viable.
  if (Cand->Viable && (Fn->isDeleted() ||
      S.isFunctionConsideredUnavailable(Fn))) {
    std::string FnDesc;
    OverloadCandidateKind FnKind = ClassifyOverloadCandidate(S, Fn, FnDesc);

    S.Diag(Fn->getLocation(), diag::note_ovl_candidate_deleted)
      << FnKind << FnDesc << Fn->isDeleted();
    MaybeEmitInheritedConstructorNote(S, Fn);
    return;
  }

  // We don't really have anything else to say about viable candidates.
  if (Cand->Viable) {
    S.NoteOverloadCandidate(Fn);
    return;
  }

  switch (Cand->FailureKind) {
  case ovl_fail_too_many_arguments:
  case ovl_fail_too_few_arguments:
    return DiagnoseArityMismatch(S, Cand, NumArgs);

  case ovl_fail_bad_deduction:
    return DiagnoseBadDeduction(S, Cand, Args, NumArgs);

  case ovl_fail_trivial_conversion:
  case ovl_fail_bad_final_conversion:
  case ovl_fail_final_conversion_not_exact:
    return S.NoteOverloadCandidate(Fn);

  case ovl_fail_bad_conversion: {
    unsigned I = (Cand->IgnoreObjectArgument ? 1 : 0);
    for (unsigned N = Cand->Conversions.size(); I != N; ++I)
      if (Cand->Conversions[I].isBad())
        return DiagnoseBadConversion(S, Cand, I);

    // FIXME: this currently happens when we're called from SemaInit
    // when user-conversion overload fails.  Figure out how to handle
    // those conditions and diagnose them well.
    return S.NoteOverloadCandidate(Fn);
  }
  }
}

void NoteSurrogateCandidate(Sema &S, OverloadCandidate *Cand) {
  // Desugar the type of the surrogate down to a function type,
  // retaining as many typedefs as possible while still showing
  // the function type (and, therefore, its parameter types).
  QualType FnType = Cand->Surrogate->getConversionType();
  bool isLValueReference = false;
  bool isRValueReference = false;
  bool isPointer = false;
  if (const LValueReferenceType *FnTypeRef =
        FnType->getAs<LValueReferenceType>()) {
    FnType = FnTypeRef->getPointeeType();
    isLValueReference = true;
  } else if (const RValueReferenceType *FnTypeRef =
               FnType->getAs<RValueReferenceType>()) {
    FnType = FnTypeRef->getPointeeType();
    isRValueReference = true;
  }
  if (const PointerType *FnTypePtr = FnType->getAs<PointerType>()) {
    FnType = FnTypePtr->getPointeeType();
    isPointer = true;
  }
  // Desugar down to a function type.
  FnType = QualType(FnType->getAs<FunctionType>(), 0);
  // Reconstruct the pointer/reference as appropriate.
  if (isPointer) FnType = S.Context.getPointerType(FnType);
  if (isRValueReference) FnType = S.Context.getRValueReferenceType(FnType);
  if (isLValueReference) FnType = S.Context.getLValueReferenceType(FnType);

  S.Diag(Cand->Surrogate->getLocation(), diag::note_ovl_surrogate_cand)
    << FnType;
  MaybeEmitInheritedConstructorNote(S, Cand->Surrogate);
}

void NoteBuiltinOperatorCandidate(Sema &S,
                                  const char *Opc,
                                  SourceLocation OpLoc,
                                  OverloadCandidate *Cand) {
  assert(Cand->Conversions.size() <= 2 && "builtin operator is not binary");
  std::string TypeStr("operator");
  TypeStr += Opc;
  TypeStr += "(";
  TypeStr += Cand->BuiltinTypes.ParamTypes[0].getAsString();
  if (Cand->Conversions.size() == 1) {
    TypeStr += ")";
    S.Diag(OpLoc, diag::note_ovl_builtin_unary_candidate) << TypeStr;
  } else {
    TypeStr += ", ";
    TypeStr += Cand->BuiltinTypes.ParamTypes[1].getAsString();
    TypeStr += ")";
    S.Diag(OpLoc, diag::note_ovl_builtin_binary_candidate) << TypeStr;
  }
}

void NoteAmbiguousUserConversions(Sema &S, SourceLocation OpLoc,
                                  OverloadCandidate *Cand) {
  unsigned NoOperands = Cand->Conversions.size();
  for (unsigned ArgIdx = 0; ArgIdx < NoOperands; ++ArgIdx) {
    const ImplicitConversionSequence &ICS = Cand->Conversions[ArgIdx];
    if (ICS.isBad()) break; // all meaningless after first invalid
    if (!ICS.isAmbiguous()) continue;

    ICS.DiagnoseAmbiguousConversion(S, OpLoc,
                              S.PDiag(diag::note_ambiguous_type_conversion));
  }
}

SourceLocation GetLocationForCandidate(const OverloadCandidate *Cand) {
  if (Cand->Function)
    return Cand->Function->getLocation();
  if (Cand->IsSurrogate)
    return Cand->Surrogate->getLocation();
  return SourceLocation();
}

struct CompareOverloadCandidatesForDisplay {
  Sema &S;
  CompareOverloadCandidatesForDisplay(Sema &S) : S(S) {}

  bool operator()(const OverloadCandidate *L,
                  const OverloadCandidate *R) {
    // Fast-path this check.
    if (L == R) return false;

    // Order first by viability.
    if (L->Viable) {
      if (!R->Viable) return true;

      // TODO: introduce a tri-valued comparison for overload
      // candidates.  Would be more worthwhile if we had a sort
      // that could exploit it.
      if (isBetterOverloadCandidate(S, *L, *R, SourceLocation())) return true;
      if (isBetterOverloadCandidate(S, *R, *L, SourceLocation())) return false;
    } else if (R->Viable)
      return false;

    assert(L->Viable == R->Viable);

    // Criteria by which we can sort non-viable candidates:
    if (!L->Viable) {
      // 1. Arity mismatches come after other candidates.
      if (L->FailureKind == ovl_fail_too_many_arguments ||
          L->FailureKind == ovl_fail_too_few_arguments)
        return false;
      if (R->FailureKind == ovl_fail_too_many_arguments ||
          R->FailureKind == ovl_fail_too_few_arguments)
        return true;

      // 2. Bad conversions come first and are ordered by the number
      // of bad conversions and quality of good conversions.
      if (L->FailureKind == ovl_fail_bad_conversion) {
        if (R->FailureKind != ovl_fail_bad_conversion)
          return true;

        // If there's any ordering between the defined conversions...
        // FIXME: this might not be transitive.
        assert(L->Conversions.size() == R->Conversions.size());

        int leftBetter = 0;
        unsigned I = (L->IgnoreObjectArgument || R->IgnoreObjectArgument);
        for (unsigned E = L->Conversions.size(); I != E; ++I) {
          switch (CompareImplicitConversionSequences(S,
                                                     L->Conversions[I],
                                                     R->Conversions[I])) {
          case ImplicitConversionSequence::Better:
            leftBetter++;
            break;

          case ImplicitConversionSequence::Worse:
            leftBetter--;
            break;

          case ImplicitConversionSequence::Indistinguishable:
            break;
          }
        }
        if (leftBetter > 0) return true;
        if (leftBetter < 0) return false;

      } else if (R->FailureKind == ovl_fail_bad_conversion)
        return false;

      // TODO: others?
    }

    // Sort everything else by location.
    SourceLocation LLoc = GetLocationForCandidate(L);
    SourceLocation RLoc = GetLocationForCandidate(R);

    // Put candidates without locations (e.g. builtins) at the end.
    if (LLoc.isInvalid()) return false;
    if (RLoc.isInvalid()) return true;

    return S.SourceMgr.isBeforeInTranslationUnit(LLoc, RLoc);
  }
};

/// CompleteNonViableCandidate - Normally, overload resolution only
/// computes up to the first
void CompleteNonViableCandidate(Sema &S, OverloadCandidate *Cand,
                                Expr **Args, unsigned NumArgs) {
  assert(!Cand->Viable);

  // Don't do anything on failures other than bad conversion.
  if (Cand->FailureKind != ovl_fail_bad_conversion) return;

  // Skip forward to the first bad conversion.
  unsigned ConvIdx = (Cand->IgnoreObjectArgument ? 1 : 0);
  unsigned ConvCount = Cand->Conversions.size();
  while (true) {
    assert(ConvIdx != ConvCount && "no bad conversion in candidate");
    ConvIdx++;
    if (Cand->Conversions[ConvIdx - 1].isBad())
      break;
  }

  if (ConvIdx == ConvCount)
    return;

  assert(!Cand->Conversions[ConvIdx].isInitialized() &&
         "remaining conversion is initialized?");

  // FIXME: this should probably be preserved from the overload
  // operation somehow.
  bool SuppressUserConversions = false;

  const FunctionProtoType* Proto;
  unsigned ArgIdx = ConvIdx;

  if (Cand->IsSurrogate) {
    QualType ConvType
      = Cand->Surrogate->getConversionType().getNonReferenceType();
    if (const PointerType *ConvPtrType = ConvType->getAs<PointerType>())
      ConvType = ConvPtrType->getPointeeType();
    Proto = ConvType->getAs<FunctionProtoType>();
    ArgIdx--;
  } else if (Cand->Function) {
    Proto = Cand->Function->getType()->getAs<FunctionProtoType>();
    if (isa<CXXMethodDecl>(Cand->Function) &&
        !isa<CXXConstructorDecl>(Cand->Function))
      ArgIdx--;
  } else {
    // Builtin binary operator with a bad first conversion.
    assert(ConvCount <= 3);
    for (; ConvIdx != ConvCount; ++ConvIdx)
      Cand->Conversions[ConvIdx]
        = TryCopyInitialization(S, Args[ConvIdx],
                                Cand->BuiltinTypes.ParamTypes[ConvIdx],
                                SuppressUserConversions,
                                /*InOverloadResolution*/ true,
                                /*AllowObjCWritebackConversion=*/
                                  S.getLangOptions().ObjCAutoRefCount);
    return;
  }

  // Fill in the rest of the conversions.
  unsigned NumArgsInProto = Proto->getNumArgs();
  for (; ConvIdx != ConvCount; ++ConvIdx, ++ArgIdx) {
    if (ArgIdx < NumArgsInProto)
      Cand->Conversions[ConvIdx]
        = TryCopyInitialization(S, Args[ArgIdx], Proto->getArgType(ArgIdx),
                                SuppressUserConversions,
                                /*InOverloadResolution=*/true,
                                /*AllowObjCWritebackConversion=*/
                                  S.getLangOptions().ObjCAutoRefCount);
    else
      Cand->Conversions[ConvIdx].setEllipsis();
  }
}

} // end anonymous namespace

/// PrintOverloadCandidates - When overload resolution fails, prints
/// diagnostic messages containing the candidates in the candidate
/// set.
void OverloadCandidateSet::NoteCandidates(Sema &S,
                                          OverloadCandidateDisplayKind OCD,
                                          Expr **Args, unsigned NumArgs,
                                          const char *Opc,
                                          SourceLocation OpLoc) {
  // Sort the candidates by viability and position.  Sorting directly would
  // be prohibitive, so we make a set of pointers and sort those.
  llvm::SmallVector<OverloadCandidate*, 32> Cands;
  if (OCD == OCD_AllCandidates) Cands.reserve(size());
  for (iterator Cand = begin(), LastCand = end(); Cand != LastCand; ++Cand) {
    if (Cand->Viable)
      Cands.push_back(Cand);
    else if (OCD == OCD_AllCandidates) {
      CompleteNonViableCandidate(S, Cand, Args, NumArgs);
      if (Cand->Function || Cand->IsSurrogate)
        Cands.push_back(Cand);
      // Otherwise, this a non-viable builtin candidate.  We do not, in general,
      // want to list every possible builtin candidate.
    }
  }

  std::sort(Cands.begin(), Cands.end(),
            CompareOverloadCandidatesForDisplay(S));

  bool ReportedAmbiguousConversions = false;

  llvm::SmallVectorImpl<OverloadCandidate*>::iterator I, E;
  const Diagnostic::OverloadsShown ShowOverloads = S.Diags.getShowOverloads();
  unsigned CandsShown = 0;
  for (I = Cands.begin(), E = Cands.end(); I != E; ++I) {
    OverloadCandidate *Cand = *I;

    // Set an arbitrary limit on the number of candidate functions we'll spam
    // the user with.  FIXME: This limit should depend on details of the
    // candidate list.
    if (CandsShown >= 4 && ShowOverloads == Diagnostic::Ovl_Best) {
      break;
    }
    ++CandsShown;

    if (Cand->Function)
      NoteFunctionCandidate(S, Cand, Args, NumArgs);
    else if (Cand->IsSurrogate)
      NoteSurrogateCandidate(S, Cand);
    else {
      assert(Cand->Viable &&
             "Non-viable built-in candidates are not added to Cands.");
      // Generally we only see ambiguities including viable builtin
      // operators if overload resolution got screwed up by an
      // ambiguous user-defined conversion.
      //
      // FIXME: It's quite possible for different conversions to see
      // different ambiguities, though.
      if (!ReportedAmbiguousConversions) {
        NoteAmbiguousUserConversions(S, OpLoc, Cand);
        ReportedAmbiguousConversions = true;
      }

      // If this is a viable builtin, print it.
      NoteBuiltinOperatorCandidate(S, Opc, OpLoc, Cand);
    }
  }

  if (I != E)
    S.Diag(OpLoc, diag::note_ovl_too_many_candidates) << int(E - I);
}

// [PossiblyAFunctionType]  -->   [Return]
// NonFunctionType --> NonFunctionType
// R (A) --> R(A)
// R (*)(A) --> R (A)
// R (&)(A) --> R (A)
// R (S::*)(A) --> R (A)
QualType Sema::ExtractUnqualifiedFunctionType(QualType PossiblyAFunctionType) {
  QualType Ret = PossiblyAFunctionType;
  if (const PointerType *ToTypePtr = 
    PossiblyAFunctionType->getAs<PointerType>())
    Ret = ToTypePtr->getPointeeType();
  else if (const ReferenceType *ToTypeRef = 
    PossiblyAFunctionType->getAs<ReferenceType>())
    Ret = ToTypeRef->getPointeeType();
  else if (const MemberPointerType *MemTypePtr =
    PossiblyAFunctionType->getAs<MemberPointerType>()) 
    Ret = MemTypePtr->getPointeeType();   
  Ret = 
    Context.getCanonicalType(Ret).getUnqualifiedType();
  return Ret;
}

// A helper class to help with address of function resolution
// - allows us to avoid passing around all those ugly parameters
class AddressOfFunctionResolver 
{
  Sema& S;
  Expr* SourceExpr;
  const QualType& TargetType; 
  QualType TargetFunctionType; // Extracted function type from target type 
   
  bool Complain;
  //DeclAccessPair& ResultFunctionAccessPair;
  ASTContext& Context;

  bool TargetTypeIsNonStaticMemberFunction;
  bool FoundNonTemplateFunction;

  OverloadExpr::FindResult OvlExprInfo; 
  OverloadExpr *OvlExpr;
  TemplateArgumentListInfo OvlExplicitTemplateArgs;
  llvm::SmallVector<std::pair<DeclAccessPair, FunctionDecl*>, 4> Matches;

public:
  AddressOfFunctionResolver(Sema &S, Expr* SourceExpr, 
                            const QualType& TargetType, bool Complain)
    : S(S), SourceExpr(SourceExpr), TargetType(TargetType), 
      Complain(Complain), Context(S.getASTContext()), 
      TargetTypeIsNonStaticMemberFunction(
                                    !!TargetType->getAs<MemberPointerType>()),
      FoundNonTemplateFunction(false),
      OvlExprInfo(OverloadExpr::find(SourceExpr)),
      OvlExpr(OvlExprInfo.Expression)
  {
    ExtractUnqualifiedFunctionTypeFromTargetType();
    
    if (!TargetFunctionType->isFunctionType()) {        
      if (OvlExpr->hasExplicitTemplateArgs()) {
        DeclAccessPair dap;
        if (FunctionDecl* Fn = S.ResolveSingleFunctionTemplateSpecialization(
                                            OvlExpr, false, &dap) ) {

          if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
            if (!Method->isStatic()) {
              // If the target type is a non-function type and the function
              // found is a non-static member function, pretend as if that was
              // the target, it's the only possible type to end up with.
              TargetTypeIsNonStaticMemberFunction = true;

              // And skip adding the function if its not in the proper form.
              // We'll diagnose this due to an empty set of functions.
              if (!OvlExprInfo.HasFormOfMemberPointer)
                return;
            }
          }

          Matches.push_back(std::make_pair(dap,Fn));
        }
      }
      return;
    }
    
    if (OvlExpr->hasExplicitTemplateArgs())
      OvlExpr->getExplicitTemplateArgs().copyInto(OvlExplicitTemplateArgs);

    if (FindAllFunctionsThatMatchTargetTypeExactly()) {
      // C++ [over.over]p4:
      //   If more than one function is selected, [...]
      if (Matches.size() > 1) {
        if (FoundNonTemplateFunction)
          EliminateAllTemplateMatches();
        else
          EliminateAllExceptMostSpecializedTemplate();
      }
    }
  }
  
private:
  bool isTargetTypeAFunction() const {
    return TargetFunctionType->isFunctionType();
  }

  // [ToType]     [Return]

  // R (*)(A) --> R (A), IsNonStaticMemberFunction = false
  // R (&)(A) --> R (A), IsNonStaticMemberFunction = false
  // R (S::*)(A) --> R (A), IsNonStaticMemberFunction = true
  void inline ExtractUnqualifiedFunctionTypeFromTargetType() {
    TargetFunctionType = S.ExtractUnqualifiedFunctionType(TargetType);
  }

  // return true if any matching specializations were found
  bool AddMatchingTemplateFunction(FunctionTemplateDecl* FunctionTemplate, 
                                   const DeclAccessPair& CurAccessFunPair) {
    if (CXXMethodDecl *Method
              = dyn_cast<CXXMethodDecl>(FunctionTemplate->getTemplatedDecl())) {
      // Skip non-static function templates when converting to pointer, and
      // static when converting to member pointer.
      if (Method->isStatic() == TargetTypeIsNonStaticMemberFunction)
        return false;
    } 
    else if (TargetTypeIsNonStaticMemberFunction)
      return false;

    // C++ [over.over]p2:
    //   If the name is a function template, template argument deduction is
    //   done (14.8.2.2), and if the argument deduction succeeds, the
    //   resulting template argument list is used to generate a single
    //   function template specialization, which is added to the set of
    //   overloaded functions considered.
    FunctionDecl *Specialization = 0;
    TemplateDeductionInfo Info(Context, OvlExpr->getNameLoc());
    if (Sema::TemplateDeductionResult Result
          = S.DeduceTemplateArguments(FunctionTemplate, 
                                      &OvlExplicitTemplateArgs,
                                      TargetFunctionType, Specialization, 
                                      Info)) {
      // FIXME: make a note of the failed deduction for diagnostics.
      (void)Result;
      return false;
    } 
    
    // Template argument deduction ensures that we have an exact match.
    // This function template specicalization works.
    Specialization = cast<FunctionDecl>(Specialization->getCanonicalDecl());
    assert(TargetFunctionType
                      == Context.getCanonicalType(Specialization->getType()));
    Matches.push_back(std::make_pair(CurAccessFunPair, Specialization));
    return true;
  }
  
  bool AddMatchingNonTemplateFunction(NamedDecl* Fn, 
                                      const DeclAccessPair& CurAccessFunPair) {
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
      // Skip non-static functions when converting to pointer, and static
      // when converting to member pointer.
      if (Method->isStatic() == TargetTypeIsNonStaticMemberFunction)
        return false;
    } 
    else if (TargetTypeIsNonStaticMemberFunction)
      return false;

    if (FunctionDecl *FunDecl = dyn_cast<FunctionDecl>(Fn)) {
      QualType ResultTy;
      if (Context.hasSameUnqualifiedType(TargetFunctionType, 
                                         FunDecl->getType()) ||
          S.IsNoReturnConversion(FunDecl->getType(), TargetFunctionType,
                                 ResultTy)) {
        Matches.push_back(std::make_pair(CurAccessFunPair,
          cast<FunctionDecl>(FunDecl->getCanonicalDecl())));
        FoundNonTemplateFunction = true;
        return true;
      }
    }
    
    return false;
  }
  
  bool FindAllFunctionsThatMatchTargetTypeExactly() {
    bool Ret = false;
    
    // If the overload expression doesn't have the form of a pointer to
    // member, don't try to convert it to a pointer-to-member type.
    if (IsInvalidFormOfPointerToMemberFunction())
      return false;

    for (UnresolvedSetIterator I = OvlExpr->decls_begin(),
                               E = OvlExpr->decls_end(); 
         I != E; ++I) {
      // Look through any using declarations to find the underlying function.
      NamedDecl *Fn = (*I)->getUnderlyingDecl();

      // C++ [over.over]p3:
      //   Non-member functions and static member functions match
      //   targets of type "pointer-to-function" or "reference-to-function."
      //   Nonstatic member functions match targets of
      //   type "pointer-to-member-function."
      // Note that according to DR 247, the containing class does not matter.
      if (FunctionTemplateDecl *FunctionTemplate
                                        = dyn_cast<FunctionTemplateDecl>(Fn)) {
        if (AddMatchingTemplateFunction(FunctionTemplate, I.getPair()))
          Ret = true;
      }
      // If we have explicit template arguments supplied, skip non-templates.
      else if (!OvlExpr->hasExplicitTemplateArgs() &&
               AddMatchingNonTemplateFunction(Fn, I.getPair()))
        Ret = true;
    }
    assert(Ret || Matches.empty());
    return Ret;
  }

  void EliminateAllExceptMostSpecializedTemplate() {
    //   [...] and any given function template specialization F1 is
    //   eliminated if the set contains a second function template
    //   specialization whose function template is more specialized
    //   than the function template of F1 according to the partial
    //   ordering rules of 14.5.5.2.

    // The algorithm specified above is quadratic. We instead use a
    // two-pass algorithm (similar to the one used to identify the
    // best viable function in an overload set) that identifies the
    // best function template (if it exists).

    UnresolvedSet<4> MatchesCopy; // TODO: avoid!
    for (unsigned I = 0, E = Matches.size(); I != E; ++I)
      MatchesCopy.addDecl(Matches[I].second, Matches[I].first.getAccess());

    UnresolvedSetIterator Result =
      S.getMostSpecialized(MatchesCopy.begin(), MatchesCopy.end(),
                           TPOC_Other, 0, SourceExpr->getLocStart(),
                           S.PDiag(),
                           S.PDiag(diag::err_addr_ovl_ambiguous)
                             << Matches[0].second->getDeclName(),
                           S.PDiag(diag::note_ovl_candidate)
                             << (unsigned) oc_function_template,
                           Complain);

    if (Result != MatchesCopy.end()) {
      // Make it the first and only element
      Matches[0].first = Matches[Result - MatchesCopy.begin()].first;
      Matches[0].second = cast<FunctionDecl>(*Result);
      Matches.resize(1);
    }
  }

  void EliminateAllTemplateMatches() {
    //   [...] any function template specializations in the set are
    //   eliminated if the set also contains a non-template function, [...]
    for (unsigned I = 0, N = Matches.size(); I != N; ) {
      if (Matches[I].second->getPrimaryTemplate() == 0)
        ++I;
      else {
        Matches[I] = Matches[--N];
        Matches.set_size(N);
      }
    }
  }

public:
  void ComplainNoMatchesFound() const {
    assert(Matches.empty());
    S.Diag(OvlExpr->getLocStart(), diag::err_addr_ovl_no_viable)
        << OvlExpr->getName() << TargetFunctionType
        << OvlExpr->getSourceRange();
    S.NoteAllOverloadCandidates(OvlExpr);
  } 
  
  bool IsInvalidFormOfPointerToMemberFunction() const {
    return TargetTypeIsNonStaticMemberFunction &&
      !OvlExprInfo.HasFormOfMemberPointer;
  }
  
  void ComplainIsInvalidFormOfPointerToMemberFunction() const {
      // TODO: Should we condition this on whether any functions might
      // have matched, or is it more appropriate to do that in callers?
      // TODO: a fixit wouldn't hurt.
      S.Diag(OvlExpr->getNameLoc(), diag::err_addr_ovl_no_qualifier)
        << TargetType << OvlExpr->getSourceRange();
  }
  
  void ComplainOfInvalidConversion() const {
    S.Diag(OvlExpr->getLocStart(), diag::err_addr_ovl_not_func_ptrref)
      << OvlExpr->getName() << TargetType;
  }

  void ComplainMultipleMatchesFound() const {
    assert(Matches.size() > 1);
    S.Diag(OvlExpr->getLocStart(), diag::err_addr_ovl_ambiguous)
      << OvlExpr->getName()
      << OvlExpr->getSourceRange();
    S.NoteAllOverloadCandidates(OvlExpr);
  }
  
  int getNumMatches() const { return Matches.size(); }
  
  FunctionDecl* getMatchingFunctionDecl() const {
    if (Matches.size() != 1) return 0;
    return Matches[0].second;
  }
  
  const DeclAccessPair* getMatchingFunctionAccessPair() const {
    if (Matches.size() != 1) return 0;
    return &Matches[0].first;
  }
};
  
/// ResolveAddressOfOverloadedFunction - Try to resolve the address of
/// an overloaded function (C++ [over.over]), where @p From is an
/// expression with overloaded function type and @p ToType is the type
/// we're trying to resolve to. For example:
///
/// @code
/// int f(double);
/// int f(int);
///
/// int (*pfd)(double) = f; // selects f(double)
/// @endcode
///
/// This routine returns the resulting FunctionDecl if it could be
/// resolved, and NULL otherwise. When @p Complain is true, this
/// routine will emit diagnostics if there is an error.
FunctionDecl *
Sema::ResolveAddressOfOverloadedFunction(Expr *AddressOfExpr, QualType TargetType,
                                    bool Complain,
                                    DeclAccessPair &FoundResult) {

  assert(AddressOfExpr->getType() == Context.OverloadTy);
  
  AddressOfFunctionResolver Resolver(*this, AddressOfExpr, TargetType, Complain);
  int NumMatches = Resolver.getNumMatches();
  FunctionDecl* Fn = 0;
  if ( NumMatches == 0 && Complain) {
    if (Resolver.IsInvalidFormOfPointerToMemberFunction())
      Resolver.ComplainIsInvalidFormOfPointerToMemberFunction();
    else
      Resolver.ComplainNoMatchesFound();
  }
  else if (NumMatches > 1 && Complain)
    Resolver.ComplainMultipleMatchesFound();
  else if (NumMatches == 1) {
    Fn = Resolver.getMatchingFunctionDecl();
    assert(Fn);
    FoundResult = *Resolver.getMatchingFunctionAccessPair();
    MarkDeclarationReferenced(AddressOfExpr->getLocStart(), Fn);
    if (Complain)
      CheckAddressOfMemberAccess(AddressOfExpr, FoundResult);
  }
  
  return Fn;
}

/// \brief Given an expression that refers to an overloaded function, try to
/// resolve that overloaded function expression down to a single function.
///
/// This routine can only resolve template-ids that refer to a single function
/// template, where that template-id refers to a single template whose template
/// arguments are either provided by the template-id or have defaults,
/// as described in C++0x [temp.arg.explicit]p3.
FunctionDecl *
Sema::ResolveSingleFunctionTemplateSpecialization(OverloadExpr *ovl, 
                                                  bool Complain,
                                                  DeclAccessPair *FoundResult) {
  // C++ [over.over]p1:
  //   [...] [Note: any redundant set of parentheses surrounding the
  //   overloaded function name is ignored (5.1). ]
  // C++ [over.over]p1:
  //   [...] The overloaded function name can be preceded by the &
  //   operator.

  // If we didn't actually find any template-ids, we're done.
  if (!ovl->hasExplicitTemplateArgs())
    return 0;

  TemplateArgumentListInfo ExplicitTemplateArgs;
  ovl->getExplicitTemplateArgs().copyInto(ExplicitTemplateArgs);

  // Look through all of the overloaded functions, searching for one
  // whose type matches exactly.
  FunctionDecl *Matched = 0;
  for (UnresolvedSetIterator I = ovl->decls_begin(),
         E = ovl->decls_end(); I != E; ++I) {
    // C++0x [temp.arg.explicit]p3:
    //   [...] In contexts where deduction is done and fails, or in contexts
    //   where deduction is not done, if a template argument list is
    //   specified and it, along with any default template arguments,
    //   identifies a single function template specialization, then the
    //   template-id is an lvalue for the function template specialization.
    FunctionTemplateDecl *FunctionTemplate
      = cast<FunctionTemplateDecl>((*I)->getUnderlyingDecl());

    // C++ [over.over]p2:
    //   If the name is a function template, template argument deduction is
    //   done (14.8.2.2), and if the argument deduction succeeds, the
    //   resulting template argument list is used to generate a single
    //   function template specialization, which is added to the set of
    //   overloaded functions considered.
    FunctionDecl *Specialization = 0;
    TemplateDeductionInfo Info(Context, ovl->getNameLoc());
    if (TemplateDeductionResult Result
          = DeduceTemplateArguments(FunctionTemplate, &ExplicitTemplateArgs,
                                    Specialization, Info)) {
      // FIXME: make a note of the failed deduction for diagnostics.
      (void)Result;
      continue;
    }

    assert(Specialization && "no specialization and no error?");

    // Multiple matches; we can't resolve to a single declaration.
    if (Matched) {
      if (Complain) {
        Diag(ovl->getExprLoc(), diag::err_addr_ovl_ambiguous)
          << ovl->getName();
        NoteAllOverloadCandidates(ovl);
      }
      return 0;
    }
    
    Matched = Specialization;
    if (FoundResult) *FoundResult = I.getPair();    
  }

  return Matched;
}




// Resolve and fix an overloaded expression that
// can be resolved because it identifies a single function
// template specialization
// Last three arguments should only be supplied if Complain = true
ExprResult Sema::ResolveAndFixSingleFunctionTemplateSpecialization(
             Expr *SrcExpr, bool doFunctionPointerConverion, bool complain,
                                  const SourceRange& OpRangeForComplaining, 
                                           QualType DestTypeForComplaining, 
                                            unsigned DiagIDForComplaining) {
  assert(SrcExpr->getType() == Context.OverloadTy);

  OverloadExpr::FindResult ovl = OverloadExpr::find(SrcExpr);

  DeclAccessPair found;
  ExprResult SingleFunctionExpression;
  if (FunctionDecl *fn = ResolveSingleFunctionTemplateSpecialization(
                           ovl.Expression, /*complain*/ false, &found)) {
    if (DiagnoseUseOfDecl(fn, SrcExpr->getSourceRange().getBegin()))
      return ExprError();

    // It is only correct to resolve to an instance method if we're
    // resolving a form that's permitted to be a pointer to member.
    // Otherwise we'll end up making a bound member expression, which
    // is illegal in all the contexts we resolve like this.
    if (!ovl.HasFormOfMemberPointer &&
        isa<CXXMethodDecl>(fn) &&
        cast<CXXMethodDecl>(fn)->isInstance()) {
      if (complain) {
        Diag(ovl.Expression->getExprLoc(),
             diag::err_invalid_use_of_bound_member_func)
          << ovl.Expression->getSourceRange();
        // TODO: I believe we only end up here if there's a mix of
        // static and non-static candidates (otherwise the expression
        // would have 'bound member' type, not 'overload' type).
        // Ideally we would note which candidate was chosen and why
        // the static candidates were rejected.
      }
      
      return ExprError();
    }

    // Fix the expresion to refer to 'fn'.
    SingleFunctionExpression =
      Owned(FixOverloadedFunctionReference(SrcExpr, found, fn));

    // If desired, do function-to-pointer decay.
    if (doFunctionPointerConverion)
      SingleFunctionExpression =
        DefaultFunctionArrayLvalueConversion(SingleFunctionExpression.take());
  }

  if (!SingleFunctionExpression.isUsable()) {
    if (complain) {
      Diag(OpRangeForComplaining.getBegin(), DiagIDForComplaining)
        << ovl.Expression->getName()
        << DestTypeForComplaining
        << OpRangeForComplaining 
        << ovl.Expression->getQualifierLoc().getSourceRange();
      NoteAllOverloadCandidates(SrcExpr);
    }      
    return ExprError();
  }

  return SingleFunctionExpression;
}

/// \brief Add a single candidate to the overload set.
static void AddOverloadedCallCandidate(Sema &S,
                                       DeclAccessPair FoundDecl,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet &CandidateSet,
                                       bool PartialOverloading,
                                       bool KnownValid) {
  NamedDecl *Callee = FoundDecl.getDecl();
  if (isa<UsingShadowDecl>(Callee))
    Callee = cast<UsingShadowDecl>(Callee)->getTargetDecl();

  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(Callee)) {
    if (ExplicitTemplateArgs) {
      assert(!KnownValid && "Explicit template arguments?");
      return;
    }
    S.AddOverloadCandidate(Func, FoundDecl, Args, NumArgs, CandidateSet,
                           false, PartialOverloading);
    return;
  }

  if (FunctionTemplateDecl *FuncTemplate
      = dyn_cast<FunctionTemplateDecl>(Callee)) {
    S.AddTemplateOverloadCandidate(FuncTemplate, FoundDecl,
                                   ExplicitTemplateArgs,
                                   Args, NumArgs, CandidateSet);
    return;
  }

  assert(!KnownValid && "unhandled case in overloaded call candidate");
}

/// \brief Add the overload candidates named by callee and/or found by argument
/// dependent lookup to the given overload set.
void Sema::AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet &CandidateSet,
                                       bool PartialOverloading) {

#ifndef NDEBUG
  // Verify that ArgumentDependentLookup is consistent with the rules
  // in C++0x [basic.lookup.argdep]p3:
  //
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains
  //
  //     -- a declaration of a class member, or
  //
  //     -- a block-scope function declaration that is not a
  //        using-declaration, or
  //
  //     -- a declaration that is neither a function or a function
  //        template
  //
  //   then Y is empty.

  if (ULE->requiresADL()) {
    for (UnresolvedLookupExpr::decls_iterator I = ULE->decls_begin(),
           E = ULE->decls_end(); I != E; ++I) {
      assert(!(*I)->getDeclContext()->isRecord());
      assert(isa<UsingShadowDecl>(*I) ||
             !(*I)->getDeclContext()->isFunctionOrMethod());
      assert((*I)->getUnderlyingDecl()->isFunctionOrFunctionTemplate());
    }
  }
#endif

  // It would be nice to avoid this copy.
  TemplateArgumentListInfo TABuffer;
  TemplateArgumentListInfo *ExplicitTemplateArgs = 0;
  if (ULE->hasExplicitTemplateArgs()) {
    ULE->copyTemplateArgumentsInto(TABuffer);
    ExplicitTemplateArgs = &TABuffer;
  }

  for (UnresolvedLookupExpr::decls_iterator I = ULE->decls_begin(),
         E = ULE->decls_end(); I != E; ++I)
    AddOverloadedCallCandidate(*this, I.getPair(), ExplicitTemplateArgs,
                               Args, NumArgs, CandidateSet,
                               PartialOverloading, /*KnownValid*/ true);

  if (ULE->requiresADL())
    AddArgumentDependentLookupCandidates(ULE->getName(), /*Operator*/ false,
                                         Args, NumArgs,
                                         ExplicitTemplateArgs,
                                         CandidateSet,
                                         PartialOverloading,
                                         ULE->isStdAssociatedNamespace());
}

/// Attempt to recover from an ill-formed use of a non-dependent name in a
/// template, where the non-dependent name was declared after the template
/// was defined. This is common in code written for a compilers which do not
/// correctly implement two-stage name lookup.
///
/// Returns true if a viable candidate was found and a diagnostic was issued.
static bool
DiagnoseTwoPhaseLookup(Sema &SemaRef, SourceLocation FnLoc,
                       const CXXScopeSpec &SS, LookupResult &R,
                       TemplateArgumentListInfo *ExplicitTemplateArgs,
                       Expr **Args, unsigned NumArgs) {
  if (SemaRef.ActiveTemplateInstantiations.empty() || !SS.isEmpty())
    return false;

  for (DeclContext *DC = SemaRef.CurContext; DC; DC = DC->getParent()) {
    SemaRef.LookupQualifiedName(R, DC);

    if (!R.empty()) {
      R.suppressDiagnostics();

      if (isa<CXXRecordDecl>(DC)) {
        // Don't diagnose names we find in classes; we get much better
        // diagnostics for these from DiagnoseEmptyLookup.
        R.clear();
        return false;
      }

      OverloadCandidateSet Candidates(FnLoc);
      for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I)
        AddOverloadedCallCandidate(SemaRef, I.getPair(),
                                   ExplicitTemplateArgs, Args, NumArgs,
                                   Candidates, false, /*KnownValid*/ false);

      OverloadCandidateSet::iterator Best;
      if (Candidates.BestViableFunction(SemaRef, FnLoc, Best) != OR_Success) {
        // No viable functions. Don't bother the user with notes for functions
        // which don't work and shouldn't be found anyway.
        R.clear();
        return false;
      }

      // Find the namespaces where ADL would have looked, and suggest
      // declaring the function there instead.
      Sema::AssociatedNamespaceSet AssociatedNamespaces;
      Sema::AssociatedClassSet AssociatedClasses;
      SemaRef.FindAssociatedClassesAndNamespaces(Args, NumArgs,
                                                 AssociatedNamespaces,
                                                 AssociatedClasses);
      // Never suggest declaring a function within namespace 'std'. 
      Sema::AssociatedNamespaceSet SuggestedNamespaces;
      if (DeclContext *Std = SemaRef.getStdNamespace()) {
        for (Sema::AssociatedNamespaceSet::iterator
               it = AssociatedNamespaces.begin(),
               end = AssociatedNamespaces.end(); it != end; ++it) {
          if (!Std->Encloses(*it))
            SuggestedNamespaces.insert(*it);
        }
      } else {
        // Lacking the 'std::' namespace, use all of the associated namespaces.
        SuggestedNamespaces = AssociatedNamespaces;
      }

      SemaRef.Diag(R.getNameLoc(), diag::err_not_found_by_two_phase_lookup)
        << R.getLookupName();
      if (SuggestedNamespaces.empty()) {
        SemaRef.Diag(Best->Function->getLocation(),
                     diag::note_not_found_by_two_phase_lookup)
          << R.getLookupName() << 0;
      } else if (SuggestedNamespaces.size() == 1) {
        SemaRef.Diag(Best->Function->getLocation(),
                     diag::note_not_found_by_two_phase_lookup)
          << R.getLookupName() << 1 << *SuggestedNamespaces.begin();
      } else {
        // FIXME: It would be useful to list the associated namespaces here,
        // but the diagnostics infrastructure doesn't provide a way to produce
        // a localized representation of a list of items.
        SemaRef.Diag(Best->Function->getLocation(),
                     diag::note_not_found_by_two_phase_lookup)
          << R.getLookupName() << 2;
      }

      // Try to recover by calling this function.
      return true;
    }

    R.clear();
  }

  return false;
}

/// Attempt to recover from ill-formed use of a non-dependent operator in a
/// template, where the non-dependent operator was declared after the template
/// was defined.
///
/// Returns true if a viable candidate was found and a diagnostic was issued.
static bool
DiagnoseTwoPhaseOperatorLookup(Sema &SemaRef, OverloadedOperatorKind Op,
                               SourceLocation OpLoc,
                               Expr **Args, unsigned NumArgs) {
  DeclarationName OpName =
    SemaRef.Context.DeclarationNames.getCXXOperatorName(Op);
  LookupResult R(SemaRef, OpName, OpLoc, Sema::LookupOperatorName);
  return DiagnoseTwoPhaseLookup(SemaRef, OpLoc, CXXScopeSpec(), R,
                                /*ExplicitTemplateArgs=*/0, Args, NumArgs);
}

/// Attempts to recover from a call where no functions were found.
///
/// Returns true if new candidates were found.
static ExprResult
BuildRecoveryCallExpr(Sema &SemaRef, Scope *S, Expr *Fn,
                      UnresolvedLookupExpr *ULE,
                      SourceLocation LParenLoc,
                      Expr **Args, unsigned NumArgs,
                      SourceLocation RParenLoc,
                      bool EmptyLookup) {

  CXXScopeSpec SS;
  SS.Adopt(ULE->getQualifierLoc());

  TemplateArgumentListInfo TABuffer;
  TemplateArgumentListInfo *ExplicitTemplateArgs = 0;
  if (ULE->hasExplicitTemplateArgs()) {
    ULE->copyTemplateArgumentsInto(TABuffer);
    ExplicitTemplateArgs = &TABuffer;
  }

  LookupResult R(SemaRef, ULE->getName(), ULE->getNameLoc(),
                 Sema::LookupOrdinaryName);
  if (!DiagnoseTwoPhaseLookup(SemaRef, Fn->getExprLoc(), SS, R,
                              ExplicitTemplateArgs, Args, NumArgs) &&
      (!EmptyLookup ||
       SemaRef.DiagnoseEmptyLookup(S, SS, R, Sema::CTC_Expression)))
    return ExprError();

  assert(!R.empty() && "lookup results empty despite recovery");

  // Build an implicit member call if appropriate.  Just drop the
  // casts and such from the call, we don't really care.
  ExprResult NewFn = ExprError();
  if ((*R.begin())->isCXXClassMember())
    NewFn = SemaRef.BuildPossibleImplicitMemberExpr(SS, R,
                                                    ExplicitTemplateArgs);
  else if (ExplicitTemplateArgs)
    NewFn = SemaRef.BuildTemplateIdExpr(SS, R, false, *ExplicitTemplateArgs);
  else
    NewFn = SemaRef.BuildDeclarationNameExpr(SS, R, false);

  if (NewFn.isInvalid())
    return ExprError();

  // This shouldn't cause an infinite loop because we're giving it
  // an expression with viable lookup results, which should never
  // end up here.
  return SemaRef.ActOnCallExpr(/*Scope*/ 0, NewFn.take(), LParenLoc,
                               MultiExprArg(Args, NumArgs), RParenLoc);
}

/// ResolveOverloadedCallFn - Given the call expression that calls Fn
/// (which eventually refers to the declaration Func) and the call
/// arguments Args/NumArgs, attempt to resolve the function call down
/// to a specific function. If overload resolution succeeds, returns
/// the function declaration produced by overload
/// resolution. Otherwise, emits diagnostics, deletes all of the
/// arguments and Fn, and returns NULL.
ExprResult
Sema::BuildOverloadedCallExpr(Scope *S, Expr *Fn, UnresolvedLookupExpr *ULE,
                              SourceLocation LParenLoc,
                              Expr **Args, unsigned NumArgs,
                              SourceLocation RParenLoc,
                              Expr *ExecConfig) {
#ifndef NDEBUG
  if (ULE->requiresADL()) {
    // To do ADL, we must have found an unqualified name.
    assert(!ULE->getQualifier() && "qualified name with ADL");

    // We don't perform ADL for implicit declarations of builtins.
    // Verify that this was correctly set up.
    FunctionDecl *F;
    if (ULE->decls_begin() + 1 == ULE->decls_end() &&
        (F = dyn_cast<FunctionDecl>(*ULE->decls_begin())) &&
        F->getBuiltinID() && F->isImplicit())
      assert(0 && "performing ADL for builtin");

    // We don't perform ADL in C.
    assert(getLangOptions().CPlusPlus && "ADL enabled in C");
  } else
    assert(!ULE->isStdAssociatedNamespace() &&
           "std is associated namespace but not doing ADL");
#endif

  OverloadCandidateSet CandidateSet(Fn->getExprLoc());

  // Add the functions denoted by the callee to the set of candidate
  // functions, including those from argument-dependent lookup.
  AddOverloadedCallCandidates(ULE, Args, NumArgs, CandidateSet);

  // If we found nothing, try to recover.
  // BuildRecoveryCallExpr diagnoses the error itself, so we just bail
  // out if it fails.
  if (CandidateSet.empty())
    return BuildRecoveryCallExpr(*this, S, Fn, ULE, LParenLoc, Args, NumArgs,
                                 RParenLoc, /*EmptyLookup=*/true);

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, Fn->getLocStart(), Best)) {
  case OR_Success: {
    FunctionDecl *FDecl = Best->Function;
    MarkDeclarationReferenced(Fn->getExprLoc(), FDecl);
    CheckUnresolvedLookupAccess(ULE, Best->FoundDecl);
    DiagnoseUseOfDecl(FDecl? FDecl : Best->FoundDecl.getDecl(),
                      ULE->getNameLoc());
    Fn = FixOverloadedFunctionReference(Fn, Best->FoundDecl, FDecl);
    return BuildResolvedCallExpr(Fn, FDecl, LParenLoc, Args, NumArgs, RParenLoc,
                                 ExecConfig);
  }

  case OR_No_Viable_Function: {
    // Try to recover by looking for viable functions which the user might
    // have meant to call.
    ExprResult Recovery = BuildRecoveryCallExpr(*this, S, Fn, ULE, LParenLoc,
                                                Args, NumArgs, RParenLoc,
                                                /*EmptyLookup=*/false);
    if (!Recovery.isInvalid())
      return Recovery;

    Diag(Fn->getSourceRange().getBegin(),
         diag::err_ovl_no_viable_function_in_call)
      << ULE->getName() << Fn->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    break;
  }

  case OR_Ambiguous:
    Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_ambiguous_call)
      << ULE->getName() << Fn->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_ViableCandidates, Args, NumArgs);
    break;

  case OR_Deleted:
    {
      Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_deleted_call)
        << Best->Function->isDeleted()
        << ULE->getName()
        << getDeletedOrUnavailableSuffix(Best->Function)
        << Fn->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    }
    break;
  }

  // Overload resolution failed.
  return ExprError();
}

static bool IsOverloaded(const UnresolvedSetImpl &Functions) {
  return Functions.size() > 1 ||
    (Functions.size() == 1 && isa<FunctionTemplateDecl>(*Functions.begin()));
}

/// \brief Create a unary operation that may resolve to an overloaded
/// operator.
///
/// \param OpLoc The location of the operator itself (e.g., '*').
///
/// \param OpcIn The UnaryOperator::Opcode that describes this
/// operator.
///
/// \param Functions The set of non-member functions that will be
/// considered by overload resolution. The caller needs to build this
/// set based on the context using, e.g.,
/// LookupOverloadedOperatorName() and ArgumentDependentLookup(). This
/// set should not contain any member functions; those will be added
/// by CreateOverloadedUnaryOp().
///
/// \param input The input argument.
ExprResult
Sema::CreateOverloadedUnaryOp(SourceLocation OpLoc, unsigned OpcIn,
                              const UnresolvedSetImpl &Fns,
                              Expr *Input) {
  UnaryOperator::Opcode Opc = static_cast<UnaryOperator::Opcode>(OpcIn);

  OverloadedOperatorKind Op = UnaryOperator::getOverloadedOperator(Opc);
  assert(Op != OO_None && "Invalid opcode for overloaded unary operator");
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);
  // TODO: provide better source location info.
  DeclarationNameInfo OpNameInfo(OpName, OpLoc);

  if (Input->getObjectKind() == OK_ObjCProperty) {
    ExprResult Result = ConvertPropertyForRValue(Input);
    if (Result.isInvalid())
      return ExprError();
    Input = Result.take();
  }

  Expr *Args[2] = { Input, 0 };
  unsigned NumArgs = 1;

  // For post-increment and post-decrement, add the implicit '0' as
  // the second argument, so that we know this is a post-increment or
  // post-decrement.
  if (Opc == UO_PostInc || Opc == UO_PostDec) {
    llvm::APSInt Zero(Context.getTypeSize(Context.IntTy), false);
    Args[1] = IntegerLiteral::Create(Context, Zero, Context.IntTy,
                                     SourceLocation());
    NumArgs = 2;
  }

  if (Input->isTypeDependent()) {
    if (Fns.empty())
      return Owned(new (Context) UnaryOperator(Input,
                                               Opc,
                                               Context.DependentTy,
                                               VK_RValue, OK_Ordinary,
                                               OpLoc));

    CXXRecordDecl *NamingClass = 0; // because lookup ignores member operators
    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, NamingClass,
                                     NestedNameSpecifierLoc(), OpNameInfo,
                                     /*ADL*/ true, IsOverloaded(Fns),
                                     Fns.begin(), Fns.end());
    return Owned(new (Context) CXXOperatorCallExpr(Context, Op, Fn,
                                                  &Args[0], NumArgs,
                                                   Context.DependentTy,
                                                   VK_RValue,
                                                   OpLoc));
  }

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet(OpLoc);

  // Add the candidates from the given function set.
  AddFunctionCandidates(Fns, &Args[0], NumArgs, CandidateSet, false);

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(Op, OpLoc, &Args[0], NumArgs, CandidateSet);

  // Add candidates from ADL.
  AddArgumentDependentLookupCandidates(OpName, /*Operator*/ true,
                                       Args, NumArgs,
                                       /*ExplicitTemplateArgs*/ 0,
                                       CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(Op, OpLoc, &Args[0], NumArgs, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, OpLoc, Best)) {
  case OR_Success: {
    // We found a built-in operator or an overloaded operator.
    FunctionDecl *FnDecl = Best->Function;

    if (FnDecl) {
      // We matched an overloaded operator. Build a call to that
      // operator.

      MarkDeclarationReferenced(OpLoc, FnDecl);

      // Convert the arguments.
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
        CheckMemberOperatorAccess(OpLoc, Args[0], 0, Best->FoundDecl);

        ExprResult InputRes =
          PerformObjectArgumentInitialization(Input, /*Qualifier=*/0,
                                              Best->FoundDecl, Method);
        if (InputRes.isInvalid())
          return ExprError();
        Input = InputRes.take();
      } else {
        // Convert the arguments.
        ExprResult InputInit
          = PerformCopyInitialization(InitializedEntity::InitializeParameter(
                                                      Context,
                                                      FnDecl->getParamDecl(0)),
                                      SourceLocation(),
                                      Input);
        if (InputInit.isInvalid())
          return ExprError();
        Input = InputInit.take();
      }

      DiagnoseUseOfDecl(Best->FoundDecl, OpLoc);

      // Determine the result type.
      QualType ResultTy = FnDecl->getResultType();
      ExprValueKind VK = Expr::getValueKindForType(ResultTy);
      ResultTy = ResultTy.getNonLValueExprType(Context);

      // Build the actual expression node.
      ExprResult FnExpr = CreateFunctionRefExpr(*this, FnDecl);
      if (FnExpr.isInvalid())
        return ExprError();

      Args[0] = Input;
      CallExpr *TheCall =
        new (Context) CXXOperatorCallExpr(Context, Op, FnExpr.take(),
                                          Args, NumArgs, ResultTy, VK, OpLoc);

      if (CheckCallReturnType(FnDecl->getResultType(), OpLoc, TheCall,
                              FnDecl))
        return ExprError();

      return MaybeBindToTemporary(TheCall);
    } else {
      // We matched a built-in operator. Convert the arguments, then
      // break out so that we will build the appropriate built-in
      // operator node.
      ExprResult InputRes =
        PerformImplicitConversion(Input, Best->BuiltinTypes.ParamTypes[0],
                                  Best->Conversions[0], AA_Passing);
      if (InputRes.isInvalid())
        return ExprError();
      Input = InputRes.take();
      break;
    }
  }

  case OR_No_Viable_Function:
    // This is an erroneous use of an operator which can be overloaded by
    // a non-member function. Check for non-member operators which were
    // defined too late to be candidates.
    if (DiagnoseTwoPhaseOperatorLookup(*this, Op, OpLoc, Args, NumArgs))
      // FIXME: Recover by calling the found function.
      return ExprError();

    // No viable function; fall through to handling this as a
    // built-in operator, which will produce an error message for us.
    break;

  case OR_Ambiguous:
    Diag(OpLoc,  diag::err_ovl_ambiguous_oper_unary)
        << UnaryOperator::getOpcodeStr(Opc)
        << Input->getType()
        << Input->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_ViableCandidates,
                                Args, NumArgs,
                                UnaryOperator::getOpcodeStr(Opc), OpLoc);
    return ExprError();

  case OR_Deleted:
    Diag(OpLoc, diag::err_ovl_deleted_oper)
      << Best->Function->isDeleted()
      << UnaryOperator::getOpcodeStr(Opc)
      << getDeletedOrUnavailableSuffix(Best->Function)
      << Input->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    return ExprError();
  }

  // Either we found no viable overloaded operator or we matched a
  // built-in operator. In either case, fall through to trying to
  // build a built-in operation.
  return CreateBuiltinUnaryOp(OpLoc, Opc, Input);
}

/// \brief Create a binary operation that may resolve to an overloaded
/// operator.
///
/// \param OpLoc The location of the operator itself (e.g., '+').
///
/// \param OpcIn The BinaryOperator::Opcode that describes this
/// operator.
///
/// \param Functions The set of non-member functions that will be
/// considered by overload resolution. The caller needs to build this
/// set based on the context using, e.g.,
/// LookupOverloadedOperatorName() and ArgumentDependentLookup(). This
/// set should not contain any member functions; those will be added
/// by CreateOverloadedBinOp().
///
/// \param LHS Left-hand argument.
/// \param RHS Right-hand argument.
ExprResult
Sema::CreateOverloadedBinOp(SourceLocation OpLoc,
                            unsigned OpcIn,
                            const UnresolvedSetImpl &Fns,
                            Expr *LHS, Expr *RHS) {
  Expr *Args[2] = { LHS, RHS };
  LHS=RHS=0; //Please use only Args instead of LHS/RHS couple

  BinaryOperator::Opcode Opc = static_cast<BinaryOperator::Opcode>(OpcIn);
  OverloadedOperatorKind Op = BinaryOperator::getOverloadedOperator(Opc);
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  // If either side is type-dependent, create an appropriate dependent
  // expression.
  if (Args[0]->isTypeDependent() || Args[1]->isTypeDependent()) {
    if (Fns.empty()) {
      // If there are no functions to store, just build a dependent
      // BinaryOperator or CompoundAssignment.
      if (Opc <= BO_Assign || Opc > BO_OrAssign)
        return Owned(new (Context) BinaryOperator(Args[0], Args[1], Opc,
                                                  Context.DependentTy,
                                                  VK_RValue, OK_Ordinary,
                                                  OpLoc));

      return Owned(new (Context) CompoundAssignOperator(Args[0], Args[1], Opc,
                                                        Context.DependentTy,
                                                        VK_LValue,
                                                        OK_Ordinary,
                                                        Context.DependentTy,
                                                        Context.DependentTy,
                                                        OpLoc));
    }

    // FIXME: save results of ADL from here?
    CXXRecordDecl *NamingClass = 0; // because lookup ignores member operators
    // TODO: provide better source location info in DNLoc component.
    DeclarationNameInfo OpNameInfo(OpName, OpLoc);
    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, NamingClass, 
                                     NestedNameSpecifierLoc(), OpNameInfo, 
                                     /*ADL*/ true, IsOverloaded(Fns),
                                     Fns.begin(), Fns.end());
    return Owned(new (Context) CXXOperatorCallExpr(Context, Op, Fn,
                                                   Args, 2,
                                                   Context.DependentTy,
                                                   VK_RValue,
                                                   OpLoc));
  }

  // Always do property rvalue conversions on the RHS.
  if (Args[1]->getObjectKind() == OK_ObjCProperty) {
    ExprResult Result = ConvertPropertyForRValue(Args[1]);
    if (Result.isInvalid())
      return ExprError();
    Args[1] = Result.take();
  }

  // The LHS is more complicated.
  if (Args[0]->getObjectKind() == OK_ObjCProperty) {

    // There's a tension for assignment operators between primitive
    // property assignment and the overloaded operators.
    if (BinaryOperator::isAssignmentOp(Opc)) {
      const ObjCPropertyRefExpr *PRE = LHS->getObjCProperty();

      // Is the property "logically" settable?
      bool Settable = (PRE->isExplicitProperty() ||
                       PRE->getImplicitPropertySetter());

      // To avoid gratuitously inventing semantics, use the primitive
      // unless it isn't.  Thoughts in case we ever really care:
      // - If the property isn't logically settable, we have to
      //   load and hope.
      // - If the property is settable and this is simple assignment,
      //   we really should use the primitive.
      // - If the property is settable, then we could try overloading
      //   on a generic lvalue of the appropriate type;  if it works
      //   out to a builtin candidate, we would do that same operation
      //   on the property, and otherwise just error.
      if (Settable)
        return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);
    }

    ExprResult Result = ConvertPropertyForRValue(Args[0]);
    if (Result.isInvalid())
      return ExprError();
    Args[0] = Result.take();
  }

  // If this is the assignment operator, we only perform overload resolution
  // if the left-hand side is a class or enumeration type. This is actually
  // a hack. The standard requires that we do overload resolution between the
  // various built-in candidates, but as DR507 points out, this can lead to
  // problems. So we do it this way, which pretty much follows what GCC does.
  // Note that we go the traditional code path for compound assignment forms.
  if (Opc == BO_Assign && !Args[0]->getType()->isOverloadableType())
    return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);

  // If this is the .* operator, which is not overloadable, just
  // create a built-in binary operator.
  if (Opc == BO_PtrMemD)
    return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet(OpLoc);

  // Add the candidates from the given function set.
  AddFunctionCandidates(Fns, Args, 2, CandidateSet, false);

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(Op, OpLoc, Args, 2, CandidateSet);

  // Add candidates from ADL.
  AddArgumentDependentLookupCandidates(OpName, /*Operator*/ true,
                                       Args, 2,
                                       /*ExplicitTemplateArgs*/ 0,
                                       CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(Op, OpLoc, Args, 2, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, OpLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        MarkDeclarationReferenced(OpLoc, FnDecl);

        // Convert the arguments.
        if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
          // Best->Access is only meaningful for class members.
          CheckMemberOperatorAccess(OpLoc, Args[0], Args[1], Best->FoundDecl);

          ExprResult Arg1 =
            PerformCopyInitialization(
              InitializedEntity::InitializeParameter(Context,
                                                     FnDecl->getParamDecl(0)),
              SourceLocation(), Owned(Args[1]));
          if (Arg1.isInvalid())
            return ExprError();

          ExprResult Arg0 =
            PerformObjectArgumentInitialization(Args[0], /*Qualifier=*/0,
                                                Best->FoundDecl, Method);
          if (Arg0.isInvalid())
            return ExprError();
          Args[0] = Arg0.takeAs<Expr>();
          Args[1] = RHS = Arg1.takeAs<Expr>();
        } else {
          // Convert the arguments.
          ExprResult Arg0 = PerformCopyInitialization(
            InitializedEntity::InitializeParameter(Context,
                                                   FnDecl->getParamDecl(0)),
            SourceLocation(), Owned(Args[0]));
          if (Arg0.isInvalid())
            return ExprError();

          ExprResult Arg1 =
            PerformCopyInitialization(
              InitializedEntity::InitializeParameter(Context,
                                                     FnDecl->getParamDecl(1)),
              SourceLocation(), Owned(Args[1]));
          if (Arg1.isInvalid())
            return ExprError();
          Args[0] = LHS = Arg0.takeAs<Expr>();
          Args[1] = RHS = Arg1.takeAs<Expr>();
        }

        DiagnoseUseOfDecl(Best->FoundDecl, OpLoc);

        // Determine the result type.
        QualType ResultTy = FnDecl->getResultType();
        ExprValueKind VK = Expr::getValueKindForType(ResultTy);
        ResultTy = ResultTy.getNonLValueExprType(Context);

        // Build the actual expression node.
        ExprResult FnExpr = CreateFunctionRefExpr(*this, FnDecl, OpLoc);
        if (FnExpr.isInvalid())
          return ExprError();

        CXXOperatorCallExpr *TheCall =
          new (Context) CXXOperatorCallExpr(Context, Op, FnExpr.take(),
                                            Args, 2, ResultTy, VK, OpLoc);

        if (CheckCallReturnType(FnDecl->getResultType(), OpLoc, TheCall,
                                FnDecl))
          return ExprError();

        return MaybeBindToTemporary(TheCall);
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        ExprResult ArgsRes0 =
          PerformImplicitConversion(Args[0], Best->BuiltinTypes.ParamTypes[0],
                                    Best->Conversions[0], AA_Passing);
        if (ArgsRes0.isInvalid())
          return ExprError();
        Args[0] = ArgsRes0.take();

        ExprResult ArgsRes1 =
          PerformImplicitConversion(Args[1], Best->BuiltinTypes.ParamTypes[1],
                                    Best->Conversions[1], AA_Passing);
        if (ArgsRes1.isInvalid())
          return ExprError();
        Args[1] = ArgsRes1.take();
        break;
      }
    }

    case OR_No_Viable_Function: {
      // C++ [over.match.oper]p9:
      //   If the operator is the operator , [...] and there are no
      //   viable functions, then the operator is assumed to be the
      //   built-in operator and interpreted according to clause 5.
      if (Opc == BO_Comma)
        break;

      // For class as left operand for assignment or compound assigment
      // operator do not fall through to handling in built-in, but report that
      // no overloaded assignment operator found
      ExprResult Result = ExprError();
      if (Args[0]->getType()->isRecordType() &&
          Opc >= BO_Assign && Opc <= BO_OrAssign) {
        Diag(OpLoc,  diag::err_ovl_no_viable_oper)
             << BinaryOperator::getOpcodeStr(Opc)
             << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      } else {
        // This is an erroneous use of an operator which can be overloaded by
        // a non-member function. Check for non-member operators which were
        // defined too late to be candidates.
        if (DiagnoseTwoPhaseOperatorLookup(*this, Op, OpLoc, Args, 2))
          // FIXME: Recover by calling the found function.
          return ExprError();

        // No viable function; try to create a built-in operation, which will
        // produce an error. Then, show the non-viable candidates.
        Result = CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);
      }
      assert(Result.isInvalid() &&
             "C++ binary operator overloading is missing candidates!");
      if (Result.isInvalid())
        CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, 2,
                                    BinaryOperator::getOpcodeStr(Opc), OpLoc);
      return move(Result);
    }

    case OR_Ambiguous:
      Diag(OpLoc,  diag::err_ovl_ambiguous_oper_binary)
          << BinaryOperator::getOpcodeStr(Opc)
          << Args[0]->getType() << Args[1]->getType()
          << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_ViableCandidates, Args, 2,
                                  BinaryOperator::getOpcodeStr(Opc), OpLoc);
      return ExprError();

    case OR_Deleted:
      Diag(OpLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << BinaryOperator::getOpcodeStr(Opc)
        << getDeletedOrUnavailableSuffix(Best->Function)
        << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, 2);
      return ExprError();
  }

  // We matched a built-in operator; build it.
  return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);
}

ExprResult
Sema::CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
                                         SourceLocation RLoc,
                                         Expr *Base, Expr *Idx) {
  Expr *Args[2] = { Base, Idx };
  DeclarationName OpName =
      Context.DeclarationNames.getCXXOperatorName(OO_Subscript);

  // If either side is type-dependent, create an appropriate dependent
  // expression.
  if (Args[0]->isTypeDependent() || Args[1]->isTypeDependent()) {

    CXXRecordDecl *NamingClass = 0; // because lookup ignores member operators
    // CHECKME: no 'operator' keyword?
    DeclarationNameInfo OpNameInfo(OpName, LLoc);
    OpNameInfo.setCXXOperatorNameRange(SourceRange(LLoc, RLoc));
    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, NamingClass,
                                     NestedNameSpecifierLoc(), OpNameInfo,
                                     /*ADL*/ true, /*Overloaded*/ false,
                                     UnresolvedSetIterator(),
                                     UnresolvedSetIterator());
    // Can't add any actual overloads yet

    return Owned(new (Context) CXXOperatorCallExpr(Context, OO_Subscript, Fn,
                                                   Args, 2,
                                                   Context.DependentTy,
                                                   VK_RValue,
                                                   RLoc));
  }

  if (Args[0]->getObjectKind() == OK_ObjCProperty) {
    ExprResult Result = ConvertPropertyForRValue(Args[0]);
    if (Result.isInvalid())
      return ExprError();
    Args[0] = Result.take();
  }
  if (Args[1]->getObjectKind() == OK_ObjCProperty) {
    ExprResult Result = ConvertPropertyForRValue(Args[1]);
    if (Result.isInvalid())
      return ExprError();
    Args[1] = Result.take();
  }

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet(LLoc);

  // Subscript can only be overloaded as a member function.

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(OO_Subscript, LLoc, Args, 2, CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(OO_Subscript, LLoc, Args, 2, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, LLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        MarkDeclarationReferenced(LLoc, FnDecl);

        CheckMemberOperatorAccess(LLoc, Args[0], Args[1], Best->FoundDecl);
        DiagnoseUseOfDecl(Best->FoundDecl, LLoc);

        // Convert the arguments.
        CXXMethodDecl *Method = cast<CXXMethodDecl>(FnDecl);
        ExprResult Arg0 =
          PerformObjectArgumentInitialization(Args[0], /*Qualifier=*/0,
                                              Best->FoundDecl, Method);
        if (Arg0.isInvalid())
          return ExprError();
        Args[0] = Arg0.take();

        // Convert the arguments.
        ExprResult InputInit
          = PerformCopyInitialization(InitializedEntity::InitializeParameter(
                                                      Context,
                                                      FnDecl->getParamDecl(0)),
                                      SourceLocation(),
                                      Owned(Args[1]));
        if (InputInit.isInvalid())
          return ExprError();

        Args[1] = InputInit.takeAs<Expr>();

        // Determine the result type
        QualType ResultTy = FnDecl->getResultType();
        ExprValueKind VK = Expr::getValueKindForType(ResultTy);
        ResultTy = ResultTy.getNonLValueExprType(Context);

        // Build the actual expression node.
        ExprResult FnExpr = CreateFunctionRefExpr(*this, FnDecl, LLoc);
        if (FnExpr.isInvalid())
          return ExprError();

        CXXOperatorCallExpr *TheCall =
          new (Context) CXXOperatorCallExpr(Context, OO_Subscript,
                                            FnExpr.take(), Args, 2,
                                            ResultTy, VK, RLoc);

        if (CheckCallReturnType(FnDecl->getResultType(), LLoc, TheCall,
                                FnDecl))
          return ExprError();

        return MaybeBindToTemporary(TheCall);
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        ExprResult ArgsRes0 =
          PerformImplicitConversion(Args[0], Best->BuiltinTypes.ParamTypes[0],
                                    Best->Conversions[0], AA_Passing);
        if (ArgsRes0.isInvalid())
          return ExprError();
        Args[0] = ArgsRes0.take();

        ExprResult ArgsRes1 =
          PerformImplicitConversion(Args[1], Best->BuiltinTypes.ParamTypes[1],
                                    Best->Conversions[1], AA_Passing);
        if (ArgsRes1.isInvalid())
          return ExprError();
        Args[1] = ArgsRes1.take();

        break;
      }
    }

    case OR_No_Viable_Function: {
      if (CandidateSet.empty())
        Diag(LLoc, diag::err_ovl_no_oper)
          << Args[0]->getType() << /*subscript*/ 0
          << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      else
        Diag(LLoc, diag::err_ovl_no_viable_subscript)
          << Args[0]->getType()
          << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, 2,
                                  "[]", LLoc);
      return ExprError();
    }

    case OR_Ambiguous:
      Diag(LLoc,  diag::err_ovl_ambiguous_oper_binary)
          << "[]"
          << Args[0]->getType() << Args[1]->getType()
          << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_ViableCandidates, Args, 2,
                                  "[]", LLoc);
      return ExprError();

    case OR_Deleted:
      Diag(LLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted() << "[]"
        << getDeletedOrUnavailableSuffix(Best->Function)
        << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, 2,
                                  "[]", LLoc);
      return ExprError();
    }

  // We matched a built-in operator; build it.
  return CreateBuiltinArraySubscriptExpr(Args[0], LLoc, Args[1], RLoc);
}

/// BuildCallToMemberFunction - Build a call to a member
/// function. MemExpr is the expression that refers to the member
/// function (and includes the object parameter), Args/NumArgs are the
/// arguments to the function call (not including the object
/// parameter). The caller needs to validate that the member
/// expression refers to a non-static member function or an overloaded
/// member function.
ExprResult
Sema::BuildCallToMemberFunction(Scope *S, Expr *MemExprE,
                                SourceLocation LParenLoc, Expr **Args,
                                unsigned NumArgs, SourceLocation RParenLoc) {
  assert(MemExprE->getType() == Context.BoundMemberTy ||
         MemExprE->getType() == Context.OverloadTy);

  // Dig out the member expression. This holds both the object
  // argument and the member function we're referring to.
  Expr *NakedMemExpr = MemExprE->IgnoreParens();

  // Determine whether this is a call to a pointer-to-member function.
  if (BinaryOperator *op = dyn_cast<BinaryOperator>(NakedMemExpr)) {
    assert(op->getType() == Context.BoundMemberTy);
    assert(op->getOpcode() == BO_PtrMemD || op->getOpcode() == BO_PtrMemI);

    QualType fnType =
      op->getRHS()->getType()->castAs<MemberPointerType>()->getPointeeType();

    const FunctionProtoType *proto = fnType->castAs<FunctionProtoType>();
    QualType resultType = proto->getCallResultType(Context);
    ExprValueKind valueKind = Expr::getValueKindForType(proto->getResultType());

    // Check that the object type isn't more qualified than the
    // member function we're calling.
    Qualifiers funcQuals = Qualifiers::fromCVRMask(proto->getTypeQuals());

    QualType objectType = op->getLHS()->getType();
    if (op->getOpcode() == BO_PtrMemI)
      objectType = objectType->castAs<PointerType>()->getPointeeType();
    Qualifiers objectQuals = objectType.getQualifiers();

    Qualifiers difference = objectQuals - funcQuals;
    difference.removeObjCGCAttr();
    difference.removeAddressSpace();
    if (difference) {
      std::string qualsString = difference.getAsString();
      Diag(LParenLoc, diag::err_pointer_to_member_call_drops_quals)
        << fnType.getUnqualifiedType()
        << qualsString
        << (qualsString.find(' ') == std::string::npos ? 1 : 2);
    }
              
    CXXMemberCallExpr *call
      = new (Context) CXXMemberCallExpr(Context, MemExprE, Args, NumArgs,
                                        resultType, valueKind, RParenLoc);

    if (CheckCallReturnType(proto->getResultType(),
                            op->getRHS()->getSourceRange().getBegin(),
                            call, 0))
      return ExprError();

    if (ConvertArgumentsForCall(call, op, 0, proto, Args, NumArgs, RParenLoc))
      return ExprError();

    return MaybeBindToTemporary(call);
  }

  MemberExpr *MemExpr;
  CXXMethodDecl *Method = 0;
  DeclAccessPair FoundDecl = DeclAccessPair::make(0, AS_public);
  NestedNameSpecifier *Qualifier = 0;
  if (isa<MemberExpr>(NakedMemExpr)) {
    MemExpr = cast<MemberExpr>(NakedMemExpr);
    Method = cast<CXXMethodDecl>(MemExpr->getMemberDecl());
    FoundDecl = MemExpr->getFoundDecl();
    Qualifier = MemExpr->getQualifier();
  } else {
    UnresolvedMemberExpr *UnresExpr = cast<UnresolvedMemberExpr>(NakedMemExpr);
    Qualifier = UnresExpr->getQualifier();

    QualType ObjectType = UnresExpr->getBaseType();
    Expr::Classification ObjectClassification
      = UnresExpr->isArrow()? Expr::Classification::makeSimpleLValue()
                            : UnresExpr->getBase()->Classify(Context);

    // Add overload candidates
    OverloadCandidateSet CandidateSet(UnresExpr->getMemberLoc());

    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (UnresExpr->hasExplicitTemplateArgs()) {
      UnresExpr->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    for (UnresolvedMemberExpr::decls_iterator I = UnresExpr->decls_begin(),
           E = UnresExpr->decls_end(); I != E; ++I) {

      NamedDecl *Func = *I;
      CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(Func->getDeclContext());
      if (isa<UsingShadowDecl>(Func))
        Func = cast<UsingShadowDecl>(Func)->getTargetDecl();


      // Microsoft supports direct constructor calls.
      if (getLangOptions().Microsoft && isa<CXXConstructorDecl>(Func)) {
        AddOverloadCandidate(cast<CXXConstructorDecl>(Func), I.getPair(), Args, NumArgs,
                             CandidateSet);
      } else if ((Method = dyn_cast<CXXMethodDecl>(Func))) {
        // If explicit template arguments were provided, we can't call a
        // non-template member function.
        if (TemplateArgs)
          continue;

        AddMethodCandidate(Method, I.getPair(), ActingDC, ObjectType,
                           ObjectClassification,
                           Args, NumArgs, CandidateSet,
                           /*SuppressUserConversions=*/false);
      } else {
        AddMethodTemplateCandidate(cast<FunctionTemplateDecl>(Func),
                                   I.getPair(), ActingDC, TemplateArgs,
                                   ObjectType,  ObjectClassification,
                                   Args, NumArgs, CandidateSet,
                                   /*SuppressUsedConversions=*/false);
      }
    }

    DeclarationName DeclName = UnresExpr->getMemberName();

    OverloadCandidateSet::iterator Best;
    switch (CandidateSet.BestViableFunction(*this, UnresExpr->getLocStart(),
                                            Best)) {
    case OR_Success:
      Method = cast<CXXMethodDecl>(Best->Function);
      MarkDeclarationReferenced(UnresExpr->getMemberLoc(), Method);
      FoundDecl = Best->FoundDecl;
      CheckUnresolvedMemberAccess(UnresExpr, Best->FoundDecl);
      DiagnoseUseOfDecl(Best->FoundDecl, UnresExpr->getNameLoc());
      break;

    case OR_No_Viable_Function:
      Diag(UnresExpr->getMemberLoc(),
           diag::err_ovl_no_viable_member_function_in_call)
        << DeclName << MemExprE->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
      // FIXME: Leaking incoming expressions!
      return ExprError();

    case OR_Ambiguous:
      Diag(UnresExpr->getMemberLoc(), diag::err_ovl_ambiguous_member_call)
        << DeclName << MemExprE->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
      // FIXME: Leaking incoming expressions!
      return ExprError();

    case OR_Deleted:
      Diag(UnresExpr->getMemberLoc(), diag::err_ovl_deleted_member_call)
        << Best->Function->isDeleted()
        << DeclName 
        << getDeletedOrUnavailableSuffix(Best->Function)
        << MemExprE->getSourceRange();
      CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
      // FIXME: Leaking incoming expressions!
      return ExprError();
    }

    MemExprE = FixOverloadedFunctionReference(MemExprE, FoundDecl, Method);

    // If overload resolution picked a static member, build a
    // non-member call based on that function.
    if (Method->isStatic()) {
      return BuildResolvedCallExpr(MemExprE, Method, LParenLoc,
                                   Args, NumArgs, RParenLoc);
    }

    MemExpr = cast<MemberExpr>(MemExprE->IgnoreParens());
  }

  QualType ResultType = Method->getResultType();
  ExprValueKind VK = Expr::getValueKindForType(ResultType);
  ResultType = ResultType.getNonLValueExprType(Context);

  assert(Method && "Member call to something that isn't a method?");
  CXXMemberCallExpr *TheCall =
    new (Context) CXXMemberCallExpr(Context, MemExprE, Args, NumArgs,
                                    ResultType, VK, RParenLoc);

  // Check for a valid return type.
  if (CheckCallReturnType(Method->getResultType(), MemExpr->getMemberLoc(),
                          TheCall, Method))
    return ExprError();

  // Convert the object argument (for a non-static member function call).
  // We only need to do this if there was actually an overload; otherwise
  // it was done at lookup.
  if (!Method->isStatic()) {
    ExprResult ObjectArg =
      PerformObjectArgumentInitialization(MemExpr->getBase(), Qualifier,
                                          FoundDecl, Method);
    if (ObjectArg.isInvalid())
      return ExprError();
    MemExpr->setBase(ObjectArg.take());
  }

  // Convert the rest of the arguments
  const FunctionProtoType *Proto =
    Method->getType()->getAs<FunctionProtoType>();
  if (ConvertArgumentsForCall(TheCall, MemExpr, Method, Proto, Args, NumArgs,
                              RParenLoc))
    return ExprError();

  if (CheckFunctionCall(Method, TheCall))
    return ExprError();

  if ((isa<CXXConstructorDecl>(CurContext) || 
       isa<CXXDestructorDecl>(CurContext)) && 
      TheCall->getMethodDecl()->isPure()) {
    const CXXMethodDecl *MD = TheCall->getMethodDecl();

    if (isa<CXXThisExpr>(MemExpr->getBase()->IgnoreParenCasts())) {
      Diag(MemExpr->getLocStart(), 
           diag::warn_call_to_pure_virtual_member_function_from_ctor_dtor)
        << MD->getDeclName() << isa<CXXDestructorDecl>(CurContext)
        << MD->getParent()->getDeclName();

      Diag(MD->getLocStart(), diag::note_previous_decl) << MD->getDeclName();
    }
  }
  return MaybeBindToTemporary(TheCall);
}

/// BuildCallToObjectOfClassType - Build a call to an object of class
/// type (C++ [over.call.object]), which can end up invoking an
/// overloaded function call operator (@c operator()) or performing a
/// user-defined conversion on the object argument.
ExprResult
Sema::BuildCallToObjectOfClassType(Scope *S, Expr *Obj,
                                   SourceLocation LParenLoc,
                                   Expr **Args, unsigned NumArgs,
                                   SourceLocation RParenLoc) {
  ExprResult Object = Owned(Obj);
  if (Object.get()->getObjectKind() == OK_ObjCProperty) {
    Object = ConvertPropertyForRValue(Object.take());
    if (Object.isInvalid())
      return ExprError();
  }

  assert(Object.get()->getType()->isRecordType() && "Requires object type argument");
  const RecordType *Record = Object.get()->getType()->getAs<RecordType>();

  // C++ [over.call.object]p1:
  //  If the primary-expression E in the function call syntax
  //  evaluates to a class object of type "cv T", then the set of
  //  candidate functions includes at least the function call
  //  operators of T. The function call operators of T are obtained by
  //  ordinary lookup of the name operator() in the context of
  //  (E).operator().
  OverloadCandidateSet CandidateSet(LParenLoc);
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(OO_Call);

  if (RequireCompleteType(LParenLoc, Object.get()->getType(),
                          PDiag(diag::err_incomplete_object_call)
                          << Object.get()->getSourceRange()))
    return true;

  LookupResult R(*this, OpName, LParenLoc, LookupOrdinaryName);
  LookupQualifiedName(R, Record->getDecl());
  R.suppressDiagnostics();

  for (LookupResult::iterator Oper = R.begin(), OperEnd = R.end();
       Oper != OperEnd; ++Oper) {
    AddMethodCandidate(Oper.getPair(), Object.get()->getType(),
                       Object.get()->Classify(Context), Args, NumArgs, CandidateSet,
                       /*SuppressUserConversions=*/ false);
  }

  // C++ [over.call.object]p2:
  //   In addition, for each conversion function declared in T of the
  //   form
  //
  //        operator conversion-type-id () cv-qualifier;
  //
  //   where cv-qualifier is the same cv-qualification as, or a
  //   greater cv-qualification than, cv, and where conversion-type-id
  //   denotes the type "pointer to function of (P1,...,Pn) returning
  //   R", or the type "reference to pointer to function of
  //   (P1,...,Pn) returning R", or the type "reference to function
  //   of (P1,...,Pn) returning R", a surrogate call function [...]
  //   is also considered as a candidate function. Similarly,
  //   surrogate call functions are added to the set of candidate
  //   functions for each conversion function declared in an
  //   accessible base class provided the function is not hidden
  //   within T by another intervening declaration.
  const UnresolvedSetImpl *Conversions
    = cast<CXXRecordDecl>(Record->getDecl())->getVisibleConversionFunctions();
  for (UnresolvedSetImpl::iterator I = Conversions->begin(),
         E = Conversions->end(); I != E; ++I) {
    NamedDecl *D = *I;
    CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(D->getDeclContext());
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();

    // Skip over templated conversion functions; they aren't
    // surrogates.
    if (isa<FunctionTemplateDecl>(D))
      continue;

    CXXConversionDecl *Conv = cast<CXXConversionDecl>(D);

    // Strip the reference type (if any) and then the pointer type (if
    // any) to get down to what might be a function type.
    QualType ConvType = Conv->getConversionType().getNonReferenceType();
    if (const PointerType *ConvPtrType = ConvType->getAs<PointerType>())
      ConvType = ConvPtrType->getPointeeType();

    if (const FunctionProtoType *Proto = ConvType->getAs<FunctionProtoType>())
      AddSurrogateCandidate(Conv, I.getPair(), ActingContext, Proto,
                            Object.get(), Args, NumArgs, CandidateSet);
  }

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, Object.get()->getLocStart(),
                             Best)) {
  case OR_Success:
    // Overload resolution succeeded; we'll build the appropriate call
    // below.
    break;

  case OR_No_Viable_Function:
    if (CandidateSet.empty())
      Diag(Object.get()->getSourceRange().getBegin(), diag::err_ovl_no_oper)
        << Object.get()->getType() << /*call*/ 1
        << Object.get()->getSourceRange();
    else
      Diag(Object.get()->getSourceRange().getBegin(),
           diag::err_ovl_no_viable_object_call)
        << Object.get()->getType() << Object.get()->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    break;

  case OR_Ambiguous:
    Diag(Object.get()->getSourceRange().getBegin(),
         diag::err_ovl_ambiguous_object_call)
      << Object.get()->getType() << Object.get()->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_ViableCandidates, Args, NumArgs);
    break;

  case OR_Deleted:
    Diag(Object.get()->getSourceRange().getBegin(),
         diag::err_ovl_deleted_object_call)
      << Best->Function->isDeleted()
      << Object.get()->getType() 
      << getDeletedOrUnavailableSuffix(Best->Function)
      << Object.get()->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    break;
  }

  if (Best == CandidateSet.end())
    return true;

  if (Best->Function == 0) {
    // Since there is no function declaration, this is one of the
    // surrogate candidates. Dig out the conversion function.
    CXXConversionDecl *Conv
      = cast<CXXConversionDecl>(
                         Best->Conversions[0].UserDefined.ConversionFunction);

    CheckMemberOperatorAccess(LParenLoc, Object.get(), 0, Best->FoundDecl);
    DiagnoseUseOfDecl(Best->FoundDecl, LParenLoc);

    // We selected one of the surrogate functions that converts the
    // object parameter to a function pointer. Perform the conversion
    // on the object argument, then let ActOnCallExpr finish the job.

    // Create an implicit member expr to refer to the conversion operator.
    // and then call it.
    ExprResult Call = BuildCXXMemberCallExpr(Object.get(), Best->FoundDecl, Conv);
    if (Call.isInvalid())
      return ExprError();

    return ActOnCallExpr(S, Call.get(), LParenLoc, MultiExprArg(Args, NumArgs),
                         RParenLoc);
  }

  MarkDeclarationReferenced(LParenLoc, Best->Function);
  CheckMemberOperatorAccess(LParenLoc, Object.get(), 0, Best->FoundDecl);
  DiagnoseUseOfDecl(Best->FoundDecl, LParenLoc);

  // We found an overloaded operator(). Build a CXXOperatorCallExpr
  // that calls this method, using Object for the implicit object
  // parameter and passing along the remaining arguments.
  CXXMethodDecl *Method = cast<CXXMethodDecl>(Best->Function);
  const FunctionProtoType *Proto =
    Method->getType()->getAs<FunctionProtoType>();

  unsigned NumArgsInProto = Proto->getNumArgs();
  unsigned NumArgsToCheck = NumArgs;

  // Build the full argument list for the method call (the
  // implicit object parameter is placed at the beginning of the
  // list).
  Expr **MethodArgs;
  if (NumArgs < NumArgsInProto) {
    NumArgsToCheck = NumArgsInProto;
    MethodArgs = new Expr*[NumArgsInProto + 1];
  } else {
    MethodArgs = new Expr*[NumArgs + 1];
  }
  MethodArgs[0] = Object.get();
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    MethodArgs[ArgIdx + 1] = Args[ArgIdx];

  ExprResult NewFn = CreateFunctionRefExpr(*this, Method);
  if (NewFn.isInvalid())
    return true;

  // Once we've built TheCall, all of the expressions are properly
  // owned.
  QualType ResultTy = Method->getResultType();
  ExprValueKind VK = Expr::getValueKindForType(ResultTy);
  ResultTy = ResultTy.getNonLValueExprType(Context);

  CXXOperatorCallExpr *TheCall =
    new (Context) CXXOperatorCallExpr(Context, OO_Call, NewFn.take(),
                                      MethodArgs, NumArgs + 1,
                                      ResultTy, VK, RParenLoc);
  delete [] MethodArgs;

  if (CheckCallReturnType(Method->getResultType(), LParenLoc, TheCall,
                          Method))
    return true;

  // We may have default arguments. If so, we need to allocate more
  // slots in the call for them.
  if (NumArgs < NumArgsInProto)
    TheCall->setNumArgs(Context, NumArgsInProto + 1);
  else if (NumArgs > NumArgsInProto)
    NumArgsToCheck = NumArgsInProto;

  bool IsError = false;

  // Initialize the implicit object parameter.
  ExprResult ObjRes =
    PerformObjectArgumentInitialization(Object.get(), /*Qualifier=*/0,
                                        Best->FoundDecl, Method);
  if (ObjRes.isInvalid())
    IsError = true;
  else
    Object = move(ObjRes);
  TheCall->setArg(0, Object.take());

  // Check the argument types.
  for (unsigned i = 0; i != NumArgsToCheck; i++) {
    Expr *Arg;
    if (i < NumArgs) {
      Arg = Args[i];

      // Pass the argument.

      ExprResult InputInit
        = PerformCopyInitialization(InitializedEntity::InitializeParameter(
                                                    Context,
                                                    Method->getParamDecl(i)),
                                    SourceLocation(), Arg);

      IsError |= InputInit.isInvalid();
      Arg = InputInit.takeAs<Expr>();
    } else {
      ExprResult DefArg
        = BuildCXXDefaultArgExpr(LParenLoc, Method, Method->getParamDecl(i));
      if (DefArg.isInvalid()) {
        IsError = true;
        break;
      }

      Arg = DefArg.takeAs<Expr>();
    }

    TheCall->setArg(i + 1, Arg);
  }

  // If this is a variadic call, handle args passed through "...".
  if (Proto->isVariadic()) {
    // Promote the arguments (C99 6.5.2.2p7).
    for (unsigned i = NumArgsInProto; i != NumArgs; i++) {
      ExprResult Arg = DefaultVariadicArgumentPromotion(Args[i], VariadicMethod, 0);
      IsError |= Arg.isInvalid();
      TheCall->setArg(i + 1, Arg.take());
    }
  }

  if (IsError) return true;

  if (CheckFunctionCall(Method, TheCall))
    return true;

  return MaybeBindToTemporary(TheCall);
}

/// BuildOverloadedArrowExpr - Build a call to an overloaded @c operator->
///  (if one exists), where @c Base is an expression of class type and
/// @c Member is the name of the member we're trying to find.
ExprResult
Sema::BuildOverloadedArrowExpr(Scope *S, Expr *Base, SourceLocation OpLoc) {
  assert(Base->getType()->isRecordType() &&
         "left-hand side must have class type");

  if (Base->getObjectKind() == OK_ObjCProperty) {
    ExprResult Result = ConvertPropertyForRValue(Base);
    if (Result.isInvalid())
      return ExprError();
    Base = Result.take();
  }

  SourceLocation Loc = Base->getExprLoc();

  // C++ [over.ref]p1:
  //
  //   [...] An expression x->m is interpreted as (x.operator->())->m
  //   for a class object x of type T if T::operator->() exists and if
  //   the operator is selected as the best match function by the
  //   overload resolution mechanism (13.3).
  DeclarationName OpName =
    Context.DeclarationNames.getCXXOperatorName(OO_Arrow);
  OverloadCandidateSet CandidateSet(Loc);
  const RecordType *BaseRecord = Base->getType()->getAs<RecordType>();

  if (RequireCompleteType(Loc, Base->getType(),
                          PDiag(diag::err_typecheck_incomplete_tag)
                            << Base->getSourceRange()))
    return ExprError();

  LookupResult R(*this, OpName, OpLoc, LookupOrdinaryName);
  LookupQualifiedName(R, BaseRecord->getDecl());
  R.suppressDiagnostics();

  for (LookupResult::iterator Oper = R.begin(), OperEnd = R.end();
       Oper != OperEnd; ++Oper) {
    AddMethodCandidate(Oper.getPair(), Base->getType(), Base->Classify(Context),
                       0, 0, CandidateSet, /*SuppressUserConversions=*/false);
  }

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(*this, OpLoc, Best)) {
  case OR_Success:
    // Overload resolution succeeded; we'll build the call below.
    break;

  case OR_No_Viable_Function:
    if (CandidateSet.empty())
      Diag(OpLoc, diag::err_typecheck_member_reference_arrow)
        << Base->getType() << Base->getSourceRange();
    else
      Diag(OpLoc, diag::err_ovl_no_viable_oper)
        << "operator->" << Base->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, &Base, 1);
    return ExprError();

  case OR_Ambiguous:
    Diag(OpLoc,  diag::err_ovl_ambiguous_oper_unary)
      << "->" << Base->getType() << Base->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_ViableCandidates, &Base, 1);
    return ExprError();

  case OR_Deleted:
    Diag(OpLoc,  diag::err_ovl_deleted_oper)
      << Best->Function->isDeleted()
      << "->" 
      << getDeletedOrUnavailableSuffix(Best->Function)
      << Base->getSourceRange();
    CandidateSet.NoteCandidates(*this, OCD_AllCandidates, &Base, 1);
    return ExprError();
  }

  MarkDeclarationReferenced(OpLoc, Best->Function);
  CheckMemberOperatorAccess(OpLoc, Base, 0, Best->FoundDecl);
  DiagnoseUseOfDecl(Best->FoundDecl, OpLoc);

  // Convert the object parameter.
  CXXMethodDecl *Method = cast<CXXMethodDecl>(Best->Function);
  ExprResult BaseResult =
    PerformObjectArgumentInitialization(Base, /*Qualifier=*/0,
                                        Best->FoundDecl, Method);
  if (BaseResult.isInvalid())
    return ExprError();
  Base = BaseResult.take();

  // Build the operator call.
  ExprResult FnExpr = CreateFunctionRefExpr(*this, Method);
  if (FnExpr.isInvalid())
    return ExprError();

  QualType ResultTy = Method->getResultType();
  ExprValueKind VK = Expr::getValueKindForType(ResultTy);
  ResultTy = ResultTy.getNonLValueExprType(Context);
  CXXOperatorCallExpr *TheCall =
    new (Context) CXXOperatorCallExpr(Context, OO_Arrow, FnExpr.take(),
                                      &Base, 1, ResultTy, VK, OpLoc);

  if (CheckCallReturnType(Method->getResultType(), OpLoc, TheCall,
                          Method))
          return ExprError();

  return MaybeBindToTemporary(TheCall);
}

/// FixOverloadedFunctionReference - E is an expression that refers to
/// a C++ overloaded function (possibly with some parentheses and
/// perhaps a '&' around it). We have resolved the overloaded function
/// to the function declaration Fn, so patch up the expression E to
/// refer (possibly indirectly) to Fn. Returns the new expr.
Expr *Sema::FixOverloadedFunctionReference(Expr *E, DeclAccessPair Found,
                                           FunctionDecl *Fn) {
  if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    Expr *SubExpr = FixOverloadedFunctionReference(PE->getSubExpr(),
                                                   Found, Fn);
    if (SubExpr == PE->getSubExpr())
      return PE;

    return new (Context) ParenExpr(PE->getLParen(), PE->getRParen(), SubExpr);
  }

  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    Expr *SubExpr = FixOverloadedFunctionReference(ICE->getSubExpr(),
                                                   Found, Fn);
    assert(Context.hasSameType(ICE->getSubExpr()->getType(),
                               SubExpr->getType()) &&
           "Implicit cast type cannot be determined from overload");
    assert(ICE->path_empty() && "fixing up hierarchy conversion?");
    if (SubExpr == ICE->getSubExpr())
      return ICE;

    return ImplicitCastExpr::Create(Context, ICE->getType(),
                                    ICE->getCastKind(),
                                    SubExpr, 0,
                                    ICE->getValueKind());
  }

  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(E)) {
    assert(UnOp->getOpcode() == UO_AddrOf &&
           "Can only take the address of an overloaded function");
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
      if (Method->isStatic()) {
        // Do nothing: static member functions aren't any different
        // from non-member functions.
      } else {
        // Fix the sub expression, which really has to be an
        // UnresolvedLookupExpr holding an overloaded member function
        // or template.
        Expr *SubExpr = FixOverloadedFunctionReference(UnOp->getSubExpr(),
                                                       Found, Fn);
        if (SubExpr == UnOp->getSubExpr())
          return UnOp;

        assert(isa<DeclRefExpr>(SubExpr)
               && "fixed to something other than a decl ref");
        assert(cast<DeclRefExpr>(SubExpr)->getQualifier()
               && "fixed to a member ref with no nested name qualifier");

        // We have taken the address of a pointer to member
        // function. Perform the computation here so that we get the
        // appropriate pointer to member type.
        QualType ClassType
          = Context.getTypeDeclType(cast<RecordDecl>(Method->getDeclContext()));
        QualType MemPtrType
          = Context.getMemberPointerType(Fn->getType(), ClassType.getTypePtr());

        return new (Context) UnaryOperator(SubExpr, UO_AddrOf, MemPtrType,
                                           VK_RValue, OK_Ordinary,
                                           UnOp->getOperatorLoc());
      }
    }
    Expr *SubExpr = FixOverloadedFunctionReference(UnOp->getSubExpr(),
                                                   Found, Fn);
    if (SubExpr == UnOp->getSubExpr())
      return UnOp;

    return new (Context) UnaryOperator(SubExpr, UO_AddrOf,
                                     Context.getPointerType(SubExpr->getType()),
                                       VK_RValue, OK_Ordinary,
                                       UnOp->getOperatorLoc());
  }

  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(E)) {
    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (ULE->hasExplicitTemplateArgs()) {
      ULE->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    return DeclRefExpr::Create(Context,
                               ULE->getQualifierLoc(),
                               Fn,
                               ULE->getNameLoc(),
                               Fn->getType(),
                               VK_LValue,
                               Found.getDecl(),
                               TemplateArgs);
  }

  if (UnresolvedMemberExpr *MemExpr = dyn_cast<UnresolvedMemberExpr>(E)) {
    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (MemExpr->hasExplicitTemplateArgs()) {
      MemExpr->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    Expr *Base;

    // If we're filling in a static method where we used to have an
    // implicit member access, rewrite to a simple decl ref.
    if (MemExpr->isImplicitAccess()) {
      if (cast<CXXMethodDecl>(Fn)->isStatic()) {
        return DeclRefExpr::Create(Context,
                                   MemExpr->getQualifierLoc(),
                                   Fn,
                                   MemExpr->getMemberLoc(),
                                   Fn->getType(),
                                   VK_LValue,
                                   Found.getDecl(),
                                   TemplateArgs);
      } else {
        SourceLocation Loc = MemExpr->getMemberLoc();
        if (MemExpr->getQualifier())
          Loc = MemExpr->getQualifierLoc().getBeginLoc();
        Base = new (Context) CXXThisExpr(Loc,
                                         MemExpr->getBaseType(),
                                         /*isImplicit=*/true);
      }
    } else
      Base = MemExpr->getBase();

    ExprValueKind valueKind;
    QualType type;
    if (cast<CXXMethodDecl>(Fn)->isStatic()) {
      valueKind = VK_LValue;
      type = Fn->getType();
    } else {
      valueKind = VK_RValue;
      type = Context.BoundMemberTy;
    }

    return MemberExpr::Create(Context, Base,
                              MemExpr->isArrow(),
                              MemExpr->getQualifierLoc(),
                              Fn,
                              Found,
                              MemExpr->getMemberNameInfo(),
                              TemplateArgs,
                              type, valueKind, OK_Ordinary);
  }

  llvm_unreachable("Invalid reference to overloaded function");
  return E;
}

ExprResult Sema::FixOverloadedFunctionReference(ExprResult E,
                                                DeclAccessPair Found,
                                                FunctionDecl *Fn) {
  return Owned(FixOverloadedFunctionReference((Expr *)E.get(), Found, Fn));
}

} // end namespace clang
