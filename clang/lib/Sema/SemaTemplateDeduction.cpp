//===------- SemaTemplateDeduction.cpp - Template Argument Deduction ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template argument deduction.
//
//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include <algorithm>

namespace clang {
  /// \brief Various flags that control template argument deduction.
  ///
  /// These flags can be bitwise-OR'd together.
  enum TemplateDeductionFlags {
    /// \brief No template argument deduction flags, which indicates the
    /// strictest results for template argument deduction (as used for, e.g.,
    /// matching class template partial specializations).
    TDF_None = 0,
    /// \brief Within template argument deduction from a function call, we are
    /// matching with a parameter type for which the original parameter was
    /// a reference.
    TDF_ParamWithReferenceType = 0x1,
    /// \brief Within template argument deduction from a function call, we
    /// are matching in a case where we ignore cv-qualifiers.
    TDF_IgnoreQualifiers = 0x02,
    /// \brief Within template argument deduction from a function call,
    /// we are matching in a case where we can perform template argument
    /// deduction from a template-id of a derived class of the argument type.
    TDF_DerivedClass = 0x04,
    /// \brief Allow non-dependent types to differ, e.g., when performing
    /// template argument deduction from a function call where conversions
    /// may apply.
    TDF_SkipNonDependent = 0x08
  };
}

using namespace clang;

/// \brief Compare two APSInts, extending and switching the sign as
/// necessary to compare their values regardless of underlying type.
static bool hasSameExtendedValue(llvm::APSInt X, llvm::APSInt Y) {
  if (Y.getBitWidth() > X.getBitWidth())
    X.extend(Y.getBitWidth());
  else if (Y.getBitWidth() < X.getBitWidth())
    Y.extend(X.getBitWidth());

  // If there is a signedness mismatch, correct it.
  if (X.isSigned() != Y.isSigned()) {
    // If the signed value is negative, then the values cannot be the same.
    if ((Y.isSigned() && Y.isNegative()) || (X.isSigned() && X.isNegative()))
      return false;

    Y.setIsSigned(true);
    X.setIsSigned(true);
  }

  return X == Y;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        const TemplateArgument &Arg,
                        Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced);

/// \brief If the given expression is of a form that permits the deduction
/// of a non-type template parameter, return the declaration of that
/// non-type template parameter.
static NonTypeTemplateParmDecl *getDeducedParameterFromExpr(Expr *E) {
  if (ImplicitCastExpr *IC = dyn_cast<ImplicitCastExpr>(E))
    E = IC->getSubExpr();

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl());

  return 0;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given constant.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                              NonTypeTemplateParmDecl *NTTP,
                              llvm::APSInt Value, QualType ValueType,
                              bool DeducedFromArrayBound,
                              Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");

  if (Deduced[NTTP->getIndex()].isNull()) {
    Deduced[NTTP->getIndex()] = DeducedTemplateArgument(Value, ValueType,
                                                        DeducedFromArrayBound);
    return Sema::TDK_Success;
  }

  if (Deduced[NTTP->getIndex()].getKind() != TemplateArgument::Integral) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(Value, ValueType);
    return Sema::TDK_Inconsistent;    
  }

  // Extent the smaller of the two values.
  llvm::APSInt PrevValue = *Deduced[NTTP->getIndex()].getAsIntegral();
  if (!hasSameExtendedValue(PrevValue, Value)) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(Value, ValueType);
    return Sema::TDK_Inconsistent;
  }

  if (!DeducedFromArrayBound)
    Deduced[NTTP->getIndex()].setDeducedFromArrayBound(false);

  return Sema::TDK_Success;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given type- or value-dependent expression.
///
/// \returns true if deduction succeeded, false otherwise.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                              NonTypeTemplateParmDecl *NTTP,
                              Expr *Value,
                              Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");
  assert((Value->isTypeDependent() || Value->isValueDependent()) &&
         "Expression template argument must be type- or value-dependent.");

  if (Deduced[NTTP->getIndex()].isNull()) {
    Deduced[NTTP->getIndex()] = TemplateArgument(Value->Retain());
    return Sema::TDK_Success;
  }

  if (Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Integral) {
    // Okay, we deduced a constant in one case and a dependent expression
    // in another case. FIXME: Later, we will check that instantiating the
    // dependent expression gives us the constant value.
    return Sema::TDK_Success;
  }

  if (Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Expression) {
    // Compare the expressions for equality
    llvm::FoldingSetNodeID ID1, ID2;
    Deduced[NTTP->getIndex()].getAsExpr()->Profile(ID1, S.Context, true);
    Value->Profile(ID2, S.Context, true);
    if (ID1 == ID2)
      return Sema::TDK_Success;
   
    // FIXME: Fill in argument mismatch information
    return Sema::TDK_NonDeducedMismatch;
  }

  return Sema::TDK_Success;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given declaration.
///
/// \returns true if deduction succeeded, false otherwise.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                              NonTypeTemplateParmDecl *NTTP,
                              Decl *D,
                              Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");
  
  if (Deduced[NTTP->getIndex()].isNull()) {
    Deduced[NTTP->getIndex()] = TemplateArgument(D->getCanonicalDecl());
    return Sema::TDK_Success;
  }
  
  if (Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Expression) {
    // Okay, we deduced a declaration in one case and a dependent expression
    // in another case.
    return Sema::TDK_Success;
  }
  
  if (Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Declaration) {
    // Compare the declarations for equality
    if (Deduced[NTTP->getIndex()].getAsDecl()->getCanonicalDecl() ==
          D->getCanonicalDecl())
      return Sema::TDK_Success;
    
    // FIXME: Fill in argument mismatch information
    return Sema::TDK_NonDeducedMismatch;
  }
  
  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        TemplateName Param,
                        TemplateName Arg,
                        Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  TemplateDecl *ParamDecl = Param.getAsTemplateDecl();
  if (!ParamDecl) {
    // The parameter type is dependent and is not a template template parameter,
    // so there is nothing that we can deduce.
    return Sema::TDK_Success;
  }
  
  if (TemplateTemplateParmDecl *TempParam
        = dyn_cast<TemplateTemplateParmDecl>(ParamDecl)) {
    // Bind the template template parameter to the given template name.
    TemplateArgument &ExistingArg = Deduced[TempParam->getIndex()];
    if (ExistingArg.isNull()) {
      // This is the first deduction for this template template parameter.
      ExistingArg = TemplateArgument(S.Context.getCanonicalTemplateName(Arg));
      return Sema::TDK_Success;
    }
    
    // Verify that the previous binding matches this deduction.
    assert(ExistingArg.getKind() == TemplateArgument::Template);
    if (S.Context.hasSameTemplateName(ExistingArg.getAsTemplate(), Arg))
      return Sema::TDK_Success;
    
    // Inconsistent deduction.
    Info.Param = TempParam;
    Info.FirstArg = ExistingArg;
    Info.SecondArg = TemplateArgument(Arg);
    return Sema::TDK_Inconsistent;
  }
  
  // Verify that the two template names are equivalent.
  if (S.Context.hasSameTemplateName(Param, Arg))
    return Sema::TDK_Success;
  
  // Mismatch of non-dependent template parameter to argument.
  Info.FirstArg = TemplateArgument(Param);
  Info.SecondArg = TemplateArgument(Arg);
  return Sema::TDK_NonDeducedMismatch;
}

/// \brief Deduce the template arguments by comparing the template parameter
/// type (which is a template-id) with the template argument type.
///
/// \param S the Sema
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param Param the parameter type
///
/// \param Arg the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateSpecializationType *Param,
                        QualType Arg,
                        Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(Arg.isCanonical() && "Argument type must be canonical");

  // Check whether the template argument is a dependent template-id.
  if (const TemplateSpecializationType *SpecArg
        = dyn_cast<TemplateSpecializationType>(Arg)) {
    // Perform template argument deduction for the template name.
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(S, TemplateParams,
                                    Param->getTemplateName(),
                                    SpecArg->getTemplateName(),
                                    Info, Deduced))
      return Result;


    // Perform template argument deduction on each template
    // argument.
    unsigned NumArgs = std::min(SpecArg->getNumArgs(), Param->getNumArgs());
    for (unsigned I = 0; I != NumArgs; ++I)
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams,
                                      Param->getArg(I),
                                      SpecArg->getArg(I),
                                      Info, Deduced))
        return Result;

    return Sema::TDK_Success;
  }

  // If the argument type is a class template specialization, we
  // perform template argument deduction using its template
  // arguments.
  const RecordType *RecordArg = dyn_cast<RecordType>(Arg);
  if (!RecordArg)
    return Sema::TDK_NonDeducedMismatch;

  ClassTemplateSpecializationDecl *SpecArg
    = dyn_cast<ClassTemplateSpecializationDecl>(RecordArg->getDecl());
  if (!SpecArg)
    return Sema::TDK_NonDeducedMismatch;

  // Perform template argument deduction for the template name.
  if (Sema::TemplateDeductionResult Result
        = DeduceTemplateArguments(S,
                                  TemplateParams,
                                  Param->getTemplateName(),
                               TemplateName(SpecArg->getSpecializedTemplate()),
                                  Info, Deduced))
    return Result;

  unsigned NumArgs = Param->getNumArgs();
  const TemplateArgumentList &ArgArgs = SpecArg->getTemplateArgs();
  if (NumArgs != ArgArgs.size())
    return Sema::TDK_NonDeducedMismatch;

  for (unsigned I = 0; I != NumArgs; ++I)
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(S, TemplateParams,
                                    Param->getArg(I),
                                    ArgArgs.get(I),
                                    Info, Deduced))
      return Result;

  return Sema::TDK_Success;
}

/// \brief Deduce the template arguments by comparing the parameter type and
/// the argument type (C++ [temp.deduct.type]).
///
/// \param S the semantic analysis object within which we are deducing
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param ParamIn the parameter type
///
/// \param ArgIn the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \param TDF bitwise OR of the TemplateDeductionFlags bits that describe
/// how template argument deduction is performed.
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        QualType ParamIn, QualType ArgIn,
                        Sema::TemplateDeductionInfo &Info,
                     llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                        unsigned TDF) {
  // We only want to look at the canonical types, since typedefs and
  // sugar are not part of template argument deduction.
  QualType Param = S.Context.getCanonicalType(ParamIn);
  QualType Arg = S.Context.getCanonicalType(ArgIn);

  // C++0x [temp.deduct.call]p4 bullet 1:
  //   - If the original P is a reference type, the deduced A (i.e., the type
  //     referred to by the reference) can be more cv-qualified than the
  //     transformed A.
  if (TDF & TDF_ParamWithReferenceType) {
    Qualifiers Quals;
    QualType UnqualParam = S.Context.getUnqualifiedArrayType(Param, Quals);
    Quals.setCVRQualifiers(Quals.getCVRQualifiers() &
                           Arg.getCVRQualifiersThroughArrayTypes());
    Param = S.Context.getQualifiedType(UnqualParam, Quals);
  }

  // If the parameter type is not dependent, there is nothing to deduce.
  if (!Param->isDependentType()) {
    if (!(TDF & TDF_SkipNonDependent) && Param != Arg) {
      
      return Sema::TDK_NonDeducedMismatch;
    }
    
    return Sema::TDK_Success;
  }

  // C++ [temp.deduct.type]p9:
  //   A template type argument T, a template template argument TT or a
  //   template non-type argument i can be deduced if P and A have one of
  //   the following forms:
  //
  //     T
  //     cv-list T
  if (const TemplateTypeParmType *TemplateTypeParm
        = Param->getAs<TemplateTypeParmType>()) {
    unsigned Index = TemplateTypeParm->getIndex();
    bool RecanonicalizeArg = false;

    // If the argument type is an array type, move the qualifiers up to the
    // top level, so they can be matched with the qualifiers on the parameter.
    // FIXME: address spaces, ObjC GC qualifiers
    if (isa<ArrayType>(Arg)) {
      Qualifiers Quals;
      Arg = S.Context.getUnqualifiedArrayType(Arg, Quals);
      if (Quals) {
        Arg = S.Context.getQualifiedType(Arg, Quals);
        RecanonicalizeArg = true;
      }
    }

    // The argument type can not be less qualified than the parameter
    // type.
    if (Param.isMoreQualifiedThan(Arg) && !(TDF & TDF_IgnoreQualifiers)) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = Deduced[Index];
      Info.SecondArg = TemplateArgument(Arg);
      return Sema::TDK_InconsistentQuals;
    }

    assert(TemplateTypeParm->getDepth() == 0 && "Can't deduce with depth > 0");
    assert(Arg != S.Context.OverloadTy && "Unresolved overloaded function");
    QualType DeducedType = Arg;
    DeducedType.removeCVRQualifiers(Param.getCVRQualifiers());
    if (RecanonicalizeArg)
      DeducedType = S.Context.getCanonicalType(DeducedType);

    if (Deduced[Index].isNull())
      Deduced[Index] = TemplateArgument(DeducedType);
    else {
      // C++ [temp.deduct.type]p2:
      //   [...] If type deduction cannot be done for any P/A pair, or if for
      //   any pair the deduction leads to more than one possible set of
      //   deduced values, or if different pairs yield different deduced
      //   values, or if any template argument remains neither deduced nor
      //   explicitly specified, template argument deduction fails.
      if (Deduced[Index].getAsType() != DeducedType) {
        Info.Param
          = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
        Info.FirstArg = Deduced[Index];
        Info.SecondArg = TemplateArgument(Arg);
        return Sema::TDK_Inconsistent;
      }
    }
    return Sema::TDK_Success;
  }

  // Set up the template argument deduction information for a failure.
  Info.FirstArg = TemplateArgument(ParamIn);
  Info.SecondArg = TemplateArgument(ArgIn);

  // Check the cv-qualifiers on the parameter and argument types.
  if (!(TDF & TDF_IgnoreQualifiers)) {
    if (TDF & TDF_ParamWithReferenceType) {
      if (Param.isMoreQualifiedThan(Arg))
        return Sema::TDK_NonDeducedMismatch;
    } else {
      if (Param.getCVRQualifiers() != Arg.getCVRQualifiers())
        return Sema::TDK_NonDeducedMismatch;
    }
  }

  switch (Param->getTypeClass()) {
    // No deduction possible for these types
    case Type::Builtin:
      return Sema::TDK_NonDeducedMismatch;

    //     T *
    case Type::Pointer: {
      const PointerType *PointerArg = Arg->getAs<PointerType>();
      if (!PointerArg)
        return Sema::TDK_NonDeducedMismatch;

      unsigned SubTDF = TDF & (TDF_IgnoreQualifiers | TDF_DerivedClass);
      return DeduceTemplateArguments(S, TemplateParams,
                                   cast<PointerType>(Param)->getPointeeType(),
                                     PointerArg->getPointeeType(),
                                     Info, Deduced, SubTDF);
    }

    //     T &
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg = Arg->getAs<LValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(S, TemplateParams,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced, 0);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg = Arg->getAs<RValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(S, TemplateParams,
                           cast<RValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced, 0);
    }

    //     T [] (implied, but not stated explicitly)
    case Type::IncompleteArray: {
      const IncompleteArrayType *IncompleteArrayArg =
        S.Context.getAsIncompleteArrayType(Arg);
      if (!IncompleteArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(S, TemplateParams,
                     S.Context.getAsIncompleteArrayType(Param)->getElementType(),
                                     IncompleteArrayArg->getElementType(),
                                     Info, Deduced, 0);
    }

    //     T [integer-constant]
    case Type::ConstantArray: {
      const ConstantArrayType *ConstantArrayArg =
        S.Context.getAsConstantArrayType(Arg);
      if (!ConstantArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      const ConstantArrayType *ConstantArrayParm =
        S.Context.getAsConstantArrayType(Param);
      if (ConstantArrayArg->getSize() != ConstantArrayParm->getSize())
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(S, TemplateParams,
                                     ConstantArrayParm->getElementType(),
                                     ConstantArrayArg->getElementType(),
                                     Info, Deduced, 0);
    }

    //     type [i]
    case Type::DependentSizedArray: {
      const ArrayType *ArrayArg = S.Context.getAsArrayType(Arg);
      if (!ArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      // Check the element type of the arrays
      const DependentSizedArrayType *DependentArrayParm
        = S.Context.getAsDependentSizedArrayType(Param);
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams,
                                      DependentArrayParm->getElementType(),
                                      ArrayArg->getElementType(),
                                      Info, Deduced, 0))
        return Result;

      // Determine the array bound is something we can deduce.
      NonTypeTemplateParmDecl *NTTP
        = getDeducedParameterFromExpr(DependentArrayParm->getSizeExpr());
      if (!NTTP)
        return Sema::TDK_Success;

      // We can perform template argument deduction for the given non-type
      // template parameter.
      assert(NTTP->getDepth() == 0 &&
             "Cannot deduce non-type template argument at depth > 0");
      if (const ConstantArrayType *ConstantArrayArg
            = dyn_cast<ConstantArrayType>(ArrayArg)) {
        llvm::APSInt Size(ConstantArrayArg->getSize());
        return DeduceNonTypeTemplateArgument(S, NTTP, Size, 
                                             S.Context.getSizeType(),
                                             /*ArrayBound=*/true,
                                             Info, Deduced);
      }
      if (const DependentSizedArrayType *DependentArrayArg
            = dyn_cast<DependentSizedArrayType>(ArrayArg))
        return DeduceNonTypeTemplateArgument(S, NTTP,
                                             DependentArrayArg->getSizeExpr(),
                                             Info, Deduced);

      // Incomplete type does not match a dependently-sized array type
      return Sema::TDK_NonDeducedMismatch;
    }

    //     type(*)(T)
    //     T(*)()
    //     T(*)(T)
    case Type::FunctionProto: {
      const FunctionProtoType *FunctionProtoArg =
        dyn_cast<FunctionProtoType>(Arg);
      if (!FunctionProtoArg)
        return Sema::TDK_NonDeducedMismatch;

      const FunctionProtoType *FunctionProtoParam =
        cast<FunctionProtoType>(Param);

      if (FunctionProtoParam->getTypeQuals() !=
          FunctionProtoArg->getTypeQuals())
        return Sema::TDK_NonDeducedMismatch;

      if (FunctionProtoParam->getNumArgs() != FunctionProtoArg->getNumArgs())
        return Sema::TDK_NonDeducedMismatch;

      if (FunctionProtoParam->isVariadic() != FunctionProtoArg->isVariadic())
        return Sema::TDK_NonDeducedMismatch;

      // Check return types.
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams,
                                      FunctionProtoParam->getResultType(),
                                      FunctionProtoArg->getResultType(),
                                      Info, Deduced, 0))
        return Result;

      for (unsigned I = 0, N = FunctionProtoParam->getNumArgs(); I != N; ++I) {
        // Check argument types.
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArguments(S, TemplateParams,
                                        FunctionProtoParam->getArgType(I),
                                        FunctionProtoArg->getArgType(I),
                                        Info, Deduced, 0))
          return Result;
      }

      return Sema::TDK_Success;
    }

    case Type::InjectedClassName: {
      // Treat a template's injected-class-name as if the template
      // specialization type had been used.
      Param = cast<InjectedClassNameType>(Param)->getUnderlyingType();
      assert(isa<TemplateSpecializationType>(Param) &&
             "injected class name is not a template specialization type");
      // fall through
    }

    //     template-name<T> (where template-name refers to a class template)
    //     template-name<i>
    //     TT<T>
    //     TT<i>
    //     TT<>
    case Type::TemplateSpecialization: {
      const TemplateSpecializationType *SpecParam
        = cast<TemplateSpecializationType>(Param);

      // Try to deduce template arguments from the template-id.
      Sema::TemplateDeductionResult Result
        = DeduceTemplateArguments(S, TemplateParams, SpecParam, Arg,
                                  Info, Deduced);

      if (Result && (TDF & TDF_DerivedClass)) {
        // C++ [temp.deduct.call]p3b3:
        //   If P is a class, and P has the form template-id, then A can be a
        //   derived class of the deduced A. Likewise, if P is a pointer to a
        //   class of the form template-id, A can be a pointer to a derived
        //   class pointed to by the deduced A.
        //
        // More importantly:
        //   These alternatives are considered only if type deduction would
        //   otherwise fail.
        if (const RecordType *RecordT = Arg->getAs<RecordType>()) {
          // We cannot inspect base classes as part of deduction when the type
          // is incomplete, so either instantiate any templates necessary to
          // complete the type, or skip over it if it cannot be completed.
          if (S.RequireCompleteType(Info.getLocation(), Arg, 0))
            return Result;

          // Use data recursion to crawl through the list of base classes.
          // Visited contains the set of nodes we have already visited, while
          // ToVisit is our stack of records that we still need to visit.
          llvm::SmallPtrSet<const RecordType *, 8> Visited;
          llvm::SmallVector<const RecordType *, 8> ToVisit;
          ToVisit.push_back(RecordT);
          bool Successful = false;
          while (!ToVisit.empty()) {
            // Retrieve the next class in the inheritance hierarchy.
            const RecordType *NextT = ToVisit.back();
            ToVisit.pop_back();

            // If we have already seen this type, skip it.
            if (!Visited.insert(NextT))
              continue;

            // If this is a base class, try to perform template argument
            // deduction from it.
            if (NextT != RecordT) {
              Sema::TemplateDeductionResult BaseResult
                = DeduceTemplateArguments(S, TemplateParams, SpecParam,
                                          QualType(NextT, 0), Info, Deduced);

              // If template argument deduction for this base was successful,
              // note that we had some success.
              if (BaseResult == Sema::TDK_Success)
                Successful = true;
            }

            // Visit base classes
            CXXRecordDecl *Next = cast<CXXRecordDecl>(NextT->getDecl());
            for (CXXRecordDecl::base_class_iterator Base = Next->bases_begin(),
                                                 BaseEnd = Next->bases_end();
                 Base != BaseEnd; ++Base) {
              assert(Base->getType()->isRecordType() &&
                     "Base class that isn't a record?");
              ToVisit.push_back(Base->getType()->getAs<RecordType>());
            }
          }

          if (Successful)
            return Sema::TDK_Success;
        }

      }

      return Result;
    }

    //     T type::*
    //     T T::*
    //     T (type::*)()
    //     type (T::*)()
    //     type (type::*)(T)
    //     type (T::*)(T)
    //     T (type::*)(T)
    //     T (T::*)()
    //     T (T::*)(T)
    case Type::MemberPointer: {
      const MemberPointerType *MemPtrParam = cast<MemberPointerType>(Param);
      const MemberPointerType *MemPtrArg = dyn_cast<MemberPointerType>(Arg);
      if (!MemPtrArg)
        return Sema::TDK_NonDeducedMismatch;

      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams,
                                      MemPtrParam->getPointeeType(),
                                      MemPtrArg->getPointeeType(),
                                      Info, Deduced,
                                      TDF & TDF_IgnoreQualifiers))
        return Result;

      return DeduceTemplateArguments(S, TemplateParams,
                                     QualType(MemPtrParam->getClass(), 0),
                                     QualType(MemPtrArg->getClass(), 0),
                                     Info, Deduced, 0);
    }

    //     (clang extension)
    //
    //     type(^)(T)
    //     T(^)()
    //     T(^)(T)
    case Type::BlockPointer: {
      const BlockPointerType *BlockPtrParam = cast<BlockPointerType>(Param);
      const BlockPointerType *BlockPtrArg = dyn_cast<BlockPointerType>(Arg);

      if (!BlockPtrArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(S, TemplateParams,
                                     BlockPtrParam->getPointeeType(),
                                     BlockPtrArg->getPointeeType(), Info,
                                     Deduced, 0);
    }

    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::DependentName:
      // No template argument deduction for these types
      return Sema::TDK_Success;

    default:
      break;
  }

  // FIXME: Many more cases to go (to go).
  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        const TemplateArgument &Arg,
                        Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  switch (Param.getKind()) {
  case TemplateArgument::Null:
    assert(false && "Null template argument in parameter list");
    break;

  case TemplateArgument::Type:
    if (Arg.getKind() == TemplateArgument::Type)
      return DeduceTemplateArguments(S, TemplateParams, Param.getAsType(),
                                     Arg.getAsType(), Info, Deduced, 0);
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;
      
  case TemplateArgument::Template:
    if (Arg.getKind() == TemplateArgument::Template)
      return DeduceTemplateArguments(S, TemplateParams, 
                                     Param.getAsTemplate(),
                                     Arg.getAsTemplate(), Info, Deduced);
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;
      
  case TemplateArgument::Declaration:
    if (Arg.getKind() == TemplateArgument::Declaration &&
        Param.getAsDecl()->getCanonicalDecl() ==
          Arg.getAsDecl()->getCanonicalDecl())
      return Sema::TDK_Success;
      
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::Integral:
    if (Arg.getKind() == TemplateArgument::Integral) {
      if (hasSameExtendedValue(*Param.getAsIntegral(), *Arg.getAsIntegral()))
        return Sema::TDK_Success;

      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    if (Arg.getKind() == TemplateArgument::Expression) {
      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    assert(false && "Type/value mismatch");
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::Expression: {
    if (NonTypeTemplateParmDecl *NTTP
          = getDeducedParameterFromExpr(Param.getAsExpr())) {
      if (Arg.getKind() == TemplateArgument::Integral)
        return DeduceNonTypeTemplateArgument(S, NTTP,
                                             *Arg.getAsIntegral(),
                                             Arg.getIntegralType(),
                                             /*ArrayBound=*/false,
                                             Info, Deduced);
      if (Arg.getKind() == TemplateArgument::Expression)
        return DeduceNonTypeTemplateArgument(S, NTTP, Arg.getAsExpr(),
                                             Info, Deduced);
      if (Arg.getKind() == TemplateArgument::Declaration)
        return DeduceNonTypeTemplateArgument(S, NTTP, Arg.getAsDecl(),
                                             Info, Deduced);
      
      assert(false && "Type/value mismatch");
      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    // Can't deduce anything, but that's okay.
    return Sema::TDK_Success;
  }
  case TemplateArgument::Pack:
    assert(0 && "FIXME: Implement!");
    break;
  }

  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgumentList &ParamList,
                        const TemplateArgumentList &ArgList,
                        Sema::TemplateDeductionInfo &Info,
                    llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(ParamList.size() == ArgList.size());
  for (unsigned I = 0, N = ParamList.size(); I != N; ++I) {
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(S, TemplateParams,
                                    ParamList[I], ArgList[I],
                                    Info, Deduced))
      return Result;
  }
  return Sema::TDK_Success;
}

/// \brief Determine whether two template arguments are the same.
static bool isSameTemplateArg(ASTContext &Context,
                              const TemplateArgument &X,
                              const TemplateArgument &Y) {
  if (X.getKind() != Y.getKind())
    return false;

  switch (X.getKind()) {
    case TemplateArgument::Null:
      assert(false && "Comparing NULL template argument");
      break;

    case TemplateArgument::Type:
      return Context.getCanonicalType(X.getAsType()) ==
             Context.getCanonicalType(Y.getAsType());

    case TemplateArgument::Declaration:
      return X.getAsDecl()->getCanonicalDecl() ==
             Y.getAsDecl()->getCanonicalDecl();

    case TemplateArgument::Template:
      return Context.getCanonicalTemplateName(X.getAsTemplate())
               .getAsVoidPointer() ==
             Context.getCanonicalTemplateName(Y.getAsTemplate())
               .getAsVoidPointer();
      
    case TemplateArgument::Integral:
      return *X.getAsIntegral() == *Y.getAsIntegral();

    case TemplateArgument::Expression: {
      llvm::FoldingSetNodeID XID, YID;
      X.getAsExpr()->Profile(XID, Context, true);
      Y.getAsExpr()->Profile(YID, Context, true);      
      return XID == YID;
    }

    case TemplateArgument::Pack:
      if (X.pack_size() != Y.pack_size())
        return false;

      for (TemplateArgument::pack_iterator XP = X.pack_begin(),
                                        XPEnd = X.pack_end(),
                                           YP = Y.pack_begin();
           XP != XPEnd; ++XP, ++YP)
        if (!isSameTemplateArg(Context, *XP, *YP))
          return false;

      return true;
  }

  return false;
}

/// \brief Helper function to build a TemplateParameter when we don't
/// know its type statically.
static TemplateParameter makeTemplateParameter(Decl *D) {
  if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(D))
    return TemplateParameter(TTP);
  else if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D))
    return TemplateParameter(NTTP);

  return TemplateParameter(cast<TemplateTemplateParmDecl>(D));
}

/// \brief Perform template argument deduction to determine whether
/// the given template arguments match the given class template
/// partial specialization per C++ [temp.class.spec.match].
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                              const TemplateArgumentList &TemplateArgs,
                              TemplateDeductionInfo &Info) {
  // C++ [temp.class.spec.match]p2:
  //   A partial specialization matches a given actual template
  //   argument list if the template arguments of the partial
  //   specialization can be deduced from the actual template argument
  //   list (14.8.2).
  SFINAETrap Trap(*this);
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(*this,
                                    Partial->getTemplateParameters(),
                                    Partial->getTemplateArgs(),
                                    TemplateArgs, Info, Deduced))
    return Result;

  InstantiatingTemplate Inst(*this, Partial->getLocation(), Partial,
                             Deduced.data(), Deduced.size());
  if (Inst)
    return TDK_InstantiationDepth;

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  TemplateArgumentListBuilder Builder(Partial->getTemplateParameters(),
                                      Deduced.size());
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    if (Deduced[I].isNull()) {
      Decl *Param
        = const_cast<NamedDecl *>(
                                Partial->getTemplateParameters()->getParam(I));
      Info.Param = makeTemplateParameter(Param);
      return TDK_Incomplete;
    }

    Builder.Append(Deduced[I]);
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList
    = new (Context) TemplateArgumentList(Context, Builder, /*TakeArgs=*/true);
  Info.reset(DeducedArgumentList);

  // Substitute the deduced template arguments into the template
  // arguments of the class template partial specialization, and
  // verify that the instantiated template arguments are both valid
  // and are equivalent to the template arguments originally provided
  // to the class template.
  // FIXME: Do we have to correct the types of deduced non-type template 
  // arguments (in particular, integral non-type template arguments?).
  Sema::LocalInstantiationScope InstScope(*this);
  ClassTemplateDecl *ClassTemplate = Partial->getSpecializedTemplate();
  const TemplateArgumentLoc *PartialTemplateArgs
    = Partial->getTemplateArgsAsWritten();
  unsigned N = Partial->getNumTemplateArgsAsWritten();

  // Note that we don't provide the langle and rangle locations.
  TemplateArgumentListInfo InstArgs;

  for (unsigned I = 0; I != N; ++I) {
    Decl *Param = const_cast<NamedDecl *>(
                    ClassTemplate->getTemplateParameters()->getParam(I));
    TemplateArgumentLoc InstArg;
    if (Subst(PartialTemplateArgs[I], InstArg,
              MultiLevelTemplateArgumentList(*DeducedArgumentList))) {
      Info.Param = makeTemplateParameter(Param);
      Info.FirstArg = PartialTemplateArgs[I].getArgument();
      return TDK_SubstitutionFailure;
    }
    InstArgs.addArgument(InstArg);
  }

  TemplateArgumentListBuilder ConvertedInstArgs(
                                  ClassTemplate->getTemplateParameters(), N);

  if (CheckTemplateArgumentList(ClassTemplate, Partial->getLocation(),
                                InstArgs, false, ConvertedInstArgs)) {
    // FIXME: fail with more useful information?
    return TDK_SubstitutionFailure;
  }
  
  for (unsigned I = 0, E = ConvertedInstArgs.flatSize(); I != E; ++I) {
    TemplateArgument InstArg = ConvertedInstArgs.getFlatArguments()[I];

    Decl *Param = const_cast<NamedDecl *>(
                    ClassTemplate->getTemplateParameters()->getParam(I));

    if (InstArg.getKind() == TemplateArgument::Expression) {
      // When the argument is an expression, check the expression result
      // against the actual template parameter to get down to the canonical
      // template argument.
      Expr *InstExpr = InstArg.getAsExpr();
      if (NonTypeTemplateParmDecl *NTTP
            = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        if (CheckTemplateArgument(NTTP, NTTP->getType(), InstExpr, InstArg)) {
          Info.Param = makeTemplateParameter(Param);
          Info.FirstArg = Partial->getTemplateArgs()[I];
          return TDK_SubstitutionFailure;
        }
      }
    }

    if (!isSameTemplateArg(Context, TemplateArgs[I], InstArg)) {
      Info.Param = makeTemplateParameter(Param);
      Info.FirstArg = TemplateArgs[I];
      Info.SecondArg = InstArg;
      return TDK_NonDeducedMismatch;
    }
  }

  if (Trap.hasErrorOccurred())
    return TDK_SubstitutionFailure;

  return TDK_Success;
}

/// \brief Determine whether the given type T is a simple-template-id type.
static bool isSimpleTemplateIdType(QualType T) {
  if (const TemplateSpecializationType *Spec
        = T->getAs<TemplateSpecializationType>())
    return Spec->getTemplateName().getAsTemplateDecl() != 0;

  return false;
}

/// \brief Substitute the explicitly-provided template arguments into the
/// given function template according to C++ [temp.arg.explicit].
///
/// \param FunctionTemplate the function template into which the explicit
/// template arguments will be substituted.
///
/// \param ExplicitTemplateArguments the explicitly-specified template
/// arguments.
///
/// \param Deduced the deduced template arguments, which will be populated
/// with the converted and checked explicit template arguments.
///
/// \param ParamTypes will be populated with the instantiated function
/// parameters.
///
/// \param FunctionType if non-NULL, the result type of the function template
/// will also be instantiated and the pointed-to value will be updated with
/// the instantiated function type.
///
/// \param Info if substitution fails for any reason, this object will be
/// populated with more information about the failure.
///
/// \returns TDK_Success if substitution was successful, or some failure
/// condition.
Sema::TemplateDeductionResult
Sema::SubstituteExplicitTemplateArguments(
                                      FunctionTemplateDecl *FunctionTemplate,
                        const TemplateArgumentListInfo &ExplicitTemplateArgs,
                       llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                 llvm::SmallVectorImpl<QualType> &ParamTypes,
                                          QualType *FunctionType,
                                          TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  if (ExplicitTemplateArgs.size() == 0) {
    // No arguments to substitute; just copy over the parameter types and
    // fill in the function type.
    for (FunctionDecl::param_iterator P = Function->param_begin(),
                                   PEnd = Function->param_end();
         P != PEnd;
         ++P)
      ParamTypes.push_back((*P)->getType());

    if (FunctionType)
      *FunctionType = Function->getType();
    return TDK_Success;
  }

  // Substitution of the explicit template arguments into a function template
  /// is a SFINAE context. Trap any errors that might occur.
  SFINAETrap Trap(*this);

  // C++ [temp.arg.explicit]p3:
  //   Template arguments that are present shall be specified in the
  //   declaration order of their corresponding template-parameters. The
  //   template argument list shall not specify more template-arguments than
  //   there are corresponding template-parameters.
  TemplateArgumentListBuilder Builder(TemplateParams,
                                      ExplicitTemplateArgs.size());

  // Enter a new template instantiation context where we check the
  // explicitly-specified template arguments against this function template,
  // and then substitute them into the function parameter types.
  InstantiatingTemplate Inst(*this, FunctionTemplate->getLocation(),
                             FunctionTemplate, Deduced.data(), Deduced.size(),
           ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution);
  if (Inst)
    return TDK_InstantiationDepth;

  if (CheckTemplateArgumentList(FunctionTemplate,
                                SourceLocation(),
                                ExplicitTemplateArgs,
                                true,
                                Builder) || Trap.hasErrorOccurred())
    return TDK_InvalidExplicitArguments;

  // Form the template argument list from the explicitly-specified
  // template arguments.
  TemplateArgumentList *ExplicitArgumentList
    = new (Context) TemplateArgumentList(Context, Builder, /*TakeArgs=*/true);
  Info.reset(ExplicitArgumentList);

  // Instantiate the types of each of the function parameters given the
  // explicitly-specified template arguments.
  for (FunctionDecl::param_iterator P = Function->param_begin(),
                                PEnd = Function->param_end();
       P != PEnd;
       ++P) {
    QualType ParamType
      = SubstType((*P)->getType(),
                  MultiLevelTemplateArgumentList(*ExplicitArgumentList),
                  (*P)->getLocation(), (*P)->getDeclName());
    if (ParamType.isNull() || Trap.hasErrorOccurred())
      return TDK_SubstitutionFailure;

    ParamTypes.push_back(ParamType);
  }

  // If the caller wants a full function type back, instantiate the return
  // type and form that function type.
  if (FunctionType) {
    // FIXME: exception-specifications?
    const FunctionProtoType *Proto
      = Function->getType()->getAs<FunctionProtoType>();
    assert(Proto && "Function template does not have a prototype?");

    QualType ResultType
      = SubstType(Proto->getResultType(),
                  MultiLevelTemplateArgumentList(*ExplicitArgumentList),
                  Function->getTypeSpecStartLoc(),
                  Function->getDeclName());
    if (ResultType.isNull() || Trap.hasErrorOccurred())
      return TDK_SubstitutionFailure;

    *FunctionType = BuildFunctionType(ResultType,
                                      ParamTypes.data(), ParamTypes.size(),
                                      Proto->isVariadic(),
                                      Proto->getTypeQuals(),
                                      Function->getLocation(),
                                      Function->getDeclName());
    if (FunctionType->isNull() || Trap.hasErrorOccurred())
      return TDK_SubstitutionFailure;
  }

  // C++ [temp.arg.explicit]p2:
  //   Trailing template arguments that can be deduced (14.8.2) may be
  //   omitted from the list of explicit template-arguments. If all of the
  //   template arguments can be deduced, they may all be omitted; in this
  //   case, the empty template argument list <> itself may also be omitted.
  //
  // Take all of the explicitly-specified arguments and put them into the
  // set of deduced template arguments.
  Deduced.reserve(TemplateParams->size());
  for (unsigned I = 0, N = ExplicitArgumentList->size(); I != N; ++I)
    Deduced.push_back(ExplicitArgumentList->get(I));

  return TDK_Success;
}

/// \brief Allocate a TemplateArgumentLoc where all locations have
/// been initialized to the given location.
///
/// \param S The semantic analysis object.
///
/// \param The template argument we are producing template argument
/// location information for.
///
/// \param NTTPType For a declaration template argument, the type of
/// the non-type template parameter that corresponds to this template
/// argument.
///
/// \param Loc The source location to use for the resulting template
/// argument.
static TemplateArgumentLoc 
getTrivialTemplateArgumentLoc(Sema &S,
                              const TemplateArgument &Arg, 
                              QualType NTTPType,
                              SourceLocation Loc) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Can't get a NULL template argument here");
    break;

  case TemplateArgument::Type:
    return TemplateArgumentLoc(Arg, 
                    S.Context.getTrivialTypeSourceInfo(Arg.getAsType(), Loc));

  case TemplateArgument::Declaration: {
    Expr *E
      = S.BuildExpressionFromDeclTemplateArgument(Arg, NTTPType, Loc)
                                                              .takeAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(E), E);
  }

  case TemplateArgument::Integral: {
    Expr *E
      = S.BuildExpressionFromIntegralTemplateArgument(Arg, Loc).takeAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(E), E);
  }

  case TemplateArgument::Template:
    return TemplateArgumentLoc(Arg, SourceRange(), Loc);

  case TemplateArgument::Expression:
    return TemplateArgumentLoc(Arg, Arg.getAsExpr());

  case TemplateArgument::Pack:
    llvm_unreachable("Template parameter packs are not yet supported");
  }

  return TemplateArgumentLoc();
}

/// \brief Finish template argument deduction for a function template,
/// checking the deduced template arguments for completeness and forming
/// the function template specialization.
Sema::TemplateDeductionResult
Sema::FinishTemplateArgumentDeduction(FunctionTemplateDecl *FunctionTemplate,
                       llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                      unsigned NumExplicitlySpecified,
                                      FunctionDecl *&Specialization,
                                      TemplateDeductionInfo &Info) {
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  // Template argument deduction for function templates in a SFINAE context.
  // Trap any errors that might occur.
  SFINAETrap Trap(*this);

  // Enter a new template instantiation context while we instantiate the
  // actual function declaration.
  InstantiatingTemplate Inst(*this, FunctionTemplate->getLocation(),
                             FunctionTemplate, Deduced.data(), Deduced.size(),
              ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution);
  if (Inst)
    return TDK_InstantiationDepth;

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  TemplateArgumentListBuilder Builder(TemplateParams, Deduced.size());
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    NamedDecl *Param = FunctionTemplate->getTemplateParameters()->getParam(I);
    if (!Deduced[I].isNull()) {
      if (I < NumExplicitlySpecified || 
          Deduced[I].getKind() == TemplateArgument::Type) {
        // We have already fully type-checked and converted this
        // argument (because it was explicitly-specified) or no
        // additional checking is necessary (because it's a template
        // type parameter). Just record the presence of this
        // parameter.
        Builder.Append(Deduced[I]);
        continue;
      }

      // We have deduced this argument, so it still needs to be
      // checked and converted.

      // First, for a non-type template parameter type that is
      // initialized by a declaration, we need the type of the
      // corresponding non-type template parameter.
      QualType NTTPType;
      if (NonTypeTemplateParmDecl *NTTP 
                                = dyn_cast<NonTypeTemplateParmDecl>(Param)) { 
        if (Deduced[I].getKind() == TemplateArgument::Declaration) {
          NTTPType = NTTP->getType();
          if (NTTPType->isDependentType()) {
            TemplateArgumentList TemplateArgs(Context, Builder, 
                                              /*TakeArgs=*/false);
            NTTPType = SubstType(NTTPType,
                                 MultiLevelTemplateArgumentList(TemplateArgs),
                                 NTTP->getLocation(),
                                 NTTP->getDeclName());
            if (NTTPType.isNull()) {
              Info.Param = makeTemplateParameter(Param);
              return TDK_SubstitutionFailure;
            }
          }
        }
      }

      // Convert the deduced template argument into a template
      // argument that we can check, almost as if the user had written
      // the template argument explicitly.
      TemplateArgumentLoc Arg = getTrivialTemplateArgumentLoc(*this,
                                                              Deduced[I],
                                                              NTTPType,
                                                           SourceLocation());

      // Check the template argument, converting it as necessary.
      if (CheckTemplateArgument(Param, Arg,
                                FunctionTemplate,
                                FunctionTemplate->getLocation(),
                                FunctionTemplate->getSourceRange().getEnd(),
                                Builder,
                                Deduced[I].wasDeducedFromArrayBound()
                                  ? CTAK_DeducedFromArrayBound 
                                  : CTAK_Deduced)) {
        Info.Param = makeTemplateParameter(
                         const_cast<NamedDecl *>(TemplateParams->getParam(I)));
        return TDK_SubstitutionFailure;
      }

      continue;
    }

    // Substitute into the default template argument, if available. 
    TemplateArgumentLoc DefArg
      = SubstDefaultTemplateArgumentIfAvailable(FunctionTemplate,
                                              FunctionTemplate->getLocation(),
                                  FunctionTemplate->getSourceRange().getEnd(),
                                                Param,
                                                Builder);

    // If there was no default argument, deduction is incomplete.
    if (DefArg.getArgument().isNull()) {
      Info.Param = makeTemplateParameter(
                         const_cast<NamedDecl *>(TemplateParams->getParam(I)));
      return TDK_Incomplete;
    }
    
    // Check whether we can actually use the default argument.
    if (CheckTemplateArgument(Param, DefArg,
                              FunctionTemplate,
                              FunctionTemplate->getLocation(),
                              FunctionTemplate->getSourceRange().getEnd(),
                              Builder,
                              CTAK_Deduced)) {
      Info.Param = makeTemplateParameter(
                         const_cast<NamedDecl *>(TemplateParams->getParam(I)));
      return TDK_SubstitutionFailure;
    }

    // If we get here, we successfully used the default template argument.
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList
    = new (Context) TemplateArgumentList(Context, Builder, /*TakeArgs=*/true);
  Info.reset(DeducedArgumentList);

  // Substitute the deduced template arguments into the function template
  // declaration to produce the function template specialization.
  Specialization = cast_or_null<FunctionDecl>(
                      SubstDecl(FunctionTemplate->getTemplatedDecl(),
                                FunctionTemplate->getDeclContext(),
                         MultiLevelTemplateArgumentList(*DeducedArgumentList)));
  if (!Specialization)
    return TDK_SubstitutionFailure;

  assert(Specialization->getPrimaryTemplate()->getCanonicalDecl() == 
         FunctionTemplate->getCanonicalDecl());
  
  // If the template argument list is owned by the function template
  // specialization, release it.
  if (Specialization->getTemplateSpecializationArgs() == DeducedArgumentList)
    Info.take();

  // There may have been an error that did not prevent us from constructing a
  // declaration. Mark the declaration invalid and return with a substitution
  // failure.
  if (Trap.hasErrorOccurred()) {
    Specialization->setInvalidDecl(true);
    return TDK_SubstitutionFailure;
  }

  return TDK_Success;
}

static QualType GetTypeOfFunction(ASTContext &Context,
                                  bool isAddressOfOperand,
                                  FunctionDecl *Fn) {
  if (!isAddressOfOperand) return Fn->getType();
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn))
    if (Method->isInstance())
      return Context.getMemberPointerType(Fn->getType(),
               Context.getTypeDeclType(Method->getParent()).getTypePtr());
  return Context.getPointerType(Fn->getType());
}

/// Apply the deduction rules for overload sets.
///
/// \return the null type if this argument should be treated as an
/// undeduced context
static QualType
ResolveOverloadForDeduction(Sema &S, TemplateParameterList *TemplateParams,
                            Expr *Arg, QualType ParamType) {
  llvm::PointerIntPair<OverloadExpr*,1> R = OverloadExpr::find(Arg);

  bool isAddressOfOperand = bool(R.getInt());
  OverloadExpr *Ovl = R.getPointer();

  // If there were explicit template arguments, we can only find
  // something via C++ [temp.arg.explicit]p3, i.e. if the arguments
  // unambiguously name a full specialization.
  if (Ovl->hasExplicitTemplateArgs()) {
    // But we can still look for an explicit specialization.
    if (FunctionDecl *ExplicitSpec
          = S.ResolveSingleFunctionTemplateSpecialization(Ovl))
      return GetTypeOfFunction(S.Context, isAddressOfOperand, ExplicitSpec);
    return QualType();
  }

  // C++0x [temp.deduct.call]p6:
  //   When P is a function type, pointer to function type, or pointer
  //   to member function type:

  if (!ParamType->isFunctionType() &&
      !ParamType->isFunctionPointerType() &&
      !ParamType->isMemberFunctionPointerType())
    return QualType();

  QualType Match;
  for (UnresolvedSetIterator I = Ovl->decls_begin(),
         E = Ovl->decls_end(); I != E; ++I) {
    NamedDecl *D = (*I)->getUnderlyingDecl();

    //   - If the argument is an overload set containing one or more
    //     function templates, the parameter is treated as a
    //     non-deduced context.
    if (isa<FunctionTemplateDecl>(D))
      return QualType();

    FunctionDecl *Fn = cast<FunctionDecl>(D);
    QualType ArgType = GetTypeOfFunction(S.Context, isAddressOfOperand, Fn);

    //   - If the argument is an overload set (not containing function
    //     templates), trial argument deduction is attempted using each
    //     of the members of the set. If deduction succeeds for only one
    //     of the overload set members, that member is used as the
    //     argument value for the deduction. If deduction succeeds for
    //     more than one member of the overload set the parameter is
    //     treated as a non-deduced context.

    // We do all of this in a fresh context per C++0x [temp.deduct.type]p2:
    //   Type deduction is done independently for each P/A pair, and
    //   the deduced template argument values are then combined.
    // So we do not reject deductions which were made elsewhere.
    llvm::SmallVector<DeducedTemplateArgument, 8> 
      Deduced(TemplateParams->size());
    Sema::TemplateDeductionInfo Info(S.Context, Ovl->getNameLoc());
    unsigned TDF = 0;

    Sema::TemplateDeductionResult Result
      = DeduceTemplateArguments(S, TemplateParams,
                                ParamType, ArgType,
                                Info, Deduced, TDF);
    if (Result) continue;
    if (!Match.isNull()) return QualType();
    Match = ArgType;
  }

  return Match;
}

/// \brief Perform template argument deduction from a function call
/// (C++ [temp.deduct.call]).
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArguments the explicit template arguments provided
/// for this call.
///
/// \param Args the function call arguments
///
/// \param NumArgs the number of arguments in Args
///
/// \param Name the name of the function being called. This is only significant
/// when the function template is a conversion function template, in which
/// case this routine will also perform template argument deduction based on
/// the function to which 
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgumentListInfo *ExplicitTemplateArgs,
                              Expr **Args, unsigned NumArgs,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();

  // C++ [temp.deduct.call]p1:
  //   Template argument deduction is done by comparing each function template
  //   parameter type (call it P) with the type of the corresponding argument
  //   of the call (call it A) as described below.
  unsigned CheckArgs = NumArgs;
  if (NumArgs < Function->getMinRequiredArguments())
    return TDK_TooFewArguments;
  else if (NumArgs > Function->getNumParams()) {
    const FunctionProtoType *Proto
      = Function->getType()->getAs<FunctionProtoType>();
    if (!Proto->isVariadic())
      return TDK_TooManyArguments;

    CheckArgs = Function->getNumParams();
  }

  // The types of the parameters from which we will perform template argument
  // deduction.
  Sema::LocalInstantiationScope InstScope(*this);
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  llvm::SmallVector<QualType, 4> ParamTypes;
  unsigned NumExplicitlySpecified = 0;
  if (ExplicitTemplateArgs) {
    TemplateDeductionResult Result =
      SubstituteExplicitTemplateArguments(FunctionTemplate,
                                          *ExplicitTemplateArgs,
                                          Deduced,
                                          ParamTypes,
                                          0,
                                          Info);
    if (Result)
      return Result;

    NumExplicitlySpecified = Deduced.size();
  } else {
    // Just fill in the parameter types from the function declaration.
    for (unsigned I = 0; I != CheckArgs; ++I)
      ParamTypes.push_back(Function->getParamDecl(I)->getType());
  }

  // Deduce template arguments from the function parameters.
  Deduced.resize(TemplateParams->size());
  for (unsigned I = 0; I != CheckArgs; ++I) {
    QualType ParamType = ParamTypes[I];
    QualType ArgType = Args[I]->getType();

    // Overload sets usually make this parameter an undeduced
    // context, but there are sometimes special circumstances.
    if (ArgType == Context.OverloadTy) {
      ArgType = ResolveOverloadForDeduction(*this, TemplateParams,
                                            Args[I], ParamType);
      if (ArgType.isNull())
        continue;
    }

    // C++ [temp.deduct.call]p2:
    //   If P is not a reference type:
    QualType CanonParamType = Context.getCanonicalType(ParamType);
    bool ParamWasReference = isa<ReferenceType>(CanonParamType);
    if (!ParamWasReference) {
      //   - If A is an array type, the pointer type produced by the
      //     array-to-pointer standard conversion (4.2) is used in place of
      //     A for type deduction; otherwise,
      if (ArgType->isArrayType())
        ArgType = Context.getArrayDecayedType(ArgType);
      //   - If A is a function type, the pointer type produced by the
      //     function-to-pointer standard conversion (4.3) is used in place
      //     of A for type deduction; otherwise,
      else if (ArgType->isFunctionType())
        ArgType = Context.getPointerType(ArgType);
      else {
        // - If A is a cv-qualified type, the top level cv-qualifiers of A’s
        //   type are ignored for type deduction.
        QualType CanonArgType = Context.getCanonicalType(ArgType);
        if (CanonArgType.getLocalCVRQualifiers())
          ArgType = CanonArgType.getLocalUnqualifiedType();
      }
    }

    // C++0x [temp.deduct.call]p3:
    //   If P is a cv-qualified type, the top level cv-qualifiers of P’s type
    //   are ignored for type deduction.
    if (CanonParamType.getLocalCVRQualifiers())
      ParamType = CanonParamType.getLocalUnqualifiedType();
    if (const ReferenceType *ParamRefType = ParamType->getAs<ReferenceType>()) {
      //   [...] If P is a reference type, the type referred to by P is used
      //   for type deduction.
      ParamType = ParamRefType->getPointeeType();

      //   [...] If P is of the form T&&, where T is a template parameter, and
      //   the argument is an lvalue, the type A& is used in place of A for
      //   type deduction.
      if (isa<RValueReferenceType>(ParamRefType) &&
          ParamRefType->getAs<TemplateTypeParmType>() &&
          Args[I]->isLvalue(Context) == Expr::LV_Valid)
        ArgType = Context.getLValueReferenceType(ArgType);
    }

    // C++0x [temp.deduct.call]p4:
    //   In general, the deduction process attempts to find template argument
    //   values that will make the deduced A identical to A (after the type A
    //   is transformed as described above). [...]
    unsigned TDF = TDF_SkipNonDependent;

    //     - If the original P is a reference type, the deduced A (i.e., the
    //       type referred to by the reference) can be more cv-qualified than
    //       the transformed A.
    if (ParamWasReference)
      TDF |= TDF_ParamWithReferenceType;
    //     - The transformed A can be another pointer or pointer to member
    //       type that can be converted to the deduced A via a qualification
    //       conversion (4.4).
    if (ArgType->isPointerType() || ArgType->isMemberPointerType())
      TDF |= TDF_IgnoreQualifiers;
    //     - If P is a class and P has the form simple-template-id, then the
    //       transformed A can be a derived class of the deduced A. Likewise,
    //       if P is a pointer to a class of the form simple-template-id, the
    //       transformed A can be a pointer to a derived class pointed to by
    //       the deduced A.
    if (isSimpleTemplateIdType(ParamType) ||
        (isa<PointerType>(ParamType) &&
         isSimpleTemplateIdType(
                              ParamType->getAs<PointerType>()->getPointeeType())))
      TDF |= TDF_DerivedClass;

    if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(*this, TemplateParams,
                                    ParamType, ArgType, Info, Deduced,
                                    TDF))
      return Result;

    // FIXME: we need to check that the deduced A is the same as A,
    // modulo the various allowed differences.
  }

  return FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
                                         NumExplicitlySpecified,
                                         Specialization, Info);
}

/// \brief Deduce template arguments when taking the address of a function
/// template (C++ [temp.deduct.funcaddr]) or matching a specialization to
/// a template.
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArguments the explicitly-specified template 
/// arguments.
///
/// \param ArgFunctionType the function type that will be used as the
/// "argument" type (A) when performing template argument deduction from the
/// function template's function type. This type may be NULL, if there is no
/// argument type to compare against, in C++0x [temp.arg.explicit]p3.
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                              QualType ArgFunctionType,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  QualType FunctionType = Function->getType();

  // Substitute any explicit template arguments.
  Sema::LocalInstantiationScope InstScope(*this);
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  unsigned NumExplicitlySpecified = 0;
  llvm::SmallVector<QualType, 4> ParamTypes;
  if (ExplicitTemplateArgs) {
    if (TemplateDeductionResult Result
          = SubstituteExplicitTemplateArguments(FunctionTemplate,
                                                *ExplicitTemplateArgs,
                                                Deduced, ParamTypes,
                                                &FunctionType, Info))
      return Result;

    NumExplicitlySpecified = Deduced.size();
  }

  // Template argument deduction for function templates in a SFINAE context.
  // Trap any errors that might occur.
  SFINAETrap Trap(*this);

  Deduced.resize(TemplateParams->size());

  if (!ArgFunctionType.isNull()) {
    // Deduce template arguments from the function type.
    if (TemplateDeductionResult Result
          = ::DeduceTemplateArguments(*this, TemplateParams,
                                      FunctionType, ArgFunctionType, Info,
                                      Deduced, 0))
      return Result;
  }
  
  return FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
                                         NumExplicitlySpecified,
                                         Specialization, Info);
}

/// \brief Deduce template arguments for a templated conversion
/// function (C++ [temp.deduct.conv]) and, if successful, produce a
/// conversion function template specialization.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                              QualType ToType,
                              CXXConversionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  CXXConversionDecl *Conv
    = cast<CXXConversionDecl>(FunctionTemplate->getTemplatedDecl());
  QualType FromType = Conv->getConversionType();

  // Canonicalize the types for deduction.
  QualType P = Context.getCanonicalType(FromType);
  QualType A = Context.getCanonicalType(ToType);

  // C++0x [temp.deduct.conv]p3:
  //   If P is a reference type, the type referred to by P is used for
  //   type deduction.
  if (const ReferenceType *PRef = P->getAs<ReferenceType>())
    P = PRef->getPointeeType();

  // C++0x [temp.deduct.conv]p3:
  //   If A is a reference type, the type referred to by A is used
  //   for type deduction.
  if (const ReferenceType *ARef = A->getAs<ReferenceType>())
    A = ARef->getPointeeType();
  // C++ [temp.deduct.conv]p2:
  //
  //   If A is not a reference type:
  else {
    assert(!A->isReferenceType() && "Reference types were handled above");

    //   - If P is an array type, the pointer type produced by the
    //     array-to-pointer standard conversion (4.2) is used in place
    //     of P for type deduction; otherwise,
    if (P->isArrayType())
      P = Context.getArrayDecayedType(P);
    //   - If P is a function type, the pointer type produced by the
    //     function-to-pointer standard conversion (4.3) is used in
    //     place of P for type deduction; otherwise,
    else if (P->isFunctionType())
      P = Context.getPointerType(P);
    //   - If P is a cv-qualified type, the top level cv-qualifiers of
    //     P’s type are ignored for type deduction.
    else
      P = P.getUnqualifiedType();

    // C++0x [temp.deduct.conv]p3:
    //   If A is a cv-qualified type, the top level cv-qualifiers of A’s
    //   type are ignored for type deduction.
    A = A.getUnqualifiedType();
  }

  // Template argument deduction for function templates in a SFINAE context.
  // Trap any errors that might occur.
  SFINAETrap Trap(*this);

  // C++ [temp.deduct.conv]p1:
  //   Template argument deduction is done by comparing the return
  //   type of the template conversion function (call it P) with the
  //   type that is required as the result of the conversion (call it
  //   A) as described in 14.8.2.4.
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(TemplateParams->size());

  // C++0x [temp.deduct.conv]p4:
  //   In general, the deduction process attempts to find template
  //   argument values that will make the deduced A identical to
  //   A. However, there are two cases that allow a difference:
  unsigned TDF = 0;
  //     - If the original A is a reference type, A can be more
  //       cv-qualified than the deduced A (i.e., the type referred to
  //       by the reference)
  if (ToType->isReferenceType())
    TDF |= TDF_ParamWithReferenceType;
  //     - The deduced A can be another pointer or pointer to member
  //       type that can be converted to A via a qualiﬁcation
  //       conversion.
  //
  // (C++0x [temp.deduct.conv]p6 clarifies that this only happens when
  // both P and A are pointers or member pointers. In this case, we
  // just ignore cv-qualifiers completely).
  if ((P->isPointerType() && A->isPointerType()) ||
      (P->isMemberPointerType() && P->isMemberPointerType()))
    TDF |= TDF_IgnoreQualifiers;
  if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(*this, TemplateParams,
                                    P, A, Info, Deduced, TDF))
    return Result;

  // FIXME: we need to check that the deduced A is the same as A,
  // modulo the various allowed differences.

  // Finish template argument deduction.
  Sema::LocalInstantiationScope InstScope(*this);
  FunctionDecl *Spec = 0;
  TemplateDeductionResult Result
    = FinishTemplateArgumentDeduction(FunctionTemplate, Deduced, 0, Spec, 
                                      Info);
  Specialization = cast_or_null<CXXConversionDecl>(Spec);
  return Result;
}

/// \brief Deduce template arguments for a function template when there is
/// nothing to deduce against (C++0x [temp.arg.explicit]p3).
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArguments the explicitly-specified template 
/// arguments.
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                           const TemplateArgumentListInfo *ExplicitTemplateArgs,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  return DeduceTemplateArguments(FunctionTemplate, ExplicitTemplateArgs,
                                 QualType(), Specialization, Info);
}

/// \brief Stores the result of comparing the qualifiers of two types.
enum DeductionQualifierComparison { 
  NeitherMoreQualified = 0, 
  ParamMoreQualified, 
  ArgMoreQualified 
};

/// \brief Deduce the template arguments during partial ordering by comparing 
/// the parameter type and the argument type (C++0x [temp.deduct.partial]).
///
/// \param S the semantic analysis object within which we are deducing
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param ParamIn the parameter type
///
/// \param ArgIn the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static Sema::TemplateDeductionResult
DeduceTemplateArgumentsDuringPartialOrdering(Sema &S,
                                        TemplateParameterList *TemplateParams,
                                             QualType ParamIn, QualType ArgIn,
                                             Sema::TemplateDeductionInfo &Info,
                      llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
   llvm::SmallVectorImpl<DeductionQualifierComparison> *QualifierComparisons) {
  CanQualType Param = S.Context.getCanonicalType(ParamIn);
  CanQualType Arg = S.Context.getCanonicalType(ArgIn);

  // C++0x [temp.deduct.partial]p5:
  //   Before the partial ordering is done, certain transformations are 
  //   performed on the types used for partial ordering: 
  //     - If P is a reference type, P is replaced by the type referred to. 
  CanQual<ReferenceType> ParamRef = Param->getAs<ReferenceType>();
  if (!ParamRef.isNull())
    Param = ParamRef->getPointeeType();
  
  //     - If A is a reference type, A is replaced by the type referred to.
  CanQual<ReferenceType> ArgRef = Arg->getAs<ReferenceType>();
  if (!ArgRef.isNull())
    Arg = ArgRef->getPointeeType();
  
  if (QualifierComparisons && !ParamRef.isNull() && !ArgRef.isNull()) {
    // C++0x [temp.deduct.partial]p6:
    //   If both P and A were reference types (before being replaced with the 
    //   type referred to above), determine which of the two types (if any) is 
    //   more cv-qualified than the other; otherwise the types are considered to 
    //   be equally cv-qualified for partial ordering purposes. The result of this
    //   determination will be used below.
    //
    // We save this information for later, using it only when deduction 
    // succeeds in both directions.
    DeductionQualifierComparison QualifierResult = NeitherMoreQualified;
    if (Param.isMoreQualifiedThan(Arg))
      QualifierResult = ParamMoreQualified;
    else if (Arg.isMoreQualifiedThan(Param))
      QualifierResult = ArgMoreQualified;
    QualifierComparisons->push_back(QualifierResult);
  }
  
  // C++0x [temp.deduct.partial]p7:
  //   Remove any top-level cv-qualifiers:
  //     - If P is a cv-qualified type, P is replaced by the cv-unqualified 
  //       version of P.
  Param = Param.getUnqualifiedType();
  //     - If A is a cv-qualified type, A is replaced by the cv-unqualified 
  //       version of A.
  Arg = Arg.getUnqualifiedType();
  
  // C++0x [temp.deduct.partial]p8:
  //   Using the resulting types P and A the deduction is then done as 
  //   described in 14.9.2.5. If deduction succeeds for a given type, the type
  //   from the argument template is considered to be at least as specialized
  //   as the type from the parameter template.
  return DeduceTemplateArguments(S, TemplateParams, Param, Arg, Info,
                                 Deduced, TDF_None);
}

static void
MarkUsedTemplateParameters(Sema &SemaRef, QualType T,
                           bool OnlyDeduced,
                           unsigned Level,
                           llvm::SmallVectorImpl<bool> &Deduced);
  
/// \brief Determine whether the function template \p FT1 is at least as
/// specialized as \p FT2.
static bool isAtLeastAsSpecializedAs(Sema &S,
                                     SourceLocation Loc,
                                     FunctionTemplateDecl *FT1,
                                     FunctionTemplateDecl *FT2,
                                     TemplatePartialOrderingContext TPOC,
    llvm::SmallVectorImpl<DeductionQualifierComparison> *QualifierComparisons) {
  FunctionDecl *FD1 = FT1->getTemplatedDecl();
  FunctionDecl *FD2 = FT2->getTemplatedDecl();  
  const FunctionProtoType *Proto1 = FD1->getType()->getAs<FunctionProtoType>();
  const FunctionProtoType *Proto2 = FD2->getType()->getAs<FunctionProtoType>();
  
  assert(Proto1 && Proto2 && "Function templates must have prototypes");
  TemplateParameterList *TemplateParams = FT2->getTemplateParameters();
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(TemplateParams->size());

  // C++0x [temp.deduct.partial]p3:
  //   The types used to determine the ordering depend on the context in which
  //   the partial ordering is done:
  Sema::TemplateDeductionInfo Info(S.Context, Loc);
  switch (TPOC) {
  case TPOC_Call: {
    //   - In the context of a function call, the function parameter types are
    //     used.
    unsigned NumParams = std::min(Proto1->getNumArgs(), Proto2->getNumArgs());
    for (unsigned I = 0; I != NumParams; ++I)
      if (DeduceTemplateArgumentsDuringPartialOrdering(S,
                                                       TemplateParams,
                                                       Proto2->getArgType(I),
                                                       Proto1->getArgType(I),
                                                       Info,
                                                       Deduced,
                                                       QualifierComparisons))
        return false;
    
    break;
  }
    
  case TPOC_Conversion:
    //   - In the context of a call to a conversion operator, the return types
    //     of the conversion function templates are used.
    if (DeduceTemplateArgumentsDuringPartialOrdering(S,
                                                     TemplateParams,
                                                     Proto2->getResultType(),
                                                     Proto1->getResultType(),
                                                     Info,
                                                     Deduced,
                                                     QualifierComparisons))
      return false;
    break;
    
  case TPOC_Other:
    //   - In other contexts (14.6.6.2) the function template’s function type 
    //     is used.
    if (DeduceTemplateArgumentsDuringPartialOrdering(S,
                                                     TemplateParams,
                                                     FD2->getType(),
                                                     FD1->getType(),
                                                     Info,
                                                     Deduced,
                                                     QualifierComparisons))
      return false;
    break;
  }
  
  // C++0x [temp.deduct.partial]p11:
  //   In most cases, all template parameters must have values in order for 
  //   deduction to succeed, but for partial ordering purposes a template 
  //   parameter may remain without a value provided it is not used in the 
  //   types being used for partial ordering. [ Note: a template parameter used
  //   in a non-deduced context is considered used. -end note]
  unsigned ArgIdx = 0, NumArgs = Deduced.size();
  for (; ArgIdx != NumArgs; ++ArgIdx)
    if (Deduced[ArgIdx].isNull())
      break;

  if (ArgIdx == NumArgs) {
    // All template arguments were deduced. FT1 is at least as specialized 
    // as FT2.
    return true;
  }

  // Figure out which template parameters were used.
  llvm::SmallVector<bool, 4> UsedParameters;
  UsedParameters.resize(TemplateParams->size());
  switch (TPOC) {
  case TPOC_Call: {
    unsigned NumParams = std::min(Proto1->getNumArgs(), Proto2->getNumArgs());
    for (unsigned I = 0; I != NumParams; ++I)
      ::MarkUsedTemplateParameters(S, Proto2->getArgType(I), false, 
                                   TemplateParams->getDepth(),
                                   UsedParameters);
    break;
  }
    
  case TPOC_Conversion:
    ::MarkUsedTemplateParameters(S, Proto2->getResultType(), false, 
                                 TemplateParams->getDepth(),
                                 UsedParameters);
    break;
    
  case TPOC_Other:
    ::MarkUsedTemplateParameters(S, FD2->getType(), false, 
                                 TemplateParams->getDepth(),
                                 UsedParameters);
    break;
  }
  
  for (; ArgIdx != NumArgs; ++ArgIdx)
    // If this argument had no value deduced but was used in one of the types
    // used for partial ordering, then deduction fails.
    if (Deduced[ArgIdx].isNull() && UsedParameters[ArgIdx])
      return false;
  
  return true;
}
                                    
                                     
/// \brief Returns the more specialized function template according
/// to the rules of function template partial ordering (C++ [temp.func.order]).
///
/// \param FT1 the first function template
///
/// \param FT2 the second function template
///
/// \param TPOC the context in which we are performing partial ordering of
/// function templates.
///
/// \returns the more specialized function template. If neither
/// template is more specialized, returns NULL.
FunctionTemplateDecl *
Sema::getMoreSpecializedTemplate(FunctionTemplateDecl *FT1,
                                 FunctionTemplateDecl *FT2,
                                 SourceLocation Loc,
                                 TemplatePartialOrderingContext TPOC) {
  llvm::SmallVector<DeductionQualifierComparison, 4> QualifierComparisons;
  bool Better1 = isAtLeastAsSpecializedAs(*this, Loc, FT1, FT2, TPOC, 0);
  bool Better2 = isAtLeastAsSpecializedAs(*this, Loc, FT2, FT1, TPOC, 
                                          &QualifierComparisons);
  
  if (Better1 != Better2) // We have a clear winner
    return Better1? FT1 : FT2;
  
  if (!Better1 && !Better2) // Neither is better than the other
    return 0;


  // C++0x [temp.deduct.partial]p10:
  //   If for each type being considered a given template is at least as 
  //   specialized for all types and more specialized for some set of types and
  //   the other template is not more specialized for any types or is not at 
  //   least as specialized for any types, then the given template is more
  //   specialized than the other template. Otherwise, neither template is more
  //   specialized than the other.
  Better1 = false;
  Better2 = false;
  for (unsigned I = 0, N = QualifierComparisons.size(); I != N; ++I) {
    // C++0x [temp.deduct.partial]p9:
    //   If, for a given type, deduction succeeds in both directions (i.e., the
    //   types are identical after the transformations above) and if the type
    //   from the argument template is more cv-qualified than the type from the
    //   parameter template (as described above) that type is considered to be
    //   more specialized than the other. If neither type is more cv-qualified 
    //   than the other then neither type is more specialized than the other.
    switch (QualifierComparisons[I]) {
      case NeitherMoreQualified:
        break;
        
      case ParamMoreQualified:
        Better1 = true;
        if (Better2)
          return 0;
        break;
        
      case ArgMoreQualified:
        Better2 = true;
        if (Better1)
          return 0;
        break;
    }
  }
   
  assert(!(Better1 && Better2) && "Should have broken out in the loop above");
  if (Better1)
    return FT1;
  else if (Better2)
    return FT2;
  else
    return 0;
}

/// \brief Determine if the two templates are equivalent.
static bool isSameTemplate(TemplateDecl *T1, TemplateDecl *T2) {
  if (T1 == T2)
    return true;
  
  if (!T1 || !T2)
    return false;
  
  return T1->getCanonicalDecl() == T2->getCanonicalDecl();
}

/// \brief Retrieve the most specialized of the given function template
/// specializations.
///
/// \param SpecBegin the start iterator of the function template
/// specializations that we will be comparing.
///
/// \param SpecEnd the end iterator of the function template
/// specializations, paired with \p SpecBegin.
///
/// \param TPOC the partial ordering context to use to compare the function
/// template specializations.
///
/// \param Loc the location where the ambiguity or no-specializations 
/// diagnostic should occur.
///
/// \param NoneDiag partial diagnostic used to diagnose cases where there are
/// no matching candidates.
///
/// \param AmbigDiag partial diagnostic used to diagnose an ambiguity, if one
/// occurs.
///
/// \param CandidateDiag partial diagnostic used for each function template
/// specialization that is a candidate in the ambiguous ordering. One parameter
/// in this diagnostic should be unbound, which will correspond to the string
/// describing the template arguments for the function template specialization.
///
/// \param Index if non-NULL and the result of this function is non-nULL, 
/// receives the index corresponding to the resulting function template
/// specialization.
///
/// \returns the most specialized function template specialization, if 
/// found. Otherwise, returns SpecEnd.
///
/// \todo FIXME: Consider passing in the "also-ran" candidates that failed 
/// template argument deduction.
UnresolvedSetIterator
Sema::getMostSpecialized(UnresolvedSetIterator SpecBegin,
                         UnresolvedSetIterator SpecEnd,
                         TemplatePartialOrderingContext TPOC,
                         SourceLocation Loc,
                         const PartialDiagnostic &NoneDiag,
                         const PartialDiagnostic &AmbigDiag,
                         const PartialDiagnostic &CandidateDiag) {
  if (SpecBegin == SpecEnd) {
    Diag(Loc, NoneDiag);
    return SpecEnd;
  }
  
  if (SpecBegin + 1 == SpecEnd)    
    return SpecBegin;
  
  // Find the function template that is better than all of the templates it
  // has been compared to.
  UnresolvedSetIterator Best = SpecBegin;
  FunctionTemplateDecl *BestTemplate 
    = cast<FunctionDecl>(*Best)->getPrimaryTemplate();
  assert(BestTemplate && "Not a function template specialization?");
  for (UnresolvedSetIterator I = SpecBegin + 1; I != SpecEnd; ++I) {
    FunctionTemplateDecl *Challenger
      = cast<FunctionDecl>(*I)->getPrimaryTemplate();
    assert(Challenger && "Not a function template specialization?");
    if (isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger,
                                                  Loc, TPOC),
                       Challenger)) {
      Best = I;
      BestTemplate = Challenger;
    }
  }
  
  // Make sure that the "best" function template is more specialized than all
  // of the others.
  bool Ambiguous = false;
  for (UnresolvedSetIterator I = SpecBegin; I != SpecEnd; ++I) {
    FunctionTemplateDecl *Challenger
      = cast<FunctionDecl>(*I)->getPrimaryTemplate();
    if (I != Best &&
        !isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger, 
                                                   Loc, TPOC),
                        BestTemplate)) {
      Ambiguous = true;
      break;
    }
  }
  
  if (!Ambiguous) {
    // We found an answer. Return it.
    return Best;
  }
  
  // Diagnose the ambiguity.
  Diag(Loc, AmbigDiag);
  
  // FIXME: Can we order the candidates in some sane way?
  for (UnresolvedSetIterator I = SpecBegin; I != SpecEnd; ++I)
    Diag((*I)->getLocation(), CandidateDiag)
      << getTemplateArgumentBindingsText(
        cast<FunctionDecl>(*I)->getPrimaryTemplate()->getTemplateParameters(),
                    *cast<FunctionDecl>(*I)->getTemplateSpecializationArgs());
  
  return SpecEnd;
}

/// \brief Returns the more specialized class template partial specialization
/// according to the rules of partial ordering of class template partial
/// specializations (C++ [temp.class.order]).
///
/// \param PS1 the first class template partial specialization
///
/// \param PS2 the second class template partial specialization
///
/// \returns the more specialized class template partial specialization. If
/// neither partial specialization is more specialized, returns NULL.
ClassTemplatePartialSpecializationDecl *
Sema::getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2,
                                              SourceLocation Loc) {
  // C++ [temp.class.order]p1:
  //   For two class template partial specializations, the first is at least as
  //   specialized as the second if, given the following rewrite to two 
  //   function templates, the first function template is at least as 
  //   specialized as the second according to the ordering rules for function 
  //   templates (14.6.6.2):
  //     - the first function template has the same template parameters as the
  //       first partial specialization and has a single function parameter 
  //       whose type is a class template specialization with the template 
  //       arguments of the first partial specialization, and
  //     - the second function template has the same template parameters as the
  //       second partial specialization and has a single function parameter 
  //       whose type is a class template specialization with the template 
  //       arguments of the second partial specialization.
  //
  // Rather than synthesize function templates, we merely perform the 
  // equivalent partial ordering by performing deduction directly on the
  // template arguments of the class template partial specializations. This
  // computation is slightly simpler than the general problem of function
  // template partial ordering, because class template partial specializations
  // are more constrained. We know that every template parameter is deduc
  llvm::SmallVector<DeducedTemplateArgument, 4> Deduced;
  Sema::TemplateDeductionInfo Info(Context, Loc);
  
  // Determine whether PS1 is at least as specialized as PS2
  Deduced.resize(PS2->getTemplateParameters()->size());
  bool Better1 = !DeduceTemplateArgumentsDuringPartialOrdering(*this,
                                                  PS2->getTemplateParameters(),
                                                  Context.getTypeDeclType(PS2),
                                                  Context.getTypeDeclType(PS1),
                                                               Info,
                                                               Deduced,
                                                               0);

  // Determine whether PS2 is at least as specialized as PS1
  Deduced.clear();
  Deduced.resize(PS1->getTemplateParameters()->size());
  bool Better2 = !DeduceTemplateArgumentsDuringPartialOrdering(*this,
                                                  PS1->getTemplateParameters(),
                                                  Context.getTypeDeclType(PS1),
                                                  Context.getTypeDeclType(PS2),
                                                               Info,
                                                               Deduced,
                                                               0);
  
  if (Better1 == Better2)
    return 0;
  
  return Better1? PS1 : PS2;
}

static void
MarkUsedTemplateParameters(Sema &SemaRef,
                           const TemplateArgument &TemplateArg,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used);

/// \brief Mark the template parameters that are used by the given
/// expression.
static void
MarkUsedTemplateParameters(Sema &SemaRef,
                           const Expr *E,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used) {
  // FIXME: if !OnlyDeduced, we have to walk the whole subexpression to 
  // find other occurrences of template parameters.
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE)
    return;

  const NonTypeTemplateParmDecl *NTTP
    = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl());
  if (!NTTP)
    return;

  if (NTTP->getDepth() == Depth)
    Used[NTTP->getIndex()] = true;
}

/// \brief Mark the template parameters that are used by the given
/// nested name specifier.
static void
MarkUsedTemplateParameters(Sema &SemaRef,
                           NestedNameSpecifier *NNS,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used) {
  if (!NNS)
    return;
  
  MarkUsedTemplateParameters(SemaRef, NNS->getPrefix(), OnlyDeduced, Depth,
                             Used);
  MarkUsedTemplateParameters(SemaRef, QualType(NNS->getAsType(), 0), 
                             OnlyDeduced, Depth, Used);
}
  
/// \brief Mark the template parameters that are used by the given
/// template name.
static void
MarkUsedTemplateParameters(Sema &SemaRef,
                           TemplateName Name,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used) {
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    if (TemplateTemplateParmDecl *TTP
          = dyn_cast<TemplateTemplateParmDecl>(Template)) {
      if (TTP->getDepth() == Depth)
        Used[TTP->getIndex()] = true;
    }
    return;
  }
  
  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName())
    MarkUsedTemplateParameters(SemaRef, QTN->getQualifier(), OnlyDeduced, 
                               Depth, Used);
  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName())
    MarkUsedTemplateParameters(SemaRef, DTN->getQualifier(), OnlyDeduced, 
                               Depth, Used);
}

/// \brief Mark the template parameters that are used by the given
/// type.
static void
MarkUsedTemplateParameters(Sema &SemaRef, QualType T,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used) {
  if (T.isNull())
    return;
  
  // Non-dependent types have nothing deducible
  if (!T->isDependentType())
    return;

  T = SemaRef.Context.getCanonicalType(T);
  switch (T->getTypeClass()) {
  case Type::Pointer:
    MarkUsedTemplateParameters(SemaRef,
                               cast<PointerType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::BlockPointer:
    MarkUsedTemplateParameters(SemaRef,
                               cast<BlockPointerType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    MarkUsedTemplateParameters(SemaRef,
                               cast<ReferenceType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::MemberPointer: {
    const MemberPointerType *MemPtr = cast<MemberPointerType>(T.getTypePtr());
    MarkUsedTemplateParameters(SemaRef, MemPtr->getPointeeType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(SemaRef, QualType(MemPtr->getClass(), 0),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::DependentSizedArray:
    MarkUsedTemplateParameters(SemaRef,
                               cast<DependentSizedArrayType>(T)->getSizeExpr(),
                               OnlyDeduced, Depth, Used);
    // Fall through to check the element type

  case Type::ConstantArray:
  case Type::IncompleteArray:
    MarkUsedTemplateParameters(SemaRef,
                               cast<ArrayType>(T)->getElementType(),
                               OnlyDeduced, Depth, Used);
    break;

  case Type::Vector:
  case Type::ExtVector:
    MarkUsedTemplateParameters(SemaRef,
                               cast<VectorType>(T)->getElementType(),
                               OnlyDeduced, Depth, Used);
    break;

  case Type::DependentSizedExtVector: {
    const DependentSizedExtVectorType *VecType
      = cast<DependentSizedExtVectorType>(T);
    MarkUsedTemplateParameters(SemaRef, VecType->getElementType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(SemaRef, VecType->getSizeExpr(), OnlyDeduced, 
                               Depth, Used);
    break;
  }

  case Type::FunctionProto: {
    const FunctionProtoType *Proto = cast<FunctionProtoType>(T);
    MarkUsedTemplateParameters(SemaRef, Proto->getResultType(), OnlyDeduced,
                               Depth, Used);
    for (unsigned I = 0, N = Proto->getNumArgs(); I != N; ++I)
      MarkUsedTemplateParameters(SemaRef, Proto->getArgType(I), OnlyDeduced,
                                 Depth, Used);
    break;
  }

  case Type::TemplateTypeParm: {
    const TemplateTypeParmType *TTP = cast<TemplateTypeParmType>(T);
    if (TTP->getDepth() == Depth)
      Used[TTP->getIndex()] = true;
    break;
  }

  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *Spec
      = cast<TemplateSpecializationType>(T);
    MarkUsedTemplateParameters(SemaRef, Spec->getTemplateName(), OnlyDeduced,
                               Depth, Used);
    for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
      MarkUsedTemplateParameters(SemaRef, Spec->getArg(I), OnlyDeduced, Depth,
                                 Used);
    break;
  }

  case Type::Complex:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef, 
                                 cast<ComplexType>(T)->getElementType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::DependentName:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef,
                                 cast<DependentNameType>(T)->getQualifier(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::TypeOf:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef,
                                 cast<TypeOfType>(T)->getUnderlyingType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::TypeOfExpr:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef,
                                 cast<TypeOfExprType>(T)->getUnderlyingExpr(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::Decltype:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef,
                                 cast<DecltypeType>(T)->getUnderlyingExpr(),
                                 OnlyDeduced, Depth, Used);
    break;

  // None of these types have any template parameters in them.
  case Type::Builtin:
  case Type::VariableArray:
  case Type::FunctionNoProto:
  case Type::Record:
  case Type::Enum:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
  case Type::UnresolvedUsing:
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    break;
  }
}

/// \brief Mark the template parameters that are used by this
/// template argument.
static void
MarkUsedTemplateParameters(Sema &SemaRef,
                           const TemplateArgument &TemplateArg,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallVectorImpl<bool> &Used) {
  switch (TemplateArg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
    case TemplateArgument::Declaration:
    break;

  case TemplateArgument::Type:
    MarkUsedTemplateParameters(SemaRef, TemplateArg.getAsType(), OnlyDeduced,
                               Depth, Used);
    break;

  case TemplateArgument::Template:
    MarkUsedTemplateParameters(SemaRef, TemplateArg.getAsTemplate(), 
                               OnlyDeduced, Depth, Used);
    break;

  case TemplateArgument::Expression:
    MarkUsedTemplateParameters(SemaRef, TemplateArg.getAsExpr(), OnlyDeduced, 
                               Depth, Used);
    break;
      
  case TemplateArgument::Pack:
    for (TemplateArgument::pack_iterator P = TemplateArg.pack_begin(),
                                      PEnd = TemplateArg.pack_end();
         P != PEnd; ++P)
      MarkUsedTemplateParameters(SemaRef, *P, OnlyDeduced, Depth, Used);
    break;
  }
}

/// \brief Mark the template parameters can be deduced by the given
/// template argument list.
///
/// \param TemplateArgs the template argument list from which template
/// parameters will be deduced.
///
/// \param Deduced a bit vector whose elements will be set to \c true
/// to indicate when the corresponding template parameter will be
/// deduced.
void
Sema::MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                 bool OnlyDeduced, unsigned Depth,
                                 llvm::SmallVectorImpl<bool> &Used) {
  for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
    ::MarkUsedTemplateParameters(*this, TemplateArgs[I], OnlyDeduced, 
                                 Depth, Used);
}

/// \brief Marks all of the template parameters that will be deduced by a
/// call to the given function template.
void 
Sema::MarkDeducedTemplateParameters(FunctionTemplateDecl *FunctionTemplate,
                                    llvm::SmallVectorImpl<bool> &Deduced) {
  TemplateParameterList *TemplateParams 
    = FunctionTemplate->getTemplateParameters();
  Deduced.clear();
  Deduced.resize(TemplateParams->size());
  
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  for (unsigned I = 0, N = Function->getNumParams(); I != N; ++I)
    ::MarkUsedTemplateParameters(*this, Function->getParamDecl(I)->getType(),
                                 true, TemplateParams->getDepth(), Deduced);
}
