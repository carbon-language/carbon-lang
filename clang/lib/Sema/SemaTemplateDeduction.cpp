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
#include "llvm/Support/Compiler.h"
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

static Sema::TemplateDeductionResult
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        const TemplateArgument &Arg,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced);

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
DeduceNonTypeTemplateArgument(ASTContext &Context,
                              NonTypeTemplateParmDecl *NTTP,
                              llvm::APSInt Value,
                              Sema::TemplateDeductionInfo &Info,
                              llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");

  if (Deduced[NTTP->getIndex()].isNull()) {
    QualType T = NTTP->getType();

    // FIXME: Make sure we didn't overflow our data type!
    unsigned AllowedBits = Context.getTypeSize(T);
    if (Value.getBitWidth() != AllowedBits)
      Value.extOrTrunc(AllowedBits);
    Value.setIsSigned(T->isSignedIntegerType());

    Deduced[NTTP->getIndex()] = TemplateArgument(Value, T);
    return Sema::TDK_Success;
  }

  assert(Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Integral);

  // If the template argument was previously deduced to a negative value,
  // then our deduction fails.
  const llvm::APSInt *PrevValuePtr = Deduced[NTTP->getIndex()].getAsIntegral();
  if (PrevValuePtr->isNegative()) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(Value, NTTP->getType());
    return Sema::TDK_Inconsistent;
  }

  llvm::APSInt PrevValue = *PrevValuePtr;
  if (Value.getBitWidth() > PrevValue.getBitWidth())
    PrevValue.zext(Value.getBitWidth());
  else if (Value.getBitWidth() < PrevValue.getBitWidth())
    Value.zext(PrevValue.getBitWidth());

  if (Value != PrevValue) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(Value, NTTP->getType());
    return Sema::TDK_Inconsistent;
  }

  return Sema::TDK_Success;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given type- or value-dependent expression.
///
/// \returns true if deduction succeeded, false otherwise.

static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(ASTContext &Context,
                              NonTypeTemplateParmDecl *NTTP,
                              Expr *Value,
                              Sema::TemplateDeductionInfo &Info,
                           llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");
  assert((Value->isTypeDependent() || Value->isValueDependent()) &&
         "Expression template argument must be type- or value-dependent.");

  if (Deduced[NTTP->getIndex()].isNull()) {
    // FIXME: Clone the Value?
    Deduced[NTTP->getIndex()] = TemplateArgument(Value);
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
    Deduced[NTTP->getIndex()].getAsExpr()->Profile(ID1, Context, true);
    Value->Profile(ID2, Context, true);
    if (ID1 == ID2)
      return Sema::TDK_Success;
   
    // FIXME: Fill in argument mismatch information
    return Sema::TDK_NonDeducedMismatch;
  }

  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        TemplateName Param,
                        TemplateName Arg,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
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
      ExistingArg = TemplateArgument(Context.getCanonicalTemplateName(Arg));
      return Sema::TDK_Success;
    }
    
    // Verify that the previous binding matches this deduction.
    assert(ExistingArg.getKind() == TemplateArgument::Template);
    if (Context.hasSameTemplateName(ExistingArg.getAsTemplate(), Arg))
      return Sema::TDK_Success;
    
    // Inconsistent deduction.
    Info.Param = TempParam;
    Info.FirstArg = ExistingArg;
    Info.SecondArg = TemplateArgument(Arg);
    return Sema::TDK_Inconsistent;
  }
  
  // Verify that the two template names are equivalent.
  if (Context.hasSameTemplateName(Param, Arg))
    return Sema::TDK_Success;
  
  // Mismatch of non-dependent template parameter to argument.
  Info.FirstArg = TemplateArgument(Param);
  Info.SecondArg = TemplateArgument(Arg);
  return Sema::TDK_NonDeducedMismatch;
}

/// \brief Deduce the template arguments by comparing the template parameter
/// type (which is a template-id) with the template argument type.
///
/// \param Context the AST context in which this deduction occurs.
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
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        const TemplateSpecializationType *Param,
                        QualType Arg,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(Arg.isCanonical() && "Argument type must be canonical");

  // Check whether the template argument is a dependent template-id.
  if (const TemplateSpecializationType *SpecArg
        = dyn_cast<TemplateSpecializationType>(Arg)) {
    // Perform template argument deduction for the template name.
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(Context, TemplateParams,
                                    Param->getTemplateName(),
                                    SpecArg->getTemplateName(),
                                    Info, Deduced))
      return Result;


    // Perform template argument deduction on each template
    // argument.
    unsigned NumArgs = std::min(SpecArg->getNumArgs(), Param->getNumArgs());
    for (unsigned I = 0; I != NumArgs; ++I)
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(Context, TemplateParams,
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
        = DeduceTemplateArguments(Context,
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
          = DeduceTemplateArguments(Context, TemplateParams,
                                    Param->getArg(I),
                                    ArgArgs.get(I),
                                    Info, Deduced))
      return Result;

  return Sema::TDK_Success;
}

/// \brief Returns a completely-unqualified array type, capturing the
/// qualifiers in Quals.
///
/// \param Context the AST context in which the array type was built.
///
/// \param T a canonical type that may be an array type.
///
/// \param Quals will receive the full set of qualifiers that were
/// applied to the element type of the array.
///
/// \returns if \p T is an array type, the completely unqualified array type
/// that corresponds to T. Otherwise, returns T.
static QualType getUnqualifiedArrayType(ASTContext &Context, QualType T,
                                        Qualifiers &Quals) {
  assert(T.isCanonical() && "Only operates on canonical types");
  if (!isa<ArrayType>(T)) {
    Quals = T.getQualifiers();
    return T.getUnqualifiedType();
  }

  assert(!T.hasQualifiers() && "canonical array type has qualifiers!");

  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(T)) {
    QualType Elt = getUnqualifiedArrayType(Context, CAT->getElementType(),
                                           Quals);
    if (Elt == CAT->getElementType())
      return T;

    return Context.getConstantArrayType(Elt, CAT->getSize(),
                                        CAT->getSizeModifier(), 0);
  }

  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(T)) {
    QualType Elt = getUnqualifiedArrayType(Context, IAT->getElementType(),
                                           Quals);
    if (Elt == IAT->getElementType())
      return T;

    return Context.getIncompleteArrayType(Elt, IAT->getSizeModifier(), 0);
  }

  const DependentSizedArrayType *DSAT = cast<DependentSizedArrayType>(T);
  QualType Elt = getUnqualifiedArrayType(Context, DSAT->getElementType(),
                                         Quals);
  if (Elt == DSAT->getElementType())
    return T;

  return Context.getDependentSizedArrayType(Elt, DSAT->getSizeExpr()->Retain(),
                                            DSAT->getSizeModifier(), 0,
                                            SourceRange());
}

/// \brief Deduce the template arguments by comparing the parameter type and
/// the argument type (C++ [temp.deduct.type]).
///
/// \param Context the AST context in which this deduction occurs.
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
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        QualType ParamIn, QualType ArgIn,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced,
                        unsigned TDF) {
  // We only want to look at the canonical types, since typedefs and
  // sugar are not part of template argument deduction.
  QualType Param = Context.getCanonicalType(ParamIn);
  QualType Arg = Context.getCanonicalType(ArgIn);

  // C++0x [temp.deduct.call]p4 bullet 1:
  //   - If the original P is a reference type, the deduced A (i.e., the type
  //     referred to by the reference) can be more cv-qualified than the
  //     transformed A.
  if (TDF & TDF_ParamWithReferenceType) {
    Qualifiers Quals = Param.getQualifiers();
    Quals.setCVRQualifiers(Quals.getCVRQualifiers() & Arg.getCVRQualifiers());
    Param = Context.getQualifiedType(Param.getUnqualifiedType(), Quals);
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
      Arg = getUnqualifiedArrayType(Context, Arg, Quals);
      if (Quals) {
        Arg = Context.getQualifiedType(Arg, Quals);
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

    QualType DeducedType = Arg;
    DeducedType.removeCVRQualifiers(Param.getCVRQualifiers());
    if (RecanonicalizeArg)
      DeducedType = Context.getCanonicalType(DeducedType);

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
      return DeduceTemplateArguments(Context, TemplateParams,
                                   cast<PointerType>(Param)->getPointeeType(),
                                     PointerArg->getPointeeType(),
                                     Info, Deduced, SubTDF);
    }

    //     T &
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg = Arg->getAs<LValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(Context, TemplateParams,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced, 0);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg = Arg->getAs<RValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(Context, TemplateParams,
                           cast<RValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced, 0);
    }

    //     T [] (implied, but not stated explicitly)
    case Type::IncompleteArray: {
      const IncompleteArrayType *IncompleteArrayArg =
        Context.getAsIncompleteArrayType(Arg);
      if (!IncompleteArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(Context, TemplateParams,
                     Context.getAsIncompleteArrayType(Param)->getElementType(),
                                     IncompleteArrayArg->getElementType(),
                                     Info, Deduced, 0);
    }

    //     T [integer-constant]
    case Type::ConstantArray: {
      const ConstantArrayType *ConstantArrayArg =
        Context.getAsConstantArrayType(Arg);
      if (!ConstantArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      const ConstantArrayType *ConstantArrayParm =
        Context.getAsConstantArrayType(Param);
      if (ConstantArrayArg->getSize() != ConstantArrayParm->getSize())
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArguments(Context, TemplateParams,
                                     ConstantArrayParm->getElementType(),
                                     ConstantArrayArg->getElementType(),
                                     Info, Deduced, 0);
    }

    //     type [i]
    case Type::DependentSizedArray: {
      const ArrayType *ArrayArg = dyn_cast<ArrayType>(Arg);
      if (!ArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      // Check the element type of the arrays
      const DependentSizedArrayType *DependentArrayParm
        = cast<DependentSizedArrayType>(Param);
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(Context, TemplateParams,
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
        return DeduceNonTypeTemplateArgument(Context, NTTP, Size,
                                             Info, Deduced);
      }
      if (const DependentSizedArrayType *DependentArrayArg
            = dyn_cast<DependentSizedArrayType>(ArrayArg))
        return DeduceNonTypeTemplateArgument(Context, NTTP,
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
            = DeduceTemplateArguments(Context, TemplateParams,
                                      FunctionProtoParam->getResultType(),
                                      FunctionProtoArg->getResultType(),
                                      Info, Deduced, 0))
        return Result;

      for (unsigned I = 0, N = FunctionProtoParam->getNumArgs(); I != N; ++I) {
        // Check argument types.
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArguments(Context, TemplateParams,
                                        FunctionProtoParam->getArgType(I),
                                        FunctionProtoArg->getArgType(I),
                                        Info, Deduced, 0))
          return Result;
      }

      return Sema::TDK_Success;
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
        = DeduceTemplateArguments(Context, TemplateParams, SpecParam, Arg,
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
        if (const RecordType *RecordT = dyn_cast<RecordType>(Arg)) {
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
                = DeduceTemplateArguments(Context, TemplateParams, SpecParam,
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
            = DeduceTemplateArguments(Context, TemplateParams,
                                      MemPtrParam->getPointeeType(),
                                      MemPtrArg->getPointeeType(),
                                      Info, Deduced,
                                      TDF & TDF_IgnoreQualifiers))
        return Result;

      return DeduceTemplateArguments(Context, TemplateParams,
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

      return DeduceTemplateArguments(Context, TemplateParams,
                                     BlockPtrParam->getPointeeType(),
                                     BlockPtrArg->getPointeeType(), Info,
                                     Deduced, 0);
    }

    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::Typename:
      // No template argument deduction for these types
      return Sema::TDK_Success;

    default:
      break;
  }

  // FIXME: Many more cases to go (to go).
  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        const TemplateArgument &Arg,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  switch (Param.getKind()) {
  case TemplateArgument::Null:
    assert(false && "Null template argument in parameter list");
    break;

  case TemplateArgument::Type:
    if (Arg.getKind() == TemplateArgument::Type)
      return DeduceTemplateArguments(Context, TemplateParams, Param.getAsType(),
                                     Arg.getAsType(), Info, Deduced, 0);
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;
      
  case TemplateArgument::Template:
    if (Arg.getKind() == TemplateArgument::Template)
      return DeduceTemplateArguments(Context, TemplateParams, 
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
      // FIXME: Zero extension + sign checking here?
      if (*Param.getAsIntegral() == *Arg.getAsIntegral())
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
        // FIXME: Sign problems here
        return DeduceNonTypeTemplateArgument(Context, NTTP,
                                             *Arg.getAsIntegral(),
                                             Info, Deduced);
      if (Arg.getKind() == TemplateArgument::Expression)
        return DeduceNonTypeTemplateArgument(Context, NTTP, Arg.getAsExpr(),
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
DeduceTemplateArguments(ASTContext &Context,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgumentList &ParamList,
                        const TemplateArgumentList &ArgList,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(ParamList.size() == ArgList.size());
  for (unsigned I = 0, N = ParamList.size(); I != N; ++I) {
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(Context, TemplateParams,
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
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(Context,
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
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
        Info.Param = TTP;
      else if (NonTypeTemplateParmDecl *NTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(Param))
        Info.Param = NTTP;
      else
        Info.Param = cast<TemplateTemplateParmDecl>(Param);
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
  ClassTemplateDecl *ClassTemplate = Partial->getSpecializedTemplate();
  const TemplateArgumentLoc *PartialTemplateArgs
    = Partial->getTemplateArgsAsWritten();
  unsigned N = Partial->getNumTemplateArgsAsWritten();
  llvm::SmallVector<TemplateArgumentLoc, 16> InstArgs(N);
  for (unsigned I = 0; I != N; ++I) {
    Decl *Param = const_cast<NamedDecl *>(
                    ClassTemplate->getTemplateParameters()->getParam(I));
    if (Subst(PartialTemplateArgs[I], InstArgs[I],
              MultiLevelTemplateArgumentList(*DeducedArgumentList))) {
      Info.Param = makeTemplateParameter(Param);
      Info.FirstArg = PartialTemplateArgs[I].getArgument();
      return TDK_SubstitutionFailure;
    }
  }

  TemplateArgumentListBuilder ConvertedInstArgs(
                                  ClassTemplate->getTemplateParameters(), N);

  if (CheckTemplateArgumentList(ClassTemplate, Partial->getLocation(),
                                /*LAngle*/ SourceLocation(),
                                InstArgs.data(), N,
                                /*RAngle*/ SourceLocation(),
                                false, ConvertedInstArgs)) {
    // FIXME: fail with more useful information?
    return TDK_SubstitutionFailure;
  }
  
  for (unsigned I = 0, E = ConvertedInstArgs.flatSize(); I != E; ++I) {
    // We don't really care if we overwrite the internal structures of
    // the arg list builder, because we're going to throw it all away.
    TemplateArgument &InstArg
      = const_cast<TemplateArgument&>(ConvertedInstArgs.getFlatArguments()[I]);

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
/// \param NumExplicitTemplateArguments the number of explicitly-specified
/// template arguments in @p ExplicitTemplateArguments. This value may be zero.
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
                             const TemplateArgumentLoc *ExplicitTemplateArgs,
                                          unsigned NumExplicitTemplateArgs,
                            llvm::SmallVectorImpl<TemplateArgument> &Deduced,
                                 llvm::SmallVectorImpl<QualType> &ParamTypes,
                                          QualType *FunctionType,
                                          TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  if (NumExplicitTemplateArgs == 0) {
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
                                      NumExplicitTemplateArgs);

  // Enter a new template instantiation context where we check the
  // explicitly-specified template arguments against this function template,
  // and then substitute them into the function parameter types.
  InstantiatingTemplate Inst(*this, FunctionTemplate->getLocation(),
                             FunctionTemplate, Deduced.data(), Deduced.size(),
           ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution);
  if (Inst)
    return TDK_InstantiationDepth;

  if (CheckTemplateArgumentList(FunctionTemplate,
                                SourceLocation(), SourceLocation(),
                                ExplicitTemplateArgs,
                                NumExplicitTemplateArgs,
                                SourceLocation(),
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

/// \brief Finish template argument deduction for a function template,
/// checking the deduced template arguments for completeness and forming
/// the function template specialization.
Sema::TemplateDeductionResult
Sema::FinishTemplateArgumentDeduction(FunctionTemplateDecl *FunctionTemplate,
                            llvm::SmallVectorImpl<TemplateArgument> &Deduced,
                                      FunctionDecl *&Specialization,
                                      TemplateDeductionInfo &Info) {
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  TemplateArgumentListBuilder Builder(TemplateParams, Deduced.size());
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    if (Deduced[I].isNull()) {
      Info.Param = makeTemplateParameter(
                            const_cast<NamedDecl *>(TemplateParams->getParam(I)));
      return TDK_Incomplete;
    }

    Builder.Append(Deduced[I]);
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList
    = new (Context) TemplateArgumentList(Context, Builder, /*TakeArgs=*/true);
  Info.reset(DeducedArgumentList);

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

/// \brief Perform template argument deduction from a function call
/// (C++ [temp.deduct.call]).
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param HasExplicitTemplateArgs whether any template arguments were
/// explicitly specified.
///
/// \param ExplicitTemplateArguments when @p HasExplicitTemplateArgs is true,
/// the explicitly-specified template arguments.
///
/// \param NumExplicitTemplateArguments when @p HasExplicitTemplateArgs is true,
/// the number of explicitly-specified template arguments in
/// @p ExplicitTemplateArguments. This value may be zero.
///
/// \param Args the function call arguments
///
/// \param NumArgs the number of arguments in Args
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
                              bool HasExplicitTemplateArgs,
                              const TemplateArgumentLoc *ExplicitTemplateArgs,
                              unsigned NumExplicitTemplateArgs,
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
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  llvm::SmallVector<QualType, 4> ParamTypes;
  if (NumExplicitTemplateArgs) {
    TemplateDeductionResult Result =
      SubstituteExplicitTemplateArguments(FunctionTemplate,
                                          ExplicitTemplateArgs,
                                          NumExplicitTemplateArgs,
                                          Deduced,
                                          ParamTypes,
                                          0,
                                          Info);
    if (Result)
      return Result;
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
        if (CanonArgType.getCVRQualifiers())
          ArgType = CanonArgType.getUnqualifiedType();
      }
    }

    // C++0x [temp.deduct.call]p3:
    //   If P is a cv-qualified type, the top level cv-qualifiers of P’s type
    //   are ignored for type deduction.
    if (CanonParamType.getCVRQualifiers())
      ParamType = CanonParamType.getUnqualifiedType();
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
        = ::DeduceTemplateArguments(Context, TemplateParams,
                                    ParamType, ArgType, Info, Deduced,
                                    TDF))
      return Result;

    // FIXME: C++0x [temp.deduct.call] paragraphs 6-9 deal with function
    // pointer parameters.

    // FIXME: we need to check that the deduced A is the same as A,
    // modulo the various allowed differences.
  }

  return FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
                                         Specialization, Info);
}

/// \brief Deduce template arguments when taking the address of a function
/// template (C++ [temp.deduct.funcaddr]) or matching a 
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param HasExplicitTemplateArgs whether any template arguments were
/// explicitly specified.
///
/// \param ExplicitTemplateArguments when @p HasExplicitTemplateArgs is true,
/// the explicitly-specified template arguments.
///
/// \param NumExplicitTemplateArguments when @p HasExplicitTemplateArgs is true,
/// the number of explicitly-specified template arguments in
/// @p ExplicitTemplateArguments. This value may be zero.
///
/// \param ArgFunctionType the function type that will be used as the
/// "argument" type (A) when performing template argument deduction from the
/// function template's function type.
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
                              bool HasExplicitTemplateArgs,
                              const TemplateArgumentLoc *ExplicitTemplateArgs,
                              unsigned NumExplicitTemplateArgs,
                              QualType ArgFunctionType,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  QualType FunctionType = Function->getType();

  // Substitute any explicit template arguments.
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  llvm::SmallVector<QualType, 4> ParamTypes;
  if (HasExplicitTemplateArgs) {
    if (TemplateDeductionResult Result
          = SubstituteExplicitTemplateArguments(FunctionTemplate,
                                                ExplicitTemplateArgs,
                                                NumExplicitTemplateArgs,
                                                Deduced, ParamTypes,
                                                &FunctionType, Info))
      return Result;
  }

  // Template argument deduction for function templates in a SFINAE context.
  // Trap any errors that might occur.
  SFINAETrap Trap(*this);

  // Deduce template arguments from the function type.
  Deduced.resize(TemplateParams->size());
  if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(Context, TemplateParams,
                                    FunctionType, ArgFunctionType, Info,
                                    Deduced, 0))
    return Result;

  return FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
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
  llvm::SmallVector<TemplateArgument, 4> Deduced;
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
        = ::DeduceTemplateArguments(Context, TemplateParams,
                                    P, A, Info, Deduced, TDF))
    return Result;

  // FIXME: we need to check that the deduced A is the same as A,
  // modulo the various allowed differences.

  // Finish template argument deduction.
  FunctionDecl *Spec = 0;
  TemplateDeductionResult Result
    = FinishTemplateArgumentDeduction(FunctionTemplate, Deduced, Spec, Info);
  Specialization = cast_or_null<CXXConversionDecl>(Spec);
  return Result;
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
/// \param Context the AST context in which this deduction occurs.
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
DeduceTemplateArgumentsDuringPartialOrdering(ASTContext &Context,
                                          TemplateParameterList *TemplateParams,
                                             QualType ParamIn, QualType ArgIn,
                                             Sema::TemplateDeductionInfo &Info,
                             llvm::SmallVectorImpl<TemplateArgument> &Deduced,
    llvm::SmallVectorImpl<DeductionQualifierComparison> *QualifierComparisons) {
  CanQualType Param = Context.getCanonicalType(ParamIn);
  CanQualType Arg = Context.getCanonicalType(ArgIn);

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
  return DeduceTemplateArguments(Context, TemplateParams, Param, Arg, Info,
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
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  Deduced.resize(TemplateParams->size());

  // C++0x [temp.deduct.partial]p3:
  //   The types used to determine the ordering depend on the context in which
  //   the partial ordering is done:
  Sema::TemplateDeductionInfo Info(S.Context);
  switch (TPOC) {
  case TPOC_Call: {
    //   - In the context of a function call, the function parameter types are
    //     used.
    unsigned NumParams = std::min(Proto1->getNumArgs(), Proto2->getNumArgs());
    for (unsigned I = 0; I != NumParams; ++I)
      if (DeduceTemplateArgumentsDuringPartialOrdering(S.Context,
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
    if (DeduceTemplateArgumentsDuringPartialOrdering(S.Context,
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
    if (DeduceTemplateArgumentsDuringPartialOrdering(S.Context,
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
                                 TemplatePartialOrderingContext TPOC) {
  llvm::SmallVector<DeductionQualifierComparison, 4> QualifierComparisons;
  bool Better1 = isAtLeastAsSpecializedAs(*this, FT1, FT2, TPOC, 0);
  bool Better2 = isAtLeastAsSpecializedAs(*this, FT2, FT1, TPOC, 
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
/// \param Specializations the set of function template specializations that
/// we will be comparing.
///
/// \param NumSpecializations the number of function template specializations in
/// \p Specializations
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
/// found. Otherwise, returns NULL.
///
/// \todo FIXME: Consider passing in the "also-ran" candidates that failed 
/// template argument deduction.
FunctionDecl *Sema::getMostSpecialized(FunctionDecl **Specializations,
                                       unsigned NumSpecializations,
                                       TemplatePartialOrderingContext TPOC,
                                       SourceLocation Loc,
                                       const PartialDiagnostic &NoneDiag,
                                       const PartialDiagnostic &AmbigDiag,
                                       const PartialDiagnostic &CandidateDiag,
                                       unsigned *Index) {
  if (NumSpecializations == 0) {
    Diag(Loc, NoneDiag);
    return 0;
  }
  
  if (NumSpecializations == 1) {
    if (Index)
      *Index = 0;
    
    return Specializations[0];
  }
    
  
  // Find the function template that is better than all of the templates it
  // has been compared to.
  unsigned Best = 0;
  FunctionTemplateDecl *BestTemplate 
    = Specializations[Best]->getPrimaryTemplate();
  assert(BestTemplate && "Not a function template specialization?");
  for (unsigned I = 1; I != NumSpecializations; ++I) {
    FunctionTemplateDecl *Challenger = Specializations[I]->getPrimaryTemplate();
    assert(Challenger && "Not a function template specialization?");
    if (isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger, 
                                                  TPOC),
                       Challenger)) {
      Best = I;
      BestTemplate = Challenger;
    }
  }
  
  // Make sure that the "best" function template is more specialized than all
  // of the others.
  bool Ambiguous = false;
  for (unsigned I = 0; I != NumSpecializations; ++I) {
    FunctionTemplateDecl *Challenger = Specializations[I]->getPrimaryTemplate();
    if (I != Best &&
        !isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger, 
                                                  TPOC),
                        BestTemplate)) {
      Ambiguous = true;
      break;
    }
  }
  
  if (!Ambiguous) {
    // We found an answer. Return it.
    if (Index)
      *Index = Best;
    return Specializations[Best];
  }
  
  // Diagnose the ambiguity.
  Diag(Loc, AmbigDiag);
  
  // FIXME: Can we order the candidates in some sane way?
  for (unsigned I = 0; I != NumSpecializations; ++I)
    Diag(Specializations[I]->getLocation(), CandidateDiag)
      << getTemplateArgumentBindingsText(
            Specializations[I]->getPrimaryTemplate()->getTemplateParameters(),
                         *Specializations[I]->getTemplateSpecializationArgs());
  
  return 0;
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
                                  ClassTemplatePartialSpecializationDecl *PS2) {
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
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  Sema::TemplateDeductionInfo Info(Context);
  
  // Determine whether PS1 is at least as specialized as PS2
  Deduced.resize(PS2->getTemplateParameters()->size());
  bool Better1 = !DeduceTemplateArgumentsDuringPartialOrdering(Context,
                                                  PS2->getTemplateParameters(),
                                                  Context.getTypeDeclType(PS2),
                                                  Context.getTypeDeclType(PS1),
                                                               Info,
                                                               Deduced,
                                                               0);

  // Determine whether PS2 is at least as specialized as PS1
  Deduced.clear();
  Deduced.resize(PS1->getTemplateParameters()->size());
  bool Better2 = !DeduceTemplateArgumentsDuringPartialOrdering(Context,
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
  if (!E)
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

  case Type::Typename:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(SemaRef,
                                 cast<TypenameType>(T)->getQualifier(),
                                 OnlyDeduced, Depth, Used);
    break;

  // None of these types have any template parameters in them.
  case Type::Builtin:
  case Type::FixedWidthInt:
  case Type::VariableArray:
  case Type::FunctionNoProto:
  case Type::Record:
  case Type::Enum:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
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
void Sema::MarkDeducedTemplateParameters(FunctionTemplateDecl *FunctionTemplate,
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
