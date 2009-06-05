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
using namespace clang;

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
///
/// \returns true if deduction succeeded, false otherwise.
static bool DeduceNonTypeTemplateArgument(ASTContext &Context, 
                                          NonTypeTemplateParmDecl *NTTP, 
                                          llvm::APInt Value,
                             llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 && 
         "Cannot deduce non-type template argument with depth > 0");
  
  if (Deduced[NTTP->getIndex()].isNull()) {
    Deduced[NTTP->getIndex()] = TemplateArgument(SourceLocation(), 
                                                 llvm::APSInt(Value),
                                                 NTTP->getType());
    return true;
  }
  
  if (Deduced[NTTP->getIndex()].getKind() != TemplateArgument::Integral)
    return false;
  
  // If the template argument was previously deduced to a negative value, 
  // then our deduction fails.
  const llvm::APSInt *PrevValuePtr = Deduced[NTTP->getIndex()].getAsIntegral();
  assert(PrevValuePtr && "Not an integral template argument?");
  if (PrevValuePtr->isSigned() && PrevValuePtr->isNegative())
    return false;
  
  llvm::APInt PrevValue = *PrevValuePtr;
  if (Value.getBitWidth() > PrevValue.getBitWidth())
    PrevValue.zext(Value.getBitWidth());
  else if (Value.getBitWidth() < PrevValue.getBitWidth())
    Value.zext(PrevValue.getBitWidth());
  return Value == PrevValue;
}

/// \brief Deduce the value of the given non-type template parameter 
/// from the given type- or value-dependent expression.
///
/// \returns true if deduction succeeded, false otherwise.

static bool DeduceNonTypeTemplateArgument(ASTContext &Context, 
                                          NonTypeTemplateParmDecl *NTTP,
                                          Expr *Value,
                            llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 && 
         "Cannot deduce non-type template argument with depth > 0");
  assert((Value->isTypeDependent() || Value->isValueDependent()) &&
         "Expression template argument must be type- or value-dependent.");
  
  if (Deduced[NTTP->getIndex()].isNull()) {
    // FIXME: Clone the Value?
    Deduced[NTTP->getIndex()] = TemplateArgument(Value);
    return true;
  }
  
  if (Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Integral) {
    // Okay, we deduced a constant in one case and a dependent expression 
    // in another case. FIXME: Later, we will check that instantiating the 
    // dependent expression gives us the constant value.
    return true;
  }
  
  // FIXME: Compare the expressions for equality!
  return true;
}

static bool DeduceTemplateArguments(ASTContext &Context, QualType Param, 
                                    QualType Arg,
                             llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  // We only want to look at the canonical types, since typedefs and
  // sugar are not part of template argument deduction.
  Param = Context.getCanonicalType(Param);
  Arg = Context.getCanonicalType(Arg);

  // If the parameter type is not dependent, just compare the types
  // directly.
  if (!Param->isDependentType())
    return Param == Arg;

  // C++ [temp.deduct.type]p9:
  //
  //   A template type argument T, a template template argument TT or a 
  //   template non-type argument i can be deduced if P and A have one of 
  //   the following forms:
  //
  //     T
  //     cv-list T
  if (const TemplateTypeParmType *TemplateTypeParm 
        = Param->getAsTemplateTypeParmType()) {
    // The argument type can not be less qualified than the parameter
    // type.
    if (Param.isMoreQualifiedThan(Arg))
      return false;

    assert(TemplateTypeParm->getDepth() == 0 && "Can't deduce with depth > 0");
	  
    unsigned Quals = Arg.getCVRQualifiers() & ~Param.getCVRQualifiers();
    QualType DeducedType = Arg.getQualifiedType(Quals);
	  unsigned Index = TemplateTypeParm->getIndex();

    if (Deduced[Index].isNull())
      Deduced[Index] = TemplateArgument(SourceLocation(), DeducedType);
    else {
      // C++ [temp.deduct.type]p2: 
      //   [...] If type deduction cannot be done for any P/A pair, or if for
      //   any pair the deduction leads to more than one possible set of 
      //   deduced values, or if different pairs yield different deduced 
      //   values, or if any template argument remains neither deduced nor 
      //   explicitly specified, template argument deduction fails.
      if (Deduced[Index].getAsType() != DeducedType)
        return false;
    }
    return true;
  }

  if (Param.getCVRQualifiers() != Arg.getCVRQualifiers())
    return false;

  switch (Param->getTypeClass()) {
    // No deduction possible for these types
    case Type::Builtin:
      return false;
      
      
    //     T *
    case Type::Pointer: {
      const PointerType *PointerArg = Arg->getAsPointerType();
      if (!PointerArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                                   cast<PointerType>(Param)->getPointeeType(),
                                     PointerArg->getPointeeType(),
                                     Deduced);
    }
      
    //     T &
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg = Arg->getAsLValueReferenceType();
      if (!ReferenceArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Deduced);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg = Arg->getAsRValueReferenceType();
      if (!ReferenceArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                           cast<RValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Deduced);
    }
      
    //     T [] (implied, but not stated explicitly)
    case Type::IncompleteArray: {
      const IncompleteArrayType *IncompleteArrayArg = 
        Context.getAsIncompleteArrayType(Arg);
      if (!IncompleteArrayArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                     Context.getAsIncompleteArrayType(Param)->getElementType(),
                                     IncompleteArrayArg->getElementType(),
                                     Deduced);
    }

    //     T [integer-constant]
    case Type::ConstantArray: {
      const ConstantArrayType *ConstantArrayArg = 
        Context.getAsConstantArrayType(Arg);
      if (!ConstantArrayArg)
        return false;
      
      const ConstantArrayType *ConstantArrayParm = 
        Context.getAsConstantArrayType(Param);
      if (ConstantArrayArg->getSize() != ConstantArrayParm->getSize())
        return false;
      
      return DeduceTemplateArguments(Context,
                                     ConstantArrayParm->getElementType(),
                                     ConstantArrayArg->getElementType(),
                                     Deduced);
    }

    //     type [i]
    case Type::DependentSizedArray: {
      const ArrayType *ArrayArg = dyn_cast<ArrayType>(Arg);
      if (!ArrayArg)
        return false;
      
      // Check the element type of the arrays
      const DependentSizedArrayType *DependentArrayParm
        = cast<DependentSizedArrayType>(Param);
      if (!DeduceTemplateArguments(Context,
                                   DependentArrayParm->getElementType(),
                                   ArrayArg->getElementType(),
                                   Deduced))
        return false;
          
      // Determine the array bound is something we can deduce.
      NonTypeTemplateParmDecl *NTTP 
        = getDeducedParameterFromExpr(DependentArrayParm->getSizeExpr());
      if (!NTTP)
        return true;
      
      // We can perform template argument deduction for the given non-type 
      // template parameter.
      assert(NTTP->getDepth() == 0 && 
             "Cannot deduce non-type template argument at depth > 0");
      if (const ConstantArrayType *ConstantArrayArg 
            = dyn_cast<ConstantArrayType>(ArrayArg))
        return DeduceNonTypeTemplateArgument(Context, NTTP, 
                                             ConstantArrayArg->getSize(),
                                             Deduced);
      if (const DependentSizedArrayType *DependentArrayArg
            = dyn_cast<DependentSizedArrayType>(ArrayArg))
        return DeduceNonTypeTemplateArgument(Context, NTTP,
                                             DependentArrayArg->getSizeExpr(),
                                             Deduced);
      
      // Incomplete type does not match a dependently-sized array type
      return false;
    }
      
    default:
      break;
  }

  // FIXME: Many more cases to go (to go).
  return false;
}

static bool
DeduceTemplateArguments(ASTContext &Context, const TemplateArgument &Param,
                        const TemplateArgument &Arg,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  switch (Param.getKind()) {
  case TemplateArgument::Null:
    assert(false && "Null template argument in parameter list");
    break;
      
  case TemplateArgument::Type: 
    assert(Arg.getKind() == TemplateArgument::Type && "Type/value mismatch");
    return DeduceTemplateArguments(Context, Param.getAsType(), 
                                   Arg.getAsType(), Deduced);

  case TemplateArgument::Declaration:
    // FIXME: Implement this check
    assert(false && "Unimplemented template argument deduction case");
    return false;
      
  case TemplateArgument::Integral:
    if (Arg.getKind() == TemplateArgument::Integral) {
      // FIXME: Zero extension + sign checking here?
      return *Param.getAsIntegral() == *Arg.getAsIntegral();
    }
    if (Arg.getKind() == TemplateArgument::Expression)
      return false;

    assert(false && "Type/value mismatch");
    return false;
      
  case TemplateArgument::Expression: {
    if (NonTypeTemplateParmDecl *NTTP 
          = getDeducedParameterFromExpr(Param.getAsExpr())) {
      if (Arg.getKind() == TemplateArgument::Integral)
        // FIXME: Sign problems here
        return DeduceNonTypeTemplateArgument(Context, NTTP, 
                                             *Arg.getAsIntegral(), Deduced);
      if (Arg.getKind() == TemplateArgument::Expression)
        return DeduceNonTypeTemplateArgument(Context, NTTP, Arg.getAsExpr(),
                                             Deduced);
      
      assert(false && "Type/value mismatch");
      return false;
    }
    
    // Can't deduce anything, but that's okay.
    return true;
  }
  }
      
  return true;
}

static bool 
DeduceTemplateArguments(ASTContext &Context,
                        const TemplateArgumentList &ParamList,
                        const TemplateArgumentList &ArgList,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(ParamList.size() == ArgList.size());
  for (unsigned I = 0, N = ParamList.size(); I != N; ++I) {
    if (!DeduceTemplateArguments(Context, ParamList[I], ArgList[I], Deduced))
      return false;
  }
  return true;
}


TemplateArgumentList * 
Sema::DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                              const TemplateArgumentList &TemplateArgs) {
  // Deduce the template arguments for the partial specialization
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (! ::DeduceTemplateArguments(Context, Partial->getTemplateArgs(), 
                                  TemplateArgs, Deduced))
    return 0;
  
  // FIXME: Substitute the deduced template arguments into the template
  // arguments of the class template partial specialization; the resulting
  // template arguments should match TemplateArgs exactly.
  
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    TemplateArgument &Arg = Deduced[I];

    // FIXME: If this template argument was not deduced, but the corresponding
    // template parameter has a default argument, instantiate the default
    // argument.
    if (Arg.isNull()) // FIXME: Result->Destroy(Context);
      return 0;
    
    if (Arg.getKind() == TemplateArgument::Integral) {
      // FIXME: Instantiate the type, but we need some context!
      const NonTypeTemplateParmDecl *Parm 
        = cast<NonTypeTemplateParmDecl>(Partial->getTemplateParameters()
                                          ->getParam(I));
      //      QualType T = InstantiateType(Parm->getType(), *Result,
      //                                   Parm->getLocation(), Parm->getDeclName());
      //      if (T.isNull()) // FIXME: Result->Destroy(Context);
      //        return 0;
      QualType T = Parm->getType();
      
      // FIXME: Make sure we didn't overflow our data type!
      llvm::APSInt &Value = *Arg.getAsIntegral();
      unsigned AllowedBits = Context.getTypeSize(T);
      if (Value.getBitWidth() != AllowedBits)
        Value.extOrTrunc(AllowedBits);
      Value.setIsSigned(T->isSignedIntegerType());
      Arg.setIntegralType(T);
    }
  }
  
  return new (Context) TemplateArgumentList(Context, Deduced.data(),
                                            Deduced.size(), /*CopyArgs=*/true);
}
