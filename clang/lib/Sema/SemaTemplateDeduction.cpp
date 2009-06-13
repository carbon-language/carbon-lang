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
                              llvm::APInt Value,
                              Sema::TemplateDeductionInfo &Info,
                              llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 && 
         "Cannot deduce non-type template argument with depth > 0");
  
  if (Deduced[NTTP->getIndex()].isNull()) {
    Deduced[NTTP->getIndex()] = TemplateArgument(SourceLocation(), 
                                                 llvm::APSInt(Value),
                                                 NTTP->getType());
    return Sema::TDK_Success;
  }
  
  assert(Deduced[NTTP->getIndex()].getKind() == TemplateArgument::Integral);
  
  // If the template argument was previously deduced to a negative value, 
  // then our deduction fails.
  const llvm::APSInt *PrevValuePtr = Deduced[NTTP->getIndex()].getAsIntegral();
  if (PrevValuePtr->isSigned() && PrevValuePtr->isNegative()) {
    // FIXME: This is wacky; we should be dealing with APSInts and
    // checking the actual signs.
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(SourceLocation(), 
                                      llvm::APSInt(Value),
                                      NTTP->getType());
    return Sema::TDK_Inconsistent;
  }

  llvm::APInt PrevValue = *PrevValuePtr;
  if (Value.getBitWidth() > PrevValue.getBitWidth())
    PrevValue.zext(Value.getBitWidth());
  else if (Value.getBitWidth() < PrevValue.getBitWidth())
    Value.zext(PrevValue.getBitWidth());

  if (Value != PrevValue) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = TemplateArgument(SourceLocation(), 
                                      llvm::APSInt(Value),
                                      NTTP->getType());
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
  
  // FIXME: Compare the expressions for equality!
  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(ASTContext &Context,
                        TemplateName Param,
                        TemplateName Arg,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  // FIXME: Implement template argument deduction for template
  // template parameters.

  // FIXME: this routine does not have enough information to produce
  // good diagnostics.

  TemplateDecl *ParamDecl = Param.getAsTemplateDecl();
  TemplateDecl *ArgDecl = Arg.getAsTemplateDecl();
  
  if (!ParamDecl || !ArgDecl) {
    // FIXME: fill in Info.Param/Info.FirstArg
    return Sema::TDK_Inconsistent;
  }

  ParamDecl = cast<TemplateDecl>(Context.getCanonicalDecl(ParamDecl));
  ArgDecl = cast<TemplateDecl>(Context.getCanonicalDecl(ArgDecl));
  if (ParamDecl != ArgDecl) {
    // FIXME: fill in Info.Param/Info.FirstArg
    return Sema::TDK_Inconsistent;
  }

  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(ASTContext &Context, 
                        TemplateParameterList *TemplateParams,
                        QualType ParamIn, QualType ArgIn,
                        Sema::TemplateDeductionInfo &Info,
                        llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  // We only want to look at the canonical types, since typedefs and
  // sugar are not part of template argument deduction.
  QualType Param = Context.getCanonicalType(ParamIn);
  QualType Arg = Context.getCanonicalType(ArgIn);

  // If the parameter type is not dependent, just compare the types
  // directly.
  if (!Param->isDependentType()) {
    if (Param == Arg)
      return Sema::TDK_Success;

    Info.FirstArg = TemplateArgument(SourceLocation(), ParamIn);
    Info.SecondArg = TemplateArgument(SourceLocation(), ArgIn);
    return Sema::TDK_NonDeducedMismatch;
  }

  // C++ [temp.deduct.type]p9:
  //   A template type argument T, a template template argument TT or a 
  //   template non-type argument i can be deduced if P and A have one of 
  //   the following forms:
  //
  //     T
  //     cv-list T
  if (const TemplateTypeParmType *TemplateTypeParm 
        = Param->getAsTemplateTypeParmType()) {
    unsigned Index = TemplateTypeParm->getIndex();

    // The argument type can not be less qualified than the parameter
    // type.
    if (Param.isMoreQualifiedThan(Arg)) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = Deduced[Index];
      Info.SecondArg = TemplateArgument(SourceLocation(), Arg);
      return Sema::TDK_InconsistentQuals;
    }

    assert(TemplateTypeParm->getDepth() == 0 && "Can't deduce with depth > 0");
	  
    unsigned Quals = Arg.getCVRQualifiers() & ~Param.getCVRQualifiers();
    QualType DeducedType = Arg.getQualifiedType(Quals);

    if (Deduced[Index].isNull())
      Deduced[Index] = TemplateArgument(SourceLocation(), DeducedType);
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
        Info.SecondArg = TemplateArgument(SourceLocation(), Arg);
        return Sema::TDK_Inconsistent;
      }
    }
    return Sema::TDK_Success;
  }

  // Set up the template argument deduction information for a failure.
  Info.FirstArg = TemplateArgument(SourceLocation(), ParamIn);
  Info.SecondArg = TemplateArgument(SourceLocation(), ArgIn);

  if (Param.getCVRQualifiers() != Arg.getCVRQualifiers())
    return Sema::TDK_NonDeducedMismatch;

  switch (Param->getTypeClass()) {
    // No deduction possible for these types
    case Type::Builtin:
      return Sema::TDK_NonDeducedMismatch;
      
    //     T *
    case Type::Pointer: {
      const PointerType *PointerArg = Arg->getAsPointerType();
      if (!PointerArg)
        return Sema::TDK_NonDeducedMismatch;
      
      return DeduceTemplateArguments(Context, TemplateParams,
                                   cast<PointerType>(Param)->getPointeeType(),
                                     PointerArg->getPointeeType(),
                                     Info, Deduced);
    }
      
    //     T &
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg = Arg->getAsLValueReferenceType();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;
      
      return DeduceTemplateArguments(Context, TemplateParams,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg = Arg->getAsRValueReferenceType();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;
      
      return DeduceTemplateArguments(Context, TemplateParams,
                           cast<RValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Info, Deduced);
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
                                     Info, Deduced);
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
                                     Info, Deduced);
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
                                      Info, Deduced))
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
            = dyn_cast<ConstantArrayType>(ArrayArg))
        return DeduceNonTypeTemplateArgument(Context, NTTP, 
                                             ConstantArrayArg->getSize(),
                                             Info, Deduced);
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
                                      Info, Deduced))
        return Result;
      
      for (unsigned I = 0, N = FunctionProtoParam->getNumArgs(); I != N; ++I) {
        // Check argument types.
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArguments(Context, TemplateParams,
                                        FunctionProtoParam->getArgType(I),
                                        FunctionProtoArg->getArgType(I),
                                        Info, Deduced))
          return Result;
      }
      
      return Sema::TDK_Success;
    }
     
    //     template-name<T> (wheretemplate-name refers to a class template)
    //     template-name<i>
    //     TT<T> (TODO)
    //     TT<i> (TODO)
    //     TT<> (TODO)
    case Type::TemplateSpecialization: {
      const TemplateSpecializationType *SpecParam
        = cast<TemplateSpecializationType>(Param);

      // Check whether the template argument is a dependent template-id.
      // FIXME: This is untested code; it can be tested when we implement
      // partial ordering of class template partial specializations.
      if (const TemplateSpecializationType *SpecArg 
            = dyn_cast<TemplateSpecializationType>(Arg)) {
        // Perform template argument deduction for the template name.
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArguments(Context,
                                        SpecParam->getTemplateName(),
                                        SpecArg->getTemplateName(),
                                        Info, Deduced))
          return Result;
            
        unsigned NumArgs = SpecParam->getNumArgs();

        // FIXME: When one of the template-names refers to a
        // declaration with default template arguments, do we need to
        // fill in those default template arguments here? Most likely,
        // the answer is "yes", but I don't see any references. This
        // issue may be resolved elsewhere, because we may want to
        // instantiate default template arguments when
        if (SpecArg->getNumArgs() != NumArgs)
          return Sema::TDK_NonDeducedMismatch;

        // Perform template argument deduction on each template
        // argument.
        for (unsigned I = 0; I != NumArgs; ++I)
          if (Sema::TemplateDeductionResult Result
                = DeduceTemplateArguments(Context, TemplateParams,
                                          SpecParam->getArg(I),
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
                                      SpecParam->getTemplateName(),
                              TemplateName(SpecArg->getSpecializedTemplate()),
                                      Info, Deduced))
          return Result;

      // FIXME: Can the # of arguments in the parameter and the argument differ?
      unsigned NumArgs = SpecParam->getNumArgs();
      const TemplateArgumentList &ArgArgs = SpecArg->getTemplateArgs();
      if (NumArgs != ArgArgs.size())
        return Sema::TDK_NonDeducedMismatch;

      for (unsigned I = 0; I != NumArgs; ++I)
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArguments(Context, TemplateParams,
                                        SpecParam->getArg(I),
                                        ArgArgs.get(I),
                                        Info, Deduced))
          return Result;
      
      return Sema::TDK_Success;
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
                                      Info, Deduced))
        return Result;

      return DeduceTemplateArguments(Context, TemplateParams,
                                     QualType(MemPtrParam->getClass(), 0),
                                     QualType(MemPtrArg->getClass(), 0),
                                     Info, Deduced);
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
                                     Deduced);
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
  return Sema::TDK_NonDeducedMismatch;
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
    assert(Arg.getKind() == TemplateArgument::Type && "Type/value mismatch");
    return DeduceTemplateArguments(Context, TemplateParams,
                                   Param.getAsType(), 
                                   Arg.getAsType(), Info, Deduced);

  case TemplateArgument::Declaration:
    // FIXME: Implement this check
    assert(false && "Unimplemented template argument deduction case");
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
  TemplateArgumentListBuilder Builder(Context);
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    if (Deduced[I].isNull()) {
      Decl *Param 
        = const_cast<Decl *>(Partial->getTemplateParameters()->getParam(I));
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
        Info.Param = TTP;
      else if (NonTypeTemplateParmDecl *NTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(Param))
        Info.Param = NTTP;
      else
        Info.Param = cast<TemplateTemplateParmDecl>(Param);
      return TDK_Incomplete;
    }

    Builder.push_back(Deduced[I]);
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList 
    = new (Context) TemplateArgumentList(Context, Builder, /*CopyArgs=*/true,
                                         /*FlattenArgs=*/true);
  Info.reset(DeducedArgumentList);

  // Now that we have all of the deduced template arguments, take
  // another pass through them to convert any integral template
  // arguments to the appropriate type.
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    TemplateArgument &Arg = Deduced[I];    
    if (Arg.getKind() == TemplateArgument::Integral) {
      const NonTypeTemplateParmDecl *Parm 
        = cast<NonTypeTemplateParmDecl>(Partial->getTemplateParameters()
                                          ->getParam(I));
      QualType T = InstantiateType(Parm->getType(), *DeducedArgumentList,
                                   Parm->getLocation(), Parm->getDeclName());
      if (T.isNull()) {
        Info.Param = const_cast<NonTypeTemplateParmDecl*>(Parm);
        Info.FirstArg = TemplateArgument(Parm->getLocation(), Parm->getType());
        return TDK_SubstitutionFailure;
      }
      
      // FIXME: Make sure we didn't overflow our data type!
      llvm::APSInt &Value = *Arg.getAsIntegral();
      unsigned AllowedBits = Context.getTypeSize(T);
      if (Value.getBitWidth() != AllowedBits)
        Value.extOrTrunc(AllowedBits);
      Value.setIsSigned(T->isSignedIntegerType());
      Arg.setIntegralType(T);
    }

    (*DeducedArgumentList)[I] = Arg;
  }

  // Substitute the deduced template arguments into the template
  // arguments of the class template partial specialization, and
  // verify that the instantiated template arguments are both valid
  // and are equivalent to the template arguments originally provided
  // to the class template. 
  ClassTemplateDecl *ClassTemplate = Partial->getSpecializedTemplate();
  const TemplateArgumentList &PartialTemplateArgs = Partial->getTemplateArgs();
  for (unsigned I = 0, N = PartialTemplateArgs.flat_size(); I != N; ++I) {
    Decl *Param = const_cast<Decl *>(
                    ClassTemplate->getTemplateParameters()->getParam(I));
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      TemplateArgument InstArg = Instantiate(PartialTemplateArgs[I],
                                             *DeducedArgumentList);
      if (InstArg.getKind() != TemplateArgument::Type) {
        Info.Param = TTP;
        Info.FirstArg = PartialTemplateArgs[I];
        return TDK_SubstitutionFailure;
      }

      if (Context.getCanonicalType(InstArg.getAsType())
            != Context.getCanonicalType(TemplateArgs[I].getAsType())) {
        Info.Param = TTP;
        Info.FirstArg = TemplateArgs[I];
        Info.SecondArg = InstArg;
        return TDK_NonDeducedMismatch;
      }

      continue;
    }

    // FIXME: Check template template arguments?
  }

  return TDK_Success;
}

static void 
MarkDeducedTemplateParameters(Sema &SemaRef,
                              const TemplateArgument &TemplateArg,
                              llvm::SmallVectorImpl<bool> &Deduced);

/// \brief Mark the template arguments that are deduced by the given
/// expression.
static void 
MarkDeducedTemplateParameters(Expr *E, llvm::SmallVectorImpl<bool> &Deduced) {
  DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
  if (!E)
    return;

  NonTypeTemplateParmDecl *NTTP 
    = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl());
  if (!NTTP)
    return;

  Deduced[NTTP->getIndex()] = true;
}

/// \brief Mark the template parameters that are deduced by the given
/// type.
static void 
MarkDeducedTemplateParameters(Sema &SemaRef, QualType T,
                              llvm::SmallVectorImpl<bool> &Deduced) {
  // Non-dependent types have nothing deducible
  if (!T->isDependentType())
    return;

  T = SemaRef.Context.getCanonicalType(T);
  switch (T->getTypeClass()) {
  case Type::ExtQual:
    MarkDeducedTemplateParameters(SemaRef, 
                QualType(cast<ExtQualType>(T.getTypePtr())->getBaseType(), 0),
                                  Deduced);
    break;

  case Type::Pointer:
    MarkDeducedTemplateParameters(SemaRef,
                          cast<PointerType>(T.getTypePtr())->getPointeeType(),
                                  Deduced);
    break;

  case Type::BlockPointer:
    MarkDeducedTemplateParameters(SemaRef,
                     cast<BlockPointerType>(T.getTypePtr())->getPointeeType(),
                                  Deduced);
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    MarkDeducedTemplateParameters(SemaRef,
                        cast<ReferenceType>(T.getTypePtr())->getPointeeType(),
                                  Deduced);
    break;

  case Type::MemberPointer: {
    const MemberPointerType *MemPtr = cast<MemberPointerType>(T.getTypePtr());
    MarkDeducedTemplateParameters(SemaRef, MemPtr->getPointeeType(), Deduced);
    MarkDeducedTemplateParameters(SemaRef, QualType(MemPtr->getClass(), 0),
                                  Deduced);
    break;
  }

  case Type::DependentSizedArray:
    MarkDeducedTemplateParameters(
                 cast<DependentSizedArrayType>(T.getTypePtr())->getSizeExpr(),
                                  Deduced);
    // Fall through to check the element type

  case Type::ConstantArray:
  case Type::IncompleteArray:
    MarkDeducedTemplateParameters(SemaRef,
                            cast<ArrayType>(T.getTypePtr())->getElementType(),
                                  Deduced);
    break;

  case Type::Vector:
  case Type::ExtVector:
    MarkDeducedTemplateParameters(SemaRef,
                           cast<VectorType>(T.getTypePtr())->getElementType(),
                                  Deduced);
    break;

  case Type::FunctionProto: {
    const FunctionProtoType *Proto = cast<FunctionProtoType>(T.getTypePtr());
    MarkDeducedTemplateParameters(SemaRef, Proto->getResultType(), Deduced);
    for (unsigned I = 0, N = Proto->getNumArgs(); I != N; ++I)
      MarkDeducedTemplateParameters(SemaRef, Proto->getArgType(I), Deduced);
    break;
  }

  case Type::TemplateTypeParm:
    Deduced[cast<TemplateTypeParmType>(T.getTypePtr())->getIndex()] = true;
    break;

  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *Spec 
      = cast<TemplateSpecializationType>(T.getTypePtr());
    if (TemplateDecl *Template = Spec->getTemplateName().getAsTemplateDecl())
      if (TemplateTemplateParmDecl *TTP 
            = dyn_cast<TemplateTemplateParmDecl>(Template))
        Deduced[TTP->getIndex()] = true;
      
      for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
        MarkDeducedTemplateParameters(SemaRef, Spec->getArg(I), Deduced);

    break;
  }

  // None of these types have any deducible parts.
  case Type::Builtin:
  case Type::FixedWidthInt:
  case Type::Complex:
  case Type::VariableArray:
  case Type::FunctionNoProto:
  case Type::Record:
  case Type::Enum:
  case Type::Typename:
  case Type::ObjCInterface:
  case Type::ObjCQualifiedInterface:
  case Type::ObjCQualifiedId:
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    break;
  }
}

/// \brief Mark the template parameters that are deduced by this
/// template argument.
static void 
MarkDeducedTemplateParameters(Sema &SemaRef,
                              const TemplateArgument &TemplateArg,
                              llvm::SmallVectorImpl<bool> &Deduced) {
  switch (TemplateArg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
    break;
    
  case TemplateArgument::Type:
    MarkDeducedTemplateParameters(SemaRef, TemplateArg.getAsType(), Deduced);
    break;

  case TemplateArgument::Declaration:
    if (TemplateTemplateParmDecl *TTP 
        = dyn_cast<TemplateTemplateParmDecl>(TemplateArg.getAsDecl()))
      Deduced[TTP->getIndex()] = true;
    break;

  case TemplateArgument::Expression:
    MarkDeducedTemplateParameters(TemplateArg.getAsExpr(), Deduced);
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
Sema::MarkDeducedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                    llvm::SmallVectorImpl<bool> &Deduced) {
  for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
    ::MarkDeducedTemplateParameters(*this, TemplateArgs[I], Deduced);
}
