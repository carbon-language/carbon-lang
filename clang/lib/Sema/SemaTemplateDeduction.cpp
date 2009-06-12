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

static bool
DeduceTemplateArguments(ASTContext &Context, const TemplateArgument &Param,
                        const TemplateArgument &Arg,
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

static bool DeduceTemplateArguments(ASTContext &Context,
                                    TemplateName Param,
                                    TemplateName Arg,
                             llvm::SmallVectorImpl<TemplateArgument> &Deduced) {
  // FIXME: Implement template argument deduction for template
  // template parameters.

  TemplateDecl *ParamDecl = Param.getAsTemplateDecl();
  TemplateDecl *ArgDecl = Arg.getAsTemplateDecl();
  
  if (!ParamDecl || !ArgDecl)
    return false;

  ParamDecl = cast<TemplateDecl>(Context.getCanonicalDecl(ParamDecl));
  ArgDecl = cast<TemplateDecl>(Context.getCanonicalDecl(ArgDecl));
  return ParamDecl == ArgDecl;
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
      
    //     type(*)(T) 
    //     T(*)() 
    //     T(*)(T) 
    case Type::FunctionProto: {
      const FunctionProtoType *FunctionProtoArg = 
        dyn_cast<FunctionProtoType>(Arg);
      if (!FunctionProtoArg)
        return false;
      
      const FunctionProtoType *FunctionProtoParam = 
        cast<FunctionProtoType>(Param);

      if (FunctionProtoParam->getTypeQuals() != 
          FunctionProtoArg->getTypeQuals())
        return false;
      
      if (FunctionProtoParam->getNumArgs() != FunctionProtoArg->getNumArgs())
        return false;
      
      if (FunctionProtoParam->isVariadic() != FunctionProtoArg->isVariadic())
        return false;

      // Check return types.
      if (!DeduceTemplateArguments(Context,
                                   FunctionProtoParam->getResultType(),
                                   FunctionProtoArg->getResultType(),
                                   Deduced))
        return false;
      
      for (unsigned I = 0, N = FunctionProtoParam->getNumArgs(); I != N; ++I) {
        // Check argument types.
        if (!DeduceTemplateArguments(Context,
                                     FunctionProtoParam->getArgType(I),
                                     FunctionProtoArg->getArgType(I),
                                     Deduced))
          return false;
      }
      
      return true;
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
        if (!DeduceTemplateArguments(Context,
                                     SpecParam->getTemplateName(),
                                     SpecArg->getTemplateName(),
                                     Deduced))
          return false;
            
        unsigned NumArgs = SpecParam->getNumArgs();

        // FIXME: When one of the template-names refers to a
        // declaration with default template arguments, do we need to
        // fill in those default template arguments here? Most likely,
        // the answer is "yes", but I don't see any references. This
        // issue may be resolved elsewhere, because we may want to
        // instantiate default template arguments when
        if (SpecArg->getNumArgs() != NumArgs)
          return false;

        // Perform template argument deduction on each template
        // argument.
        for (unsigned I = 0; I != NumArgs; ++I)
          if (!DeduceTemplateArguments(Context,
                                       SpecParam->getArg(I),
                                       SpecArg->getArg(I),
                                       Deduced))
            return false;

        return true;
      } 

      // If the argument type is a class template specialization, we
      // perform template argument deduction using its template
      // arguments.
      const RecordType *RecordArg = dyn_cast<RecordType>(Arg);
      if (!RecordArg)
        return false;

      ClassTemplateSpecializationDecl *SpecArg 
        = dyn_cast<ClassTemplateSpecializationDecl>(RecordArg->getDecl());
      if (!SpecArg)
        return false;

      // Perform template argument deduction for the template name.
      if (!DeduceTemplateArguments(Context,
                                   SpecParam->getTemplateName(),
                                   TemplateName(SpecArg->getSpecializedTemplate()),
                                   Deduced))
          return false;

      // FIXME: Can the # of arguments in the parameter and the argument differ?
      unsigned NumArgs = SpecParam->getNumArgs();
      const TemplateArgumentList &ArgArgs = SpecArg->getTemplateArgs();
      if (NumArgs != ArgArgs.size())
        return false;

      for (unsigned I = 0; I != NumArgs; ++I)
        if (!DeduceTemplateArguments(Context,
                                     SpecParam->getArg(I),
                                     ArgArgs.get(I),
                                     Deduced))
          return false;
      
      return true;
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
        return false;

      return DeduceTemplateArguments(Context,
                                     MemPtrParam->getPointeeType(),
                                     MemPtrArg->getPointeeType(),
                                     Deduced) &&
        DeduceTemplateArguments(Context,
                                QualType(MemPtrParam->getClass(), 0),
                                QualType(MemPtrArg->getClass(), 0),
                                Deduced);
    }

    //     type(^)(T) 
    //     T(^)() 
    //     T(^)(T) 
    case Type::BlockPointer: {
      const BlockPointerType *BlockPtrParam = cast<BlockPointerType>(Param);
      const BlockPointerType *BlockPtrArg = dyn_cast<BlockPointerType>(Arg);
      
      if (!BlockPtrArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                                     BlockPtrParam->getPointeeType(),
                                     BlockPtrArg->getPointeeType(), Deduced);
    }

    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::Typename:
      // No template argument deduction for these types
      return true;

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

  // FIXME: It isn't clear whether we want the diagnostic to point at
  // the partial specialization itself or at the actual point of
  // instantiation.
  InstantiatingTemplate Inst(*this, Partial->getLocation(), Partial,
                             Deduced.data(), Deduced.size());
  if (Inst)
    return 0;

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  TemplateArgumentListBuilder Builder(Context);
  for (unsigned I = 0, N = Deduced.size(); I != N; ++I) {
    if (Deduced[I].isNull())
      return 0;

    Builder.push_back(Deduced[I]);
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList 
    = new (Context) TemplateArgumentList(Context, Builder, /*CopyArgs=*/true,
                                         /*FlattenArgs=*/true);

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
      if (T.isNull()) // FIXME: DeducedArgumentList->Destroy(Context);
        return 0;
      
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
    TemplateArgument InstArg = Instantiate(PartialTemplateArgs[I],
                                           *DeducedArgumentList);
    if (InstArg.isNull()) {
      // FIXME: DeducedArgumentList->Destroy(Context); (or use RAII)
      return 0;
    }

    Decl *Param 
      = const_cast<Decl *>(ClassTemplate->getTemplateParameters()->getParam(I));
    if (isa<TemplateTypeParmDecl>(Param)) {
      if (InstArg.getKind() != TemplateArgument::Type ||
          Context.getCanonicalType(InstArg.getAsType())
            != Context.getCanonicalType(TemplateArgs[I].getAsType()))
        // FIXME: DeducedArgumentList->Destroy(Context); (or use RAII)
        return 0;
    } else if (NonTypeTemplateParmDecl *NTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      QualType T = InstantiateType(NTTP->getType(), TemplateArgs,
                                   NTTP->getLocation(), NTTP->getDeclName());
      if (T.isNull())
        // FIXME: DeducedArgumentList->Destroy(Context); (or use RAII)
        return 0;

      if (InstArg.getKind() == TemplateArgument::Declaration ||
          InstArg.getKind() == TemplateArgument::Expression) {
        // Turn the template argument into an expression, so that we can
        // perform type checking on it and convert it to the type of the
        // non-type template parameter. FIXME: Will this expression be
        // leaked? It's hard to tell, since our ownership model for
        // expressions in template arguments is so poor.
        Expr *E = 0;
        if (InstArg.getKind() == TemplateArgument::Declaration) {
          NamedDecl *D = cast<NamedDecl>(InstArg.getAsDecl());
          QualType T = Context.OverloadTy;
          if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
            T = VD->getType().getNonReferenceType();
          E = new (Context) DeclRefExpr(D, T, InstArg.getLocation());
        } else {
          E = InstArg.getAsExpr();
        }

        // Check that the template argument can be used to initialize
        // the corresponding template parameter.
        if (CheckTemplateArgument(NTTP, T, E, InstArg))
          return 0;
      }

      switch (InstArg.getKind()) {
      case TemplateArgument::Null:
        assert(false && "Null template arguments cannot get here");
        return 0;

      case TemplateArgument::Type:
        assert(false && "Type/value mismatch");
        return 0;

      case TemplateArgument::Integral: {
        llvm::APSInt &Value = *InstArg.getAsIntegral();
        if (T->isIntegralType() || T->isEnumeralType()) {
          QualType IntegerType = Context.getCanonicalType(T);
          if (const EnumType *Enum = dyn_cast<EnumType>(IntegerType))
            IntegerType = Context.getCanonicalType(
                                           Enum->getDecl()->getIntegerType());

          // Check that an unsigned parameter does not receive a negative
          // value.
          if (IntegerType->isUnsignedIntegerType()
              && (Value.isSigned() && Value.isNegative()))
            return 0;

          // Check for truncation. If the number of bits in the
          // instantiated template argument exceeds what is allowed by
          // the type, template argument deduction fails.
          unsigned AllowedBits = Context.getTypeSize(IntegerType);
          if (Value.getActiveBits() > AllowedBits)
            return 0;

          if (Value.getBitWidth() != AllowedBits)
            Value.extOrTrunc(AllowedBits);
          Value.setIsSigned(IntegerType->isSignedIntegerType());

          // Check that the instantiated value is the same as the
          // value provided as a template argument.
          if (Value != *TemplateArgs[I].getAsIntegral())
            return 0;
        } else if (T->isPointerType() || T->isMemberPointerType()) {
          // Deal with NULL pointers that are used to initialize
          // pointer and pointer-to-member non-type template
          // parameters (C++0x).
          if (TemplateArgs[I].getAsDecl()) 
            return 0; // Not a NULL declaration

          // Check that the integral value is 0, the NULL pointer
          // constant.
          if (Value != 0)
            return 0;
        } else
          return 0;
        break;
      }

      case TemplateArgument::Declaration:
        if (Context.getCanonicalDecl(InstArg.getAsDecl())
              != Context.getCanonicalDecl(TemplateArgs[I].getAsDecl()))
          return 0;
        break;

      case TemplateArgument::Expression:
        // FIXME: Check equality of expressions
        break;
      }
    } else {
      assert(isa<TemplateTemplateParmDecl>(Param));
      // FIXME: Check template template arguments
    }
  }

  return DeducedArgumentList;
}
