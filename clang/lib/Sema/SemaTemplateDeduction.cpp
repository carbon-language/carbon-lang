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

  // FIXME: Use a visitor or switch to handle all of the kinds of
  // types that the parameter may be.
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
    case Type::Pointer: {
      const PointerType *PointerArg = Arg->getAsPointerType();
      if (!PointerArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                                   cast<PointerType>(Param)->getPointeeType(),
                                     PointerArg->getPointeeType(),
                                     Deduced);
    }
      
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg = Arg->getAsLValueReferenceType();
      if (!ReferenceArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Deduced);
    }

    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg = Arg->getAsRValueReferenceType();
      if (!ReferenceArg)
        return false;
      
      return DeduceTemplateArguments(Context,
                           cast<RValueReferenceType>(Param)->getPointeeType(),
                                     ReferenceArg->getPointeeType(),
                                     Deduced);
    }
      
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
  assert(Param.getKind() == Arg.getKind() &&
         "Template argument kind mismatch during deduction");
  switch (Param.getKind()) {
  case TemplateArgument::Type: 
    return DeduceTemplateArguments(Context, Param.getAsType(), 
                                   Arg.getAsType(), Deduced);

  default:
    return false;
  }
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


bool 
Sema::DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                              const TemplateArgumentList &TemplateArgs) {
  llvm::SmallVector<TemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  return ::DeduceTemplateArguments(Context, Partial->getTemplateArgs(), 
                                  TemplateArgs, Deduced);
}
