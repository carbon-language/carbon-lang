//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CGCall.h"
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ParameterAttributes.h"
using namespace clang;
using namespace CodeGen;

/***/

static void 
constructParamAttrListInternal(const Decl *TargetDecl,
                               const llvm::SmallVector<QualType, 16> &ArgTypes,
                               ParamAttrListType &PAL) {
  unsigned FuncAttrs = 0;

  if (TargetDecl) {
    if (TargetDecl->getAttr<NoThrowAttr>())
      FuncAttrs |= llvm::ParamAttr::NoUnwind;
    if (TargetDecl->getAttr<NoReturnAttr>())
      FuncAttrs |= llvm::ParamAttr::NoReturn;
  }

  unsigned Index = 1;
  if (CodeGenFunction::hasAggregateLLVMType(ArgTypes[0])) {
    PAL.push_back(llvm::ParamAttrsWithIndex::get(Index, 
                                                 llvm::ParamAttr::StructRet));
    ++Index;
  } else if (ArgTypes[0]->isPromotableIntegerType()) {
    if (ArgTypes[0]->isSignedIntegerType()) {
      FuncAttrs |= llvm::ParamAttr::SExt;
    } else if (ArgTypes[0]->isUnsignedIntegerType()) {
      FuncAttrs |= llvm::ParamAttr::ZExt;
    }
  }
  if (FuncAttrs)
    PAL.push_back(llvm::ParamAttrsWithIndex::get(0, FuncAttrs));
  for (llvm::SmallVector<QualType, 8>::const_iterator i = ArgTypes.begin() + 1,
         e = ArgTypes.end(); i != e; ++i, ++Index) {
    QualType ParamType = *i;
    unsigned ParamAttrs = 0;
    if (ParamType->isRecordType())
      ParamAttrs |= llvm::ParamAttr::ByVal;
    if (ParamType->isPromotableIntegerType()) {
      if (ParamType->isSignedIntegerType()) {
        ParamAttrs |= llvm::ParamAttr::SExt;
      } else if (ParamType->isUnsignedIntegerType()) {
        ParamAttrs |= llvm::ParamAttr::ZExt;
      }
    }
    if (ParamAttrs)
      PAL.push_back(llvm::ParamAttrsWithIndex::get(Index, ParamAttrs));
  }
}

/***/

// FIXME: Use iterator and sidestep silly type array creation.

CGFunctionInfo::CGFunctionInfo(const FunctionDecl *FD)
  : TheDecl(FD) 
{
  const FunctionType *FTy = FD->getType()->getAsFunctionType();
  const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(FTy);
  
  ArgTypes.push_back(FTy->getResultType());
  if (FTP)
    for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
      ArgTypes.push_back(FTP->getArgType(i));
}

CGFunctionInfo::CGFunctionInfo(const ObjCMethodDecl *MD,
                               const ASTContext &Context)
  : TheDecl(MD) 
{
  ArgTypes.push_back(MD->getResultType());
  ArgTypes.push_back(MD->getSelfDecl()->getType());
  ArgTypes.push_back(Context.getObjCSelType());
  for (ObjCMethodDecl::param_const_iterator i = MD->param_begin(),
         e = MD->param_end(); i != e; ++i)
    ArgTypes.push_back((*i)->getType());
}

void CGFunctionInfo::constructParamAttrList(ParamAttrListType &PAL) const {
  constructParamAttrListInternal(TheDecl, ArgTypes, PAL);
}

/***/

CGCallInfo::CGCallInfo(QualType _ResultType,
                       const llvm::SmallVector<std::pair<llvm::Value*, QualType>, 16> &_Args) 
  : ResultType(_ResultType),
    Args(_Args) {
  ArgTypes.push_back(ResultType);
  for (CallArgList::const_iterator i = Args.begin(), e = Args.end(); i!=e; ++i)
    ArgTypes.push_back(i->second);
}

void CGCallInfo::constructParamAttrList(ParamAttrListType &PAL) const {
  // FIXME: Provide TargetDecl so nounwind, noreturn, etc, etc get set.
  constructParamAttrListInternal(0, ArgTypes, PAL);
}
