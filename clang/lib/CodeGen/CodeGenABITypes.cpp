//==--- CodeGenABITypes.cpp - Convert Clang types to LLVM types for ABI ----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CodeGenABITypes is a simple interface for getting LLVM types for
// the parameters and the return value of a function given the Clang
// types.
//
// The class is implemented as a public wrapper around the private
// CodeGenTypes class in lib/CodeGen.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenABITypes.h"
#include "CodeGenModule.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"

using namespace clang;
using namespace CodeGen;

CodeGenABITypes::CodeGenABITypes(ASTContext &C,
                                 llvm::Module &M,
                                 const llvm::DataLayout &TD)
  : CGO(new CodeGenOptions),
    CGM(new CodeGen::CodeGenModule(C, *CGO, M, TD, C.getDiagnostics())) {
}

CodeGenABITypes::~CodeGenABITypes()
{
  delete CGO;
  delete CGM;
}

const CGFunctionInfo &
CodeGenABITypes::arrangeObjCMessageSendSignature(const ObjCMethodDecl *MD,
                                                 QualType receiverType) {
  return CGM->getTypes().arrangeObjCMessageSendSignature(MD, receiverType);
}

const CGFunctionInfo &
CodeGenABITypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> Ty) {
  return CGM->getTypes().arrangeFreeFunctionType(Ty);
}

const CGFunctionInfo &
CodeGenABITypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> Ty) {
  return CGM->getTypes().arrangeFreeFunctionType(Ty);
}

const CGFunctionInfo &
CodeGenABITypes::arrangeCXXMethodType(const CXXRecordDecl *RD,
                                      const FunctionProtoType *FTP) {
  return CGM->getTypes().arrangeCXXMethodType(RD, FTP);
}

const CGFunctionInfo &
CodeGenABITypes::arrangeFreeFunctionCall(CanQualType returnType,
                                         llvm::ArrayRef<CanQualType> argTypes,
                                         FunctionType::ExtInfo info,
                                         RequiredArgs args) {
  return CGM->getTypes().arrangeLLVMFunctionInfo(
      returnType, /*IsInstanceMethod=*/false, argTypes, info, args);
}
