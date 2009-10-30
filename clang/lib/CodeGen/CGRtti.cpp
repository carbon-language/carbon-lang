//===--- CGCXXRtti.cpp - Emit LLVM Code for C++ RTTI descriptors ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of RTTI descriptors.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
using namespace clang;
using namespace CodeGen;

llvm::Constant *CodeGenModule::GenerateRtti(const CXXRecordDecl *RD) {
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  if (!getContext().getLangOptions().Rtti)
    return llvm::Constant::getNullValue(Int8PtrTy);

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  mangleCXXRtti(getMangleContext(), 
                Context.getTagDeclType(RD).getTypePtr(), Out);
  
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> info;
  // assert(0 && "FIXME: implement rtti descriptor");
  // FIXME: descriptor
  info.push_back(llvm::Constant::getNullValue(Int8PtrTy));
  // assert(0 && "FIXME: implement rtti ts");
  // FIXME: TS
  info.push_back(llvm::Constant::getNullValue(Int8PtrTy));

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Int8PtrTy, info.size());
  C = llvm::ConstantArray::get(type, info);
  llvm::Constant *Rtti = 
    new llvm::GlobalVariable(getModule(), type, true, linktype, C,
                             Out.str());
  Rtti = llvm::ConstantExpr::getBitCast(Rtti, Int8PtrTy);
  return Rtti;
}
