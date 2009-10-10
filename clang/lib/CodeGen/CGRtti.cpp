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
  llvm::Type *Ptr8Ty;
  Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  llvm::Constant *Rtti = llvm::Constant::getNullValue(Ptr8Ty);

  if (!getContext().getLangOptions().Rtti)
    return Rtti;

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  QualType ClassTy;
  ClassTy = getContext().getTagDeclType(RD);
  mangleCXXRtti(getMangleContext(), ClassTy, Out);
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> info;
  // assert(0 && "FIXME: implement rtti descriptor");
  // FIXME: descriptor
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));
  // assert(0 && "FIXME: implement rtti ts");
  // FIXME: TS
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, info.size());
  C = llvm::ConstantArray::get(type, info);
  Rtti = new llvm::GlobalVariable(getModule(), type, true, linktype, C,
                                  Out.str());
  Rtti = llvm::ConstantExpr::getBitCast(Rtti, Ptr8Ty);
  return Rtti;
}
