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

class RttiBuilder {
  CodeGenModule &CGM;  // Per-module state.
  llvm::LLVMContext &VMContext;
  const llvm::Type *Int8PtrTy;
public:
  RttiBuilder(CodeGenModule &cgm)
    : CGM(cgm), VMContext(cgm.getModule().getContext()),
      Int8PtrTy(llvm::Type::getInt8PtrTy(VMContext)) { }

  llvm::Constant *BuildName(const CXXRecordDecl *RD) {
    llvm::SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    mangleCXXRttiName(CGM.getMangleContext(),
                      CGM.getContext().getTagDeclType(RD), Out);
  
    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::LinkOnceODRLinkage;

    llvm::Constant *C;
    C = llvm::ConstantArray::get(VMContext, Out.str().substr(4));
    
    llvm::Constant *s = new llvm::GlobalVariable(CGM.getModule(), C->getType(),
                                                 true, linktype, C,
                                                 Out.str());
    s = llvm::ConstantExpr::getBitCast(s, Int8PtrTy);
    return s;
  };
};

llvm::Constant *CodeGenModule::GenerateRtti(const CXXRecordDecl *RD) {
  RttiBuilder b(*this);

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  if (!getContext().getLangOptions().Rtti)
    return llvm::Constant::getNullValue(Int8PtrTy);

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  mangleCXXRtti(getMangleContext(), Context.getTagDeclType(RD), Out);
  
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::LinkOnceODRLinkage;
  std::vector<llvm::Constant *> info;
  // assert(0 && "FIXME: implement rtti descriptor");
  // FIXME: descriptor
  info.push_back(llvm::Constant::getNullValue(Int8PtrTy));
  info.push_back(b.BuildName(RD));

  // FIXME: rest of rtti bits

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Int8PtrTy, info.size());
  C = llvm::ConstantArray::get(type, info);
  llvm::Constant *Rtti = 
    new llvm::GlobalVariable(getModule(), type, true, linktype, C,
                             Out.str());
  Rtti = llvm::ConstantExpr::getBitCast(Rtti, Int8PtrTy);
  return Rtti;
}
