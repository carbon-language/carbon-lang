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

  llvm::Constant *Buildclass_type_infoDesc() {
    // Build a descriptor for class_type_info.
    llvm::StringRef Name = "_ZTVN10__cxxabiv121__vmi_class_type_infoE";
    llvm::Constant *GV = CGM.getModule().getGlobalVariable(Name);
    if (GV)
      GV = llvm::ConstantExpr::getBitCast(GV,
                                          llvm::PointerType::get(Int8PtrTy, 0));
    else {
      llvm::GlobalVariable::LinkageTypes linktype;
      linktype = llvm::GlobalValue::ExternalLinkage;
      GV = new llvm::GlobalVariable(CGM.getModule(), Int8PtrTy,
                                    true, linktype, 0, Name);
    }  
    llvm::Constant *C;
    C = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), 2);
    C = llvm::ConstantExpr::getInBoundsGetElementPtr(GV, &C, 1);
    return llvm::ConstantExpr::getBitCast(C, Int8PtrTy);
  }
  
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

  llvm::Constant *Buildclass_type_info(const CXXRecordDecl *RD) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

    if (!CGM.getContext().getLangOptions().Rtti)
      return llvm::Constant::getNullValue(Int8PtrTy);

    llvm::SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    mangleCXXRtti(CGM.getMangleContext(), CGM.getContext().getTagDeclType(RD),
                  Out);
  
    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::LinkOnceODRLinkage;
    std::vector<llvm::Constant *> info;

    info.push_back(Buildclass_type_infoDesc());
    info.push_back(BuildName(RD));

    // FIXME: rest of rtti bits

    llvm::Constant *C;
    llvm::ArrayType *type = llvm::ArrayType::get(Int8PtrTy, info.size());
    C = llvm::ConstantArray::get(type, info);
    llvm::Constant *Rtti = 
      new llvm::GlobalVariable(CGM.getModule(), type, true, linktype, C,
                               Out.str());
    Rtti = llvm::ConstantExpr::getBitCast(Rtti, Int8PtrTy);
    return Rtti;
  }
};

llvm::Constant *CodeGenModule::GenerateRtti(const CXXRecordDecl *RD) {
  RttiBuilder b(*this);

  return  b.Buildclass_type_info(RD);
}
