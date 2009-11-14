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
#include "clang/AST/RecordLayout.h"
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

  /// BuildVtableRef - Build a reference to a vtable.
  llvm::Constant *BuildVtableRef(const char *Name) {
    // Build a descriptor for Name
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

  /// - BuildFlags - Build a psABI __flags value for __vmi_class_type_info.
  llvm::Constant *BuildFlags(int f) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), f);
  }

  /// BuildBaseCount - Build a psABI __base_count value for
  /// __vmi_class_type_info.
  llvm::Constant *BuildBaseCount(unsigned c) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), c);
  }

  llvm::Constant *Buildclass_type_infoRef(const CXXRecordDecl *RD) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    llvm::Constant *C;

    if (!CGM.getContext().getLangOptions().Rtti)
      return llvm::Constant::getNullValue(Int8PtrTy);

    llvm::SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    mangleCXXRtti(CGM.getMangleContext(), CGM.getContext().getTagDeclType(RD),
                  Out);

    C = CGM.getModule().getGlobalVariable(Out.str());
    if (C)
      return llvm::ConstantExpr::getBitCast(C, Int8PtrTy);

    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::ExternalLinkage;;

    C = new llvm::GlobalVariable(CGM.getModule(), Int8PtrTy, true, linktype,
                                 0, Out.str());
    return llvm::ConstantExpr::getBitCast(C, Int8PtrTy);
  }

  llvm::Constant *Buildclass_type_info(const CXXRecordDecl *RD) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    llvm::Constant *C;

    if (!CGM.getContext().getLangOptions().Rtti)
      return llvm::Constant::getNullValue(Int8PtrTy);

    llvm::SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    mangleCXXRtti(CGM.getMangleContext(), CGM.getContext().getTagDeclType(RD),
                  Out);

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Out.str());
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::LinkOnceODRLinkage;
    std::vector<llvm::Constant *> info;

    if (RD->getNumBases() == 0)
      C = BuildVtableRef("_ZTVN10__cxxabiv117__class_type_infoE");
    // FIXME: Add si_class_type_info optimization
    else
      C = BuildVtableRef("_ZTVN10__cxxabiv121__vmi_class_type_infoE");
    info.push_back(C);
    info.push_back(BuildName(RD));

    // If we have no bases, there are no more fields.
    if (RD->getNumBases()) {

      // FIXME: Calculate is_diamond and non-diamond repeated inheritance, 3 is
      // conservative.
      info.push_back(BuildFlags(3));
      info.push_back(BuildBaseCount(RD->getNumBases()));

      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
             e = RD->bases_end(); i != e; ++i) {
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
        info.push_back(CGM.GenerateRtti(Base));
        int64_t offset;
        if (!i->isVirtual())
          offset = Layout.getBaseClassOffset(Base);
        else
          offset = CGM.getVtableInfo().getVirtualBaseOffsetIndex(RD, Base);
        offset <<= 8;
        // Now set the flags.
        offset += i->isVirtual() ? 1 : 0;;
        offset += i->getAccessSpecifier() == AS_public ? 2 : 0;
        const llvm::Type *LongTy =
          CGM.getTypes().ConvertType(CGM.getContext().LongTy);
        C = llvm::ConstantInt::get(LongTy, offset);
        info.push_back(C);
      }
    }

    std::vector<const llvm::Type *> Types(info.size());
    for (unsigned i=0; i<info.size(); ++i)
      Types[i] = info[i]->getType();
    // llvm::StructType *Ty = llvm::StructType::get(VMContext, Types, true);
    C = llvm::ConstantStruct::get(VMContext, &info[0], info.size(), false);

    if (GV == 0)
      GV = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, linktype,
                                    C, Out.str());
    else {
      llvm::GlobalVariable *OGV = GV;
      GV = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, linktype,
                                    C, Out.str());
      GV->takeName(OGV);
      llvm::Constant *NewPtr = llvm::ConstantExpr::getBitCast(GV, OGV->getType());
      OGV->replaceAllUsesWith(NewPtr);
      OGV->eraseFromParent();
    }
    return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

#if 0
    llvm::ArrayType *type = llvm::ArrayType::get(Int8PtrTy, info.size());
    C = llvm::ConstantArray::get(type, info);
    llvm::Constant *Rtti =
      new llvm::GlobalVariable(CGM.getModule(), type, true, linktype, C,
                               Out.str());
    Rtti = llvm::ConstantExpr::getBitCast(Rtti, Int8PtrTy);
    return Rtti;
#endif
  }
};

llvm::Constant *CodeGenModule::GenerateRttiRef(const CXXRecordDecl *RD) {
  RttiBuilder b(*this);

  return b.Buildclass_type_infoRef(RD);
}

llvm::Constant *CodeGenModule::GenerateRtti(const CXXRecordDecl *RD) {
  RttiBuilder b(*this);

  return b.Buildclass_type_info(RD);
}
