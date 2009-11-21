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

#include "clang/AST/Type.h"
#include "clang/AST/RecordLayout.h"
#include "CodeGenModule.h"
using namespace clang;
using namespace CodeGen;

class RttiBuilder {
  CodeGenModule &CGM;  // Per-module state.
  llvm::LLVMContext &VMContext;
  const llvm::Type *Int8PtrTy;
  llvm::SmallSet<const CXXRecordDecl *, 16> SeenVBase;
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenBase;
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

  llvm::Constant *BuildName(QualType Ty, bool Hidden, bool Extern) {
    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRttiName(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::LinkOnceODRLinkage;
    if (!Extern)
      linktype = llvm::GlobalValue::InternalLinkage;

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    llvm::Constant *C;
    C = llvm::ConstantArray::get(VMContext, Name.substr(4));

    llvm::GlobalVariable *OGV = GV;
    GV = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, linktype,
                                  C, Name);
    if (OGV) {
      GV->takeName(OGV);
      llvm::Constant *NewPtr = llvm::ConstantExpr::getBitCast(GV,
                                                              OGV->getType());
      OGV->replaceAllUsesWith(NewPtr);
      OGV->eraseFromParent();
    }
    if (Hidden)
      GV->setVisibility(llvm::GlobalVariable::HiddenVisibility);
    return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);
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

  llvm::Constant *BuildTypeRef(QualType Ty) {
    llvm::Constant *C;

    if (!CGM.getContext().getLangOptions().Rtti)
      return llvm::Constant::getNullValue(Int8PtrTy);

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRtti(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    C = CGM.getModule().getGlobalVariable(Name);
    if (C)
      return llvm::ConstantExpr::getBitCast(C, Int8PtrTy);

    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::ExternalLinkage;;

    C = new llvm::GlobalVariable(CGM.getModule(), Int8PtrTy, true, linktype,
                                 0, Name);
    return llvm::ConstantExpr::getBitCast(C, Int8PtrTy);
  }

  llvm::Constant *Buildclass_type_infoRef(const CXXRecordDecl *RD) {
    return BuildTypeRef(CGM.getContext().getTagDeclType(RD));
  }

  /// CalculateFlags - Calculate the flags for the __vmi_class_type_info
  /// datastructure.  1 for non-diamond repeated inheritance, 2 for a dimond
  /// shaped class.
  int CalculateFlags(const CXXRecordDecl*RD) {
    int flags = 0;
    if (SeenBase.count(RD))
      flags |= 1;
    else
      SeenBase.insert(RD);
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual()) {
        if (SeenVBase.count(Base))
          flags |= 2;
        else
          SeenVBase.insert(Base);
      }
      flags |= CalculateFlags(Base);
    }
    return flags;
  }

  bool SimpleInheritance(const CXXRecordDecl *RD) {
    if (RD->getNumBases() != 1)
      return false;
    CXXRecordDecl::base_class_const_iterator i = RD->bases_begin();
    if (i->isVirtual())
      return false;
    if (i->getAccessSpecifier() != AS_public)
      return false;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (Layout.getBaseClassOffset(Base) != 0)
      return false;
    return true;
  }

  llvm::Constant *finish(std::vector<llvm::Constant *> &info,
                         llvm::GlobalVariable *GV,
                         llvm::StringRef Name, bool Hidden, bool Extern) {
    llvm::GlobalVariable::LinkageTypes linktype;
    linktype = llvm::GlobalValue::LinkOnceODRLinkage;
    if (!Extern)
      linktype = llvm::GlobalValue::InternalLinkage;

    llvm::Constant *C;
    C = llvm::ConstantStruct::get(VMContext, &info[0], info.size(), false);

    llvm::GlobalVariable *OGV = GV;
    GV = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, linktype,
                                  C, Name);
    if (OGV) {
      GV->takeName(OGV);
      llvm::Constant *NewPtr = llvm::ConstantExpr::getBitCast(GV,
                                                              OGV->getType());
      OGV->replaceAllUsesWith(NewPtr);
      OGV->eraseFromParent();
    }
    if (Hidden)
      GV->setVisibility(llvm::GlobalVariable::HiddenVisibility);
    return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);
  }


  llvm::Constant *Buildclass_type_info(const CXXRecordDecl *RD) {
    if (!CGM.getContext().getLangOptions().Rtti)
      return llvm::Constant::getNullValue(Int8PtrTy);

    llvm::Constant *C;

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRtti(CGM.getContext().getTagDeclType(RD),
                                         OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    std::vector<llvm::Constant *> info;

    bool Hidden = CGM.getDeclVisibilityMode(RD) == LangOptions::Hidden;
    bool Extern = !RD->isInAnonymousNamespace();

    bool simple = false;
    if (RD->getNumBases() == 0)
      C = BuildVtableRef("_ZTVN10__cxxabiv117__class_type_infoE");
    else if (SimpleInheritance(RD)) {
      simple = true;
      C = BuildVtableRef("_ZTVN10__cxxabiv120__si_class_type_infoE");
    } else
      C = BuildVtableRef("_ZTVN10__cxxabiv121__vmi_class_type_infoE");
    info.push_back(C);
    info.push_back(BuildName(CGM.getContext().getTagDeclType(RD), Hidden,
                             Extern));

    // If we have no bases, there are no more fields.
    if (RD->getNumBases()) {
      if (!simple) {
        info.push_back(BuildFlags(CalculateFlags(RD)));
        info.push_back(BuildBaseCount(RD->getNumBases()));
      }

      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
             e = RD->bases_end(); i != e; ++i) {
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
        info.push_back(CGM.GenerateRttiRef(Base));
        if (simple)
          break;
        int64_t offset;
        if (!i->isVirtual())
          offset = Layout.getBaseClassOffset(Base)/8;
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

    return finish(info, GV, Name, Hidden, Extern);
  }

  /// - BuildFlags - Build a __flags value for __pbase_type_info.
  llvm::Constant *BuildInt(int f) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), f);
  }

  bool DecideExtern(QualType Ty) {
    // For this type, see if all components are never in an anonymous namespace.
    if (const MemberPointerType *MPT = Ty->getAs<MemberPointerType>())
      return (DecideExtern(MPT->getPointeeType())
              && DecideExtern(QualType(MPT->getClass(), 0)));
    if (const PointerType *PT = Ty->getAs<PointerType>())
      return DecideExtern(PT->getPointeeType());
    if (const RecordType *RT = Ty->getAs<RecordType>())
      if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
        return !RD->isInAnonymousNamespace();
    return true;
  }

  bool DecideHidden(QualType Ty) {
    // For this type, see if all components are never hidden.
    if (const MemberPointerType *MPT = Ty->getAs<MemberPointerType>())
      return (DecideHidden(MPT->getPointeeType())
              && DecideHidden(QualType(MPT->getClass(), 0)));
    if (const PointerType *PT = Ty->getAs<PointerType>())
      return DecideHidden(PT->getPointeeType());
    if (const RecordType *RT = Ty->getAs<RecordType>())
      if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
        return CGM.getDeclVisibilityMode(RD) == LangOptions::Hidden;
    return false;
  }

  llvm::Constant *BuildPointerType(QualType Ty) {
    llvm::Constant *C;

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRtti(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    std::vector<llvm::Constant *> info;

    bool Extern = DecideExtern(Ty);
    bool Hidden = DecideHidden(Ty);

    QualType PTy = Ty->getPointeeType();
    QualType BTy;
    bool PtrMem = false;
    if (const MemberPointerType *MPT = dyn_cast<MemberPointerType>(Ty)) {
      PtrMem = true;
      BTy = QualType(MPT->getClass(), 0);
      PTy = MPT->getPointeeType();
    }

    if (PtrMem)
      C = BuildVtableRef("_ZTVN10__cxxabiv129__pointer_to_member_type_infoE");
    else
      C = BuildVtableRef("_ZTVN10__cxxabiv119__pointer_type_infoE");
    info.push_back(C);
    info.push_back(BuildName(Ty, Hidden, Extern));
    Qualifiers Q = PTy.getQualifiers();
    PTy = CGM.getContext().getCanonicalType(PTy).getUnqualifiedType();
    int flags = 0;
    flags += Q.hasConst() ? 0x1 : 0;
    flags += Q.hasVolatile() ? 0x2 : 0;
    flags += Q.hasRestrict() ? 0x4 : 0;
    flags += Ty.getTypePtr()->isIncompleteType() ? 0x8 : 0;
    if (PtrMem && BTy.getTypePtr()->isIncompleteType())
      flags += 0x10;

    info.push_back(BuildInt(flags));
    info.push_back(BuildInt(0));
    info.push_back(BuildType(PTy));

    if (PtrMem)
      info.push_back(BuildType(BTy));

    // We always generate these as hidden, only the name isn't hidden.
    return finish(info, GV, Name, true, Extern);
  }

  llvm::Constant *BuildSimpleType(QualType Ty, const char *vtbl) {
    llvm::Constant *C;

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRtti(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    std::vector<llvm::Constant *> info;

    bool Extern = DecideExtern(Ty);
    bool Hidden = DecideHidden(Ty);

    C = BuildVtableRef(vtbl);
    info.push_back(C);
    info.push_back(BuildName(Ty, Hidden, Extern));

    // We always generate these as hidden, only the name isn't hidden.
    return finish(info, GV, Name, true, Extern);
  }

  llvm::Constant *BuildType(QualType Ty) {
    const clang::Type &Type
      = *CGM.getContext().getCanonicalType(Ty).getTypePtr();

    if (const RecordType *RT = Ty.getTypePtr()->getAs<RecordType>())
      if (const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl()))
        return Buildclass_type_info(RD);

    switch (Type.getTypeClass()) {
    default: {
      assert(0 && "typeid expression");
      return llvm::Constant::getNullValue(Int8PtrTy);
    }

    case Type::Builtin: {
      // We expect all type_info objects for builtin types to be in the library.
      return BuildTypeRef(Ty);
    }

    case Type::Pointer: {
      QualType PTy = Ty->getPointeeType();
      Qualifiers Q = PTy.getQualifiers();
      Q.removeConst();
      // T* and const T* for all builtin types T are expected in the library.
      if (isa<BuiltinType>(PTy) && Q.empty())
        return BuildTypeRef(Ty);

      return BuildPointerType(Ty);
    }
    case Type::MemberPointer:
      return BuildPointerType(Ty);
    case Type::FunctionProto:
    case Type::FunctionNoProto:
      return BuildSimpleType(Ty, "_ZTVN10__cxxabiv120__function_type_infoE");
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Vector:
    case Type::ExtVector:
      return BuildSimpleType(Ty, "_ZTVN10__cxxabiv117__array_type_infoE");
    case Type::Enum:
      return BuildSimpleType(Ty, "_ZTVN10__cxxabiv116__enum_type_infoE");
    }
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

llvm::Constant *CodeGenModule::GenerateRtti(QualType Ty) {
  RttiBuilder b(*this);

  return b.BuildType(Ty);
}
