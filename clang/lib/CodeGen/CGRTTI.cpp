//===--- CGCXXRTTI.cpp - Emit LLVM Code for C++ RTTI descriptors ----------===//
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

namespace {
class RTTIBuilder {
  CodeGenModule &CGM;  // Per-module state.
  llvm::LLVMContext &VMContext;
  const llvm::Type *Int8PtrTy;
  llvm::SmallSet<const CXXRecordDecl *, 16> SeenVBase;
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenBase;

  std::vector<llvm::Constant *> Info;
  
  // Type info flags.
  enum {
    /// TI_Const - Type has const qualifier.
    TI_Const = 0x1,
    
    /// TI_Volatile - Type has volatile qualifier.
    TI_Volatile = 0x2,

    /// TI_Restrict - Type has restrict qualifier.
    TI_Restrict = 0x4,
    
    /// TI_Incomplete - Type is incomplete.
    TI_Incomplete = 0x8,

    /// TI_ContainingClassIncomplete - Containing class is incomplete.
    /// (in pointer to member).
    TI_ContainingClassIncomplete = 0x10
  };
  
  /// GetAddrOfExternalRTTIDescriptor - Returns the constant for the RTTI 
  /// descriptor of the given type.
  llvm::Constant *GetAddrOfExternalRTTIDescriptor(QualType Ty);
  
public:
  RTTIBuilder(CodeGenModule &cgm)
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

  // FIXME: This should be removed, and clients should pass in the linkage
  // directly instead.
  static inline llvm::GlobalVariable::LinkageTypes
  GetLinkageFromExternFlag(bool Extern) {
    if (Extern)
      return llvm::GlobalValue::WeakODRLinkage;
    
    return llvm::GlobalValue::InternalLinkage;
  }
  
  // FIXME: This should be removed, and clients should pass in the linkage
  // directly instead.
  llvm::Constant *BuildName(QualType Ty, bool Hidden, bool Extern) {
    return BuildName(Ty, Hidden, GetLinkageFromExternFlag(Extern));
  }

  llvm::Constant *BuildName(QualType Ty, bool Hidden, 
                            llvm::GlobalVariable::LinkageTypes Linkage) {
    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRTTIName(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *OGV = CGM.getModule().getGlobalVariable(Name);
    if (OGV && !OGV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(OGV, Int8PtrTy);

    llvm::Constant *C = llvm::ConstantArray::get(VMContext, Name.substr(4));

    llvm::GlobalVariable *GV = 
      new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, Linkage,
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

  /// - BuildFlags - Build a psABI __flags value for __vmi_class_type_info.
  llvm::Constant *BuildFlags(int f) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), f);
  }

  /// BuildBaseCount - Build a psABI __base_count value for
  /// __vmi_class_type_info.
  llvm::Constant *BuildBaseCount(unsigned c) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), c);
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

  llvm::Constant *finish(llvm::GlobalVariable *GV,
                         llvm::StringRef Name, bool Hidden, 
                         llvm::GlobalVariable::LinkageTypes Linkage) {
    llvm::Constant *C = 
      llvm::ConstantStruct::get(VMContext, &Info[0], Info.size(), 
                                /*Packed=*/false);

    llvm::GlobalVariable *OGV = GV;
    GV = new llvm::GlobalVariable(CGM.getModule(), C->getType(), true, Linkage,
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


  llvm::Constant *
  Buildclass_type_info(const CXXRecordDecl *RD,
                       llvm::GlobalVariable::LinkageTypes Linkage) {
    assert(Info.empty() && "Info vector must be empty!");
    
    llvm::Constant *C;

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRTTI(CGM.getContext().getTagDeclType(RD),
                                         OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    // If we're in an anonymous namespace, then we always want internal linkage.
    if (RD->isInAnonymousNamespace() || !RD->hasLinkage())
      Linkage = llvm::GlobalVariable::InternalLinkage;
    
    bool Hidden = CGM.getDeclVisibilityMode(RD) == LangOptions::Hidden;

    bool simple = false;
    if (RD->getNumBases() == 0)
      C = BuildVtableRef("_ZTVN10__cxxabiv117__class_type_infoE");
    else if (SimpleInheritance(RD)) {
      simple = true;
      C = BuildVtableRef("_ZTVN10__cxxabiv120__si_class_type_infoE");
    } else
      C = BuildVtableRef("_ZTVN10__cxxabiv121__vmi_class_type_infoE");
    Info.push_back(C);
    Info.push_back(BuildName(CGM.getContext().getTagDeclType(RD), Hidden,
                             Linkage));

    // If we have no bases, there are no more fields.
    if (RD->getNumBases()) {
      if (!simple) {
        Info.push_back(BuildFlags(CalculateFlags(RD)));
        Info.push_back(BuildBaseCount(RD->getNumBases()));
      }

      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
             e = RD->bases_end(); i != e; ++i) {
        QualType BaseType = i->getType();
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());
        Info.push_back(CGM.GetAddrOfRTTIDescriptor(BaseType));
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
        Info.push_back(C);
      }
    }

    return finish(GV, Name, Hidden, Linkage);
  }

  /// - BuildFlags - Build a __flags value for __pbase_type_info.
  llvm::Constant *BuildInt(unsigned n) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), n);
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
        return !RD->isInAnonymousNamespace() && RD->hasLinkage();
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
    assert(Info.empty() && "Info vector must be empty!");
    
    llvm::Constant *C;

    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    bool Extern = DecideExtern(Ty);
    bool Hidden = DecideHidden(Ty);

    const MemberPointerType *PtrMemTy = dyn_cast<MemberPointerType>(Ty);
    QualType PointeeTy;
    
    if (PtrMemTy)
      PointeeTy = PtrMemTy->getPointeeType();
    else
      PointeeTy = Ty->getPointeeType();

    if (PtrMemTy)
      C = BuildVtableRef("_ZTVN10__cxxabiv129__pointer_to_member_type_infoE");
    else
      C = BuildVtableRef("_ZTVN10__cxxabiv119__pointer_type_infoE");
    
    Info.push_back(C);
    Info.push_back(BuildName(Ty, Hidden, Extern));
    Qualifiers Q = PointeeTy.getQualifiers();
    
    PointeeTy = 
      CGM.getContext().getCanonicalType(PointeeTy).getUnqualifiedType();
    
    unsigned Flags = 0;
    if (Q.hasConst())
      Flags |= TI_Const;
    if (Q.hasVolatile())
      Flags |= TI_Volatile;
    if (Q.hasRestrict())
      Flags |= TI_Restrict;
    
    if (Ty->isIncompleteType())
      Flags |= TI_Incomplete;
  
    if (PtrMemTy && PtrMemTy->getClass()->isIncompleteType())
      Flags |= TI_ContainingClassIncomplete;
    
    Info.push_back(BuildInt(Flags));
    Info.push_back(BuildInt(0));
    Info.push_back(RTTIBuilder(CGM).BuildType(PointeeTy));

    if (PtrMemTy)
      Info.push_back(RTTIBuilder(CGM).BuildType(
                                            QualType(PtrMemTy->getClass(), 0)));

    // We always generate these as hidden, only the name isn't hidden.
    return finish(GV, Name, /*Hidden=*/true, GetLinkageFromExternFlag(Extern));
  }

  llvm::Constant *BuildSimpleType(QualType Ty, const char *vtbl) {
    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getGlobalVariable(Name);
    if (GV && !GV->isDeclaration())
      return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);

    bool Extern = DecideExtern(Ty);
    bool Hidden = DecideHidden(Ty);

    Info.push_back(BuildVtableRef(vtbl));
    Info.push_back(BuildName(Ty, Hidden, Extern));
    
    // We always generate these as hidden, only the name isn't hidden.
    return finish(GV, Name, /*Hidden=*/true, 
                  GetLinkageFromExternFlag(Extern));
  }

  /// BuildType - Builds the type info for the given type.
  llvm::Constant *BuildType(QualType Ty) {
    const clang::Type &Type
      = *CGM.getContext().getCanonicalType(Ty).getTypePtr();

    if (const RecordType *RT = Ty.getTypePtr()->getAs<RecordType>())
      if (const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl()))
        return BuildClassTypeInfo(RD);

    switch (Type.getTypeClass()) {
    default: {
      assert(0 && "typeid expression");
      return llvm::Constant::getNullValue(Int8PtrTy);
    }

    case Type::Builtin: {
      // We expect all type_info objects for builtin types to be in the library.
      return GetAddrOfExternalRTTIDescriptor(Ty);
    }

    case Type::Pointer: {
      QualType PTy = Ty->getPointeeType();
      Qualifiers Q = PTy.getQualifiers();
      Q.removeConst();
      // T* and const T* for all builtin types T are expected in the library.
      if (isa<BuiltinType>(PTy) && Q.empty())
        return GetAddrOfExternalRTTIDescriptor(Ty);

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
  
  /// BuildClassTypeInfo - Builds the class type info (or a reference to it)
  /// for the given record decl.
  llvm::Constant *BuildClassTypeInfo(const CXXRecordDecl *RD) {
    const CXXMethodDecl *KeyFunction = 0;

    if (RD->isDynamicClass())
      KeyFunction = CGM.getContext().getKeyFunction(RD);
    
    if (KeyFunction) {
      // If the key function is defined in this translation unit, then the RTTI
      // related constants should also be emitted here, with external linkage.
      if (KeyFunction->getBody())
        return Buildclass_type_info(RD, llvm::GlobalValue::ExternalLinkage);
      
      // Otherwise, we just want a reference to the type info.
      QualType Ty = CGM.getContext().getTagDeclType(RD);
      return GetAddrOfExternalRTTIDescriptor(Ty);
    }
    
    // If there is no key function (or if the record doesn't have any virtual
    // member functions or virtual bases), emit the type info with weak_odr
    // linkage.
    return Buildclass_type_info(RD, llvm::GlobalValue::WeakODRLinkage);
  }
};
}

llvm::Constant *RTTIBuilder::GetAddrOfExternalRTTIDescriptor(QualType Ty) {
  // Mangle the RTTI name.
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
  llvm::StringRef Name = OutName.str();

  // Look for an existing global variable.
  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name);
  
  if (!GV) {
    // Create a new global variable.
    GV = new llvm::GlobalVariable(CGM.getModule(), Int8PtrTy, /*Constant=*/true,
                                  llvm::GlobalValue::ExternalLinkage, 0, Name);
  }
  
  return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);
}

llvm::Constant *CodeGenModule::GetAddrOfRTTIDescriptor(QualType Ty) {
  if (!getContext().getLangOptions().RTTI) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    return llvm::Constant::getNullValue(Int8PtrTy);
  }
  
  return RTTIBuilder(*this).BuildType(Ty);
}
