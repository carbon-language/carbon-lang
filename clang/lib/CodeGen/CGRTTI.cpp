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

  /// GetAddrOfExternalRTTIDescriptor - Returns the constant for the RTTI 
  /// descriptor of the given type.
  llvm::Constant *GetAddrOfExternalRTTIDescriptor(QualType Ty);
  
  /// BuildTypeInfo - Build the RTTI type info struct for the given type.
  llvm::Constant *BuildTypeInfo(QualType Ty);

  /// BuildVtablePointer - Build the vtable pointer for the given type.
  void BuildVtablePointer(const Type *Ty);
  
  /// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct,
  /// used for pointer types.
  void BuildPointerTypeInfo(const PointerType *Ty);
  
  /// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info 
  /// struct, used for member pointer types.
  void BuildPointerToMemberTypeInfo(const MemberPointerType *Ty);
  
public:
  RTTIBuilder(CodeGenModule &cgm)
    : CGM(cgm), VMContext(cgm.getModule().getContext()),
      Int8PtrTy(llvm::Type::getInt8PtrTy(VMContext)) { }

  /// BuildVtableRef - Build a reference to a vtable.
  llvm::Constant *BuildVtableRef(const char *Name) {
    // Build a descriptor for Name
    llvm::Constant *GV = CGM.getModule().getNamedGlobal(Name);
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

    llvm::GlobalVariable *OGV = CGM.getModule().getNamedGlobal(Name);
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
  int CalculateFlags(const CXXRecordDecl *RD) {
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
    GV = CGM.getModule().getNamedGlobal(Name);
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

  llvm::Constant *BuildSimpleType(QualType Ty, const char *vtbl) {
    llvm::SmallString<256> OutName;
    CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
    llvm::StringRef Name = OutName.str();

    llvm::GlobalVariable *GV;
    GV = CGM.getModule().getNamedGlobal(Name);
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

    case Type::Pointer:
    case Type::MemberPointer:
        
      return BuildTypeInfo(Ty);
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
  
  // Pointer type info flags.
  enum {
    /// PTI_Const - Type has const qualifier.
    PTI_Const = 0x1,
    
    /// PTI_Volatile - Type has volatile qualifier.
    PTI_Volatile = 0x2,
    
    /// PTI_Restrict - Type has restrict qualifier.
    PTI_Restrict = 0x4,
    
    /// PTI_Incomplete - Type is incomplete.
    PTI_Incomplete = 0x8,
    
    /// PTI_ContainingClassIncomplete - Containing class is incomplete.
    /// (in pointer to member).
    PTI_ContainingClassIncomplete = 0x10
  };
};
}

llvm::Constant *RTTIBuilder::GetAddrOfExternalRTTIDescriptor(QualType Ty) {
  // Mangle the RTTI name.
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
  llvm::StringRef Name = OutName.str();

  // Look for an existing global.
  llvm::GlobalVariable *GV = CGM.getModule().getNamedGlobal(Name);
  
  if (!GV) {
    // Create a new global variable.
    GV = new llvm::GlobalVariable(CGM.getModule(), Int8PtrTy, /*Constant=*/true,
                                  llvm::GlobalValue::ExternalLinkage, 0, Name);
  }
  
  return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);
}

/// TypeInfoIsInStandardLibrary - Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
static bool TypeInfoIsInStandardLibrary(const BuiltinType *Ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and 
  //   X const*, for every X in: void, bool, wchar_t, char, unsigned char, 
  //   signed char, short, unsigned short, int, unsigned int, long, 
  //   unsigned long, long long, unsigned long long, float, double, long double, 
  //   char16_t, char32_t, and the IEEE 754r decimal and half-precision 
  //   floating point types.
  switch (Ty->getKind()) {
    case BuiltinType::Void:
    case BuiltinType::Bool:
    case BuiltinType::WChar:
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Int128:
    case BuiltinType::UInt128:
      return true;
      
    case BuiltinType::Overload:
    case BuiltinType::Dependent:
    case BuiltinType::UndeducedAuto:
      assert(false && "Should not see this type here!");
      
    case BuiltinType::NullPtr:
      assert(false && "FIXME: nullptr_t is not handled!");

    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      assert(false && "FIXME: Objective-C types are unsupported!");
  }
  
  // Silent gcc.
  return false;
}

static bool TypeInfoIsInStandardLibrary(const PointerType *PointerTy) {
  QualType PointeeTy = PointerTy->getPointeeType();
  const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(PointeeTy);
  if (!BuiltinTy)
    return false;
    
  // Check the qualifiers.
  Qualifiers Quals = PointeeTy.getQualifiers();
  Quals.removeConst();
    
  if (!Quals.empty())
    return false;
    
  return TypeInfoIsInStandardLibrary(BuiltinTy);
}

/// ShouldUseExternalRTTIDescriptor - Returns whether the type information for
/// the given type exists somewhere else, and that we should not emit the typ
/// information in this translation unit.
bool ShouldUseExternalRTTIDescriptor(QualType Ty) {
  // Type info for builtin types is defined in the standard library.
  if (const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(Ty))
    return TypeInfoIsInStandardLibrary(BuiltinTy);
  
  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return TypeInfoIsInStandardLibrary(PointerTy);

  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    (void)RecordTy;
    assert(false && "FIXME");
  }
  
  return false;
}

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
static bool IsIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getDecl()->isDefinition();
}  

/// IsPointerToIncompleteClassType - Returns whether the given pointer type
/// is an indirect or direct pointer to an incomplete class type.
static bool IsPointerToIncompleteClassType(const PointerType *PointerTy) {
  QualType PointeeTy = PointerTy->getPointeeType();
  while ((PointerTy = dyn_cast<PointerType>(PointeeTy)))
    PointeeTy = PointerTy->getPointeeType();

  if (const RecordType *RecordTy = dyn_cast<RecordType>(PointeeTy)) {
    // Check if the record type is incomplete.
    return IsIncompleteClassType(RecordTy);
  }
  
  return false;
}

/// getTypeInfoLinkage - Return the linkage that the type info and type info
/// name constants should have for the given type.
static llvm::GlobalVariable::LinkageTypes getTypeInfoLinkage(QualType Ty) {
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty)) {
    // Itanium C++ ABI 2.9.5p7:
    //   In addition, it and all of the intermediate abi::__pointer_type_info 
    //   structs in the chain down to the abi::__class_type_info for the
    //   incomplete class type must be prevented from resolving to the 
    //   corresponding type_info structs for the complete class type, possibly
    //   by making them local static objects. Finally, a dummy class RTTI is
    //   generated for the incomplete type that will not resolve to the final 
    //   complete class RTTI (because the latter need not exist), possibly by 
    //   making it a local static object.
    if (IsPointerToIncompleteClassType(PointerTy))
      return llvm::GlobalValue::InternalLinkage;
   
    // FIXME: Check linkage and anonymous namespace.
    return llvm::GlobalValue::WeakODRLinkage;
  } else if (const MemberPointerType *MemberPointerTy = 
              dyn_cast<MemberPointerType>(Ty)) {
    // If the class type is incomplete, then the type info constants should 
    // have internal linkage.
    const RecordType *ClassType = cast<RecordType>(MemberPointerTy->getClass());
    if (!ClassType->getDecl()->isDefinition())
      return llvm::GlobalValue::InternalLinkage;
    
    // FIXME: Check linkage and anonymous namespace.
    return llvm::GlobalValue::WeakODRLinkage;
  }

  assert(false && "FIXME!");
  return llvm::GlobalValue::WeakODRLinkage;
}

void RTTIBuilder::BuildVtablePointer(const Type *Ty) {
  const char *VtableName;

  switch (Ty->getTypeClass()) {
  default: assert(0 && "Unhandled type!");
  case Type::Pointer:
    // abi::__pointer_type_info
    VtableName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
    break;
  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info
    VtableName =  "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";
    break;
  }

  llvm::Constant *Vtable = 
    CGM.getModule().getOrInsertGlobal(VtableName, Int8PtrTy);
    
  const llvm::Type *PtrDiffTy = 
    CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());

  // The vtable address point is 2.
  llvm::Constant *Two = llvm::ConstantInt::get(PtrDiffTy, 2);
  Vtable = llvm::ConstantExpr::getInBoundsGetElementPtr(Vtable, &Two, 1);
  Vtable = llvm::ConstantExpr::getBitCast(Vtable, Int8PtrTy);

  Info.push_back(Vtable);
}

llvm::Constant *RTTIBuilder::BuildTypeInfo(QualType Ty) {
  // We want to operate on the canonical type.
  Ty = CGM.getContext().getCanonicalType(Ty);

  // Check if we've already emitted an RTTI descriptor for this type.
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXRTTI(Ty, OutName);
  llvm::StringRef Name = OutName.str();
  
  llvm::GlobalVariable *OldGV = CGM.getModule().getNamedGlobal(Name);
  if (OldGV && !OldGV->isDeclaration())
    return llvm::ConstantExpr::getBitCast(OldGV, Int8PtrTy);
  
  // Check if there is already an external RTTI descriptor for this type.
  if (ShouldUseExternalRTTIDescriptor(Ty))
    return GetAddrOfExternalRTTIDescriptor(Ty);

  llvm::GlobalVariable::LinkageTypes Linkage = getTypeInfoLinkage(Ty);

  // Add the vtable pointer.
  BuildVtablePointer(cast<Type>(Ty));
  
  // And the name.
  Info.push_back(BuildName(Ty, DecideHidden(Ty), Linkage));
  
  switch (Ty->getTypeClass()) {
  default: assert(false && "Unhandled type class!");
  case Type::Builtin:
    assert(false && "Builtin type info must be in the standard library!");
    break;

  case Type::Pointer:
    BuildPointerTypeInfo(cast<PointerType>(Ty));
    break;
  
  case Type::MemberPointer:
    BuildPointerToMemberTypeInfo(cast<MemberPointerType>(Ty));
    break;
  }

  llvm::Constant *Init = 
    llvm::ConstantStruct::get(VMContext, &Info[0], Info.size(), 
                              /*Packed=*/false);

  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(CGM.getModule(), Init->getType(), 
                             /*Constant=*/true, Linkage, Init, Name);
  
  // If there's already an old global variable, replace it with the new one.
  if (OldGV) {
    GV->takeName(OldGV);
    llvm::Constant *NewPtr = 
      llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
    OldGV->replaceAllUsesWith(NewPtr);
    OldGV->eraseFromParent();
  }
    
  return llvm::ConstantExpr::getBitCast(GV, Int8PtrTy);
}

/// DetermineQualifierFlags - Deterine the pointer type info flags from the
/// given qualifier.
static unsigned DetermineQualifierFlags(Qualifiers Quals) {
  unsigned Flags = 0;

  if (Quals.hasConst())
    Flags |= RTTIBuilder::PTI_Const;
  if (Quals.hasVolatile())
    Flags |= RTTIBuilder::PTI_Volatile;
  if (Quals.hasRestrict())
    Flags |= RTTIBuilder::PTI_Restrict;

  return Flags;
}

/// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct,
/// used for pointer types.
void RTTIBuilder::BuildPointerTypeInfo(const PointerType *Ty) {
  const PointerType *PointerTy = cast<PointerType>(Ty);
  QualType PointeeTy = PointerTy->getPointeeType();
  
  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other 
  //   attributes of the type pointed to
  unsigned Flags = DetermineQualifierFlags(PointeeTy.getQualifiers());

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set. 
  if (IsPointerToIncompleteClassType(PointerTy))
    Flags |= PTI_Incomplete;

  const llvm::Type *UnsignedIntLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Info.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));
  
  // Itanium C++ ABI 2.9.5p7:
  //  __pointee is a pointer to the std::type_info derivation for the 
  //  unqualified type being pointed to.
  Info.push_back(RTTIBuilder(CGM).BuildType(PointeeTy.getUnqualifiedType()));
}

/// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info 
/// struct, used for member pointer types.
void RTTIBuilder::BuildPointerToMemberTypeInfo(const MemberPointerType *Ty) {
  QualType PointeeTy = Ty->getPointeeType();
  
  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other 
  //   attributes of the type pointed to.
  unsigned Flags = DetermineQualifierFlags(PointeeTy.getQualifiers());

  const RecordType *ClassType = cast<RecordType>(Ty->getClass());
  
  if (IsIncompleteClassType(ClassType))
    Flags |= PTI_ContainingClassIncomplete;
  
  // FIXME: Handle PTI_Incomplete.
  
  const llvm::Type *UnsignedIntLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Info.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));
  
  // Itanium C++ ABI 2.9.5p7:
  //   __pointee is a pointer to the std::type_info derivation for the 
  //   unqualified type being pointed to.
  Info.push_back(RTTIBuilder(CGM).BuildType(PointeeTy.getUnqualifiedType()));

  // Itanium C++ ABI 2.9.5p9:
  //   __context is a pointer to an abi::__class_type_info corresponding to the
  //   class type containing the member pointed to 
  //   (e.g., the "A" in "int A::*").
  Info.push_back(RTTIBuilder(CGM).BuildType(QualType(ClassType, 0)));
}

llvm::Constant *CodeGenModule::GetAddrOfRTTIDescriptor(QualType Ty) {
  if (!getContext().getLangOptions().RTTI) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    return llvm::Constant::getNullValue(Int8PtrTy);
  }
  
  return RTTIBuilder(*this).BuildType(Ty);
}
