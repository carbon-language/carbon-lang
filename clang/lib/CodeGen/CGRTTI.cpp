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
  
  /// Fields - The fields of the RTTI descriptor currently being built.
  llvm::SmallVector<llvm::Constant *, 16> Fields;

  /// GetAddrOfExternalRTTIDescriptor - Returns the constant for the RTTI 
  /// descriptor of the given type.
  llvm::Constant *GetAddrOfExternalRTTIDescriptor(QualType Ty);
  
  /// BuildVtablePointer - Build the vtable pointer for the given type.
  void BuildVtablePointer(const Type *Ty);
  
  /// BuildSIClassTypeInfo - Build an abi::__si_class_type_info, used for single
  /// inheritance, according to the Itanium C++ ABI, 2.9.5p6b.
  void BuildSIClassTypeInfo(const CXXRecordDecl *RD);
  
  /// BuildVMIClassTypeInfo - Build an abi::__vmi_class_type_info, used for
  /// classes with bases that do not satisfy the abi::__si_class_type_info 
  /// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
  void BuildVMIClassTypeInfo(const CXXRecordDecl *RD);
  
  /// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct, used
  /// for pointer types.
  void BuildPointerTypeInfo(const PointerType *Ty);
  
  /// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info 
  /// struct, used for member pointer types.
  void BuildPointerToMemberTypeInfo(const MemberPointerType *Ty);
  
public:
  RTTIBuilder(CodeGenModule &cgm)
    : CGM(cgm), VMContext(cgm.getModule().getContext()),
      Int8PtrTy(llvm::Type::getInt8PtrTy(VMContext)) { }

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

  // FIXME: unify with DecideExtern
  bool DecideHidden(QualType Ty) {
    // For this type, see if all components are never hidden.
    if (const MemberPointerType *MPT = Ty->getAs<MemberPointerType>())
      return (DecideHidden(MPT->getPointeeType())
              && DecideHidden(QualType(MPT->getClass(), 0)));
    if (const PointerType *PT = Ty->getAs<PointerType>())
      return DecideHidden(PT->getPointeeType());
    if (const FunctionType *FT = Ty->getAs<FunctionType>()) {
      if (DecideHidden(FT->getResultType()) == false)
        return false;
      if (const FunctionProtoType *FPT = Ty->getAs<FunctionProtoType>()) {
        for (unsigned i = 0; i <FPT->getNumArgs(); ++i)
          if (DecideHidden(FPT->getArgType(i)) == false)
            return false;
        for (unsigned i = 0; i <FPT->getNumExceptions(); ++i)
          if (DecideHidden(FPT->getExceptionType(i)) == false)
            return false;
        return true;
      }
    }
    if (const RecordType *RT = Ty->getAs<RecordType>())
      if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
        return CGM.getDeclVisibilityMode(RD) == LangOptions::Hidden;
    return false;
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
  
  // VMI type info flags.
  enum {
    /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
    VMI_NonDiamondRepeat = 0x1,
    
    /// VMI_DiamondShaped - Class is diamond shaped.
    VMI_DiamondShaped = 0x2
  };
  
  // Base class type info flags.
  enum {
    /// BCTI_Virtual - Base class is virtual.
    BCTI_Virtual = 0x1,
    
    /// BCTI_Public - Base class is public.
    BCTI_Public = 0x2
  };
  
  /// BuildTypeInfo - Build the RTTI type info struct for the given type.
  llvm::Constant *BuildTypeInfo(QualType Ty);
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
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());
    if (!RD->isDynamicClass())
      return false;

    // Get the key function.
    const CXXMethodDecl *KeyFunction = RD->getASTContext().getKeyFunction(RD);
    if (KeyFunction && !KeyFunction->getBody()) {
      // The class has a key function, but it is not defined in this translation
      // unit, so we should use the external descriptor for it.
      return true;
    }
  }
  
  return false;
}

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
static bool IsIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getDecl()->isDefinition();
}  

/// ContainsIncompleteClassType - Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an 
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
static bool ContainsIncompleteClassType(QualType Ty) {
  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    if (IsIncompleteClassType(RecordTy))
      return true;
  }
  
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return ContainsIncompleteClassType(PointerTy->getPointeeType());
  
  if (const MemberPointerType *MemberPointerTy = 
      dyn_cast<MemberPointerType>(Ty)) {
    // Check if the class type is incomplete.
    const RecordType *ClassType = cast<RecordType>(MemberPointerTy->getClass());
    if (IsIncompleteClassType(ClassType))
      return true;
    
    return ContainsIncompleteClassType(MemberPointerTy->getPointeeType());
  }
  
  return false;
}

/// getTypeInfoLinkage - Return the linkage that the type info and type info
/// name constants should have for the given type.
static llvm::GlobalVariable::LinkageTypes getTypeInfoLinkage(QualType Ty) {
  // Itanium C++ ABI 2.9.5p7:
  //   In addition, it and all of the intermediate abi::__pointer_type_info 
  //   structs in the chain down to the abi::__class_type_info for the
  //   incomplete class type must be prevented from resolving to the 
  //   corresponding type_info structs for the complete class type, possibly
  //   by making them local static objects. Finally, a dummy class RTTI is
  //   generated for the incomplete type that will not resolve to the final 
  //   complete class RTTI (because the latter need not exist), possibly by 
  //   making it a local static object.
  if (ContainsIncompleteClassType(Ty))
    return llvm::GlobalValue::InternalLinkage;
  
  switch (Ty->getTypeClass()) {
  default:   
    // FIXME: We need to add code to handle all types.
    assert(false && "Unhandled type!");
    break;

  case Type::Pointer: {
    const PointerType *PointerTy = cast<PointerType>(Ty);
 
    // If the pointee type has internal linkage, then the pointer type needs to
    // have it as well.
    if (getTypeInfoLinkage(PointerTy->getPointeeType()) == 
        llvm::GlobalVariable::InternalLinkage)
      return llvm::GlobalVariable::InternalLinkage;
    
    return llvm::GlobalVariable::WeakODRLinkage;
  }

  case Type::Enum: {
    const EnumType *EnumTy = cast<EnumType>(Ty);
    const EnumDecl *ED = EnumTy->getDecl();
    
    // If we're in an anonymous namespace, then we always want internal linkage.
    if (ED->isInAnonymousNamespace() || !ED->hasLinkage())
      return llvm::GlobalVariable::InternalLinkage;
    
    return llvm::GlobalValue::WeakODRLinkage;
  }

  case Type::Record: {
    const RecordType *RecordTy = cast<RecordType>(Ty);
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());

    // If we're in an anonymous namespace, then we always want internal linkage.
    if (RD->isInAnonymousNamespace() || !RD->hasLinkage())
      return llvm::GlobalVariable::InternalLinkage;
    
    if (!RD->isDynamicClass())
      return llvm::GlobalValue::WeakODRLinkage;
    
    // Get the key function.
    const CXXMethodDecl *KeyFunction = RD->getASTContext().getKeyFunction(RD);
    if (!KeyFunction) {
      // There is no key function, the RTTI descriptor is emitted with weak_odr
      // linkage.
      return llvm::GlobalValue::WeakODRLinkage;
    }

    // Otherwise, the RTTI descriptor is emitted with external linkage.
    return llvm::GlobalValue::ExternalLinkage;
  }

  case Type::Vector:
  case Type::ExtVector:
  case Type::Builtin:
    return llvm::GlobalValue::WeakODRLinkage;

  case Type::FunctionProto: {
    const FunctionProtoType *FPT = cast<FunctionProtoType>(Ty);

    // Check the return type.
    if (getTypeInfoLinkage(FPT->getResultType()) == 
        llvm::GlobalValue::InternalLinkage)
      return llvm::GlobalValue::InternalLinkage;
    
    // Check the parameter types.
    for (unsigned i = 0; i != FPT->getNumArgs(); ++i) {
      if (getTypeInfoLinkage(FPT->getArgType(i)) == 
          llvm::GlobalValue::InternalLinkage)
        return llvm::GlobalValue::InternalLinkage;
    }
    
    return llvm::GlobalValue::WeakODRLinkage;
  }
  
  case Type::ConstantArray: 
  case Type::IncompleteArray: {
    const ArrayType *AT = cast<ArrayType>(Ty);

    // Check the element type.
    if (getTypeInfoLinkage(AT->getElementType()) ==
        llvm::GlobalValue::InternalLinkage)
      return llvm::GlobalValue::InternalLinkage;
  }

  }

  return llvm::GlobalValue::WeakODRLinkage;
}

// CanUseSingleInheritance - Return whether the given record decl has a "single, 
// public, non-virtual base at offset zero (i.e. the derived class is dynamic 
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
static bool CanUseSingleInheritance(const CXXRecordDecl *RD) {
  // Check the number of bases.
  if (RD->getNumBases() != 1)
    return false;
  
  // Get the base.
  CXXRecordDecl::base_class_const_iterator Base = RD->bases_begin();
  
  // Check that the base is not virtual.
  if (Base->isVirtual())
    return false;
  
  // Check that the base is public.
  if (Base->getAccessSpecifier() != AS_public)
    return false;
  
  // Check that the class is dynamic iff the base is.
  const CXXRecordDecl *BaseDecl = 
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
  if (!BaseDecl->isEmpty() && 
      BaseDecl->isDynamicClass() != RD->isDynamicClass())
    return false;
  
  return true;
}

void RTTIBuilder::BuildVtablePointer(const Type *Ty) {
  const char *VtableName;

  switch (Ty->getTypeClass()) {
  default: assert(0 && "Unhandled type!");

  // GCC treats vector types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
    // abi::__fundamental_type_info.
    VtableName = "_ZTVN10__cxxabiv123__fundamental_type_infoE";
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
    // abi::__array_type_info.
    VtableName = "_ZTVN10__cxxabiv117__array_type_infoE";
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // abi::__function_type_info.
    VtableName = "_ZTVN10__cxxabiv120__function_type_infoE";
    break;

  case Type::Enum:
    // abi::__enum_type_info.
    VtableName = "_ZTVN10__cxxabiv116__enum_type_infoE";
    break;
      
  case Type::Record: {
    const CXXRecordDecl *RD = 
      cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());
    
    if (!RD->getNumBases()) {
      // abi::__class_type_info.
      VtableName = "_ZTVN10__cxxabiv117__class_type_infoE";
    } else if (CanUseSingleInheritance(RD)) {
      // abi::__si_class_type_info.
      VtableName = "_ZTVN10__cxxabiv120__si_class_type_infoE";
    } else {
      // abi::__vmi_class_type_info.
      VtableName = "_ZTVN10__cxxabiv121__vmi_class_type_infoE";
    }
    
    break;
  }

  case Type::Pointer:
    // abi::__pointer_type_info.
    VtableName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
    break;

  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info.
    VtableName = "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";
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

  Fields.push_back(Vtable);
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
  Fields.push_back(BuildName(Ty, DecideHidden(Ty), Linkage));
  
  switch (Ty->getTypeClass()) {
  default: assert(false && "Unhandled type class!");
  case Type::Builtin:
    assert(false && "Builtin type info must be in the standard library!");
    break;

  // GCC treats vector types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
    // Itanium C++ ABI 2.9.5p4:
    // abi::__fundamental_type_info adds no data members to std::type_info.
    break;
      
  case Type::ConstantArray:
  case Type::IncompleteArray:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__array_type_info adds no data members to std::type_info.
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__function_type_info adds no data members to std::type_info.
    break;

  case Type::Enum:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__enum_type_info adds no data members to std::type_info.
    break;

  case Type::Record: {
    const CXXRecordDecl *RD = 
      cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());
    if (!RD->getNumBases()) {
      // We don't need to emit any fields.
      break;
    }
    
    if (CanUseSingleInheritance(RD))
      BuildSIClassTypeInfo(RD);
    else 
      BuildVMIClassTypeInfo(RD);

    break;
  }
      
  case Type::Pointer:
    BuildPointerTypeInfo(cast<PointerType>(Ty));
    break;
  
  case Type::MemberPointer:
    BuildPointerToMemberTypeInfo(cast<MemberPointerType>(Ty));
    break;
  }

  llvm::Constant *Init = 
    llvm::ConstantStruct::get(VMContext, &Fields[0], Fields.size(), 
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

/// ComputeQualifierFlags - Compute the pointer type info flags from the
/// given qualifier.
static unsigned ComputeQualifierFlags(Qualifiers Quals) {
  unsigned Flags = 0;

  if (Quals.hasConst())
    Flags |= RTTIBuilder::PTI_Const;
  if (Quals.hasVolatile())
    Flags |= RTTIBuilder::PTI_Volatile;
  if (Quals.hasRestrict())
    Flags |= RTTIBuilder::PTI_Restrict;

  return Flags;
}

/// BuildSIClassTypeInfo - Build an abi::__si_class_type_info, used for single
/// inheritance, according to the Itanium C++ ABI, 2.95p6b.
void RTTIBuilder::BuildSIClassTypeInfo(const CXXRecordDecl *RD) {
  // Itanium C++ ABI 2.9.5p6b:
  // It adds to abi::__class_type_info a single member pointing to the 
  // type_info structure for the base type,
  llvm::Constant *BaseTypeInfo = 
    RTTIBuilder(CGM).BuildTypeInfo(RD->bases_begin()->getType());
  Fields.push_back(BaseTypeInfo);
}

/// SeenBases - Contains virtual and non-virtual bases seen when traversing
/// a class hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> NonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> VirtualBases;
};

/// ComputeVMIClassTypeInfoFlags - Compute the value of the flags member in
/// abi::__vmi_class_type_info.
///
static unsigned ComputeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base, 
                                             SeenBases &Bases) {
  
  unsigned Flags = 0;
  
  const CXXRecordDecl *BaseDecl = 
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
  
  if (Base->isVirtual()) {
    if (Bases.VirtualBases.count(BaseDecl)) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      Flags |= RTTIBuilder::VMI_DiamondShaped;
    } else {
      if (Bases.NonVirtualBases.count(BaseDecl))
        Flags |= RTTIBuilder::VMI_NonDiamondRepeat;

      // Mark the virtual base as seen.
      Bases.VirtualBases.insert(BaseDecl);
    }
  } else {
    if (Bases.NonVirtualBases.count(BaseDecl)) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      Flags |= RTTIBuilder::VMI_NonDiamondRepeat;
    } else {
      if (Bases.VirtualBases.count(BaseDecl))
        Flags |= RTTIBuilder::VMI_NonDiamondRepeat;
        
      // Mark the non-virtual base as seen.
      Bases.NonVirtualBases.insert(BaseDecl);
    }
  }

  // Walk all bases.
  for (CXXRecordDecl::base_class_const_iterator I = BaseDecl->bases_begin(),
       E = BaseDecl->bases_end(); I != E; ++I) 
    Flags |= ComputeVMIClassTypeInfoFlags(I, Bases);
  
  return Flags;
}

static unsigned ComputeVMIClassTypeInfoFlags(const CXXRecordDecl *RD) {
  unsigned Flags = 0;
  SeenBases Bases;
  
  // Walk all bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) 
    Flags |= ComputeVMIClassTypeInfoFlags(I, Bases);
  
  return Flags;
}

/// BuildVMIClassTypeInfo - Build an abi::__vmi_class_type_info, used for
/// classes with bases that do not satisfy the abi::__si_class_type_info 
/// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
void RTTIBuilder::BuildVMIClassTypeInfo(const CXXRecordDecl *RD) {
  const llvm::Type *UnsignedIntLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  
  // Itanium C++ ABI 2.9.5p6c:
  //   __flags is a word with flags describing details about the class 
  //   structure, which may be referenced by using the __flags_masks 
  //   enumeration. These flags refer to both direct and indirect bases. 
  unsigned Flags = ComputeVMIClassTypeInfoFlags(RD);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_count is a word with the number of direct proper base class 
  //   descriptions that follow.
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, RD->getNumBases()));
  
  if (!RD->getNumBases())
    return;
  
  const llvm::Type *LongLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().LongTy);

  // Now add the base class descriptions.
  
  // Itanium C++ ABI 2.9.5p6c:
  //   __base_info[] is an array of base class descriptions -- one for every 
  //   direct proper base. Each description is of the type:
  //
  //   struct abi::__base_class_type_info {
	//   public:
  //     const __class_type_info *__base_type;
  //     long __offset_flags;
  //
  //     enum __offset_flags_masks {
  //       __virtual_mask = 0x1,
  //       __public_mask = 0x2,
  //       __offset_shift = 8
  //     };
  //   };
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXBaseSpecifier *Base = I;

    // The __base_type member points to the RTTI for the base type.
    Fields.push_back(RTTIBuilder(CGM).BuildTypeInfo(Base->getType()));

    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());

    int64_t OffsetFlags = 0;
    
    // All but the lower 8 bits of __offset_flags are a signed offset. 
    // For a non-virtual base, this is the offset in the object of the base
    // subobject. For a virtual base, this is the offset in the virtual table of
    // the virtual base offset for the virtual base referenced (negative).
    if (Base->isVirtual())
      OffsetFlags = CGM.getVtableInfo().getVirtualBaseOffsetIndex(RD, BaseDecl);
    else {
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      OffsetFlags = Layout.getBaseClassOffset(BaseDecl) / 8;
    };
    
    OffsetFlags <<= 8;
    
    // The low-order byte of __offset_flags contains flags, as given by the 
    // masks from the enumeration __offset_flags_masks.
    if (Base->isVirtual())
      OffsetFlags |= BCTI_Virtual;
    if (Base->getAccessSpecifier() == AS_public)
      OffsetFlags |= BCTI_Public;

    Fields.push_back(llvm::ConstantInt::get(LongLTy, OffsetFlags));
  }
}

/// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct,
/// used for pointer types.
void RTTIBuilder::BuildPointerTypeInfo(const PointerType *Ty) {
  QualType PointeeTy = Ty->getPointeeType();
  
  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other 
  //   attributes of the type pointed to
  unsigned Flags = ComputeQualifierFlags(PointeeTy.getQualifiers());

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set. 
  if (ContainsIncompleteClassType(PointeeTy))
    Flags |= PTI_Incomplete;

  const llvm::Type *UnsignedIntLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));
  
  // Itanium C++ ABI 2.9.5p7:
  //  __pointee is a pointer to the std::type_info derivation for the 
  //  unqualified type being pointed to.
  llvm::Constant *PointeeTypeInfo = 
    RTTIBuilder(CGM).BuildTypeInfo(PointeeTy.getUnqualifiedType());
  Fields.push_back(PointeeTypeInfo);
}

/// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info 
/// struct, used for member pointer types.
void RTTIBuilder::BuildPointerToMemberTypeInfo(const MemberPointerType *Ty) {
  QualType PointeeTy = Ty->getPointeeType();
  
  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other 
  //   attributes of the type pointed to.
  unsigned Flags = ComputeQualifierFlags(PointeeTy.getQualifiers());

  const RecordType *ClassType = cast<RecordType>(Ty->getClass());

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set. 
  if (ContainsIncompleteClassType(PointeeTy))
    Flags |= PTI_Incomplete;

  if (IsIncompleteClassType(ClassType))
    Flags |= PTI_ContainingClassIncomplete;
  
  const llvm::Type *UnsignedIntLTy = 
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));
  
  // Itanium C++ ABI 2.9.5p7:
  //   __pointee is a pointer to the std::type_info derivation for the 
  //   unqualified type being pointed to.
  llvm::Constant *PointeeTypeInfo = 
    RTTIBuilder(CGM).BuildTypeInfo(PointeeTy.getUnqualifiedType());
  Fields.push_back(PointeeTypeInfo);

  // Itanium C++ ABI 2.9.5p9:
  //   __context is a pointer to an abi::__class_type_info corresponding to the
  //   class type containing the member pointed to 
  //   (e.g., the "A" in "int A::*").
  Fields.push_back(RTTIBuilder(CGM).BuildTypeInfo(QualType(ClassType, 0)));
}

llvm::Constant *CodeGenModule::GetAddrOfRTTIDescriptor(QualType Ty) {
  if (!getContext().getLangOptions().RTTI) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    return llvm::Constant::getNullValue(Int8PtrTy);
  }
  
  return RTTIBuilder(*this).BuildTypeInfo(Ty);
}
