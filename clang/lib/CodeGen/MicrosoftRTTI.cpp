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

#include "CodeGenModule.h"
#include "CGCXXABI.h"
#include "CGObjCRuntime.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CodeGenOptions.h"

using namespace clang;
using namespace CodeGen;

// MS RTTI Overview:
// The run time type information emitted by cl.exe contains 5 distinct types of
// structures.  Many of them reference each other.
//
// TypeInfo:  Static classes that are returned by typeid.
//
// CompleteObjectLocator:  Referenced by vftables.  They contain information
//   required for dynamic casting, including OffsetFromTop.  They also contain
//   a reference to the TypeInfo for the type and a reference to the
//   CompleteHierarchyDescriptor for the type.
//
// ClassHieararchyDescriptor: Contains information about a class hierarchy.
//   Used during dynamic_cast to walk a class hierarchy.  References a base
//   class array and the size of said array.
//
// BaseClassArray: Contains a list of classes in a hierarchy.  BaseClassArray is
//   somewhat of a misnomer because the most derived class is also in the list
//   as well as multiple copies of virtual bases (if they occur multiple times
//   in the hiearchy.)  The BaseClassArray contains one BaseClassDescriptor for
//   every path in the hierarchy, in pre-order depth first order.  Note, we do
//   not declare a specific llvm type for BaseClassArray, it's merely an array
//   of BaseClassDescriptor pointers.
//
// BaseClassDescriptor: Contains information about a class in a class hierarchy.
//   BaseClassDescriptor is also somewhat of a misnomer for the same reason that
//   BaseClassArray is.  It contains information about a class within a
//   hierarchy such as: is this base is ambiguous and what is its offset in the
//   vbtable.  The names of the BaseClassDescriptors have all of their fields
//   mangled into them so they can be aggressively deduplicated by the linker.

// 5 routines for constructing the llvm types for MS RTTI structs.
static llvm::StructType *getClassHierarchyDescriptorType(CodeGenModule &CGM);

static llvm::StructType *getTypeDescriptorType(CodeGenModule &CGM,
                                               StringRef TypeInfoString) {
  llvm::SmallString<32> TDTypeName("MSRTTITypeDescriptor");
  TDTypeName += TypeInfoString.size();
  if (auto Type = CGM.getModule().getTypeByName(TDTypeName))
    return Type;
  llvm::Type *FieldTypes[] = {
      CGM.Int8PtrPtrTy,
      CGM.Int8PtrTy,
      llvm::ArrayType::get(CGM.Int8Ty, TypeInfoString.size() + 1)};
  return llvm::StructType::create(CGM.getLLVMContext(), FieldTypes, TDTypeName);
}

static llvm::StructType *getBaseClassDescriptorType(CodeGenModule &CGM) {
  static const char Name[] = "MSRTTIBaseClassDescriptor";
  if (auto Type = CGM.getModule().getTypeByName(Name))
    return Type;
  llvm::Type *FieldTypes[] = {
      CGM.Int8PtrTy,
      CGM.IntTy,
      CGM.IntTy,
      CGM.IntTy,
      CGM.IntTy,
      CGM.IntTy,
      getClassHierarchyDescriptorType(CGM)->getPointerTo()};
  return llvm::StructType::create(CGM.getLLVMContext(), FieldTypes, Name);
}

static llvm::StructType *getClassHierarchyDescriptorType(CodeGenModule &CGM) {
  static const char Name[] = "MSRTTIClassHierarchyDescriptor";
  if (auto Type = CGM.getModule().getTypeByName(Name))
    return Type;
  // Forward declare RTTIClassHierarchyDescriptor to break a cycle.
  llvm::StructType *Type = llvm::StructType::create(CGM.getLLVMContext(), Name);
  llvm::Type *FieldTypes[] = {
    CGM.IntTy,
    CGM.IntTy,
    CGM.IntTy,
    getBaseClassDescriptorType(CGM)->getPointerTo()->getPointerTo()};
  Type->setBody(FieldTypes);
  return Type;
}

static llvm::StructType *getCompleteObjectLocatorType(CodeGenModule &CGM) {
  static const char Name[] = "MSRTTICompleteObjectLocator";
  if (auto Type = CGM.getModule().getTypeByName(Name))
    return Type;
  llvm::Type *FieldTypes[] = {
    CGM.IntTy,
    CGM.IntTy,
    CGM.IntTy,
    CGM.Int8PtrTy,
    getClassHierarchyDescriptorType(CGM)->getPointerTo() };
  return llvm::StructType::create(CGM.getLLVMContext(), FieldTypes, Name);
}

static llvm::GlobalVariable *getTypeInfoVTable(CodeGenModule &CGM) {
  StringRef MangledName("\01??_7type_info@@6B@");
  if (auto VTable = CGM.getModule().getNamedGlobal(MangledName))
    return VTable;
  return new llvm::GlobalVariable(CGM.getModule(), CGM.Int8PtrTy,
                                  /*Constant=*/true,
                                  llvm::GlobalVariable::ExternalLinkage,
                                  /*Initializer=*/0, MangledName);
}

namespace {

/// \brief A Helper struct that stores information about a class in a class
/// hierarchy.  The information stored in these structs struct is used during
/// the generation of ClassHierarchyDescriptors and BaseClassDescriptors.
// During RTTI creation, MSRTTIClasses are stored in a contiguous array with
// implicit depth first pre-order tree connectivity.  getFirstChild and
// getNextSibling allow us to walk the tree efficiently.
struct MSRTTIClass {
  enum {
    IsPrivateOnPath = 1 | 8,
    IsAmbiguous = 2,
    IsPrivate = 4,
    IsVirtual = 16,
    HasHierarchyDescriptor = 64
  };
  MSRTTIClass(const CXXRecordDecl *RD) : RD(RD) {}
  uint32_t initialize(const MSRTTIClass *Parent,
                      const CXXBaseSpecifier *Specifier);

  MSRTTIClass *getFirstChild() { return this + 1; }
  static MSRTTIClass *getNextChild(MSRTTIClass *Child) {
    return Child + 1 + Child->NumBases;
  }

  const CXXRecordDecl *RD, *VirtualRoot;
  uint32_t Flags, NumBases, OffsetInVBase;
};

/// \brief Recursively initialize the base class array.
uint32_t MSRTTIClass::initialize(const MSRTTIClass *Parent,
                                 const CXXBaseSpecifier *Specifier) {
  Flags = HasHierarchyDescriptor;
  if (!Parent) {
    VirtualRoot = 0;
    OffsetInVBase = 0;
  } else {
    if (Specifier->getAccessSpecifier() != AS_public)
      Flags |= IsPrivate | IsPrivateOnPath;
    if (Specifier->isVirtual()) {
      Flags |= IsVirtual;
      VirtualRoot = RD;
      OffsetInVBase = 0;
    } else {
      if (Parent->Flags & IsPrivateOnPath)
        Flags |= IsPrivateOnPath;
      VirtualRoot = Parent->VirtualRoot;
      OffsetInVBase = Parent->OffsetInVBase + RD->getASTContext()
          .getASTRecordLayout(Parent->RD).getBaseClassOffset(RD).getQuantity();
    }
  }
  NumBases = 0;
  MSRTTIClass *Child = getFirstChild();
  for (const CXXBaseSpecifier &Base : RD->bases()) {
    NumBases += Child->initialize(this, &Base) + 1;
    Child = getNextChild(Child);
  }
  return NumBases;
}

/// \brief An ephemeral helper class for building MS RTTI types.  It caches some
/// calls to the module and information about the most derived class in a
/// hierarchy.
struct MSRTTIBuilder {
  enum {
    HasBranchingHierarchy = 1,
    HasVirtualBranchingHierarchy = 2,
    HasAmbiguousBases = 4
  };

  MSRTTIBuilder(CodeGenModule &CGM, const CXXRecordDecl *RD)
      : CGM(CGM), Context(CGM.getContext()), VMContext(CGM.getLLVMContext()),
        Module(CGM.getModule()), RD(RD), Linkage(CGM.getVTableLinkage(RD)),
        Mangler(
            cast<MicrosoftMangleContext>(CGM.getCXXABI().getMangleContext())) {}

  llvm::GlobalVariable *getBaseClassDescriptor(const MSRTTIClass &Classes);
  llvm::GlobalVariable *
  getBaseClassArray(SmallVectorImpl<MSRTTIClass> &Classes);
  llvm::GlobalVariable *getClassHierarchyDescriptor();
  llvm::GlobalVariable *getCompleteObjectLocator(const VPtrInfo *Info);

  CodeGenModule &CGM;
  ASTContext &Context;
  llvm::LLVMContext &VMContext;
  llvm::Module &Module;
  const CXXRecordDecl *RD;
  llvm::GlobalVariable::LinkageTypes Linkage;
  MicrosoftMangleContext &Mangler;
};

} // namespace

/// \brief Recursively serializes a class hierarchy in pre-order depth first
/// order.
static void serializeClassHierarchy(SmallVectorImpl<MSRTTIClass> &Classes,
                                    const CXXRecordDecl *RD) {
  Classes.push_back(MSRTTIClass(RD));
  for (const CXXBaseSpecifier &Base : RD->bases())
    serializeClassHierarchy(Classes, Base.getType()->getAsCXXRecordDecl());
}

/// \brief Find ambiguity among base classes.
static void
detectAmbiguousBases(SmallVectorImpl<MSRTTIClass> &Classes) {
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> VirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> UniqueBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> AmbiguousBases;
  for (MSRTTIClass *Class = &Classes.front(); Class <= &Classes.back();) {
    if ((Class->Flags & MSRTTIClass::IsVirtual) &&
        !VirtualBases.insert(Class->RD)) {
      Class = MSRTTIClass::getNextChild(Class);
      continue;
    }
    if (!UniqueBases.insert(Class->RD))
      AmbiguousBases.insert(Class->RD);
    Class++;
  }
  if (AmbiguousBases.empty())
    return;
  for (MSRTTIClass &Class : Classes)
    if (AmbiguousBases.count(Class.RD))
      Class.Flags |= MSRTTIClass::IsAmbiguous;
}

llvm::GlobalVariable *MSRTTIBuilder::getClassHierarchyDescriptor() {
  SmallString<256> MangledName;
  {
    llvm::raw_svector_ostream Out(MangledName);
    Mangler.mangleCXXRTTIClassHierarchyDescriptor(RD, Out);
  }

  // Check to see if we've already declared this ClassHierarchyDescriptor.
  if (auto CHD = Module.getNamedGlobal(MangledName))
    return CHD;

  // Serialize the class hierarchy and initalize the CHD Fields.
  SmallVector<MSRTTIClass, 8> Classes;
  serializeClassHierarchy(Classes, RD);
  Classes.front().initialize(/*Parent=*/0, /*Specifier=*/0);
  detectAmbiguousBases(Classes);
  int Flags = 0;
  for (auto Class : Classes) {
    if (Class.RD->getNumBases() > 1)
      Flags |= HasBranchingHierarchy;
    // Note: cl.exe does not calculate "HasAmbiguousBases" correctly.  We
    // believe the field isn't actually used.
    if (Class.Flags & MSRTTIClass::IsAmbiguous)
      Flags |= HasAmbiguousBases;
  }
  if ((Flags & HasBranchingHierarchy) && RD->getNumVBases() != 0)
    Flags |= HasVirtualBranchingHierarchy;
  // These gep indices are used to get the address of the first element of the
  // base class array.
  llvm::Value *GEPIndices[] = {llvm::ConstantInt::get(CGM.IntTy, 0),
                               llvm::ConstantInt::get(CGM.IntTy, 0)};

  // Forward declare the class hierarchy descriptor
  auto Type = getClassHierarchyDescriptorType(CGM);
  auto CHD = new llvm::GlobalVariable(Module, Type, /*Constant=*/true, Linkage,
                                      /*Initializer=*/0, MangledName.c_str());

  // Initialize the base class ClassHierarchyDescriptor.
  llvm::Constant *Fields[] = {
    llvm::ConstantInt::get(CGM.IntTy, 0), // Unknown
    llvm::ConstantInt::get(CGM.IntTy, Flags),
    llvm::ConstantInt::get(CGM.IntTy, Classes.size()),
    llvm::ConstantExpr::getInBoundsGetElementPtr(
        getBaseClassArray(Classes),
        llvm::ArrayRef<llvm::Value *>(GEPIndices))};
  CHD->setInitializer(llvm::ConstantStruct::get(Type, Fields));
  return CHD;
}

llvm::GlobalVariable *
MSRTTIBuilder::getBaseClassArray(SmallVectorImpl<MSRTTIClass> &Classes) {
  SmallString<256> MangledName;
  {
    llvm::raw_svector_ostream Out(MangledName);
    Mangler.mangleCXXRTTIBaseClassArray(RD, Out);
  }

  // Foward declare the base class array.
  // cl.exe pads the base class array with 1 (in 32 bit mode) or 4 (in 64 bit
  // mode) bytes of padding.  We provide a pointer sized amount of padding by
  // adding +1 to Classes.size().  The sections have pointer alignment and are
  // marked pick-any so it shouldn't matter.
  auto PtrType = getBaseClassDescriptorType(CGM)->getPointerTo();
  auto ArrayType = llvm::ArrayType::get(PtrType, Classes.size() + 1);
  auto BCA = new llvm::GlobalVariable(Module, ArrayType,
      /*Constant=*/true, Linkage, /*Initializer=*/0, MangledName.c_str());

  // Initialize the BaseClassArray.
  SmallVector<llvm::Constant *, 8> BaseClassArrayData;
  for (MSRTTIClass &Class : Classes)
    BaseClassArrayData.push_back(getBaseClassDescriptor(Class));
  BaseClassArrayData.push_back(llvm::ConstantPointerNull::get(PtrType));
  BCA->setInitializer(llvm::ConstantArray::get(ArrayType, BaseClassArrayData));
  return BCA;
}

llvm::GlobalVariable *
MSRTTIBuilder::getBaseClassDescriptor(const MSRTTIClass &Class) {
  // Compute the fields for the BaseClassDescriptor.  They are computed up front
  // because they are mangled into the name of the object.
  uint32_t OffsetInVBTable = 0;
  int32_t VBPtrOffset = -1;
  if (Class.VirtualRoot) {
    auto &VTableContext = CGM.getMicrosoftVTableContext();
    OffsetInVBTable = VTableContext.getVBTableIndex(RD, Class.VirtualRoot) * 4;
    VBPtrOffset = Context.getASTRecordLayout(RD).getVBPtrOffset().getQuantity();
  }

  SmallString<256> MangledName;
  {
    llvm::raw_svector_ostream Out(MangledName);
    Mangler.mangleCXXRTTIBaseClassDescriptor(Class.RD, Class.OffsetInVBase,
                                             VBPtrOffset, OffsetInVBTable,
                                             Class.Flags, Out);
  }

  // Check to see if we've already declared declared this object.
  if (auto BCD = Module.getNamedGlobal(MangledName))
    return BCD;

  // Forward declare the base class descriptor.
  auto Type = getBaseClassDescriptorType(CGM);
  auto BCD = new llvm::GlobalVariable(Module, Type, /*Constant=*/true, Linkage,
                                      /*Initializer=*/0, MangledName.c_str());

  // Initialize the BaseClassDescriptor.
  llvm::Constant *Fields[] = {
    CGM.getMSTypeDescriptor(Context.getTypeDeclType(Class.RD)),
    llvm::ConstantInt::get(CGM.IntTy, Class.NumBases),
    llvm::ConstantInt::get(CGM.IntTy, Class.OffsetInVBase),
    llvm::ConstantInt::get(CGM.IntTy, VBPtrOffset),
    llvm::ConstantInt::get(CGM.IntTy, OffsetInVBTable),
    llvm::ConstantInt::get(CGM.IntTy, Class.Flags),
    MSRTTIBuilder(CGM, Class.RD).getClassHierarchyDescriptor()};
  BCD->setInitializer(llvm::ConstantStruct::get(Type, Fields));
  return BCD;
}

llvm::GlobalVariable *
MSRTTIBuilder::getCompleteObjectLocator(const VPtrInfo *Info) {
  SmallString<256> MangledName;
  {
    llvm::raw_svector_ostream Out(MangledName);
    Mangler.mangleCXXRTTICompleteObjectLocator(RD, Info->MangledPath, Out);
  }

  // Check to see if we've already computed this complete object locator.
  if (auto COL = Module.getNamedGlobal(MangledName))
    return COL;

  // Compute the fields of the complete object locator.
  int OffsetToTop = Info->FullOffsetInMDC.getQuantity();
  int VFPtrOffset = 0;
  // The offset includes the vtordisp if one exists.
  if (const CXXRecordDecl *VBase = Info->getVBaseWithVPtr())
    if (Context.getASTRecordLayout(RD)
      .getVBaseOffsetsMap()
      .find(VBase)
      ->second.hasVtorDisp())
      VFPtrOffset = Info->NonVirtualOffset.getQuantity() + 4;

  // Forward declare the complete object locator.
  llvm::StructType *Type = getCompleteObjectLocatorType(CGM);
  auto COL = new llvm::GlobalVariable(Module, Type, /*Constant=*/true, Linkage,
    /*Initializer=*/0, MangledName.c_str());

  // Initialize the CompleteObjectLocator.
  llvm::Constant *Fields[] = {
    llvm::ConstantInt::get(CGM.IntTy, 0), // IsDeltaEncoded
    llvm::ConstantInt::get(CGM.IntTy, OffsetToTop),
    llvm::ConstantInt::get(CGM.IntTy, VFPtrOffset),
    CGM.getMSTypeDescriptor(Context.getTypeDeclType(RD)),
    getClassHierarchyDescriptor()};
  COL->setInitializer(llvm::ConstantStruct::get(Type, Fields));
  return COL;
}


/// \brief Gets a TypeDescriptor.  Returns a llvm::Constant * rather than a
/// llvm::GlobalVariable * because different type descriptors have different
/// types, and need to be abstracted.  They are abstracting by casting the
/// address to an Int8PtrTy.
llvm::Constant *CodeGenModule::getMSTypeDescriptor(QualType Type) {
  auto &Mangler(cast<MicrosoftMangleContext>(getCXXABI().getMangleContext()));
  SmallString<256> MangledName, TypeInfoString;
  {
    llvm::raw_svector_ostream Out(MangledName);
    Mangler.mangleCXXRTTI(Type, Out);
  }

  // Check to see if we've already declared this TypeDescriptor.
  if (auto TypeDescriptor = getModule().getNamedGlobal(MangledName))
    return llvm::ConstantExpr::getBitCast(TypeDescriptor, Int8PtrTy);

  // Compute the fields for the TypeDescriptor.
  {
    llvm::raw_svector_ostream Out(TypeInfoString);
    Mangler.mangleCXXRTTIName(Type, Out);
  }

  // Declare and initialize the TypeDescriptor.
  llvm::Constant *Fields[] = {
    getTypeInfoVTable(*this),                  // VFPtr
    llvm::ConstantPointerNull::get(Int8PtrTy), // Runtime data
    llvm::ConstantDataArray::getString(VMContext, TypeInfoString)};
  auto TypeDescriptorType = getTypeDescriptorType(*this, TypeInfoString);
  return llvm::ConstantExpr::getBitCast(
      new llvm::GlobalVariable(
          getModule(), TypeDescriptorType, /*Constant=*/false,
          getTypeInfoLinkage(Type),
          llvm::ConstantStruct::get(TypeDescriptorType, Fields),
          MangledName.c_str()),
      Int8PtrTy);
}

llvm::GlobalVariable *
CodeGenModule::getMSCompleteObjectLocator(const CXXRecordDecl *RD,
                                          const VPtrInfo *Info) {
  return MSRTTIBuilder(*this, RD).getCompleteObjectLocator(Info);
}
