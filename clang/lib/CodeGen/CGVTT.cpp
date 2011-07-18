//===--- CGVTT.cpp - Emit LLVM Code for C++ VTTs --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of VTTs (vtable tables).
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CGCXXABI.h"
#include "clang/AST/RecordLayout.h"
using namespace clang;
using namespace CodeGen;

#define D1(x)

namespace {

/// VTT builder - Class for building VTT layout information.
class VTTBuilder {
  
  CodeGenModule &CGM;

  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  typedef llvm::SmallVector<llvm::Constant *, 64> VTTComponentsVectorTy;
  
  /// VTTComponents - The VTT components.
  VTTComponentsVectorTy VTTComponents;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;

  typedef llvm::DenseMap<BaseSubobject, uint64_t> AddressPointsMapTy;

  /// SubVTTIndicies - The sub-VTT indices for the bases of the most derived
  /// class.
  llvm::DenseMap<BaseSubobject, uint64_t> SubVTTIndicies;

  /// SecondaryVirtualPointerIndices - The secondary virtual pointer indices of
  /// all subobjects of the most derived class.
  llvm::DenseMap<BaseSubobject, uint64_t> SecondaryVirtualPointerIndices;

  /// GenerateDefinition - Whether the VTT builder should generate LLVM IR for
  /// the VTT.
  bool GenerateDefinition;

  /// The linkage to use for any construction vtables required by this VTT.
  /// Only required if we're building a definition.
  llvm::GlobalVariable::LinkageTypes LinkageForConstructionVTables;
  
  /// GetAddrOfVTable - Returns the address of the vtable for the base class in
  /// the given vtable class.
  ///
  /// \param AddressPoints - If the returned vtable is a construction vtable,
  /// this will hold the address points for it.
  llvm::Constant *GetAddrOfVTable(BaseSubobject Base, bool BaseIsVirtual,
                                  AddressPointsMapTy& AddressPoints);

  /// AddVTablePointer - Add a vtable pointer to the VTT currently being built.
  ///
  /// \param AddressPoints - If the vtable is a construction vtable, this has
  /// the address points for it.
  void AddVTablePointer(BaseSubobject Base, llvm::Constant *VTable,
                        const CXXRecordDecl *VTableClass,
                        const AddressPointsMapTy& AddressPoints);
                        
  /// LayoutSecondaryVTTs - Lay out the secondary VTTs of the given base 
  /// subobject.
  void LayoutSecondaryVTTs(BaseSubobject Base);
  
  /// LayoutSecondaryVirtualPointers - Lay out the secondary virtual pointers
  /// for the given base subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  ///
  /// \param AddressPoints - If the vtable is a construction vtable, this has
  /// the address points for it.
  void LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                      bool BaseIsMorallyVirtual,
                                      llvm::Constant *VTable,
                                      const CXXRecordDecl *VTableClass,
                                      const AddressPointsMapTy& AddressPoints,
                                      VisitedVirtualBasesSetTy &VBases);
  
  /// LayoutSecondaryVirtualPointers - Lay out the secondary virtual pointers
  /// for the given base subobject.
  ///
  /// \param AddressPoints - If the vtable is a construction vtable, this has
  /// the address points for it.
  void LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                      llvm::Constant *VTable,
                                      const AddressPointsMapTy& AddressPoints);

  /// LayoutVirtualVTTs - Lay out the VTTs for the virtual base classes of the
  /// given record decl.
  void LayoutVirtualVTTs(const CXXRecordDecl *RD,
                         VisitedVirtualBasesSetTy &VBases);
  
  /// LayoutVTT - Will lay out the VTT for the given subobject, including any
  /// secondary VTTs, secondary virtual pointers and virtual VTTs.
  void LayoutVTT(BaseSubobject Base, bool BaseIsVirtual);
  
public:
  VTTBuilder(CodeGenModule &CGM, const CXXRecordDecl *MostDerivedClass,
             bool GenerateDefinition,
             llvm::GlobalVariable::LinkageTypes LinkageForConstructionVTables
               = (llvm::GlobalVariable::LinkageTypes) -1);

  // getVTTComponents - Returns a reference to the VTT components.
  const VTTComponentsVectorTy &getVTTComponents() const {
    return VTTComponents;
  }
  
  /// getSubVTTIndicies - Returns a reference to the sub-VTT indices.
  const llvm::DenseMap<BaseSubobject, uint64_t> &getSubVTTIndicies() const {
    return SubVTTIndicies;
  }
  
  /// getSecondaryVirtualPointerIndices - Returns a reference to the secondary
  /// virtual pointer indices.
  const llvm::DenseMap<BaseSubobject, uint64_t> &
  getSecondaryVirtualPointerIndices() const {
    return SecondaryVirtualPointerIndices;
  }

};

VTTBuilder::VTTBuilder(CodeGenModule &CGM,
                       const CXXRecordDecl *MostDerivedClass,
                       bool GenerateDefinition,
          llvm::GlobalVariable::LinkageTypes LinkageForConstructionVTables)
  : CGM(CGM), MostDerivedClass(MostDerivedClass), 
  MostDerivedClassLayout(CGM.getContext().getASTRecordLayout(MostDerivedClass)),
    GenerateDefinition(GenerateDefinition),
    LinkageForConstructionVTables(LinkageForConstructionVTables) {
  assert(!GenerateDefinition ||
         LinkageForConstructionVTables
           != (llvm::GlobalVariable::LinkageTypes) -1);
    
  // Lay out this VTT.
  LayoutVTT(BaseSubobject(MostDerivedClass, CharUnits::Zero()), 
            /*BaseIsVirtual=*/false);
}

llvm::Constant *
VTTBuilder::GetAddrOfVTable(BaseSubobject Base, bool BaseIsVirtual, 
                            AddressPointsMapTy& AddressPoints) {
  if (!GenerateDefinition)
    return 0;
  
  if (Base.getBase() == MostDerivedClass) {
    assert(Base.getBaseOffset().isZero() &&
           "Most derived class vtable must have a zero offset!");
    // This is a regular vtable.
    return CGM.getVTables().GetAddrOfVTable(MostDerivedClass);
  }
  
  return CGM.getVTables().GenerateConstructionVTable(MostDerivedClass, 
                                                     Base, BaseIsVirtual,
                                           LinkageForConstructionVTables,
                                                     AddressPoints);
}

void VTTBuilder::AddVTablePointer(BaseSubobject Base, llvm::Constant *VTable,
                                  const CXXRecordDecl *VTableClass,
                                  const AddressPointsMapTy& AddressPoints) {
  // Store the vtable pointer index if we're generating the primary VTT.
  if (VTableClass == MostDerivedClass) {
    assert(!SecondaryVirtualPointerIndices.count(Base) &&
           "A virtual pointer index already exists for this base subobject!");
    SecondaryVirtualPointerIndices[Base] = VTTComponents.size();
  }

  if (!GenerateDefinition) {
    VTTComponents.push_back(0);
    return;
  }

  uint64_t AddressPoint;
  if (VTableClass != MostDerivedClass) {
    // The vtable is a construction vtable, look in the construction vtable
    // address points.
    AddressPoint = AddressPoints.lookup(Base);
    assert(AddressPoint != 0 && "Did not find ctor vtable address point!");
  } else {
    // Just get the address point for the regular vtable.
    AddressPoint = CGM.getVTables().getAddressPoint(Base, VTableClass);
    assert(AddressPoint != 0 && "Did not find vtable address point!");
  }

  if (!AddressPoint) AddressPoint = 0;
  
  llvm::Value *Idxs[] = {
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()), 0),
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()), 
                           AddressPoint)
  };
  
  llvm::Constant *Init = 
    llvm::ConstantExpr::getInBoundsGetElementPtr(VTable, Idxs, 2);
  
  llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
  
  VTTComponents.push_back(Init);
}

void VTTBuilder::LayoutSecondaryVTTs(BaseSubobject Base) {
  const CXXRecordDecl *RD = Base.getBase();

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    
    // Don't layout virtual bases.
    if (I->isVirtual())
        continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    CharUnits BaseOffset = Base.getBaseOffset() + 
      Layout.getBaseClassOffset(BaseDecl);
   
    // Layout the VTT for this base.
    LayoutVTT(BaseSubobject(BaseDecl, BaseOffset), /*BaseIsVirtual=*/false);
  }
}

void
VTTBuilder::LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                        bool BaseIsMorallyVirtual,
                                        llvm::Constant *VTable,
                                        const CXXRecordDecl *VTableClass,
                                        const AddressPointsMapTy& AddressPoints,
                                        VisitedVirtualBasesSetTy &VBases) {
  const CXXRecordDecl *RD = Base.getBase();
  
  // We're not interested in bases that don't have virtual bases, and not
  // morally virtual bases.
  if (!RD->getNumVBases() && !BaseIsMorallyVirtual)
    return;

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Itanium C++ ABI 2.6.2:
    //   Secondary virtual pointers are present for all bases with either
    //   virtual bases or virtual function declarations overridden along a 
    //   virtual path.
    //
    // If the base class is not dynamic, we don't want to add it, nor any
    // of its base classes.
    if (!BaseDecl->isDynamicClass())
      continue;
    
    bool BaseDeclIsMorallyVirtual = BaseIsMorallyVirtual;
    bool BaseDeclIsNonVirtualPrimaryBase = false;
    CharUnits BaseOffset;
    if (I->isVirtual()) {
      // Ignore virtual bases that we've already visited.
      if (!VBases.insert(BaseDecl))
        continue;
      
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      BaseDeclIsMorallyVirtual = true;
    } else {
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      
      BaseOffset = Base.getBaseOffset() + 
        Layout.getBaseClassOffset(BaseDecl);
      
      if (!Layout.isPrimaryBaseVirtual() &&
          Layout.getPrimaryBase() == BaseDecl)
        BaseDeclIsNonVirtualPrimaryBase = true;
    }

    // Itanium C++ ABI 2.6.2:
    //   Secondary virtual pointers: for each base class X which (a) has virtual
    //   bases or is reachable along a virtual path from D, and (b) is not a
    //   non-virtual primary base, the address of the virtual table for X-in-D
    //   or an appropriate construction virtual table.
    if (!BaseDeclIsNonVirtualPrimaryBase &&
        (BaseDecl->getNumVBases() || BaseDeclIsMorallyVirtual)) {
      // Add the vtable pointer.
      AddVTablePointer(BaseSubobject(BaseDecl, BaseOffset), VTable, 
                       VTableClass, AddressPoints);
    }

    // And lay out the secondary virtual pointers for the base class.
    LayoutSecondaryVirtualPointers(BaseSubobject(BaseDecl, BaseOffset),
                                   BaseDeclIsMorallyVirtual, VTable, 
                                   VTableClass, AddressPoints, VBases);
  }
}

void 
VTTBuilder::LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                      llvm::Constant *VTable,
                                      const AddressPointsMapTy& AddressPoints) {
  VisitedVirtualBasesSetTy VBases;
  LayoutSecondaryVirtualPointers(Base, /*BaseIsMorallyVirtual=*/false,
                                 VTable, Base.getBase(), AddressPoints, VBases);
}

void VTTBuilder::LayoutVirtualVTTs(const CXXRecordDecl *RD,
                                   VisitedVirtualBasesSetTy &VBases) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    // Check if this is a virtual base.
    if (I->isVirtual()) {
      // Check if we've seen this base before.
      if (!VBases.insert(BaseDecl))
        continue;
    
      CharUnits BaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      
      LayoutVTT(BaseSubobject(BaseDecl, BaseOffset), /*BaseIsVirtual=*/true);
    }
    
    // We only need to layout virtual VTTs for this base if it actually has
    // virtual bases.
    if (BaseDecl->getNumVBases())
      LayoutVirtualVTTs(BaseDecl, VBases);
  }
}

void VTTBuilder::LayoutVTT(BaseSubobject Base, bool BaseIsVirtual) {
  const CXXRecordDecl *RD = Base.getBase();

  // Itanium C++ ABI 2.6.2:
  //   An array of virtual table addresses, called the VTT, is declared for 
  //   each class type that has indirect or direct virtual base classes.
  if (RD->getNumVBases() == 0)
    return;

  bool IsPrimaryVTT = Base.getBase() == MostDerivedClass;

  if (!IsPrimaryVTT) {
    // Remember the sub-VTT index.
    SubVTTIndicies[Base] = VTTComponents.size();
  }

  AddressPointsMapTy AddressPoints;
  llvm::Constant *VTable = GetAddrOfVTable(Base, BaseIsVirtual, AddressPoints);

  // Add the primary vtable pointer.
  AddVTablePointer(Base, VTable, RD, AddressPoints);

  // Add the secondary VTTs.
  LayoutSecondaryVTTs(Base);
  
  // Add the secondary virtual pointers.
  LayoutSecondaryVirtualPointers(Base, VTable, AddressPoints);
  
  // If this is the primary VTT, we want to lay out virtual VTTs as well.
  if (IsPrimaryVTT) {
    VisitedVirtualBasesSetTy VBases;
    LayoutVirtualVTTs(Base.getBase(), VBases);
  }
}
  
}

void
CodeGenVTables::EmitVTTDefinition(llvm::GlobalVariable *VTT,
                                  llvm::GlobalVariable::LinkageTypes Linkage,
                                  const CXXRecordDecl *RD) {
  VTTBuilder Builder(CGM, RD, /*GenerateDefinition=*/true, Linkage);

  llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(Int8PtrTy, Builder.getVTTComponents().size());
  
  llvm::Constant *Init = 
    llvm::ConstantArray::get(ArrayType, Builder.getVTTComponents());

  VTT->setInitializer(Init);

  // Set the correct linkage.
  VTT->setLinkage(Linkage);

  // Set the right visibility.
  CGM.setTypeVisibility(VTT, RD, CodeGenModule::TVK_ForVTT);
}

llvm::GlobalVariable *CodeGenVTables::GetAddrOfVTT(const CXXRecordDecl *RD) {
  assert(RD->getNumVBases() && "Only classes with virtual bases need a VTT");

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXVTT(RD, Out);
  Out.flush();
  llvm::StringRef Name = OutName.str();

  ComputeVTableRelatedInformation(RD, /*VTableRequired=*/true);

  VTTBuilder Builder(CGM, RD, /*GenerateDefinition=*/false);

  llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(Int8PtrTy, Builder.getVTTComponents().size());

  llvm::GlobalVariable *GV =
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, ArrayType, 
                                          llvm::GlobalValue::ExternalLinkage);
  GV->setUnnamedAddr(true);
  return GV;
}

bool CodeGenVTables::needsVTTParameter(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  
  // We don't have any virtual bases, just return early.
  if (!MD->getParent()->getNumVBases())
    return false;
  
  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return true;
  
  return false;
}

uint64_t CodeGenVTables::getSubVTTIndex(const CXXRecordDecl *RD, 
                                        BaseSubobject Base) {
  BaseSubobjectPairTy ClassSubobjectPair(RD, Base);

  SubVTTIndiciesMapTy::iterator I = SubVTTIndicies.find(ClassSubobjectPair);
  if (I != SubVTTIndicies.end())
    return I->second;
  
  VTTBuilder Builder(CGM, RD, /*GenerateDefinition=*/false);

  for (llvm::DenseMap<BaseSubobject, uint64_t>::const_iterator I =
       Builder.getSubVTTIndicies().begin(), 
       E = Builder.getSubVTTIndicies().end(); I != E; ++I) {
    // Insert all indices.
    BaseSubobjectPairTy ClassSubobjectPair(RD, I->first);
    
    SubVTTIndicies.insert(std::make_pair(ClassSubobjectPair, I->second));
  }
    
  I = SubVTTIndicies.find(ClassSubobjectPair);
  assert(I != SubVTTIndicies.end() && "Did not find index!");
  
  return I->second;
}

uint64_t 
CodeGenVTables::getSecondaryVirtualPointerIndex(const CXXRecordDecl *RD,
                                                BaseSubobject Base) {
  SecondaryVirtualPointerIndicesMapTy::iterator I =
    SecondaryVirtualPointerIndices.find(std::make_pair(RD, Base));

  if (I != SecondaryVirtualPointerIndices.end())
    return I->second;

  VTTBuilder Builder(CGM, RD, /*GenerateDefinition=*/false);

  // Insert all secondary vpointer indices.
  for (llvm::DenseMap<BaseSubobject, uint64_t>::const_iterator I = 
       Builder.getSecondaryVirtualPointerIndices().begin(),
       E = Builder.getSecondaryVirtualPointerIndices().end(); I != E; ++I) {
    std::pair<const CXXRecordDecl *, BaseSubobject> Pair =
      std::make_pair(RD, I->first);
    
    SecondaryVirtualPointerIndices.insert(std::make_pair(Pair, I->second));
  }

  I = SecondaryVirtualPointerIndices.find(std::make_pair(RD, Base));
  assert(I != SecondaryVirtualPointerIndices.end() && "Did not find index!");
  
  return I->second;
}

