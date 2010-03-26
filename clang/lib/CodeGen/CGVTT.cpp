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
#include "clang/AST/RecordLayout.h"
using namespace clang;
using namespace CodeGen;

#define D1(x)

namespace {

/// VTT builder - Class for building VTT layout information.
class VTTBuilder {
  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  /// Inits - The list of values built for the VTT.
  std::vector<llvm::Constant *> &Inits;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  CodeGenModule &CGM;  // Per-module state.

  CodeGenVTables::AddrMap_t &AddressPoints;
  // vtbl - A pointer to the vtable for Class.
  llvm::Constant *ClassVtbl;
  llvm::LLVMContext &VMContext;

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;

  typedef llvm::DenseMap<BaseSubobject, uint64_t> AddressPointsMapTy;

  /// SeenVBasesInSecondary - The seen virtual bases when building the 
  /// secondary virtual pointers.
  llvm::SmallPtrSet<const CXXRecordDecl *, 32> SeenVBasesInSecondary;

  llvm::DenseMap<const CXXRecordDecl *, uint64_t> SubVTTIndicies;
  
  bool GenerateDefinition;

  llvm::DenseMap<BaseSubobject, llvm::Constant *> CtorVtables;
  llvm::DenseMap<std::pair<const CXXRecordDecl *, BaseSubobject>, uint64_t> 
    CtorVtableAddressPoints;
  
  llvm::Constant *getCtorVtable(const BaseSubobject &Base,
                                bool BaseIsVirtual) {
    if (!GenerateDefinition)
      return 0;

    llvm::Constant *&CtorVtable = CtorVtables[Base];
    return CtorVtable;
  }
  
  
  /// BuildVtablePtr - Build up a referene to the given secondary vtable
  llvm::Constant *BuildVtablePtr(llvm::Constant *Vtable,
                                 const CXXRecordDecl *VtableClass,
                                 const CXXRecordDecl *RD,
                                 uint64_t Offset) {
    if (!GenerateDefinition)
      return 0;

    uint64_t AddressPoint;
    
    if (VtableClass != MostDerivedClass) {
      // We have a ctor vtable, look for the address point in the ctor vtable
      // address points.
      AddressPoint = 
        CtorVtableAddressPoints[std::make_pair(VtableClass, 
                                               BaseSubobject(RD, Offset))];
    } else { 
      AddressPoint = 
        (*AddressPoints[VtableClass])[std::make_pair(RD, Offset)];
    }

    // FIXME: We can never have 0 address point.  Do this for now so gepping
    // retains the same structure.  Later we'll just assert.
    if (AddressPoint == 0)
      AddressPoint = 1;
    D1(printf("XXX address point for %s in %s layout %s at offset %d was %d\n",
              RD->getNameAsCString(), VtblClass->getNameAsCString(),
              Class->getNameAsCString(), (int)Offset, (int)AddressPoint));

    llvm::Value *Idxs[] = {
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), 0),
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), AddressPoint)
    };
    
    llvm::Constant *Init = 
      llvm::ConstantExpr::getInBoundsGetElementPtr(Vtable, Idxs, 2);

    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);
    return llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
  }

  /// Secondary - Add the secondary vtable pointers to Inits.  Offset is the
  /// current offset in bits to the object we're working on.
  void Secondary(const CXXRecordDecl *RD, llvm::Constant *vtbl,
                 const CXXRecordDecl *VtblClass, uint64_t Offset,
                 bool MorallyVirtual) {
    if (RD->getNumVBases() == 0 && !MorallyVirtual)
      return;

    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

      // We only want to visit each virtual base once.
      if (i->isVirtual() && SeenVBasesInSecondary.count(Base))
        continue;
      
      // Itanium C++ ABI 2.6.2:
      //   Secondary virtual pointers are present for all bases with either
      //   virtual bases or virtual function declarations overridden along a 
      //   virtual path.
      //
      // If the base class is not dynamic, we don't want to add it, nor any
      // of its base classes.
      if (!Base->isDynamicClass())
        continue;

      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
      const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();
      bool NonVirtualPrimaryBase;
      NonVirtualPrimaryBase = !PrimaryBaseWasVirtual && Base == PrimaryBase;
      bool BaseMorallyVirtual = MorallyVirtual | i->isVirtual();
      uint64_t BaseOffset;
      if (!i->isVirtual()) {
        const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      } else
        BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(Base);
      llvm::Constant *subvtbl = vtbl;
      const CXXRecordDecl *subVtblClass = VtblClass;
      if ((Base->getNumVBases() || BaseMorallyVirtual)
          && !NonVirtualPrimaryBase) {
        llvm::Constant *init;
        if (BaseMorallyVirtual || VtblClass == MostDerivedClass)
          init = BuildVtablePtr(vtbl, VtblClass, Base, BaseOffset);
        else {
          init = getCtorVtable(BaseSubobject(Base, BaseOffset), i->isVirtual());
          
          subvtbl = init;
          subVtblClass = Base;
          
          init = BuildVtablePtr(init, MostDerivedClass, Base, BaseOffset);
        }

        Inits.push_back(init);
      }
      
      if (i->isVirtual())
        SeenVBasesInSecondary.insert(Base);
      
      Secondary(Base, subvtbl, subVtblClass, BaseOffset, BaseMorallyVirtual);
    }
  }

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
  VTTBuilder(std::vector<llvm::Constant *> &inits, 
             const CXXRecordDecl *MostDerivedClass,
             CodeGenModule &cgm, bool GenerateDefinition) 
    : MostDerivedClass(MostDerivedClass), 
     Inits(inits), 
  MostDerivedClassLayout(cgm.getContext().getASTRecordLayout(MostDerivedClass)),
      CGM(cgm),
      AddressPoints(*cgm.getVTables().OldAddressPoints[MostDerivedClass]),
      VMContext(cgm.getModule().getContext()),
      GenerateDefinition(GenerateDefinition) {
    
        LayoutVTT(BaseSubobject(MostDerivedClass, 0), /*BaseIsVirtual=*/false);
#if 0
    // First comes the primary virtual table pointer for the complete class...
    ClassVtbl = GenerateDefinition ? 
          CGM.getVTables().GetAddrOfVTable(MostDerivedClass) :0;

    llvm::Constant *Init = BuildVtablePtr(ClassVtbl, MostDerivedClass, 
                                          MostDerivedClass, 0);
    Inits.push_back(Init);
    
    // then the secondary VTTs...
    LayoutSecondaryVTTs(BaseSubobject(MostDerivedClass, 0));

    // Make sure to clear the set of seen virtual bases.
    SeenVBasesInSecondary.clear();

    // then the secondary vtable pointers...
    Secondary(MostDerivedClass, ClassVtbl, MostDerivedClass, 0, false);

    // and last, the virtual VTTs.
    VisitedVirtualBasesSetTy VBases;
    LayoutVirtualVTTs(MostDerivedClass, VBases);
#endif
  }
  
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> &getSubVTTIndicies() {
    return SubVTTIndicies;
  }
};

llvm::Constant *
VTTBuilder::GetAddrOfVTable(BaseSubobject Base, bool BaseIsVirtual, 
                            AddressPointsMapTy& AddressPoints) {
  if (!GenerateDefinition)
    return 0;
  
  if (Base.getBase() == MostDerivedClass) {
    assert(Base.getBaseOffset() == 0 &&
           "Most derived class vtable must have a zero offset!");
    // This is a regular vtable.
    return CGM.getVTables().GetAddrOfVTable(MostDerivedClass);
  }
  
  return CGM.getVTables().GenerateConstructionVTable(MostDerivedClass, 
                                                     Base, BaseIsVirtual,
                                                     AddressPoints);
}

void VTTBuilder::AddVTablePointer(BaseSubobject Base, llvm::Constant *VTable,
                                  const CXXRecordDecl *VTableClass,
                                  const AddressPointsMapTy& AddressPoints) {
  if (!GenerateDefinition) {
    Inits.push_back(0);
    return;
  }

  uint64_t AddressPoint;
  if (VTableClass != MostDerivedClass) {
    // The vtable is a construction vtable, look in the construction vtable
    // address points.
    AddressPoint = AddressPoints.lookup(Base);
  } else {
    AddressPoint = 
      (*this->AddressPoints[VTableClass])[std::make_pair(Base.getBase(), 
                                                         Base.getBaseOffset())];
  }

  if (!AddressPoint) AddressPoint = 0;
  assert(AddressPoint != 0 && "Did not find an address point!");
  
  llvm::Value *Idxs[] = {
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()), 0),
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()), 
                           AddressPoint)
  };
  
  llvm::Constant *Init = 
    llvm::ConstantExpr::getInBoundsGetElementPtr(VTable, Idxs, 2);
  
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
  
  Inits.push_back(Init);
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
    uint64_t BaseOffset = Base.getBaseOffset() + 
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
    uint64_t BaseOffset;
    if (I->isVirtual()) {
      // Ignore virtual bases that we've already visited.
      if (!VBases.insert(BaseDecl))
        continue;
      
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      BaseDeclIsMorallyVirtual = true;
    } else {
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      
      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      
      if (!Layout.getPrimaryBaseWasVirtual() &&
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
      AddVTablePointer(BaseSubobject(BaseDecl, BaseOffset), VTable, VTableClass, 
                       AddressPoints);
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
    
      uint64_t BaseOffset = 
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
  
  // Remember the sub-VTT index.
  SubVTTIndicies[RD] = Inits.size();

  AddressPointsMapTy AddressPoints;
  llvm::Constant *VTable = GetAddrOfVTable(Base, BaseIsVirtual, AddressPoints);

  // Add the primary vtable pointer.
  AddVTablePointer(Base, VTable, RD, AddressPoints);

  // Add the secondary VTTs.
  LayoutSecondaryVTTs(Base);
  
  // Add the secondary virtual pointers.
  LayoutSecondaryVirtualPointers(Base, VTable, AddressPoints);
  
  // If this is the primary VTT, we want to lay out virtual VTTs as well.
  if (Base.getBase() == MostDerivedClass) {
    VisitedVirtualBasesSetTy VBases;
    LayoutVirtualVTTs(Base.getBase(), VBases);
  }
}
  
}

llvm::GlobalVariable *
CodeGenVTables::GenerateVTT(llvm::GlobalVariable::LinkageTypes Linkage,
                            bool GenerateDefinition,
                            const CXXRecordDecl *RD) {
  // Only classes that have virtual bases need a VTT.
  if (RD->getNumVBases() == 0)
    return 0;

  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXVTT(RD, OutName);
  llvm::StringRef Name = OutName.str();

  D1(printf("vtt %s\n", RD->getNameAsCString()));

  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name);
  if (GV == 0 || GV->isDeclaration()) {
    const llvm::Type *Int8PtrTy = 
      llvm::Type::getInt8PtrTy(CGM.getLLVMContext());

    std::vector<llvm::Constant *> inits;
    VTTBuilder b(inits, RD, CGM, GenerateDefinition);

    const llvm::ArrayType *Type = llvm::ArrayType::get(Int8PtrTy, inits.size());
    llvm::Constant *Init = 0;
    if (GenerateDefinition)
      Init = llvm::ConstantArray::get(Type, inits);

    llvm::GlobalVariable *OldGV = GV;
    GV = new llvm::GlobalVariable(CGM.getModule(), Type, /*isConstant=*/true, 
                                  Linkage, Init, Name);
    CGM.setGlobalVisibility(GV, RD);
    
    if (OldGV) {
      GV->takeName(OldGV);
      llvm::Constant *NewPtr = 
        llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
      OldGV->replaceAllUsesWith(NewPtr);
      OldGV->eraseFromParent();
    }
  }
  
  return GV;
}

llvm::GlobalVariable *CodeGenVTables::getVTT(const CXXRecordDecl *RD) {
  return GenerateVTT(llvm::GlobalValue::ExternalLinkage, 
                     /*GenerateDefinition=*/false, RD);
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
                                        const CXXRecordDecl *Base) {
  ClassPairTy ClassPair(RD, Base);

  SubVTTIndiciesTy::iterator I = 
    SubVTTIndicies.find(ClassPair);
  if (I != SubVTTIndicies.end())
    return I->second;
  
  std::vector<llvm::Constant *> inits;
  VTTBuilder Builder(inits, RD, CGM, /*GenerateDefinition=*/false);

  for (llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I =
       Builder.getSubVTTIndicies().begin(), 
       E = Builder.getSubVTTIndicies().end(); I != E; ++I) {
    // Insert all indices.
    ClassPairTy ClassPair(RD, I->first);
    
    SubVTTIndicies.insert(std::make_pair(ClassPair, I->second));
  }
    
  I = SubVTTIndicies.find(ClassPair);
  assert(I != SubVTTIndicies.end() && "Did not find index!");
  
  return I->second;
}
