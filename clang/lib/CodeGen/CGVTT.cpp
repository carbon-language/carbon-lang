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
    if (!CtorVtable) {
      // Get the vtable.
      CtorVtable = 
        CGM.getVTables().GenerateConstructionVTable(MostDerivedClass, 
                                                    Base, BaseIsVirtual, 
                                                    CtorVtableAddressPoints);
    }
    
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
    if (RD->getNumVBases() == 0 && ! MorallyVirtual)
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

  /// LayoutSecondaryVTTs - Lay out the secondary VTTs of the given base 
  /// subobject.
  void LayoutSecondaryVTTs(BaseSubobject Base);
  
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
  }
  
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> &getSubVTTIndicies() {
    return SubVTTIndicies;
  }
};
  
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

/// LayoutVirtualVTTs - Lay out the VTTs for the virtual base classes of the
/// given record decl.
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

  llvm::Constant *Vtable;
  const CXXRecordDecl *VtableClass;

  // First comes the primary virtual table pointer...
  Vtable = getCtorVtable(Base, /*IsVirtual=*/BaseIsVirtual);
  VtableClass = RD;
  
  llvm::Constant *Init = BuildVtablePtr(Vtable, VtableClass, RD, 
                                        Base.getBaseOffset());
  Inits.push_back(Init);

  // then the secondary VTTs....
  LayoutSecondaryVTTs(Base);

  // Make sure to clear the set of seen virtual bases.
  SeenVBasesInSecondary.clear();

  // and last the secondary vtable pointers.
  Secondary(RD, Vtable, VtableClass, Base.getBaseOffset(), false);
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
