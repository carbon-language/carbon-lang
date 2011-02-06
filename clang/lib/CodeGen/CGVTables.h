//===--- CGVTables.h - Emit LLVM Code for C++ vtables ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGVTABLE_H
#define CLANG_CODEGEN_CGVTABLE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/GlobalVariable.h"
#include "clang/Basic/ABI.h"
#include "GlobalDecl.h"

namespace clang {
  class CXXRecordDecl;

namespace CodeGen {
  class CodeGenModule;

// BaseSubobject - Uniquely identifies a direct or indirect base class. 
// Stores both the base class decl and the offset from the most derived class to
// the base class.
class BaseSubobject {
  /// Base - The base class declaration.
  const CXXRecordDecl *Base;
  
  /// BaseOffset - The offset from the most derived class to the base class.
  uint64_t BaseOffset;
  
public:
  BaseSubobject(const CXXRecordDecl *Base, uint64_t BaseOffset)
    : Base(Base), BaseOffset(BaseOffset) { }
  
  /// getBase - Returns the base class declaration.
  const CXXRecordDecl *getBase() const { return Base; }

  /// getBaseOffset - Returns the base class offset.
  uint64_t getBaseOffset() const { return BaseOffset; }

  friend bool operator==(const BaseSubobject &LHS, const BaseSubobject &RHS) {
    return LHS.Base == RHS.Base && LHS.BaseOffset == RHS.BaseOffset;
 }
};

} // end namespace CodeGen
} // end namespace clang

namespace llvm {

template<> struct DenseMapInfo<clang::CodeGen::BaseSubobject> {
  static clang::CodeGen::BaseSubobject getEmptyKey() {
    return clang::CodeGen::BaseSubobject(
      DenseMapInfo<const clang::CXXRecordDecl *>::getEmptyKey(),
      DenseMapInfo<uint64_t>::getEmptyKey());
  }

  static clang::CodeGen::BaseSubobject getTombstoneKey() {
    return clang::CodeGen::BaseSubobject(
      DenseMapInfo<const clang::CXXRecordDecl *>::getTombstoneKey(),
      DenseMapInfo<uint64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const clang::CodeGen::BaseSubobject &Base) {
    return 
      DenseMapInfo<const clang::CXXRecordDecl *>::getHashValue(Base.getBase()) ^
      DenseMapInfo<uint64_t>::getHashValue(Base.getBaseOffset());
  }

  static bool isEqual(const clang::CodeGen::BaseSubobject &LHS, 
                      const clang::CodeGen::BaseSubobject &RHS) {
    return LHS == RHS;
  }
};

// It's OK to treat BaseSubobject as a POD type.
template <> struct isPodLike<clang::CodeGen::BaseSubobject> {
  static const bool value = true;
};

}

namespace clang {
namespace CodeGen {

class CodeGenVTables {
  CodeGenModule &CGM;

  /// MethodVTableIndices - Contains the index (relative to the vtable address
  /// point) where the function pointer for a virtual function is stored.
  typedef llvm::DenseMap<GlobalDecl, int64_t> MethodVTableIndicesTy;
  MethodVTableIndicesTy MethodVTableIndices;

  typedef std::pair<const CXXRecordDecl *,
                    const CXXRecordDecl *> ClassPairTy;

  /// VirtualBaseClassOffsetOffsets - Contains the vtable offset (relative to 
  /// the address point) in bytes where the offsets for virtual bases of a class
  /// are stored.
  typedef llvm::DenseMap<ClassPairTy, int64_t> 
    VirtualBaseClassOffsetOffsetsMapTy;
  VirtualBaseClassOffsetOffsetsMapTy VirtualBaseClassOffsetOffsets;

  /// VTables - All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, llvm::GlobalVariable *> VTables;
  
  /// NumVirtualFunctionPointers - Contains the number of virtual function 
  /// pointers in the vtable for a given record decl.
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> NumVirtualFunctionPointers;

  typedef llvm::SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;
  
  /// Thunks - Contains all thunks that a given method decl will need.
  ThunksMapTy Thunks;

  // The layout entry and a bool indicating whether we've actually emitted
  // the vtable.
  typedef llvm::PointerIntPair<uint64_t *, 1, bool> VTableLayoutData;
  typedef llvm::DenseMap<const CXXRecordDecl *, VTableLayoutData>
    VTableLayoutMapTy;
  
  /// VTableLayoutMap - Stores the vtable layout for all record decls.
  /// The layout is stored as an array of 64-bit integers, where the first
  /// integer is the number of vtable entries in the layout, and the subsequent
  /// integers are the vtable components.
  VTableLayoutMapTy VTableLayoutMap;

  typedef std::pair<const CXXRecordDecl *, BaseSubobject> BaseSubobjectPairTy;
  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t> AddressPointsMapTy;
  
  /// Address points - Address points for all vtables.
  AddressPointsMapTy AddressPoints;

  /// VTableAddressPointsMapTy - Address points for a single vtable.
  typedef llvm::DenseMap<BaseSubobject, uint64_t> VTableAddressPointsMapTy;

  typedef llvm::SmallVector<std::pair<uint64_t, ThunkInfo>, 1> 
    VTableThunksTy;
  
  typedef llvm::DenseMap<const CXXRecordDecl *, VTableThunksTy>
    VTableThunksMapTy;
  
  /// VTableThunksMap - Contains thunks needed by vtables.
  VTableThunksMapTy VTableThunksMap;
  
  uint64_t getNumVTableComponents(const CXXRecordDecl *RD) const {
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");
    
    return VTableLayoutMap.lookup(RD).getPointer()[0];
  }

  const uint64_t *getVTableComponentsData(const CXXRecordDecl *RD) const {
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");

    uint64_t *Components = VTableLayoutMap.lookup(RD).getPointer();
    return &Components[1];
  }

  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t> SubVTTIndiciesMapTy;
  
  /// SubVTTIndicies - Contains indices into the various sub-VTTs.
  SubVTTIndiciesMapTy SubVTTIndicies;

  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t>
    SecondaryVirtualPointerIndicesMapTy;

  /// SecondaryVirtualPointerIndices - Contains the secondary virtual pointer
  /// indices.
  SecondaryVirtualPointerIndicesMapTy SecondaryVirtualPointerIndices;

  /// getNumVirtualFunctionPointers - Return the number of virtual function
  /// pointers in the vtable for a given record decl.
  uint64_t getNumVirtualFunctionPointers(const CXXRecordDecl *RD);
  
  void ComputeMethodVTableIndices(const CXXRecordDecl *RD);

  /// EmitThunk - Emit a single thunk.
  void EmitThunk(GlobalDecl GD, const ThunkInfo &Thunk, 
                 bool UseAvailableExternallyLinkage);

  /// MaybeEmitThunkAvailableExternally - Try to emit the given thunk with
  /// available_externally linkage to allow for inlining of thunks.
  /// This will be done iff optimizations are enabled and the member function
  /// doesn't contain any incomplete types.
  void MaybeEmitThunkAvailableExternally(GlobalDecl GD, const ThunkInfo &Thunk);

  /// ComputeVTableRelatedInformation - Compute and store all vtable related
  /// information (vtable layout, vbase offset offsets, thunks etc) for the
  /// given record decl.
  void ComputeVTableRelatedInformation(const CXXRecordDecl *RD,
                                       bool VTableRequired);

  /// CreateVTableInitializer - Create a vtable initializer for the given record
  /// decl.
  /// \param Components - The vtable components; this is really an array of
  /// VTableComponents.
  llvm::Constant *CreateVTableInitializer(const CXXRecordDecl *RD,
                                          const uint64_t *Components, 
                                          unsigned NumComponents,
                                          const VTableThunksTy &VTableThunks);

public:
  CodeGenVTables(CodeGenModule &CGM)
    : CGM(CGM) { }

  /// \brief True if the VTable of this record must be emitted in the
  /// translation unit.
  bool ShouldEmitVTableInThisTU(const CXXRecordDecl *RD);

  /// needsVTTParameter - Return whether the given global decl needs a VTT
  /// parameter, which it does if it's a base constructor or destructor with
  /// virtual bases.
  static bool needsVTTParameter(GlobalDecl GD);

  /// getSubVTTIndex - Return the index of the sub-VTT for the base class of the
  /// given record decl.
  uint64_t getSubVTTIndex(const CXXRecordDecl *RD, BaseSubobject Base);
  
  /// getSecondaryVirtualPointerIndex - Return the index in the VTT where the
  /// virtual pointer for the given subobject is located.
  uint64_t getSecondaryVirtualPointerIndex(const CXXRecordDecl *RD,
                                           BaseSubobject Base);

  /// getMethodVTableIndex - Return the index (relative to the vtable address
  /// point) where the function pointer for the given virtual function is
  /// stored.
  uint64_t getMethodVTableIndex(GlobalDecl GD);

  /// getVirtualBaseOffsetOffset - Return the offset in bytes (relative to the
  /// vtable address point) where the offset of the virtual base that contains 
  /// the given base is stored, otherwise, if no virtual base contains the given
  /// class, return 0.  Base must be a virtual base class or an unambigious
  /// base.
  int64_t getVirtualBaseOffsetOffset(const CXXRecordDecl *RD,
                                     const CXXRecordDecl *VBase);

  /// getAddressPoint - Get the address point of the given subobject in the
  /// class decl.
  uint64_t getAddressPoint(BaseSubobject Base, const CXXRecordDecl *RD);
  
  /// GetAddrOfVTable - Get the address of the vtable for the given record decl.
  llvm::GlobalVariable *GetAddrOfVTable(const CXXRecordDecl *RD);

  /// EmitVTableDefinition - Emit the definition of the given vtable.
  void EmitVTableDefinition(llvm::GlobalVariable *VTable,
                            llvm::GlobalVariable::LinkageTypes Linkage,
                            const CXXRecordDecl *RD);
  
  /// GenerateConstructionVTable - Generate a construction vtable for the given 
  /// base subobject.
  llvm::GlobalVariable *
  GenerateConstructionVTable(const CXXRecordDecl *RD, const BaseSubobject &Base, 
                             bool BaseIsVirtual, 
                             VTableAddressPointsMapTy& AddressPoints);

    
  /// GetAddrOfVTable - Get the address of the VTT for the given record decl.
  llvm::GlobalVariable *GetAddrOfVTT(const CXXRecordDecl *RD);

  /// EmitVTTDefinition - Emit the definition of the given vtable.
  void EmitVTTDefinition(llvm::GlobalVariable *VTT,
                         llvm::GlobalVariable::LinkageTypes Linkage,
                         const CXXRecordDecl *RD);

  /// EmitThunks - Emit the associated thunks for the given global decl.
  void EmitThunks(GlobalDecl GD);
    
  /// GenerateClassData - Generate all the class data required to be generated
  /// upon definition of a KeyFunction.  This includes the vtable, the
  /// rtti data structure and the VTT.
  ///
  /// \param Linkage - The desired linkage of the vtable, the RTTI and the VTT.
  void GenerateClassData(llvm::GlobalVariable::LinkageTypes Linkage,
                         const CXXRecordDecl *RD);
};

} // end namespace CodeGen
} // end namespace clang
#endif
