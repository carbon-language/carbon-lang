//===--- CGVTables.h - Emit LLVM Code for C++ vtables -----------*- C++ -*-===//
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
#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/GlobalDecl.h"

namespace clang {
  class CXXRecordDecl;

namespace CodeGen {
  class CodeGenModule;
  class CodeGenVTables;

class VTableContext {
  ASTContext &Context;

  /// MethodVTableIndices - Contains the index (relative to the vtable address
  /// point) where the function pointer for a virtual function is stored.
  typedef llvm::DenseMap<GlobalDecl, int64_t> MethodVTableIndicesTy;
  MethodVTableIndicesTy MethodVTableIndices;

  /// NumVirtualFunctionPointers - Contains the number of virtual function 
  /// pointers in the vtable for a given record decl.
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> NumVirtualFunctionPointers;

  typedef std::pair<const CXXRecordDecl *,
                    const CXXRecordDecl *> ClassPairTy;

  /// VirtualBaseClassOffsetOffsets - Contains the vtable offset (relative to 
  /// the address point) in chars where the offsets for virtual bases of a class
  /// are stored.
  typedef llvm::DenseMap<ClassPairTy, CharUnits> 
    VirtualBaseClassOffsetOffsetsMapTy;
  VirtualBaseClassOffsetOffsetsMapTy VirtualBaseClassOffsetOffsets;

  void ComputeMethodVTableIndices(const CXXRecordDecl *RD);

public:
  VTableContext(ASTContext &Context) : Context(Context) {}

  /// getNumVirtualFunctionPointers - Return the number of virtual function
  /// pointers in the vtable for a given record decl.
  uint64_t getNumVirtualFunctionPointers(const CXXRecordDecl *RD);
  
  /// getMethodVTableIndex - Return the index (relative to the vtable address
  /// point) where the function pointer for the given virtual function is
  /// stored.
  uint64_t getMethodVTableIndex(GlobalDecl GD);

  /// getVirtualBaseOffsetOffset - Return the offset in chars (relative to the
  /// vtable address point) where the offset of the virtual base that contains 
  /// the given base is stored, otherwise, if no virtual base contains the given
  /// class, return 0.  Base must be a virtual base class or an unambigious
  /// base.
  CharUnits getVirtualBaseOffsetOffset(const CXXRecordDecl *RD,
                                       const CXXRecordDecl *VBase);

  friend class CodeGenVTables;
};

class CodeGenVTables {
  CodeGenModule &CGM;

  VTableContext VTContext;

  /// VTables - All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, llvm::GlobalVariable *> VTables;
  
  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;
  
  /// Thunks - Contains all thunks that a given method decl will need.
  ThunksMapTy Thunks;

  // The layout entry.
  typedef llvm::DenseMap<const CXXRecordDecl *, uint64_t *> VTableLayoutMapTy;
  
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

  typedef SmallVector<std::pair<uint64_t, ThunkInfo>, 1> 
    VTableThunksTy;
  
  typedef llvm::DenseMap<const CXXRecordDecl *, VTableThunksTy>
    VTableThunksMapTy;
  
  /// VTableThunksMap - Contains thunks needed by vtables.
  VTableThunksMapTy VTableThunksMap;
  
  uint64_t getNumVTableComponents(const CXXRecordDecl *RD) const {
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");
    
    return VTableLayoutMap.lookup(RD)[0];
  }

  const uint64_t *getVTableComponentsData(const CXXRecordDecl *RD) const {
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");

    uint64_t *Components = VTableLayoutMap.lookup(RD);
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
  void ComputeVTableRelatedInformation(const CXXRecordDecl *RD);

  /// CreateVTableInitializer - Create a vtable initializer for the given record
  /// decl.
  /// \param Components - The vtable components; this is really an array of
  /// VTableComponents.
  llvm::Constant *CreateVTableInitializer(const CXXRecordDecl *RD,
                                          const uint64_t *Components, 
                                          unsigned NumComponents,
                                          const VTableThunksTy &VTableThunks);

public:
  CodeGenVTables(CodeGenModule &CGM);

  VTableContext &getVTableContext() { return VTContext; }

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
                             llvm::GlobalVariable::LinkageTypes Linkage,
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
