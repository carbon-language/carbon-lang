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

/// VTableComponent - Represents a single component in a vtable.
class VTableComponent {
public:
  enum Kind {
    CK_VCallOffset,
    CK_VBaseOffset,
    CK_OffsetToTop,
    CK_RTTI,
    CK_FunctionPointer,
    
    /// CK_CompleteDtorPointer - A pointer to the complete destructor.
    CK_CompleteDtorPointer,
    
    /// CK_DeletingDtorPointer - A pointer to the deleting destructor.
    CK_DeletingDtorPointer,
    
    /// CK_UnusedFunctionPointer - In some cases, a vtable function pointer
    /// will end up never being called. Such vtable function pointers are
    /// represented as a CK_UnusedFunctionPointer. 
    CK_UnusedFunctionPointer
  };

  static VTableComponent MakeVCallOffset(CharUnits Offset) {
    return VTableComponent(CK_VCallOffset, Offset);
  }

  static VTableComponent MakeVBaseOffset(CharUnits Offset) {
    return VTableComponent(CK_VBaseOffset, Offset);
  }

  static VTableComponent MakeOffsetToTop(CharUnits Offset) {
    return VTableComponent(CK_OffsetToTop, Offset);
  }
  
  static VTableComponent MakeRTTI(const CXXRecordDecl *RD) {
    return VTableComponent(CK_RTTI, reinterpret_cast<uintptr_t>(RD));
  }

  static VTableComponent MakeFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeFunction with destructors!");

    return VTableComponent(CK_FunctionPointer, 
                           reinterpret_cast<uintptr_t>(MD));
  }
  
  static VTableComponent MakeCompleteDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_CompleteDtorPointer,
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeDeletingDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_DeletingDtorPointer, 
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeUnusedFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeUnusedFunction with destructors!");
    return VTableComponent(CK_UnusedFunctionPointer,
                           reinterpret_cast<uintptr_t>(MD));                           
  }

  static VTableComponent getFromOpaqueInteger(uint64_t I) {
    return VTableComponent(I);
  }

  /// getKind - Get the kind of this vtable component.
  Kind getKind() const {
    return (Kind)(Value & 0x7);
  }

  CharUnits getVCallOffset() const {
    assert(getKind() == CK_VCallOffset && "Invalid component kind!");
    
    return getOffset();
  }

  CharUnits getVBaseOffset() const {
    assert(getKind() == CK_VBaseOffset && "Invalid component kind!");
    
    return getOffset();
  }

  CharUnits getOffsetToTop() const {
    assert(getKind() == CK_OffsetToTop && "Invalid component kind!");
    
    return getOffset();
  }
  
  const CXXRecordDecl *getRTTIDecl() const {
    assert(getKind() == CK_RTTI && "Invalid component kind!");
    
    return reinterpret_cast<CXXRecordDecl *>(getPointer());
  }
  
  const CXXMethodDecl *getFunctionDecl() const {
    assert(getKind() == CK_FunctionPointer);
    
    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }

  const CXXDestructorDecl *getDestructorDecl() const {
    assert((getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer) && "Invalid component kind!");
    
    return reinterpret_cast<CXXDestructorDecl *>(getPointer());
  }

  const CXXMethodDecl *getUnusedFunctionDecl() const {
    assert(getKind() == CK_UnusedFunctionPointer);
    
    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }
  
private:
  VTableComponent(Kind ComponentKind, CharUnits Offset) {
    assert((ComponentKind == CK_VCallOffset || 
            ComponentKind == CK_VBaseOffset ||
            ComponentKind == CK_OffsetToTop) && "Invalid component kind!");
    assert(Offset.getQuantity() <= ((1LL << 56) - 1) && "Offset is too big!");
    
    Value = ((Offset.getQuantity() << 3) | ComponentKind);
  }

  VTableComponent(Kind ComponentKind, uintptr_t Ptr) {
    assert((ComponentKind == CK_RTTI || 
            ComponentKind == CK_FunctionPointer ||
            ComponentKind == CK_CompleteDtorPointer ||
            ComponentKind == CK_DeletingDtorPointer ||
            ComponentKind == CK_UnusedFunctionPointer) &&
            "Invalid component kind!");
    
    assert((Ptr & 7) == 0 && "Pointer not sufficiently aligned!");
    
    Value = Ptr | ComponentKind;
  }
  
  CharUnits getOffset() const {
    assert((getKind() == CK_VCallOffset || getKind() == CK_VBaseOffset ||
            getKind() == CK_OffsetToTop) && "Invalid component kind!");
    
    return CharUnits::fromQuantity(Value >> 3);
  }

  uintptr_t getPointer() const {
    assert((getKind() == CK_RTTI || 
            getKind() == CK_FunctionPointer ||
            getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer ||
            getKind() == CK_UnusedFunctionPointer) &&
           "Invalid component kind!");
    
    return static_cast<uintptr_t>(Value & ~7ULL);
  }
  
  explicit VTableComponent(uint64_t Value)
    : Value(Value) { }

  /// The kind is stored in the lower 3 bits of the value. For offsets, we
  /// make use of the facts that classes can't be larger than 2^55 bytes,
  /// so we store the offset in the lower part of the 61 bytes that remain.
  /// (The reason that we're not simply using a PointerIntPair here is that we
  /// need the offsets to be 64-bit, even when on a 32-bit machine).
  int64_t Value;
};

class VTableContext {
  ASTContext &Context;

public:
  typedef SmallVector<std::pair<uint64_t, ThunkInfo>, 1> 
    VTableThunksTy;
  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  
private:
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

  // The layout entry.
  typedef llvm::DenseMap<const CXXRecordDecl *, uint64_t *> VTableLayoutMapTy;
  
  /// VTableLayoutMap - Stores the vtable layout for all record decls.
  /// The layout is stored as an array of 64-bit integers, where the first
  /// integer is the number of vtable entries in the layout, and the subsequent
  /// integers are the vtable components.
  VTableLayoutMapTy VTableLayoutMap;

  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;
  
  /// Thunks - Contains all thunks that a given method decl will need.
  ThunksMapTy Thunks;

  typedef llvm::DenseMap<const CXXRecordDecl *, VTableThunksTy>
    VTableThunksMapTy;
  
  /// VTableThunksMap - Contains thunks needed by vtables.
  VTableThunksMapTy VTableThunksMap;
  
  typedef std::pair<const CXXRecordDecl *, BaseSubobject> BaseSubobjectPairTy;
  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t> AddressPointsMapTy;
  
  /// Address points - Address points for all vtables.
  AddressPointsMapTy AddressPoints;

  void ComputeMethodVTableIndices(const CXXRecordDecl *RD);

  /// ComputeVTableRelatedInformation - Compute and store all vtable related
  /// information (vtable layout, vbase offset offsets, thunks etc) for the
  /// given record decl.
  void ComputeVTableRelatedInformation(const CXXRecordDecl *RD);

public:
  VTableContext(ASTContext &Context) : Context(Context) {}

  uint64_t getNumVTableComponents(const CXXRecordDecl *RD) {
    ComputeVTableRelatedInformation(RD);
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");
    
    return VTableLayoutMap.lookup(RD)[0];
  }

  const uint64_t *getVTableComponentsData(const CXXRecordDecl *RD) {
    ComputeVTableRelatedInformation(RD);
    assert(VTableLayoutMap.count(RD) && "No vtable layout for this class!");

    uint64_t *Components = VTableLayoutMap.lookup(RD);
    return &Components[1];
  }

  const ThunkInfoVectorTy *getThunkInfo(const CXXMethodDecl *MD) {
    ComputeVTableRelatedInformation(MD->getParent());

    ThunksMapTy::const_iterator I = Thunks.find(MD);
    if (I == Thunks.end()) {
      // We did not find a thunk for this method.
      return 0;
    }

    return &I->second;
  }

  const VTableThunksTy &getVTableThunks(const CXXRecordDecl *RD) {
    ComputeVTableRelatedInformation(RD);
    assert(VTableThunksMap.count(RD) && 
           "No thunk status for this record decl!");
    
    return VTableThunksMap[RD];
  }

  uint64_t getAddressPoint(BaseSubobject Base, const CXXRecordDecl *RD) {
    ComputeVTableRelatedInformation(RD);
    assert(AddressPoints.count(std::make_pair(RD, Base)) &&
           "Did not find address point!");

    uint64_t AddressPoint = AddressPoints.lookup(std::make_pair(RD, Base));
    assert(AddressPoint && "Address point must not be zero!");

    return AddressPoint;
  }

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
};

class CodeGenVTables {
  CodeGenModule &CGM;

  VTableContext VTContext;

  /// VTables - All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, llvm::GlobalVariable *> VTables;
  
  /// VTableAddressPointsMapTy - Address points for a single vtable.
  typedef llvm::DenseMap<BaseSubobject, uint64_t> VTableAddressPointsMapTy;

  typedef std::pair<const CXXRecordDecl *, BaseSubobject> BaseSubobjectPairTy;
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

  /// CreateVTableInitializer - Create a vtable initializer for the given record
  /// decl.
  /// \param Components - The vtable components; this is really an array of
  /// VTableComponents.
  llvm::Constant *CreateVTableInitializer(const CXXRecordDecl *RD,
                                          const uint64_t *Components, 
                                          unsigned NumComponents,
                             const VTableContext::VTableThunksTy &VTableThunks);

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
