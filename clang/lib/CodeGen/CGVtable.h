//===--- CGVtable.h - Emit LLVM Code for C++ vtables ----------------------===//
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
#include "GlobalDecl.h"

namespace llvm {
  class Constant;
}

namespace clang {
  class CXXRecordDecl;

namespace CodeGen {
  class CodeGenModule;

/// ThunkAdjustment - Virtual and non-virtual adjustment for thunks.
class ThunkAdjustment {
public:
  ThunkAdjustment(int64_t NonVirtual, int64_t Virtual)
  : NonVirtual(NonVirtual),
    Virtual(Virtual) { }

  ThunkAdjustment()
    : NonVirtual(0), Virtual(0) { }

  // isEmpty - Return whether this thunk adjustment is empty.
  bool isEmpty() const {
    return NonVirtual == 0 && Virtual == 0;
  }

  /// NonVirtual - The non-virtual adjustment.
  int64_t NonVirtual;

  /// Virtual - The virtual adjustment.
  int64_t Virtual;
};

/// CovariantThunkAdjustment - Adjustment of the 'this' pointer and the
/// return pointer for covariant thunks.
class CovariantThunkAdjustment {
public:
  CovariantThunkAdjustment(const ThunkAdjustment &ThisAdjustment,
                           const ThunkAdjustment &ReturnAdjustment)
  : ThisAdjustment(ThisAdjustment), ReturnAdjustment(ReturnAdjustment) { }

  CovariantThunkAdjustment() { }

  ThunkAdjustment ThisAdjustment;
  ThunkAdjustment ReturnAdjustment;
};

class CGVtableInfo {
  CodeGenModule &CGM;

  /// MethodVtableIndices - Contains the index (relative to the vtable address
  /// point) where the function pointer for a virtual function is stored.
  typedef llvm::DenseMap<GlobalDecl, int64_t> MethodVtableIndicesTy;
  MethodVtableIndicesTy MethodVtableIndices;

  typedef std::pair<const CXXRecordDecl *,
                    const CXXRecordDecl *> ClassPairTy;

  /// VirtualBaseClassIndicies - Contains the index into the vtable where the
  /// offsets for virtual bases of a class are stored.
  typedef llvm::DenseMap<ClassPairTy, int64_t> VirtualBaseClassIndiciesTy;
  VirtualBaseClassIndiciesTy VirtualBaseClassIndicies;

  llvm::DenseMap<const CXXRecordDecl *, llvm::Constant *> Vtables;
  
  /// NumVirtualFunctionPointers - Contains the number of virtual function 
  /// pointers in the vtable for a given record decl.
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> NumVirtualFunctionPointers;

  /// getNumVirtualFunctionPointers - Return the number of virtual function
  /// pointers in the vtable for a given record decl.
  uint64_t getNumVirtualFunctionPointers(const CXXRecordDecl *RD);
  
  void ComputeMethodVtableIndices(const CXXRecordDecl *RD);
  
  /// GenerateClassData - Generate all the class data requires to be generated
  /// upon definition of a KeyFunction.  This includes the vtable, the
  /// rtti data structure and the VTT.
  void GenerateClassData(const CXXRecordDecl *RD);
  
public:
  CGVtableInfo(CodeGenModule &CGM)
    : CGM(CGM) { }

  /// getMethodVtableIndex - Return the index (relative to the vtable address
  /// point) where the function pointer for the given virtual function is
  /// stored.
  uint64_t getMethodVtableIndex(GlobalDecl GD);

  /// getVirtualBaseOffsetIndex - Return the index (relative to the vtable
  /// address point) where the offset of the virtual base that contains the
  /// given Base is stored, otherwise, if no virtual base contains the given
  /// class, return 0.  Base must be a virtual base class or an unambigious
  /// base.
  int64_t getVirtualBaseOffsetIndex(const CXXRecordDecl *RD,
                                    const CXXRecordDecl *VBase);

  llvm::Constant *getVtable(const CXXRecordDecl *RD);
  llvm::Constant *getCtorVtable(const CXXRecordDecl *RD,
                                const CXXRecordDecl *Class, uint64_t Offset);
  
  
  void MaybeEmitVtable(GlobalDecl GD);
};

}
}
#endif
