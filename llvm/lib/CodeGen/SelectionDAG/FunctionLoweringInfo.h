//===-- FunctionLoweringInfo.h - Lower functions from LLVM IR to CodeGen --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating functions from LLVM IR into
// Machine IR.
//
//===----------------------------------------------------------------------===//

#ifndef FUNCTIONLOWERINGINFO_H
#define FUNCTIONLOWERINGINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#ifndef NDEBUG
#include "llvm/ADT/SmallSet.h"
#endif
#include "llvm/CodeGen/ValueTypes.h"
#include <vector>

namespace llvm {

class AllocaInst;
class BasicBlock;
class CallInst;
class Function;
class GlobalVariable;
class Instruction;
class MachineBasicBlock;
class MachineFunction;
class MachineModuleInfo;
class MachineRegisterInfo;
class TargetLowering;
class Value;

//===--------------------------------------------------------------------===//
/// FunctionLoweringInfo - This contains information that is global to a
/// function that is used when lowering a region of the function.
///
class FunctionLoweringInfo {
public:
  TargetLowering &TLI;
  Function *Fn;
  MachineFunction *MF;
  MachineRegisterInfo *RegInfo;

  /// CanLowerReturn - true iff the function's return value can be lowered to
  /// registers.
  bool CanLowerReturn;

  /// DemoteRegister - if CanLowerReturn is false, DemoteRegister is a vreg
  /// allocated to hold a pointer to the hidden sret parameter.
  unsigned DemoteRegister;

  explicit FunctionLoweringInfo(TargetLowering &TLI);

  /// set - Initialize this FunctionLoweringInfo with the given Function
  /// and its associated MachineFunction.
  ///
  void set(Function &Fn, MachineFunction &MF, bool EnableFastISel);

  /// MBBMap - A mapping from LLVM basic blocks to their machine code entry.
  DenseMap<const BasicBlock*, MachineBasicBlock *> MBBMap;

  /// ValueMap - Since we emit code for the function a basic block at a time,
  /// we must remember which virtual registers hold the values for
  /// cross-basic-block values.
  DenseMap<const Value*, unsigned> ValueMap;

  /// StaticAllocaMap - Keep track of frame indices for fixed sized allocas in
  /// the entry block.  This allows the allocas to be efficiently referenced
  /// anywhere in the function.
  DenseMap<const AllocaInst*, int> StaticAllocaMap;

#ifndef NDEBUG
  SmallSet<Instruction*, 8> CatchInfoLost;
  SmallSet<Instruction*, 8> CatchInfoFound;
#endif

  unsigned MakeReg(EVT VT);
  
  /// isExportedInst - Return true if the specified value is an instruction
  /// exported from its block.
  bool isExportedInst(const Value *V) {
    return ValueMap.count(V);
  }

  unsigned CreateRegForValue(const Value *V);
  
  unsigned InitializeRegForValue(const Value *V) {
    unsigned &R = ValueMap[V];
    assert(R == 0 && "Already initialized this value register!");
    return R = CreateRegForValue(V);
  }
  
  struct LiveOutInfo {
    unsigned NumSignBits;
    APInt KnownOne, KnownZero;
    LiveOutInfo() : NumSignBits(0), KnownOne(1, 0), KnownZero(1, 0) {}
  };
  
  /// LiveOutRegInfo - Information about live out vregs, indexed by their
  /// register number offset by 'FirstVirtualRegister'.
  std::vector<LiveOutInfo> LiveOutRegInfo;

  /// clear - Clear out all the function-specific state. This returns this
  /// FunctionLoweringInfo to an empty state, ready to be used for a
  /// different function.
  void clear();
};

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// of insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
unsigned ComputeLinearIndex(const TargetLowering &TLI, const Type *Ty,
                            const unsigned *Indices,
                            const unsigned *IndicesEnd,
                            unsigned CurIndex = 0);

/// ComputeValueVTs - Given an LLVM IR type, compute a sequence of
/// EVTs that represent all the individual underlying
/// non-aggregate types that comprise it.
///
/// If Offsets is non-null, it points to a vector to be filled in
/// with the in-memory offsets of each of the individual values.
///
void ComputeValueVTs(const TargetLowering &TLI, const Type *Ty,
                     SmallVectorImpl<EVT> &ValueVTs,
                     SmallVectorImpl<uint64_t> *Offsets = 0,
                     uint64_t StartingOffset = 0);

/// ExtractTypeInfo - Returns the type info, possibly bitcast, encoded in V.
GlobalVariable *ExtractTypeInfo(Value *V);

/// AddCatchInfo - Extract the personality and type infos from an eh.selector
/// call, and add them to the specified machine basic block.
void AddCatchInfo(CallInst &I, MachineModuleInfo *MMI, MachineBasicBlock *MBB);

/// CopyCatchInfo - Copy catch information from DestBB to SrcBB.
void CopyCatchInfo(BasicBlock *SrcBB, BasicBlock *DestBB,
                   MachineModuleInfo *MMI, FunctionLoweringInfo &FLI);

} // end namespace llvm

#endif
