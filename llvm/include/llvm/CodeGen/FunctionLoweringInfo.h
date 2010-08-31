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

#ifndef LLVM_CODEGEN_FUNCTIONLOWERINGINFO_H
#define LLVM_CODEGEN_FUNCTIONLOWERINGINFO_H

#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#ifndef NDEBUG
#include "llvm/ADT/SmallSet.h"
#endif
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/CallSite.h"
#include <vector>

namespace llvm {

class AllocaInst;
class BasicBlock;
class CallInst;
class Function;
class GlobalVariable;
class Instruction;
class MachineInstr;
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
  const TargetLowering &TLI;
  const Function *Fn;
  MachineFunction *MF;
  MachineRegisterInfo *RegInfo;

  /// CanLowerReturn - true iff the function's return value can be lowered to
  /// registers.
  bool CanLowerReturn;

  /// DemoteRegister - if CanLowerReturn is false, DemoteRegister is a vreg
  /// allocated to hold a pointer to the hidden sret parameter.
  unsigned DemoteRegister;

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

  /// ByValArgFrameIndexMap - Keep track of frame indices for byval arguments.
  DenseMap<const Argument*, int> ByValArgFrameIndexMap;

  /// ArgDbgValues - A list of DBG_VALUE instructions created during isel for
  /// function arguments that are inserted after scheduling is completed.
  SmallVector<MachineInstr*, 8> ArgDbgValues;

  /// RegFixups - Registers which need to be replaced after isel is done.
  DenseMap<unsigned, unsigned> RegFixups;

  /// MBB - The current block.
  MachineBasicBlock *MBB;

  /// MBB - The current insert position inside the current block.
  MachineBasicBlock::iterator InsertPt;

#ifndef NDEBUG
  SmallSet<const Instruction *, 8> CatchInfoLost;
  SmallSet<const Instruction *, 8> CatchInfoFound;
#endif

  struct LiveOutInfo {
    unsigned NumSignBits;
    APInt KnownOne, KnownZero;
    LiveOutInfo() : NumSignBits(0), KnownOne(1, 0), KnownZero(1, 0) {}
  };
  
  /// LiveOutRegInfo - Information about live out vregs, indexed by their
  /// register number offset by 'FirstVirtualRegister'.
  std::vector<LiveOutInfo> LiveOutRegInfo;

  /// PHINodesToUpdate - A list of phi instructions whose operand list will
  /// be updated after processing the current basic block.
  /// TODO: This isn't per-function state, it's per-basic-block state. But
  /// there's no other convenient place for it to live right now.
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;

  explicit FunctionLoweringInfo(const TargetLowering &TLI);

  /// set - Initialize this FunctionLoweringInfo with the given Function
  /// and its associated MachineFunction.
  ///
  void set(const Function &Fn, MachineFunction &MF);

  /// clear - Clear out all the function-specific state. This returns this
  /// FunctionLoweringInfo to an empty state, ready to be used for a
  /// different function.
  void clear();

  /// isExportedInst - Return true if the specified value is an instruction
  /// exported from its block.
  bool isExportedInst(const Value *V) {
    return ValueMap.count(V);
  }

  unsigned CreateReg(EVT VT);
  
  unsigned CreateRegs(const Type *Ty);
  
  unsigned InitializeRegForValue(const Value *V) {
    unsigned &R = ValueMap[V];
    assert(R == 0 && "Already initialized this value register!");
    return R = CreateRegs(V->getType());
  }

  /// setByValArgumentFrameIndex - Record frame index for the byval
  /// argument.
  void setByValArgumentFrameIndex(const Argument *A, int FI);
  
  /// getByValArgumentFrameIndex - Get frame index for the byval argument.
  int getByValArgumentFrameIndex(const Argument *A);
};

/// AddCatchInfo - Extract the personality and type infos from an eh.selector
/// call, and add them to the specified machine basic block.
void AddCatchInfo(const CallInst &I,
                  MachineModuleInfo *MMI, MachineBasicBlock *MBB);

/// CopyCatchInfo - Copy catch information from DestBB to SrcBB.
void CopyCatchInfo(const BasicBlock *SrcBB, const BasicBlock *DestBB,
                   MachineModuleInfo *MMI, FunctionLoweringInfo &FLI);

} // end namespace llvm

#endif
