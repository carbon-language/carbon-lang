//===- MachineSSAContext.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares a specialization of the GenericSSAContext<X>
/// template class for Machine IR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESSACONTEXT_H
#define LLVM_CODEGEN_MACHINESSACONTEXT_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/Printable.h"

namespace llvm {
class MachineRegisterInfo;
class MachineInstr;
class MachineFunction;
class Register;
template <typename _FunctionT> class GenericSSAContext;
template <typename, bool> class DominatorTreeBase;

inline auto successors(MachineBasicBlock *BB) { return BB->successors(); }
inline auto predecessors(MachineBasicBlock *BB) { return BB->predecessors(); }
inline unsigned succ_size(MachineBasicBlock *BB) { return BB->succ_size(); }
inline unsigned pred_size(MachineBasicBlock *BB) { return BB->pred_size(); }

template <> class GenericSSAContext<MachineFunction> {
  const MachineRegisterInfo *RegInfo = nullptr;
  MachineFunction *MF;

public:
  using BlockT = MachineBasicBlock;
  using FunctionT = MachineFunction;
  using InstructionT = MachineInstr;
  using ValueRefT = Register;
  using DominatorTreeT = DominatorTreeBase<BlockT, false>;

  static MachineBasicBlock *getEntryBlock(MachineFunction &F);

  void setFunction(MachineFunction &Fn);
  MachineFunction *getFunction() const { return MF; }

  Printable print(MachineBasicBlock *Block) const;
  Printable print(MachineInstr *Inst) const;
  Printable print(Register Value) const;
};

using MachineSSAContext = GenericSSAContext<MachineFunction>;
} // namespace llvm

#endif // LLVM_CODEGEN_MACHINESSACONTEXT_H
