//===- lib/CodeGen/MachineStableHash.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable hashing for MachineInstr and MachineOperand. Useful or getting a
// hash across runs, modules, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/MIRFormatter.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StableHashing.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "machine-stable-hash"

using namespace llvm;

STATISTIC(StableHashBailingMachineBasicBlock,
          "Number of encountered unsupported MachineOperands that were "
          "MachineBasicBlocks while computing stable hashes");
STATISTIC(StableHashBailingConstantPoolIndex,
          "Number of encountered unsupported MachineOperands that were "
          "ConstantPoolIndex while computing stable hashes");
STATISTIC(StableHashBailingTargetIndexNoName,
          "Number of encountered unsupported MachineOperands that were "
          "TargetIndex with no name");
STATISTIC(StableHashBailingGlobalAddress,
          "Number of encountered unsupported MachineOperands that were "
          "GlobalAddress while computing stable hashes");
STATISTIC(StableHashBailingBlockAddress,
          "Number of encountered unsupported MachineOperands that were "
          "BlockAddress while computing stable hashes");
STATISTIC(StableHashBailingMetadataUnsupported,
          "Number of encountered unsupported MachineOperands that were "
          "Metadata of an unsupported kind while computing stable hashes");

stable_hash llvm::stableHashValue(const MachineOperand &MO) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (Register::isVirtualRegister(MO.getReg())) {
      const MachineRegisterInfo &MRI = MO.getParent()->getMF()->getRegInfo();
      return MRI.getVRegDef(MO.getReg())->getOpcode();
    }

    // Register operands don't have target flags.
    return stable_hash_combine(MO.getType(), MO.getReg(), MO.getSubReg(),
                               MO.isDef());
  case MachineOperand::MO_Immediate:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(), MO.getImm());
  case MachineOperand::MO_CImmediate:
  case MachineOperand::MO_FPImmediate: {
    auto Val = MO.isCImm() ? MO.getCImm()->getValue()
                           : MO.getFPImm()->getValueAPF().bitcastToAPInt();
    auto ValHash =
        stable_hash_combine_array(Val.getRawData(), Val.getNumWords());
    return hash_combine(MO.getType(), MO.getTargetFlags(), ValHash);
  }

  case MachineOperand::MO_MachineBasicBlock:
    StableHashBailingMachineBasicBlock++;
    return 0;
  case MachineOperand::MO_ConstantPoolIndex:
    StableHashBailingConstantPoolIndex++;
    return 0;
  case MachineOperand::MO_BlockAddress:
    StableHashBailingBlockAddress++;
    return 0;
  case MachineOperand::MO_Metadata:
    StableHashBailingMetadataUnsupported++;
    return 0;
  case MachineOperand::MO_GlobalAddress:
    StableHashBailingGlobalAddress++;
    return 0;
  case MachineOperand::MO_TargetIndex: {
    if (const char *Name = MO.getTargetIndexName())
      return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                                 stable_hash_combine_string(Name),
                                 MO.getOffset());
    StableHashBailingTargetIndexNoName++;
    return 0;
  }

  case MachineOperand::MO_FrameIndex:
  case MachineOperand::MO_JumpTableIndex:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getIndex());

  case MachineOperand::MO_ExternalSymbol:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getOffset(),
                        stable_hash_combine_string(MO.getSymbolName()));

  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut:
    return hash_combine(MO.getType(), MO.getTargetFlags(), MO.getRegMask());

  case MachineOperand::MO_ShuffleMask: {
    std::vector<llvm::stable_hash> ShuffleMaskHashes;

    llvm::transform(
        MO.getShuffleMask(), std::back_inserter(ShuffleMaskHashes),
        [](int S) -> llvm::stable_hash { return llvm::stable_hash(S); });

    return hash_combine(MO.getType(), MO.getTargetFlags(),
                        stable_hash_combine_array(ShuffleMaskHashes.data(),
                                                  ShuffleMaskHashes.size()));
  }
  case MachineOperand::MO_MCSymbol: {
    auto SymbolName = MO.getMCSymbol()->getName();
    return hash_combine(MO.getType(), MO.getTargetFlags(),
                        stable_hash_combine_string(SymbolName));
  }
  case MachineOperand::MO_CFIIndex:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getCFIIndex());
  case MachineOperand::MO_IntrinsicID:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getIntrinsicID());
  case MachineOperand::MO_Predicate:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getPredicate());
  }
  llvm_unreachable("Invalid machine operand type");
}

/// A stable hash value for machine instructions.
/// Returns 0 if no stable hash could be computed.
/// The hashing and equality testing functions ignore definitions so this is
/// useful for CSE, etc.
stable_hash llvm::stableHashValue(const MachineInstr &MI, bool HashVRegs,
                                  bool HashConstantPoolIndices,
                                  bool HashMemOperands) {
  // Build up a buffer of hash code components.
  SmallVector<stable_hash, 16> HashComponents;
  HashComponents.reserve(MI.getNumOperands() + MI.getNumMemOperands() + 2);
  HashComponents.push_back(MI.getOpcode());
  HashComponents.push_back(MI.getFlags());
  for (const MachineOperand &MO : MI.operands()) {
    if (!HashVRegs && MO.isReg() && MO.isDef() &&
        Register::isVirtualRegister(MO.getReg()))
      continue; // Skip virtual register defs.

    if (MO.isCPI()) {
      HashComponents.push_back(stable_hash_combine(
          MO.getType(), MO.getTargetFlags(), MO.getIndex()));
      continue;
    }

    stable_hash StableHash = stableHashValue(MO);
    if (!StableHash)
      return 0;
    HashComponents.push_back(StableHash);
  }

  for (const auto *Op : MI.memoperands()) {
    if (!HashMemOperands)
      break;
    HashComponents.push_back(static_cast<unsigned>(Op->getSize()));
    HashComponents.push_back(static_cast<unsigned>(Op->getFlags()));
    HashComponents.push_back(static_cast<unsigned>(Op->getOffset()));
    HashComponents.push_back(static_cast<unsigned>(Op->getSuccessOrdering()));
    HashComponents.push_back(static_cast<unsigned>(Op->getAddrSpace()));
    HashComponents.push_back(static_cast<unsigned>(Op->getSyncScopeID()));
    HashComponents.push_back(static_cast<unsigned>(Op->getBaseAlign().value()));
    HashComponents.push_back(static_cast<unsigned>(Op->getFailureOrdering()));
  }

  return stable_hash_combine_range(HashComponents.begin(),
                                   HashComponents.end());
}
