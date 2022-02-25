//===- SILoadStoreOptimizer.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to fuse DS instructions with close by immediate offsets.
// This will fuse operations such as
//  ds_read_b32 v0, v2 offset:16
//  ds_read_b32 v1, v2 offset:32
// ==>
//   ds_read2_b32 v[0:1], v2, offset0:4 offset1:8
//
// The same is done for certain SMEM and VMEM opcodes, e.g.:
//  s_buffer_load_dword s4, s[0:3], 4
//  s_buffer_load_dword s5, s[0:3], 8
// ==>
//  s_buffer_load_dwordx2 s[4:5], s[0:3], 4
//
// This pass also tries to promote constant offset to the immediate by
// adjusting the base. It tries to use a base from the nearby instructions that
// allows it to have a 13bit constant offset and then promotes the 13bit offset
// to the immediate.
// E.g.
//  s_movk_i32 s0, 0x1800
//  v_add_co_u32_e32 v0, vcc, s0, v2
//  v_addc_co_u32_e32 v1, vcc, 0, v6, vcc
//
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[0:1], off
// =>
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[5:6], off offset:2048
//
// Future improvements:
//
// - This is currently missing stores of constants because loading
//   the constant into the data register is placed between the stores, although
//   this is arguably a scheduling problem.
//
// - Live interval recomputing seems inefficient. This currently only matches
//   one pair, and recomputes live intervals and moves on to the next pair. It
//   would be better to compute a list of all merges that need to occur.
//
// - With a list of instructions to process, we can also merge more. If a
//   cluster of loads have offsets that are too large to fit in the 8-bit
//   offsets, but are close enough to fit in the 8 bits, we can add to the base
//   pointer and use the new reduced offsets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-load-store-opt"

namespace {
enum InstClassEnum {
  UNKNOWN,
  DS_READ,
  DS_WRITE,
  S_BUFFER_LOAD_IMM,
  BUFFER_LOAD,
  BUFFER_STORE,
  MIMG,
  TBUFFER_LOAD,
  TBUFFER_STORE,
};

struct AddressRegs {
  unsigned char NumVAddrs = 0;
  bool SBase = false;
  bool SRsrc = false;
  bool SOffset = false;
  bool VAddr = false;
  bool Addr = false;
  bool SSamp = false;
};

// GFX10 image_sample instructions can have 12 vaddrs + srsrc + ssamp.
const unsigned MaxAddressRegs = 12 + 1 + 1;

class SILoadStoreOptimizer : public MachineFunctionPass {
  struct CombineInfo {
    MachineBasicBlock::iterator I;
    unsigned EltSize;
    unsigned Offset;
    unsigned Width;
    unsigned Format;
    unsigned BaseOff;
    unsigned DMask;
    InstClassEnum InstClass;
    unsigned CPol = 0;
    bool UseST64;
    int AddrIdx[MaxAddressRegs];
    const MachineOperand *AddrReg[MaxAddressRegs];
    unsigned NumAddresses;
    unsigned Order;

    bool hasSameBaseAddress(const MachineInstr &MI) {
      for (unsigned i = 0; i < NumAddresses; i++) {
        const MachineOperand &AddrRegNext = MI.getOperand(AddrIdx[i]);

        if (AddrReg[i]->isImm() || AddrRegNext.isImm()) {
          if (AddrReg[i]->isImm() != AddrRegNext.isImm() ||
              AddrReg[i]->getImm() != AddrRegNext.getImm()) {
            return false;
          }
          continue;
        }

        // Check same base pointer. Be careful of subregisters, which can occur
        // with vectors of pointers.
        if (AddrReg[i]->getReg() != AddrRegNext.getReg() ||
            AddrReg[i]->getSubReg() != AddrRegNext.getSubReg()) {
         return false;
        }
      }
      return true;
    }

    bool hasMergeableAddress(const MachineRegisterInfo &MRI) {
      for (unsigned i = 0; i < NumAddresses; ++i) {
        const MachineOperand *AddrOp = AddrReg[i];
        // Immediates are always OK.
        if (AddrOp->isImm())
          continue;

        // Don't try to merge addresses that aren't either immediates or registers.
        // TODO: Should be possible to merge FrameIndexes and maybe some other
        // non-register
        if (!AddrOp->isReg())
          return false;

        // TODO: We should be able to merge physical reg addreses.
        if (AddrOp->getReg().isPhysical())
          return false;

        // If an address has only one use then there will be on other
        // instructions with the same address, so we can't merge this one.
        if (MRI.hasOneNonDBGUse(AddrOp->getReg()))
          return false;
      }
      return true;
    }

    void setMI(MachineBasicBlock::iterator MI, const SIInstrInfo &TII,
               const GCNSubtarget &STM);
  };

  struct BaseRegisters {
    Register LoReg;
    Register HiReg;

    unsigned LoSubReg = 0;
    unsigned HiSubReg = 0;
  };

  struct MemAddress {
    BaseRegisters Base;
    int64_t Offset = 0;
  };

  using MemInfoMap = DenseMap<MachineInstr *, MemAddress>;

private:
  const GCNSubtarget *STM = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AliasAnalysis *AA = nullptr;
  bool OptimizeAgain;

  static bool dmasksCanBeCombined(const CombineInfo &CI,
                                  const SIInstrInfo &TII,
                                  const CombineInfo &Paired);
  static bool offsetsCanBeCombined(CombineInfo &CI, const GCNSubtarget &STI,
                                   CombineInfo &Paired, bool Modify = false);
  static bool widthsFit(const GCNSubtarget &STI, const CombineInfo &CI,
                        const CombineInfo &Paired);
  static unsigned getNewOpcode(const CombineInfo &CI, const CombineInfo &Paired);
  static std::pair<unsigned, unsigned> getSubRegIdxs(const CombineInfo &CI,
                                                     const CombineInfo &Paired);
  const TargetRegisterClass *getTargetRegisterClass(const CombineInfo &CI,
                                                    const CombineInfo &Paired);
  const TargetRegisterClass *getDataRegClass(const MachineInstr &MI) const;

  bool checkAndPrepareMerge(CombineInfo &CI, CombineInfo &Paired,
                            SmallVectorImpl<MachineInstr *> &InstsToMove);

  unsigned read2Opcode(unsigned EltSize) const;
  unsigned read2ST64Opcode(unsigned EltSize) const;
  MachineBasicBlock::iterator mergeRead2Pair(CombineInfo &CI,
                                             CombineInfo &Paired,
                  const SmallVectorImpl<MachineInstr *> &InstsToMove);

  unsigned write2Opcode(unsigned EltSize) const;
  unsigned write2ST64Opcode(unsigned EltSize) const;
  MachineBasicBlock::iterator
  mergeWrite2Pair(CombineInfo &CI, CombineInfo &Paired,
                  const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeImagePair(CombineInfo &CI, CombineInfo &Paired,
                 const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeSBufferLoadImmPair(CombineInfo &CI, CombineInfo &Paired,
                          const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeBufferLoadPair(CombineInfo &CI, CombineInfo &Paired,
                      const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeBufferStorePair(CombineInfo &CI, CombineInfo &Paired,
                       const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeTBufferLoadPair(CombineInfo &CI, CombineInfo &Paired,
                       const SmallVectorImpl<MachineInstr *> &InstsToMove);
  MachineBasicBlock::iterator
  mergeTBufferStorePair(CombineInfo &CI, CombineInfo &Paired,
                        const SmallVectorImpl<MachineInstr *> &InstsToMove);

  void updateBaseAndOffset(MachineInstr &I, Register NewBase,
                           int32_t NewOffset) const;
  Register computeBase(MachineInstr &MI, const MemAddress &Addr) const;
  MachineOperand createRegOrImm(int32_t Val, MachineInstr &MI) const;
  Optional<int32_t> extractConstOffset(const MachineOperand &Op) const;
  void processBaseWithConstOffset(const MachineOperand &Base, MemAddress &Addr) const;
  /// Promotes constant offset to the immediate by adjusting the base. It
  /// tries to use a base from the nearby instructions that allows it to have
  /// a 13bit constant offset which gets promoted to the immediate.
  bool promoteConstantOffsetToImm(MachineInstr &CI,
                                  MemInfoMap &Visited,
                                  SmallPtrSet<MachineInstr *, 4> &Promoted) const;
  void addInstToMergeableList(const CombineInfo &CI,
                  std::list<std::list<CombineInfo> > &MergeableInsts) const;

  std::pair<MachineBasicBlock::iterator, bool> collectMergeableInsts(
      MachineBasicBlock::iterator Begin, MachineBasicBlock::iterator End,
      MemInfoMap &Visited, SmallPtrSet<MachineInstr *, 4> &AnchorList,
      std::list<std::list<CombineInfo>> &MergeableInsts) const;

public:
  static char ID;

  SILoadStoreOptimizer() : MachineFunctionPass(ID) {
    initializeSILoadStoreOptimizerPass(*PassRegistry::getPassRegistry());
  }

  bool optimizeInstsWithSameBaseAddr(std::list<CombineInfo> &MergeList,
                                     bool &OptimizeListAgain);
  bool optimizeBlock(std::list<std::list<CombineInfo> > &MergeableInsts);

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Load Store Optimizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties()
      .set(MachineFunctionProperties::Property::IsSSA);
  }
};

static unsigned getOpcodeWidth(const MachineInstr &MI, const SIInstrInfo &TII) {
  const unsigned Opc = MI.getOpcode();

  if (TII.isMUBUF(Opc)) {
    // FIXME: Handle d16 correctly
    return AMDGPU::getMUBUFElements(Opc);
  }
  if (TII.isMIMG(MI)) {
    uint64_t DMaskImm =
        TII.getNamedOperand(MI, AMDGPU::OpName::dmask)->getImm();
    return countPopulation(DMaskImm);
  }
  if (TII.isMTBUF(Opc)) {
    return AMDGPU::getMTBUFElements(Opc);
  }

  switch (Opc) {
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
    return 1;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
    return 2;
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return 4;
  case AMDGPU::DS_READ_B32:      LLVM_FALLTHROUGH;
  case AMDGPU::DS_READ_B32_gfx9: LLVM_FALLTHROUGH;
  case AMDGPU::DS_WRITE_B32:     LLVM_FALLTHROUGH;
  case AMDGPU::DS_WRITE_B32_gfx9:
    return 1;
  case AMDGPU::DS_READ_B64:      LLVM_FALLTHROUGH;
  case AMDGPU::DS_READ_B64_gfx9: LLVM_FALLTHROUGH;
  case AMDGPU::DS_WRITE_B64:     LLVM_FALLTHROUGH;
  case AMDGPU::DS_WRITE_B64_gfx9:
    return 2;
  default:
    return 0;
  }
}

/// Maps instruction opcode to enum InstClassEnum.
static InstClassEnum getInstClass(unsigned Opc, const SIInstrInfo &TII) {
  switch (Opc) {
  default:
    if (TII.isMUBUF(Opc)) {
      switch (AMDGPU::getMUBUFBaseOpcode(Opc)) {
      default:
        return UNKNOWN;
      case AMDGPU::BUFFER_LOAD_DWORD_OFFEN:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFEN_exact:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFSET:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFSET_exact:
        return BUFFER_LOAD;
      case AMDGPU::BUFFER_STORE_DWORD_OFFEN:
      case AMDGPU::BUFFER_STORE_DWORD_OFFEN_exact:
      case AMDGPU::BUFFER_STORE_DWORD_OFFSET:
      case AMDGPU::BUFFER_STORE_DWORD_OFFSET_exact:
        return BUFFER_STORE;
      }
    }
    if (TII.isMIMG(Opc)) {
      // Ignore instructions encoded without vaddr.
      if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr) == -1 &&
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr0) == -1)
        return UNKNOWN;
      // Ignore BVH instructions
      if (AMDGPU::getMIMGBaseOpcode(Opc)->BVH)
        return UNKNOWN;
      // TODO: Support IMAGE_GET_RESINFO and IMAGE_GET_LOD.
      if (TII.get(Opc).mayStore() || !TII.get(Opc).mayLoad() ||
          TII.isGather4(Opc))
        return UNKNOWN;
      return MIMG;
    }
    if (TII.isMTBUF(Opc)) {
      switch (AMDGPU::getMTBUFBaseOpcode(Opc)) {
      default:
        return UNKNOWN;
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFEN:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFEN_exact:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET_exact:
        return TBUFFER_LOAD;
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFEN:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFEN_exact:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFSET:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFSET_exact:
        return TBUFFER_STORE;
      }
    }
    return UNKNOWN;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return S_BUFFER_LOAD_IMM;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B64_gfx9:
    return DS_READ;
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B64_gfx9:
    return DS_WRITE;
  }
}

/// Determines instruction subclass from opcode. Only instructions
/// of the same subclass can be merged together.
static unsigned getInstSubclass(unsigned Opc, const SIInstrInfo &TII) {
  switch (Opc) {
  default:
    if (TII.isMUBUF(Opc))
      return AMDGPU::getMUBUFBaseOpcode(Opc);
    if (TII.isMIMG(Opc)) {
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opc);
      assert(Info);
      return Info->BaseOpcode;
    }
    if (TII.isMTBUF(Opc))
      return AMDGPU::getMTBUFBaseOpcode(Opc);
    return -1;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B64_gfx9:
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B64_gfx9:
    return Opc;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return AMDGPU::S_BUFFER_LOAD_DWORD_IMM;
  }
}

static AddressRegs getRegs(unsigned Opc, const SIInstrInfo &TII) {
  AddressRegs Result;

  if (TII.isMUBUF(Opc)) {
    if (AMDGPU::getMUBUFHasVAddr(Opc))
      Result.VAddr = true;
    if (AMDGPU::getMUBUFHasSrsrc(Opc))
      Result.SRsrc = true;
    if (AMDGPU::getMUBUFHasSoffset(Opc))
      Result.SOffset = true;

    return Result;
  }

  if (TII.isMIMG(Opc)) {
    int VAddr0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr0);
    if (VAddr0Idx >= 0) {
      int SRsrcIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::srsrc);
      Result.NumVAddrs = SRsrcIdx - VAddr0Idx;
    } else {
      Result.VAddr = true;
    }
    Result.SRsrc = true;
    const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opc);
    if (Info && AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode)->Sampler)
      Result.SSamp = true;

    return Result;
  }
  if (TII.isMTBUF(Opc)) {
    if (AMDGPU::getMTBUFHasVAddr(Opc))
      Result.VAddr = true;
    if (AMDGPU::getMTBUFHasSrsrc(Opc))
      Result.SRsrc = true;
    if (AMDGPU::getMTBUFHasSoffset(Opc))
      Result.SOffset = true;

    return Result;
  }

  switch (Opc) {
  default:
    return Result;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    Result.SBase = true;
    return Result;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64_gfx9:
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64_gfx9:
    Result.Addr = true;
    return Result;
  }
}

void SILoadStoreOptimizer::CombineInfo::setMI(MachineBasicBlock::iterator MI,
                                              const SIInstrInfo &TII,
                                              const GCNSubtarget &STM) {
  I = MI;
  unsigned Opc = MI->getOpcode();
  InstClass = getInstClass(Opc, TII);

  if (InstClass == UNKNOWN)
    return;

  switch (InstClass) {
  case DS_READ:
   EltSize =
          (Opc == AMDGPU::DS_READ_B64 || Opc == AMDGPU::DS_READ_B64_gfx9) ? 8
                                                                          : 4;
   break;
  case DS_WRITE:
    EltSize =
          (Opc == AMDGPU::DS_WRITE_B64 || Opc == AMDGPU::DS_WRITE_B64_gfx9) ? 8
                                                                            : 4;
    break;
  case S_BUFFER_LOAD_IMM:
    EltSize = AMDGPU::convertSMRDOffsetUnits(STM, 4);
    break;
  default:
    EltSize = 4;
    break;
  }

  if (InstClass == MIMG) {
    DMask = TII.getNamedOperand(*I, AMDGPU::OpName::dmask)->getImm();
    // Offset is not considered for MIMG instructions.
    Offset = 0;
  } else {
    int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
    Offset = I->getOperand(OffsetIdx).getImm();
  }

  if (InstClass == TBUFFER_LOAD || InstClass == TBUFFER_STORE)
    Format = TII.getNamedOperand(*I, AMDGPU::OpName::format)->getImm();

  Width = getOpcodeWidth(*I, TII);

  if ((InstClass == DS_READ) || (InstClass == DS_WRITE)) {
    Offset &= 0xffff;
  } else if (InstClass != MIMG) {
    CPol = TII.getNamedOperand(*I, AMDGPU::OpName::cpol)->getImm();
  }

  AddressRegs Regs = getRegs(Opc, TII);

  NumAddresses = 0;
  for (unsigned J = 0; J < Regs.NumVAddrs; J++)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr0) + J;
  if (Regs.Addr)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::addr);
  if (Regs.SBase)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::sbase);
  if (Regs.SRsrc)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::srsrc);
  if (Regs.SOffset)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::soffset);
  if (Regs.VAddr)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr);
  if (Regs.SSamp)
    AddrIdx[NumAddresses++] =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::ssamp);
  assert(NumAddresses <= MaxAddressRegs);

  for (unsigned J = 0; J < NumAddresses; J++)
    AddrReg[J] = &I->getOperand(AddrIdx[J]);
}

} // end anonymous namespace.

INITIALIZE_PASS_BEGIN(SILoadStoreOptimizer, DEBUG_TYPE,
                      "SI Load Store Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SILoadStoreOptimizer, DEBUG_TYPE, "SI Load Store Optimizer",
                    false, false)

char SILoadStoreOptimizer::ID = 0;

char &llvm::SILoadStoreOptimizerID = SILoadStoreOptimizer::ID;

FunctionPass *llvm::createSILoadStoreOptimizerPass() {
  return new SILoadStoreOptimizer();
}

static void moveInstsAfter(MachineBasicBlock::iterator I,
                           ArrayRef<MachineInstr *> InstsToMove) {
  MachineBasicBlock *MBB = I->getParent();
  ++I;
  for (MachineInstr *MI : InstsToMove) {
    MI->removeFromParent();
    MBB->insert(I, MI);
  }
}

static void addDefsUsesToList(const MachineInstr &MI,
                              DenseSet<Register> &RegDefs,
                              DenseSet<Register> &PhysRegUses) {
  for (const MachineOperand &Op : MI.operands()) {
    if (Op.isReg()) {
      if (Op.isDef())
        RegDefs.insert(Op.getReg());
      else if (Op.readsReg() && Op.getReg().isPhysical())
        PhysRegUses.insert(Op.getReg());
    }
  }
}

static bool memAccessesCanBeReordered(MachineBasicBlock::iterator A,
                                      MachineBasicBlock::iterator B,
                                      AliasAnalysis *AA) {
  // RAW or WAR - cannot reorder
  // WAW - cannot reorder
  // RAR - safe to reorder
  return !(A->mayStore() || B->mayStore()) || !A->mayAlias(AA, *B, true);
}

// Add MI and its defs to the lists if MI reads one of the defs that are
// already in the list. Returns true in that case.
static bool addToListsIfDependent(MachineInstr &MI, DenseSet<Register> &RegDefs,
                                  DenseSet<Register> &PhysRegUses,
                                  SmallVectorImpl<MachineInstr *> &Insts) {
  for (MachineOperand &Use : MI.operands()) {
    // If one of the defs is read, then there is a use of Def between I and the
    // instruction that I will potentially be merged with. We will need to move
    // this instruction after the merged instructions.
    //
    // Similarly, if there is a def which is read by an instruction that is to
    // be moved for merging, then we need to move the def-instruction as well.
    // This can only happen for physical registers such as M0; virtual
    // registers are in SSA form.
    if (Use.isReg() && ((Use.readsReg() && RegDefs.count(Use.getReg())) ||
                        (Use.isDef() && RegDefs.count(Use.getReg())) ||
                        (Use.isDef() && Use.getReg().isPhysical() &&
                         PhysRegUses.count(Use.getReg())))) {
      Insts.push_back(&MI);
      addDefsUsesToList(MI, RegDefs, PhysRegUses);
      return true;
    }
  }

  return false;
}

static bool canMoveInstsAcrossMemOp(MachineInstr &MemOp,
                                    ArrayRef<MachineInstr *> InstsToMove,
                                    AliasAnalysis *AA) {
  assert(MemOp.mayLoadOrStore());

  for (MachineInstr *InstToMove : InstsToMove) {
    if (!InstToMove->mayLoadOrStore())
      continue;
    if (!memAccessesCanBeReordered(MemOp, *InstToMove, AA))
      return false;
  }
  return true;
}

// This function assumes that \p A and \p B have are identical except for
// size and offset, and they referecne adjacent memory.
static MachineMemOperand *combineKnownAdjacentMMOs(MachineFunction &MF,
                                                   const MachineMemOperand *A,
                                                   const MachineMemOperand *B) {
  unsigned MinOffset = std::min(A->getOffset(), B->getOffset());
  unsigned Size = A->getSize() + B->getSize();
  // This function adds the offset parameter to the existing offset for A,
  // so we pass 0 here as the offset and then manually set it to the correct
  // value after the call.
  MachineMemOperand *MMO = MF.getMachineMemOperand(A, 0, Size);
  MMO->setOffset(MinOffset);
  return MMO;
}

bool SILoadStoreOptimizer::dmasksCanBeCombined(const CombineInfo &CI,
                                               const SIInstrInfo &TII,
                                               const CombineInfo &Paired) {
  assert(CI.InstClass == MIMG);

  // Ignore instructions with tfe/lwe set.
  const auto *TFEOp = TII.getNamedOperand(*CI.I, AMDGPU::OpName::tfe);
  const auto *LWEOp = TII.getNamedOperand(*CI.I, AMDGPU::OpName::lwe);

  if ((TFEOp && TFEOp->getImm()) || (LWEOp && LWEOp->getImm()))
    return false;

  // Check other optional immediate operands for equality.
  unsigned OperandsToMatch[] = {AMDGPU::OpName::cpol, AMDGPU::OpName::d16,
                                AMDGPU::OpName::unorm, AMDGPU::OpName::da,
                                AMDGPU::OpName::r128, AMDGPU::OpName::a16};

  for (auto op : OperandsToMatch) {
    int Idx = AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), op);
    if (AMDGPU::getNamedOperandIdx(Paired.I->getOpcode(), op) != Idx)
      return false;
    if (Idx != -1 &&
        CI.I->getOperand(Idx).getImm() != Paired.I->getOperand(Idx).getImm())
      return false;
  }

  // Check DMask for overlaps.
  unsigned MaxMask = std::max(CI.DMask, Paired.DMask);
  unsigned MinMask = std::min(CI.DMask, Paired.DMask);

  unsigned AllowedBitsForMin = llvm::countTrailingZeros(MaxMask);
  if ((1u << AllowedBitsForMin) <= MinMask)
    return false;

  return true;
}

static unsigned getBufferFormatWithCompCount(unsigned OldFormat,
                                       unsigned ComponentCount,
                                       const GCNSubtarget &STI) {
  if (ComponentCount > 4)
    return 0;

  const llvm::AMDGPU::GcnBufferFormatInfo *OldFormatInfo =
      llvm::AMDGPU::getGcnBufferFormatInfo(OldFormat, STI);
  if (!OldFormatInfo)
    return 0;

  const llvm::AMDGPU::GcnBufferFormatInfo *NewFormatInfo =
      llvm::AMDGPU::getGcnBufferFormatInfo(OldFormatInfo->BitsPerComp,
                                           ComponentCount,
                                           OldFormatInfo->NumFormat, STI);

  if (!NewFormatInfo)
    return 0;

  assert(NewFormatInfo->NumFormat == OldFormatInfo->NumFormat &&
         NewFormatInfo->BitsPerComp == OldFormatInfo->BitsPerComp);

  return NewFormatInfo->Format;
}

// Return the value in the inclusive range [Lo,Hi] that is aligned to the
// highest power of two. Note that the result is well defined for all inputs
// including corner cases like:
// - if Lo == Hi, return that value
// - if Lo == 0, return 0 (even though the "- 1" below underflows
// - if Lo > Hi, return 0 (as if the range wrapped around)
static uint32_t mostAlignedValueInRange(uint32_t Lo, uint32_t Hi) {
  return Hi & maskLeadingOnes<uint32_t>(countLeadingZeros((Lo - 1) ^ Hi) + 1);
}

bool SILoadStoreOptimizer::offsetsCanBeCombined(CombineInfo &CI,
                                                const GCNSubtarget &STI,
                                                CombineInfo &Paired,
                                                bool Modify) {
  assert(CI.InstClass != MIMG);

  // XXX - Would the same offset be OK? Is there any reason this would happen or
  // be useful?
  if (CI.Offset == Paired.Offset)
    return false;

  // This won't be valid if the offset isn't aligned.
  if ((CI.Offset % CI.EltSize != 0) || (Paired.Offset % CI.EltSize != 0))
    return false;

  if (CI.InstClass == TBUFFER_LOAD || CI.InstClass == TBUFFER_STORE) {

    const llvm::AMDGPU::GcnBufferFormatInfo *Info0 =
        llvm::AMDGPU::getGcnBufferFormatInfo(CI.Format, STI);
    if (!Info0)
      return false;
    const llvm::AMDGPU::GcnBufferFormatInfo *Info1 =
        llvm::AMDGPU::getGcnBufferFormatInfo(Paired.Format, STI);
    if (!Info1)
      return false;

    if (Info0->BitsPerComp != Info1->BitsPerComp ||
        Info0->NumFormat != Info1->NumFormat)
      return false;

    // TODO: Should be possible to support more formats, but if format loads
    // are not dword-aligned, the merged load might not be valid.
    if (Info0->BitsPerComp != 32)
      return false;

    if (getBufferFormatWithCompCount(CI.Format, CI.Width + Paired.Width, STI) == 0)
      return false;
  }

  uint32_t EltOffset0 = CI.Offset / CI.EltSize;
  uint32_t EltOffset1 = Paired.Offset / CI.EltSize;
  CI.UseST64 = false;
  CI.BaseOff = 0;

  // Handle all non-DS instructions.
  if ((CI.InstClass != DS_READ) && (CI.InstClass != DS_WRITE)) {
    return (EltOffset0 + CI.Width == EltOffset1 ||
            EltOffset1 + Paired.Width == EltOffset0) &&
           CI.CPol == Paired.CPol &&
           (CI.InstClass == S_BUFFER_LOAD_IMM || CI.CPol == Paired.CPol);
  }

  // If the offset in elements doesn't fit in 8-bits, we might be able to use
  // the stride 64 versions.
  if ((EltOffset0 % 64 == 0) && (EltOffset1 % 64) == 0 &&
      isUInt<8>(EltOffset0 / 64) && isUInt<8>(EltOffset1 / 64)) {
    if (Modify) {
      CI.Offset = EltOffset0 / 64;
      Paired.Offset = EltOffset1 / 64;
      CI.UseST64 = true;
    }
    return true;
  }

  // Check if the new offsets fit in the reduced 8-bit range.
  if (isUInt<8>(EltOffset0) && isUInt<8>(EltOffset1)) {
    if (Modify) {
      CI.Offset = EltOffset0;
      Paired.Offset = EltOffset1;
    }
    return true;
  }

  // Try to shift base address to decrease offsets.
  uint32_t Min = std::min(EltOffset0, EltOffset1);
  uint32_t Max = std::max(EltOffset0, EltOffset1);

  const uint32_t Mask = maskTrailingOnes<uint32_t>(8) * 64;
  if (((Max - Min) & ~Mask) == 0) {
    if (Modify) {
      // From the range of values we could use for BaseOff, choose the one that
      // is aligned to the highest power of two, to maximise the chance that
      // the same offset can be reused for other load/store pairs.
      uint32_t BaseOff = mostAlignedValueInRange(Max - 0xff * 64, Min);
      // Copy the low bits of the offsets, so that when we adjust them by
      // subtracting BaseOff they will be multiples of 64.
      BaseOff |= Min & maskTrailingOnes<uint32_t>(6);
      CI.BaseOff = BaseOff * CI.EltSize;
      CI.Offset = (EltOffset0 - BaseOff) / 64;
      Paired.Offset = (EltOffset1 - BaseOff) / 64;
      CI.UseST64 = true;
    }
    return true;
  }

  if (isUInt<8>(Max - Min)) {
    if (Modify) {
      // From the range of values we could use for BaseOff, choose the one that
      // is aligned to the highest power of two, to maximise the chance that
      // the same offset can be reused for other load/store pairs.
      uint32_t BaseOff = mostAlignedValueInRange(Max - 0xff, Min);
      CI.BaseOff = BaseOff * CI.EltSize;
      CI.Offset = EltOffset0 - BaseOff;
      Paired.Offset = EltOffset1 - BaseOff;
    }
    return true;
  }

  return false;
}

bool SILoadStoreOptimizer::widthsFit(const GCNSubtarget &STM,
                                     const CombineInfo &CI,
                                     const CombineInfo &Paired) {
  const unsigned Width = (CI.Width + Paired.Width);
  switch (CI.InstClass) {
  default:
    return (Width <= 4) && (STM.hasDwordx3LoadStores() || (Width != 3));
  case S_BUFFER_LOAD_IMM:
    switch (Width) {
    default:
      return false;
    case 2:
    case 4:
      return true;
    }
  }
}

const TargetRegisterClass *
SILoadStoreOptimizer::getDataRegClass(const MachineInstr &MI) const {
  if (const auto *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst)) {
    return TRI->getRegClassForReg(*MRI, Dst->getReg());
  }
  if (const auto *Src = TII->getNamedOperand(MI, AMDGPU::OpName::vdata)) {
    return TRI->getRegClassForReg(*MRI, Src->getReg());
  }
  if (const auto *Src = TII->getNamedOperand(MI, AMDGPU::OpName::data0)) {
    return TRI->getRegClassForReg(*MRI, Src->getReg());
  }
  if (const auto *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst)) {
    return TRI->getRegClassForReg(*MRI, Dst->getReg());
  }
  if (const auto *Src = TII->getNamedOperand(MI, AMDGPU::OpName::sdata)) {
    return TRI->getRegClassForReg(*MRI, Src->getReg());
  }
  return nullptr;
}

/// This function assumes that CI comes before Paired in a basic block.
bool SILoadStoreOptimizer::checkAndPrepareMerge(
    CombineInfo &CI, CombineInfo &Paired,
    SmallVectorImpl<MachineInstr *> &InstsToMove) {

  // Check both offsets (or masks for MIMG) can be combined and fit in the
  // reduced range.
  if (CI.InstClass == MIMG && !dmasksCanBeCombined(CI, *TII, Paired))
    return false;

  if (CI.InstClass != MIMG &&
      (!widthsFit(*STM, CI, Paired) || !offsetsCanBeCombined(CI, *STM, Paired)))
    return false;

  const unsigned Opc = CI.I->getOpcode();
  const InstClassEnum InstClass = getInstClass(Opc, *TII);

  if (InstClass == UNKNOWN) {
    return false;
  }
  const unsigned InstSubclass = getInstSubclass(Opc, *TII);

  // Do not merge VMEM buffer instructions with "swizzled" bit set.
  int Swizzled =
      AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), AMDGPU::OpName::swz);
  if (Swizzled != -1 && CI.I->getOperand(Swizzled).getImm())
    return false;

  DenseSet<Register> RegDefsToMove;
  DenseSet<Register> PhysRegUsesToMove;
  addDefsUsesToList(*CI.I, RegDefsToMove, PhysRegUsesToMove);

  const TargetRegisterClass *DataRC = getDataRegClass(*CI.I);
  bool IsAGPR = TRI->hasAGPRs(DataRC);

  MachineBasicBlock::iterator E = std::next(Paired.I);
  MachineBasicBlock::iterator MBBI = std::next(CI.I);
  MachineBasicBlock::iterator MBBE = CI.I->getParent()->end();
  for (; MBBI != E; ++MBBI) {

    if (MBBI == MBBE) {
      // CombineInfo::Order is a hint on the instruction ordering within the
      // basic block. This hint suggests that CI precedes Paired, which is
      // true most of the time. However, moveInstsAfter() processing a
      // previous list may have changed this order in a situation when it
      // moves an instruction which exists in some other merge list.
      // In this case it must be dependent.
      return false;
    }

    if ((getInstClass(MBBI->getOpcode(), *TII) != InstClass) ||
        (getInstSubclass(MBBI->getOpcode(), *TII) != InstSubclass)) {
      // This is not a matching instruction, but we can keep looking as
      // long as one of these conditions are met:
      // 1. It is safe to move I down past MBBI.
      // 2. It is safe to move MBBI down past the instruction that I will
      //    be merged into.

      if (MBBI->hasUnmodeledSideEffects()) {
        // We can't re-order this instruction with respect to other memory
        // operations, so we fail both conditions mentioned above.
        return false;
      }

      if (MBBI->mayLoadOrStore() &&
          (!memAccessesCanBeReordered(*CI.I, *MBBI, AA) ||
           !canMoveInstsAcrossMemOp(*MBBI, InstsToMove, AA))) {
        // We fail condition #1, but we may still be able to satisfy condition
        // #2.  Add this instruction to the move list and then we will check
        // if condition #2 holds once we have selected the matching instruction.
        InstsToMove.push_back(&*MBBI);
        addDefsUsesToList(*MBBI, RegDefsToMove, PhysRegUsesToMove);
        continue;
      }

      // When we match I with another DS instruction we will be moving I down
      // to the location of the matched instruction any uses of I will need to
      // be moved down as well.
      addToListsIfDependent(*MBBI, RegDefsToMove, PhysRegUsesToMove,
                            InstsToMove);
      continue;
    }

    // Don't merge volatiles.
    if (MBBI->hasOrderedMemoryRef())
      return false;

    int Swizzled =
        AMDGPU::getNamedOperandIdx(MBBI->getOpcode(), AMDGPU::OpName::swz);
    if (Swizzled != -1 && MBBI->getOperand(Swizzled).getImm())
      return false;

    // Handle a case like
    //   DS_WRITE_B32 addr, v, idx0
    //   w = DS_READ_B32 addr, idx0
    //   DS_WRITE_B32 addr, f(w), idx1
    // where the DS_READ_B32 ends up in InstsToMove and therefore prevents
    // merging of the two writes.
    if (addToListsIfDependent(*MBBI, RegDefsToMove, PhysRegUsesToMove,
                              InstsToMove))
      continue;

    if (&*MBBI == &*Paired.I) {
      if (TRI->hasAGPRs(getDataRegClass(*MBBI)) != IsAGPR)
        return false;
      // FIXME: nothing is illegal in a ds_write2 opcode with two AGPR data
      //        operands. However we are reporting that ds_write2 shall have
      //        only VGPR data so that machine copy propagation does not
      //        create an illegal instruction with a VGPR and AGPR sources.
      //        Consequenctially if we create such instruction the verifier
      //        will complain.
      if (IsAGPR && CI.InstClass == DS_WRITE)
        return false;

      // We need to go through the list of instructions that we plan to
      // move and make sure they are all safe to move down past the merged
      // instruction.
      if (canMoveInstsAcrossMemOp(*MBBI, InstsToMove, AA)) {

        // Call offsetsCanBeCombined with modify = true so that the offsets are
        // correct for the new instruction.  This should return true, because
        // this function should only be called on CombineInfo objects that
        // have already been confirmed to be mergeable.
        if (CI.InstClass != MIMG)
          offsetsCanBeCombined(CI, *STM, Paired, true);
        return true;
      }
      return false;
    }

    // We've found a load/store that we couldn't merge for some reason.
    // We could potentially keep looking, but we'd need to make sure that
    // it was safe to move I and also all the instruction in InstsToMove
    // down past this instruction.
    // check if we can move I across MBBI and if we can move all I's users
    if (!memAccessesCanBeReordered(*CI.I, *MBBI, AA) ||
        !canMoveInstsAcrossMemOp(*MBBI, InstsToMove, AA))
      break;
  }
  return false;
}

unsigned SILoadStoreOptimizer::read2Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_READ2_B32 : AMDGPU::DS_READ2_B64;
  return (EltSize == 4) ? AMDGPU::DS_READ2_B32_gfx9 : AMDGPU::DS_READ2_B64_gfx9;
}

unsigned SILoadStoreOptimizer::read2ST64Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_READ2ST64_B32 : AMDGPU::DS_READ2ST64_B64;

  return (EltSize == 4) ? AMDGPU::DS_READ2ST64_B32_gfx9
                        : AMDGPU::DS_READ2ST64_B64_gfx9;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeRead2Pair(CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be careful, since the addresses could be subregisters themselves in weird
  // cases, like vectors of pointers.
  const auto *AddrReg = TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);

  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdst);
  const auto *Dest1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdst);

  unsigned NewOffset0 = CI.Offset;
  unsigned NewOffset1 = Paired.Offset;
  unsigned Opc =
      CI.UseST64 ? read2ST64Opcode(CI.EltSize) : read2Opcode(CI.EltSize);

  unsigned SubRegIdx0 = (CI.EltSize == 4) ? AMDGPU::sub0 : AMDGPU::sub0_sub1;
  unsigned SubRegIdx1 = (CI.EltSize == 4) ? AMDGPU::sub1 : AMDGPU::sub2_sub3;

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(SubRegIdx0, SubRegIdx1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) && "Computed offset doesn't fit");

  const MCInstrDesc &Read2Desc = TII->get(Opc);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);
  Register DestReg = MRI->createVirtualRegister(SuperRC);

  DebugLoc DL = CI.I->getDebugLoc();

  Register BaseReg = AddrReg->getReg();
  unsigned BaseSubReg = AddrReg->getSubReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    Register ImmReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
    BuildMI(*MBB, Paired.I, DL, TII->get(AMDGPU::S_MOV_B32), ImmReg)
        .addImm(CI.BaseOff);

    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;

    TII->getAddNoCarry(*MBB, Paired.I, DL, BaseReg)
        .addReg(ImmReg)
        .addReg(AddrReg->getReg(), 0, BaseSubReg)
        .addImm(0); // clamp bit
    BaseSubReg = 0;
  }

  MachineInstrBuilder Read2 =
      BuildMI(*MBB, Paired.I, DL, Read2Desc, DestReg)
          .addReg(BaseReg, BaseRegFlags, BaseSubReg) // addr
          .addImm(NewOffset0)                        // offset0
          .addImm(NewOffset1)                        // offset1
          .addImm(0)                                 // gds
          .cloneMergedMemRefs({&*CI.I, &*Paired.I});

  (void)Read2;

  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);

  // Copy to the old destination registers.
  BuildMI(*MBB, Paired.I, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, Paired.I, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();

  LLVM_DEBUG(dbgs() << "Inserted read2: " << *Read2 << '\n');
  return Read2;
}

unsigned SILoadStoreOptimizer::write2Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_WRITE2_B32 : AMDGPU::DS_WRITE2_B64;
  return (EltSize == 4) ? AMDGPU::DS_WRITE2_B32_gfx9
                        : AMDGPU::DS_WRITE2_B64_gfx9;
}

unsigned SILoadStoreOptimizer::write2ST64Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_WRITE2ST64_B32
                          : AMDGPU::DS_WRITE2ST64_B64;

  return (EltSize == 4) ? AMDGPU::DS_WRITE2ST64_B32_gfx9
                        : AMDGPU::DS_WRITE2ST64_B64_gfx9;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeWrite2Pair(CombineInfo &CI, CombineInfo &Paired,
                                      const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be sure to use .addOperand(), and not .addReg() with these. We want to be
  // sure we preserve the subregister index and any register flags set on them.
  const MachineOperand *AddrReg =
      TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);
  const MachineOperand *Data0 =
      TII->getNamedOperand(*CI.I, AMDGPU::OpName::data0);
  const MachineOperand *Data1 =
      TII->getNamedOperand(*Paired.I, AMDGPU::OpName::data0);

  unsigned NewOffset0 = CI.Offset;
  unsigned NewOffset1 = Paired.Offset;
  unsigned Opc =
      CI.UseST64 ? write2ST64Opcode(CI.EltSize) : write2Opcode(CI.EltSize);

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(Data0, Data1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) && "Computed offset doesn't fit");

  const MCInstrDesc &Write2Desc = TII->get(Opc);
  DebugLoc DL = CI.I->getDebugLoc();

  Register BaseReg = AddrReg->getReg();
  unsigned BaseSubReg = AddrReg->getSubReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    Register ImmReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
    BuildMI(*MBB, Paired.I, DL, TII->get(AMDGPU::S_MOV_B32), ImmReg)
        .addImm(CI.BaseOff);

    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;

    TII->getAddNoCarry(*MBB, Paired.I, DL, BaseReg)
        .addReg(ImmReg)
        .addReg(AddrReg->getReg(), 0, BaseSubReg)
        .addImm(0); // clamp bit
    BaseSubReg = 0;
  }

  MachineInstrBuilder Write2 =
      BuildMI(*MBB, Paired.I, DL, Write2Desc)
          .addReg(BaseReg, BaseRegFlags, BaseSubReg) // addr
          .add(*Data0)                               // data0
          .add(*Data1)                               // data1
          .addImm(NewOffset0)                        // offset0
          .addImm(NewOffset1)                        // offset1
          .addImm(0)                                 // gds
          .cloneMergedMemRefs({&*CI.I, &*Paired.I});

  moveInstsAfter(Write2, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();

  LLVM_DEBUG(dbgs() << "Inserted write2 inst: " << *Write2 << '\n');
  return Write2;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeImagePair(CombineInfo &CI, CombineInfo &Paired,
                           const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  const unsigned Opcode = getNewOpcode(CI, Paired);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);

  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedDMask = CI.DMask | Paired.DMask;
  unsigned DMaskIdx =
      AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), AMDGPU::OpName::dmask);

  auto MIB = BuildMI(*MBB, Paired.I, DL, TII->get(Opcode), DestReg);
  for (unsigned I = 1, E = (*CI.I).getNumOperands(); I != E; ++I) {
    if (I == DMaskIdx)
      MIB.addImm(MergedDMask);
    else
      MIB.add((*CI.I).getOperand(I));
  }

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New = MIB.addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  unsigned SubRegIdx0, SubRegIdx1;
  std::tie(SubRegIdx0, SubRegIdx1) = getSubRegIdxs(CI, Paired);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdata);

  BuildMI(*MBB, Paired.I, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, Paired.I, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeSBufferLoadImmPair(
    CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  const unsigned Opcode = getNewOpcode(CI, Paired);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);

  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset, Paired.Offset);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New =
    BuildMI(*MBB, Paired.I, DL, TII->get(Opcode), DestReg)
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::sbase))
        .addImm(MergedOffset) // offset
        .addImm(CI.CPol)      // cpol
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI, Paired);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::sdst);
  const auto *Dest1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::sdst);

  BuildMI(*MBB, Paired.I, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, Paired.I, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeBufferLoadPair(
    CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI, Paired);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);

  // Copy to the new source register.
  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset, Paired.Offset);

  auto MIB = BuildMI(*MBB, Paired.I, DL, TII->get(Opcode), DestReg);

  AddressRegs Regs = getRegs(Opcode, *TII);

  if (Regs.VAddr)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New =
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
        .addImm(MergedOffset) // offset
        .addImm(CI.CPol)      // cpol
        .addImm(0)            // tfe
        .addImm(0)            // swz
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI, Paired);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdata);

  BuildMI(*MBB, Paired.I, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, Paired.I, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeTBufferLoadPair(
    CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI, Paired);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);

  // Copy to the new source register.
  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset, Paired.Offset);

  auto MIB = BuildMI(*MBB, Paired.I, DL, TII->get(Opcode), DestReg);

  AddressRegs Regs = getRegs(Opcode, *TII);

  if (Regs.VAddr)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  unsigned JoinedFormat =
      getBufferFormatWithCompCount(CI.Format, CI.Width + Paired.Width, *STM);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New =
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
          .addImm(MergedOffset) // offset
          .addImm(JoinedFormat) // format
          .addImm(CI.CPol)      // cpol
          .addImm(0)            // tfe
          .addImm(0)            // swz
          .addMemOperand(
              combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI, Paired);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdata);

  BuildMI(*MBB, Paired.I, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, Paired.I, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeTBufferStorePair(
    CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI, Paired);

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI, Paired);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the new source register.
  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);
  Register SrcReg = MRI->createVirtualRegister(SuperRC);

  const auto *Src0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Src1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdata);

  BuildMI(*MBB, Paired.I, DL, TII->get(AMDGPU::REG_SEQUENCE), SrcReg)
      .add(*Src0)
      .addImm(SubRegIdx0)
      .add(*Src1)
      .addImm(SubRegIdx1);

  auto MIB = BuildMI(*MBB, Paired.I, DL, TII->get(Opcode))
                 .addReg(SrcReg, RegState::Kill);

  AddressRegs Regs = getRegs(Opcode, *TII);

  if (Regs.VAddr)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  unsigned JoinedFormat =
      getBufferFormatWithCompCount(CI.Format, CI.Width + Paired.Width, *STM);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New =
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
          .addImm(std::min(CI.Offset, Paired.Offset)) // offset
          .addImm(JoinedFormat)                     // format
          .addImm(CI.CPol)                          // cpol
          .addImm(0)                                // tfe
          .addImm(0)                                // swz
          .addMemOperand(
              combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  moveInstsAfter(MIB, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

unsigned SILoadStoreOptimizer::getNewOpcode(const CombineInfo &CI,
                                            const CombineInfo &Paired) {
  const unsigned Width = CI.Width + Paired.Width;

  switch (CI.InstClass) {
  default:
    assert(CI.InstClass == BUFFER_LOAD || CI.InstClass == BUFFER_STORE);
    // FIXME: Handle d16 correctly
    return AMDGPU::getMUBUFOpcode(AMDGPU::getMUBUFBaseOpcode(CI.I->getOpcode()),
                                  Width);
  case TBUFFER_LOAD:
  case TBUFFER_STORE:
    return AMDGPU::getMTBUFOpcode(AMDGPU::getMTBUFBaseOpcode(CI.I->getOpcode()),
                                  Width);

  case UNKNOWN:
    llvm_unreachable("Unknown instruction class");
  case S_BUFFER_LOAD_IMM:
    switch (Width) {
    default:
      return 0;
    case 2:
      return AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM;
    case 4:
      return AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM;
    }
  case MIMG:
    assert("No overlaps" && (countPopulation(CI.DMask | Paired.DMask) == Width));
    return AMDGPU::getMaskedMIMGOp(CI.I->getOpcode(), Width);
  }
}

std::pair<unsigned, unsigned>
SILoadStoreOptimizer::getSubRegIdxs(const CombineInfo &CI, const CombineInfo &Paired) {

  if (CI.Width == 0 || Paired.Width == 0 || CI.Width + Paired.Width > 4)
    return std::make_pair(0, 0);

  bool ReverseOrder;
  if (CI.InstClass == MIMG) {
    assert((countPopulation(CI.DMask | Paired.DMask) == CI.Width + Paired.Width) &&
           "No overlaps");
    ReverseOrder = CI.DMask > Paired.DMask;
  } else
    ReverseOrder = CI.Offset > Paired.Offset;

  static const unsigned Idxs[4][4] = {
      {AMDGPU::sub0, AMDGPU::sub0_sub1, AMDGPU::sub0_sub1_sub2, AMDGPU::sub0_sub1_sub2_sub3},
      {AMDGPU::sub1, AMDGPU::sub1_sub2, AMDGPU::sub1_sub2_sub3, 0},
      {AMDGPU::sub2, AMDGPU::sub2_sub3, 0, 0},
      {AMDGPU::sub3, 0, 0, 0},
  };
  unsigned Idx0;
  unsigned Idx1;

  assert(CI.Width >= 1 && CI.Width <= 3);
  assert(Paired.Width >= 1 && Paired.Width <= 3);

  if (ReverseOrder) {
    Idx1 = Idxs[0][Paired.Width - 1];
    Idx0 = Idxs[Paired.Width][CI.Width - 1];
  } else {
    Idx0 = Idxs[0][CI.Width - 1];
    Idx1 = Idxs[CI.Width][Paired.Width - 1];
  }

  return std::make_pair(Idx0, Idx1);
}

const TargetRegisterClass *
SILoadStoreOptimizer::getTargetRegisterClass(const CombineInfo &CI,
                                             const CombineInfo &Paired) {
  if (CI.InstClass == S_BUFFER_LOAD_IMM) {
    switch (CI.Width + Paired.Width) {
    default:
      return nullptr;
    case 2:
      return &AMDGPU::SReg_64_XEXECRegClass;
    case 4:
      return &AMDGPU::SGPR_128RegClass;
    case 8:
      return &AMDGPU::SGPR_256RegClass;
    case 16:
      return &AMDGPU::SGPR_512RegClass;
    }
  }

  unsigned BitWidth = 32 * (CI.Width + Paired.Width);
  return TRI->hasAGPRs(getDataRegClass(*CI.I))
             ? TRI->getAGPRClassForBitWidth(BitWidth)
             : TRI->getVGPRClassForBitWidth(BitWidth);
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeBufferStorePair(
    CombineInfo &CI, CombineInfo &Paired,
    const SmallVectorImpl<MachineInstr *> &InstsToMove) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI, Paired);

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI, Paired);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the new source register.
  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI, Paired);
  Register SrcReg = MRI->createVirtualRegister(SuperRC);

  const auto *Src0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Src1 = TII->getNamedOperand(*Paired.I, AMDGPU::OpName::vdata);

  BuildMI(*MBB, Paired.I, DL, TII->get(AMDGPU::REG_SEQUENCE), SrcReg)
      .add(*Src0)
      .addImm(SubRegIdx0)
      .add(*Src1)
      .addImm(SubRegIdx1);

  auto MIB = BuildMI(*MBB, Paired.I, DL, TII->get(Opcode))
                 .addReg(SrcReg, RegState::Kill);

  AddressRegs Regs = getRegs(Opcode, *TII);

  if (Regs.VAddr)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));


  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && Paired.I->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *Paired.I->memoperands_begin();

  MachineInstr *New =
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
        .addImm(std::min(CI.Offset, Paired.Offset)) // offset
        .addImm(CI.CPol)      // cpol
        .addImm(0)            // tfe
        .addImm(0)            // swz
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  moveInstsAfter(MIB, InstsToMove);

  CI.I->eraseFromParent();
  Paired.I->eraseFromParent();
  return New;
}

MachineOperand
SILoadStoreOptimizer::createRegOrImm(int32_t Val, MachineInstr &MI) const {
  APInt V(32, Val, true);
  if (TII->isInlineConstant(V))
    return MachineOperand::CreateImm(Val);

  Register Reg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
  MachineInstr *Mov =
  BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
          TII->get(AMDGPU::S_MOV_B32), Reg)
    .addImm(Val);
  (void)Mov;
  LLVM_DEBUG(dbgs() << "    "; Mov->dump());
  return MachineOperand::CreateReg(Reg, false);
}

// Compute base address using Addr and return the final register.
Register SILoadStoreOptimizer::computeBase(MachineInstr &MI,
                                           const MemAddress &Addr) const {
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator MBBI = MI.getIterator();
  DebugLoc DL = MI.getDebugLoc();

  assert((TRI->getRegSizeInBits(Addr.Base.LoReg, *MRI) == 32 ||
          Addr.Base.LoSubReg) &&
         "Expected 32-bit Base-Register-Low!!");

  assert((TRI->getRegSizeInBits(Addr.Base.HiReg, *MRI) == 32 ||
          Addr.Base.HiSubReg) &&
         "Expected 32-bit Base-Register-Hi!!");

  LLVM_DEBUG(dbgs() << "  Re-Computed Anchor-Base:\n");
  MachineOperand OffsetLo = createRegOrImm(static_cast<int32_t>(Addr.Offset), MI);
  MachineOperand OffsetHi =
    createRegOrImm(static_cast<int32_t>(Addr.Offset >> 32), MI);

  const auto *CarryRC = TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);
  Register CarryReg = MRI->createVirtualRegister(CarryRC);
  Register DeadCarryReg = MRI->createVirtualRegister(CarryRC);

  Register DestSub0 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register DestSub1 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *LoHalf =
    BuildMI(*MBB, MBBI, DL, TII->get(AMDGPU::V_ADD_CO_U32_e64), DestSub0)
      .addReg(CarryReg, RegState::Define)
      .addReg(Addr.Base.LoReg, 0, Addr.Base.LoSubReg)
      .add(OffsetLo)
      .addImm(0); // clamp bit
  (void)LoHalf;
  LLVM_DEBUG(dbgs() << "    "; LoHalf->dump(););

  MachineInstr *HiHalf =
  BuildMI(*MBB, MBBI, DL, TII->get(AMDGPU::V_ADDC_U32_e64), DestSub1)
    .addReg(DeadCarryReg, RegState::Define | RegState::Dead)
    .addReg(Addr.Base.HiReg, 0, Addr.Base.HiSubReg)
    .add(OffsetHi)
    .addReg(CarryReg, RegState::Kill)
    .addImm(0); // clamp bit
  (void)HiHalf;
  LLVM_DEBUG(dbgs() << "    "; HiHalf->dump(););

  Register FullDestReg = MRI->createVirtualRegister(TRI->getVGPR64Class());
  MachineInstr *FullBase =
    BuildMI(*MBB, MBBI, DL, TII->get(TargetOpcode::REG_SEQUENCE), FullDestReg)
      .addReg(DestSub0)
      .addImm(AMDGPU::sub0)
      .addReg(DestSub1)
      .addImm(AMDGPU::sub1);
  (void)FullBase;
  LLVM_DEBUG(dbgs() << "    "; FullBase->dump(); dbgs() << "\n";);

  return FullDestReg;
}

// Update base and offset with the NewBase and NewOffset in MI.
void SILoadStoreOptimizer::updateBaseAndOffset(MachineInstr &MI,
                                               Register NewBase,
                                               int32_t NewOffset) const {
  auto Base = TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  Base->setReg(NewBase);
  Base->setIsKill(false);
  TII->getNamedOperand(MI, AMDGPU::OpName::offset)->setImm(NewOffset);
}

Optional<int32_t>
SILoadStoreOptimizer::extractConstOffset(const MachineOperand &Op) const {
  if (Op.isImm())
    return Op.getImm();

  if (!Op.isReg())
    return None;

  MachineInstr *Def = MRI->getUniqueVRegDef(Op.getReg());
  if (!Def || Def->getOpcode() != AMDGPU::S_MOV_B32 ||
      !Def->getOperand(1).isImm())
    return None;

  return Def->getOperand(1).getImm();
}

// Analyze Base and extracts:
//  - 32bit base registers, subregisters
//  - 64bit constant offset
// Expecting base computation as:
//   %OFFSET0:sgpr_32 = S_MOV_B32 8000
//   %LO:vgpr_32, %c:sreg_64_xexec =
//       V_ADD_CO_U32_e64 %BASE_LO:vgpr_32, %103:sgpr_32,
//   %HI:vgpr_32, = V_ADDC_U32_e64 %BASE_HI:vgpr_32, 0, killed %c:sreg_64_xexec
//   %Base:vreg_64 =
//       REG_SEQUENCE %LO:vgpr_32, %subreg.sub0, %HI:vgpr_32, %subreg.sub1
void SILoadStoreOptimizer::processBaseWithConstOffset(const MachineOperand &Base,
                                                      MemAddress &Addr) const {
  if (!Base.isReg())
    return;

  MachineInstr *Def = MRI->getUniqueVRegDef(Base.getReg());
  if (!Def || Def->getOpcode() != AMDGPU::REG_SEQUENCE
      || Def->getNumOperands() != 5)
    return;

  MachineOperand BaseLo = Def->getOperand(1);
  MachineOperand BaseHi = Def->getOperand(3);
  if (!BaseLo.isReg() || !BaseHi.isReg())
    return;

  MachineInstr *BaseLoDef = MRI->getUniqueVRegDef(BaseLo.getReg());
  MachineInstr *BaseHiDef = MRI->getUniqueVRegDef(BaseHi.getReg());

  if (!BaseLoDef || BaseLoDef->getOpcode() != AMDGPU::V_ADD_CO_U32_e64 ||
      !BaseHiDef || BaseHiDef->getOpcode() != AMDGPU::V_ADDC_U32_e64)
    return;

  const auto *Src0 = TII->getNamedOperand(*BaseLoDef, AMDGPU::OpName::src0);
  const auto *Src1 = TII->getNamedOperand(*BaseLoDef, AMDGPU::OpName::src1);

  auto Offset0P = extractConstOffset(*Src0);
  if (Offset0P)
    BaseLo = *Src1;
  else {
    if (!(Offset0P = extractConstOffset(*Src1)))
      return;
    BaseLo = *Src0;
  }

  Src0 = TII->getNamedOperand(*BaseHiDef, AMDGPU::OpName::src0);
  Src1 = TII->getNamedOperand(*BaseHiDef, AMDGPU::OpName::src1);

  if (Src0->isImm())
    std::swap(Src0, Src1);

  if (!Src1->isImm())
    return;

  uint64_t Offset1 = Src1->getImm();
  BaseHi = *Src0;

  Addr.Base.LoReg = BaseLo.getReg();
  Addr.Base.HiReg = BaseHi.getReg();
  Addr.Base.LoSubReg = BaseLo.getSubReg();
  Addr.Base.HiSubReg = BaseHi.getSubReg();
  Addr.Offset = (*Offset0P & 0x00000000ffffffff) | (Offset1 << 32);
}

bool SILoadStoreOptimizer::promoteConstantOffsetToImm(
    MachineInstr &MI,
    MemInfoMap &Visited,
    SmallPtrSet<MachineInstr *, 4> &AnchorList) const {

  if (!(MI.mayLoad() ^ MI.mayStore()))
    return false;

  // TODO: Support flat and scratch.
  if (AMDGPU::getGlobalSaddrOp(MI.getOpcode()) < 0)
    return false;

  if (MI.mayLoad() && TII->getNamedOperand(MI, AMDGPU::OpName::vdata) != NULL)
    return false;

  if (AnchorList.count(&MI))
    return false;

  LLVM_DEBUG(dbgs() << "\nTryToPromoteConstantOffsetToImmFor "; MI.dump());

  if (TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm()) {
    LLVM_DEBUG(dbgs() << "  Const-offset is already promoted.\n";);
    return false;
  }

  // Step1: Find the base-registers and a 64bit constant offset.
  MachineOperand &Base = *TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  MemAddress MAddr;
  if (Visited.find(&MI) == Visited.end()) {
    processBaseWithConstOffset(Base, MAddr);
    Visited[&MI] = MAddr;
  } else
    MAddr = Visited[&MI];

  if (MAddr.Offset == 0) {
    LLVM_DEBUG(dbgs() << "  Failed to extract constant-offset or there are no"
                         " constant offsets that can be promoted.\n";);
    return false;
  }

  LLVM_DEBUG(dbgs() << "  BASE: {" << MAddr.Base.HiReg << ", "
             << MAddr.Base.LoReg << "} Offset: " << MAddr.Offset << "\n\n";);

  // Step2: Traverse through MI's basic block and find an anchor(that has the
  // same base-registers) with the highest 13bit distance from MI's offset.
  // E.g. (64bit loads)
  // bb:
  //   addr1 = &a + 4096;   load1 = load(addr1,  0)
  //   addr2 = &a + 6144;   load2 = load(addr2,  0)
  //   addr3 = &a + 8192;   load3 = load(addr3,  0)
  //   addr4 = &a + 10240;  load4 = load(addr4,  0)
  //   addr5 = &a + 12288;  load5 = load(addr5,  0)
  //
  // Starting from the first load, the optimization will try to find a new base
  // from which (&a + 4096) has 13 bit distance. Both &a + 6144 and &a + 8192
  // has 13bit distance from &a + 4096. The heuristic considers &a + 8192
  // as the new-base(anchor) because of the maximum distance which can
  // accomodate more intermediate bases presumeably.
  //
  // Step3: move (&a + 8192) above load1. Compute and promote offsets from
  // (&a + 8192) for load1, load2, load4.
  //   addr = &a + 8192
  //   load1 = load(addr,       -4096)
  //   load2 = load(addr,       -2048)
  //   load3 = load(addr,       0)
  //   load4 = load(addr,       2048)
  //   addr5 = &a + 12288;  load5 = load(addr5,  0)
  //
  MachineInstr *AnchorInst = nullptr;
  MemAddress AnchorAddr;
  uint32_t MaxDist = std::numeric_limits<uint32_t>::min();
  SmallVector<std::pair<MachineInstr *, int64_t>, 4> InstsWCommonBase;

  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator E = MBB->end();
  MachineBasicBlock::iterator MBBI = MI.getIterator();
  ++MBBI;
  const SITargetLowering *TLI =
    static_cast<const SITargetLowering *>(STM->getTargetLowering());

  for ( ; MBBI != E; ++MBBI) {
    MachineInstr &MINext = *MBBI;
    // TODO: Support finding an anchor(with same base) from store addresses or
    // any other load addresses where the opcodes are different.
    if (MINext.getOpcode() != MI.getOpcode() ||
        TII->getNamedOperand(MINext, AMDGPU::OpName::offset)->getImm())
      continue;

    const MachineOperand &BaseNext =
      *TII->getNamedOperand(MINext, AMDGPU::OpName::vaddr);
    MemAddress MAddrNext;
    if (Visited.find(&MINext) == Visited.end()) {
      processBaseWithConstOffset(BaseNext, MAddrNext);
      Visited[&MINext] = MAddrNext;
    } else
      MAddrNext = Visited[&MINext];

    if (MAddrNext.Base.LoReg != MAddr.Base.LoReg ||
        MAddrNext.Base.HiReg != MAddr.Base.HiReg ||
        MAddrNext.Base.LoSubReg != MAddr.Base.LoSubReg ||
        MAddrNext.Base.HiSubReg != MAddr.Base.HiSubReg)
      continue;

    InstsWCommonBase.push_back(std::make_pair(&MINext, MAddrNext.Offset));

    int64_t Dist = MAddr.Offset - MAddrNext.Offset;
    TargetLoweringBase::AddrMode AM;
    AM.HasBaseReg = true;
    AM.BaseOffs = Dist;
    if (TLI->isLegalGlobalAddressingMode(AM) &&
        (uint32_t)std::abs(Dist) > MaxDist) {
      MaxDist = std::abs(Dist);

      AnchorAddr = MAddrNext;
      AnchorInst = &MINext;
    }
  }

  if (AnchorInst) {
    LLVM_DEBUG(dbgs() << "  Anchor-Inst(with max-distance from Offset): ";
               AnchorInst->dump());
    LLVM_DEBUG(dbgs() << "  Anchor-Offset from BASE: "
               <<  AnchorAddr.Offset << "\n\n");

    // Instead of moving up, just re-compute anchor-instruction's base address.
    Register Base = computeBase(MI, AnchorAddr);

    updateBaseAndOffset(MI, Base, MAddr.Offset - AnchorAddr.Offset);
    LLVM_DEBUG(dbgs() << "  After promotion: "; MI.dump(););

    for (auto P : InstsWCommonBase) {
      TargetLoweringBase::AddrMode AM;
      AM.HasBaseReg = true;
      AM.BaseOffs = P.second - AnchorAddr.Offset;

      if (TLI->isLegalGlobalAddressingMode(AM)) {
        LLVM_DEBUG(dbgs() << "  Promote Offset(" << P.second;
                   dbgs() << ")"; P.first->dump());
        updateBaseAndOffset(*P.first, Base, P.second - AnchorAddr.Offset);
        LLVM_DEBUG(dbgs() << "     After promotion: "; P.first->dump());
      }
    }
    AnchorList.insert(AnchorInst);
    return true;
  }

  return false;
}

void SILoadStoreOptimizer::addInstToMergeableList(const CombineInfo &CI,
                 std::list<std::list<CombineInfo> > &MergeableInsts) const {
  for (std::list<CombineInfo> &AddrList : MergeableInsts) {
    if (AddrList.front().InstClass == CI.InstClass &&
        AddrList.front().hasSameBaseAddress(*CI.I)) {
      AddrList.emplace_back(CI);
      return;
    }
  }

  // Base address not found, so add a new list.
  MergeableInsts.emplace_back(1, CI);
}

std::pair<MachineBasicBlock::iterator, bool>
SILoadStoreOptimizer::collectMergeableInsts(
    MachineBasicBlock::iterator Begin, MachineBasicBlock::iterator End,
    MemInfoMap &Visited, SmallPtrSet<MachineInstr *, 4> &AnchorList,
    std::list<std::list<CombineInfo>> &MergeableInsts) const {
  bool Modified = false;

  // Sort potential mergeable instructions into lists.  One list per base address.
  unsigned Order = 0;
  MachineBasicBlock::iterator BlockI = Begin;
  for (; BlockI != End; ++BlockI) {
    MachineInstr &MI = *BlockI;

    // We run this before checking if an address is mergeable, because it can produce
    // better code even if the instructions aren't mergeable.
    if (promoteConstantOffsetToImm(MI, Visited, AnchorList))
      Modified = true;

    // Don't combine if volatile. We also won't be able to merge across this, so
    // break the search. We can look after this barrier for separate merges.
    if (MI.hasOrderedMemoryRef()) {
      LLVM_DEBUG(dbgs() << "Breaking search on memory fence: " << MI);

      // Search will resume after this instruction in a separate merge list.
      ++BlockI;
      break;
    }

    const InstClassEnum InstClass = getInstClass(MI.getOpcode(), *TII);
    if (InstClass == UNKNOWN)
      continue;

    CombineInfo CI;
    CI.setMI(MI, *TII, *STM);
    CI.Order = Order++;

    if (!CI.hasMergeableAddress(*MRI))
      continue;

    LLVM_DEBUG(dbgs() << "Mergeable: " << MI);

    addInstToMergeableList(CI, MergeableInsts);
  }

  // At this point we have lists of Mergeable instructions.
  //
  // Part 2: Sort lists by offset and then for each CombineInfo object in the
  // list try to find an instruction that can be merged with I.  If an instruction
  // is found, it is stored in the Paired field.  If no instructions are found, then
  // the CombineInfo object is deleted from the list.

  for (std::list<std::list<CombineInfo>>::iterator I = MergeableInsts.begin(),
                                                   E = MergeableInsts.end(); I != E;) {

    std::list<CombineInfo> &MergeList = *I;
    if (MergeList.size() <= 1) {
      // This means we have found only one instruction with a given address
      // that can be merged, and we need at least 2 instructions to do a merge,
      // so this list can be discarded.
      I = MergeableInsts.erase(I);
      continue;
    }

    // Sort the lists by offsets, this way mergeable instructions will be
    // adjacent to each other in the list, which will make it easier to find
    // matches.
    MergeList.sort(
        [] (const CombineInfo &A, CombineInfo &B) {
          return A.Offset < B.Offset;
        });
    ++I;
  }

  return std::make_pair(BlockI, Modified);
}

// Scan through looking for adjacent LDS operations with constant offsets from
// the same base register. We rely on the scheduler to do the hard work of
// clustering nearby loads, and assume these are all adjacent.
bool SILoadStoreOptimizer::optimizeBlock(
                       std::list<std::list<CombineInfo> > &MergeableInsts) {
  bool Modified = false;

  for (std::list<std::list<CombineInfo>>::iterator I = MergeableInsts.begin(),
                                                   E = MergeableInsts.end(); I != E;) {
    std::list<CombineInfo> &MergeList = *I;

    bool OptimizeListAgain = false;
    if (!optimizeInstsWithSameBaseAddr(MergeList, OptimizeListAgain)) {
      // We weren't able to make any changes, so delete the list so we don't
      // process the same instructions the next time we try to optimize this
      // block.
      I = MergeableInsts.erase(I);
      continue;
    }

    Modified = true;

    // We made changes, but also determined that there were no more optimization
    // opportunities, so we don't need to reprocess the list
    if (!OptimizeListAgain) {
      I = MergeableInsts.erase(I);
      continue;
    }
    OptimizeAgain = true;
  }
  return Modified;
}

bool
SILoadStoreOptimizer::optimizeInstsWithSameBaseAddr(
                                          std::list<CombineInfo> &MergeList,
                                          bool &OptimizeListAgain) {
  if (MergeList.empty())
    return false;

  bool Modified = false;

  for (auto I = MergeList.begin(), Next = std::next(I); Next != MergeList.end();
       Next = std::next(I)) {

    auto First = I;
    auto Second = Next;

    if ((*First).Order > (*Second).Order)
      std::swap(First, Second);
    CombineInfo &CI = *First;
    CombineInfo &Paired = *Second;

    SmallVector<MachineInstr *, 8> InstsToMove;
    if (!checkAndPrepareMerge(CI, Paired, InstsToMove)) {
      ++I;
      continue;
    }

    Modified = true;

    LLVM_DEBUG(dbgs() << "Merging: " << *CI.I << "   with: " << *Paired.I);

    switch (CI.InstClass) {
    default:
      llvm_unreachable("unknown InstClass");
      break;
    case DS_READ: {
      MachineBasicBlock::iterator NewMI =
          mergeRead2Pair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      break;
    }
    case DS_WRITE: {
      MachineBasicBlock::iterator NewMI =
          mergeWrite2Pair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      break;
    }
    case S_BUFFER_LOAD_IMM: {
      MachineBasicBlock::iterator NewMI =
          mergeSBufferLoadImmPair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 16;
      break;
    }
    case BUFFER_LOAD: {
      MachineBasicBlock::iterator NewMI =
          mergeBufferLoadPair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 4;
      break;
    }
    case BUFFER_STORE: {
      MachineBasicBlock::iterator NewMI =
          mergeBufferStorePair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 4;
      break;
    }
    case MIMG: {
      MachineBasicBlock::iterator NewMI =
          mergeImagePair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 4;
      break;
    }
    case TBUFFER_LOAD: {
      MachineBasicBlock::iterator NewMI =
          mergeTBufferLoadPair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 4;
      break;
    }
    case TBUFFER_STORE: {
      MachineBasicBlock::iterator NewMI =
          mergeTBufferStorePair(CI, Paired, InstsToMove);
      CI.setMI(NewMI, *TII, *STM);
      OptimizeListAgain |= (CI.Width + Paired.Width) < 4;
      break;
    }
    }
    CI.Order = Paired.Order;
    if (I == Second)
      I = Next;

    MergeList.erase(Second);
  }

  return Modified;
}

bool SILoadStoreOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  STM = &MF.getSubtarget<GCNSubtarget>();
  if (!STM->loadStoreOptEnabled())
    return false;

  TII = STM->getInstrInfo();
  TRI = &TII->getRegisterInfo();

  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  LLVM_DEBUG(dbgs() << "Running SILoadStoreOptimizer\n");

  bool Modified = false;

  // Contains the list of instructions for which constant offsets are being
  // promoted to the IMM. This is tracked for an entire block at time.
  SmallPtrSet<MachineInstr *, 4> AnchorList;
  MemInfoMap Visited;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator SectionEnd;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;
         I = SectionEnd) {
      bool CollectModified;
      std::list<std::list<CombineInfo>> MergeableInsts;

      // First pass: Collect list of all instructions we know how to merge in a
      // subset of the block.
      std::tie(SectionEnd, CollectModified) =
          collectMergeableInsts(I, E, Visited, AnchorList, MergeableInsts);

      Modified |= CollectModified;

      do {
        OptimizeAgain = false;
        Modified |= optimizeBlock(MergeableInsts);
      } while (OptimizeAgain);
    }

    Visited.clear();
    AnchorList.clear();
  }

  return Modified;
}
