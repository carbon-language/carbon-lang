//===-- R600EmitClauseMarkers.cpp - Emit CF_ALU ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Add CF_ALU. R600 Alu instructions are grouped in clause which can hold
/// 128 Alu instructions ; these instructions can access up to 4 prefetched
/// 4 lines of 16 registers from constant buffers. Such ALU clauses are
/// initiated by CF_ALU instructions.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

class R600EmitClauseMarkersPass : public MachineFunctionPass {

private:
  static char ID;
  const R600InstrInfo *TII;

  unsigned OccupiedDwords(MachineInstr *MI) const {
    switch (MI->getOpcode()) {
    case AMDGPU::INTERP_PAIR_XY:
    case AMDGPU::INTERP_PAIR_ZW:
    case AMDGPU::INTERP_VEC_LOAD:
    case AMDGPU::DOT4_eg_pseudo:
    case AMDGPU::DOT4_r600_pseudo:
      return 4;
    case AMDGPU::KILL:
      return 0;
    default:
      break;
    }

    if(TII->isVector(*MI) ||
        TII->isCubeOp(MI->getOpcode()) ||
        TII->isReductionOp(MI->getOpcode()))
      return 4;

    unsigned NumLiteral = 0;
    for (MachineInstr::mop_iterator It = MI->operands_begin(),
        E = MI->operands_end(); It != E; ++It) {
      MachineOperand &MO = *It;
      if (MO.isReg() && MO.getReg() == AMDGPU::ALU_LITERAL_X)
        ++NumLiteral;
    }
    return 1 + NumLiteral;
  }

  bool isALU(const MachineInstr *MI) const {
    if (TII->isALUInstr(MI->getOpcode()))
      return true;
    if (TII->isVector(*MI) || TII->isCubeOp(MI->getOpcode()))
      return true;
    switch (MI->getOpcode()) {
    case AMDGPU::PRED_X:
    case AMDGPU::INTERP_PAIR_XY:
    case AMDGPU::INTERP_PAIR_ZW:
    case AMDGPU::INTERP_VEC_LOAD:
    case AMDGPU::COPY:
    case AMDGPU::DOT4_eg_pseudo:
    case AMDGPU::DOT4_r600_pseudo:
      return true;
    default:
      return false;
    }
  }

  bool IsTrivialInst(MachineInstr *MI) const {
    switch (MI->getOpcode()) {
    case AMDGPU::KILL:
    case AMDGPU::RETURN:
      return true;
    default:
      return false;
    }
  }

  // Register Idx, then Const value
  std::vector<std::pair<unsigned, unsigned> > ExtractConstRead(MachineInstr *MI)
      const {
    const R600Operands::Ops OpTable[3][2] = {
      {R600Operands::SRC0, R600Operands::SRC0_SEL},
      {R600Operands::SRC1, R600Operands::SRC1_SEL},
      {R600Operands::SRC2, R600Operands::SRC2_SEL},
    };
    std::vector<std::pair<unsigned, unsigned> > Result;

    if (!TII->isALUInstr(MI->getOpcode()))
      return Result;
    for (unsigned j = 0; j < 3; j++) {
      int SrcIdx = TII->getOperandIdx(MI->getOpcode(), OpTable[j][0]);
      if (SrcIdx < 0)
        break;
      if (MI->getOperand(SrcIdx).getReg() == AMDGPU::ALU_CONST) {
        unsigned Const = MI->getOperand(
            TII->getOperandIdx(MI->getOpcode(), OpTable[j][1])).getImm();
        Result.push_back(std::pair<unsigned, unsigned>(SrcIdx, Const));
      }
    }
    return Result;
  }

  std::pair<unsigned, unsigned> getAccessedBankLine(unsigned Sel) const {
    // Sel is (512 + (kc_bank << 12) + ConstIndex) << 2
    // (See also R600ISelLowering.cpp)
    // ConstIndex value is in [0, 4095];
    return std::pair<unsigned, unsigned>(
        ((Sel >> 2) - 512) >> 12, // KC_BANK
        // Line Number of ConstIndex
        // A line contains 16 constant registers however KCX bank can lock
        // two line at the same time ; thus we want to get an even line number.
        // Line number can be retrieved with (>>4), using (>>5) <<1 generates
        // an even number.
        ((((Sel >> 2) - 512) & 4095) >> 5) << 1);
  }

  bool SubstituteKCacheBank(MachineInstr *MI,
      std::vector<std::pair<unsigned, unsigned> > &CachedConsts) const {
    std::vector<std::pair<unsigned, unsigned> > UsedKCache;
    std::vector<std::pair<unsigned, unsigned> > Consts = ExtractConstRead(MI);
    assert(TII->isALUInstr(MI->getOpcode()) && "Can't assign Const");
    for (unsigned i = 0, n = Consts.size(); i < n; ++i) {
      unsigned Sel = Consts[i].second;
      unsigned Chan = Sel & 3, Index = ((Sel >> 2) - 512) & 31;
      unsigned KCacheIndex = Index * 4 + Chan;
      const std::pair<unsigned, unsigned> &BankLine = getAccessedBankLine(Sel);
      if (CachedConsts.empty()) {
        CachedConsts.push_back(BankLine);
        UsedKCache.push_back(std::pair<unsigned, unsigned>(0, KCacheIndex));
        continue;
      }
      if (CachedConsts[0] == BankLine) {
        UsedKCache.push_back(std::pair<unsigned, unsigned>(0, KCacheIndex));
        continue;
      }
      if (CachedConsts.size() == 1) {
        CachedConsts.push_back(BankLine);
        UsedKCache.push_back(std::pair<unsigned, unsigned>(1, KCacheIndex));
        continue;
      }
      if (CachedConsts[1] == BankLine) {
        UsedKCache.push_back(std::pair<unsigned, unsigned>(1, KCacheIndex));
        continue;
      }
      return false;
    }

    for (unsigned i = 0, n = Consts.size(); i < n; ++i) {
      switch(UsedKCache[i].first) {
      case 0:
        MI->getOperand(Consts[i].first).setReg(
            AMDGPU::R600_KC0RegClass.getRegister(UsedKCache[i].second));
        break;
      case 1:
        MI->getOperand(Consts[i].first).setReg(
            AMDGPU::R600_KC1RegClass.getRegister(UsedKCache[i].second));
        break;
      default:
        llvm_unreachable("Wrong Cache Line");
      }
    }
    return true;
  }

  MachineBasicBlock::iterator
  MakeALUClause(MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const {
    MachineBasicBlock::iterator ClauseHead = I;
    std::vector<std::pair<unsigned, unsigned> > KCacheBanks;
    bool PushBeforeModifier = false;
    unsigned AluInstCount = 0;
    for (MachineBasicBlock::iterator E = MBB.end(); I != E; ++I) {
      if (IsTrivialInst(I))
        continue;
      if (!isALU(I))
        break;
      if (I->getOpcode() == AMDGPU::PRED_X) {
        if (TII->getFlagOp(I).getImm() & MO_FLAG_PUSH)
          PushBeforeModifier = true;
        AluInstCount ++;
        continue;
      }
      if (I->getOpcode() == AMDGPU::KILLGT) {
        I++;
        break;
      }
      if (TII->isALUInstr(I->getOpcode()) &&
          !SubstituteKCacheBank(I, KCacheBanks))
        break;
      AluInstCount += OccupiedDwords(I);
      if (AluInstCount > TII->getMaxAlusPerClause())
        break;
    }
    unsigned Opcode = PushBeforeModifier ?
        AMDGPU::CF_ALU_PUSH_BEFORE : AMDGPU::CF_ALU;
    BuildMI(MBB, ClauseHead, MBB.findDebugLoc(ClauseHead), TII->get(Opcode))
        .addImm(0) // ADDR
        .addImm(KCacheBanks.empty()?0:KCacheBanks[0].first) // KB0
        .addImm((KCacheBanks.size() < 2)?0:KCacheBanks[1].first) // KB1
        .addImm(KCacheBanks.empty()?0:2) // KM0
        .addImm((KCacheBanks.size() < 2)?0:2) // KM1
        .addImm(KCacheBanks.empty()?0:KCacheBanks[0].second) // KLINE0
        .addImm((KCacheBanks.size() < 2)?0:KCacheBanks[1].second) // KLINE1
        .addImm(AluInstCount); // COUNT
    return I;
  }

public:
  R600EmitClauseMarkersPass(TargetMachine &tm) : MachineFunctionPass(ID),
    TII (static_cast<const R600InstrInfo *>(tm.getInstrInfo())) { }

  virtual bool runOnMachineFunction(MachineFunction &MF) {
    for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                    BB != BB_E; ++BB) {
      MachineBasicBlock &MBB = *BB;
      MachineBasicBlock::iterator I = MBB.begin();
      if (I->getOpcode() == AMDGPU::CF_ALU)
        continue; // BB was already parsed
      for (MachineBasicBlock::iterator E = MBB.end(); I != E;) {
        if (isALU(I))
          I = MakeALUClause(MBB, I);
        else
          ++I;
      }
    }
    return false;
  }

  const char *getPassName() const {
    return "R600 Emit Clause Markers Pass";
  }
};

char R600EmitClauseMarkersPass::ID = 0;

}


llvm::FunctionPass *llvm::createR600EmitClauseMarkers(TargetMachine &TM) {
  return new R600EmitClauseMarkersPass(TM);
}

