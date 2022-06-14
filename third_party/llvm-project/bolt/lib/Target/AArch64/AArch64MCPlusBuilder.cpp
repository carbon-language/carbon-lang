//===- bolt/Target/AArch64/AArch64MCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class AArch64MCPlusBuilder : public MCPlusBuilder {
public:
  AArch64MCPlusBuilder(const MCInstrAnalysis *Analysis, const MCInstrInfo *Info,
                       const MCRegisterInfo *RegInfo)
      : MCPlusBuilder(Analysis, Info, RegInfo) {}

  bool equals(const MCTargetExpr &A, const MCTargetExpr &B,
              CompFuncTy Comp) const override {
    const auto &AArch64ExprA = cast<AArch64MCExpr>(A);
    const auto &AArch64ExprB = cast<AArch64MCExpr>(B);
    if (AArch64ExprA.getKind() != AArch64ExprB.getKind())
      return false;

    return MCPlusBuilder::equals(*AArch64ExprA.getSubExpr(),
                                 *AArch64ExprB.getSubExpr(), Comp);
  }

  bool hasEVEXEncoding(const MCInst &) const override { return false; }

  bool isMacroOpFusionPair(ArrayRef<MCInst> Insts) const override {
    return false;
  }

  bool shortenInstruction(MCInst &, const MCSubtargetInfo &) const override {
    return false;
  }

  bool isADRP(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADRP;
  }

  bool isADR(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADR;
  }

  void getADRReg(const MCInst &Inst, MCPhysReg &RegName) const override {
    assert((isADR(Inst) || isADRP(Inst)) && "Not an ADR instruction");
    assert(MCPlus::getNumPrimeOperands(Inst) != 0 &&
           "No operands for ADR instruction");
    assert(Inst.getOperand(0).isReg() &&
           "Unexpected operand in ADR instruction");
    RegName = Inst.getOperand(0).getReg();
  }

  bool isTB(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::TBNZW ||
            Inst.getOpcode() == AArch64::TBNZX ||
            Inst.getOpcode() == AArch64::TBZW ||
            Inst.getOpcode() == AArch64::TBZX);
  }

  bool isCB(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::CBNZW ||
            Inst.getOpcode() == AArch64::CBNZX ||
            Inst.getOpcode() == AArch64::CBZW ||
            Inst.getOpcode() == AArch64::CBZX);
  }

  bool isMOVW(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::MOVKWi ||
            Inst.getOpcode() == AArch64::MOVKXi ||
            Inst.getOpcode() == AArch64::MOVNWi ||
            Inst.getOpcode() == AArch64::MOVNXi ||
            Inst.getOpcode() == AArch64::MOVZXi ||
            Inst.getOpcode() == AArch64::MOVZWi);
  }

  bool isADD(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::ADDSWri ||
            Inst.getOpcode() == AArch64::ADDSWrr ||
            Inst.getOpcode() == AArch64::ADDSWrs ||
            Inst.getOpcode() == AArch64::ADDSWrx ||
            Inst.getOpcode() == AArch64::ADDSXri ||
            Inst.getOpcode() == AArch64::ADDSXrr ||
            Inst.getOpcode() == AArch64::ADDSXrs ||
            Inst.getOpcode() == AArch64::ADDSXrx ||
            Inst.getOpcode() == AArch64::ADDSXrx64 ||
            Inst.getOpcode() == AArch64::ADDWri ||
            Inst.getOpcode() == AArch64::ADDWrr ||
            Inst.getOpcode() == AArch64::ADDWrs ||
            Inst.getOpcode() == AArch64::ADDWrx ||
            Inst.getOpcode() == AArch64::ADDXri ||
            Inst.getOpcode() == AArch64::ADDXrr ||
            Inst.getOpcode() == AArch64::ADDXrs ||
            Inst.getOpcode() == AArch64::ADDXrx ||
            Inst.getOpcode() == AArch64::ADDXrx64);
  }

  bool isLDRB(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::LDRBBpost ||
            Inst.getOpcode() == AArch64::LDRBBpre ||
            Inst.getOpcode() == AArch64::LDRBBroW ||
            Inst.getOpcode() == AArch64::LDRBBroX ||
            Inst.getOpcode() == AArch64::LDRBBui ||
            Inst.getOpcode() == AArch64::LDRSBWpost ||
            Inst.getOpcode() == AArch64::LDRSBWpre ||
            Inst.getOpcode() == AArch64::LDRSBWroW ||
            Inst.getOpcode() == AArch64::LDRSBWroX ||
            Inst.getOpcode() == AArch64::LDRSBWui ||
            Inst.getOpcode() == AArch64::LDRSBXpost ||
            Inst.getOpcode() == AArch64::LDRSBXpre ||
            Inst.getOpcode() == AArch64::LDRSBXroW ||
            Inst.getOpcode() == AArch64::LDRSBXroX ||
            Inst.getOpcode() == AArch64::LDRSBXui);
  }

  bool isLDRH(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::LDRHHpost ||
            Inst.getOpcode() == AArch64::LDRHHpre ||
            Inst.getOpcode() == AArch64::LDRHHroW ||
            Inst.getOpcode() == AArch64::LDRHHroX ||
            Inst.getOpcode() == AArch64::LDRHHui ||
            Inst.getOpcode() == AArch64::LDRSHWpost ||
            Inst.getOpcode() == AArch64::LDRSHWpre ||
            Inst.getOpcode() == AArch64::LDRSHWroW ||
            Inst.getOpcode() == AArch64::LDRSHWroX ||
            Inst.getOpcode() == AArch64::LDRSHWui ||
            Inst.getOpcode() == AArch64::LDRSHXpost ||
            Inst.getOpcode() == AArch64::LDRSHXpre ||
            Inst.getOpcode() == AArch64::LDRSHXroW ||
            Inst.getOpcode() == AArch64::LDRSHXroX ||
            Inst.getOpcode() == AArch64::LDRSHXui);
  }

  bool isLDRW(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::LDRWpost ||
            Inst.getOpcode() == AArch64::LDRWpre ||
            Inst.getOpcode() == AArch64::LDRWroW ||
            Inst.getOpcode() == AArch64::LDRWroX ||
            Inst.getOpcode() == AArch64::LDRWui);
  }

  bool isLDRX(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::LDRXpost ||
            Inst.getOpcode() == AArch64::LDRXpre ||
            Inst.getOpcode() == AArch64::LDRXroW ||
            Inst.getOpcode() == AArch64::LDRXroX ||
            Inst.getOpcode() == AArch64::LDRXui);
  }

  bool isLoad(const MCInst &Inst) const override {
    return isLDRB(Inst) || isLDRH(Inst) || isLDRW(Inst) || isLDRX(Inst);
  }

  bool isLoadFromStack(const MCInst &Inst) const {
    if (!isLoad(Inst))
      return false;
    const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());
    unsigned NumDefs = InstInfo.getNumDefs();
    for (unsigned I = NumDefs, E = InstInfo.getNumOperands(); I < E; ++I) {
      const MCOperand &Operand = Inst.getOperand(I);
      if (!Operand.isReg())
        continue;
      unsigned Reg = Operand.getReg();
      if (Reg == AArch64::SP || Reg == AArch64::WSP || Reg == AArch64::FP ||
          Reg == AArch64::W29)
        return true;
    }
    return false;
  }

  bool isRegToRegMove(const MCInst &Inst, MCPhysReg &From,
                      MCPhysReg &To) const override {
    if (Inst.getOpcode() != AArch64::ORRXrs)
      return false;
    if (Inst.getOperand(1).getReg() != AArch64::XZR)
      return false;
    if (Inst.getOperand(3).getImm() != 0)
      return false;
    From = Inst.getOperand(2).getReg();
    To = Inst.getOperand(0).getReg();
    return true;
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::BLR;
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    // ADRP is blacklisted and is an exception. Even though it has a
    // PC-relative operand, this operand is not a complete symbol reference
    // and BOLT shouldn't try to process it in isolation.
    if (isADRP(Inst))
      return false;

    if (isADR(Inst))
      return true;

    // Look for literal addressing mode (see C1-143 ARM DDI 0487B.a)
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I)
      if (MCII.OpInfo[I].OperandType == MCOI::OPERAND_PCREL)
        return true;

    return false;
  }

  bool evaluateADR(const MCInst &Inst, int64_t &Imm,
                   const MCExpr **DispExpr) const {
    assert((isADR(Inst) || isADRP(Inst)) && "Not an ADR instruction");

    const MCOperand &Label = Inst.getOperand(1);
    if (!Label.isImm()) {
      assert(Label.isExpr() && "Unexpected ADR operand");
      assert(DispExpr && "DispExpr must be set");
      *DispExpr = Label.getExpr();
      return false;
    }

    if (Inst.getOpcode() == AArch64::ADR) {
      Imm = Label.getImm();
      return true;
    }
    Imm = Label.getImm() << 12;
    return true;
  }

  bool evaluateAArch64MemoryOperand(const MCInst &Inst, int64_t &DispImm,
                                    const MCExpr **DispExpr = nullptr) const {
    if (isADR(Inst) || isADRP(Inst))
      return evaluateADR(Inst, DispImm, DispExpr);

    // Literal addressing mode
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
      if (MCII.OpInfo[I].OperandType != MCOI::OPERAND_PCREL)
        continue;

      if (!Inst.getOperand(I).isImm()) {
        assert(Inst.getOperand(I).isExpr() && "Unexpected PCREL operand");
        assert(DispExpr && "DispExpr must be set");
        *DispExpr = Inst.getOperand(I).getExpr();
        return true;
      }

      DispImm = Inst.getOperand(I).getImm() << 2;
      return true;
    }
    return false;
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    int64_t DispValue;
    const MCExpr *DispExpr = nullptr;
    if (!evaluateAArch64MemoryOperand(Inst, DispValue, &DispExpr))
      return false;

    // Make sure it's a well-formed addressing we can statically evaluate.
    if (DispExpr)
      return false;

    Target = DispValue;
    if (Inst.getOpcode() == AArch64::ADRP)
      Target += Address & ~0xFFFULL;
    else
      Target += Address;
    return true;
  }

  bool replaceMemOperandDisp(MCInst &Inst, MCOperand Operand) const override {
    MCInst::iterator OI = Inst.begin();
    if (isADR(Inst) || isADRP(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 2 &&
             "Unexpected number of operands");
      ++OI;
    } else {
      const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
      for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
        if (MCII.OpInfo[I].OperandType == MCOI::OPERAND_PCREL)
          break;
        ++OI;
      }
      assert(OI != Inst.end() && "Literal operand not found");
    }
    *OI = Operand;
    return true;
  }

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint64_t RelType) const override {

    if (isADR(Inst) || RelType == ELF::R_AARCH64_ADR_PREL_LO21 ||
        RelType == ELF::R_AARCH64_TLSDESC_ADR_PREL21) {
      return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS, Ctx);
    } else if (isADRP(Inst) || RelType == ELF::R_AARCH64_ADR_PREL_PG_HI21 ||
               RelType == ELF::R_AARCH64_ADR_PREL_PG_HI21_NC ||
               RelType == ELF::R_AARCH64_TLSDESC_ADR_PAGE21 ||
               RelType == ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 ||
               RelType == ELF::R_AARCH64_ADR_GOT_PAGE) {
      // Never emit a GOT reloc, we handled this in
      // RewriteInstance::readRelocations().
      return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_PAGE, Ctx);
    } else {
      switch (RelType) {
      case ELF::R_AARCH64_ADD_ABS_LO12_NC:
      case ELF::R_AARCH64_LD64_GOT_LO12_NC:
      case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
      case ELF::R_AARCH64_TLSDESC_ADD_LO12:
      case ELF::R_AARCH64_TLSDESC_LD64_LO12:
      case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
      case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_LO12, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G3:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_G3, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G2:
      case ELF::R_AARCH64_MOVW_UABS_G2_NC:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_G2_NC, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G1:
      case ELF::R_AARCH64_MOVW_UABS_G1_NC:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_G1_NC, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G0:
      case ELF::R_AARCH64_MOVW_UABS_G0_NC:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_G0_NC, Ctx);
      default:
        break;
      }
    }
    return Expr;
  }

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    if (OpNum >= MCPlus::getNumPrimeOperands(Inst))
      return false;

    // Auto-select correct operand number
    if (OpNum == 0) {
      if (isConditionalBranch(Inst) || isADR(Inst) || isADRP(Inst))
        OpNum = 1;
      if (isTB(Inst))
        OpNum = 2;
      if (isMOVW(Inst))
        OpNum = 1;
    }

    return true;
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *AArchExpr = dyn_cast<AArch64MCExpr>(Expr);
    if (AArchExpr && AArchExpr->getSubExpr())
      return getTargetSymbol(AArchExpr->getSubExpr());

    auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr);
    if (BinExpr)
      return getTargetSymbol(BinExpr->getLHS());

    auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr);
    if (SymExpr && SymExpr->getKind() == MCSymbolRefExpr::VK_None)
      return &SymExpr->getSymbol();

    return nullptr;
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (!getSymbolRefOperandNum(Inst, OpNum))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
  }

  int64_t getTargetAddend(const MCExpr *Expr) const override {
    auto *AArchExpr = dyn_cast<AArch64MCExpr>(Expr);
    if (AArchExpr && AArchExpr->getSubExpr())
      return getTargetAddend(AArchExpr->getSubExpr());

    auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr);
    if (BinExpr && BinExpr->getOpcode() == MCBinaryExpr::Add)
      return getTargetAddend(BinExpr->getRHS());

    auto *ConstExpr = dyn_cast<MCConstantExpr>(Expr);
    if (ConstExpr)
      return ConstExpr->getValue();

    return 0;
  }

  int64_t getTargetAddend(const MCInst &Inst,
                          unsigned OpNum = 0) const override {
    if (!getSymbolRefOperandNum(Inst, OpNum))
      return 0;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return 0;

    return getTargetAddend(Op.getExpr());
  }

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    size_t OpNum = 0;

    if (isConditionalBranch(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 2 &&
             "Invalid number of operands");
      OpNum = 1;
    }

    if (isTB(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 3 &&
             "Invalid number of operands");
      OpNum = 2;
    }

    if (Info->get(Inst.getOpcode()).OpInfo[OpNum].OperandType !=
        MCOI::OPERAND_PCREL) {
      assert((isIndirectBranch(Inst) || isIndirectCall(Inst)) &&
             "FAILED evaluateBranch");
      return false;
    }

    int64_t Imm = Inst.getOperand(OpNum).getImm() << 2;
    Target = Addr + Imm;
    return true;
  }

  bool replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");
    assert(MCPlus::getNumPrimeOperands(Inst) >= 1 &&
           "Invalid number of operands");
    MCInst::iterator OI = Inst.begin();

    if (isConditionalBranch(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 2 &&
             "Invalid number of operands");
      ++OI;
    }

    if (isTB(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 3 &&
             "Invalid number of operands");
      OI = Inst.begin() + 2;
    }

    *OI = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
    return true;
  }

  /// Matches indirect branch patterns in AArch64 related to a jump table (JT),
  /// helping us to build the complete CFG. A typical indirect branch to
  /// a jump table entry in AArch64 looks like the following:
  ///
  ///   adrp    x1, #-7585792           # Get JT Page location
  ///   add     x1, x1, #692            # Complement with JT Page offset
  ///   ldrh    w0, [x1, w0, uxtw #1]   # Loads JT entry
  ///   adr     x1, #12                 # Get PC + 12 (end of this BB) used next
  ///   add     x0, x1, w0, sxth #2     # Finish building branch target
  ///                                   # (entries in JT are relative to the end
  ///                                   #  of this BB)
  ///   br      x0                      # Indirect jump instruction
  ///
  bool analyzeIndirectBranchFragment(
      const MCInst &Inst,
      DenseMap<const MCInst *, SmallVector<MCInst *, 4>> &UDChain,
      const MCExpr *&JumpTable, int64_t &Offset, int64_t &ScaleValue,
      MCInst *&PCRelBase) const {
    // Expect AArch64 BR
    assert(Inst.getOpcode() == AArch64::BR && "Unexpected opcode");

    // Match the indirect branch pattern for aarch64
    SmallVector<MCInst *, 4> &UsesRoot = UDChain[&Inst];
    if (UsesRoot.size() == 0 || UsesRoot[0] == nullptr)
      return false;

    const MCInst *DefAdd = UsesRoot[0];

    // Now we match an ADD
    if (!isADD(*DefAdd)) {
      // If the address is not broken up in two parts, this is not branching
      // according to a jump table entry. Fail.
      return false;
    }
    if (DefAdd->getOpcode() == AArch64::ADDXri) {
      // This can happen when there is no offset, but a direct jump that was
      // transformed into an indirect one  (indirect tail call) :
      //   ADRP   x2, Perl_re_compiler
      //   ADD    x2, x2, :lo12:Perl_re_compiler
      //   BR     x2
      return false;
    }
    if (DefAdd->getOpcode() == AArch64::ADDXrs) {
      // Covers the less common pattern where JT entries are relative to
      // the JT itself (like x86). Seems less efficient since we can't
      // assume the JT is aligned at 4B boundary and thus drop 2 bits from
      // JT values.
      // cde264:
      //    adrp    x12, #21544960  ; 216a000
      //    add     x12, x12, #1696 ; 216a6a0  (JT object in .rodata)
      //    ldrsw   x8, [x12, x8, lsl #2]   --> loads e.g. 0xfeb73bd8
      //  * add     x8, x8, x12   --> = cde278, next block
      //    br      x8
      // cde278:
      //
      // Parsed as ADDXrs reg:x8 reg:x8 reg:x12 imm:0
      return false;
    }
    assert(DefAdd->getOpcode() == AArch64::ADDXrx &&
           "Failed to match indirect branch!");

    // Validate ADD operands
    int64_t OperandExtension = DefAdd->getOperand(3).getImm();
    unsigned ShiftVal = AArch64_AM::getArithShiftValue(OperandExtension);
    AArch64_AM::ShiftExtendType ExtendType =
        AArch64_AM::getArithExtendType(OperandExtension);
    if (ShiftVal != 2)
      llvm_unreachable("Failed to match indirect branch! (fragment 2)");

    if (ExtendType == AArch64_AM::SXTB)
      ScaleValue = 1LL;
    else if (ExtendType == AArch64_AM::SXTH)
      ScaleValue = 2LL;
    else if (ExtendType == AArch64_AM::SXTW)
      ScaleValue = 4LL;
    else
      llvm_unreachable("Failed to match indirect branch! (fragment 3)");

    // Match an ADR to load base address to be used when addressing JT targets
    SmallVector<MCInst *, 4> &UsesAdd = UDChain[DefAdd];
    if (UsesAdd.size() <= 1 || UsesAdd[1] == nullptr || UsesAdd[2] == nullptr) {
      // This happens when we don't have enough context about this jump table
      // because the jumping code sequence was split in multiple basic blocks.
      // This was observed in the wild in HHVM code (dispatchImpl).
      return false;
    }
    MCInst *DefBaseAddr = UsesAdd[1];
    assert(DefBaseAddr->getOpcode() == AArch64::ADR &&
           "Failed to match indirect branch pattern! (fragment 3)");

    PCRelBase = DefBaseAddr;
    // Match LOAD to load the jump table (relative) target
    const MCInst *DefLoad = UsesAdd[2];
    assert(isLoad(*DefLoad) &&
           "Failed to match indirect branch load pattern! (1)");
    assert((ScaleValue != 1LL || isLDRB(*DefLoad)) &&
           "Failed to match indirect branch load pattern! (2)");
    assert((ScaleValue != 2LL || isLDRH(*DefLoad)) &&
           "Failed to match indirect branch load pattern! (3)");

    // Match ADD that calculates the JumpTable Base Address (not the offset)
    SmallVector<MCInst *, 4> &UsesLoad = UDChain[DefLoad];
    const MCInst *DefJTBaseAdd = UsesLoad[1];
    MCPhysReg From, To;
    if (DefJTBaseAdd == nullptr || isLoadFromStack(*DefJTBaseAdd) ||
        isRegToRegMove(*DefJTBaseAdd, From, To)) {
      // Sometimes base address may have been defined in another basic block
      // (hoisted). Return with no jump table info.
      JumpTable = nullptr;
      return true;
    }

    assert(DefJTBaseAdd->getOpcode() == AArch64::ADDXri &&
           "Failed to match jump table base address pattern! (1)");

    if (DefJTBaseAdd->getOperand(2).isImm())
      Offset = DefJTBaseAdd->getOperand(2).getImm();
    SmallVector<MCInst *, 4> &UsesJTBaseAdd = UDChain[DefJTBaseAdd];
    const MCInst *DefJTBasePage = UsesJTBaseAdd[1];
    if (DefJTBasePage == nullptr || isLoadFromStack(*DefJTBasePage)) {
      JumpTable = nullptr;
      return true;
    }
    assert(DefJTBasePage->getOpcode() == AArch64::ADRP &&
           "Failed to match jump table base page pattern! (2)");
    if (DefJTBasePage->getOperand(1).isExpr())
      JumpTable = DefJTBasePage->getOperand(1).getExpr();
    return true;
  }

  DenseMap<const MCInst *, SmallVector<MCInst *, 4>>
  computeLocalUDChain(const MCInst *CurInstr, InstructionIterator Begin,
                      InstructionIterator End) const {
    DenseMap<int, MCInst *> RegAliasTable;
    DenseMap<const MCInst *, SmallVector<MCInst *, 4>> Uses;

    auto addInstrOperands = [&](const MCInst &Instr) {
      // Update Uses table
      for (const MCOperand &Operand : MCPlus::primeOperands(Instr)) {
        if (!Operand.isReg())
          continue;
        unsigned Reg = Operand.getReg();
        MCInst *AliasInst = RegAliasTable[Reg];
        Uses[&Instr].push_back(AliasInst);
        LLVM_DEBUG({
          dbgs() << "Adding reg operand " << Reg << " refs ";
          if (AliasInst != nullptr)
            AliasInst->dump();
          else
            dbgs() << "\n";
        });
      }
    };

    LLVM_DEBUG(dbgs() << "computeLocalUDChain\n");
    bool TerminatorSeen = false;
    for (auto II = Begin; II != End; ++II) {
      MCInst &Instr = *II;
      // Ignore nops and CFIs
      if (isPseudo(Instr) || isNoop(Instr))
        continue;
      if (TerminatorSeen) {
        RegAliasTable.clear();
        Uses.clear();
      }

      LLVM_DEBUG(dbgs() << "Now updating for:\n ");
      LLVM_DEBUG(Instr.dump());
      addInstrOperands(Instr);

      BitVector Regs = BitVector(RegInfo->getNumRegs(), false);
      getWrittenRegs(Instr, Regs);

      // Update register definitions after this point
      for (int Idx : Regs.set_bits()) {
        RegAliasTable[Idx] = &Instr;
        LLVM_DEBUG(dbgs() << "Setting reg " << Idx
                          << " def to current instr.\n");
      }

      TerminatorSeen = isTerminator(Instr);
    }

    // Process the last instruction, which is not currently added into the
    // instruction stream
    if (CurInstr)
      addInstrOperands(*CurInstr);

    return Uses;
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
      unsigned &IndexRegNumOut, int64_t &DispValueOut,
      const MCExpr *&DispExprOut, MCInst *&PCRelBaseOut) const override {
    MemLocInstrOut = nullptr;
    BaseRegNumOut = AArch64::NoRegister;
    IndexRegNumOut = AArch64::NoRegister;
    DispValueOut = 0;
    DispExprOut = nullptr;

    // An instruction referencing memory used by jump instruction (directly or
    // via register). This location could be an array of function pointers
    // in case of indirect tail call, or a jump table.
    MCInst *MemLocInstr = nullptr;

    // Analyze the memory location.
    int64_t ScaleValue, DispValue;
    const MCExpr *DispExpr;

    DenseMap<const MCInst *, SmallVector<llvm::MCInst *, 4>> UDChain =
        computeLocalUDChain(&Instruction, Begin, End);
    MCInst *PCRelBase;
    if (!analyzeIndirectBranchFragment(Instruction, UDChain, DispExpr,
                                       DispValue, ScaleValue, PCRelBase))
      return IndirectBranchType::UNKNOWN;

    MemLocInstrOut = MemLocInstr;
    DispValueOut = DispValue;
    DispExprOut = DispExpr;
    PCRelBaseOut = PCRelBase;
    return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
  }

  ///  Matches PLT entry pattern and returns the associated GOT entry address.
  ///  Typical PLT entry looks like the following:
  ///
  ///    adrp    x16, 230000
  ///    ldr     x17, [x16, #3040]
  ///    add     x16, x16, #0xbe0
  ///    br      x17
  ///
  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override {
    // Check branch instruction
    MCInst *Branch = &Instruction;
    assert(Branch->getOpcode() == AArch64::BR && "Unexpected opcode");

    DenseMap<const MCInst *, SmallVector<llvm::MCInst *, 4>> UDChain =
        computeLocalUDChain(Branch, Begin, End);

    // Match ldr instruction
    SmallVector<MCInst *, 4> &BranchUses = UDChain[Branch];
    if (BranchUses.size() < 1 || BranchUses[0] == nullptr)
      return 0;

    // Check ldr instruction
    const MCInst *Ldr = BranchUses[0];
    if (Ldr->getOpcode() != AArch64::LDRXui)
      return 0;

    // Get ldr value
    const unsigned ScaleLdr = 8; // LDRX operates on 8 bytes segments
    assert(Ldr->getOperand(2).isImm() && "Unexpected ldr operand");
    const uint64_t Offset = Ldr->getOperand(2).getImm() * ScaleLdr;

    // Match adrp instruction
    SmallVector<MCInst *, 4> &LdrUses = UDChain[Ldr];
    if (LdrUses.size() < 2 || LdrUses[1] == nullptr)
      return 0;

    // Check adrp instruction
    MCInst *Adrp = LdrUses[1];
    if (Adrp->getOpcode() != AArch64::ADRP)
      return 0;

    // Get adrp instruction PC
    const unsigned InstSize = 4;
    uint64_t AdrpPC = BeginPC;
    for (InstructionIterator It = Begin; It != End; ++It) {
      if (&(*It) == Adrp)
        break;
      AdrpPC += InstSize;
    }

    // Get adrp value
    uint64_t Base;
    assert(Adrp->getOperand(1).isImm() && "Unexpected adrp operand");
    bool Ret = evaluateMemOperandTarget(*Adrp, Base, AdrpPC, InstSize);
    assert(Ret && "Failed to evaluate adrp");
    (void)Ret;

    return Base + Offset;
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case AArch64::TBZW:     return AArch64::TBNZW;
    case AArch64::TBZX:     return AArch64::TBNZX;
    case AArch64::TBNZW:    return AArch64::TBZW;
    case AArch64::TBNZX:    return AArch64::TBZX;
    case AArch64::CBZW:     return AArch64::CBNZW;
    case AArch64::CBZX:     return AArch64::CBNZX;
    case AArch64::CBNZW:    return AArch64::CBZW;
    case AArch64::CBNZX:    return AArch64::CBZX;
    }
  }

  unsigned getCondCode(const MCInst &Inst) const override {
    // AArch64 does not use conditional codes, so we just return the opcode
    // of the conditional branch here.
    return Inst.getOpcode();
  }

  unsigned getCanonicalBranchCondCode(unsigned Opcode) const override {
    switch (Opcode) {
    default:
      return Opcode;
    case AArch64::TBNZW:    return AArch64::TBZW;
    case AArch64::TBNZX:    return AArch64::TBZX;
    case AArch64::CBNZW:    return AArch64::CBZW;
    case AArch64::CBNZX:    return AArch64::CBZX;
    }
  }

  bool reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    if (isTB(Inst) || isCB(Inst)) {
      Inst.setOpcode(getInvertedBranchOpcode(Inst.getOpcode()));
      assert(Inst.getOpcode() != 0 && "Invalid branch instruction");
    } else if (Inst.getOpcode() == AArch64::Bcc) {
      Inst.getOperand(0).setImm(AArch64CC::getInvertedCondCode(
          static_cast<AArch64CC::CondCode>(Inst.getOperand(0).getImm())));
      assert(Inst.getOperand(0).getImm() != AArch64CC::AL &&
             Inst.getOperand(0).getImm() != AArch64CC::NV &&
             "Can't reverse ALWAYS cond code");
    } else {
      LLVM_DEBUG(Inst.dump());
      llvm_unreachable("Unrecognized branch instruction");
    }
    return replaceBranchTarget(Inst, TBB, Ctx);
  }

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("Failed to get pcrel encoding size");
      return 0;
    case AArch64::TBZW:     return 16;
    case AArch64::TBZX:     return 16;
    case AArch64::TBNZW:    return 16;
    case AArch64::TBNZX:    return 16;
    case AArch64::CBZW:     return 21;
    case AArch64::CBZX:     return 21;
    case AArch64::CBNZW:    return 21;
    case AArch64::CBNZX:    return 21;
    case AArch64::B:        return 28;
    case AArch64::BL:       return 28;
    case AArch64::Bcc:      return 21;
    }
  }

  int getShortJmpEncodingSize() const override { return 33; }

  int getUncondBranchEncodingSize() const override { return 28; }

  bool createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    Inst.setOpcode(AArch64::B);
    Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
        Inst, MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        *Ctx, 0)));
    setTailCall(Inst);
    return true;
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    createShortJmp(Seq, Target, Ctx, /*IsTailCall*/ true);
  }

  bool createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(AArch64::BRK);
    Inst.addOperand(MCOperand::createImm(1));
    return true;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    setTailCall(Inst);
    return true;
  }

  bool convertTailCallToJmp(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    clearOffset(Inst);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  bool isNoop(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::HINT &&
           Inst.getOperand(0).getImm() == 0;
  }

  bool createNoop(MCInst &Inst) const override {
    Inst.setOpcode(AArch64::HINT);
    Inst.clear();
    Inst.addOperand(MCOperand::createImm(0));
    return true;
  }

  bool isStore(const MCInst &Inst) const override { return false; }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I) || isNoop(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I) || isTailCall(*I) || !isBranch(*I))
        break;

      // Handle unconditional branches.
      if (isUnconditionalBranch(*I)) {
        // If any code was seen after this unconditional branch, we've seen
        // unreachable code. Ignore them.
        CondBranch = nullptr;
        UncondBranch = &*I;
        const MCSymbol *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I))
        return false;

      if (CondBranch == nullptr) {
        const MCSymbol *TargetBB = getTargetSymbol(*I);
        if (TargetBB == nullptr) {
          // Unrecognized branch target
          return false;
        }
        FBB = TBB;
        TBB = TargetBB;
        CondBranch = &*I;
        continue;
      }

      llvm_unreachable("multiple conditional branches in one BB");
    }
    return true;
  }

  void createLongJmp(InstructionListType &Seq, const MCSymbol *Target,
                     MCContext *Ctx, bool IsTailCall) override {
    // ip0 (r16) is reserved to the linker (refer to 5.3.1.1 of "Procedure Call
    //   Standard for the ARM 64-bit Architecture (AArch64)".
    // The sequence of instructions we create here is the following:
    //  movz ip0, #:abs_g3:<addr>
    //  movk ip0, #:abs_g2_nc:<addr>
    //  movk ip0, #:abs_g1_nc:<addr>
    //  movk ip0, #:abs_g0_nc:<addr>
    //  br ip0
    MCInst Inst;
    Inst.setOpcode(AArch64::MOVZXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(AArch64MCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        AArch64MCExpr::VK_ABS_G3, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x30));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(AArch64MCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        AArch64MCExpr::VK_ABS_G2_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x20));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(AArch64MCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        AArch64MCExpr::VK_ABS_G1_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x10));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(AArch64MCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        AArch64MCExpr::VK_ABS_G0_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::BR);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    if (IsTailCall)
      setTailCall(Inst);
    Seq.emplace_back(Inst);
  }

  void createShortJmp(InstructionListType &Seq, const MCSymbol *Target,
                      MCContext *Ctx, bool IsTailCall) override {
    // ip0 (r16) is reserved to the linker (refer to 5.3.1.1 of "Procedure Call
    //   Standard for the ARM 64-bit Architecture (AArch64)".
    // The sequence of instructions we create here is the following:
    //  adrp ip0, imm
    //  add ip0, ip0, imm
    //  br ip0
    MCPhysReg Reg = AArch64::X16;
    InstructionListType Insts = materializeAddress(Target, Ctx, Reg);
    Insts.emplace_back();
    MCInst &Inst = Insts.back();
    Inst.clear();
    Inst.setOpcode(AArch64::BR);
    Inst.addOperand(MCOperand::createReg(Reg));
    if (IsTailCall)
      setTailCall(Inst);
    Seq.swap(Insts);
  }

  /// Matching pattern here is
  ///
  ///    ADRP  x16, imm
  ///    ADD   x16, x16, imm
  ///    BR    x16
  ///
  bool matchLinkerVeneer(InstructionIterator Begin, InstructionIterator End,
                         uint64_t Address, const MCInst &CurInst,
                         MCInst *&TargetHiBits, MCInst *&TargetLowBits,
                         uint64_t &Target) const override {
    if (CurInst.getOpcode() != AArch64::BR || !CurInst.getOperand(0).isReg() ||
        CurInst.getOperand(0).getReg() != AArch64::X16)
      return false;

    auto I = End;
    if (I == Begin)
      return false;

    --I;
    Address -= 4;
    if (I == Begin || I->getOpcode() != AArch64::ADDXri ||
        MCPlus::getNumPrimeOperands(*I) < 3 || !I->getOperand(0).isReg() ||
        !I->getOperand(1).isReg() ||
        I->getOperand(0).getReg() != AArch64::X16 ||
        I->getOperand(1).getReg() != AArch64::X16 || !I->getOperand(2).isImm())
      return false;
    TargetLowBits = &*I;
    uint64_t Addr = I->getOperand(2).getImm() & 0xFFF;

    --I;
    Address -= 4;
    if (I->getOpcode() != AArch64::ADRP ||
        MCPlus::getNumPrimeOperands(*I) < 2 || !I->getOperand(0).isReg() ||
        !I->getOperand(1).isImm() || I->getOperand(0).getReg() != AArch64::X16)
      return false;
    TargetHiBits = &*I;
    Addr |= (Address + ((int64_t)I->getOperand(1).getImm() << 12)) &
            0xFFFFFFFFFFFFF000ULL;
    Target = Addr;
    return true;
  }

  bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                               int64_t Addend, MCContext *Ctx, int64_t &Value,
                               uint64_t RelType) const override {
    unsigned ImmOpNo = -1U;
    for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
         ++Index) {
      if (Inst.getOperand(Index).isImm()) {
        ImmOpNo = Index;
        break;
      }
    }
    if (ImmOpNo == -1U)
      return false;

    Value = Inst.getOperand(ImmOpNo).getImm();

    setOperandToSymbolRef(Inst, ImmOpNo, Symbol, Addend, Ctx, RelType);

    return true;
  }

  bool createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(AArch64::B);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
        Inst, MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx),
        *Ctx, 0)));
    return true;
  }

  bool isMoveMem2Reg(const MCInst &Inst) const override { return false; }

  bool isLeave(const MCInst &Inst) const override { return false; }

  bool isPop(const MCInst &Inst) const override { return false; }

  bool isPrefix(const MCInst &Inst) const override { return false; }

  bool deleteREPPrefix(MCInst &Inst) const override { return false; }

  bool createReturn(MCInst &Inst) const override {
    Inst.setOpcode(AArch64::RET);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(AArch64::LR));
    return true;
  }

  InstructionListType materializeAddress(const MCSymbol *Target, MCContext *Ctx,
                                         MCPhysReg RegName,
                                         int64_t Addend = 0) const override {
    // Get page-aligned address and add page offset
    InstructionListType Insts(2);
    Insts[0].setOpcode(AArch64::ADRP);
    Insts[0].clear();
    Insts[0].addOperand(MCOperand::createReg(RegName));
    Insts[0].addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(Insts[0], /* OpNum */ 1, Target, Addend, Ctx,
                          ELF::R_AARCH64_NONE);
    Insts[1].setOpcode(AArch64::ADDXri);
    Insts[1].clear();
    Insts[1].addOperand(MCOperand::createReg(RegName));
    Insts[1].addOperand(MCOperand::createReg(RegName));
    Insts[1].addOperand(MCOperand::createImm(0));
    Insts[1].addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(Insts[1], /* OpNum */ 2, Target, Addend, Ctx,
                          ELF::R_AARCH64_ADD_ABS_LO12_NC);
    return Insts;
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createAArch64MCPlusBuilder(const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo) {
  return new AArch64MCPlusBuilder(Analysis, Info, RegInfo);
}

} // namespace bolt
} // namespace llvm
