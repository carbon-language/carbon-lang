//===-- AArch64MCPlusBuilder.cpp - --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64-specific MC+ builder.
//
//===----------------------------------------------------------------------===//

#include "MCPlus.h"
#include "MCPlusBuilder.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "InstPrinter/AArch64InstPrinter.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "MCTargetDesc/AArch64ELFStreamer.h"
#include "MCTargetDesc/AArch64MCAsmInfo.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "MCTargetDesc/AArch64WinCOFFStreamer.h"
#include "Utils/AArch64BaseInfo.h"

#define DEBUG_TYPE "bolt-aarch64"

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
    if (AArch64ExprA.getKind() !=  AArch64ExprB.getKind())
      return false;

    return MCPlusBuilder::equals(*AArch64ExprA.getSubExpr(),
                                 *AArch64ExprB.getSubExpr(), Comp);
  }

  bool hasEVEXEncoding(const MCInst &) const override {
    return false;
  }

  bool isMacroOpFusionPair(ArrayRef<MCInst> Insts) const override {
    return false;
  }

  bool shortenInstruction(MCInst &) const override {
    return false;
  }

  bool isADRP(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADRP;
  }

  bool isADR(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADR;
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
    const auto InstInfo = Info->get(Inst.getOpcode());
    auto NumDefs = InstInfo.getNumDefs();
    for (unsigned I = NumDefs, E = InstInfo.getNumOperands(); I < E; ++I) {
      auto &Operand = Inst.getOperand(I);
      if (!Operand.isReg())
        continue;
      auto Reg = Operand.getReg();
      if (Reg == AArch64::SP || Reg == AArch64::WSP ||
          Reg == AArch64::FP || Reg == AArch64::W29)
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

  MCPhysReg getNoRegister() const override {
    return AArch64::NoRegister;
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
    const auto MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
      if (MCII.OpInfo[I].OperandType == MCOI::OPERAND_PCREL)
        return true;
    }
    return false;
  }

  bool evaluateADR(const MCInst &Inst, int64_t &Imm,
                   const MCExpr **DispExpr) const {
    assert((isADR(Inst) || isADRP(Inst)) && "Not an ADR instruction");

    auto &Label = Inst.getOperand(1);
    if (!Label.isImm()) {
      assert (Label.isExpr() && "Unexpected ADR operand");
      assert (DispExpr && "DispExpr must be set");
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

  bool evaluateAArch64MemoryOperand(const MCInst &Inst,
                                    int64_t &DispImm,
                                    const MCExpr **DispExpr = nullptr)
                                                                const {
    if (isADR(Inst) || isADRP(Inst))
      return evaluateADR(Inst, DispImm, DispExpr);

    // Literal addressing mode
    const auto MCII = Info->get(Inst.getOpcode());
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
    int64_t       DispValue;
    const MCExpr* DispExpr{nullptr};
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
      const auto MCII = Info->get(Inst.getOpcode());
      for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
        if (MCII.OpInfo[I].OperandType == MCOI::OPERAND_PCREL) {
          break;
        }
        ++OI;
      }
      assert(OI != Inst.end() && "Literal operand not found");
    }
    OI = Inst.erase(OI);
    Inst.insert(OI, Operand);
    return true;
  }

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint64_t RelType) const override {
    if (RelType == ELF::R_AARCH64_ADR_GOT_PAGE ||
        RelType == ELF::R_AARCH64_TLSDESC_ADR_PAGE21) {
      // Never emit a GOT reloc, we handled this in
      // RewriteInstance::readRelocations().
      return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_PAGE, Ctx);
    } else if (Inst.getOpcode() == AArch64::ADRP) {
      return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_ABS_PAGE, Ctx);
    } else {
      switch(RelType) {
      case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
      case ELF::R_AARCH64_LD64_GOT_LO12_NC:
      case ELF::R_AARCH64_TLSDESC_LD64_LO12:
      case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
      case ELF::R_AARCH64_ADD_ABS_LO12_NC:
        return AArch64MCExpr::create(Expr, AArch64MCExpr::VK_LO12, Ctx);
      default:
        break;
      }
    }
    return Expr;
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *AArchExpr = dyn_cast<AArch64MCExpr>(Expr);
    if (AArchExpr && AArchExpr->getSubExpr()) {
      return getTargetSymbol(AArchExpr->getSubExpr());
    }

    auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr);
    if (!SymExpr || SymExpr->getKind() != MCSymbolRefExpr::VK_None)
      return nullptr;

    return &SymExpr->getSymbol();
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (OpNum >= MCPlus::getNumPrimeOperands(Inst))
      return nullptr;

    // Auto-select correct operand number
    if (OpNum == 0) {
      if (isConditionalBranch(Inst))
        OpNum = 1;
      if (isTB(Inst))
        OpNum = 2;
      if (isMOVW(Inst))
        OpNum = 1;
    }

    auto &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
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

    OI = Inst.erase(OI);
    Inst.insert(OI, MCOperand::createExpr(MCSymbolRefExpr::create(
                        TBB, MCSymbolRefExpr::VK_None, *Ctx)));
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
    auto &UsesRoot = UDChain[&Inst];
    if (UsesRoot.size() == 0 || UsesRoot[0] == nullptr) {
      return false;
    }
    const auto *DefAdd = UsesRoot[0];

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
    auto OperandExtension = DefAdd->getOperand(3).getImm();
    auto ShiftVal = AArch64_AM::getArithShiftValue(OperandExtension);
    auto ExtendType = AArch64_AM::getArithExtendType(OperandExtension);
    if (ShiftVal != 2) {
      llvm_unreachable("Failed to match indirect branch! (fragment 2)");
    }
    if (ExtendType == AArch64_AM::SXTB) {
      ScaleValue = 1LL;
    } else if (ExtendType == AArch64_AM::SXTH) {
      ScaleValue = 2LL;
    } else if (ExtendType == AArch64_AM::SXTW) {
      ScaleValue = 4LL;
    } else {
      llvm_unreachable("Failed to match indirect branch! (fragment 3)");
    }

    // Match an ADR to load base address to be used when addressing JT targets
    auto &UsesAdd = UDChain[DefAdd];
    if (UsesAdd.size() <= 1 || UsesAdd[1] == nullptr || UsesAdd[2] == nullptr) {
      // This happens when we don't have enough context about this jump table
      // because the jumping code sequence was split in multiple basic blocks.
      // This was observed in the wild in HHVM code (dispatchImpl).
      return false;
    }
    auto *DefBaseAddr = UsesAdd[1];
    assert(DefBaseAddr->getOpcode() == AArch64::ADR &&
           "Failed to match indirect branch pattern! (fragment 3)");

    PCRelBase = DefBaseAddr;
    // Match LOAD to load the jump table (relative) target
    const auto *DefLoad = UsesAdd[2];
    assert(isLoad(*DefLoad) &&
           "Failed to match indirect branch load pattern! (1)");
    assert((ScaleValue != 1LL || isLDRB(*DefLoad)) &&
           "Failed to match indirect branch load pattern! (2)");
    assert((ScaleValue != 2LL || isLDRH(*DefLoad)) &&
           "Failed to match indirect branch load pattern! (3)");

    // Match ADD that calculates the JumpTable Base Address (not the offset)
    auto &UsesLoad = UDChain[DefLoad];
    const auto *DefJTBaseAdd = UsesLoad[1];
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
    auto &UsesJTBaseAdd = UDChain[DefJTBaseAdd];
    const auto *DefJTBasePage = UsesJTBaseAdd[1];
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
  computeLocalUDChain(const MCInst *CurInstr,
                      InstructionIterator Begin,
                      InstructionIterator End) const {
    DenseMap<int, MCInst *> RegAliasTable;
    DenseMap<const MCInst *, SmallVector<MCInst *, 4>> Uses;

    auto addInstrOperands = [&](const MCInst &Instr) {
      // Update Uses table
      for (unsigned OpNum = 0, OpEnd = MCPlus::getNumPrimeOperands(Instr);
           OpNum != OpEnd; ++OpNum) {
        if (!Instr.getOperand(OpNum).isReg())
          continue;
        Uses[&Instr].push_back(RegAliasTable[Instr.getOperand(OpNum).getReg()]);
        DEBUG({
          dbgs() << "Adding reg operand " << Instr.getOperand(OpNum).getReg()
                 << " refs ";
          if (RegAliasTable[Instr.getOperand(OpNum).getReg()] != nullptr)
            RegAliasTable[Instr.getOperand(OpNum).getReg()]->dump();
          else
            dbgs() << "\n";
        });
      }
    };

    DEBUG(dbgs() << "computeLocalUDChain\n");
    bool TerminatorSeen = false;
    for (auto II = Begin; II != End; ++II) {
      auto &Instr = *II;
      // Ignore nops and CFIs
      if (Info->get(Instr.getOpcode()).isPseudo() || isNoop(Instr))
        continue;
      if (TerminatorSeen) {
        RegAliasTable.clear();
        Uses.clear();
      }

      DEBUG(dbgs() << "Now updating for:\n ");
      DEBUG(Instr.dump());
      addInstrOperands(Instr);

      BitVector Regs = BitVector(RegInfo->getNumRegs(), false);
      getWrittenRegs(Instr, Regs);

      // Update register definitions after this point
      int Idx = Regs.find_first();
      while (Idx != -1) {
        RegAliasTable[Idx] = &Instr;
        DEBUG(dbgs() << "Setting reg " << Idx << " def to current instr.\n");
        Idx = Regs.find_next(Idx);
      }

      TerminatorSeen = isTerminator(Instr);
    }

    // Process the last instruction, which is not currently added into the
    // instruction stream
    if (CurInstr) {
      addInstrOperands(*CurInstr);
    }
    return Uses;
  }

  IndirectBranchType analyzeIndirectBranch(
     MCInst &Instruction,
     InstructionIterator Begin,
     InstructionIterator End,
     const unsigned PtrSize,
     MCInst *&MemLocInstrOut,
     unsigned &BaseRegNumOut,
     unsigned &IndexRegNumOut,
     int64_t &DispValueOut,
     const MCExpr *&DispExprOut,
     MCInst *&PCRelBaseOut
  ) const override {
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
    int64_t       ScaleValue, DispValue;
    const MCExpr *DispExpr;

    auto UDChain = computeLocalUDChain(&Instruction, Begin, End);
    MCInst *PCRelBase;
    if (!analyzeIndirectBranchFragment(Instruction, UDChain, DispExpr,
                                       DispValue, ScaleValue, PCRelBase)) {
      return IndirectBranchType::UNKNOWN;
    }

    MemLocInstrOut = MemLocInstr;
    DispValueOut = DispValue;
    DispExprOut = DispExpr;
    PCRelBaseOut = PCRelBase;
    return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
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

  unsigned getCanonicalBranchOpcode(unsigned Opcode) const override {
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
      DEBUG(Inst.dump());
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

  int getShortJmpEncodingSize() const override {
    return 32;
  }

  int getUncondBranchEncodingSize() const override {
    return 28;
  }

  bool isTailCall(const MCInst &Inst) const override {
    auto IsTCOrErr = tryGetAnnotationAs<bool>(Inst, "TC");
    if (IsTCOrErr)
      return *IsTCOrErr;
    if (getConditionalTailCall(Inst))
      return true;
    return false;
  }

  bool createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    Inst.setOpcode(AArch64::B);
    Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
        Inst, MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        *Ctx, 0)));
    addAnnotation(Inst, "TC", true);
    return true;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    addAnnotation(Inst, "TC", true);
    return true;
  }

  bool convertTailCallToJmp(MCInst &Inst) override {
    removeAnnotation(Inst, "TC");
    removeAnnotation(Inst, "Offset");
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, "TC");
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

  bool isStore(const MCInst &Inst) const override {
    return false;
  }

  bool analyzeBranch(InstructionIterator Begin,
                     InstructionIterator End,
                     const MCSymbol *&TBB,
                     const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (Info->get(I->getOpcode()).isPseudo() || isNoop(*I))
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
        const auto *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I)) {
        return false;
      }

      if (CondBranch == nullptr) {
        const auto *TargetBB = getTargetSymbol(*I);
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

  void createLongJmp(std::vector<MCInst> &Seq, const MCSymbol *Target,
                     MCContext *Ctx) const override {
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
    Seq.emplace_back(Inst);
  }

  void createShortJmp(std::vector<MCInst> &Seq, const MCSymbol *Target,
                      MCContext *Ctx) const override {
    // ip0 (r16) is reserved to the linker (refer to 5.3.1.1 of "Procedure Call
    //   Standard for the ARM 64-bit Architecture (AArch64)".
    // The sequence of instructions we create here is the following:
    //  movz ip0, #:abs_g1_nc:<addr>
    //  movk ip0, #:abs_g0_nc:<addr>
    //  br ip0
    MCInst Inst;
    Inst.setOpcode(AArch64::MOVZXi);
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
    Seq.emplace_back(Inst);
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
    if (I == Begin ||
        I->getOpcode() != AArch64::ADDXri ||
        MCPlus::getNumPrimeOperands(*I) < 3 ||
        !I->getOperand(0).isReg() ||
        !I->getOperand(1).isReg() ||
        I->getOperand(0).getReg() != AArch64::X16 ||
        I->getOperand(1).getReg() != AArch64::X16 ||
        !I->getOperand(2).isImm())
      return false;
    TargetLowBits = &*I;
    uint64_t Addr = I->getOperand(2).getImm() & 0xFFF;

    --I;
    Address -= 4;
    if (I->getOpcode() != AArch64::ADRP ||
        MCPlus::getNumPrimeOperands(*I) < 2 ||
        !I->getOperand(0).isReg() ||
        !I->getOperand(1).isImm() ||
        I->getOperand(0).getReg() != AArch64::X16)
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

  bool isMoveMem2Reg(const MCInst &Inst) const override {
    return false;
  }

  bool isADD64rr(const MCInst &Inst) const override {
    return false;
  }

  bool isLeave(const MCInst &Inst) const override {
    return false;
  }

  bool isPop(const MCInst &Inst) const override {
    return false;
  }

  bool isPrefix(const MCInst &Inst) const override {
    return false;
  }

  bool deleteREPPrefix(MCInst &Inst) const override {
    return false;
  }

  bool createReturn(MCInst &Inst) const override {
    Inst.setOpcode(AArch64::RET);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(AArch64::LR));
    return true;
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

}
}
