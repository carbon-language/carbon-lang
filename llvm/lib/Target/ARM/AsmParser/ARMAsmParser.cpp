//===-- ARMAsmParser.cpp - Parse ARM assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMMCExpr.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace {

class ARMOperand;

class ARMAsmParser : public TargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  int TryParseRegister();
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);
  bool TryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &);
  int TryParseShiftRegister(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool ParseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &,
                   ARMII::AddrMode AddrMode);
  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &, StringRef Mnemonic);
  bool ParsePrefix(ARMMCExpr::VariantKind &RefKind);
  const MCExpr *ApplyPrefixToExpr(const MCExpr *E,
                                  MCSymbolRefExpr::VariantKind Variant);


  bool ParseMemoryOffsetReg(bool &Negative,
                            bool &OffsetRegShifted,
                            enum ARM_AM::ShiftOpc &ShiftType,
                            const MCExpr *&ShiftAmount,
                            const MCExpr *&Offset,
                            bool &OffsetIsReg,
                            int &OffsetRegNum,
                            SMLoc &E);
  bool ParseShift(enum ARM_AM::ShiftOpc &St,
                  const MCExpr *&ShiftAmount, SMLoc &E);
  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveThumb(SMLoc L);
  bool ParseDirectiveThumbFunc(SMLoc L);
  bool ParseDirectiveCode(SMLoc L);
  bool ParseDirectiveSyntax(SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);
  void GetMnemonicAcceptInfo(StringRef Mnemonic, bool &CanAcceptCarrySet,
                             bool &CanAcceptPredicationCode);

  bool isThumb() const {
    // FIXME: Can tablegen auto-generate this?
    return (STI.getFeatureBits() & ARM::ModeThumb) != 0;
  }
  bool isThumbOne() const {
    return isThumb() && (STI.getFeatureBits() & ARM::FeatureThumb2) == 0;
  }
  void SwitchMode() {
    unsigned FB = ComputeAvailableFeatures(STI.ToggleFeature(ARM::ModeThumb));
    setAvailableFeatures(FB);
  }

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "ARMGenAsmMatcher.inc"

  /// }

  OperandMatchResultTy tryParseCoprocNumOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseCoprocRegOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseMemBarrierOptOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseProcIFlagsOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseMSRMaskOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseMemMode2Operand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy tryParseMemMode3Operand(
    SmallVectorImpl<MCParsedAsmOperand*>&);

  // Asm Match Converter Methods
  bool CvtLdWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool CvtStWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool CvtLdWriteBackRegAddrMode3(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool CvtStWriteBackRegAddrMode3(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);

public:
  ARMAsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser)
    : TargetAsmParser(), STI(_STI), Parser(_Parser) {
    MCAsmParserExtension::Initialize(_Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  virtual bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);
  virtual bool ParseDirective(AsmToken DirectiveID);
};
} // end anonymous namespace

namespace {

/// ARMOperand - Instances of this class represent a parsed ARM machine
/// instruction.
class ARMOperand : public MCParsedAsmOperand {
  enum KindTy {
    CondCode,
    CCOut,
    CoprocNum,
    CoprocReg,
    Immediate,
    MemBarrierOpt,
    Memory,
    MSRMask,
    ProcIFlags,
    Register,
    RegisterList,
    DPRRegisterList,
    SPRRegisterList,
    ShiftedRegister,
    Shifter,
    Token
  } Kind;

  SMLoc StartLoc, EndLoc;
  SmallVector<unsigned, 8> Registers;

  union {
    struct {
      ARMCC::CondCodes Val;
    } CC;

    struct {
      ARM_MB::MemBOpt Val;
    } MBOpt;

    struct {
      unsigned Val;
    } Cop;

    struct {
      ARM_PROC::IFlags Val;
    } IFlags;

    struct {
      unsigned Val;
    } MMask;

    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNum;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;

    /// Combined record for all forms of ARM address expressions.
    struct {
      ARMII::AddrMode AddrMode;
      unsigned BaseRegNum;
      union {
        unsigned RegNum;     ///< Offset register num, when OffsetIsReg.
        const MCExpr *Value; ///< Offset value, when !OffsetIsReg.
      } Offset;
      const MCExpr *ShiftAmount;     // used when OffsetRegShifted is true
      enum ARM_AM::ShiftOpc ShiftType; // used when OffsetRegShifted is true
      unsigned OffsetRegShifted : 1; // only used when OffsetIsReg is true
      unsigned Preindexed       : 1;
      unsigned Postindexed      : 1;
      unsigned OffsetIsReg      : 1;
      unsigned Negative         : 1; // only used when OffsetIsReg is true
      unsigned Writeback        : 1;
    } Mem;

    struct {
      ARM_AM::ShiftOpc ShiftTy;
      unsigned Imm;
    } Shift;
    struct {
      ARM_AM::ShiftOpc ShiftTy;
      unsigned SrcReg;
      unsigned ShiftReg;
      unsigned ShiftImm;
    } ShiftedReg;
  };

  ARMOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}
public:
  ARMOperand(const ARMOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case CondCode:
      CC = o.CC;
      break;
    case Token:
      Tok = o.Tok;
      break;
    case CCOut:
    case Register:
      Reg = o.Reg;
      break;
    case RegisterList:
    case DPRRegisterList:
    case SPRRegisterList:
      Registers = o.Registers;
      break;
    case CoprocNum:
    case CoprocReg:
      Cop = o.Cop;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case MemBarrierOpt:
      MBOpt = o.MBOpt;
      break;
    case Memory:
      Mem = o.Mem;
      break;
    case MSRMask:
      MMask = o.MMask;
      break;
    case ProcIFlags:
      IFlags = o.IFlags;
      break;
    case Shifter:
      Shift = o.Shift;
      break;
    case ShiftedRegister:
      ShiftedReg = o.ShiftedReg;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  ARMCC::CondCodes getCondCode() const {
    assert(Kind == CondCode && "Invalid access!");
    return CC.Val;
  }

  unsigned getCoproc() const {
    assert((Kind == CoprocNum || Kind == CoprocReg) && "Invalid access!");
    return Cop.Val;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == Register || Kind == CCOut) && "Invalid access!");
    return Reg.RegNum;
  }

  const SmallVectorImpl<unsigned> &getRegList() const {
    assert((Kind == RegisterList || Kind == DPRRegisterList ||
            Kind == SPRRegisterList) && "Invalid access!");
    return Registers;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  ARM_MB::MemBOpt getMemBarrierOpt() const {
    assert(Kind == MemBarrierOpt && "Invalid access!");
    return MBOpt.Val;
  }

  ARM_PROC::IFlags getProcIFlags() const {
    assert(Kind == ProcIFlags && "Invalid access!");
    return IFlags.Val;
  }

  unsigned getMSRMask() const {
    assert(Kind == MSRMask && "Invalid access!");
    return MMask.Val;
  }

  /// @name Memory Operand Accessors
  /// @{
  ARMII::AddrMode getMemAddrMode() const {
    return Mem.AddrMode;
  }
  unsigned getMemBaseRegNum() const {
    return Mem.BaseRegNum;
  }
  unsigned getMemOffsetRegNum() const {
    assert(Mem.OffsetIsReg && "Invalid access!");
    return Mem.Offset.RegNum;
  }
  const MCExpr *getMemOffset() const {
    assert(!Mem.OffsetIsReg && "Invalid access!");
    return Mem.Offset.Value;
  }
  unsigned getMemOffsetRegShifted() const {
    assert(Mem.OffsetIsReg && "Invalid access!");
    return Mem.OffsetRegShifted;
  }
  const MCExpr *getMemShiftAmount() const {
    assert(Mem.OffsetIsReg && Mem.OffsetRegShifted && "Invalid access!");
    return Mem.ShiftAmount;
  }
  enum ARM_AM::ShiftOpc getMemShiftType() const {
    assert(Mem.OffsetIsReg && Mem.OffsetRegShifted && "Invalid access!");
    return Mem.ShiftType;
  }
  bool getMemPreindexed() const { return Mem.Preindexed; }
  bool getMemPostindexed() const { return Mem.Postindexed; }
  bool getMemOffsetIsReg() const { return Mem.OffsetIsReg; }
  bool getMemNegative() const { return Mem.Negative; }
  bool getMemWriteback() const { return Mem.Writeback; }

  /// @}

  bool isCoprocNum() const { return Kind == CoprocNum; }
  bool isCoprocReg() const { return Kind == CoprocReg; }
  bool isCondCode() const { return Kind == CondCode; }
  bool isCCOut() const { return Kind == CCOut; }
  bool isImm() const { return Kind == Immediate; }
  bool isImm0_255() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 256;
  }
  bool isImm0_7() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 8;
  }
  bool isImm0_15() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 16;
  }
  bool isImm0_65535() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 65536;
  }
  bool isImm0_65535Expr() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    // If it's not a constant expression, it'll generate a fixup and be
    // handled later.
    if (!CE) return true;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 65536;
  }
  bool isARMSOImm() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return ARM_AM::getSOImmVal(Value) != -1;
  }
  bool isT2SOImm() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return ARM_AM::getT2SOImmVal(Value) != -1;
  }
  bool isReg() const { return Kind == Register; }
  bool isRegList() const { return Kind == RegisterList; }
  bool isDPRRegList() const { return Kind == DPRRegisterList; }
  bool isSPRRegList() const { return Kind == SPRRegisterList; }
  bool isToken() const { return Kind == Token; }
  bool isMemBarrierOpt() const { return Kind == MemBarrierOpt; }
  bool isMemory() const { return Kind == Memory; }
  bool isShifter() const { return Kind == Shifter; }
  bool isShiftedReg() const { return Kind == ShiftedRegister; }
  bool isMemMode2() const {
    if (getMemAddrMode() != ARMII::AddrMode2)
      return false;

    if (getMemOffsetIsReg())
      return true;

    if (getMemNegative() &&
        !(getMemPostindexed() || getMemPreindexed()))
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    if (!CE) return false;
    int64_t Value = CE->getValue();

    // The offset must be in the range 0-4095 (imm12).
    if (Value > 4095 || Value < -4095)
      return false;

    return true;
  }
  bool isMemMode3() const {
    if (getMemAddrMode() != ARMII::AddrMode3)
      return false;

    if (getMemOffsetIsReg()) {
      if (getMemOffsetRegShifted())
        return false; // No shift with offset reg allowed
      return true;
    }

    if (getMemNegative() &&
        !(getMemPostindexed() || getMemPreindexed()))
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    if (!CE) return false;
    int64_t Value = CE->getValue();

    // The offset must be in the range 0-255 (imm8).
    if (Value > 255 || Value < -255)
      return false;

    return true;
  }
  bool isMemMode5() const {
    if (!isMemory() || getMemOffsetIsReg() || getMemWriteback() ||
        getMemNegative())
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    if (!CE) return false;

    // The offset must be a multiple of 4 in the range 0-1020.
    int64_t Value = CE->getValue();
    return ((Value & 0x3) == 0 && Value <= 1020 && Value >= -1020);
  }
  bool isMemMode7() const {
    if (!isMemory() ||
        getMemPreindexed() ||
        getMemPostindexed() ||
        getMemOffsetIsReg() ||
        getMemNegative() ||
        getMemWriteback())
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    if (!CE) return false;

    if (CE->getValue())
      return false;

    return true;
  }
  bool isMemModeRegThumb() const {
    if (!isMemory() || !getMemOffsetIsReg() || getMemWriteback())
      return false;
    return true;
  }
  bool isMemModeImmThumb() const {
    if (!isMemory() || getMemOffsetIsReg() || getMemWriteback())
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    if (!CE) return false;

    // The offset must be a multiple of 4 in the range 0-124.
    uint64_t Value = CE->getValue();
    return ((Value & 0x3) == 0 && Value <= 124);
  }
  bool isMSRMask() const { return Kind == MSRMask; }
  bool isProcIFlags() const { return Kind == ProcIFlags; }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addCondCodeOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getCondCode())));
    unsigned RegNum = getCondCode() == ARMCC::AL ? 0: ARM::CPSR;
    Inst.addOperand(MCOperand::CreateReg(RegNum));
  }

  void addCoprocNumOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getCoproc()));
  }

  void addCoprocRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getCoproc()));
  }

  void addCCOutOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addShiftedRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");
    assert(isShiftedReg() && "addShiftedRegOperands() on non ShiftedReg!");
    assert((ShiftedReg.ShiftReg == 0 ||
            ARM_AM::getSORegOffset(ShiftedReg.ShiftImm) == 0) &&
           "Invalid shifted register operand!");
    Inst.addOperand(MCOperand::CreateReg(ShiftedReg.SrcReg));
    Inst.addOperand(MCOperand::CreateReg(ShiftedReg.ShiftReg));
    Inst.addOperand(MCOperand::CreateImm(
      ARM_AM::getSORegOpc(ShiftedReg.ShiftTy, ShiftedReg.ShiftImm)));
  }

  void addShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(
      ARM_AM::getSORegOpc(Shift.ShiftTy, 0)));
  }

  void addRegListOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const SmallVectorImpl<unsigned> &RegList = getRegList();
    for (SmallVectorImpl<unsigned>::const_iterator
           I = RegList.begin(), E = RegList.end(); I != E; ++I)
      Inst.addOperand(MCOperand::CreateReg(*I));
  }

  void addDPRRegListOperands(MCInst &Inst, unsigned N) const {
    addRegListOperands(Inst, N);
  }

  void addSPRRegListOperands(MCInst &Inst, unsigned N) const {
    addRegListOperands(Inst, N);
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_255Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_7Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_15Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_65535Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_65535ExprOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addARMSOImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addT2SOImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addMemBarrierOptOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getMemBarrierOpt())));
  }

  void addMemMode7Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isMemMode7() && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    (void)CE;
    assert((CE || CE->getValue() == 0) &&
           "No offset operand support in mode 7");
  }

  void addMemMode2Operands(MCInst &Inst, unsigned N) const {
    assert(isMemMode2() && "Invalid mode or number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));
    unsigned IdxMode = (getMemPreindexed() | getMemPostindexed() << 1);

    if (getMemOffsetIsReg()) {
      Inst.addOperand(MCOperand::CreateReg(getMemOffsetRegNum()));

      ARM_AM::AddrOpc AMOpc = getMemNegative() ? ARM_AM::sub : ARM_AM::add;
      ARM_AM::ShiftOpc ShOpc = ARM_AM::no_shift;
      int64_t ShiftAmount = 0;

      if (getMemOffsetRegShifted()) {
        ShOpc = getMemShiftType();
        const MCConstantExpr *CE =
                   dyn_cast<MCConstantExpr>(getMemShiftAmount());
        ShiftAmount = CE->getValue();
      }

      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM2Opc(AMOpc, ShiftAmount,
                                           ShOpc, IdxMode)));
      return;
    }

    // Create a operand placeholder to always yield the same number of operands.
    Inst.addOperand(MCOperand::CreateReg(0));

    // FIXME: #-0 is encoded differently than #0. Does the parser preserve
    // the difference?
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    assert(CE && "Non-constant mode 2 offset operand!");
    int64_t Offset = CE->getValue();

    if (Offset >= 0)
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM2Opc(ARM_AM::add,
                                           Offset, ARM_AM::no_shift, IdxMode)));
    else
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM2Opc(ARM_AM::sub,
                                          -Offset, ARM_AM::no_shift, IdxMode)));
  }

  void addMemMode3Operands(MCInst &Inst, unsigned N) const {
    assert(isMemMode3() && "Invalid mode or number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));
    unsigned IdxMode = (getMemPreindexed() | getMemPostindexed() << 1);

    if (getMemOffsetIsReg()) {
      Inst.addOperand(MCOperand::CreateReg(getMemOffsetRegNum()));

      ARM_AM::AddrOpc AMOpc = getMemNegative() ? ARM_AM::sub : ARM_AM::add;
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM3Opc(AMOpc, 0,
                                                             IdxMode)));
      return;
    }

    // Create a operand placeholder to always yield the same number of operands.
    Inst.addOperand(MCOperand::CreateReg(0));

    // FIXME: #-0 is encoded differently than #0. Does the parser preserve
    // the difference?
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    assert(CE && "Non-constant mode 3 offset operand!");
    int64_t Offset = CE->getValue();

    if (Offset >= 0)
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM3Opc(ARM_AM::add,
                                           Offset, IdxMode)));
    else
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM3Opc(ARM_AM::sub,
                                           -Offset, IdxMode)));
  }

  void addMemMode5Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemMode5() && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));
    assert(!getMemOffsetIsReg() && "Invalid mode 5 operand");

    // FIXME: #-0 is encoded differently than #0. Does the parser preserve
    // the difference?
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    assert(CE && "Non-constant mode 5 offset operand!");

    // The MCInst offset operand doesn't include the low two bits (like
    // the instruction encoding).
    int64_t Offset = CE->getValue() / 4;
    if (Offset >= 0)
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM5Opc(ARM_AM::add,
                                                             Offset)));
    else
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM5Opc(ARM_AM::sub,
                                                             -Offset)));
  }

  void addMemModeRegThumbOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemModeRegThumb() && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));
    Inst.addOperand(MCOperand::CreateReg(getMemOffsetRegNum()));
  }

  void addMemModeImmThumbOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemModeImmThumb() && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseRegNum()));
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemOffset());
    assert(CE && "Non-constant mode offset operand!");
    Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
  }

  void addMSRMaskOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getMSRMask())));
  }

  void addProcIFlagsOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getProcIFlags())));
  }

  virtual void print(raw_ostream &OS) const;

  static ARMOperand *CreateCondCode(ARMCC::CondCodes CC, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CondCode);
    Op->CC.Val = CC;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateCoprocNum(unsigned CopVal, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CoprocNum);
    Op->Cop.Val = CopVal;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateCoprocReg(unsigned CopVal, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CoprocReg);
    Op->Cop.Val = CopVal;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateCCOut(unsigned RegNum, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CCOut);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateToken(StringRef Str, SMLoc S) {
    ARMOperand *Op = new ARMOperand(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateShiftedRegister(ARM_AM::ShiftOpc ShTy,
                                           unsigned SrcReg,
                                           unsigned ShiftReg,
                                           unsigned ShiftImm,
                                           SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(ShiftedRegister);
    Op->ShiftedReg.ShiftTy = ShTy;
    Op->ShiftedReg.SrcReg = SrcReg;
    Op->ShiftedReg.ShiftReg = ShiftReg;
    Op->ShiftedReg.ShiftImm = ShiftImm;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateShifter(ARM_AM::ShiftOpc ShTy,
                                   SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Shifter);
    Op->Shift.ShiftTy = ShTy;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *
  CreateRegList(const SmallVectorImpl<std::pair<unsigned, SMLoc> > &Regs,
                SMLoc StartLoc, SMLoc EndLoc) {
    KindTy Kind = RegisterList;

    if (ARM::DPRRegClass.contains(Regs.front().first))
      Kind = DPRRegisterList;
    else if (ARM::SPRRegClass.contains(Regs.front().first))
      Kind = SPRRegisterList;

    ARMOperand *Op = new ARMOperand(Kind);
    for (SmallVectorImpl<std::pair<unsigned, SMLoc> >::const_iterator
           I = Regs.begin(), E = Regs.end(); I != E; ++I)
      Op->Registers.push_back(I->first);
    array_pod_sort(Op->Registers.begin(), Op->Registers.end());
    Op->StartLoc = StartLoc;
    Op->EndLoc = EndLoc;
    return Op;
  }

  static ARMOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateMem(ARMII::AddrMode AddrMode, unsigned BaseRegNum,
                               bool OffsetIsReg, const MCExpr *Offset,
                               int OffsetRegNum, bool OffsetRegShifted,
                               enum ARM_AM::ShiftOpc ShiftType,
                               const MCExpr *ShiftAmount, bool Preindexed,
                               bool Postindexed, bool Negative, bool Writeback,
                               SMLoc S, SMLoc E) {
    assert((OffsetRegNum == -1 || OffsetIsReg) &&
           "OffsetRegNum must imply OffsetIsReg!");
    assert((!OffsetRegShifted || OffsetIsReg) &&
           "OffsetRegShifted must imply OffsetIsReg!");
    assert((Offset || OffsetIsReg) &&
           "Offset must exists unless register offset is used!");
    assert((!ShiftAmount || (OffsetIsReg && OffsetRegShifted)) &&
           "Cannot have shift amount without shifted register offset!");
    assert((!Offset || !OffsetIsReg) &&
           "Cannot have expression offset and register offset!");

    ARMOperand *Op = new ARMOperand(Memory);
    Op->Mem.AddrMode = AddrMode;
    Op->Mem.BaseRegNum = BaseRegNum;
    Op->Mem.OffsetIsReg = OffsetIsReg;
    if (OffsetIsReg)
      Op->Mem.Offset.RegNum = OffsetRegNum;
    else
      Op->Mem.Offset.Value = Offset;
    Op->Mem.OffsetRegShifted = OffsetRegShifted;
    Op->Mem.ShiftType = ShiftType;
    Op->Mem.ShiftAmount = ShiftAmount;
    Op->Mem.Preindexed = Preindexed;
    Op->Mem.Postindexed = Postindexed;
    Op->Mem.Negative = Negative;
    Op->Mem.Writeback = Writeback;

    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateMemBarrierOpt(ARM_MB::MemBOpt Opt, SMLoc S) {
    ARMOperand *Op = new ARMOperand(MemBarrierOpt);
    Op->MBOpt.Val = Opt;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateProcIFlags(ARM_PROC::IFlags IFlags, SMLoc S) {
    ARMOperand *Op = new ARMOperand(ProcIFlags);
    Op->IFlags.Val = IFlags;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateMSRMask(unsigned MMask, SMLoc S) {
    ARMOperand *Op = new ARMOperand(MSRMask);
    Op->MMask.Val = MMask;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }
};

} // end anonymous namespace.

void ARMOperand::print(raw_ostream &OS) const {
  switch (Kind) {
  case CondCode:
    OS << "<ARMCC::" << ARMCondCodeToString(getCondCode()) << ">";
    break;
  case CCOut:
    OS << "<ccout " << getReg() << ">";
    break;
  case CoprocNum:
    OS << "<coprocessor number: " << getCoproc() << ">";
    break;
  case CoprocReg:
    OS << "<coprocessor register: " << getCoproc() << ">";
    break;
  case MSRMask:
    OS << "<mask: " << getMSRMask() << ">";
    break;
  case Immediate:
    getImm()->print(OS);
    break;
  case MemBarrierOpt:
    OS << "<ARM_MB::" << MemBOptToString(getMemBarrierOpt()) << ">";
    break;
  case Memory:
    OS << "<memory "
       << "am:" << ARMII::AddrModeToString(getMemAddrMode())
       << " base:" << getMemBaseRegNum();
    if (getMemOffsetIsReg()) {
      OS << " offset:<register " << getMemOffsetRegNum();
      if (getMemOffsetRegShifted()) {
        OS << " offset-shift-type:" << getMemShiftType();
        OS << " offset-shift-amount:" << *getMemShiftAmount();
      }
    } else {
      OS << " offset:" << *getMemOffset();
    }
    if (getMemOffsetIsReg())
      OS << " (offset-is-reg)";
    if (getMemPreindexed())
      OS << " (pre-indexed)";
    if (getMemPostindexed())
      OS << " (post-indexed)";
    if (getMemNegative())
      OS << " (negative)";
    if (getMemWriteback())
      OS << " (writeback)";
    OS << ">";
    break;
  case ProcIFlags: {
    OS << "<ARM_PROC::";
    unsigned IFlags = getProcIFlags();
    for (int i=2; i >= 0; --i)
      if (IFlags & (1 << i))
        OS << ARM_PROC::IFlagsToString(1 << i);
    OS << ">";
    break;
  }
  case Register:
    OS << "<register " << getReg() << ">";
    break;
  case Shifter:
    OS << "<shifter " << ARM_AM::getShiftOpcStr(Shift.ShiftTy) << ">";
    break;
  case ShiftedRegister:
    OS << "<so_reg"
       << ShiftedReg.SrcReg
       << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(ShiftedReg.ShiftImm))
       << ", " << ShiftedReg.ShiftReg << ", "
       << ARM_AM::getSORegOffset(ShiftedReg.ShiftImm)
       << ">";
    break;
  case RegisterList:
  case DPRRegisterList:
  case SPRRegisterList: {
    OS << "<register_list ";

    const SmallVectorImpl<unsigned> &RegList = getRegList();
    for (SmallVectorImpl<unsigned>::const_iterator
           I = RegList.begin(), E = RegList.end(); I != E; ) {
      OS << *I;
      if (++I < E) OS << ", ";
    }

    OS << ">";
    break;
  }
  case Token:
    OS << "'" << getToken() << "'";
    break;
  }
}

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

bool ARMAsmParser::ParseRegister(unsigned &RegNo,
                                 SMLoc &StartLoc, SMLoc &EndLoc) {
  RegNo = TryParseRegister();

  return (RegNo == (unsigned)-1);
}

/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name the token is eaten and the register number is
/// returned.  Otherwise return -1.
///
int ARMAsmParser::TryParseRegister() {
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  std::string upperCase = Tok.getString().str();
  std::string lowerCase = LowercaseString(upperCase);
  unsigned RegNum = MatchRegisterName(lowerCase);
  if (!RegNum) {
    RegNum = StringSwitch<unsigned>(lowerCase)
      .Case("r13", ARM::SP)
      .Case("r14", ARM::LR)
      .Case("r15", ARM::PC)
      .Case("ip", ARM::R12)
      .Default(0);
  }
  if (!RegNum) return -1;

  Parser.Lex(); // Eat identifier token.
  return RegNum;
}

// Try to parse a shifter  (e.g., "lsl <amt>"). On success, return 0.
// If a recoverable error occurs, return 1. If an irrecoverable error
// occurs, return -1. An irrecoverable error is one where tokens have been
// consumed in the process of trying to parse the shifter (i.e., when it is
// indeed a shifter operand, but malformed).
int ARMAsmParser::TryParseShiftRegister(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  std::string upperCase = Tok.getString().str();
  std::string lowerCase = LowercaseString(upperCase);
  ARM_AM::ShiftOpc ShiftTy = StringSwitch<ARM_AM::ShiftOpc>(lowerCase)
      .Case("lsl", ARM_AM::lsl)
      .Case("lsr", ARM_AM::lsr)
      .Case("asr", ARM_AM::asr)
      .Case("ror", ARM_AM::ror)
      .Case("rrx", ARM_AM::rrx)
      .Default(ARM_AM::no_shift);

  if (ShiftTy == ARM_AM::no_shift)
    return 1;

  Parser.Lex(); // Eat the operator.

  // The source register for the shift has already been added to the
  // operand list, so we need to pop it off and combine it into the shifted
  // register operand instead.
  OwningPtr<ARMOperand> PrevOp((ARMOperand*)Operands.pop_back_val());
  if (!PrevOp->isReg())
    return Error(PrevOp->getStartLoc(), "shift must be of a register");
  int SrcReg = PrevOp->getReg();
  int64_t Imm = 0;
  int ShiftReg = 0;
  if (ShiftTy == ARM_AM::rrx) {
    // RRX Doesn't have an explicit shift amount. The encoder expects
    // the shift register to be the same as the source register. Seems odd,
    // but OK.
    ShiftReg = SrcReg;
  } else {
    // Figure out if this is shifted by a constant or a register (for non-RRX).
    if (Parser.getTok().is(AsmToken::Hash)) {
      Parser.Lex(); // Eat hash.
      SMLoc ImmLoc = Parser.getTok().getLoc();
      const MCExpr *ShiftExpr = 0;
      if (getParser().ParseExpression(ShiftExpr)) {
        Error(ImmLoc, "invalid immediate shift value");
        return -1;
      }
      // The expression must be evaluatable as an immediate.
      const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ShiftExpr);
      if (!CE) {
        Error(ImmLoc, "invalid immediate shift value");
        return -1;
      }
      // Range check the immediate.
      // lsl, ror: 0 <= imm <= 31
      // lsr, asr: 0 <= imm <= 32
      Imm = CE->getValue();
      if (Imm < 0 ||
          ((ShiftTy == ARM_AM::lsl || ShiftTy == ARM_AM::ror) && Imm > 31) ||
          ((ShiftTy == ARM_AM::lsr || ShiftTy == ARM_AM::asr) && Imm > 32)) {
        Error(ImmLoc, "immediate shift value out of range");
        return -1;
      }
    } else if (Parser.getTok().is(AsmToken::Identifier)) {
      ShiftReg = TryParseRegister();
      SMLoc L = Parser.getTok().getLoc();
      if (ShiftReg == -1) {
        Error (L, "expected immediate or register in shift operand");
        return -1;
      }
    } else {
      Error (Parser.getTok().getLoc(),
                    "expected immediate or register in shift operand");
      return -1;
    }
  }

  Operands.push_back(ARMOperand::CreateShiftedRegister(ShiftTy, SrcReg,
                                                       ShiftReg, Imm,
                                               S, Parser.getTok().getLoc()));

  return 0;
}


/// Try to parse a register name.  The token must be an Identifier when called.
/// If it's a register, an AsmOperand is created. Another AsmOperand is created
/// if there is a "writeback". 'true' if it's not a register.
///
/// TODO this is likely to change to allow different register types and or to
/// parse for a specific register type.
bool ARMAsmParser::
TryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  int RegNo = TryParseRegister();
  if (RegNo == -1)
    return true;

  Operands.push_back(ARMOperand::CreateReg(RegNo, S, Parser.getTok().getLoc()));

  const AsmToken &ExclaimTok = Parser.getTok();
  if (ExclaimTok.is(AsmToken::Exclaim)) {
    Operands.push_back(ARMOperand::CreateToken(ExclaimTok.getString(),
                                               ExclaimTok.getLoc()));
    Parser.Lex(); // Eat exclaim token
  }

  return false;
}

/// MatchCoprocessorOperandName - Try to parse an coprocessor related
/// instruction with a symbolic operand name. Example: "p1", "p7", "c3",
/// "c5", ...
static int MatchCoprocessorOperandName(StringRef Name, char CoprocOp) {
  // Use the same layout as the tablegen'erated register name matcher. Ugly,
  // but efficient.
  switch (Name.size()) {
  default: break;
  case 2:
    if (Name[0] != CoprocOp)
      return -1;
    switch (Name[1]) {
    default:  return -1;
    case '0': return 0;
    case '1': return 1;
    case '2': return 2;
    case '3': return 3;
    case '4': return 4;
    case '5': return 5;
    case '6': return 6;
    case '7': return 7;
    case '8': return 8;
    case '9': return 9;
    }
    break;
  case 3:
    if (Name[0] != CoprocOp || Name[1] != '1')
      return -1;
    switch (Name[2]) {
    default:  return -1;
    case '0': return 10;
    case '1': return 11;
    case '2': return 12;
    case '3': return 13;
    case '4': return 14;
    case '5': return 15;
    }
    break;
  }

  return -1;
}

/// tryParseCoprocNumOperand - Try to parse an coprocessor number operand. The
/// token must be an Identifier when called, and if it is a coprocessor
/// number, the token is eaten and the operand is added to the operand list.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseCoprocNumOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  int Num = MatchCoprocessorOperandName(Tok.getString(), 'p');
  if (Num == -1)
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateCoprocNum(Num, S));
  return MatchOperand_Success;
}

/// tryParseCoprocRegOperand - Try to parse an coprocessor register operand. The
/// token must be an Identifier when called, and if it is a coprocessor
/// number, the token is eaten and the operand is added to the operand list.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseCoprocRegOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  int Reg = MatchCoprocessorOperandName(Tok.getString(), 'c');
  if (Reg == -1)
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateCoprocReg(Reg, S));
  return MatchOperand_Success;
}

/// Parse a register list, return it if successful else return null.  The first
/// token must be a '{' when called.
bool ARMAsmParser::
ParseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  assert(Parser.getTok().is(AsmToken::LCurly) &&
         "Token is not a Left Curly Brace");
  SMLoc S = Parser.getTok().getLoc();

  // Read the rest of the registers in the list.
  unsigned PrevRegNum = 0;
  SmallVector<std::pair<unsigned, SMLoc>, 32> Registers;

  do {
    bool IsRange = Parser.getTok().is(AsmToken::Minus);
    Parser.Lex(); // Eat non-identifier token.

    const AsmToken &RegTok = Parser.getTok();
    SMLoc RegLoc = RegTok.getLoc();
    if (RegTok.isNot(AsmToken::Identifier)) {
      Error(RegLoc, "register expected");
      return true;
    }

    int RegNum = TryParseRegister();
    if (RegNum == -1) {
      Error(RegLoc, "register expected");
      return true;
    }

    if (IsRange) {
      int Reg = PrevRegNum;
      do {
        ++Reg;
        Registers.push_back(std::make_pair(Reg, RegLoc));
      } while (Reg != RegNum);
    } else {
      Registers.push_back(std::make_pair(RegNum, RegLoc));
    }

    PrevRegNum = RegNum;
  } while (Parser.getTok().is(AsmToken::Comma) ||
           Parser.getTok().is(AsmToken::Minus));

  // Process the right curly brace of the list.
  const AsmToken &RCurlyTok = Parser.getTok();
  if (RCurlyTok.isNot(AsmToken::RCurly)) {
    Error(RCurlyTok.getLoc(), "'}' expected");
    return true;
  }

  SMLoc E = RCurlyTok.getLoc();
  Parser.Lex(); // Eat right curly brace token.

  // Verify the register list.
  SmallVectorImpl<std::pair<unsigned, SMLoc> >::const_iterator
    RI = Registers.begin(), RE = Registers.end();

  unsigned HighRegNum = getARMRegisterNumbering(RI->first);
  bool EmittedWarning = false;

  DenseMap<unsigned, bool> RegMap;
  RegMap[HighRegNum] = true;

  for (++RI; RI != RE; ++RI) {
    const std::pair<unsigned, SMLoc> &RegInfo = *RI;
    unsigned Reg = getARMRegisterNumbering(RegInfo.first);

    if (RegMap[Reg]) {
      Error(RegInfo.second, "register duplicated in register list");
      return true;
    }

    if (!EmittedWarning && Reg < HighRegNum)
      Warning(RegInfo.second,
              "register not in ascending order in register list");

    RegMap[Reg] = true;
    HighRegNum = std::max(Reg, HighRegNum);
  }

  Operands.push_back(ARMOperand::CreateRegList(Registers, S, E));
  return false;
}

/// tryParseMemBarrierOptOperand - Try to parse DSB/DMB data barrier options.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseMemBarrierOptOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");
  StringRef OptStr = Tok.getString();

  unsigned Opt = StringSwitch<unsigned>(OptStr.slice(0, OptStr.size()))
    .Case("sy",    ARM_MB::SY)
    .Case("st",    ARM_MB::ST)
    .Case("sh",    ARM_MB::ISH)
    .Case("ish",   ARM_MB::ISH)
    .Case("shst",  ARM_MB::ISHST)
    .Case("ishst", ARM_MB::ISHST)
    .Case("nsh",   ARM_MB::NSH)
    .Case("un",    ARM_MB::NSH)
    .Case("nshst", ARM_MB::NSHST)
    .Case("unst",  ARM_MB::NSHST)
    .Case("osh",   ARM_MB::OSH)
    .Case("oshst", ARM_MB::OSHST)
    .Default(~0U);

  if (Opt == ~0U)
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateMemBarrierOpt((ARM_MB::MemBOpt)Opt, S));
  return MatchOperand_Success;
}

/// tryParseProcIFlagsOperand - Try to parse iflags from CPS instruction.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseProcIFlagsOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");
  StringRef IFlagsStr = Tok.getString();

  unsigned IFlags = 0;
  for (int i = 0, e = IFlagsStr.size(); i != e; ++i) {
    unsigned Flag = StringSwitch<unsigned>(IFlagsStr.substr(i, 1))
    .Case("a", ARM_PROC::A)
    .Case("i", ARM_PROC::I)
    .Case("f", ARM_PROC::F)
    .Default(~0U);

    // If some specific iflag is already set, it means that some letter is
    // present more than once, this is not acceptable.
    if (Flag == ~0U || (IFlags & Flag))
      return MatchOperand_NoMatch;

    IFlags |= Flag;
  }

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateProcIFlags((ARM_PROC::IFlags)IFlags, S));
  return MatchOperand_Success;
}

/// tryParseMSRMaskOperand - Try to parse mask flags from MSR instruction.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseMSRMaskOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");
  StringRef Mask = Tok.getString();

  // Split spec_reg from flag, example: CPSR_sxf => "CPSR" and "sxf"
  size_t Start = 0, Next = Mask.find('_');
  StringRef Flags = "";
  StringRef SpecReg = Mask.slice(Start, Next);
  if (Next != StringRef::npos)
    Flags = Mask.slice(Next+1, Mask.size());

  // FlagsVal contains the complete mask:
  // 3-0: Mask
  // 4: Special Reg (cpsr, apsr => 0; spsr => 1)
  unsigned FlagsVal = 0;

  if (SpecReg == "apsr") {
    FlagsVal = StringSwitch<unsigned>(Flags)
    .Case("nzcvq",  0x8) // same as CPSR_c
    .Case("g",      0x4) // same as CPSR_s
    .Case("nzcvqg", 0xc) // same as CPSR_fs
    .Default(~0U);

    if (FlagsVal == ~0U) {
      if (!Flags.empty())
        return MatchOperand_NoMatch;
      else
        FlagsVal = 0; // No flag
    }
  } else if (SpecReg == "cpsr" || SpecReg == "spsr") {
    if (Flags == "all") // cpsr_all is an alias for cpsr_fc
      Flags = "fc";
    for (int i = 0, e = Flags.size(); i != e; ++i) {
      unsigned Flag = StringSwitch<unsigned>(Flags.substr(i, 1))
      .Case("c", 1)
      .Case("x", 2)
      .Case("s", 4)
      .Case("f", 8)
      .Default(~0U);

      // If some specific flag is already set, it means that some letter is
      // present more than once, this is not acceptable.
      if (FlagsVal == ~0U || (FlagsVal & Flag))
        return MatchOperand_NoMatch;
      FlagsVal |= Flag;
    }
  } else // No match for special register.
    return MatchOperand_NoMatch;

  // Special register without flags are equivalent to "fc" flags.
  if (!FlagsVal)
    FlagsVal = 0x9;

  // Bit 4: Special Reg (cpsr, apsr => 0; spsr => 1)
  if (SpecReg == "spsr")
    FlagsVal |= 16;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateMSRMask(FlagsVal, S));
  return MatchOperand_Success;
}

/// tryParseMemMode2Operand - Try to parse memory addressing mode 2 operand.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseMemMode2Operand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  assert(Parser.getTok().is(AsmToken::LBrac) && "Token is not a \"[\"");

  if (ParseMemory(Operands, ARMII::AddrMode2))
    return MatchOperand_NoMatch;

  return MatchOperand_Success;
}

/// tryParseMemMode3Operand - Try to parse memory addressing mode 3 operand.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
tryParseMemMode3Operand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  assert(Parser.getTok().is(AsmToken::LBrac) && "Token is not a \"[\"");

  if (ParseMemory(Operands, ARMII::AddrMode3))
    return MatchOperand_NoMatch;

  return MatchOperand_Success;
}

/// CvtLdWriteBackRegAddrMode2 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
CvtLdWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);

  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));

  ((ARMOperand*)Operands[3])->addMemMode2Operands(Inst, 3);
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// CvtStWriteBackRegAddrMode2 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
CvtStWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  ((ARMOperand*)Operands[3])->addMemMode2Operands(Inst, 3);
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// CvtLdWriteBackRegAddrMode3 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
CvtLdWriteBackRegAddrMode3(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);

  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));

  ((ARMOperand*)Operands[3])->addMemMode3Operands(Inst, 3);
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// CvtStWriteBackRegAddrMode3 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
CvtStWriteBackRegAddrMode3(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  ((ARMOperand*)Operands[3])->addMemMode3Operands(Inst, 3);
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// Parse an ARM memory expression, return false if successful else return true
/// or an error.  The first token must be a '[' when called.
///
/// TODO Only preindexing and postindexing addressing are started, unindexed
/// with option, etc are still to do.
bool ARMAsmParser::
ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
            ARMII::AddrMode AddrMode = ARMII::AddrModeNone) {
  SMLoc S, E;
  assert(Parser.getTok().is(AsmToken::LBrac) &&
         "Token is not a Left Bracket");
  S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier)) {
    Error(BaseRegTok.getLoc(), "register expected");
    return true;
  }
  int BaseRegNum = TryParseRegister();
  if (BaseRegNum == -1) {
    Error(BaseRegTok.getLoc(), "register expected");
    return true;
  }

  // The next token must either be a comma or a closing bracket.
  const AsmToken &Tok = Parser.getTok();
  if (!Tok.is(AsmToken::Comma) && !Tok.is(AsmToken::RBrac))
    return true;

  bool Preindexed = false;
  bool Postindexed = false;
  bool OffsetIsReg = false;
  bool Negative = false;
  bool Writeback = false;
  ARMOperand *WBOp = 0;
  int OffsetRegNum = -1;
  bool OffsetRegShifted = false;
  enum ARM_AM::ShiftOpc ShiftType = ARM_AM::lsl;
  const MCExpr *ShiftAmount = 0;
  const MCExpr *Offset = 0;

  // First look for preindexed address forms, that is after the "[Rn" we now
  // have to see if the next token is a comma.
  if (Tok.is(AsmToken::Comma)) {
    Preindexed = true;
    Parser.Lex(); // Eat comma token.

    if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType, ShiftAmount,
                             Offset, OffsetIsReg, OffsetRegNum, E))
      return true;
    const AsmToken &RBracTok = Parser.getTok();
    if (RBracTok.isNot(AsmToken::RBrac)) {
      Error(RBracTok.getLoc(), "']' expected");
      return true;
    }
    E = RBracTok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &ExclaimTok = Parser.getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      // None of addrmode3 instruction uses "!"
      if (AddrMode == ARMII::AddrMode3)
        return true;

      WBOp = ARMOperand::CreateToken(ExclaimTok.getString(),
                                     ExclaimTok.getLoc());
      Writeback = true;
      Parser.Lex(); // Eat exclaim token
    } else { // In addressing mode 2, pre-indexed mode always end with "!"
      if (AddrMode == ARMII::AddrMode2)
        Preindexed = false;
    }
  } else {
    // The "[Rn" we have so far was not followed by a comma.

    // If there's anything other than the right brace, this is a post indexing
    // addressing form.
    E = Tok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &NextTok = Parser.getTok();

    if (NextTok.isNot(AsmToken::EndOfStatement)) {
      Postindexed = true;
      Writeback = true;

      if (NextTok.isNot(AsmToken::Comma)) {
        Error(NextTok.getLoc(), "',' expected");
        return true;
      }

      Parser.Lex(); // Eat comma token.

      if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType,
                               ShiftAmount, Offset, OffsetIsReg, OffsetRegNum,
                               E))
        return true;
    }
  }

  // Force Offset to exist if used.
  if (!OffsetIsReg) {
    if (!Offset)
      Offset = MCConstantExpr::Create(0, getContext());
  } else {
    if (AddrMode == ARMII::AddrMode3 && OffsetRegShifted) {
      Error(E, "shift amount not supported");
      return true;
    }
  }

  Operands.push_back(ARMOperand::CreateMem(AddrMode, BaseRegNum, OffsetIsReg,
                                     Offset, OffsetRegNum, OffsetRegShifted,
                                     ShiftType, ShiftAmount, Preindexed,
                                     Postindexed, Negative, Writeback, S, E));
  if (WBOp)
    Operands.push_back(WBOp);

  return false;
}

/// Parse the offset of a memory operand after we have seen "[Rn," or "[Rn],"
/// we will parse the following (were +/- means that a plus or minus is
/// optional):
///   +/-Rm
///   +/-Rm, shift
///   #offset
/// we return false on success or an error otherwise.
bool ARMAsmParser::ParseMemoryOffsetReg(bool &Negative,
                                        bool &OffsetRegShifted,
                                        enum ARM_AM::ShiftOpc &ShiftType,
                                        const MCExpr *&ShiftAmount,
                                        const MCExpr *&Offset,
                                        bool &OffsetIsReg,
                                        int &OffsetRegNum,
                                        SMLoc &E) {
  Negative = false;
  OffsetRegShifted = false;
  OffsetIsReg = false;
  OffsetRegNum = -1;
  const AsmToken &NextTok = Parser.getTok();
  E = NextTok.getLoc();
  if (NextTok.is(AsmToken::Plus))
    Parser.Lex(); // Eat plus token.
  else if (NextTok.is(AsmToken::Minus)) {
    Negative = true;
    Parser.Lex(); // Eat minus token
  }
  // See if there is a register following the "[Rn," or "[Rn]," we have so far.
  const AsmToken &OffsetRegTok = Parser.getTok();
  if (OffsetRegTok.is(AsmToken::Identifier)) {
    SMLoc CurLoc = OffsetRegTok.getLoc();
    OffsetRegNum = TryParseRegister();
    if (OffsetRegNum != -1) {
      OffsetIsReg = true;
      E = CurLoc;
    }
  }

  // If we parsed a register as the offset then there can be a shift after that.
  if (OffsetRegNum != -1) {
    // Look for a comma then a shift
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Comma)) {
      Parser.Lex(); // Eat comma token.

      const AsmToken &Tok = Parser.getTok();
      if (ParseShift(ShiftType, ShiftAmount, E))
        return Error(Tok.getLoc(), "shift expected");
      OffsetRegShifted = true;
    }
  }
  else { // the "[Rn," or "[Rn,]" we have so far was not followed by "Rm"
    // Look for #offset following the "[Rn," or "[Rn],"
    const AsmToken &HashTok = Parser.getTok();
    if (HashTok.isNot(AsmToken::Hash))
      return Error(HashTok.getLoc(), "'#' expected");

    Parser.Lex(); // Eat hash token.

    if (getParser().ParseExpression(Offset))
     return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  }
  return false;
}

/// ParseShift as one of these two:
///   ( lsl | lsr | asr | ror ) , # shift_amount
///   rrx
/// and returns true if it parses a shift otherwise it returns false.
bool ARMAsmParser::ParseShift(ARM_AM::ShiftOpc &St,
                              const MCExpr *&ShiftAmount, SMLoc &E) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return true;
  StringRef ShiftName = Tok.getString();
  if (ShiftName == "lsl" || ShiftName == "LSL")
    St = ARM_AM::lsl;
  else if (ShiftName == "lsr" || ShiftName == "LSR")
    St = ARM_AM::lsr;
  else if (ShiftName == "asr" || ShiftName == "ASR")
    St = ARM_AM::asr;
  else if (ShiftName == "ror" || ShiftName == "ROR")
    St = ARM_AM::ror;
  else if (ShiftName == "rrx" || ShiftName == "RRX")
    St = ARM_AM::rrx;
  else
    return true;
  Parser.Lex(); // Eat shift type token.

  // Rrx stands alone.
  if (St == ARM_AM::rrx)
    return false;

  // Otherwise, there must be a '#' and a shift amount.
  const AsmToken &HashTok = Parser.getTok();
  if (HashTok.isNot(AsmToken::Hash))
    return Error(HashTok.getLoc(), "'#' expected");
  Parser.Lex(); // Eat hash token.

  if (getParser().ParseExpression(ShiftAmount))
    return true;

  return false;
}

/// Parse a arm instruction operand.  For now this parses the operand regardless
/// of the mnemonic.
bool ARMAsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                StringRef Mnemonic) {
  SMLoc S, E;

  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);
  if (ResTy == MatchOperand_Success)
    return false;
  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_ParseFail)
    return true;

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Identifier: {
    if (!TryParseRegisterWithWriteBack(Operands))
      return false;
    int Res = TryParseShiftRegister(Operands);
    if (Res == 0) // success
      return false;
    else if (Res == -1) // irrecoverable error
      return true;

    // Fall though for the Identifier case that is not a register or a
    // special name.
  }
  case AsmToken::Integer: // things like 1f and 2b as a branch targets
  case AsmToken::Dot: {   // . as a branch target
    // This was not a register so parse other operands that start with an
    // identifier (like labels) as expressions and create them as immediates.
    const MCExpr *IdVal;
    S = Parser.getTok().getLoc();
    if (getParser().ParseExpression(IdVal))
      return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(IdVal, S, E));
    return false;
  }
  case AsmToken::LBrac:
    return ParseMemory(Operands);
  case AsmToken::LCurly:
    return ParseRegisterList(Operands);
  case AsmToken::Hash:
    // #42 -> immediate.
    // TODO: ":lower16:" and ":upper16:" modifiers after # before immediate
    S = Parser.getTok().getLoc();
    Parser.Lex();
    const MCExpr *ImmVal;
    if (getParser().ParseExpression(ImmVal))
      return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(ImmVal, S, E));
    return false;
  case AsmToken::Colon: {
    // ":lower16:" and ":upper16:" expression prefixes
    // FIXME: Check it's an expression prefix,
    // e.g. (FOO - :lower16:BAR) isn't legal.
    ARMMCExpr::VariantKind RefKind;
    if (ParsePrefix(RefKind))
      return true;

    const MCExpr *SubExprVal;
    if (getParser().ParseExpression(SubExprVal))
      return true;

    const MCExpr *ExprVal = ARMMCExpr::Create(RefKind, SubExprVal,
                                                   getContext());
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(ExprVal, S, E));
    return false;
  }
  }
}

// ParsePrefix - Parse ARM 16-bit relocations expression prefix, i.e.
//  :lower16: and :upper16:.
bool ARMAsmParser::ParsePrefix(ARMMCExpr::VariantKind &RefKind) {
  RefKind = ARMMCExpr::VK_ARM_None;

  // :lower16: and :upper16: modifiers
  assert(getLexer().is(AsmToken::Colon) && "expected a :");
  Parser.Lex(); // Eat ':'

  if (getLexer().isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(), "expected prefix identifier in operand");
    return true;
  }

  StringRef IDVal = Parser.getTok().getIdentifier();
  if (IDVal == "lower16") {
    RefKind = ARMMCExpr::VK_ARM_LO16;
  } else if (IDVal == "upper16") {
    RefKind = ARMMCExpr::VK_ARM_HI16;
  } else {
    Error(Parser.getTok().getLoc(), "unexpected prefix in operand");
    return true;
  }
  Parser.Lex();

  if (getLexer().isNot(AsmToken::Colon)) {
    Error(Parser.getTok().getLoc(), "unexpected token after prefix");
    return true;
  }
  Parser.Lex(); // Eat the last ':'
  return false;
}

const MCExpr *
ARMAsmParser::ApplyPrefixToExpr(const MCExpr *E,
                                MCSymbolRefExpr::VariantKind Variant) {
  // Recurse over the given expression, rebuilding it to apply the given variant
  // to the leftmost symbol.
  if (Variant == MCSymbolRefExpr::VK_None)
    return E;

  switch (E->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle target expr yet");
  case MCExpr::Constant:
    llvm_unreachable("Can't handle lower16/upper16 of constant yet");

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    if (SRE->getKind() != MCSymbolRefExpr::VK_None)
      return 0;

    return MCSymbolRefExpr::Create(&SRE->getSymbol(), Variant, getContext());
  }

  case MCExpr::Unary:
    llvm_unreachable("Can't handle unary expressions yet");

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = ApplyPrefixToExpr(BE->getLHS(), Variant);
    const MCExpr *RHS = BE->getRHS();
    if (!LHS)
      return 0;

    return MCBinaryExpr::Create(BE->getOpcode(), LHS, RHS, getContext());
  }
  }

  assert(0 && "Invalid expression kind!");
  return 0;
}

/// \brief Given a mnemonic, split out possible predication code and carry
/// setting letters to form a canonical mnemonic and flags.
//
// FIXME: Would be nice to autogen this.
static StringRef SplitMnemonic(StringRef Mnemonic,
                               unsigned &PredicationCode,
                               bool &CarrySetting,
                               unsigned &ProcessorIMod) {
  PredicationCode = ARMCC::AL;
  CarrySetting = false;
  ProcessorIMod = 0;

  // Ignore some mnemonics we know aren't predicated forms.
  //
  // FIXME: Would be nice to autogen this.
  if (Mnemonic == "teq"   || Mnemonic == "vceq"  || Mnemonic == "movs"   ||
      Mnemonic == "svc"   || Mnemonic == "mls"   || Mnemonic == "smmls"  ||
      Mnemonic == "vcls"  || Mnemonic == "vmls"  || Mnemonic == "vnmls"  ||
      Mnemonic == "vacge" || Mnemonic == "vcge"  || Mnemonic == "vclt"   ||
      Mnemonic == "vacgt" || Mnemonic == "vcgt"  || Mnemonic == "vcle"   ||
      Mnemonic == "smlal" || Mnemonic == "umaal" || Mnemonic == "umlal"  ||
      Mnemonic == "vabal" || Mnemonic == "vmlal" || Mnemonic == "vpadal" ||
      Mnemonic == "vqdmlal")
    return Mnemonic;

  // First, split out any predication code. Ignore mnemonics we know aren't
  // predicated but do have a carry-set and so weren't caught above.
  if (Mnemonic != "adcs" && Mnemonic != "bics") {
    unsigned CC = StringSwitch<unsigned>(Mnemonic.substr(Mnemonic.size()-2))
      .Case("eq", ARMCC::EQ)
      .Case("ne", ARMCC::NE)
      .Case("hs", ARMCC::HS)
      .Case("cs", ARMCC::HS)
      .Case("lo", ARMCC::LO)
      .Case("cc", ARMCC::LO)
      .Case("mi", ARMCC::MI)
      .Case("pl", ARMCC::PL)
      .Case("vs", ARMCC::VS)
      .Case("vc", ARMCC::VC)
      .Case("hi", ARMCC::HI)
      .Case("ls", ARMCC::LS)
      .Case("ge", ARMCC::GE)
      .Case("lt", ARMCC::LT)
      .Case("gt", ARMCC::GT)
      .Case("le", ARMCC::LE)
      .Case("al", ARMCC::AL)
      .Default(~0U);
    if (CC != ~0U) {
      Mnemonic = Mnemonic.slice(0, Mnemonic.size() - 2);
      PredicationCode = CC;
    }
  }

  // Next, determine if we have a carry setting bit. We explicitly ignore all
  // the instructions we know end in 's'.
  if (Mnemonic.endswith("s") &&
      !(Mnemonic == "asrs" || Mnemonic == "cps" || Mnemonic == "mls" ||
        Mnemonic == "movs" || Mnemonic == "mrs" || Mnemonic == "smmls" ||
        Mnemonic == "vabs" || Mnemonic == "vcls" || Mnemonic == "vmls" ||
        Mnemonic == "vmrs" || Mnemonic == "vnmls" || Mnemonic == "vqabs" ||
        Mnemonic == "vrecps" || Mnemonic == "vrsqrts")) {
    Mnemonic = Mnemonic.slice(0, Mnemonic.size() - 1);
    CarrySetting = true;
  }

  // The "cps" instruction can have a interrupt mode operand which is glued into
  // the mnemonic. Check if this is the case, split it and parse the imod op
  if (Mnemonic.startswith("cps")) {
    // Split out any imod code.
    unsigned IMod =
      StringSwitch<unsigned>(Mnemonic.substr(Mnemonic.size()-2, 2))
      .Case("ie", ARM_PROC::IE)
      .Case("id", ARM_PROC::ID)
      .Default(~0U);
    if (IMod != ~0U) {
      Mnemonic = Mnemonic.slice(0, Mnemonic.size()-2);
      ProcessorIMod = IMod;
    }
  }

  return Mnemonic;
}

/// \brief Given a canonical mnemonic, determine if the instruction ever allows
/// inclusion of carry set or predication code operands.
//
// FIXME: It would be nice to autogen this.
void ARMAsmParser::
GetMnemonicAcceptInfo(StringRef Mnemonic, bool &CanAcceptCarrySet,
                      bool &CanAcceptPredicationCode) {
  if (Mnemonic == "and" || Mnemonic == "lsl" || Mnemonic == "lsr" ||
      Mnemonic == "rrx" || Mnemonic == "ror" || Mnemonic == "sub" ||
      Mnemonic == "smull" || Mnemonic == "add" || Mnemonic == "adc" ||
      Mnemonic == "mul" || Mnemonic == "bic" || Mnemonic == "asr" ||
      Mnemonic == "umlal" || Mnemonic == "orr" || Mnemonic == "mvn" ||
      Mnemonic == "rsb" || Mnemonic == "rsc" || Mnemonic == "orn" ||
      Mnemonic == "sbc" || Mnemonic == "mla" || Mnemonic == "umull" ||
      Mnemonic == "eor" || Mnemonic == "smlal" ||
      (Mnemonic == "mov" && !isThumbOne())) {
    CanAcceptCarrySet = true;
  } else {
    CanAcceptCarrySet = false;
  }

  if (Mnemonic == "cbnz" || Mnemonic == "setend" || Mnemonic == "dmb" ||
      Mnemonic == "cps" || Mnemonic == "mcr2" || Mnemonic == "it" ||
      Mnemonic == "mcrr2" || Mnemonic == "cbz" || Mnemonic == "cdp2" ||
      Mnemonic == "trap" || Mnemonic == "mrc2" || Mnemonic == "mrrc2" ||
      Mnemonic == "dsb" || Mnemonic == "movs" || Mnemonic == "isb" ||
      Mnemonic == "clrex" || Mnemonic.startswith("cps")) {
    CanAcceptPredicationCode = false;
  } else {
    CanAcceptPredicationCode = true;
  }

  if (isThumb())
    if (Mnemonic == "bkpt" || Mnemonic == "mcr" || Mnemonic == "mcrr" ||
        Mnemonic == "mrc" || Mnemonic == "mrrc" || Mnemonic == "cdp")
      CanAcceptPredicationCode = false;
}

/// Parse an arm instruction mnemonic followed by its operands.
bool ARMAsmParser::ParseInstruction(StringRef Name, SMLoc NameLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create the leading tokens for the mnemonic, split by '.' characters.
  size_t Start = 0, Next = Name.find('.');
  StringRef Mnemonic = Name.slice(Start, Next);

  // Split out the predication code and carry setting flag from the mnemonic.
  unsigned PredicationCode;
  unsigned ProcessorIMod;
  bool CarrySetting;
  Mnemonic = SplitMnemonic(Mnemonic, PredicationCode, CarrySetting,
                       ProcessorIMod);

  Operands.push_back(ARMOperand::CreateToken(Mnemonic, NameLoc));

  // FIXME: This is all a pretty gross hack. We should automatically handle
  // optional operands like this via tblgen.

  // Next, add the CCOut and ConditionCode operands, if needed.
  //
  // For mnemonics which can ever incorporate a carry setting bit or predication
  // code, our matching model involves us always generating CCOut and
  // ConditionCode operands to match the mnemonic "as written" and then we let
  // the matcher deal with finding the right instruction or generating an
  // appropriate error.
  bool CanAcceptCarrySet, CanAcceptPredicationCode;
  GetMnemonicAcceptInfo(Mnemonic, CanAcceptCarrySet, CanAcceptPredicationCode);

  // If we had a carry-set on an instruction that can't do that, issue an
  // error.
  if (!CanAcceptCarrySet && CarrySetting) {
    Parser.EatToEndOfStatement();
    return Error(NameLoc, "instruction '" + Mnemonic +
                 "' can not set flags, but 's' suffix specified");
  }

  // Add the carry setting operand, if necessary.
  //
  // FIXME: It would be awesome if we could somehow invent a location such that
  // match errors on this operand would print a nice diagnostic about how the
  // 's' character in the mnemonic resulted in a CCOut operand.
  if (CanAcceptCarrySet)
    Operands.push_back(ARMOperand::CreateCCOut(CarrySetting ? ARM::CPSR : 0,
                                               NameLoc));

  // Add the predication code operand, if necessary.
  if (CanAcceptPredicationCode) {
    Operands.push_back(ARMOperand::CreateCondCode(
                         ARMCC::CondCodes(PredicationCode), NameLoc));
  } else {
    // This mnemonic can't ever accept a predication code, but the user wrote
    // one (or misspelled another mnemonic).

    // FIXME: Issue a nice error.
  }

  // Add the processor imod operand, if necessary.
  if (ProcessorIMod) {
    Operands.push_back(ARMOperand::CreateImm(
          MCConstantExpr::Create(ProcessorIMod, getContext()),
                                 NameLoc, NameLoc));
  } else {
    // This mnemonic can't ever accept a imod, but the user wrote
    // one (or misspelled another mnemonic).

    // FIXME: Issue a nice error.
  }

  // Add the remaining tokens in the mnemonic.
  while (Next != StringRef::npos) {
    Start = Next;
    Next = Name.find('.', Start + 1);
    StringRef ExtraToken = Name.slice(Start, Next);

    Operands.push_back(ARMOperand::CreateToken(ExtraToken, NameLoc));
  }

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Mnemonic)) {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, Mnemonic)) {
        Parser.EatToEndOfStatement();
        return true;
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Parser.EatToEndOfStatement();
    return TokError("unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement


  // The 'mov' mnemonic is special. One variant has a cc_out operand, while
  // another does not. Specifically, the MOVW instruction does not. So we
  // special case it here and remove the defaulted (non-setting) cc_out
  // operand if that's the instruction we're trying to match.
  //
  // We do this post-processing of the explicit operands rather than just
  // conditionally adding the cc_out in the first place because we need
  // to check the type of the parsed immediate operand.
  if (Mnemonic == "mov" && Operands.size() > 4 &&
      !static_cast<ARMOperand*>(Operands[4])->isARMSOImm() &&
      static_cast<ARMOperand*>(Operands[4])->isImm0_65535Expr() &&
      static_cast<ARMOperand*>(Operands[1])->getReg() == 0) {
    ARMOperand *Op = static_cast<ARMOperand*>(Operands[1]);
    Operands.erase(Operands.begin() + 1);
    delete Op;
  }



  return false;
}

bool ARMAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  MCInst Inst;
  unsigned ErrorInfo;
  MatchResultTy MatchResult;
  MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo);
  switch (MatchResult) {
  case Match_Success:
    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((ARMOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_ConversionFail:
    return Error(IDLoc, "unable to convert operands to instruction");
  }

  llvm_unreachable("Implement any new match types added!");
  return true;
}

/// ParseDirective parses the arm specific directives
bool ARMAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(4, DirectiveID.getLoc());
  else if (IDVal == ".thumb")
    return ParseDirectiveThumb(DirectiveID.getLoc());
  else if (IDVal == ".thumb_func")
    return ParseDirectiveThumbFunc(DirectiveID.getLoc());
  else if (IDVal == ".code")
    return ParseDirectiveCode(DirectiveID.getLoc());
  else if (IDVal == ".syntax")
    return ParseDirectiveSyntax(DirectiveID.getLoc());
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool ARMAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().ParseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size, 0/*addrspace*/);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// ParseDirectiveThumb
///  ::= .thumb
bool ARMAsmParser::ParseDirectiveThumb(SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // TODO: set thumb mode
  // TODO: tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// ParseDirectiveThumbFunc
///  ::= .thumbfunc symbol_name
bool ARMAsmParser::ParseDirectiveThumbFunc(SMLoc L) {
  const MCAsmInfo &MAI = getParser().getStreamer().getContext().getAsmInfo();
  bool isMachO = MAI.hasSubsectionsViaSymbols();
  StringRef Name;

  // Darwin asm has function name after .thumb_func direction
  // ELF doesn't
  if (isMachO) {
    const AsmToken &Tok = Parser.getTok();
    if (Tok.isNot(AsmToken::Identifier) && Tok.isNot(AsmToken::String))
      return Error(L, "unexpected token in .thumb_func directive");
    Name = Tok.getString();
    Parser.Lex(); // Consume the identifier token.
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // FIXME: assuming function name will be the line following .thumb_func
  if (!isMachO) {
    Name = Parser.getTok().getString();
  }

  // Mark symbol as a thumb symbol.
  MCSymbol *Func = getParser().getContext().GetOrCreateSymbol(Name);
  getParser().getStreamer().EmitThumbFunc(Func);
  return false;
}

/// ParseDirectiveSyntax
///  ::= .syntax unified | divided
bool ARMAsmParser::ParseDirectiveSyntax(SMLoc L) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return Error(L, "unexpected token in .syntax directive");
  StringRef Mode = Tok.getString();
  if (Mode == "unified" || Mode == "UNIFIED")
    Parser.Lex();
  else if (Mode == "divided" || Mode == "DIVIDED")
    return Error(L, "'.syntax divided' arm asssembly not supported");
  else
    return Error(L, "unrecognized syntax mode in .syntax directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(Parser.getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  // TODO tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// ParseDirectiveCode
///  ::= .code 16 | 32
bool ARMAsmParser::ParseDirectiveCode(SMLoc L) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Integer))
    return Error(L, "unexpected token in .code directive");
  int64_t Val = Parser.getTok().getIntVal();
  if (Val == 16)
    Parser.Lex();
  else if (Val == 32)
    Parser.Lex();
  else
    return Error(L, "invalid operand to .code directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(Parser.getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  if (Val == 16) {
    if (!isThumb())
      SwitchMode();
    getParser().getStreamer().EmitAssemblerFlag(MCAF_Code16);
  } else {
    if (isThumb())
      SwitchMode();
    getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
  }

  return false;
}

extern "C" void LLVMInitializeARMAsmLexer();

/// Force static initialization.
extern "C" void LLVMInitializeARMAsmParser() {
  RegisterAsmParser<ARMAsmParser> X(TheARMTarget);
  RegisterAsmParser<ARMAsmParser> Y(TheThumbTarget);
  LLVMInitializeARMAsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "ARMGenAsmMatcher.inc"
