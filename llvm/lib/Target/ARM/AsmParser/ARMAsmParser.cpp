//===-- ARMAsmParser.cpp - Parse ARM assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace {

class ARMOperand;

class ARMAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  int tryParseRegister();
  bool tryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &);
  int tryParseShiftRegister(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool parseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool parseMemory(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &, StringRef Mnemonic);
  bool parsePrefix(ARMMCExpr::VariantKind &RefKind);
  const MCExpr *applyPrefixToExpr(const MCExpr *E,
                                  MCSymbolRefExpr::VariantKind Variant);


  bool parseMemRegOffsetShift(ARM_AM::ShiftOpc &ShiftType,
                              unsigned &ShiftAmount);
  bool parseDirectiveWord(unsigned Size, SMLoc L);
  bool parseDirectiveThumb(SMLoc L);
  bool parseDirectiveThumbFunc(SMLoc L);
  bool parseDirectiveCode(SMLoc L);
  bool parseDirectiveSyntax(SMLoc L);

  StringRef splitMnemonic(StringRef Mnemonic, unsigned &PredicationCode,
                          bool &CarrySetting, unsigned &ProcessorIMod);
  void getMnemonicAcceptInfo(StringRef Mnemonic, bool &CanAcceptCarrySet,
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

  OperandMatchResultTy parseCoprocNumOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseCoprocRegOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseMemBarrierOptOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseProcIFlagsOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseMSRMaskOperand(
    SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parsePKHImm(SmallVectorImpl<MCParsedAsmOperand*> &O,
                                   StringRef Op, int Low, int High);
  OperandMatchResultTy parsePKHLSLImm(SmallVectorImpl<MCParsedAsmOperand*> &O) {
    return parsePKHImm(O, "lsl", 0, 31);
  }
  OperandMatchResultTy parsePKHASRImm(SmallVectorImpl<MCParsedAsmOperand*> &O) {
    return parsePKHImm(O, "asr", 1, 32);
  }
  OperandMatchResultTy parseSetEndImm(SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseShifterImm(SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseRotImm(SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parseBitfield(SmallVectorImpl<MCParsedAsmOperand*>&);
  OperandMatchResultTy parsePostIdxReg(SmallVectorImpl<MCParsedAsmOperand*>&);

  // Asm Match Converter Methods
  bool cvtLdWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool cvtStWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                                  const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool cvtLdExtTWriteBackImm(MCInst &Inst, unsigned Opcode,
                             const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool cvtLdExtTWriteBackReg(MCInst &Inst, unsigned Opcode,
                             const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool cvtStExtTWriteBackImm(MCInst &Inst, unsigned Opcode,
                             const SmallVectorImpl<MCParsedAsmOperand*> &);
  bool cvtStExtTWriteBackReg(MCInst &Inst, unsigned Opcode,
                             const SmallVectorImpl<MCParsedAsmOperand*> &);

  bool validateInstruction(MCInst &Inst,
                           const SmallVectorImpl<MCParsedAsmOperand*> &Ops);

public:
  ARMAsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser)
    : MCTargetAsmParser(), STI(_STI), Parser(_Parser) {
    MCAsmParserExtension::Initialize(_Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  // Implementation of the MCTargetAsmParser interface:
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);
  bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);
  bool ParseDirective(AsmToken DirectiveID);

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);
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
    PostIndexRegister,
    MSRMask,
    ProcIFlags,
    Register,
    RegisterList,
    DPRRegisterList,
    SPRRegisterList,
    ShiftedRegister,
    ShiftedImmediate,
    ShifterImmediate,
    RotateImmediate,
    BitfieldDescriptor,
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
      unsigned BaseRegNum;
      // Offset is in OffsetReg or OffsetImm. If both are zero, no offset
      // was specified.
      const MCConstantExpr *OffsetImm;  // Offset immediate value
      unsigned OffsetRegNum;    // Offset register num, when OffsetImm == NULL
      ARM_AM::ShiftOpc ShiftType; // Shift type for OffsetReg
      unsigned ShiftValue;      // shift for OffsetReg.
      unsigned isNegative : 1;  // Negated OffsetReg? (~'U' bit)
    } Mem;

    struct {
      unsigned RegNum;
      bool isAdd;
      ARM_AM::ShiftOpc ShiftTy;
      unsigned ShiftImm;
    } PostIdxReg;

    struct {
      bool isASR;
      unsigned Imm;
    } ShifterImm;
    struct {
      ARM_AM::ShiftOpc ShiftTy;
      unsigned SrcReg;
      unsigned ShiftReg;
      unsigned ShiftImm;
    } RegShiftedReg;
    struct {
      ARM_AM::ShiftOpc ShiftTy;
      unsigned SrcReg;
      unsigned ShiftImm;
    } RegShiftedImm;
    struct {
      unsigned Imm;
    } RotImm;
    struct {
      unsigned LSB;
      unsigned Width;
    } Bitfield;
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
    case PostIndexRegister:
      PostIdxReg = o.PostIdxReg;
      break;
    case MSRMask:
      MMask = o.MMask;
      break;
    case ProcIFlags:
      IFlags = o.IFlags;
      break;
    case ShifterImmediate:
      ShifterImm = o.ShifterImm;
      break;
    case ShiftedRegister:
      RegShiftedReg = o.RegShiftedReg;
      break;
    case ShiftedImmediate:
      RegShiftedImm = o.RegShiftedImm;
      break;
    case RotateImmediate:
      RotImm = o.RotImm;
      break;
    case BitfieldDescriptor:
      Bitfield = o.Bitfield;
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
  bool isImm0_31() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 32;
  }
  bool isImm1_16() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value > 0 && Value < 17;
  }
  bool isImm1_32() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value > 0 && Value < 33;
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
  bool isImm24bit() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value <= 0xffffff;
  }
  bool isPKHLSLImm() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < 32;
  }
  bool isPKHASRImm() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value > 0 && Value <= 32;
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
  bool isSetEndImm() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Value = CE->getValue();
    return Value == 1 || Value == 0;
  }
  bool isReg() const { return Kind == Register; }
  bool isRegList() const { return Kind == RegisterList; }
  bool isDPRRegList() const { return Kind == DPRRegisterList; }
  bool isSPRRegList() const { return Kind == SPRRegisterList; }
  bool isToken() const { return Kind == Token; }
  bool isMemBarrierOpt() const { return Kind == MemBarrierOpt; }
  bool isMemory() const { return Kind == Memory; }
  bool isShifterImm() const { return Kind == ShifterImmediate; }
  bool isRegShiftedReg() const { return Kind == ShiftedRegister; }
  bool isRegShiftedImm() const { return Kind == ShiftedImmediate; }
  bool isRotImm() const { return Kind == RotateImmediate; }
  bool isBitfield() const { return Kind == BitfieldDescriptor; }
  bool isPostIdxRegShifted() const { return Kind == PostIndexRegister; }
  bool isPostIdxReg() const {
    return Kind == PostIndexRegister && PostIdxReg.ShiftTy == ARM_AM::no_shift;
  }
  bool isMemNoOffset() const {
    if (Kind != Memory)
      return false;
    // No offset of any kind.
    return Mem.OffsetRegNum == 0 && Mem.OffsetImm == 0;
  }
  bool isAddrMode2() const {
    if (Kind != Memory)
      return false;
    // Check for register offset.
    if (Mem.OffsetRegNum) return true;
    // Immediate offset in range [-4095, 4095].
    if (!Mem.OffsetImm) return true;
    int64_t Val = Mem.OffsetImm->getValue();
    return Val > -4096 && Val < 4096;
  }
  bool isAM2OffsetImm() const {
    if (Kind != Immediate)
      return false;
    // Immediate offset in range [-4095, 4095].
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Val = CE->getValue();
    return Val > -4096 && Val < 4096;
  }
  bool isAddrMode5() const {
    if (Kind != Memory)
      return false;
    // Check for register offset.
    if (Mem.OffsetRegNum) return false;
    // Immediate offset in range [-1020, 1020] and a multiple of 4.
    if (!Mem.OffsetImm) return true;
    int64_t Val = Mem.OffsetImm->getValue();
    return Val >= -1020 && Val <= 1020 && ((Val & 3) == 0);
  }
  bool isMemRegOffset() const {
    if (Kind != Memory || !Mem.OffsetRegNum)
      return false;
    return true;
  }
  bool isMemThumbRR() const {
    // Thumb reg+reg addressing is simple. Just two registers, a base and
    // an offset. No shifts, negations or any other complicating factors.
    if (Kind != Memory || !Mem.OffsetRegNum || Mem.isNegative ||
        Mem.ShiftType != ARM_AM::no_shift)
      return false;
    return true;
  }
  bool isMemImm8Offset() const {
    if (Kind != Memory || Mem.OffsetRegNum != 0)
      return false;
    // Immediate offset in range [-255, 255].
    if (!Mem.OffsetImm) return true;
    int64_t Val = Mem.OffsetImm->getValue();
    return Val > -256 && Val < 256;
  }
  bool isMemImm12Offset() const {
    if (Kind != Memory || Mem.OffsetRegNum != 0)
      return false;
    // Immediate offset in range [-4095, 4095].
    if (!Mem.OffsetImm) return true;
    int64_t Val = Mem.OffsetImm->getValue();
    return Val > -4096 && Val < 4096;
  }
  bool isPostIdxImm8() const {
    if (Kind != Immediate)
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;
    int64_t Val = CE->getValue();
    return Val > -256 && Val < 256;
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

  void addRegShiftedRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");
    assert(isRegShiftedReg() && "addRegShiftedRegOperands() on non RegShiftedReg!");
    Inst.addOperand(MCOperand::CreateReg(RegShiftedReg.SrcReg));
    Inst.addOperand(MCOperand::CreateReg(RegShiftedReg.ShiftReg));
    Inst.addOperand(MCOperand::CreateImm(
      ARM_AM::getSORegOpc(RegShiftedReg.ShiftTy, RegShiftedReg.ShiftImm)));
  }

  void addRegShiftedImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    assert(isRegShiftedImm() && "addRegShiftedImmOperands() on non RegShiftedImm!");
    Inst.addOperand(MCOperand::CreateReg(RegShiftedImm.SrcReg));
    Inst.addOperand(MCOperand::CreateImm(
      ARM_AM::getSORegOpc(RegShiftedImm.ShiftTy, RegShiftedImm.ShiftImm)));
  }


  void addShifterImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm((ShifterImm.isASR << 5) |
                                         ShifterImm.Imm));
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

  void addRotImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // Encoded as val>>3. The printer handles display as 8, 16, 24.
    Inst.addOperand(MCOperand::CreateImm(RotImm.Imm >> 3));
  }

  void addBitfieldOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // Munge the lsb/width into a bitfield mask.
    unsigned lsb = Bitfield.LSB;
    unsigned width = Bitfield.Width;
    // Make a 32-bit mask w/ the referenced bits clear and all other bits set.
    uint32_t Mask = ~(((uint32_t)0xffffffff >> lsb) << (32 - width) >>
                      (32 - (lsb + width)));
    Inst.addOperand(MCOperand::CreateImm(Mask));
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

  void addImm0_31Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm1_16Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // The constant encodes as the immediate-1, and we store in the instruction
    // the bits as encoded, so subtract off one here.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    Inst.addOperand(MCOperand::CreateImm(CE->getValue() - 1));
  }

  void addImm1_32Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // The constant encodes as the immediate-1, and we store in the instruction
    // the bits as encoded, so subtract off one here.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    Inst.addOperand(MCOperand::CreateImm(CE->getValue() - 1));
  }

  void addImm0_65535Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm0_65535ExprOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addImm24bitOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addPKHLSLImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addPKHASRImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // An ASR value of 32 encodes as 0, so that's how we want to add it to
    // the instruction as well.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    int Val = CE->getValue();
    Inst.addOperand(MCOperand::CreateImm(Val == 32 ? 0 : Val));
  }

  void addARMSOImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addT2SOImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addSetEndImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addMemBarrierOptOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getMemBarrierOpt())));
  }

  void addMemNoOffsetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
  }

  void addAddrMode2Operands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");
    int32_t Val = Mem.OffsetImm ? Mem.OffsetImm->getValue() : 0;
    if (!Mem.OffsetRegNum) {
      ARM_AM::AddrOpc AddSub = Val < 0 ? ARM_AM::sub : ARM_AM::add;
      // Special case for #-0
      if (Val == INT32_MIN) Val = 0;
      if (Val < 0) Val = -Val;
      Val = ARM_AM::getAM2Opc(AddSub, Val, ARM_AM::no_shift);
    } else {
      // For register offset, we encode the shift type and negation flag
      // here.
      Val = ARM_AM::getAM2Opc(Mem.isNegative ? ARM_AM::sub : ARM_AM::add,
                              0, Mem.ShiftType);
    }
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateReg(Mem.OffsetRegNum));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addAM2OffsetImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    assert(CE && "non-constant AM2OffsetImm operand!");
    int32_t Val = CE->getValue();
    ARM_AM::AddrOpc AddSub = Val < 0 ? ARM_AM::sub : ARM_AM::add;
    // Special case for #-0
    if (Val == INT32_MIN) Val = 0;
    if (Val < 0) Val = -Val;
    Val = ARM_AM::getAM2Opc(AddSub, Val, ARM_AM::no_shift);
    Inst.addOperand(MCOperand::CreateReg(0));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addAddrMode5Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    // The lower two bits are always zero and as such are not encoded.
    int32_t Val = Mem.OffsetImm ? Mem.OffsetImm->getValue() / 4 : 0;
    ARM_AM::AddrOpc AddSub = Val < 0 ? ARM_AM::sub : ARM_AM::add;
    // Special case for #-0
    if (Val == INT32_MIN) Val = 0;
    if (Val < 0) Val = -Val;
    Val = ARM_AM::getAM5Opc(AddSub, Val);
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addMemImm8OffsetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    int64_t Val = Mem.OffsetImm ? Mem.OffsetImm->getValue() : 0;
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addMemImm12OffsetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    int64_t Val = Mem.OffsetImm ? Mem.OffsetImm->getValue() : 0;
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addMemRegOffsetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");
    unsigned Val = ARM_AM::getAM2Opc(Mem.isNegative ? ARM_AM::sub : ARM_AM::add,
                                     Mem.ShiftValue, Mem.ShiftType);
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateReg(Mem.OffsetRegNum));
    Inst.addOperand(MCOperand::CreateImm(Val));
  }

  void addMemThumbRROperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateReg(Mem.OffsetRegNum));
  }

  void addPostIdxImm8Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    assert(CE && "non-constant post-idx-imm8 operand!");
    int Imm = CE->getValue();
    bool isAdd = Imm >= 0;
    Imm = (Imm < 0 ? -Imm : Imm) | (int)isAdd << 8;
    Inst.addOperand(MCOperand::CreateImm(Imm));
  }

  void addPostIdxRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(PostIdxReg.RegNum));
    Inst.addOperand(MCOperand::CreateImm(PostIdxReg.isAdd));
  }

  void addPostIdxRegShiftedOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(PostIdxReg.RegNum));
    // The sign, shift type, and shift amount are encoded in a single operand
    // using the AM2 encoding helpers.
    ARM_AM::AddrOpc opc = PostIdxReg.isAdd ? ARM_AM::add : ARM_AM::sub;
    unsigned Imm = ARM_AM::getAM2Opc(opc, PostIdxReg.ShiftImm,
                                     PostIdxReg.ShiftTy);
    Inst.addOperand(MCOperand::CreateImm(Imm));
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
    Op->RegShiftedReg.ShiftTy = ShTy;
    Op->RegShiftedReg.SrcReg = SrcReg;
    Op->RegShiftedReg.ShiftReg = ShiftReg;
    Op->RegShiftedReg.ShiftImm = ShiftImm;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateShiftedImmediate(ARM_AM::ShiftOpc ShTy,
                                            unsigned SrcReg,
                                            unsigned ShiftImm,
                                            SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(ShiftedImmediate);
    Op->RegShiftedImm.ShiftTy = ShTy;
    Op->RegShiftedImm.SrcReg = SrcReg;
    Op->RegShiftedImm.ShiftImm = ShiftImm;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateShifterImm(bool isASR, unsigned Imm,
                                   SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(ShifterImmediate);
    Op->ShifterImm.isASR = isASR;
    Op->ShifterImm.Imm = Imm;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateRotImm(unsigned Imm, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(RotateImmediate);
    Op->RotImm.Imm = Imm;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateBitfield(unsigned LSB, unsigned Width,
                                    SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(BitfieldDescriptor);
    Op->Bitfield.LSB = LSB;
    Op->Bitfield.Width = Width;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *
  CreateRegList(const SmallVectorImpl<std::pair<unsigned, SMLoc> > &Regs,
                SMLoc StartLoc, SMLoc EndLoc) {
    KindTy Kind = RegisterList;

    if (llvm::ARMMCRegisterClasses[ARM::DPRRegClassID].
        contains(Regs.front().first))
      Kind = DPRRegisterList;
    else if (llvm::ARMMCRegisterClasses[ARM::SPRRegClassID].
             contains(Regs.front().first))
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

  static ARMOperand *CreateMem(unsigned BaseRegNum,
                               const MCConstantExpr *OffsetImm,
                               unsigned OffsetRegNum,
                               ARM_AM::ShiftOpc ShiftType,
                               unsigned ShiftValue,
                               bool isNegative,
                               SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Memory);
    Op->Mem.BaseRegNum = BaseRegNum;
    Op->Mem.OffsetImm = OffsetImm;
    Op->Mem.OffsetRegNum = OffsetRegNum;
    Op->Mem.ShiftType = ShiftType;
    Op->Mem.ShiftValue = ShiftValue;
    Op->Mem.isNegative = isNegative;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreatePostIdxReg(unsigned RegNum, bool isAdd,
                                      ARM_AM::ShiftOpc ShiftTy,
                                      unsigned ShiftImm,
                                      SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(PostIndexRegister);
    Op->PostIdxReg.RegNum = RegNum;
    Op->PostIdxReg.isAdd = isAdd;
    Op->PostIdxReg.ShiftTy = ShiftTy;
    Op->PostIdxReg.ShiftImm = ShiftImm;
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
       << " base:" << Mem.BaseRegNum;
    OS << ">";
    break;
  case PostIndexRegister:
    OS << "post-idx register " << (PostIdxReg.isAdd ? "" : "-")
       << PostIdxReg.RegNum;
    if (PostIdxReg.ShiftTy != ARM_AM::no_shift)
      OS << ARM_AM::getShiftOpcStr(PostIdxReg.ShiftTy) << " "
         << PostIdxReg.ShiftImm;
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
  case ShifterImmediate:
    OS << "<shift " << (ShifterImm.isASR ? "asr" : "lsl")
       << " #" << ShifterImm.Imm << ">";
    break;
  case ShiftedRegister:
    OS << "<so_reg_reg "
       << RegShiftedReg.SrcReg
       << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(RegShiftedReg.ShiftImm))
       << ", " << RegShiftedReg.ShiftReg << ", "
       << ARM_AM::getSORegOffset(RegShiftedReg.ShiftImm)
       << ">";
    break;
  case ShiftedImmediate:
    OS << "<so_reg_imm "
       << RegShiftedImm.SrcReg
       << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(RegShiftedImm.ShiftImm))
       << ", " << ARM_AM::getSORegOffset(RegShiftedImm.ShiftImm)
       << ">";
    break;
  case RotateImmediate:
    OS << "<ror " << " #" << (RotImm.Imm * 8) << ">";
    break;
  case BitfieldDescriptor:
    OS << "<bitfield " << "lsb: " << Bitfield.LSB
       << ", width: " << Bitfield.Width << ">";
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
  RegNo = tryParseRegister();

  return (RegNo == (unsigned)-1);
}

/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name the token is eaten and the register number is
/// returned.  Otherwise return -1.
///
int ARMAsmParser::tryParseRegister() {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier)) return -1;

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
int ARMAsmParser::tryParseShiftRegister(
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
      ShiftReg = tryParseRegister();
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

  if (ShiftReg && ShiftTy != ARM_AM::rrx)
    Operands.push_back(ARMOperand::CreateShiftedRegister(ShiftTy, SrcReg,
                                                         ShiftReg, Imm,
                                               S, Parser.getTok().getLoc()));
  else
    Operands.push_back(ARMOperand::CreateShiftedImmediate(ShiftTy, SrcReg, Imm,
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
tryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  int RegNo = tryParseRegister();
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

/// parseCoprocNumOperand - Try to parse an coprocessor number operand. The
/// token must be an Identifier when called, and if it is a coprocessor
/// number, the token is eaten and the operand is added to the operand list.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseCoprocNumOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
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

/// parseCoprocRegOperand - Try to parse an coprocessor register operand. The
/// token must be an Identifier when called, and if it is a coprocessor
/// number, the token is eaten and the operand is added to the operand list.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseCoprocRegOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
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
parseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
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

    int RegNum = tryParseRegister();
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

/// parseMemBarrierOptOperand - Try to parse DSB/DMB data barrier options.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseMemBarrierOptOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
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

/// parseProcIFlagsOperand - Try to parse iflags from CPS instruction.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseProcIFlagsOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
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

/// parseMSRMaskOperand - Try to parse mask flags from MSR instruction.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseMSRMaskOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");
  StringRef Mask = Tok.getString();

  // Split spec_reg from flag, example: CPSR_sxf => "CPSR" and "sxf"
  size_t Start = 0, Next = Mask.find('_');
  StringRef Flags = "";
  std::string SpecReg = LowercaseString(Mask.slice(Start, Next));
  if (Next != StringRef::npos)
    Flags = Mask.slice(Next+1, Mask.size());

  // FlagsVal contains the complete mask:
  // 3-0: Mask
  // 4: Special Reg (cpsr, apsr => 0; spsr => 1)
  unsigned FlagsVal = 0;

  if (SpecReg == "apsr") {
    FlagsVal = StringSwitch<unsigned>(Flags)
    .Case("nzcvq",  0x8) // same as CPSR_f
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

ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parsePKHImm(SmallVectorImpl<MCParsedAsmOperand*> &Operands, StringRef Op,
            int Low, int High) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(), Op + " operand expected.");
    return MatchOperand_ParseFail;
  }
  StringRef ShiftName = Tok.getString();
  std::string LowerOp = LowercaseString(Op);
  std::string UpperOp = UppercaseString(Op);
  if (ShiftName != LowerOp && ShiftName != UpperOp) {
    Error(Parser.getTok().getLoc(), Op + " operand expected.");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat shift type token.

  // There must be a '#' and a shift amount.
  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "'#' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.

  const MCExpr *ShiftAmount;
  SMLoc Loc = Parser.getTok().getLoc();
  if (getParser().ParseExpression(ShiftAmount)) {
    Error(Loc, "illegal expression");
    return MatchOperand_ParseFail;
  }
  const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ShiftAmount);
  if (!CE) {
    Error(Loc, "constant expression expected");
    return MatchOperand_ParseFail;
  }
  int Val = CE->getValue();
  if (Val < Low || Val > High) {
    Error(Loc, "immediate value out of range");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(ARMOperand::CreateImm(CE, Loc, Parser.getTok().getLoc()));

  return MatchOperand_Success;
}

ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseSetEndImm(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(Tok.getLoc(), "'be' or 'le' operand expected");
    return MatchOperand_ParseFail;
  }
  int Val = StringSwitch<int>(Tok.getString())
    .Case("be", 1)
    .Case("le", 0)
    .Default(-1);
  Parser.Lex(); // Eat the token.

  if (Val == -1) {
    Error(Tok.getLoc(), "'be' or 'le' operand expected");
    return MatchOperand_ParseFail;
  }
  Operands.push_back(ARMOperand::CreateImm(MCConstantExpr::Create(Val,
                                                                  getContext()),
                                           S, Parser.getTok().getLoc()));
  return MatchOperand_Success;
}

/// parseShifterImm - Parse the shifter immediate operand for SSAT/USAT
/// instructions. Legal values are:
///     lsl #n  'n' in [0,31]
///     asr #n  'n' in [1,32]
///             n == 32 encoded as n == 0.
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseShifterImm(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(S, "shift operator 'asr' or 'lsl' expected");
    return MatchOperand_ParseFail;
  }
  StringRef ShiftName = Tok.getString();
  bool isASR;
  if (ShiftName == "lsl" || ShiftName == "LSL")
    isASR = false;
  else if (ShiftName == "asr" || ShiftName == "ASR")
    isASR = true;
  else {
    Error(S, "shift operator 'asr' or 'lsl' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat the operator.

  // A '#' and a shift amount.
  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "'#' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.

  const MCExpr *ShiftAmount;
  SMLoc E = Parser.getTok().getLoc();
  if (getParser().ParseExpression(ShiftAmount)) {
    Error(E, "malformed shift expression");
    return MatchOperand_ParseFail;
  }
  const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ShiftAmount);
  if (!CE) {
    Error(E, "shift amount must be an immediate");
    return MatchOperand_ParseFail;
  }

  int64_t Val = CE->getValue();
  if (isASR) {
    // Shift amount must be in [1,32]
    if (Val < 1 || Val > 32) {
      Error(E, "'asr' shift amount must be in range [1,32]");
      return MatchOperand_ParseFail;
    }
    // asr #32 encoded as asr #0.
    if (Val == 32) Val = 0;
  } else {
    // Shift amount must be in [1,32]
    if (Val < 0 || Val > 31) {
      Error(E, "'lsr' shift amount must be in range [0,31]");
      return MatchOperand_ParseFail;
    }
  }

  E = Parser.getTok().getLoc();
  Operands.push_back(ARMOperand::CreateShifterImm(isASR, Val, S, E));

  return MatchOperand_Success;
}

/// parseRotImm - Parse the shifter immediate operand for SXTB/UXTB family
/// of instructions. Legal values are:
///     ror #n  'n' in {0, 8, 16, 24}
ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseRotImm(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(S, "rotate operator 'ror' expected");
    return MatchOperand_ParseFail;
  }
  StringRef ShiftName = Tok.getString();
  if (ShiftName != "ror" && ShiftName != "ROR") {
    Error(S, "rotate operator 'ror' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat the operator.

  // A '#' and a rotate amount.
  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "'#' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.

  const MCExpr *ShiftAmount;
  SMLoc E = Parser.getTok().getLoc();
  if (getParser().ParseExpression(ShiftAmount)) {
    Error(E, "malformed rotate expression");
    return MatchOperand_ParseFail;
  }
  const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ShiftAmount);
  if (!CE) {
    Error(E, "rotate amount must be an immediate");
    return MatchOperand_ParseFail;
  }

  int64_t Val = CE->getValue();
  // Shift amount must be in {0, 8, 16, 24} (0 is undocumented extension)
  // normally, zero is represented in asm by omitting the rotate operand
  // entirely.
  if (Val != 8 && Val != 16 && Val != 24 && Val != 0) {
    Error(E, "'ror' rotate amount must be 8, 16, or 24");
    return MatchOperand_ParseFail;
  }

  E = Parser.getTok().getLoc();
  Operands.push_back(ARMOperand::CreateRotImm(Val, S, E));

  return MatchOperand_Success;
}

ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parseBitfield(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  // The bitfield descriptor is really two operands, the LSB and the width.
  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "'#' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.

  const MCExpr *LSBExpr;
  SMLoc E = Parser.getTok().getLoc();
  if (getParser().ParseExpression(LSBExpr)) {
    Error(E, "malformed immediate expression");
    return MatchOperand_ParseFail;
  }
  const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(LSBExpr);
  if (!CE) {
    Error(E, "'lsb' operand must be an immediate");
    return MatchOperand_ParseFail;
  }

  int64_t LSB = CE->getValue();
  // The LSB must be in the range [0,31]
  if (LSB < 0 || LSB > 31) {
    Error(E, "'lsb' operand must be in the range [0,31]");
    return MatchOperand_ParseFail;
  }
  E = Parser.getTok().getLoc();

  // Expect another immediate operand.
  if (Parser.getTok().isNot(AsmToken::Comma)) {
    Error(Parser.getTok().getLoc(), "too few operands");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.
  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "'#' expected");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat hash token.

  const MCExpr *WidthExpr;
  if (getParser().ParseExpression(WidthExpr)) {
    Error(E, "malformed immediate expression");
    return MatchOperand_ParseFail;
  }
  CE = dyn_cast<MCConstantExpr>(WidthExpr);
  if (!CE) {
    Error(E, "'width' operand must be an immediate");
    return MatchOperand_ParseFail;
  }

  int64_t Width = CE->getValue();
  // The LSB must be in the range [1,32-lsb]
  if (Width < 1 || Width > 32 - LSB) {
    Error(E, "'width' operand must be in the range [1,32-lsb]");
    return MatchOperand_ParseFail;
  }
  E = Parser.getTok().getLoc();

  Operands.push_back(ARMOperand::CreateBitfield(LSB, Width, S, E));

  return MatchOperand_Success;
}

ARMAsmParser::OperandMatchResultTy ARMAsmParser::
parsePostIdxReg(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Check for a post-index addressing register operand. Specifically:
  // postidx_reg := '+' register {, shift}
  //              | '-' register {, shift}
  //              | register {, shift}

  // This method must return MatchOperand_NoMatch without consuming any tokens
  // in the case where there is no match, as other alternatives take other
  // parse methods.
  AsmToken Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  bool haveEaten = false;
  bool isAdd = true;
  int Reg = -1;
  if (Tok.is(AsmToken::Plus)) {
    Parser.Lex(); // Eat the '+' token.
    haveEaten = true;
  } else if (Tok.is(AsmToken::Minus)) {
    Parser.Lex(); // Eat the '-' token.
    isAdd = false;
    haveEaten = true;
  }
  if (Parser.getTok().is(AsmToken::Identifier))
    Reg = tryParseRegister();
  if (Reg == -1) {
    if (!haveEaten)
      return MatchOperand_NoMatch;
    Error(Parser.getTok().getLoc(), "register expected");
    return MatchOperand_ParseFail;
  }
  SMLoc E = Parser.getTok().getLoc();

  ARM_AM::ShiftOpc ShiftTy = ARM_AM::no_shift;
  unsigned ShiftImm = 0;

  Operands.push_back(ARMOperand::CreatePostIdxReg(Reg, isAdd, ShiftTy,
                                                  ShiftImm, S, E));

  return MatchOperand_Success;
}

/// cvtLdWriteBackRegAddrMode2 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtLdWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);

  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));

  ((ARMOperand*)Operands[3])->addAddrMode2Operands(Inst, 3);
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// cvtStWriteBackRegAddrMode2 - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtStWriteBackRegAddrMode2(MCInst &Inst, unsigned Opcode,
                         const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  assert(0 && "cvtStWriteBackRegAddrMode2 not implemented yet!");
  return true;
}

/// cvtLdExtTWriteBackImm - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtLdExtTWriteBackImm(MCInst &Inst, unsigned Opcode,
                      const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Rt
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  // addr
  ((ARMOperand*)Operands[3])->addMemNoOffsetOperands(Inst, 1);
  // offset
  ((ARMOperand*)Operands[4])->addPostIdxImm8Operands(Inst, 1);
  // pred
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// cvtLdExtTWriteBackReg - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtLdExtTWriteBackReg(MCInst &Inst, unsigned Opcode,
                      const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Rt
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  // addr
  ((ARMOperand*)Operands[3])->addMemNoOffsetOperands(Inst, 1);
  // offset
  ((ARMOperand*)Operands[4])->addPostIdxRegOperands(Inst, 2);
  // pred
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// cvtStExtTWriteBackImm - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtStExtTWriteBackImm(MCInst &Inst, unsigned Opcode,
                      const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  // Rt
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  // addr
  ((ARMOperand*)Operands[3])->addMemNoOffsetOperands(Inst, 1);
  // offset
  ((ARMOperand*)Operands[4])->addPostIdxImm8Operands(Inst, 1);
  // pred
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// cvtStExtTWriteBackReg - Convert parsed operands to MCInst.
/// Needed here because the Asm Gen Matcher can't handle properly tied operands
/// when they refer multiple MIOperands inside a single one.
bool ARMAsmParser::
cvtStExtTWriteBackReg(MCInst &Inst, unsigned Opcode,
                      const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create a writeback register dummy placeholder.
  Inst.addOperand(MCOperand::CreateImm(0));
  // Rt
  ((ARMOperand*)Operands[2])->addRegOperands(Inst, 1);
  // addr
  ((ARMOperand*)Operands[3])->addMemNoOffsetOperands(Inst, 1);
  // offset
  ((ARMOperand*)Operands[4])->addPostIdxRegOperands(Inst, 2);
  // pred
  ((ARMOperand*)Operands[1])->addCondCodeOperands(Inst, 2);
  return true;
}

/// Parse an ARM memory expression, return false if successful else return true
/// or an error.  The first token must be a '[' when called.
bool ARMAsmParser::
parseMemory(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S, E;
  assert(Parser.getTok().is(AsmToken::LBrac) &&
         "Token is not a Left Bracket");
  S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  int BaseRegNum = tryParseRegister();
  if (BaseRegNum == -1)
    return Error(BaseRegTok.getLoc(), "register expected");

  // The next token must either be a comma or a closing bracket.
  const AsmToken &Tok = Parser.getTok();
  if (!Tok.is(AsmToken::Comma) && !Tok.is(AsmToken::RBrac))
    return Error(Tok.getLoc(), "malformed memory operand");

  if (Tok.is(AsmToken::RBrac)) {
    E = Tok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    Operands.push_back(ARMOperand::CreateMem(BaseRegNum, 0, 0, ARM_AM::no_shift,
                                             0, false, S, E));

    return false;
  }

  assert(Tok.is(AsmToken::Comma) && "Lost comma in memory operand?!");
  Parser.Lex(); // Eat the comma.

  // If we have a '#' it's an immediate offset, else assume it's a register
  // offset.
  if (Parser.getTok().is(AsmToken::Hash)) {
    Parser.Lex(); // Eat the '#'.
    E = Parser.getTok().getLoc();

    // FIXME: Special case #-0 so we can correctly set the U bit.

    const MCExpr *Offset;
    if (getParser().ParseExpression(Offset))
     return true;

    // The expression has to be a constant. Memory references with relocations
    // don't come through here, as they use the <label> forms of the relevant
    // instructions.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Offset);
    if (!CE)
      return Error (E, "constant expression expected");

    // Now we should have the closing ']'
    E = Parser.getTok().getLoc();
    if (Parser.getTok().isNot(AsmToken::RBrac))
      return Error(E, "']' expected");
    Parser.Lex(); // Eat right bracket token.

    // Don't worry about range checking the value here. That's handled by
    // the is*() predicates.
    Operands.push_back(ARMOperand::CreateMem(BaseRegNum, CE, 0,
                                             ARM_AM::no_shift, 0, false, S,E));

    // If there's a pre-indexing writeback marker, '!', just add it as a token
    // operand.
    if (Parser.getTok().is(AsmToken::Exclaim)) {
      Operands.push_back(ARMOperand::CreateToken("!",Parser.getTok().getLoc()));
      Parser.Lex(); // Eat the '!'.
    }

    return false;
  }

  // The register offset is optionally preceded by a '+' or '-'
  bool isNegative = false;
  if (Parser.getTok().is(AsmToken::Minus)) {
    isNegative = true;
    Parser.Lex(); // Eat the '-'.
  } else if (Parser.getTok().is(AsmToken::Plus)) {
    // Nothing to do.
    Parser.Lex(); // Eat the '+'.
  }

  E = Parser.getTok().getLoc();
  int OffsetRegNum = tryParseRegister();
  if (OffsetRegNum == -1)
    return Error(E, "register expected");

  // If there's a shift operator, handle it.
  ARM_AM::ShiftOpc ShiftType = ARM_AM::no_shift;
  unsigned ShiftValue = 0;
  if (Parser.getTok().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the ','.
    if (parseMemRegOffsetShift(ShiftType, ShiftValue))
      return true;
  }

  // Now we should have the closing ']'
  E = Parser.getTok().getLoc();
  if (Parser.getTok().isNot(AsmToken::RBrac))
    return Error(E, "']' expected");
  Parser.Lex(); // Eat right bracket token.

  Operands.push_back(ARMOperand::CreateMem(BaseRegNum, 0, OffsetRegNum,
                                           ShiftType, ShiftValue, isNegative,
                                           S, E));

  // If there's a pre-indexing writeback marker, '!', just add it as a token
  // operand.
  if (Parser.getTok().is(AsmToken::Exclaim)) {
    Operands.push_back(ARMOperand::CreateToken("!",Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the '!'.
  }

  return false;
}

/// parseMemRegOffsetShift - one of these two:
///   ( lsl | lsr | asr | ror ) , # shift_amount
///   rrx
/// return true if it parses a shift otherwise it returns false.
bool ARMAsmParser::parseMemRegOffsetShift(ARM_AM::ShiftOpc &St,
                                          unsigned &Amount) {
  SMLoc Loc = Parser.getTok().getLoc();
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
    return Error(Loc, "illegal shift operator");
  Parser.Lex(); // Eat shift type token.

  // rrx stands alone.
  Amount = 0;
  if (St != ARM_AM::rrx) {
    Loc = Parser.getTok().getLoc();
    // A '#' and a shift amount.
    const AsmToken &HashTok = Parser.getTok();
    if (HashTok.isNot(AsmToken::Hash))
      return Error(HashTok.getLoc(), "'#' expected");
    Parser.Lex(); // Eat hash token.

    const MCExpr *Expr;
    if (getParser().ParseExpression(Expr))
      return true;
    // Range check the immediate.
    // lsl, ror: 0 <= imm <= 31
    // lsr, asr: 0 <= imm <= 32
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr);
    if (!CE)
      return Error(Loc, "shift amount must be an immediate");
    int64_t Imm = CE->getValue();
    if (Imm < 0 ||
        ((St == ARM_AM::lsl || St == ARM_AM::ror) && Imm > 31) ||
        ((St == ARM_AM::lsr || St == ARM_AM::asr) && Imm > 32))
      return Error(Loc, "immediate shift value out of range");
    Amount = Imm;
  }

  return false;
}

/// Parse a arm instruction operand.  For now this parses the operand regardless
/// of the mnemonic.
bool ARMAsmParser::parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
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
    if (!tryParseRegisterWithWriteBack(Operands))
      return false;
    int Res = tryParseShiftRegister(Operands);
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
    return parseMemory(Operands);
  case AsmToken::LCurly:
    return parseRegisterList(Operands);
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
    if (parsePrefix(RefKind))
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

// parsePrefix - Parse ARM 16-bit relocations expression prefix, i.e.
//  :lower16: and :upper16:.
bool ARMAsmParser::parsePrefix(ARMMCExpr::VariantKind &RefKind) {
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
ARMAsmParser::applyPrefixToExpr(const MCExpr *E,
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
    const MCExpr *LHS = applyPrefixToExpr(BE->getLHS(), Variant);
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
StringRef ARMAsmParser::splitMnemonic(StringRef Mnemonic,
                                      unsigned &PredicationCode,
                                      bool &CarrySetting,
                                      unsigned &ProcessorIMod) {
  PredicationCode = ARMCC::AL;
  CarrySetting = false;
  ProcessorIMod = 0;

  // Ignore some mnemonics we know aren't predicated forms.
  //
  // FIXME: Would be nice to autogen this.
  if ((Mnemonic == "movs" && isThumb()) ||
      Mnemonic == "teq"   || Mnemonic == "vceq"   || Mnemonic == "svc"   ||
      Mnemonic == "mls"   || Mnemonic == "smmls"  || Mnemonic == "vcls"  ||
      Mnemonic == "vmls"  || Mnemonic == "vnmls"  || Mnemonic == "vacge" ||
      Mnemonic == "vcge"  || Mnemonic == "vclt"   || Mnemonic == "vacgt" ||
      Mnemonic == "vcgt"  || Mnemonic == "vcle"   || Mnemonic == "smlal" ||
      Mnemonic == "umaal" || Mnemonic == "umlal"  || Mnemonic == "vabal" ||
      Mnemonic == "vmlal" || Mnemonic == "vpadal" || Mnemonic == "vqdmlal")
    return Mnemonic;

  // First, split out any predication code. Ignore mnemonics we know aren't
  // predicated but do have a carry-set and so weren't caught above.
  if (Mnemonic != "adcs" && Mnemonic != "bics" && Mnemonic != "movs" &&
      Mnemonic != "muls" && Mnemonic != "smlals" && Mnemonic != "smulls" &&
      Mnemonic != "umlals" && Mnemonic != "umulls") {
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
        Mnemonic == "mrs" || Mnemonic == "smmls" || Mnemonic == "vabs" ||
        Mnemonic == "vcls" || Mnemonic == "vmls" || Mnemonic == "vmrs" ||
        Mnemonic == "vnmls" || Mnemonic == "vqabs" || Mnemonic == "vrecps" ||
        Mnemonic == "vrsqrts" || Mnemonic == "srs" ||
        (Mnemonic == "movs" && isThumb()))) {
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
getMnemonicAcceptInfo(StringRef Mnemonic, bool &CanAcceptCarrySet,
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
      Mnemonic == "dsb" || Mnemonic == "isb" || Mnemonic == "clrex" ||
      Mnemonic == "setend" ||
      ((Mnemonic == "pld" || Mnemonic == "pli") && !isThumb()) ||
      ((Mnemonic.startswith("rfe") || Mnemonic.startswith("srs"))
        && !isThumb()) ||
      Mnemonic.startswith("cps") || (Mnemonic == "movs" && isThumb())) {
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
  Mnemonic = splitMnemonic(Mnemonic, PredicationCode, CarrySetting,
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
  getMnemonicAcceptInfo(Mnemonic, CanAcceptCarrySet, CanAcceptPredicationCode);

  // If we had a carry-set on an instruction that can't do that, issue an
  // error.
  if (!CanAcceptCarrySet && CarrySetting) {
    Parser.EatToEndOfStatement();
    return Error(NameLoc, "instruction '" + Mnemonic +
                 "' can not set flags, but 's' suffix specified");
  }
  // If we had a predication code on an instruction that can't do that, issue an
  // error.
  if (!CanAcceptPredicationCode && PredicationCode != ARMCC::AL) {
    Parser.EatToEndOfStatement();
    return Error(NameLoc, "instruction '" + Mnemonic +
                 "' is not predicable, but condition code specified");
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
    if (parseOperand(Operands, Mnemonic)) {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (parseOperand(Operands, Mnemonic)) {
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

  // ARM mode 'blx' need special handling, as the register operand version
  // is predicable, but the label operand version is not. So, we can't rely
  // on the Mnemonic based checking to correctly figure out when to put
  // a CondCode operand in the list. If we're trying to match the label
  // version, remove the CondCode operand here.
  if (!isThumb() && Mnemonic == "blx" && Operands.size() == 3 &&
      static_cast<ARMOperand*>(Operands[2])->isImm()) {
    ARMOperand *Op = static_cast<ARMOperand*>(Operands[1]);
    Operands.erase(Operands.begin() + 1);
    delete Op;
  }
  return false;
}

// Validate context-sensitive operand constraints.
// FIXME: We would really like to be able to tablegen'erate this.
bool ARMAsmParser::
validateInstruction(MCInst &Inst,
                    const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  switch (Inst.getOpcode()) {
  case ARM::LDREXD: {
    // Rt2 must be Rt + 1.
    unsigned Rt = getARMRegisterNumbering(Inst.getOperand(0).getReg());
    unsigned Rt2 = getARMRegisterNumbering(Inst.getOperand(1).getReg());
    if (Rt2 != Rt + 1)
      return Error(Operands[3]->getStartLoc(),
                   "destination operands must be sequential");
    return false;
  }
  case ARM::STREXD: {
    // Rt2 must be Rt + 1.
    unsigned Rt = getARMRegisterNumbering(Inst.getOperand(1).getReg());
    unsigned Rt2 = getARMRegisterNumbering(Inst.getOperand(2).getReg());
    if (Rt2 != Rt + 1)
      return Error(Operands[4]->getStartLoc(),
                   "source operands must be sequential");
    return false;
  }
  case ARM::SBFX:
  case ARM::UBFX: {
    // width must be in range [1, 32-lsb]
    unsigned lsb = Inst.getOperand(2).getImm();
    unsigned widthm1 = Inst.getOperand(3).getImm();
    if (widthm1 >= 32 - lsb)
      return Error(Operands[5]->getStartLoc(),
                   "bitfield width must be in range [1,32-lsb]");
  }
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
    // Context sensitive operand constraints aren't handled by the matcher,
    // so check them here.
    if (validateInstruction(Inst, Operands))
      return true;

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

/// parseDirective parses the arm specific directives
bool ARMAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return parseDirectiveWord(4, DirectiveID.getLoc());
  else if (IDVal == ".thumb")
    return parseDirectiveThumb(DirectiveID.getLoc());
  else if (IDVal == ".thumb_func")
    return parseDirectiveThumbFunc(DirectiveID.getLoc());
  else if (IDVal == ".code")
    return parseDirectiveCode(DirectiveID.getLoc());
  else if (IDVal == ".syntax")
    return parseDirectiveSyntax(DirectiveID.getLoc());
  return true;
}

/// parseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool ARMAsmParser::parseDirectiveWord(unsigned Size, SMLoc L) {
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

/// parseDirectiveThumb
///  ::= .thumb
bool ARMAsmParser::parseDirectiveThumb(SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // TODO: set thumb mode
  // TODO: tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// parseDirectiveThumbFunc
///  ::= .thumbfunc symbol_name
bool ARMAsmParser::parseDirectiveThumbFunc(SMLoc L) {
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

/// parseDirectiveSyntax
///  ::= .syntax unified | divided
bool ARMAsmParser::parseDirectiveSyntax(SMLoc L) {
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

/// parseDirectiveCode
///  ::= .code 16 | 32
bool ARMAsmParser::parseDirectiveCode(SMLoc L) {
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
    if (!isThumb()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code16);
    }
  } else {
    if (isThumb()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
    }
  }

  return false;
}

extern "C" void LLVMInitializeARMAsmLexer();

/// Force static initialization.
extern "C" void LLVMInitializeARMAsmParser() {
  RegisterMCAsmParser<ARMAsmParser> X(TheARMTarget);
  RegisterMCAsmParser<ARMAsmParser> Y(TheThumbTarget);
  LLVMInitializeARMAsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "ARMGenAsmMatcher.inc"
