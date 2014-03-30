//===-- ARM64AsmParser.cpp - Parse ARM64 assembly to MCInst instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARM64AddressingModes.h"
#include "MCTargetDesc/ARM64BaseInfo.h"
#include "MCTargetDesc/ARM64MCExpr.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <cstdio>
using namespace llvm;

namespace {

class ARM64Operand;

class ARM64AsmParser : public MCTargetAsmParser {
public:
  typedef SmallVectorImpl<MCParsedAsmOperand *> OperandVector;

private:
  StringRef Mnemonic; ///< Instruction mnemonic.
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  SMLoc getLoc() const { return Parser.getTok().getLoc(); }

  bool parseSysAlias(StringRef Name, SMLoc NameLoc, OperandVector &Operands);
  unsigned parseCondCodeString(StringRef Cond);
  bool parseCondCode(OperandVector &Operands, bool invertCondCode);
  int tryParseRegister();
  int tryMatchVectorRegister(StringRef &Kind);
  bool parseOptionalShift(OperandVector &Operands);
  bool parseOptionalExtend(OperandVector &Operands);
  bool parseRegister(OperandVector &Operands);
  bool parseMemory(OperandVector &Operands);
  bool parseSymbolicImmVal(const MCExpr *&ImmVal);
  bool parseVectorList(OperandVector &Operands);
  bool parseOperand(OperandVector &Operands, bool isCondCode,
                    bool invertCondCode);

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }
  bool showMatchError(SMLoc Loc, unsigned ErrCode);

  bool parseDirectiveWord(unsigned Size, SMLoc L);
  bool parseDirectiveTLSDescCall(SMLoc L);

  bool parseDirectiveLOH(StringRef LOH, SMLoc L);

  bool validateInstruction(MCInst &Inst, SmallVectorImpl<SMLoc> &Loc);
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               unsigned &ErrorInfo, bool MatchingInlineAsm);
/// @name Auto-generated Match Functions
/// {

#define GET_ASSEMBLER_HEADER
#include "ARM64GenAsmMatcher.inc"

  /// }

  OperandMatchResultTy tryParseNoIndexMemory(OperandVector &Operands);
  OperandMatchResultTy tryParseBarrierOperand(OperandVector &Operands);
  OperandMatchResultTy tryParseSystemRegister(OperandVector &Operands);
  OperandMatchResultTy tryParseCPSRField(OperandVector &Operands);
  OperandMatchResultTy tryParseSysCROperand(OperandVector &Operands);
  OperandMatchResultTy tryParsePrefetch(OperandVector &Operands);
  OperandMatchResultTy tryParseAdrpLabel(OperandVector &Operands);
  OperandMatchResultTy tryParseAdrLabel(OperandVector &Operands);
  OperandMatchResultTy tryParseFPImm(OperandVector &Operands);
  bool tryParseVectorRegister(OperandVector &Operands);

public:
  enum ARM64MatchResultTy {
    Match_InvalidSuffix = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "ARM64GenAsmMatcher.inc"
  };
  ARM64AsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser,
                 const MCInstrInfo &MII)
      : MCTargetAsmParser(), STI(_STI), Parser(_Parser) {
    MCAsmParserExtension::Initialize(_Parser);
  }

  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc, OperandVector &Operands);
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);
  virtual bool ParseDirective(AsmToken DirectiveID);
  unsigned validateTargetOperandClass(MCParsedAsmOperand *Op, unsigned Kind);

  static bool classifySymbolRef(const MCExpr *Expr,
                                ARM64MCExpr::VariantKind &ELFRefKind,
                                MCSymbolRefExpr::VariantKind &DarwinRefKind,
                                const MCConstantExpr *&Addend);
};
} // end anonymous namespace

namespace {

/// ARM64Operand - Instances of this class represent a parsed ARM64 machine
/// instruction.
class ARM64Operand : public MCParsedAsmOperand {
public:
  enum MemIdxKindTy {
    ImmediateOffset, // pre-indexed, no writeback
    RegisterOffset   // register offset, with optional extend
  };

private:
  enum KindTy {
    k_Immediate,
    k_Memory,
    k_Register,
    k_VectorList,
    k_VectorIndex,
    k_Token,
    k_SysCR,
    k_Prefetch,
    k_Shifter,
    k_Extend,
    k_FPImm,
    k_Barrier,
    k_SystemRegister,
    k_CPSRField
  } Kind;

  SMLoc StartLoc, EndLoc, OffsetLoc;

  struct TokOp {
    const char *Data;
    unsigned Length;
    bool IsSuffix; // Is the operand actually a suffix on the mnemonic.
  };

  struct RegOp {
    unsigned RegNum;
    bool isVector;
  };

  struct VectorListOp {
    unsigned RegNum;
    unsigned Count;
    unsigned NumElements;
    unsigned ElementKind;
  };

  struct VectorIndexOp {
    unsigned Val;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct FPImmOp {
    unsigned Val; // Encoded 8-bit representation.
  };

  struct BarrierOp {
    unsigned Val; // Not the enum since not all values have names.
  };

  struct SystemRegisterOp {
    // 16-bit immediate, usually from the ARM64SYS::SystermRegister enum,
    // but not limited to those values.
    uint16_t Val;
  };

  struct CPSRFieldOp {
    ARM64SYS::CPSRField Field;
  };

  struct SysCRImmOp {
    unsigned Val;
  };

  struct PrefetchOp {
    unsigned Val;
  };

  struct ShifterOp {
    unsigned Val;
  };

  struct ExtendOp {
    unsigned Val;
  };

  // This is for all forms of ARM64 address expressions
  struct MemOp {
    unsigned BaseRegNum, OffsetRegNum;
    ARM64_AM::ExtendType ExtType;
    unsigned ShiftVal;
    bool ExplicitShift;
    const MCExpr *OffsetImm;
    MemIdxKindTy Mode;
  };

  union {
    struct TokOp Tok;
    struct RegOp Reg;
    struct VectorListOp VectorList;
    struct VectorIndexOp VectorIndex;
    struct ImmOp Imm;
    struct FPImmOp FPImm;
    struct BarrierOp Barrier;
    struct SystemRegisterOp SystemRegister;
    struct CPSRFieldOp CPSRField;
    struct SysCRImmOp SysCRImm;
    struct PrefetchOp Prefetch;
    struct ShifterOp Shifter;
    struct ExtendOp Extend;
    struct MemOp Mem;
  };

  // Keep the MCContext around as the MCExprs may need manipulated during
  // the add<>Operands() calls.
  MCContext &Ctx;

  ARM64Operand(KindTy K, MCContext &_Ctx)
      : MCParsedAsmOperand(), Kind(K), Ctx(_Ctx) {}

public:
  ARM64Operand(const ARM64Operand &o) : MCParsedAsmOperand(), Ctx(o.Ctx) {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case k_Token:
      Tok = o.Tok;
      break;
    case k_Immediate:
      Imm = o.Imm;
      break;
    case k_FPImm:
      FPImm = o.FPImm;
      break;
    case k_Barrier:
      Barrier = o.Barrier;
      break;
    case k_SystemRegister:
      SystemRegister = o.SystemRegister;
      break;
    case k_CPSRField:
      CPSRField = o.CPSRField;
      break;
    case k_Register:
      Reg = o.Reg;
      break;
    case k_VectorList:
      VectorList = o.VectorList;
      break;
    case k_VectorIndex:
      VectorIndex = o.VectorIndex;
      break;
    case k_SysCR:
      SysCRImm = o.SysCRImm;
      break;
    case k_Prefetch:
      Prefetch = o.Prefetch;
      break;
    case k_Memory:
      Mem = o.Mem;
      break;
    case k_Shifter:
      Shifter = o.Shifter;
      break;
    case k_Extend:
      Extend = o.Extend;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }
  /// getOffsetLoc - Get the location of the offset of this memory operand.
  SMLoc getOffsetLoc() const { return OffsetLoc; }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  bool isTokenSuffix() const {
    assert(Kind == k_Token && "Invalid access!");
    return Tok.IsSuffix;
  }

  const MCExpr *getImm() const {
    assert(Kind == k_Immediate && "Invalid access!");
    return Imm.Val;
  }

  unsigned getFPImm() const {
    assert(Kind == k_FPImm && "Invalid access!");
    return FPImm.Val;
  }

  unsigned getBarrier() const {
    assert(Kind == k_Barrier && "Invalid access!");
    return Barrier.Val;
  }

  uint16_t getSystemRegister() const {
    assert(Kind == k_SystemRegister && "Invalid access!");
    return SystemRegister.Val;
  }

  ARM64SYS::CPSRField getCPSRField() const {
    assert(Kind == k_CPSRField && "Invalid access!");
    return CPSRField.Field;
  }

  unsigned getReg() const {
    assert(Kind == k_Register && "Invalid access!");
    return Reg.RegNum;
  }

  unsigned getVectorListStart() const {
    assert(Kind == k_VectorList && "Invalid access!");
    return VectorList.RegNum;
  }

  unsigned getVectorListCount() const {
    assert(Kind == k_VectorList && "Invalid access!");
    return VectorList.Count;
  }

  unsigned getVectorIndex() const {
    assert(Kind == k_VectorIndex && "Invalid access!");
    return VectorIndex.Val;
  }

  unsigned getSysCR() const {
    assert(Kind == k_SysCR && "Invalid access!");
    return SysCRImm.Val;
  }

  unsigned getPrefetch() const {
    assert(Kind == k_Prefetch && "Invalid access!");
    return Prefetch.Val;
  }

  unsigned getShifter() const {
    assert(Kind == k_Shifter && "Invalid access!");
    return Shifter.Val;
  }

  unsigned getExtend() const {
    assert(Kind == k_Extend && "Invalid access!");
    return Extend.Val;
  }

  bool isImm() const { return Kind == k_Immediate; }
  bool isSImm9() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= -256 && Val < 256);
  }
  bool isSImm7s4() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= -256 && Val <= 252 && (Val & 3) == 0);
  }
  bool isSImm7s8() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= -512 && Val <= 504 && (Val & 7) == 0);
  }
  bool isSImm7s16() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= -1024 && Val <= 1008 && (Val & 15) == 0);
  }
  bool isImm0_7() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 8);
  }
  bool isImm1_8() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val > 0 && Val < 9);
  }
  bool isImm0_15() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 16);
  }
  bool isImm1_16() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val > 0 && Val < 17);
  }
  bool isImm0_31() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 32);
  }
  bool isImm1_31() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 1 && Val < 32);
  }
  bool isImm1_32() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 1 && Val < 33);
  }
  bool isImm0_63() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 64);
  }
  bool isImm1_63() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 1 && Val < 64);
  }
  bool isImm1_64() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 1 && Val < 65);
  }
  bool isImm0_127() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 128);
  }
  bool isImm0_255() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 256);
  }
  bool isImm0_65535() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    int64_t Val = MCE->getValue();
    return (Val >= 0 && Val < 65536);
  }
  bool isLogicalImm32() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    return ARM64_AM::isLogicalImmediate(MCE->getValue(), 32);
  }
  bool isLogicalImm64() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    return ARM64_AM::isLogicalImmediate(MCE->getValue(), 64);
  }
  bool isSIMDImmType10() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return false;
    return ARM64_AM::isAdvSIMDModImmType10(MCE->getValue());
  }
  bool isBranchTarget26() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return true;
    int64_t Val = MCE->getValue();
    if (Val & 0x3)
      return false;
    return (Val >= -(0x2000000 << 2) && Val <= (0x1ffffff << 2));
  }
  bool isBranchTarget19() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return true;
    int64_t Val = MCE->getValue();
    if (Val & 0x3)
      return false;
    return (Val >= -(0x40000 << 2) && Val <= (0x3ffff << 2));
  }
  bool isBranchTarget14() const {
    if (!isImm())
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      return true;
    int64_t Val = MCE->getValue();
    if (Val & 0x3)
      return false;
    return (Val >= -(0x2000 << 2) && Val <= (0x1fff << 2));
  }

  bool isMovWSymbol(ArrayRef<ARM64MCExpr::VariantKind> AllowedModifiers) const {
    if (!isImm())
      return false;

    ARM64MCExpr::VariantKind ELFRefKind;
    MCSymbolRefExpr::VariantKind DarwinRefKind;
    const MCConstantExpr *Addend;
    if (!ARM64AsmParser::classifySymbolRef(getImm(), ELFRefKind, DarwinRefKind,
                                           Addend)) {
      return false;
    }
    if (DarwinRefKind != MCSymbolRefExpr::VK_None)
      return false;

    for (unsigned i = 0; i != AllowedModifiers.size(); ++i) {
      if (ELFRefKind == AllowedModifiers[i])
        return Addend == 0;
    }

    return false;
  }

  bool isMovZSymbolG3() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G3 };
    return isMovWSymbol(Variants);
  }

  bool isMovZSymbolG2() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G2,
                                                   ARM64MCExpr::VK_TPREL_G2,
                                                   ARM64MCExpr::VK_DTPREL_G2 };
    return isMovWSymbol(Variants);
  }

  bool isMovZSymbolG1() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G1,
                                                   ARM64MCExpr::VK_GOTTPREL_G1,
                                                   ARM64MCExpr::VK_TPREL_G1,
                                                   ARM64MCExpr::VK_DTPREL_G1, };
    return isMovWSymbol(Variants);
  }

  bool isMovZSymbolG0() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G0,
                                                   ARM64MCExpr::VK_TPREL_G0,
                                                   ARM64MCExpr::VK_DTPREL_G0 };
    return isMovWSymbol(Variants);
  }

  bool isMovKSymbolG2() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G2_NC };
    return isMovWSymbol(Variants);
  }

  bool isMovKSymbolG1() const {
    static ARM64MCExpr::VariantKind Variants[] = {
      ARM64MCExpr::VK_ABS_G1_NC, ARM64MCExpr::VK_TPREL_G1_NC,
      ARM64MCExpr::VK_DTPREL_G1_NC
    };
    return isMovWSymbol(Variants);
  }

  bool isMovKSymbolG0() const {
    static ARM64MCExpr::VariantKind Variants[] = {
      ARM64MCExpr::VK_ABS_G0_NC,   ARM64MCExpr::VK_GOTTPREL_G0_NC,
      ARM64MCExpr::VK_TPREL_G0_NC, ARM64MCExpr::VK_DTPREL_G0_NC
    };
    return isMovWSymbol(Variants);
  }

  bool isFPImm() const { return Kind == k_FPImm; }
  bool isBarrier() const { return Kind == k_Barrier; }
  bool isSystemRegister() const {
    if (Kind == k_SystemRegister)
      return true;
    // SPSel is legal for both the system register and the CPSR-field
    // variants of MSR, so special case that. Fugly.
    return (Kind == k_CPSRField && getCPSRField() == ARM64SYS::cpsr_SPSel);
  }
  bool isSystemCPSRField() const { return Kind == k_CPSRField; }
  bool isReg() const { return Kind == k_Register && !Reg.isVector; }
  bool isVectorReg() const { return Kind == k_Register && Reg.isVector; }

  /// Is this a vector list with the type implicit (presumably attached to the
  /// instruction itself)?
  template <unsigned NumRegs> bool isImplicitlyTypedVectorList() const {
    return Kind == k_VectorList && VectorList.Count == NumRegs &&
           !VectorList.ElementKind;
  }

  template <unsigned NumRegs, unsigned NumElements, char ElementKind>
  bool isTypedVectorList() const {
    if (Kind != k_VectorList)
      return false;
    if (VectorList.Count != NumRegs)
      return false;
    if (VectorList.ElementKind != ElementKind)
      return false;
    return VectorList.NumElements == NumElements;
  }

  bool isVectorIndexB() const {
    return Kind == k_VectorIndex && VectorIndex.Val < 16;
  }
  bool isVectorIndexH() const {
    return Kind == k_VectorIndex && VectorIndex.Val < 8;
  }
  bool isVectorIndexS() const {
    return Kind == k_VectorIndex && VectorIndex.Val < 4;
  }
  bool isVectorIndexD() const {
    return Kind == k_VectorIndex && VectorIndex.Val < 2;
  }
  bool isToken() const { return Kind == k_Token; }
  bool isTokenEqual(StringRef Str) const {
    return Kind == k_Token && getToken() == Str;
  }
  bool isMem() const { return Kind == k_Memory; }
  bool isSysCR() const { return Kind == k_SysCR; }
  bool isPrefetch() const { return Kind == k_Prefetch; }
  bool isShifter() const { return Kind == k_Shifter; }
  bool isExtend() const {
    // lsl is an alias for UXTX but will be a parsed as a k_Shifter operand.
    if (isShifter()) {
      ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(Shifter.Val);
      return ST == ARM64_AM::LSL;
    }
    return Kind == k_Extend;
  }
  bool isExtend64() const {
    if (Kind != k_Extend)
      return false;
    // UXTX and SXTX require a 64-bit source register (the ExtendLSL64 class).
    ARM64_AM::ExtendType ET = ARM64_AM::getArithExtendType(Extend.Val);
    return ET != ARM64_AM::UXTX && ET != ARM64_AM::SXTX;
  }
  bool isExtendLSL64() const {
    // lsl is an alias for UXTX but will be a parsed as a k_Shifter operand.
    if (isShifter()) {
      ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(Shifter.Val);
      return ST == ARM64_AM::LSL;
    }
    if (Kind != k_Extend)
      return false;
    ARM64_AM::ExtendType ET = ARM64_AM::getArithExtendType(Extend.Val);
    return ET == ARM64_AM::UXTX || ET == ARM64_AM::SXTX;
  }

  bool isArithmeticShifter() const {
    if (!isShifter())
      return false;

    // An arithmetic shifter is LSL, LSR, or ASR.
    ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(Shifter.Val);
    return ST == ARM64_AM::LSL || ST == ARM64_AM::LSR || ST == ARM64_AM::ASR;
  }

  bool isMovImm32Shifter() const {
    if (!isShifter())
      return false;

    // A MOVi shifter is LSL of 0, 16, 32, or 48.
    ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(Shifter.Val);
    if (ST != ARM64_AM::LSL)
      return false;
    uint64_t Val = ARM64_AM::getShiftValue(Shifter.Val);
    return (Val == 0 || Val == 16);
  }

  bool isMovImm64Shifter() const {
    if (!isShifter())
      return false;

    // A MOVi shifter is LSL of 0 or 16.
    ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(Shifter.Val);
    if (ST != ARM64_AM::LSL)
      return false;
    uint64_t Val = ARM64_AM::getShiftValue(Shifter.Val);
    return (Val == 0 || Val == 16 || Val == 32 || Val == 48);
  }

  bool isAddSubShifter() const {
    if (!isShifter())
      return false;

    // An ADD/SUB shifter is either 'lsl #0' or 'lsl #12'.
    unsigned Val = Shifter.Val;
    return ARM64_AM::getShiftType(Val) == ARM64_AM::LSL &&
           (ARM64_AM::getShiftValue(Val) == 0 ||
            ARM64_AM::getShiftValue(Val) == 12);
  }

  bool isLogicalVecShifter() const {
    if (!isShifter())
      return false;

    // A logical vector shifter is a left shift by 0, 8, 16, or 24.
    unsigned Val = Shifter.Val;
    unsigned Shift = ARM64_AM::getShiftValue(Val);
    return ARM64_AM::getShiftType(Val) == ARM64_AM::LSL &&
           (Shift == 0 || Shift == 8 || Shift == 16 || Shift == 24);
  }

  bool isLogicalVecHalfWordShifter() const {
    if (!isLogicalVecShifter())
      return false;

    // A logical vector shifter is a left shift by 0 or 8.
    unsigned Val = Shifter.Val;
    unsigned Shift = ARM64_AM::getShiftValue(Val);
    return ARM64_AM::getShiftType(Val) == ARM64_AM::LSL &&
           (Shift == 0 || Shift == 8);
  }

  bool isMoveVecShifter() const {
    if (!isShifter())
      return false;

    // A logical vector shifter is a left shift by 8 or 16.
    unsigned Val = Shifter.Val;
    unsigned Shift = ARM64_AM::getShiftValue(Val);
    return ARM64_AM::getShiftType(Val) == ARM64_AM::MSL &&
           (Shift == 8 || Shift == 16);
  }

  bool isMemoryRegisterOffset8() const {
    return isMem() && Mem.Mode == RegisterOffset && Mem.ShiftVal == 0;
  }

  bool isMemoryRegisterOffset16() const {
    return isMem() && Mem.Mode == RegisterOffset &&
           (Mem.ShiftVal == 0 || Mem.ShiftVal == 1);
  }

  bool isMemoryRegisterOffset32() const {
    return isMem() && Mem.Mode == RegisterOffset &&
           (Mem.ShiftVal == 0 || Mem.ShiftVal == 2);
  }

  bool isMemoryRegisterOffset64() const {
    return isMem() && Mem.Mode == RegisterOffset &&
           (Mem.ShiftVal == 0 || Mem.ShiftVal == 3);
  }

  bool isMemoryRegisterOffset128() const {
    return isMem() && Mem.Mode == RegisterOffset &&
           (Mem.ShiftVal == 0 || Mem.ShiftVal == 4);
  }

  bool isMemoryUnscaled() const {
    if (!isMem())
      return false;
    if (Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    // Make sure the immediate value is valid.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    if (!CE)
      return false;
    // The offset must fit in a signed 9-bit unscaled immediate.
    int64_t Value = CE->getValue();
    return (Value >= -256 && Value < 256);
  }
  // Fallback unscaled operands are for aliases of LDR/STR that fall back
  // to LDUR/STUR when the offset is not legal for the former but is for
  // the latter. As such, in addition to checking for being a legal unscaled
  // address, also check that it is not a legal scaled address. This avoids
  // ambiguity in the matcher.
  bool isMemoryUnscaledFB8() const {
    return isMemoryUnscaled() && !isMemoryIndexed8();
  }
  bool isMemoryUnscaledFB16() const {
    return isMemoryUnscaled() && !isMemoryIndexed16();
  }
  bool isMemoryUnscaledFB32() const {
    return isMemoryUnscaled() && !isMemoryIndexed32();
  }
  bool isMemoryUnscaledFB64() const {
    return isMemoryUnscaled() && !isMemoryIndexed64();
  }
  bool isMemoryUnscaledFB128() const {
    return isMemoryUnscaled() && !isMemoryIndexed128();
  }
  bool isMemoryIndexed(unsigned Scale) const {
    if (!isMem())
      return false;
    if (Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    // Make sure the immediate value is valid.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);

    if (CE) {
      // The offset must be a positive multiple of the scale and in range of
      // encoding with a 12-bit immediate.
      int64_t Value = CE->getValue();
      return (Value >= 0 && (Value % Scale) == 0 && Value <= (4095 * Scale));
    }

    // If it's not a constant, check for some expressions we know.
    const MCExpr *Expr = Mem.OffsetImm;
    ARM64MCExpr::VariantKind ELFRefKind;
    MCSymbolRefExpr::VariantKind DarwinRefKind;
    const MCConstantExpr *Addend;
    if (!ARM64AsmParser::classifySymbolRef(Expr, ELFRefKind, DarwinRefKind,
                                           Addend)) {
      // If we don't understand the expression, assume the best and
      // let the fixup and relocation code deal with it.
      return true;
    }

    if (DarwinRefKind == MCSymbolRefExpr::VK_PAGEOFF ||
        ELFRefKind == ARM64MCExpr::VK_LO12 ||
        ELFRefKind == ARM64MCExpr::VK_GOT_LO12 ||
        ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12 ||
        ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12_NC ||
        ELFRefKind == ARM64MCExpr::VK_TPREL_LO12 ||
        ELFRefKind == ARM64MCExpr::VK_TPREL_LO12_NC ||
        ELFRefKind == ARM64MCExpr::VK_GOTTPREL_LO12_NC ||
        ELFRefKind == ARM64MCExpr::VK_TLSDESC_LO12) {
      // Note that we don't range-check the addend. It's adjusted modulo page
      // size when converted, so there is no "out of range" condition when using
      // @pageoff.
      int64_t Value = Addend ? Addend->getValue() : 0;
      return Value >= 0 && (Value % Scale) == 0;
    } else if (DarwinRefKind == MCSymbolRefExpr::VK_GOTPAGEOFF ||
               DarwinRefKind == MCSymbolRefExpr::VK_TLVPPAGEOFF) {
      // @gotpageoff/@tlvppageoff can only be used directly, not with an addend.
      return Addend == 0;
    }

    return false;
  }
  bool isMemoryIndexed128() const { return isMemoryIndexed(16); }
  bool isMemoryIndexed64() const { return isMemoryIndexed(8); }
  bool isMemoryIndexed32() const { return isMemoryIndexed(4); }
  bool isMemoryIndexed16() const { return isMemoryIndexed(2); }
  bool isMemoryIndexed8() const { return isMemoryIndexed(1); }
  bool isMemoryNoIndex() const {
    if (!isMem())
      return false;
    if (Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;

    // Make sure the immediate value is valid. Only zero is allowed.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    if (!CE || CE->getValue() != 0)
      return false;
    return true;
  }
  bool isMemorySIMDNoIndex() const {
    if (!isMem())
      return false;
    if (Mem.Mode != ImmediateOffset)
      return false;
    return Mem.OffsetImm == 0;
  }
  bool isMemoryIndexedSImm9() const {
    if (!isMem() || Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    assert(CE && "Non-constant pre-indexed offset!");
    int64_t Value = CE->getValue();
    return Value >= -256 && Value <= 255;
  }
  bool isMemoryIndexed32SImm7() const {
    if (!isMem() || Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    assert(CE && "Non-constant pre-indexed offset!");
    int64_t Value = CE->getValue();
    return ((Value % 4) == 0) && Value >= -256 && Value <= 252;
  }
  bool isMemoryIndexed64SImm7() const {
    if (!isMem() || Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    assert(CE && "Non-constant pre-indexed offset!");
    int64_t Value = CE->getValue();
    return ((Value % 8) == 0) && Value >= -512 && Value <= 504;
  }
  bool isMemoryIndexed128SImm7() const {
    if (!isMem() || Mem.Mode != ImmediateOffset)
      return false;
    if (!Mem.OffsetImm)
      return true;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
    assert(CE && "Non-constant pre-indexed offset!");
    int64_t Value = CE->getValue();
    return ((Value % 16) == 0) && Value >= -1024 && Value <= 1008;
  }

  bool isAdrpLabel() const {
    // Validation was handled during parsing, so we just sanity check that
    // something didn't go haywire.
    return isImm();
  }

  bool isAdrLabel() const {
    // Validation was handled during parsing, so we just sanity check that
    // something didn't go haywire.
    return isImm();
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addVectorRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  template <unsigned NumRegs>
  void addVectorList64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    static unsigned FirstRegs[] = { ARM64::D0,       ARM64::D0_D1,
                                    ARM64::D0_D1_D2, ARM64::D0_D1_D2_D3 };
    unsigned FirstReg = FirstRegs[NumRegs - 1];

    Inst.addOperand(
        MCOperand::CreateReg(FirstReg + getVectorListStart() - ARM64::Q0));
  }

  template <unsigned NumRegs>
  void addVectorList128Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    static unsigned FirstRegs[] = { ARM64::Q0,       ARM64::Q0_Q1,
                                    ARM64::Q0_Q1_Q2, ARM64::Q0_Q1_Q2_Q3 };
    unsigned FirstReg = FirstRegs[NumRegs - 1];

    Inst.addOperand(
        MCOperand::CreateReg(FirstReg + getVectorListStart() - ARM64::Q0));
  }

  void addVectorIndexBOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getVectorIndex()));
  }

  void addVectorIndexHOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getVectorIndex()));
  }

  void addVectorIndexSOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getVectorIndex()));
  }

  void addVectorIndexDOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getVectorIndex()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // If this is a pageoff symrefexpr with an addend, adjust the addend
    // to be only the page-offset portion. Otherwise, just add the expr
    // as-is.
    addExpr(Inst, getImm());
  }

  void addAdrpLabelOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addAdrLabelOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addSImm9Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addSImm7s4Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() / 4));
  }

  void addSImm7s8Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() / 8));
  }

  void addSImm7s16Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() / 16));
  }

  void addImm0_7Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_8Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_15Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_16Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_31Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_31Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_32Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_63Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_63Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm1_64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_127Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_255Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addImm0_65535Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue()));
  }

  void addLogicalImm32Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid logical immediate operand!");
    uint64_t encoding = ARM64_AM::encodeLogicalImmediate(MCE->getValue(), 32);
    Inst.addOperand(MCOperand::CreateImm(encoding));
  }

  void addLogicalImm64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid logical immediate operand!");
    uint64_t encoding = ARM64_AM::encodeLogicalImmediate(MCE->getValue(), 64);
    Inst.addOperand(MCOperand::CreateImm(encoding));
  }

  void addSIMDImmType10Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    assert(MCE && "Invalid immediate operand!");
    uint64_t encoding = ARM64_AM::encodeAdvSIMDModImmType10(MCE->getValue());
    Inst.addOperand(MCOperand::CreateImm(encoding));
  }

  void addBranchTarget26Operands(MCInst &Inst, unsigned N) const {
    // Branch operands don't encode the low bits, so shift them off
    // here. If it's a label, however, just put it on directly as there's
    // not enough information now to do anything.
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE) {
      addExpr(Inst, getImm());
      return;
    }
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() >> 2));
  }

  void addBranchTarget19Operands(MCInst &Inst, unsigned N) const {
    // Branch operands don't encode the low bits, so shift them off
    // here. If it's a label, however, just put it on directly as there's
    // not enough information now to do anything.
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE) {
      addExpr(Inst, getImm());
      return;
    }
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() >> 2));
  }

  void addBranchTarget14Operands(MCInst &Inst, unsigned N) const {
    // Branch operands don't encode the low bits, so shift them off
    // here. If it's a label, however, just put it on directly as there's
    // not enough information now to do anything.
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE) {
      addExpr(Inst, getImm());
      return;
    }
    assert(MCE && "Invalid constant immediate operand!");
    Inst.addOperand(MCOperand::CreateImm(MCE->getValue() >> 2));
  }

  void addFPImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getFPImm()));
  }

  void addBarrierOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getBarrier()));
  }

  void addSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == k_SystemRegister)
      Inst.addOperand(MCOperand::CreateImm(getSystemRegister()));
    else {
      assert(Kind == k_CPSRField && getCPSRField() == ARM64SYS::cpsr_SPSel);
      Inst.addOperand(MCOperand::CreateImm(ARM64SYS::SPSel));
    }
  }

  void addSystemCPSRFieldOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getCPSRField()));
  }

  void addSysCROperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getSysCR()));
  }

  void addPrefetchOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getPrefetch()));
  }

  void addShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addArithmeticShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addMovImm32ShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addMovImm64ShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addAddSubShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addLogicalVecShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addLogicalVecHalfWordShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addMoveVecShifterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getShifter()));
  }

  void addExtendOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // lsl is an alias for UXTX but will be a parsed as a k_Shifter operand.
    if (isShifter()) {
      assert(ARM64_AM::getShiftType(getShifter()) == ARM64_AM::LSL);
      unsigned imm = getArithExtendImm(ARM64_AM::UXTX,
                                       ARM64_AM::getShiftValue(getShifter()));
      Inst.addOperand(MCOperand::CreateImm(imm));
    } else
      Inst.addOperand(MCOperand::CreateImm(getExtend()));
  }

  void addExtend64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getExtend()));
  }

  void addExtendLSL64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // lsl is an alias for UXTX but will be a parsed as a k_Shifter operand.
    if (isShifter()) {
      assert(ARM64_AM::getShiftType(getShifter()) == ARM64_AM::LSL);
      unsigned imm = getArithExtendImm(ARM64_AM::UXTX,
                                       ARM64_AM::getShiftValue(getShifter()));
      Inst.addOperand(MCOperand::CreateImm(imm));
    } else
      Inst.addOperand(MCOperand::CreateImm(getExtend()));
  }

  void addMemoryRegisterOffsetOperands(MCInst &Inst, unsigned N, bool DoShift) {
    assert(N == 3 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateReg(Mem.OffsetRegNum));
    unsigned ExtendImm = ARM64_AM::getMemExtendImm(Mem.ExtType, DoShift);
    Inst.addOperand(MCOperand::CreateImm(ExtendImm));
  }

  void addMemoryRegisterOffset8Operands(MCInst &Inst, unsigned N) {
    addMemoryRegisterOffsetOperands(Inst, N, Mem.ExplicitShift);
  }

  void addMemoryRegisterOffset16Operands(MCInst &Inst, unsigned N) {
    addMemoryRegisterOffsetOperands(Inst, N, Mem.ShiftVal == 1);
  }

  void addMemoryRegisterOffset32Operands(MCInst &Inst, unsigned N) {
    addMemoryRegisterOffsetOperands(Inst, N, Mem.ShiftVal == 2);
  }

  void addMemoryRegisterOffset64Operands(MCInst &Inst, unsigned N) {
    addMemoryRegisterOffsetOperands(Inst, N, Mem.ShiftVal == 3);
  }

  void addMemoryRegisterOffset128Operands(MCInst &Inst, unsigned N) {
    addMemoryRegisterOffsetOperands(Inst, N, Mem.ShiftVal == 4);
  }

  void addMemoryIndexedOperands(MCInst &Inst, unsigned N,
                                unsigned Scale) const {
    // Add the base register operand.
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));

    if (!Mem.OffsetImm) {
      // There isn't an offset.
      Inst.addOperand(MCOperand::CreateImm(0));
      return;
    }

    // Add the offset operand.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm)) {
      assert(CE->getValue() % Scale == 0 &&
             "Offset operand must be multiple of the scale!");

      // The MCInst offset operand doesn't include the low bits (like the
      // instruction encoding).
      Inst.addOperand(MCOperand::CreateImm(CE->getValue() / Scale));
    }

    // If this is a pageoff symrefexpr with an addend, the linker will
    // do the scaling of the addend.
    //
    // Otherwise we don't know what this is, so just add the scaling divide to
    // the expression and let the MC fixup evaluation code deal with it.
    const MCExpr *Expr = Mem.OffsetImm;
    ARM64MCExpr::VariantKind ELFRefKind;
    MCSymbolRefExpr::VariantKind DarwinRefKind;
    const MCConstantExpr *Addend;
    if (Scale > 1 &&
        (!ARM64AsmParser::classifySymbolRef(Expr, ELFRefKind, DarwinRefKind,
                                            Addend) ||
         (Addend != 0 && DarwinRefKind != MCSymbolRefExpr::VK_PAGEOFF))) {
      Expr = MCBinaryExpr::CreateDiv(Expr, MCConstantExpr::Create(Scale, Ctx),
                                     Ctx);
    }

    Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addMemoryUnscaledOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryUnscaled() && "Invalid number of operands!");
    // Add the base register operand.
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));

    // Add the offset operand.
    if (!Mem.OffsetImm)
      Inst.addOperand(MCOperand::CreateImm(0));
    else {
      // Only constant offsets supported.
      const MCConstantExpr *CE = cast<MCConstantExpr>(Mem.OffsetImm);
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    }
  }

  void addMemoryIndexed128Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryIndexed128() && "Invalid number of operands!");
    addMemoryIndexedOperands(Inst, N, 16);
  }

  void addMemoryIndexed64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryIndexed64() && "Invalid number of operands!");
    addMemoryIndexedOperands(Inst, N, 8);
  }

  void addMemoryIndexed32Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryIndexed32() && "Invalid number of operands!");
    addMemoryIndexedOperands(Inst, N, 4);
  }

  void addMemoryIndexed16Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryIndexed16() && "Invalid number of operands!");
    addMemoryIndexedOperands(Inst, N, 2);
  }

  void addMemoryIndexed8Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemoryIndexed8() && "Invalid number of operands!");
    addMemoryIndexedOperands(Inst, N, 1);
  }

  void addMemoryNoIndexOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isMemoryNoIndex() && "Invalid number of operands!");
    // Add the base register operand (the offset is always zero, so ignore it).
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
  }

  void addMemorySIMDNoIndexOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isMemorySIMDNoIndex() && "Invalid number of operands!");
    // Add the base register operand (the offset is always zero, so ignore it).
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
  }

  void addMemoryWritebackIndexedOperands(MCInst &Inst, unsigned N,
                                         unsigned Scale) const {
    assert(N == 2 && "Invalid number of operands!");

    // Add the base register operand.
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));

    // Add the offset operand.
    int64_t Offset = 0;
    if (Mem.OffsetImm) {
      const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.OffsetImm);
      assert(CE && "Non-constant indexed offset operand!");
      Offset = CE->getValue();
    }

    if (Scale != 1) {
      assert(Offset % Scale == 0 &&
             "Offset operand must be a multiple of the scale!");
      Offset /= Scale;
    }

    Inst.addOperand(MCOperand::CreateImm(Offset));
  }

  void addMemoryIndexedSImm9Operands(MCInst &Inst, unsigned N) const {
    addMemoryWritebackIndexedOperands(Inst, N, 1);
  }

  void addMemoryIndexed32SImm7Operands(MCInst &Inst, unsigned N) const {
    addMemoryWritebackIndexedOperands(Inst, N, 4);
  }

  void addMemoryIndexed64SImm7Operands(MCInst &Inst, unsigned N) const {
    addMemoryWritebackIndexedOperands(Inst, N, 8);
  }

  void addMemoryIndexed128SImm7Operands(MCInst &Inst, unsigned N) const {
    addMemoryWritebackIndexedOperands(Inst, N, 16);
  }

  virtual void print(raw_ostream &OS) const;

  static ARM64Operand *CreateToken(StringRef Str, bool IsSuffix, SMLoc S,
                                   MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Token, Ctx);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->Tok.IsSuffix = IsSuffix;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateReg(unsigned RegNum, bool isVector, SMLoc S,
                                 SMLoc E, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Register, Ctx);
    Op->Reg.RegNum = RegNum;
    Op->Reg.isVector = isVector;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateVectorList(unsigned RegNum, unsigned Count,
                                        unsigned NumElements, char ElementKind,
                                        SMLoc S, SMLoc E, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_VectorList, Ctx);
    Op->VectorList.RegNum = RegNum;
    Op->VectorList.Count = Count;
    Op->VectorList.NumElements = NumElements;
    Op->VectorList.ElementKind = ElementKind;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateVectorIndex(unsigned Idx, SMLoc S, SMLoc E,
                                         MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_VectorIndex, Ctx);
    Op->VectorIndex.Val = Idx;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E,
                                 MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Immediate, Ctx);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateFPImm(unsigned Val, SMLoc S, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_FPImm, Ctx);
    Op->FPImm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateBarrier(unsigned Val, SMLoc S, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Barrier, Ctx);
    Op->Barrier.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateSystemRegister(uint16_t Val, SMLoc S,
                                            MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_SystemRegister, Ctx);
    Op->SystemRegister.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateCPSRField(ARM64SYS::CPSRField Field, SMLoc S,
                                       MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_CPSRField, Ctx);
    Op->CPSRField.Field = Field;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateMem(unsigned BaseRegNum, const MCExpr *Off,
                                 SMLoc S, SMLoc E, SMLoc OffsetLoc,
                                 MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Memory, Ctx);
    Op->Mem.BaseRegNum = BaseRegNum;
    Op->Mem.OffsetRegNum = 0;
    Op->Mem.OffsetImm = Off;
    Op->Mem.ExtType = ARM64_AM::UXTX;
    Op->Mem.ShiftVal = 0;
    Op->Mem.ExplicitShift = false;
    Op->Mem.Mode = ImmediateOffset;
    Op->OffsetLoc = OffsetLoc;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateRegOffsetMem(unsigned BaseReg, unsigned OffsetReg,
                                          ARM64_AM::ExtendType ExtType,
                                          unsigned ShiftVal, bool ExplicitShift,
                                          SMLoc S, SMLoc E, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Memory, Ctx);
    Op->Mem.BaseRegNum = BaseReg;
    Op->Mem.OffsetRegNum = OffsetReg;
    Op->Mem.OffsetImm = 0;
    Op->Mem.ExtType = ExtType;
    Op->Mem.ShiftVal = ShiftVal;
    Op->Mem.ExplicitShift = ExplicitShift;
    Op->Mem.Mode = RegisterOffset;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateSysCR(unsigned Val, SMLoc S, SMLoc E,
                                   MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_SysCR, Ctx);
    Op->SysCRImm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreatePrefetch(unsigned Val, SMLoc S, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Prefetch, Ctx);
    Op->Prefetch.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARM64Operand *CreateShifter(ARM64_AM::ShiftType ShOp, unsigned Val,
                                     SMLoc S, SMLoc E, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Shifter, Ctx);
    Op->Shifter.Val = ARM64_AM::getShifterImm(ShOp, Val);
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARM64Operand *CreateExtend(ARM64_AM::ExtendType ExtOp, unsigned Val,
                                    SMLoc S, SMLoc E, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_Extend, Ctx);
    Op->Extend.Val = ARM64_AM::getArithExtendImm(ExtOp, Val);
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
};

} // end anonymous namespace.

void ARM64Operand::print(raw_ostream &OS) const {
  switch (Kind) {
  case k_FPImm:
    OS << "<fpimm " << getFPImm() << "(" << ARM64_AM::getFPImmFloat(getFPImm())
       << ") >";
    break;
  case k_Barrier: {
    const char *Name =
        ARM64SYS::getBarrierOptName((ARM64SYS::BarrierOption)getBarrier());
    OS << "<barrier ";
    if (Name)
      OS << Name;
    else
      OS << getBarrier();
    OS << ">";
    break;
  }
  case k_SystemRegister: {
    const char *Name = ARM64SYS::getSystemRegisterName(
        (ARM64SYS::SystemRegister)getSystemRegister());
    OS << "<systemreg ";
    if (Name)
      OS << Name;
    else
      OS << "#" << getSystemRegister();
    OS << ">";
    break;
  }
  case k_CPSRField: {
    const char *Name = ARM64SYS::getCPSRFieldName(getCPSRField());
    OS << "<cpsrfield " << Name << ">";
    break;
  }
  case k_Immediate:
    getImm()->print(OS);
    break;
  case k_Memory:
    OS << "<memory>";
    break;
  case k_Register:
    OS << "<register " << getReg() << ">";
    break;
  case k_VectorList: {
    OS << "<vectorlist ";
    unsigned Reg = getVectorListStart();
    for (unsigned i = 0, e = getVectorListCount(); i != e; ++i)
      OS << Reg + i << " ";
    OS << ">";
    break;
  }
  case k_VectorIndex:
    OS << "<vectorindex " << getVectorIndex() << ">";
    break;
  case k_Token:
    OS << "'" << getToken() << "'";
    break;
  case k_SysCR:
    OS << "c" << getSysCR();
    break;
  case k_Prefetch:
    OS << "<prfop ";
    if (ARM64_AM::isNamedPrefetchOp(getPrefetch()))
      OS << ARM64_AM::getPrefetchOpName((ARM64_AM::PrefetchOp)getPrefetch());
    else
      OS << "#" << getPrefetch();
    OS << ">";
    break;
  case k_Shifter: {
    unsigned Val = getShifter();
    OS << "<" << ARM64_AM::getShiftName(ARM64_AM::getShiftType(Val)) << " #"
       << ARM64_AM::getShiftValue(Val) << ">";
    break;
  }
  case k_Extend: {
    unsigned Val = getExtend();
    OS << "<" << ARM64_AM::getExtendName(ARM64_AM::getArithExtendType(Val))
       << " #" << ARM64_AM::getArithShiftValue(Val) << ">";
    break;
  }
  }
}

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

static unsigned matchVectorRegName(StringRef Name) {
  return StringSwitch<unsigned>(Name)
      .Case("v0", ARM64::Q0)
      .Case("v1", ARM64::Q1)
      .Case("v2", ARM64::Q2)
      .Case("v3", ARM64::Q3)
      .Case("v4", ARM64::Q4)
      .Case("v5", ARM64::Q5)
      .Case("v6", ARM64::Q6)
      .Case("v7", ARM64::Q7)
      .Case("v8", ARM64::Q8)
      .Case("v9", ARM64::Q9)
      .Case("v10", ARM64::Q10)
      .Case("v11", ARM64::Q11)
      .Case("v12", ARM64::Q12)
      .Case("v13", ARM64::Q13)
      .Case("v14", ARM64::Q14)
      .Case("v15", ARM64::Q15)
      .Case("v16", ARM64::Q16)
      .Case("v17", ARM64::Q17)
      .Case("v18", ARM64::Q18)
      .Case("v19", ARM64::Q19)
      .Case("v20", ARM64::Q20)
      .Case("v21", ARM64::Q21)
      .Case("v22", ARM64::Q22)
      .Case("v23", ARM64::Q23)
      .Case("v24", ARM64::Q24)
      .Case("v25", ARM64::Q25)
      .Case("v26", ARM64::Q26)
      .Case("v27", ARM64::Q27)
      .Case("v28", ARM64::Q28)
      .Case("v29", ARM64::Q29)
      .Case("v30", ARM64::Q30)
      .Case("v31", ARM64::Q31)
      .Default(0);
}

static bool isValidVectorKind(StringRef Name) {
  return StringSwitch<bool>(Name.lower())
      .Case(".8b", true)
      .Case(".16b", true)
      .Case(".4h", true)
      .Case(".8h", true)
      .Case(".2s", true)
      .Case(".4s", true)
      .Case(".1d", true)
      .Case(".2d", true)
      .Case(".1q", true)
      // Accept the width neutral ones, too, for verbose syntax. If those
      // aren't used in the right places, the token operand won't match so
      // all will work out.
      .Case(".b", true)
      .Case(".h", true)
      .Case(".s", true)
      .Case(".d", true)
      .Default(false);
}

static void parseValidVectorKind(StringRef Name, unsigned &NumElements,
                                 char &ElementKind) {
  assert(isValidVectorKind(Name));

  ElementKind = Name.lower()[Name.size() - 1];
  NumElements = 0;

  if (Name.size() == 2)
    return;

  // Parse the lane count
  Name = Name.drop_front();
  while (isdigit(Name.front())) {
    NumElements = 10 * NumElements + (Name.front() - '0');
    Name = Name.drop_front();
  }
}

bool ARM64AsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                   SMLoc &EndLoc) {
  StartLoc = getLoc();
  RegNo = tryParseRegister();
  EndLoc = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  return (RegNo == (unsigned)-1);
}

/// tryParseRegister - Try to parse a register name. The token must be an
/// Identifier when called, and if it is a register name the token is eaten and
/// the register is added to the operand list.
int ARM64AsmParser::tryParseRegister() {
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  std::string lowerCase = Tok.getString().lower();
  unsigned RegNum = MatchRegisterName(lowerCase);
  // Also handle a few aliases of registers.
  if (RegNum == 0)
    RegNum = StringSwitch<unsigned>(lowerCase)
                 .Case("x29", ARM64::FP)
                 .Case("x30", ARM64::LR)
                 .Case("x31", ARM64::XZR)
                 .Case("w31", ARM64::WZR)
                 .Default(0);

  if (RegNum == 0)
    return -1;

  Parser.Lex(); // Eat identifier token.
  return RegNum;
}

/// tryMatchVectorRegister - Try to parse a vector register name with optional
/// kind specifier. If it is a register specifier, eat the token and return it.
int ARM64AsmParser::tryMatchVectorRegister(StringRef &Kind) {
  if (Parser.getTok().isNot(AsmToken::Identifier)) {
    TokError("vector register expected");
    return -1;
  }

  StringRef Name = Parser.getTok().getString();
  // If there is a kind specifier, it's separated from the register name by
  // a '.'.
  size_t Start = 0, Next = Name.find('.');
  StringRef Head = Name.slice(Start, Next);
  unsigned RegNum = matchVectorRegName(Head);
  if (RegNum) {
    if (Next != StringRef::npos) {
      Kind = Name.slice(Next, StringRef::npos);
      if (!isValidVectorKind(Kind)) {
        TokError("invalid vector kind qualifier");
        return -1;
      }
    }
    Parser.Lex(); // Eat the register token.
    return RegNum;
  }
  return -1;
}

static int MatchSysCRName(StringRef Name) {
  // Use the same layout as the tablegen'erated register name matcher. Ugly,
  // but efficient.
  switch (Name.size()) {
  default:
    break;
  case 2:
    if (Name[0] != 'c' && Name[0] != 'C')
      return -1;
    switch (Name[1]) {
    default:
      return -1;
    case '0':
      return 0;
    case '1':
      return 1;
    case '2':
      return 2;
    case '3':
      return 3;
    case '4':
      return 4;
    case '5':
      return 5;
    case '6':
      return 6;
    case '7':
      return 7;
    case '8':
      return 8;
    case '9':
      return 9;
    }
    break;
  case 3:
    if ((Name[0] != 'c' && Name[0] != 'C') || Name[1] != '1')
      return -1;
    switch (Name[2]) {
    default:
      return -1;
    case '0':
      return 10;
    case '1':
      return 11;
    case '2':
      return 12;
    case '3':
      return 13;
    case '4':
      return 14;
    case '5':
      return 15;
    }
    break;
  }

  llvm_unreachable("Unhandled SysCR operand string!");
  return -1;
}

/// tryParseSysCROperand - Try to parse a system instruction CR operand name.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseSysCROperand(OperandVector &Operands) {
  SMLoc S = getLoc();
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  int Num = MatchSysCRName(Tok.getString());
  if (Num == -1)
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARM64Operand::CreateSysCR(Num, S, getLoc(), getContext()));
  return MatchOperand_Success;
}

/// tryParsePrefetch - Try to parse a prefetch operand.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParsePrefetch(OperandVector &Operands) {
  SMLoc S = getLoc();
  const AsmToken &Tok = Parser.getTok();
  // Either an identifier for named values or a 5-bit immediate.
  if (Tok.is(AsmToken::Hash)) {
    Parser.Lex(); // Eat hash token.
    const MCExpr *ImmVal;
    if (getParser().parseExpression(ImmVal))
      return MatchOperand_ParseFail;

    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      TokError("immediate value expected for prefetch operand");
      return MatchOperand_ParseFail;
    }
    unsigned prfop = MCE->getValue();
    if (prfop > 31) {
      TokError("prefetch operand out of range, [0,31] expected");
      return MatchOperand_ParseFail;
    }

    Operands.push_back(ARM64Operand::CreatePrefetch(prfop, S, getContext()));
    return MatchOperand_Success;
  }

  if (Tok.isNot(AsmToken::Identifier)) {
    TokError("pre-fetch hint expected");
    return MatchOperand_ParseFail;
  }

  unsigned prfop = StringSwitch<unsigned>(Tok.getString())
                       .Case("pldl1keep", ARM64_AM::PLDL1KEEP)
                       .Case("pldl1strm", ARM64_AM::PLDL1STRM)
                       .Case("pldl2keep", ARM64_AM::PLDL2KEEP)
                       .Case("pldl2strm", ARM64_AM::PLDL2STRM)
                       .Case("pldl3keep", ARM64_AM::PLDL3KEEP)
                       .Case("pldl3strm", ARM64_AM::PLDL3STRM)
                       .Case("pstl1keep", ARM64_AM::PSTL1KEEP)
                       .Case("pstl1strm", ARM64_AM::PSTL1STRM)
                       .Case("pstl2keep", ARM64_AM::PSTL2KEEP)
                       .Case("pstl2strm", ARM64_AM::PSTL2STRM)
                       .Case("pstl3keep", ARM64_AM::PSTL3KEEP)
                       .Case("pstl3strm", ARM64_AM::PSTL3STRM)
                       .Default(0xff);
  if (prfop == 0xff) {
    TokError("pre-fetch hint expected");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARM64Operand::CreatePrefetch(prfop, S, getContext()));
  return MatchOperand_Success;
}

/// tryParseAdrpLabel - Parse and validate a source label for the ADRP
/// instruction.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseAdrpLabel(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Expr;
  if (parseSymbolicImmVal(Expr))
    return MatchOperand_ParseFail;

  ARM64MCExpr::VariantKind ELFRefKind;
  MCSymbolRefExpr::VariantKind DarwinRefKind;
  const MCConstantExpr *Addend;
  if (!classifySymbolRef(Expr, ELFRefKind, DarwinRefKind, Addend)) {
    Error(S, "modified label reference + constant expected");
    return MatchOperand_ParseFail;
  }

  if (DarwinRefKind == MCSymbolRefExpr::VK_None &&
      ELFRefKind == ARM64MCExpr::VK_INVALID) {
    // No modifier was specified at all; this is the syntax for an ELF basic
    // ADRP relocation (unfortunately).
    Expr = ARM64MCExpr::Create(Expr, ARM64MCExpr::VK_ABS_PAGE, getContext());
  } else if ((DarwinRefKind == MCSymbolRefExpr::VK_GOTPAGE ||
              DarwinRefKind == MCSymbolRefExpr::VK_TLVPPAGE) &&
             Addend != 0) {
    Error(S, "gotpage label reference not allowed an addend");
    return MatchOperand_ParseFail;
  } else if (DarwinRefKind != MCSymbolRefExpr::VK_PAGE &&
             DarwinRefKind != MCSymbolRefExpr::VK_GOTPAGE &&
             DarwinRefKind != MCSymbolRefExpr::VK_TLVPPAGE &&
             ELFRefKind != ARM64MCExpr::VK_GOT_PAGE &&
             ELFRefKind != ARM64MCExpr::VK_GOTTPREL_PAGE &&
             ELFRefKind != ARM64MCExpr::VK_TLSDESC_PAGE) {
    // The operand must be an @page or @gotpage qualified symbolref.
    Error(S, "page or gotpage label reference expected");
    return MatchOperand_ParseFail;
  }

  // We have a label reference possibly with addend. The addend is a raw value
  // here. The linker will adjust it to only reference the page.
  SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  Operands.push_back(ARM64Operand::CreateImm(Expr, S, E, getContext()));

  return MatchOperand_Success;
}

/// tryParseAdrLabel - Parse and validate a source label for the ADR
/// instruction.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseAdrLabel(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Expr;
  if (getParser().parseExpression(Expr))
    return MatchOperand_ParseFail;

  // The operand must be an un-qualified assembler local symbolref.
  // FIXME: wrong for ELF.
  if (const MCSymbolRefExpr *SRE = dyn_cast<const MCSymbolRefExpr>(Expr)) {
    // FIXME: Should reference the MachineAsmInfo to get the private prefix.
    bool isTemporary = SRE->getSymbol().getName().startswith("L");
    if (!isTemporary || SRE->getKind() != MCSymbolRefExpr::VK_None) {
      Error(S, "unqualified, assembler-local label name expected");
      return MatchOperand_ParseFail;
    }
  }

  SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  Operands.push_back(ARM64Operand::CreateImm(Expr, S, E, getContext()));

  return MatchOperand_Success;
}

/// tryParseFPImm - A floating point immediate expression operand.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseFPImm(OperandVector &Operands) {
  SMLoc S = getLoc();

  if (Parser.getTok().isNot(AsmToken::Hash))
    return MatchOperand_NoMatch;
  Parser.Lex(); // Eat the '#'.

  // Handle negation, as that still comes through as a separate token.
  bool isNegative = false;
  if (Parser.getTok().is(AsmToken::Minus)) {
    isNegative = true;
    Parser.Lex();
  }
  const AsmToken &Tok = Parser.getTok();
  if (Tok.is(AsmToken::Real)) {
    APFloat RealVal(APFloat::IEEEdouble, Tok.getString());
    uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
    // If we had a '-' in front, toggle the sign bit.
    IntVal ^= (uint64_t)isNegative << 63;
    int Val = ARM64_AM::getFP64Imm(APInt(64, IntVal));
    Parser.Lex(); // Eat the token.
    // Check for out of range values. As an exception, we let Zero through,
    // as we handle that special case in post-processing before matching in
    // order to use the zero register for it.
    if (Val == -1 && !RealVal.isZero()) {
      TokError("floating point value out of range");
      return MatchOperand_ParseFail;
    }
    Operands.push_back(ARM64Operand::CreateFPImm(Val, S, getContext()));
    return MatchOperand_Success;
  }
  if (Tok.is(AsmToken::Integer)) {
    int64_t Val;
    if (!isNegative && Tok.getString().startswith("0x")) {
      Val = Tok.getIntVal();
      if (Val > 255 || Val < 0) {
        TokError("encoded floating point value out of range");
        return MatchOperand_ParseFail;
      }
    } else {
      APFloat RealVal(APFloat::IEEEdouble, Tok.getString());
      uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
      // If we had a '-' in front, toggle the sign bit.
      IntVal ^= (uint64_t)isNegative << 63;
      Val = ARM64_AM::getFP64Imm(APInt(64, IntVal));
    }
    Parser.Lex(); // Eat the token.
    Operands.push_back(ARM64Operand::CreateFPImm(Val, S, getContext()));
    return MatchOperand_Success;
  }

  TokError("invalid floating point immediate");
  return MatchOperand_ParseFail;
}

/// parseCondCodeString - Parse a Condition Code string.
unsigned ARM64AsmParser::parseCondCodeString(StringRef Cond) {
  unsigned CC = StringSwitch<unsigned>(Cond)
                    .Case("eq", ARM64CC::EQ)
                    .Case("ne", ARM64CC::NE)
                    .Case("cs", ARM64CC::CS)
                    .Case("hs", ARM64CC::CS)
                    .Case("cc", ARM64CC::CC)
                    .Case("lo", ARM64CC::CC)
                    .Case("mi", ARM64CC::MI)
                    .Case("pl", ARM64CC::PL)
                    .Case("vs", ARM64CC::VS)
                    .Case("vc", ARM64CC::VC)
                    .Case("hi", ARM64CC::HI)
                    .Case("ls", ARM64CC::LS)
                    .Case("ge", ARM64CC::GE)
                    .Case("lt", ARM64CC::LT)
                    .Case("gt", ARM64CC::GT)
                    .Case("le", ARM64CC::LE)
                    .Case("al", ARM64CC::AL)
                // Upper case works too. Not mixed case, though.
                    .Case("EQ", ARM64CC::EQ)
                    .Case("NE", ARM64CC::NE)
                    .Case("CS", ARM64CC::CS)
                    .Case("HS", ARM64CC::CS)
                    .Case("CC", ARM64CC::CC)
                    .Case("LO", ARM64CC::CC)
                    .Case("MI", ARM64CC::MI)
                    .Case("PL", ARM64CC::PL)
                    .Case("VS", ARM64CC::VS)
                    .Case("VC", ARM64CC::VC)
                    .Case("HI", ARM64CC::HI)
                    .Case("LS", ARM64CC::LS)
                    .Case("GE", ARM64CC::GE)
                    .Case("LT", ARM64CC::LT)
                    .Case("GT", ARM64CC::GT)
                    .Case("LE", ARM64CC::LE)
                    .Case("AL", ARM64CC::AL)
                    .Default(~0U);
  return CC;
}

/// parseCondCode - Parse a Condition Code operand.
bool ARM64AsmParser::parseCondCode(OperandVector &Operands,
                                   bool invertCondCode) {
  SMLoc S = getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  StringRef Cond = Tok.getString();
  unsigned CC = parseCondCodeString(Cond);
  if (CC == ~0U)
    return TokError("invalid condition code");
  Parser.Lex(); // Eat identifier token.

  if (invertCondCode)
    CC = ARM64CC::getInvertedCondCode(ARM64CC::CondCode(CC));

  const MCExpr *CCExpr = MCConstantExpr::Create(CC, getContext());
  Operands.push_back(
      ARM64Operand::CreateImm(CCExpr, S, getLoc(), getContext()));
  return false;
}

/// ParseOptionalShift - Some operands take an optional shift argument. Parse
/// them if present.
bool ARM64AsmParser::parseOptionalShift(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();
  ARM64_AM::ShiftType ShOp = StringSwitch<ARM64_AM::ShiftType>(Tok.getString())
                                 .Case("lsl", ARM64_AM::LSL)
                                 .Case("lsr", ARM64_AM::LSR)
                                 .Case("asr", ARM64_AM::ASR)
                                 .Case("ror", ARM64_AM::ROR)
                                 .Case("msl", ARM64_AM::MSL)
                                 .Case("LSL", ARM64_AM::LSL)
                                 .Case("LSR", ARM64_AM::LSR)
                                 .Case("ASR", ARM64_AM::ASR)
                                 .Case("ROR", ARM64_AM::ROR)
                                 .Case("MSL", ARM64_AM::MSL)
                                 .Default(ARM64_AM::InvalidShift);
  if (ShOp == ARM64_AM::InvalidShift)
    return true;

  SMLoc S = Tok.getLoc();
  Parser.Lex();

  // We expect a number here.
  if (getLexer().isNot(AsmToken::Hash))
    return TokError("immediate value expected for shifter operand");
  Parser.Lex(); // Eat the '#'.

  SMLoc ExprLoc = getLoc();
  const MCExpr *ImmVal;
  if (getParser().parseExpression(ImmVal))
    return true;

  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
  if (!MCE)
    return TokError("immediate value expected for shifter operand");

  if ((MCE->getValue() & 0x3f) != MCE->getValue())
    return Error(ExprLoc, "immediate value too large for shifter operand");

  SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  Operands.push_back(
      ARM64Operand::CreateShifter(ShOp, MCE->getValue(), S, E, getContext()));
  return false;
}

/// parseOptionalExtend - Some operands take an optional extend argument. Parse
/// them if present.
bool ARM64AsmParser::parseOptionalExtend(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();
  ARM64_AM::ExtendType ExtOp =
      StringSwitch<ARM64_AM::ExtendType>(Tok.getString())
          .Case("uxtb", ARM64_AM::UXTB)
          .Case("uxth", ARM64_AM::UXTH)
          .Case("uxtw", ARM64_AM::UXTW)
          .Case("uxtx", ARM64_AM::UXTX)
          .Case("lsl", ARM64_AM::UXTX) // Alias for UXTX
          .Case("sxtb", ARM64_AM::SXTB)
          .Case("sxth", ARM64_AM::SXTH)
          .Case("sxtw", ARM64_AM::SXTW)
          .Case("sxtx", ARM64_AM::SXTX)
          .Case("UXTB", ARM64_AM::UXTB)
          .Case("UXTH", ARM64_AM::UXTH)
          .Case("UXTW", ARM64_AM::UXTW)
          .Case("UXTX", ARM64_AM::UXTX)
          .Case("LSL", ARM64_AM::UXTX) // Alias for UXTX
          .Case("SXTB", ARM64_AM::SXTB)
          .Case("SXTH", ARM64_AM::SXTH)
          .Case("SXTW", ARM64_AM::SXTW)
          .Case("SXTX", ARM64_AM::SXTX)
          .Default(ARM64_AM::InvalidExtend);
  if (ExtOp == ARM64_AM::InvalidExtend)
    return true;

  SMLoc S = Tok.getLoc();
  Parser.Lex();

  if (getLexer().is(AsmToken::EndOfStatement) ||
      getLexer().is(AsmToken::Comma)) {
    SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(
        ARM64Operand::CreateExtend(ExtOp, 0, S, E, getContext()));
    return false;
  }

  if (getLexer().isNot(AsmToken::Hash)) {
    SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(
        ARM64Operand::CreateExtend(ExtOp, 0, S, E, getContext()));
    return false;
  }

  Parser.Lex(); // Eat the '#'.

  const MCExpr *ImmVal;
  if (getParser().parseExpression(ImmVal))
    return true;

  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
  if (!MCE)
    return TokError("immediate value expected for extend operand");

  SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  Operands.push_back(
      ARM64Operand::CreateExtend(ExtOp, MCE->getValue(), S, E, getContext()));
  return false;
}

/// parseSysAlias - The IC, DC, AT, and TLBI instructions are simple aliases for
/// the SYS instruction. Parse them specially so that we create a SYS MCInst.
bool ARM64AsmParser::parseSysAlias(StringRef Name, SMLoc NameLoc,
                                   OperandVector &Operands) {
  if (Name.find('.') != StringRef::npos)
    return TokError("invalid operand");

  Mnemonic = Name;
  Operands.push_back(
      ARM64Operand::CreateToken("sys", false, NameLoc, getContext()));

  const AsmToken &Tok = Parser.getTok();
  StringRef Op = Tok.getString();
  SMLoc S = Tok.getLoc();

  const MCExpr *Expr = 0;

#define SYS_ALIAS(op1, Cn, Cm, op2)                                            \
  do {                                                                         \
    Expr = MCConstantExpr::Create(op1, getContext());                          \
    Operands.push_back(                                                        \
        ARM64Operand::CreateImm(Expr, S, getLoc(), getContext()));             \
    Operands.push_back(                                                        \
        ARM64Operand::CreateSysCR(Cn, S, getLoc(), getContext()));             \
    Operands.push_back(                                                        \
        ARM64Operand::CreateSysCR(Cm, S, getLoc(), getContext()));             \
    Expr = MCConstantExpr::Create(op2, getContext());                          \
    Operands.push_back(                                                        \
        ARM64Operand::CreateImm(Expr, S, getLoc(), getContext()));             \
  } while (0)

  if (Mnemonic == "ic") {
    if (!Op.compare_lower("ialluis")) {
      // SYS #0, C7, C1, #0
      SYS_ALIAS(0, 7, 1, 0);
    } else if (!Op.compare_lower("iallu")) {
      // SYS #0, C7, C5, #0
      SYS_ALIAS(0, 7, 5, 0);
    } else if (!Op.compare_lower("ivau")) {
      // SYS #3, C7, C5, #1
      SYS_ALIAS(3, 7, 5, 1);
    } else {
      return TokError("invalid operand for IC instruction");
    }
  } else if (Mnemonic == "dc") {
    if (!Op.compare_lower("zva")) {
      // SYS #3, C7, C4, #1
      SYS_ALIAS(3, 7, 4, 1);
    } else if (!Op.compare_lower("ivac")) {
      // SYS #3, C7, C6, #1
      SYS_ALIAS(0, 7, 6, 1);
    } else if (!Op.compare_lower("isw")) {
      // SYS #0, C7, C6, #2
      SYS_ALIAS(0, 7, 6, 2);
    } else if (!Op.compare_lower("cvac")) {
      // SYS #3, C7, C10, #1
      SYS_ALIAS(3, 7, 10, 1);
    } else if (!Op.compare_lower("csw")) {
      // SYS #0, C7, C10, #2
      SYS_ALIAS(0, 7, 10, 2);
    } else if (!Op.compare_lower("cvau")) {
      // SYS #3, C7, C11, #1
      SYS_ALIAS(3, 7, 11, 1);
    } else if (!Op.compare_lower("civac")) {
      // SYS #3, C7, C14, #1
      SYS_ALIAS(3, 7, 14, 1);
    } else if (!Op.compare_lower("cisw")) {
      // SYS #0, C7, C14, #2
      SYS_ALIAS(0, 7, 14, 2);
    } else {
      return TokError("invalid operand for DC instruction");
    }
  } else if (Mnemonic == "at") {
    if (!Op.compare_lower("s1e1r")) {
      // SYS #0, C7, C8, #0
      SYS_ALIAS(0, 7, 8, 0);
    } else if (!Op.compare_lower("s1e2r")) {
      // SYS #4, C7, C8, #0
      SYS_ALIAS(4, 7, 8, 0);
    } else if (!Op.compare_lower("s1e3r")) {
      // SYS #6, C7, C8, #0
      SYS_ALIAS(6, 7, 8, 0);
    } else if (!Op.compare_lower("s1e1w")) {
      // SYS #0, C7, C8, #1
      SYS_ALIAS(0, 7, 8, 1);
    } else if (!Op.compare_lower("s1e2w")) {
      // SYS #4, C7, C8, #1
      SYS_ALIAS(4, 7, 8, 1);
    } else if (!Op.compare_lower("s1e3w")) {
      // SYS #6, C7, C8, #1
      SYS_ALIAS(6, 7, 8, 1);
    } else if (!Op.compare_lower("s1e0r")) {
      // SYS #0, C7, C8, #3
      SYS_ALIAS(0, 7, 8, 2);
    } else if (!Op.compare_lower("s1e0w")) {
      // SYS #0, C7, C8, #3
      SYS_ALIAS(0, 7, 8, 3);
    } else if (!Op.compare_lower("s12e1r")) {
      // SYS #4, C7, C8, #4
      SYS_ALIAS(4, 7, 8, 4);
    } else if (!Op.compare_lower("s12e1w")) {
      // SYS #4, C7, C8, #5
      SYS_ALIAS(4, 7, 8, 5);
    } else if (!Op.compare_lower("s12e0r")) {
      // SYS #4, C7, C8, #6
      SYS_ALIAS(4, 7, 8, 6);
    } else if (!Op.compare_lower("s12e0w")) {
      // SYS #4, C7, C8, #7
      SYS_ALIAS(4, 7, 8, 7);
    } else {
      return TokError("invalid operand for AT instruction");
    }
  } else if (Mnemonic == "tlbi") {
    if (!Op.compare_lower("vmalle1is")) {
      // SYS #0, C8, C3, #0
      SYS_ALIAS(0, 8, 3, 0);
    } else if (!Op.compare_lower("alle2is")) {
      // SYS #4, C8, C3, #0
      SYS_ALIAS(4, 8, 3, 0);
    } else if (!Op.compare_lower("alle3is")) {
      // SYS #6, C8, C3, #0
      SYS_ALIAS(6, 8, 3, 0);
    } else if (!Op.compare_lower("vae1is")) {
      // SYS #0, C8, C3, #1
      SYS_ALIAS(0, 8, 3, 1);
    } else if (!Op.compare_lower("vae2is")) {
      // SYS #4, C8, C3, #1
      SYS_ALIAS(4, 8, 3, 1);
    } else if (!Op.compare_lower("vae3is")) {
      // SYS #6, C8, C3, #1
      SYS_ALIAS(6, 8, 3, 1);
    } else if (!Op.compare_lower("aside1is")) {
      // SYS #0, C8, C3, #2
      SYS_ALIAS(0, 8, 3, 2);
    } else if (!Op.compare_lower("vaae1is")) {
      // SYS #0, C8, C3, #3
      SYS_ALIAS(0, 8, 3, 3);
    } else if (!Op.compare_lower("alle1is")) {
      // SYS #4, C8, C3, #4
      SYS_ALIAS(4, 8, 3, 4);
    } else if (!Op.compare_lower("vale1is")) {
      // SYS #0, C8, C3, #5
      SYS_ALIAS(0, 8, 3, 5);
    } else if (!Op.compare_lower("vaale1is")) {
      // SYS #0, C8, C3, #7
      SYS_ALIAS(0, 8, 3, 7);
    } else if (!Op.compare_lower("vmalle1")) {
      // SYS #0, C8, C7, #0
      SYS_ALIAS(0, 8, 7, 0);
    } else if (!Op.compare_lower("alle2")) {
      // SYS #4, C8, C7, #0
      SYS_ALIAS(4, 8, 7, 0);
    } else if (!Op.compare_lower("vale2is")) {
      // SYS #4, C8, C3, #5
      SYS_ALIAS(4, 8, 3, 5);
    } else if (!Op.compare_lower("vale3is")) {
      // SYS #6, C8, C3, #5
      SYS_ALIAS(6, 8, 3, 5);
    } else if (!Op.compare_lower("alle3")) {
      // SYS #6, C8, C7, #0
      SYS_ALIAS(6, 8, 7, 0);
    } else if (!Op.compare_lower("vae1")) {
      // SYS #0, C8, C7, #1
      SYS_ALIAS(0, 8, 7, 1);
    } else if (!Op.compare_lower("vae2")) {
      // SYS #4, C8, C7, #1
      SYS_ALIAS(4, 8, 7, 1);
    } else if (!Op.compare_lower("vae3")) {
      // SYS #6, C8, C7, #1
      SYS_ALIAS(6, 8, 7, 1);
    } else if (!Op.compare_lower("aside1")) {
      // SYS #0, C8, C7, #2
      SYS_ALIAS(0, 8, 7, 2);
    } else if (!Op.compare_lower("vaae1")) {
      // SYS #0, C8, C7, #3
      SYS_ALIAS(0, 8, 7, 3);
    } else if (!Op.compare_lower("alle1")) {
      // SYS #4, C8, C7, #4
      SYS_ALIAS(4, 8, 7, 4);
    } else if (!Op.compare_lower("vale1")) {
      // SYS #0, C8, C7, #5
      SYS_ALIAS(0, 8, 7, 5);
    } else if (!Op.compare_lower("vale2")) {
      // SYS #4, C8, C7, #5
      SYS_ALIAS(4, 8, 7, 5);
    } else if (!Op.compare_lower("vale3")) {
      // SYS #6, C8, C7, #5
      SYS_ALIAS(6, 8, 7, 5);
    } else if (!Op.compare_lower("vaale1")) {
      // SYS #0, C8, C7, #7
      SYS_ALIAS(0, 8, 7, 7);
    } else if (!Op.compare_lower("ipas2e1")) {
      // SYS #4, C8, C4, #1
      SYS_ALIAS(4, 8, 4, 1);
    } else if (!Op.compare_lower("ipas2le1")) {
      // SYS #4, C8, C4, #5
      SYS_ALIAS(4, 8, 4, 5);
    } else if (!Op.compare_lower("vmalls12e1")) {
      // SYS #4, C8, C7, #6
      SYS_ALIAS(4, 8, 7, 6);
    } else if (!Op.compare_lower("vmalls12e1is")) {
      // SYS #4, C8, C3, #6
      SYS_ALIAS(4, 8, 3, 6);
    } else {
      return TokError("invalid operand for TLBI instruction");
    }
  }

#undef SYS_ALIAS

  Parser.Lex(); // Eat operand.

  // Check for the optional register operand.
  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat comma.

    if (Tok.isNot(AsmToken::Identifier) || parseRegister(Operands))
      return TokError("expected register operand");
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Parser.eatToEndOfStatement();
    return TokError("unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseBarrierOperand(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();

  // Can be either a #imm style literal or an option name
  if (Tok.is(AsmToken::Hash)) {
    // Immediate operand.
    Parser.Lex(); // Eat the '#'
    const MCExpr *ImmVal;
    SMLoc ExprLoc = getLoc();
    if (getParser().parseExpression(ImmVal))
      return MatchOperand_ParseFail;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      Error(ExprLoc, "immediate value expected for barrier operand");
      return MatchOperand_ParseFail;
    }
    if (MCE->getValue() < 0 || MCE->getValue() > 15) {
      Error(ExprLoc, "barrier operand out of range");
      return MatchOperand_ParseFail;
    }
    Operands.push_back(
        ARM64Operand::CreateBarrier(MCE->getValue(), ExprLoc, getContext()));
    return MatchOperand_Success;
  }

  if (Tok.isNot(AsmToken::Identifier)) {
    TokError("invalid operand for instruction");
    return MatchOperand_ParseFail;
  }

  unsigned Opt = StringSwitch<unsigned>(Tok.getString())
                     .Case("oshld", ARM64SYS::OSHLD)
                     .Case("oshst", ARM64SYS::OSHST)
                     .Case("osh", ARM64SYS::OSH)
                     .Case("nshld", ARM64SYS::NSHLD)
                     .Case("nshst", ARM64SYS::NSHST)
                     .Case("nsh", ARM64SYS::NSH)
                     .Case("ishld", ARM64SYS::ISHLD)
                     .Case("ishst", ARM64SYS::ISHST)
                     .Case("ish", ARM64SYS::ISH)
                     .Case("ld", ARM64SYS::LD)
                     .Case("st", ARM64SYS::ST)
                     .Case("sy", ARM64SYS::SY)
                     .Default(ARM64SYS::InvalidBarrier);
  if (Opt == ARM64SYS::InvalidBarrier) {
    TokError("invalid barrier option name");
    return MatchOperand_ParseFail;
  }

  // The only valid named option for ISB is 'sy'
  if (Mnemonic == "isb" && Opt != ARM64SYS::SY) {
    TokError("'sy' or #imm operand expected");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(ARM64Operand::CreateBarrier(Opt, getLoc(), getContext()));
  Parser.Lex(); // Consume the option

  return MatchOperand_Success;
}

ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseSystemRegister(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();

  // It can be specified as a symbolic name.
  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  auto ID = Tok.getString().lower();
  ARM64SYS::SystemRegister Reg =
      StringSwitch<ARM64SYS::SystemRegister>(ID)
          .Case("spsr_el1", ARM64SYS::SPSR_svc)
          .Case("spsr_svc", ARM64SYS::SPSR_svc)
          .Case("elr_el1", ARM64SYS::ELR_EL1)
          .Case("sp_el0", ARM64SYS::SP_EL0)
          .Case("spsel", ARM64SYS::SPSel)
          .Case("daif", ARM64SYS::DAIF)
          .Case("currentel", ARM64SYS::CurrentEL)
          .Case("nzcv", ARM64SYS::NZCV)
          .Case("fpcr", ARM64SYS::FPCR)
          .Case("fpsr", ARM64SYS::FPSR)
          .Case("dspsr", ARM64SYS::DSPSR)
          .Case("dlr", ARM64SYS::DLR)
          .Case("spsr_el2", ARM64SYS::SPSR_hyp)
          .Case("spsr_hyp", ARM64SYS::SPSR_hyp)
          .Case("elr_el2", ARM64SYS::ELR_EL2)
          .Case("sp_el1", ARM64SYS::SP_EL1)
          .Case("spsr_irq", ARM64SYS::SPSR_irq)
          .Case("spsr_abt", ARM64SYS::SPSR_abt)
          .Case("spsr_und", ARM64SYS::SPSR_und)
          .Case("spsr_fiq", ARM64SYS::SPSR_fiq)
          .Case("spsr_el3", ARM64SYS::SPSR_EL3)
          .Case("elr_el3", ARM64SYS::ELR_EL3)
          .Case("sp_el2", ARM64SYS::SP_EL2)
          .Case("midr_el1", ARM64SYS::MIDR_EL1)
          .Case("ctr_el0", ARM64SYS::CTR_EL0)
          .Case("mpidr_el1", ARM64SYS::MPIDR_EL1)
          .Case("ecoidr_el1", ARM64SYS::ECOIDR_EL1)
          .Case("dczid_el0", ARM64SYS::DCZID_EL0)
          .Case("mvfr0_el1", ARM64SYS::MVFR0_EL1)
          .Case("mvfr1_el1", ARM64SYS::MVFR1_EL1)
          .Case("id_aa64pfr0_el1", ARM64SYS::ID_AA64PFR0_EL1)
          .Case("id_aa64pfr1_el1", ARM64SYS::ID_AA64PFR1_EL1)
          .Case("id_aa64dfr0_el1", ARM64SYS::ID_AA64DFR0_EL1)
          .Case("id_aa64dfr1_el1", ARM64SYS::ID_AA64DFR1_EL1)
          .Case("id_aa64isar0_el1", ARM64SYS::ID_AA64ISAR0_EL1)
          .Case("id_aa64isar1_el1", ARM64SYS::ID_AA64ISAR1_EL1)
          .Case("id_aa64mmfr0_el1", ARM64SYS::ID_AA64MMFR0_EL1)
          .Case("id_aa64mmfr1_el1", ARM64SYS::ID_AA64MMFR1_EL1)
          .Case("ccsidr_el1", ARM64SYS::CCSIDR_EL1)
          .Case("clidr_el1", ARM64SYS::CLIDR_EL1)
          .Case("aidr_el1", ARM64SYS::AIDR_EL1)
          .Case("csselr_el1", ARM64SYS::CSSELR_EL1)
          .Case("vpidr_el2", ARM64SYS::VPIDR_EL2)
          .Case("vmpidr_el2", ARM64SYS::VMPIDR_EL2)
          .Case("sctlr_el1", ARM64SYS::SCTLR_EL1)
          .Case("sctlr_el2", ARM64SYS::SCTLR_EL2)
          .Case("sctlr_el3", ARM64SYS::SCTLR_EL3)
          .Case("actlr_el1", ARM64SYS::ACTLR_EL1)
          .Case("actlr_el2", ARM64SYS::ACTLR_EL2)
          .Case("actlr_el3", ARM64SYS::ACTLR_EL3)
          .Case("cpacr_el1", ARM64SYS::CPACR_EL1)
          .Case("cptr_el2", ARM64SYS::CPTR_EL2)
          .Case("cptr_el3", ARM64SYS::CPTR_EL3)
          .Case("scr_el3", ARM64SYS::SCR_EL3)
          .Case("hcr_el2", ARM64SYS::HCR_EL2)
          .Case("mdcr_el2", ARM64SYS::MDCR_EL2)
          .Case("mdcr_el3", ARM64SYS::MDCR_EL3)
          .Case("hstr_el2", ARM64SYS::HSTR_EL2)
          .Case("hacr_el2", ARM64SYS::HACR_EL2)
          .Case("ttbr0_el1", ARM64SYS::TTBR0_EL1)
          .Case("ttbr1_el1", ARM64SYS::TTBR1_EL1)
          .Case("ttbr0_el2", ARM64SYS::TTBR0_EL2)
          .Case("ttbr0_el3", ARM64SYS::TTBR0_EL3)
          .Case("vttbr_el2", ARM64SYS::VTTBR_EL2)
          .Case("tcr_el1", ARM64SYS::TCR_EL1)
          .Case("tcr_el2", ARM64SYS::TCR_EL2)
          .Case("tcr_el3", ARM64SYS::TCR_EL3)
          .Case("vtcr_el2", ARM64SYS::VTCR_EL2)
          .Case("adfsr_el1", ARM64SYS::ADFSR_EL1)
          .Case("aifsr_el1", ARM64SYS::AIFSR_EL1)
          .Case("adfsr_el2", ARM64SYS::ADFSR_EL2)
          .Case("aifsr_el2", ARM64SYS::AIFSR_EL2)
          .Case("adfsr_el3", ARM64SYS::ADFSR_EL3)
          .Case("aifsr_el3", ARM64SYS::AIFSR_EL3)
          .Case("esr_el1", ARM64SYS::ESR_EL1)
          .Case("esr_el2", ARM64SYS::ESR_EL2)
          .Case("esr_el3", ARM64SYS::ESR_EL3)
          .Case("far_el1", ARM64SYS::FAR_EL1)
          .Case("far_el2", ARM64SYS::FAR_EL2)
          .Case("far_el3", ARM64SYS::FAR_EL3)
          .Case("hpfar_el2", ARM64SYS::HPFAR_EL2)
          .Case("par_el1", ARM64SYS::PAR_EL1)
          .Case("mair_el1", ARM64SYS::MAIR_EL1)
          .Case("mair_el2", ARM64SYS::MAIR_EL2)
          .Case("mair_el3", ARM64SYS::MAIR_EL3)
          .Case("amair_el1", ARM64SYS::AMAIR_EL1)
          .Case("amair_el2", ARM64SYS::AMAIR_EL2)
          .Case("amair_el3", ARM64SYS::AMAIR_EL3)
          .Case("vbar_el1", ARM64SYS::VBAR_EL1)
          .Case("vbar_el2", ARM64SYS::VBAR_EL2)
          .Case("vbar_el3", ARM64SYS::VBAR_EL3)
          .Case("rvbar_el1", ARM64SYS::RVBAR_EL1)
          .Case("rvbar_el2", ARM64SYS::RVBAR_EL2)
          .Case("rvbar_el3", ARM64SYS::RVBAR_EL3)
          .Case("isr_el1", ARM64SYS::ISR_EL1)
          .Case("contextidr_el1", ARM64SYS::CONTEXTIDR_EL1)
          .Case("tpidr_el0", ARM64SYS::TPIDR_EL0)
          .Case("tpidrro_el0", ARM64SYS::TPIDRRO_EL0)
          .Case("tpidr_el1", ARM64SYS::TPIDR_EL1)
          .Case("tpidr_el2", ARM64SYS::TPIDR_EL2)
          .Case("tpidr_el3", ARM64SYS::TPIDR_EL3)
          .Case("teecr32_el1", ARM64SYS::TEECR32_EL1)
          .Case("cntfrq_el0", ARM64SYS::CNTFRQ_EL0)
          .Case("cntpct_el0", ARM64SYS::CNTPCT_EL0)
          .Case("cntvct_el0", ARM64SYS::CNTVCT_EL0)
          .Case("cntvoff_el2", ARM64SYS::CNTVOFF_EL2)
          .Case("cntkctl_el1", ARM64SYS::CNTKCTL_EL1)
          .Case("cnthctl_el2", ARM64SYS::CNTHCTL_EL2)
          .Case("cntp_tval_el0", ARM64SYS::CNTP_TVAL_EL0)
          .Case("cntp_ctl_el0", ARM64SYS::CNTP_CTL_EL0)
          .Case("cntp_cval_el0", ARM64SYS::CNTP_CVAL_EL0)
          .Case("cntv_tval_el0", ARM64SYS::CNTV_TVAL_EL0)
          .Case("cntv_ctl_el0", ARM64SYS::CNTV_CTL_EL0)
          .Case("cntv_cval_el0", ARM64SYS::CNTV_CVAL_EL0)
          .Case("cnthp_tval_el2", ARM64SYS::CNTHP_TVAL_EL2)
          .Case("cnthp_ctl_el2", ARM64SYS::CNTHP_CTL_EL2)
          .Case("cnthp_cval_el2", ARM64SYS::CNTHP_CVAL_EL2)
          .Case("cntps_tval_el1", ARM64SYS::CNTPS_TVAL_EL1)
          .Case("cntps_ctl_el1", ARM64SYS::CNTPS_CTL_EL1)
          .Case("cntps_cval_el1", ARM64SYS::CNTPS_CVAL_EL1)
          .Case("dacr32_el2", ARM64SYS::DACR32_EL2)
          .Case("ifsr32_el2", ARM64SYS::IFSR32_EL2)
          .Case("teehbr32_el1", ARM64SYS::TEEHBR32_EL1)
          .Case("sder32_el3", ARM64SYS::SDER32_EL3)
          .Case("fpexc32_el2", ARM64SYS::FPEXC32_EL2)
          .Case("current_el", ARM64SYS::CurrentEL)
          .Case("pmevcntr0_el0", ARM64SYS::PMEVCNTR0_EL0)
          .Case("pmevcntr1_el0", ARM64SYS::PMEVCNTR1_EL0)
          .Case("pmevcntr2_el0", ARM64SYS::PMEVCNTR2_EL0)
          .Case("pmevcntr3_el0", ARM64SYS::PMEVCNTR3_EL0)
          .Case("pmevcntr4_el0", ARM64SYS::PMEVCNTR4_EL0)
          .Case("pmevcntr5_el0", ARM64SYS::PMEVCNTR5_EL0)
          .Case("pmevcntr6_el0", ARM64SYS::PMEVCNTR6_EL0)
          .Case("pmevcntr7_el0", ARM64SYS::PMEVCNTR7_EL0)
          .Case("pmevcntr8_el0", ARM64SYS::PMEVCNTR8_EL0)
          .Case("pmevcntr9_el0", ARM64SYS::PMEVCNTR9_EL0)
          .Case("pmevcntr10_el0", ARM64SYS::PMEVCNTR10_EL0)
          .Case("pmevcntr11_el0", ARM64SYS::PMEVCNTR11_EL0)
          .Case("pmevcntr12_el0", ARM64SYS::PMEVCNTR12_EL0)
          .Case("pmevcntr13_el0", ARM64SYS::PMEVCNTR13_EL0)
          .Case("pmevcntr14_el0", ARM64SYS::PMEVCNTR14_EL0)
          .Case("pmevcntr15_el0", ARM64SYS::PMEVCNTR15_EL0)
          .Case("pmevcntr16_el0", ARM64SYS::PMEVCNTR16_EL0)
          .Case("pmevcntr17_el0", ARM64SYS::PMEVCNTR17_EL0)
          .Case("pmevcntr18_el0", ARM64SYS::PMEVCNTR18_EL0)
          .Case("pmevcntr19_el0", ARM64SYS::PMEVCNTR19_EL0)
          .Case("pmevcntr20_el0", ARM64SYS::PMEVCNTR20_EL0)
          .Case("pmevcntr21_el0", ARM64SYS::PMEVCNTR21_EL0)
          .Case("pmevcntr22_el0", ARM64SYS::PMEVCNTR22_EL0)
          .Case("pmevcntr23_el0", ARM64SYS::PMEVCNTR23_EL0)
          .Case("pmevcntr24_el0", ARM64SYS::PMEVCNTR24_EL0)
          .Case("pmevcntr25_el0", ARM64SYS::PMEVCNTR25_EL0)
          .Case("pmevcntr26_el0", ARM64SYS::PMEVCNTR26_EL0)
          .Case("pmevcntr27_el0", ARM64SYS::PMEVCNTR27_EL0)
          .Case("pmevcntr28_el0", ARM64SYS::PMEVCNTR28_EL0)
          .Case("pmevcntr29_el0", ARM64SYS::PMEVCNTR29_EL0)
          .Case("pmevcntr30_el0", ARM64SYS::PMEVCNTR30_EL0)
          .Case("pmevtyper0_el0", ARM64SYS::PMEVTYPER0_EL0)
          .Case("pmevtyper1_el0", ARM64SYS::PMEVTYPER1_EL0)
          .Case("pmevtyper2_el0", ARM64SYS::PMEVTYPER2_EL0)
          .Case("pmevtyper3_el0", ARM64SYS::PMEVTYPER3_EL0)
          .Case("pmevtyper4_el0", ARM64SYS::PMEVTYPER4_EL0)
          .Case("pmevtyper5_el0", ARM64SYS::PMEVTYPER5_EL0)
          .Case("pmevtyper6_el0", ARM64SYS::PMEVTYPER6_EL0)
          .Case("pmevtyper7_el0", ARM64SYS::PMEVTYPER7_EL0)
          .Case("pmevtyper8_el0", ARM64SYS::PMEVTYPER8_EL0)
          .Case("pmevtyper9_el0", ARM64SYS::PMEVTYPER9_EL0)
          .Case("pmevtyper10_el0", ARM64SYS::PMEVTYPER10_EL0)
          .Case("pmevtyper11_el0", ARM64SYS::PMEVTYPER11_EL0)
          .Case("pmevtyper12_el0", ARM64SYS::PMEVTYPER12_EL0)
          .Case("pmevtyper13_el0", ARM64SYS::PMEVTYPER13_EL0)
          .Case("pmevtyper14_el0", ARM64SYS::PMEVTYPER14_EL0)
          .Case("pmevtyper15_el0", ARM64SYS::PMEVTYPER15_EL0)
          .Case("pmevtyper16_el0", ARM64SYS::PMEVTYPER16_EL0)
          .Case("pmevtyper17_el0", ARM64SYS::PMEVTYPER17_EL0)
          .Case("pmevtyper18_el0", ARM64SYS::PMEVTYPER18_EL0)
          .Case("pmevtyper19_el0", ARM64SYS::PMEVTYPER19_EL0)
          .Case("pmevtyper20_el0", ARM64SYS::PMEVTYPER20_EL0)
          .Case("pmevtyper21_el0", ARM64SYS::PMEVTYPER21_EL0)
          .Case("pmevtyper22_el0", ARM64SYS::PMEVTYPER22_EL0)
          .Case("pmevtyper23_el0", ARM64SYS::PMEVTYPER23_EL0)
          .Case("pmevtyper24_el0", ARM64SYS::PMEVTYPER24_EL0)
          .Case("pmevtyper25_el0", ARM64SYS::PMEVTYPER25_EL0)
          .Case("pmevtyper26_el0", ARM64SYS::PMEVTYPER26_EL0)
          .Case("pmevtyper27_el0", ARM64SYS::PMEVTYPER27_EL0)
          .Case("pmevtyper28_el0", ARM64SYS::PMEVTYPER28_EL0)
          .Case("pmevtyper29_el0", ARM64SYS::PMEVTYPER29_EL0)
          .Case("pmevtyper30_el0", ARM64SYS::PMEVTYPER30_EL0)
          .Case("pmccfiltr_el0", ARM64SYS::PMCCFILTR_EL0)
          .Case("rmr_el3", ARM64SYS::RMR_EL3)
          .Case("rmr_el2", ARM64SYS::RMR_EL2)
          .Case("rmr_el1", ARM64SYS::RMR_EL1)
          .Case("cpm_ioacc_ctl_el3", ARM64SYS::CPM_IOACC_CTL_EL3)
          .Case("mdccsr_el0", ARM64SYS::MDCCSR_EL0)
          .Case("mdccint_el1", ARM64SYS::MDCCINT_EL1)
          .Case("dbgdtr_el0", ARM64SYS::DBGDTR_EL0)
          .Case("dbgdtrrx_el0", ARM64SYS::DBGDTRRX_EL0)
          .Case("dbgdtrtx_el0", ARM64SYS::DBGDTRTX_EL0)
          .Case("dbgvcr32_el2", ARM64SYS::DBGVCR32_EL2)
          .Case("osdtrrx_el1", ARM64SYS::OSDTRRX_EL1)
          .Case("mdscr_el1", ARM64SYS::MDSCR_EL1)
          .Case("osdtrtx_el1", ARM64SYS::OSDTRTX_EL1)
          .Case("oseccr_el11", ARM64SYS::OSECCR_EL11)
          .Case("dbgbvr0_el1", ARM64SYS::DBGBVR0_EL1)
          .Case("dbgbvr1_el1", ARM64SYS::DBGBVR1_EL1)
          .Case("dbgbvr2_el1", ARM64SYS::DBGBVR2_EL1)
          .Case("dbgbvr3_el1", ARM64SYS::DBGBVR3_EL1)
          .Case("dbgbvr4_el1", ARM64SYS::DBGBVR4_EL1)
          .Case("dbgbvr5_el1", ARM64SYS::DBGBVR5_EL1)
          .Case("dbgbvr6_el1", ARM64SYS::DBGBVR6_EL1)
          .Case("dbgbvr7_el1", ARM64SYS::DBGBVR7_EL1)
          .Case("dbgbvr8_el1", ARM64SYS::DBGBVR8_EL1)
          .Case("dbgbvr9_el1", ARM64SYS::DBGBVR9_EL1)
          .Case("dbgbvr10_el1", ARM64SYS::DBGBVR10_EL1)
          .Case("dbgbvr11_el1", ARM64SYS::DBGBVR11_EL1)
          .Case("dbgbvr12_el1", ARM64SYS::DBGBVR12_EL1)
          .Case("dbgbvr13_el1", ARM64SYS::DBGBVR13_EL1)
          .Case("dbgbvr14_el1", ARM64SYS::DBGBVR14_EL1)
          .Case("dbgbvr15_el1", ARM64SYS::DBGBVR15_EL1)
          .Case("dbgbcr0_el1", ARM64SYS::DBGBCR0_EL1)
          .Case("dbgbcr1_el1", ARM64SYS::DBGBCR1_EL1)
          .Case("dbgbcr2_el1", ARM64SYS::DBGBCR2_EL1)
          .Case("dbgbcr3_el1", ARM64SYS::DBGBCR3_EL1)
          .Case("dbgbcr4_el1", ARM64SYS::DBGBCR4_EL1)
          .Case("dbgbcr5_el1", ARM64SYS::DBGBCR5_EL1)
          .Case("dbgbcr6_el1", ARM64SYS::DBGBCR6_EL1)
          .Case("dbgbcr7_el1", ARM64SYS::DBGBCR7_EL1)
          .Case("dbgbcr8_el1", ARM64SYS::DBGBCR8_EL1)
          .Case("dbgbcr9_el1", ARM64SYS::DBGBCR9_EL1)
          .Case("dbgbcr10_el1", ARM64SYS::DBGBCR10_EL1)
          .Case("dbgbcr11_el1", ARM64SYS::DBGBCR11_EL1)
          .Case("dbgbcr12_el1", ARM64SYS::DBGBCR12_EL1)
          .Case("dbgbcr13_el1", ARM64SYS::DBGBCR13_EL1)
          .Case("dbgbcr14_el1", ARM64SYS::DBGBCR14_EL1)
          .Case("dbgbcr15_el1", ARM64SYS::DBGBCR15_EL1)
          .Case("dbgwvr0_el1", ARM64SYS::DBGWVR0_EL1)
          .Case("dbgwvr1_el1", ARM64SYS::DBGWVR1_EL1)
          .Case("dbgwvr2_el1", ARM64SYS::DBGWVR2_EL1)
          .Case("dbgwvr3_el1", ARM64SYS::DBGWVR3_EL1)
          .Case("dbgwvr4_el1", ARM64SYS::DBGWVR4_EL1)
          .Case("dbgwvr5_el1", ARM64SYS::DBGWVR5_EL1)
          .Case("dbgwvr6_el1", ARM64SYS::DBGWVR6_EL1)
          .Case("dbgwvr7_el1", ARM64SYS::DBGWVR7_EL1)
          .Case("dbgwvr8_el1", ARM64SYS::DBGWVR8_EL1)
          .Case("dbgwvr9_el1", ARM64SYS::DBGWVR9_EL1)
          .Case("dbgwvr10_el1", ARM64SYS::DBGWVR10_EL1)
          .Case("dbgwvr11_el1", ARM64SYS::DBGWVR11_EL1)
          .Case("dbgwvr12_el1", ARM64SYS::DBGWVR12_EL1)
          .Case("dbgwvr13_el1", ARM64SYS::DBGWVR13_EL1)
          .Case("dbgwvr14_el1", ARM64SYS::DBGWVR14_EL1)
          .Case("dbgwvr15_el1", ARM64SYS::DBGWVR15_EL1)
          .Case("dbgwcr0_el1", ARM64SYS::DBGWCR0_EL1)
          .Case("dbgwcr1_el1", ARM64SYS::DBGWCR1_EL1)
          .Case("dbgwcr2_el1", ARM64SYS::DBGWCR2_EL1)
          .Case("dbgwcr3_el1", ARM64SYS::DBGWCR3_EL1)
          .Case("dbgwcr4_el1", ARM64SYS::DBGWCR4_EL1)
          .Case("dbgwcr5_el1", ARM64SYS::DBGWCR5_EL1)
          .Case("dbgwcr6_el1", ARM64SYS::DBGWCR6_EL1)
          .Case("dbgwcr7_el1", ARM64SYS::DBGWCR7_EL1)
          .Case("dbgwcr8_el1", ARM64SYS::DBGWCR8_EL1)
          .Case("dbgwcr9_el1", ARM64SYS::DBGWCR9_EL1)
          .Case("dbgwcr10_el1", ARM64SYS::DBGWCR10_EL1)
          .Case("dbgwcr11_el1", ARM64SYS::DBGWCR11_EL1)
          .Case("dbgwcr12_el1", ARM64SYS::DBGWCR12_EL1)
          .Case("dbgwcr13_el1", ARM64SYS::DBGWCR13_EL1)
          .Case("dbgwcr14_el1", ARM64SYS::DBGWCR14_EL1)
          .Case("dbgwcr15_el1", ARM64SYS::DBGWCR15_EL1)
          .Case("mdrar_el1", ARM64SYS::MDRAR_EL1)
          .Case("oslar_el1", ARM64SYS::OSLAR_EL1)
          .Case("oslsr_el1", ARM64SYS::OSLSR_EL1)
          .Case("osdlr_el1", ARM64SYS::OSDLR_EL1)
          .Case("dbgprcr_el1", ARM64SYS::DBGPRCR_EL1)
          .Case("dbgclaimset_el1", ARM64SYS::DBGCLAIMSET_EL1)
          .Case("dbgclaimclr_el1", ARM64SYS::DBGCLAIMCLR_EL1)
          .Case("dbgauthstatus_el1", ARM64SYS::DBGAUTHSTATUS_EL1)
          .Case("dbgdevid2", ARM64SYS::DBGDEVID2)
          .Case("dbgdevid1", ARM64SYS::DBGDEVID1)
          .Case("dbgdevid0", ARM64SYS::DBGDEVID0)
          .Case("id_pfr0_el1", ARM64SYS::ID_PFR0_EL1)
          .Case("id_pfr1_el1", ARM64SYS::ID_PFR1_EL1)
          .Case("id_dfr0_el1", ARM64SYS::ID_DFR0_EL1)
          .Case("id_afr0_el1", ARM64SYS::ID_AFR0_EL1)
          .Case("id_isar0_el1", ARM64SYS::ID_ISAR0_EL1)
          .Case("id_isar1_el1", ARM64SYS::ID_ISAR1_EL1)
          .Case("id_isar2_el1", ARM64SYS::ID_ISAR2_EL1)
          .Case("id_isar3_el1", ARM64SYS::ID_ISAR3_EL1)
          .Case("id_isar4_el1", ARM64SYS::ID_ISAR4_EL1)
          .Case("id_isar5_el1", ARM64SYS::ID_ISAR5_EL1)
          .Case("afsr1_el1", ARM64SYS::AFSR1_EL1)
          .Case("afsr0_el1", ARM64SYS::AFSR0_EL1)
          .Case("revidr_el1", ARM64SYS::REVIDR_EL1)
          .Default(ARM64SYS::InvalidSystemReg);
  if (Reg != ARM64SYS::InvalidSystemReg) {
    // We matched a reg name, so create the operand.
    Operands.push_back(
        ARM64Operand::CreateSystemRegister(Reg, getLoc(), getContext()));
    Parser.Lex(); // Consume the register name.
    return MatchOperand_Success;
  }

  // Or we may have an identifier that encodes the sub-operands.
  // For example, s3_2_c15_c0_0.
  unsigned op0, op1, CRn, CRm, op2;
  std::string Desc = ID;
  if (std::sscanf(Desc.c_str(), "s%u_%u_c%u_c%u_%u", &op0, &op1, &CRn, &CRm,
                  &op2) != 5)
    return MatchOperand_NoMatch;
  if ((op0 != 2 && op0 != 3) || op1 > 7 || CRn > 15 || CRm > 15 || op2 > 7)
    return MatchOperand_NoMatch;

  unsigned Val = op0 << 14 | op1 << 11 | CRn << 7 | CRm << 3 | op2;
  Operands.push_back(
      ARM64Operand::CreateSystemRegister(Val, getLoc(), getContext()));
  Parser.Lex(); // Consume the register name.

  return MatchOperand_Success;
}

ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseCPSRField(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();

  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  ARM64SYS::CPSRField Field =
      StringSwitch<ARM64SYS::CPSRField>(Tok.getString().lower())
          .Case("spsel", ARM64SYS::cpsr_SPSel)
          .Case("daifset", ARM64SYS::cpsr_DAIFSet)
          .Case("daifclr", ARM64SYS::cpsr_DAIFClr)
          .Default(ARM64SYS::InvalidCPSRField);
  if (Field == ARM64SYS::InvalidCPSRField)
    return MatchOperand_NoMatch;
  Operands.push_back(
      ARM64Operand::CreateCPSRField(Field, getLoc(), getContext()));
  Parser.Lex(); // Consume the register name.

  return MatchOperand_Success;
}

/// tryParseVectorRegister - Parse a vector register operand.
bool ARM64AsmParser::tryParseVectorRegister(OperandVector &Operands) {
  if (Parser.getTok().isNot(AsmToken::Identifier))
    return true;

  SMLoc S = getLoc();
  // Check for a vector register specifier first.
  StringRef Kind;
  int64_t Reg = tryMatchVectorRegister(Kind);
  if (Reg == -1)
    return true;
  Operands.push_back(
      ARM64Operand::CreateReg(Reg, true, S, getLoc(), getContext()));
  // If there was an explicit qualifier, that goes on as a literal text
  // operand.
  if (!Kind.empty())
    Operands.push_back(ARM64Operand::CreateToken(Kind, false, S, getContext()));

  // If there is an index specifier following the register, parse that too.
  if (Parser.getTok().is(AsmToken::LBrac)) {
    SMLoc SIdx = getLoc();
    Parser.Lex(); // Eat left bracket token.

    const MCExpr *ImmVal;
    if (getParser().parseExpression(ImmVal))
      return MatchOperand_ParseFail;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      TokError("immediate value expected for vector index");
      return MatchOperand_ParseFail;
    }

    SMLoc E = getLoc();
    if (Parser.getTok().isNot(AsmToken::RBrac)) {
      Error(E, "']' expected");
      return MatchOperand_ParseFail;
    }

    Parser.Lex(); // Eat right bracket token.

    Operands.push_back(ARM64Operand::CreateVectorIndex(MCE->getValue(), SIdx, E,
                                                       getContext()));
  }

  return false;
}

/// parseRegister - Parse a non-vector register operand.
bool ARM64AsmParser::parseRegister(OperandVector &Operands) {
  SMLoc S = getLoc();
  // Try for a vector register.
  if (!tryParseVectorRegister(Operands))
    return false;

  // Try for a scalar register.
  int64_t Reg = tryParseRegister();
  if (Reg == -1)
    return true;
  Operands.push_back(
      ARM64Operand::CreateReg(Reg, false, S, getLoc(), getContext()));

  // A small number of instructions (FMOVXDhighr, for example) have "[1]"
  // as a string token in the instruction itself.
  if (getLexer().getKind() == AsmToken::LBrac) {
    SMLoc LBracS = getLoc();
    Parser.Lex();
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Integer)) {
      SMLoc IntS = getLoc();
      int64_t Val = Tok.getIntVal();
      if (Val == 1) {
        Parser.Lex();
        if (getLexer().getKind() == AsmToken::RBrac) {
          SMLoc RBracS = getLoc();
          Parser.Lex();
          Operands.push_back(
              ARM64Operand::CreateToken("[", false, LBracS, getContext()));
          Operands.push_back(
              ARM64Operand::CreateToken("1", false, IntS, getContext()));
          Operands.push_back(
              ARM64Operand::CreateToken("]", false, RBracS, getContext()));
          return false;
        }
      }
    }
  }

  return false;
}

/// tryParseNoIndexMemory - Custom parser method for memory operands that
///                         do not allow base regisrer writeback modes,
///                         or those that handle writeback separately from
///                         the memory operand (like the AdvSIMD ldX/stX
///                         instructions.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseNoIndexMemory(OperandVector &Operands) {
  if (Parser.getTok().isNot(AsmToken::LBrac))
    return MatchOperand_NoMatch;
  SMLoc S = getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier)) {
    Error(BaseRegTok.getLoc(), "register expected");
    return MatchOperand_ParseFail;
  }

  int64_t Reg = tryParseRegister();
  if (Reg == -1) {
    Error(BaseRegTok.getLoc(), "register expected");
    return MatchOperand_ParseFail;
  }

  SMLoc E = getLoc();
  if (Parser.getTok().isNot(AsmToken::RBrac)) {
    Error(E, "']' expected");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // Eat right bracket token.

  Operands.push_back(ARM64Operand::CreateMem(Reg, 0, S, E, E, getContext()));
  return MatchOperand_Success;
}

/// parseMemory - Parse a memory operand for a basic load/store instruction.
bool ARM64AsmParser::parseMemory(OperandVector &Operands) {
  assert(Parser.getTok().is(AsmToken::LBrac) && "Token is not a Left Bracket");
  SMLoc S = getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier))
    return Error(BaseRegTok.getLoc(), "register expected");

  int64_t Reg = tryParseRegister();
  if (Reg == -1)
    return Error(BaseRegTok.getLoc(), "register expected");

  // If there is an offset expression, parse it.
  const MCExpr *OffsetExpr = 0;
  SMLoc OffsetLoc;
  if (Parser.getTok().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the comma.
    OffsetLoc = getLoc();

    // Register offset
    const AsmToken &OffsetRegTok = Parser.getTok();
    int Reg2 = OffsetRegTok.is(AsmToken::Identifier) ? tryParseRegister() : -1;
    if (Reg2 != -1) {
      // Default shift is LSL, with an omitted shift.  We use the third bit of
      // the extend value to indicate presence/omission of the immediate offset.
      ARM64_AM::ExtendType ExtOp = ARM64_AM::UXTX;
      int64_t ShiftVal = 0;
      bool ExplicitShift = false;

      if (Parser.getTok().is(AsmToken::Comma)) {
        // Embedded extend operand.
        Parser.Lex(); // Eat the comma

        SMLoc ExtLoc = getLoc();
        const AsmToken &Tok = Parser.getTok();
        ExtOp = StringSwitch<ARM64_AM::ExtendType>(Tok.getString())
                    .Case("uxtw", ARM64_AM::UXTW)
                    .Case("lsl", ARM64_AM::UXTX) // Alias for UXTX
                    .Case("sxtw", ARM64_AM::SXTW)
                    .Case("sxtx", ARM64_AM::SXTX)
                    .Case("UXTW", ARM64_AM::UXTW)
                    .Case("LSL", ARM64_AM::UXTX) // Alias for UXTX
                    .Case("SXTW", ARM64_AM::SXTW)
                    .Case("SXTX", ARM64_AM::SXTX)
                    .Default(ARM64_AM::InvalidExtend);
        if (ExtOp == ARM64_AM::InvalidExtend)
          return Error(ExtLoc, "expected valid extend operation");

        Parser.Lex(); // Eat the extend op.

        if (getLexer().is(AsmToken::RBrac)) {
          // No immediate operand.
          if (ExtOp == ARM64_AM::UXTX)
            return Error(ExtLoc, "LSL extend requires immediate operand");
        } else if (getLexer().is(AsmToken::Hash)) {
          // Immediate operand.
          Parser.Lex(); // Eat the '#'
          const MCExpr *ImmVal;
          SMLoc ExprLoc = getLoc();
          if (getParser().parseExpression(ImmVal))
            return true;
          const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
          if (!MCE)
            return TokError("immediate value expected for extend operand");

          ExplicitShift = true;
          ShiftVal = MCE->getValue();
          if (ShiftVal < 0 || ShiftVal > 4)
            return Error(ExprLoc, "immediate operand out of range");
        } else
          return Error(getLoc(), "expected immediate operand");
      }

      if (Parser.getTok().isNot(AsmToken::RBrac))
        return Error(getLoc(), "']' expected");

      Parser.Lex(); // Eat right bracket token.

      SMLoc E = getLoc();
      Operands.push_back(ARM64Operand::CreateRegOffsetMem(
          Reg, Reg2, ExtOp, ShiftVal, ExplicitShift, S, E, getContext()));
      return false;

      // Immediate expressions.
    } else if (Parser.getTok().is(AsmToken::Hash)) {
      Parser.Lex(); // Eat hash token.

      if (parseSymbolicImmVal(OffsetExpr))
        return true;
    } else {
      // FIXME: We really should make sure that we're dealing with a LDR/STR
      // instruction that can legally have a symbolic expression here.
      // Symbol reference.
      if (Parser.getTok().isNot(AsmToken::Identifier) &&
          Parser.getTok().isNot(AsmToken::String))
        return Error(getLoc(), "identifier or immediate expression expected");
      if (getParser().parseExpression(OffsetExpr))
        return true;
      // If this is a plain ref, Make sure a legal variant kind was specified.
      // Otherwise, it's a more complicated expression and we have to just
      // assume it's OK and let the relocation stuff puke if it's not.
      ARM64MCExpr::VariantKind ELFRefKind;
      MCSymbolRefExpr::VariantKind DarwinRefKind;
      const MCConstantExpr *Addend;
      if (classifySymbolRef(OffsetExpr, ELFRefKind, DarwinRefKind, Addend) &&
          Addend == 0) {
        assert(ELFRefKind == ARM64MCExpr::VK_INVALID &&
               "ELF symbol modifiers not supported here yet");

        switch (DarwinRefKind) {
        default:
          return Error(getLoc(), "expected @pageoff or @gotpageoff modifier");
        case MCSymbolRefExpr::VK_GOTPAGEOFF:
        case MCSymbolRefExpr::VK_PAGEOFF:
        case MCSymbolRefExpr::VK_TLVPPAGEOFF:
          // These are what we're expecting.
          break;
        }
      }
    }
  }

  SMLoc E = getLoc();
  if (Parser.getTok().isNot(AsmToken::RBrac))
    return Error(E, "']' expected");

  Parser.Lex(); // Eat right bracket token.

  // Create the memory operand.
  Operands.push_back(
      ARM64Operand::CreateMem(Reg, OffsetExpr, S, E, OffsetLoc, getContext()));

  // Check for a '!', indicating pre-indexed addressing with writeback.
  if (Parser.getTok().is(AsmToken::Exclaim)) {
    // There needs to have been an immediate or wback doesn't make sense.
    if (!OffsetExpr)
      return Error(E, "missing offset for pre-indexed addressing");
    // Pre-indexed with writeback must have a constant expression for the
    // offset. FIXME: Theoretically, we'd like to allow fixups so long
    // as they don't require a relocation.
    if (!isa<MCConstantExpr>(OffsetExpr))
      return Error(OffsetLoc, "constant immediate expression expected");

    // Create the Token operand for the '!'.
    Operands.push_back(ARM64Operand::CreateToken(
        "!", false, Parser.getTok().getLoc(), getContext()));
    Parser.Lex(); // Eat the '!' token.
  }

  return false;
}

bool ARM64AsmParser::parseSymbolicImmVal(const MCExpr *&ImmVal) {
  bool HasELFModifier = false;
  ARM64MCExpr::VariantKind RefKind;

  if (Parser.getTok().is(AsmToken::Colon)) {
    Parser.Lex(); // Eat ':"
    HasELFModifier = true;

    if (Parser.getTok().isNot(AsmToken::Identifier)) {
      Error(Parser.getTok().getLoc(),
            "expect relocation specifier in operand after ':'");
      return true;
    }

    std::string LowerCase = Parser.getTok().getIdentifier().lower();
    RefKind = StringSwitch<ARM64MCExpr::VariantKind>(LowerCase)
                  .Case("lo12", ARM64MCExpr::VK_LO12)
                  .Case("abs_g3", ARM64MCExpr::VK_ABS_G3)
                  .Case("abs_g2", ARM64MCExpr::VK_ABS_G2)
                  .Case("abs_g2_nc", ARM64MCExpr::VK_ABS_G2_NC)
                  .Case("abs_g1", ARM64MCExpr::VK_ABS_G1)
                  .Case("abs_g1_nc", ARM64MCExpr::VK_ABS_G1_NC)
                  .Case("abs_g0", ARM64MCExpr::VK_ABS_G0)
                  .Case("abs_g0_nc", ARM64MCExpr::VK_ABS_G0_NC)
                  .Case("dtprel_g2", ARM64MCExpr::VK_DTPREL_G2)
                  .Case("dtprel_g1", ARM64MCExpr::VK_DTPREL_G1)
                  .Case("dtprel_g1_nc", ARM64MCExpr::VK_DTPREL_G1_NC)
                  .Case("dtprel_g0", ARM64MCExpr::VK_DTPREL_G0)
                  .Case("dtprel_g0_nc", ARM64MCExpr::VK_DTPREL_G0_NC)
                  .Case("dtprel_lo12", ARM64MCExpr::VK_DTPREL_LO12)
                  .Case("dtprel_lo12_nc", ARM64MCExpr::VK_DTPREL_LO12_NC)
                  .Case("tprel_g2", ARM64MCExpr::VK_TPREL_G2)
                  .Case("tprel_g1", ARM64MCExpr::VK_TPREL_G1)
                  .Case("tprel_g1_nc", ARM64MCExpr::VK_TPREL_G1_NC)
                  .Case("tprel_g0", ARM64MCExpr::VK_TPREL_G0)
                  .Case("tprel_g0_nc", ARM64MCExpr::VK_TPREL_G0_NC)
                  .Case("tprel_lo12", ARM64MCExpr::VK_TPREL_LO12)
                  .Case("tprel_lo12_nc", ARM64MCExpr::VK_TPREL_LO12_NC)
                  .Case("tlsdesc_lo12", ARM64MCExpr::VK_TLSDESC_LO12)
                  .Case("got", ARM64MCExpr::VK_GOT_PAGE)
                  .Case("got_lo12", ARM64MCExpr::VK_GOT_LO12)
                  .Case("gottprel", ARM64MCExpr::VK_GOTTPREL_PAGE)
                  .Case("gottprel_lo12", ARM64MCExpr::VK_GOTTPREL_LO12_NC)
                  .Case("gottprel_g1", ARM64MCExpr::VK_GOTTPREL_G1)
                  .Case("gottprel_g0_nc", ARM64MCExpr::VK_GOTTPREL_G0_NC)
                  .Case("tlsdesc", ARM64MCExpr::VK_TLSDESC_PAGE)
                  .Default(ARM64MCExpr::VK_INVALID);

    if (RefKind == ARM64MCExpr::VK_INVALID) {
      Error(Parser.getTok().getLoc(),
            "expect relocation specifier in operand after ':'");
      return true;
    }

    Parser.Lex(); // Eat identifier

    if (Parser.getTok().isNot(AsmToken::Colon)) {
      Error(Parser.getTok().getLoc(), "expect ':' after relocation specifier");
      return true;
    }
    Parser.Lex(); // Eat ':'
  }

  if (getParser().parseExpression(ImmVal))
    return true;

  if (HasELFModifier)
    ImmVal = ARM64MCExpr::Create(ImmVal, RefKind, getContext());

  return false;
}

/// parseVectorList - Parse a vector list operand for AdvSIMD instructions.
bool ARM64AsmParser::parseVectorList(OperandVector &Operands) {
  assert(Parser.getTok().is(AsmToken::LCurly) && "Token is not a Left Bracket");
  SMLoc S = getLoc();
  Parser.Lex(); // Eat left bracket token.
  StringRef Kind;
  int64_t FirstReg = tryMatchVectorRegister(Kind);
  if (FirstReg == -1)
    return Error(getLoc(), "vector register expected");
  int64_t PrevReg = FirstReg;
  unsigned Count = 1;
  while (Parser.getTok().isNot(AsmToken::RCurly)) {
    if (Parser.getTok().is(AsmToken::EndOfStatement))
      Error(getLoc(), "'}' expected");

    if (Parser.getTok().isNot(AsmToken::Comma))
      return Error(getLoc(), "',' expected");
    Parser.Lex(); // Eat the comma token.

    SMLoc Loc = getLoc();
    StringRef NextKind;
    int64_t Reg = tryMatchVectorRegister(NextKind);
    if (Reg == -1)
      return Error(Loc, "vector register expected");
    // Any Kind suffices must match on all regs in the list.
    if (Kind != NextKind)
      return Error(Loc, "mismatched register size suffix");

    // Registers must be incremental (with wraparound at 31)
    if (getContext().getRegisterInfo()->getEncodingValue(Reg) !=
        (getContext().getRegisterInfo()->getEncodingValue(PrevReg) + 1) % 32)
      return Error(Loc, "registers must be sequential");

    PrevReg = Reg;
    ++Count;
  }
  Parser.Lex(); // Eat the '}' token.

  unsigned NumElements = 0;
  char ElementKind = 0;
  if (!Kind.empty())
    parseValidVectorKind(Kind, NumElements, ElementKind);

  Operands.push_back(ARM64Operand::CreateVectorList(
      FirstReg, Count, NumElements, ElementKind, S, getLoc(), getContext()));

  // If there is an index specifier following the list, parse that too.
  if (Parser.getTok().is(AsmToken::LBrac)) {
    SMLoc SIdx = getLoc();
    Parser.Lex(); // Eat left bracket token.

    const MCExpr *ImmVal;
    if (getParser().parseExpression(ImmVal))
      return MatchOperand_ParseFail;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      TokError("immediate value expected for vector index");
      return MatchOperand_ParseFail;
    }

    SMLoc E = getLoc();
    if (Parser.getTok().isNot(AsmToken::RBrac)) {
      Error(E, "']' expected");
      return MatchOperand_ParseFail;
    }

    Parser.Lex(); // Eat right bracket token.

    Operands.push_back(ARM64Operand::CreateVectorIndex(MCE->getValue(), SIdx, E,
                                                       getContext()));
  }
  return false;
}

/// parseOperand - Parse a arm instruction operand.  For now this parses the
/// operand regardless of the mnemonic.
bool ARM64AsmParser::parseOperand(OperandVector &Operands, bool isCondCode,
                                  bool invertCondCode) {
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

  // Nothing custom, so do general case parsing.
  SMLoc S, E;
  switch (getLexer().getKind()) {
  default: {
    SMLoc S = getLoc();
    const MCExpr *Expr;
    if (parseSymbolicImmVal(Expr))
      return Error(S, "invalid operand");

    SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(ARM64Operand::CreateImm(Expr, S, E, getContext()));
    return false;
  }
  case AsmToken::LBrac:
    return parseMemory(Operands);
  case AsmToken::LCurly:
    return parseVectorList(Operands);
  case AsmToken::Identifier: {
    // If we're expecting a Condition Code operand, then just parse that.
    if (isCondCode)
      return parseCondCode(Operands, invertCondCode);

    // If it's a register name, parse it.
    if (!parseRegister(Operands))
      return false;

    // This could be an optional "shift" operand.
    if (!parseOptionalShift(Operands))
      return false;

    // Or maybe it could be an optional "extend" operand.
    if (!parseOptionalExtend(Operands))
      return false;

    // This was not a register so parse other operands that start with an
    // identifier (like labels) as expressions and create them as immediates.
    const MCExpr *IdVal;
    S = getLoc();
    if (getParser().parseExpression(IdVal))
      return true;

    E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(ARM64Operand::CreateImm(IdVal, S, E, getContext()));
    return false;
  }
  case AsmToken::Hash: {
    // #42 -> immediate.
    S = getLoc();
    Parser.Lex();

    // The only Real that should come through here is a literal #0.0 for
    // the fcmp[e] r, #0.0 instructions. They expect raw token operands,
    // so convert the value.
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Real)) {
      APFloat RealVal(APFloat::IEEEdouble, Tok.getString());
      uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
      if (IntVal != 0 || (Mnemonic != "fcmp" && Mnemonic != "fcmpe"))
        return TokError("unexpected floating point literal");
      Parser.Lex(); // Eat the token.

      Operands.push_back(
          ARM64Operand::CreateToken("#0", false, S, getContext()));
      Operands.push_back(
          ARM64Operand::CreateToken(".0", false, S, getContext()));
      return false;
    }

    const MCExpr *ImmVal;
    if (parseSymbolicImmVal(ImmVal))
      return true;

    E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(ARM64Operand::CreateImm(ImmVal, S, E, getContext()));
    return false;
  }
  }
}

/// ParseInstruction - Parse an ARM64 instruction mnemonic followed by its
/// operands.
bool ARM64AsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                      StringRef Name, SMLoc NameLoc,
                                      OperandVector &Operands) {
  // Create the leading tokens for the mnemonic, split by '.' characters.
  size_t Start = 0, Next = Name.find('.');
  StringRef Head = Name.slice(Start, Next);

  // IC, DC, AT, and TLBI instructions are aliases for the SYS instruction.
  if (Head == "ic" || Head == "dc" || Head == "at" || Head == "tlbi")
    return parseSysAlias(Head, NameLoc, Operands);

  Operands.push_back(
      ARM64Operand::CreateToken(Head, false, NameLoc, getContext()));
  Mnemonic = Head;

  // Handle condition codes for a branch mnemonic
  if (Head == "b" && Next != StringRef::npos) {
    Start = Next;
    Next = Name.find('.', Start + 1);
    Head = Name.slice(Start + 1, Next);

    SMLoc SuffixLoc = SMLoc::getFromPointer(NameLoc.getPointer() +
                                            (Head.data() - Name.data()));
    unsigned CC = parseCondCodeString(Head);
    if (CC == ~0U)
      return Error(SuffixLoc, "invalid condition code");
    const MCExpr *CCExpr = MCConstantExpr::Create(CC, getContext());
    Operands.push_back(
        ARM64Operand::CreateImm(CCExpr, NameLoc, NameLoc, getContext()));
  }

  // Add the remaining tokens in the mnemonic.
  while (Next != StringRef::npos) {
    Start = Next;
    Next = Name.find('.', Start + 1);
    Head = Name.slice(Start, Next);
    SMLoc SuffixLoc = SMLoc::getFromPointer(NameLoc.getPointer() +
                                            (Head.data() - Name.data()) + 1);
    Operands.push_back(
        ARM64Operand::CreateToken(Head, true, SuffixLoc, getContext()));
  }

  // Conditional compare instructions have a Condition Code operand, which needs
  // to be parsed and an immediate operand created.
  bool condCodeFourthOperand =
      (Head == "ccmp" || Head == "ccmn" || Head == "fccmp" ||
       Head == "fccmpe" || Head == "fcsel" || Head == "csel" ||
       Head == "csinc" || Head == "csinv" || Head == "csneg");

  // These instructions are aliases to some of the conditional select
  // instructions. However, the condition code is inverted in the aliased
  // instruction.
  //
  // FIXME: Is this the correct way to handle these? Or should the parser
  //        generate the aliased instructions directly?
  bool condCodeSecondOperand = (Head == "cset" || Head == "csetm");
  bool condCodeThirdOperand =
      (Head == "cinc" || Head == "cinv" || Head == "cneg");

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, false, false)) {
      Parser.eatToEndOfStatement();
      return true;
    }

    unsigned N = 2;
    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.

      // Parse and remember the operand.
      if (parseOperand(Operands, (N == 4 && condCodeFourthOperand) ||
                                     (N == 3 && condCodeThirdOperand) ||
                                     (N == 2 && condCodeSecondOperand),
                       condCodeSecondOperand || condCodeThirdOperand)) {
        Parser.eatToEndOfStatement();
        return true;
      }

      ++N;
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = Parser.getTok().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

/// isFPR32Register - Check if a register is in the FPR32 register class.
/// (The parser does not have the target register info to check the register
/// class directly.)
static bool isFPR32Register(unsigned Reg) {
  using namespace ARM64;
  switch (Reg) {
  default:
    break;
  case S0:  case S1:  case S2:  case S3:  case S4:  case S5:  case S6:
  case S7:  case S8:  case S9:  case S10:  case S11:  case S12:  case S13:
  case S14:  case S15:  case S16:  case S17:  case S18:  case S19:  case S20:
  case S21:  case S22:  case S23:  case S24:  case S25:  case S26:  case S27:
  case S28:  case S29:  case S30:  case S31:
    return true;
  }
  return false;
}

/// isGPR32Register - Check if a register is in the GPR32sp register class.
/// (The parser does not have the target register info to check the register
/// class directly.)
static bool isGPR32Register(unsigned Reg) {
  using namespace ARM64;
  switch (Reg) {
  default:
    break;
  case W0:  case W1:  case W2:  case W3:  case W4:  case W5:  case W6:
  case W7:  case W8:  case W9:  case W10:  case W11:  case W12:  case W13:
  case W14:  case W15:  case W16:  case W17:  case W18:  case W19:  case W20:
  case W21:  case W22:  case W23:  case W24:  case W25:  case W26:  case W27:
  case W28:  case W29:  case W30:  case WSP:
    return true;
  }
  return false;
}

static bool isGPR64Reg(unsigned Reg) {
  using namespace ARM64;
  switch (Reg) {
  case X0:  case X1:  case X2:  case X3:  case X4:  case X5:  case X6:
  case X7:  case X8:  case X9:  case X10:  case X11:  case X12:  case X13:
  case X14:  case X15:  case X16:  case X17:  case X18:  case X19:  case X20:
  case X21:  case X22:  case X23:  case X24:  case X25:  case X26:  case X27:
  case X28:  case FP:  case LR:  case SP:  case XZR:
    return true;
  default:
    return false;
  }
}


// FIXME: This entire function is a giant hack to provide us with decent
// operand range validation/diagnostics until TableGen/MC can be extended
// to support autogeneration of this kind of validation.
bool ARM64AsmParser::validateInstruction(MCInst &Inst,
                                         SmallVectorImpl<SMLoc> &Loc) {
  const MCRegisterInfo *RI = getContext().getRegisterInfo();
  // Check for indexed addressing modes w/ the base register being the
  // same as a destination/source register or pair load where
  // the Rt == Rt2. All of those are undefined behaviour.
  switch (Inst.getOpcode()) {
  case ARM64::LDPSWpre:
  case ARM64::LDPWpost:
  case ARM64::LDPWpre:
  case ARM64::LDPXpost:
  case ARM64::LDPXpre: {
    unsigned Rt = Inst.getOperand(0).getReg();
    unsigned Rt2 = Inst.getOperand(1).getReg();
    unsigned Rn = Inst.getOperand(2).getReg();
    if (RI->isSubRegisterEq(Rn, Rt))
      return Error(Loc[0], "unpredictable LDP instruction, writeback base "
                           "is also a destination");
    if (RI->isSubRegisterEq(Rn, Rt2))
      return Error(Loc[1], "unpredictable LDP instruction, writeback base "
                           "is also a destination");
    // FALLTHROUGH
  }
  case ARM64::LDPDpost:
  case ARM64::LDPDpre:
  case ARM64::LDPQpost:
  case ARM64::LDPQpre:
  case ARM64::LDPSpost:
  case ARM64::LDPSpre:
  case ARM64::LDPSWpost:
  case ARM64::LDPDi:
  case ARM64::LDPQi:
  case ARM64::LDPSi:
  case ARM64::LDPSWi:
  case ARM64::LDPWi:
  case ARM64::LDPXi: {
    unsigned Rt = Inst.getOperand(0).getReg();
    unsigned Rt2 = Inst.getOperand(1).getReg();
    if (Rt == Rt2)
      return Error(Loc[1], "unpredictable LDP instruction, Rt2==Rt");
    break;
  }
  case ARM64::STPDpost:
  case ARM64::STPDpre:
  case ARM64::STPQpost:
  case ARM64::STPQpre:
  case ARM64::STPSpost:
  case ARM64::STPSpre:
  case ARM64::STPWpost:
  case ARM64::STPWpre:
  case ARM64::STPXpost:
  case ARM64::STPXpre: {
    unsigned Rt = Inst.getOperand(0).getReg();
    unsigned Rt2 = Inst.getOperand(1).getReg();
    unsigned Rn = Inst.getOperand(2).getReg();
    if (RI->isSubRegisterEq(Rn, Rt))
      return Error(Loc[0], "unpredictable STP instruction, writeback base "
                           "is also a source");
    if (RI->isSubRegisterEq(Rn, Rt2))
      return Error(Loc[1], "unpredictable STP instruction, writeback base "
                           "is also a source");
    break;
  }
  case ARM64::LDRBBpre:
  case ARM64::LDRBpre:
  case ARM64::LDRHHpre:
  case ARM64::LDRHpre:
  case ARM64::LDRSBWpre:
  case ARM64::LDRSBXpre:
  case ARM64::LDRSHWpre:
  case ARM64::LDRSHXpre:
  case ARM64::LDRSWpre:
  case ARM64::LDRWpre:
  case ARM64::LDRXpre:
  case ARM64::LDRBBpost:
  case ARM64::LDRBpost:
  case ARM64::LDRHHpost:
  case ARM64::LDRHpost:
  case ARM64::LDRSBWpost:
  case ARM64::LDRSBXpost:
  case ARM64::LDRSHWpost:
  case ARM64::LDRSHXpost:
  case ARM64::LDRSWpost:
  case ARM64::LDRWpost:
  case ARM64::LDRXpost: {
    unsigned Rt = Inst.getOperand(0).getReg();
    unsigned Rn = Inst.getOperand(1).getReg();
    if (RI->isSubRegisterEq(Rn, Rt))
      return Error(Loc[0], "unpredictable LDR instruction, writeback base "
                           "is also a source");
    break;
  }
  case ARM64::STRBBpost:
  case ARM64::STRBpost:
  case ARM64::STRHHpost:
  case ARM64::STRHpost:
  case ARM64::STRWpost:
  case ARM64::STRXpost:
  case ARM64::STRBBpre:
  case ARM64::STRBpre:
  case ARM64::STRHHpre:
  case ARM64::STRHpre:
  case ARM64::STRWpre:
  case ARM64::STRXpre: {
    unsigned Rt = Inst.getOperand(0).getReg();
    unsigned Rn = Inst.getOperand(1).getReg();
    if (RI->isSubRegisterEq(Rn, Rt))
      return Error(Loc[0], "unpredictable STR instruction, writeback base "
                           "is also a source");
    break;
  }
  }

  // Now check immediate ranges. Separate from the above as there is overlap
  // in the instructions being checked and this keeps the nested conditionals
  // to a minimum.
  switch (Inst.getOpcode()) {
  case ARM64::ANDWrs:
  case ARM64::ANDSWrs:
  case ARM64::EORWrs:
  case ARM64::ORRWrs: {
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[3], "immediate value expected");
    int64_t shifter = Inst.getOperand(3).getImm();
    ARM64_AM::ShiftType ST = ARM64_AM::getShiftType(shifter);
    if (ST == ARM64_AM::LSL && shifter > 31)
      return Error(Loc[3], "shift value out of range");
    return false;
  }
  case ARM64::ADDSWri:
  case ARM64::ADDSXri:
  case ARM64::ADDWri:
  case ARM64::ADDXri:
  case ARM64::SUBSWri:
  case ARM64::SUBSXri:
  case ARM64::SUBWri:
  case ARM64::SUBXri: {
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[3], "immediate value expected");
    int64_t shifter = Inst.getOperand(3).getImm();
    if (shifter != 0 && shifter != 12)
      return Error(Loc[3], "shift value out of range");
    // The imm12 operand can be an expression. Validate that it's legit.
    // FIXME: We really, really want to allow arbitrary expressions here
    // and resolve the value and validate the result at fixup time, but
    // that's hard as we have long since lost any source information we
    // need to generate good diagnostics by that point.
    if (Inst.getOpcode() == ARM64::ADDXri && Inst.getOperand(2).isExpr()) {
      const MCExpr *Expr = Inst.getOperand(2).getExpr();
      ARM64MCExpr::VariantKind ELFRefKind;
      MCSymbolRefExpr::VariantKind DarwinRefKind;
      const MCConstantExpr *Addend;
      if (!classifySymbolRef(Expr, ELFRefKind, DarwinRefKind, Addend)) {
        return Error(Loc[2], "invalid immediate expression");
      }

      if (DarwinRefKind == MCSymbolRefExpr::VK_PAGEOFF ||
          DarwinRefKind == MCSymbolRefExpr::VK_TLVPPAGEOFF ||
          ELFRefKind == ARM64MCExpr::VK_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12_NC ||
          ELFRefKind == ARM64MCExpr::VK_TPREL_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_TPREL_LO12_NC ||
          ELFRefKind == ARM64MCExpr::VK_TLSDESC_LO12) {
        // Note that we don't range-check the addend. It's adjusted
        // modulo page size when converted, so there is no "out of range"
        // condition when using @pageoff. Any validity checking for the value
        // was done in the is*() predicate function.
        return false;
      } else if (DarwinRefKind == MCSymbolRefExpr::VK_GOTPAGEOFF) {
        // @gotpageoff can only be used directly, not with an addend.
        return Addend != 0;
      }

      // Otherwise, we're not sure, so don't allow it for now.
      return Error(Loc[2], "invalid immediate expression");
    }

    // If it's anything but an immediate, it's not legit.
    if (!Inst.getOperand(2).isImm())
      return Error(Loc[2], "invalid immediate expression");
    int64_t imm = Inst.getOperand(2).getImm();
    if (imm > 4095 || imm < 0)
      return Error(Loc[2], "immediate value out of range");
    return false;
  }
  case ARM64::LDRBpre:
  case ARM64::LDRHpre:
  case ARM64::LDRSBWpre:
  case ARM64::LDRSBXpre:
  case ARM64::LDRSHWpre:
  case ARM64::LDRSHXpre:
  case ARM64::LDRWpre:
  case ARM64::LDRXpre:
  case ARM64::LDRSpre:
  case ARM64::LDRDpre:
  case ARM64::LDRQpre:
  case ARM64::STRBpre:
  case ARM64::STRHpre:
  case ARM64::STRWpre:
  case ARM64::STRXpre:
  case ARM64::STRSpre:
  case ARM64::STRDpre:
  case ARM64::STRQpre:
  case ARM64::LDRBpost:
  case ARM64::LDRHpost:
  case ARM64::LDRSBWpost:
  case ARM64::LDRSBXpost:
  case ARM64::LDRSHWpost:
  case ARM64::LDRSHXpost:
  case ARM64::LDRWpost:
  case ARM64::LDRXpost:
  case ARM64::LDRSpost:
  case ARM64::LDRDpost:
  case ARM64::LDRQpost:
  case ARM64::STRBpost:
  case ARM64::STRHpost:
  case ARM64::STRWpost:
  case ARM64::STRXpost:
  case ARM64::STRSpost:
  case ARM64::STRDpost:
  case ARM64::STRQpost:
  case ARM64::LDTRXi:
  case ARM64::LDTRWi:
  case ARM64::LDTRHi:
  case ARM64::LDTRBi:
  case ARM64::LDTRSHWi:
  case ARM64::LDTRSHXi:
  case ARM64::LDTRSBWi:
  case ARM64::LDTRSBXi:
  case ARM64::LDTRSWi:
  case ARM64::STTRWi:
  case ARM64::STTRXi:
  case ARM64::STTRHi:
  case ARM64::STTRBi:
  case ARM64::LDURWi:
  case ARM64::LDURXi:
  case ARM64::LDURSi:
  case ARM64::LDURDi:
  case ARM64::LDURQi:
  case ARM64::LDURHi:
  case ARM64::LDURBi:
  case ARM64::LDURSHWi:
  case ARM64::LDURSHXi:
  case ARM64::LDURSBWi:
  case ARM64::LDURSBXi:
  case ARM64::LDURSWi:
  case ARM64::PRFUMi:
  case ARM64::STURWi:
  case ARM64::STURXi:
  case ARM64::STURSi:
  case ARM64::STURDi:
  case ARM64::STURQi:
  case ARM64::STURHi:
  case ARM64::STURBi: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(2).isImm())
      return Error(Loc[1], "immediate value expected");
    int64_t offset = Inst.getOperand(2).getImm();
    if (offset > 255 || offset < -256)
      return Error(Loc[1], "offset value out of range");
    return false;
  }
  case ARM64::LDRSro:
  case ARM64::LDRWro:
  case ARM64::LDRSWro:
  case ARM64::STRWro:
  case ARM64::STRSro: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[1], "immediate value expected");
    int64_t shift = Inst.getOperand(3).getImm();
    ARM64_AM::ExtendType type = ARM64_AM::getMemExtendType(shift);
    if (type != ARM64_AM::UXTW && type != ARM64_AM::UXTX &&
        type != ARM64_AM::SXTW && type != ARM64_AM::SXTX)
      return Error(Loc[1], "shift type invalid");
    return false;
  }
  case ARM64::LDRDro:
  case ARM64::LDRQro:
  case ARM64::LDRXro:
  case ARM64::PRFMro:
  case ARM64::STRXro:
  case ARM64::STRDro:
  case ARM64::STRQro: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[1], "immediate value expected");
    int64_t shift = Inst.getOperand(3).getImm();
    ARM64_AM::ExtendType type = ARM64_AM::getMemExtendType(shift);
    if (type != ARM64_AM::UXTW && type != ARM64_AM::UXTX &&
        type != ARM64_AM::SXTW && type != ARM64_AM::SXTX)
      return Error(Loc[1], "shift type invalid");
    return false;
  }
  case ARM64::LDRHro:
  case ARM64::LDRHHro:
  case ARM64::LDRSHWro:
  case ARM64::LDRSHXro:
  case ARM64::STRHro:
  case ARM64::STRHHro: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[1], "immediate value expected");
    int64_t shift = Inst.getOperand(3).getImm();
    ARM64_AM::ExtendType type = ARM64_AM::getMemExtendType(shift);
    if (type != ARM64_AM::UXTW && type != ARM64_AM::UXTX &&
        type != ARM64_AM::SXTW && type != ARM64_AM::SXTX)
      return Error(Loc[1], "shift type invalid");
    return false;
  }
  case ARM64::LDRBro:
  case ARM64::LDRBBro:
  case ARM64::LDRSBWro:
  case ARM64::LDRSBXro:
  case ARM64::STRBro:
  case ARM64::STRBBro: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[1], "immediate value expected");
    int64_t shift = Inst.getOperand(3).getImm();
    ARM64_AM::ExtendType type = ARM64_AM::getMemExtendType(shift);
    if (type != ARM64_AM::UXTW && type != ARM64_AM::UXTX &&
        type != ARM64_AM::SXTW && type != ARM64_AM::SXTX)
      return Error(Loc[1], "shift type invalid");
    return false;
  }
  case ARM64::LDPWi:
  case ARM64::LDPXi:
  case ARM64::LDPSi:
  case ARM64::LDPDi:
  case ARM64::LDPQi:
  case ARM64::LDPSWi:
  case ARM64::STPWi:
  case ARM64::STPXi:
  case ARM64::STPSi:
  case ARM64::STPDi:
  case ARM64::STPQi:
  case ARM64::LDPWpre:
  case ARM64::LDPXpre:
  case ARM64::LDPSpre:
  case ARM64::LDPDpre:
  case ARM64::LDPQpre:
  case ARM64::LDPSWpre:
  case ARM64::STPWpre:
  case ARM64::STPXpre:
  case ARM64::STPSpre:
  case ARM64::STPDpre:
  case ARM64::STPQpre:
  case ARM64::LDPWpost:
  case ARM64::LDPXpost:
  case ARM64::LDPSpost:
  case ARM64::LDPDpost:
  case ARM64::LDPQpost:
  case ARM64::LDPSWpost:
  case ARM64::STPWpost:
  case ARM64::STPXpost:
  case ARM64::STPSpost:
  case ARM64::STPDpost:
  case ARM64::STPQpost:
  case ARM64::LDNPWi:
  case ARM64::LDNPXi:
  case ARM64::LDNPSi:
  case ARM64::LDNPDi:
  case ARM64::LDNPQi:
  case ARM64::STNPWi:
  case ARM64::STNPXi:
  case ARM64::STNPSi:
  case ARM64::STNPDi:
  case ARM64::STNPQi: {
    // FIXME: Should accept expressions and error in fixup evaluation
    // if out of range.
    if (!Inst.getOperand(3).isImm())
      return Error(Loc[2], "immediate value expected");
    int64_t offset = Inst.getOperand(3).getImm();
    if (offset > 63 || offset < -64)
      return Error(Loc[2], "offset value out of range");
    return false;
  }
  default:
    return false;
  }
}

static void rewriteMOV(ARM64AsmParser::OperandVector &Operands,
                       StringRef mnemonic, uint64_t imm, unsigned shift,
                       MCContext &Context) {
  ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[0]);
  ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
  Operands[0] =
      ARM64Operand::CreateToken(mnemonic, false, Op->getStartLoc(), Context);

  const MCExpr *NewImm = MCConstantExpr::Create(imm >> shift, Context);
  Operands[2] = ARM64Operand::CreateImm(NewImm, Op2->getStartLoc(),
                                        Op2->getEndLoc(), Context);

  Operands.push_back(ARM64Operand::CreateShifter(
      ARM64_AM::LSL, shift, Op2->getStartLoc(), Op2->getEndLoc(), Context));
  delete Op2;
  delete Op;
}

bool ARM64AsmParser::showMatchError(SMLoc Loc, unsigned ErrCode) {
  switch (ErrCode) {
  case Match_MissingFeature:
    return Error(Loc,
                 "instruction requires a CPU feature not currently enabled");
  case Match_InvalidOperand:
    return Error(Loc, "invalid operand for instruction");
  case Match_InvalidSuffix:
    return Error(Loc, "invalid type suffix for instruction");
  case Match_InvalidMemoryIndexedSImm9:
    return Error(Loc, "index must be an integer in range [-256,255].");
  case Match_InvalidMemoryIndexed32SImm7:
    return Error(Loc, "index must be a multiple of 4 in range [-256,252].");
  case Match_InvalidMemoryIndexed64SImm7:
    return Error(Loc, "index must be a multiple of 8 in range [-512,504].");
  case Match_InvalidMemoryIndexed128SImm7:
    return Error(Loc, "index must be a multiple of 16 in range [-1024,1008].");
  case Match_InvalidMemoryIndexed8:
    return Error(Loc, "index must be an integer in range [0,4095].");
  case Match_InvalidMemoryIndexed16:
    return Error(Loc, "index must be a multiple of 2 in range [0,8190].");
  case Match_InvalidMemoryIndexed32:
    return Error(Loc, "index must be a multiple of 4 in range [0,16380].");
  case Match_InvalidMemoryIndexed64:
    return Error(Loc, "index must be a multiple of 8 in range [0,32760].");
  case Match_InvalidMemoryIndexed128:
    return Error(Loc, "index must be a multiple of 16 in range [0,65520].");
  case Match_InvalidImm1_8:
    return Error(Loc, "immediate must be an integer in range [1,8].");
  case Match_InvalidImm1_16:
    return Error(Loc, "immediate must be an integer in range [1,16].");
  case Match_InvalidImm1_32:
    return Error(Loc, "immediate must be an integer in range [1,32].");
  case Match_InvalidImm1_64:
    return Error(Loc, "immediate must be an integer in range [1,64].");
  case Match_MnemonicFail:
    return Error(Loc, "unrecognized instruction mnemonic");
  default:
    assert(0 && "unexpected error code!");
    return Error(Loc, "invalid instruction format");
  }
}

bool ARM64AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                             OperandVector &Operands,
                                             MCStreamer &Out,
                                             unsigned &ErrorInfo,
                                             bool MatchingInlineAsm) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[0]);
  assert(Op->isToken() && "Leading operand should always be a mnemonic!");

  StringRef Tok = Op->getToken();
  // Translate CMN/CMP pseudos to ADDS/SUBS with zero register destination.
  // This needs to be done before the special handling of ADD/SUB immediates.
  if (Tok == "cmp" || Tok == "cmn") {
    // Replace the opcode with either ADDS or SUBS.
    const char *Repl = StringSwitch<const char *>(Tok)
                           .Case("cmp", "subs")
                           .Case("cmn", "adds")
                           .Default(0);
    assert(Repl && "Unknown compare instruction");
    delete Operands[0];
    Operands[0] = ARM64Operand::CreateToken(Repl, false, IDLoc, getContext());

    // Insert WZR or XZR as destination operand.
    ARM64Operand *RegOp = static_cast<ARM64Operand *>(Operands[1]);
    unsigned ZeroReg;
    if (RegOp->isReg() &&
        (isGPR32Register(RegOp->getReg()) || RegOp->getReg() == ARM64::WZR))
      ZeroReg = ARM64::WZR;
    else
      ZeroReg = ARM64::XZR;
    Operands.insert(
        Operands.begin() + 1,
        ARM64Operand::CreateReg(ZeroReg, false, IDLoc, IDLoc, getContext()));
    // Update since we modified it above.
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[0]);
    Tok = Op->getToken();
  }

  unsigned NumOperands = Operands.size();

  if (Tok == "mov" && NumOperands == 3) {
    // The MOV mnemomic is aliased to movn/movz, depending on the value of
    // the immediate being instantiated.
    // FIXME: Catching this here is a total hack, and we should use tblgen
    // support to implement this instead as soon as it is available.

    ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
    if (Op2->isImm()) {
      if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Op2->getImm())) {
        uint64_t Val = CE->getValue();
        uint64_t NVal = ~Val;

        // If this is a 32-bit register and the value has none of the upper
        // set, clear the complemented upper 32-bits so the logic below works
        // for 32-bit registers too.
        ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
        if (Op1->isReg() && isGPR32Register(Op1->getReg()) &&
            (Val & 0xFFFFFFFFULL) == Val)
          NVal &= 0x00000000FFFFFFFFULL;

        // MOVK Rd, imm << 0
        if ((Val & 0xFFFF) == Val)
          rewriteMOV(Operands, "movz", Val, 0, getContext());

        // MOVK Rd, imm << 16
        else if ((Val & 0xFFFF0000ULL) == Val)
          rewriteMOV(Operands, "movz", Val, 16, getContext());

        // MOVK Rd, imm << 32
        else if ((Val & 0xFFFF00000000ULL) == Val)
          rewriteMOV(Operands, "movz", Val, 32, getContext());

        // MOVK Rd, imm << 48
        else if ((Val & 0xFFFF000000000000ULL) == Val)
          rewriteMOV(Operands, "movz", Val, 48, getContext());

        // MOVN Rd, (~imm << 0)
        else if ((NVal & 0xFFFFULL) == NVal)
          rewriteMOV(Operands, "movn", NVal, 0, getContext());

        // MOVN Rd, ~(imm << 16)
        else if ((NVal & 0xFFFF0000ULL) == NVal)
          rewriteMOV(Operands, "movn", NVal, 16, getContext());

        // MOVN Rd, ~(imm << 32)
        else if ((NVal & 0xFFFF00000000ULL) == NVal)
          rewriteMOV(Operands, "movn", NVal, 32, getContext());

        // MOVN Rd, ~(imm << 48)
        else if ((NVal & 0xFFFF000000000000ULL) == NVal)
          rewriteMOV(Operands, "movn", NVal, 48, getContext());
      }
    }
  } else if (NumOperands == 4) {
    if (Tok == "add" || Tok == "adds" || Tok == "sub" || Tok == "subs") {
      // Handle the uimm24 immediate form, where the shift is not specified.
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      if (Op3->isImm()) {
        if (const MCConstantExpr *CE =
                dyn_cast<MCConstantExpr>(Op3->getImm())) {
          uint64_t Val = CE->getValue();
          if (Val >= (1 << 24)) {
            Error(IDLoc, "immediate value is too large");
            return true;
          }
          if (Val < (1 << 12)) {
            Operands.push_back(ARM64Operand::CreateShifter(
                ARM64_AM::LSL, 0, IDLoc, IDLoc, getContext()));
          } else if ((Val & 0xfff) == 0) {
            delete Operands[3];
            CE = MCConstantExpr::Create(Val >> 12, getContext());
            Operands[3] =
                ARM64Operand::CreateImm(CE, IDLoc, IDLoc, getContext());
            Operands.push_back(ARM64Operand::CreateShifter(
                ARM64_AM::LSL, 12, IDLoc, IDLoc, getContext()));
          } else {
            Error(IDLoc, "immediate value is too large");
            return true;
          }
        } else {
          Operands.push_back(ARM64Operand::CreateShifter(
              ARM64_AM::LSL, 0, IDLoc, IDLoc, getContext()));
        }
      }

      // FIXME: Horible hack to handle the LSL -> UBFM alias.
    } else if (NumOperands == 4 && Tok == "lsl") {
      ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      if (Op2->isReg() && Op3->isImm()) {
        const MCConstantExpr *Op3CE = dyn_cast<MCConstantExpr>(Op3->getImm());
        if (Op3CE) {
          uint64_t Op3Val = Op3CE->getValue();
          uint64_t NewOp3Val = 0;
          uint64_t NewOp4Val = 0;
          if (isGPR32Register(Op2->getReg()) || Op2->getReg() == ARM64::WZR) {
            NewOp3Val = (32 - Op3Val) & 0x1f;
            NewOp4Val = 31 - Op3Val;
          } else {
            NewOp3Val = (64 - Op3Val) & 0x3f;
            NewOp4Val = 63 - Op3Val;
          }

          const MCExpr *NewOp3 =
              MCConstantExpr::Create(NewOp3Val, getContext());
          const MCExpr *NewOp4 =
              MCConstantExpr::Create(NewOp4Val, getContext());

          Operands[0] = ARM64Operand::CreateToken(
              "ubfm", false, Op->getStartLoc(), getContext());
          Operands[3] = ARM64Operand::CreateImm(NewOp3, Op3->getStartLoc(),
                                                Op3->getEndLoc(), getContext());
          Operands.push_back(ARM64Operand::CreateImm(
              NewOp4, Op3->getStartLoc(), Op3->getEndLoc(), getContext()));
          delete Op3;
          delete Op;
        }
      }

      // FIXME: Horrible hack to handle the optional LSL shift for vector
      //        instructions.
    } else if (NumOperands == 4 && (Tok == "bic" || Tok == "orr")) {
      ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
      ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      if ((Op1->isToken() && Op2->isVectorReg() && Op3->isImm()) ||
          (Op1->isVectorReg() && Op2->isToken() && Op3->isImm()))
        Operands.push_back(ARM64Operand::CreateShifter(ARM64_AM::LSL, 0, IDLoc,
                                                       IDLoc, getContext()));
    } else if (NumOperands == 4 && (Tok == "movi" || Tok == "mvni")) {
      ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
      ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      if ((Op1->isToken() && Op2->isVectorReg() && Op3->isImm()) ||
          (Op1->isVectorReg() && Op2->isToken() && Op3->isImm())) {
        StringRef Suffix = Op1->isToken() ? Op1->getToken() : Op2->getToken();
        // Canonicalize on lower-case for ease of comparison.
        std::string CanonicalSuffix = Suffix.lower();
        if (Tok != "movi" ||
            (CanonicalSuffix != ".1d" && CanonicalSuffix != ".2d" &&
             CanonicalSuffix != ".8b" && CanonicalSuffix != ".16b"))
          Operands.push_back(ARM64Operand::CreateShifter(
              ARM64_AM::LSL, 0, IDLoc, IDLoc, getContext()));
      }
    }
  } else if (NumOperands == 5) {
    // FIXME: Horrible hack to handle the BFI -> BFM, SBFIZ->SBFM, and
    // UBFIZ -> UBFM aliases.
    if (Tok == "bfi" || Tok == "sbfiz" || Tok == "ubfiz") {
      ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      ARM64Operand *Op4 = static_cast<ARM64Operand *>(Operands[4]);

      if (Op1->isReg() && Op3->isImm() && Op4->isImm()) {
        const MCConstantExpr *Op3CE = dyn_cast<MCConstantExpr>(Op3->getImm());
        const MCConstantExpr *Op4CE = dyn_cast<MCConstantExpr>(Op4->getImm());

        if (Op3CE && Op4CE) {
          uint64_t Op3Val = Op3CE->getValue();
          uint64_t Op4Val = Op4CE->getValue();

          uint64_t NewOp3Val = 0;
          if (isGPR32Register(Op1->getReg()))
            NewOp3Val = (32 - Op3Val) & 0x1f;
          else
            NewOp3Val = (64 - Op3Val) & 0x3f;

          uint64_t NewOp4Val = Op4Val - 1;

          const MCExpr *NewOp3 =
              MCConstantExpr::Create(NewOp3Val, getContext());
          const MCExpr *NewOp4 =
              MCConstantExpr::Create(NewOp4Val, getContext());
          Operands[3] = ARM64Operand::CreateImm(NewOp3, Op3->getStartLoc(),
                                                Op3->getEndLoc(), getContext());
          Operands[4] = ARM64Operand::CreateImm(NewOp4, Op4->getStartLoc(),
                                                Op4->getEndLoc(), getContext());
          if (Tok == "bfi")
            Operands[0] = ARM64Operand::CreateToken(
                "bfm", false, Op->getStartLoc(), getContext());
          else if (Tok == "sbfiz")
            Operands[0] = ARM64Operand::CreateToken(
                "sbfm", false, Op->getStartLoc(), getContext());
          else if (Tok == "ubfiz")
            Operands[0] = ARM64Operand::CreateToken(
                "ubfm", false, Op->getStartLoc(), getContext());
          else
            llvm_unreachable("No valid mnemonic for alias?");

          delete Op;
          delete Op3;
          delete Op4;
        }
      }

      // FIXME: Horrible hack to handle the BFXIL->BFM, SBFX->SBFM, and
      // UBFX -> UBFM aliases.
    } else if (NumOperands == 5 &&
               (Tok == "bfxil" || Tok == "sbfx" || Tok == "ubfx")) {
      ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
      ARM64Operand *Op3 = static_cast<ARM64Operand *>(Operands[3]);
      ARM64Operand *Op4 = static_cast<ARM64Operand *>(Operands[4]);

      if (Op1->isReg() && Op3->isImm() && Op4->isImm()) {
        const MCConstantExpr *Op3CE = dyn_cast<MCConstantExpr>(Op3->getImm());
        const MCConstantExpr *Op4CE = dyn_cast<MCConstantExpr>(Op4->getImm());

        if (Op3CE && Op4CE) {
          uint64_t Op3Val = Op3CE->getValue();
          uint64_t Op4Val = Op4CE->getValue();
          uint64_t NewOp4Val = Op3Val + Op4Val - 1;

          if (NewOp4Val >= Op3Val) {
            const MCExpr *NewOp4 =
                MCConstantExpr::Create(NewOp4Val, getContext());
            Operands[4] = ARM64Operand::CreateImm(
                NewOp4, Op4->getStartLoc(), Op4->getEndLoc(), getContext());
            if (Tok == "bfxil")
              Operands[0] = ARM64Operand::CreateToken(
                  "bfm", false, Op->getStartLoc(), getContext());
            else if (Tok == "sbfx")
              Operands[0] = ARM64Operand::CreateToken(
                  "sbfm", false, Op->getStartLoc(), getContext());
            else if (Tok == "ubfx")
              Operands[0] = ARM64Operand::CreateToken(
                  "ubfm", false, Op->getStartLoc(), getContext());
            else
              llvm_unreachable("No valid mnemonic for alias?");

            delete Op;
            delete Op4;
          }
        }
      }
    }
  }
  // FIXME: Horrible hack for tbz and tbnz with Wn register operand.
  //        InstAlias can't quite handle this since the reg classes aren't
  //        subclasses.
  if (NumOperands == 4 && (Tok == "tbz" || Tok == "tbnz")) {
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[2]);
    if (Op->isImm()) {
      if (const MCConstantExpr *OpCE = dyn_cast<MCConstantExpr>(Op->getImm())) {
        if (OpCE->getValue() < 32) {
          // The source register can be Wn here, but the matcher expects a
          // GPR64. Twiddle it here if necessary.
          ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[1]);
          if (Op->isReg()) {
            unsigned Reg = getXRegFromWReg(Op->getReg());
            Operands[1] = ARM64Operand::CreateReg(
                Reg, false, Op->getStartLoc(), Op->getEndLoc(), getContext());
            delete Op;
          }
        }
      }
    }
  }
  // FIXME: Horrible hack for sxtw and uxtw with Wn src and Xd dst operands.
  //        InstAlias can't quite handle this since the reg classes aren't
  //        subclasses.
  if (NumOperands == 3 && (Tok == "sxtw" || Tok == "uxtw")) {
    // The source register can be Wn here, but the matcher expects a
    // GPR64. Twiddle it here if necessary.
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[2]);
    if (Op->isReg()) {
      unsigned Reg = getXRegFromWReg(Op->getReg());
      Operands[2] = ARM64Operand::CreateReg(Reg, false, Op->getStartLoc(),
                                            Op->getEndLoc(), getContext());
      delete Op;
    }
  }
  // FIXME: Likewise for [su]xt[bh] with a Xd dst operand
  else if (NumOperands == 3 &&
           (Tok == "sxtb" || Tok == "uxtb" || Tok == "sxth" || Tok == "uxth")) {
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[1]);
    if (Op->isReg() && isGPR64Reg(Op->getReg())) {
      // The source register can be Wn here, but the matcher expects a
      // GPR64. Twiddle it here if necessary.
      ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[2]);
      if (Op->isReg()) {
        unsigned Reg = getXRegFromWReg(Op->getReg());
        Operands[2] = ARM64Operand::CreateReg(Reg, false, Op->getStartLoc(),
                                              Op->getEndLoc(), getContext());
        delete Op;
      }
    }
  }

  // Yet another horrible hack to handle FMOV Rd, #0.0 using [WX]ZR.
  if (NumOperands == 3 && Tok == "fmov") {
    ARM64Operand *RegOp = static_cast<ARM64Operand *>(Operands[1]);
    ARM64Operand *ImmOp = static_cast<ARM64Operand *>(Operands[2]);
    if (RegOp->isReg() && ImmOp->isFPImm() &&
        ImmOp->getFPImm() == (unsigned)-1) {
      unsigned zreg =
          isFPR32Register(RegOp->getReg()) ? ARM64::WZR : ARM64::XZR;
      Operands[2] = ARM64Operand::CreateReg(zreg, false, Op->getStartLoc(),
                                            Op->getEndLoc(), getContext());
      delete ImmOp;
    }
  }

  // FIXME: Horrible hack to handle the literal .d[1] vector index on
  // FMOV instructions. The index isn't an actual instruction operand
  // but rather syntactic sugar. It really should be part of the mnemonic,
  // not the operand, but whatever.
  if ((NumOperands == 5) && Tok == "fmov") {
    // If the last operand is a vectorindex of '1', then replace it with
    // a '[' '1' ']' token sequence, which is what the matcher
    // (annoyingly) expects for a literal vector index operand.
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[NumOperands - 1]);
    if (Op->isVectorIndexD() && Op->getVectorIndex() == 1) {
      SMLoc Loc = Op->getStartLoc();
      Operands.pop_back();
      Operands.push_back(
          ARM64Operand::CreateToken("[", false, Loc, getContext()));
      Operands.push_back(
          ARM64Operand::CreateToken("1", false, Loc, getContext()));
      Operands.push_back(
          ARM64Operand::CreateToken("]", false, Loc, getContext()));
    } else if (Op->isReg()) {
      // Similarly, check the destination operand for the GPR->High-lane
      // variant.
      unsigned OpNo = NumOperands - 2;
      ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[OpNo]);
      if (Op->isVectorIndexD() && Op->getVectorIndex() == 1) {
        SMLoc Loc = Op->getStartLoc();
        Operands[OpNo] =
            ARM64Operand::CreateToken("[", false, Loc, getContext());
        Operands.insert(
            Operands.begin() + OpNo + 1,
            ARM64Operand::CreateToken("1", false, Loc, getContext()));
        Operands.insert(
            Operands.begin() + OpNo + 2,
            ARM64Operand::CreateToken("]", false, Loc, getContext()));
      }
    }
  }

  MCInst Inst;
  // First try to match against the secondary set of tables containing the
  // short-form NEON instructions (e.g. "fadd.2s v0, v1, v2").
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm, 1);

  // If that fails, try against the alternate table containing long-form NEON:
  // "fadd v0.2s, v1.2s, v2.2s"
  if (MatchResult != Match_Success)
    MatchResult =
        MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm, 0);

  switch (MatchResult) {
  case Match_Success: {
    // Perform range checking and other semantic validations
    SmallVector<SMLoc, 8> OperandLocs;
    NumOperands = Operands.size();
    for (unsigned i = 1; i < NumOperands; ++i)
      OperandLocs.push_back(Operands[i]->getStartLoc());
    if (validateInstruction(Inst, OperandLocs))
      return true;

    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst, STI);
    return false;
  }
  case Match_MissingFeature:
  case Match_MnemonicFail:
    return showMatchError(IDLoc, MatchResult);
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((ARM64Operand *)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    // If the match failed on a suffix token operand, tweak the diagnostic
    // accordingly.
    if (((ARM64Operand *)Operands[ErrorInfo])->isToken() &&
        ((ARM64Operand *)Operands[ErrorInfo])->isTokenSuffix())
      MatchResult = Match_InvalidSuffix;

    return showMatchError(ErrorLoc, MatchResult);
  }
  case Match_InvalidMemoryIndexedSImm9: {
    // If there is not a '!' after the memory operand that failed, we really
    // want the diagnostic for the non-pre-indexed instruction variant instead.
    // Be careful to check for the post-indexed variant as well, which also
    // uses this match diagnostic. Also exclude the explicitly unscaled
    // mnemonics, as they want the unscaled diagnostic as well.
    if (Operands.size() == ErrorInfo + 1 &&
        !((ARM64Operand *)Operands[ErrorInfo])->isImm() &&
        !Tok.startswith("stur") && !Tok.startswith("ldur")) {
      // whether we want an Indexed64 or Indexed32 diagnostic depends on
      // the register class of the previous operand. Default to 64 in case
      // we see something unexpected.
      MatchResult = Match_InvalidMemoryIndexed64;
      if (ErrorInfo) {
        ARM64Operand *PrevOp = (ARM64Operand *)Operands[ErrorInfo - 1];
        if (PrevOp->isReg() && ARM64MCRegisterClasses[ARM64::GPR32RegClassID]
                                   .contains(PrevOp->getReg()))
          MatchResult = Match_InvalidMemoryIndexed32;
      }
    }
    SMLoc ErrorLoc = ((ARM64Operand *)Operands[ErrorInfo])->getStartLoc();
    if (ErrorLoc == SMLoc())
      ErrorLoc = IDLoc;
    return showMatchError(ErrorLoc, MatchResult);
  }
  case Match_InvalidMemoryIndexed32:
  case Match_InvalidMemoryIndexed64:
  case Match_InvalidMemoryIndexed128:
    // If there is a '!' after the memory operand that failed, we really
    // want the diagnostic for the pre-indexed instruction variant instead.
    if (Operands.size() > ErrorInfo + 1 &&
        ((ARM64Operand *)Operands[ErrorInfo + 1])->isTokenEqual("!"))
      MatchResult = Match_InvalidMemoryIndexedSImm9;
  // FALL THROUGH
  case Match_InvalidMemoryIndexed8:
  case Match_InvalidMemoryIndexed16:
  case Match_InvalidMemoryIndexed32SImm7:
  case Match_InvalidMemoryIndexed64SImm7:
  case Match_InvalidMemoryIndexed128SImm7:
  case Match_InvalidImm1_8:
  case Match_InvalidImm1_16:
  case Match_InvalidImm1_32:
  case Match_InvalidImm1_64: {
    // Any time we get here, there's nothing fancy to do. Just get the
    // operand SMLoc and display the diagnostic.
    SMLoc ErrorLoc = ((ARM64Operand *)Operands[ErrorInfo])->getStartLoc();
    // If it's a memory operand, the error is with the offset immediate,
    // so get that location instead.
    if (((ARM64Operand *)Operands[ErrorInfo])->isMem())
      ErrorLoc = ((ARM64Operand *)Operands[ErrorInfo])->getOffsetLoc();
    if (ErrorLoc == SMLoc())
      ErrorLoc = IDLoc;
    return showMatchError(ErrorLoc, MatchResult);
  }
  }

  llvm_unreachable("Implement any new match types added!");
  return true;
}

/// ParseDirective parses the arm specific directives
bool ARM64AsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  SMLoc Loc = DirectiveID.getLoc();
  if (IDVal == ".hword")
    return parseDirectiveWord(2, Loc);
  if (IDVal == ".word")
    return parseDirectiveWord(4, Loc);
  if (IDVal == ".xword")
    return parseDirectiveWord(8, Loc);
  if (IDVal == ".tlsdesccall")
    return parseDirectiveTLSDescCall(Loc);

  return parseDirectiveLOH(IDVal, Loc);
}

/// parseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool ARM64AsmParser::parseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

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

// parseDirectiveTLSDescCall:
//   ::= .tlsdesccall symbol
bool ARM64AsmParser::parseDirectiveTLSDescCall(SMLoc L) {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return Error(L, "expected symbol after directive");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, getContext());
  Expr = ARM64MCExpr::Create(Expr, ARM64MCExpr::VK_TLSDESC, getContext());

  MCInst Inst;
  Inst.setOpcode(ARM64::TLSDESCCALL);
  Inst.addOperand(MCOperand::CreateExpr(Expr));

  getParser().getStreamer().EmitInstruction(Inst, STI);
  return false;
}

/// ::= .loh <lohName | lohId> label1, ..., labelN
/// The number of arguments depends on the loh identifier.
bool ARM64AsmParser::parseDirectiveLOH(StringRef IDVal, SMLoc Loc) {
  if (IDVal != MCLOHDirectiveName())
    return true;
  MCLOHType Kind;
  if (getParser().getTok().isNot(AsmToken::Identifier)) {
    if (getParser().getTok().isNot(AsmToken::Integer))
      return TokError("expected an identifier or a number in directive");
    // We successfully get a numeric value for the identifier.
    // Check if it is valid.
    int64_t Id = getParser().getTok().getIntVal();
    Kind = (MCLOHType)Id;
    // Check that Id does not overflow MCLOHType.
    if (!isValidMCLOHType(Kind) || Id != Kind)
      return TokError("invalid numeric identifier in directive");
  } else {
    StringRef Name = getTok().getIdentifier();
    // We successfully parse an identifier.
    // Check if it is a recognized one.
    int Id = MCLOHNameToId(Name);

    if (Id == -1)
      return TokError("invalid identifier in directive");
    Kind = (MCLOHType)Id;
  }
  // Consume the identifier.
  Lex();
  // Get the number of arguments of this LOH.
  int NbArgs = MCLOHIdToNbArgs(Kind);

  assert(NbArgs != -1 && "Invalid number of arguments");

  SmallVector<MCSymbol *, 3> Args;
  for (int Idx = 0; Idx < NbArgs; ++Idx) {
    StringRef Name;
    if (getParser().parseIdentifier(Name))
      return TokError("expected identifier in directive");
    Args.push_back(getContext().GetOrCreateSymbol(Name));

    if (Idx + 1 == NbArgs)
      break;
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
    Lex();
  }
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Twine(IDVal) + "' directive");

  getStreamer().EmitLOHDirective((MCLOHType)Kind, Args);
  return false;
}

bool
ARM64AsmParser::classifySymbolRef(const MCExpr *Expr,
                                  ARM64MCExpr::VariantKind &ELFRefKind,
                                  MCSymbolRefExpr::VariantKind &DarwinRefKind,
                                  const MCConstantExpr *&Addend) {
  ELFRefKind = ARM64MCExpr::VK_INVALID;
  DarwinRefKind = MCSymbolRefExpr::VK_None;

  if (const ARM64MCExpr *AE = dyn_cast<ARM64MCExpr>(Expr)) {
    ELFRefKind = AE->getKind();
    Expr = AE->getSubExpr();
  }

  const MCSymbolRefExpr *SE = dyn_cast<MCSymbolRefExpr>(Expr);
  if (SE) {
    // It's a simple symbol reference with no addend.
    DarwinRefKind = SE->getKind();
    Addend = 0;
    return true;
  }

  const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr);
  if (!BE)
    return false;

  SE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
  if (!SE)
    return false;
  DarwinRefKind = SE->getKind();

  if (BE->getOpcode() != MCBinaryExpr::Add)
    return false;

  // See if the addend is is a constant, otherwise there's more going
  // on here than we can deal with.
  Addend = dyn_cast<MCConstantExpr>(BE->getRHS());
  if (!Addend)
    return false;

  // It's some symbol reference + a constant addend, but really
  // shouldn't use both Darwin and ELF syntax.
  return ELFRefKind == ARM64MCExpr::VK_INVALID ||
         DarwinRefKind == MCSymbolRefExpr::VK_None;
}

/// Force static initialization.
extern "C" void LLVMInitializeARM64AsmParser() {
  RegisterMCAsmParser<ARM64AsmParser> X(TheARM64Target);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "ARM64GenAsmMatcher.inc"

// Define this matcher function after the auto-generated include so we
// have the match class enum definitions.
unsigned ARM64AsmParser::validateTargetOperandClass(MCParsedAsmOperand *AsmOp,
                                                    unsigned Kind) {
  ARM64Operand *Op = static_cast<ARM64Operand *>(AsmOp);
  // If the kind is a token for a literal immediate, check if our asm
  // operand matches. This is for InstAliases which have a fixed-value
  // immediate in the syntax.
  int64_t ExpectedVal;
  switch (Kind) {
  default:
    return Match_InvalidOperand;
  case MCK__35_0:
    ExpectedVal = 0;
    break;
  case MCK__35_1:
    ExpectedVal = 1;
    break;
  case MCK__35_12:
    ExpectedVal = 12;
    break;
  case MCK__35_16:
    ExpectedVal = 16;
    break;
  case MCK__35_2:
    ExpectedVal = 2;
    break;
  case MCK__35_24:
    ExpectedVal = 24;
    break;
  case MCK__35_3:
    ExpectedVal = 3;
    break;
  case MCK__35_32:
    ExpectedVal = 32;
    break;
  case MCK__35_4:
    ExpectedVal = 4;
    break;
  case MCK__35_48:
    ExpectedVal = 48;
    break;
  case MCK__35_6:
    ExpectedVal = 6;
    break;
  case MCK__35_64:
    ExpectedVal = 64;
    break;
  case MCK__35_8:
    ExpectedVal = 8;
    break;
  }
  if (!Op->isImm())
    return Match_InvalidOperand;
  const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Op->getImm());
  if (!CE)
    return Match_InvalidOperand;
  if (CE->getValue() == ExpectedVal)
    return Match_Success;
  return Match_InvalidOperand;
}
