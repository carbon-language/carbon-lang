//===-- ARM64AsmParser.cpp - Parse ARM64 assembly to MCInst instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARM64AddressingModes.h"
#include "MCTargetDesc/ARM64MCExpr.h"
#include "Utils/ARM64BaseInfo.h"
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
  int tryMatchVectorRegister(StringRef &Kind, bool expected);
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
                               unsigned &ErrorInfo,
                               bool MatchingInlineAsm) override;
/// @name Auto-generated Match Functions
/// {

#define GET_ASSEMBLER_HEADER
#include "ARM64GenAsmMatcher.inc"

  /// }

  OperandMatchResultTy tryParseNoIndexMemory(OperandVector &Operands);
  OperandMatchResultTy tryParseBarrierOperand(OperandVector &Operands);
  OperandMatchResultTy tryParseMRSSystemRegister(OperandVector &Operands);
  OperandMatchResultTy tryParseSysReg(OperandVector &Operands);
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
                 const MCInstrInfo &MII,
                 const MCTargetOptions &Options)
      : MCTargetAsmParser(), STI(_STI), Parser(_Parser) {
    MCAsmParserExtension::Initialize(_Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  bool ParseDirective(AsmToken DirectiveID) override;
  unsigned validateTargetOperandClass(MCParsedAsmOperand *Op,
                                      unsigned Kind) override;

  static bool classifySymbolRef(const MCExpr *Expr,
                                ARM64MCExpr::VariantKind &ELFRefKind,
                                MCSymbolRefExpr::VariantKind &DarwinRefKind,
                                int64_t &Addend);
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
    k_SysReg,
    k_SysCR,
    k_Prefetch,
    k_Shifter,
    k_Extend,
    k_FPImm,
    k_Barrier
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

  struct SysRegOp {
    const char *Data;
    unsigned Length;
    uint64_t FeatureBits; // We need to pass through information about which
                          // core we are compiling for so that the SysReg
                          // Mappers can appropriately conditionalize.
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
    struct SysRegOp SysReg;
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
    case k_Register:
      Reg = o.Reg;
      break;
    case k_VectorList:
      VectorList = o.VectorList;
      break;
    case k_VectorIndex:
      VectorIndex = o.VectorIndex;
      break;
    case k_SysReg:
      SysReg = o.SysReg;
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
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }
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

  unsigned getReg() const override {
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

  StringRef getSysReg() const {
    assert(Kind == k_SysReg && "Invalid access!");
    return StringRef(SysReg.Data, SysReg.Length);
  }

  uint64_t getSysRegFeatureBits() const {
    assert(Kind == k_SysReg && "Invalid access!");
    return SysReg.FeatureBits;
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

  bool isImm() const override { return Kind == k_Immediate; }
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
  bool isPCRelLabel19() const {
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
    int64_t Addend;
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
                                                   ARM64MCExpr::VK_ABS_G2_S,
                                                   ARM64MCExpr::VK_TPREL_G2,
                                                   ARM64MCExpr::VK_DTPREL_G2 };
    return isMovWSymbol(Variants);
  }

  bool isMovZSymbolG1() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G1,
                                                   ARM64MCExpr::VK_ABS_G1_S,
                                                   ARM64MCExpr::VK_GOTTPREL_G1,
                                                   ARM64MCExpr::VK_TPREL_G1,
                                                   ARM64MCExpr::VK_DTPREL_G1, };
    return isMovWSymbol(Variants);
  }

  bool isMovZSymbolG0() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G0,
                                                   ARM64MCExpr::VK_ABS_G0_S,
                                                   ARM64MCExpr::VK_TPREL_G0,
                                                   ARM64MCExpr::VK_DTPREL_G0 };
    return isMovWSymbol(Variants);
  }

  bool isMovKSymbolG3() const {
    static ARM64MCExpr::VariantKind Variants[] = { ARM64MCExpr::VK_ABS_G3 };
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
  bool isSysReg() const { return Kind == k_SysReg; }
  bool isMRSSystemRegister() const {
    if (!isSysReg()) return false;

    bool IsKnownRegister;
    auto Mapper = ARM64SysReg::MRSMapper(getSysRegFeatureBits());
    Mapper.fromString(getSysReg(), IsKnownRegister);

    return IsKnownRegister;
  }
  bool isMSRSystemRegister() const {
    if (!isSysReg()) return false;

    bool IsKnownRegister;
    auto Mapper = ARM64SysReg::MSRMapper(getSysRegFeatureBits());
    Mapper.fromString(getSysReg(), IsKnownRegister);

    return IsKnownRegister;
  }
  bool isSystemPStateField() const {
    if (!isSysReg()) return false;

    bool IsKnownRegister;
    ARM64PState::PStateMapper().fromString(getSysReg(), IsKnownRegister);

    return IsKnownRegister;
  }
  bool isReg() const override { return Kind == k_Register && !Reg.isVector; }
  bool isVectorReg() const { return Kind == k_Register && Reg.isVector; }
  bool isVectorRegLo() const {
    return Kind == k_Register && Reg.isVector &&
      ARM64MCRegisterClasses[ARM64::FPR128_loRegClassID].contains(Reg.RegNum);
  }

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
  bool isToken() const override { return Kind == k_Token; }
  bool isTokenEqual(StringRef Str) const {
    return Kind == k_Token && getToken() == Str;
  }
  bool isMem() const override { return Kind == k_Memory; }
  bool isSysCR() const { return Kind == k_SysCR; }
  bool isPrefetch() const { return Kind == k_Prefetch; }
  bool isShifter() const { return Kind == k_Shifter; }
  bool isExtend() const {
    // lsl is an alias for UXTW but will be a parsed as a k_Shifter operand.
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
    int64_t Addend;
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
      return Addend >= 0 && (Addend % Scale) == 0;
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
    return Mem.OffsetImm == nullptr;
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
    if (!isImm())
        return false;

    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Val = CE->getValue();
      int64_t Min = - (4096 * (1LL << (21 - 1)));
      int64_t Max = 4096 * ((1LL << (21 - 1)) - 1);
      return (Val % 4096) == 0 && Val >= Min && Val <= Max;
    }

    return true;
  }

  bool isAdrLabel() const {
    // Validation was handled during parsing, so we just sanity check that
    // something didn't go haywire.
    if (!isImm())
        return false;

    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Val = CE->getValue();
      int64_t Min = - (1LL << (21 - 1));
      int64_t Max = ((1LL << (21 - 1)) - 1);
      return Val >= Min && Val <= Max;
    }

    return true;
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (!Expr)
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

  void addVectorRegLoOperands(MCInst &Inst, unsigned N) const {
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
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(getImm());
    if (!MCE)
      addExpr(Inst, getImm());
    else
      Inst.addOperand(MCOperand::CreateImm(MCE->getValue() >> 12));
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

  void addPCRelLabel19Operands(MCInst &Inst, unsigned N) const {
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

  void addMRSSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    auto Mapper = ARM64SysReg::MRSMapper(getSysRegFeatureBits());
    uint32_t Bits = Mapper.fromString(getSysReg(), Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addMSRSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    auto Mapper = ARM64SysReg::MSRMapper(getSysRegFeatureBits());
    uint32_t Bits = Mapper.fromString(getSysReg(), Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addSystemPStateFieldOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    uint32_t Bits = ARM64PState::PStateMapper().fromString(getSysReg(), Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
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
    // lsl is an alias for UXTW but will be a parsed as a k_Shifter operand.
    if (isShifter()) {
      assert(ARM64_AM::getShiftType(getShifter()) == ARM64_AM::LSL);
      unsigned imm = getArithExtendImm(ARM64_AM::UXTW,
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
    Inst.addOperand(MCOperand::CreateReg(getXRegFromWReg(Mem.OffsetRegNum)));
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
    int64_t Addend;
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

  void print(raw_ostream &OS) const override;

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

  static ARM64Operand *CreateSysReg(StringRef Str, SMLoc S,
                                    uint64_t FeatureBits, MCContext &Ctx) {
    ARM64Operand *Op = new ARM64Operand(k_SysReg, Ctx);
    Op->SysReg.Data = Str.data();
    Op->SysReg.Length = Str.size();
    Op->SysReg.FeatureBits = FeatureBits;
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
    Op->Mem.OffsetImm = nullptr;
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
    bool Valid;
    StringRef Name = ARM64DB::DBarrierMapper().toString(getBarrier(), Valid);
    if (Valid)
      OS << "<barrier " << Name << ">";
    else
      OS << "<barrier invalid #" << getBarrier() << ">";
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
  case k_SysReg:
    OS << "<sysreg: " << getSysReg() << '>';
    break;
  case k_Token:
    OS << "'" << getToken() << "'";
    break;
  case k_SysCR:
    OS << "c" << getSysCR();
    break;
  case k_Prefetch: {
    bool Valid;
    StringRef Name = ARM64PRFM::PRFMMapper().toString(getPrefetch(), Valid);
    if (Valid)
      OS << "<prfop " << Name << ">";
    else
      OS << "<prfop invalid #" << getPrefetch() << ">";
    break;
  }
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
                 .Case("fp",  ARM64::FP)
                 .Case("lr",  ARM64::LR)
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
int ARM64AsmParser::tryMatchVectorRegister(StringRef &Kind, bool expected) {
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

  if (expected)
    TokError("vector register expected");
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
  bool Hash = Tok.is(AsmToken::Hash);
  if (Hash || Tok.is(AsmToken::Integer)) {
    if (Hash)
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

  bool Valid;
  unsigned prfop = ARM64PRFM::PRFMMapper().fromString(Tok.getString(), Valid);
  if (!Valid) {
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

  if (Parser.getTok().is(AsmToken::Hash)) {
    Parser.Lex(); // Eat hash token.
  }

  if (parseSymbolicImmVal(Expr))
    return MatchOperand_ParseFail;

  ARM64MCExpr::VariantKind ELFRefKind;
  MCSymbolRefExpr::VariantKind DarwinRefKind;
  int64_t Addend;
  if (classifySymbolRef(Expr, ELFRefKind, DarwinRefKind, Addend)) {
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
  }

  // We have either a label reference possibly with addend or an immediate. The
  // addend is a raw value here. The linker will adjust it to only reference the
  // page.
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

  if (Parser.getTok().is(AsmToken::Hash)) {
    Parser.Lex(); // Eat hash token.
  }

  if (getParser().parseExpression(Expr))
    return MatchOperand_ParseFail;

  SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
  Operands.push_back(ARM64Operand::CreateImm(Expr, S, E, getContext()));

  return MatchOperand_Success;
}

/// tryParseFPImm - A floating point immediate expression operand.
ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseFPImm(OperandVector &Operands) {
  SMLoc S = getLoc();

  bool Hash = false;
  if (Parser.getTok().is(AsmToken::Hash)) {
    Parser.Lex(); // Eat '#'
    Hash = true;
  }

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

  if (!Hash)
    return MatchOperand_NoMatch;

  TokError("invalid floating point immediate");
  return MatchOperand_ParseFail;
}

/// parseCondCodeString - Parse a Condition Code string.
unsigned ARM64AsmParser::parseCondCodeString(StringRef Cond) {
  unsigned CC = StringSwitch<unsigned>(Cond.lower())
                    .Case("eq", ARM64CC::EQ)
                    .Case("ne", ARM64CC::NE)
                    .Case("cs", ARM64CC::HS)
                    .Case("hs", ARM64CC::HS)
                    .Case("cc", ARM64CC::LO)
                    .Case("lo", ARM64CC::LO)
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
                    .Case("nv", ARM64CC::NV)
                    .Default(ARM64CC::Invalid);
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
  if (CC == ARM64CC::Invalid)
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
  bool Hash = getLexer().is(AsmToken::Hash);
  if (!Hash && getLexer().isNot(AsmToken::Integer))
    return TokError("immediate value expected for shifter operand");

  if (Hash)
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

  bool Hash = getLexer().is(AsmToken::Hash);
  if (!Hash && getLexer().isNot(AsmToken::Integer)) {
    SMLoc E = SMLoc::getFromPointer(getLoc().getPointer() - 1);
    Operands.push_back(
        ARM64Operand::CreateExtend(ExtOp, 0, S, E, getContext()));
    return false;
  }

  if (Hash)
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

  const MCExpr *Expr = nullptr;

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
    } else if (!Op.compare_lower("ipas2e1is")) {
      // SYS #4, C8, C4, #1
      SYS_ALIAS(4, 8, 0, 1);
    } else if (!Op.compare_lower("ipas2le1is")) {
      // SYS #4, C8, C4, #5
      SYS_ALIAS(4, 8, 0, 5);
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

  bool ExpectRegister = (Op.lower().find("all") == StringRef::npos);
  bool HasRegister = false;

  // Check for the optional register operand.
  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat comma.

    if (Tok.isNot(AsmToken::Identifier) || parseRegister(Operands))
      return TokError("expected register operand");

    HasRegister = true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Parser.eatToEndOfStatement();
    return TokError("unexpected token in argument list");
  }

  if (ExpectRegister && !HasRegister) {
    return TokError("specified " + Mnemonic + " op requires a register");
  }
  else if (!ExpectRegister && HasRegister) {
    return TokError("specified " + Mnemonic + " op does not use a register");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseBarrierOperand(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();

  // Can be either a #imm style literal or an option name
  bool Hash = Tok.is(AsmToken::Hash);
  if (Hash || Tok.is(AsmToken::Integer)) {
    // Immediate operand.
    if (Hash)
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

  bool Valid;
  unsigned Opt = ARM64DB::DBarrierMapper().fromString(Tok.getString(), Valid);
  if (!Valid) {
    TokError("invalid barrier option name");
    return MatchOperand_ParseFail;
  }

  // The only valid named option for ISB is 'sy'
  if (Mnemonic == "isb" && Opt != ARM64DB::SY) {
    TokError("'sy' or #imm operand expected");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(ARM64Operand::CreateBarrier(Opt, getLoc(), getContext()));
  Parser.Lex(); // Consume the option

  return MatchOperand_Success;
}

ARM64AsmParser::OperandMatchResultTy
ARM64AsmParser::tryParseSysReg(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();

  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  Operands.push_back(ARM64Operand::CreateSysReg(Tok.getString(), getLoc(),
                     STI.getFeatureBits(), getContext()));
  Parser.Lex(); // Eat identifier

  return MatchOperand_Success;
}

/// tryParseVectorRegister - Parse a vector register operand.
bool ARM64AsmParser::tryParseVectorRegister(OperandVector &Operands) {
  if (Parser.getTok().isNot(AsmToken::Identifier))
    return true;

  SMLoc S = getLoc();
  // Check for a vector register specifier first.
  StringRef Kind;
  int64_t Reg = tryMatchVectorRegister(Kind, false);
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
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      TokError("immediate value expected for vector index");
      return false;
    }

    SMLoc E = getLoc();
    if (Parser.getTok().isNot(AsmToken::RBrac)) {
      Error(E, "']' expected");
      return false;
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

  Operands.push_back(ARM64Operand::CreateMem(Reg, nullptr, S, E, E, getContext()));
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
  const MCExpr *OffsetExpr = nullptr;
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

        // A 32-bit offset register is only valid for [SU]/XTW extend
        // operators.
        if (ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(Reg2)) {
         if (ExtOp != ARM64_AM::UXTW &&
            ExtOp != ARM64_AM::SXTW)
          return Error(ExtLoc, "32-bit general purpose offset register "
                               "requires sxtw or uxtw extend");
        } else if (!ARM64MCRegisterClasses[ARM64::GPR64allRegClassID].contains(
                       Reg2))
          return Error(OffsetLoc,
                       "64-bit general purpose offset register expected");

        bool Hash = getLexer().is(AsmToken::Hash);
        if (getLexer().is(AsmToken::RBrac)) {
          // No immediate operand.
          if (ExtOp == ARM64_AM::UXTX)
            return Error(ExtLoc, "LSL extend requires immediate operand");
        } else if (Hash || getLexer().is(AsmToken::Integer)) {
          // Immediate operand.
          if (Hash)
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
    } else if (Parser.getTok().is(AsmToken::Hash) ||
               Parser.getTok().is(AsmToken::Colon) ||
               Parser.getTok().is(AsmToken::Integer)) {
      if (Parser.getTok().is(AsmToken::Hash))
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
      int64_t Addend;
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
                  .Case("abs_g2_s", ARM64MCExpr::VK_ABS_G2_S)
                  .Case("abs_g2_nc", ARM64MCExpr::VK_ABS_G2_NC)
                  .Case("abs_g1", ARM64MCExpr::VK_ABS_G1)
                  .Case("abs_g1_s", ARM64MCExpr::VK_ABS_G1_S)
                  .Case("abs_g1_nc", ARM64MCExpr::VK_ABS_G1_NC)
                  .Case("abs_g0", ARM64MCExpr::VK_ABS_G0)
                  .Case("abs_g0_s", ARM64MCExpr::VK_ABS_G0_S)
                  .Case("abs_g0_nc", ARM64MCExpr::VK_ABS_G0_NC)
                  .Case("dtprel_g2", ARM64MCExpr::VK_DTPREL_G2)
                  .Case("dtprel_g1", ARM64MCExpr::VK_DTPREL_G1)
                  .Case("dtprel_g1_nc", ARM64MCExpr::VK_DTPREL_G1_NC)
                  .Case("dtprel_g0", ARM64MCExpr::VK_DTPREL_G0)
                  .Case("dtprel_g0_nc", ARM64MCExpr::VK_DTPREL_G0_NC)
                  .Case("dtprel_hi12", ARM64MCExpr::VK_DTPREL_HI12)
                  .Case("dtprel_lo12", ARM64MCExpr::VK_DTPREL_LO12)
                  .Case("dtprel_lo12_nc", ARM64MCExpr::VK_DTPREL_LO12_NC)
                  .Case("tprel_g2", ARM64MCExpr::VK_TPREL_G2)
                  .Case("tprel_g1", ARM64MCExpr::VK_TPREL_G1)
                  .Case("tprel_g1_nc", ARM64MCExpr::VK_TPREL_G1_NC)
                  .Case("tprel_g0", ARM64MCExpr::VK_TPREL_G0)
                  .Case("tprel_g0_nc", ARM64MCExpr::VK_TPREL_G0_NC)
                  .Case("tprel_hi12", ARM64MCExpr::VK_TPREL_HI12)
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
  int64_t FirstReg = tryMatchVectorRegister(Kind, true);
  if (FirstReg == -1)
    return true;
  int64_t PrevReg = FirstReg;
  unsigned Count = 1;

  if (Parser.getTok().is(AsmToken::Minus)) {
    Parser.Lex(); // Eat the minus.

    SMLoc Loc = getLoc();
    StringRef NextKind;
    int64_t Reg = tryMatchVectorRegister(NextKind, true);
    if (Reg == -1)
      return true;
    // Any Kind suffices must match on all regs in the list.
    if (Kind != NextKind)
      return Error(Loc, "mismatched register size suffix");

    unsigned Space = (PrevReg < Reg) ? (Reg - PrevReg) : (Reg + 32 - PrevReg);

    if (Space == 0 || Space > 3) {
      return Error(Loc, "invalid number of vectors");
    }

    Count += Space;
  }
  else {
    while (Parser.getTok().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma token.

      SMLoc Loc = getLoc();
      StringRef NextKind;
      int64_t Reg = tryMatchVectorRegister(NextKind, true);
      if (Reg == -1)
        return true;
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
  }

  if (Parser.getTok().is(AsmToken::EndOfStatement))
    Error(getLoc(), "'}' expected");
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
      return false;
    const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!MCE) {
      TokError("immediate value expected for vector index");
      return false;
    }

    SMLoc E = getLoc();
    if (Parser.getTok().isNot(AsmToken::RBrac)) {
      Error(E, "']' expected");
      return false;
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
  case AsmToken::Integer:
  case AsmToken::Real:
  case AsmToken::Hash: {
    // #42 -> immediate.
    S = getLoc();
    if (getLexer().is(AsmToken::Hash))
      Parser.Lex();

    // The only Real that should come through here is a literal #0.0 for
    // the fcmp[e] r, #0.0 instructions. They expect raw token operands,
    // so convert the value.
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Real)) {
      APFloat RealVal(APFloat::IEEEdouble, Tok.getString());
      uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
      if (IntVal != 0 ||
          (Mnemonic != "fcmp" && Mnemonic != "fcmpe" && Mnemonic != "fcmeq" &&
           Mnemonic != "fcmge" && Mnemonic != "fcmgt" && Mnemonic != "fcmle" &&
           Mnemonic != "fcmlt"))
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
  Name = StringSwitch<StringRef>(Name.lower())
             .Case("beq", "b.eq")
             .Case("bne", "b.ne")
             .Case("bhs", "b.hs")
             .Case("bcs", "b.cs")
             .Case("blo", "b.lo")
             .Case("bcc", "b.cc")
             .Case("bmi", "b.mi")
             .Case("bpl", "b.pl")
             .Case("bvs", "b.vs")
             .Case("bvc", "b.vc")
             .Case("bhi", "b.hi")
             .Case("bls", "b.ls")
             .Case("bge", "b.ge")
             .Case("blt", "b.lt")
             .Case("bgt", "b.gt")
             .Case("ble", "b.le")
             .Case("bal", "b.al")
             .Case("bnv", "b.nv")
             .Default(Name);

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
    if (CC == ARM64CC::Invalid)
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
    if ((Inst.getOpcode() == ARM64::ADDXri ||
         Inst.getOpcode() == ARM64::ADDWri) &&
        Inst.getOperand(2).isExpr()) {
      const MCExpr *Expr = Inst.getOperand(2).getExpr();
      ARM64MCExpr::VariantKind ELFRefKind;
      MCSymbolRefExpr::VariantKind DarwinRefKind;
      int64_t Addend;
      if (!classifySymbolRef(Expr, ELFRefKind, DarwinRefKind, Addend)) {
        return Error(Loc[2], "invalid immediate expression");
      }

      // Note that we don't range-check the addend. It's adjusted modulo page
      // size when converted, so there is no "out of range" condition when using
      // @pageoff. Any validity checking for the value was done in the is*()
      // predicate function.
      if ((DarwinRefKind == MCSymbolRefExpr::VK_PAGEOFF ||
           DarwinRefKind == MCSymbolRefExpr::VK_TLVPPAGEOFF) &&
          Inst.getOpcode() == ARM64::ADDXri)
        return false;
      if (ELFRefKind == ARM64MCExpr::VK_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_DTPREL_HI12 ||
          ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_DTPREL_LO12_NC ||
          ELFRefKind == ARM64MCExpr::VK_TPREL_HI12 ||
          ELFRefKind == ARM64MCExpr::VK_TPREL_LO12 ||
          ELFRefKind == ARM64MCExpr::VK_TPREL_LO12_NC ||
          ELFRefKind == ARM64MCExpr::VK_TLSDESC_LO12) {
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

static void rewriteMOVI(ARM64AsmParser::OperandVector &Operands,
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

static void rewriteMOVRSP(ARM64AsmParser::OperandVector &Operands,
                        MCContext &Context) {
  ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[0]);
  ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
  Operands[0] =
    ARM64Operand::CreateToken("add", false, Op->getStartLoc(), Context);

  const MCExpr *Imm = MCConstantExpr::Create(0, Context);
  Operands.push_back(ARM64Operand::CreateImm(Imm, Op2->getStartLoc(),
                                             Op2->getEndLoc(), Context));
  Operands.push_back(ARM64Operand::CreateShifter(
      ARM64_AM::LSL, 0, Op2->getStartLoc(), Op2->getEndLoc(), Context));

  delete Op;
}

static void rewriteMOVR(ARM64AsmParser::OperandVector &Operands,
                        MCContext &Context) {
  ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[0]);
  ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
  Operands[0] =
    ARM64Operand::CreateToken("orr", false, Op->getStartLoc(), Context);

  // Operands[2] becomes Operands[3].
  Operands.push_back(Operands[2]);
  // And Operands[2] becomes ZR.
  unsigned ZeroReg = ARM64::XZR;
  if (ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(
          Operands[2]->getReg()))
    ZeroReg = ARM64::WZR;

  Operands[2] =
    ARM64Operand::CreateReg(ZeroReg, false, Op2->getStartLoc(),
                            Op2->getEndLoc(), Context);

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
    return Error(Loc, "index must be an integer in range [-256, 255].");
  case Match_InvalidMemoryIndexed32SImm7:
    return Error(Loc, "index must be a multiple of 4 in range [-256, 252].");
  case Match_InvalidMemoryIndexed64SImm7:
    return Error(Loc, "index must be a multiple of 8 in range [-512, 504].");
  case Match_InvalidMemoryIndexed128SImm7:
    return Error(Loc, "index must be a multiple of 16 in range [-1024, 1008].");
  case Match_InvalidMemoryIndexed8:
    return Error(Loc, "index must be an integer in range [0, 4095].");
  case Match_InvalidMemoryIndexed16:
    return Error(Loc, "index must be a multiple of 2 in range [0, 8190].");
  case Match_InvalidMemoryIndexed32:
    return Error(Loc, "index must be a multiple of 4 in range [0, 16380].");
  case Match_InvalidMemoryIndexed64:
    return Error(Loc, "index must be a multiple of 8 in range [0, 32760].");
  case Match_InvalidMemoryIndexed128:
    return Error(Loc, "index must be a multiple of 16 in range [0, 65520].");
  case Match_InvalidImm0_7:
    return Error(Loc, "immediate must be an integer in range [0, 7].");
  case Match_InvalidImm0_15:
    return Error(Loc, "immediate must be an integer in range [0, 15].");
  case Match_InvalidImm0_31:
    return Error(Loc, "immediate must be an integer in range [0, 31].");
  case Match_InvalidImm0_63:
    return Error(Loc, "immediate must be an integer in range [0, 63].");
  case Match_InvalidImm1_8:
    return Error(Loc, "immediate must be an integer in range [1, 8].");
  case Match_InvalidImm1_16:
    return Error(Loc, "immediate must be an integer in range [1, 16].");
  case Match_InvalidImm1_32:
    return Error(Loc, "immediate must be an integer in range [1, 32].");
  case Match_InvalidImm1_64:
    return Error(Loc, "immediate must be an integer in range [1, 64].");
  case Match_InvalidLabel:
    return Error(Loc, "expected label or encodable integer pc offset");
  case Match_MRS:
    return Error(Loc, "expected readable system register");
  case Match_MSR:
    return Error(Loc, "expected writable system register or pstate");
  case Match_MnemonicFail:
    return Error(Loc, "unrecognized instruction mnemonic");
  default:
    assert(0 && "unexpected error code!");
    return Error(Loc, "invalid instruction format");
  }
}

static const char *getSubtargetFeatureName(unsigned Val);

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
                           .Default(nullptr);
    assert(Repl && "Unknown compare instruction");
    delete Operands[0];
    Operands[0] = ARM64Operand::CreateToken(Repl, false, IDLoc, getContext());

    // Insert WZR or XZR as destination operand.
    ARM64Operand *RegOp = static_cast<ARM64Operand *>(Operands[1]);
    unsigned ZeroReg;
    if (RegOp->isReg() &&
        ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(
            RegOp->getReg()))
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

    ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
    ARM64Operand *Op2 = static_cast<ARM64Operand *>(Operands[2]);
    if (Op2->isImm()) {
      if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Op2->getImm())) {
        uint64_t Val = CE->getValue();
        uint64_t NVal = ~Val;

        // If this is a 32-bit register and the value has none of the upper
        // set, clear the complemented upper 32-bits so the logic below works
        // for 32-bit registers too.
        ARM64Operand *Op1 = static_cast<ARM64Operand *>(Operands[1]);
        if (Op1->isReg() &&
            ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(
                Op1->getReg()) &&
            (Val & 0xFFFFFFFFULL) == Val)
          NVal &= 0x00000000FFFFFFFFULL;

        // MOVK Rd, imm << 0
        if ((Val & 0xFFFF) == Val)
          rewriteMOVI(Operands, "movz", Val, 0, getContext());

        // MOVK Rd, imm << 16
        else if ((Val & 0xFFFF0000ULL) == Val)
          rewriteMOVI(Operands, "movz", Val, 16, getContext());

        // MOVK Rd, imm << 32
        else if ((Val & 0xFFFF00000000ULL) == Val)
          rewriteMOVI(Operands, "movz", Val, 32, getContext());

        // MOVK Rd, imm << 48
        else if ((Val & 0xFFFF000000000000ULL) == Val)
          rewriteMOVI(Operands, "movz", Val, 48, getContext());

        // MOVN Rd, (~imm << 0)
        else if ((NVal & 0xFFFFULL) == NVal)
          rewriteMOVI(Operands, "movn", NVal, 0, getContext());

        // MOVN Rd, ~(imm << 16)
        else if ((NVal & 0xFFFF0000ULL) == NVal)
          rewriteMOVI(Operands, "movn", NVal, 16, getContext());

        // MOVN Rd, ~(imm << 32)
        else if ((NVal & 0xFFFF00000000ULL) == NVal)
          rewriteMOVI(Operands, "movn", NVal, 32, getContext());

        // MOVN Rd, ~(imm << 48)
        else if ((NVal & 0xFFFF000000000000ULL) == NVal)
          rewriteMOVI(Operands, "movn", NVal, 48, getContext());
      }
    } else if (Op1->isReg() && Op2->isReg()) {
      // reg->reg move.
      unsigned Reg1 = Op1->getReg();
      unsigned Reg2 = Op2->getReg();
      if ((Reg1 == ARM64::SP &&
           ARM64MCRegisterClasses[ARM64::GPR64allRegClassID].contains(Reg2)) ||
          (Reg2 == ARM64::SP &&
           ARM64MCRegisterClasses[ARM64::GPR64allRegClassID].contains(Reg1)) ||
          (Reg1 == ARM64::WSP &&
           ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(Reg2)) ||
          (Reg2 == ARM64::WSP &&
           ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(Reg1)))
        rewriteMOVRSP(Operands, getContext());
      else
        rewriteMOVR(Operands, getContext());
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
          if (ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(
                  Op2->getReg())) {
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
          if (ARM64MCRegisterClasses[ARM64::GPR32allRegClassID].contains(
                  Op1->getReg()))
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
  // FIXME: Likewise for sxt[bh] with a Xd dst operand
  else if (NumOperands == 3 && (Tok == "sxtb" || Tok == "sxth")) {
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[1]);
    if (Op->isReg() &&
        ARM64MCRegisterClasses[ARM64::GPR64allRegClassID].contains(
            Op->getReg())) {
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
  // FIXME: Likewise for uxt[bh] with a Xd dst operand
  else if (NumOperands == 3 && (Tok == "uxtb" || Tok == "uxth")) {
    ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[1]);
    if (Op->isReg() &&
        ARM64MCRegisterClasses[ARM64::GPR64allRegClassID].contains(
            Op->getReg())) {
      // The source register can be Wn here, but the matcher expects a
      // GPR32. Twiddle it here if necessary.
      ARM64Operand *Op = static_cast<ARM64Operand *>(Operands[1]);
      if (Op->isReg()) {
        unsigned Reg = getWRegFromXReg(Op->getReg());
        Operands[1] = ARM64Operand::CreateReg(Reg, false, Op->getStartLoc(),
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
      unsigned zreg = ARM64MCRegisterClasses[ARM64::FPR32RegClassID].contains(
                          RegOp->getReg())
                          ? ARM64::WZR
                          : ARM64::XZR;
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
      delete Op;
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
        delete Op;
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
  case Match_MissingFeature: {
    assert(ErrorInfo && "Unknown missing feature!");
    // Special case the error message for the very common case where only
    // a single subtarget feature is missing (neon, e.g.).
    std::string Msg = "instruction requires:";
    unsigned Mask = 1;
    for (unsigned i = 0; i < (sizeof(ErrorInfo)*8-1); ++i) {
      if (ErrorInfo & Mask) {
        Msg += " ";
        Msg += getSubtargetFeatureName(ErrorInfo & Mask);
      }
      Mask <<= 1;
    }
    return Error(IDLoc, Msg);
  }
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
        if (PrevOp->isReg() &&
            ARM64MCRegisterClasses[ARM64::GPR32RegClassID].contains(
                PrevOp->getReg()))
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
  case Match_InvalidImm0_7:
  case Match_InvalidImm0_15:
  case Match_InvalidImm0_31:
  case Match_InvalidImm0_63:
  case Match_InvalidImm1_8:
  case Match_InvalidImm1_16:
  case Match_InvalidImm1_32:
  case Match_InvalidImm1_64:
  case Match_InvalidLabel:
  case Match_MSR:
  case Match_MRS: {
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
                                  int64_t &Addend) {
  ELFRefKind = ARM64MCExpr::VK_INVALID;
  DarwinRefKind = MCSymbolRefExpr::VK_None;
  Addend = 0;

  if (const ARM64MCExpr *AE = dyn_cast<ARM64MCExpr>(Expr)) {
    ELFRefKind = AE->getKind();
    Expr = AE->getSubExpr();
  }

  const MCSymbolRefExpr *SE = dyn_cast<MCSymbolRefExpr>(Expr);
  if (SE) {
    // It's a simple symbol reference with no addend.
    DarwinRefKind = SE->getKind();
    return true;
  }

  const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr);
  if (!BE)
    return false;

  SE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
  if (!SE)
    return false;
  DarwinRefKind = SE->getKind();

  if (BE->getOpcode() != MCBinaryExpr::Add &&
      BE->getOpcode() != MCBinaryExpr::Sub)
    return false;

  // See if the addend is is a constant, otherwise there's more going
  // on here than we can deal with.
  auto AddendExpr = dyn_cast<MCConstantExpr>(BE->getRHS());
  if (!AddendExpr)
    return false;

  Addend = AddendExpr->getValue();
  if (BE->getOpcode() == MCBinaryExpr::Sub)
    Addend = -Addend;

  // It's some symbol reference + a constant addend, but really
  // shouldn't use both Darwin and ELF syntax.
  return ELFRefKind == ARM64MCExpr::VK_INVALID ||
         DarwinRefKind == MCSymbolRefExpr::VK_None;
}

/// Force static initialization.
extern "C" void LLVMInitializeARM64AsmParser() {
  RegisterMCAsmParser<ARM64AsmParser> X(TheARM64leTarget);
  RegisterMCAsmParser<ARM64AsmParser> Y(TheARM64beTarget);
}

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
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
