//===-- AMDGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {

struct OptionalOperand;

class AMDGPUOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Register,
    Expression
  } Kind;

  SMLoc StartLoc, EndLoc;

public:
  AMDGPUOperand(enum KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  MCContext *Ctx;

  enum ImmTy {
    ImmTyNone,
    ImmTyDSOffset0,
    ImmTyDSOffset1,
    ImmTyGDS,
    ImmTyOffset,
    ImmTyGLC,
    ImmTySLC,
    ImmTyTFE,
    ImmTyClamp,
    ImmTyOMod
  };

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    bool IsFPImm;
    ImmTy Type;
    int64_t Val;
  };

  struct RegOp {
    unsigned RegNo;
    int Modifiers;
    const MCRegisterInfo *TRI;
    bool IsForcedVOP3;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    const MCExpr *Expr;
  };

  void addImmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createImm(getImm()));
  }

  StringRef getToken() const {
    return StringRef(Tok.Data, Tok.Length);
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    if (isReg())
      addRegOperands(Inst, N);
    else
      addImmOperands(Inst, N);
  }

  void addRegWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createImm(
        Reg.Modifiers == -1 ? 0 : Reg.Modifiers));
    addRegOperands(Inst, N);
  }

  void addSoppBrTargetOperands(MCInst &Inst, unsigned N) const {
    if (isImm())
      addImmOperands(Inst, N);
    else {
      assert(isExpr());
      Inst.addOperand(MCOperand::createExpr(Expr));
    }
  }

  bool defaultTokenHasSuffix() const {
    StringRef Token(Tok.Data, Tok.Length);

    return Token.endswith("_e32") || Token.endswith("_e64");
  }

  bool isToken() const override {
    return Kind == Token;
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  bool isInlineImm() const {
    float F = BitsToFloat(Imm.Val);
    // TODO: Add 0.5pi for VI
    return isImm() && ((Imm.Val <= 64 && Imm.Val >= -16) ||
           (F == 0.0 || F == 0.5 || F == -0.5 || F == 1.0 || F == -1.0 ||
           F == 2.0 || F == -2.0 || F == 4.0 || F == -4.0));
  }

  bool isDSOffset0() const {
    assert(isImm());
    return Imm.Type == ImmTyDSOffset0;
  }

  bool isDSOffset1() const {
    assert(isImm());
    return Imm.Type == ImmTyDSOffset1;
  }

  int64_t getImm() const {
    return Imm.Val;
  }

  enum ImmTy getImmTy() const {
    assert(isImm());
    return Imm.Type;
  }

  bool isRegKind() const {
    return Kind == Register;
  }

  bool isReg() const override {
    return Kind == Register && Reg.Modifiers == -1;
  }

  bool isRegWithInputMods() const {
    return Kind == Register && (Reg.IsForcedVOP3 || Reg.Modifiers != -1);
  }

  void setModifiers(unsigned Mods) {
    assert(isReg());
    Reg.Modifiers = Mods;
  }

  bool hasModifiers() const {
    assert(isRegKind());
    return Reg.Modifiers != -1;
  }

  unsigned getReg() const override {
    return Reg.RegNo;
  }

  bool isRegOrImm() const {
    return isReg() || isImm();
  }

  bool isRegClass(unsigned RCID) const {
    return Reg.TRI->getRegClass(RCID).contains(getReg());
  }

  bool isSCSrc32() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::SReg_32RegClassID));
  }

  bool isSSrc32() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::SReg_32RegClassID));
  }

  bool isSSrc64() const {
    return isImm() || isInlineImm() ||
           (isReg() && isRegClass(AMDGPU::SReg_64RegClassID));
  }

  bool isVCSrc32() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::VS_32RegClassID));
  }

  bool isVCSrc64() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::VS_64RegClassID));
  }

  bool isVSrc32() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::VS_32RegClassID));
  }

  bool isVSrc64() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::VS_64RegClassID));
  }

  bool isMem() const override {
    return false;
  }

  bool isExpr() const {
    return Kind == Expression;
  }

  bool isSoppBrTarget() const {
    return isExpr() || isImm();
  }

  SMLoc getStartLoc() const override {
    return StartLoc;
  }

  SMLoc getEndLoc() const override {
    return EndLoc;
  }

  void print(raw_ostream &OS) const override { }

  static std::unique_ptr<AMDGPUOperand> CreateImm(int64_t Val, SMLoc Loc,
                                                  enum ImmTy Type = ImmTyNone,
                                                  bool IsFPImm = false) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Immediate);
    Op->Imm.Val = Val;
    Op->Imm.IsFPImm = IsFPImm;
    Op->Imm.Type = Type;
    Op->StartLoc = Loc;
    Op->EndLoc = Loc;
    return Op;
  }

  static std::unique_ptr<AMDGPUOperand> CreateToken(StringRef Str, SMLoc Loc,
                                           bool HasExplicitEncodingSize = true) {
    auto Res = llvm::make_unique<AMDGPUOperand>(Token);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    Res->StartLoc = Loc;
    Res->EndLoc = Loc;
    return Res;
  }

  static std::unique_ptr<AMDGPUOperand> CreateReg(unsigned RegNo, SMLoc S,
                                                  SMLoc E,
                                                  const MCRegisterInfo *TRI,
                                                  bool ForceVOP3) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Register);
    Op->Reg.RegNo = RegNo;
    Op->Reg.TRI = TRI;
    Op->Reg.Modifiers = -1;
    Op->Reg.IsForcedVOP3 = ForceVOP3;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<AMDGPUOperand> CreateExpr(const class MCExpr *Expr, SMLoc S) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Expression);
    Op->Expr = Expr;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  bool isDSOffset() const;
  bool isDSOffset01() const;
  bool isSWaitCnt() const;
  bool isMubufOffset() const;
};

class AMDGPUAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  const MCInstrInfo &MII;
  MCAsmParser &Parser;

  unsigned ForcedEncodingSize;
  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "AMDGPUGenAsmMatcher.inc"

  /// }

public:
  AMDGPUAsmParser(MCSubtargetInfo &STI, MCAsmParser &_Parser,
               const MCInstrInfo &MII,
               const MCTargetOptions &Options)
      : MCTargetAsmParser(), STI(STI), MII(MII), Parser(_Parser),
        ForcedEncodingSize(0){

    if (!STI.getFeatureBits()) {
      // Set default features.
      STI.ToggleFeature("SOUTHERN_ISLANDS");
    }

    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  unsigned getForcedEncodingSize() const {
    return ForcedEncodingSize;
  }

  void setForcedEncodingSize(unsigned Size) {
    ForcedEncodingSize = Size;
  }

  bool isForcedVOP3() const {
    return ForcedEncodingSize == 64;
  }

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  unsigned checkTargetMatchPredicate(MCInst &Inst) override;
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseDirective(AsmToken DirectiveID) override;
  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Mnemonic);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  OperandMatchResultTy parseIntWithPrefix(const char *Prefix, int64_t &Int,
                                          int64_t Default = 0);
  OperandMatchResultTy parseIntWithPrefix(const char *Prefix,
                                          OperandVector &Operands,
                                          enum AMDGPUOperand::ImmTy ImmTy =
                                                      AMDGPUOperand::ImmTyNone);
  OperandMatchResultTy parseNamedBit(const char *Name, OperandVector &Operands,
                                     enum AMDGPUOperand::ImmTy ImmTy =
                                                      AMDGPUOperand::ImmTyNone);
  OperandMatchResultTy parseOptionalOps(
                                   const ArrayRef<OptionalOperand> &OptionalOps,
                                   OperandVector &Operands);


  void cvtDSOffset01(MCInst &Inst, const OperandVector &Operands);
  void cvtDS(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseDSOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseDSOff01OptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseDSOffsetOptional(OperandVector &Operands);

  bool parseCnt(int64_t &IntVal);
  OperandMatchResultTy parseSWaitCntOps(OperandVector &Operands);
  OperandMatchResultTy parseSOppBrTarget(OperandVector &Operands);

  void cvtMubuf(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseOffset(OperandVector &Operands);
  OperandMatchResultTy parseMubufOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseGLC(OperandVector &Operands);
  OperandMatchResultTy parseSLC(OperandVector &Operands);
  OperandMatchResultTy parseTFE(OperandVector &Operands);

  OperandMatchResultTy parseDMask(OperandVector &Operands);
  OperandMatchResultTy parseUNorm(OperandVector &Operands);
  OperandMatchResultTy parseR128(OperandVector &Operands);

  void cvtVOP3(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseVOP3OptionalOps(OperandVector &Operands);
};

struct OptionalOperand {
  const char *Name;
  AMDGPUOperand::ImmTy Type;
  bool IsBit;
  int64_t Default;
  bool (*ConvertResult)(int64_t&);
};

}

static unsigned getRegClass(bool IsVgpr, unsigned RegWidth) {
  if (IsVgpr) {
    switch (RegWidth) {
      default: llvm_unreachable("Unknown register width");
      case 1: return AMDGPU::VGPR_32RegClassID;
      case 2: return AMDGPU::VReg_64RegClassID;
      case 3: return AMDGPU::VReg_96RegClassID;
      case 4: return AMDGPU::VReg_128RegClassID;
      case 8: return AMDGPU::VReg_256RegClassID;
      case 16: return AMDGPU::VReg_512RegClassID;
    }
  }

  switch (RegWidth) {
    default: llvm_unreachable("Unknown register width");
    case 1: return AMDGPU::SGPR_32RegClassID;
    case 2: return AMDGPU::SGPR_64RegClassID;
    case 4: return AMDGPU::SReg_128RegClassID;
    case 8: return AMDGPU::SReg_256RegClassID;
    case 16: return AMDGPU::SReg_512RegClassID;
  }
}

static unsigned getRegForName(const StringRef &RegName) {

  return StringSwitch<unsigned>(RegName)
    .Case("exec", AMDGPU::EXEC)
    .Case("vcc", AMDGPU::VCC)
    .Case("flat_scr", AMDGPU::FLAT_SCR)
    .Case("m0", AMDGPU::M0)
    .Case("scc", AMDGPU::SCC)
    .Case("flat_scr_lo", AMDGPU::FLAT_SCR_LO)
    .Case("flat_scr_hi", AMDGPU::FLAT_SCR_HI)
    .Case("vcc_lo", AMDGPU::VCC_LO)
    .Case("vcc_hi", AMDGPU::VCC_HI)
    .Case("exec_lo", AMDGPU::EXEC_LO)
    .Case("exec_hi", AMDGPU::EXEC_HI)
    .Default(0);
}

bool AMDGPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  const StringRef &RegName = Tok.getString();
  RegNo = getRegForName(RegName);

  if (RegNo) {
    Parser.Lex();
    return false;
  }

  // Match vgprs and sgprs
  if (RegName[0] != 's' && RegName[0] != 'v')
    return true;

  bool IsVgpr = RegName[0] == 'v';
  unsigned RegWidth;
  unsigned RegIndexInClass;
  if (RegName.size() > 1) {
    // We have a 32-bit register
    RegWidth = 1;
    if (RegName.substr(1).getAsInteger(10, RegIndexInClass))
      return true;
    Parser.Lex();
  } else {
    // We have a register greater than 32-bits.

    int64_t RegLo, RegHi;
    Parser.Lex();
    if (getLexer().isNot(AsmToken::LBrac))
      return true;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegLo))
      return true;

    if (getLexer().isNot(AsmToken::Colon))
      return true;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegHi))
      return true;

    if (getLexer().isNot(AsmToken::RBrac))
      return true;

    Parser.Lex();
    RegWidth = (RegHi - RegLo) + 1;
    if (IsVgpr) {
      // VGPR registers aren't aligned.
      RegIndexInClass = RegLo;
    } else {
      // SGPR registers are aligned.  Max alignment is 4 dwords.
      RegIndexInClass = RegLo / std::min(RegWidth, 4u);
    }
  }

  const MCRegisterInfo *TRC = getContext().getRegisterInfo();
  unsigned RC = getRegClass(IsVgpr, RegWidth);
  if (RegIndexInClass > TRC->getRegClass(RC).getNumRegs())
    return true;
  RegNo = TRC->getRegClass(RC).getRegister(RegIndexInClass);
  return false;
}

unsigned AMDGPUAsmParser::checkTargetMatchPredicate(MCInst &Inst) {

  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;

  if ((getForcedEncodingSize() == 32 && (TSFlags & SIInstrFlags::VOP3)) ||
      (getForcedEncodingSize() == 64 && !(TSFlags & SIInstrFlags::VOP3)))
    return Match_InvalidOperand;

  return Match_Success;
}


bool AMDGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
    default: break;
    case Match_Success:
      Inst.setLoc(IDLoc);
      Out.EmitInstruction(Inst, STI);
      return false;
    case Match_MissingFeature:
      return Error(IDLoc, "instruction not supported on this GPU");

    case Match_MnemonicFail:
      return Error(IDLoc, "unrecognized instruction mnemonic");

    case Match_InvalidOperand: {
      SMLoc ErrorLoc = IDLoc;
      if (ErrorInfo != ~0ULL) {
        if (ErrorInfo >= Operands.size()) {
          if (isForcedVOP3()) {
            // If 64-bit encoding has been forced we can end up with no
            // clamp or omod operands if none of the registers have modifiers,
            // so we need to add these to the operand list.
            AMDGPUOperand &LastOp =
                ((AMDGPUOperand &)*Operands[Operands.size() - 1]);
            if (LastOp.isRegKind() ||
               (LastOp.isImm() &&
                LastOp.getImmTy() != AMDGPUOperand::ImmTyNone)) {
              SMLoc S = Parser.getTok().getLoc();
              Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                                 AMDGPUOperand::ImmTyClamp));
              Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                                 AMDGPUOperand::ImmTyOMod));
              bool Res = MatchAndEmitInstruction(IDLoc, Opcode, Operands,
                                                 Out, ErrorInfo,
                                                 MatchingInlineAsm);
              if (!Res)
                return Res;
            }

          }
          return Error(IDLoc, "too few operands for instruction");
        }

        ErrorLoc = ((AMDGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
        if (ErrorLoc == SMLoc())
          ErrorLoc = IDLoc;
      }
      return Error(ErrorLoc, "invalid operand for instruction");
    }
  }
  llvm_unreachable("Implement any new match types added!");
}

bool AMDGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  return true;
}

static bool operandsHaveModifiers(const OperandVector &Operands) {

  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    const AMDGPUOperand &Op = ((AMDGPUOperand&)*Operands[i]);
    if (Op.isRegKind() && Op.hasModifiers())
      return true;
    if (Op.isImm() && (Op.getImmTy() == AMDGPUOperand::ImmTyOMod ||
                       Op.getImmTy() == AMDGPUOperand::ImmTyClamp))
      return true;
  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {

  // Try to parse with a custom parser
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If we successfully parsed the operand or if there as an error parsing,
  // we are done.
  //
  // If we are parsing after we reach EndOfStatement then this means we
  // are appending default values to the Operands list.  This is only done
  // by custom parser, so we shouldn't continue on to the generic parsing.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail ||
      getLexer().is(AsmToken::EndOfStatement))
    return ResTy;

  bool Negate = false, Abs = false;
  if (getLexer().getKind()== AsmToken::Minus) {
    Parser.Lex();
    Negate = true;
  }

  if (getLexer().getKind() == AsmToken::Pipe) {
    Parser.Lex();
    Abs = true;
  }

  switch(getLexer().getKind()) {
    case AsmToken::Integer: {
      SMLoc S = Parser.getTok().getLoc();
      int64_t IntVal;
      if (getParser().parseAbsoluteExpression(IntVal))
        return MatchOperand_ParseFail;
      APInt IntVal32(32, IntVal);
      if (IntVal32.getSExtValue() != IntVal) {
        Error(S, "invalid immediate: only 32-bit values are legal");
        return MatchOperand_ParseFail;
      }

      IntVal = IntVal32.getSExtValue();
      if (Negate)
        IntVal *= -1;
      Operands.push_back(AMDGPUOperand::CreateImm(IntVal, S));
      return MatchOperand_Success;
    }
    case AsmToken::Real: {
      // FIXME: We should emit an error if a double precisions floating-point
      // value is used.  I'm not sure the best way to detect this.
      SMLoc S = Parser.getTok().getLoc();
      int64_t IntVal;
      if (getParser().parseAbsoluteExpression(IntVal))
        return MatchOperand_ParseFail;

      APFloat F((float)BitsToDouble(IntVal));
      if (Negate)
        F.changeSign();
      Operands.push_back(
          AMDGPUOperand::CreateImm(F.bitcastToAPInt().getZExtValue(), S));
      return MatchOperand_Success;
    }
    case AsmToken::Identifier: {
      SMLoc S, E;
      unsigned RegNo;
      if (!ParseRegister(RegNo, S, E)) {

        bool HasModifiers = operandsHaveModifiers(Operands);
        unsigned Modifiers = 0;

        if (Negate)
          Modifiers |= 0x1;

        if (Abs) {
          if (getLexer().getKind() != AsmToken::Pipe)
            return MatchOperand_ParseFail;
          Parser.Lex();
          Modifiers |= 0x2;
        }

        if (Modifiers && !HasModifiers) {
          // We are adding a modifier to src1 or src2 and previous sources
          // don't have modifiers, so we need to go back and empty modifers
          // for each previous source.
          for (unsigned PrevRegIdx = Operands.size() - 1; PrevRegIdx > 1;
               --PrevRegIdx) {

            AMDGPUOperand &RegOp = ((AMDGPUOperand&)*Operands[PrevRegIdx]);
            RegOp.setModifiers(0);
          }
        }


        Operands.push_back(AMDGPUOperand::CreateReg(
            RegNo, S, E, getContext().getRegisterInfo(),
            isForcedVOP3()));

        if (HasModifiers || Modifiers) {
          AMDGPUOperand &RegOp = ((AMDGPUOperand&)*Operands[Operands.size() - 1]);
          RegOp.setModifiers(Modifiers);

        }
     }  else {
      Operands.push_back(AMDGPUOperand::CreateToken(Parser.getTok().getString(),
                                                    S));
      Parser.Lex();
     }
     return MatchOperand_Success;
    }
    default:
      return MatchOperand_NoMatch;
  }
}

bool AMDGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name,
                                       SMLoc NameLoc, OperandVector &Operands) {

  // Clear any forced encodings from the previous instruction.
  setForcedEncodingSize(0);

  if (Name.endswith("_e64"))
    setForcedEncodingSize(64);
  else if (Name.endswith("_e32"))
    setForcedEncodingSize(32);

  // Add the instruction mnemonic
  Operands.push_back(AMDGPUOperand::CreateToken(Name, NameLoc));

  while (!getLexer().is(AsmToken::EndOfStatement)) {
    AMDGPUAsmParser::OperandMatchResultTy Res = parseOperand(Operands, Name);

    // Eat the comma or space if there is one.
    if (getLexer().is(AsmToken::Comma))
      Parser.Lex();

    switch (Res) {
      case MatchOperand_Success: break;
      case MatchOperand_ParseFail: return Error(getLexer().getLoc(),
                                                "failed parsing operand.");
      case MatchOperand_NoMatch: return Error(getLexer().getLoc(),
                                              "not a valid operand.");
    }
  }

  // Once we reach end of statement, continue parsing so we can add default
  // values for optional arguments.
  AMDGPUAsmParser::OperandMatchResultTy Res;
  while ((Res = parseOperand(Operands, Name)) != MatchOperand_NoMatch) {
    if (Res != MatchOperand_Success)
      return Error(getLexer().getLoc(), "failed parsing operand.");
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, int64_t &Int,
                                    int64_t Default) {

  // We are at the end of the statement, and this is a default argument, so
  // use a default value.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Int = Default;
    return MatchOperand_Success;
  }

  switch(getLexer().getKind()) {
    default: return MatchOperand_NoMatch;
    case AsmToken::Identifier: {
      StringRef OffsetName = Parser.getTok().getString();
      if (!OffsetName.equals(Prefix))
        return MatchOperand_NoMatch;

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Colon))
        return MatchOperand_ParseFail;

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;

      if (getParser().parseAbsoluteExpression(Int))
        return MatchOperand_ParseFail;
      break;
    }
  }
  return MatchOperand_Success;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, OperandVector &Operands,
                                    enum AMDGPUOperand::ImmTy ImmTy) {

  SMLoc S = Parser.getTok().getLoc();
  int64_t Offset = 0;

  AMDGPUAsmParser::OperandMatchResultTy Res = parseIntWithPrefix(Prefix, Offset);
  if (Res != MatchOperand_Success)
    return Res;

  Operands.push_back(AMDGPUOperand::CreateImm(Offset, S, ImmTy));
  return MatchOperand_Success;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseNamedBit(const char *Name, OperandVector &Operands,
                               enum AMDGPUOperand::ImmTy ImmTy) {
  int64_t Bit = 0;
  SMLoc S = Parser.getTok().getLoc();

  // We are at the end of the statement, and this is a default argument, so
  // use a default value.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    switch(getLexer().getKind()) {
      case AsmToken::Identifier: {
        StringRef Tok = Parser.getTok().getString();
        if (Tok == Name) {
          Bit = 1;
          Parser.Lex();
        } else if (Tok.startswith("no") && Tok.endswith(Name)) {
          Bit = 0;
          Parser.Lex();
        } else {
          return MatchOperand_NoMatch;
        }
        break;
      }
      default:
        return MatchOperand_NoMatch;
    }
  }

  Operands.push_back(AMDGPUOperand::CreateImm(Bit, S, ImmTy));
  return MatchOperand_Success;
}

static bool operandsHasOptionalOp(const OperandVector &Operands,
                                  const OptionalOperand &OOp) {
  for (unsigned i = 0; i < Operands.size(); i++) {
    const AMDGPUOperand &ParsedOp = ((const AMDGPUOperand &)*Operands[i]);
    if ((ParsedOp.isImm() && ParsedOp.getImmTy() == OOp.Type) ||
        (ParsedOp.isToken() && ParsedOp.getToken() == OOp.Name))
      return true;

  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOptionalOps(const ArrayRef<OptionalOperand> &OptionalOps,
                                   OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  for (const OptionalOperand &Op : OptionalOps) {
    if (operandsHasOptionalOp(Operands, Op))
      continue;
    AMDGPUAsmParser::OperandMatchResultTy Res;
    int64_t Value;
    if (Op.IsBit) {
      Res = parseNamedBit(Op.Name, Operands, Op.Type);
      if (Res == MatchOperand_NoMatch)
        continue;
      return Res;
    }

    Res = parseIntWithPrefix(Op.Name, Value, Op.Default);

    if (Res == MatchOperand_NoMatch)
      continue;

    if (Res != MatchOperand_Success)
      return Res;

    if (Op.ConvertResult && !Op.ConvertResult(Value)) {
      return MatchOperand_ParseFail;
    }

    Operands.push_back(AMDGPUOperand::CreateImm(Value, S, Op.Type));
    return MatchOperand_Success;
  }
  return MatchOperand_NoMatch;
}

//===----------------------------------------------------------------------===//
// ds
//===----------------------------------------------------------------------===//

static const OptionalOperand DSOptionalOps [] = {
  {"offset",  AMDGPUOperand::ImmTyOffset, false, 0, nullptr},
  {"gds",     AMDGPUOperand::ImmTyGDS, true, 0, nullptr}
};

static const OptionalOperand DSOptionalOpsOff01 [] = {
  {"offset0", AMDGPUOperand::ImmTyDSOffset0, false, 0, nullptr},
  {"offset1", AMDGPUOperand::ImmTyDSOffset1, false, 0, nullptr},
  {"gds",     AMDGPUOperand::ImmTyGDS, true, 0, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(DSOptionalOps, Operands);
}
AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOff01OptionalOps(OperandVector &Operands) {
  return parseOptionalOps(DSOptionalOpsOff01, Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOffsetOptional(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  AMDGPUAsmParser::OperandMatchResultTy Res =
    parseIntWithPrefix("offset", Operands, AMDGPUOperand::ImmTyOffset);
  if (Res == MatchOperand_NoMatch) {
    Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                       AMDGPUOperand::ImmTyOffset));
    Res = MatchOperand_Success;
  }
  return Res;
}

bool AMDGPUOperand::isDSOffset() const {
  return isImm() && isUInt<16>(getImm());
}

bool AMDGPUOperand::isDSOffset01() const {
  return isImm() && isUInt<8>(getImm());
}

void AMDGPUAsmParser::cvtDSOffset01(MCInst &Inst,
                                    const OperandVector &Operands) {

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  unsigned Offset0Idx = OptionalIdx[AMDGPUOperand::ImmTyDSOffset0];
  unsigned Offset1Idx = OptionalIdx[AMDGPUOperand::ImmTyDSOffset1];
  unsigned GDSIdx = OptionalIdx[AMDGPUOperand::ImmTyGDS];

  ((AMDGPUOperand &)*Operands[Offset0Idx]).addImmOperands(Inst, 1); // offset0
  ((AMDGPUOperand &)*Operands[Offset1Idx]).addImmOperands(Inst, 1); // offset1
  ((AMDGPUOperand &)*Operands[GDSIdx]).addImmOperands(Inst, 1); // gds
  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}

void AMDGPUAsmParser::cvtDS(MCInst &Inst, const OperandVector &Operands) {

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;
  bool GDSOnly = false;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    if (Op.isToken() && Op.getToken() == "gds") {
      GDSOnly = true;
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  unsigned OffsetIdx = OptionalIdx[AMDGPUOperand::ImmTyOffset];
  ((AMDGPUOperand &)*Operands[OffsetIdx]).addImmOperands(Inst, 1); // offset

  if (!GDSOnly) {
    unsigned GDSIdx = OptionalIdx[AMDGPUOperand::ImmTyGDS];
    ((AMDGPUOperand &)*Operands[GDSIdx]).addImmOperands(Inst, 1); // gds
  }
  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}


//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

bool AMDGPUAsmParser::parseCnt(int64_t &IntVal) {
  StringRef CntName = Parser.getTok().getString();
  int64_t CntVal;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::LParen))
    return true;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Integer))
    return true;

  if (getParser().parseAbsoluteExpression(CntVal))
    return true;

  if (getLexer().isNot(AsmToken::RParen))
    return true;

  Parser.Lex();
  if (getLexer().is(AsmToken::Amp) || getLexer().is(AsmToken::Comma))
    Parser.Lex();

  int CntShift;
  int CntMask;

  if (CntName == "vmcnt") {
    CntMask = 0xf;
    CntShift = 0;
  } else if (CntName == "expcnt") {
    CntMask = 0x7;
    CntShift = 4;
  } else if (CntName == "lgkmcnt") {
    CntMask = 0x7;
    CntShift = 8;
  } else {
    return true;
  }

  IntVal &= ~(CntMask << CntShift);
  IntVal |= (CntVal << CntShift);
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSWaitCntOps(OperandVector &Operands) {
  // Disable all counters by default.
  // vmcnt   [3:0]
  // expcnt  [6:4]
  // lgkmcnt [10:8]
  int64_t CntVal = 0x77f;
  SMLoc S = Parser.getTok().getLoc();

  switch(getLexer().getKind()) {
    default: return MatchOperand_ParseFail;
    case AsmToken::Integer:
      // The operand can be an integer value.
      if (getParser().parseAbsoluteExpression(CntVal))
        return MatchOperand_ParseFail;
      break;

    case AsmToken::Identifier:
      do {
        if (parseCnt(CntVal))
          return MatchOperand_ParseFail;
      } while(getLexer().isNot(AsmToken::EndOfStatement));
      break;
  }
  Operands.push_back(AMDGPUOperand::CreateImm(CntVal, S));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isSWaitCnt() const {
  return isImm();
}

//===----------------------------------------------------------------------===//
// sopp branch targets
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSOppBrTarget(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();

  switch (getLexer().getKind()) {
    default: return MatchOperand_ParseFail;
    case AsmToken::Integer: {
      int64_t Imm;
      if (getParser().parseAbsoluteExpression(Imm))
        return MatchOperand_ParseFail;
      Operands.push_back(AMDGPUOperand::CreateImm(Imm, S));
      return MatchOperand_Success;
    }

    case AsmToken::Identifier:
      Operands.push_back(AMDGPUOperand::CreateExpr(
          MCSymbolRefExpr::Create(getContext().getOrCreateSymbol(
                                  Parser.getTok().getString()), getContext()), S));
      Parser.Lex();
      return MatchOperand_Success;
  }
}

//===----------------------------------------------------------------------===//
// mubuf
//===----------------------------------------------------------------------===//

static const OptionalOperand MubufOptionalOps [] = {
  {"offset", AMDGPUOperand::ImmTyOffset, false, 0, nullptr},
  {"glc",    AMDGPUOperand::ImmTyGLC, true, 0, nullptr},
  {"slc",    AMDGPUOperand::ImmTySLC, true, 0, nullptr},
  {"tfe",    AMDGPUOperand::ImmTyTFE, true, 0, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseMubufOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(MubufOptionalOps, Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOffset(OperandVector &Operands) {
  return parseIntWithPrefix("offset", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseGLC(OperandVector &Operands) {
  return parseNamedBit("glc", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSLC(OperandVector &Operands) {
  return parseNamedBit("slc", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseTFE(OperandVector &Operands) {
  return parseNamedBit("tfe", Operands);
}

bool AMDGPUOperand::isMubufOffset() const {
  return isImm() && isUInt<12>(getImm());
}

void AMDGPUAsmParser::cvtMubuf(MCInst &Inst,
                               const OperandVector &Operands) {
  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle the case where soffset is an immediate
    if (Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyNone) {
      Op.addImmOperands(Inst, 1);
      continue;
    }

    // Handle tokens like 'offen' which are sometimes hard-coded into the
    // asm string.  There are no MCInst operands for these.
    if (Op.isToken()) {
      continue;
    }
    assert(Op.isImm());

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  assert(OptionalIdx.size() == 4);

  unsigned OffsetIdx = OptionalIdx[AMDGPUOperand::ImmTyOffset];
  unsigned GLCIdx = OptionalIdx[AMDGPUOperand::ImmTyGLC];
  unsigned SLCIdx = OptionalIdx[AMDGPUOperand::ImmTySLC];
  unsigned TFEIdx = OptionalIdx[AMDGPUOperand::ImmTyTFE];

  ((AMDGPUOperand &)*Operands[OffsetIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[GLCIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[SLCIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[TFEIdx]).addImmOperands(Inst, 1);
}

//===----------------------------------------------------------------------===//
// mimg
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDMask(OperandVector &Operands) {
  return parseIntWithPrefix("dmask", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseUNorm(OperandVector &Operands) {
  return parseNamedBit("unorm", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseR128(OperandVector &Operands) {
  return parseNamedBit("r128", Operands);
}

//===----------------------------------------------------------------------===//
// vop3
//===----------------------------------------------------------------------===//

static bool ConvertOmodMul(int64_t &Mul) {
  if (Mul != 1 && Mul != 2 && Mul != 4)
    return false;

  Mul >>= 1;
  return true;
}

static bool ConvertOmodDiv(int64_t &Div) {
  if (Div == 1) {
    Div = 0;
    return true;
  }

  if (Div == 2) {
    Div = 3;
    return true;
  }

  return false;
}

static const OptionalOperand VOP3OptionalOps [] = {
  {"clamp", AMDGPUOperand::ImmTyClamp, true, 0, nullptr},
  {"mul",   AMDGPUOperand::ImmTyOMod, false, 1, ConvertOmodMul},
  {"div",   AMDGPUOperand::ImmTyOMod, false, 1, ConvertOmodDiv},
};

static bool isVOP3(OperandVector &Operands) {
  if (operandsHaveModifiers(Operands))
    return true;

  AMDGPUOperand &DstOp = ((AMDGPUOperand&)*Operands[1]);

  if (DstOp.isReg() && DstOp.isRegClass(AMDGPU::SGPR_64RegClassID))
    return true;

  if (Operands.size() >= 5)
    return true;

  if (Operands.size() > 3) {
    AMDGPUOperand &Src1Op = ((AMDGPUOperand&)*Operands[3]);
    if (Src1Op.getReg() && (Src1Op.isRegClass(AMDGPU::SReg_32RegClassID) ||
                            Src1Op.isRegClass(AMDGPU::SReg_64RegClassID)))
      return true;
  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseVOP3OptionalOps(OperandVector &Operands) {

  // The value returned by this function may change after parsing
  // an operand so store the original value here.
  bool HasModifiers = operandsHaveModifiers(Operands);

  bool IsVOP3 = isVOP3(Operands);
  if (HasModifiers || IsVOP3 ||
      getLexer().isNot(AsmToken::EndOfStatement) ||
      getForcedEncodingSize() == 64) {

    AMDGPUAsmParser::OperandMatchResultTy Res =
        parseOptionalOps(VOP3OptionalOps, Operands);

    if (!HasModifiers && Res == MatchOperand_Success) {
      // We have added a modifier operation, so we need to make sure all
      // previous register operands have modifiers
      for (unsigned i = 2, e = Operands.size(); i != e; ++i) {
        AMDGPUOperand &Op = ((AMDGPUOperand&)*Operands[i]);
        if (Op.isReg())
          Op.setModifiers(0);
      }
    }
    return Res;
  }
  return MatchOperand_NoMatch;
}

void AMDGPUAsmParser::cvtVOP3(MCInst &Inst, const OperandVector &Operands) {
  ((AMDGPUOperand &)*Operands[1]).addRegOperands(Inst, 1);
  unsigned i = 2;

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  if (operandsHaveModifiers(Operands)) {
    for (unsigned e = Operands.size(); i != e; ++i) {
      AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

      if (Op.isRegWithInputMods()) {
        ((AMDGPUOperand &)*Operands[i]).addRegWithInputModsOperands(Inst, 2);
        continue;
      }
      OptionalIdx[Op.getImmTy()] = i;
    }

    unsigned ClampIdx = OptionalIdx[AMDGPUOperand::ImmTyClamp];
    unsigned OModIdx = OptionalIdx[AMDGPUOperand::ImmTyOMod];

    ((AMDGPUOperand &)*Operands[ClampIdx]).addImmOperands(Inst, 1);
    ((AMDGPUOperand &)*Operands[OModIdx]).addImmOperands(Inst, 1);
  } else {
    for (unsigned e = Operands.size(); i != e; ++i)
      ((AMDGPUOperand &)*Operands[i]).addRegOrImmOperands(Inst, 1);
  }
}

/// Force static initialization.
extern "C" void LLVMInitializeR600AsmParser() {
  RegisterMCAsmParser<AMDGPUAsmParser> A(TheAMDGPUTarget);
  RegisterMCAsmParser<AMDGPUAsmParser> B(TheGCNTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "AMDGPUGenAsmMatcher.inc"

