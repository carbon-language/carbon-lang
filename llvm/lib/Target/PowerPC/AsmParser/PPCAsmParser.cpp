//===-- PPCAsmParser.cpp - Parse PowerPC asm to MCInst instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCMCExpr.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static unsigned RRegs[32] = {
  PPC::R0,  PPC::R1,  PPC::R2,  PPC::R3,
  PPC::R4,  PPC::R5,  PPC::R6,  PPC::R7,
  PPC::R8,  PPC::R9,  PPC::R10, PPC::R11,
  PPC::R12, PPC::R13, PPC::R14, PPC::R15,
  PPC::R16, PPC::R17, PPC::R18, PPC::R19,
  PPC::R20, PPC::R21, PPC::R22, PPC::R23,
  PPC::R24, PPC::R25, PPC::R26, PPC::R27,
  PPC::R28, PPC::R29, PPC::R30, PPC::R31
};
static unsigned RRegsNoR0[32] = {
  PPC::ZERO,
            PPC::R1,  PPC::R2,  PPC::R3,
  PPC::R4,  PPC::R5,  PPC::R6,  PPC::R7,
  PPC::R8,  PPC::R9,  PPC::R10, PPC::R11,
  PPC::R12, PPC::R13, PPC::R14, PPC::R15,
  PPC::R16, PPC::R17, PPC::R18, PPC::R19,
  PPC::R20, PPC::R21, PPC::R22, PPC::R23,
  PPC::R24, PPC::R25, PPC::R26, PPC::R27,
  PPC::R28, PPC::R29, PPC::R30, PPC::R31
};
static unsigned XRegs[32] = {
  PPC::X0,  PPC::X1,  PPC::X2,  PPC::X3,
  PPC::X4,  PPC::X5,  PPC::X6,  PPC::X7,
  PPC::X8,  PPC::X9,  PPC::X10, PPC::X11,
  PPC::X12, PPC::X13, PPC::X14, PPC::X15,
  PPC::X16, PPC::X17, PPC::X18, PPC::X19,
  PPC::X20, PPC::X21, PPC::X22, PPC::X23,
  PPC::X24, PPC::X25, PPC::X26, PPC::X27,
  PPC::X28, PPC::X29, PPC::X30, PPC::X31
};
static unsigned XRegsNoX0[32] = {
  PPC::ZERO8,
            PPC::X1,  PPC::X2,  PPC::X3,
  PPC::X4,  PPC::X5,  PPC::X6,  PPC::X7,
  PPC::X8,  PPC::X9,  PPC::X10, PPC::X11,
  PPC::X12, PPC::X13, PPC::X14, PPC::X15,
  PPC::X16, PPC::X17, PPC::X18, PPC::X19,
  PPC::X20, PPC::X21, PPC::X22, PPC::X23,
  PPC::X24, PPC::X25, PPC::X26, PPC::X27,
  PPC::X28, PPC::X29, PPC::X30, PPC::X31
};
static unsigned FRegs[32] = {
  PPC::F0,  PPC::F1,  PPC::F2,  PPC::F3,
  PPC::F4,  PPC::F5,  PPC::F6,  PPC::F7,
  PPC::F8,  PPC::F9,  PPC::F10, PPC::F11,
  PPC::F12, PPC::F13, PPC::F14, PPC::F15,
  PPC::F16, PPC::F17, PPC::F18, PPC::F19,
  PPC::F20, PPC::F21, PPC::F22, PPC::F23,
  PPC::F24, PPC::F25, PPC::F26, PPC::F27,
  PPC::F28, PPC::F29, PPC::F30, PPC::F31
};
static unsigned VRegs[32] = {
  PPC::V0,  PPC::V1,  PPC::V2,  PPC::V3,
  PPC::V4,  PPC::V5,  PPC::V6,  PPC::V7,
  PPC::V8,  PPC::V9,  PPC::V10, PPC::V11,
  PPC::V12, PPC::V13, PPC::V14, PPC::V15,
  PPC::V16, PPC::V17, PPC::V18, PPC::V19,
  PPC::V20, PPC::V21, PPC::V22, PPC::V23,
  PPC::V24, PPC::V25, PPC::V26, PPC::V27,
  PPC::V28, PPC::V29, PPC::V30, PPC::V31
};
static unsigned CRBITRegs[32] = {
  PPC::CR0LT, PPC::CR0GT, PPC::CR0EQ, PPC::CR0UN,
  PPC::CR1LT, PPC::CR1GT, PPC::CR1EQ, PPC::CR1UN,
  PPC::CR2LT, PPC::CR2GT, PPC::CR2EQ, PPC::CR2UN,
  PPC::CR3LT, PPC::CR3GT, PPC::CR3EQ, PPC::CR3UN,
  PPC::CR4LT, PPC::CR4GT, PPC::CR4EQ, PPC::CR4UN,
  PPC::CR5LT, PPC::CR5GT, PPC::CR5EQ, PPC::CR5UN,
  PPC::CR6LT, PPC::CR6GT, PPC::CR6EQ, PPC::CR6UN,
  PPC::CR7LT, PPC::CR7GT, PPC::CR7EQ, PPC::CR7UN
};
static unsigned CRRegs[8] = {
  PPC::CR0, PPC::CR1, PPC::CR2, PPC::CR3,
  PPC::CR4, PPC::CR5, PPC::CR6, PPC::CR7
};

struct PPCOperand;

class PPCAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  bool IsPPC64;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  bool isPPC64() const { return IsPPC64; }

  bool MatchRegisterName(const AsmToken &Tok,
                         unsigned &RegNo, int64_t &IntVal);

  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  const MCExpr *ExtractModifierFromExpr(const MCExpr *E,
                                        PPCMCExpr::VariantKind &Variant);
  bool ParseExpression(const MCExpr *&EVal);

  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveTC(unsigned Size, SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  void ProcessInstruction(MCInst &Inst,
                          const SmallVectorImpl<MCParsedAsmOperand*> &Ops);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "PPCGenAsmMatcher.inc"

  /// }


public:
  PPCAsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser)
    : MCTargetAsmParser(), STI(_STI), Parser(_Parser) {
    // Check for 64-bit vs. 32-bit pointer mode.
    Triple TheTriple(STI.getTargetTriple());
    IsPPC64 = TheTriple.getArch() == Triple::ppc64;
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  virtual bool ParseInstruction(ParseInstructionInfo &Info,
                                StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};

/// PPCOperand - Instances of this class represent a parsed PowerPC machine
/// instruction.
struct PPCOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Expression
  } Kind;

  SMLoc StartLoc, EndLoc;
  bool IsPPC64;

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    int64_t Val;
  };

  struct ExprOp {
    const MCExpr *Val;
  };

  union {
    struct TokOp Tok;
    struct ImmOp Imm;
    struct ExprOp Expr;
  };

  PPCOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}
public:
  PPCOperand(const PPCOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    IsPPC64 = o.IsPPC64;
    switch (Kind) {
    case Token:
      Tok = o.Tok;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case Expression:
      Expr = o.Expr;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }

  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  /// isPPC64 - True if this operand is for an instruction in 64-bit mode.
  bool isPPC64() const { return IsPPC64; }

  int64_t getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  const MCExpr *getExpr() const {
    assert(Kind == Expression && "Invalid access!");
    return Expr.Val;
  }

  unsigned getReg() const {
    assert(isRegNumber() && "Invalid access!");
    return (unsigned) Imm.Val;
  }

  unsigned getCCReg() const {
    assert(isCCRegNumber() && "Invalid access!");
    return (unsigned) Imm.Val;
  }

  unsigned getCRBitMask() const {
    assert(isCRBitMask() && "Invalid access!");
    return 7 - countTrailingZeros<uint64_t>(Imm.Val);
  }

  bool isToken() const { return Kind == Token; }
  bool isImm() const { return Kind == Immediate || Kind == Expression; }
  bool isU5Imm() const { return Kind == Immediate && isUInt<5>(getImm()); }
  bool isS5Imm() const { return Kind == Immediate && isInt<5>(getImm()); }
  bool isU6Imm() const { return Kind == Immediate && isUInt<6>(getImm()); }
  bool isU16Imm() const { return Kind == Expression ||
                                 (Kind == Immediate && isUInt<16>(getImm())); }
  bool isS16Imm() const { return Kind == Expression ||
                                 (Kind == Immediate && isInt<16>(getImm())); }
  bool isS16ImmX4() const { return Kind == Expression ||
                                   (Kind == Immediate && isInt<16>(getImm()) &&
                                    (getImm() & 3) == 0); }
  bool isDirectBr() const { return Kind == Expression ||
                                   (Kind == Immediate && isInt<26>(getImm()) &&
                                    (getImm() & 3) == 0); }
  bool isCondBr() const { return Kind == Expression ||
                                 (Kind == Immediate && isInt<16>(getImm()) &&
                                  (getImm() & 3) == 0); }
  bool isRegNumber() const { return Kind == Immediate && isUInt<5>(getImm()); }
  bool isCCRegNumber() const { return Kind == Immediate &&
                                      isUInt<3>(getImm()); }
  bool isCRBitMask() const { return Kind == Immediate && isUInt<8>(getImm()) &&
                                    isPowerOf2_32(getImm()); }
  bool isMem() const { return false; }
  bool isReg() const { return false; }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("addRegOperands");
  }

  void addRegGPRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(RRegs[getReg()]));
  }

  void addRegGPRCNoR0Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(RRegsNoR0[getReg()]));
  }

  void addRegG8RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(XRegs[getReg()]));
  }

  void addRegG8RCNoX0Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(XRegsNoX0[getReg()]));
  }

  void addRegGxRCOperands(MCInst &Inst, unsigned N) const {
    if (isPPC64())
      addRegG8RCOperands(Inst, N);
    else
      addRegGPRCOperands(Inst, N);
  }

  void addRegGxRCNoR0Operands(MCInst &Inst, unsigned N) const {
    if (isPPC64())
      addRegG8RCNoX0Operands(Inst, N);
    else
      addRegGPRCNoR0Operands(Inst, N);
  }

  void addRegF4RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(FRegs[getReg()]));
  }

  void addRegF8RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(FRegs[getReg()]));
  }

  void addRegVRRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(VRegs[getReg()]));
  }

  void addRegCRBITRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(CRBITRegs[getReg()]));
  }

  void addRegCRRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(CRRegs[getCCReg()]));
  }

  void addCRBitMaskOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(CRRegs[getCRBitMask()]));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == Immediate)
      Inst.addOperand(MCOperand::CreateImm(getImm()));
    else
      Inst.addOperand(MCOperand::CreateExpr(getExpr()));
  }

  void addBranchTargetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == Immediate)
      Inst.addOperand(MCOperand::CreateImm(getImm() / 4));
    else
      Inst.addOperand(MCOperand::CreateExpr(getExpr()));
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  virtual void print(raw_ostream &OS) const;

  static PPCOperand *CreateToken(StringRef Str, SMLoc S, bool IsPPC64) {
    PPCOperand *Op = new PPCOperand(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static PPCOperand *CreateImm(int64_t Val, SMLoc S, SMLoc E, bool IsPPC64) {
    PPCOperand *Op = new PPCOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static PPCOperand *CreateExpr(const MCExpr *Val,
                                SMLoc S, SMLoc E, bool IsPPC64) {
    PPCOperand *Op = new PPCOperand(Expression);
    Op->Expr.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }
};

} // end anonymous namespace.

void PPCOperand::print(raw_ostream &OS) const {
  switch (Kind) {
  case Token:
    OS << "'" << getToken() << "'";
    break;
  case Immediate:
    OS << getImm();
    break;
  case Expression:
    getExpr()->print(OS);
    break;
  }
}


void PPCAsmParser::
ProcessInstruction(MCInst &Inst,
                   const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  switch (Inst.getOpcode()) {
  case PPC::SLWI: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::RLWINM);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRWI: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::RLWINM);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(32 - N));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::SLDI: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::RLDICR);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(63 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRDI: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::RLDICL);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(64 - N));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    Inst = TmpInst;
    break;
  }
  }
}

bool PPCAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  MCInst Inst;

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  default: break;
  case Match_Success:
    // Post-process instructions (typically extended mnemonics)
    ProcessInstruction(Inst, Operands);
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction use requires an option to be enabled");
  case Match_MnemonicFail:
      return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((PPCOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  llvm_unreachable("Implement any new match types added!");
}

bool PPCAsmParser::
MatchRegisterName(const AsmToken &Tok, unsigned &RegNo, int64_t &IntVal) {
  if (Tok.is(AsmToken::Identifier)) {
    StringRef Name = Tok.getString();

    if (Name.equals_lower("lr")) {
      RegNo = isPPC64()? PPC::LR8 : PPC::LR;
      IntVal = 8;
      return false;
    } else if (Name.equals_lower("ctr")) {
      RegNo = isPPC64()? PPC::CTR8 : PPC::CTR;
      IntVal = 9;
      return false;
    } else if (Name.substr(0, 1).equals_lower("r") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = isPPC64()? XRegs[IntVal] : RRegs[IntVal];
      return false;
    } else if (Name.substr(0, 1).equals_lower("f") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = FRegs[IntVal];
      return false;
    } else if (Name.substr(0, 1).equals_lower("v") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = VRegs[IntVal];
      return false;
    } else if (Name.substr(0, 2).equals_lower("cr") &&
               !Name.substr(2).getAsInteger(10, IntVal) && IntVal < 8) {
      RegNo = CRRegs[IntVal];
      return false;
    }
  }

  return true;
}

bool PPCAsmParser::
ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  int64_t IntVal;

  if (!MatchRegisterName(Tok, RegNo, IntVal)) {
    Parser.Lex(); // Eat identifier token.
    return false;
  }

  return Error(StartLoc, "invalid register name");
}

/// Extract @l/@ha modifier from expression.  Recursively scan
/// the expression and check for VK_PPC_LO/HI/HA
/// symbol variants.  If all symbols with modifier use the same
/// variant, return the corresponding PPCMCExpr::VariantKind,
/// and a modified expression using the default symbol variant.
/// Otherwise, return NULL.
const MCExpr *PPCAsmParser::
ExtractModifierFromExpr(const MCExpr *E,
                        PPCMCExpr::VariantKind &Variant) {
  MCContext &Context = getParser().getContext();
  Variant = PPCMCExpr::VK_PPC_None;

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return 0;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    switch (SRE->getKind()) {
    case MCSymbolRefExpr::VK_PPC_LO:
      Variant = PPCMCExpr::VK_PPC_LO;
      break;
    case MCSymbolRefExpr::VK_PPC_HI:
      Variant = PPCMCExpr::VK_PPC_HI;
      break;
    case MCSymbolRefExpr::VK_PPC_HA:
      Variant = PPCMCExpr::VK_PPC_HA;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHER:
      Variant = PPCMCExpr::VK_PPC_HIGHER;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHERA:
      Variant = PPCMCExpr::VK_PPC_HIGHERA;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHEST:
      Variant = PPCMCExpr::VK_PPC_HIGHEST;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHESTA:
      Variant = PPCMCExpr::VK_PPC_HIGHESTA;
      break;
    default:
      return 0;
    }

    return MCSymbolRefExpr::Create(&SRE->getSymbol(), Context);
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = ExtractModifierFromExpr(UE->getSubExpr(), Variant);
    if (!Sub)
      return 0;
    return MCUnaryExpr::Create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    PPCMCExpr::VariantKind LHSVariant, RHSVariant;
    const MCExpr *LHS = ExtractModifierFromExpr(BE->getLHS(), LHSVariant);
    const MCExpr *RHS = ExtractModifierFromExpr(BE->getRHS(), RHSVariant);

    if (!LHS && !RHS)
      return 0;

    if (!LHS) LHS = BE->getLHS();
    if (!RHS) RHS = BE->getRHS();

    if (LHSVariant == PPCMCExpr::VK_PPC_None)
      Variant = RHSVariant;
    else if (RHSVariant == PPCMCExpr::VK_PPC_None)
      Variant = LHSVariant;
    else if (LHSVariant == RHSVariant)
      Variant = LHSVariant;
    else
      return 0;

    return MCBinaryExpr::Create(BE->getOpcode(), LHS, RHS, Context);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// Parse an expression.  This differs from the default "parseExpression"
/// in that it handles complex @l/@ha modifiers.
bool PPCAsmParser::
ParseExpression(const MCExpr *&EVal) {
  if (getParser().parseExpression(EVal))
    return true;

  PPCMCExpr::VariantKind Variant;
  const MCExpr *E = ExtractModifierFromExpr(EVal, Variant);
  if (E)
    EVal = PPCMCExpr::Create(Variant, E, getParser().getContext());

  return false;
}

bool PPCAsmParser::
ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  const MCExpr *EVal;
  PPCOperand *Op;

  // Attempt to parse the next token as an immediate
  switch (getLexer().getKind()) {
  // Special handling for register names.  These are interpreted
  // as immediates corresponding to the register number.
  case AsmToken::Percent:
    Parser.Lex(); // Eat the '%'.
    unsigned RegNo;
    int64_t IntVal;
    if (!MatchRegisterName(Parser.getTok(), RegNo, IntVal)) {
      Parser.Lex(); // Eat the identifier token.
      Op = PPCOperand::CreateImm(IntVal, S, E, isPPC64());
      Operands.push_back(Op);
      return false;
    }
    return Error(S, "invalid register name");

  // All other expressions
  case AsmToken::LParen:
  case AsmToken::Plus:
  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Identifier:
  case AsmToken::Dot:
  case AsmToken::Dollar:
    if (!ParseExpression(EVal))
      break;
    /* fall through */
  default:
    return Error(S, "unknown operand");
  }

  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(EVal))
    Op = PPCOperand::CreateImm(CE->getValue(), S, E, isPPC64());
  else
    Op = PPCOperand::CreateExpr(EVal, S, E, isPPC64());

  // Push the parsed operand into the list of operands
  Operands.push_back(Op);

  // Check for D-form memory operands
  if (getLexer().is(AsmToken::LParen)) {
    Parser.Lex(); // Eat the '('.
    S = Parser.getTok().getLoc();

    int64_t IntVal;
    switch (getLexer().getKind()) {
    case AsmToken::Percent:
      Parser.Lex(); // Eat the '%'.
      unsigned RegNo;
      if (MatchRegisterName(Parser.getTok(), RegNo, IntVal))
        return Error(S, "invalid register name");
      Parser.Lex(); // Eat the identifier token.
      break;

    case AsmToken::Integer:
      if (getParser().parseAbsoluteExpression(IntVal) ||
          IntVal < 0 || IntVal > 31)
        return Error(S, "invalid register number");
      break;

    default:
      return Error(S, "invalid memory operand");
    }

    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "missing ')'");
    E = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the ')'.

    Op = PPCOperand::CreateImm(IntVal, S, E, isPPC64());
    Operands.push_back(Op);
  }

  return false;
}

/// Parse an instruction mnemonic followed by its operands.
bool PPCAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // The first operand is the token for the instruction name.
  // If the instruction ends in a '.', we need to create a separate
  // token for it, to match what TableGen is doing.
  size_t Dot = Name.find('.');
  StringRef Mnemonic = Name.slice(0, Dot);
  Operands.push_back(PPCOperand::CreateToken(Mnemonic, NameLoc, isPPC64()));
  if (Dot != StringRef::npos) {
    SMLoc DotLoc = SMLoc::getFromPointer(NameLoc.getPointer() + Dot);
    StringRef DotStr = Name.slice(Dot, StringRef::npos);
    Operands.push_back(PPCOperand::CreateToken(DotStr, DotLoc, isPPC64()));
  }

  // If there are no more operands then finish
  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Parse the first operand
  if (ParseOperand(Operands))
    return true;

  while (getLexer().isNot(AsmToken::EndOfStatement) &&
         getLexer().is(AsmToken::Comma)) {
    // Consume the comma token
    getLexer().Lex();

    // Parse the next operand
    if (ParseOperand(Operands))
      return true;
  }

  return false;
}

/// ParseDirective parses the PPC specific directives
bool PPCAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(4, DirectiveID.getLoc());
  if (IDVal == ".tc")
    return ParseDirectiveTC(isPPC64()? 8 : 4, DirectiveID.getLoc());
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool PPCAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// ParseDirectiveTC
///  ::= .tc [ symbol (, expression)* ]
bool PPCAsmParser::ParseDirectiveTC(unsigned Size, SMLoc L) {
  // Skip TC symbol, which is only used with XCOFF.
  while (getLexer().isNot(AsmToken::EndOfStatement)
         && getLexer().isNot(AsmToken::Comma))
    Parser.Lex();
  if (getLexer().isNot(AsmToken::Comma))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // Align to word size.
  getParser().getStreamer().EmitValueToAlignment(Size);

  // Emit expressions.
  return ParseDirectiveWord(Size, L);
}

/// Force static initialization.
extern "C" void LLVMInitializePowerPCAsmParser() {
  RegisterMCAsmParser<PPCAsmParser> A(ThePPC32Target);
  RegisterMCAsmParser<PPCAsmParser> B(ThePPC64Target);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "PPCGenAsmMatcher.inc"
