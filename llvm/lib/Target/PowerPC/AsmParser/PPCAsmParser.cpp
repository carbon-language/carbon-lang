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
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/ADT/STLExtras.h"
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

// Evaluate an expression containing condition register
// or condition register field symbols.  Returns positive
// value on success, or -1 on error.
static int64_t
EvaluateCRExpr(const MCExpr *E) {
  switch (E->getKind()) {
  case MCExpr::Target:
    return -1;

  case MCExpr::Constant: {
    int64_t Res = cast<MCConstantExpr>(E)->getValue();
    return Res < 0 ? -1 : Res;
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);
    StringRef Name = SRE->getSymbol().getName();

    if (Name == "lt") return 0;
    if (Name == "gt") return 1;
    if (Name == "eq") return 2;
    if (Name == "so") return 3;
    if (Name == "un") return 3;

    if (Name == "cr0") return 0;
    if (Name == "cr1") return 1;
    if (Name == "cr2") return 2;
    if (Name == "cr3") return 3;
    if (Name == "cr4") return 4;
    if (Name == "cr5") return 5;
    if (Name == "cr6") return 6;
    if (Name == "cr7") return 7;

    return -1;
  }

  case MCExpr::Unary:
    return -1;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    int64_t LHSVal = EvaluateCRExpr(BE->getLHS());
    int64_t RHSVal = EvaluateCRExpr(BE->getRHS());
    int64_t Res;

    if (LHSVal < 0 || RHSVal < 0)
      return -1;

    switch (BE->getOpcode()) {
    default: return -1;
    case MCBinaryExpr::Add: Res = LHSVal + RHSVal; break;
    case MCBinaryExpr::Mul: Res = LHSVal * RHSVal; break;
    }

    return Res < 0 ? -1 : Res;
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

struct PPCOperand;

class PPCAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  const MCInstrInfo &MII;
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
  const MCExpr *FixupVariantKind(const MCExpr *E);
  bool ParseExpression(const MCExpr *&EVal);

  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveTC(unsigned Size, SMLoc L);
  bool ParseDirectiveMachine(SMLoc L);

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
  PPCAsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser,
               const MCInstrInfo &_MII)
      : MCTargetAsmParser(), STI(_STI), Parser(_Parser), MII(_MII) {
    // Check for 64-bit vs. 32-bit pointer mode.
    Triple TheTriple(STI.getTargetTriple());
    IsPPC64 = (TheTriple.getArch() == Triple::ppc64 ||
               TheTriple.getArch() == Triple::ppc64le);
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  virtual bool ParseInstruction(ParseInstructionInfo &Info,
                                StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);

  unsigned validateTargetOperandClass(MCParsedAsmOperand *Op, unsigned Kind);

  virtual const MCExpr *applyModifierToExpr(const MCExpr *E,
                                            MCSymbolRefExpr::VariantKind,
                                            MCContext &Ctx);
};

/// PPCOperand - Instances of this class represent a parsed PowerPC machine
/// instruction.
struct PPCOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Expression,
    TLSRegister
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
    int64_t CRVal;     // Cached result of EvaluateCRExpr(Val)
  };

  struct TLSRegOp {
    const MCSymbolRefExpr *Sym;
  };

  union {
    struct TokOp Tok;
    struct ImmOp Imm;
    struct ExprOp Expr;
    struct TLSRegOp TLSReg;
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
    case TLSRegister:
      TLSReg = o.TLSReg;
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

  int64_t getExprCRVal() const {
    assert(Kind == Expression && "Invalid access!");
    return Expr.CRVal;
  }

  const MCExpr *getTLSReg() const {
    assert(Kind == TLSRegister && "Invalid access!");
    return TLSReg.Sym;
  }

  unsigned getReg() const {
    assert(isRegNumber() && "Invalid access!");
    return (unsigned) Imm.Val;
  }

  unsigned getCCReg() const {
    assert(isCCRegNumber() && "Invalid access!");
    return (unsigned) (Kind == Immediate ? Imm.Val : Expr.CRVal);
  }

  unsigned getCRBit() const {
    assert(isCRBitNumber() && "Invalid access!");
    return (unsigned) (Kind == Immediate ? Imm.Val : Expr.CRVal);
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
  bool isS17Imm() const { return Kind == Expression ||
                                 (Kind == Immediate && isInt<17>(getImm())); }
  bool isTLSReg() const { return Kind == TLSRegister; }
  bool isDirectBr() const { return Kind == Expression ||
                                   (Kind == Immediate && isInt<26>(getImm()) &&
                                    (getImm() & 3) == 0); }
  bool isCondBr() const { return Kind == Expression ||
                                 (Kind == Immediate && isInt<16>(getImm()) &&
                                  (getImm() & 3) == 0); }
  bool isRegNumber() const { return Kind == Immediate && isUInt<5>(getImm()); }
  bool isCCRegNumber() const { return (Kind == Expression
                                       && isUInt<3>(getExprCRVal())) ||
                                      (Kind == Immediate
                                       && isUInt<3>(getImm())); }
  bool isCRBitNumber() const { return (Kind == Expression
                                       && isUInt<5>(getExprCRVal())) ||
                                      (Kind == Immediate
                                       && isUInt<5>(getImm())); }
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
    Inst.addOperand(MCOperand::CreateReg(CRBITRegs[getCRBit()]));
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

  void addTLSRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateExpr(getTLSReg()));
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

  static PPCOperand *CreateTokenWithStringCopy(StringRef Str, SMLoc S,
                                               bool IsPPC64) {
    // Allocate extra memory for the string and copy it.
    void *Mem = ::operator new(sizeof(PPCOperand) + Str.size());
    PPCOperand *Op = new (Mem) PPCOperand(Token);
    Op->Tok.Data = (const char *)(Op + 1);
    Op->Tok.Length = Str.size();
    std::memcpy((char *)(Op + 1), Str.data(), Str.size());
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
    Op->Expr.CRVal = EvaluateCRExpr(Val);
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static PPCOperand *CreateTLSReg(const MCSymbolRefExpr *Sym,
                                  SMLoc S, SMLoc E, bool IsPPC64) {
    PPCOperand *Op = new PPCOperand(TLSRegister);
    Op->TLSReg.Sym = Sym;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static PPCOperand *CreateFromMCExpr(const MCExpr *Val,
                                      SMLoc S, SMLoc E, bool IsPPC64) {
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Val))
      return CreateImm(CE->getValue(), S, E, IsPPC64);

    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(Val))
      if (SRE->getKind() == MCSymbolRefExpr::VK_PPC_TLS)
        return CreateTLSReg(SRE, S, E, IsPPC64);

    return CreateExpr(Val, S, E, IsPPC64);
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
  case TLSRegister:
    getTLSReg()->print(OS);
    break;
  }
}


void PPCAsmParser::
ProcessInstruction(MCInst &Inst,
                   const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  int Opcode = Inst.getOpcode();
  switch (Opcode) {
  case PPC::LAx: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::LA);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::SUBI: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::ADDI);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(-N));
    Inst = TmpInst;
    break;
  }
  case PPC::SUBIS: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::ADDIS);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(-N));
    Inst = TmpInst;
    break;
  }
  case PPC::SUBIC: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::ADDIC);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(-N));
    Inst = TmpInst;
    break;
  }
  case PPC::SUBICo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(PPC::ADDICo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(-N));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTLWI:
  case PPC::EXTLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(B));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(N - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTRWI:
  case PPC::EXTRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(B + N));
    TmpInst.addOperand(MCOperand::CreateImm(32 - N));
    TmpInst.addOperand(MCOperand::CreateImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::INSLWI:
  case PPC::INSLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSLWI? PPC::RLWIMI : PPC::RLWIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(32 - B));
    TmpInst.addOperand(MCOperand::CreateImm(B));
    TmpInst.addOperand(MCOperand::CreateImm((B + N) - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::INSRWI:
  case PPC::INSRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSRWI? PPC::RLWIMI : PPC::RLWIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(32 - (B + N)));
    TmpInst.addOperand(MCOperand::CreateImm(B));
    TmpInst.addOperand(MCOperand::CreateImm((B + N) - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::ROTRWI:
  case PPC::ROTRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::ROTRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(32 - N));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::SLWI:
  case PPC::SLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRWI:
  case PPC::SRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(32 - N));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRRWI:
  case PPC::CLRRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRLSLWI:
  case PPC::CLRLSLWIo: {
    MCInst TmpInst;
    int64_t B = Inst.getOperand(2).getImm();
    int64_t N = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRLSLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(B - N));
    TmpInst.addOperand(MCOperand::CreateImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTLDI:
  case PPC::EXTLDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTLDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(B));
    TmpInst.addOperand(MCOperand::CreateImm(N - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTRDI:
  case PPC::EXTRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(B + N));
    TmpInst.addOperand(MCOperand::CreateImm(64 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::INSRDI:
  case PPC::INSRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSRDI? PPC::RLDIMI : PPC::RLDIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(64 - (B + N)));
    TmpInst.addOperand(MCOperand::CreateImm(B));
    Inst = TmpInst;
    break;
  }
  case PPC::ROTRDI:
  case PPC::ROTRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::ROTRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(64 - N));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    Inst = TmpInst;
    break;
  }
  case PPC::SLDI:
  case PPC::SLDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SLDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(63 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRDI:
  case PPC::SRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(64 - N));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRRDI:
  case PPC::CLRRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRRDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(0));
    TmpInst.addOperand(MCOperand::CreateImm(63 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRLSLDI:
  case PPC::CLRLSLDIo: {
    MCInst TmpInst;
    int64_t B = Inst.getOperand(2).getImm();
    int64_t N = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRLSLDI? PPC::RLDIC : PPC::RLDICo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::CreateImm(N));
    TmpInst.addOperand(MCOperand::CreateImm(B - N));
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
    } else if (Name.equals_lower("vrsave")) {
      RegNo = PPC::VRSAVE;
      IntVal = 256;
      return false;
    } else if (Name.startswith_lower("r") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = isPPC64()? XRegs[IntVal] : RRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("f") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = FRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("v") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = VRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("cr") &&
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

/// Extract \code @l/@ha \endcode modifier from expression.  Recursively scan
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

/// Find all VK_TLSGD/VK_TLSLD symbol references in expression and replace
/// them by VK_PPC_TLSGD/VK_PPC_TLSLD.  This is necessary to avoid having
/// _GLOBAL_OFFSET_TABLE_ created via ELFObjectWriter::RelocNeedsGOT.
/// FIXME: This is a hack.
const MCExpr *PPCAsmParser::
FixupVariantKind(const MCExpr *E) {
  MCContext &Context = getParser().getContext();

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return E;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);
    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;

    switch (SRE->getKind()) {
    case MCSymbolRefExpr::VK_TLSGD:
      Variant = MCSymbolRefExpr::VK_PPC_TLSGD;
      break;
    case MCSymbolRefExpr::VK_TLSLD:
      Variant = MCSymbolRefExpr::VK_PPC_TLSLD;
      break;
    default:
      return E;
    }
    return MCSymbolRefExpr::Create(&SRE->getSymbol(), Variant, Context);
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = FixupVariantKind(UE->getSubExpr());
    if (Sub == UE->getSubExpr())
      return E;
    return MCUnaryExpr::Create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = FixupVariantKind(BE->getLHS());
    const MCExpr *RHS = FixupVariantKind(BE->getRHS());
    if (LHS == BE->getLHS() && RHS == BE->getRHS())
      return E;
    return MCBinaryExpr::Create(BE->getOpcode(), LHS, RHS, Context);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// Parse an expression.  This differs from the default "parseExpression"
/// in that it handles complex \code @l/@ha \endcode modifiers.
bool PPCAsmParser::
ParseExpression(const MCExpr *&EVal) {
  if (getParser().parseExpression(EVal))
    return true;

  EVal = FixupVariantKind(EVal);

  PPCMCExpr::VariantKind Variant;
  const MCExpr *E = ExtractModifierFromExpr(EVal, Variant);
  if (E)
    EVal = PPCMCExpr::Create(Variant, E, false, getParser().getContext());

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

  // Push the parsed operand into the list of operands
  Op = PPCOperand::CreateFromMCExpr(EVal, S, E, isPPC64());
  Operands.push_back(Op);

  // Check whether this is a TLS call expression
  bool TLSCall = false;
  if (const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(EVal))
    TLSCall = Ref->getSymbol().getName() == "__tls_get_addr";

  if (TLSCall && getLexer().is(AsmToken::LParen)) {
    const MCExpr *TLSSym;

    Parser.Lex(); // Eat the '('.
    S = Parser.getTok().getLoc();
    if (ParseExpression(TLSSym))
      return Error(S, "invalid TLS call expression");
    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "missing ')'");
    E = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the ')'.

    Op = PPCOperand::CreateFromMCExpr(TLSSym, S, E, isPPC64());
    Operands.push_back(Op);
  }

  // Otherwise, check for D-form memory operands
  if (!TLSCall && getLexer().is(AsmToken::LParen)) {
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
  // If the next character is a '+' or '-', we need to add it to the
  // instruction name, to match what TableGen is doing.
  std::string NewOpcode;
  if (getLexer().is(AsmToken::Plus)) {
    getLexer().Lex();
    NewOpcode = Name;
    NewOpcode += '+';
    Name = NewOpcode;
  }
  if (getLexer().is(AsmToken::Minus)) {
    getLexer().Lex();
    NewOpcode = Name;
    NewOpcode += '-';
    Name = NewOpcode;
  }
  // If the instruction ends in a '.', we need to create a separate
  // token for it, to match what TableGen is doing.
  size_t Dot = Name.find('.');
  StringRef Mnemonic = Name.slice(0, Dot);
  if (!NewOpcode.empty()) // Underlying memory for Name is volatile.
    Operands.push_back(
        PPCOperand::CreateTokenWithStringCopy(Mnemonic, NameLoc, isPPC64()));
  else
    Operands.push_back(PPCOperand::CreateToken(Mnemonic, NameLoc, isPPC64()));
  if (Dot != StringRef::npos) {
    SMLoc DotLoc = SMLoc::getFromPointer(NameLoc.getPointer() + Dot);
    StringRef DotStr = Name.slice(Dot, StringRef::npos);
    if (!NewOpcode.empty()) // Underlying memory for Name is volatile.
      Operands.push_back(
          PPCOperand::CreateTokenWithStringCopy(DotStr, DotLoc, isPPC64()));
    else
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
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  if (IDVal == ".llong")
    return ParseDirectiveWord(8, DirectiveID.getLoc());
  if (IDVal == ".tc")
    return ParseDirectiveTC(isPPC64()? 8 : 4, DirectiveID.getLoc());
  if (IDVal == ".machine")
    return ParseDirectiveMachine(DirectiveID.getLoc());
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

/// ParseDirectiveMachine
///  ::= .machine [ cpu | "push" | "pop" ]
bool PPCAsmParser::ParseDirectiveMachine(SMLoc L) {
  if (getLexer().isNot(AsmToken::Identifier) &&
      getLexer().isNot(AsmToken::String))
    return Error(L, "unexpected token in directive");

  StringRef CPU = Parser.getTok().getIdentifier();
  Parser.Lex();

  // FIXME: Right now, the parser always allows any available
  // instruction, so the .machine directive is not useful.
  // Implement ".machine any" (by doing nothing) for the benefit
  // of existing assembler code.  Likewise, we can then implement
  // ".machine push" and ".machine pop" as no-op.
  if (CPU != "any" && CPU != "push" && CPU != "pop")
    return Error(L, "unrecognized machine type");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");

  return false;
}

/// Force static initialization.
extern "C" void LLVMInitializePowerPCAsmParser() {
  RegisterMCAsmParser<PPCAsmParser> A(ThePPC32Target);
  RegisterMCAsmParser<PPCAsmParser> B(ThePPC64Target);
  RegisterMCAsmParser<PPCAsmParser> C(ThePPC64LETarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "PPCGenAsmMatcher.inc"

// Define this matcher function after the auto-generated include so we
// have the match class enum definitions.
unsigned PPCAsmParser::validateTargetOperandClass(MCParsedAsmOperand *AsmOp,
                                                  unsigned Kind) {
  // If the kind is a token for a literal immediate, check if our asm
  // operand matches. This is for InstAliases which have a fixed-value
  // immediate in the syntax.
  int64_t ImmVal;
  switch (Kind) {
    case MCK_0: ImmVal = 0; break;
    case MCK_1: ImmVal = 1; break;
    case MCK_2: ImmVal = 2; break;
    case MCK_3: ImmVal = 3; break;
    default: return Match_InvalidOperand;
  }

  PPCOperand *Op = static_cast<PPCOperand*>(AsmOp);
  if (Op->isImm() && Op->getImm() == ImmVal)
    return Match_Success;

  return Match_InvalidOperand;
}

const MCExpr *
PPCAsmParser::applyModifierToExpr(const MCExpr *E,
                                  MCSymbolRefExpr::VariantKind Variant,
                                  MCContext &Ctx) {
  switch (Variant) {
  case MCSymbolRefExpr::VK_PPC_LO:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_LO, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HI:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HI, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HA:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HA, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHER:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HIGHER, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHERA:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HIGHERA, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHEST:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HIGHEST, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHESTA:
    return PPCMCExpr::Create(PPCMCExpr::VK_PPC_HIGHESTA, E, false, Ctx);
  default:
    return 0;
  }
}
