//===-- SparcAsmParser.cpp - Parse Sparc assembly to MCInst instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SparcMCTargetDesc.h"
#include "MCTargetDesc/SparcMCExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// The generated AsmMatcher SparcGenAsmMatcher uses "Sparc" as the target
// namespace. But SPARC backend uses "SP" as its namespace.
namespace llvm {
  namespace Sparc {
    using namespace SP;
  }
}

namespace {
class SparcOperand;
class SparcAsmParser : public MCTargetAsmParser {

  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "SparcGenAsmMatcher.inc"

  /// }

  // public interface of the MCTargetAsmParser.
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);
  bool ParseDirective(AsmToken DirectiveID);

  virtual unsigned validateTargetOperandClass(MCParsedAsmOperand *Op,
                                              unsigned Kind);

  // Custom parse functions for Sparc specific operands.
  OperandMatchResultTy
  parseMEMOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
               StringRef Name);

  OperandMatchResultTy
  parseSparcAsmOperand(SparcOperand *&Operand, bool isCall = false);

  // returns true if Tok is matched to a register and returns register in RegNo.
  bool matchRegisterName(const AsmToken &Tok, unsigned &RegNo,
                         unsigned &RegKind);

  bool matchSparcAsmModifiers(const MCExpr *&EVal, SMLoc &EndLoc);
  bool parseDirectiveWord(unsigned Size, SMLoc L);

  bool is64Bit() const { return STI.getTargetTriple().startswith("sparcv9"); }
public:
  SparcAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser,
                const MCInstrInfo &MII)
      : MCTargetAsmParser(), STI(sti), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

};

  static unsigned IntRegs[32] = {
    Sparc::G0, Sparc::G1, Sparc::G2, Sparc::G3,
    Sparc::G4, Sparc::G5, Sparc::G6, Sparc::G7,
    Sparc::O0, Sparc::O1, Sparc::O2, Sparc::O3,
    Sparc::O4, Sparc::O5, Sparc::O6, Sparc::O7,
    Sparc::L0, Sparc::L1, Sparc::L2, Sparc::L3,
    Sparc::L4, Sparc::L5, Sparc::L6, Sparc::L7,
    Sparc::I0, Sparc::I1, Sparc::I2, Sparc::I3,
    Sparc::I4, Sparc::I5, Sparc::I6, Sparc::I7 };

  static unsigned FloatRegs[32] = {
    Sparc::F0,  Sparc::F1,  Sparc::F2,  Sparc::F3,
    Sparc::F4,  Sparc::F5,  Sparc::F6,  Sparc::F7,
    Sparc::F8,  Sparc::F9,  Sparc::F10, Sparc::F11,
    Sparc::F12, Sparc::F13, Sparc::F14, Sparc::F15,
    Sparc::F16, Sparc::F17, Sparc::F18, Sparc::F19,
    Sparc::F20, Sparc::F21, Sparc::F22, Sparc::F23,
    Sparc::F24, Sparc::F25, Sparc::F26, Sparc::F27,
    Sparc::F28, Sparc::F29, Sparc::F30, Sparc::F31 };

  static unsigned DoubleRegs[32] = {
    Sparc::D0,  Sparc::D1,  Sparc::D2,  Sparc::D3,
    Sparc::D4,  Sparc::D5,  Sparc::D6,  Sparc::D7,
    Sparc::D8,  Sparc::D7,  Sparc::D8,  Sparc::D9,
    Sparc::D12, Sparc::D13, Sparc::D14, Sparc::D15,
    Sparc::D16, Sparc::D17, Sparc::D18, Sparc::D19,
    Sparc::D20, Sparc::D21, Sparc::D22, Sparc::D23,
    Sparc::D24, Sparc::D25, Sparc::D26, Sparc::D27,
    Sparc::D28, Sparc::D29, Sparc::D30, Sparc::D31 };

  static unsigned QuadFPRegs[32] = {
    Sparc::Q0,  Sparc::Q1,  Sparc::Q2,  Sparc::Q3,
    Sparc::Q4,  Sparc::Q5,  Sparc::Q6,  Sparc::Q7,
    Sparc::Q8,  Sparc::Q9,  Sparc::Q10, Sparc::Q11,
    Sparc::Q12, Sparc::Q13, Sparc::Q14, Sparc::Q15 };


/// SparcOperand - Instances of this class represent a parsed Sparc machine
/// instruction.
class SparcOperand : public MCParsedAsmOperand {
public:
  enum RegisterKind {
    rk_None,
    rk_IntReg,
    rk_FloatReg,
    rk_DoubleReg,
    rk_QuadReg,
    rk_CCReg,
    rk_Y
  };
private:
  enum KindTy {
    k_Token,
    k_Register,
    k_Immediate,
    k_MemoryReg,
    k_MemoryImm
  } Kind;

  SMLoc StartLoc, EndLoc;

  SparcOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
    RegisterKind Kind;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned Base;
    unsigned OffsetReg;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };
public:
  bool isToken() const { return Kind == k_Token; }
  bool isReg() const { return Kind == k_Register; }
  bool isImm() const { return Kind == k_Immediate; }
  bool isMem() const { return isMEMrr() || isMEMri(); }
  bool isMEMrr() const { return Kind == k_MemoryReg; }
  bool isMEMri() const { return Kind == k_MemoryImm; }

  bool isFloatReg() const {
    return (Kind == k_Register && Reg.Kind == rk_FloatReg);
  }

  bool isFloatOrDoubleReg() const {
    return (Kind == k_Register && (Reg.Kind == rk_FloatReg
                                   || Reg.Kind == rk_DoubleReg));
  }


  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_MemoryReg || Kind == k_MemoryImm) && "Invalid access!");
    return Mem.Base;
  }

  unsigned getMemOffsetReg() const {
    assert((Kind == k_MemoryReg) && "Invalid access!");
    return Mem.OffsetReg;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_MemoryImm) && "Invalid access!");
    return Mem.Off;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const {
    return StartLoc;
  }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const {
    return EndLoc;
  }

  virtual void print(raw_ostream &OS) const {
    switch (Kind) {
    case k_Token:     OS << "Token: " << getToken() << "\n"; break;
    case k_Register:  OS << "Reg: #" << getReg() << "\n"; break;
    case k_Immediate: OS << "Imm: " << getImm() << "\n"; break;
    case k_MemoryReg: OS << "Mem: " << getMemBase() << "+"
                         << getMemOffsetReg() << "\n"; break;
    case k_MemoryImm: assert(getMemOff() != 0);
      OS << "Mem: " << getMemBase()
         << "+" << *getMemOff()
         << "\n"; break;
    }
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const{
    // Add as immediate when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addMEMrrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()));

    assert(getMemOffsetReg() != 0 && "Invalid offset");
    Inst.addOperand(MCOperand::CreateReg(getMemOffsetReg()));
  }

  void addMEMriOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  static SparcOperand *CreateToken(StringRef Str, SMLoc S) {
    SparcOperand *Op = new SparcOperand(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static SparcOperand *CreateReg(unsigned RegNum,
                                 unsigned Kind,
                                 SMLoc S, SMLoc E) {
    SparcOperand *Op = new SparcOperand(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->Reg.Kind   = (SparcOperand::RegisterKind)Kind;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static SparcOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    SparcOperand *Op = new SparcOperand(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static SparcOperand *MorphToDoubleReg(SparcOperand *Op) {
    unsigned Reg = Op->getReg();
    assert(Op->Reg.Kind == rk_FloatReg);
    unsigned regIdx = Reg - Sparc::F0;
    if (regIdx % 2 || regIdx > 31)
      return 0;
    Op->Reg.RegNum = DoubleRegs[regIdx / 2];
    Op->Reg.Kind = rk_DoubleReg;
    return Op;
  }

  static SparcOperand *MorphToQuadReg(SparcOperand *Op) {
    unsigned Reg = Op->getReg();
    unsigned regIdx = 0;
    switch (Op->Reg.Kind) {
    default: assert(0 && "Unexpected register kind!");
    case rk_FloatReg:
      regIdx = Reg - Sparc::F0;
      if (regIdx % 4 || regIdx > 31)
        return 0;
      Reg = QuadFPRegs[regIdx / 4];
      break;
    case rk_DoubleReg:
      regIdx =  Reg - Sparc::D0;
      if (regIdx % 2 || regIdx > 31)
        return 0;
      Reg = QuadFPRegs[regIdx / 2];
      break;
    }
    Op->Reg.RegNum  = Reg;
    Op->Reg.Kind = rk_QuadReg;
    return Op;
  }

  static SparcOperand *MorphToMEMrr(unsigned Base, SparcOperand *Op) {
    unsigned offsetReg = Op->getReg();
    Op->Kind = k_MemoryReg;
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = offsetReg;
    Op->Mem.Off = 0;
    return Op;
  }

  static SparcOperand *CreateMEMri(unsigned Base,
                                 const MCExpr *Off,
                                 SMLoc S, SMLoc E) {
    SparcOperand *Op = new SparcOperand(k_MemoryImm);
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = 0;
    Op->Mem.Off = Off;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static SparcOperand *MorphToMEMri(unsigned Base, SparcOperand *Op) {
    const MCExpr *Imm  = Op->getImm();
    Op->Kind = k_MemoryImm;
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = 0;
    Op->Mem.Off = Imm;
    return Op;
  }
};

} // end namespace

bool SparcAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  MCInst Inst;
  SmallVector<MCInst, 8> Instructions;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                              MatchingInlineAsm);
  switch (MatchResult) {
  default:
    break;

  case Match_Success: {
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst, STI);
    return false;
  }

  case Match_MissingFeature:
    return Error(IDLoc,
                 "instruction requires a CPU feature not currently enabled");

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((SparcOperand*) Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }
  return true;
}

bool SparcAsmParser::
ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc)
{
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  if (getLexer().getKind() != AsmToken::Percent)
    return false;
  Parser.Lex();
  unsigned regKind = SparcOperand::rk_None;
  if (matchRegisterName(Tok, RegNo, regKind)) {
    Parser.Lex();
    return false;
  }

  return Error(StartLoc, "invalid register name");
}

bool SparcAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                 SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands)
{
  // Check if we have valid mnemonic.
  if (!mnemonicIsValid(Name, 0)) {
    Parser.eatToEndOfStatement();
    return Error(NameLoc, "Unknown instruction");
  }
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(SparcOperand::CreateToken(Name, NameLoc));

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Name) != MatchOperand_Success) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token");
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      // Parse and remember the operand.
      if (parseOperand(Operands, Name) != MatchOperand_Success) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token");
      }
    }
  }
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token");
  }
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool SparcAsmParser::
ParseDirective(AsmToken DirectiveID)
{
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".byte")
    return parseDirectiveWord(1, DirectiveID.getLoc());

  if (IDVal == ".half")
    return parseDirectiveWord(2, DirectiveID.getLoc());

  if (IDVal == ".word")
    return parseDirectiveWord(4, DirectiveID.getLoc());

  if (IDVal == ".nword")
    return parseDirectiveWord(is64Bit() ? 8 : 4, DirectiveID.getLoc());

  if (is64Bit() && IDVal == ".xword")
    return parseDirectiveWord(8, DirectiveID.getLoc());

  if (IDVal == ".register") {
    // For now, ignore .register directive.
    Parser.eatToEndOfStatement();
    return false;
  }

  // Let the MC layer to handle other directives.
  return true;
}

bool SparcAsmParser:: parseDirectiveWord(unsigned Size, SMLoc L) {
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

SparcAsmParser::OperandMatchResultTy SparcAsmParser::
parseMEMOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands)
{

  SMLoc S, E;
  unsigned BaseReg = 0;

  if (ParseRegister(BaseReg, S, E)) {
    return MatchOperand_NoMatch;
  }

  switch (getLexer().getKind()) {
  default: return MatchOperand_NoMatch;

  case AsmToken::Comma:
  case AsmToken::RBrac:
  case AsmToken::EndOfStatement:
    Operands.push_back(SparcOperand::CreateMEMri(BaseReg, 0, S, E));
    return MatchOperand_Success;

  case AsmToken:: Plus:
    Parser.Lex(); // Eat the '+'
    break;
  case AsmToken::Minus:
    break;
  }

  SparcOperand *Offset = 0;
  OperandMatchResultTy ResTy = parseSparcAsmOperand(Offset);
  if (ResTy != MatchOperand_Success || !Offset)
    return MatchOperand_NoMatch;

  Offset = (Offset->isImm()
            ? SparcOperand::MorphToMEMri(BaseReg, Offset)
            : SparcOperand::MorphToMEMrr(BaseReg, Offset));

  Operands.push_back(Offset);
  return MatchOperand_Success;
}

SparcAsmParser::OperandMatchResultTy SparcAsmParser::
parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
             StringRef Mnemonic)
{

  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail)
    return ResTy;

  if (getLexer().is(AsmToken::LBrac)) {
    // Memory operand
    Operands.push_back(SparcOperand::CreateToken("[",
                                                 Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the [

    if (Mnemonic == "cas" || Mnemonic == "casx") {
      SMLoc S = Parser.getTok().getLoc();
      if (getLexer().getKind() != AsmToken::Percent)
        return MatchOperand_NoMatch;
      Parser.Lex(); // eat %

      unsigned RegNo, RegKind;
      if (!matchRegisterName(Parser.getTok(), RegNo, RegKind))
        return MatchOperand_NoMatch;

      Parser.Lex(); // Eat the identifier token.
      SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer()-1);
      Operands.push_back(SparcOperand::CreateReg(RegNo, RegKind, S, E));
      ResTy = MatchOperand_Success;
    } else {
      ResTy = parseMEMOperand(Operands);
    }

    if (ResTy != MatchOperand_Success)
      return ResTy;

    if (!getLexer().is(AsmToken::RBrac))
      return MatchOperand_ParseFail;

    Operands.push_back(SparcOperand::CreateToken("]",
                                                 Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the ]
    return MatchOperand_Success;
  }

  SparcOperand *Op = 0;
  ResTy = parseSparcAsmOperand(Op, (Mnemonic == "call"));
  if (ResTy != MatchOperand_Success || !Op)
    return MatchOperand_ParseFail;

  // Push the parsed operand into the list of operands
  Operands.push_back(Op);

  return MatchOperand_Success;
}

SparcAsmParser::OperandMatchResultTy
SparcAsmParser::parseSparcAsmOperand(SparcOperand *&Op, bool isCall)
{

  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  const MCExpr *EVal;

  Op = 0;
  switch (getLexer().getKind()) {
  default:  break;

  case AsmToken::Percent:
    Parser.Lex(); // Eat the '%'.
    unsigned RegNo;
    unsigned RegKind;
    if (matchRegisterName(Parser.getTok(), RegNo, RegKind)) {
      StringRef name = Parser.getTok().getString();
      Parser.Lex(); // Eat the identifier token.
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      switch (RegNo) {
      default:
        Op = SparcOperand::CreateReg(RegNo, RegKind, S, E);
        break;
      case Sparc::Y:
        Op = SparcOperand::CreateToken("%y", S);
        break;

      case Sparc::ICC:
        if (name == "xcc")
          Op = SparcOperand::CreateToken("%xcc", S);
        else
          Op = SparcOperand::CreateToken("%icc", S);
        break;

      case Sparc::FCC:
        assert(name == "fcc0" && "Cannot handle %fcc other than %fcc0 yet");
        Op = SparcOperand::CreateToken("%fcc0", S);
        break;
      }
      break;
    }
    if (matchSparcAsmModifiers(EVal, E)) {
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      Op = SparcOperand::CreateImm(EVal, S, E);
    }
    break;

  case AsmToken::Minus:
  case AsmToken::Integer:
    if (!getParser().parseExpression(EVal, E))
      Op = SparcOperand::CreateImm(EVal, S, E);
    break;

  case AsmToken::Identifier: {
    StringRef Identifier;
    if (!getParser().parseIdentifier(Identifier)) {
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      MCSymbol *Sym = getContext().GetOrCreateSymbol(Identifier);

      const MCExpr *Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None,
                                                  getContext());
      if (isCall &&
          getContext().getObjectFileInfo()->getRelocM() == Reloc::PIC_)
        Res = SparcMCExpr::Create(SparcMCExpr::VK_Sparc_WPLT30, Res,
                                  getContext());
      Op = SparcOperand::CreateImm(Res, S, E);
    }
    break;
  }
  }
  return (Op) ? MatchOperand_Success : MatchOperand_ParseFail;
}

bool SparcAsmParser::matchRegisterName(const AsmToken &Tok,
                                       unsigned &RegNo,
                                       unsigned &RegKind)
{
  int64_t intVal = 0;
  RegNo = 0;
  RegKind = SparcOperand::rk_None;
  if (Tok.is(AsmToken::Identifier)) {
    StringRef name = Tok.getString();

    // %fp
    if (name.equals("fp")) {
      RegNo = Sparc::I6;
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
    // %sp
    if (name.equals("sp")) {
      RegNo = Sparc::O6;
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }

    if (name.equals("y")) {
      RegNo = Sparc::Y;
      RegKind = SparcOperand::rk_Y;
      return true;
    }

    if (name.equals("icc")) {
      RegNo = Sparc::ICC;
      RegKind = SparcOperand::rk_CCReg;
      return true;
    }

    if (name.equals("xcc")) {
      // FIXME:: check 64bit.
      RegNo = Sparc::ICC;
      RegKind = SparcOperand::rk_CCReg;
      return true;
    }

    // %fcc0 - %fcc3
    if (name.substr(0, 3).equals_lower("fcc")
        && !name.substr(3).getAsInteger(10, intVal)
        && intVal < 4) {
      // FIXME: check 64bit and  handle %fcc1 - %fcc3
      RegNo = Sparc::FCC;
      RegKind = SparcOperand::rk_CCReg;
      return true;
    }

    // %g0 - %g7
    if (name.substr(0, 1).equals_lower("g")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[intVal];
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
    // %o0 - %o7
    if (name.substr(0, 1).equals_lower("o")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[8 + intVal];
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
    if (name.substr(0, 1).equals_lower("l")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[16 + intVal];
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
    if (name.substr(0, 1).equals_lower("i")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[24 + intVal];
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
    // %f0 - %f31
    if (name.substr(0, 1).equals_lower("f")
        && !name.substr(1, 2).getAsInteger(10, intVal) && intVal < 32) {
      RegNo = FloatRegs[intVal];
      RegKind = SparcOperand::rk_FloatReg;
      return true;
    }
    // %f32 - %f62
    if (name.substr(0, 1).equals_lower("f")
        && !name.substr(1, 2).getAsInteger(10, intVal)
        && intVal >= 32 && intVal <= 62 && (intVal % 2 == 0)) {
      // FIXME: Check V9
      RegNo = DoubleRegs[intVal/2];
      RegKind = SparcOperand::rk_DoubleReg;
      return true;
    }

    // %r0 - %r31
    if (name.substr(0, 1).equals_lower("r")
        && !name.substr(1, 2).getAsInteger(10, intVal) && intVal < 31) {
      RegNo = IntRegs[intVal];
      RegKind = SparcOperand::rk_IntReg;
      return true;
    }
  }
  return false;
}

static bool hasGOTReference(const MCExpr *Expr) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    if (const SparcMCExpr *SE = dyn_cast<SparcMCExpr>(Expr))
      return hasGOTReference(SE->getSubExpr());
    break;

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Expr);
    return hasGOTReference(BE->getLHS()) || hasGOTReference(BE->getRHS());
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(Expr);
    return (SymRef.getSymbol().getName() == "_GLOBAL_OFFSET_TABLE_");
  }

  case MCExpr::Unary:
    return hasGOTReference(cast<MCUnaryExpr>(Expr)->getSubExpr());
  }
  return false;
}

bool SparcAsmParser::matchSparcAsmModifiers(const MCExpr *&EVal,
                                            SMLoc &EndLoc)
{
  AsmToken Tok = Parser.getTok();
  if (!Tok.is(AsmToken::Identifier))
    return false;

  StringRef name = Tok.getString();

  SparcMCExpr::VariantKind VK = SparcMCExpr::parseVariantKind(name);

  if (VK == SparcMCExpr::VK_Sparc_None)
    return false;

  Parser.Lex(); // Eat the identifier.
  if (Parser.getTok().getKind() != AsmToken::LParen)
    return false;

  Parser.Lex(); // Eat the LParen token.
  const MCExpr *subExpr;
  if (Parser.parseParenExpression(subExpr, EndLoc))
    return false;

  bool isPIC = getContext().getObjectFileInfo()->getRelocM() == Reloc::PIC_;

  switch(VK) {
  default: break;
  case SparcMCExpr::VK_Sparc_LO:
    VK =  (hasGOTReference(subExpr)
           ? SparcMCExpr::VK_Sparc_PC10
           : (isPIC ? SparcMCExpr::VK_Sparc_GOT10 : VK));
    break;
  case SparcMCExpr::VK_Sparc_HI:
    VK =  (hasGOTReference(subExpr)
           ? SparcMCExpr::VK_Sparc_PC22
           : (isPIC ? SparcMCExpr::VK_Sparc_GOT22 : VK));
    break;
  }

  EVal = SparcMCExpr::Create(VK, subExpr, getContext());
  return true;
}


extern "C" void LLVMInitializeSparcAsmParser() {
  RegisterMCAsmParser<SparcAsmParser> A(TheSparcTarget);
  RegisterMCAsmParser<SparcAsmParser> B(TheSparcV9Target);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "SparcGenAsmMatcher.inc"



unsigned SparcAsmParser::
validateTargetOperandClass(MCParsedAsmOperand *GOp,
                           unsigned Kind)
{
  SparcOperand *Op = (SparcOperand*)GOp;
  if (Op->isFloatOrDoubleReg()) {
    switch (Kind) {
    default: break;
    case MCK_DFPRegs:
      if (!Op->isFloatReg() || SparcOperand::MorphToDoubleReg(Op))
        return MCTargetAsmParser::Match_Success;
      break;
    case MCK_QFPRegs:
      if (SparcOperand::MorphToQuadReg(Op))
        return MCTargetAsmParser::Match_Success;
      break;
    }
  }
  return Match_InvalidOperand;
}
