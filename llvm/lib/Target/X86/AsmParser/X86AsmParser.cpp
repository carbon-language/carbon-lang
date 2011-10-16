//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct X86Operand;

class X86ATTAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  bool Error(SMLoc L, const Twine &Msg,
             ArrayRef<SMRange> Ranges = ArrayRef<SMRange>()) {
    return Parser.Error(L, Msg, Ranges);
  }

  X86Operand *ParseOperand();
  X86Operand *ParseMemOperand(unsigned SegReg, SMLoc StartLoc);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveCode(StringRef IDVal, SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);

  /// isSrcOp - Returns true if operand is either (%rsi) or %ds:%(rsi)
  /// in 64bit mode or (%edi) or %es:(%edi) in 32bit mode.
  bool isSrcOp(X86Operand &Op);

  /// isDstOp - Returns true if operand is either %es:(%rdi) in 64bit mode
  /// or %es:(%edi) in 32bit mode.
  bool isDstOp(X86Operand &Op);

  bool is64BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return (STI.getFeatureBits() & X86::Mode64Bit) != 0;
  }
  void SwitchMode() {
    unsigned FB = ComputeAvailableFeatures(STI.ToggleFeature(X86::Mode64Bit));
    setAvailableFeatures(FB);
  }

  /// @name Auto-generated Matcher Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "X86GenAsmMatcher.inc"

  /// }

public:
  X86ATTAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser(), STI(sti), Parser(parser) {

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  virtual bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
} // end anonymous namespace

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

namespace {

/// X86Operand - Instances of this class represent a parsed X86 machine
/// instruction.
struct X86Operand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Register,
    Immediate,
    Memory
  } Kind;

  SMLoc StartLoc, EndLoc;

  union {
    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNo;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;

    struct {
      unsigned SegReg;
      const MCExpr *Disp;
      unsigned BaseReg;
      unsigned IndexReg;
      unsigned Scale;
    } Mem;
  };

  X86Operand(KindTy K, SMLoc Start, SMLoc End)
    : Kind(K), StartLoc(Start), EndLoc(End) {}

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }
  
  SMRange getLocRange() const { return SMRange(StartLoc, EndLoc); }

  virtual void print(raw_ostream &OS) const {}

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }
  void setTokenValue(StringRef Value) {
    assert(Kind == Token && "Invalid access!");
    Tok.Data = Value.data();
    Tok.Length = Value.size();
  }

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNo;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  const MCExpr *getMemDisp() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Disp;
  }
  unsigned getMemSegReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.SegReg;
  }
  unsigned getMemBaseReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.BaseReg;
  }
  unsigned getMemIndexReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.IndexReg;
  }
  unsigned getMemScale() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Scale;
  }

  bool isToken() const {return Kind == Token; }

  bool isImm() const { return Kind == Immediate; }

  bool isImmSExti16i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    uint64_t Value = CE->getValue();
    return ((                                  Value <= 0x000000000000007FULL)||
            (0x000000000000FF80ULL <= Value && Value <= 0x000000000000FFFFULL)||
            (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
  }
  bool isImmSExti32i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    uint64_t Value = CE->getValue();
    return ((                                  Value <= 0x000000000000007FULL)||
            (0x00000000FFFFFF80ULL <= Value && Value <= 0x00000000FFFFFFFFULL)||
            (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
  }
  bool isImmZExtu32u8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    uint64_t Value = CE->getValue();
    return (Value <= 0x00000000000000FFULL);
  }
  bool isImmSExti64i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    uint64_t Value = CE->getValue();
    return ((                                  Value <= 0x000000000000007FULL)||
            (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
  }
  bool isImmSExti64i32() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    uint64_t Value = CE->getValue();
    return ((                                  Value <= 0x000000007FFFFFFFULL)||
            (0xFFFFFFFF80000000ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
  }

  bool isMem() const { return Kind == Memory; }

  bool isAbsMem() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1;
  }

  bool isReg() const { return Kind == Register; }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert((N == 5) && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseReg()));
    Inst.addOperand(MCOperand::CreateImm(getMemScale()));
    Inst.addOperand(MCOperand::CreateReg(getMemIndexReg()));
    addExpr(Inst, getMemDisp());
    Inst.addOperand(MCOperand::CreateReg(getMemSegReg()));
  }

  void addAbsMemOperands(MCInst &Inst, unsigned N) const {
    assert((N == 1) && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateExpr(getMemDisp()));
  }

  static X86Operand *CreateToken(StringRef Str, SMLoc Loc) {
    SMLoc EndLoc = SMLoc::getFromPointer(Loc.getPointer() + Str.size() - 1);
    X86Operand *Res = new X86Operand(Token, Loc, EndLoc);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    return Res;
  }

  static X86Operand *CreateReg(unsigned RegNo, SMLoc StartLoc, SMLoc EndLoc) {
    X86Operand *Res = new X86Operand(Register, StartLoc, EndLoc);
    Res->Reg.RegNo = RegNo;
    return Res;
  }

  static X86Operand *CreateImm(const MCExpr *Val, SMLoc StartLoc, SMLoc EndLoc){
    X86Operand *Res = new X86Operand(Immediate, StartLoc, EndLoc);
    Res->Imm.Val = Val;
    return Res;
  }

  /// Create an absolute memory operand.
  static X86Operand *CreateMem(const MCExpr *Disp, SMLoc StartLoc,
                               SMLoc EndLoc) {
    X86Operand *Res = new X86Operand(Memory, StartLoc, EndLoc);
    Res->Mem.SegReg   = 0;
    Res->Mem.Disp     = Disp;
    Res->Mem.BaseReg  = 0;
    Res->Mem.IndexReg = 0;
    Res->Mem.Scale    = 1;
    return Res;
  }

  /// Create a generalized memory operand.
  static X86Operand *CreateMem(unsigned SegReg, const MCExpr *Disp,
                               unsigned BaseReg, unsigned IndexReg,
                               unsigned Scale, SMLoc StartLoc, SMLoc EndLoc) {
    // We should never just have a displacement, that should be parsed as an
    // absolute memory operand.
    assert((SegReg || BaseReg || IndexReg) && "Invalid memory operand!");

    // The scale should always be one of {1,2,4,8}.
    assert(((Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8)) &&
           "Invalid scale!");
    X86Operand *Res = new X86Operand(Memory, StartLoc, EndLoc);
    Res->Mem.SegReg   = SegReg;
    Res->Mem.Disp     = Disp;
    Res->Mem.BaseReg  = BaseReg;
    Res->Mem.IndexReg = IndexReg;
    Res->Mem.Scale    = Scale;
    return Res;
  }
};

} // end anonymous namespace.

bool X86ATTAsmParser::isSrcOp(X86Operand &Op) {
  unsigned basereg = is64BitMode() ? X86::RSI : X86::ESI;

  return (Op.isMem() &&
    (Op.Mem.SegReg == 0 || Op.Mem.SegReg == X86::DS) &&
    isa<MCConstantExpr>(Op.Mem.Disp) &&
    cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
    Op.Mem.BaseReg == basereg && Op.Mem.IndexReg == 0);
}

bool X86ATTAsmParser::isDstOp(X86Operand &Op) {
  unsigned basereg = is64BitMode() ? X86::RDI : X86::EDI;

  return Op.isMem() && Op.Mem.SegReg == X86::ES &&
    isa<MCConstantExpr>(Op.Mem.Disp) &&
    cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
    Op.Mem.BaseReg == basereg && Op.Mem.IndexReg == 0;
}

bool X86ATTAsmParser::ParseRegister(unsigned &RegNo,
                                    SMLoc &StartLoc, SMLoc &EndLoc) {
  RegNo = 0;
  const AsmToken &TokPercent = Parser.getTok();
  assert(TokPercent.is(AsmToken::Percent) && "Invalid token kind!");
  StartLoc = TokPercent.getLoc();
  Parser.Lex(); // Eat percent token.

  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return Error(Tok.getLoc(), "invalid register name");

  RegNo = MatchRegisterName(Tok.getString());

  // If the match failed, try the register name as lowercase.
  if (RegNo == 0)
    RegNo = MatchRegisterName(LowercaseString(Tok.getString()));

  if (!is64BitMode()) {
    // FIXME: This should be done using Requires<In32BitMode> and
    // Requires<In64BitMode> so "eiz" usage in 64-bit instructions can be also
    // checked.
    // FIXME: Check AH, CH, DH, BH cannot be used in an instruction requiring a
    // REX prefix.
    if (RegNo == X86::RIZ ||
        X86MCRegisterClasses[X86::GR64RegClassID].contains(RegNo) ||
        X86II::isX86_64NonExtLowByteReg(RegNo) ||
        X86II::isX86_64ExtendedReg(RegNo))
      return Error(Tok.getLoc(), "register %"
                   + Tok.getString() + " is only available in 64-bit mode");
  }

  // Parse "%st" as "%st(0)" and "%st(1)", which is multiple tokens.
  if (RegNo == 0 && (Tok.getString() == "st" || Tok.getString() == "ST")) {
    RegNo = X86::ST0;
    EndLoc = Tok.getLoc();
    Parser.Lex(); // Eat 'st'

    // Check to see if we have '(4)' after %st.
    if (getLexer().isNot(AsmToken::LParen))
      return false;
    // Lex the paren.
    getParser().Lex();

    const AsmToken &IntTok = Parser.getTok();
    if (IntTok.isNot(AsmToken::Integer))
      return Error(IntTok.getLoc(), "expected stack index");
    switch (IntTok.getIntVal()) {
    case 0: RegNo = X86::ST0; break;
    case 1: RegNo = X86::ST1; break;
    case 2: RegNo = X86::ST2; break;
    case 3: RegNo = X86::ST3; break;
    case 4: RegNo = X86::ST4; break;
    case 5: RegNo = X86::ST5; break;
    case 6: RegNo = X86::ST6; break;
    case 7: RegNo = X86::ST7; break;
    default: return Error(IntTok.getLoc(), "invalid stack index");
    }

    if (getParser().Lex().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "expected ')'");

    EndLoc = Tok.getLoc();
    Parser.Lex(); // Eat ')'
    return false;
  }

  // If this is "db[0-7]", match it as an alias
  // for dr[0-7].
  if (RegNo == 0 && Tok.getString().size() == 3 &&
      Tok.getString().startswith("db")) {
    switch (Tok.getString()[2]) {
    case '0': RegNo = X86::DR0; break;
    case '1': RegNo = X86::DR1; break;
    case '2': RegNo = X86::DR2; break;
    case '3': RegNo = X86::DR3; break;
    case '4': RegNo = X86::DR4; break;
    case '5': RegNo = X86::DR5; break;
    case '6': RegNo = X86::DR6; break;
    case '7': RegNo = X86::DR7; break;
    }

    if (RegNo != 0) {
      EndLoc = Tok.getLoc();
      Parser.Lex(); // Eat it.
      return false;
    }
  }

  if (RegNo == 0)
    return Error(Tok.getLoc(), "invalid register name");

  EndLoc = Tok.getLoc();
  Parser.Lex(); // Eat identifier token.
  return false;
}

X86Operand *X86ATTAsmParser::ParseOperand() {
  switch (getLexer().getKind()) {
  default:
    // Parse a memory operand with no segment register.
    return ParseMemOperand(0, Parser.getTok().getLoc());
  case AsmToken::Percent: {
    // Read the register.
    unsigned RegNo;
    SMLoc Start, End;
    if (ParseRegister(RegNo, Start, End)) return 0;
    if (RegNo == X86::EIZ || RegNo == X86::RIZ) {
      Error(Start, "%eiz and %riz can only be used as index registers");
      return 0;
    }

    // If this is a segment register followed by a ':', then this is the start
    // of a memory reference, otherwise this is a normal register reference.
    if (getLexer().isNot(AsmToken::Colon))
      return X86Operand::CreateReg(RegNo, Start, End);


    getParser().Lex(); // Eat the colon.
    return ParseMemOperand(RegNo, Start);
  }
  case AsmToken::Dollar: {
    // $42 -> immediate.
    SMLoc Start = Parser.getTok().getLoc(), End;
    Parser.Lex();
    const MCExpr *Val;
    if (getParser().ParseExpression(Val, End))
      return 0;
    return X86Operand::CreateImm(Val, Start, End);
  }
  }
}

/// ParseMemOperand: segment: disp(basereg, indexreg, scale).  The '%ds:' prefix
/// has already been parsed if present.
X86Operand *X86ATTAsmParser::ParseMemOperand(unsigned SegReg, SMLoc MemStart) {

  // We have to disambiguate a parenthesized expression "(4+5)" from the start
  // of a memory operand with a missing displacement "(%ebx)" or "(,%eax)".  The
  // only way to do this without lookahead is to eat the '(' and see what is
  // after it.
  const MCExpr *Disp = MCConstantExpr::Create(0, getParser().getContext());
  if (getLexer().isNot(AsmToken::LParen)) {
    SMLoc ExprEnd;
    if (getParser().ParseExpression(Disp, ExprEnd)) return 0;

    // After parsing the base expression we could either have a parenthesized
    // memory address or not.  If not, return now.  If so, eat the (.
    if (getLexer().isNot(AsmToken::LParen)) {
      // Unless we have a segment register, treat this as an immediate.
      if (SegReg == 0)
        return X86Operand::CreateMem(Disp, MemStart, ExprEnd);
      return X86Operand::CreateMem(SegReg, Disp, 0, 0, 1, MemStart, ExprEnd);
    }

    // Eat the '('.
    Parser.Lex();
  } else {
    // Okay, we have a '('.  We don't know if this is an expression or not, but
    // so we have to eat the ( to see beyond it.
    SMLoc LParenLoc = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the '('.

    if (getLexer().is(AsmToken::Percent) || getLexer().is(AsmToken::Comma)) {
      // Nothing to do here, fall into the code below with the '(' part of the
      // memory operand consumed.
    } else {
      SMLoc ExprEnd;

      // It must be an parenthesized expression, parse it now.
      if (getParser().ParseParenExpression(Disp, ExprEnd))
        return 0;

      // After parsing the base expression we could either have a parenthesized
      // memory address or not.  If not, return now.  If so, eat the (.
      if (getLexer().isNot(AsmToken::LParen)) {
        // Unless we have a segment register, treat this as an immediate.
        if (SegReg == 0)
          return X86Operand::CreateMem(Disp, LParenLoc, ExprEnd);
        return X86Operand::CreateMem(SegReg, Disp, 0, 0, 1, MemStart, ExprEnd);
      }

      // Eat the '('.
      Parser.Lex();
    }
  }

  // If we reached here, then we just ate the ( of the memory operand.  Process
  // the rest of the memory operand.
  unsigned BaseReg = 0, IndexReg = 0, Scale = 1;

  if (getLexer().is(AsmToken::Percent)) {
    SMLoc L;
    if (ParseRegister(BaseReg, L, L)) return 0;
    if (BaseReg == X86::EIZ || BaseReg == X86::RIZ) {
      Error(L, "eiz and riz can only be used as index registers");
      return 0;
    }
  }

  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the comma.

    // Following the comma we should have either an index register, or a scale
    // value. We don't support the later form, but we want to parse it
    // correctly.
    //
    // Not that even though it would be completely consistent to support syntax
    // like "1(%eax,,1)", the assembler doesn't. Use "eiz" or "riz" for this.
    if (getLexer().is(AsmToken::Percent)) {
      SMLoc L;
      if (ParseRegister(IndexReg, L, L)) return 0;

      if (getLexer().isNot(AsmToken::RParen)) {
        // Parse the scale amount:
        //  ::= ',' [scale-expression]
        if (getLexer().isNot(AsmToken::Comma)) {
          Error(Parser.getTok().getLoc(),
                "expected comma in scale expression");
          return 0;
        }
        Parser.Lex(); // Eat the comma.

        if (getLexer().isNot(AsmToken::RParen)) {
          SMLoc Loc = Parser.getTok().getLoc();

          int64_t ScaleVal;
          if (getParser().ParseAbsoluteExpression(ScaleVal))
            return 0;

          // Validate the scale amount.
          if (ScaleVal != 1 && ScaleVal != 2 && ScaleVal != 4 && ScaleVal != 8){
            Error(Loc, "scale factor in address must be 1, 2, 4 or 8");
            return 0;
          }
          Scale = (unsigned)ScaleVal;
        }
      }
    } else if (getLexer().isNot(AsmToken::RParen)) {
      // A scale amount without an index is ignored.
      // index.
      SMLoc Loc = Parser.getTok().getLoc();

      int64_t Value;
      if (getParser().ParseAbsoluteExpression(Value))
        return 0;

      if (Value != 1)
        Warning(Loc, "scale factor without index register is ignored");
      Scale = 1;
    }
  }

  // Ok, we've eaten the memory operand, verify we have a ')' and eat it too.
  if (getLexer().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "unexpected token in memory operand");
    return 0;
  }
  SMLoc MemEnd = Parser.getTok().getLoc();
  Parser.Lex(); // Eat the ')'.

  return X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale,
                               MemStart, MemEnd);
}

bool X86ATTAsmParser::
ParseInstruction(StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  StringRef PatchedName = Name;

  // FIXME: Hack to recognize setneb as setne.
  if (PatchedName.startswith("set") && PatchedName.endswith("b") &&
      PatchedName != "setb" && PatchedName != "setnb")
    PatchedName = PatchedName.substr(0, Name.size()-1);
  
  // FIXME: Hack to recognize cmp<comparison code>{ss,sd,ps,pd}.
  const MCExpr *ExtraImmOp = 0;
  if ((PatchedName.startswith("cmp") || PatchedName.startswith("vcmp")) &&
      (PatchedName.endswith("ss") || PatchedName.endswith("sd") ||
       PatchedName.endswith("ps") || PatchedName.endswith("pd"))) {
    bool IsVCMP = PatchedName.startswith("vcmp");
    unsigned SSECCIdx = IsVCMP ? 4 : 3;
    unsigned SSEComparisonCode = StringSwitch<unsigned>(
      PatchedName.slice(SSECCIdx, PatchedName.size() - 2))
      .Case("eq",          0)
      .Case("lt",          1)
      .Case("le",          2)
      .Case("unord",       3)
      .Case("neq",         4)
      .Case("nlt",         5)
      .Case("nle",         6)
      .Case("ord",         7)
      .Case("eq_uq",       8)
      .Case("nge",         9)
      .Case("ngt",      0x0A)
      .Case("false",    0x0B)
      .Case("neq_oq",   0x0C)
      .Case("ge",       0x0D)
      .Case("gt",       0x0E)
      .Case("true",     0x0F)
      .Case("eq_os",    0x10)
      .Case("lt_oq",    0x11)
      .Case("le_oq",    0x12)
      .Case("unord_s",  0x13)
      .Case("neq_us",   0x14)
      .Case("nlt_uq",   0x15)
      .Case("nle_uq",   0x16)
      .Case("ord_s",    0x17)
      .Case("eq_us",    0x18)
      .Case("nge_uq",   0x19)
      .Case("ngt_uq",   0x1A)
      .Case("false_os", 0x1B)
      .Case("neq_os",   0x1C)
      .Case("ge_oq",    0x1D)
      .Case("gt_oq",    0x1E)
      .Case("true_us",  0x1F)
      .Default(~0U);
    if (SSEComparisonCode != ~0U) {
      ExtraImmOp = MCConstantExpr::Create(SSEComparisonCode,
                                          getParser().getContext());
      if (PatchedName.endswith("ss")) {
        PatchedName = IsVCMP ? "vcmpss" : "cmpss";
      } else if (PatchedName.endswith("sd")) {
        PatchedName = IsVCMP ? "vcmpsd" : "cmpsd";
      } else if (PatchedName.endswith("ps")) {
        PatchedName = IsVCMP ? "vcmpps" : "cmpps";
      } else {
        assert(PatchedName.endswith("pd") && "Unexpected mnemonic!");
        PatchedName = IsVCMP ? "vcmppd" : "cmppd";
      }
    }
  }

  Operands.push_back(X86Operand::CreateToken(PatchedName, NameLoc));

  if (ExtraImmOp)
    Operands.push_back(X86Operand::CreateImm(ExtraImmOp, NameLoc, NameLoc));


  // Determine whether this is an instruction prefix.
  bool isPrefix =
    Name == "lock" || Name == "rep" ||
    Name == "repe" || Name == "repz" ||
    Name == "repne" || Name == "repnz" ||
    Name == "rex64" || Name == "data16";


  // This does the actual operand parsing.  Don't parse any more if we have a
  // prefix juxtaposed with an operation like "lock incl 4(%rax)", because we
  // just want to parse the "lock" as the first instruction and the "incl" as
  // the next one.
  if (getLexer().isNot(AsmToken::EndOfStatement) && !isPrefix) {

    // Parse '*' modifier.
    if (getLexer().is(AsmToken::Star)) {
      SMLoc Loc = Parser.getTok().getLoc();
      Operands.push_back(X86Operand::CreateToken("*", Loc));
      Parser.Lex(); // Eat the star.
    }

    // Read the first operand.
    if (X86Operand *Op = ParseOperand())
      Operands.push_back(Op);
    else {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (X86Operand *Op = ParseOperand())
        Operands.push_back(Op);
      else {
        Parser.EatToEndOfStatement();
        return true;
      }
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.EatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  if (getLexer().is(AsmToken::EndOfStatement))
    Parser.Lex(); // Consume the EndOfStatement
  else if (isPrefix && getLexer().is(AsmToken::Slash))
    Parser.Lex(); // Consume the prefix separator Slash

  // This is a terrible hack to handle "out[bwl]? %al, (%dx)" ->
  // "outb %al, %dx".  Out doesn't take a memory form, but this is a widely
  // documented form in various unofficial manuals, so a lot of code uses it.
  if ((Name == "outb" || Name == "outw" || Name == "outl" || Name == "out") &&
      Operands.size() == 3) {
    X86Operand &Op = *(X86Operand*)Operands.back();
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands.back() = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
      delete &Op;
    }
  }
  // Same hack for "in[bwl]? (%dx), %al" -> "inb %dx, %al".
  if ((Name == "inb" || Name == "inw" || Name == "inl" || Name == "in") &&
      Operands.size() == 3) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands.begin()[1] = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
      delete &Op;
    }
  }
  // Transform "ins[bwl] %dx, %es:(%edi)" into "ins[bwl]"
  if (Name.startswith("ins") && Operands.size() == 3 &&
      (Name == "insb" || Name == "insw" || Name == "insl")) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (Op.isReg() && Op.getReg() == X86::DX && isDstOp(Op2)) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }

  // Transform "outs[bwl] %ds:(%esi), %dx" into "out[bwl]"
  if (Name.startswith("outs") && Operands.size() == 3 &&
      (Name == "outsb" || Name == "outsw" || Name == "outsl")) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (isSrcOp(Op) && Op2.isReg() && Op2.getReg() == X86::DX) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }

  // Transform "movs[bwl] %ds:(%esi), %es:(%edi)" into "movs[bwl]"
  if (Name.startswith("movs") && Operands.size() == 3 &&
      (Name == "movsb" || Name == "movsw" || Name == "movsl" ||
       (is64BitMode() && Name == "movsq"))) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (isSrcOp(Op) && isDstOp(Op2)) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }
  // Transform "lods[bwl] %ds:(%esi),{%al,%ax,%eax,%rax}" into "lods[bwl]"
  if (Name.startswith("lods") && Operands.size() == 3 &&
      (Name == "lods" || Name == "lodsb" || Name == "lodsw" ||
       Name == "lodsl" || (is64BitMode() && Name == "lodsq"))) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    X86Operand *Op2 = static_cast<X86Operand*>(Operands[2]);
    if (isSrcOp(*Op1) && Op2->isReg()) {
      const char *ins;
      unsigned reg = Op2->getReg();
      bool isLods = Name == "lods";
      if (reg == X86::AL && (isLods || Name == "lodsb"))
        ins = "lodsb";
      else if (reg == X86::AX && (isLods || Name == "lodsw"))
        ins = "lodsw";
      else if (reg == X86::EAX && (isLods || Name == "lodsl"))
        ins = "lodsl";
      else if (reg == X86::RAX && (isLods || Name == "lodsq"))
        ins = "lodsq";
      else
        ins = NULL;
      if (ins != NULL) {
        Operands.pop_back();
        Operands.pop_back();
        delete Op1;
        delete Op2;
        if (Name != ins)
          static_cast<X86Operand*>(Operands[0])->setTokenValue(ins);
      }
    }
  }
  // Transform "stos[bwl] {%al,%ax,%eax,%rax},%es:(%edi)" into "stos[bwl]"
  if (Name.startswith("stos") && Operands.size() == 3 &&
      (Name == "stos" || Name == "stosb" || Name == "stosw" ||
       Name == "stosl" || (is64BitMode() && Name == "stosq"))) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    X86Operand *Op2 = static_cast<X86Operand*>(Operands[2]);
    if (isDstOp(*Op2) && Op1->isReg()) {
      const char *ins;
      unsigned reg = Op1->getReg();
      bool isStos = Name == "stos";
      if (reg == X86::AL && (isStos || Name == "stosb"))
        ins = "stosb";
      else if (reg == X86::AX && (isStos || Name == "stosw"))
        ins = "stosw";
      else if (reg == X86::EAX && (isStos || Name == "stosl"))
        ins = "stosl";
      else if (reg == X86::RAX && (isStos || Name == "stosq"))
        ins = "stosq";
      else
        ins = NULL;
      if (ins != NULL) {
        Operands.pop_back();
        Operands.pop_back();
        delete Op1;
        delete Op2;
        if (Name != ins)
          static_cast<X86Operand*>(Operands[0])->setTokenValue(ins);
      }
    }
  }

  // FIXME: Hack to handle recognize s{hr,ar,hl} $1, <op>.  Canonicalize to
  // "shift <op>".
  if ((Name.startswith("shr") || Name.startswith("sar") ||
       Name.startswith("shl") || Name.startswith("sal") ||
       Name.startswith("rcl") || Name.startswith("rcr") ||
       Name.startswith("rol") || Name.startswith("ror")) &&
      Operands.size() == 3) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    if (Op1->isImm() && isa<MCConstantExpr>(Op1->getImm()) &&
        cast<MCConstantExpr>(Op1->getImm())->getValue() == 1) {
      delete Operands[1];
      Operands.erase(Operands.begin() + 1);
    }
  }
  
  // Transforms "int $3" into "int3" as a size optimization.  We can't write an
  // instalias with an immediate operand yet.
  if (Name == "int" && Operands.size() == 2) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    if (Op1->isImm() && isa<MCConstantExpr>(Op1->getImm()) &&
        cast<MCConstantExpr>(Op1->getImm())->getValue() == 3) {
      delete Operands[1];
      Operands.erase(Operands.begin() + 1);
      static_cast<X86Operand*>(Operands[0])->setTokenValue("int3");
    }
  }

  return false;
}

bool X86ATTAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  X86Operand *Op = static_cast<X86Operand*>(Operands[0]);
  assert(Op->isToken() && "Leading operand should always be a mnemonic!");

  // First, handle aliases that expand to multiple instructions.
  // FIXME: This should be replaced with a real .td file alias mechanism.
  // Also, MatchInstructionImpl should do actually *do* the EmitInstruction
  // call.
  if (Op->getToken() == "fstsw" || Op->getToken() == "fstcw" ||
      Op->getToken() == "fstsww" || Op->getToken() == "fstcww" ||
      Op->getToken() == "finit" || Op->getToken() == "fsave" ||
      Op->getToken() == "fstenv" || Op->getToken() == "fclex") {
    MCInst Inst;
    Inst.setOpcode(X86::WAIT);
    Out.EmitInstruction(Inst);

    const char *Repl =
      StringSwitch<const char*>(Op->getToken())
        .Case("finit",  "fninit")
        .Case("fsave",  "fnsave")
        .Case("fstcw",  "fnstcw")
        .Case("fstcww",  "fnstcw")
        .Case("fstenv", "fnstenv")
        .Case("fstsw",  "fnstsw")
        .Case("fstsww", "fnstsw")
        .Case("fclex",  "fnclex")
        .Default(0);
    assert(Repl && "Unknown wait-prefixed instruction");
    delete Operands[0];
    Operands[0] = X86Operand::CreateToken(Repl, IDLoc);
  }

  bool WasOriginallyInvalidOperand = false;
  unsigned OrigErrorInfo;
  MCInst Inst;

  // First, try a direct match.
  switch (MatchInstructionImpl(Operands, Inst, OrigErrorInfo)) {
  default: break;
  case Match_Success:
    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_ConversionFail:
    return Error(IDLoc, "unable to convert operands to instruction");
  case Match_InvalidOperand:
    WasOriginallyInvalidOperand = true;
    break;
  case Match_MnemonicFail:
    break;
  }

  // FIXME: Ideally, we would only attempt suffix matches for things which are
  // valid prefixes, and we could just infer the right unambiguous
  // type. However, that requires substantially more matcher support than the
  // following hack.

  // Change the operand to point to a temporary token.
  StringRef Base = Op->getToken();
  SmallString<16> Tmp;
  Tmp += Base;
  Tmp += ' ';
  Op->setTokenValue(Tmp.str());

  // If this instruction starts with an 'f', then it is a floating point stack
  // instruction.  These come in up to three forms for 32-bit, 64-bit, and
  // 80-bit floating point, which use the suffixes s,l,t respectively.
  //
  // Otherwise, we assume that this may be an integer instruction, which comes
  // in 8/16/32/64-bit forms using the b,w,l,q suffixes respectively.
  const char *Suffixes = Base[0] != 'f' ? "bwlq" : "slt\0";
  
  // Check for the various suffix matches.
  Tmp[Base.size()] = Suffixes[0];
  unsigned ErrorInfoIgnore;
  unsigned Match1, Match2, Match3, Match4;
  
  Match1 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore);
  Tmp[Base.size()] = Suffixes[1];
  Match2 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore);
  Tmp[Base.size()] = Suffixes[2];
  Match3 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore);
  Tmp[Base.size()] = Suffixes[3];
  Match4 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore);

  // Restore the old token.
  Op->setTokenValue(Base);

  // If exactly one matched, then we treat that as a successful match (and the
  // instruction will already have been filled in correctly, since the failing
  // matches won't have modified it).
  unsigned NumSuccessfulMatches =
    (Match1 == Match_Success) + (Match2 == Match_Success) +
    (Match3 == Match_Success) + (Match4 == Match_Success);
  if (NumSuccessfulMatches == 1) {
    Out.EmitInstruction(Inst);
    return false;
  }

  // Otherwise, the match failed, try to produce a decent error message.

  // If we had multiple suffix matches, then identify this as an ambiguous
  // match.
  if (NumSuccessfulMatches > 1) {
    char MatchChars[4];
    unsigned NumMatches = 0;
    if (Match1 == Match_Success) MatchChars[NumMatches++] = Suffixes[0];
    if (Match2 == Match_Success) MatchChars[NumMatches++] = Suffixes[1];
    if (Match3 == Match_Success) MatchChars[NumMatches++] = Suffixes[2];
    if (Match4 == Match_Success) MatchChars[NumMatches++] = Suffixes[3];

    SmallString<126> Msg;
    raw_svector_ostream OS(Msg);
    OS << "ambiguous instructions require an explicit suffix (could be ";
    for (unsigned i = 0; i != NumMatches; ++i) {
      if (i != 0)
        OS << ", ";
      if (i + 1 == NumMatches)
        OS << "or ";
      OS << "'" << Base << MatchChars[i] << "'";
    }
    OS << ")";
    Error(IDLoc, OS.str());
    return true;
  }

  // Okay, we know that none of the variants matched successfully.

  // If all of the instructions reported an invalid mnemonic, then the original
  // mnemonic was invalid.
  if ((Match1 == Match_MnemonicFail) && (Match2 == Match_MnemonicFail) &&
      (Match3 == Match_MnemonicFail) && (Match4 == Match_MnemonicFail)) {
    if (!WasOriginallyInvalidOperand) {
      return Error(IDLoc, "invalid instruction mnemonic '" + Base + "'",
                   Op->getLocRange());
    }

    // Recover location info for the operand if we know which was the problem.
    if (OrigErrorInfo != ~0U) {
      if (OrigErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      X86Operand *Operand = (X86Operand*)Operands[OrigErrorInfo];
      if (Operand->getStartLoc().isValid()) {
        SMRange OperandRange = Operand->getLocRange();
        return Error(Operand->getStartLoc(), "invalid operand for instruction",
                     OperandRange);
      }
    }

    return Error(IDLoc, "invalid operand for instruction");
  }

  // If one instruction matched with a missing feature, report this as a
  // missing feature.
  if ((Match1 == Match_MissingFeature) + (Match2 == Match_MissingFeature) +
      (Match3 == Match_MissingFeature) + (Match4 == Match_MissingFeature) == 1){
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  }

  // If one instruction matched with an invalid operand, report this as an
  // operand failure.
  if ((Match1 == Match_InvalidOperand) + (Match2 == Match_InvalidOperand) +
      (Match3 == Match_InvalidOperand) + (Match4 == Match_InvalidOperand) == 1){
    Error(IDLoc, "invalid operand for instruction");
    return true;
  }

  // If all of these were an outright failure, report it in a useless way.
  Error(IDLoc, "unknown use of instruction mnemonic without a size suffix");
  return true;
}


bool X86ATTAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  else if (IDVal.startswith(".code"))
    return ParseDirectiveCode(IDVal, DirectiveID.getLoc());
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool X86ATTAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().ParseExpression(Value))
        return true;
      
      getParser().getStreamer().EmitValue(Value, Size, 0 /*addrspace*/);
      
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

/// ParseDirectiveCode
///  ::= .code32 | .code64
bool X86ATTAsmParser::ParseDirectiveCode(StringRef IDVal, SMLoc L) {
  if (IDVal == ".code32") {
    Parser.Lex();
    if (is64BitMode()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
    }
  } else if (IDVal == ".code64") {
    Parser.Lex();
    if (!is64BitMode()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code64);
    }
  } else {
    return Error(L, "unexpected directive " + IDVal);
  }

  return false;
}


extern "C" void LLVMInitializeX86AsmLexer();

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  RegisterMCAsmParser<X86ATTAsmParser> X(TheX86_32Target);
  RegisterMCAsmParser<X86ATTAsmParser> Y(TheX86_64Target);
  LLVMInitializeX86AsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "X86GenAsmMatcher.inc"
