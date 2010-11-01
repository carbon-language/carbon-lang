//===-- ARMAsmParser.cpp - Parse ARM assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

// The shift types for register controlled shifts in arm memory addressing
enum ShiftType {
  Lsl,
  Lsr,
  Asr,
  Ror,
  Rrx
};

namespace {
  struct ARMOperand;

class ARMAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;
  TargetMachine &TM;

private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }

  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  int TryParseRegister();
  ARMOperand *TryParseRegisterWithWriteBack();
  ARMOperand *ParseRegisterList();
  ARMOperand *ParseMemory();

  bool ParseMemoryOffsetReg(bool &Negative,
                            bool &OffsetRegShifted,
                            enum ShiftType &ShiftType,
                            const MCExpr *&ShiftAmount,
                            const MCExpr *&Offset,
                            bool &OffsetIsReg,
                            int &OffsetRegNum,
                            SMLoc &E);

  bool ParseShift(enum ShiftType &St, const MCExpr *&ShiftAmount, SMLoc &E);

  ARMOperand *ParseOperand();

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  bool ParseDirectiveThumb(SMLoc L);

  bool ParseDirectiveThumbFunc(SMLoc L);

  bool ParseDirectiveCode(SMLoc L);

  bool ParseDirectiveSyntax(SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "ARMGenAsmMatcher.inc"

  /// }


public:
  ARMAsmParser(const Target &T, MCAsmParser &_Parser, TargetMachine &_TM)
    : TargetAsmParser(T), Parser(_Parser), TM(_TM) {
      // Initialize the set of available features.
      setAvailableFeatures(ComputeAvailableFeatures(
          &TM.getSubtarget<ARMSubtarget>()));
    }

  virtual bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
} // end anonymous namespace

namespace {

/// ARMOperand - Instances of this class represent a parsed ARM machine
/// instruction.
struct ARMOperand : public MCParsedAsmOperand {
public:
  enum KindTy {
    CondCode,
    Immediate,
    Memory,
    Register,
    Token
  } Kind;

  SMLoc StartLoc, EndLoc;

  union {
    struct {
      ARMCC::CondCodes Val;
    } CC;

    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNum;
      bool Writeback;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;

    // This is for all forms of ARM address expressions
    struct {
      unsigned BaseRegNum;
      unsigned OffsetRegNum; // used when OffsetIsReg is true
      const MCExpr *Offset; // used when OffsetIsReg is false
      const MCExpr *ShiftAmount; // used when OffsetRegShifted is true
      enum ShiftType ShiftType;  // used when OffsetRegShifted is true
      unsigned
        OffsetRegShifted : 1, // only used when OffsetIsReg is true
        Preindexed : 1,
        Postindexed : 1,
        OffsetIsReg : 1,
        Negative : 1, // only used when OffsetIsReg is true
        Writeback : 1;
    } Mem;

  };

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
    case Register:
      Reg = o.Reg;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case Memory:
      Mem = o.Mem;
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

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  bool isCondCode() const { return Kind == CondCode; }
  bool isImm() const { return Kind == Immediate; }
  bool isReg() const { return Kind == Register; }
  bool isToken() const { return Kind == Token; }
  bool isMemory() const { return Kind == Memory; }

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
    // FIXME: What belongs here?
    Inst.addOperand(MCOperand::CreateReg(0));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }


  bool isMemMode5() const {
    if (!isMemory() || Mem.OffsetIsReg || Mem.OffsetRegShifted ||
        Mem.Writeback || Mem.Negative)
      return false;
    // If there is an offset expression, make sure it's valid.
    if (!Mem.Offset)
      return true;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset);
    if (!CE)
      return false;
    // The offset must be a multiple of 4 in the range 0-1020.
    int64_t Value = CE->getValue();
    return ((Value & 0x3) == 0 && Value <= 1020 && Value >= -1020);
  }

  void addMemMode5Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemMode5() && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    assert(!Mem.OffsetIsReg && "invalid mode 5 operand");
    // FIXME: #-0 is encoded differently than #0. Does the parser preserve
    // the difference?
    if (Mem.Offset) {
      const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset);
      assert(CE && "non-constant mode 5 offset operand!");
      // The MCInst offset operand doesn't include the low two bits (like
      // the instruction encoding).
      Inst.addOperand(MCOperand::CreateImm(CE->getValue() / 4));
    } else
      Inst.addOperand(MCOperand::CreateImm(0));
  }

  virtual void dump(raw_ostream &OS) const;

  static ARMOperand *CreateCondCode(ARMCC::CondCodes CC, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CondCode);
    Op->CC.Val = CC;
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

  static ARMOperand *CreateReg(unsigned RegNum, bool Writeback, SMLoc S,
                               SMLoc E) {
    ARMOperand *Op = new ARMOperand(Register);
    Op->Reg.RegNum = RegNum;
    Op->Reg.Writeback = Writeback;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateMem(unsigned BaseRegNum, bool OffsetIsReg,
                               const MCExpr *Offset, unsigned OffsetRegNum,
                               bool OffsetRegShifted, enum ShiftType ShiftType,
                               const MCExpr *ShiftAmount, bool Preindexed,
                               bool Postindexed, bool Negative, bool Writeback,
                               SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Memory);
    Op->Mem.BaseRegNum = BaseRegNum;
    Op->Mem.OffsetIsReg = OffsetIsReg;
    Op->Mem.Offset = Offset;
    Op->Mem.OffsetRegNum = OffsetRegNum;
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

private:
  ARMOperand(KindTy K) : Kind(K) {}
};

} // end anonymous namespace.

void ARMOperand::dump(raw_ostream &OS) const {
  switch (Kind) {
  case CondCode:
    OS << ARMCondCodeToString(getCondCode());
    break;
  case Immediate:
    getImm()->print(OS);
    break;
  case Memory:
    OS << "<memory>";
    break;
  case Register:
    OS << "<register " << getReg() << ">";
    break;
  case Token:
    OS << "'" << getToken() << "'";
    break;
  }
}

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name the token is eaten and the register number is
/// returned.  Otherwise return -1.
///
int ARMAsmParser::TryParseRegister() {
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  int RegNum = MatchRegisterName(Tok.getString());
  if (RegNum == -1)
    return -1;
  Parser.Lex(); // Eat identifier token.
  return RegNum;
}


/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name the token is eaten and the register number is
/// returned.  Otherwise return -1.
///
/// TODO this is likely to change to allow different register types and or to
/// parse for a specific register type.
ARMOperand *ARMAsmParser::TryParseRegisterWithWriteBack() {
  SMLoc S = Parser.getTok().getLoc();
  int RegNo = TryParseRegister();
  if (RegNo == -1) return 0;

  SMLoc E = Parser.getTok().getLoc();

  bool Writeback = false;
  const AsmToken &ExclaimTok = Parser.getTok();
  if (ExclaimTok.is(AsmToken::Exclaim)) {
    E = ExclaimTok.getLoc();
    Writeback = true;
    Parser.Lex(); // Eat exclaim token
  }

  return ARMOperand::CreateReg(RegNo, Writeback, S, E);
}

/// Parse a register list, return it if successful else return null.  The first
/// token must be a '{' when called.
ARMOperand *ARMAsmParser::ParseRegisterList() {
  SMLoc S, E;
  assert(Parser.getTok().is(AsmToken::LCurly) &&
         "Token is not an Left Curly Brace");
  S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat left curly brace token.

  const AsmToken &RegTok = Parser.getTok();
  SMLoc RegLoc = RegTok.getLoc();
  if (RegTok.isNot(AsmToken::Identifier)) {
    Error(RegLoc, "register expected");
    return 0;
  }
  int RegNum = MatchRegisterName(RegTok.getString());
  if (RegNum == -1) {
    Error(RegLoc, "register expected");
    return 0;
  }

  Parser.Lex(); // Eat identifier token.
  unsigned RegList = 1 << RegNum;

  int HighRegNum = RegNum;
  // TODO ranges like "{Rn-Rm}"
  while (Parser.getTok().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat comma token.

    const AsmToken &RegTok = Parser.getTok();
    SMLoc RegLoc = RegTok.getLoc();
    if (RegTok.isNot(AsmToken::Identifier)) {
      Error(RegLoc, "register expected");
      return 0;
    }
    int RegNum = MatchRegisterName(RegTok.getString());
    if (RegNum == -1) {
      Error(RegLoc, "register expected");
      return 0;
    }

    if (RegList & (1 << RegNum))
      Warning(RegLoc, "register duplicated in register list");
    else if (RegNum <= HighRegNum)
      Warning(RegLoc, "register not in ascending order in register list");
    RegList |= 1 << RegNum;
    HighRegNum = RegNum;

    Parser.Lex(); // Eat identifier token.
  }
  const AsmToken &RCurlyTok = Parser.getTok();
  if (RCurlyTok.isNot(AsmToken::RCurly)) {
    Error(RCurlyTok.getLoc(), "'}' expected");
    return 0;
  }
  E = RCurlyTok.getLoc();
  Parser.Lex(); // Eat left curly brace token.

  // FIXME: Need to return an operand!
  Error(E, "FIXME: register list parsing not implemented");
  return 0;
}

/// Parse an arm memory expression, return false if successful else return true
/// or an error.  The first token must be a '[' when called.
/// TODO Only preindexing and postindexing addressing are started, unindexed
/// with option, etc are still to do.
ARMOperand *ARMAsmParser::ParseMemory() {
  SMLoc S, E;
  assert(Parser.getTok().is(AsmToken::LBrac) &&
         "Token is not an Left Bracket");
  S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier)) {
    Error(BaseRegTok.getLoc(), "register expected");
    return 0;
  }
  int BaseRegNum = TryParseRegister();
  if (BaseRegNum == -1) {
    Error(BaseRegTok.getLoc(), "register expected");
    return 0;
  }

  bool Preindexed = false;
  bool Postindexed = false;
  bool OffsetIsReg = false;
  bool Negative = false;
  bool Writeback = false;

  // First look for preindexed address forms, that is after the "[Rn" we now
  // have to see if the next token is a comma.
  const AsmToken &Tok = Parser.getTok();
  if (Tok.is(AsmToken::Comma)) {
    Preindexed = true;
    Parser.Lex(); // Eat comma token.
    int OffsetRegNum;
    bool OffsetRegShifted;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset;
    if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType, ShiftAmount,
                             Offset, OffsetIsReg, OffsetRegNum, E))
      return 0;
    const AsmToken &RBracTok = Parser.getTok();
    if (RBracTok.isNot(AsmToken::RBrac)) {
      Error(RBracTok.getLoc(), "']' expected");
      return 0;
    }
    E = RBracTok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &ExclaimTok = Parser.getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      E = ExclaimTok.getLoc();
      Writeback = true;
      Parser.Lex(); // Eat exclaim token
    }
    return ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                                 OffsetRegShifted, ShiftType, ShiftAmount,
                                 Preindexed, Postindexed, Negative, Writeback,
                                 S, E);
  }
  // The "[Rn" we have so far was not followed by a comma.
  else if (Tok.is(AsmToken::RBrac)) {
    // If there's anything other than the right brace, this is a post indexing
    // addressing form.
    E = Tok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    int OffsetRegNum = 0;
    bool OffsetRegShifted = false;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset = 0;

    const AsmToken &NextTok = Parser.getTok();
    if (NextTok.isNot(AsmToken::EndOfStatement)) {
      Postindexed = true;
      Writeback = true;
      if (NextTok.isNot(AsmToken::Comma)) {
        Error(NextTok.getLoc(), "',' expected");
        return 0;
      }
      Parser.Lex(); // Eat comma token.
      if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType,
                               ShiftAmount, Offset, OffsetIsReg, OffsetRegNum,
                               E))
        return 0;
    }

    return ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                                 OffsetRegShifted, ShiftType, ShiftAmount,
                                 Preindexed, Postindexed, Negative, Writeback,
                                 S, E);
  }

  return 0;
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
                                        enum ShiftType &ShiftType,
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

  // If we parsed a register as the offset then their can be a shift after that
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
bool ARMAsmParser::ParseShift(ShiftType &St, const MCExpr *&ShiftAmount,
                              SMLoc &E) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return true;
  StringRef ShiftName = Tok.getString();
  if (ShiftName == "lsl" || ShiftName == "LSL")
    St = Lsl;
  else if (ShiftName == "lsr" || ShiftName == "LSR")
    St = Lsr;
  else if (ShiftName == "asr" || ShiftName == "ASR")
    St = Asr;
  else if (ShiftName == "ror" || ShiftName == "ROR")
    St = Ror;
  else if (ShiftName == "rrx" || ShiftName == "RRX")
    St = Rrx;
  else
    return true;
  Parser.Lex(); // Eat shift type token.

  // Rrx stands alone.
  if (St == Rrx)
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
ARMOperand *ARMAsmParser::ParseOperand() {
  SMLoc S, E;

  switch (getLexer().getKind()) {
  case AsmToken::Identifier:
    if (ARMOperand *Op = TryParseRegisterWithWriteBack())
      return Op;

    // This was not a register so parse other operands that start with an
    // identifier (like labels) as expressions and create them as immediates.
    const MCExpr *IdVal;
    S = Parser.getTok().getLoc();
    if (getParser().ParseExpression(IdVal))
      return 0;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    return ARMOperand::CreateImm(IdVal, S, E);
  case AsmToken::LBrac:
    return ParseMemory();
  case AsmToken::LCurly:
    return ParseRegisterList();
  case AsmToken::Hash:
    // #42 -> immediate.
    // TODO: ":lower16:" and ":upper16:" modifiers after # before immediate
    S = Parser.getTok().getLoc();
    Parser.Lex();
    const MCExpr *ImmVal;
    if (getParser().ParseExpression(ImmVal))
      return 0;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    return ARMOperand::CreateImm(ImmVal, S, E);
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return 0;
  }
}

/// Parse an arm instruction mnemonic followed by its operands.
bool ARMAsmParser::ParseInstruction(StringRef Name, SMLoc NameLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create the leading tokens for the mnemonic, split by '.' characters.
  size_t Start = 0, Next = Name.find('.');
  StringRef Head = Name.slice(Start, Next);

  // Determine the predicate, if any.
  //
  // FIXME: We need a way to check whether a prefix supports predication,
  // otherwise we will end up with an ambiguity for instructions that happen to
  // end with a predicate name.
  // FIXME: Likewise, some arithmetic instructions have an 's' prefix which
  // indicates to update the condition codes. Those instructions have an
  // additional immediate operand which encodes the prefix as reg0 or CPSR.
  // Just checking for a suffix of 's' definitely creates ambiguities; e.g,
  // the SMMLS instruction.
  unsigned CC = StringSwitch<unsigned>(Head.substr(Head.size()-2))
    .Case("eq", ARMCC::EQ)
    .Case("ne", ARMCC::NE)
    .Case("hs", ARMCC::HS)
    .Case("lo", ARMCC::LO)
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

  if (CC == ~0U ||
      (CC == ARMCC::LS && (Head == "vmls" || Head == "vnmls"))) {
    CC = ARMCC::AL;
  } else {
    Head = Head.slice(0, Head.size() - 2);
  }

  Operands.push_back(ARMOperand::CreateToken(Head, NameLoc));
  // FIXME: Should only add this operand for predicated instructions
  Operands.push_back(ARMOperand::CreateCondCode(ARMCC::CondCodes(CC), NameLoc));

  // Add the remaining tokens in the mnemonic.
  while (Next != StringRef::npos) {
    Start = Next;
    Next = Name.find('.', Start + 1);
    Head = Name.slice(Start, Next);

    Operands.push_back(ARMOperand::CreateToken(Head, NameLoc));
  }

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ARMOperand *Op = ParseOperand())
      Operands.push_back(Op);
    else {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ARMOperand *Op = ParseOperand())
        Operands.push_back(Op);
      else {
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
  return false;
}

bool ARMAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  MCInst Inst;
  unsigned ErrorInfo;
  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo)) {
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
  }

  llvm_unreachable("Implement any new match types added!");
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
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier) && Tok.isNot(AsmToken::String))
    return Error(L, "unexpected token in .syntax directive");
  Parser.Lex(); // Consume the identifier token.

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // TODO: mark symbol as a thumb symbol
  // getParser().getStreamer().Emit???();
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
    Parser.Lex();
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

  // TODO tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
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
