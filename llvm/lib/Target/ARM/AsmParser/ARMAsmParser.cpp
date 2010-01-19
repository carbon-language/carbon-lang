//===-- ARMAsmParser.cpp - Parse ARM assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmLexer.h"
#include "llvm/MC/MCAsmParser.h"
#include "llvm/MC/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

namespace {
struct ARMOperand;

// The shift types for register controlled shifts in arm memory addressing
enum ShiftType {
  Lsl,
  Lsr,
  Asr,
  Ror,
  Rrx
};

class ARMAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;

private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }

  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  bool MaybeParseRegister(ARMOperand &Op, bool ParseWriteBack);

  bool ParseRegisterList(ARMOperand &Op);

  bool ParseMemory(ARMOperand &Op);

  bool ParseMemoryOffsetReg(bool &Negative,
                            bool &OffsetRegShifted,
                            enum ShiftType &ShiftType,
                            const MCExpr *&ShiftAmount,
                            const MCExpr *&Offset,
                            bool &OffsetIsReg,
                            int &OffsetRegNum);

  bool ParseShift(enum ShiftType &St, const MCExpr *&ShiftAmount);

  bool ParseOperand(ARMOperand &Op);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  bool ParseDirectiveThumb(SMLoc L);

  bool ParseDirectiveThumbFunc(SMLoc L);

  bool ParseDirectiveCode(SMLoc L);

  bool ParseDirectiveSyntax(SMLoc L);

  // TODO - For now hacked versions of the next two are in here in this file to
  // allow some parser testing until the table gen versions are implemented.

  /// @name Auto-generated Match Functions
  /// {
  bool MatchInstruction(const SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCInst &Inst);

  /// MatchRegisterName - Match the given string to a register name and return
  /// its register number, or -1 if there is no match.  To allow return values
  /// to be used directly in register lists, arm registers have values between
  /// 0 and 15.
  int MatchRegisterName(const StringRef &Name);

  /// }


public:
  ARMAsmParser(const Target &T, MCAsmParser &_Parser)
    : TargetAsmParser(T), Parser(_Parser) {}

  virtual bool ParseInstruction(const StringRef &Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
  
/// ARMOperand - Instances of this class represent a parsed ARM machine
/// instruction.
struct ARMOperand : public MCParsedAsmOperand {
  enum {
    Token,
    Register,
    Immediate,
    Memory
  } Kind;


  union {
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

  bool isToken() const {return Kind == Token; }

  bool isReg() const { return Kind == Register; }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  static ARMOperand CreateToken(StringRef Str) {
    ARMOperand Res;
    Res.Kind = Token;
    Res.Tok.Data = Str.data();
    Res.Tok.Length = Str.size();
    return Res;
  }

  static ARMOperand CreateReg(unsigned RegNum, bool Writeback) {
    ARMOperand Res;
    Res.Kind = Register;
    Res.Reg.RegNum = RegNum;
    Res.Reg.Writeback = Writeback;
    return Res;
  }

  static ARMOperand CreateImm(const MCExpr *Val) {
    ARMOperand Res;
    Res.Kind = Immediate;
    Res.Imm.Val = Val;
    return Res;
  }

  static ARMOperand CreateMem(unsigned BaseRegNum, bool OffsetIsReg,
                              const MCExpr *Offset, unsigned OffsetRegNum,
                              bool OffsetRegShifted, enum ShiftType ShiftType,
                              const MCExpr *ShiftAmount, bool Preindexed,
                              bool Postindexed, bool Negative, bool Writeback) {
    ARMOperand Res;
    Res.Kind = Memory;
    Res.Mem.BaseRegNum = BaseRegNum;
    Res.Mem.OffsetIsReg = OffsetIsReg;
    Res.Mem.Offset = Offset;
    Res.Mem.OffsetRegNum = OffsetRegNum;
    Res.Mem.OffsetRegShifted = OffsetRegShifted;
    Res.Mem.ShiftType = ShiftType;
    Res.Mem.ShiftAmount = ShiftAmount;
    Res.Mem.Preindexed = Preindexed;
    Res.Mem.Postindexed = Postindexed;
    Res.Mem.Negative = Negative;
    Res.Mem.Writeback = Writeback;
    return Res;
  }
};

} // end anonymous namespace.

/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name a Reg operand is created, the token is eaten
/// and false is returned.  Else true is returned and no token is eaten.
/// TODO this is likely to change to allow different register types and or to
/// parse for a specific register type.
bool ARMAsmParser::MaybeParseRegister(ARMOperand &Op, bool ParseWriteBack) {
  const AsmToken &Tok = getLexer().getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  int RegNum;

  RegNum = MatchRegisterName(Tok.getString());
  if (RegNum == -1)
    return true;
  Parser.Lex(); // Eat identifier token.

  bool Writeback = false;
  if (ParseWriteBack) {
    const AsmToken &ExclaimTok = getLexer().getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      Writeback = true;
      Parser.Lex(); // Eat exclaim token
    }
  }

  Op = ARMOperand::CreateReg(RegNum, Writeback);

  return false;
}

/// Parse a register list, return false if successful else return true or an 
/// error.  The first token must be a '{' when called.
bool ARMAsmParser::ParseRegisterList(ARMOperand &Op) {
  assert(getLexer().getTok().is(AsmToken::LCurly) &&
         "Token is not an Left Curly Brace");
  Parser.Lex(); // Eat left curly brace token.

  const AsmToken &RegTok = getLexer().getTok();
  SMLoc RegLoc = RegTok.getLoc();
  if (RegTok.isNot(AsmToken::Identifier))
    return Error(RegLoc, "register expected");
  int RegNum = MatchRegisterName(RegTok.getString());
  if (RegNum == -1)
    return Error(RegLoc, "register expected");
  Parser.Lex(); // Eat identifier token.
  unsigned RegList = 1 << RegNum;

  int HighRegNum = RegNum;
  // TODO ranges like "{Rn-Rm}"
  while (getLexer().getTok().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat comma token.

    const AsmToken &RegTok = getLexer().getTok();
    SMLoc RegLoc = RegTok.getLoc();
    if (RegTok.isNot(AsmToken::Identifier))
      return Error(RegLoc, "register expected");
    int RegNum = MatchRegisterName(RegTok.getString());
    if (RegNum == -1)
      return Error(RegLoc, "register expected");

    if (RegList & (1 << RegNum))
      Warning(RegLoc, "register duplicated in register list");
    else if (RegNum <= HighRegNum)
      Warning(RegLoc, "register not in ascending order in register list");
    RegList |= 1 << RegNum;
    HighRegNum = RegNum;

    Parser.Lex(); // Eat identifier token.
  }
  const AsmToken &RCurlyTok = getLexer().getTok();
  if (RCurlyTok.isNot(AsmToken::RCurly))
    return Error(RCurlyTok.getLoc(), "'}' expected");
  Parser.Lex(); // Eat left curly brace token.

  return false;
}

/// Parse an arm memory expression, return false if successful else return true
/// or an error.  The first token must be a '[' when called.
/// TODO Only preindexing and postindexing addressing are started, unindexed
/// with option, etc are still to do.
bool ARMAsmParser::ParseMemory(ARMOperand &Op) {
  assert(getLexer().getTok().is(AsmToken::LBrac) &&
         "Token is not an Left Bracket");
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = getLexer().getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier))
    return Error(BaseRegTok.getLoc(), "register expected");
  if (MaybeParseRegister(Op, false))
    return Error(BaseRegTok.getLoc(), "register expected");
  int BaseRegNum = Op.getReg();

  bool Preindexed = false;
  bool Postindexed = false;
  bool OffsetIsReg = false;
  bool Negative = false;
  bool Writeback = false;

  // First look for preindexed address forms, that is after the "[Rn" we now
  // have to see if the next token is a comma.
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.is(AsmToken::Comma)) {
    Preindexed = true;
    Parser.Lex(); // Eat comma token.
    int OffsetRegNum;
    bool OffsetRegShifted;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset;
    if(ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType, ShiftAmount,
                            Offset, OffsetIsReg, OffsetRegNum))
      return true;
    const AsmToken &RBracTok = getLexer().getTok();
    if (RBracTok.isNot(AsmToken::RBrac))
      return Error(RBracTok.getLoc(), "']' expected");
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &ExclaimTok = getLexer().getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      Writeback = true;
      Parser.Lex(); // Eat exclaim token
    }
    Op = ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                               OffsetRegShifted, ShiftType, ShiftAmount,
                               Preindexed, Postindexed, Negative, Writeback);
    return false;
  }
  // The "[Rn" we have so far was not followed by a comma.
  else if (Tok.is(AsmToken::RBrac)) {
    // This is a post indexing addressing forms, that is a ']' follows after
    // the "[Rn".
    Postindexed = true;
    Writeback = true;
    Parser.Lex(); // Eat right bracket token.

    int OffsetRegNum = 0;
    bool OffsetRegShifted = false;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset;

    const AsmToken &NextTok = getLexer().getTok();
    if (NextTok.isNot(AsmToken::EndOfStatement)) {
      if (NextTok.isNot(AsmToken::Comma))
	return Error(NextTok.getLoc(), "',' expected");
      Parser.Lex(); // Eat comma token.
      if(ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType,
                              ShiftAmount, Offset, OffsetIsReg, OffsetRegNum))
        return true;
    }

    Op = ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                               OffsetRegShifted, ShiftType, ShiftAmount,
                               Preindexed, Postindexed, Negative, Writeback);
    return false;
  }

  return true;
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
                                        int &OffsetRegNum) {
  ARMOperand Op;
  Negative = false;
  OffsetRegShifted = false;
  OffsetIsReg = false;
  OffsetRegNum = -1;
  const AsmToken &NextTok = getLexer().getTok();
  if (NextTok.is(AsmToken::Plus))
    Parser.Lex(); // Eat plus token.
  else if (NextTok.is(AsmToken::Minus)) {
    Negative = true;
    Parser.Lex(); // Eat minus token
  }
  // See if there is a register following the "[Rn," or "[Rn]," we have so far.
  const AsmToken &OffsetRegTok = getLexer().getTok();
  if (OffsetRegTok.is(AsmToken::Identifier)) {
    OffsetIsReg = !MaybeParseRegister(Op, false);
    if (OffsetIsReg)
      OffsetRegNum = Op.getReg();
  }
  // If we parsed a register as the offset then their can be a shift after that
  if (OffsetRegNum != -1) {
    // Look for a comma then a shift
    const AsmToken &Tok = getLexer().getTok();
    if (Tok.is(AsmToken::Comma)) {
      Parser.Lex(); // Eat comma token.

      const AsmToken &Tok = getLexer().getTok();
      if (ParseShift(ShiftType, ShiftAmount))
	return Error(Tok.getLoc(), "shift expected");
      OffsetRegShifted = true;
    }
  }
  else { // the "[Rn," or "[Rn,]" we have so far was not followed by "Rm"
    // Look for #offset following the "[Rn," or "[Rn],"
    const AsmToken &HashTok = getLexer().getTok();
    if (HashTok.isNot(AsmToken::Hash))
      return Error(HashTok.getLoc(), "'#' expected");
    Parser.Lex(); // Eat hash token.

    if (getParser().ParseExpression(Offset))
     return true;
  }
  return false;
}

/// ParseShift as one of these two:
///   ( lsl | lsr | asr | ror ) , # shift_amount
///   rrx
/// and returns true if it parses a shift otherwise it returns false.
bool ARMAsmParser::ParseShift(ShiftType &St, const MCExpr *&ShiftAmount) {
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return true;
  const StringRef &ShiftName = Tok.getString();
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
  const AsmToken &HashTok = getLexer().getTok();
  if (HashTok.isNot(AsmToken::Hash))
    return Error(HashTok.getLoc(), "'#' expected");
  Parser.Lex(); // Eat hash token.

  if (getParser().ParseExpression(ShiftAmount))
    return true;

  return false;
}

/// A hack to allow some testing, to be replaced by a real table gen version.
int ARMAsmParser::MatchRegisterName(const StringRef &Name) {
  if (Name == "r0" || Name == "R0")
    return 0;
  else if (Name == "r1" || Name == "R1")
    return 1;
  else if (Name == "r2" || Name == "R2")
    return 2;
  else if (Name == "r3" || Name == "R3")
    return 3;
  else if (Name == "r3" || Name == "R3")
    return 3;
  else if (Name == "r4" || Name == "R4")
    return 4;
  else if (Name == "r5" || Name == "R5")
    return 5;
  else if (Name == "r6" || Name == "R6")
    return 6;
  else if (Name == "r7" || Name == "R7")
    return 7;
  else if (Name == "r8" || Name == "R8")
    return 8;
  else if (Name == "r9" || Name == "R9")
    return 9;
  else if (Name == "r10" || Name == "R10")
    return 10;
  else if (Name == "r11" || Name == "R11" || Name == "fp")
    return 11;
  else if (Name == "r12" || Name == "R12" || Name == "ip")
    return 12;
  else if (Name == "r13" || Name == "R13" || Name == "sp")
    return 13;
  else if (Name == "r14" || Name == "R14" || Name == "lr")
      return 14;
  else if (Name == "r15" || Name == "R15" || Name == "pc")
    return 15;
  return -1;
}

/// A hack to allow some testing, to be replaced by a real table gen version.
bool ARMAsmParser::
MatchInstruction(const SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                 MCInst &Inst) {
  ARMOperand &Op0 = *(ARMOperand*)Operands[0];
  assert(Op0.Kind == ARMOperand::Token && "First operand not a Token");
  const StringRef &Mnemonic = Op0.getToken();
  if (Mnemonic == "add" ||
      Mnemonic == "stmfd" ||
      Mnemonic == "str" ||
      Mnemonic == "ldmfd" ||
      Mnemonic == "ldr" ||
      Mnemonic == "mov" ||
      Mnemonic == "sub" ||
      Mnemonic == "bl" ||
      Mnemonic == "push" ||
      Mnemonic == "blx" ||
      Mnemonic == "pop") {
    // Hard-coded to a valid instruction, till we have a real matcher.
    Inst = MCInst();
    Inst.setOpcode(ARM::MOVr);
    Inst.addOperand(MCOperand::CreateReg(2));
    Inst.addOperand(MCOperand::CreateReg(2));
    Inst.addOperand(MCOperand::CreateImm(0));
    Inst.addOperand(MCOperand::CreateImm(0));
    Inst.addOperand(MCOperand::CreateReg(0));
    return false;
  }

  return true;
}

/// Parse a arm instruction operand.  For now this parses the operand regardless
/// of the mnemonic.
bool ARMAsmParser::ParseOperand(ARMOperand &Op) {
  switch (getLexer().getKind()) {
  case AsmToken::Identifier:
    if (!MaybeParseRegister(Op, true))
      return false;
    // This was not a register so parse other operands that start with an
    // identifier (like labels) as expressions and create them as immediates.
    const MCExpr *IdVal;
    if (getParser().ParseExpression(IdVal))
      return true;
    Op = ARMOperand::CreateImm(IdVal);
    return false;
  case AsmToken::LBrac:
    return ParseMemory(Op);
  case AsmToken::LCurly:
    return ParseRegisterList(Op);
  case AsmToken::Hash:
    // #42 -> immediate.
    // TODO: ":lower16:" and ":upper16:" modifiers after # before immediate
    Parser.Lex();
    const MCExpr *ImmVal;
    if (getParser().ParseExpression(ImmVal))
      return true;
    Op = ARMOperand::CreateImm(ImmVal);
    return false;
  default:
    return Error(getLexer().getTok().getLoc(), "unexpected token in operand");
  }
}

/// Parse an arm instruction mnemonic followed by its operands.
bool ARMAsmParser::ParseInstruction(const StringRef &Name, SMLoc NameLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  Operands.push_back(new ARMOperand(ARMOperand::CreateToken(Name)));

  SMLoc Loc = getLexer().getTok().getLoc();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {

    // Read the first operand.
    ARMOperand Op;
    if (ParseOperand(Op)) return true;
    Operands.push_back(new ARMOperand(Op));

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Op)) return true;
      Operands.push_back(new ARMOperand(Op));
    }
  }
  return false;
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
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.isNot(AsmToken::Identifier) && Tok.isNot(AsmToken::String))
    return Error(L, "unexpected token in .syntax directive");
  StringRef ATTRIBUTE_UNUSED SymbolName = getLexer().getTok().getIdentifier();
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
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return Error(L, "unexpected token in .syntax directive");
  const StringRef &Mode = Tok.getString();
  bool unified_syntax;
  if (Mode == "unified" || Mode == "UNIFIED") {
    Parser.Lex();
    unified_syntax = true;
  }
  else if (Mode == "divided" || Mode == "DIVIDED") {
    Parser.Lex();
    unified_syntax = false;
  }
  else
    return Error(L, "unrecognized syntax mode in .syntax directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  // TODO tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// ParseDirectiveCode
///  ::= .code 16 | 32
bool ARMAsmParser::ParseDirectiveCode(SMLoc L) {
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.isNot(AsmToken::Integer))
    return Error(L, "unexpected token in .code directive");
  int64_t Val = getLexer().getTok().getIntVal();
  bool thumb_mode;
  if (Val == 16) {
    Parser.Lex();
    thumb_mode = true;
  }
  else if (Val == 32) {
    Parser.Lex();
    thumb_mode = false;
  }
  else
    return Error(L, "invalid operand to .code directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  // TODO tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// Force static initialization.
extern "C" void LLVMInitializeARMAsmParser() {
  RegisterAsmParser<ARMAsmParser> X(TheARMTarget);
  RegisterAsmParser<ARMAsmParser> Y(TheThumbTarget);
}
