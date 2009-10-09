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
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
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

  bool ParseRegister(ARMOperand &Op);

  bool ParseRegisterList(ARMOperand &Op);

  bool ParseMemory(ARMOperand &Op);

  bool ParseShift(enum ShiftType *St, const MCExpr *ShiftAmount);

  bool ParseOperand(ARMOperand &Op);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  // TODO - For now hacked versions of the next two are in here in this file to
  // allow some parser testing until the table gen versions are implemented.

  /// @name Auto-generated Match Functions
  /// {
  bool MatchInstruction(SmallVectorImpl<ARMOperand> &Operands,
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

  virtual bool ParseInstruction(const StringRef &Name, MCInst &Inst);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
  
} // end anonymous namespace

namespace {

/// ARMOperand - Instances of this class represent a parsed ARM machine
/// instruction.
struct ARMOperand {
  enum {
    Token,
    Register,
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

    // This is for all forms of ARM address expressions
    struct {
      unsigned BaseRegNum;
      bool OffsetIsReg;
      const MCExpr *Offset; // used when OffsetIsReg is false
      unsigned OffsetRegNum; // used when OffsetIsReg is true
      bool OffsetRegShifted; // only used when OffsetIsReg is true
      enum ShiftType ShiftType;  // used when OffsetRegShifted is true
      const MCExpr *ShiftAmount; // used when OffsetRegShifted is true
      bool Preindexed;
      bool Postindexed;
      bool Negative; // only used when OffsetIsReg is true
      bool Writeback;
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

// Try to parse a register name.  The token must be an Identifier when called,
// and if it is a register name a Reg operand is created, the token is eaten
// and false is returned.  Else true is returned and no token is eaten.
// TODO this is likely to change to allow different register types and or to
// parse for a specific register type.
bool ARMAsmParser::ParseRegister(ARMOperand &Op) {
  const AsmToken &Tok = getLexer().getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  int RegNum;

  RegNum = MatchRegisterName(Tok.getString());
  if (RegNum == -1)
    return true;
  getLexer().Lex(); // Eat identifier token.

  bool Writeback = false;
  const AsmToken &ExclaimTok = getLexer().getTok();
  if (ExclaimTok.is(AsmToken::Exclaim)) {
    Writeback = true;
    getLexer().Lex(); // Eat exclaim token
  }

  Op = ARMOperand::CreateReg(RegNum, Writeback);

  return false;
}

// Try to parse a register list.  The first token must be a '{' when called
// for now.
bool ARMAsmParser::ParseRegisterList(ARMOperand &Op) {
  const AsmToken &LCurlyTok = getLexer().getTok();
  assert(LCurlyTok.is(AsmToken::LCurly) && "Token is not an Left Curly Brace");
  getLexer().Lex(); // Eat left curly brace token.

  const AsmToken &RegTok = getLexer().getTok();
  SMLoc RegLoc = RegTok.getLoc();
  if (RegTok.isNot(AsmToken::Identifier))
    return Error(RegLoc, "register expected");
  int RegNum = MatchRegisterName(RegTok.getString());
  if (RegNum == -1)
    return Error(RegLoc, "register expected");
  getLexer().Lex(); // Eat identifier token.
  unsigned RegList = 1 << RegNum;

  int HighRegNum = RegNum;
  // TODO ranges like "{Rn-Rm}"
  while (getLexer().getTok().is(AsmToken::Comma)) {
    getLexer().Lex(); // Eat comma token.

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

    getLexer().Lex(); // Eat identifier token.
  }
  const AsmToken &RCurlyTok = getLexer().getTok();
  if (RCurlyTok.isNot(AsmToken::RCurly))
    return Error(RCurlyTok.getLoc(), "'}' expected");
  getLexer().Lex(); // Eat left curly brace token.

  return false;
}

// Try to parse an arm memory expression.  It must start with a '[' token.
// TODO Only preindexing and postindexing addressing are started, unindexed
// with option, etc are still to do.
bool ARMAsmParser::ParseMemory(ARMOperand &Op) {
  const AsmToken &LBracTok = getLexer().getTok();
  assert(LBracTok.is(AsmToken::LBrac) && "Token is not an Left Bracket");
  getLexer().Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = getLexer().getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier))
    return Error(BaseRegTok.getLoc(), "register expected");
  int BaseRegNum = MatchRegisterName(BaseRegTok.getString());
  if (BaseRegNum == -1)
    return Error(BaseRegTok.getLoc(), "register expected");
  getLexer().Lex(); // Eat identifier token.

  bool Preindexed = false;
  bool Postindexed = false;
  bool OffsetIsReg = false;
  bool Negative = false;
  bool Writeback = false;

  // First look for preindexed address forms:
  //  [Rn, +/-Rm]
  //  [Rn, #offset]
  //  [Rn, +/-Rm, shift]
  // that is after the "[Rn" we now have see if the next token is a comma.
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.is(AsmToken::Comma)) {
    Preindexed = true;
    getLexer().Lex(); // Eat comma token.

    const AsmToken &NextTok = getLexer().getTok();
    if (NextTok.is(AsmToken::Plus))
      getLexer().Lex(); // Eat plus token.
    else if (NextTok.is(AsmToken::Minus)) {
      Negative = true;
      getLexer().Lex(); // Eat minus token
    }

    // See if there is a register following the "[Rn," we have so far.
    const AsmToken &OffsetRegTok = getLexer().getTok();
    int OffsetRegNum = MatchRegisterName(OffsetRegTok.getString());
    bool OffsetRegShifted = false;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset;
    if (OffsetRegNum != -1) {
      OffsetIsReg = true;
      getLexer().Lex(); // Eat identifier token for the offset register.
      // Look for a comma then a shift
      const AsmToken &Tok = getLexer().getTok();
      if (Tok.is(AsmToken::Comma)) {
        getLexer().Lex(); // Eat comma token.

        const AsmToken &Tok = getLexer().getTok();
        if (ParseShift(&ShiftType, ShiftAmount))
          return Error(Tok.getLoc(), "shift expected");
        OffsetRegShifted = true;
      }
    }
    else { // "[Rn," we have so far was not followed by "Rm"
      // Look for #offset following the "[Rn,"
      const AsmToken &HashTok = getLexer().getTok();
      if (HashTok.isNot(AsmToken::Hash))
        return Error(HashTok.getLoc(), "'#' expected");
      getLexer().Lex(); // Eat hash token.

      if (getParser().ParseExpression(Offset))
       return true;
    }
    const AsmToken &RBracTok = getLexer().getTok();
    if (RBracTok.isNot(AsmToken::RBrac))
      return Error(RBracTok.getLoc(), "']' expected");
    getLexer().Lex(); // Eat right bracket token.

    const AsmToken &ExclaimTok = getLexer().getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      Writeback = true;
      getLexer().Lex(); // Eat exclaim token
    }
    Op = ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                               OffsetRegShifted, ShiftType, ShiftAmount,
                               Preindexed, Postindexed, Negative, Writeback);
    return false;
  }
  // The "[Rn" we have so far was not followed by a comma.
  else if (Tok.is(AsmToken::RBrac)) {
    // This is a post indexing addressing forms:
    //  [Rn], #offset
    //  [Rn], +/-Rm
    //  [Rn], +/-Rm, shift
    // that is a ']' follows after the "[Rn".
    Postindexed = true;
    Writeback = true;
    getLexer().Lex(); // Eat right bracket token.

    const AsmToken &CommaTok = getLexer().getTok();
    if (CommaTok.isNot(AsmToken::Comma))
      return Error(CommaTok.getLoc(), "',' expected");
    getLexer().Lex(); // Eat comma token.

    const AsmToken &NextTok = getLexer().getTok();
    if (NextTok.is(AsmToken::Plus))
      getLexer().Lex(); // Eat plus token.
    else if (NextTok.is(AsmToken::Minus)) {
      Negative = true;
      getLexer().Lex(); // Eat minus token
    }

    // See if there is a register following the "[Rn]," we have so far.
    const AsmToken &OffsetRegTok = getLexer().getTok();
    int OffsetRegNum = MatchRegisterName(OffsetRegTok.getString());
    bool OffsetRegShifted = false;
    enum ShiftType ShiftType;
    const MCExpr *ShiftAmount;
    const MCExpr *Offset;
    if (OffsetRegNum != -1) {
      OffsetIsReg = true;
      getLexer().Lex(); // Eat identifier token for the offset register.
      // Look for a comma then a shift
      const AsmToken &Tok = getLexer().getTok();
      if (Tok.is(AsmToken::Comma)) {
        getLexer().Lex(); // Eat comma token.

        const AsmToken &Tok = getLexer().getTok();
        if (ParseShift(&ShiftType, ShiftAmount))
          return Error(Tok.getLoc(), "shift expected");
        OffsetRegShifted = true;
      }
    }
    else { // "[Rn]," we have so far was not followed by "Rm"
      // Look for #offset following the "[Rn],"
      const AsmToken &HashTok = getLexer().getTok();
      if (HashTok.isNot(AsmToken::Hash))
        return Error(HashTok.getLoc(), "'#' expected");
      getLexer().Lex(); // Eat hash token.

      if (getParser().ParseExpression(Offset))
       return true;
    }
    Op = ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset, OffsetRegNum,
                               OffsetRegShifted, ShiftType, ShiftAmount,
                               Preindexed, Postindexed, Negative, Writeback);
    return false;
  }

  return true;
}

/// ParseShift as one of these two:
///   ( lsl | lsr | asr | ror ) , # shift_amount
///   rrx
/// and returns true if it parses a shift otherwise it returns false.
bool ARMAsmParser::ParseShift(ShiftType *St, const MCExpr *ShiftAmount) {
  const AsmToken &Tok = getLexer().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return true;
  const StringRef &ShiftName = Tok.getString();
  if (ShiftName == "lsl" || ShiftName == "LSL")
    *St = Lsl;
  else if (ShiftName == "lsr" || ShiftName == "LSR")
    *St = Lsr;
  else if (ShiftName == "asr" || ShiftName == "ASR")
    *St = Asr;
  else if (ShiftName == "ror" || ShiftName == "ROR")
    *St = Ror;
  else if (ShiftName == "rrx" || ShiftName == "RRX")
    *St = Rrx;
  else
    return true;
  getLexer().Lex(); // Eat shift type token.

  // For all but a Rotate right there must be a '#' and a shift amount
  if (*St != Rrx) {
    // Look for # following the shift type
    const AsmToken &HashTok = getLexer().getTok();
    if (HashTok.isNot(AsmToken::Hash))
      return Error(HashTok.getLoc(), "'#' expected");
    getLexer().Lex(); // Eat hash token.

    if (getParser().ParseExpression(ShiftAmount))
      return true;
  }

  return false;
}

// A hack to allow some testing
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

// A hack to allow some testing
bool ARMAsmParser::MatchInstruction(SmallVectorImpl<ARMOperand> &Operands,
                                    MCInst &Inst) {
  struct ARMOperand Op0 = Operands[0];
  assert(Op0.Kind == ARMOperand::Token && "First operand not a Token");
  const StringRef &Mnemonic = Op0.getToken();
  if (Mnemonic == "add" ||
      Mnemonic == "stmfd" ||
      Mnemonic == "str" ||
      Mnemonic == "ldmfd" ||
      Mnemonic == "ldr" ||
      Mnemonic == "mov")
    return false;

  return true;
}

// TODO - this is a work in progress
bool ARMAsmParser::ParseOperand(ARMOperand &Op) {
  switch (getLexer().getKind()) {
  case AsmToken::Identifier:
    if (!ParseRegister(Op))
      return false;
    // TODO parse other operands that start with an identifier like labels
    return Error(getLexer().getTok().getLoc(), "labels not yet supported");
  case AsmToken::LBrac:
    if (!ParseMemory(Op))
      return false;
  case AsmToken::LCurly:
    if (!ParseRegisterList(Op))
      return(false);
  case AsmToken::Hash:
    return Error(getLexer().getTok().getLoc(), "immediates not yet supported");
  default:
    return Error(getLexer().getTok().getLoc(), "unexpected token in operand");
  }
}

bool ARMAsmParser::ParseInstruction(const StringRef &Name, MCInst &Inst) {
  SmallVector<ARMOperand, 7> Operands;

  Operands.push_back(ARMOperand::CreateToken(Name));

  SMLoc Loc = getLexer().getTok().getLoc();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {

    // Read the first operand.
    Operands.push_back(ARMOperand());
    if (ParseOperand(Operands.back()))
      return true;

    while (getLexer().is(AsmToken::Comma)) {
      getLexer().Lex();  // Eat the comma.

      // Parse and remember the operand.
      Operands.push_back(ARMOperand());
      if (ParseOperand(Operands.back()))
        return true;
    }
  }
  if (!MatchInstruction(Operands, Inst))
    return false;

  Error(Loc, "ARMAsmParser::ParseInstruction only partly implemented");
  return true;
}

bool ARMAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(4, DirectiveID.getLoc());
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

      getParser().getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;
      
      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      getLexer().Lex();
    }
  }

  getLexer().Lex();
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeARMAsmParser() {
  RegisterAsmParser<ARMAsmParser> X(TheARMTarget);
  RegisterAsmParser<ARMAsmParser> Y(TheThumbTarget);
}
