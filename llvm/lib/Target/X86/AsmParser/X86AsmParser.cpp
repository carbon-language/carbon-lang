//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmParser.h"
#include "X86.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

namespace {
struct X86Operand;

class X86ATTAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;

protected:
  unsigned Is64Bit : 1;

private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }

  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  X86Operand *ParseOperand();
  X86Operand *ParseMemOperand(unsigned SegReg, SMLoc StartLoc);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  void InstructionCleanup(MCInst &Inst);

  /// @name Auto-generated Match Functions
  /// {

  bool MatchInstruction(const SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCInst &Inst);

  bool MatchInstructionImpl(
    const SmallVectorImpl<MCParsedAsmOperand*> &Operands, MCInst &Inst);

  /// }

public:
  X86ATTAsmParser(const Target &T, MCAsmParser &_Parser)
    : TargetAsmParser(T), Parser(_Parser) {}

  virtual bool ParseInstruction(const StringRef &Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
 
class X86_32ATTAsmParser : public X86ATTAsmParser {
public:
  X86_32ATTAsmParser(const Target &T, MCAsmParser &_Parser)
    : X86ATTAsmParser(T, _Parser) {
    Is64Bit = false;
  }
};

class X86_64ATTAsmParser : public X86ATTAsmParser {
public:
  X86_64ATTAsmParser(const Target &T, MCAsmParser &_Parser)
    : X86ATTAsmParser(T, _Parser) {
    Is64Bit = true;
  }
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
  
  bool isImmSExt8() const { 
    // Accept immediates which fit in 8 bits when sign extended, and
    // non-absolute immediates.
    if (!isImm())
      return false;

    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm())) {
      int64_t Value = CE->getValue();
      return Value == (int64_t) (int8_t) Value;
    }

    return true;
  }
  
  bool isMem() const { return Kind == Memory; }

  bool isAbsMem() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1;
  }

  bool isNoSegMem() const {
    return Kind == Memory && !getMemSegReg();
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

  void addImmSExt8Operands(MCInst &Inst, unsigned N) const {
    // FIXME: Support user customization of the render method.
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

  void addNoSegMemOperands(MCInst &Inst, unsigned N) const {
    assert((N == 4) && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseReg()));
    Inst.addOperand(MCOperand::CreateImm(getMemScale()));
    Inst.addOperand(MCOperand::CreateReg(getMemIndexReg()));
    addExpr(Inst, getMemDisp());
  }

  static X86Operand *CreateToken(StringRef Str, SMLoc Loc) {
    X86Operand *Res = new X86Operand(Token, Loc, Loc);
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

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  RegNo = MatchRegisterName(Tok.getString());
  
  // Parse %st(1) and "%st" as "%st(0)"
  if (RegNo == 0 && Tok.getString() == "st") {
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
  }
  
  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the comma.

    // Following the comma we should have either an index register, or a scale
    // value. We don't support the later form, but we want to parse it
    // correctly.
    //
    // Not that even though it would be completely consistent to support syntax
    // like "1(%eax,,1)", the assembler doesn't.
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
      // Otherwise we have the unsupported form of a scale amount without an
      // index.
      SMLoc Loc = Parser.getTok().getLoc();

      int64_t Value;
      if (getParser().ParseAbsoluteExpression(Value))
        return 0;
      
      Error(Loc, "cannot have scale factor without index register");
      return 0;
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
ParseInstruction(const StringRef &Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // FIXME: Hack to recognize "sal..." and "rep..." for now. We need a way to
  // represent alternative syntaxes in the .td file, without requiring
  // instruction duplication.
  StringRef PatchedName = StringSwitch<StringRef>(Name)
    .Case("sal", "shl")
    .Case("salb", "shlb")
    .Case("sall", "shll")
    .Case("salq", "shlq")
    .Case("salw", "shlw")
    .Case("repe", "rep")
    .Case("repz", "rep")
    .Case("repnz", "repne")
    .Default(Name);
  Operands.push_back(X86Operand::CreateToken(PatchedName, NameLoc));

  if (getLexer().isNot(AsmToken::EndOfStatement)) {

    // Parse '*' modifier.
    if (getLexer().is(AsmToken::Star)) {
      SMLoc Loc = Parser.getTok().getLoc();
      Operands.push_back(X86Operand::CreateToken("*", Loc));
      Parser.Lex(); // Eat the star.
    }

    // Read the first operand.
    if (X86Operand *Op = ParseOperand())
      Operands.push_back(Op);
    else
      return true;
    
    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (X86Operand *Op = ParseOperand())
        Operands.push_back(Op);
      else
        return true;
    }
  }

  // FIXME: Hack to handle recognizing s{hr,ar,hl}? $1.
  if ((Name.startswith("shr") || Name.startswith("sar") ||
       Name.startswith("shl")) &&
      Operands.size() == 3 &&
      static_cast<X86Operand*>(Operands[1])->isImm() &&
      isa<MCConstantExpr>(static_cast<X86Operand*>(Operands[1])->getImm()) &&
      cast<MCConstantExpr>(static_cast<X86Operand*>(Operands[1])->getImm())->getValue() == 1) {
    delete Operands[1];
    Operands.erase(Operands.begin() + 1);
  }

  return false;
}

bool X86ATTAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
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

// FIXME: Custom X86 cleanup function to implement a temporary hack to handle
// matching INCL/DECL correctly for x86_64. This needs to be replaced by a
// proper mechanism for supporting (ambiguous) feature dependent instructions.
void X86ATTAsmParser::InstructionCleanup(MCInst &Inst) {
  if (!Is64Bit) return;

  switch (Inst.getOpcode()) {
  case X86::DEC16r: Inst.setOpcode(X86::DEC64_16r); break;
  case X86::DEC16m: Inst.setOpcode(X86::DEC64_16m); break;
  case X86::DEC32r: Inst.setOpcode(X86::DEC64_32r); break;
  case X86::DEC32m: Inst.setOpcode(X86::DEC64_32m); break;
  case X86::INC16r: Inst.setOpcode(X86::INC64_16r); break;
  case X86::INC16m: Inst.setOpcode(X86::INC64_16m); break;
  case X86::INC32r: Inst.setOpcode(X86::INC64_32r); break;
  case X86::INC32m: Inst.setOpcode(X86::INC64_32m); break;
  }
}

bool
X86ATTAsmParser::MatchInstruction(const SmallVectorImpl<MCParsedAsmOperand*>
                                    &Operands,
                                  MCInst &Inst) {
  // First, try a direct match.
  if (!MatchInstructionImpl(Operands, Inst))
    return false;

  // Ignore anything which is obviously not a suffix match.
  if (Operands.size() == 0)
    return true;
  X86Operand *Op = static_cast<X86Operand*>(Operands[0]);
  if (!Op->isToken() || Op->getToken().size() > 15)
    return true;

  // FIXME: Ideally, we would only attempt suffix matches for things which are
  // valid prefixes, and we could just infer the right unambiguous
  // type. However, that requires substantially more matcher support than the
  // following hack.

  // Change the operand to point to a temporary token.
  char Tmp[16];
  StringRef Base = Op->getToken();
  memcpy(Tmp, Base.data(), Base.size());
  Op->setTokenValue(StringRef(Tmp, Base.size() + 1));

  // Check for the various suffix matches.
  Tmp[Base.size()] = 'b';
  bool MatchB = MatchInstructionImpl(Operands, Inst);
  Tmp[Base.size()] = 'w';
  bool MatchW = MatchInstructionImpl(Operands, Inst);
  Tmp[Base.size()] = 'l';
  bool MatchL = MatchInstructionImpl(Operands, Inst);

  // Restore the old token.
  Op->setTokenValue(Base);

  // If exactly one matched, then we treat that as a successful match (and the
  // instruction will already have been filled in correctly, since the failing
  // matches won't have modified it).
  if (MatchB + MatchW + MatchL == 2)
    return false;

  // Otherwise, the match failed.
  return true;
}


extern "C" void LLVMInitializeX86AsmLexer();

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  RegisterAsmParser<X86_32ATTAsmParser> X(TheX86_32Target);
  RegisterAsmParser<X86_64ATTAsmParser> Y(TheX86_64Target);
  LLVMInitializeX86AsmLexer();
}

#include "X86GenAsmMatcher.inc"
