//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmLexer.h"
#include "llvm/MC/MCAsmParser.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

namespace {
struct X86Operand;

class X86ATTAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;

private:
  bool MatchInstruction(const StringRef &Name,
                        SmallVectorImpl<X86Operand> &Operands,
                        MCInst &Inst);

  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }

  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  bool ParseRegister(X86Operand &Op);

  bool ParseOperand(X86Operand &Op);

  bool ParseMemOperand(X86Operand &Op);
  
  /// @name Auto-generated Match Functions
  /// {  

  bool MatchRegisterName(const StringRef &Name, unsigned &RegNo);

  /// }

public:
  X86ATTAsmParser(const Target &T, MCAsmParser &_Parser)
    : TargetAsmParser(T), Parser(_Parser) {}

  virtual bool ParseInstruction(const StringRef &Name, MCInst &Inst);
};
  
} // end anonymous namespace


namespace {

/// X86Operand - Instances of this class represent a parsed X86 machine
/// instruction.
struct X86Operand {
  enum {
    Register,
    Immediate,
    Memory
  } Kind;

  union {
    struct {
      unsigned RegNo;
    } Reg;

    struct {
      MCValue Val;
    } Imm;

    struct {
      unsigned SegReg;
      MCValue Disp;
      unsigned BaseReg;
      unsigned IndexReg;
      unsigned Scale;
    } Mem;
  };

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNo;
  }

  static X86Operand CreateReg(unsigned RegNo) {
    X86Operand Res;
    Res.Kind = Register;
    Res.Reg.RegNo = RegNo;
    return Res;
  }
  static X86Operand CreateImm(MCValue Val) {
    X86Operand Res;
    Res.Kind = Immediate;
    Res.Imm.Val = Val;
    return Res;
  }
  static X86Operand CreateMem(unsigned SegReg, MCValue Disp, unsigned BaseReg,
                              unsigned IndexReg, unsigned Scale) {
    // If there is no index register, we should never have a scale, and we
    // should always have a scale (in {1,2,4,8}) if we do.
    assert(((Scale == 0 && !IndexReg) ||
            (IndexReg && (Scale == 1 || Scale == 2 ||
                          Scale == 4 || Scale == 8))) &&
           "Invalid scale!");
    X86Operand Res;
    Res.Kind = Memory;
    Res.Mem.SegReg   = SegReg;
    Res.Mem.Disp     = Disp;
    Res.Mem.BaseReg  = BaseReg;
    Res.Mem.IndexReg = IndexReg;
    Res.Mem.Scale    = Scale;
    return Res;
  }
};

} // end anonymous namespace.


bool X86ATTAsmParser::ParseRegister(X86Operand &Op) {
  const AsmToken &Tok = getLexer().getTok();
  assert(Tok.is(AsmToken::Register) && "Invalid token kind!");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  unsigned RegNo;
  assert(Tok.getString().startswith("%") && "Invalid register name!");
  if (MatchRegisterName(Tok.getString().substr(1), RegNo))
    return Error(Tok.getLoc(), "invalid register name");

  Op = X86Operand::CreateReg(RegNo);
  getLexer().Lex(); // Eat register token.

  return false;
}

bool X86ATTAsmParser::ParseOperand(X86Operand &Op) {
  switch (getLexer().getKind()) {
  default:
    return ParseMemOperand(Op);
  case AsmToken::Register:
    // FIXME: if a segment register, this could either be just the seg reg, or
    // the start of a memory operand.
    return ParseRegister(Op);
  case AsmToken::Dollar: {
    // $42 -> immediate.
    getLexer().Lex();
    MCValue Val;
    if (getParser().ParseRelocatableExpression(Val))
      return true;
    Op = X86Operand::CreateImm(Val);
    return false;
  }
  case AsmToken::Star:
    getLexer().Lex(); // Eat the star.
    
    if (getLexer().is(AsmToken::Register)) {
      if (ParseRegister(Op))
        return true;
    } else if (ParseMemOperand(Op))
      return true;

    // FIXME: Note the '*' in the operand for use by the matcher.
    return false;
  }
}

/// ParseMemOperand: segment: disp(basereg, indexreg, scale)
bool X86ATTAsmParser::ParseMemOperand(X86Operand &Op) {
  // FIXME: If SegReg ':'  (e.g. %gs:), eat and remember.
  unsigned SegReg = 0;
  
  // We have to disambiguate a parenthesized expression "(4+5)" from the start
  // of a memory operand with a missing displacement "(%ebx)" or "(,%eax)".  The
  // only way to do this without lookahead is to eat the ( and see what is after
  // it.
  MCValue Disp = MCValue::get(0, 0, 0);
  if (getLexer().isNot(AsmToken::LParen)) {
    if (getParser().ParseRelocatableExpression(Disp)) return true;
    
    // After parsing the base expression we could either have a parenthesized
    // memory address or not.  If not, return now.  If so, eat the (.
    if (getLexer().isNot(AsmToken::LParen)) {
      Op = X86Operand::CreateMem(SegReg, Disp, 0, 0, 0);
      return false;
    }
    
    // Eat the '('.
    getLexer().Lex();
  } else {
    // Okay, we have a '('.  We don't know if this is an expression or not, but
    // so we have to eat the ( to see beyond it.
    getLexer().Lex(); // Eat the '('.
    
    if (getLexer().is(AsmToken::Register) || getLexer().is(AsmToken::Comma)) {
      // Nothing to do here, fall into the code below with the '(' part of the
      // memory operand consumed.
    } else {
      // It must be an parenthesized expression, parse it now.
      if (getParser().ParseParenRelocatableExpression(Disp))
        return true;
      
      // After parsing the base expression we could either have a parenthesized
      // memory address or not.  If not, return now.  If so, eat the (.
      if (getLexer().isNot(AsmToken::LParen)) {
        Op = X86Operand::CreateMem(SegReg, Disp, 0, 0, 0);
        return false;
      }
      
      // Eat the '('.
      getLexer().Lex();
    }
  }
  
  // If we reached here, then we just ate the ( of the memory operand.  Process
  // the rest of the memory operand.
  unsigned BaseReg = 0, IndexReg = 0, Scale = 0;
  
  if (getLexer().is(AsmToken::Register)) {
    if (ParseRegister(Op))
      return true;
    BaseReg = Op.getReg();
  }
  
  if (getLexer().is(AsmToken::Comma)) {
    getLexer().Lex(); // Eat the comma.

    // Following the comma we should have either an index register, or a scale
    // value. We don't support the later form, but we want to parse it
    // correctly.
    //
    // Not that even though it would be completely consistent to support syntax
    // like "1(%eax,,1)", the assembler doesn't.
    if (getLexer().is(AsmToken::Register)) {
      if (ParseRegister(Op))
        return true;
      IndexReg = Op.getReg();
      Scale = 1;      // If not specified, the scale defaults to 1.
    
      if (getLexer().isNot(AsmToken::RParen)) {
        // Parse the scale amount:
        //  ::= ',' [scale-expression]
        if (getLexer().isNot(AsmToken::Comma))
          return true;
        getLexer().Lex(); // Eat the comma.

        if (getLexer().isNot(AsmToken::RParen)) {
          SMLoc Loc = getLexer().getTok().getLoc();

          int64_t ScaleVal;
          if (getParser().ParseAbsoluteExpression(ScaleVal))
            return true;
          
          // Validate the scale amount.
          if (ScaleVal != 1 && ScaleVal != 2 && ScaleVal != 4 && ScaleVal != 8)
            return Error(Loc, "scale factor in address must be 1, 2, 4 or 8");
          Scale = (unsigned)ScaleVal;
        }
      }
    } else if (getLexer().isNot(AsmToken::RParen)) {
      // Otherwise we have the unsupported form of a scale amount without an
      // index.
      SMLoc Loc = getLexer().getTok().getLoc();

      int64_t Value;
      if (getParser().ParseAbsoluteExpression(Value))
        return true;
      
      return Error(Loc, "cannot have scale factor without index register");
    }
  }
  
  // Ok, we've eaten the memory operand, verify we have a ')' and eat it too.
  if (getLexer().isNot(AsmToken::RParen))
    return Error(getLexer().getTok().getLoc(),
                    "unexpected token in memory operand");
  getLexer().Lex(); // Eat the ')'.
  
  Op = X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale);
  return false;
}

bool X86ATTAsmParser::ParseInstruction(const StringRef &Name, MCInst &Inst) {
  SmallVector<X86Operand, 3> Operands;

  SMLoc Loc = getLexer().getTok().getLoc();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    Operands.push_back(X86Operand());
    if (ParseOperand(Operands.back()))
      return true;

    while (getLexer().is(AsmToken::Comma)) {
      getLexer().Lex();  // Eat the comma.

      // Parse and remember the operand.
      Operands.push_back(X86Operand());
      if (ParseOperand(Operands.back()))
        return true;
    }
  }

  if (!MatchInstruction(Name, Operands, Inst))
    return false;

  // FIXME: We should give nicer diagnostics about the exact failure.

  // FIXME: For now we just treat unrecognized instructions as "warnings".
  Warning(Loc, "unrecognized instruction");

  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  RegisterAsmParser<X86ATTAsmParser> X(TheX86_32Target);
  RegisterAsmParser<X86ATTAsmParser> Y(TheX86_64Target);
}

// FIXME: These should come from tblgen.

// Match_X86_Op_GR8
static bool 
Match_X86_Op_GR8(const X86Operand &Op, MCOperand *MCOps, unsigned NumOps) {
  assert(NumOps == 1 && "Invalid number of ops!");

  // FIXME: Match correct registers.
  if (Op.Kind != X86Operand::Register)
    return true;

  MCOps[0].MakeReg(Op.getReg());
  return false;
}

#define DUMMY(name) \
  static bool Match_X86_Op_##name(const X86Operand &Op, \
                                  MCOperand *MCOps,     \
                                  unsigned NumMCOps) {  \
    return true;                                        \
  }

DUMMY(FR32)
DUMMY(FR64)
DUMMY(GR16)
DUMMY(GR32)
DUMMY(GR32_NOREX)
DUMMY(GR64)
DUMMY(GR8_NOREX)
DUMMY(RST)
DUMMY(VR128)
DUMMY(VR64)
DUMMY(brtarget)
DUMMY(brtarget8)
DUMMY(f128mem)
DUMMY(f32mem)
DUMMY(f64mem)
DUMMY(f80mem)
DUMMY(i128mem)
DUMMY(i16i8imm)
DUMMY(i16imm)
DUMMY(i16mem)
DUMMY(i32i8imm)
DUMMY(i32imm_pcrel)
DUMMY(i32imm)
DUMMY(i32mem)
DUMMY(i64i32imm_pcrel)
DUMMY(i64i32imm)
DUMMY(i64i8imm)
DUMMY(i64imm)
DUMMY(i64mem)
DUMMY(i8imm)
DUMMY(i8mem_NOREX)
DUMMY(i8mem)
DUMMY(lea32mem)
DUMMY(lea64_32mem)
DUMMY(lea64mem)
DUMMY(sdmem)
DUMMY(ssmem)

#include "X86GenAsmMatcher.inc"
