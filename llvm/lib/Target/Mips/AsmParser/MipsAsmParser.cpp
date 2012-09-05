//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsRegisterInfo.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {

class MipsAsmParser : public MCTargetAsmParser {

  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

#define GET_ASSEMBLER_HEADER
#include "MipsGenAsmMatcher.inc"

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirective(AsmToken DirectiveID);

  MipsAsmParser::OperandMatchResultTy
  parseMemOperand(SmallVectorImpl<MCParsedAsmOperand*>&);

  unsigned
  getMCInstOperandNum(unsigned Kind, MCInst &Inst,
                      const SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                      unsigned OperandNum, unsigned &NumMCOperands);

  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &,
                    StringRef Mnemonic);

  int tryParseRegister(StringRef Mnemonic);

  bool tryParseRegisterOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               StringRef Mnemonic);

  bool isMips64() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64) != 0;
  }

  int matchRegisterName(StringRef Symbol);

  int matchRegisterByNumber(unsigned RegNum, StringRef Mnemonic);

  unsigned getReg(int RC,int RegNo);

public:
  MipsAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser(), STI(sti), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

};
}

namespace {

/// MipsOperand - Instances of this class represent a parsed Mips machine
/// instruction.
class MipsOperand : public MCParsedAsmOperand {

  enum KindTy {
    k_CondCode,
    k_CoprocNum,
    k_Immediate,
    k_Memory,
    k_PostIndexRegister,
    k_Register,
    k_Token
  } Kind;

  MipsOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  union {
    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNum;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;
  };

  SMLoc StartLoc, EndLoc;

public:
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
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

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst,Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("unimplemented!");
  }

  bool isReg() const { return Kind == k_Register; }
  bool isImm() const { return Kind == k_Immediate; }
  bool isToken() const { return Kind == k_Token; }
  bool isMem() const { return Kind == k_Memory; }

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

  static MipsOperand *CreateToken(StringRef Str, SMLoc S) {
    MipsOperand *Op = new MipsOperand(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static MipsOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  virtual void print(raw_ostream &OS) const {
    llvm_unreachable("unimplemented!");
  }
};
}

unsigned MipsAsmParser::
getMCInstOperandNum(unsigned Kind, MCInst &Inst,
                    const SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                    unsigned OperandNum, unsigned &NumMCOperands) {
  assert (0 && "getMCInstOperandNum() not supported by the Mips target.");
  // The Mips backend doesn't currently include the matcher implementation, so
  // the getMCInstOperandNumImpl() is undefined.  This is a temporary
  // work around.
  NumMCOperands = 0;
  return 0;
}

bool MipsAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  MCInst Inst;
  unsigned ErrorInfo;
  unsigned Kind;
  unsigned MatchResult = MatchInstructionImpl(Operands, Kind, Inst, ErrorInfo);

  switch (MatchResult) {
  default: break;
  case Match_Success: {
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst);
    return false;
  }
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((MipsOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }
  return true;
}

int MipsAsmParser::matchRegisterName(StringRef Name) {

   int CC = StringSwitch<unsigned>(Name)
    .Case("zero",  Mips::ZERO)
    .Case("a0",  Mips::A0)
    .Case("a1",  Mips::A1)
    .Case("a2",  Mips::A2)
    .Case("a3",  Mips::A3)
    .Case("v0",  Mips::V0)
    .Case("v1",  Mips::V1)
    .Case("s0",  Mips::S0)
    .Case("s1",  Mips::S1)
    .Case("s2",  Mips::S2)
    .Case("s3",  Mips::S3)
    .Case("s4",  Mips::S4)
    .Case("s5",  Mips::S5)
    .Case("s6",  Mips::S6)
    .Case("s7",  Mips::S7)
    .Case("k0",  Mips::K0)
    .Case("k1",  Mips::K1)
    .Case("sp",  Mips::SP)
    .Case("fp",  Mips::FP)
    .Case("gp",  Mips::GP)
    .Case("ra",  Mips::RA)
    .Case("t0",  Mips::T0)
    .Case("t1",  Mips::T1)
    .Case("t2",  Mips::T2)
    .Case("t3",  Mips::T3)
    .Case("t4",  Mips::T4)
    .Case("t5",  Mips::T5)
    .Case("t6",  Mips::T6)
    .Case("t7",  Mips::T7)
    .Case("t8",  Mips::T8)
    .Case("t9",  Mips::T9)
    .Case("at",  Mips::AT)
    .Case("fcc0",  Mips::FCC0)
    .Default(-1);

  if (CC != -1) {
    //64 bit register in Mips are following 32 bit definitions.
    if (isMips64())
      CC++;
    return CC;
  }

  return -1;
}

unsigned MipsAsmParser::getReg(int RC,int RegNo){
  return *(getContext().getRegisterInfo().getRegClass(RC).begin() + RegNo);
}

int MipsAsmParser::matchRegisterByNumber(unsigned RegNum,StringRef Mnemonic) {

  if (Mnemonic.lower() == "rdhwr") {
    //at the moment only hwreg29 is supported
    if (RegNum != 29)
      return -1;
    return Mips::HWR29;
  }

  if (RegNum > 31)
    return -1;

  return getReg(Mips::CPURegsRegClassID,RegNum);
}

int MipsAsmParser::tryParseRegister(StringRef Mnemonic) {
  const AsmToken &Tok = Parser.getTok();
  int RegNum = -1;

  if (Tok.is(AsmToken::Identifier)) {
    std::string lowerCase = Tok.getString().lower();
    RegNum = matchRegisterName(lowerCase);
  } else if (Tok.is(AsmToken::Integer))
      RegNum = matchRegisterByNumber(static_cast<unsigned> (Tok.getIntVal()),
                                     Mnemonic.lower());
  return RegNum;
}

bool MipsAsmParser::
  tryParseRegisterOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                          StringRef Mnemonic){

  SMLoc S = Parser.getTok().getLoc();
  int RegNo = -1;
  RegNo = tryParseRegister(Mnemonic);
  if (RegNo == -1)
    return true;

  Operands.push_back(MipsOperand::CreateReg(RegNo, S,
                     Parser.getTok().getLoc()));
  Parser.Lex(); // Eat register token.
  return false;
}

bool MipsAsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand*>&Operands,
                                 StringRef Mnemonic) {
  //Check if the current operand has a custom associated parser, if so, try to
  //custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);
  if (ResTy == MatchOperand_Success)
    return false;
  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_ParseFail)
    return true;

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Dollar: {
    //parse register
    SMLoc S = Parser.getTok().getLoc();
    Parser.Lex(); // Eat dollar token.
    //parse register operand
    if (!tryParseRegisterOperand(Operands,Mnemonic)) {
      if (getLexer().is(AsmToken::LParen)) {
        //check if it is indexed addressing operand
        Operands.push_back(MipsOperand::CreateToken("(", S));
        Parser.Lex(); //eat parenthesis
        if (getLexer().isNot(AsmToken::Dollar))
          return true;

        Parser.Lex(); //eat dollar
        if (tryParseRegisterOperand(Operands,Mnemonic))
          return true;

        if (!getLexer().is(AsmToken::RParen))
          return true;

        S = Parser.getTok().getLoc();
        Operands.push_back(MipsOperand::CreateToken(")", S));
        Parser.Lex();
      }
      return false;
    }
    //maybe it is a symbol reference
    StringRef Identifier;
    if (Parser.ParseIdentifier(Identifier))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer()-1);

    StringRef Id = StringRef("$" + Identifier.str());
    MCSymbol *Sym = getContext().GetOrCreateSymbol(Id);

    // Otherwise create a symbol ref.
    const MCExpr *Res = MCSymbolRefExpr::Create(Sym,
                                                MCSymbolRefExpr::VK_None,
                                                getContext());

    Operands.push_back(MipsOperand::CreateImm(Res, S, E));
    return false;
  }
  case AsmToken::Identifier:
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::String: {
     // quoted label names
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc();
    if (getParser().ParseExpression(IdVal))
      return true;
    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer()-1);
    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
    return false;
  }
  }//switch(getLexer().getKind())
  return true;
}

bool MipsAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                  SMLoc &EndLoc) {

  StartLoc = Parser.getTok().getLoc();
  RegNo = tryParseRegister("");
  EndLoc = Parser.getTok().getLoc();
  return (RegNo == (unsigned)-1);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMemOperand(
               SmallVectorImpl<MCParsedAsmOperand*>&Operands) {
  return MatchOperand_Success;
}

bool MipsAsmParser::
ParseInstruction(StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  //first operand is a instruction mnemonic
  Operands.push_back(MipsOperand::CreateToken(Name, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.EatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }

    while (getLexer().is(AsmToken::Comma) ) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.EatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.EatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::
ParseDirective(AsmToken DirectiveID) {
  return true;
}

extern "C" void LLVMInitializeMipsAsmParser() {
  RegisterMCAsmParser<MipsAsmParser> X(TheMipsTarget);
  RegisterMCAsmParser<MipsAsmParser> Y(TheMipselTarget);
  RegisterMCAsmParser<MipsAsmParser> A(TheMips64Target);
  RegisterMCAsmParser<MipsAsmParser> B(TheMips64elTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MipsGenAsmMatcher.inc"
