//===-- AMDGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class AMDGPUAsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;


  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "AMDGPUGenAsmMatcher.inc"

  /// }

public:
  AMDGPUAsmParser(MCSubtargetInfo &STI, MCAsmParser &Parser,
                  const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(), STI(STI), Parser(Parser) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseDirective(AsmToken DirectiveID) override;
  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Mnemonic);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool parseCnt(int64_t &IntVal);
  OperandMatchResultTy parseSWaitCntOps(OperandVector &Operands);
};

class AMDGPUOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate
  } Kind;

public:
  AMDGPUOperand(enum KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    int64_t Val;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
  };

  void addImmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateImm(getImm()));
  }
  void addRegOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("addRegOperands");
  }
  StringRef getToken() const {
    return StringRef(Tok.Data, Tok.Length);
  }
  bool isToken() const override {
    return Kind == Token;
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  int64_t getImm() const {
    return Imm.Val;
  }

  bool isReg() const override {
    return false;
  }

  unsigned getReg() const override {
    return 0;
  }

  bool isMem() const override {
    return false;
  }

  SMLoc getStartLoc() const override {
    return SMLoc();
  }

  SMLoc getEndLoc() const override {
    return SMLoc();
  }

  void print(raw_ostream &OS) const override { }

  static std::unique_ptr<AMDGPUOperand> CreateImm(int64_t Val) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Immediate);
    Op->Imm.Val = Val;
    return Op;
  }

  static std::unique_ptr<AMDGPUOperand> CreateToken(StringRef Str, SMLoc Loc) {
    auto Res = llvm::make_unique<AMDGPUOperand>(Token);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    return Res;
  }

  bool isSWaitCnt() const;
};

}

bool AMDGPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  return true;
}


bool AMDGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst, STI);
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction use requires an option to be enabled");
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_InvalidOperand: {
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

    }
    return Error(IDLoc, "invalid operand for instruction");
  }
  }
  llvm_unreachable("Implement any new match types added!");
}

bool AMDGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  return true;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {

  // Try to parse with a custom parser
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If we successfully parsed the operand or if there as an error parsing,
  // we are done.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail)
    return ResTy;

  switch(getLexer().getKind()) {
    case AsmToken::Integer: {
      int64_t IntVal;
      if (getParser().parseAbsoluteExpression(IntVal))
        return MatchOperand_ParseFail;
      Operands.push_back(AMDGPUOperand::CreateImm(IntVal));
      return MatchOperand_Success;
    }
    default:
      return MatchOperand_NoMatch;
  }
}

bool AMDGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name,
                                       SMLoc NameLoc, OperandVector &Operands) {
  // Add the instruction mnemonic
  Operands.push_back(AMDGPUOperand::CreateToken(Name, NameLoc));

  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  AMDGPUAsmParser::OperandMatchResultTy Res = parseOperand(Operands, Name);
  switch (Res) {
    case MatchOperand_Success: return false;
    case MatchOperand_ParseFail: return Error(NameLoc,
                                              "Failed parsing operand");
    case MatchOperand_NoMatch: return Error(NameLoc, "Not a valid operand");
  }
  return true;
}

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

bool AMDGPUAsmParser::parseCnt(int64_t &IntVal) {
  StringRef CntName = Parser.getTok().getString();
  int64_t CntVal;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::LParen))
    return true;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Integer))
    return true;

  if (getParser().parseAbsoluteExpression(CntVal))
    return true;

  if (getLexer().isNot(AsmToken::RParen))
    return true;

  Parser.Lex();
  if (getLexer().is(AsmToken::Amp) || getLexer().is(AsmToken::Comma))
    Parser.Lex();

  int CntShift;
  int CntMask;

  if (CntName == "vmcnt") {
    CntMask = 0xf;
    CntShift = 0;
  } else if (CntName == "expcnt") {
    CntMask = 0x7;
    CntShift = 4;
  } else if (CntName == "lgkmcnt") {
    CntMask = 0x7;
    CntShift = 8;
  } else {
    return true;
  }

  IntVal &= ~(CntMask << CntShift);
  IntVal |= (CntVal << CntShift);
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSWaitCntOps(OperandVector &Operands) {
  // Disable all counters by default.
  // vmcnt   [3:0]
  // expcnt  [6:4]
  // lgkmcnt [10:8]
  int64_t CntVal = 0x77f;

  switch(getLexer().getKind()) {
    default: return MatchOperand_ParseFail;
    case AsmToken::Integer:
      // The operand can be an integer value.
      if (getParser().parseAbsoluteExpression(CntVal))
        return MatchOperand_ParseFail;
      break;

    case AsmToken::Identifier:
      do {
        if (parseCnt(CntVal))
          return MatchOperand_ParseFail;
      } while(getLexer().isNot(AsmToken::EndOfStatement));
      break;
  }
  Operands.push_back(AMDGPUOperand::CreateImm(CntVal));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isSWaitCnt() const {
  return isImm();
}

/// Force static initialization.
extern "C" void LLVMInitializeR600AsmParser() {
  RegisterMCAsmParser<AMDGPUAsmParser> A(TheAMDGPUTarget);
  RegisterMCAsmParser<AMDGPUAsmParser> B(TheGCNTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "AMDGPUGenAsmMatcher.inc"

