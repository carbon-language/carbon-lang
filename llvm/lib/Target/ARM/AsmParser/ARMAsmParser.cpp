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

class ARMAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;

private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }

  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

public:
  ARMAsmParser(const Target &T, MCAsmParser &_Parser)
    : TargetAsmParser(T), Parser(_Parser) {}

  virtual bool ParseInstruction(const StringRef &Name, MCInst &Inst);

  virtual bool ParseDirective(AsmToken DirectiveID);
};
  
} // end anonymous namespace

bool ARMAsmParser::ParseInstruction(const StringRef &Name, MCInst &Inst) {
  SMLoc Loc = getLexer().getTok().getLoc();
  Error(Loc, "ARMAsmParser::ParseInstruction currently unimplemented");
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
