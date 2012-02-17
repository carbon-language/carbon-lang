//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {
class MipsAsmParser : public MCTargetAsmParser {
  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirective(AsmToken DirectiveID);

public:
  MipsAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser() {
  }

};
}

bool MipsAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  return true;
}

bool MipsAsmParser::
ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  return true;
}

bool MipsAsmParser::
ParseInstruction(StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  return true;
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
