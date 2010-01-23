//===-- X86AsmLexer.cpp - Tokenize X86 assembly to AsmTokens --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmLexer.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "X86.h"

using namespace llvm;

namespace {
  
class X86AsmLexer : public TargetAsmLexer {
  const MCAsmInfo &AsmInfo;
protected:
  AsmToken LexToken();
public:
  X86AsmLexer(const Target &T, const MCAsmInfo &MAI)
    : TargetAsmLexer(T), AsmInfo(MAI) {
  }
};

}

AsmToken X86AsmLexer::LexToken() {
  return AsmToken(AsmToken::Error, "", 0);
}

extern "C" void LLVMInitializeX86AsmLexer() {
  RegisterAsmLexer<X86AsmLexer> X(TheX86_32Target);
  RegisterAsmLexer<X86AsmLexer> Y(TheX86_64Target);
}

//#define REGISTERS_ONLY
//#include "../X86GenAsmMatcher.inc"
//#undef REGISTERS_ONLY
