//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/MC/MCAsmParser.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

namespace {

class X86ATTAsmParser : public TargetAsmParser {
 public:
  explicit X86ATTAsmParser(const Target &);

  virtual bool ParseInstruction(MCAsmParser &AP, const char *Name, 
                                MCInst &Inst);
};

}

X86ATTAsmParser::X86ATTAsmParser(const Target &T) 
  : TargetAsmParser(T)
{
}

bool X86ATTAsmParser::ParseInstruction(MCAsmParser &AP, const char *Name, 
                                       MCInst &Inst) {
  return true;
}

namespace {
  TargetAsmParser *createAsmParser(const Target &T) {
    return new X86ATTAsmParser(T);
  }
}

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  TargetRegistry::RegisterAsmParser(TheX86_32Target, &createAsmParser);
  TargetRegistry::RegisterAsmParser(TheX86_64Target, &createAsmParser);
}
