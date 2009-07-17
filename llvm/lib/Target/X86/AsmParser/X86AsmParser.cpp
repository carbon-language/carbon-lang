//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

namespace {

class X86ATTAsmParser : public TargetAsmParser {
 public:
  explicit X86ATTAsmParser(const Target &);
};

}

X86ATTAsmParser::X86ATTAsmParser(const Target &T) 
  : TargetAsmParser(T)
{
}

namespace {
  TargetAsmParser *createAsmParser(const Target &T) {
    return new X86ATTAsmParser(T);
  }
}

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  extern Target TheX86_32Target;
  TargetRegistry::RegisterAsmParser(TheX86_32Target, &createAsmParser);
  extern Target TheX86_64Target;
  TargetRegistry::RegisterAsmParser(TheX86_64Target, &createAsmParser);
}
