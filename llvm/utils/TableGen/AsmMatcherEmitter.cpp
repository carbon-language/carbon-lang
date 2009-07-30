//===- AsmMatcherEmitter.cpp - Generate an assembly matcher ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a target specifier matcher for converting parsed
// assembly operands in the MCInst structures.
//
//===----------------------------------------------------------------------===//

#include "AsmMatcherEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
using namespace llvm;

void AsmMatcherEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target;
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();
  Record *AsmParser = Target.getAsmParser();
  std::string ClassName = AsmParser->getValueAsString("AsmParserClassName");

  std::string Namespace = Registers[0].TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Assembly Matcher Source Fragment", OS);

  // Emit the function to match a register name to number.

  OS << "bool " << Target.getName() << ClassName
     << "::MatchRegisterName(const StringRef &Name, unsigned &RegNo) {\n";

  // FIXME: TableGen should have a fast string matcher generator.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister &Reg = Registers[i];
    if (Reg.TheDef->getValueAsString("AsmName").empty())
      continue;

    OS << "  if (Name == \"" 
       << Reg.TheDef->getValueAsString("AsmName") << "\")\n"
       << "    return RegNo=" << i + 1 << ", false;\n";
  }
  OS << "  return true;\n";
  OS << "}\n";
}
