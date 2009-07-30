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

static std::string FlattenVariants(const std::string &AsmString,
                                   unsigned Index) {
  StringRef Cur = AsmString;
  std::string Res = "";
  
  for (;;) {
    std::pair<StringRef, StringRef> Split = Cur.split('{');

    Res += Split.first;
    if (Split.second.empty())
      break;

    std::pair<StringRef, StringRef> Inner = Cur.split('}');
    StringRef Selection = Inner.first;
    for (unsigned i = 0; i != Index; ++i)
      Selection = Selection.split('|').second;
    Selection = Selection.split('|').first;

    Res += Selection;

    Cur = Inner.second;
  } 

  return Res;
}

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

  // Emit the function to match instructions.  
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);

  const std::map<std::string, CodeGenInstruction> &Instructions =
    Target.getInstructions();
  for (std::map<std::string, CodeGenInstruction>::const_iterator 
         it = Instructions.begin(), ie = Instructions.end(); it != ie; ++it) {
    const CodeGenInstruction &CGI = it->second;

    if (it->first != "SUB8rr")
      continue;

    /*
def SUB8rr  : I<0x28, MRMDestReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                "sub{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (sub GR8:$src1, GR8:$src2)),
                 (implicit EFLAGS)]>;
    */

    outs() << it->first << " "
           << FlattenVariants(CGI.AsmString, 0)
           << "\n";
  }
}
