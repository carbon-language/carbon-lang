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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <list>
using namespace llvm;

/// FlattenVariants - Flatten an .td file assembly string by selecting the
/// variant at index \arg N.
static std::string FlattenVariants(const std::string &AsmString,
                                   unsigned N) {
  StringRef Cur = AsmString;
  std::string Res = "";
  
  for (;;) {
    // Add the prefix until the next '{', and split out the contents in the
    // braces.
    std::pair<StringRef, StringRef> Inner, Split = Cur.split('{');

    Res += Split.first;
    if (Split.second.empty())
      break;

    Inner = Split.second.split('}');

    // Select the Nth variant (or empty).
    StringRef Selection = Inner.first;
    for (unsigned i = 0; i != N; ++i)
      Selection = Selection.split('|').second;
    Res += Selection.split('|').first;

    Cur = Inner.second;
  } 

  return Res;
}

/// TokenizeAsmString - Tokenize a simplified assembly string.
static void TokenizeAsmString(const std::string &AsmString, 
                              SmallVectorImpl<StringRef> &Tokens) {
  unsigned Prev = 0;
  bool InTok = true;
  for (unsigned i = 0, e = AsmString.size(); i != e; ++i) {
    switch (AsmString[i]) {
    case '*':
    case '!':
    case ' ':
    case '\t':
    case ',':
      if (InTok) {
        Tokens.push_back(StringRef(&AsmString[Prev], i - Prev));
        InTok = false;
      }
      if (AsmString[i] == '*' || AsmString[i] == '!')
        Tokens.push_back(StringRef(&AsmString[i], 1));
      Prev = i + 1;
      break;

    default:
      InTok = true;
    }
  }
  if (InTok && Prev != AsmString.size())
    Tokens.push_back(StringRef(&AsmString[Prev], AsmString.size() - Prev));
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

  std::list<std::string> MatchFns;

  OS << "\n";
  const std::map<std::string, CodeGenInstruction> &Instructions =
    Target.getInstructions();
  for (std::map<std::string, CodeGenInstruction>::const_iterator 
         it = Instructions.begin(), ie = Instructions.end(); it != ie; ++it) {
    const CodeGenInstruction &CGI = it->second;

    // Ignore psuedo ops.
    //
    // FIXME: This is a hack.
    if (const RecordVal *Form = CGI.TheDef->getValue("Form"))
      if (Form->getValue()->getAsString() == "Pseudo")
        continue;

    // Ignore instructions with no .s string.
    //
    // FIXME: What are these?
    if (CGI.AsmString.empty())
      continue;

    // FIXME: Hack; ignore "lock".
    if (StringRef(CGI.AsmString).startswith("lock"))
      continue;

    // FIXME: Hack.
#if 0
    if (1 && it->first != "SUB8mr")
      continue;
#endif

    std::string Flattened = FlattenVariants(CGI.AsmString, 0);
    SmallVector<StringRef, 8> Tokens;

    TokenizeAsmString(Flattened, Tokens);

    DEBUG({
        outs() << it->first << " -- flattened:\"" 
               << Flattened << "\", tokens:[";
        for (unsigned i = 0, e = Tokens.size(); i != e; ++i) {
          outs() << Tokens[i];
          if (i + 1 != e)
            outs() << ", ";
        }
        outs() << "]\n";

        for (unsigned i = 0, e = CGI.OperandList.size(); i != e; ++i) {
          const CodeGenInstruction::OperandInfo &OI = CGI.OperandList[i];
          outs() << "  op[" << i << "] = " << OI.Name
                 << " " << OI.Rec->getName()
                 << " (" << OI.MIOperandNo << ", " << OI.MINumOperands << ")\n";
        }
      });

    // FIXME: Ignore non-literal tokens.
    if (std::find(Tokens[0].begin(), Tokens[0].end(), '$') != Tokens[0].end())
      continue;

    std::string FnName = "Match_" + Target.getName() + "_Inst_" + it->first;
    MatchFns.push_back(FnName);

    OS << "static bool " << FnName
       << "(const StringRef &Name,"
       << " SmallVectorImpl<X86Operand> &Operands,"
       << " MCInst &Inst) {\n\n";

    OS << "  // Match name.\n";
    OS << "  if (Name != \"" << Tokens[0] << "\")\n";
    OS << "    return true;\n\n";
    
    OS << "  // Match number of operands.\n";
    OS << "  if (Operands.size() != " << Tokens.size() - 1 << ")\n";
    OS << "    return true;\n\n";

    // Compute the total number of MCOperands.
    //
    // FIXME: Isn't this somewhere else?
    unsigned NumMIOperands = 0;
    for (unsigned i = 0, e = CGI.OperandList.size(); i != e; ++i) {
      const CodeGenInstruction::OperandInfo &OI = CGI.OperandList[i];
      NumMIOperands = std::max(NumMIOperands, 
                               OI.MIOperandNo + OI.MINumOperands);
    }

    std::set<unsigned> MatchedOperands;
    // This the list of operands we need to fill in.
    if (NumMIOperands)
      OS << "  MCOperand Ops[" << NumMIOperands << "];\n\n";

    unsigned ParsedOpIdx = 0;
    for (unsigned i = 1, e = Tokens.size(); i < e; ++i) {
      // FIXME: Can only match simple operands.
      if (Tokens[i][0] != '$') {
        OS << "  // FIXME: unable to match token: '" << Tokens[i] << "'!\n";
        OS << "  return true;\n\n";
        continue;
      }

      // Map this token to an operand. FIXME: Move elsewhere.

      unsigned Idx;
      try {
        Idx = CGI.getOperandNamed(Tokens[i].substr(1));
      } catch(...) {
        OS << "  // FIXME: unable to find operand: '" << Tokens[i] << "'!\n";
        OS << "  return true;\n\n";
        continue;
      }

      // FIXME: Each match routine should always end up filling the same number
      // of operands, we should just check that the number matches what the
      // match routine expects here instead of passing it. We can do this once
      // we start generating the class match functions.
      const CodeGenInstruction::OperandInfo &OI = CGI.OperandList[Idx];

      // Track that we have matched these operands.
      //
      // FIXME: Verify that we don't parse something to the same operand twice.
      for (unsigned j = 0; j != OI.MINumOperands; ++j)
        MatchedOperands.insert(OI.MIOperandNo + j);

      OS << "  // Match '" << Tokens[i] << "' (parsed operand " << ParsedOpIdx 
         << ") to machine operands [" << OI.MIOperandNo << ", " 
         << OI.MIOperandNo + OI.MINumOperands << ").\n";
      OS << "  if (Match_" << Target.getName() 
         << "_Op_" << OI.Rec->getName()  << "("
         << "Operands[" << ParsedOpIdx << "], "
         << "&Ops[" << OI.MIOperandNo << "], " 
         << OI.MINumOperands << "))\n";
      OS << "    return true;\n\n";

      ++ParsedOpIdx;
    }

    // Generate code to construct the MCInst.

    OS << "  // Construct MCInst.\n";
    OS << "  Inst.setOpcode(" << Target.getName() << "::" 
       << it->first << ");\n";
    for (unsigned i = 0, e = NumMIOperands; i != e; ++i) {
      // FIXME: Oops! Ignore this for now, the instruction should print ok. If
      // we need to evaluate the constraints.
      if (!MatchedOperands.count(i)) {
        OS << "\n";
        OS << "  // FIXME: Nothing matched Ops[" << i << "]!\n";
        OS << "  Ops[" << i << "] = MCOperand::CreateReg(0);\n";
        OS << "\n";
      }

      OS << "  Inst.addOperand(Ops[" << i << "]);\n";
    }
    OS << "\n";
    OS << "  return false;\n";
    OS << "}\n\n";
  }

  // Generate the top level match function.

  OS << "bool " << Target.getName() << ClassName
     << "::MatchInstruction(const StringRef &Name, "
     << "SmallVectorImpl<" << Target.getName() << "Operand> &Operands, "
     << "MCInst &Inst) {\n";
  for (std::list<std::string>::iterator it = MatchFns.begin(), 
         ie = MatchFns.end(); it != ie; ++it) {
    OS << "  if (!" << *it << "(Name, Operands, Inst))\n";
    OS << "    return false;\n\n";
  }

  OS << "  return true;\n";
  OS << "}\n\n";
}
