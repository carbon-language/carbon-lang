//===- AsmWriterEmitter.cpp - Generate an assembly writer -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is emits an assembly printer for the current target.
// Note that this is currently fairly skeletal, but will grow over time.
//
//===----------------------------------------------------------------------===//

#include "AsmWriterEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <ostream>
using namespace llvm;

static bool isIdentChar(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z') ||
         (C >= '0' && C <= '9') ||
         C == '_';
}

namespace {
  struct AsmWriterOperand {
    enum { isLiteralTextOperand, isMachineInstrOperand } OperandType;

    /// Str - For isLiteralTextOperand, this IS the literal text.  For
    /// isMachineInstrOperand, this is the PrinterMethodName for the operand.
    std::string Str;

    /// MiOpNo - For isMachineInstrOperand, this is the operand number of the
    /// machine instruction.
    unsigned MIOpNo;
    
    /// MiModifier - For isMachineInstrOperand, this is the modifier string for
    /// an operand, specified with syntax like ${opname:modifier}.
    std::string MiModifier;

    AsmWriterOperand(const std::string &LitStr)
      : OperandType(isLiteralTextOperand), Str(LitStr) {}

    AsmWriterOperand(const std::string &Printer, unsigned OpNo, 
                     const std::string &Modifier) 
      : OperandType(isMachineInstrOperand), Str(Printer), MIOpNo(OpNo),
      MiModifier(Modifier) {}

    bool operator!=(const AsmWriterOperand &Other) const {
      if (OperandType != Other.OperandType || Str != Other.Str) return true;
      if (OperandType == isMachineInstrOperand)
        return MIOpNo != Other.MIOpNo || MiModifier != Other.MiModifier;
      return false;
    }
    bool operator==(const AsmWriterOperand &Other) const {
      return !operator!=(Other);
    }
    void EmitCode(std::ostream &OS) const;
  };

  struct AsmWriterInst {
    std::vector<AsmWriterOperand> Operands;
    const CodeGenInstruction *CGI;

    AsmWriterInst(const CodeGenInstruction &CGI, unsigned Variant);

    /// MatchesAllButOneOp - If this instruction is exactly identical to the
    /// specified instruction except for one differing operand, return the
    /// differing operand number.  Otherwise return ~0.
    unsigned MatchesAllButOneOp(const AsmWriterInst &Other) const;

  private:
    void AddLiteralString(const std::string &Str) {
      // If the last operand was already a literal text string, append this to
      // it, otherwise add a new operand.
      if (!Operands.empty() &&
          Operands.back().OperandType == AsmWriterOperand::isLiteralTextOperand)
        Operands.back().Str.append(Str);
      else
        Operands.push_back(AsmWriterOperand(Str));
    }
  };
}


void AsmWriterOperand::EmitCode(std::ostream &OS) const {
  if (OperandType == isLiteralTextOperand)
    OS << "O << \"" << Str << "\"; ";
  else {
    OS << Str << "(MI, " << MIOpNo;
    if (!MiModifier.empty())
      OS << ", \"" << MiModifier << '"';
    OS << "); ";
  }
}


/// ParseAsmString - Parse the specified Instruction's AsmString into this
/// AsmWriterInst.
///
AsmWriterInst::AsmWriterInst(const CodeGenInstruction &CGI, unsigned Variant) {
  this->CGI = &CGI;
  unsigned CurVariant = ~0U;  // ~0 if we are outside a {.|.|.} region, other #.

  // NOTE: Any extensions to this code need to be mirrored in the 
  // AsmPrinter::printInlineAsm code that executes as compile time (assuming
  // that inline asm strings should also get the new feature)!
  const std::string &AsmString = CGI.AsmString;
  std::string::size_type LastEmitted = 0;
  while (LastEmitted != AsmString.size()) {
    std::string::size_type DollarPos =
      AsmString.find_first_of("${|}", LastEmitted);
    if (DollarPos == std::string::npos) DollarPos = AsmString.size();

    // Emit a constant string fragment.
    if (DollarPos != LastEmitted) {
      // TODO: this should eventually handle escaping.
      if (CurVariant == Variant || CurVariant == ~0U)
        AddLiteralString(std::string(AsmString.begin()+LastEmitted,
                                     AsmString.begin()+DollarPos));
      LastEmitted = DollarPos;
    } else if (AsmString[DollarPos] == '{') {
      if (CurVariant != ~0U)
        throw "Nested variants found for instruction '" +
              CGI.TheDef->getName() + "'!";
      LastEmitted = DollarPos+1;
      CurVariant = 0;   // We are now inside of the variant!
    } else if (AsmString[DollarPos] == '|') {
      if (CurVariant == ~0U)
        throw "'|' character found outside of a variant in instruction '"
          + CGI.TheDef->getName() + "'!";
      ++CurVariant;
      ++LastEmitted;
    } else if (AsmString[DollarPos] == '}') {
      if (CurVariant == ~0U)
        throw "'}' character found outside of a variant in instruction '"
          + CGI.TheDef->getName() + "'!";
      ++LastEmitted;
      CurVariant = ~0U;
    } else if (DollarPos+1 != AsmString.size() &&
               AsmString[DollarPos+1] == '$') {
      if (CurVariant == Variant || CurVariant == ~0U) 
        AddLiteralString("$");  // "$$" -> $
      LastEmitted = DollarPos+2;
    } else {
      // Get the name of the variable.
      std::string::size_type VarEnd = DollarPos+1;

      // handle ${foo}bar as $foo by detecting whether the character following
      // the dollar sign is a curly brace.  If so, advance VarEnd and DollarPos
      // so the variable name does not contain the leading curly brace.
      bool hasCurlyBraces = false;
      if (VarEnd < AsmString.size() && '{' == AsmString[VarEnd]) {
        hasCurlyBraces = true;
        ++DollarPos;
        ++VarEnd;
      }

      while (VarEnd < AsmString.size() && isIdentChar(AsmString[VarEnd]))
        ++VarEnd;
      std::string VarName(AsmString.begin()+DollarPos+1,
                          AsmString.begin()+VarEnd);

      // Modifier - Support ${foo:modifier} syntax, where "modifier" is passed
      // into printOperand.
      std::string Modifier;
      
      // In order to avoid starting the next string at the terminating curly
      // brace, advance the end position past it if we found an opening curly
      // brace.
      if (hasCurlyBraces) {
        if (VarEnd >= AsmString.size())
          throw "Reached end of string before terminating curly brace in '"
                + CGI.TheDef->getName() + "'";
        
        // Look for a modifier string.
        if (AsmString[VarEnd] == ':') {
          ++VarEnd;
          if (VarEnd >= AsmString.size())
            throw "Reached end of string before terminating curly brace in '"
              + CGI.TheDef->getName() + "'";
          
          unsigned ModifierStart = VarEnd;
          while (VarEnd < AsmString.size() && isIdentChar(AsmString[VarEnd]))
            ++VarEnd;
          Modifier = std::string(AsmString.begin()+ModifierStart,
                                 AsmString.begin()+VarEnd);
          if (Modifier.empty())
            throw "Bad operand modifier name in '"+ CGI.TheDef->getName() + "'";
        }
        
        if (AsmString[VarEnd] != '}')
          throw "Variable name beginning with '{' did not end with '}' in '"
                + CGI.TheDef->getName() + "'";
        ++VarEnd;
      }
      if (VarName.empty())
        throw "Stray '$' in '" + CGI.TheDef->getName() +
              "' asm string, maybe you want $$?";

      unsigned OpNo = CGI.getOperandNamed(VarName);
      CodeGenInstruction::OperandInfo OpInfo = CGI.OperandList[OpNo];

      // If this is a two-address instruction and we are not accessing the
      // 0th operand, remove an operand.
      unsigned MIOp = OpInfo.MIOperandNo;
      if (CGI.isTwoAddress && MIOp != 0) {
        if (MIOp == 1)
          throw "Should refer to operand #0 instead of #1 for two-address"
            " instruction '" + CGI.TheDef->getName() + "'!";
        --MIOp;
      }

      if (CurVariant == Variant || CurVariant == ~0U) 
        Operands.push_back(AsmWriterOperand(OpInfo.PrinterMethodName, MIOp,
                                            Modifier));
      LastEmitted = VarEnd;
    }
  }

  AddLiteralString("\\n");
}

/// MatchesAllButOneOp - If this instruction is exactly identical to the
/// specified instruction except for one differing operand, return the differing
/// operand number.  If more than one operand mismatches, return ~1, otherwise
/// if the instructions are identical return ~0.
unsigned AsmWriterInst::MatchesAllButOneOp(const AsmWriterInst &Other)const{
  if (Operands.size() != Other.Operands.size()) return ~1;

  unsigned MismatchOperand = ~0U;
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    if (Operands[i] != Other.Operands[i])
      if (MismatchOperand != ~0U)  // Already have one mismatch?
        return ~1U;
      else
        MismatchOperand = i;
  }
  return MismatchOperand;
}

static void PrintCases(std::vector<std::pair<std::string,
                       AsmWriterOperand> > &OpsToPrint, std::ostream &O) {
  O << "    case " << OpsToPrint.back().first << ": ";
  AsmWriterOperand TheOp = OpsToPrint.back().second;
  OpsToPrint.pop_back();

  // Check to see if any other operands are identical in this list, and if so,
  // emit a case label for them.
  for (unsigned i = OpsToPrint.size(); i != 0; --i)
    if (OpsToPrint[i-1].second == TheOp) {
      O << "\n    case " << OpsToPrint[i-1].first << ": ";
      OpsToPrint.erase(OpsToPrint.begin()+i-1);
    }

  // Finally, emit the code.
  TheOp.EmitCode(O);
  O << "break;\n";
}


/// EmitInstructions - Emit the last instruction in the vector and any other
/// instructions that are suitably similar to it.
static void EmitInstructions(std::vector<AsmWriterInst> &Insts,
                             std::ostream &O) {
  AsmWriterInst FirstInst = Insts.back();
  Insts.pop_back();

  std::vector<AsmWriterInst> SimilarInsts;
  unsigned DifferingOperand = ~0;
  for (unsigned i = Insts.size(); i != 0; --i) {
    unsigned DiffOp = Insts[i-1].MatchesAllButOneOp(FirstInst);
    if (DiffOp != ~1U) {
      if (DifferingOperand == ~0U)  // First match!
        DifferingOperand = DiffOp;

      // If this differs in the same operand as the rest of the instructions in
      // this class, move it to the SimilarInsts list.
      if (DifferingOperand == DiffOp || DiffOp == ~0U) {
        SimilarInsts.push_back(Insts[i-1]);
        Insts.erase(Insts.begin()+i-1);
      }
    }
  }

  O << "  case " << FirstInst.CGI->Namespace << "::"
    << FirstInst.CGI->TheDef->getName() << ":\n";
  for (unsigned i = 0, e = SimilarInsts.size(); i != e; ++i)
    O << "  case " << SimilarInsts[i].CGI->Namespace << "::"
      << SimilarInsts[i].CGI->TheDef->getName() << ":\n";
  for (unsigned i = 0, e = FirstInst.Operands.size(); i != e; ++i) {
    if (i != DifferingOperand) {
      // If the operand is the same for all instructions, just print it.
      O << "    ";
      FirstInst.Operands[i].EmitCode(O);
    } else {
      // If this is the operand that varies between all of the instructions,
      // emit a switch for just this operand now.
      O << "    switch (MI->getOpcode()) {\n";
      std::vector<std::pair<std::string, AsmWriterOperand> > OpsToPrint;
      OpsToPrint.push_back(std::make_pair(FirstInst.CGI->Namespace + "::" +
                                          FirstInst.CGI->TheDef->getName(),
                                          FirstInst.Operands[i]));

      for (unsigned si = 0, e = SimilarInsts.size(); si != e; ++si) {
        AsmWriterInst &AWI = SimilarInsts[si];
        OpsToPrint.push_back(std::make_pair(AWI.CGI->Namespace+"::"+
                                            AWI.CGI->TheDef->getName(),
                                            AWI.Operands[i]));
      }
      std::reverse(OpsToPrint.begin(), OpsToPrint.end());
      while (!OpsToPrint.empty())
        PrintCases(OpsToPrint, O);
      O << "    }";
    }
    O << "\n";
  }

  O << "    break;\n";
}

void AsmWriterEmitter::run(std::ostream &O) {
  EmitSourceFileHeader("Assembly Writer Source Fragment", O);

  CodeGenTarget Target;
  Record *AsmWriter = Target.getAsmWriter();
  std::string ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  unsigned Variant = AsmWriter->getValueAsInt("Variant");

  O <<
  "/// printInstruction - This method is automatically generated by tablegen\n"
  "/// from the instruction set description.  This method returns true if the\n"
  "/// machine instruction was sufficiently described to print it, otherwise\n"
  "/// it returns false.\n"
    "bool " << Target.getName() << ClassName
            << "::printInstruction(const MachineInstr *MI) {\n";

  std::vector<AsmWriterInst> Instructions;

  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I)
    if (!I->second.AsmString.empty())
      Instructions.push_back(AsmWriterInst(I->second, Variant));

  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);
  
  // Compute the CodeGenInstruction -> AsmWriterInst mapping.  Note that not
  // all machine instructions are necessarily being printed, so there may be
  // target instructions not in this map.
  std::map<const CodeGenInstruction*, AsmWriterInst*> CGIAWIMap;
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i)
    CGIAWIMap.insert(std::make_pair(Instructions[i].CGI, &Instructions[i]));

  // Build an aggregate string, and build a table of offsets into it.
  std::map<std::string, unsigned> StringOffset;
  std::string AggregateString;
  AggregateString += '\0';
  
  O << "  static unsigned short OpStrIdxs[] = {\n";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    AsmWriterInst *AWI = CGIAWIMap[NumberedInstructions[i]];
    unsigned Idx;
    if (AWI == 0 || AWI->Operands[0].Str.empty()) {
      // Something not handled by the asmwriter printer.
      Idx = 0;
    } else {
      unsigned &Entry = StringOffset[AWI->Operands[0].Str];
      if (Entry == 0) {
        // Add the string to the aggregate if this is the first time found.
        Entry = AggregateString.size();
        std::string Str = AWI->Operands[0].Str;
        UnescapeString(Str);
        AggregateString += Str;
        AggregateString += '\0';
      }
      Idx = Entry;
      assert(Entry < 65536 && "Must not use unsigned short for table idx!");

      // Nuke the string from the operand list.  It is now handled!
      AWI->Operands.erase(AWI->Operands.begin());
    }
    O << "    " << Idx << ",\t// " << NumberedInstructions[i]->TheDef->getName()
      << "\n";
  }
  O << "  };\n\n";
  
  // Emit the string itself.
  O << "  const char *AsmStrs = \n    \"";
  unsigned CharsPrinted = 0;
  EscapeString(AggregateString);
  for (unsigned i = 0, e = AggregateString.size(); i != e; ++i) {
    if (CharsPrinted > 70) {
      O << "\"\n    \"";
      CharsPrinted = 0;
    }
    O << AggregateString[i];
    ++CharsPrinted;
    
    // Print escape sequences all together.
    if (AggregateString[i] == '\\') {
      assert(i+1 < AggregateString.size() && "Incomplete escape sequence!");
      if (isdigit(AggregateString[i+1])) {
        assert(isdigit(AggregateString[i+2]) && isdigit(AggregateString[i+3]) &&
               "Expected 3 digit octal escape!");
        O << AggregateString[++i];
        O << AggregateString[++i];
        O << AggregateString[++i];
        CharsPrinted += 3;
      } else {
        O << AggregateString[++i];
        ++CharsPrinted;
      }
    }
  }
  O << "\";\n\n";

  O << "  // Emit the opcode for the instruction.\n"
    << "  O << AsmStrs+OpStrIdxs[MI->getOpcode()];\n\n";

  // Because this is a vector we want to emit from the end.  Reverse all of the
  // elements in the vector.
  std::reverse(Instructions.begin(), Instructions.end());

  // Find the opcode # of inline asm
  O << "  switch (MI->getOpcode()) {\n"
       "  default: return false;\n"
       "  case " << NumberedInstructions.back()->Namespace
    << "::INLINEASM: printInlineAsm(MI); break;\n";

  while (!Instructions.empty())
    EmitInstructions(Instructions, O);

  O << "  }\n"
       "  return true;\n"
       "}\n";
}
