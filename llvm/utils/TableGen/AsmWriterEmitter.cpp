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

    /// OpVT - For isMachineInstrOperand, this is the value type for the
    /// operand.
    MVT::ValueType OpVT;

    AsmWriterOperand(const std::string &LitStr)
      : OperandType(isLiteralTextOperand),  Str(LitStr) {}

    AsmWriterOperand(const std::string &Printer, unsigned OpNo,
                     MVT::ValueType VT) : OperandType(isMachineInstrOperand),
                                          Str(Printer), MIOpNo(OpNo), OpVT(VT){}

    void EmitCode(std::ostream &OS) const;
  };

  struct AsmWriterInst {
    std::vector<AsmWriterOperand> Operands;
    
    void ParseAsmString(const CodeGenInstruction &CGI, unsigned Variant);
    void EmitCode(std::ostream &OS) const {
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        Operands[i].EmitCode(OS);
    }
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
  else
    OS << Str << "(MI, " << MIOpNo << ", MVT::" << getName(OpVT) << "); ";
}


/// ParseAsmString - Parse the specified Instruction's AsmString into this
/// AsmWriterInst.
///
void AsmWriterInst::ParseAsmString(const CodeGenInstruction &CGI,
                                   unsigned Variant) {
  bool inVariant = false;  // True if we are inside a {.|.|.} region.

  const std::string &AsmString = CGI.AsmString;
  std::string::size_type LastEmitted = 0;
  while (LastEmitted != AsmString.size()) {
    std::string::size_type DollarPos =
      AsmString.find_first_of("${|}", LastEmitted);
    if (DollarPos == std::string::npos) DollarPos = AsmString.size();

    // Emit a constant string fragment.
    if (DollarPos != LastEmitted) {
      // TODO: this should eventually handle escaping.
      AddLiteralString(std::string(AsmString.begin()+LastEmitted,
                                   AsmString.begin()+DollarPos));
      LastEmitted = DollarPos;
    } else if (AsmString[DollarPos] == '{') {
      if (inVariant)
        throw "Nested variants found for instruction '" + CGI.Name + "'!";
      LastEmitted = DollarPos+1;
      inVariant = true;   // We are now inside of the variant!
      for (unsigned i = 0; i != Variant; ++i) {
        // Skip over all of the text for an irrelevant variant here.  The
        // next variant starts at |, or there may not be text for this
        // variant if we see a }.
        std::string::size_type NP =
          AsmString.find_first_of("|}", LastEmitted);
        if (NP == std::string::npos)
          throw "Incomplete variant for instruction '" + CGI.Name + "'!";
        LastEmitted = NP+1;
        if (AsmString[NP] == '}') {
          inVariant = false;        // No text for this variant.
          break;
        }
      }
    } else if (AsmString[DollarPos] == '|') {
      if (!inVariant)
        throw "'|' character found outside of a variant in instruction '"
          + CGI.Name + "'!";
      // Move to the end of variant list.
      std::string::size_type NP = AsmString.find('}', LastEmitted);
      if (NP == std::string::npos)
        throw "Incomplete variant for instruction '" + CGI.Name + "'!";
      LastEmitted = NP+1;
      inVariant = false;
    } else if (AsmString[DollarPos] == '}') {
      if (!inVariant)
        throw "'}' character found outside of a variant in instruction '"
          + CGI.Name + "'!";
      LastEmitted = DollarPos+1;
      inVariant = false;
    } else if (DollarPos+1 != AsmString.size() &&
               AsmString[DollarPos+1] == '$') {
      AddLiteralString("$");  // "$$" -> $
      LastEmitted = DollarPos+2;
    } else {
      // Get the name of the variable.
      // TODO: should eventually handle ${foo}bar as $foo
      std::string::size_type VarEnd = DollarPos+1;
      while (VarEnd < AsmString.size() && isIdentChar(AsmString[VarEnd]))
        ++VarEnd;
      std::string VarName(AsmString.begin()+DollarPos+1,
                          AsmString.begin()+VarEnd);
      if (VarName.empty())
        throw "Stray '$' in '" + CGI.Name + "' asm string, maybe you want $$?";

      unsigned OpNo = CGI.getOperandNamed(VarName);

      // If this is a two-address instruction and we are not accessing the
      // 0th operand, remove an operand.
      unsigned MIOp = CGI.OperandList[OpNo].MIOperandNo;
      if (CGI.isTwoAddress && MIOp != 0) {
        if (MIOp == 1)
          throw "Should refer to operand #0 instead of #1 for two-address"
            " instruction '" + CGI.Name + "'!";
        --MIOp;
      }

      Operands.push_back(AsmWriterOperand(CGI.OperandList[OpNo].PrinterMethodName,
                                          MIOp, CGI.OperandList[OpNo].Ty));
      LastEmitted = VarEnd;
    }
  }

  AddLiteralString("\\n");
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
  O << "  switch (MI->getOpcode()) {\n"
       "  default: return false;\n";

  std::string Namespace = Target.inst_begin()->second.Namespace;

  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I)
    if (!I->second.AsmString.empty()) {
      O << "  case " << Namespace << "::" << I->first << ": ";

      AsmWriterInst AWI;
      AWI.ParseAsmString(I->second, Variant);
      AWI.EmitCode(O);
      O << " break;\n";
    }

  O << "  }\n"
       "  return true;\n"
       "}\n";
}
