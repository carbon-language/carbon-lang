//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrInfoEmitter.h"
#include "Record.h"

static void EmitSourceHeader(const std::string &Desc, std::ostream &o) {
  o << "//===- TableGen'erated file -------------------------------------*-"
       " C++ -*-===//\n//\n// " << Desc << "\n//\n// Automatically generate"
       "d file, do not edit!\n//\n//===------------------------------------"
       "----------------------------------===//\n\n";
}

static std::string getQualifiedName(Record *R) {
  std::string Namespace = R->getValueAsString("Namespace");
  if (Namespace.empty()) return R->getName();
  return Namespace + "::" + R->getName();
}

static Record *getTarget(RecordKeeper &RC) {
  std::vector<Record*> Targets = RC.getAllDerivedDefinitions("Target");

  if (Targets.size() != 1)
    throw std::string("ERROR: Multiple subclasses of Target defined!");
  return Targets[0];
}

// runEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::runEnums(std::ostream &OS) {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  if (Insts.size() == 0)
    throw std::string("No 'Instruction' subclasses defined!");

  std::string Namespace = Insts[0]->getValueAsString("Namespace");

  EmitSourceHeader("Target Instruction Enum Values", OS);

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";

  // We must emit the PHI and NOOP opcodes first...
  Record *Target = getTarget(Records);
  Record *InstrInfo = Target->getValueAsDef("InstructionSet");

  Record *PHI = InstrInfo->getValueAsDef("PHIInst");
  Record *NOOP = InstrInfo->getValueAsDef("NOOPInst");

  OS << "    " << PHI->getName() << ", \t// 0 (fixed for all targets)\n"
     << "    " << NOOP->getName() << ", \t// 1 (fixed for all targets)\n";
  
  // Print out the rest of the instructions now...
  for (unsigned i = 0, e = Insts.size(); i != e; ++i)
    if (Insts[i] != PHI && Insts[i] != NOOP)
      OS << "    " << Insts[i]->getName() << ", \t// " << i+2 << "\n";
  
  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
}
