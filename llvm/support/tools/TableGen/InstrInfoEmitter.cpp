//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrInfoEmitter.h"
#include "Record.h"

// runEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::runEnums(std::ostream &OS) {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  if (Insts.size() == 0)
    throw std::string("No 'Instruction' subclasses defined!");

  std::string Namespace = Insts[0]->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Instruction Enum Values", OS);

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";

  // We must emit the PHI opcode first...
  Record *Target = getTarget(Records);
  Record *InstrInfo = Target->getValueAsDef("InstructionSet");
  Record *PHI = InstrInfo->getValueAsDef("PHIInst");

  OS << "    " << PHI->getName() << ", \t// 0 (fixed for all targets)\n";
  
  // Print out the rest of the instructions now...
  for (unsigned i = 0, e = Insts.size(); i != e; ++i)
    if (Insts[i] != PHI)
      OS << "    " << Insts[i]->getName() << ", \t// " << i+1 << "\n";
  
  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
}

void InstrInfoEmitter::printDefList(ListInit *LI, const std::string &Name,
                                    std::ostream &OS) const {
  OS << "static const unsigned " << Name << "[] = { ";
  for (unsigned j = 0, e = LI->getSize(); j != e; ++j)
    if (DefInit *DI = dynamic_cast<DefInit*>(LI->getElement(j)))
      OS << getQualifiedName(DI->getDef()) << ", ";
    else
      throw "Illegal value in '" + Name + "' list!";
  OS << "0 };\n";
}


// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Target Instruction Descriptors", OS);
  Record *Target = getTarget(Records);
  const std::string &TargetName = Target->getName();
  Record *InstrInfo = Target->getValueAsDef("InstructionSet");
  Record *PHI = InstrInfo->getValueAsDef("PHIInst");

  std::vector<Record*> Instructions =
    Records.getAllDerivedDefinitions("Instruction");
  
  // Emit all of the instruction's implicit uses and defs...
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i) {
    Record *Inst = Instructions[i];
    ListInit *LI = Inst->getValueAsListInit("Uses");
    if (LI->getSize()) printDefList(LI, Inst->getName()+"ImpUses", OS);
    LI = Inst->getValueAsListInit("Defs");
    if (LI->getSize()) printDefList(LI, Inst->getName()+"ImpDefs", OS);
  }

  OS << "\nstatic const TargetInstrDescriptor " << TargetName
     << "Insts[] = {\n";
  emitRecord(PHI, 0, InstrInfo, OS);

  for (unsigned i = 0, e = Instructions.size(); i != e; ++i)
    if (Instructions[i] != PHI)
      emitRecord(Instructions[i], i+1, InstrInfo, OS);
  OS << "};\n";
}

void InstrInfoEmitter::emitRecord(Record *R, unsigned Num, Record *InstrInfo,
                                  std::ostream &OS) {
  OS << "  { \"" << R->getValueAsString("Name")
     << "\",\t-1, -1, 0, false, 0, 0, 0, 0";

  // Emit all of the target indepedent flags...
  if (R->getValueAsBit("isReturn"))     OS << "|M_RET_FLAG";
  if (R->getValueAsBit("isBranch"))     OS << "|M_BRANCH_FLAG";
  if (R->getValueAsBit("isCall"  ))     OS << "|M_CALL_FLAG";
  if (R->getValueAsBit("isTwoAddress")) OS << "|M_2_ADDR_FLAG";
  if (R->getValueAsBit("isTerminator")) OS << "|M_TERMINATOR_FLAG";
  OS << ", 0";

  // Emit all of the target-specific flags...
  ListInit *LI    = InstrInfo->getValueAsListInit("TSFlagsFields");
  ListInit *Shift = InstrInfo->getValueAsListInit("TSFlagsShifts");
  if (LI->getSize() != Shift->getSize())
    throw "Lengths of " + InstrInfo->getName() +
          ":(TargetInfoFields, TargetInfoPositions) must be equal!";

  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    emitShiftedValue(R, dynamic_cast<StringInit*>(LI->getElement(i)),
                     dynamic_cast<IntInit*>(Shift->getElement(i)), OS);

  OS << ", ";

  // Emit the implicit uses and defs lists...
  LI = R->getValueAsListInit("Uses");
  if (!LI->getSize())
    OS << "0, ";
  else 
    OS << R->getName() << "ImpUses, ";

  LI = R->getValueAsListInit("Defs");
  if (!LI->getSize())
    OS << "0 ";
  else 
    OS << R->getName() << "ImpDefs ";

  OS << " },  // Inst #" << Num << " = " << R->getName() << "\n";
}

void InstrInfoEmitter::emitShiftedValue(Record *R, StringInit *Val,
                                        IntInit *ShiftInt, std::ostream &OS) {
  if (Val == 0 || ShiftInt == 0)
    throw std::string("Illegal value or shift amount in TargetInfo*!");
  RecordVal *RV = R->getValue(Val->getValue());
  int Shift = ShiftInt->getValue();

  if (RV == 0 || RV->getValue() == 0)
    throw R->getName() + " doesn't have a field named '" + Val->getValue()+"'!";

  Init *Value = RV->getValue();
  if (BitInit *BI = dynamic_cast<BitInit*>(Value)) {
    if (BI->getValue()) OS << "|(1<<" << Shift << ")";
    return;
  } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Value)) {
    // Convert the Bits to an integer to print...
    Init *I = BI->convertInitializerTo(new IntRecTy());
    if (I)
      if (IntInit *II = dynamic_cast<IntInit*>(I)) {
        if (II->getValue())
          OS << "|(" << II->getValue() << "<<" << Shift << ")";
        return;
      }

  } else if (IntInit *II = dynamic_cast<IntInit*>(Value)) {
    if (II->getValue()) OS << "|(" << II->getValue() << "<<" << Shift << ")";
    return;
  }

  std::cerr << "Unhandled initializer: " << *Val << "\n";
  throw "In record '" + R->getName() + "' for TSFlag emission.";
}
