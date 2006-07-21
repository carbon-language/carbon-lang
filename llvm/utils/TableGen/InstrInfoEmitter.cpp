//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrInfoEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include <algorithm>
using namespace llvm;

// runEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::runEnums(std::ostream &OS) {
  EmitSourceFileHeader("Target Instruction Enum Values", OS);
  OS << "namespace llvm {\n\n";

  CodeGenTarget Target;

  // We must emit the PHI opcode first...
  std::string Namespace;
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(), 
       E = Target.inst_end(); II != E; ++II) {
    if (II->second.Namespace != "TargetInstrInfo") {
      Namespace = II->second.Namespace;
      break;
    }
  }
  
  if (Namespace.empty()) {
    std::cerr << "No instructions defined!\n";
    exit(1);
  }

  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);

  OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    OS << "    " << NumberedInstructions[i]->TheDef->getName()
       << "\t= " << i << ",\n";
  }
  OS << "    INSTRUCTION_LIST_END = " << NumberedInstructions.size() << "\n";
  OS << "  };\n}\n";
  OS << "} // End llvm namespace \n";
}

void InstrInfoEmitter::printDefList(const std::vector<Record*> &Uses,
                                    unsigned Num, std::ostream &OS) const {
  OS << "static const unsigned ImplicitList" << Num << "[] = { ";
  for (unsigned i = 0, e = Uses.size(); i != e; ++i)
    OS << getQualifiedName(Uses[i]) << ", ";
  OS << "0 };\n";
}

static std::vector<Record*> GetOperandInfo(const CodeGenInstruction &Inst) {
  std::vector<Record*> Result;
  for (unsigned i = 0, e = Inst.OperandList.size(); i != e; ++i) {
    if (Inst.OperandList[i].Rec->isSubClassOf("RegisterClass")) {
      Result.push_back(Inst.OperandList[i].Rec);
    } else {
      // This might be a multiple operand thing.
      // Targets like X86 have registers in their multi-operand operands.
      DagInit *MIOI = Inst.OperandList[i].MIOperandInfo;
      unsigned NumDefs = MIOI->getNumArgs();
      for (unsigned j = 0, e = Inst.OperandList[i].MINumOperands; j != e; ++j) {
        if (NumDefs <= j) {
          Result.push_back(0);
        } else {
          DefInit *Def = dynamic_cast<DefInit*>(MIOI->getArg(j));
          Result.push_back(Def ? Def->getDef() : 0);
        }
      }
    }
  }
  return Result;
}


// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(std::ostream &OS) {
  GatherItinClasses();

  EmitSourceFileHeader("Target Instruction Descriptors", OS);
  OS << "namespace llvm {\n\n";

  CodeGenTarget Target;
  const std::string &TargetName = Target.getName();
  Record *InstrInfo = Target.getInstructionSet();

  // Emit empty implicit uses and defs lists
  OS << "static const unsigned EmptyImpList[] = { 0 };\n";

  // Keep track of all of the def lists we have emitted already.
  std::map<std::vector<Record*>, unsigned> EmittedLists;
  unsigned ListNumber = 0;
 
  // Emit all of the instruction's implicit uses and defs.
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
         E = Target.inst_end(); II != E; ++II) {
    Record *Inst = II->second.TheDef;
    std::vector<Record*> Uses = Inst->getValueAsListOfDefs("Uses");
    if (!Uses.empty()) {
      unsigned &IL = EmittedLists[Uses];
      if (!IL) printDefList(Uses, IL = ++ListNumber, OS);
    }
    std::vector<Record*> Defs = Inst->getValueAsListOfDefs("Defs");
    if (!Defs.empty()) {
      unsigned &IL = EmittedLists[Defs];
      if (!IL) printDefList(Defs, IL = ++ListNumber, OS);
    }
  }

  std::map<std::vector<Record*>, unsigned> OperandInfosEmitted;
  unsigned OperandListNum = 0;
  OperandInfosEmitted[std::vector<Record*>()] = ++OperandListNum;
  
  // Emit all of the operand info records.
  OS << "\n";
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
       E = Target.inst_end(); II != E; ++II) {
    std::vector<Record*> OperandInfo = GetOperandInfo(II->second);
    unsigned &N = OperandInfosEmitted[OperandInfo];
    if (N == 0) {
      N = ++OperandListNum;
      OS << "static const TargetOperandInfo OperandInfo" << N << "[] = { ";
      for (unsigned i = 0, e = OperandInfo.size(); i != e; ++i) {
        Record *RC = OperandInfo[i];
        // FIXME: We only care about register operands for now.
        if (RC && RC->isSubClassOf("RegisterClass"))
          OS << "{ " << getQualifiedName(RC) << "RegClassID, 0 }, ";
        else if (RC && RC->getName() == "ptr_rc")
          // Ptr value whose register class is resolved via callback.
          OS << "{ 0, 1 }, ";
        else
          OS << "{ 0, 0 }, ";
      }
      OS << "};\n";
    }
  }
  
  // Emit all of the TargetInstrDescriptor records in their ENUM ordering.
  //
  OS << "\nstatic const TargetInstrDescriptor " << TargetName
     << "Insts[] = {\n";
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);

  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i)
    emitRecord(*NumberedInstructions[i], i, InstrInfo, EmittedLists,
               OperandInfosEmitted, OS);
  OS << "};\n";
  OS << "} // End llvm namespace \n";
}

void InstrInfoEmitter::emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                                  Record *InstrInfo,
                         std::map<std::vector<Record*>, unsigned> &EmittedLists,
                               std::map<std::vector<Record*>, unsigned> &OpInfo,
                                  std::ostream &OS) {
  int MinOperands;
  if (!Inst.OperandList.empty())
    // Each logical operand can be multiple MI operands.
    MinOperands = Inst.OperandList.back().MIOperandNo +
                  Inst.OperandList.back().MINumOperands;
  else
    MinOperands = 0;
  
  OS << "  { \"";
  if (Inst.Name.empty())
    OS << Inst.TheDef->getName();
  else
    OS << Inst.Name;
  
  unsigned ItinClass = !IsItineraries ? 0 :
            ItinClassNumber(Inst.TheDef->getValueAsDef("Itinerary")->getName());
  
  OS << "\",\t" << MinOperands << ", " << ItinClass
     << ", 0";

  // Try to determine (from the pattern), if the instruction is a store.
  bool isStore = false;
  if (dynamic_cast<ListInit*>(Inst.TheDef->getValueInit("Pattern"))) {
    ListInit *LI = Inst.TheDef->getValueAsListInit("Pattern");
    if (LI && LI->getSize() > 0) {
      DagInit *Dag = (DagInit *)LI->getElement(0);
      DefInit *OpDef = dynamic_cast<DefInit*>(Dag->getOperator());
      if (OpDef) {
        Record *Operator = OpDef->getDef();
        if (Operator->isSubClassOf("SDNode")) {
          const std::string Opcode = Operator->getValueAsString("Opcode");
          if (Opcode == "ISD::STORE" || Opcode == "ISD::TRUNCSTORE")
            isStore = true;
        }
      }
    }
  }

  // Emit all of the target indepedent flags...
  if (Inst.isReturn)     OS << "|M_RET_FLAG";
  if (Inst.isBranch)     OS << "|M_BRANCH_FLAG";
  if (Inst.isBarrier)    OS << "|M_BARRIER_FLAG";
  if (Inst.hasDelaySlot) OS << "|M_DELAY_SLOT_FLAG";
  if (Inst.isCall)       OS << "|M_CALL_FLAG";
  if (Inst.isLoad)       OS << "|M_LOAD_FLAG";
  if (Inst.isStore || isStore) OS << "|M_STORE_FLAG";
  if (Inst.isTwoAddress) OS << "|M_2_ADDR_FLAG";
  if (Inst.isConvertibleToThreeAddress) OS << "|M_CONVERTIBLE_TO_3_ADDR";
  if (Inst.isCommutable) OS << "|M_COMMUTABLE";
  if (Inst.isTerminator) OS << "|M_TERMINATOR_FLAG";
  if (Inst.usesCustomDAGSchedInserter)
    OS << "|M_USES_CUSTOM_DAG_SCHED_INSERTION";
  if (Inst.hasVariableNumberOfOperands)
    OS << "|M_VARIABLE_OPS";
  OS << ", 0";

  // Emit all of the target-specific flags...
  ListInit *LI    = InstrInfo->getValueAsListInit("TSFlagsFields");
  ListInit *Shift = InstrInfo->getValueAsListInit("TSFlagsShifts");
  if (LI->getSize() != Shift->getSize())
    throw "Lengths of " + InstrInfo->getName() +
          ":(TargetInfoFields, TargetInfoPositions) must be equal!";

  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    emitShiftedValue(Inst.TheDef, dynamic_cast<StringInit*>(LI->getElement(i)),
                     dynamic_cast<IntInit*>(Shift->getElement(i)), OS);

  OS << ", ";

  // Emit the implicit uses and defs lists...
  std::vector<Record*> UseList = Inst.TheDef->getValueAsListOfDefs("Uses");
  if (UseList.empty())
    OS << "EmptyImpList, ";
  else
    OS << "ImplicitList" << EmittedLists[UseList] << ", ";

  std::vector<Record*> DefList = Inst.TheDef->getValueAsListOfDefs("Defs");
  if (DefList.empty())
    OS << "EmptyImpList, ";
  else
    OS << "ImplicitList" << EmittedLists[DefList] << ", ";

  // Emit the operand info.
  std::vector<Record*> OperandInfo = GetOperandInfo(Inst);
  if (OperandInfo.empty())
    OS << "0";
  else
    OS << "OperandInfo" << OpInfo[OperandInfo];
  
  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << "\n";
}

struct LessRecord {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getName() < Rec2->getName();
  }
};
void InstrInfoEmitter::GatherItinClasses() {
  std::vector<Record*> DefList =
                          Records.getAllDerivedDefinitions("InstrItinClass");
  IsItineraries = !DefList.empty();
  
  if (!IsItineraries) return;
  
  std::sort(DefList.begin(), DefList.end(), LessRecord());

  for (unsigned i = 0, N = DefList.size(); i < N; i++) {
    Record *Def = DefList[i];
    ItinClassMap[Def->getName()] = i;
  }
}  
  
unsigned InstrInfoEmitter::ItinClassNumber(std::string ItinName) {
  return ItinClassMap[ItinName];
}

void InstrInfoEmitter::emitShiftedValue(Record *R, StringInit *Val,
                                        IntInit *ShiftInt, std::ostream &OS) {
  if (Val == 0 || ShiftInt == 0)
    throw std::string("Illegal value or shift amount in TargetInfo*!");
  RecordVal *RV = R->getValue(Val->getValue());
  int Shift = ShiftInt->getValue();

  if (RV == 0 || RV->getValue() == 0) {
    // This isn't an error if this is a builtin instruction.
    if (R->getName() != "PHI" && R->getName() != "INLINEASM")
      throw R->getName() + " doesn't have a field named '" + 
            Val->getValue() + "'!";
    return;
  }

  Init *Value = RV->getValue();
  if (BitInit *BI = dynamic_cast<BitInit*>(Value)) {
    if (BI->getValue()) OS << "|(1<<" << Shift << ")";
    return;
  } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Value)) {
    // Convert the Bits to an integer to print...
    Init *I = BI->convertInitializerTo(new IntRecTy());
    if (I)
      if (IntInit *II = dynamic_cast<IntInit*>(I)) {
        if (II->getValue()) {
          if (Shift)
            OS << "|(" << II->getValue() << "<<" << Shift << ")";
          else
            OS << "|" << II->getValue();
        }
        return;
      }

  } else if (IntInit *II = dynamic_cast<IntInit*>(Value)) {
    if (II->getValue()) {
      if (Shift)
        OS << "|(" << II->getValue() << "<<" << Shift << ")";
      else
        OS << II->getValue();
    }
    return;
  }

  std::cerr << "Unhandled initializer: " << *Val << "\n";
  throw "In record '" + R->getName() + "' for TSFlag emission.";
}

