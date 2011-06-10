//===- CodeGenRegisters.cpp - Register and RegisterClass Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenRegisters.h"
#include "CodeGenTarget.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//                              CodeGenRegister
//===----------------------------------------------------------------------===//

CodeGenRegister::CodeGenRegister(Record *R) : TheDef(R) {
  CostPerUse = R->getValueAsInt("CostPerUse");
}

const std::string &CodeGenRegister::getName() const {
  return TheDef->getName();
}

//===----------------------------------------------------------------------===//
//                            CodeGenRegisterClass
//===----------------------------------------------------------------------===//

CodeGenRegisterClass::CodeGenRegisterClass(Record *R) : TheDef(R) {
  // Rename anonymous register classes.
  if (R->getName().size() > 9 && R->getName()[9] == '.') {
    static unsigned AnonCounter = 0;
    R->setName("AnonRegClass_"+utostr(AnonCounter++));
  }

  std::vector<Record*> TypeList = R->getValueAsListOfDefs("RegTypes");
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    Record *Type = TypeList[i];
    if (!Type->isSubClassOf("ValueType"))
      throw "RegTypes list member '" + Type->getName() +
        "' does not derive from the ValueType class!";
    VTs.push_back(getValueType(Type));
  }
  assert(!VTs.empty() && "RegisterClass must contain at least one ValueType!");

  std::vector<Record*> RegList = R->getValueAsListOfDefs("MemberList");
  for (unsigned i = 0, e = RegList.size(); i != e; ++i) {
    Record *Reg = RegList[i];
    if (!Reg->isSubClassOf("Register"))
      throw "Register Class member '" + Reg->getName() +
            "' does not derive from the Register class!";
    Elements.push_back(Reg);
  }

  // SubRegClasses is a list<dag> containing (RC, subregindex, ...) dags.
  ListInit *SRC = R->getValueAsListInit("SubRegClasses");
  for (ListInit::const_iterator i = SRC->begin(), e = SRC->end(); i != e; ++i) {
    DagInit *DAG = dynamic_cast<DagInit*>(*i);
    if (!DAG) throw "SubRegClasses must contain DAGs";
    DefInit *DAGOp = dynamic_cast<DefInit*>(DAG->getOperator());
    Record *RCRec;
    if (!DAGOp || !(RCRec = DAGOp->getDef())->isSubClassOf("RegisterClass"))
      throw "Operator '" + DAG->getOperator()->getAsString() +
        "' in SubRegClasses is not a RegisterClass";
    // Iterate over args, all SubRegIndex instances.
    for (DagInit::const_arg_iterator ai = DAG->arg_begin(), ae = DAG->arg_end();
         ai != ae; ++ai) {
      DefInit *Idx = dynamic_cast<DefInit*>(*ai);
      Record *IdxRec;
      if (!Idx || !(IdxRec = Idx->getDef())->isSubClassOf("SubRegIndex"))
        throw "Argument '" + (*ai)->getAsString() +
          "' in SubRegClasses is not a SubRegIndex";
      if (!SubRegClasses.insert(std::make_pair(IdxRec, RCRec)).second)
        throw "SubRegIndex '" + IdxRec->getName() + "' mentioned twice";
    }
  }

  // Allow targets to override the size in bits of the RegisterClass.
  unsigned Size = R->getValueAsInt("Size");

  Namespace = R->getValueAsString("Namespace");
  SpillSize = Size ? Size : EVT(VTs[0]).getSizeInBits();
  SpillAlignment = R->getValueAsInt("Alignment");
  CopyCost = R->getValueAsInt("CopyCost");
  Allocatable = R->getValueAsBit("isAllocatable");
  MethodBodies = R->getValueAsCode("MethodBodies");
  MethodProtos = R->getValueAsCode("MethodProtos");
}

const std::string &CodeGenRegisterClass::getName() const {
  return TheDef->getName();
}

//===----------------------------------------------------------------------===//
//                               CodeGenRegBank
//===----------------------------------------------------------------------===//

CodeGenRegBank::CodeGenRegBank(RecordKeeper &Records) : Records(Records) {
  // Read in the user-defined (named) sub-register indices. More indices will
  // be synthesized.
  SubRegIndices = Records.getAllDerivedDefinitions("SubRegIndex");
  std::sort(SubRegIndices.begin(), SubRegIndices.end(), LessRecord());
  NumNamedIndices = SubRegIndices.size();
}

Record *CodeGenRegBank::getCompositeSubRegIndex(Record *A, Record *B) {
  std::string Name = A->getName() + "_then_" + B->getName();
  Record *R = new Record(Name, SMLoc(), Records);
  Records.addDef(R);
  SubRegIndices.push_back(R);
  return R;
}

unsigned CodeGenRegBank::getSubRegIndexNo(Record *idx) {
  std::vector<Record*>::const_iterator i =
    std::find(SubRegIndices.begin(), SubRegIndices.end(), idx);
  assert(i != SubRegIndices.end() && "Not a SubRegIndex");
  return (i - SubRegIndices.begin()) + 1;
}

