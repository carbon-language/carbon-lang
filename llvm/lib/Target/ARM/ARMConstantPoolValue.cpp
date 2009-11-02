//===- ARMConstantPoolValue.cpp - ARM constantpool value --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "ARMConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Constant.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

ARMConstantPoolValue::ARMConstantPoolValue(Constant *cval, unsigned id,
                                           ARMCP::ARMCPKind K,
                                           unsigned char PCAdj,
                                           const char *Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((const Type*)cval->getType()),
    CVal(cval), S(NULL), LabelId(id), Kind(K), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(LLVMContext &C,
                                           const char *s, unsigned id,
                                           unsigned char PCAdj,
                                           const char *Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((const Type*)Type::getInt32Ty(C)),
    CVal(NULL), S(strdup(s)), LabelId(id), Kind(ARMCP::CPExtSymbol),
    PCAdjust(PCAdj), Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(GlobalValue *gv, const char *Modif)
  : MachineConstantPoolValue((const Type*)Type::getInt32Ty(gv->getContext())),
    CVal(gv), S(NULL), LabelId(0), Kind(ARMCP::CPValue), PCAdjust(0),
    Modifier(Modif) {}

GlobalValue *ARMConstantPoolValue::getGV() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

BlockAddress *ARMConstantPoolValue::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

int ARMConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                    unsigned Alignment) {
  unsigned AlignMask = Alignment - 1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].getAlignment() & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      if (CPV->CVal == CVal &&
          CPV->S == S &&
          CPV->LabelId == LabelId &&
          CPV->PCAdjust == PCAdjust)
        return i;
    }
  }

  return -1;
}

ARMConstantPoolValue::~ARMConstantPoolValue() {
  free((void*)S);
}

void
ARMConstantPoolValue::AddSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  ID.AddPointer(S);
  ID.AddInteger(LabelId);
  ID.AddInteger(PCAdjust);
}

void ARMConstantPoolValue::dump() const {
  errs() << "  " << *this;
}


void ARMConstantPoolValue::print(raw_ostream &O) const {
  if (CVal)
    O << CVal->getName();
  else
    O << S;
  if (Modifier) O << "(" << Modifier << ")";
  if (PCAdjust != 0) {
    O << "-(LPC" << LabelId << "+" << (unsigned)PCAdjust;
    if (AddCurrentAddress) O << "-.";
    O << ")";
  }
}
