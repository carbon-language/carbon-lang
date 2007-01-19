//===- ARMConstantPoolValue.cpp - ARM constantpool value --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "ARMConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/GlobalValue.h"
using namespace llvm;

ARMConstantPoolValue::ARMConstantPoolValue(GlobalValue *gv, unsigned id,
                                         bool isNonLazy, unsigned char PCAdj)
  : MachineConstantPoolValue((const Type*)gv->getType()),
    GV(gv), LabelId(id), isNonLazyPtr(isNonLazy), PCAdjust(PCAdj) {}

int ARMConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                    unsigned Alignment) {
  unsigned AlignMask = (1 << Alignment)-1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].Offset & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      if (CPV->GV == GV && CPV->LabelId == LabelId &&
          CPV->isNonLazyPtr == isNonLazyPtr)
        return i;
    }
  }

  return -1;
}

void
ARMConstantPoolValue::AddSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(GV);
  ID.AddInteger(LabelId);
  ID.AddInteger((unsigned)isNonLazyPtr);
  ID.AddInteger(PCAdjust);
}

void ARMConstantPoolValue::print(std::ostream &O) const {
  O << GV->getName();
  if (isNonLazyPtr) O << "$non_lazy_ptr";
  if (PCAdjust != 0) O << "-(LPIC" << LabelId << "+"
                       << (unsigned)PCAdjust << ")";
}
