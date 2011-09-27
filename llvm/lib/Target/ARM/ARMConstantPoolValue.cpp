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

ARMConstantPoolValue::ARMConstantPoolValue(const Constant *cval, unsigned id,
                                           ARMCP::ARMCPKind K,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((Type*)cval->getType()),
    CVal(cval), S(NULL), LabelId(id), Kind(K), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(LLVMContext &C,
                                           const char *s, unsigned id,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((Type*)Type::getInt32Ty(C)),
    CVal(NULL), S(strdup(s)), LabelId(id), Kind(ARMCP::CPExtSymbol),
    PCAdjust(PCAdj), Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(const GlobalValue *gv,
                                           ARMCP::ARMCPModifier Modif)
  : MachineConstantPoolValue((Type*)Type::getInt32Ty(gv->getContext())),
    CVal(gv), S(NULL), LabelId(0), Kind(ARMCP::CPValue), PCAdjust(0),
    Modifier(Modif), AddCurrentAddress(false) {}

const GlobalValue *ARMConstantPoolValue::getGV() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

const BlockAddress *ARMConstantPoolValue::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

static bool CPV_streq(const char *S1, const char *S2) {
  if (S1 == S2)
    return true;
  if (S1 && S2 && strcmp(S1, S2) == 0)
    return true;
  return false;
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
          CPV->LabelId == LabelId &&
          CPV->PCAdjust == PCAdjust &&
          CPV_streq(CPV->S, S) &&
          CPV->Modifier == Modifier)
        return i;
    }
  }

  return -1;
}

ARMConstantPoolValue::~ARMConstantPoolValue() {
  free((void*)S);
}

void
ARMConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  ID.AddPointer(S);
  ID.AddInteger(LabelId);
  ID.AddInteger(PCAdjust);
}

bool
ARMConstantPoolValue::hasSameValue(ARMConstantPoolValue *ACPV) {
  if (ACPV->Kind == Kind &&
      ACPV->CVal == CVal &&
      ACPV->PCAdjust == PCAdjust &&
      CPV_streq(ACPV->S, S) &&
      ACPV->Modifier == Modifier) {
    if (ACPV->LabelId == LabelId)
      return true;
    // Two PC relative constpool entries containing the same GV address or
    // external symbols. FIXME: What about blockaddress?
    if (Kind == ARMCP::CPValue || Kind == ARMCP::CPExtSymbol)
      return true;
  }
  return false;
}

void ARMConstantPoolValue::dump() const {
  errs() << "  " << *this;
}


void ARMConstantPoolValue::print(raw_ostream &O) const {
  if (CVal)
    O << CVal->getName();
  else
    O << S;
  if (Modifier) O << "(" << getModifierText() << ")";
  if (PCAdjust != 0) {
    O << "-(LPC" << LabelId << "+" << (unsigned)PCAdjust;
    if (AddCurrentAddress) O << "-.";
    O << ")";
  }
}
