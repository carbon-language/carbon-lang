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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

//===----------------------------------------------------------------------===//
// ARMConstantPoolValue
//===----------------------------------------------------------------------===//

ARMConstantPoolValue::ARMConstantPoolValue(Type *Ty, unsigned id,
                                           ARMCP::ARMCPKind kind,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier modifier,
                                           bool addCurrentAddress)
  : MachineConstantPoolValue(Ty), LabelId(id), Kind(kind), PCAdjust(PCAdj),
    Modifier(modifier), AddCurrentAddress(addCurrentAddress) {}

ARMConstantPoolValue::ARMConstantPoolValue(const Constant *cval, unsigned id,
                                           ARMCP::ARMCPKind K,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((Type*)cval->getType()),
    CVal(cval), S(NULL), LabelId(id), Kind(K), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(LLVMContext &C,
                                           const MachineBasicBlock *mbb,
                                           unsigned id,
                                           ARMCP::ARMCPKind K,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((Type*)Type::getInt8PtrTy(C)),
    CVal(NULL), MBB(mbb), S(NULL), LabelId(id), Kind(K), PCAdjust(PCAdj),
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

const MachineBasicBlock *ARMConstantPoolValue::getMBB() const {
  return MBB;
}

const char *ARMConstantPoolValue::getModifierText() const {
  switch (Modifier) {
  default: llvm_unreachable("Unknown modifier!");
    // FIXME: Are these case sensitive? It'd be nice to lower-case all the
    // strings if that's legal.
  case ARMCP::no_modifier: return "none";
  case ARMCP::TLSGD:       return "tlsgd";
  case ARMCP::GOT:         return "GOT";
  case ARMCP::GOTOFF:      return "GOTOFF";
  case ARMCP::GOTTPOFF:    return "gottpoff";
  case ARMCP::TPOFF:       return "tpoff";
  }
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
  else if (MBB)
    O << "";
  else
    O << S;
  if (Modifier) O << "(" << getModifierText() << ")";
  if (PCAdjust != 0) {
    O << "-(LPC" << LabelId << "+" << (unsigned)PCAdjust;
    if (AddCurrentAddress) O << "-.";
    O << ")";
  }
}

//===----------------------------------------------------------------------===//
// ARMConstantPoolConstant
//===----------------------------------------------------------------------===//

ARMConstantPoolConstant::ARMConstantPoolConstant(const Constant *C,
                                                 unsigned ID,
                                                 ARMCP::ARMCPKind Kind,
                                                 unsigned char PCAdj,
                                                 ARMCP::ARMCPModifier Modifier,
                                                 bool AddCurrentAddress)
  : ARMConstantPoolValue((Type*)C->getType(), ID, Kind, PCAdj, Modifier,
                         AddCurrentAddress),
    CVal(C) {}

ARMConstantPoolConstant *
ARMConstantPoolConstant::Create(const Constant *C, unsigned ID) {
  return new ARMConstantPoolConstant(C, ID, ARMCP::CPValue, 0,
                                     ARMCP::no_modifier, false);
}

const GlobalValue *ARMConstantPoolConstant::getGV() const {
  return dyn_cast<GlobalValue>(CVal);
}

bool ARMConstantPoolConstant::hasSameValue(ARMConstantPoolValue *ACPV) {
  const ARMConstantPoolConstant *ACPC = dyn_cast<ARMConstantPoolConstant>(ACPV);

  return (ACPC ? ACPC->CVal == CVal : true) &&
    ARMConstantPoolValue::hasSameValue(ACPV);
}

void ARMConstantPoolConstant::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  ARMConstantPoolValue::addSelectionDAGCSEId(ID);
}

void ARMConstantPoolConstant::print(raw_ostream &O) const {
  O << CVal->getName();
  ARMConstantPoolValue::print(O);
}
