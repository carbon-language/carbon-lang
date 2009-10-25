//===-- llvm/CodeGen/PseudoSourceValue.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PseudoSourceValue class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
using namespace llvm;

static ManagedStatic<PseudoSourceValue[4]> PSVs;

const PseudoSourceValue *PseudoSourceValue::getStack()
{ return &(*PSVs)[0]; }
const PseudoSourceValue *PseudoSourceValue::getGOT()
{ return &(*PSVs)[1]; }
const PseudoSourceValue *PseudoSourceValue::getJumpTable()
{ return &(*PSVs)[2]; }
const PseudoSourceValue *PseudoSourceValue::getConstantPool()
{ return &(*PSVs)[3]; }

static const char *const PSVNames[] = {
  "Stack",
  "GOT",
  "JumpTable",
  "ConstantPool"
};

// FIXME: THIS IS A HACK!!!!
// Eventually these should be uniqued on LLVMContext rather than in a managed
// static.  For now, we can safely use the global context for the time being to
// squeak by.
PseudoSourceValue::PseudoSourceValue() :
  Value(Type::getInt8PtrTy(getGlobalContext()),
        PseudoSourceValueVal) {}

void PseudoSourceValue::printCustom(raw_ostream &O) const {
  O << PSVNames[this - *PSVs];
}

namespace {
  /// FixedStackPseudoSourceValue - A specialized PseudoSourceValue
  /// for holding FixedStack values, which must include a frame
  /// index.
  class FixedStackPseudoSourceValue : public PseudoSourceValue {
    const int FI;
  public:
    explicit FixedStackPseudoSourceValue(int fi) : FI(fi) {}

    virtual bool isConstant(const MachineFrameInfo *MFI) const;

    virtual bool isAliased(const MachineFrameInfo *MFI) const;

    virtual void printCustom(raw_ostream &OS) const {
      OS << "FixedStack" << FI;
    }
  };
}

static ManagedStatic<std::map<int, const PseudoSourceValue *> > FSValues;

const PseudoSourceValue *PseudoSourceValue::getFixedStack(int FI) {
  const PseudoSourceValue *&V = (*FSValues)[FI];
  if (!V)
    V = new FixedStackPseudoSourceValue(FI);
  return V;
}

bool PseudoSourceValue::isConstant(const MachineFrameInfo *) const {
  if (this == getStack())
    return false;
  if (this == getGOT() ||
      this == getConstantPool() ||
      this == getJumpTable())
    return true;
  llvm_unreachable("Unknown PseudoSourceValue!");
  return false;
}

bool PseudoSourceValue::isAliased(const MachineFrameInfo *MFI) const {
  if (this == getStack() ||
      this == getGOT() ||
      this == getConstantPool() ||
      this == getJumpTable())
    return false;
  llvm_unreachable("Unknown PseudoSourceValue!");
  return true;
}

bool FixedStackPseudoSourceValue::isConstant(const MachineFrameInfo *MFI) const{
  return MFI && MFI->isImmutableObjectIndex(FI);
}

bool FixedStackPseudoSourceValue::isAliased(const MachineFrameInfo *MFI) const {
  // Negative frame indices are used for special things that don't
  // appear in LLVM IR. Non-negative indices may be used for things
  // like static allocas.
  if (!MFI)
    return FI >= 0;
  // Spill slots should not alias others.
  return !MFI->isFixedObjectIndex(FI) && !MFI->isSpillSlotObjectIndex(FI);
}
