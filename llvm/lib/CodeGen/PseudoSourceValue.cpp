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
  /// StackObjectPseudoSourceValue - A specialized PseudoSourceValue
  /// for holding StackObject values, which must include a frame
  /// index.
  class VISIBILITY_HIDDEN StackObjectPseudoSourceValue
    : public PseudoSourceValue {
    const int FI;
  public:
    explicit StackObjectPseudoSourceValue(int fi) : FI(fi) {}

    virtual bool isConstant(const MachineFrameInfo *MFI) const;

    virtual void printCustom(raw_ostream &OS) const {
      if (FI < 0)
        OS << "Fixed";
      OS << "StackObject" << FI;
    }
  };
}

static ManagedStatic<std::map<int, const PseudoSourceValue *> > FSValues;

const PseudoSourceValue *PseudoSourceValue::getStackObject(int FI) {
  const PseudoSourceValue *&V = (*FSValues)[FI];
  if (!V)
    V = new StackObjectPseudoSourceValue(FI);
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

bool
StackObjectPseudoSourceValue::isConstant(const MachineFrameInfo *MFI) const {
  return MFI && MFI->isImmutableObjectIndex(FI);
}
