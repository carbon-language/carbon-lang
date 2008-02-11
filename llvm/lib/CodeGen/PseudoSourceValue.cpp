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

#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/ManagedStatic.h"

namespace llvm {
  static ManagedStatic<PseudoSourceValue[5]> PSVs;

  const PseudoSourceValue *PseudoSourceValue::getFixedStack()
  { return &(*PSVs)[0]; }
  const PseudoSourceValue *PseudoSourceValue::getStack()
  { return &(*PSVs)[1]; }
  const PseudoSourceValue *PseudoSourceValue::getGOT()
  { return &(*PSVs)[2]; }
  const PseudoSourceValue *PseudoSourceValue::getConstantPool()
  { return &(*PSVs)[3]; }
  const PseudoSourceValue *PseudoSourceValue::getJumpTable()
  { return &(*PSVs)[4]; }

  static const char *PSVNames[] = {
    "FixedStack",
    "Stack",
    "GOT",
    "ConstantPool",
    "JumpTable"
  };

  PseudoSourceValue::PseudoSourceValue() :
    Value(PointerType::getUnqual(Type::Int8Ty), PseudoSourceValueVal) {}

  void PseudoSourceValue::print(std::ostream &OS) const {
    OS << PSVNames[this - *PSVs];
  }
}
