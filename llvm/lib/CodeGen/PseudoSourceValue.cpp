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

namespace llvm {
  const PseudoSourceValue PseudoSourceValue::FPRel("FPRel");
  const PseudoSourceValue PseudoSourceValue::SPRel("SPRel");
  const PseudoSourceValue PseudoSourceValue::GPRel("GPRel");
  const PseudoSourceValue PseudoSourceValue::TPRel("TPRel");
  const PseudoSourceValue PseudoSourceValue::CPRel("CPRel");
  const PseudoSourceValue PseudoSourceValue::JTRel("JTRel");

  PseudoSourceValue::PseudoSourceValue(const char *_name) :
    Value(PointerType::getUnqual(Type::Int8Ty), PseudoSourceValueVal),
          name(_name) {
  }

  void PseudoSourceValue::print(std::ostream &OS) const {
    OS << name;
  }
}
