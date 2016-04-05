//===- llvm/CodeGen/GlobalISel/RegisterBank.cpp - Register Bank --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegisterBank class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegisterBank.h"

#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "registerbank"

using namespace llvm;

void RegisterBank::verify(const TargetRegisterInfo &TRI) const {
  // Verify that the Size of the register bank is big enough to cover all the
  // register classes it covers.
  // Verify that the register bank covers all the sub and super classes of the
  // classes it covers.
}

bool RegisterBank::contains(const TargetRegisterClass &RC) const {
  return ContainedRegClass.test(RC.getID());
}

bool RegisterBank::operator==(const RegisterBank &OtherRB) const {
  // There must be only one instance of a given register bank alive
  // for the whole compilation.
  // The RegisterBankInfo is supposed to enforce that.
  assert((OtherRB.getID() != getID() || &OtherRB == this) &&
         "ID does not uniquely identify a RegisterBank");
  return &OtherRB == this;
}
