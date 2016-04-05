//===- llvm/CodeGen/GlobalISel/RegisterBankInfo.cpp --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegisterBankInfo class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "registerbankinfo"

using namespace llvm;

RegisterBankInfo::RegisterBankInfo(unsigned NbOfRegBanks)
    : NbOfRegBanks(NbOfRegBanks) {
  RegBanks.reset(new RegisterBank[NbOfRegBanks]);
}

RegisterBankInfo::~RegisterBankInfo() {}

void RegisterBankInfo::verify(const TargetRegisterInfo &TRI) const {}
