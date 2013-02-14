//===-- AArch64MachineFuctionInfo.cpp - AArch64 machine function info -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file just contains the anchor for the AArch64MachineFunctionInfo to
// force vtable emission.
//
//===----------------------------------------------------------------------===//
#include "AArch64MachineFunctionInfo.h"

using namespace llvm;

void AArch64MachineFunctionInfo::anchor() { }
