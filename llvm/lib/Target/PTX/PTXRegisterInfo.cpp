//===- PTXRegisterInfo.cpp - PTX Register Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXRegisterInfo.h"

using namespace llvm;

#include "PTXGenRegisterInfo.inc"
