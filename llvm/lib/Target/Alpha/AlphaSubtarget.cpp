//===- AlphaSubtarget.cpp - Alpha Subtarget Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Alpha specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AlphaSubtarget.h"
#include "Alpha.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/SubtargetFeature.h"

using namespace llvm;

//"alphaev67-unknown-linux-gnu"

AlphaSubtarget::AlphaSubtarget(const Module &M, const std::string &FS)
  :HasF2I(false), HasCT(false)
{
//TODO: figure out host
}
