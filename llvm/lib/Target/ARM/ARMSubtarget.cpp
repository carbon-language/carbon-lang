//===-- ARMSubtarget.cpp - ARM Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMGenSubtarget.inc"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// FIXME: this is temporary.
static cl::opt<bool> Thumb("enable-thumb",
                           cl::desc("Switch to thumb mode in ARM backend"));

ARMSubtarget::ARMSubtarget(const Module &M, const std::string &FS)
  : ARMArchVersion(V4T), HasVFP2(false), IsDarwin(false),
    UseThumbBacktraces(false), IsR9Reserved(false), stackAlignment(8) {

  // Determine default and user specified characteristics
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  IsThumb = Thumb;
  
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    IsDarwin = TT.find("-darwin") != std::string::npos;
  } else if (TT.empty()) {
#if defined(__APPLE__)
    IsDarwin = true;
#endif
  }

  if (IsDarwin) {
    UseThumbBacktraces = true;
    IsR9Reserved = true;
    stackAlignment = 4;
  } 
}
