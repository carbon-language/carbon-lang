//===-- SparcSubtarget.cpp - SPARC Subtarget Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SparcSubtarget.h"
#include "Sparc.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "sparc-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SparcGenSubtargetInfo.inc"

void SparcSubtarget::anchor() { }

static std::string computeDataLayout(const SparcSubtarget &ST) {
  // Sparc is big endian.
  std::string Ret = "E-m:e";

  // Some ABIs have 32bit pointers.
  if (!ST.is64Bit())
    Ret += "-p:32:32";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // On SparcV9 128 floats are aligned to 128 bits, on others only to 64.
  // On SparcV9 registers can hold 64 or 32 bits, on others only 32.
  if (ST.is64Bit())
    Ret += "-n32:64";
  else
    Ret += "-f128:64-n32";

  if (ST.is64Bit())
    Ret += "-S128";
  else
    Ret += "-S64";

  return Ret;
}

SparcSubtarget &SparcSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  IsV9 = false;
  V8DeprecatedInsts = false;
  IsVIS = false;
  HasHardQuad = false;
  UsePopc = false;

  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = (Is64Bit) ? "v9" : "v8";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Popc is a v9-only instruction.
  if (!IsV9)
    UsePopc = false;

  return *this;
}

SparcSubtarget::SparcSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, TargetMachine &TM,
                               bool is64Bit)
    : SparcGenSubtargetInfo(TT, CPU, FS), Is64Bit(is64Bit),
      DL(computeDataLayout(initializeSubtargetDependencies(CPU, FS))),
      InstrInfo(*this), TLInfo(TM), TSInfo(DL), FrameLowering(*this) {}

int SparcSubtarget::getAdjustedFrameSize(int frameSize) const {

  if (is64Bit()) {
    // All 64-bit stack frames must be 16-byte aligned, and must reserve space
    // for spilling the 16 window registers at %sp+BIAS..%sp+BIAS+128.
    frameSize += 128;
    // Frames with calls must also reserve space for 6 outgoing arguments
    // whether they are used or not. LowerCall_64 takes care of that.
    assert(frameSize % 16 == 0 && "Stack size not 16-byte aligned");
  } else {
    // Emit the correct save instruction based on the number of bytes in
    // the frame. Minimum stack frame size according to V8 ABI is:
    //   16 words for register window spill
    //    1 word for address of returned aggregate-value
    // +  6 words for passing parameters on the stack
    // ----------
    //   23 words * 4 bytes per word = 92 bytes
    frameSize += 92;

    // Round up to next doubleword boundary -- a double-word boundary
    // is required by the ABI.
    frameSize = RoundUpToAlignment(frameSize, 8);
  }
  return frameSize;
}
