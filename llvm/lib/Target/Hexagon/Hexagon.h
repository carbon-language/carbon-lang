//=-- Hexagon.h - Top-level interface for Hexagon representation --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Hexagon back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_Hexagon_H
#define TARGET_Hexagon_H

#include <cassert>
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  class FunctionPass;
  class TargetMachine;
  class HexagonTargetMachine;
  class raw_ostream;

  FunctionPass *createHexagonISelDag(HexagonTargetMachine &TM);
  FunctionPass *createHexagonDelaySlotFillerPass(TargetMachine &TM);
  FunctionPass *createHexagonFPMoverPass(TargetMachine &TM);
  FunctionPass *createHexagonRemoveExtendOps(HexagonTargetMachine &TM);
  FunctionPass *createHexagonCFGOptimizer(HexagonTargetMachine &TM);

  FunctionPass* createHexagonSplitTFRCondSets(HexagonTargetMachine &TM);
  FunctionPass* createHexagonExpandPredSpillCode(HexagonTargetMachine &TM);

  FunctionPass *createHexagonHardwareLoops();
  FunctionPass *createHexagonOptimizeSZExtends();
  FunctionPass *createHexagonFixupHwLoops();

} // end namespace llvm;

#define Hexagon_POINTER_SIZE 4

#define Hexagon_PointerSize (Hexagon_POINTER_SIZE)
#define Hexagon_PointerSize_Bits (Hexagon_POINTER_SIZE * 8)
#define Hexagon_WordSize Hexagon_PointerSize
#define Hexagon_WordSize_Bits Hexagon_PointerSize_Bits

// allocframe saves LR and FP on stack before allocating
// a new stack frame. This takes 8 bytes.
#define HEXAGON_LRFP_SIZE 8

#endif
