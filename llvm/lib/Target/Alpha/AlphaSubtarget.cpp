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
#include "llvm/Support/Debug.h"

using namespace llvm;

enum AlphaFeature {
  AlphaFeatureCIX   = 1 << 0,
  AlphaFeatureFIX = 1 << 1,
};

/// Sorted (by key) array of values for CPU subtype.
static const SubtargetFeatureKV AlphaSubTypeKV[] = {
  { "ev56"    , "Select the Alpha EV56 processor", 0 },
  { "ev6"    , "Select the Alpha EV6 processor", AlphaFeatureFIX },
  { "ev67"    , "Select the Alpha EV67 processor", AlphaFeatureFIX | AlphaFeatureCIX },
  { "generic", "Select instructions for a generic Alpha processor (EV56)", 0 },
  { "pca56"    , "Select the Alpha PCA56 processor", 0 },
};

/// Length of AlphaSubTypeKV.
static const unsigned AlphaSubTypeKVSize = sizeof(AlphaSubTypeKV)
                                             / sizeof(SubtargetFeatureKV);

/// Sorted (by key) array of values for CPU features.
static SubtargetFeatureKV AlphaFeatureKV[] = {
  { "CIX", "Should CIX extentions be used" , AlphaFeatureCIX },
  { "FIX"  , "Should FIX extentions be used"  , AlphaFeatureFIX },
 };
/// Length of AlphaFeatureKV.
static const unsigned AlphaFeatureKVSize = sizeof(AlphaFeatureKV)
                                          / sizeof(SubtargetFeatureKV);

AlphaSubtarget::AlphaSubtarget(const Module &M, const std::string &FS)
  :HasF2I(false), HasCT(false)
{
  std::string CPU = "generic";
  uint32_t Bits =
  SubtargetFeatures::Parse(FS, CPU,
                           AlphaSubTypeKV, AlphaSubTypeKVSize,
                           AlphaFeatureKV, AlphaFeatureKVSize);
  HasF2I = (Bits & AlphaFeatureFIX) != 0;
  HasCT  = (Bits & AlphaFeatureCIX) != 0;

}
