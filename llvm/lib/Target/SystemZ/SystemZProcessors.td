//===-- SystemZ.td - SystemZ processors and features ---------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Processor and feature definitions.
//
//===----------------------------------------------------------------------===//

class SystemZFeature<string extname, string intname, string desc>
  : Predicate<"Subtarget.has"##intname##"()">,
    AssemblerPredicate<"Feature"##intname, extname>,
    SubtargetFeature<extname, "Has"##intname, "true", desc>;

def FeatureDistinctOps : SystemZFeature<
  "distinct-ops", "DistinctOps",
  "Assume that the distinct-operands facility is installed"
>;

def : Processor<"z10",   NoItineraries, []>;
def : Processor<"z196",  NoItineraries, [FeatureDistinctOps]>;
def : Processor<"zEC12", NoItineraries, [FeatureDistinctOps]>;
