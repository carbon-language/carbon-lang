//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class ModulePass;
class FunctionPass;

// Insert edge profiling instrumentation
ModulePass *createEdgeProfilerPass();

// Insert optimal edge profiling instrumentation
ModulePass *createOptimalEdgeProfilerPass();

// Insert path profiling instrumentation
ModulePass *createPathProfilerPass();

// Insert GCOV profiling instrumentation
static const char DefaultGCovVersion[4] = {'*', '2', '0', '4'};
ModulePass *createGCOVProfilerPass(bool EmitNotes = true, bool EmitData = true,
                                   const char (&Version)[4] =DefaultGCovVersion,
                                   bool UseExtraChecksum = false,
                                   bool NoRedZone = false,
                                   bool NoFunctionNamesInData = false);

// Insert AddressSanitizer (address sanity checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass(
    bool CheckInitOrder = false, bool CheckUseAfterReturn = false,
    bool CheckLifetime = false, StringRef BlacklistFile = StringRef(),
    bool ZeroBaseShadow = false);
ModulePass *createAddressSanitizerModulePass(
    bool CheckInitOrder = false, StringRef BlacklistFile = StringRef(),
    bool ZeroBaseShadow = false);

// Insert MemorySanitizer instrumentation (detection of uninitialized reads)
FunctionPass *createMemorySanitizerPass(bool TrackOrigins = false,
                                        StringRef BlacklistFile = StringRef());

// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerPass(StringRef BlacklistFile = StringRef());


// BoundsChecking - This pass instruments the code to perform run-time bounds
// checking on loads, stores, and other memory intrinsics.
FunctionPass *createBoundsCheckingPass();

} // End llvm namespace

#endif
