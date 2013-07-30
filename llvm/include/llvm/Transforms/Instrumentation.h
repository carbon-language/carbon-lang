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
struct GCOVOptions {
  static GCOVOptions getDefault();

  // Specify whether to emit .gcno files.
  bool EmitNotes;

  // Specify whether to modify the program to emit .gcda files when run.
  bool EmitData;

  // A four-byte version string. The meaning of a version string is described in
  // gcc's gcov-io.h
  char Version[4];

  // Emit a "cfg checksum" that follows the "line number checksum" of a
  // function. This affects both .gcno and .gcda files.
  bool UseCfgChecksum;

  // Add the 'noredzone' attribute to added runtime library calls.
  bool NoRedZone;

  // Emit the name of the function in the .gcda files. This is redundant, as
  // the function identifier can be used to find the name from the .gcno file.
  bool FunctionNamesInData;
};
ModulePass *createGCOVProfilerPass(const GCOVOptions &Options =
                                   GCOVOptions::getDefault());

// Insert AddressSanitizer (address sanity checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass(
    bool CheckInitOrder = true, bool CheckUseAfterReturn = false,
    bool CheckLifetime = false, StringRef BlacklistFile = StringRef(),
    bool ZeroBaseShadow = false);
ModulePass *createAddressSanitizerModulePass(
    bool CheckInitOrder = true, StringRef BlacklistFile = StringRef(),
    bool ZeroBaseShadow = false);

// Insert MemorySanitizer instrumentation (detection of uninitialized reads)
FunctionPass *createMemorySanitizerPass(bool TrackOrigins = false,
                                        StringRef BlacklistFile = StringRef());

// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerPass(StringRef BlacklistFile = StringRef());

// BoundsChecking - This pass instruments the code to perform run-time bounds
// checking on loads, stores, and other memory intrinsics.
FunctionPass *createBoundsCheckingPass();

/// createDebugIRPass - Enable interactive stepping through LLVM IR in LLDB (or
///                     GDB) and generate a file with the LLVM IR to be
///                     displayed in the debugger.
///
/// Existing debug metadata is preserved (but may be modified) in order to allow
/// accessing variables in the original source. The line table and file
/// information is modified to correspond to the lines in the LLVM IR. If
/// Filename and Directory are empty, a file name is generated based on existing
/// debug information. If no debug information is available, a temporary file
/// name is generated.
///
/// @param HideDebugIntrinsics  Omit debug intrinsics in emitted IR source file.
/// @param HideDebugMetadata    Omit debug metadata in emitted IR source file.
/// @param Directory            Embed this directory in the debug information.
/// @param Filename             Embed this file name in the debug information.
ModulePass *createDebugIRPass(bool HideDebugIntrinsics,
                              bool HideDebugMetadata,
                              StringRef Directory = StringRef(),
                              StringRef Filename = StringRef());

/// createDebugIRPass - Enable interactive stepping through LLVM IR in LLDB
///                     (or GDB) with an existing IR file on disk. When creating
///                     a DebugIR pass with this function, no source file is
///                     output to disk and the existing one is unmodified. Debug
///                     metadata in the Module is created/updated to point to
///                     the existing textual IR file on disk.
/// NOTE: If the IR file to be debugged is not on disk, use the version of this
///       function with parameters in order to generate the file that will be
///       seen by the debugger.
ModulePass *createDebugIRPass();

} // End llvm namespace

#endif
