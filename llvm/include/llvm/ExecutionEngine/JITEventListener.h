//===- JITEventListener.h - Exposes events from JIT compilation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the JITEventListener interface, which lets users get
// callbacks when significant events happen during the JIT compilation process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_EVENTLISTENER_H
#define LLVM_EXECUTION_ENGINE_JIT_EVENTLISTENER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/DebugLoc.h"

#include <vector>

namespace llvm {
class Function;
class MachineFunction;

/// Empty for now, but this object will contain all details about the
/// generated machine code that a Listener might care about.
struct JITEvent_EmittedFunctionDetails {
  const MachineFunction *MF;

  struct LineStart {
    // The address at which the current line changes.
    uintptr_t Address;
    // The new location information.  These can be translated to
    // DebugLocTuples using MF->getDebugLocTuple().
    DebugLoc Loc;
  };
  // This holds line boundary information sorted by address.
  std::vector<LineStart> LineStarts;
};

/// JITEventListener - This interface is used by the JIT to notify clients about
/// significant events during compilation.  For example, we could have
/// implementations for profilers and debuggers that need to know where
/// functions have been emitted.
///
/// Each method defaults to doing nothing, so you only need to override the ones
/// you care about.
class JITEventListener {
public:
  JITEventListener() {}
  virtual ~JITEventListener();  // Defined in JIT.cpp.

  typedef JITEvent_EmittedFunctionDetails EmittedFunctionDetails;
  /// NotifyFunctionEmitted - Called after a function has been successfully
  /// emitted to memory.  The function still has its MachineFunction attached,
  /// if you should happen to need that.
  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *Code, size_t Size,
                                     const EmittedFunctionDetails &Details) {}

  /// NotifyFreeingMachineCode - This is called inside of
  /// freeMachineCodeForFunction(), after the global mapping is removed, but
  /// before the machine code is returned to the allocator.  OldPtr is the
  /// address of the machine code.
  virtual void NotifyFreeingMachineCode(const Function &F, void *OldPtr) {}
};

// These return NULL if support isn't available.
JITEventListener *createMacOSJITEventListener();
JITEventListener *createOProfileJITEventListener();

} // end namespace llvm.

#endif
