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

#include "llvm/Config/config.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/DebugLoc.h"

#include <vector>

namespace llvm {
class Function;
class MachineFunction;
class OProfileWrapper;
class IntelJITEventsWrapper;

/// JITEvent_EmittedFunctionDetails - Helper struct for containing information
/// about a generated machine code function.
struct JITEvent_EmittedFunctionDetails {
  struct LineStart {
    /// The address at which the current line changes.
    uintptr_t Address;

    /// The new location information.  These can be translated to DebugLocTuples
    /// using MF->getDebugLocTuple().
    DebugLoc Loc;
  };

  /// The machine function the struct contains information for.
  const MachineFunction *MF;

  /// The list of line boundary information, sorted by address.
  std::vector<LineStart> LineStarts;
};

/// JITEventListener - Abstract interface for use by the JIT to notify clients
/// about significant events during compilation. For example, to notify
/// profilers and debuggers that need to know where functions have been emitted.
///
/// The default implementation of each method does nothing.
class JITEventListener {
public:
  typedef JITEvent_EmittedFunctionDetails EmittedFunctionDetails;

public:
  JITEventListener() {}
  virtual ~JITEventListener();

  /// NotifyFunctionEmitted - Called after a function has been successfully
  /// emitted to memory.  The function still has its MachineFunction attached,
  /// if you should happen to need that.
  virtual void NotifyFunctionEmitted(const Function &,
                                     void *, size_t,
                                     const EmittedFunctionDetails &) {}

  /// NotifyFreeingMachineCode - Called from freeMachineCodeForFunction(), after
  /// the global mapping is removed, but before the machine code is returned to
  /// the allocator.
  ///
  /// OldPtr is the address of the machine code and will be the same as the Code
  /// parameter to a previous NotifyFunctionEmitted call.  The Function passed
  /// to NotifyFunctionEmitted may have been destroyed by the time of the
  /// matching NotifyFreeingMachineCode call.
  virtual void NotifyFreeingMachineCode(void *) {}

#if LLVM_USE_INTEL_JITEVENTS
  // Construct an IntelJITEventListener
  static JITEventListener *createIntelJITEventListener();

  // Construct an IntelJITEventListener with a test Intel JIT API implementation
  static JITEventListener *createIntelJITEventListener(
                                      IntelJITEventsWrapper* AlternativeImpl);
#else
  static JITEventListener *createIntelJITEventListener() { return 0; }

  static JITEventListener *createIntelJITEventListener(
                                      IntelJITEventsWrapper* AlternativeImpl) {
    return 0;
  }
#endif // USE_INTEL_JITEVENTS

#if LLVM_USE_OPROFILE
  // Construct an OProfileJITEventListener
  static JITEventListener *createOProfileJITEventListener();

  // Construct an OProfileJITEventListener with a test opagent implementation
  static JITEventListener *createOProfileJITEventListener(
                                      OProfileWrapper* AlternativeImpl);
#else

  static JITEventListener *createOProfileJITEventListener() { return 0; }

  static JITEventListener *createOProfileJITEventListener(
                                      OProfileWrapper* AlternativeImpl) {
    return 0;
  }
#endif // USE_OPROFILE

};

} // end namespace llvm.

#endif // defined LLVM_EXECUTION_ENGINE_JIT_EVENTLISTENER_H
