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
  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *Code, size_t Size,
                                     const EmittedFunctionDetails &Details) {}

  /// NotifyFreeingMachineCode - Called from freeMachineCodeForFunction(), after
  /// the global mapping is removed, but before the machine code is returned to
  /// the allocator.
  ///
  /// OldPtr is the address of the machine code and will be the same as the Code
  /// parameter to a previous NotifyFunctionEmitted call.  The Function passed
  /// to NotifyFunctionEmitted may have been destroyed by the time of the
  /// matching NotifyFreeingMachineCode call.
  virtual void NotifyFreeingMachineCode(void *OldPtr) {}
};

// This returns NULL if support isn't available.
JITEventListener *createOProfileJITEventListener();

} // end namespace llvm.

#endif
