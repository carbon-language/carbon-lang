//===-- llvm/IR/ModuleSlotTracker.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MODULESLOTTRACKER_H
#define LLVM_IR_MODULESLOTTRACKER_H

#include <memory>

namespace llvm {

class Module;
class Function;
class SlotTracker;

/// Manage lifetime of a slot tracker for printing IR.
///
/// Wrapper around the \a SlotTracker used internally by \a AsmWriter.  This
/// class allows callers to share the cost of incorporating the metadata in a
/// module or a function.
///
/// If the IR changes from underneath \a ModuleSlotTracker, strings like
/// "<badref>" will be printed, or, worse, the wrong slots entirely.
class ModuleSlotTracker {
  /// Storage for a slot tracker.
  std::unique_ptr<SlotTracker> MachineStorage;

  const Module *M = nullptr;
  const Function *F = nullptr;
  SlotTracker *Machine = nullptr;

public:
  /// Wrap a preinitialized SlotTracker.
  ModuleSlotTracker(SlotTracker &Machine, const Module *M,
                    const Function *F = nullptr);

  /// Construct a slot tracker from a module.
  ///
  /// If \a M is \c nullptr, uses a null slot tracker.
  explicit ModuleSlotTracker(const Module *M);

  /// Destructor to clean up storage.
  ~ModuleSlotTracker();

  SlotTracker *getMachine() const { return Machine; }
  const Module *getModule() const { return M; }
  const Function *getCurrentFunction() const { return F; }

  /// Incorporate the given function.
  ///
  /// Purge the currently incorporated function and incorporate \c F.  If \c F
  /// is currently incorporated, this is a no-op.
  void incorporateFunction(const Function &F);
};

} // end namespace llvm

#endif
