//===---------------------- Stage.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a stage.
/// A chain of stages compose an instruction pipeline.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_STAGES_STAGE_H
#define LLVM_MCA_STAGES_STAGE_H

#include "llvm/MCA/HWEventListener.h"
#include "llvm/Support/Error.h"
#include <set>

namespace llvm {
namespace mca {

class InstRef;

class Stage {
  Stage *NextInSequence;
  std::set<HWEventListener *> Listeners;

  Stage(const Stage &Other) = delete;
  Stage &operator=(const Stage &Other) = delete;

protected:
  const std::set<HWEventListener *> &getListeners() const { return Listeners; }

public:
  Stage() : NextInSequence(nullptr) {}
  virtual ~Stage();

  /// Returns true if it can execute IR during this cycle.
  virtual bool isAvailable(const InstRef &IR) const { return true; }

  /// Returns true if some instructions are still executing this stage.
  virtual bool hasWorkToComplete() const = 0;

  /// Called once at the start of each cycle.  This can be used as a setup
  /// phase to prepare for the executions during the cycle.
  virtual Error cycleStart() { return ErrorSuccess(); }

  /// Called once at the end of each cycle.
  virtual Error cycleEnd() { return ErrorSuccess(); }

  /// The primary action that this stage performs on instruction IR.
  virtual Error execute(InstRef &IR) = 0;

  void setNextInSequence(Stage *NextStage) {
    assert(!NextInSequence && "This stage already has a NextInSequence!");
    NextInSequence = NextStage;
  }

  bool checkNextStage(const InstRef &IR) const {
    return NextInSequence && NextInSequence->isAvailable(IR);
  }

  /// Called when an instruction is ready to move the next pipeline stage.
  ///
  /// Stages are responsible for moving instructions to their immediate
  /// successor stages.
  Error moveToTheNextStage(InstRef &IR) {
    assert(checkNextStage(IR) && "Next stage is not ready!");
    return NextInSequence->execute(IR);
  }

  /// Add a listener to receive callbacks during the execution of this stage.
  void addListener(HWEventListener *Listener);

  /// Notify listeners of a particular hardware event.
  template <typename EventT> void notifyEvent(const EventT &Event) const {
    for (HWEventListener *Listener : Listeners)
      Listener->onEvent(Event);
  }
};

} // namespace mca
} // namespace llvm
#endif // LLVM_MCA_STAGES_STAGE_H
