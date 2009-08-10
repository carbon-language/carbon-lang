//=- llvm/CodeGen/SimpleHazardRecognizer.h - Scheduling Support -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SimpleHazardRecognizer class, which
// implements hazard-avoidance heuristics for scheduling, based on the
// scheduling itineraries specified for the target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SIMPLEHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_SIMPLEHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
  /// SimpleHazardRecognizer - A *very* simple hazard recognizer. It uses
  /// a coarse classification and attempts to avoid that instructions of
  /// a given class aren't grouped too densely together.
  class SimpleHazardRecognizer : public ScheduleHazardRecognizer {
    /// Class - A simple classification for SUnits.
    enum Class {
      Other, Load, Store
    };

    /// Window - The Class values of the most recently issued
    /// instructions.
    Class Window[8];

    /// getClass - Classify the given SUnit.
    Class getClass(const SUnit *SU) {
      const MachineInstr *MI = SU->getInstr();
      const TargetInstrDesc &TID = MI->getDesc();
      if (TID.mayLoad())
        return Load;
      if (TID.mayStore())
        return Store;
      return Other;
    }

    /// Step - Rotate the existing entries in Window and insert the
    /// given class value in position as the most recent.
    void Step(Class C) {
      std::copy(Window+1, array_endof(Window), Window);
      Window[array_lengthof(Window)-1] = C;
    }

  public:
    SimpleHazardRecognizer() : Window() {
      Reset();
    }

    virtual HazardType getHazardType(SUnit *SU) {
      Class C = getClass(SU);
      if (C == Other)
        return NoHazard;
      unsigned Score = 0;
      for (unsigned i = 0; i != array_lengthof(Window); ++i)
        if (Window[i] == C)
          Score += i + 1;
      if (Score > array_lengthof(Window) * 2)
        return Hazard;
      return NoHazard;
    }

    virtual void Reset() {
      for (unsigned i = 0; i != array_lengthof(Window); ++i)
        Window[i] = Other;
    }

    virtual void EmitInstruction(SUnit *SU) {
      Step(getClass(SU));
    }

    virtual void AdvanceCycle() {
      Step(Other);
    }
  };
}

#endif
