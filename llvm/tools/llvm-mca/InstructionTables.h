//===--------------------- InstructionTables.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a custom driver to generate instruction tables.
/// See the description of command-line flag -instruction-tables in
/// docs/CommandGuide/lvm-mca.rst
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H
#define LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H

#include "HWEventListener.h"
#include "InstrBuilder.h"
#include "SourceMgr.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

class InstructionTables {
  const llvm::MCSchedModel &SM;
  InstrBuilder &IB;
  SourceMgr &S;
  std::set<HWEventListener *> Listeners;

public:
  InstructionTables(const llvm::MCSchedModel &Model, InstrBuilder &Builder,
                    SourceMgr &Source)
      : SM(Model), IB(Builder), S(Source) {}

  void addEventListener(HWEventListener *Listener) {
    if (Listener)
      Listeners.insert(Listener);
  }

  void run();
};
} // namespace mca

#endif
