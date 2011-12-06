//=- llvm/CodeGen/DFAPacketizer.h - DFA Packetizer for VLIW ---*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class implements a deterministic finite automaton (DFA) based
// packetizing mechanism for VLIW architectures. It provides APIs to
// determine whether there exists a legal mapping of instructions to
// functional unit assignments in a packet. The DFA is auto-generated from
// the target's Schedule.td file.
//
// A DFA consists of 3 major elements: states, inputs, and transitions. For
// the packetizing mechanism, the input is the set of instruction classes for
// a target. The state models all possible combinations of functional unit
// consumption for a given set of instructions in a packet. A transition
// models the addition of an instruction to a packet. In the DFA constructed
// by this class, if an instruction can be added to a packet, then a valid
// transition exists from the corresponding state. Invalid transitions
// indicate that the instruction cannot be added to the current packet.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DFAPACKETIZER_H
#define LLVM_CODEGEN_DFAPACKETIZER_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class MCInstrDesc;
class MachineInstr;
class InstrItineraryData;

class DFAPacketizer {
private:
  typedef std::pair<unsigned, unsigned> UnsignPair;
  const InstrItineraryData *InstrItins;
  int CurrentState;
  const int (*DFAStateInputTable)[2];
  const unsigned *DFAStateEntryTable;

  // CachedTable is a map from <FromState, Input> to ToState.
  DenseMap<UnsignPair, unsigned> CachedTable;

  // ReadTable - Read the DFA transition table and update CachedTable.
  void ReadTable(unsigned int state);

public:
  DFAPacketizer(const InstrItineraryData *I, const int (*SIT)[2],
                const unsigned *SET);

  // Reset the current state to make all resources available.
  void clearResources() {
    CurrentState = 0;
  }

  // canReserveResources - Check if the resources occupied by a MCInstrDesc
  // are available in the current state.
  bool canReserveResources(const llvm::MCInstrDesc *MID);

  // reserveResources - Reserve the resources occupied by a MCInstrDesc and
  // change the current state to reflect that change.
  void reserveResources(const llvm::MCInstrDesc *MID);

  // canReserveResources - Check if the resources occupied by a machine
  // instruction are available in the current state.
  bool canReserveResources(llvm::MachineInstr *MI);

  // reserveResources - Reserve the resources occupied by a machine
  // instruction and change the current state to reflect that change.
  void reserveResources(llvm::MachineInstr *MI);
};
}

#endif
