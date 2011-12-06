//===- DFAPacketizerEmitter.h - Packetization DFA for a VLIW machine-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class parses the Schedule.td file and produces an API that can be used
// to reason about whether an instruction can be added to a packet on a VLIW
// architecture. The class internally generates a deterministic finite
// automaton (DFA) that models all possible mappings of machine instructions
// to functional units as instructions are added to a packet.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>

namespace llvm {
//
// class DFAGen: class that generates and prints out the DFA for resource
// tracking.
//
class DFAGen : public TableGenBackend {
private:
  std::string TargetName;
  //
  // allInsnClasses is the set of all possible resources consumed by an
  // InstrStage.
  //
  DenseSet<unsigned> allInsnClasses;
  RecordKeeper &Records;

public:
  DFAGen(RecordKeeper &R);

  //
  // collectAllInsnClasses: Populate allInsnClasses which is a set of units
  // used in each stage.
  //
  void collectAllInsnClasses(const std::string &Name,
                            Record *ItinData,
                            unsigned &NStages,
                            raw_ostream &OS);

  void run(raw_ostream &OS);
};
}
