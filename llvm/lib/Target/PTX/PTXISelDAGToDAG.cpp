//===-- PTXISelDAGToDAG.cpp - A dag to dag inst selector for PTX ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"

using namespace llvm;

namespace {
// PTXDAGToDAGISel - PTX specific code to select PTX machine
// instructions for SelectionDAG operations.
class PTXDAGToDAGISel : public SelectionDAGISel {
  public:
    PTXDAGToDAGISel(PTXTargetMachine &TM, CodeGenOpt::Level OptLevel);

    virtual const char *getPassName() const {
      return "PTX DAG->DAG Pattern Instruction Selection";
    }

    SDNode *Select(SDNode *Node);

    // Include the pieces auto'gened from the target description
#include "PTXGenDAGISel.inc"

}; // class PTXDAGToDAGISel
} // namespace

// createPTXISelDag - This pass converts a legalized DAG into a
// PTX-specific DAG, ready for instruction scheduling
FunctionPass *llvm::createPTXISelDag(PTXTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel) {
  return new PTXDAGToDAGISel(TM, OptLevel);
}

PTXDAGToDAGISel::PTXDAGToDAGISel(PTXTargetMachine &TM,
                                 CodeGenOpt::Level OptLevel)
  : SelectionDAGISel(TM, OptLevel) {}

SDNode *PTXDAGToDAGISel::Select(SDNode *Node) {
  // SelectCode() is auto'gened
  return SelectCode(Node);
}
