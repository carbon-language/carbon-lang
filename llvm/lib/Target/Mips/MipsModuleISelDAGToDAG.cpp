//===----------------------------------------------------------------------===//
// Instruction Selector Subtarget Control
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file defines a pass used to change the subtarget for the
// Mips Instruction selector.
//
//===----------------------------------------------------------------------===//

#include "MipsISelDAGToDAG.h"
#include "MipsModuleISelDAGToDAG.h"
#include "MipsTargetMachine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "mips-isel"

namespace {
//===----------------------------------------------------------------------===//
// MipsModuleDAGToDAGISel - MIPS specific code to select MIPS machine
// instructions for SelectionDAG operations.
//===----------------------------------------------------------------------===//
class MipsModuleDAGToDAGISel : public MachineFunctionPass {
public:

  static char ID;

  explicit MipsModuleDAGToDAGISel(MipsTargetMachine &TM_)
      : MachineFunctionPass(ID), TM(TM_) {}

  // Pass Name
  const char *getPassName() const override {
    return "MIPS DAG->DAG Pattern Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

protected:
  MipsTargetMachine &TM;
};
} // namespace

bool MipsModuleDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(errs() << "In MipsModuleDAGToDAGISel::runMachineFunction\n");
  TM.resetSubtarget(&MF);
  return false;
}

char MipsModuleDAGToDAGISel::ID = 0;

llvm::FunctionPass *llvm::createMipsModuleISelDag(MipsTargetMachine &TM) {
  return new MipsModuleDAGToDAGISel(TM);
}


