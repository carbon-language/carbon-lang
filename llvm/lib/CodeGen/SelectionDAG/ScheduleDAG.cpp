//===-- ScheduleDAG.cpp - Implement a trivial DAG scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple code linearizer for DAGs.  This is not a very good
// way to emit code, but gets working code quickly.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

#ifndef _NDEBUG
static cl::opt<bool>
ViewDAGs("view-sched-dags", cl::Hidden,
         cl::desc("Pop up a window to show sched dags as they are processed"));
#else
static const bool ViewDAGS = 0;
#endif

namespace {
  class SimpleSched {
    SelectionDAG &DAG;
    MachineBasicBlock *BB;
    const TargetMachine &TM;
    const TargetInstrInfo &TII;
    
    std::map<SDNode *, unsigned> EmittedOps;
  public:
    SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
      : DAG(D), BB(bb), TM(D.getTarget()), TII(*TM.getInstrInfo()) {
      assert(&TII && "Target doesn't provide instr info?");
    }
    
    void Run() {
      Emit(DAG.getRoot());
    }
    
  private:
    unsigned Emit(SDOperand Op);
  };
}

unsigned SimpleSched::Emit(SDOperand Op) {
  // Check to see if we have already emitted this.  If so, return the value
  // already emitted.  Note that if a node has a single use it cannot be
  // revisited, so don't bother putting it in the map.
  unsigned *OpSlot;
  if (Op.Val->hasOneUse()) {
    OpSlot = 0;  // No reuse possible.
  } else {
    std::map<SDNode *, unsigned>::iterator OpI = EmittedOps.lower_bound(Op.Val);
    if (OpI != EmittedOps.end() && OpI->first == Op.Val)
      return OpI->second + Op.ResNo;
    OpSlot = &EmittedOps.insert(OpI, std::make_pair(Op.Val, 0))->second;
  }
  
  unsigned ResultReg = 0;
  if (Op.isTargetOpcode()) {
    unsigned Opc = Op.getTargetOpcode();
    const TargetInstrDescriptor &II = TII.get(Opc);

    // Target nodes have any register or immediate operands before any chain
    // nodes.  Check that the DAG matches the TD files's expectation of #
    // operands.
    assert((unsigned(II.numOperands) == Op.getNumOperands() ||
            // It could be some number of operands followed by a token chain.
           (unsigned(II.numOperands)+1 == Op.getNumOperands() &&
            Op.getOperand(II.numOperands).getValueType() == MVT::Other)) &&
           "#operands for dag node doesn't match .td file!"); 

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, II.numOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    assert(Op.Val->getNumValues() == 1 &&
           Op.getValue(0).getValueType() == MVT::Other &&
           "Return values not implemented yet");
    
    // Emit all of the operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =dyn_cast<RegisterSDNode>(Op.getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else {
        unsigned R = Emit(Op.getOperand(i));
        // Add an operand, unless this corresponds to a chain node.
        if (Op.getOperand(i).getValueType() != MVT::Other)
          MI->addRegOperand(R, MachineOperand::Use);
      }
    }

    // Now that we have emitted all operands, emit this instruction itself.
    BB->insert(BB->end(), MI);
  } else {
    switch (Op.getOpcode()) {
    default: assert(0 &&
                    "This target-independent node should have been selected!");
    case ISD::EntryToken: break;
    }
  }
  
  if (OpSlot) *OpSlot = ResultReg;
  return ResultReg+Op.ResNo;
}


/// Pick a safe ordering and emit instructions for each target node in the
/// graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &SD) {
  if (ViewDAGs) SD.viewGraph();
  SimpleSched(SD, BB).Run();  
}
