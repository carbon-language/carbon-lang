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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
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
    const MRegisterInfo &MRI;
    SSARegMap *RegMap;
    
    std::map<SDNode *, unsigned> EmittedOps;
  public:
    SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
      : DAG(D), BB(bb), TM(D.getTarget()), TII(*TM.getInstrInfo()),
        MRI(*TM.getRegisterInfo()), RegMap(BB->getParent()->getSSARegMap()) {
      assert(&TII && "Target doesn't provide instr info?");
      assert(&MRI && "Target doesn't provide register info?");
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
    unsigned NumResults = Op.Val->getNumValues();
    if (NumResults && Op.getOperand(NumResults-1).getValueType() == MVT::Other)
      --NumResults;
#ifndef _NDEBUG
    unsigned Operands = Op.getNumOperands();
    if (Operands && Op.getOperand(Operands-1).getValueType() == MVT::Other)
      --Operands;
    assert(unsigned(II.numOperands) == Operands+NumResults &&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, II.numOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults) {
      // Create the result registers for this node and add the result regs to
      // the machine instruction.
      const TargetOperandInfo *OpInfo = II.OpInfo;
      ResultReg = RegMap->createVirtualRegister(OpInfo[0].RegClass);
      MI->addRegOperand(ResultReg, MachineOperand::Def);
      for (unsigned i = 1; i != NumResults; ++i) {
        assert(OpInfo[i].RegClass && "Isn't a register operand!");
        MI->addRegOperand(RegMap->createVirtualRegister(OpInfo[0].RegClass),
                          MachineOperand::Def);
      }
    }
    
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
    default:
      Op.Val->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: break;
    case ISD::CopyToReg: {
      unsigned Val = Emit(Op.getOperand(2));
      MRI.copyRegToReg(*BB, BB->end(),
                       cast<RegisterSDNode>(Op.getOperand(1))->getReg(), Val,
                       RegMap->getRegClass(Val));
      break;
    }
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
