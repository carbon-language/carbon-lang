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
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

#ifndef NDEBUG
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
    MachineConstantPool *ConstPool;
    
    std::map<SDNode *, unsigned> EmittedOps;
  public:
    SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
      : DAG(D), BB(bb), TM(D.getTarget()), TII(*TM.getInstrInfo()),
        MRI(*TM.getRegisterInfo()), RegMap(BB->getParent()->getSSARegMap()),
        ConstPool(BB->getParent()->getConstantPool()) {
      assert(&TII && "Target doesn't provide instr info?");
      assert(&MRI && "Target doesn't provide register info?");
    }
    
    MachineBasicBlock *Run() {
      Emit(DAG.getRoot());
      return BB;
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

    // The results of target nodes have register or immediate operands first,
    // then an optional chain, and optional flag operands (which do not go into
    // the machine instrs).
    unsigned NumResults = Op.Val->getNumValues();
    while (NumResults && Op.Val->getValueType(NumResults-1) == MVT::Flag)
      --NumResults;
    if (NumResults && Op.Val->getValueType(NumResults-1) == MVT::Other)
      --NumResults;    // Skip over chain result.

    // The inputs to target nodes have any actual inputs first, followed by an
    // optional chain operand, then flag operands.  Compute the number of actual
    // operands that  will go into the machine instr.
    unsigned NodeOperands = Op.getNumOperands();
    while (NodeOperands &&
           Op.getOperand(NodeOperands-1).getValueType() == MVT::Flag)
      --NodeOperands;
    
    if (NodeOperands &&    // Ignore chain if it exists.
        Op.getOperand(NodeOperands-1).getValueType() == MVT::Other)
      --NodeOperands;
   
    unsigned NumMIOperands = NodeOperands+NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
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
    
    // If there is a token chain operand, emit it first, as a hack to get avoid
    // really bad cases.
    if (Op.getNumOperands() > NodeOperands &&
        Op.getOperand(NodeOperands).getValueType() == MVT::Other)
      Emit(Op.getOperand(NodeOperands));
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i) {
      if (Op.getOperand(i).isTargetOpcode()) {
        // Note that this case is redundant with the final else block, but we
        // include it because it is the most common and it makes the logic
        // simpler here.
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        
        MI->addRegOperand(Emit(Op.getOperand(i)), MachineOperand::Use);
      } else if (ConstantSDNode *C =
                                   dyn_cast<ConstantSDNode>(Op.getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =dyn_cast<RegisterSDNode>(Op.getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else if (GlobalAddressSDNode *TGA =
                       dyn_cast<GlobalAddressSDNode>(Op.getOperand(i))) {
        MI->addGlobalAddressOperand(TGA->getGlobal(), false, 0);
      } else if (BasicBlockSDNode *BB =
                       dyn_cast<BasicBlockSDNode>(Op.getOperand(i))) {
        MI->addMachineBasicBlockOperand(BB->getBasicBlock());
      } else if (FrameIndexSDNode *FI =
                       dyn_cast<FrameIndexSDNode>(Op.getOperand(i))) {
        MI->addFrameIndexOperand(FI->getIndex());
      } else if (ConstantPoolSDNode *CP = 
                    dyn_cast<ConstantPoolSDNode>(Op.getOperand(i))) {
        unsigned Idx = ConstPool->getConstantPoolIndex(CP->get());
        MI->addConstantPoolIndexOperand(Idx);
      } else if (ExternalSymbolSDNode *ES = 
                 dyn_cast<ExternalSymbolSDNode>(Op.getOperand(i))) {
        MI->addExternalSymbolOperand(ES->getSymbol(), false);
      } else {
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        MI->addRegOperand(Emit(Op.getOperand(i)), MachineOperand::Use);
      }
    }

    // Finally, if this node has any flag operands, we *must* emit them last, to
    // avoid emitting operations that might clobber the flags.
    if (Op.getNumOperands() > NodeOperands) {
      unsigned i = NodeOperands;
      if (Op.getOperand(i).getValueType() == MVT::Other)
        ++i;  // the chain is already selected.
      for (; i != Op.getNumOperands(); ++i) {
        assert(Op.getOperand(i).getValueType() == MVT::Flag &&
               "Must be flag operands!");
        Emit(Op.getOperand(i));
      }
    }
    
    // Now that we have emitted all operands, emit this instruction itself.
    if ((II.Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION) == 0) {
      BB->insert(BB->end(), MI);
    } else {
      // Insert this instruction into the end of the basic block, potentially
      // taking some custom action.
      BB = DAG.getTargetLoweringInfo().InsertAtEndOfBasicBlock(MI, BB);
    }
  } else {
    switch (Op.getOpcode()) {
    default:
      Op.Val->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: break;
    case ISD::TokenFactor:
      for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i)
        Emit(Op.getOperand(i));
      break;
    case ISD::CopyToReg: {
      SDOperand FlagOp;
      if (Op.getNumOperands() == 4)
        FlagOp = Op.getOperand(3);
      if (Op.getOperand(0).Val != FlagOp.Val)
        Emit(Op.getOperand(0));   // Emit the chain.
      unsigned Val = Emit(Op.getOperand(2));
      if (FlagOp.Val) Emit(FlagOp);
      MRI.copyRegToReg(*BB, BB->end(),
                       cast<RegisterSDNode>(Op.getOperand(1))->getReg(), Val,
                       RegMap->getRegClass(Val));
      break;
    }
    case ISD::CopyFromReg: {
      Emit(Op.getOperand(0));   // Emit the chain.
      unsigned SrcReg = cast<RegisterSDNode>(Op.getOperand(1))->getReg();
      
      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        TRC = RegMap->getRegClass(SrcReg);
      } else {
        // FIXME: we don't know what register class to generate this for.  Do
        // a brute force search and pick the first match. :(
        for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
               E = MRI.regclass_end(); I != E; ++I)
          if ((*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      }
      
      // Create the reg, emit the copy.
      ResultReg = RegMap->createVirtualRegister(TRC);
      MRI.copyRegToReg(*BB, BB->end(), ResultReg, SrcReg, TRC);
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
  BB = SimpleSched(SD, BB).Run();  
}
