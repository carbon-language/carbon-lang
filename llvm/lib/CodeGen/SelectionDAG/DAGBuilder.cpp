//===-- DAGBuilder.cpp - Turn an LLVM BasicBlock into a DAG for selection -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file turns an LLVM BasicBlock into a target independent SelectionDAG in
// preparation for target specific optimizations and instruction selection.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/InstVisitor.h"
using namespace llvm;

namespace llvm {
struct SelectionDAGBuilder : public InstVisitor<SelectionDAGBuilder> {
  // DAG - the current dag we are building.
  SelectionDAG &DAG;

  // SDTB - The target-specific builder interface, which indicates how to expand
  // extremely target-specific aspects of the representation, such as function
  // calls and arguments.
  SelectionDAGTargetBuilder &SDTB;

  // BB - The current machine basic block we are working on.
  MachineBasicBlock *BB;

  // CurRoot - The root built for the current basic block.
  SelectionDAGNode *CurRoot;

  SelectionDAGBuilder(SelectionDAG &dag, SelectionDAGTargetBuilder &sdtb)
    : DAG(dag), SDTB(sdtb), BB(0), CurRoot(0) {}

  void visitBB(BasicBlock &bb);

  // Visitation methods for instructions: Create the appropriate DAG nodes for
  // the instruction.
  void visitAdd(BinaryOperator &BO);
  void visitSub(BinaryOperator &BO);
  void visitMul(BinaryOperator &BO);

  void visitAnd(BinaryOperator &BO);
  void visitOr (BinaryOperator &BO);
  void visitXor(BinaryOperator &BO);

  void visitSetEQ(BinaryOperator &BO);

  void visitLoad(LoadInst &LI);
  void visitCall(CallInst &CI);

  void visitBr(BranchInst &BI);
  void visitRet(ReturnInst &RI);

  void visitInstruction(Instruction &I) {
    std::cerr << "DAGBuilder: Cannot instruction select: " << I;
    abort();
  }

private:
  SelectionDAGNode *getNodeFor(Value *V);
  SelectionDAGNode *getNodeFor(Value &V) { return getNodeFor(&V); }
  
  SelectionDAGNode *addSeqNode(SelectionDAGNode *N);
};
}  // end llvm namespace

/// addSeqNode - The same as addNode, but the node is also included in the
/// sequence nodes for this block.  This method should be called for any
/// instructions which have a specified sequence they must be evaluated in.
///
SelectionDAGNode *SelectionDAGBuilder::addSeqNode(SelectionDAGNode *N) {
  DAG.addNode(N);   // First, add the node to the selection DAG
  
  if (!CurRoot)
    CurRoot = N;
  else {
    // Create and add a new chain node for the existing root and this node...
    CurRoot = DAG.addNode(new SelectionDAGNode(ISD::ChainNode, MVT::isVoid,
                                               BB, CurRoot, N));
  }
  return N;
}

/// getNodeFor - This method returns the SelectionDAGNode for the specified LLVM
/// value, creating a node as necessary.
///
SelectionDAGNode *SelectionDAGBuilder::getNodeFor(Value *V) {
  // If we already have the entry, return it.
  SelectionDAGNode*& Entry = DAG.ValueMap[V];
  if (Entry) return Entry;
  
  // Otherwise, we need to create a node to return now... start by figuring out
  // which type the node will be...
  MVT::ValueType ValueType = DAG.getValueType(V->getType());

  if (Instruction *I = dyn_cast<Instruction>(V))
    // Instructions will be filled in later.  For now, just create and return a
    // dummy node.
    return Entry = new SelectionDAGNode(ISD::ProtoNode, ValueType);

  if (Constant *C = dyn_cast<Constant>(V)) {
    if (ConstantBool *CB = dyn_cast<ConstantBool>(C)) {
      Entry = new SelectionDAGNode(ISD::Constant, ValueType);
      Entry->addValue(new ReducedValue_Constant_i1(CB->getValue()));
    } else if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      Entry = new SelectionDAGNode(ISD::Constant, ValueType);
      switch (ValueType) {
      case MVT::i8:
        Entry->addValue(new ReducedValue_Constant_i8(CI->getRawValue()));
        break;
      case MVT::i16:
        Entry->addValue(new ReducedValue_Constant_i16(CI->getRawValue()));
        break;
      case MVT::i32:
        Entry->addValue(new ReducedValue_Constant_i32(CI->getRawValue()));
        break;
      case MVT::i64:
        Entry->addValue(new ReducedValue_Constant_i64(CI->getRawValue()));
        break;
      default:
        assert(0 && "Invalid ValueType for an integer constant!");
      }

    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      Entry = new SelectionDAGNode(ISD::Constant, ValueType);
      if (ValueType == MVT::f32)
        Entry->addValue(new ReducedValue_Constant_f32(CFP->getValue()));
      else
        Entry->addValue(new ReducedValue_Constant_f64(CFP->getValue()));
    }
    if (Entry) return Entry;
  } else if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    Entry = new SelectionDAGNode(ISD::BasicBlock, ValueType);
    Entry->addValue(new ReducedValue_BasicBlock_i32(DAG.BlockMap[BB]));
    return Entry;
  }

  std::cerr << "Unhandled LLVM value in DAG Builder!: " << *V << "\n";
  abort();
  return 0;
}


// visitBB - This method is used to visit a basic block in the program.  It
// manages the CurRoot instance variable so that all of the visit(Instruction)
// methods can be written to assume that there is only one basic block being
// constructed.
//
void SelectionDAGBuilder::visitBB(BasicBlock &bb) {
  BB = DAG.BlockMap[&bb];       // Update BB instance var
  
  // Save the current global DAG...
  SelectionDAGNode *OldRoot = CurRoot;
  CurRoot = 0;
  
  visit(bb.begin(), bb.end());  // Visit all of the instructions...
  
  if (OldRoot) {
    if (!CurRoot)
      CurRoot = OldRoot;   // This block had no root of its own..
    else {
      // The previous basic block AND this basic block had roots, insert a
      // block chain node now...
      CurRoot = DAG.addNode(new SelectionDAGNode(ISD::BlockChainNode,
                                                 MVT::isVoid,
                                                 BB, OldRoot, CurRoot));
    }
  }
}

//===----------------------------------------------------------------------===//
//                         ...Visitation Methods...
//===----------------------------------------------------------------------===//

void SelectionDAGBuilder::visitAdd(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::Plus, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}
void SelectionDAGBuilder::visitSub(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::Minus, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}
void SelectionDAGBuilder::visitMul(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::Times, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}

void SelectionDAGBuilder::visitAnd(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::And, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}
void SelectionDAGBuilder::visitOr(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::Or, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}
void SelectionDAGBuilder::visitXor(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::Xor, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}
void SelectionDAGBuilder::visitSetEQ(BinaryOperator &BO) {
  getNodeFor(BO)->setNode(ISD::SetEQ, BB, getNodeFor(BO.getOperand(0)),
                          getNodeFor(BO.getOperand(1)));
}


void SelectionDAGBuilder::visitRet(ReturnInst &RI) {
  if (RI.getNumOperands()) {         // Value return
    addSeqNode(new SelectionDAGNode(ISD::Ret, MVT::isVoid, BB,
                                    getNodeFor(RI.getOperand(0))));
  } else {                           // Void return
    addSeqNode(new SelectionDAGNode(ISD::RetVoid, MVT::isVoid, BB));
  }
}


void SelectionDAGBuilder::visitBr(BranchInst &BI) {
  if (BI.isUnconditional())
    addSeqNode(new SelectionDAGNode(ISD::Br, MVT::isVoid, BB,
                                    getNodeFor(BI.getOperand(0))));
  else
    addSeqNode(new SelectionDAGNode(ISD::BrCond, MVT::isVoid, BB,
                                    getNodeFor(BI.getCondition()),
                                    getNodeFor(BI.getSuccessor(0)),
                                    getNodeFor(BI.getSuccessor(1))));
}


void SelectionDAGBuilder::visitLoad(LoadInst &LI) {
  // FIXME: this won't prevent reordering of loads!
  getNodeFor(LI)->setNode(ISD::Load, BB, getNodeFor(LI.getOperand(0)));  
}

void SelectionDAGBuilder::visitCall(CallInst &CI) {
  SDTB.expandCall(DAG, CI);
}



// SelectionDAG constructor - Just use the SelectionDAGBuilder to do all of the
// dirty work...
SelectionDAG::SelectionDAG(MachineFunction &f, const TargetMachine &tm,
                           SelectionDAGTargetBuilder &SDTB)
  : F(f), TM(tm) {

  switch (TM.getTargetData().getPointerSize()) {
  default: assert(0 && "Unknown pointer size!"); abort();
  case 1: PointerType = MVT::i8;  break;
  case 2: PointerType = MVT::i16; break;
  case 3: PointerType = MVT::i32; break;
  case 4: PointerType = MVT::i64; break;
  }

  // Create all of the machine basic blocks for the function... building the
  // BlockMap.  This map is used for PHI node conversion.
  const Function &Fn = *F.getFunction();
  for (Function::const_iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    F.getBasicBlockList().push_back(BlockMap[I] = new MachineBasicBlock(I));

  SDTB.expandArguments(*this);

  SelectionDAGBuilder SDB(*this, SDTB);
  for (Function::const_iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    SDB.visitBB(const_cast<BasicBlock&>(*I));
  Root = SDB.CurRoot;
}

