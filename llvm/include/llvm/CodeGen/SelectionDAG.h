//===-- llvm/CodeGen/SelectionDAG.h - InstSelection DAG ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the SelectionDAG class, and transitively defines the
// SDNode class and subclasses.
//   
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAG_H
#define LLVM_CODEGEN_SELECTIONDAG_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include <map>
#include <string> // FIXME remove eventually, turning map into const char* map.

namespace llvm {
  class TargetLowering;
  class TargetMachine;
  class MachineFunction;

/// SelectionDAG class - This is used to represent a portion of an LLVM function
/// in a low-level Data Dependence DAG representation suitable for instruction
/// selection.  This DAG is constructed as the first step of instruction
/// selection in order to allow implementation of machine specific optimizations
/// and code simplifications.
///
/// The representation used by the SelectionDAG is a target-independent
/// representation, which has some similarities to the GCC RTL representation,
/// but is significantly more simple, powerful, and is a graph form instead of a
/// linear form.
///
class SelectionDAG {
  const TargetMachine &TM;
  MachineFunction &MF;

  // Root - The root of the entire DAG.  EntryNode - The starting token.
  SDOperand Root, EntryNode;

  // AllNodes - All of the nodes in the DAG
  std::vector<SDNode*> AllNodes;

  // Maps to auto-CSE operations.
  std::map<std::pair<unsigned, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> UnaryOps;
  std::map<std::pair<unsigned, std::pair<SDOperand, SDOperand> >,
           SDNode *> BinaryOps;

  std::map<std::pair<std::pair<SDOperand, SDOperand>, ISD::CondCode>,
           SetCCSDNode*> SetCCs;

  std::map<std::pair<SDOperand, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> Loads;

  std::map<const GlobalValue*, SDNode*> GlobalValues;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> Constants;
  std::map<std::pair<double, MVT::ValueType>, SDNode*> ConstantFPs;
  std::map<int, SDNode*> FrameIndices;
  std::map<unsigned, SDNode*> ConstantPoolIndices;
  std::map<MachineBasicBlock *, SDNode*> BBNodes;
  std::map<std::string, SDNode*> ExternalSymbols;
public:
  SelectionDAG(const TargetMachine &tm, MachineFunction &mf) : TM(tm), MF(mf) {
    EntryNode = Root = getNode(ISD::EntryToken, MVT::Other);
  }
  ~SelectionDAG();

  MachineFunction &getMachineFunction() const { return MF; }
  const TargetMachine &getTarget() { return TM; }

  /// getRoot - Return the root tag of the SelectionDAG.
  ///
  const SDOperand &getRoot() const { return Root; }

  /// getEntryNode - Return the token chain corresponding to the entry of the
  /// function.
  const SDOperand &getEntryNode() const { return EntryNode; }

  /// setRoot - Set the current root tag of the SelectionDAG.
  ///
  const SDOperand &setRoot(SDOperand N) { return Root = N; }

  /// Legalize - This transforms the SelectionDAG into a SelectionDAG that is
  /// compatible with the target instruction selector, as indicated by the
  /// TargetLowering object.
  ///
  /// Note that this is an involved process that may invalidate pointers into
  /// the graph.
  void Legalize(TargetLowering &TLI);

  SDOperand getConstant(uint64_t Val, MVT::ValueType VT);
  SDOperand getConstantFP(double Val, MVT::ValueType VT);
  SDOperand getGlobalAddress(const GlobalValue *GV, MVT::ValueType VT);
  SDOperand getFrameIndex(int FI, MVT::ValueType VT);
  SDOperand getConstantPool(unsigned CPIdx, MVT::ValueType VT);
  SDOperand getBasicBlock(MachineBasicBlock *MBB);
  SDOperand getExternalSymbol(const char *Sym, MVT::ValueType VT);

  SDOperand getCopyToReg(SDOperand Chain, SDOperand N, unsigned VReg) {
    // Note: these are auto-CSE'd because the caller doesn't make requests that
    // could cause duplicates to occur.
    SDNode *NN = new CopyRegSDNode(Chain, N, VReg);
    AllNodes.push_back(NN);
    return SDOperand(NN, 0);
  }

  SDOperand getCopyFromReg(unsigned VReg, MVT::ValueType VT) {
    // Note: These nodes are auto-CSE'd by the caller of this method.
    SDNode *NN = new CopyRegSDNode(VReg, VT);
    AllNodes.push_back(NN);
    return SDOperand(NN, 0);
  }

  /// getCall - Note that this destroys the vector of RetVals passed in.
  ///
  SDNode *getCall(std::vector<MVT::ValueType> &RetVals, SDOperand Chain,
                  SDOperand Callee) {
    SDNode *NN = new SDNode(ISD::CALL, Chain, Callee);
    NN->setValueTypes(RetVals);
    AllNodes.push_back(NN);
    return NN;
  }

  SDOperand getSetCC(ISD::CondCode, SDOperand LHS, SDOperand RHS);

  /// getNode - Gets or creates the specified node.
  ///
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT, SDOperand N);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2, SDOperand N3);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    std::vector<SDOperand> &Children);

  /// getLoad - Loads are not normal binary operators: their result type is not
  /// determined by their operands, and they produce a value AND a token chain.
  ///
  SDOperand getLoad(MVT::ValueType VT, SDOperand Chain, SDOperand Ptr);

  void replaceAllUsesWith(SDOperand Old, SDOperand New) {
    assert(Old != New && "RAUW self!");
    assert(0 && "Unimplemented!");
  }

  void dump() const;
};

}

#endif
