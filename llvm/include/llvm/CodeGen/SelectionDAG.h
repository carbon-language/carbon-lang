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
  TargetLowering &TLI;
  MachineFunction &MF;

  // Root - The root of the entire DAG.  EntryNode - The starting token.
  SDOperand Root, EntryNode;

  // AllNodes - All of the nodes in the DAG
  std::vector<SDNode*> AllNodes;
public:
  SelectionDAG(TargetLowering &tli, MachineFunction &mf) : TLI(tli), MF(mf) {
    EntryNode = Root = getNode(ISD::EntryToken, MVT::Other);
  }
  ~SelectionDAG();

  MachineFunction &getMachineFunction() const { return MF; }
  const TargetMachine &getTarget() const;
  TargetLowering &getTargetLoweringInfo() const { return TLI; }

  /// viewGraph - Pop up a ghostview window with the DAG rendered using 'dot'.
  ///
  void viewGraph();


  typedef std::vector<SDNode*>::const_iterator allnodes_iterator;
  allnodes_iterator allnodes_begin() const { return AllNodes.begin(); }
  allnodes_iterator allnodes_end() const { return AllNodes.end(); }

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
  void Legalize();

  /// RemoveDeadNodes - This method deletes all unreachable nodes in the
  /// SelectionDAG, including nodes (like loads) that have uses of their token
  /// chain but no other uses and no side effect.  If a node is passed in as an
  /// argument, it is used as the seed for node deletion.
  void RemoveDeadNodes(SDNode *N = 0);

  SDOperand getConstant(uint64_t Val, MVT::ValueType VT);
  SDOperand getConstantFP(double Val, MVT::ValueType VT);
  SDOperand getGlobalAddress(const GlobalValue *GV, MVT::ValueType VT);
  SDOperand getFrameIndex(int FI, MVT::ValueType VT);
  SDOperand getConstantPool(unsigned CPIdx, MVT::ValueType VT);
  SDOperand getBasicBlock(MachineBasicBlock *MBB);
  SDOperand getExternalSymbol(const char *Sym, MVT::ValueType VT);

  SDOperand getCopyToReg(SDOperand Chain, SDOperand N, unsigned Reg) {
    // Note: these are auto-CSE'd because the caller doesn't make requests that
    // could cause duplicates to occur.
    SDNode *NN = new RegSDNode(ISD::CopyToReg, Chain, N, Reg);
    NN->setValueTypes(MVT::Other);
    AllNodes.push_back(NN);
    return SDOperand(NN, 0);
  }

  SDOperand getCopyFromReg(unsigned Reg, MVT::ValueType VT, SDOperand Chain) {
    // Note: These nodes are auto-CSE'd by the caller of this method.
    SDNode *NN = new RegSDNode(ISD::CopyFromReg, Chain, Reg);
    NN->setValueTypes(VT, MVT::Other);
    AllNodes.push_back(NN);
    return SDOperand(NN, 0);
  }

  SDOperand getImplicitDef(SDOperand Chain, unsigned Reg) {
    // Note: These nodes are auto-CSE'd by the caller of this method.
    SDNode *NN = new RegSDNode(ISD::ImplicitDef, Chain, Reg);
    NN->setValueTypes(MVT::Other);
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

  /// getCall - This is identical to the one above, and should be used for calls
  /// where arguments are passed in physical registers.  This destroys the
  /// RetVals and ArgsInRegs vectors.
  SDNode *getCall(std::vector<MVT::ValueType> &RetVals, SDOperand Chain,
                  SDOperand Callee, std::vector<SDOperand> &ArgsInRegs) {
    ArgsInRegs.insert(ArgsInRegs.begin(), Callee);
    ArgsInRegs.insert(ArgsInRegs.begin(), Chain);
    SDNode *NN = new SDNode(ISD::CALL, ArgsInRegs);
    NN->setValueTypes(RetVals);
    AllNodes.push_back(NN);
    return NN;
  }


  SDOperand getSetCC(ISD::CondCode, MVT::ValueType VT,
                     SDOperand LHS, SDOperand RHS);

  /// getZeroExtendInReg - Return the expression required to zero extend the Op
  /// value assuming it was the smaller SrcTy value.
  SDOperand getZeroExtendInReg(SDOperand Op, MVT::ValueType SrcTy);

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

  // getNode - These versions take an extra value type for extending and
  // truncating loads, stores, rounds, extends etc.
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT, SDOperand N1,
                    SDOperand N2, MVT::ValueType EVT);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N, MVT::ValueType EVT);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT, SDOperand N1,
                    SDOperand N2, SDOperand N3, MVT::ValueType EVT);

  /// getLoad - Loads are not normal binary operators: their result type is not
  /// determined by their operands, and they produce a value AND a token chain.
  ///
  SDOperand getLoad(MVT::ValueType VT, SDOperand Chain, SDOperand Ptr);

  void replaceAllUsesWith(SDOperand Old, SDOperand New) {
    assert(Old != New && "RAUW self!");
    assert(0 && "Unimplemented!");
  }

  void dump() const;

private:
  void DeleteNodeIfDead(SDNode *N, void *NodeSet);

  // Maps to auto-CSE operations.
  std::map<std::pair<unsigned, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> UnaryOps;
  std::map<std::pair<unsigned, std::pair<SDOperand, SDOperand> >,
           SDNode *> BinaryOps;

  std::map<std::pair<std::pair<SDOperand, SDOperand>,
                     std::pair<ISD::CondCode, MVT::ValueType> >,
           SetCCSDNode*> SetCCs;

  std::map<std::pair<SDOperand, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> Loads;

  std::map<const GlobalValue*, SDNode*> GlobalValues;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> Constants;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> ConstantFPs;
  std::map<int, SDNode*> FrameIndices;
  std::map<unsigned, SDNode*> ConstantPoolIndices;
  std::map<MachineBasicBlock *, SDNode*> BBNodes;
  std::map<std::string, SDNode*> ExternalSymbols;
  struct EVTStruct {
    unsigned Opcode;
    MVT::ValueType VT, EVT;
    std::vector<SDOperand> Ops;
    bool operator<(const EVTStruct &RHS) const {
      if (Opcode < RHS.Opcode) return true;
      if (Opcode > RHS.Opcode) return false;
      if (VT < RHS.VT) return true;
      if (VT > RHS.VT) return false;
      if (EVT < RHS.EVT) return true;
      if (EVT > RHS.EVT) return false;
      return Ops < RHS.Ops;
    }
  };
  std::map<EVTStruct, SDNode*> MVTSDNodes;
};

template <> struct GraphTraits<SelectionDAG*> : public GraphTraits<SDNode*> {
  typedef SelectionDAG::allnodes_iterator nodes_iterator;
  static nodes_iterator nodes_begin(SelectionDAG *G) {
    return G->allnodes_begin();
  }
  static nodes_iterator nodes_end(SelectionDAG *G) {
    return G->allnodes_end();
  }
};

}

#endif
