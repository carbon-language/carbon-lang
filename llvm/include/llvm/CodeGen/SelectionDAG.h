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
#include "llvm/CodeGen/SelectionDAGCSEMap.h"
#include "llvm/ADT/ilist"

#include <list>
#include <map>
#include <set>
#include <string>

namespace llvm {
  class TargetLowering;
  class TargetMachine;
  class MachineDebugInfo;
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
  MachineDebugInfo *DI;

  /// Root - The root of the entire DAG.  EntryNode - The starting token.
  SDOperand Root, EntryNode;

  /// AllNodes - A linked list of nodes in the current DAG.
  ilist<SDNode> AllNodes;

  /// CSEMap - This structure is used to memoize nodes, automatically performing
  /// CSE with existing nodes with a duplicate is requested.
  SelectionDAGCSEMap CSEMap;

public:
  SelectionDAG(TargetLowering &tli, MachineFunction &mf, MachineDebugInfo *di)
  : TLI(tli), MF(mf), DI(di) {
    EntryNode = Root = getNode(ISD::EntryToken, MVT::Other);
  }
  ~SelectionDAG();

  MachineFunction &getMachineFunction() const { return MF; }
  const TargetMachine &getTarget() const;
  TargetLowering &getTargetLoweringInfo() const { return TLI; }
  MachineDebugInfo *getMachineDebugInfo() const { return DI; }

  /// viewGraph - Pop up a ghostview window with the DAG rendered using 'dot'.
  ///
  void viewGraph();


  typedef ilist<SDNode>::const_iterator allnodes_const_iterator;
  allnodes_const_iterator allnodes_begin() const { return AllNodes.begin(); }
  allnodes_const_iterator allnodes_end() const { return AllNodes.end(); }
  typedef ilist<SDNode>::iterator allnodes_iterator;
  allnodes_iterator allnodes_begin() { return AllNodes.begin(); }
  allnodes_iterator allnodes_end() { return AllNodes.end(); }
  
  /// getRoot - Return the root tag of the SelectionDAG.
  ///
  const SDOperand &getRoot() const { return Root; }

  /// getEntryNode - Return the token chain corresponding to the entry of the
  /// function.
  const SDOperand &getEntryNode() const { return EntryNode; }

  /// setRoot - Set the current root tag of the SelectionDAG.
  ///
  const SDOperand &setRoot(SDOperand N) { return Root = N; }

  /// Combine - This iterates over the nodes in the SelectionDAG, folding
  /// certain types of nodes together, or eliminating superfluous nodes.  When
  /// the AfterLegalize argument is set to 'true', Combine takes care not to
  /// generate any nodes that will be illegal on the target.
  void Combine(bool AfterLegalize);
  
  /// Legalize - This transforms the SelectionDAG into a SelectionDAG that is
  /// compatible with the target instruction selector, as indicated by the
  /// TargetLowering object.
  ///
  /// Note that this is an involved process that may invalidate pointers into
  /// the graph.
  void Legalize();

  /// RemoveDeadNodes - This method deletes all unreachable nodes in the
  /// SelectionDAG.
  void RemoveDeadNodes();

  SDOperand getString(const std::string &Val);
  SDOperand getConstant(uint64_t Val, MVT::ValueType VT, bool isTarget = false);
  SDOperand getTargetConstant(uint64_t Val, MVT::ValueType VT) {
    return getConstant(Val, VT, true);
  }
  SDOperand getConstantFP(double Val, MVT::ValueType VT, bool isTarget = false);
  SDOperand getTargetConstantFP(double Val, MVT::ValueType VT) {
    return getConstantFP(Val, VT, true);
  }
  SDOperand getGlobalAddress(const GlobalValue *GV, MVT::ValueType VT,
                             int offset = 0, bool isTargetGA = false);
  SDOperand getTargetGlobalAddress(const GlobalValue *GV, MVT::ValueType VT,
                                   int offset = 0) {
    return getGlobalAddress(GV, VT, offset, true);
  }
  SDOperand getFrameIndex(int FI, MVT::ValueType VT, bool isTarget = false);
  SDOperand getTargetFrameIndex(int FI, MVT::ValueType VT) {
    return getFrameIndex(FI, VT, true);
  }
  SDOperand getJumpTable(int JTI, MVT::ValueType VT, bool isTarget = false);
  SDOperand getTargetJumpTable(int JTI, MVT::ValueType VT) {
    return getJumpTable(JTI, VT, true);
  }
  SDOperand getConstantPool(Constant *C, MVT::ValueType VT,
                            unsigned Align = 0, int Offs = 0, bool isT=false);
  SDOperand getTargetConstantPool(Constant *C, MVT::ValueType VT,
                                  unsigned Align = 0, int Offset = 0) {
    return getConstantPool(C, VT, Align, Offset, true);
  }
  SDOperand getBasicBlock(MachineBasicBlock *MBB);
  SDOperand getExternalSymbol(const char *Sym, MVT::ValueType VT);
  SDOperand getTargetExternalSymbol(const char *Sym, MVT::ValueType VT);
  SDOperand getValueType(MVT::ValueType);
  SDOperand getRegister(unsigned Reg, MVT::ValueType VT);

  SDOperand getCopyToReg(SDOperand Chain, unsigned Reg, SDOperand N) {
    return getNode(ISD::CopyToReg, MVT::Other, Chain,
                   getRegister(Reg, N.getValueType()), N);
  }

  // This version of the getCopyToReg method takes an extra operand, which
  // indicates that there is potentially an incoming flag value (if Flag is not
  // null) and that there should be a flag result.
  SDOperand getCopyToReg(SDOperand Chain, unsigned Reg, SDOperand N,
                         SDOperand Flag) {
    const MVT::ValueType *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDOperand Ops[] = { Chain, getRegister(Reg, N.getValueType()), N, Flag };
    return getNode(ISD::CopyToReg, VTs, 2, Ops, Flag.Val ? 4 : 3);
  }

  // Similar to last getCopyToReg() except parameter Reg is a SDOperand
  SDOperand getCopyToReg(SDOperand Chain, SDOperand Reg, SDOperand N,
                         SDOperand Flag) {
    const MVT::ValueType *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDOperand Ops[] = { Chain, Reg, N, Flag };
    return getNode(ISD::CopyToReg, VTs, 2, Ops, Flag.Val ? 4 : 3);
  }
  
  SDOperand getCopyFromReg(SDOperand Chain, unsigned Reg, MVT::ValueType VT) {
    const MVT::ValueType *VTs = getNodeValueTypes(VT, MVT::Other);
    SDOperand Ops[] = { Chain, getRegister(Reg, VT) };
    return getNode(ISD::CopyFromReg, VTs, 2, Ops, 2);
  }
  
  // This version of the getCopyFromReg method takes an extra operand, which
  // indicates that there is potentially an incoming flag value (if Flag is not
  // null) and that there should be a flag result.
  SDOperand getCopyFromReg(SDOperand Chain, unsigned Reg, MVT::ValueType VT,
                           SDOperand Flag) {
    const MVT::ValueType *VTs = getNodeValueTypes(VT, MVT::Other, MVT::Flag);
    SDOperand Ops[] = { Chain, getRegister(Reg, VT), Flag };
    return getNode(ISD::CopyFromReg, VTs, 3, Ops, Flag.Val ? 3 : 2);
  }

  SDOperand getCondCode(ISD::CondCode Cond);

  /// getZeroExtendInReg - Return the expression required to zero extend the Op
  /// value assuming it was the smaller SrcTy value.
  SDOperand getZeroExtendInReg(SDOperand Op, MVT::ValueType SrcTy);
  
  /// getCALLSEQ_START - Return a new CALLSEQ_START node, which always must have
  /// a flag result (to ensure it's not CSE'd).
  SDOperand getCALLSEQ_START(SDOperand Chain, SDOperand Op) {
    const MVT::ValueType *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDOperand Ops[] = { Chain,  Op };
    return getNode(ISD::CALLSEQ_START, VTs, 2, Ops, 2);
  }

  /// getNode - Gets or creates the specified node.
  ///
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT, SDOperand N);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2, SDOperand N3);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4,
                    SDOperand N5);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    const SDOperand *Ops, unsigned NumOps);
  SDOperand getNode(unsigned Opcode, std::vector<MVT::ValueType> &ResultTys,
                    const SDOperand *Ops, unsigned NumOps);
  SDOperand getNode(unsigned Opcode, const MVT::ValueType *VTs, unsigned NumVTs,
                    const SDOperand *Ops, unsigned NumOps);
  
  /// getSetCC - Helper function to make it easier to build SetCC's if you just
  /// have an ISD::CondCode instead of an SDOperand.
  ///
  SDOperand getSetCC(MVT::ValueType VT, SDOperand LHS, SDOperand RHS,
                     ISD::CondCode Cond) {
    return getNode(ISD::SETCC, VT, LHS, RHS, getCondCode(Cond));
  }

  /// getSelectCC - Helper function to make it easier to build SelectCC's if you
  /// just have an ISD::CondCode instead of an SDOperand.
  ///
  SDOperand getSelectCC(SDOperand LHS, SDOperand RHS,
                        SDOperand True, SDOperand False, ISD::CondCode Cond) {
    return getNode(ISD::SELECT_CC, True.getValueType(), LHS, RHS, True, False,
                   getCondCode(Cond));
  }
  
  /// getVAArg - VAArg produces a result and token chain, and takes a pointer
  /// and a source value as input.
  SDOperand getVAArg(MVT::ValueType VT, SDOperand Chain, SDOperand Ptr,
                     SDOperand SV);

  /// getLoad - Loads are not normal binary operators: their result type is not
  /// determined by their operands, and they produce a value AND a token chain.
  ///
  SDOperand getLoad(MVT::ValueType VT, SDOperand Chain, SDOperand Ptr,
                    SDOperand SV);
  SDOperand getVecLoad(unsigned Count, MVT::ValueType VT, SDOperand Chain, 
                       SDOperand Ptr, SDOperand SV);
  SDOperand getExtLoad(unsigned Opcode, MVT::ValueType VT, SDOperand Chain,
                       SDOperand Ptr, SDOperand SV, MVT::ValueType EVT);

  // getSrcValue - construct a node to track a Value* through the backend
  SDOperand getSrcValue(const Value* I, int offset = 0);

  /// UpdateNodeOperands - *Mutate* the specified node in-place to have the
  /// specified operands.  If the resultant node already exists in the DAG,
  /// this does not modify the specified node, instead it returns the node that
  /// already exists.  If the resultant node does not exist in the DAG, the
  /// input node is returned.  As a degenerate case, if you specify the same
  /// input operands as the node already has, the input node is returned.
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand Op);
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2);
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2,
                               SDOperand Op3);
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2,
                               SDOperand Op3, SDOperand Op4);
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2,
                               SDOperand Op3, SDOperand Op4, SDOperand Op5);
  SDOperand UpdateNodeOperands(SDOperand N, SDOperand *Ops, unsigned NumOps);
  
  /// SelectNodeTo - These are used for target selectors to *mutate* the
  /// specified node to have the specified return type, Target opcode, and
  /// operands.  Note that target opcodes are stored as
  /// ISD::BUILTIN_OP_END+TargetOpcode in the node opcode field.  The 0th value
  /// of the resultant node is returned.
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3, 
                         SDOperand Op4);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3,
                         SDOperand Op4, SDOperand Op5);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3, 
                         SDOperand Op4, SDOperand Op5, SDOperand Op6);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3,
                         SDOperand Op4, SDOperand Op5, SDOperand Op6,
			 SDOperand Op7);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT, 
                         SDOperand Op1, SDOperand Op2, SDOperand Op3,
                         SDOperand Op4, SDOperand Op5, SDOperand Op6,
			 SDOperand Op7, SDOperand Op8);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT1, 
                         MVT::ValueType VT2, SDOperand Op1, SDOperand Op2);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT1,
                         MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                         SDOperand Op3);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT1,
                         MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                         SDOperand Op3, SDOperand Op4);
  SDOperand SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT::ValueType VT1,
                         MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                         SDOperand Op3, SDOperand Op4, SDOperand Op5);

  /// getTargetNode - These are used for target selectors to create a new node
  /// with specified return type(s), target opcode, and operands.
  ///
  /// Note that getTargetNode returns the resultant node.  If there is already a
  /// node of the specified opcode and operands, it returns that node instead of
  /// the current one.
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3,
                        SDOperand Op4);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3,
                        SDOperand Op4, SDOperand Op5);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3,
                        SDOperand Op4, SDOperand Op5, SDOperand Op6);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3,
                        SDOperand Op4, SDOperand Op5, SDOperand Op6,
                        SDOperand Op7);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        SDOperand Op1, SDOperand Op2, SDOperand Op3,
                        SDOperand Op4, SDOperand Op5, SDOperand Op6,
                        SDOperand Op7, SDOperand Op8);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT,
                        const SDOperand *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                        SDOperand Op3);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5,
                        SDOperand Op6);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5,
                        SDOperand Op6, SDOperand Op7);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, MVT::ValueType VT3,
                        SDOperand Op1, SDOperand Op2);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, MVT::ValueType VT3,
                        SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, MVT::ValueType VT3,
                        SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5,
                        SDOperand Op6);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                        MVT::ValueType VT2, MVT::ValueType VT3,
                        SDOperand Op1, SDOperand Op2,
                        SDOperand Op3, SDOperand Op4, SDOperand Op5,
                        SDOperand Op6, SDOperand Op7);
  SDNode *getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                        MVT::ValueType VT2,
                        const SDOperand *Ops, unsigned NumOps);
  
  /// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
  /// This can cause recursive merging of nodes in the DAG.  Use the first
  /// version if 'From' is known to have a single result, use the second
  /// if you have two nodes with identical results, use the third otherwise.
  ///
  /// These methods all take an optional vector, which (if not null) is 
  /// populated with any nodes that are deleted from the SelectionDAG, due to
  /// new equivalences that are discovered.
  ///
  void ReplaceAllUsesWith(SDOperand From, SDOperand Op,
                          std::vector<SDNode*> *Deleted = 0);
  void ReplaceAllUsesWith(SDNode *From, SDNode *To,
                          std::vector<SDNode*> *Deleted = 0);
  void ReplaceAllUsesWith(SDNode *From, const SDOperand *To,
                          std::vector<SDNode*> *Deleted = 0);

  /// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
  /// uses of other values produced by From.Val alone.  The Deleted vector is
  /// handled the same was as for ReplaceAllUsesWith, but it is required for
  /// this method.
  void ReplaceAllUsesOfValueWith(SDOperand From, SDOperand To,
                                 std::vector<SDNode*> &Deleted);

  /// DeleteNode - Remove the specified node from the system.  This node must
  /// have no referrers.
  void DeleteNode(SDNode *N);

  /// AssignNodeIds - Assign a unique node id for each node in the DAG based on
  /// their allnodes order. It returns the maximum id.
  unsigned AssignNodeIds();

  /// AssignTopologicalOrder - Assign a unique node id for each node in the DAG
  /// based on their topological order. It returns the maximum id and a vector
  /// of the SDNodes* in assigned order by reference.
  unsigned AssignTopologicalOrder(std::vector<SDNode*> &TopOrder);

  void dump() const;

private:
  void RemoveNodeFromCSEMaps(SDNode *N);
  SDNode *AddNonLeafNodeToCSEMaps(SDNode *N);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDOperand Op, void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDOperand Op1, SDOperand Op2,
                               void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, const SDOperand *Ops, unsigned NumOps,
                               void *&InsertPos);

  void DeleteNodeNotInCSEMaps(SDNode *N);
  MVT::ValueType *getNodeValueTypes(MVT::ValueType VT1);
  MVT::ValueType *getNodeValueTypes(MVT::ValueType VT1, MVT::ValueType VT2);
  MVT::ValueType *getNodeValueTypes(MVT::ValueType VT1, MVT::ValueType VT2,
                                    MVT::ValueType VT3);
  MVT::ValueType *getNodeValueTypes(std::vector<MVT::ValueType> &RetVals);
  
  
  /// SimplifySetCC - Try to simplify a setcc built with the specified operands 
  /// and cc.  If unable to simplify it, return a null SDOperand.
  SDOperand SimplifySetCC(MVT::ValueType VT, SDOperand N1,
                          SDOperand N2, ISD::CondCode Cond);
  
  // List of non-single value types.
  std::list<std::vector<MVT::ValueType> > VTList;
  
  // Maps to auto-CSE operations.
  std::vector<CondCodeSDNode*> CondCodeNodes;

  std::vector<SDNode*> ValueTypeNodes;
  std::map<std::string, SDNode*> ExternalSymbols;
  std::map<std::string, SDNode*> TargetExternalSymbols;
  std::map<std::string, StringSDNode*> StringNodes;
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

}  // end namespace llvm

#endif
