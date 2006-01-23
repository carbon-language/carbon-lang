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
#include "llvm/ADT/ilist"

#include <map>
#include <list>
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

  // Root - The root of the entire DAG.  EntryNode - The starting token.
  SDOperand Root, EntryNode;

  // AllNodes - A linked list of nodes in the current DAG.
  ilist<SDNode> AllNodes;

  // ValueNodes - track SrcValue nodes
  std::map<std::pair<const Value*, int>, SDNode*> ValueNodes;

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
  /// SelectionDAG, including nodes (like loads) that have uses of their token
  /// chain but no other uses and no side effect.  If a node is passed in as an
  /// argument, it is used as the seed for node deletion.
  void RemoveDeadNodes(SDNode *N = 0);

  SDOperand getString(const std::string &Val);
  SDOperand getConstant(uint64_t Val, MVT::ValueType VT);
  SDOperand getTargetConstant(uint64_t Val, MVT::ValueType VT);
  SDOperand getConstantFP(double Val, MVT::ValueType VT);
  SDOperand getGlobalAddress(const GlobalValue *GV, MVT::ValueType VT,
                             int offset = 0);
  SDOperand getTargetGlobalAddress(const GlobalValue *GV, MVT::ValueType VT,
                                   int offset = 0);
  SDOperand getFrameIndex(int FI, MVT::ValueType VT);
  SDOperand getTargetFrameIndex(int FI, MVT::ValueType VT);
  SDOperand getConstantPool(Constant *C, MVT::ValueType VT);
  SDOperand getTargetConstantPool(Constant *C, MVT::ValueType VT);
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
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(MVT::Other);
    VTs.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(getRegister(Reg, N.getValueType()));
    Ops.push_back(N);
    if (Flag.Val) Ops.push_back(Flag);
    return getNode(ISD::CopyToReg, VTs, Ops);
  }

  // Similar to last getCopyToReg() except parameter Reg is a SDOperand
  SDOperand getCopyToReg(SDOperand Chain, SDOperand Reg, SDOperand N,
                         SDOperand Flag) {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(MVT::Other);
    VTs.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Reg);
    Ops.push_back(N);
    if (Flag.Val) Ops.push_back(Flag);
    return getNode(ISD::CopyToReg, VTs, Ops);
  }
  
  SDOperand getCopyFromReg(SDOperand Chain, unsigned Reg, MVT::ValueType VT) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT);
    ResultTys.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(getRegister(Reg, VT));
    return getNode(ISD::CopyFromReg, ResultTys, Ops);
  }
  
  // This version of the getCopyFromReg method takes an extra operand, which
  // indicates that there is potentially an incoming flag value (if Flag is not
  // null) and that there should be a flag result.
  SDOperand getCopyFromReg(SDOperand Chain, unsigned Reg, MVT::ValueType VT,
                           SDOperand Flag) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT);
    ResultTys.push_back(MVT::Other);
    ResultTys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(getRegister(Reg, VT));
    if (Flag.Val) Ops.push_back(Flag);
    return getNode(ISD::CopyFromReg, ResultTys, Ops);
  }

  /// getCall - Note that this destroys the vector of RetVals passed in.
  ///
  SDNode *getCall(std::vector<MVT::ValueType> &RetVals, SDOperand Chain,
                  SDOperand Callee, bool isTailCall = false) {
    SDNode *NN = new SDNode(isTailCall ? ISD::TAILCALL : ISD::CALL, Chain,
                            Callee);
    setNodeValueTypes(NN, RetVals);
    AllNodes.push_back(NN);
    return NN;
  }
  /// getCall - Note that this destroys the vector of RetVals passed in.
  ///
  SDNode *getCall(std::vector<MVT::ValueType> &RetVals, SDOperand Chain,
                  SDOperand Callee, SDOperand Flag, bool isTailCall = false) {
    SDNode *NN = new SDNode(isTailCall ? ISD::TAILCALL : ISD::CALL, Chain,
                            Callee, Flag);
    setNodeValueTypes(NN, RetVals);
    AllNodes.push_back(NN);
    return NN;
  }
  
  /// getCall - This is identical to the one above, and should be used for calls
  /// where arguments are passed in physical registers.  This destroys the
  /// RetVals and ArgsInRegs vectors.
  SDNode *getCall(std::vector<MVT::ValueType> &RetVals, SDOperand Chain,
                  SDOperand Callee, std::vector<SDOperand> &ArgsInRegs,
                  bool isTailCall = false) {
    ArgsInRegs.insert(ArgsInRegs.begin(), Callee);
    ArgsInRegs.insert(ArgsInRegs.begin(), Chain);
    SDNode *NN = new SDNode(isTailCall ? ISD::TAILCALL : ISD::CALL, ArgsInRegs);
    setNodeValueTypes(NN, RetVals);
    AllNodes.push_back(NN);
    return NN;
  }

  SDOperand getCondCode(ISD::CondCode Cond);

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
                    SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4,
                    SDOperand N5);
  SDOperand getNode(unsigned Opcode, MVT::ValueType VT,
                    std::vector<SDOperand> &Children);
  SDOperand getNode(unsigned Opcode, std::vector<MVT::ValueType> &ResultTys,
                    std::vector<SDOperand> &Ops);

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
    MVT::ValueType VT = True.getValueType();
    return getNode(ISD::SELECT_CC, VT, LHS, RHS, True, False,getCondCode(Cond));
  }
  
  /// getBR2Way_CC - Helper function to make it easier to build BRTWOWAY_CC
  /// nodes.
  ///
  SDOperand getBR2Way_CC(SDOperand Chain, SDOperand CCNode, SDOperand LHS, 
                         SDOperand RHS, SDOperand True, SDOperand False) {
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(CCNode);
    Ops.push_back(LHS);
    Ops.push_back(RHS);
    Ops.push_back(True);
    Ops.push_back(False);
    return getNode(ISD::BRTWOWAY_CC, MVT::Other, Ops);
  }

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

  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3,
                          SDOperand Op4) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3, Op4);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3,
                          SDOperand Op4, SDOperand Op5) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3, Op4, Op5);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3,
                          SDOperand Op4, SDOperand Op5, SDOperand Op6) {
    std::vector<SDOperand> Ops;
    Ops.reserve(6);
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3,
                          SDOperand Op4, SDOperand Op5, SDOperand Op6,
                          SDOperand Op7) {
    std::vector<SDOperand> Ops;
    Ops.reserve(7);
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    Ops.push_back(Op7);
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          SDOperand Op1, SDOperand Op2, SDOperand Op3,
                          SDOperand Op4, SDOperand Op5, SDOperand Op6,
                          SDOperand Op7, SDOperand Op8) {
    std::vector<SDOperand> Ops;
    Ops.reserve(8);
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    Ops.push_back(Op7);
    Ops.push_back(Op8);
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT,
                          std::vector<SDOperand> &Ops) {
    return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                          SDOperand Op3) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4, SDOperand Op5) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4, SDOperand Op5,
                          SDOperand Op6) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4, SDOperand Op5,
                          SDOperand Op6, SDOperand Op7) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6); 
    Ops.push_back(Op7);
   return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, MVT::ValueType VT3,
                          SDOperand Op1, SDOperand Op2) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    ResultTys.push_back(VT3);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, MVT::ValueType VT3,
                          SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4, SDOperand Op5,
                          SDOperand Op6) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    ResultTys.push_back(VT3);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                          MVT::ValueType VT2, MVT::ValueType VT3,
                          SDOperand Op1, SDOperand Op2,
                          SDOperand Op3, SDOperand Op4, SDOperand Op5,
                          SDOperand Op6, SDOperand Op7) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    ResultTys.push_back(VT3);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(Op3);
    Ops.push_back(Op4);
    Ops.push_back(Op5);
    Ops.push_back(Op6);
    Ops.push_back(Op7);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  SDOperand getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                          MVT::ValueType VT2, std::vector<SDOperand> &Ops) {
    std::vector<MVT::ValueType> ResultTys;
    ResultTys.push_back(VT1);
    ResultTys.push_back(VT2);
    return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops);
  }
  
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
  void ReplaceAllUsesWith(SDNode *From, const std::vector<SDOperand> &To,
                          std::vector<SDNode*> *Deleted = 0);
  
  
  /// DeleteNode - Remove the specified node from the system.  This node must
  /// have no referrers.
  void DeleteNode(SDNode *N);
  
  void dump() const;

private:
  void RemoveNodeFromCSEMaps(SDNode *N);
  SDNode *AddNonLeafNodeToCSEMaps(SDNode *N);
  void DestroyDeadNode(SDNode *N);
  void DeleteNodeNotInCSEMaps(SDNode *N);
  void setNodeValueTypes(SDNode *N, std::vector<MVT::ValueType> &RetVals);
  void setNodeValueTypes(SDNode *N, MVT::ValueType VT1, MVT::ValueType VT2);
  
  
  /// SimplifySetCC - Try to simplify a setcc built with the specified operands 
  /// and cc.  If unable to simplify it, return a null SDOperand.
  SDOperand SimplifySetCC(MVT::ValueType VT, SDOperand N1,
                          SDOperand N2, ISD::CondCode Cond);
  
  // List of non-single value types.
  std::list<std::vector<MVT::ValueType> > VTList;
  
  // Maps to auto-CSE operations.
  std::map<std::pair<unsigned, MVT::ValueType>, SDNode *> NullaryOps;
  std::map<std::pair<unsigned, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> UnaryOps;
  std::map<std::pair<unsigned, std::pair<SDOperand, SDOperand> >,
           SDNode *> BinaryOps;

  std::map<std::pair<unsigned, MVT::ValueType>, RegisterSDNode*> RegNodes;
  std::vector<CondCodeSDNode*> CondCodeNodes;

  std::map<std::pair<SDOperand, std::pair<SDOperand, MVT::ValueType> >,
           SDNode *> Loads;

  std::map<std::pair<const GlobalValue*, int>, SDNode*> GlobalValues;
  std::map<std::pair<const GlobalValue*, int>, SDNode*> TargetGlobalValues;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> Constants;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> TargetConstants;
  std::map<std::pair<uint64_t, MVT::ValueType>, SDNode*> ConstantFPs;
  std::map<int, SDNode*> FrameIndices, TargetFrameIndices;
  std::map<Constant *, SDNode*> ConstantPoolIndices;
  std::map<Constant *, SDNode*> TargetConstantPoolIndices;
  std::map<MachineBasicBlock *, SDNode*> BBNodes;
  std::vector<SDNode*> ValueTypeNodes;
  std::map<std::string, SDNode*> ExternalSymbols;
  std::map<std::string, SDNode*> TargetExternalSymbols;
  std::map<std::string, StringSDNode*> StringNodes;
  std::map<std::pair<unsigned,
                     std::pair<MVT::ValueType, std::vector<SDOperand> > >,
           SDNode*> OneResultNodes;
  std::map<std::pair<unsigned,
                     std::pair<std::vector<MVT::ValueType>,
                               std::vector<SDOperand> > >,
           SDNode*> ArbitraryNodes;
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
