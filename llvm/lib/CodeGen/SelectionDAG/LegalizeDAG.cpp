//===-- LegalizeDAG.cpp - Implement SelectionDAG::Legalize ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::Legalize method.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
/// SelectionDAGLegalize - This takes an arbitrary SelectionDAG as input and
/// hacks on it until the target machine can handle it.  This involves
/// eliminating value sizes the machine cannot handle (promoting small sizes to
/// large sizes or splitting up large values into small values) as well as
/// eliminating operations the machine cannot handle.
///
/// This code also does a small amount of optimization and recognition of idioms
/// as part of its processing.  For example, if a target does not support a
/// 'setcc' instruction efficiently, but does support 'brcc' instruction, this
/// will attempt merge setcc and brc instructions into brcc's.
///
namespace {
class SelectionDAGLegalize {
  TargetLowering &TLI;
  SelectionDAG &DAG;

  /// LegalizeAction - This enum indicates what action we should take for each
  /// value type the can occur in the program.
  enum LegalizeAction {
    Legal,            // The target natively supports this value type.
    Promote,          // This should be promoted to the next larger type.
    Expand,           // This integer type should be broken into smaller pieces.
  };

  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// value type, where the two bits correspond to the LegalizeAction enum.
  /// This can be queried with "getTypeAction(VT)".
  unsigned ValueTypeActions;

  /// NeedsAnotherIteration - This is set when we expand a large integer
  /// operation into smaller integer operations, but the smaller operations are
  /// not set.  This occurs only rarely in practice, for targets that don't have
  /// 32-bit or larger integer registers.
  bool NeedsAnotherIteration;

  /// LegalizedNodes - For nodes that are of legal width, and that have more
  /// than one use, this map indicates what regularized operand to use.  This
  /// allows us to avoid legalizing the same thing more than once.
  std::map<SDOperand, SDOperand> LegalizedNodes;

  /// PromotedNodes - For nodes that are below legal width, and that have more
  /// than one use, this map indicates what promoted value to use.  This allows
  /// us to avoid promoting the same thing more than once.
  std::map<SDOperand, SDOperand> PromotedNodes;

  /// ExpandedNodes - For nodes that need to be expanded, and which have more
  /// than one use, this map indicates which which operands are the expanded
  /// version of the input.  This allows us to avoid expanding the same node
  /// more than once.
  std::map<SDOperand, std::pair<SDOperand, SDOperand> > ExpandedNodes;

  void AddLegalizedOperand(SDOperand From, SDOperand To) {
    bool isNew = LegalizedNodes.insert(std::make_pair(From, To)).second;
    assert(isNew && "Got into the map somehow?");
  }
  void AddPromotedOperand(SDOperand From, SDOperand To) {
    bool isNew = PromotedNodes.insert(std::make_pair(From, To)).second;
    assert(isNew && "Got into the map somehow?");
  }

public:

  SelectionDAGLegalize(SelectionDAG &DAG);

  /// Run - While there is still lowering to do, perform a pass over the DAG.
  /// Most regularization can be done in a single pass, but targets that require
  /// large values to be split into registers multiple times (e.g. i64 -> 4x
  /// i16) require iteration for these values (the first iteration will demote
  /// to i32, the second will demote to i16).
  void Run() {
    do {
      NeedsAnotherIteration = false;
      LegalizeDAG();
    } while (NeedsAnotherIteration);
  }

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return (LegalizeAction)((ValueTypeActions >> (2*VT)) & 3);
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT::ValueType VT) const {
    return getTypeAction(VT) == Legal;
  }

private:
  void LegalizeDAG();

  SDOperand LegalizeOp(SDOperand O);
  void ExpandOp(SDOperand O, SDOperand &Lo, SDOperand &Hi);
  SDOperand PromoteOp(SDOperand O);

  SDOperand ExpandLibCall(const char *Name, SDNode *Node,
                          SDOperand &Hi);
  SDOperand ExpandIntToFP(bool isSigned, MVT::ValueType DestTy,
                          SDOperand Source);
  bool ExpandShift(unsigned Opc, SDOperand Op, SDOperand Amt,
                   SDOperand &Lo, SDOperand &Hi);
  void ExpandShiftParts(unsigned NodeOp, SDOperand Op, SDOperand Amt,
                        SDOperand &Lo, SDOperand &Hi);
  void ExpandByParts(unsigned NodeOp, SDOperand LHS, SDOperand RHS,
                     SDOperand &Lo, SDOperand &Hi);

  void SpliceCallInto(const SDOperand &CallResult, SDNode *OutChain);

  SDOperand getIntPtrConstant(uint64_t Val) {
    return DAG.getConstant(Val, TLI.getPointerTy());
  }
};
}


SelectionDAGLegalize::SelectionDAGLegalize(SelectionDAG &dag)
  : TLI(dag.getTargetLoweringInfo()), DAG(dag),
    ValueTypeActions(TLI.getValueTypeActions()) {
  assert(MVT::LAST_VALUETYPE <= 16 &&
         "Too many value types for ValueTypeActions to hold!");
}

void SelectionDAGLegalize::LegalizeDAG() {
  SDOperand OldRoot = DAG.getRoot();
  SDOperand NewRoot = LegalizeOp(OldRoot);
  DAG.setRoot(NewRoot);

  ExpandedNodes.clear();
  LegalizedNodes.clear();
  PromotedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes(OldRoot.Val);
}

SDOperand SelectionDAGLegalize::LegalizeOp(SDOperand Op) {
  assert(getTypeAction(Op.getValueType()) == Legal &&
         "Caller should expand or promote operands that are not legal!");
  SDNode *Node = Op.Val;

  // If this operation defines any values that cannot be represented in a
  // register on this target, make sure to expand or promote them.
  if (Node->getNumValues() > 1) {
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      switch (getTypeAction(Node->getValueType(i))) {
      case Legal: break;  // Nothing to do.
      case Expand: {
        SDOperand T1, T2;
        ExpandOp(Op.getValue(i), T1, T2);
        assert(LegalizedNodes.count(Op) &&
               "Expansion didn't add legal operands!");
        return LegalizedNodes[Op];
      }
      case Promote:
        PromoteOp(Op.getValue(i));
        assert(LegalizedNodes.count(Op) &&
               "Expansion didn't add legal operands!");
        return LegalizedNodes[Op];
      }
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  std::map<SDOperand, SDOperand>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDOperand Tmp1, Tmp2, Tmp3;

  SDOperand Result = Op;

  switch (Node->getOpcode()) {
  default:
    if (Node->getOpcode() >= ISD::BUILTIN_OP_END) {
      // If this is a target node, legalize it by legalizing the operands then
      // passing it through.
      std::vector<SDOperand> Ops;
      bool Changed = false;
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
        Ops.push_back(LegalizeOp(Node->getOperand(i)));
        Changed = Changed || Node->getOperand(i) != Ops.back();
      }
      if (Changed)
        if (Node->getNumValues() == 1)
          Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Ops);
        else {
          std::vector<MVT::ValueType> VTs(Node->value_begin(),
                                          Node->value_end());
          Result = DAG.getNode(Node->getOpcode(), VTs, Ops);
        }

      for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
        AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
      return Result.getValue(Op.ResNo);
    }
    // Otherwise this is an unhandled builtin node.  splat.
    std::cerr << "NODE: "; Node->dump(); std::cerr << "\n";
    assert(0 && "Do not know how to legalize this operator!");
    abort();
  case ISD::EntryToken:
  case ISD::FrameIndex:
  case ISD::GlobalAddress:
  case ISD::ExternalSymbol:
  case ISD::ConstantPool:           // Nothing to do.
    assert(getTypeAction(Node->getValueType(0)) == Legal &&
           "This must be legal!");
    break;
  case ISD::CopyFromReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getCopyFromReg(cast<RegSDNode>(Node)->getReg(),
                                  Node->getValueType(0), Tmp1);
    else
      Result = Op.getValue(0);

    // Since CopyFromReg produces two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(Op.getValue(0), Result);
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  case ISD::ImplicitDef:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getImplicitDef(Tmp1, cast<RegSDNode>(Node)->getReg());
    break;
  case ISD::UNDEF: {
    MVT::ValueType VT = Op.getValueType();
    switch (TLI.getOperationAction(ISD::UNDEF, VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand:
    case TargetLowering::Promote:
      if (MVT::isInteger(VT))
        Result = DAG.getConstant(0, VT);
      else if (MVT::isFloatingPoint(VT))
        Result = DAG.getConstantFP(0, VT);
      else
        assert(0 && "Unknown value type!");
      break;
    case TargetLowering::Legal:
      break;
    }
    break;
  }
  case ISD::Constant:
    // We know we don't need to expand constants here, constants only have one
    // value and we check that it is fine above.

    // FIXME: Maybe we should handle things like targets that don't support full
    // 32-bit immediates?
    break;
  case ISD::ConstantFP: {
    // Spill FP immediates to the constant pool if the target cannot directly
    // codegen them.  Targets often have some immediate values that can be
    // efficiently generated into an FP register without a load.  We explicitly
    // leave these constants as ConstantFP nodes for the target to deal with.

    ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Node);

    // Check to see if this FP immediate is already legal.
    bool isLegal = false;
    for (TargetLowering::legal_fpimm_iterator I = TLI.legal_fpimm_begin(),
           E = TLI.legal_fpimm_end(); I != E; ++I)
      if (CFP->isExactlyValue(*I)) {
        isLegal = true;
        break;
      }

    if (!isLegal) {
      // Otherwise we need to spill the constant to memory.
      MachineConstantPool *CP = DAG.getMachineFunction().getConstantPool();

      bool Extend = false;

      // If a FP immediate is precise when represented as a float, we put it
      // into the constant pool as a float, even if it's is statically typed
      // as a double.
      MVT::ValueType VT = CFP->getValueType(0);
      bool isDouble = VT == MVT::f64;
      ConstantFP *LLVMC = ConstantFP::get(isDouble ? Type::DoubleTy :
                                             Type::FloatTy, CFP->getValue());
      if (isDouble && CFP->isExactlyValue((float)CFP->getValue()) &&
          // Only do this if the target has a native EXTLOAD instruction from
          // f32.
          TLI.getOperationAction(ISD::EXTLOAD,
                                 MVT::f32) == TargetLowering::Legal) {
        LLVMC = cast<ConstantFP>(ConstantExpr::getCast(LLVMC, Type::FloatTy));
        VT = MVT::f32;
        Extend = true;
      }

      SDOperand CPIdx = DAG.getConstantPool(CP->getConstantPoolIndex(LLVMC),
                                            TLI.getPointerTy());
      if (Extend) {
        Result = DAG.getExtLoad(ISD::EXTLOAD, MVT::f64, DAG.getEntryNode(),
                                CPIdx, DAG.getSrcValue(NULL), MVT::f32);
      } else {
        Result = DAG.getLoad(VT, DAG.getEntryNode(), CPIdx,
                             DAG.getSrcValue(NULL));
      }
    }
    break;
  }
  case ISD::TokenFactor: {
    std::vector<SDOperand> Ops;
    bool Changed = false;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
      SDOperand Op = Node->getOperand(i);
      // Fold single-use TokenFactor nodes into this token factor as we go.
      // FIXME: This is something that the DAGCombiner should do!!
      if (Op.getOpcode() == ISD::TokenFactor && Op.hasOneUse()) {
        Changed = true;
        for (unsigned j = 0, e = Op.getNumOperands(); j != e; ++j)
          Ops.push_back(LegalizeOp(Op.getOperand(j)));
      } else {
        Ops.push_back(LegalizeOp(Op));  // Legalize the operands
        Changed |= Ops[i] != Op;
      }
    }
    if (Changed)
      Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Ops);
    break;
  }

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Do not try to legalize the target-specific arguments (#1+)
    Tmp2 = Node->getOperand(0);
    if (Tmp1 != Tmp2) {
      Node->setAdjCallChain(Tmp1);

      // If moving the operand from pointing to Tmp2 dropped its use count to 1,
      // this will cause the maps used to memoize results to get confused.
      // Create and add a dummy use, just to increase its use count.  This will
      // be removed at the end of legalize when dead nodes are removed.
      if (Tmp2.Val->hasOneUse())
        DAG.getNode(ISD::PCMARKER, MVT::Other, Tmp2,
                    DAG.getConstant(0, MVT::i32));
    }
    // Note that we do not create new CALLSEQ_DOWN/UP nodes here.  These
    // nodes are treated specially and are mutated in place.  This makes the dag
    // legalization process more efficient and also makes libcall insertion
    // easier.
    break;
  case ISD::DYNAMIC_STACKALLOC:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the size.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the alignment.
    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
        Tmp3 != Node->getOperand(2)) {
      std::vector<MVT::ValueType> VTs(Node->value_begin(), Node->value_end());
      std::vector<SDOperand> Ops;
      Ops.push_back(Tmp1); Ops.push_back(Tmp2); Ops.push_back(Tmp3);
      Result = DAG.getNode(ISD::DYNAMIC_STACKALLOC, VTs, Ops);
    } else
      Result = Op.getValue(0);

    // Since this op produces two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);

  case ISD::TAILCALL:
  case ISD::CALL: {
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the callee.

    bool Changed = false;
    std::vector<SDOperand> Ops;
    for (unsigned i = 2, e = Node->getNumOperands(); i != e; ++i) {
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Changed |= Ops.back() != Node->getOperand(i);
    }

    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) || Changed) {
      std::vector<MVT::ValueType> RetTyVTs;
      RetTyVTs.reserve(Node->getNumValues());
      for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
        RetTyVTs.push_back(Node->getValueType(i));
      Result = SDOperand(DAG.getCall(RetTyVTs, Tmp1, Tmp2, Ops,
                                     Node->getOpcode() == ISD::TAILCALL), 0);
    } else {
      Result = Result.getValue(0);
    }
    // Since calls produce multiple values, make sure to remember that we
    // legalized all of them.
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      AddLegalizedOperand(SDOperand(Node, i), Result.getValue(i));
    return Result.getValue(Op.ResNo);
  }
  case ISD::BR:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getNode(ISD::BR, MVT::Other, Tmp1, Node->getOperand(1));
    break;

  case ISD::BRCOND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the condition.
      break;
    case Promote:
      Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the condition.
      break;
    }
    // Basic block destination (Op#2) is always legal.
    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
      Result = DAG.getNode(ISD::BRCOND, MVT::Other, Tmp1, Tmp2,
                           Node->getOperand(2));
    break;
  case ISD::BRCONDTWOWAY:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the condition.
      break;
    case Promote:
      Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the condition.
      break;
    }
    // If this target does not support BRCONDTWOWAY, lower it to a BRCOND/BR
    // pair.
    switch (TLI.getOperationAction(ISD::BRCONDTWOWAY, MVT::Other)) {
    case TargetLowering::Promote:
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1)) {
        std::vector<SDOperand> Ops;
        Ops.push_back(Tmp1);
        Ops.push_back(Tmp2);
        Ops.push_back(Node->getOperand(2));
        Ops.push_back(Node->getOperand(3));
        Result = DAG.getNode(ISD::BRCONDTWOWAY, MVT::Other, Ops);
      }
      break;
    case TargetLowering::Expand:
      Result = DAG.getNode(ISD::BRCOND, MVT::Other, Tmp1, Tmp2,
                           Node->getOperand(2));
      Result = DAG.getNode(ISD::BR, MVT::Other, Result, Node->getOperand(3));
      break;
    }
    break;

  case ISD::LOAD:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    if (Tmp1 != Node->getOperand(0) ||
        Tmp2 != Node->getOperand(1))
      Result = DAG.getLoad(Node->getValueType(0), Tmp1, Tmp2,
                           Node->getOperand(2));
    else
      Result = SDOperand(Node, 0);

    // Since loads produce two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);

  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD: {
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    MVT::ValueType SrcVT = cast<VTSDNode>(Node->getOperand(3))->getVT();
    switch (TLI.getOperationAction(Node->getOpcode(), SrcVT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Promote:
      assert(SrcVT == MVT::i1 && "Can only promote EXTLOAD from i1 -> i8!");
      Result = DAG.getExtLoad(Node->getOpcode(), Node->getValueType(0),
                              Tmp1, Tmp2, Node->getOperand(2), MVT::i8);
      // Since loads produce two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDOperand(Node, 0), Result);
      AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
      return Result.getValue(Op.ResNo);

    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(1))
        Result = DAG.getExtLoad(Node->getOpcode(), Node->getValueType(0),
                                Tmp1, Tmp2, Node->getOperand(2), SrcVT);
      else
        Result = SDOperand(Node, 0);

      // Since loads produce two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDOperand(Node, 0), Result);
      AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
      return Result.getValue(Op.ResNo);
    case TargetLowering::Expand:
      //f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
      if (SrcVT == MVT::f32 && Node->getValueType(0) == MVT::f64) {
        SDOperand Load = DAG.getLoad(SrcVT, Tmp1, Tmp2, Node->getOperand(2));
        Result = DAG.getNode(ISD::FP_EXTEND, Node->getValueType(0), Load);
        if (Op.ResNo)
          return Load.getValue(1);
        return Result;
      }
      assert(Node->getOpcode() != ISD::EXTLOAD &&
             "EXTLOAD should always be supported!");
      // Turn the unsupported load into an EXTLOAD followed by an explicit
      // zero/sign extend inreg.
      Result = DAG.getExtLoad(ISD::EXTLOAD, Node->getValueType(0),
                              Tmp1, Tmp2, Node->getOperand(2), SrcVT);
      SDOperand ValRes;
      if (Node->getOpcode() == ISD::SEXTLOAD)
        ValRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result, DAG.getValueType(SrcVT));
      else
        ValRes = DAG.getZeroExtendInReg(Result, SrcVT);
      AddLegalizedOperand(SDOperand(Node, 0), ValRes);
      AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
      if (Op.ResNo)
        return Result.getValue(1);
      return ValRes;
    }
    assert(0 && "Unreachable");
  }
  case ISD::EXTRACT_ELEMENT:
    // Get both the low and high parts.
    ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
    if (cast<ConstantSDNode>(Node->getOperand(1))->getValue())
      Result = Tmp2;  // 1 -> Hi
    else
      Result = Tmp1;  // 0 -> Lo
    break;

  case ISD::CopyToReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Legal:
      // Legalize the incoming value (must be legal).
      Tmp2 = LegalizeOp(Node->getOperand(1));
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getCopyToReg(Tmp1, Tmp2, cast<RegSDNode>(Node)->getReg());
      break;
    case Promote:
      Tmp2 = PromoteOp(Node->getOperand(1));
      Result = DAG.getCopyToReg(Tmp1, Tmp2, cast<RegSDNode>(Node)->getReg());
      break;
    case Expand:
      SDOperand Lo, Hi;
      ExpandOp(Node->getOperand(1), Lo, Hi);
      unsigned Reg = cast<RegSDNode>(Node)->getReg();
      Lo = DAG.getCopyToReg(Tmp1, Lo, Reg);
      Hi = DAG.getCopyToReg(Tmp1, Hi, Reg+1);
      // Note that the copytoreg nodes are independent of each other.
      Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
      assert(isTypeLegal(Result.getValueType()) &&
             "Cannot expand multiple times yet (i64 -> i16)");
      break;
    }
    break;

  case ISD::RET:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    switch (Node->getNumOperands()) {
    case 2:  // ret val
      switch (getTypeAction(Node->getOperand(1).getValueType())) {
      case Legal:
        Tmp2 = LegalizeOp(Node->getOperand(1));
        if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
          Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Tmp2);
        break;
      case Expand: {
        SDOperand Lo, Hi;
        ExpandOp(Node->getOperand(1), Lo, Hi);
        Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Lo, Hi);
        break;
      }
      case Promote:
        Tmp2 = PromoteOp(Node->getOperand(1));
        Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Tmp2);
        break;
      }
      break;
    case 1:  // ret void
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1);
      break;
    default: { // ret <values>
      std::vector<SDOperand> NewValues;
      NewValues.push_back(Tmp1);
      for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i)
        switch (getTypeAction(Node->getOperand(i).getValueType())) {
        case Legal:
          NewValues.push_back(LegalizeOp(Node->getOperand(i)));
          break;
        case Expand: {
          SDOperand Lo, Hi;
          ExpandOp(Node->getOperand(i), Lo, Hi);
          NewValues.push_back(Lo);
          NewValues.push_back(Hi);
          break;
        }
        case Promote:
          assert(0 && "Can't promote multiple return value yet!");
        }
      Result = DAG.getNode(ISD::RET, MVT::Other, NewValues);
      break;
    }
    }
    break;
  case ISD::STORE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(2));  // Legalize the pointer.

    // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
    if (ConstantFPSDNode *CFP =dyn_cast<ConstantFPSDNode>(Node->getOperand(1))){
      if (CFP->getValueType(0) == MVT::f32) {
        union {
          unsigned I;
          float    F;
        } V;
        V.F = CFP->getValue();
        Result = DAG.getNode(ISD::STORE, MVT::Other, Tmp1, 
                             DAG.getConstant(V.I, MVT::i32), Tmp2,
                             Node->getOperand(3));
      } else {
        assert(CFP->getValueType(0) == MVT::f64 && "Unknown FP type!");
        union {
          uint64_t I;
          double   F;
        } V;
        V.F = CFP->getValue();
        Result = DAG.getNode(ISD::STORE, MVT::Other, Tmp1, 
                             DAG.getConstant(V.I, MVT::i64), Tmp2,
                             Node->getOperand(3));
      }
      Node = Result.Val;
    }

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Legal: {
      SDOperand Val = LegalizeOp(Node->getOperand(1));
      if (Val != Node->getOperand(1) || Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(2))
        Result = DAG.getNode(ISD::STORE, MVT::Other, Tmp1, Val, Tmp2,
                             Node->getOperand(3));
      break;
    }
    case Promote:
      // Truncate the value and store the result.
      Tmp3 = PromoteOp(Node->getOperand(1));
      Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Tmp1, Tmp3, Tmp2,
                           Node->getOperand(3),
                          DAG.getValueType(Node->getOperand(1).getValueType()));
      break;

    case Expand:
      SDOperand Lo, Hi;
      ExpandOp(Node->getOperand(1), Lo, Hi);

      if (!TLI.isLittleEndian())
        std::swap(Lo, Hi);

      Lo = DAG.getNode(ISD::STORE, MVT::Other, Tmp1, Lo, Tmp2,
                       Node->getOperand(3));
      unsigned IncrementSize = MVT::getSizeInBits(Hi.getValueType())/8;
      Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                         getIntPtrConstant(IncrementSize));
      assert(isTypeLegal(Tmp2.getValueType()) &&
             "Pointers must be legal!");
      //Again, claiming both parts of the store came form the same Instr
      Hi = DAG.getNode(ISD::STORE, MVT::Other, Tmp1, Hi, Tmp2,
                       Node->getOperand(3));
      Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
      break;
    }
    break;
  case ISD::PCMARKER:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getNode(ISD::PCMARKER, MVT::Other, Tmp1,Node->getOperand(1));
    break;
  case ISD::TRUNCSTORE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the pointer.

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1));
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
          Tmp3 != Node->getOperand(2))
        Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Tmp1, Tmp2, Tmp3,
                             Node->getOperand(3), Node->getOperand(4));
      break;
    case Promote:
    case Expand:
      assert(0 && "Cannot handle illegal TRUNCSTORE yet!");
    }
    break;
  case ISD::SELECT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the condition.
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));  // Promote the condition.
      break;
    }
    Tmp2 = LegalizeOp(Node->getOperand(1));   // TrueVal
    Tmp3 = LegalizeOp(Node->getOperand(2));   // FalseVal

    switch (TLI.getOperationAction(Node->getOpcode(), Tmp2.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
          Tmp3 != Node->getOperand(2))
        Result = DAG.getNode(ISD::SELECT, Node->getValueType(0),
                             Tmp1, Tmp2, Tmp3);
      break;
    case TargetLowering::Promote: {
      MVT::ValueType NVT =
        TLI.getTypeToPromoteTo(ISD::SELECT, Tmp2.getValueType());
      unsigned ExtOp, TruncOp;
      if (MVT::isInteger(Tmp2.getValueType())) {
        ExtOp = ISD::ZERO_EXTEND;
        TruncOp  = ISD::TRUNCATE;
      } else {
        ExtOp = ISD::FP_EXTEND;
        TruncOp  = ISD::FP_ROUND;
      }
      // Promote each of the values to the new type.
      Tmp2 = DAG.getNode(ExtOp, NVT, Tmp2);
      Tmp3 = DAG.getNode(ExtOp, NVT, Tmp3);
      // Perform the larger operation, then round down.
      Result = DAG.getNode(ISD::SELECT, NVT, Tmp1, Tmp2,Tmp3);
      Result = DAG.getNode(TruncOp, Node->getValueType(0), Result);
      break;
    }
    }
    break;
  case ISD::SETCC:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
      Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                              Node->getValueType(0), Tmp1, Tmp2);
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));   // LHS
      Tmp2 = PromoteOp(Node->getOperand(1));   // RHS

      // If this is an FP compare, the operands have already been extended.
      if (MVT::isInteger(Node->getOperand(0).getValueType())) {
        MVT::ValueType VT = Node->getOperand(0).getValueType();
        MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);

        // Otherwise, we have to insert explicit sign or zero extends.  Note
        // that we could insert sign extends for ALL conditions, but zero extend
        // is cheaper on many machines (an AND instead of two shifts), so prefer
        // it.
        switch (cast<SetCCSDNode>(Node)->getCondition()) {
        default: assert(0 && "Unknown integer comparison!");
        case ISD::SETEQ:
        case ISD::SETNE:
        case ISD::SETUGE:
        case ISD::SETUGT:
        case ISD::SETULE:
        case ISD::SETULT:
          // ALL of these operations will work if we either sign or zero extend
          // the operands (including the unsigned comparisons!).  Zero extend is
          // usually a simpler/cheaper operation, so prefer it.
          Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
          Tmp2 = DAG.getZeroExtendInReg(Tmp2, VT);
          break;
        case ISD::SETGE:
        case ISD::SETGT:
        case ISD::SETLT:
        case ISD::SETLE:
          Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                             DAG.getValueType(VT));
          Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp2,
                             DAG.getValueType(VT));
          break;
        }

      }
      Result = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                            Node->getValueType(0), Tmp1, Tmp2);
      break;
    case Expand:
      SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
      ExpandOp(Node->getOperand(0), LHSLo, LHSHi);
      ExpandOp(Node->getOperand(1), RHSLo, RHSHi);
      switch (cast<SetCCSDNode>(Node)->getCondition()) {
      case ISD::SETEQ:
      case ISD::SETNE:
        if (RHSLo == RHSHi)
          if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
            if (RHSCST->isAllOnesValue()) {
              // Comparison to -1.
              Tmp1 = DAG.getNode(ISD::AND, LHSLo.getValueType(), LHSLo, LHSHi);
              Result = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                                    Node->getValueType(0), Tmp1, RHSLo);
              break;
            }

        Tmp1 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSLo, RHSLo);
        Tmp2 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSHi, RHSHi);
        Tmp1 = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp2);
        Result = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                              Node->getValueType(0), Tmp1,
                              DAG.getConstant(0, Tmp1.getValueType()));
        break;
      default:
        // If this is a comparison of the sign bit, just look at the top part.
        // X > -1,  x < 0
        if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(Node->getOperand(1)))
          if ((cast<SetCCSDNode>(Node)->getCondition() == ISD::SETLT &&
               CST->getValue() == 0) ||              // X < 0
              (cast<SetCCSDNode>(Node)->getCondition() == ISD::SETGT &&
               (CST->isAllOnesValue())))             // X > -1
            return DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                                Node->getValueType(0), LHSHi, RHSHi);

        // FIXME: This generated code sucks.
        ISD::CondCode LowCC;
        switch (cast<SetCCSDNode>(Node)->getCondition()) {
        default: assert(0 && "Unknown integer setcc!");
        case ISD::SETLT:
        case ISD::SETULT: LowCC = ISD::SETULT; break;
        case ISD::SETGT:
        case ISD::SETUGT: LowCC = ISD::SETUGT; break;
        case ISD::SETLE:
        case ISD::SETULE: LowCC = ISD::SETULE; break;
        case ISD::SETGE:
        case ISD::SETUGE: LowCC = ISD::SETUGE; break;
        }

        // Tmp1 = lo(op1) < lo(op2)   // Always unsigned comparison
        // Tmp2 = hi(op1) < hi(op2)   // Signedness depends on operands
        // dest = hi(op1) == hi(op2) ? Tmp1 : Tmp2;

        // NOTE: on targets without efficient SELECT of bools, we can always use
        // this identity: (B1 ? B2 : B3) --> (B1 & B2)|(!B1&B3)
        Tmp1 = DAG.getSetCC(LowCC, Node->getValueType(0), LHSLo, RHSLo);
        Tmp2 = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                            Node->getValueType(0), LHSHi, RHSHi);
        Result = DAG.getSetCC(ISD::SETEQ, Node->getValueType(0), LHSHi, RHSHi);
        Result = DAG.getNode(ISD::SELECT, Tmp1.getValueType(),
                             Result, Tmp1, Tmp2);
        break;
      }
    }
    break;

  case ISD::MEMSET:
  case ISD::MEMCPY:
  case ISD::MEMMOVE: {
    Tmp1 = LegalizeOp(Node->getOperand(0));      // Chain
    Tmp2 = LegalizeOp(Node->getOperand(1));      // Pointer

    if (Node->getOpcode() == ISD::MEMSET) {      // memset = ubyte
      switch (getTypeAction(Node->getOperand(2).getValueType())) {
      case Expand: assert(0 && "Cannot expand a byte!");
      case Legal:
        Tmp3 = LegalizeOp(Node->getOperand(2));
        break;
      case Promote:
        Tmp3 = PromoteOp(Node->getOperand(2));
        break;
      }
    } else {
      Tmp3 = LegalizeOp(Node->getOperand(2));    // memcpy/move = pointer,
    }

    SDOperand Tmp4;
    switch (getTypeAction(Node->getOperand(3).getValueType())) {
    case Expand: {
      // Length is too big, just take the lo-part of the length.
      SDOperand HiPart;
      ExpandOp(Node->getOperand(3), HiPart, Tmp4);
      break;
    }
    case Legal:
      Tmp4 = LegalizeOp(Node->getOperand(3));
      break;
    case Promote:
      Tmp4 = PromoteOp(Node->getOperand(3));
      break;
    }

    SDOperand Tmp5;
    switch (getTypeAction(Node->getOperand(4).getValueType())) {  // uint
    case Expand: assert(0 && "Cannot expand this yet!");
    case Legal:
      Tmp5 = LegalizeOp(Node->getOperand(4));
      break;
    case Promote:
      Tmp5 = PromoteOp(Node->getOperand(4));
      break;
    }

    switch (TLI.getOperationAction(Node->getOpcode(), MVT::Other)) {
    default: assert(0 && "This action not implemented for this operation!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
          Tmp3 != Node->getOperand(2) || Tmp4 != Node->getOperand(3) ||
          Tmp5 != Node->getOperand(4)) {
        std::vector<SDOperand> Ops;
        Ops.push_back(Tmp1); Ops.push_back(Tmp2); Ops.push_back(Tmp3);
        Ops.push_back(Tmp4); Ops.push_back(Tmp5);
        Result = DAG.getNode(Node->getOpcode(), MVT::Other, Ops);
      }
      break;
    case TargetLowering::Expand: {
      // Otherwise, the target does not support this operation.  Lower the
      // operation to an explicit libcall as appropriate.
      MVT::ValueType IntPtr = TLI.getPointerTy();
      const Type *IntPtrTy = TLI.getTargetData().getIntPtrType();
      std::vector<std::pair<SDOperand, const Type*> > Args;

      const char *FnName = 0;
      if (Node->getOpcode() == ISD::MEMSET) {
        Args.push_back(std::make_pair(Tmp2, IntPtrTy));
        // Extend the ubyte argument to be an int value for the call.
        Tmp3 = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Tmp3);
        Args.push_back(std::make_pair(Tmp3, Type::IntTy));
        Args.push_back(std::make_pair(Tmp4, IntPtrTy));

        FnName = "memset";
      } else if (Node->getOpcode() == ISD::MEMCPY ||
                 Node->getOpcode() == ISD::MEMMOVE) {
        Args.push_back(std::make_pair(Tmp2, IntPtrTy));
        Args.push_back(std::make_pair(Tmp3, IntPtrTy));
        Args.push_back(std::make_pair(Tmp4, IntPtrTy));
        FnName = Node->getOpcode() == ISD::MEMMOVE ? "memmove" : "memcpy";
      } else {
        assert(0 && "Unknown op!");
      }

      std::pair<SDOperand,SDOperand> CallResult =
        TLI.LowerCallTo(Tmp1, Type::VoidTy, false, CallingConv::C, false,
                        DAG.getExternalSymbol(FnName, IntPtr), Args, DAG);
      Result = CallResult.second;
      NeedsAnotherIteration = true;
      break;
    }
    case TargetLowering::Custom:
      std::vector<SDOperand> Ops;
      Ops.push_back(Tmp1); Ops.push_back(Tmp2); Ops.push_back(Tmp3);
      Ops.push_back(Tmp4); Ops.push_back(Tmp5);
      Result = DAG.getNode(Node->getOpcode(), MVT::Other, Ops);
      Result = TLI.LowerOperation(Result, DAG);
      Result = LegalizeOp(Result);
      break;
    }
    break;
  }

  case ISD::READPORT:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));

    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1)) {
      std::vector<MVT::ValueType> VTs(Node->value_begin(), Node->value_end());
      std::vector<SDOperand> Ops;
      Ops.push_back(Tmp1);
      Ops.push_back(Tmp2);
      Result = DAG.getNode(ISD::READPORT, VTs, Ops);
    } else
      Result = SDOperand(Node, 0);
    // Since these produce two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  case ISD::WRITEPORT:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Tmp3 = LegalizeOp(Node->getOperand(2));
    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
        Tmp3 != Node->getOperand(2))
      Result = DAG.getNode(Node->getOpcode(), MVT::Other, Tmp1, Tmp2, Tmp3);
    break;

  case ISD::READIO:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Custom:
    default: assert(0 && "This action not implemented for this operation!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1)) {
        std::vector<MVT::ValueType> VTs(Node->value_begin(), Node->value_end());
        std::vector<SDOperand> Ops;
        Ops.push_back(Tmp1);
        Ops.push_back(Tmp2);
        Result = DAG.getNode(ISD::READPORT, VTs, Ops);
      } else
        Result = SDOperand(Node, 0);
      break;
    case TargetLowering::Expand:
      // Replace this with a load from memory.
      Result = DAG.getLoad(Node->getValueType(0), Node->getOperand(0),
                           Node->getOperand(1), DAG.getSrcValue(NULL));
      Result = LegalizeOp(Result);
      break;
    }

    // Since these produce two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);

  case ISD::WRITEIO:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Tmp3 = LegalizeOp(Node->getOperand(2));

    switch (TLI.getOperationAction(Node->getOpcode(),
                                   Node->getOperand(1).getValueType())) {
    case TargetLowering::Custom:
    default: assert(0 && "This action not implemented for this operation!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
          Tmp3 != Node->getOperand(2))
        Result = DAG.getNode(Node->getOpcode(), MVT::Other, Tmp1, Tmp2, Tmp3);
      break;
    case TargetLowering::Expand:
      // Replace this with a store to memory.
      Result = DAG.getNode(ISD::STORE, MVT::Other, Node->getOperand(0),
                           Node->getOperand(1), Node->getOperand(2),
                           DAG.getSrcValue(NULL));
      Result = LegalizeOp(Result);
      break;
    }
    break;

  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS:
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    std::vector<SDOperand> Ops;
    bool Changed = false;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Changed |= Ops.back() != Node->getOperand(i);
    }
    if (Changed) {
      std::vector<MVT::ValueType> VTs(Node->value_begin(), Node->value_end());
      Result = DAG.getNode(Node->getOpcode(), VTs, Ops);
    }

    // Since these produce multiple values, make sure to remember that we
    // legalized all of them.
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      AddLegalizedOperand(SDOperand(Node, i), Result.getValue(i));
    return Result.getValue(Op.ResNo);
  }

    // Binary operators
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
  case ISD::UDIV:
  case ISD::SDIV:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "Not possible");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the RHS.
      break;
    case Promote:
      Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the RHS.
      break;
    }
    if (Tmp1 != Node->getOperand(0) ||
        Tmp2 != Node->getOperand(1))
      Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1,Tmp2);
    break;

  case ISD::UREM:
  case ISD::SREM:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1,
                             Tmp2);
      break;
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom handle this yet!");
    case TargetLowering::Expand: {
      MVT::ValueType VT = Node->getValueType(0);
      unsigned Opc = (Node->getOpcode() == ISD::UREM) ? ISD::UDIV : ISD::SDIV;
      Result = DAG.getNode(Opc, VT, Tmp1, Tmp2);
      Result = DAG.getNode(ISD::MUL, VT, Result, Tmp2);
      Result = DAG.getNode(ISD::SUB, VT, Tmp1, Result);
      }
      break;
    }
    break;

  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Op
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case TargetLowering::Promote: {
      MVT::ValueType OVT = Tmp1.getValueType();
      MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Zero extend the argument.
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Tmp1);
      // Perform the larger operation, then subtract if needed.
      Tmp1 = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      switch(Node->getOpcode())
      {
      case ISD::CTPOP:
        Result = Tmp1;
        break;
      case ISD::CTTZ:
        //if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
        Tmp2 = DAG.getSetCC(ISD::SETEQ, TLI.getSetCCResultTy(), Tmp1, 
                            DAG.getConstant(getSizeInBits(NVT), NVT));
        Result = DAG.getNode(ISD::SELECT, NVT, Tmp2, 
                           DAG.getConstant(getSizeInBits(OVT),NVT), Tmp1);
        break;
      case ISD::CTLZ:
        //Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
        Result = DAG.getNode(ISD::SUB, NVT, Tmp1, 
                             DAG.getConstant(getSizeInBits(NVT) - 
                                             getSizeInBits(OVT), NVT));
        break;
      }
      break;
    }
    case TargetLowering::Custom:
      assert(0 && "Cannot custom handle this yet!");
    case TargetLowering::Expand:
      switch(Node->getOpcode())
      {
      case ISD::CTPOP: {
        static const uint64_t mask[6] = {
          0x5555555555555555ULL, 0x3333333333333333ULL,
          0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
          0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
        };
        MVT::ValueType VT = Tmp1.getValueType();
        MVT::ValueType ShVT = TLI.getShiftAmountTy();
        unsigned len = getSizeInBits(VT);
        for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
          //x = (x & mask[i][len/8]) + (x >> (1 << i) & mask[i][len/8])
          Tmp2 = DAG.getConstant(mask[i], VT);
          Tmp3 = DAG.getConstant(1ULL << i, ShVT);
          Tmp1 = DAG.getNode(ISD::ADD, VT, 
                             DAG.getNode(ISD::AND, VT, Tmp1, Tmp2),
                             DAG.getNode(ISD::AND, VT,
                                         DAG.getNode(ISD::SRL, VT, Tmp1, Tmp3),
                                         Tmp2));
        }
        Result = Tmp1;
        break;
      }
      case ISD::CTLZ: {
        /* for now, we do this:
           x = x | (x >> 1);
           x = x | (x >> 2);
           ...
           x = x | (x >>16);
           x = x | (x >>32); // for 64-bit input 
           return popcount(~x);
        
           but see also: http://www.hackersdelight.org/HDcode/nlz.cc */
        MVT::ValueType VT = Tmp1.getValueType();
        MVT::ValueType ShVT = TLI.getShiftAmountTy();
        unsigned len = getSizeInBits(VT);
        for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
          Tmp3 = DAG.getConstant(1ULL << i, ShVT);
          Tmp1 = DAG.getNode(ISD::OR, VT, Tmp1,  
                             DAG.getNode(ISD::SRL, VT, Tmp1, Tmp3));
        }
        Tmp3 = DAG.getNode(ISD::XOR, VT, Tmp1, DAG.getConstant(~0ULL, VT));
        Result = LegalizeOp(DAG.getNode(ISD::CTPOP, VT, Tmp3));
        break;
      }
      case ISD::CTTZ: {
        // for now, we use: { return popcount(~x & (x - 1)); } 
        // unless the target has ctlz but not ctpop, in which case we use:
        // { return 32 - nlz(~x & (x-1)); }
        // see also http://www.hackersdelight.org/HDcode/ntz.cc
        MVT::ValueType VT = Tmp1.getValueType();
        Tmp2 = DAG.getConstant(~0ULL, VT);
        Tmp3 = DAG.getNode(ISD::AND, VT, 
                           DAG.getNode(ISD::XOR, VT, Tmp1, Tmp2),
                           DAG.getNode(ISD::SUB, VT, Tmp1,
                                       DAG.getConstant(1, VT)));
        // If ISD::CTLZ is legal and CTPOP isn't, then do that instead
        if (TLI.getOperationAction(ISD::CTPOP, VT) != TargetLowering::Legal &&
            TLI.getOperationAction(ISD::CTLZ, VT) == TargetLowering::Legal) {
          Result = LegalizeOp(DAG.getNode(ISD::SUB, VT, 
                                        DAG.getConstant(getSizeInBits(VT), VT),
                                        DAG.getNode(ISD::CTLZ, VT, Tmp3)));
        } else {
          Result = LegalizeOp(DAG.getNode(ISD::CTPOP, VT, Tmp3));
        }
        break;
      }
      default:
        assert(0 && "Cannot expand this yet!");
        break;
      }
      break;
    }
    break;
    
    // Unary operators
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom handle this yet!");
    case TargetLowering::Expand:
      switch(Node->getOpcode()) {
      case ISD::FNEG: {
        // Expand Y = FNEG(X) ->  Y = SUB -0.0, X
        Tmp2 = DAG.getConstantFP(-0.0, Node->getValueType(0));
        Result = LegalizeOp(DAG.getNode(ISD::SUB, Node->getValueType(0),
                                        Tmp2, Tmp1));
        break;
      }
      case ISD::FABS: {
        // Expand Y = FABS(X) -> Y = (X >u 0.0) ? X : fneg(X).
        MVT::ValueType VT = Node->getValueType(0);
        Tmp2 = DAG.getConstantFP(0.0, VT);
        Tmp2 = DAG.getSetCC(ISD::SETUGT, TLI.getSetCCResultTy(), Tmp1, Tmp2);
        Tmp3 = DAG.getNode(ISD::FNEG, VT, Tmp1);
        Result = DAG.getNode(ISD::SELECT, VT, Tmp2, Tmp1, Tmp3);
        Result = LegalizeOp(Result);
        break;
      }
      case ISD::FSQRT:
      case ISD::FSIN:
      case ISD::FCOS: {
        MVT::ValueType VT = Node->getValueType(0);
        Type *T = VT == MVT::f32 ? Type::FloatTy : Type::DoubleTy;
        const char *FnName = 0;
        switch(Node->getOpcode()) {
        case ISD::FSQRT: FnName = VT == MVT::f32 ? "sqrtf" : "sqrt"; break;
        case ISD::FSIN:  FnName = VT == MVT::f32 ? "sinf"  : "sin"; break;
        case ISD::FCOS:  FnName = VT == MVT::f32 ? "cosf"  : "cos"; break;
        default: assert(0 && "Unreachable!");
        }
        std::vector<std::pair<SDOperand, const Type*> > Args;
        Args.push_back(std::make_pair(Tmp1, T));
        // FIXME: should use ExpandLibCall!
        std::pair<SDOperand,SDOperand> CallResult =
          TLI.LowerCallTo(DAG.getEntryNode(), T, false, CallingConv::C, true,
                          DAG.getExternalSymbol(FnName, VT), Args, DAG);
        Result = LegalizeOp(CallResult.first);
        break;
      }
      default:
        assert(0 && "Unreachable!");
      }
      break;
    }
    break;

    // Conversion operators.  The source and destination have different types.
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::TRUNCATE:
  case ISD::FP_EXTEND:
  case ISD::FP_ROUND:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      //still made need to expand if the op is illegal, but the types are legal
      if (Node->getOpcode() == ISD::UINT_TO_FP &&
          TLI.getOperationAction(Node->getOpcode(), 
                                 Node->getOperand(0).getValueType()) 
          == TargetLowering::Expand) {
        SDOperand Op0 = LegalizeOp(Node->getOperand(0));
        Tmp1 = DAG.getNode(ISD::SINT_TO_FP, Node->getValueType(0), 
                           Op0);
        
        SDOperand SignSet = DAG.getSetCC(ISD::SETLT, TLI.getSetCCResultTy(), 
                                         Op0,
                                         DAG.getConstant(0, 
                                         Op0.getValueType()));
        SDOperand Zero = getIntPtrConstant(0), Four = getIntPtrConstant(4);
        SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                          SignSet, Four, Zero);
        uint64_t FF = 0x5f800000ULL;
        if (TLI.isLittleEndian()) FF <<= 32;
        static Constant *FudgeFactor = ConstantUInt::get(Type::ULongTy, FF);

        MachineConstantPool *CP = DAG.getMachineFunction().getConstantPool();
        SDOperand CPIdx = DAG.getConstantPool(CP->getConstantPoolIndex(FudgeFactor),
                                              TLI.getPointerTy());
        CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
        SDOperand FudgeInReg;
        if (Node->getValueType(0) == MVT::f32)
          FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                                   DAG.getSrcValue(NULL));
        else {
          assert(Node->getValueType(0) == MVT::f64 && "Unexpected conversion");
          FudgeInReg = 
            LegalizeOp(DAG.getExtLoad(ISD::EXTLOAD, MVT::f64,
                                      DAG.getEntryNode(), CPIdx,
                                      DAG.getSrcValue(NULL), MVT::f32));
        }
        Result = DAG.getNode(ISD::ADD, Node->getValueType(0), Tmp1, FudgeInReg);
        break;
      }
      Tmp1 = LegalizeOp(Node->getOperand(0));
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case Expand:
      if (Node->getOpcode() == ISD::SINT_TO_FP ||
          Node->getOpcode() == ISD::UINT_TO_FP) {
        Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP,
                               Node->getValueType(0), Node->getOperand(0));
        break;
      } else if (Node->getOpcode() == ISD::TRUNCATE) {
        // In the expand case, we must be dealing with a truncate, because
        // otherwise the result would be larger than the source.
        ExpandOp(Node->getOperand(0), Tmp1, Tmp2);

        // Since the result is legal, we should just be able to truncate the low
        // part of the source.
        Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), Tmp1);
        break;
      }
      assert(0 && "Shouldn't need to expand other operators here!");

    case Promote:
      switch (Node->getOpcode()) {
      case ISD::ZERO_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        // NOTE: Any extend would work here...
        Result = DAG.getNode(ISD::ZERO_EXTEND, Op.getValueType(), Result);
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
        break;
      case ISD::SIGN_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        // NOTE: Any extend would work here...
        Result = DAG.getNode(ISD::ZERO_EXTEND, Op.getValueType(), Result);
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                          DAG.getValueType(Node->getOperand(0).getValueType()));
        break;
      case ISD::TRUNCATE:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::TRUNCATE, Op.getValueType(), Result);
        break;
      case ISD::FP_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        if (Result.getValueType() != Op.getValueType())
          // Dynamically dead while we have only 2 FP types.
          Result = DAG.getNode(ISD::FP_EXTEND, Op.getValueType(), Result);
        break;
      case ISD::FP_ROUND:
      case ISD::FP_TO_SINT:
      case ISD::FP_TO_UINT:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(Node->getOpcode(), Op.getValueType(), Result);
        break;
      case ISD::SINT_TO_FP:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
        Result = DAG.getNode(ISD::SINT_TO_FP, Op.getValueType(), Result);
        break;
      case ISD::UINT_TO_FP:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
        Result = DAG.getNode(ISD::UINT_TO_FP, Op.getValueType(), Result);
        break;
      }
    }
    break;
  case ISD::FP_ROUND_INREG:
  case ISD::SIGN_EXTEND_INREG: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    MVT::ValueType ExtraVT = cast<VTSDNode>(Node->getOperand(1))->getVT();

    // If this operation is not supported, convert it to a shl/shr or load/store
    // pair.
    switch (TLI.getOperationAction(Node->getOpcode(), ExtraVT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1,
                             DAG.getValueType(ExtraVT));
      break;
    case TargetLowering::Expand:
      // If this is an integer extend and shifts are supported, do that.
      if (Node->getOpcode() == ISD::SIGN_EXTEND_INREG) {
        // NOTE: we could fall back on load/store here too for targets without
        // SAR.  However, it is doubtful that any exist.
        unsigned BitsDiff = MVT::getSizeInBits(Node->getValueType(0)) -
                            MVT::getSizeInBits(ExtraVT);
        SDOperand ShiftCst = DAG.getConstant(BitsDiff, TLI.getShiftAmountTy());
        Result = DAG.getNode(ISD::SHL, Node->getValueType(0),
                             Node->getOperand(0), ShiftCst);
        Result = DAG.getNode(ISD::SRA, Node->getValueType(0),
                             Result, ShiftCst);
      } else if (Node->getOpcode() == ISD::FP_ROUND_INREG) {
        // The only way we can lower this is to turn it into a STORETRUNC,
        // EXTLOAD pair, targetting a temporary location (a stack slot).

        // NOTE: there is a choice here between constantly creating new stack
        // slots and always reusing the same one.  We currently always create
        // new ones, as reuse may inhibit scheduling.
        const Type *Ty = MVT::getTypeForValueType(ExtraVT);
        unsigned TySize = (unsigned)TLI.getTargetData().getTypeSize(Ty);
        unsigned Align  = TLI.getTargetData().getTypeAlignment(Ty);
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI =
          MF.getFrameInfo()->CreateStackObject((unsigned)TySize, Align);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, TLI.getPointerTy());
        Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, DAG.getEntryNode(),
                             Node->getOperand(0), StackSlot,
                             DAG.getSrcValue(NULL), DAG.getValueType(ExtraVT));
        Result = DAG.getExtLoad(ISD::EXTLOAD, Node->getValueType(0),
                                Result, StackSlot, DAG.getSrcValue(NULL),
                                ExtraVT);
      } else {
        assert(0 && "Unknown op");
      }
      Result = LegalizeOp(Result);
      break;
    }
    break;
  }
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

/// PromoteOp - Given an operation that produces a value in an invalid type,
/// promote it to compute the value into a larger type.  The produced value will
/// have the correct bits for the low portion of the register, but no guarantee
/// is made about the top bits: it may be zero, sign-extended, or garbage.
SDOperand SelectionDAGLegalize::PromoteOp(SDOperand Op) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  assert(getTypeAction(VT) == Promote &&
         "Caller should expand or legalize operands that are not promotable!");
  assert(NVT > VT && MVT::isInteger(NVT) == MVT::isInteger(VT) &&
         "Cannot promote to smaller type!");

  SDOperand Tmp1, Tmp2, Tmp3;

  SDOperand Result;
  SDNode *Node = Op.Val;

  if (!Node->hasOneUse()) {
    std::map<SDOperand, SDOperand>::iterator I = PromotedNodes.find(Op);
    if (I != PromotedNodes.end()) return I->second;
  } else {
    assert(!PromotedNodes.count(Op) && "Repromoted this node??");
  }

  // Promotion needs an optimization step to clean up after it, and is not
  // careful to avoid operations the target does not support.  Make sure that
  // all generated operations are legalized in the next iteration.
  NeedsAnotherIteration = true;

  switch (Node->getOpcode()) {
  default:
    std::cerr << "NODE: "; Node->dump(); std::cerr << "\n";
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::UNDEF:
    Result = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::Constant:
    Result = DAG.getNode(ISD::ZERO_EXTEND, NVT, Op);
    assert(isa<ConstantSDNode>(Result) && "Didn't constant fold zext?");
    break;
  case ISD::ConstantFP:
    Result = DAG.getNode(ISD::FP_EXTEND, NVT, Op);
    assert(isa<ConstantFPSDNode>(Result) && "Didn't constant fold fp_extend?");
    break;
  case ISD::CopyFromReg:
    Result = DAG.getCopyFromReg(cast<RegSDNode>(Node)->getReg(), NVT,
                                Node->getOperand(0));
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    break;

  case ISD::SETCC:
    assert(getTypeAction(TLI.getSetCCResultTy()) == Legal &&
           "SetCC type is not legal??");
    Result = DAG.getSetCC(cast<SetCCSDNode>(Node)->getCondition(),
                          TLI.getSetCCResultTy(), Node->getOperand(0),
                          Node->getOperand(1));
    Result = LegalizeOp(Result);
    break;

  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Result = LegalizeOp(Node->getOperand(0));
      assert(Result.getValueType() >= NVT &&
             "This truncation doesn't make sense!");
      if (Result.getValueType() > NVT)    // Truncate to NVT instead of VT
        Result = DAG.getNode(ISD::TRUNCATE, NVT, Result);
      break;
    case Promote:
      // The truncation is not required, because we don't guarantee anything
      // about high bits anyway.
      Result = PromoteOp(Node->getOperand(0));
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      // Truncate the low part of the expanded value to the result type
      Result = DAG.getNode(ISD::TRUNCATE, VT, Tmp1);
    }
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "BUG: Smaller reg should have been promoted!");
    case Legal:
      // Input is legal?  Just do extend all the way to the larger type.
      Result = LegalizeOp(Node->getOperand(0));
      Result = DAG.getNode(Node->getOpcode(), NVT, Result);
      break;
    case Promote:
      // Promote the reg if it's smaller.
      Result = PromoteOp(Node->getOperand(0));
      // The high bits are not guaranteed to be anything.  Insert an extend.
      if (Node->getOpcode() == ISD::SIGN_EXTEND)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
      break;
    }
    break;

  case ISD::FP_EXTEND:
    assert(0 && "Case not implemented.  Dynamically dead with 2 FP types!");
  case ISD::FP_ROUND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "BUG: Cannot expand FP regs!");
    case Promote:  assert(0 && "Unreachable with 2 FP types!");
    case Legal:
      // Input is legal?  Do an FP_ROUND_INREG.
      Result = LegalizeOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
      break;
    }
    break;

  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Result = LegalizeOp(Node->getOperand(0));
      // No extra round required here.
      Result = DAG.getNode(Node->getOpcode(), NVT, Result);
      break;

    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      if (Node->getOpcode() == ISD::SINT_TO_FP)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
      // No extra round required here.
      Result = DAG.getNode(Node->getOpcode(), NVT, Result);
      break;
    case Expand:
      Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP, NVT,
                             Node->getOperand(0));
      // Round if we cannot tolerate excess precision.
      if (NoExcessFPPrecision)
        Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                             DAG.getValueType(VT));
      break;
    }
    break;

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      break;
    case Promote:
      // The input result is prerounded, so we don't have to do anything
      // special.
      Tmp1 = PromoteOp(Node->getOperand(0));
      break;
    case Expand:
      assert(0 && "not implemented");
    }
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    break;

  case ISD::FABS:
  case ISD::FNEG:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    // NOTE: we do not have to do any extra rounding here for
    // NoExcessFPPrecision, because we know the input will have the appropriate
    // precision, and these operations don't modify precision at all.
    break;

  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    if(NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
    // The input may have strange things in the top bits of the registers, but
    // these operations don't care.  They may have wierd bits going out, but
    // that too is okay if they are integer operations.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);

    // However, if this is a floating point operation, they will give excess
    // precision that we may not be able to tolerate.  If we DO allow excess
    // precision, just leave it, otherwise excise it.
    // FIXME: Why would we need to round FP ops more than integer ones?
    //     Is Round(Add(Add(A,B),C)) != Round(Add(Round(Add(A,B)), C))
    if (MVT::isFloatingPoint(NVT) && NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::SDIV:
  case ISD::SREM:
    // These operators require that their input be sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    if (MVT::isInteger(NVT)) {
      Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                         DAG.getValueType(VT));
      Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp2,
                         DAG.getValueType(VT));
    }
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);

    // Perform FP_ROUND: this is probably overly pessimistic.
    if (MVT::isFloatingPoint(NVT) && NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::UDIV:
  case ISD::UREM:
    // These operators require that their input be zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(MVT::isInteger(NVT) && "Operators don't apply to FP!");
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
    Tmp2 = DAG.getZeroExtendInReg(Tmp2, VT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    break;

  case ISD::SHL:
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.getNode(ISD::SHL, NVT, Tmp1, Tmp2);
    break;
  case ISD::SRA:
    // The input value must be properly sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                       DAG.getValueType(VT));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.getNode(ISD::SRA, NVT, Tmp1, Tmp2);
    break;
  case ISD::SRL:
    // The input value must be properly zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.getNode(ISD::SRL, NVT, Tmp1, Tmp2);
    break;
  case ISD::LOAD:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Legalize the pointer.
    // FIXME: When the DAG combiner exists, change this to use EXTLOAD!
    if (MVT::isInteger(NVT))
      Result = DAG.getExtLoad(ISD::ZEXTLOAD, NVT, Tmp1, Tmp2,
                              Node->getOperand(2), VT);
    else
      Result = DAG.getExtLoad(ISD::EXTLOAD, NVT, Tmp1, Tmp2,
                              Node->getOperand(2), VT);

    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    break;
  case ISD::SELECT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));// Legalize the condition.
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0)); // Promote the condition.
      break;
    }
    Tmp2 = PromoteOp(Node->getOperand(1));   // Legalize the op0
    Tmp3 = PromoteOp(Node->getOperand(2));   // Legalize the op1
    Result = DAG.getNode(ISD::SELECT, NVT, Tmp1, Tmp2, Tmp3);
    break;
  case ISD::TAILCALL:
  case ISD::CALL: {
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the callee.

    std::vector<SDOperand> Ops;
    for (unsigned i = 2, e = Node->getNumOperands(); i != e; ++i)
      Ops.push_back(LegalizeOp(Node->getOperand(i)));

    assert(Node->getNumValues() == 2 && Op.ResNo == 0 &&
           "Can only promote single result calls");
    std::vector<MVT::ValueType> RetTyVTs;
    RetTyVTs.reserve(2);
    RetTyVTs.push_back(NVT);
    RetTyVTs.push_back(MVT::Other);
    SDNode *NC = DAG.getCall(RetTyVTs, Tmp1, Tmp2, Ops,
                             Node->getOpcode() == ISD::TAILCALL);
    Result = SDOperand(NC, 0);

    // Insert the new chain mapping.
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    break;
  }
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    Tmp1 = Node->getOperand(0);
    //Zero extend the argument
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Tmp1);
    // Perform the larger operation, then subtract if needed.
    Tmp1 = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    switch(Node->getOpcode())
    {
    case ISD::CTPOP:
      Result = Tmp1;
      break;
    case ISD::CTTZ:
      //if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
      Tmp2 = DAG.getSetCC(ISD::SETEQ, MVT::i1, Tmp1, 
                          DAG.getConstant(getSizeInBits(NVT), NVT));
      Result = DAG.getNode(ISD::SELECT, NVT, Tmp2, 
                           DAG.getConstant(getSizeInBits(VT),NVT), Tmp1);
      break;
    case ISD::CTLZ:
      //Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
      Result = DAG.getNode(ISD::SUB, NVT, Tmp1, 
                           DAG.getConstant(getSizeInBits(NVT) - 
                                           getSizeInBits(VT), NVT));
      break;
    }
    break;
  }

  assert(Result.Val && "Didn't set a result!");
  AddPromotedOperand(Op, Result);
  return Result;
}

/// ExpandAddSub - Find a clever way to expand this add operation into
/// subcomponents.
void SelectionDAGLegalize::
ExpandByParts(unsigned NodeOp, SDOperand LHS, SDOperand RHS,
              SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH, RHSL, RHSH;
  ExpandOp(LHS, LHSL, LHSH);
  ExpandOp(RHS, RHSL, RHSH);

  // FIXME: this should be moved to the dag combiner someday.
  assert(NodeOp == ISD::ADD_PARTS || NodeOp == ISD::SUB_PARTS);
  if (LHSL.getValueType() == MVT::i32) {
    SDOperand LowEl;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(LHSL))
      if (C->getValue() == 0)
        LowEl = RHSL;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHSL))
      if (C->getValue() == 0)
        LowEl = LHSL;
    if (LowEl.Val) {
      // Turn this into an add/sub of the high part only.
      SDOperand HiEl =
        DAG.getNode(NodeOp == ISD::ADD_PARTS ? ISD::ADD : ISD::SUB,
                    LowEl.getValueType(), LHSH, RHSH);
      Lo = LowEl;
      Hi = HiEl;
      return;
    }
  }

  std::vector<SDOperand> Ops;
  Ops.push_back(LHSL);
  Ops.push_back(LHSH);
  Ops.push_back(RHSL);
  Ops.push_back(RHSH);

  std::vector<MVT::ValueType> VTs(2, LHSL.getValueType());
  Lo = DAG.getNode(NodeOp, VTs, Ops);
  Hi = Lo.getValue(1);
}

void SelectionDAGLegalize::ExpandShiftParts(unsigned NodeOp,
                                            SDOperand Op, SDOperand Amt,
                                            SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH;
  ExpandOp(Op, LHSL, LHSH);

  std::vector<SDOperand> Ops;
  Ops.push_back(LHSL);
  Ops.push_back(LHSH);
  Ops.push_back(Amt);
  std::vector<MVT::ValueType> VTs;
  VTs.push_back(LHSL.getValueType());
  VTs.push_back(LHSH.getValueType());
  VTs.push_back(Amt.getValueType());
  Lo = DAG.getNode(NodeOp, VTs, Ops);
  Hi = Lo.getValue(1);
}


/// ExpandShift - Try to find a clever way to expand this shift operation out to
/// smaller elements.  If we can't find a way that is more efficient than a
/// libcall on this target, return false.  Otherwise, return true with the
/// low-parts expanded into Lo and Hi.
bool SelectionDAGLegalize::ExpandShift(unsigned Opc, SDOperand Op,SDOperand Amt,
                                       SDOperand &Lo, SDOperand &Hi) {
  assert((Opc == ISD::SHL || Opc == ISD::SRA || Opc == ISD::SRL) &&
         "This is not a shift!");

  MVT::ValueType NVT = TLI.getTypeToTransformTo(Op.getValueType());
  SDOperand ShAmt = LegalizeOp(Amt);
  MVT::ValueType ShTy = ShAmt.getValueType();
  unsigned VTBits = MVT::getSizeInBits(Op.getValueType());
  unsigned NVTBits = MVT::getSizeInBits(NVT);

  // Handle the case when Amt is an immediate.  Other cases are currently broken
  // and are disabled.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Amt.Val)) {
    unsigned Cst = CN->getValue();
    // Expand the incoming operand to be shifted, so that we have its parts
    SDOperand InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Cst-NVTBits,ShTy));
      } else if (Cst == NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = InL;
      } else {
        Lo = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Cst, ShTy));
        Hi = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(NVTBits-Cst, ShTy)));
      }
      return true;
    case ISD::SRL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Cst-NVTBits,ShTy));
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getConstant(0, NVT);
      } else {
        Lo = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Cst, ShTy));
      }
      return true;
    case ISD::SRA:
      if (Cst > VTBits) {
        Hi = Lo = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRA, NVT, InH,
                           DAG.getConstant(Cst-NVTBits, ShTy));
        Hi = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else {
        Lo = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRA, NVT, InH, DAG.getConstant(Cst, ShTy));
      }
      return true;
    }
  }
  // FIXME: The following code for expanding shifts using ISD::SELECT is buggy,
  // so disable it for now.  Currently targets are handling this via SHL_PARTS
  // and friends.
  return false;

  // If we have an efficient select operation (or if the selects will all fold
  // away), lower to some complex code, otherwise just emit the libcall.
  if (TLI.getOperationAction(ISD::SELECT, NVT) != TargetLowering::Legal &&
      !isa<ConstantSDNode>(Amt))
    return false;

  SDOperand InL, InH;
  ExpandOp(Op, InL, InH);
  SDOperand NAmt = DAG.getNode(ISD::SUB, ShTy,           // NAmt = 32-ShAmt
                               DAG.getConstant(NVTBits, ShTy), ShAmt);

  // Compare the unmasked shift amount against 32.
  SDOperand Cond = DAG.getSetCC(ISD::SETGE, TLI.getSetCCResultTy(), ShAmt,
                                DAG.getConstant(NVTBits, ShTy));

  if (TLI.getShiftAmountFlavor() != TargetLowering::Mask) {
    ShAmt = DAG.getNode(ISD::AND, ShTy, ShAmt,             // ShAmt &= 31
                        DAG.getConstant(NVTBits-1, ShTy));
    NAmt  = DAG.getNode(ISD::AND, ShTy, NAmt,              // NAmt &= 31
                        DAG.getConstant(NVTBits-1, ShTy));
  }

  if (Opc == ISD::SHL) {
    SDOperand T1 = DAG.getNode(ISD::OR, NVT,// T1 = (Hi << Amt) | (Lo >> NAmt)
                               DAG.getNode(ISD::SHL, NVT, InH, ShAmt),
                               DAG.getNode(ISD::SRL, NVT, InL, NAmt));
    SDOperand T2 = DAG.getNode(ISD::SHL, NVT, InL, ShAmt); // T2 = Lo << Amt&31

    Hi = DAG.getNode(ISD::SELECT, NVT, Cond, T2, T1);
    Lo = DAG.getNode(ISD::SELECT, NVT, Cond, DAG.getConstant(0, NVT), T2);
  } else {
    SDOperand HiLoPart = DAG.getNode(ISD::SELECT, NVT,
                                     DAG.getSetCC(ISD::SETEQ,
                                                  TLI.getSetCCResultTy(), NAmt,
                                                  DAG.getConstant(32, ShTy)),
                                     DAG.getConstant(0, NVT),
                                     DAG.getNode(ISD::SHL, NVT, InH, NAmt));
    SDOperand T1 = DAG.getNode(ISD::OR, NVT,// T1 = (Hi << NAmt) | (Lo >> Amt)
                               HiLoPart,
                               DAG.getNode(ISD::SRL, NVT, InL, ShAmt));
    SDOperand T2 = DAG.getNode(Opc, NVT, InH, ShAmt);  // T2 = InH >> ShAmt&31

    SDOperand HiPart;
    if (Opc == ISD::SRA)
      HiPart = DAG.getNode(ISD::SRA, NVT, InH,
                           DAG.getConstant(NVTBits-1, ShTy));
    else
      HiPart = DAG.getConstant(0, NVT);
    Lo = DAG.getNode(ISD::SELECT, NVT, Cond, T2, T1);
    Hi = DAG.getNode(ISD::SELECT, NVT, Cond, HiPart, T2);
  }
  return true;
}

/// FindLatestCallSeqStart - Scan up the dag to find the latest (highest
/// NodeDepth) node that is an CallSeqStart operation and occurs later than
/// Found.
static void FindLatestCallSeqStart(SDNode *Node, SDNode *&Found) {
  if (Node->getNodeDepth() <= Found->getNodeDepth()) return;

  // If we found an CALLSEQ_START, we already know this node occurs later
  // than the Found node. Just remember this node and return.
  if (Node->getOpcode() == ISD::CALLSEQ_START) {
    Found = Node;
    return;
  }

  // Otherwise, scan the operands of Node to see if any of them is a call.
  assert(Node->getNumOperands() != 0 &&
         "All leaves should have depth equal to the entry node!");
  for (unsigned i = 0, e = Node->getNumOperands()-1; i != e; ++i)
    FindLatestCallSeqStart(Node->getOperand(i).Val, Found);

  // Tail recurse for the last iteration.
  FindLatestCallSeqStart(Node->getOperand(Node->getNumOperands()-1).Val,
                             Found);
}


/// FindEarliestCallSeqEnd - Scan down the dag to find the earliest (lowest
/// NodeDepth) node that is an CallSeqEnd operation and occurs more recent
/// than Found.
static void FindEarliestCallSeqEnd(SDNode *Node, SDNode *&Found) {
  if (Found && Node->getNodeDepth() >= Found->getNodeDepth()) return;

  // If we found an CALLSEQ_END, we already know this node occurs earlier
  // than the Found node. Just remember this node and return.
  if (Node->getOpcode() == ISD::CALLSEQ_END) {
    Found = Node;
    return;
  }

  // Otherwise, scan the operands of Node to see if any of them is a call.
  SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
  if (UI == E) return;
  for (--E; UI != E; ++UI)
    FindEarliestCallSeqEnd(*UI, Found);

  // Tail recurse for the last iteration.
  FindEarliestCallSeqEnd(*UI, Found);
}

/// FindCallSeqEnd - Given a chained node that is part of a call sequence,
/// find the CALLSEQ_END node that terminates the call sequence.
static SDNode *FindCallSeqEnd(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_END)
    return Node;
  if (Node->use_empty())
    return 0;   // No CallSeqEnd

  if (Node->hasOneUse())  // Simple case, only has one user to check.
    return FindCallSeqEnd(*Node->use_begin());

  SDOperand TheChain(Node, Node->getNumValues()-1);
  if (TheChain.getValueType() != MVT::Other)
    TheChain = SDOperand(Node, 0);
  assert(TheChain.getValueType() == MVT::Other && "Is not a token chain!");

  for (SDNode::use_iterator UI = Node->use_begin(),
         E = Node->use_end(); ; ++UI) {
    assert(UI != E && "Didn't find a user of the tokchain, no CALLSEQ_END!");

    // Make sure to only follow users of our token chain.
    SDNode *User = *UI;
    for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
      if (User->getOperand(i) == TheChain)
        if (SDNode *Result = FindCallSeqEnd(User))
          return Result;
  }
  assert(0 && "Unreachable");
  abort();
}

/// FindCallSeqStart - Given a chained node that is part of a call sequence,
/// find the CALLSEQ_START node that initiates the call sequence.
static SDNode *FindCallSeqStart(SDNode *Node) {
  assert(Node && "Didn't find callseq_start for a call??");
  if (Node->getOpcode() == ISD::CALLSEQ_START) return Node;

  assert(Node->getOperand(0).getValueType() == MVT::Other &&
         "Node doesn't have a token chain argument!");
  return FindCallSeqStart(Node->getOperand(0).Val);
}


/// FindInputOutputChains - If we are replacing an operation with a call we need
/// to find the call that occurs before and the call that occurs after it to
/// properly serialize the calls in the block.  The returned operand is the
/// input chain value for the new call (e.g. the entry node or the previous
/// call), and OutChain is set to be the chain node to update to point to the
/// end of the call chain.
static SDOperand FindInputOutputChains(SDNode *OpNode, SDNode *&OutChain,
                                       SDOperand Entry) {
  SDNode *LatestCallSeqStart = Entry.Val;
  SDNode *LatestCallSeqEnd = 0;
  FindLatestCallSeqStart(OpNode, LatestCallSeqStart);
  //std::cerr<<"Found node: "; LatestCallSeqStart->dump(); std::cerr <<"\n";

  // It is possible that no ISD::CALLSEQ_START was found because there is no
  // previous call in the function.  LatestCallStackDown may in that case be
  // the entry node itself.  Do not attempt to find a matching CALLSEQ_END
  // unless LatestCallStackDown is an CALLSEQ_START.
  if (LatestCallSeqStart->getOpcode() == ISD::CALLSEQ_START)
    LatestCallSeqEnd = FindCallSeqEnd(LatestCallSeqStart);
  else
    LatestCallSeqEnd = Entry.Val;
  assert(LatestCallSeqEnd && "NULL return from FindCallSeqEnd");

  // Finally, find the first call that this must come before, first we find the
  // CallSeqEnd that ends the call.
  OutChain = 0;
  FindEarliestCallSeqEnd(OpNode, OutChain);

  // If we found one, translate from the adj up to the callseq_start.
  if (OutChain)
    OutChain = FindCallSeqStart(OutChain);

  return SDOperand(LatestCallSeqEnd, 0);
}

/// SpliceCallInto - Given the result chain of a libcall (CallResult), and a 
void SelectionDAGLegalize::SpliceCallInto(const SDOperand &CallResult,
                                          SDNode *OutChain) {
  // Nothing to splice it into?
  if (OutChain == 0) return;

  assert(OutChain->getOperand(0).getValueType() == MVT::Other);
  //OutChain->dump();

  // Form a token factor node merging the old inval and the new inval.
  SDOperand InToken = DAG.getNode(ISD::TokenFactor, MVT::Other, CallResult,
                                  OutChain->getOperand(0));
  // Change the node to refer to the new token.
  OutChain->setAdjCallChain(InToken);
}


// ExpandLibCall - Expand a node into a call to a libcall.  If the result value
// does not fit into a register, return the lo part and set the hi part to the
// by-reg argument.  If it does fit into a single register, return the result
// and leave the Hi part unset.
SDOperand SelectionDAGLegalize::ExpandLibCall(const char *Name, SDNode *Node,
                                              SDOperand &Hi) {
  SDNode *OutChain;
  SDOperand InChain = FindInputOutputChains(Node, OutChain,
                                            DAG.getEntryNode());
  if (InChain.Val == 0)
    InChain = DAG.getEntryNode();

  TargetLowering::ArgListTy Args;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    MVT::ValueType ArgVT = Node->getOperand(i).getValueType();
    const Type *ArgTy = MVT::getTypeForValueType(ArgVT);
    Args.push_back(std::make_pair(Node->getOperand(i), ArgTy));
  }
  SDOperand Callee = DAG.getExternalSymbol(Name, TLI.getPointerTy());

  // Splice the libcall in wherever FindInputOutputChains tells us to.
  const Type *RetTy = MVT::getTypeForValueType(Node->getValueType(0));
  std::pair<SDOperand,SDOperand> CallInfo =
    TLI.LowerCallTo(InChain, RetTy, false, CallingConv::C, false,
                    Callee, Args, DAG);
  SpliceCallInto(CallInfo.second, OutChain);

  NeedsAnotherIteration = true;

  switch (getTypeAction(CallInfo.first.getValueType())) {
  default: assert(0 && "Unknown thing");
  case Legal:
    return CallInfo.first;
  case Promote:
    assert(0 && "Cannot promote this yet!");
  case Expand:
    SDOperand Lo;
    ExpandOp(CallInfo.first, Lo, Hi);
    return Lo;
  }
}


/// ExpandIntToFP - Expand a [US]INT_TO_FP operation, assuming that the
/// destination type is legal.
SDOperand SelectionDAGLegalize::
ExpandIntToFP(bool isSigned, MVT::ValueType DestTy, SDOperand Source) {
  assert(getTypeAction(DestTy) == Legal && "Destination type is not legal!");
  assert(getTypeAction(Source.getValueType()) == Expand &&
         "This is not an expansion!");
  assert(Source.getValueType() == MVT::i64 && "Only handle expand from i64!");

  if (!isSigned) {
    assert(Source.getValueType() == MVT::i64 &&
           "This only works for 64-bit -> FP");
    // The 64-bit value loaded will be incorrectly if the 'sign bit' of the
    // incoming integer is set.  To handle this, we dynamically test to see if
    // it is set, and, if so, add a fudge factor.
    SDOperand Lo, Hi;
    ExpandOp(Source, Lo, Hi);

    // If this is unsigned, and not supported, first perform the conversion to
    // signed, then adjust the result if the sign bit is set.
    SDOperand SignedConv = ExpandIntToFP(true, DestTy,
                   DAG.getNode(ISD::BUILD_PAIR, Source.getValueType(), Lo, Hi));

    SDOperand SignSet = DAG.getSetCC(ISD::SETLT, TLI.getSetCCResultTy(), Hi,
                                     DAG.getConstant(0, Hi.getValueType()));
    SDOperand Zero = getIntPtrConstant(0), Four = getIntPtrConstant(4);
    SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                      SignSet, Four, Zero);
    uint64_t FF = 0x5f800000ULL;
    if (TLI.isLittleEndian()) FF <<= 32;
    static Constant *FudgeFactor = ConstantUInt::get(Type::ULongTy, FF);

    MachineConstantPool *CP = DAG.getMachineFunction().getConstantPool();
    SDOperand CPIdx = DAG.getConstantPool(CP->getConstantPoolIndex(FudgeFactor),
                                          TLI.getPointerTy());
    CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
    SDOperand FudgeInReg;
    if (DestTy == MVT::f32)
      FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                               DAG.getSrcValue(NULL));
    else {
      assert(DestTy == MVT::f64 && "Unexpected conversion");
      FudgeInReg = DAG.getExtLoad(ISD::EXTLOAD, MVT::f64, DAG.getEntryNode(),
                                  CPIdx, DAG.getSrcValue(NULL), MVT::f32);
    }
    return DAG.getNode(ISD::ADD, DestTy, SignedConv, FudgeInReg);
  }

  // Check to see if the target has a custom way to lower this.  If so, use it.
  switch (TLI.getOperationAction(ISD::SINT_TO_FP, Source.getValueType())) {
  default: assert(0 && "This action not implemented for this operation!");
  case TargetLowering::Legal:
  case TargetLowering::Expand:
    break;   // This case is handled below.
  case TargetLowering::Custom:
    Source = DAG.getNode(ISD::SINT_TO_FP, DestTy, Source);
    return LegalizeOp(TLI.LowerOperation(Source, DAG));
  }

  // Expand the source, then glue it back together for the call.  We must expand
  // the source in case it is shared (this pass of legalize must traverse it).
  SDOperand SrcLo, SrcHi;
  ExpandOp(Source, SrcLo, SrcHi);
  Source = DAG.getNode(ISD::BUILD_PAIR, Source.getValueType(), SrcLo, SrcHi);

  SDNode *OutChain = 0;
  SDOperand InChain = FindInputOutputChains(Source.Val, OutChain,
                                            DAG.getEntryNode());
  const char *FnName = 0;
  if (DestTy == MVT::f32)
    FnName = "__floatdisf";
  else {
    assert(DestTy == MVT::f64 && "Unknown fp value type!");
    FnName = "__floatdidf";
  }

  SDOperand Callee = DAG.getExternalSymbol(FnName, TLI.getPointerTy());

  TargetLowering::ArgListTy Args;
  const Type *ArgTy = MVT::getTypeForValueType(Source.getValueType());

  Args.push_back(std::make_pair(Source, ArgTy));

  // We don't care about token chains for libcalls.  We just use the entry
  // node as our input and ignore the output chain.  This allows us to place
  // calls wherever we need them to satisfy data dependences.
  const Type *RetTy = MVT::getTypeForValueType(DestTy);

  std::pair<SDOperand,SDOperand> CallResult =
    TLI.LowerCallTo(InChain, RetTy, false, CallingConv::C, true,
                    Callee, Args, DAG);

  SpliceCallInto(CallResult.second, OutChain);
  return CallResult.first;
}



/// ExpandOp - Expand the specified SDOperand into its two component pieces
/// Lo&Hi.  Note that the Op MUST be an expanded type.  As a result of this, the
/// LegalizeNodes map is filled in for any results that are not expanded, the
/// ExpandedNodes map is filled in for any results that are expanded, and the
/// Lo/Hi values are returned.
void SelectionDAGLegalize::ExpandOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi){
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDNode *Node = Op.Val;
  assert(getTypeAction(VT) == Expand && "Not an expanded type!");
  assert(MVT::isInteger(VT) && "Cannot expand FP values!");
  assert(MVT::isInteger(NVT) && NVT < VT &&
         "Cannot expand to FP value or to larger int value!");

  // If there is more than one use of this, see if we already expanded it.
  // There is no use remembering values that only have a single use, as the map
  // entries will never be reused.
  if (!Node->hasOneUse()) {
    std::map<SDOperand, std::pair<SDOperand, SDOperand> >::iterator I
      = ExpandedNodes.find(Op);
    if (I != ExpandedNodes.end()) {
      Lo = I->second.first;
      Hi = I->second.second;
      return;
    }
  } else {
    assert(!ExpandedNodes.count(Op) && "Re-expanding a node!");
  }

  // Expanding to multiple registers needs to perform an optimization step, and
  // is not careful to avoid operations the target does not support.  Make sure
  // that all generated operations are legalized in the next iteration.
  NeedsAnotherIteration = true;

  switch (Node->getOpcode()) {
  default:
    std::cerr << "NODE: "; Node->dump(); std::cerr << "\n";
    assert(0 && "Do not know how to expand this operator!");
    abort();
  case ISD::UNDEF:
    Lo = DAG.getNode(ISD::UNDEF, NVT);
    Hi = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::Constant: {
    uint64_t Cst = cast<ConstantSDNode>(Node)->getValue();
    Lo = DAG.getConstant(Cst, NVT);
    Hi = DAG.getConstant(Cst >> MVT::getSizeInBits(NVT), NVT);
    break;
  }

  case ISD::CopyFromReg: {
    unsigned Reg = cast<RegSDNode>(Node)->getReg();
    // Aggregate register values are always in consequtive pairs.
    Lo = DAG.getCopyFromReg(Reg, NVT, Node->getOperand(0));
    Hi = DAG.getCopyFromReg(Reg+1, NVT, Lo.getValue(1));

    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), Hi.getValue(1));

    assert(isTypeLegal(NVT) && "Cannot expand this multiple times yet!");
    break;
  }

  case ISD::BUILD_PAIR:
    // Legalize both operands.  FIXME: in the future we should handle the case
    // where the two elements are not legal.
    assert(isTypeLegal(NVT) && "Cannot expand this multiple times yet!");
    Lo = LegalizeOp(Node->getOperand(0));
    Hi = LegalizeOp(Node->getOperand(1));
    break;

  case ISD::CTPOP:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    Lo = DAG.getNode(ISD::ADD, NVT,          // ctpop(HL) -> ctpop(H)+ctpop(L)
                     DAG.getNode(ISD::CTPOP, NVT, Lo),
                     DAG.getNode(ISD::CTPOP, NVT, Hi));
    Hi = DAG.getConstant(0, NVT);
    break;

  case ISD::CTLZ: {
    // ctlz (HL) -> ctlz(H) != 32 ? ctlz(H) : (ctlz(L)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDOperand BitsC = DAG.getConstant(MVT::getSizeInBits(NVT), NVT);
    SDOperand HLZ = DAG.getNode(ISD::CTLZ, NVT, Hi);
    SDOperand TopNotZero = DAG.getSetCC(ISD::SETNE, TLI.getSetCCResultTy(),
                                        HLZ, BitsC);
    SDOperand LowPart = DAG.getNode(ISD::CTLZ, NVT, Lo);
    LowPart = DAG.getNode(ISD::ADD, NVT, LowPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, NVT, TopNotZero, HLZ, LowPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::CTTZ: {
    // cttz (HL) -> cttz(L) != 32 ? cttz(L) : (cttz(H)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDOperand BitsC = DAG.getConstant(MVT::getSizeInBits(NVT), NVT);
    SDOperand LTZ = DAG.getNode(ISD::CTTZ, NVT, Lo);
    SDOperand BotNotZero = DAG.getSetCC(ISD::SETNE, TLI.getSetCCResultTy(),
                                        LTZ, BitsC);
    SDOperand HiPart = DAG.getNode(ISD::CTTZ, NVT, Hi);
    HiPart = DAG.getNode(ISD::ADD, NVT, HiPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, NVT, BotNotZero, LTZ, HiPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::LOAD: {
    SDOperand Ch = LegalizeOp(Node->getOperand(0));   // Legalize the chain.
    SDOperand Ptr = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.
    Lo = DAG.getLoad(NVT, Ch, Ptr, Node->getOperand(2));

    // Increment the pointer to the other half.
    unsigned IncrementSize = MVT::getSizeInBits(Lo.getValueType())/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    //Is this safe?  declaring that the two parts of the split load  
    //are from the same instruction?
    Hi = DAG.getLoad(NVT, Ch, Ptr, Node->getOperand(2));

    // Build a factor node to remember that this load is independent of the
    // other one.
    SDOperand TF = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                               Hi.getValue(1));

    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), TF);
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
    break;
  }
  case ISD::TAILCALL:
  case ISD::CALL: {
    SDOperand Chain  = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    SDOperand Callee = LegalizeOp(Node->getOperand(1));  // Legalize the callee.

    bool Changed = false;
    std::vector<SDOperand> Ops;
    for (unsigned i = 2, e = Node->getNumOperands(); i != e; ++i) {
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Changed |= Ops.back() != Node->getOperand(i);
    }

    assert(Node->getNumValues() == 2 && Op.ResNo == 0 &&
           "Can only expand a call once so far, not i64 -> i16!");

    std::vector<MVT::ValueType> RetTyVTs;
    RetTyVTs.reserve(3);
    RetTyVTs.push_back(NVT);
    RetTyVTs.push_back(NVT);
    RetTyVTs.push_back(MVT::Other);
    SDNode *NC = DAG.getCall(RetTyVTs, Chain, Callee, Ops,
                             Node->getOpcode() == ISD::TAILCALL);
    Lo = SDOperand(NC, 0);
    Hi = SDOperand(NC, 1);

    // Insert the new chain mapping.
    AddLegalizedOperand(Op.getValue(1), Hi.getValue(2));
    break;
  }
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {   // Simple logical operators -> two trivial pieces.
    SDOperand LL, LH, RL, RH;
    ExpandOp(Node->getOperand(0), LL, LH);
    ExpandOp(Node->getOperand(1), RL, RH);
    Lo = DAG.getNode(Node->getOpcode(), NVT, LL, RL);
    Hi = DAG.getNode(Node->getOpcode(), NVT, LH, RH);
    break;
  }
  case ISD::SELECT: {
    SDOperand C, LL, LH, RL, RH;

    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      C = LegalizeOp(Node->getOperand(0)); // Legalize the condition.
      break;
    case Promote:
      C = PromoteOp(Node->getOperand(0));  // Promote the condition.
      break;
    }
    ExpandOp(Node->getOperand(1), LL, LH);
    ExpandOp(Node->getOperand(2), RL, RH);
    Lo = DAG.getNode(ISD::SELECT, NVT, C, LL, RL);
    Hi = DAG.getNode(ISD::SELECT, NVT, C, LH, RH);
    break;
  }
  case ISD::SIGN_EXTEND: {
    SDOperand In;
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "expand-expand not implemented yet!");
    case Legal: In = LegalizeOp(Node->getOperand(0)); break;
    case Promote:
      In = PromoteOp(Node->getOperand(0));
      // Emit the appropriate sign_extend_inreg to get the value we want.
      In = DAG.getNode(ISD::SIGN_EXTEND_INREG, In.getValueType(), In,
                       DAG.getValueType(Node->getOperand(0).getValueType()));
      break;
    }

    // The low part is just a sign extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::SIGN_EXTEND, NVT, In);

    // The high part is obtained by SRA'ing all but one of the bits of the lo
    // part.
    unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
    Hi = DAG.getNode(ISD::SRA, NVT, Lo, DAG.getConstant(LoSize-1,
                                                       TLI.getShiftAmountTy()));
    break;
  }
  case ISD::ZERO_EXTEND: {
    SDOperand In;
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "expand-expand not implemented yet!");
    case Legal: In = LegalizeOp(Node->getOperand(0)); break;
    case Promote:
      In = PromoteOp(Node->getOperand(0));
      // Emit the appropriate zero_extend_inreg to get the value we want.
      In = DAG.getZeroExtendInReg(In, Node->getOperand(0).getValueType());
      break;
    }

    // The low part is just a zero extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, In);

    // The high part is just a zero.
    Hi = DAG.getConstant(0, NVT);
    break;
  }
    // These operators cannot be expanded directly, emit them as calls to
    // library functions.
  case ISD::FP_TO_SINT:
    if (Node->getOperand(0).getValueType() == MVT::f32)
      Lo = ExpandLibCall("__fixsfdi", Node, Hi);
    else
      Lo = ExpandLibCall("__fixdfdi", Node, Hi);
    break;
  case ISD::FP_TO_UINT:
    if (Node->getOperand(0).getValueType() == MVT::f32)
      Lo = ExpandLibCall("__fixunssfdi", Node, Hi);
    else
      Lo = ExpandLibCall("__fixunsdfdi", Node, Hi);
    break;

  case ISD::SHL:
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SHL, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SHL_PARTS, use it.
    if (TLI.getOperationAction(ISD::SHL_PARTS, NVT) == TargetLowering::Legal) {
      ExpandShiftParts(ISD::SHL_PARTS, Node->getOperand(0), Node->getOperand(1),
                       Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall("__ashldi3", Node, Hi);
    break;

  case ISD::SRA:
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRA, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SRA_PARTS, use it.
    if (TLI.getOperationAction(ISD::SRA_PARTS, NVT) == TargetLowering::Legal) {
      ExpandShiftParts(ISD::SRA_PARTS, Node->getOperand(0), Node->getOperand(1),
                       Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall("__ashrdi3", Node, Hi);
    break;
  case ISD::SRL:
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRL, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SRL_PARTS, use it.
    if (TLI.getOperationAction(ISD::SRL_PARTS, NVT) == TargetLowering::Legal) {
      ExpandShiftParts(ISD::SRL_PARTS, Node->getOperand(0), Node->getOperand(1),
                       Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall("__lshrdi3", Node, Hi);
    break;

  case ISD::ADD:
    ExpandByParts(ISD::ADD_PARTS, Node->getOperand(0), Node->getOperand(1),
                  Lo, Hi);
    break;
  case ISD::SUB:
    ExpandByParts(ISD::SUB_PARTS, Node->getOperand(0), Node->getOperand(1),
                  Lo, Hi);
    break;
  case ISD::MUL: {
    if (TLI.getOperationAction(ISD::MULHU, NVT) == TargetLowering::Legal) {
      SDOperand LL, LH, RL, RH;
      ExpandOp(Node->getOperand(0), LL, LH);
      ExpandOp(Node->getOperand(1), RL, RH);
      Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
      RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
      Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
    } else {
      Lo = ExpandLibCall("__muldi3" , Node, Hi); break;
    }
    break;
  }
  case ISD::SDIV: Lo = ExpandLibCall("__divdi3" , Node, Hi); break;
  case ISD::UDIV: Lo = ExpandLibCall("__udivdi3", Node, Hi); break;
  case ISD::SREM: Lo = ExpandLibCall("__moddi3" , Node, Hi); break;
  case ISD::UREM: Lo = ExpandLibCall("__umoddi3", Node, Hi); break;
  }

  // Remember in a map if the values will be reused later.
  if (!Node->hasOneUse()) {
    bool isNew = ExpandedNodes.insert(std::make_pair(Op,
                                            std::make_pair(Lo, Hi))).second;
    assert(isNew && "Value already expanded?!?");
  }
}


// SelectionDAG::Legalize - This is the entry point for the file.
//
void SelectionDAG::Legalize() {
  /// run - This is the main entry point to this class.
  ///
  SelectionDAGLegalize(*this).Run();
}

