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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include <iostream>
#include <set>
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

  SDOperand ExpandLegalINT_TO_FP(bool isSigned,
                                 SDOperand LegalOp,
                                 MVT::ValueType DestVT);
  SDOperand PromoteLegalINT_TO_FP(SDOperand LegalOp, MVT::ValueType DestVT,
                                  bool isSigned);
  SDOperand PromoteLegalFP_TO_INT(SDOperand LegalOp, MVT::ValueType DestVT,
                                  bool isSigned);

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

static unsigned getScalarizedOpcode(unsigned VecOp, MVT::ValueType VT) {
  switch (VecOp) {
  default: assert(0 && "Don't know how to scalarize this opcode!");
  case ISD::VADD: return MVT::isInteger(VT) ? ISD::ADD : ISD::FADD;
  case ISD::VSUB: return MVT::isInteger(VT) ? ISD::SUB : ISD::FSUB;
  case ISD::VMUL: return MVT::isInteger(VT) ? ISD::MUL : ISD::FMUL;
  }
}

SelectionDAGLegalize::SelectionDAGLegalize(SelectionDAG &dag)
  : TLI(dag.getTargetLoweringInfo()), DAG(dag),
    ValueTypeActions(TLI.getValueTypeActions()) {
  assert(MVT::LAST_VALUETYPE <= 16 &&
         "Too many value types for ValueTypeActions to hold!");
}

/// ExpandLegalINT_TO_FP - This function is responsible for legalizing a
/// INT_TO_FP operation of the specified operand when the target requests that
/// we expand it.  At this point, we know that the result and operand types are
/// legal for the target.
SDOperand SelectionDAGLegalize::ExpandLegalINT_TO_FP(bool isSigned,
                                                     SDOperand Op0,
                                                     MVT::ValueType DestVT) {
  if (Op0.getValueType() == MVT::i32) {
    // simple 32-bit [signed|unsigned] integer to float/double expansion
    
    // get the stack frame index of a 8 byte buffer
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
    // get address of 8 byte buffer
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, TLI.getPointerTy());
    // word offset constant for Hi/Lo address computation
    SDOperand WordOff = DAG.getConstant(sizeof(int), TLI.getPointerTy());
    // set up Hi and Lo (into buffer) address based on endian
    SDOperand Hi, Lo;
    if (TLI.isLittleEndian()) {
      Hi = DAG.getNode(ISD::ADD, TLI.getPointerTy(), StackSlot, WordOff);
      Lo = StackSlot;
    } else {
      Hi = StackSlot;
      Lo = DAG.getNode(ISD::ADD, TLI.getPointerTy(), StackSlot, WordOff);
    }
    // if signed map to unsigned space
    SDOperand Op0Mapped;
    if (isSigned) {
      // constant used to invert sign bit (signed to unsigned mapping)
      SDOperand SignBit = DAG.getConstant(0x80000000u, MVT::i32);
      Op0Mapped = DAG.getNode(ISD::XOR, MVT::i32, Op0, SignBit);
    } else {
      Op0Mapped = Op0;
    }
    // store the lo of the constructed double - based on integer input
    SDOperand Store1 = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                   Op0Mapped, Lo, DAG.getSrcValue(NULL));
    // initial hi portion of constructed double
    SDOperand InitialHi = DAG.getConstant(0x43300000u, MVT::i32);
    // store the hi of the constructed double - biased exponent
    SDOperand Store2 = DAG.getNode(ISD::STORE, MVT::Other, Store1,
                                   InitialHi, Hi, DAG.getSrcValue(NULL));
    // load the constructed double
    SDOperand Load = DAG.getLoad(MVT::f64, Store2, StackSlot,
                               DAG.getSrcValue(NULL));
    // FP constant to bias correct the final result
    SDOperand Bias = DAG.getConstantFP(isSigned ?
                                            BitsToDouble(0x4330000080000000ULL)
                                          : BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    // subtract the bias
    SDOperand Sub = DAG.getNode(ISD::FSUB, MVT::f64, Load, Bias);
    // final result
    SDOperand Result;
    // handle final rounding
    if (DestVT == MVT::f64) {
      // do nothing
      Result = Sub;
    } else {
     // if f32 then cast to f32
      Result = DAG.getNode(ISD::FP_ROUND, MVT::f32, Sub);
    }
    NeedsAnotherIteration = true;
    return Result;
  }
  assert(!isSigned && "Legalize cannot Expand SINT_TO_FP for i64 yet");
  SDOperand Tmp1 = DAG.getNode(ISD::SINT_TO_FP, DestVT, Op0);

  SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultTy(), Op0,
                                   DAG.getConstant(0, Op0.getValueType()),
                                   ISD::SETLT);
  SDOperand Zero = getIntPtrConstant(0), Four = getIntPtrConstant(4);
  SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                    SignSet, Four, Zero);

  // If the sign bit of the integer is set, the large number will be treated
  // as a negative number.  To counteract this, the dynamic code adds an
  // offset depending on the data type.
  uint64_t FF;
  switch (Op0.getValueType()) {
  default: assert(0 && "Unsupported integer type!");
  case MVT::i8 : FF = 0x43800000ULL; break;  // 2^8  (as a float)
  case MVT::i16: FF = 0x47800000ULL; break;  // 2^16 (as a float)
  case MVT::i32: FF = 0x4F800000ULL; break;  // 2^32 (as a float)
  case MVT::i64: FF = 0x5F800000ULL; break;  // 2^64 (as a float)
  }
  if (TLI.isLittleEndian()) FF <<= 32;
  static Constant *FudgeFactor = ConstantUInt::get(Type::ULongTy, FF);

  SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
  SDOperand FudgeInReg;
  if (DestVT == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                             DAG.getSrcValue(NULL));
  else {
    assert(DestVT == MVT::f64 && "Unexpected conversion");
    FudgeInReg = LegalizeOp(DAG.getExtLoad(ISD::EXTLOAD, MVT::f64,
                                           DAG.getEntryNode(), CPIdx,
                                           DAG.getSrcValue(NULL), MVT::f32));
  }

  NeedsAnotherIteration = true;
  return DAG.getNode(ISD::FADD, DestVT, Tmp1, FudgeInReg);
}

/// PromoteLegalINT_TO_FP - This function is responsible for legalizing a
/// *INT_TO_FP operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal UINT_TO_FP or SINT_TO_FP
/// operation that takes a larger input.
SDOperand SelectionDAGLegalize::PromoteLegalINT_TO_FP(SDOperand LegalOp,
                                                      MVT::ValueType DestVT,
                                                      bool isSigned) {
  // First step, figure out the appropriate *INT_TO_FP operation to use.
  MVT::ValueType NewInTy = LegalOp.getValueType();

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewInTy = (MVT::ValueType)(NewInTy+1);
    assert(MVT::isInteger(NewInTy) && "Ran out of possibilities!");

    // If the target supports SINT_TO_FP of this type, use it.
    switch (TLI.getOperationAction(ISD::SINT_TO_FP, NewInTy)) {
      default: break;
      case TargetLowering::Legal:
        if (!TLI.isTypeLegal(NewInTy))
          break;  // Can't use this datatype.
        // FALL THROUGH.
      case TargetLowering::Custom:
        OpToUse = ISD::SINT_TO_FP;
        break;
    }
    if (OpToUse) break;
    if (isSigned) continue;

    // If the target supports UINT_TO_FP of this type, use it.
    switch (TLI.getOperationAction(ISD::UINT_TO_FP, NewInTy)) {
      default: break;
      case TargetLowering::Legal:
        if (!TLI.isTypeLegal(NewInTy))
          break;  // Can't use this datatype.
        // FALL THROUGH.
      case TargetLowering::Custom:
        OpToUse = ISD::UINT_TO_FP;
        break;
    }
    if (OpToUse) break;

    // Otherwise, try a larger type.
  }

  // Make sure to legalize any nodes we create here in the next pass.
  NeedsAnotherIteration = true;

  // Okay, we found the operation and type to use.  Zero extend our input to the
  // desired type then run the operation on it.
  return DAG.getNode(OpToUse, DestVT,
                     DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND,
                                 NewInTy, LegalOp));
}

/// PromoteLegalFP_TO_INT - This function is responsible for legalizing a
/// FP_TO_*INT operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal FP_TO_UINT or FP_TO_SINT
/// operation that returns a larger result.
SDOperand SelectionDAGLegalize::PromoteLegalFP_TO_INT(SDOperand LegalOp,
                                                      MVT::ValueType DestVT,
                                                      bool isSigned) {
  // First step, figure out the appropriate FP_TO*INT operation to use.
  MVT::ValueType NewOutTy = DestVT;

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewOutTy = (MVT::ValueType)(NewOutTy+1);
    assert(MVT::isInteger(NewOutTy) && "Ran out of possibilities!");

    // If the target supports FP_TO_SINT returning this type, use it.
    switch (TLI.getOperationAction(ISD::FP_TO_SINT, NewOutTy)) {
    default: break;
    case TargetLowering::Legal:
      if (!TLI.isTypeLegal(NewOutTy))
        break;  // Can't use this datatype.
      // FALL THROUGH.
    case TargetLowering::Custom:
      OpToUse = ISD::FP_TO_SINT;
      break;
    }
    if (OpToUse) break;

    // If the target supports FP_TO_UINT of this type, use it.
    switch (TLI.getOperationAction(ISD::FP_TO_UINT, NewOutTy)) {
    default: break;
    case TargetLowering::Legal:
      if (!TLI.isTypeLegal(NewOutTy))
        break;  // Can't use this datatype.
      // FALL THROUGH.
    case TargetLowering::Custom:
      OpToUse = ISD::FP_TO_UINT;
      break;
    }
    if (OpToUse) break;

    // Otherwise, try a larger type.
  }

  // Make sure to legalize any nodes we create here in the next pass.
  NeedsAnotherIteration = true;

  // Okay, we found the operation and type to use.  Truncate the result of the
  // extended FP_TO_*INT operation to the desired size.
  return DAG.getNode(ISD::TRUNCATE, DestVT,
                     DAG.getNode(OpToUse, NewOutTy, LegalOp));
}

/// ComputeTopDownOrdering - Add the specified node to the Order list if it has
/// not been visited yet and if all of its operands have already been visited.
static void ComputeTopDownOrdering(SDNode *N, std::vector<SDNode*> &Order,
                                   std::map<SDNode*, unsigned> &Visited) {
  if (++Visited[N] != N->getNumOperands())
    return;  // Haven't visited all operands yet
  
  Order.push_back(N);
  
  if (N->hasOneUse()) { // Tail recurse in common case.
    ComputeTopDownOrdering(*N->use_begin(), Order, Visited);
    return;
  }
  
  // Now that we have N in, add anything that uses it if all of their operands
  // are now done.
  for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); UI != E;++UI)
    ComputeTopDownOrdering(*UI, Order, Visited);
}


void SelectionDAGLegalize::LegalizeDAG() {
  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  std::map<SDNode*, unsigned> Visited;
  std::vector<SDNode*> Order;
  
  // Compute ordering from all of the leaves in the graphs, those (like the
  // entry node) that have no operands.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNumOperands() == 0) {
      Visited[I] = 0 - 1U;
      ComputeTopDownOrdering(I, Order, Visited);
    }
  }
  
  assert(Order.size() == Visited.size() &&
         Order.size() == 
            (unsigned)std::distance(DAG.allnodes_begin(), DAG.allnodes_end()) &&
         "Error: DAG is cyclic!");
  Visited.clear();
  
  for (unsigned i = 0, e = Order.size(); i != e; ++i) {
    SDNode *N = Order[i];
    switch (getTypeAction(N->getValueType(0))) {
    default: assert(0 && "Bad type action!");
    case Legal:
      LegalizeOp(SDOperand(N, 0));
      break;
    case Promote:
      PromoteOp(SDOperand(N, 0));
      break;
    case Expand: {
      SDOperand X, Y;
      ExpandOp(SDOperand(N, 0), X, Y);
      break;
    }
    }
  }

  // Finally, it's possible the root changed.  Get the new root.
  SDOperand OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  ExpandedNodes.clear();
  LegalizedNodes.clear();
  PromotedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes(OldRoot.Val);
}

SDOperand SelectionDAGLegalize::LegalizeOp(SDOperand Op) {
  assert(isTypeLegal(Op.getValueType()) &&
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

  SDOperand Tmp1, Tmp2, Tmp3, Tmp4;

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
  case ISD::TargetFrameIndex:
  case ISD::Register:
  case ISD::TargetConstant:
  case ISD::GlobalAddress:
  case ISD::TargetGlobalAddress:
  case ISD::ExternalSymbol:
  case ISD::ConstantPool:           // Nothing to do.
  case ISD::BasicBlock:
  case ISD::CONDCODE:
  case ISD::VALUETYPE:
  case ISD::SRCVALUE:
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom: {
      SDOperand Tmp = TLI.LowerOperation(Op, DAG);
      if (Tmp.Val) {
        Result = LegalizeOp(Tmp);
        break;
      }
    } // FALLTHROUGH if the target doesn't want to lower this op after all.
    case TargetLowering::Legal:
      assert(isTypeLegal(Node->getValueType(0)) && "This must be legal!");
      break;
    }
    break;
  case ISD::AssertSext:
  case ISD::AssertZext:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1,
                           Node->getOperand(1));
    break;
  case ISD::CopyFromReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getCopyFromReg(Tmp1, 
                            cast<RegisterSDNode>(Node->getOperand(1))->getReg(),
                                  Node->getValueType(0));
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
      Result = DAG.getNode(ISD::ImplicitDef, MVT::Other,
                           Tmp1, Node->getOperand(1));
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
          TLI.isOperationLegal(ISD::EXTLOAD, MVT::f32)) {
        LLVMC = cast<ConstantFP>(ConstantExpr::getCast(LLVMC, Type::FloatTy));
        VT = MVT::f32;
        Extend = true;
      }

      SDOperand CPIdx = DAG.getConstantPool(LLVMC, TLI.getPointerTy());
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
  case ISD::TokenFactor:
    if (Node->getNumOperands() == 2) {
      bool Changed = false;
      SDOperand Op0 = LegalizeOp(Node->getOperand(0));
      SDOperand Op1 = LegalizeOp(Node->getOperand(1));
      if (Op0 != Node->getOperand(0) || Op1 != Node->getOperand(1))
        Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Op0, Op1);
    } else {
      std::vector<SDOperand> Ops;
      bool Changed = false;
      // Legalize the operands.
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
        SDOperand Op = Node->getOperand(i);
        Ops.push_back(LegalizeOp(Op));
        Changed |= Ops[i] != Op;
      }
      if (Changed)
        Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Ops);
    }
    break;

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Do not try to legalize the target-specific arguments (#1+)
    Tmp2 = Node->getOperand(0);
    if (Tmp1 != Tmp2)
      Node->setAdjCallChain(Tmp1);
      
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
      
    switch (TLI.getOperationAction(ISD::BRCOND, MVT::Other)) {  
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand:
      // Expand brcond's setcc into its constituent parts and create a BR_CC
      // Node.
      if (Tmp2.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, Tmp2.getOperand(2),
                             Tmp2.getOperand(0), Tmp2.getOperand(1),
                             Node->getOperand(2));
      } else {
        // Make sure the condition is either zero or one.  It may have been
        // promoted from something else.
        Tmp2 = DAG.getZeroExtendInReg(Tmp2, MVT::i1);
        
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, 
                             DAG.getCondCode(ISD::SETNE), Tmp2,
                             DAG.getConstant(0, Tmp2.getValueType()),
                             Node->getOperand(2));
      }
      break;
    case TargetLowering::Legal:
      // Basic block destination (Op#2) is always legal.
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::BRCOND, MVT::Other, Tmp1, Tmp2,
                             Node->getOperand(2));
        break;
    }
    break;
  case ISD::BR_CC:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    
    if (isTypeLegal(Node->getOperand(2).getValueType())) {
      Tmp2 = LegalizeOp(Node->getOperand(2));   // LHS
      Tmp3 = LegalizeOp(Node->getOperand(3));   // RHS
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(2) ||
          Tmp3 != Node->getOperand(3)) {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, Node->getOperand(1),
                             Tmp2, Tmp3, Node->getOperand(4));
      }
      break;
    } else {
      Tmp2 = LegalizeOp(DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(),
                                    Node->getOperand(2),  // LHS
                                    Node->getOperand(3),  // RHS
                                    Node->getOperand(1)));
      // If we get a SETCC back from legalizing the SETCC node we just
      // created, then use its LHS, RHS, and CC directly in creating a new
      // node.  Otherwise, select between the true and false value based on
      // comparing the result of the legalized with zero.
      if (Tmp2.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, Tmp2.getOperand(2),
                             Tmp2.getOperand(0), Tmp2.getOperand(1),
                             Node->getOperand(4));
      } else {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, 
                             DAG.getCondCode(ISD::SETNE),
                             Tmp2, DAG.getConstant(0, Tmp2.getValueType()), 
                             Node->getOperand(4));
      }
    }
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
      // If BRTWOWAY_CC is legal for this target, then simply expand this node
      // to that.  Otherwise, skip BRTWOWAY_CC and expand directly to a
      // BRCOND/BR pair.
      if (TLI.isOperationLegal(ISD::BRTWOWAY_CC, MVT::Other)) {
        if (Tmp2.getOpcode() == ISD::SETCC) {
          Result = DAG.getBR2Way_CC(Tmp1, Tmp2.getOperand(2),
                                    Tmp2.getOperand(0), Tmp2.getOperand(1),
                                    Node->getOperand(2), Node->getOperand(3));
        } else {
          Result = DAG.getBR2Way_CC(Tmp1, DAG.getCondCode(ISD::SETNE), Tmp2, 
                                    DAG.getConstant(0, Tmp2.getValueType()),
                                    Node->getOperand(2), Node->getOperand(3));
        }
      } else {
        Result = DAG.getNode(ISD::BRCOND, MVT::Other, Tmp1, Tmp2,
                           Node->getOperand(2));
        Result = DAG.getNode(ISD::BR, MVT::Other, Result, Node->getOperand(3));
      }
      break;
    }
    break;
  case ISD::BRTWOWAY_CC:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    if (isTypeLegal(Node->getOperand(2).getValueType())) {
      Tmp2 = LegalizeOp(Node->getOperand(2));   // LHS
      Tmp3 = LegalizeOp(Node->getOperand(3));   // RHS
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(2) ||
          Tmp3 != Node->getOperand(3)) {
        Result = DAG.getBR2Way_CC(Tmp1, Node->getOperand(1), Tmp2, Tmp3,
                                  Node->getOperand(4), Node->getOperand(5));
      }
      break;
    } else {
      Tmp2 = LegalizeOp(DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(),
                                    Node->getOperand(2),  // LHS
                                    Node->getOperand(3),  // RHS
                                    Node->getOperand(1)));
      // If this target does not support BRTWOWAY_CC, lower it to a BRCOND/BR
      // pair.
      switch (TLI.getOperationAction(ISD::BRTWOWAY_CC, MVT::Other)) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Legal:
        // If we get a SETCC back from legalizing the SETCC node we just
        // created, then use its LHS, RHS, and CC directly in creating a new
        // node.  Otherwise, select between the true and false value based on
        // comparing the result of the legalized with zero.
        if (Tmp2.getOpcode() == ISD::SETCC) {
          Result = DAG.getBR2Way_CC(Tmp1, Tmp2.getOperand(2),
                                    Tmp2.getOperand(0), Tmp2.getOperand(1),
                                    Node->getOperand(4), Node->getOperand(5));
        } else {
          Result = DAG.getBR2Way_CC(Tmp1, DAG.getCondCode(ISD::SETNE), Tmp2, 
                                    DAG.getConstant(0, Tmp2.getValueType()),
                                    Node->getOperand(4), Node->getOperand(5));
        }
        break;
      case TargetLowering::Expand: 
        Result = DAG.getNode(ISD::BRCOND, MVT::Other, Tmp1, Tmp2,
                             Node->getOperand(4));
        Result = DAG.getNode(ISD::BR, MVT::Other, Result, Node->getOperand(5));
        break;
      }
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
    
  case ISD::VLOAD:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.
 
    // If we just have one element, scalarize the result.  Otherwise, check to
    // see if we support this operation on this type at this width.  If not,
    // split the vector in half and try again.
    if (1 == cast<ConstantSDNode>(Node->getOperand(2))->getValue()) {
      MVT::ValueType SVT = cast<VTSDNode>(Node->getOperand(3))->getVT();
      Result = LegalizeOp(DAG.getLoad(SVT, Tmp1, Tmp2, Node->getOperand(4)));
    } else {
      assert(0 && "Expand case for vectors unimplemented");
    }
    
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
  case ISD::EXTRACT_ELEMENT: {
    MVT::ValueType OpTy = Node->getOperand(0).getValueType();
    switch (getTypeAction(OpTy)) {
    default:
      assert(0 && "EXTRACT_ELEMENT action for type unimplemented!");
      break;
    case Legal:
      if (cast<ConstantSDNode>(Node->getOperand(1))->getValue()) {
        // 1 -> Hi
        Result = DAG.getNode(ISD::SRL, OpTy, Node->getOperand(0),
                             DAG.getConstant(MVT::getSizeInBits(OpTy)/2, 
                                             TLI.getShiftAmountTy()));
        Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), Result);
      } else {
        // 0 -> Lo
        Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), 
                             Node->getOperand(0));
      }
      Result = LegalizeOp(Result);
      break;
    case Expand:
      // Get both the low and high parts.
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      if (cast<ConstantSDNode>(Node->getOperand(1))->getValue())
        Result = Tmp2;  // 1 -> Hi
      else
        Result = Tmp1;  // 0 -> Lo
      break;
    }
    break;
  }

  case ISD::CopyToReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    assert(isTypeLegal(Node->getOperand(2).getValueType()) &&
           "Register type must be legal!");
    // Legalize the incoming value (must be legal).
    Tmp2 = LegalizeOp(Node->getOperand(2));
    if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(2))
      Result = DAG.getNode(ISD::CopyToReg, MVT::Other, Tmp1,
                           Node->getOperand(1), Tmp2);
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
        Result = DAG.getNode(ISD::STORE, MVT::Other, Tmp1,
                             DAG.getConstant(FloatToBits(CFP->getValue()),
                                             MVT::i32),
                             Tmp2,
                             Node->getOperand(3));
      } else {
        assert(CFP->getValueType(0) == MVT::f64 && "Unknown FP type!");
        Result = DAG.getNode(ISD::STORE, MVT::Other, Tmp1,
                             DAG.getConstant(DoubleToBits(CFP->getValue()),
                                             MVT::i64),
                             Tmp2,
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
  case ISD::READCYCLECOUNTER:
    Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the chain
    if (Tmp1 != Node->getOperand(0))
      Result = DAG.getNode(ISD::READCYCLECOUNTER, MVT::i64, Tmp1);
    break;
  case ISD::TRUNCSTORE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the pointer.

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1));
      
      // The only promote case we handle is TRUNCSTORE:i1 X into
      //   -> TRUNCSTORE:i8 (and X, 1)
      if (cast<VTSDNode>(Node->getOperand(4))->getVT() == MVT::i1 &&
          TLI.getOperationAction(ISD::TRUNCSTORE, MVT::i1) == 
                TargetLowering::Promote) {
        // Promote the bool to a mask then store.
        Tmp2 = DAG.getNode(ISD::AND, Tmp2.getValueType(), Tmp2,
                           DAG.getConstant(1, Tmp2.getValueType()));
        Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Tmp1, Tmp2, Tmp3,
                             Node->getOperand(3), DAG.getValueType(MVT::i8));

      } else if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
                 Tmp3 != Node->getOperand(2)) {
        Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Tmp1, Tmp2, Tmp3,
                             Node->getOperand(3), Node->getOperand(4));
      }
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

    switch (TLI.getOperationAction(ISD::SELECT, Tmp2.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand:
      if (Tmp1.getOpcode() == ISD::SETCC) {
        Result = DAG.getSelectCC(Tmp1.getOperand(0), Tmp1.getOperand(1), 
                              Tmp2, Tmp3,
                              cast<CondCodeSDNode>(Tmp1.getOperand(2))->get());
      } else {
        // Make sure the condition is either zero or one.  It may have been
        // promoted from something else.
        Tmp1 = DAG.getZeroExtendInReg(Tmp1, MVT::i1);
        Result = DAG.getSelectCC(Tmp1, 
                                 DAG.getConstant(0, Tmp1.getValueType()),
                                 Tmp2, Tmp3, ISD::SETNE);
      }
      break;
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
        ExtOp = ISD::ANY_EXTEND;
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
  case ISD::SELECT_CC:
    Tmp3 = LegalizeOp(Node->getOperand(2));   // True
    Tmp4 = LegalizeOp(Node->getOperand(3));   // False
    
    if (isTypeLegal(Node->getOperand(0).getValueType())) {
      // Everything is legal, see if we should expand this op or something.
      switch (TLI.getOperationAction(ISD::SELECT_CC,
                                     Node->getOperand(0).getValueType())) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Custom: {
        SDOperand Tmp =
          TLI.LowerOperation(DAG.getNode(ISD::SELECT_CC, Node->getValueType(0),
                                         Node->getOperand(0),
                                         Node->getOperand(1), Tmp3, Tmp4,
                                         Node->getOperand(4)), DAG);
        if (Tmp.Val) {
          Result = LegalizeOp(Tmp);
          break;
        }
      } // FALLTHROUGH if the target can't lower this operation after all.
      case TargetLowering::Legal:
        Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
        Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
        if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1) ||
            Tmp3 != Node->getOperand(2) || Tmp4 != Node->getOperand(3)) {
          Result = DAG.getNode(ISD::SELECT_CC, Node->getValueType(0), Tmp1, Tmp2, 
                               Tmp3, Tmp4, Node->getOperand(4));
        }
        break;
      }
      break;
    } else {
      Tmp1 = LegalizeOp(DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(),
                                    Node->getOperand(0),  // LHS
                                    Node->getOperand(1),  // RHS
                                    Node->getOperand(4)));
      // If we get a SETCC back from legalizing the SETCC node we just
      // created, then use its LHS, RHS, and CC directly in creating a new
      // node.  Otherwise, select between the true and false value based on
      // comparing the result of the legalized with zero.
      if (Tmp1.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::SELECT_CC, Tmp3.getValueType(),
                             Tmp1.getOperand(0), Tmp1.getOperand(1),
                             Tmp3, Tmp4, Tmp1.getOperand(2));
      } else {
        Result = DAG.getSelectCC(Tmp1,
                                 DAG.getConstant(0, Tmp1.getValueType()), 
                                 Tmp3, Tmp4, ISD::SETNE);
      }
    }
    break;
  case ISD::SETCC:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
      Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
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
        switch (cast<CondCodeSDNode>(Node->getOperand(2))->get()) {
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
      break;
    case Expand:
      SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
      ExpandOp(Node->getOperand(0), LHSLo, LHSHi);
      ExpandOp(Node->getOperand(1), RHSLo, RHSHi);
      switch (cast<CondCodeSDNode>(Node->getOperand(2))->get()) {
      case ISD::SETEQ:
      case ISD::SETNE:
        if (RHSLo == RHSHi)
          if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
            if (RHSCST->isAllOnesValue()) {
              // Comparison to -1.
              Tmp1 = DAG.getNode(ISD::AND, LHSLo.getValueType(), LHSLo, LHSHi);
              Tmp2 = RHSLo;
              break;
            }

        Tmp1 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSLo, RHSLo);
        Tmp2 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSHi, RHSHi);
        Tmp1 = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp2);
        Tmp2 = DAG.getConstant(0, Tmp1.getValueType());
        break;
      default:
        // If this is a comparison of the sign bit, just look at the top part.
        // X > -1,  x < 0
        if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(Node->getOperand(1)))
          if ((cast<CondCodeSDNode>(Node->getOperand(2))->get() == ISD::SETLT &&
               CST->getValue() == 0) ||              // X < 0
              (cast<CondCodeSDNode>(Node->getOperand(2))->get() == ISD::SETGT &&
               (CST->isAllOnesValue()))) {            // X > -1
            Tmp1 = LHSHi;
            Tmp2 = RHSHi;
            break;
          }

        // FIXME: This generated code sucks.
        ISD::CondCode LowCC;
        switch (cast<CondCodeSDNode>(Node->getOperand(2))->get()) {
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
        Tmp1 = DAG.getSetCC(Node->getValueType(0), LHSLo, RHSLo, LowCC);
        Tmp2 = DAG.getNode(ISD::SETCC, Node->getValueType(0), LHSHi, RHSHi,
                           Node->getOperand(2));
        Result = DAG.getSetCC(Node->getValueType(0), LHSHi, RHSHi, ISD::SETEQ);
        Result = LegalizeOp(DAG.getNode(ISD::SELECT, Tmp1.getValueType(),
                                        Result, Tmp1, Tmp2));
        return Result;
      }
    }

    switch(TLI.getOperationAction(ISD::SETCC, Node->getOperand(0).getValueType())) {
    default: 
      assert(0 && "Cannot handle this action for SETCC yet!");
      break;
    case TargetLowering::Promote:
      Result = DAG.getNode(ISD::SETCC, Node->getValueType(0), Tmp1, Tmp2,
                           Node->getOperand(2));
      break;
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::SETCC, Node->getValueType(0), Tmp1, Tmp2,
                             Node->getOperand(2));
      break;
    case TargetLowering::Expand:
      // Expand a setcc node into a select_cc of the same condition, lhs, and
      // rhs that selects between const 1 (true) and const 0 (false).
      MVT::ValueType VT = Node->getValueType(0);
      Result = DAG.getNode(ISD::SELECT_CC, VT, Tmp1, Tmp2, 
                           DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                           Node->getOperand(2));
      Result = LegalizeOp(Result);
      break;
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
    case TargetLowering::Custom: {
      SDOperand Tmp =
        TLI.LowerOperation(DAG.getNode(Node->getOpcode(), MVT::Other, Tmp1, 
                                       Tmp2, Tmp3, Tmp4, Tmp5), DAG);
      if (Tmp.Val) {
        Result = LegalizeOp(Tmp);
        break;
      }
      // FALLTHROUGH if the target thinks it is legal.
    }
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
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
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

    // Vector binary operators
  case ISD::VADD:
  case ISD::VSUB:
  case ISD::VMUL: {
    Tmp1 = Node->getOperand(0); // Element Count
    Tmp2 = Node->getOperand(1); // Element Type

    // If we just have one element, scalarize the result.  Otherwise, check to
    // see if we support this operation on this type at this width.  If not,
    // split the vector in half and try again.
    if (1 == cast<ConstantSDNode>(Tmp1)->getValue()) {
      MVT::ValueType SVT = cast<VTSDNode>(Tmp2)->getVT();
  
      Result = DAG.getNode(getScalarizedOpcode(Node->getOpcode(), SVT), SVT,
                           LegalizeOp(Node->getOperand(2)),
                           LegalizeOp(Node->getOperand(3)));
    } else {
      assert(0 && "Expand case for vectors unimplemented");
    }
    break;
  }
    
  case ISD::BUILD_PAIR: {
    MVT::ValueType PairTy = Node->getValueType(0);
    // TODO: handle the case where the Lo and Hi operands are not of legal type
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Lo
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Hi
    switch (TLI.getOperationAction(ISD::BUILD_PAIR, PairTy)) {
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::BUILD_PAIR, PairTy, Tmp1, Tmp2);
      break;
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom this yet!");
    case TargetLowering::Expand:
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, PairTy, Tmp1);
      Tmp2 = DAG.getNode(ISD::ANY_EXTEND, PairTy, Tmp2);
      Tmp2 = DAG.getNode(ISD::SHL, PairTy, Tmp2,
                         DAG.getConstant(MVT::getSizeInBits(PairTy)/2, 
                                         TLI.getShiftAmountTy()));
      Result = LegalizeOp(DAG.getNode(ISD::OR, PairTy, Tmp1, Tmp2));
      break;
    }
    break;
  }

  case ISD::UREM:
  case ISD::SREM:
  case ISD::FREM:
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
    case TargetLowering::Expand:
      if (MVT::isInteger(Node->getValueType(0))) {
        MVT::ValueType VT = Node->getValueType(0);
        unsigned Opc = (Node->getOpcode() == ISD::UREM) ? ISD::UDIV : ISD::SDIV;
        Result = DAG.getNode(Opc, VT, Tmp1, Tmp2);
        Result = DAG.getNode(ISD::MUL, VT, Result, Tmp2);
        Result = DAG.getNode(ISD::SUB, VT, Tmp1, Result);
      } else {
        // Floating point mod -> fmod libcall.
        const char *FnName = Node->getValueType(0) == MVT::f32 ? "fmodf":"fmod";
        SDOperand Dummy;
        Result = ExpandLibCall(FnName, Node, Dummy);
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
        Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), Tmp1,
                            DAG.getConstant(getSizeInBits(NVT), NVT),
                            ISD::SETEQ);
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
        if (!TLI.isOperationLegal(ISD::CTPOP, VT) &&
            TLI.isOperationLegal(ISD::CTLZ, VT)) {
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
        Result = LegalizeOp(DAG.getNode(ISD::FSUB, Node->getValueType(0),
                                        Tmp2, Tmp1));
        break;
      }
      case ISD::FABS: {
        // Expand Y = FABS(X) -> Y = (X >u 0.0) ? X : fneg(X).
        MVT::ValueType VT = Node->getValueType(0);
        Tmp2 = DAG.getConstantFP(0.0, VT);
        Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), Tmp1, Tmp2, ISD::SETUGT);
        Tmp3 = DAG.getNode(ISD::FNEG, VT, Tmp1);
        Result = DAG.getNode(ISD::SELECT, VT, Tmp2, Tmp1, Tmp3);
        Result = LegalizeOp(Result);
        break;
      }
      case ISD::FSQRT:
      case ISD::FSIN:
      case ISD::FCOS: {
        MVT::ValueType VT = Node->getValueType(0);
        const char *FnName = 0;
        switch(Node->getOpcode()) {
        case ISD::FSQRT: FnName = VT == MVT::f32 ? "sqrtf" : "sqrt"; break;
        case ISD::FSIN:  FnName = VT == MVT::f32 ? "sinf"  : "sin"; break;
        case ISD::FCOS:  FnName = VT == MVT::f32 ? "cosf"  : "cos"; break;
        default: assert(0 && "Unreachable!");
        }
        SDOperand Dummy;
        Result = ExpandLibCall(FnName, Node, Dummy);
        break;
      }
      default:
        assert(0 && "Unreachable!");
      }
      break;
    }
    break;

    // Conversion operators.  The source and destination have different types.
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    bool isSigned = Node->getOpcode() == ISD::SINT_TO_FP;
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      switch (TLI.getOperationAction(Node->getOpcode(),
                                     Node->getOperand(0).getValueType())) {
      default: assert(0 && "Unknown operation action!");
      case TargetLowering::Expand:
        Result = ExpandLegalINT_TO_FP(isSigned,
                                      LegalizeOp(Node->getOperand(0)),
                                      Node->getValueType(0));
        AddLegalizedOperand(Op, Result);
        return Result;
      case TargetLowering::Promote:
        Result = PromoteLegalINT_TO_FP(LegalizeOp(Node->getOperand(0)),
                                       Node->getValueType(0),
                                       isSigned);
        AddLegalizedOperand(Op, Result);
        return Result;
      case TargetLowering::Legal:
        break;
      }

      Tmp1 = LegalizeOp(Node->getOperand(0));
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case Expand:
      Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP,
                             Node->getValueType(0), Node->getOperand(0));
      break;
    case Promote:
      if (isSigned) {
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                 Result, DAG.getValueType(Node->getOperand(0).getValueType()));
        Result = DAG.getNode(ISD::SINT_TO_FP, Op.getValueType(), Result);
      } else {
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
        Result = DAG.getNode(ISD::UINT_TO_FP, Op.getValueType(), Result);
      }
      break;
    }
    break;
  }
  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);

      // Since the result is legal, we should just be able to truncate the low
      // part of the source.
      Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), Tmp1);
      break;
    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::TRUNCATE, Op.getValueType(), Result);
      break;
    }
    break;

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));

      switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))){
      default: assert(0 && "Unknown operation action!");
      case TargetLowering::Expand:
        if (Node->getOpcode() == ISD::FP_TO_UINT) {
          SDOperand True, False;
          MVT::ValueType VT =  Node->getOperand(0).getValueType();
          MVT::ValueType NVT = Node->getValueType(0);
          unsigned ShiftAmt = MVT::getSizeInBits(Node->getValueType(0))-1;
          Tmp2 = DAG.getConstantFP((double)(1ULL << ShiftAmt), VT);
          Tmp3 = DAG.getSetCC(TLI.getSetCCResultTy(),
                            Node->getOperand(0), Tmp2, ISD::SETLT);
          True = DAG.getNode(ISD::FP_TO_SINT, NVT, Node->getOperand(0));
          False = DAG.getNode(ISD::FP_TO_SINT, NVT,
                              DAG.getNode(ISD::FSUB, VT, Node->getOperand(0),
                                          Tmp2));
          False = DAG.getNode(ISD::XOR, NVT, False, 
                              DAG.getConstant(1ULL << ShiftAmt, NVT));
          Result = LegalizeOp(DAG.getNode(ISD::SELECT, NVT, Tmp3, True, False));
          return Result;
        } else {
          assert(0 && "Do not know how to expand FP_TO_SINT yet!");
        }
        break;
      case TargetLowering::Promote:
        Result = PromoteLegalFP_TO_INT(Tmp1, Node->getValueType(0),
                                       Node->getOpcode() == ISD::FP_TO_SINT);
        AddLegalizedOperand(Op, Result);
        return Result;
      case TargetLowering::Custom: {
        SDOperand Tmp =
          DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
        Tmp = TLI.LowerOperation(Tmp, DAG);
        if (Tmp.Val) {
          AddLegalizedOperand(Op, Tmp);
          NeedsAnotherIteration = true;
          return Tmp;
        } else {
          // The target thinks this is legal afterall.
          break;
        }
      }
      case TargetLowering::Legal:
        break;
      }

      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case Expand:
      assert(0 && "Shouldn't need to expand other operators here!");
    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(Node->getOpcode(), Op.getValueType(), Result);
      break;
    }
    break;

  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::FP_EXTEND:
  case ISD::FP_ROUND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      if (Tmp1 != Node->getOperand(0))
        Result = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      break;
    case Expand:
      assert(0 && "Shouldn't need to expand other operators here!");

    case Promote:
      switch (Node->getOpcode()) {
      case ISD::ANY_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Result);
        break;
      case ISD::ZERO_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Result);
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
        break;
      case ISD::SIGN_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Result);
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                          DAG.getValueType(Node->getOperand(0).getValueType()));
        break;
      case ISD::FP_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        if (Result.getValueType() != Op.getValueType())
          // Dynamically dead while we have only 2 FP types.
          Result = DAG.getNode(ISD::FP_EXTEND, Op.getValueType(), Result);
        break;
      case ISD::FP_ROUND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(Node->getOpcode(), Op.getValueType(), Result);
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

  std::map<SDOperand, SDOperand>::iterator I = PromotedNodes.find(Op);
  if (I != PromotedNodes.end()) return I->second;

  // Promotion needs an optimization step to clean up after it, and is not
  // careful to avoid operations the target does not support.  Make sure that
  // all generated operations are legalized in the next iteration.
  NeedsAnotherIteration = true;

  switch (Node->getOpcode()) {
  case ISD::CopyFromReg:
    assert(0 && "CopyFromReg must be legal!");
  default:
    std::cerr << "NODE: "; Node->dump(); std::cerr << "\n";
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::UNDEF:
    Result = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::Constant:
    if (VT != MVT::i1)
      Result = DAG.getNode(ISD::SIGN_EXTEND, NVT, Op);
    else
      Result = DAG.getNode(ISD::ZERO_EXTEND, NVT, Op);
    assert(isa<ConstantSDNode>(Result) && "Didn't constant fold zext?");
    break;
  case ISD::ConstantFP:
    Result = DAG.getNode(ISD::FP_EXTEND, NVT, Op);
    assert(isa<ConstantFPSDNode>(Result) && "Didn't constant fold fp_extend?");
    break;

  case ISD::SETCC:
    assert(isTypeLegal(TLI.getSetCCResultTy()) && "SetCC type is not legal??");
    Result = DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(),Node->getOperand(0),
                         Node->getOperand(1), Node->getOperand(2));
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
      Result = DAG.getNode(ISD::TRUNCATE, NVT, Tmp1);
    }
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
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
      else if (Node->getOpcode() == ISD::ZERO_EXTEND)
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
    // If we're promoting a UINT to a larger size, check to see if the new node
    // will be legal.  If it isn't, check to see if FP_TO_SINT is legal, since
    // we can use that instead.  This allows us to generate better code for
    // FP_TO_UINT for small destination sizes on targets where FP_TO_UINT is not
    // legal, such as PowerPC.
    if (Node->getOpcode() == ISD::FP_TO_UINT && 
        !TLI.isOperationLegal(ISD::FP_TO_UINT, NVT) &&
        (TLI.isOperationLegal(ISD::FP_TO_SINT, NVT) ||
         TLI.getOperationAction(ISD::FP_TO_SINT, NVT)==TargetLowering::Custom)){
      Result = DAG.getNode(ISD::FP_TO_SINT, NVT, Tmp1);
    } else {
      Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    }
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
    // these operations don't care.  They may have weird bits going out, but
    // that too is okay if they are integer operations.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    // The input may have strange things in the top bits of the registers, but
    // these operations don't care.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    
    // Floating point operations will give excess precision that we may not be
    // able to tolerate.  If we DO allow excess precision, just leave it,
    // otherwise excise it.
    // FIXME: Why would we need to round FP ops more than integer ones?
    //     Is Round(Add(Add(A,B),C)) != Round(Add(Round(Add(A,B)), C))
    if (NoExcessFPPrecision)
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
  case ISD::FDIV:
  case ISD::FREM:
    // These operators require that their input be fp extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    
    // Perform FP_ROUND: this is probably overly pessimistic.
    if (NoExcessFPPrecision)
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
    Result = DAG.getExtLoad(ISD::EXTLOAD, NVT, Tmp1, Tmp2,
                            Node->getOperand(2), VT);
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    break;
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::EXTLOAD:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Legalize the pointer.
    Result = DAG.getExtLoad(Node->getOpcode(), NVT, Tmp1, Tmp2,
                         Node->getOperand(2),
                            cast<VTSDNode>(Node->getOperand(3))->getVT());
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
  case ISD::SELECT_CC:
    Tmp2 = PromoteOp(Node->getOperand(2));   // True
    Tmp3 = PromoteOp(Node->getOperand(3));   // False
    Result = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                         Node->getOperand(1), Tmp2, Tmp3,
                         Node->getOperand(4));
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
      Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), Tmp1,
                          DAG.getConstant(getSizeInBits(NVT), NVT), ISD::SETEQ);
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
  std::vector<MVT::ValueType> VTs(2, LHSL.getValueType());
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
  if (!TLI.isOperationLegal(ISD::SELECT, NVT) && !isa<ConstantSDNode>(Amt))
    return false;

  SDOperand InL, InH;
  ExpandOp(Op, InL, InH);
  SDOperand NAmt = DAG.getNode(ISD::SUB, ShTy,           // NAmt = 32-ShAmt
                               DAG.getConstant(NVTBits, ShTy), ShAmt);

  // Compare the unmasked shift amount against 32.
  SDOperand Cond = DAG.getSetCC(TLI.getSetCCResultTy(), ShAmt,
                                DAG.getConstant(NVTBits, ShTy), ISD::SETGE);

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
                                     DAG.getSetCC(TLI.getSetCCResultTy(), NAmt,
                                                  DAG.getConstant(32, ShTy),
                                                  ISD::SETEQ),
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
static void FindEarliestCallSeqEnd(SDNode *Node, SDNode *&Found,
                                   std::set<SDNode*> &Visited) {
  if ((Found && Node->getNodeDepth() >= Found->getNodeDepth()) ||
      !Visited.insert(Node).second) return;

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
    FindEarliestCallSeqEnd(*UI, Found, Visited);

  // Tail recurse for the last iteration.
  FindEarliestCallSeqEnd(*UI, Found, Visited);
}

/// FindCallSeqEnd - Given a chained node that is part of a call sequence,
/// find the CALLSEQ_END node that terminates the call sequence.
static SDNode *FindCallSeqEnd(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_END)
    return Node;
  if (Node->use_empty())
    return 0;   // No CallSeqEnd

  SDOperand TheChain(Node, Node->getNumValues()-1);
  if (TheChain.getValueType() != MVT::Other)
    TheChain = SDOperand(Node, 0);
  if (TheChain.getValueType() != MVT::Other)
    return 0;

  for (SDNode::use_iterator UI = Node->use_begin(),
         E = Node->use_end(); UI != E; ++UI) {

    // Make sure to only follow users of our token chain.
    SDNode *User = *UI;
    for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
      if (User->getOperand(i) == TheChain)
        if (SDNode *Result = FindCallSeqEnd(User))
          return Result;
  }
  return 0;
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
  if (LatestCallSeqStart->getOpcode() == ISD::CALLSEQ_START) {
    LatestCallSeqEnd = FindCallSeqEnd(LatestCallSeqStart);
    //std::cerr<<"Found end node: "; LatestCallSeqEnd->dump(); std::cerr <<"\n";
  } else {
    LatestCallSeqEnd = Entry.Val;
  }
  assert(LatestCallSeqEnd && "NULL return from FindCallSeqEnd");

  // Finally, find the first call that this must come before, first we find the
  // CallSeqEnd that ends the call.
  OutChain = 0;
  std::set<SDNode*> Visited;
  FindEarliestCallSeqEnd(OpNode, OutChain, Visited);

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

  SDOperand Result;
  switch (getTypeAction(CallInfo.first.getValueType())) {
  default: assert(0 && "Unknown thing");
  case Legal:
    Result = CallInfo.first;
    break;
  case Promote:
    assert(0 && "Cannot promote this yet!");
  case Expand:
    ExpandOp(CallInfo.first, Result, Hi);
    CallInfo.second = LegalizeOp(CallInfo.second);
    break;
  }
  
  SpliceCallInto(CallInfo.second, OutChain);
  NeedsAnotherIteration = true;
  return Result;
}


/// ExpandIntToFP - Expand a [US]INT_TO_FP operation, assuming that the
/// destination type is legal.
SDOperand SelectionDAGLegalize::
ExpandIntToFP(bool isSigned, MVT::ValueType DestTy, SDOperand Source) {
  assert(isTypeLegal(DestTy) && "Destination type is not legal!");
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

    SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultTy(), Hi,
                                     DAG.getConstant(0, Hi.getValueType()),
                                     ISD::SETLT);
    SDOperand Zero = getIntPtrConstant(0), Four = getIntPtrConstant(4);
    SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                      SignSet, Four, Zero);
    uint64_t FF = 0x5f800000ULL;
    if (TLI.isLittleEndian()) FF <<= 32;
    static Constant *FudgeFactor = ConstantUInt::get(Type::ULongTy, FF);

    SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
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
    return DAG.getNode(ISD::FADD, DestTy, SignedConv, FudgeInReg);
  }

  // Check to see if the target has a custom way to lower this.  If so, use it.
  switch (TLI.getOperationAction(ISD::SINT_TO_FP, Source.getValueType())) {
  default: assert(0 && "This action not implemented for this operation!");
  case TargetLowering::Legal:
  case TargetLowering::Expand:
    break;   // This case is handled below.
  case TargetLowering::Custom: {
    SDOperand NV = TLI.LowerOperation(DAG.getNode(ISD::SINT_TO_FP, DestTy,
                                                  Source), DAG);
    if (NV.Val)
      return LegalizeOp(NV);
    break;   // The target decided this was legal after all
  }
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

  // See if we already expanded it.
  std::map<SDOperand, std::pair<SDOperand, SDOperand> >::iterator I
    = ExpandedNodes.find(Op);
  if (I != ExpandedNodes.end()) {
    Lo = I->second.first;
    Hi = I->second.second;
    return;
  }

  // Expanding to multiple registers needs to perform an optimization step, and
  // is not careful to avoid operations the target does not support.  Make sure
  // that all generated operations are legalized in the next iteration.
  NeedsAnotherIteration = true;

  switch (Node->getOpcode()) {
   case ISD::CopyFromReg:
      assert(0 && "CopyFromReg must be legal!");
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
    SDOperand TopNotZero = DAG.getSetCC(TLI.getSetCCResultTy(), HLZ, BitsC,
                                        ISD::SETNE);
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
    SDOperand BotNotZero = DAG.getSetCC(TLI.getSetCCResultTy(), LTZ, BitsC,
                                        ISD::SETNE);
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
  case ISD::SELECT_CC: {
    SDOperand TL, TH, FL, FH;
    ExpandOp(Node->getOperand(2), TL, TH);
    ExpandOp(Node->getOperand(3), FL, FH);
    Lo = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                     Node->getOperand(1), TL, FL, Node->getOperand(4));
    Hi = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                     Node->getOperand(1), TH, FH, Node->getOperand(4));
    Lo = LegalizeOp(Lo);
    Hi = LegalizeOp(Hi);
    break;
  }
  case ISD::SEXTLOAD: {
    SDOperand Chain = LegalizeOp(Node->getOperand(0));
    SDOperand Ptr   = LegalizeOp(Node->getOperand(1));
    MVT::ValueType EVT = cast<VTSDNode>(Node->getOperand(3))->getVT();
    
    if (EVT == NVT)
      Lo = DAG.getLoad(NVT, Chain, Ptr, Node->getOperand(2));
    else
      Lo = DAG.getExtLoad(ISD::SEXTLOAD, NVT, Chain, Ptr, Node->getOperand(2),
                          EVT);
    
    // Remember that we legalized the chain.
    AddLegalizedOperand(SDOperand(Node, 1), Lo.getValue(1));
    
    // The high part is obtained by SRA'ing all but one of the bits of the lo
    // part.
    unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
    Hi = DAG.getNode(ISD::SRA, NVT, Lo, DAG.getConstant(LoSize-1,
                                                       TLI.getShiftAmountTy()));
    Lo = LegalizeOp(Lo);
    Hi = LegalizeOp(Hi);
    break;
  }
  case ISD::ZEXTLOAD: {
    SDOperand Chain = LegalizeOp(Node->getOperand(0));
    SDOperand Ptr   = LegalizeOp(Node->getOperand(1));
    MVT::ValueType EVT = cast<VTSDNode>(Node->getOperand(3))->getVT();
    
    if (EVT == NVT)
      Lo = DAG.getLoad(NVT, Chain, Ptr, Node->getOperand(2));
    else
      Lo = DAG.getExtLoad(ISD::ZEXTLOAD, NVT, Chain, Ptr, Node->getOperand(2),
                          EVT);
    
    // Remember that we legalized the chain.
    AddLegalizedOperand(SDOperand(Node, 1), Lo.getValue(1));

    // The high part is just a zero.
    Hi = LegalizeOp(DAG.getConstant(0, NVT));
    Lo = LegalizeOp(Lo);
    break;
  }
  case ISD::EXTLOAD: {
    SDOperand Chain = LegalizeOp(Node->getOperand(0));
    SDOperand Ptr   = LegalizeOp(Node->getOperand(1));
    MVT::ValueType EVT = cast<VTSDNode>(Node->getOperand(3))->getVT();
    
    if (EVT == NVT)
      Lo = DAG.getLoad(NVT, Chain, Ptr, Node->getOperand(2));
    else
      Lo = DAG.getExtLoad(ISD::EXTLOAD, NVT, Chain, Ptr, Node->getOperand(2),
                          EVT);
    
    // Remember that we legalized the chain.
    AddLegalizedOperand(SDOperand(Node, 1), Lo.getValue(1));
    
    // The high part is undefined.
    Hi = LegalizeOp(DAG.getNode(ISD::UNDEF, NVT));
    Lo = LegalizeOp(Lo);
    break;
  }
  case ISD::ANY_EXTEND: {
    SDOperand In;
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "expand-expand not implemented yet!");
    case Legal: In = LegalizeOp(Node->getOperand(0)); break;
    case Promote:
      In = PromoteOp(Node->getOperand(0));
      break;
    }
    
    // The low part is any extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ANY_EXTEND, NVT, In);
    // The high part is undefined.
    Hi = DAG.getNode(ISD::UNDEF, NVT);
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
    if (TLI.getOperationAction(ISD::FP_TO_SINT, VT) == TargetLowering::Custom) {
      SDOperand Op;
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "cannot expand FP!");
      case Legal: Op = LegalizeOp(Node->getOperand(0)); break;
      case Promote: Op = PromoteOp(Node->getOperand(0)); break;
      }

      Op = TLI.LowerOperation(DAG.getNode(ISD::FP_TO_SINT, VT, Op), DAG);

      // Now that the custom expander is done, expand the result, which is still
      // VT.
      if (Op.Val) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    if (Node->getOperand(0).getValueType() == MVT::f32)
      Lo = ExpandLibCall("__fixsfdi", Node, Hi);
    else
      Lo = ExpandLibCall("__fixdfdi", Node, Hi);
    break;

  case ISD::FP_TO_UINT:
    if (TLI.getOperationAction(ISD::FP_TO_UINT, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::FP_TO_UINT, VT,
                                 LegalizeOp(Node->getOperand(0)));
      // Now that the custom expander is done, expand the result, which is still
      // VT.
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    if (Node->getOperand(0).getValueType() == MVT::f32)
      Lo = ExpandLibCall("__fixunssfdi", Node, Hi);
    else
      Lo = ExpandLibCall("__fixunsdfdi", Node, Hi);
    break;

  case ISD::SHL:
    // If the target wants custom lowering, do so.
    if (TLI.getOperationAction(ISD::SHL, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SHL, VT, Node->getOperand(0),
                                 LegalizeOp(Node->getOperand(1)));
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SHL, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SHL_PARTS, use it.
    if (TLI.isOperationLegal(ISD::SHL_PARTS, NVT)) {
      ExpandShiftParts(ISD::SHL_PARTS, Node->getOperand(0), Node->getOperand(1),
                       Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall("__ashldi3", Node, Hi);
    break;

  case ISD::SRA:
    // If the target wants custom lowering, do so.
    if (TLI.getOperationAction(ISD::SRA, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SRA, VT, Node->getOperand(0),
                                 LegalizeOp(Node->getOperand(1)));
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRA, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SRA_PARTS, use it.
    if (TLI.isOperationLegal(ISD::SRA_PARTS, NVT)) {
      ExpandShiftParts(ISD::SRA_PARTS, Node->getOperand(0), Node->getOperand(1),
                       Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall("__ashrdi3", Node, Hi);
    break;
  case ISD::SRL:
    // If the target wants custom lowering, do so.
    if (TLI.getOperationAction(ISD::SRL, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SRL, VT, Node->getOperand(0),
                                 LegalizeOp(Node->getOperand(1)));
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRL, Node->getOperand(0), Node->getOperand(1), Lo, Hi))
      break;

    // If this target supports SRL_PARTS, use it.
    if (TLI.isOperationLegal(ISD::SRL_PARTS, NVT)) {
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
    if (TLI.isOperationLegal(ISD::MULHU, NVT)) {
      SDOperand LL, LH, RL, RH;
      ExpandOp(Node->getOperand(0), LL, LH);
      ExpandOp(Node->getOperand(1), RL, RH);
      unsigned SH = MVT::getSizeInBits(RH.getValueType())-1;
      // MULHS implicitly sign extends its inputs.  Check to see if ExpandOp
      // extended the sign bit of the low half through the upper half, and if so
      // emit a MULHS instead of the alternate sequence that is valid for any
      // i64 x i64 multiply.
      if (TLI.isOperationLegal(ISD::MULHS, NVT) &&
          // is RH an extension of the sign bit of RL?
          RH.getOpcode() == ISD::SRA && RH.getOperand(0) == RL &&
          RH.getOperand(1).getOpcode() == ISD::Constant &&
          cast<ConstantSDNode>(RH.getOperand(1))->getValue() == SH &&
          // is LH an extension of the sign bit of LL?
          LH.getOpcode() == ISD::SRA && LH.getOperand(0) == LL &&
          LH.getOperand(1).getOpcode() == ISD::Constant &&
          cast<ConstantSDNode>(LH.getOperand(1))->getValue() == SH) {
        Hi = DAG.getNode(ISD::MULHS, NVT, LL, RL);
      } else {
        Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
        RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
        LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
      }
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
  bool isNew = ExpandedNodes.insert(std::make_pair(Op,
                                          std::make_pair(Lo, Hi))).second;
  assert(isNew && "Value already expanded?!?");
}


// SelectionDAG::Legalize - This is the entry point for the file.
//
void SelectionDAG::Legalize() {
  /// run - This is the main entry point to this class.
  ///
  SelectionDAGLegalize(*this).Run();
}

