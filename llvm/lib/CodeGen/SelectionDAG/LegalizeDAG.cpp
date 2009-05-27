//===-- LegalizeDAG.cpp - Implement SelectionDAG::Legalize ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::Legalize method.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
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
class VISIBILITY_HIDDEN SelectionDAGLegalize {
  TargetLowering &TLI;
  SelectionDAG &DAG;
  CodeGenOpt::Level OptLevel;

  // Libcall insertion helpers.

  /// LastCALLSEQ_END - This keeps track of the CALLSEQ_END node that has been
  /// legalized.  We use this to ensure that calls are properly serialized
  /// against each other, including inserted libcalls.
  SDValue LastCALLSEQ_END;

  /// IsLegalizingCall - This member is used *only* for purposes of providing
  /// helpful assertions that a libcall isn't created while another call is
  /// being legalized (which could lead to non-serialized call sequences).
  bool IsLegalizingCall;

  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand      // Try to expand this to other ops, otherwise use a libcall.
  };

  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// value type, where the two bits correspond to the LegalizeAction enum.
  /// This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;

  /// LegalizedNodes - For nodes that are of legal width, and that have more
  /// than one use, this map indicates what regularized operand to use.  This
  /// allows us to avoid legalizing the same thing more than once.
  DenseMap<SDValue, SDValue> LegalizedNodes;

  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }

public:
  SelectionDAGLegalize(SelectionDAG &DAG, CodeGenOpt::Level ol);

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT VT) const {
    return (LegalizeAction)ValueTypeActions.getTypeAction(VT);
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT VT) const {
    return getTypeAction(VT) == Legal;
  }

  void LegalizeDAG();

private:
  /// HandleOp - Legalize, Promote, or Expand the specified operand as
  /// appropriate for its type.
  void HandleOp(SDValue Op);

  /// LegalizeOp - We know that the specified value has a legal type.
  /// Recursively ensure that the operands have legal types, then return the
  /// result.
  SDValue LegalizeOp(SDValue O);

  /// PerformInsertVectorEltInMemory - Some target cannot handle a variable
  /// insertion index for the INSERT_VECTOR_ELT instruction.  In this case, it
  /// is necessary to spill the vector being inserted into to memory, perform
  /// the insert there, and then read the result back.
  SDValue PerformInsertVectorEltInMemory(SDValue Vec, SDValue Val,
                                         SDValue Idx, DebugLoc dl);
  SDValue ExpandINSERT_VECTOR_ELT(SDValue Vec, SDValue Val,
                                  SDValue Idx, DebugLoc dl);

  /// Useful 16 element vector type that is used to pass operands for widening.
  typedef SmallVector<SDValue, 16> SDValueVector;

  /// ShuffleWithNarrowerEltType - Return a vector shuffle operation which
  /// performs the same shuffe in terms of order or result bytes, but on a type
  /// whose vector element type is narrower than the original shuffle type.
  /// e.g. <v4i32> <0, 1, 0, 1> -> v8i16 <0, 1, 2, 3, 0, 1, 2, 3>
  SDValue ShuffleWithNarrowerEltType(MVT NVT, MVT VT, DebugLoc dl,
                                     SDValue N1, SDValue N2, 
                                     SmallVectorImpl<int> &Mask) const;

  bool LegalizeAllNodesNotLeadingTo(SDNode *N, SDNode *Dest,
                                    SmallPtrSet<SDNode*, 32> &NodesLeadingTo);

  void LegalizeSetCCOperands(SDValue &LHS, SDValue &RHS, SDValue &CC,
                             DebugLoc dl);
  void LegalizeSetCCCondCode(MVT VT, SDValue &LHS, SDValue &RHS, SDValue &CC,
                             DebugLoc dl);
  void LegalizeSetCC(MVT VT, SDValue &LHS, SDValue &RHS, SDValue &CC,
                     DebugLoc dl) {
    LegalizeSetCCOperands(LHS, RHS, CC, dl);
    LegalizeSetCCCondCode(VT, LHS, RHS, CC, dl);
  }

  SDValue ExpandLibCall(RTLIB::Libcall LC, SDNode *Node, bool isSigned);
  SDValue ExpandFPLibCall(SDNode *Node, RTLIB::Libcall Call_F32,
                          RTLIB::Libcall Call_F64, RTLIB::Libcall Call_F80,
                          RTLIB::Libcall Call_PPCF128);
  SDValue ExpandIntLibCall(SDNode *Node, bool isSigned, RTLIB::Libcall Call_I16,
                           RTLIB::Libcall Call_I32, RTLIB::Libcall Call_I64,
                           RTLIB::Libcall Call_I128);

  SDValue EmitStackConvert(SDValue SrcOp, MVT SlotVT, MVT DestVT, DebugLoc dl);
  SDValue ExpandBUILD_VECTOR(SDNode *Node);
  SDValue ExpandSCALAR_TO_VECTOR(SDNode *Node);
  SDValue ExpandLegalINT_TO_FP(bool isSigned, SDValue LegalOp, MVT DestVT,
                               DebugLoc dl);
  SDValue PromoteLegalINT_TO_FP(SDValue LegalOp, MVT DestVT, bool isSigned,
                                DebugLoc dl);
  SDValue PromoteLegalFP_TO_INT(SDValue LegalOp, MVT DestVT, bool isSigned,
                                DebugLoc dl);

  SDValue ExpandBSWAP(SDValue Op, DebugLoc dl);
  SDValue ExpandBitCount(unsigned Opc, SDValue Op, DebugLoc dl);

  SDValue ExpandExtractFromVectorThroughStack(SDValue Op);

  void ExpandNode(SDNode *Node, SmallVectorImpl<SDValue> &Results);
  void PromoteNode(SDNode *Node, SmallVectorImpl<SDValue> &Results);
};
}

/// ShuffleWithNarrowerEltType - Return a vector shuffle operation which
/// performs the same shuffe in terms of order or result bytes, but on a type
/// whose vector element type is narrower than the original shuffle type.
/// e.g. <v4i32> <0, 1, 0, 1> -> v8i16 <0, 1, 2, 3, 0, 1, 2, 3>
SDValue 
SelectionDAGLegalize::ShuffleWithNarrowerEltType(MVT NVT, MVT VT,  DebugLoc dl, 
                                                 SDValue N1, SDValue N2,
                                             SmallVectorImpl<int> &Mask) const {
  MVT EltVT = NVT.getVectorElementType();
  unsigned NumMaskElts = VT.getVectorNumElements();
  unsigned NumDestElts = NVT.getVectorNumElements();
  unsigned NumEltsGrowth = NumDestElts / NumMaskElts;

  assert(NumEltsGrowth && "Cannot promote to vector type with fewer elts!");

  if (NumEltsGrowth == 1)
    return DAG.getVectorShuffle(NVT, dl, N1, N2, &Mask[0]);
  
  SmallVector<int, 8> NewMask;
  for (unsigned i = 0; i != NumMaskElts; ++i) {
    int Idx = Mask[i];
    for (unsigned j = 0; j != NumEltsGrowth; ++j) {
      if (Idx < 0) 
        NewMask.push_back(-1);
      else
        NewMask.push_back(Idx * NumEltsGrowth + j);
    }
  }
  assert(NewMask.size() == NumDestElts && "Non-integer NumEltsGrowth?");
  assert(TLI.isShuffleMaskLegal(NewMask, NVT) && "Shuffle not legal?");
  return DAG.getVectorShuffle(NVT, dl, N1, N2, &NewMask[0]);
}

SelectionDAGLegalize::SelectionDAGLegalize(SelectionDAG &dag,
                                           CodeGenOpt::Level ol)
  : TLI(dag.getTargetLoweringInfo()), DAG(dag), OptLevel(ol),
    ValueTypeActions(TLI.getValueTypeActions()) {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");
}

void SelectionDAGLegalize::LegalizeDAG() {
  LastCALLSEQ_END = DAG.getEntryNode();
  IsLegalizingCall = false;

  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = prior(DAG.allnodes_end()); I != next(E); ++I)
    HandleOp(SDValue(I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  LegalizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();
}


/// FindCallEndFromCallStart - Given a chained node that is part of a call
/// sequence, find the CALLSEQ_END node that terminates the call sequence.
static SDNode *FindCallEndFromCallStart(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_END)
    return Node;
  if (Node->use_empty())
    return 0;   // No CallSeqEnd

  // The chain is usually at the end.
  SDValue TheChain(Node, Node->getNumValues()-1);
  if (TheChain.getValueType() != MVT::Other) {
    // Sometimes it's at the beginning.
    TheChain = SDValue(Node, 0);
    if (TheChain.getValueType() != MVT::Other) {
      // Otherwise, hunt for it.
      for (unsigned i = 1, e = Node->getNumValues(); i != e; ++i)
        if (Node->getValueType(i) == MVT::Other) {
          TheChain = SDValue(Node, i);
          break;
        }

      // Otherwise, we walked into a node without a chain.
      if (TheChain.getValueType() != MVT::Other)
        return 0;
    }
  }

  for (SDNode::use_iterator UI = Node->use_begin(),
       E = Node->use_end(); UI != E; ++UI) {

    // Make sure to only follow users of our token chain.
    SDNode *User = *UI;
    for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
      if (User->getOperand(i) == TheChain)
        if (SDNode *Result = FindCallEndFromCallStart(User))
          return Result;
  }
  return 0;
}

/// FindCallStartFromCallEnd - Given a chained node that is part of a call
/// sequence, find the CALLSEQ_START node that initiates the call sequence.
static SDNode *FindCallStartFromCallEnd(SDNode *Node) {
  assert(Node && "Didn't find callseq_start for a call??");
  if (Node->getOpcode() == ISD::CALLSEQ_START) return Node;

  assert(Node->getOperand(0).getValueType() == MVT::Other &&
         "Node doesn't have a token chain argument!");
  return FindCallStartFromCallEnd(Node->getOperand(0).getNode());
}

/// LegalizeAllNodesNotLeadingTo - Recursively walk the uses of N, looking to
/// see if any uses can reach Dest.  If no dest operands can get to dest,
/// legalize them, legalize ourself, and return false, otherwise, return true.
///
/// Keep track of the nodes we fine that actually do lead to Dest in
/// NodesLeadingTo.  This avoids retraversing them exponential number of times.
///
bool SelectionDAGLegalize::LegalizeAllNodesNotLeadingTo(SDNode *N, SDNode *Dest,
                                     SmallPtrSet<SDNode*, 32> &NodesLeadingTo) {
  if (N == Dest) return true;  // N certainly leads to Dest :)

  // If we've already processed this node and it does lead to Dest, there is no
  // need to reprocess it.
  if (NodesLeadingTo.count(N)) return true;

  // If the first result of this node has been already legalized, then it cannot
  // reach N.
  if (LegalizedNodes.count(SDValue(N, 0))) return false;

  // Okay, this node has not already been legalized.  Check and legalize all
  // operands.  If none lead to Dest, then we can legalize this node.
  bool OperandsLeadToDest = false;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    OperandsLeadToDest |=     // If an operand leads to Dest, so do we.
      LegalizeAllNodesNotLeadingTo(N->getOperand(i).getNode(), Dest, NodesLeadingTo);

  if (OperandsLeadToDest) {
    NodesLeadingTo.insert(N);
    return true;
  }

  // Okay, this node looks safe, legalize it and return false.
  HandleOp(SDValue(N, 0));
  return false;
}

/// HandleOp - Legalize, Promote, Widen, or Expand the specified operand as
/// appropriate for its type.
void SelectionDAGLegalize::HandleOp(SDValue Op) {
  // Don't touch TargetConstants
  if (Op.getOpcode() == ISD::TargetConstant)
    return;
  MVT VT = Op.getValueType();
  // We should never see any illegal result types here.
  assert(isTypeLegal(VT) && "Illegal type introduced after type legalization?");
  (void)LegalizeOp(Op);
}

/// ExpandConstantFP - Expands the ConstantFP node to an integer constant or
/// a load from the constant pool.
static SDValue ExpandConstantFP(ConstantFPSDNode *CFP, bool UseCP,
                                SelectionDAG &DAG, const TargetLowering &TLI) {
  bool Extend = false;
  DebugLoc dl = CFP->getDebugLoc();

  // If a FP immediate is precise when represented as a float and if the
  // target can do an extending load from float to double, we put it into
  // the constant pool as a float, even if it's is statically typed as a
  // double.  This shrinks FP constants and canonicalizes them for targets where
  // an FP extending load is the same cost as a normal load (such as on the x87
  // fp stack or PPC FP unit).
  MVT VT = CFP->getValueType(0);
  ConstantFP *LLVMC = const_cast<ConstantFP*>(CFP->getConstantFPValue());
  if (!UseCP) {
    assert((VT == MVT::f64 || VT == MVT::f32) && "Invalid type expansion");
    return DAG.getConstant(LLVMC->getValueAPF().bitcastToAPInt(),
                           (VT == MVT::f64) ? MVT::i64 : MVT::i32);
  }

  MVT OrigVT = VT;
  MVT SVT = VT;
  while (SVT != MVT::f32) {
    SVT = (MVT::SimpleValueType)(SVT.getSimpleVT() - 1);
    if (CFP->isValueValidForType(SVT, CFP->getValueAPF()) &&
        // Only do this if the target has a native EXTLOAD instruction from
        // smaller type.
        TLI.isLoadExtLegal(ISD::EXTLOAD, SVT) &&
        TLI.ShouldShrinkFPConstant(OrigVT)) {
      const Type *SType = SVT.getTypeForMVT();
      LLVMC = cast<ConstantFP>(ConstantExpr::getFPTrunc(LLVMC, SType));
      VT = SVT;
      Extend = true;
    }
  }

  SDValue CPIdx = DAG.getConstantPool(LLVMC, TLI.getPointerTy());
  unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  if (Extend)
    return DAG.getExtLoad(ISD::EXTLOAD, dl,
                          OrigVT, DAG.getEntryNode(),
                          CPIdx, PseudoSourceValue::getConstantPool(),
                          0, VT, false, Alignment);
  return DAG.getLoad(OrigVT, dl, DAG.getEntryNode(), CPIdx,
                     PseudoSourceValue::getConstantPool(), 0, false, Alignment);
}

/// ExpandUnalignedStore - Expands an unaligned store to 2 half-size stores.
static
SDValue ExpandUnalignedStore(StoreSDNode *ST, SelectionDAG &DAG,
                             const TargetLowering &TLI) {
  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();
  SDValue Val = ST->getValue();
  MVT VT = Val.getValueType();
  int Alignment = ST->getAlignment();
  int SVOffset = ST->getSrcValueOffset();
  DebugLoc dl = ST->getDebugLoc();
  if (ST->getMemoryVT().isFloatingPoint() ||
      ST->getMemoryVT().isVector()) {
    MVT intVT = MVT::getIntegerVT(VT.getSizeInBits());
    if (TLI.isTypeLegal(intVT)) {
      // Expand to a bitconvert of the value to the integer type of the
      // same size, then a (misaligned) int store.
      // FIXME: Does not handle truncating floating point stores!
      SDValue Result = DAG.getNode(ISD::BIT_CONVERT, dl, intVT, Val);
      return DAG.getStore(Chain, dl, Result, Ptr, ST->getSrcValue(),
                          SVOffset, ST->isVolatile(), Alignment);
    } else {
      // Do a (aligned) store to a stack slot, then copy from the stack slot
      // to the final destination using (unaligned) integer loads and stores.
      MVT StoredVT = ST->getMemoryVT();
      MVT RegVT =
        TLI.getRegisterType(MVT::getIntegerVT(StoredVT.getSizeInBits()));
      unsigned StoredBytes = StoredVT.getSizeInBits() / 8;
      unsigned RegBytes = RegVT.getSizeInBits() / 8;
      unsigned NumRegs = (StoredBytes + RegBytes - 1) / RegBytes;

      // Make sure the stack slot is also aligned for the register type.
      SDValue StackPtr = DAG.CreateStackTemporary(StoredVT, RegVT);

      // Perform the original store, only redirected to the stack slot.
      SDValue Store = DAG.getTruncStore(Chain, dl,
                                        Val, StackPtr, NULL, 0, StoredVT);
      SDValue Increment = DAG.getConstant(RegBytes, TLI.getPointerTy());
      SmallVector<SDValue, 8> Stores;
      unsigned Offset = 0;

      // Do all but one copies using the full register width.
      for (unsigned i = 1; i < NumRegs; i++) {
        // Load one integer register's worth from the stack slot.
        SDValue Load = DAG.getLoad(RegVT, dl, Store, StackPtr, NULL, 0);
        // Store it to the final location.  Remember the store.
        Stores.push_back(DAG.getStore(Load.getValue(1), dl, Load, Ptr,
                                      ST->getSrcValue(), SVOffset + Offset,
                                      ST->isVolatile(),
                                      MinAlign(ST->getAlignment(), Offset)));
        // Increment the pointers.
        Offset += RegBytes;
        StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                               Increment);
        Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
      }

      // The last store may be partial.  Do a truncating store.  On big-endian
      // machines this requires an extending load from the stack slot to ensure
      // that the bits are in the right place.
      MVT MemVT = MVT::getIntegerVT(8 * (StoredBytes - Offset));

      // Load from the stack slot.
      SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, RegVT, Store, StackPtr,
                                    NULL, 0, MemVT);

      Stores.push_back(DAG.getTruncStore(Load.getValue(1), dl, Load, Ptr,
                                         ST->getSrcValue(), SVOffset + Offset,
                                         MemVT, ST->isVolatile(),
                                         MinAlign(ST->getAlignment(), Offset)));
      // The order of the stores doesn't matter - say it with a TokenFactor.
      return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Stores[0],
                         Stores.size());
    }
  }
  assert(ST->getMemoryVT().isInteger() &&
         !ST->getMemoryVT().isVector() &&
         "Unaligned store of unknown type.");
  // Get the half-size VT
  MVT NewStoredVT =
    (MVT::SimpleValueType)(ST->getMemoryVT().getSimpleVT() - 1);
  int NumBits = NewStoredVT.getSizeInBits();
  int IncrementSize = NumBits / 8;

  // Divide the stored value in two parts.
  SDValue ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDValue Lo = Val;
  SDValue Hi = DAG.getNode(ISD::SRL, dl, VT, Val, ShiftAmount);

  // Store the two parts
  SDValue Store1, Store2;
  Store1 = DAG.getTruncStore(Chain, dl, TLI.isLittleEndian()?Lo:Hi, Ptr,
                             ST->getSrcValue(), SVOffset, NewStoredVT,
                             ST->isVolatile(), Alignment);
  Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                    DAG.getConstant(IncrementSize, TLI.getPointerTy()));
  Alignment = MinAlign(Alignment, IncrementSize);
  Store2 = DAG.getTruncStore(Chain, dl, TLI.isLittleEndian()?Hi:Lo, Ptr,
                             ST->getSrcValue(), SVOffset + IncrementSize,
                             NewStoredVT, ST->isVolatile(), Alignment);

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2);
}

/// ExpandUnalignedLoad - Expands an unaligned load to 2 half-size loads.
static
SDValue ExpandUnalignedLoad(LoadSDNode *LD, SelectionDAG &DAG,
                            const TargetLowering &TLI) {
  int SVOffset = LD->getSrcValueOffset();
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  MVT VT = LD->getValueType(0);
  MVT LoadedVT = LD->getMemoryVT();
  DebugLoc dl = LD->getDebugLoc();
  if (VT.isFloatingPoint() || VT.isVector()) {
    MVT intVT = MVT::getIntegerVT(LoadedVT.getSizeInBits());
    if (TLI.isTypeLegal(intVT)) {
      // Expand to a (misaligned) integer load of the same size,
      // then bitconvert to floating point or vector.
      SDValue newLoad = DAG.getLoad(intVT, dl, Chain, Ptr, LD->getSrcValue(),
                                    SVOffset, LD->isVolatile(),
                                    LD->getAlignment());
      SDValue Result = DAG.getNode(ISD::BIT_CONVERT, dl, LoadedVT, newLoad);
      if (VT.isFloatingPoint() && LoadedVT != VT)
        Result = DAG.getNode(ISD::FP_EXTEND, dl, VT, Result);

      SDValue Ops[] = { Result, Chain };
      return DAG.getMergeValues(Ops, 2, dl);
    } else {
      // Copy the value to a (aligned) stack slot using (unaligned) integer
      // loads and stores, then do a (aligned) load from the stack slot.
      MVT RegVT = TLI.getRegisterType(intVT);
      unsigned LoadedBytes = LoadedVT.getSizeInBits() / 8;
      unsigned RegBytes = RegVT.getSizeInBits() / 8;
      unsigned NumRegs = (LoadedBytes + RegBytes - 1) / RegBytes;

      // Make sure the stack slot is also aligned for the register type.
      SDValue StackBase = DAG.CreateStackTemporary(LoadedVT, RegVT);

      SDValue Increment = DAG.getConstant(RegBytes, TLI.getPointerTy());
      SmallVector<SDValue, 8> Stores;
      SDValue StackPtr = StackBase;
      unsigned Offset = 0;

      // Do all but one copies using the full register width.
      for (unsigned i = 1; i < NumRegs; i++) {
        // Load one integer register's worth from the original location.
        SDValue Load = DAG.getLoad(RegVT, dl, Chain, Ptr, LD->getSrcValue(),
                                   SVOffset + Offset, LD->isVolatile(),
                                   MinAlign(LD->getAlignment(), Offset));
        // Follow the load with a store to the stack slot.  Remember the store.
        Stores.push_back(DAG.getStore(Load.getValue(1), dl, Load, StackPtr,
                                      NULL, 0));
        // Increment the pointers.
        Offset += RegBytes;
        Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
        StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                               Increment);
      }

      // The last copy may be partial.  Do an extending load.
      MVT MemVT = MVT::getIntegerVT(8 * (LoadedBytes - Offset));
      SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, RegVT, Chain, Ptr,
                                    LD->getSrcValue(), SVOffset + Offset,
                                    MemVT, LD->isVolatile(),
                                    MinAlign(LD->getAlignment(), Offset));
      // Follow the load with a store to the stack slot.  Remember the store.
      // On big-endian machines this requires a truncating store to ensure
      // that the bits end up in the right place.
      Stores.push_back(DAG.getTruncStore(Load.getValue(1), dl, Load, StackPtr,
                                         NULL, 0, MemVT));

      // The order of the stores doesn't matter - say it with a TokenFactor.
      SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Stores[0],
                               Stores.size());

      // Finally, perform the original load only redirected to the stack slot.
      Load = DAG.getExtLoad(LD->getExtensionType(), dl, VT, TF, StackBase,
                            NULL, 0, LoadedVT);

      // Callers expect a MERGE_VALUES node.
      SDValue Ops[] = { Load, TF };
      return DAG.getMergeValues(Ops, 2, dl);
    }
  }
  assert(LoadedVT.isInteger() && !LoadedVT.isVector() &&
         "Unaligned load of unsupported type.");

  // Compute the new VT that is half the size of the old one.  This is an
  // integer MVT.
  unsigned NumBits = LoadedVT.getSizeInBits();
  MVT NewLoadedVT;
  NewLoadedVT = MVT::getIntegerVT(NumBits/2);
  NumBits >>= 1;

  unsigned Alignment = LD->getAlignment();
  unsigned IncrementSize = NumBits / 8;
  ISD::LoadExtType HiExtType = LD->getExtensionType();

  // If the original load is NON_EXTLOAD, the hi part load must be ZEXTLOAD.
  if (HiExtType == ISD::NON_EXTLOAD)
    HiExtType = ISD::ZEXTLOAD;

  // Load the value in two parts
  SDValue Lo, Hi;
  if (TLI.isLittleEndian()) {
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset, NewLoadedVT, LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Hi = DAG.getExtLoad(HiExtType, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  } else {
    Hi = DAG.getExtLoad(HiExtType, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset, NewLoadedVT, LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  }

  // aggregate the two parts
  SDValue ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDValue Result = DAG.getNode(ISD::SHL, dl, VT, Hi, ShiftAmount);
  Result = DAG.getNode(ISD::OR, dl, VT, Result, Lo);

  SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                             Hi.getValue(1));

  SDValue Ops[] = { Result, TF };
  return DAG.getMergeValues(Ops, 2, dl);
}

/// PerformInsertVectorEltInMemory - Some target cannot handle a variable
/// insertion index for the INSERT_VECTOR_ELT instruction.  In this case, it
/// is necessary to spill the vector being inserted into to memory, perform
/// the insert there, and then read the result back.
SDValue SelectionDAGLegalize::
PerformInsertVectorEltInMemory(SDValue Vec, SDValue Val, SDValue Idx,
                               DebugLoc dl) {
  SDValue Tmp1 = Vec;
  SDValue Tmp2 = Val;
  SDValue Tmp3 = Idx;

  // If the target doesn't support this, we have to spill the input vector
  // to a temporary stack slot, update the element, then reload it.  This is
  // badness.  We could also load the value into a vector register (either
  // with a "move to register" or "extload into register" instruction, then
  // permute it into place, if the idx is a constant and if the idx is
  // supported by the target.
  MVT VT    = Tmp1.getValueType();
  MVT EltVT = VT.getVectorElementType();
  MVT IdxVT = Tmp3.getValueType();
  MVT PtrVT = TLI.getPointerTy();
  SDValue StackPtr = DAG.CreateStackTemporary(VT);

  int SPFI = cast<FrameIndexSDNode>(StackPtr.getNode())->getIndex();

  // Store the vector.
  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, Tmp1, StackPtr,
                            PseudoSourceValue::getFixedStack(SPFI), 0);

  // Truncate or zero extend offset to target pointer type.
  unsigned CastOpc = IdxVT.bitsGT(PtrVT) ? ISD::TRUNCATE : ISD::ZERO_EXTEND;
  Tmp3 = DAG.getNode(CastOpc, dl, PtrVT, Tmp3);
  // Add the offset to the index.
  unsigned EltSize = EltVT.getSizeInBits()/8;
  Tmp3 = DAG.getNode(ISD::MUL, dl, IdxVT, Tmp3,DAG.getConstant(EltSize, IdxVT));
  SDValue StackPtr2 = DAG.getNode(ISD::ADD, dl, IdxVT, Tmp3, StackPtr);
  // Store the scalar value.
  Ch = DAG.getTruncStore(Ch, dl, Tmp2, StackPtr2,
                         PseudoSourceValue::getFixedStack(SPFI), 0, EltVT);
  // Load the updated vector.
  return DAG.getLoad(VT, dl, Ch, StackPtr,
                     PseudoSourceValue::getFixedStack(SPFI), 0);
}


SDValue SelectionDAGLegalize::
ExpandINSERT_VECTOR_ELT(SDValue Vec, SDValue Val, SDValue Idx, DebugLoc dl) {
  if (ConstantSDNode *InsertPos = dyn_cast<ConstantSDNode>(Idx)) {
    // SCALAR_TO_VECTOR requires that the type of the value being inserted
    // match the element type of the vector being created, except for
    // integers in which case the inserted value can be over width.
    MVT EltVT = Vec.getValueType().getVectorElementType();
    if (Val.getValueType() == EltVT ||
        (EltVT.isInteger() && Val.getValueType().bitsGE(EltVT))) {
      SDValue ScVec = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                  Vec.getValueType(), Val);

      unsigned NumElts = Vec.getValueType().getVectorNumElements();
      // We generate a shuffle of InVec and ScVec, so the shuffle mask
      // should be 0,1,2,3,4,5... with the appropriate element replaced with
      // elt 0 of the RHS.
      SmallVector<int, 8> ShufOps;
      for (unsigned i = 0; i != NumElts; ++i)
        ShufOps.push_back(i != InsertPos->getZExtValue() ? i : NumElts);

      return DAG.getVectorShuffle(Vec.getValueType(), dl, Vec, ScVec,
                                  &ShufOps[0]);
    }
  }
  return PerformInsertVectorEltInMemory(Vec, Val, Idx, dl);
}

/// LegalizeOp - We know that the specified value has a legal type, and
/// that its operands are legal.  Now ensure that the operation itself
/// is legal, recursively ensuring that the operands' operations remain
/// legal.
SDValue SelectionDAGLegalize::LegalizeOp(SDValue Op) {
  if (Op.getOpcode() == ISD::TargetConstant) // Allow illegal target nodes.
    return Op;

  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();

  for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
    assert(getTypeAction(Node->getValueType(i)) == Legal &&
           "Unexpected illegal type!");

  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
    assert((isTypeLegal(Node->getOperand(i).getValueType()) || 
            Node->getOperand(i).getOpcode() == ISD::TargetConstant) &&
           "Unexpected illegal type!");

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDValue Tmp1, Tmp2, Tmp3, Tmp4;
  SDValue Result = Op;
  bool isCustom = false;

  // Figure out the correct action; the way to query this varies by opcode
  TargetLowering::LegalizeAction Action;
  bool SimpleFinishLegalizing = true;
  switch (Node->getOpcode()) {
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID:
  case ISD::VAARG:
  case ISD::STACKSAVE:
    Action = TLI.getOperationAction(Node->getOpcode(), MVT::Other);
    break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::EXTRACT_VECTOR_ELT:
    Action = TLI.getOperationAction(Node->getOpcode(),
                                    Node->getOperand(0).getValueType());
    break;
  case ISD::FP_ROUND_INREG:
  case ISD::SIGN_EXTEND_INREG: {
    MVT InnerType = cast<VTSDNode>(Node->getOperand(1))->getVT();
    Action = TLI.getOperationAction(Node->getOpcode(), InnerType);
    break;
  }
  case ISD::LOAD:
  case ISD::STORE:
  case ISD::BR_CC:
  case ISD::FORMAL_ARGUMENTS:
  case ISD::CALL:
  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
  case ISD::SELECT_CC:
  case ISD::SETCC:
    // These instructions have properties that aren't modeled in the
    // generic codepath
    SimpleFinishLegalizing = false;
    break;
  case ISD::EXTRACT_ELEMENT:
  case ISD::FLT_ROUNDS_:
  case ISD::SADDO:
  case ISD::SSUBO:
  case ISD::UADDO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
  case ISD::FPOWI:
  case ISD::MERGE_VALUES:
  case ISD::EH_RETURN:
  case ISD::FRAME_TO_ARGS_OFFSET:
    // These operations lie about being legal: when they claim to be legal,
    // they should actually be expanded.
    Action = TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0));
    if (Action == TargetLowering::Legal)
      Action = TargetLowering::Expand;
    break;
  case ISD::TRAMPOLINE:
  case ISD::FRAMEADDR:
  case ISD::RETURNADDR:
    // These operations lie about being legal: they must always be
    // custom-lowered.
    Action = TargetLowering::Custom;
    break;
  case ISD::BUILD_VECTOR:
    // A weird case: when a BUILD_VECTOR is custom-lowered, it doesn't legalize
    // its operands first!
    SimpleFinishLegalizing = false;
    break;
  default:
    if (Node->getOpcode() >= ISD::BUILTIN_OP_END) {
      Action = TargetLowering::Legal;
    } else {
      Action = TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0));
    }
    break;
  }

  if (SimpleFinishLegalizing) {
    SmallVector<SDValue, 8> Ops, ResultVals;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
    switch (Node->getOpcode()) {
    default: break;
    case ISD::BR:
    case ISD::BRIND:
    case ISD::BR_JT:
    case ISD::BR_CC:
    case ISD::BRCOND:
    case ISD::RET:
      // Branches tweak the chain to include LastCALLSEQ_END
      Ops[0] = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Ops[0],
                            LastCALLSEQ_END);
      Ops[0] = LegalizeOp(Ops[0]);
      LastCALLSEQ_END = DAG.getEntryNode();
      break;
    case ISD::SHL:
    case ISD::SRL:
    case ISD::SRA:
    case ISD::ROTL:
    case ISD::ROTR:
      // Legalizing shifts/rotates requires adjusting the shift amount
      // to the appropriate width.
      if (!Ops[1].getValueType().isVector())
        Ops[1] = LegalizeOp(DAG.getShiftAmountOperand(Ops[1]));
      break;
    }

    Result = DAG.UpdateNodeOperands(Result.getValue(0), Ops.data(),
                                    Ops.size());
    switch (Action) {
    case TargetLowering::Legal:
      for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
        ResultVals.push_back(Result.getValue(i));
      break;
    case TargetLowering::Custom:
      // FIXME: The handling for custom lowering with multiple results is
      // a complete mess.
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) {
        for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i) {
          if (e == 1)
            ResultVals.push_back(Tmp1);
          else
            ResultVals.push_back(Tmp1.getValue(i));
        }
        break;
      }

      // FALL THROUGH
    case TargetLowering::Expand:
      ExpandNode(Result.getNode(), ResultVals);
      break;
    case TargetLowering::Promote:
      PromoteNode(Result.getNode(), ResultVals);
      break;
    }
    if (!ResultVals.empty()) {
      for (unsigned i = 0, e = ResultVals.size(); i != e; ++i) {
        if (ResultVals[i] != SDValue(Node, i))
          ResultVals[i] = LegalizeOp(ResultVals[i]);
        AddLegalizedOperand(SDValue(Node, i), ResultVals[i]);
      }
      return ResultVals[Op.getResNo()];
    }
  }

  switch (Node->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "NODE: "; Node->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to legalize this operator!");
    abort();
  case ISD::DBG_STOPPOINT:
    assert(Node->getNumOperands() == 1 && "Invalid DBG_STOPPOINT node!");
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the input chain.

    switch (TLI.getOperationAction(ISD::DBG_STOPPOINT, MVT::Other)) {
    case TargetLowering::Promote:
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
      DwarfWriter *DW = DAG.getDwarfWriter();
      bool useDEBUG_LOC = TLI.isOperationLegalOrCustom(ISD::DEBUG_LOC,
                                                       MVT::Other);
      bool useLABEL = TLI.isOperationLegalOrCustom(ISD::DBG_LABEL, MVT::Other);

      const DbgStopPointSDNode *DSP = cast<DbgStopPointSDNode>(Node);
      GlobalVariable *CU_GV = cast<GlobalVariable>(DSP->getCompileUnit());
      if (DW && (useDEBUG_LOC || useLABEL) && !CU_GV->isDeclaration()) {
        DICompileUnit CU(cast<GlobalVariable>(DSP->getCompileUnit()));

        unsigned Line = DSP->getLine();
        unsigned Col = DSP->getColumn();

        if (OptLevel == CodeGenOpt::None) {
          // A bit self-referential to have DebugLoc on Debug_Loc nodes, but it
          // won't hurt anything.
          if (useDEBUG_LOC) {
            SDValue Ops[] = { Tmp1, DAG.getConstant(Line, MVT::i32),
                              DAG.getConstant(Col, MVT::i32),
                              DAG.getSrcValue(CU.getGV()) };
            Result = DAG.getNode(ISD::DEBUG_LOC, dl, MVT::Other, Ops, 4);
          } else {
            unsigned ID = DW->RecordSourceLine(Line, Col, CU);
            Result = DAG.getLabel(ISD::DBG_LABEL, dl, Tmp1, ID);
          }
        } else {
          Result = Tmp1;  // chain
        }
      } else {
        Result = Tmp1;  // chain
      }
      break;
    }
   case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode())
        break;
    case TargetLowering::Legal: {
      if (Tmp1 == Node->getOperand(0))
        break;

      SmallVector<SDValue, 8> Ops;
      Ops.push_back(Tmp1);
      Ops.push_back(Node->getOperand(1));  // line # must be legal.
      Ops.push_back(Node->getOperand(2));  // col # must be legal.
      Ops.push_back(Node->getOperand(3));  // filename must be legal.
      Ops.push_back(Node->getOperand(4));  // working dir # must be legal.
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      break;
    }
    }
    break;
  case ISD::ConstantFP: {
    // Spill FP immediates to the constant pool if the target cannot directly
    // codegen them.  Targets often have some immediate values that can be
    // efficiently generated into an FP register without a load.  We explicitly
    // leave these constants as ConstantFP nodes for the target to deal with.
    ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Node);

    switch (TLI.getOperationAction(ISD::ConstantFP, CFP->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      // Check to see if this FP immediate is already legal.
      bool isLegal = false;
      for (TargetLowering::legal_fpimm_iterator I = TLI.legal_fpimm_begin(),
             E = TLI.legal_fpimm_end(); I != E; ++I) {
        if (CFP->isExactlyValue(*I)) {
          isLegal = true;
          break;
        }
      }
      // If this is a legal constant, turn it into a TargetConstantFP node.
      if (isLegal)
        break;
      Result = ExpandConstantFP(CFP, true, DAG, TLI);
    }
    }
    break;
  }
  case ISD::FORMAL_ARGUMENTS:
  case ISD::CALL:
    // The only option for this is to custom lower it.
    Tmp3 = TLI.LowerOperation(Result.getValue(0), DAG);
    assert(Tmp3.getNode() && "Target didn't custom lower this node!");
    // A call within a calling sequence must be legalized to something
    // other than the normal CALLSEQ_END.  Violating this gets Legalize
    // into an infinite loop.
    assert ((!IsLegalizingCall ||
             Node->getOpcode() != ISD::CALL ||
             Tmp3.getNode()->getOpcode() != ISD::CALLSEQ_END) &&
            "Nested CALLSEQ_START..CALLSEQ_END not supported.");

    // The number of incoming and outgoing values should match; unless the final
    // outgoing value is a flag.
    assert((Tmp3.getNode()->getNumValues() == Result.getNode()->getNumValues() ||
            (Tmp3.getNode()->getNumValues() == Result.getNode()->getNumValues() + 1 &&
             Tmp3.getNode()->getValueType(Tmp3.getNode()->getNumValues() - 1) ==
               MVT::Flag)) &&
           "Lowering call/formal_arguments produced unexpected # results!");

    // Since CALL/FORMAL_ARGUMENTS nodes produce multiple values, make sure to
    // remember that we legalized all of them, so it doesn't get relegalized.
    for (unsigned i = 0, e = Tmp3.getNode()->getNumValues(); i != e; ++i) {
      if (Tmp3.getNode()->getValueType(i) == MVT::Flag)
        continue;
      Tmp1 = LegalizeOp(Tmp3.getValue(i));
      if (Op.getResNo() == i)
        Tmp2 = Tmp1;
      AddLegalizedOperand(SDValue(Node, i), Tmp1);
    }
    return Tmp2;
  case ISD::BUILD_VECTOR:
    switch (TLI.getOperationAction(ISD::BUILD_VECTOR, Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand:
      Result = ExpandBUILD_VECTOR(Result.getNode());
      break;
    }
    break;
  case ISD::VECTOR_SHUFFLE: {
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Legalize the input vectors,
    Tmp2 = LegalizeOp(Node->getOperand(1));   // but not the shuffle mask.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    MVT VT = Result.getValueType();

    // Copy the Mask to a local SmallVector for use with isShuffleMaskLegal.
    SmallVector<int, 8> Mask;
    cast<ShuffleVectorSDNode>(Result)->getMask(Mask);

    // Allow targets to custom lower the SHUFFLEs they support.
    switch (TLI.getOperationAction(ISD::VECTOR_SHUFFLE, VT)) {
    default: assert(0 && "Unknown operation action!");
    case TargetLowering::Legal:
      assert(TLI.isShuffleMaskLegal(Mask, VT) &&
             "vector shuffle should not be created if not legal!");
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      MVT EltVT = VT.getVectorElementType();
      unsigned NumElems = VT.getVectorNumElements();
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i != NumElems; ++i) {
        if (Mask[i] < 0) {
          Ops.push_back(DAG.getUNDEF(EltVT));
          continue;
        }
        unsigned Idx = Mask[i];
        if (Idx < NumElems)
          Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Tmp1,
                                    DAG.getIntPtrConstant(Idx)));
        else
          Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Tmp2,
                                    DAG.getIntPtrConstant(Idx - NumElems)));
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], Ops.size());
      break;
    }
    case TargetLowering::Promote: {
      // Change base type to a different vector type.
      MVT OVT = Node->getValueType(0);
      MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Cast the two input vectors.
      Tmp1 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp1);
      Tmp2 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp2);

      // Convert the shuffle mask to the right # elements.
      Result = ShuffleWithNarrowerEltType(NVT, OVT, dl, Tmp1, Tmp2, Mask);
      Result = DAG.getNode(ISD::BIT_CONVERT, dl, OVT, Result);
      break;
    }
    }
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // Legalize the operands.
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
    Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());

    switch (TLI.getOperationAction(ISD::CONCAT_VECTORS,
                                   Node->getValueType(0))) {
    default: assert(0 && "Unknown operation action!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      // Use extract/insert/build vector for now. We might try to be
      // more clever later.
      MVT PtrVT = TLI.getPointerTy();
      SmallVector<SDValue, 8> Ops;
      unsigned NumOperands = Node->getNumOperands();
      for (unsigned i=0; i < NumOperands; ++i) {
        SDValue SubOp = Node->getOperand(i);
        MVT VVT = SubOp.getNode()->getValueType(0);
        MVT EltVT = VVT.getVectorElementType();
        unsigned NumSubElem = VVT.getVectorNumElements();
        for (unsigned j=0; j < NumSubElem; ++j) {
          Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, SubOp,
                                    DAG.getConstant(j, PtrVT)));
        }
      }
      return LegalizeOp(DAG.getNode(ISD::BUILD_VECTOR, dl,
                                    Node->getValueType(0),
                                    &Ops[0], Ops.size()));
    }
    }
    break;
  }

  case ISD::CALLSEQ_START: {
    SDNode *CallEnd = FindCallEndFromCallStart(Node);

    // Recursively Legalize all of the inputs of the call end that do not lead
    // to this call start.  This ensures that any libcalls that need be inserted
    // are inserted *before* the CALLSEQ_START.
    {SmallPtrSet<SDNode*, 32> NodesLeadingTo;
    for (unsigned i = 0, e = CallEnd->getNumOperands(); i != e; ++i)
      LegalizeAllNodesNotLeadingTo(CallEnd->getOperand(i).getNode(), Node,
                                   NodesLeadingTo);
    }

    // Now that we legalized all of the inputs (which may have inserted
    // libcalls) create the new CALLSEQ_START node.
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    // Merge in the last call, to ensure that this call start after the last
    // call ended.
    if (LastCALLSEQ_END.getOpcode() != ISD::EntryToken) {
      Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                         Tmp1, LastCALLSEQ_END);
      Tmp1 = LegalizeOp(Tmp1);
    }

    // Do not try to legalize the target-specific arguments (#1+).
    if (Tmp1 != Node->getOperand(0)) {
      SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
      Ops[0] = Tmp1;
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
    }

    // Remember that the CALLSEQ_START is legalized.
    AddLegalizedOperand(Op.getValue(0), Result);
    if (Node->getNumValues() == 2)    // If this has a flag result, remember it.
      AddLegalizedOperand(Op.getValue(1), Result.getValue(1));

    // Now that the callseq_start and all of the non-call nodes above this call
    // sequence have been legalized, legalize the call itself.  During this
    // process, no libcalls can/will be inserted, guaranteeing that no calls
    // can overlap.
    assert(!IsLegalizingCall && "Inconsistent sequentialization of calls!");
    // Note that we are selecting this call!
    LastCALLSEQ_END = SDValue(CallEnd, 0);
    IsLegalizingCall = true;

    // Legalize the call, starting from the CALLSEQ_END.
    LegalizeOp(LastCALLSEQ_END);
    assert(!IsLegalizingCall && "CALLSEQ_END should have cleared this!");
    return Result;
  }
  case ISD::CALLSEQ_END:
    // If the CALLSEQ_START node hasn't been legalized first, legalize it.  This
    // will cause this node to be legalized as well as handling libcalls right.
    if (LastCALLSEQ_END.getNode() != Node) {
      LegalizeOp(SDValue(FindCallStartFromCallEnd(Node), 0));
      DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
      assert(I != LegalizedNodes.end() &&
             "Legalizing the call start should have legalized this node!");
      return I->second;
    }

    // Otherwise, the call start has been legalized and everything is going
    // according to plan.  Just legalize ourselves normally here.
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Do not try to legalize the target-specific arguments (#1+), except for
    // an optional flag input.
    if (Node->getOperand(Node->getNumOperands()-1).getValueType() != MVT::Flag){
      if (Tmp1 != Node->getOperand(0)) {
        SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    } else {
      Tmp2 = LegalizeOp(Node->getOperand(Node->getNumOperands()-1));
      if (Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(Node->getNumOperands()-1)) {
        SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Ops.back() = Tmp2;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    }
    assert(IsLegalizingCall && "Call sequence imbalance between start/end?");
    // This finishes up call legalization.
    IsLegalizingCall = false;

    // If the CALLSEQ_END node has a flag, remember that we legalized it.
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    if (Node->getNumValues() == 2)
      AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  case ISD::DYNAMIC_STACKALLOC: {
    MVT VT = Node->getValueType(0);
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the size.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the alignment.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);

    Tmp1 = Result.getValue(0);
    Tmp2 = Result.getValue(1);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
      unsigned SPReg = TLI.getStackPointerRegisterToSaveRestore();
      assert(SPReg && "Target cannot require DYNAMIC_STACKALLOC expansion and"
             " not tell us which reg is the stack pointer!");
      SDValue Chain = Tmp1.getOperand(0);

      // Chain the dynamic stack allocation so that it doesn't modify the stack
      // pointer when other instructions are using the stack.
      Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(0, true));

      SDValue Size  = Tmp2.getOperand(1);
      SDValue SP = DAG.getCopyFromReg(Chain, dl, SPReg, VT);
      Chain = SP.getValue(1);
      unsigned Align = cast<ConstantSDNode>(Tmp3)->getZExtValue();
      unsigned StackAlign =
        TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
      if (Align > StackAlign)
        SP = DAG.getNode(ISD::AND, dl, VT, SP,
                         DAG.getConstant(-(uint64_t)Align, VT));
      Tmp1 = DAG.getNode(ISD::SUB, dl, VT, SP, Size);       // Value
      Chain = DAG.getCopyToReg(Chain, dl, SPReg, Tmp1);     // Output chain

      Tmp2 = DAG.getCALLSEQ_END(Chain,  DAG.getIntPtrConstant(0, true),
                                DAG.getIntPtrConstant(0, true), SDValue());

      Tmp1 = LegalizeOp(Tmp1);
      Tmp2 = LegalizeOp(Tmp2);
      break;
    }
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Tmp1, DAG);
      if (Tmp3.getNode()) {
        Tmp1 = LegalizeOp(Tmp3);
        Tmp2 = LegalizeOp(Tmp3.getValue(1));
      }
      break;
    case TargetLowering::Legal:
      break;
    }
    // Since this op produce two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Tmp1);
    AddLegalizedOperand(SDValue(Node, 1), Tmp2);
    return Op.getResNo() ? Tmp2 : Tmp1;
  }
  case ISD::BR_JT:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();

    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the jumptable node.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));

    switch (TLI.getOperationAction(ISD::BR_JT, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Expand: {
      SDValue Chain = Result.getOperand(0);
      SDValue Table = Result.getOperand(1);
      SDValue Index = Result.getOperand(2);

      MVT PTy = TLI.getPointerTy();
      MachineFunction &MF = DAG.getMachineFunction();
      unsigned EntrySize = MF.getJumpTableInfo()->getEntrySize();
      Index= DAG.getNode(ISD::MUL, dl, PTy,
                         Index, DAG.getConstant(EntrySize, PTy));
      SDValue Addr = DAG.getNode(ISD::ADD, dl, PTy, Index, Table);

      MVT MemVT = MVT::getIntegerVT(EntrySize * 8);
      SDValue LD = DAG.getExtLoad(ISD::SEXTLOAD, dl, PTy, Chain, Addr,
                                  PseudoSourceValue::getJumpTable(), 0, MemVT);
      Addr = LD;
      if (TLI.getTargetMachine().getRelocationModel() == Reloc::PIC_) {
        // For PIC, the sequence is:
        // BRIND(load(Jumptable + index) + RelocBase)
        // RelocBase can be JumpTable, GOT or some sort of global base.
        Addr = DAG.getNode(ISD::ADD, dl, PTy, Addr,
                           TLI.getPICJumpTableRelocBase(Table, DAG));
      }
      Result = DAG.getNode(ISD::BRIND, dl, MVT::Other, LD.getValue(1), Addr);
    }
    }
    break;
  case ISD::BRCOND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a return.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();

    Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the condition.

    // Basic block destination (Op#2) is always legal.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));

    switch (TLI.getOperationAction(ISD::BRCOND, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Expand:
      // Expand brcond's setcc into its constituent parts and create a BR_CC
      // Node.
      if (Tmp2.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::BR_CC, dl, MVT::Other,
                             Tmp1, Tmp2.getOperand(2),
                             Tmp2.getOperand(0), Tmp2.getOperand(1),
                             Node->getOperand(2));
      } else {
        Result = DAG.getNode(ISD::BR_CC, dl, MVT::Other, Tmp1,
                             DAG.getCondCode(ISD::SETNE), Tmp2,
                             DAG.getConstant(0, Tmp2.getValueType()),
                             Node->getOperand(2));
      }
      break;
    }
    break;
  case ISD::BR_CC:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    Tmp2 = Node->getOperand(2);              // LHS
    Tmp3 = Node->getOperand(3);              // RHS
    Tmp4 = Node->getOperand(1);              // CC

    LegalizeSetCC(TLI.getSetCCResultType(Tmp2.getValueType()),
                  Tmp2, Tmp3, Tmp4, dl);
    LastCALLSEQ_END = DAG.getEntryNode();

    // If we didn't get both a LHS and RHS back from LegalizeSetCC,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp3.getNode() == 0) {
      Tmp3 = DAG.getConstant(0, Tmp2.getValueType());
      Tmp4 = DAG.getCondCode(ISD::SETNE);
    }

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp4, Tmp2, Tmp3,
                                    Node->getOperand(4));

    switch (TLI.getOperationAction(ISD::BR_CC, Tmp3.getValueType())) {
    default: assert(0 && "Unexpected action for BR_CC!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp4 = TLI.LowerOperation(Result, DAG);
      if (Tmp4.getNode()) Result = Tmp4;
      break;
    }
    break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    Tmp1 = LegalizeOp(LD->getChain());   // Legalize the chain.
    Tmp2 = LegalizeOp(LD->getBasePtr()); // Legalize the base pointer.

    ISD::LoadExtType ExtType = LD->getExtensionType();
    if (ExtType == ISD::NON_EXTLOAD) {
      MVT VT = Node->getValueType(0);
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, LD->getOffset());
      Tmp3 = Result.getValue(0);
      Tmp4 = Result.getValue(1);

      switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Legal:
        // If this is an unaligned load and the target doesn't support it,
        // expand it.
        if (!TLI.allowsUnalignedMemoryAccesses()) {
          unsigned ABIAlignment = TLI.getTargetData()->
            getABITypeAlignment(LD->getMemoryVT().getTypeForMVT());
          if (LD->getAlignment() < ABIAlignment){
            Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.getNode()), DAG,
                                         TLI);
            Tmp3 = Result.getOperand(0);
            Tmp4 = Result.getOperand(1);
            Tmp3 = LegalizeOp(Tmp3);
            Tmp4 = LegalizeOp(Tmp4);
          }
        }
        break;
      case TargetLowering::Custom:
        Tmp1 = TLI.LowerOperation(Tmp3, DAG);
        if (Tmp1.getNode()) {
          Tmp3 = LegalizeOp(Tmp1);
          Tmp4 = LegalizeOp(Tmp1.getValue(1));
        }
        break;
      case TargetLowering::Promote: {
        // Only promote a load of vector type to another.
        assert(VT.isVector() && "Cannot promote this load!");
        // Change base type to a different vector type.
        MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), VT);

        Tmp1 = DAG.getLoad(NVT, dl, Tmp1, Tmp2, LD->getSrcValue(),
                           LD->getSrcValueOffset(),
                           LD->isVolatile(), LD->getAlignment());
        Tmp3 = LegalizeOp(DAG.getNode(ISD::BIT_CONVERT, dl, VT, Tmp1));
        Tmp4 = LegalizeOp(Tmp1.getValue(1));
        break;
      }
      }
      // Since loads produce two values, make sure to remember that we
      // legalized both of them.
      AddLegalizedOperand(SDValue(Node, 0), Tmp3);
      AddLegalizedOperand(SDValue(Node, 1), Tmp4);
      return Op.getResNo() ? Tmp4 : Tmp3;
    } else {
      MVT SrcVT = LD->getMemoryVT();
      unsigned SrcWidth = SrcVT.getSizeInBits();
      int SVOffset = LD->getSrcValueOffset();
      unsigned Alignment = LD->getAlignment();
      bool isVolatile = LD->isVolatile();

      if (SrcWidth != SrcVT.getStoreSizeInBits() &&
          // Some targets pretend to have an i1 loading operation, and actually
          // load an i8.  This trick is correct for ZEXTLOAD because the top 7
          // bits are guaranteed to be zero; it helps the optimizers understand
          // that these bits are zero.  It is also useful for EXTLOAD, since it
          // tells the optimizers that those bits are undefined.  It would be
          // nice to have an effective generic way of getting these benefits...
          // Until such a way is found, don't insist on promoting i1 here.
          (SrcVT != MVT::i1 ||
           TLI.getLoadExtAction(ExtType, MVT::i1) == TargetLowering::Promote)) {
        // Promote to a byte-sized load if not loading an integral number of
        // bytes.  For example, promote EXTLOAD:i20 -> EXTLOAD:i24.
        unsigned NewWidth = SrcVT.getStoreSizeInBits();
        MVT NVT = MVT::getIntegerVT(NewWidth);
        SDValue Ch;

        // The extra bits are guaranteed to be zero, since we stored them that
        // way.  A zext load from NVT thus automatically gives zext from SrcVT.

        ISD::LoadExtType NewExtType =
          ExtType == ISD::ZEXTLOAD ? ISD::ZEXTLOAD : ISD::EXTLOAD;

        Result = DAG.getExtLoad(NewExtType, dl, Node->getValueType(0),
                                Tmp1, Tmp2, LD->getSrcValue(), SVOffset,
                                NVT, isVolatile, Alignment);

        Ch = Result.getValue(1); // The chain.

        if (ExtType == ISD::SEXTLOAD)
          // Having the top bits zero doesn't help when sign extending.
          Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl,
                               Result.getValueType(),
                               Result, DAG.getValueType(SrcVT));
        else if (ExtType == ISD::ZEXTLOAD || NVT == Result.getValueType())
          // All the top bits are guaranteed to be zero - inform the optimizers.
          Result = DAG.getNode(ISD::AssertZext, dl,
                               Result.getValueType(), Result,
                               DAG.getValueType(SrcVT));

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else if (SrcWidth & (SrcWidth - 1)) {
        // If not loading a power-of-2 number of bits, expand as two loads.
        assert(SrcVT.isExtended() && !SrcVT.isVector() &&
               "Unsupported extload!");
        unsigned RoundWidth = 1 << Log2_32(SrcWidth);
        assert(RoundWidth < SrcWidth);
        unsigned ExtraWidth = SrcWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Load size not an integral number of bytes!");
        MVT RoundVT = MVT::getIntegerVT(RoundWidth);
        MVT ExtraVT = MVT::getIntegerVT(ExtraWidth);
        SDValue Lo, Hi, Ch;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // EXTLOAD:i24 -> ZEXTLOAD:i16 | (shl EXTLOAD@+2:i8, 16)
          // Load the bottom RoundWidth bits.
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl,
                              Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getExtLoad(ExtType, dl, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, dl, Hi.getValueType(), Hi,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, dl, Node->getValueType(0), Lo, Hi);
        } else {
          // Big endian - avoid unaligned loads.
          // EXTLOAD:i24 -> (shl EXTLOAD:i16, 8) | ZEXTLOAD@+2:i8
          // Load the top RoundWidth bits.
          Hi = DAG.getExtLoad(ExtType, dl, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl,
                              Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, dl, Hi.getValueType(), Hi,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, dl, Node->getValueType(0), Lo, Hi);
        }

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else {
        switch (TLI.getLoadExtAction(ExtType, SrcVT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Custom:
          isCustom = true;
          // FALLTHROUGH
        case TargetLowering::Legal:
          Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, LD->getOffset());
          Tmp1 = Result.getValue(0);
          Tmp2 = Result.getValue(1);

          if (isCustom) {
            Tmp3 = TLI.LowerOperation(Result, DAG);
            if (Tmp3.getNode()) {
              Tmp1 = LegalizeOp(Tmp3);
              Tmp2 = LegalizeOp(Tmp3.getValue(1));
            }
          } else {
            // If this is an unaligned load and the target doesn't support it,
            // expand it.
            if (!TLI.allowsUnalignedMemoryAccesses()) {
              unsigned ABIAlignment = TLI.getTargetData()->
                getABITypeAlignment(LD->getMemoryVT().getTypeForMVT());
              if (LD->getAlignment() < ABIAlignment){
                Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.getNode()), DAG,
                                             TLI);
                Tmp1 = Result.getOperand(0);
                Tmp2 = Result.getOperand(1);
                Tmp1 = LegalizeOp(Tmp1);
                Tmp2 = LegalizeOp(Tmp2);
              }
            }
          }
          break;
        case TargetLowering::Expand:
          // f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
          if (SrcVT == MVT::f32 && Node->getValueType(0) == MVT::f64) {
            SDValue Load = DAG.getLoad(SrcVT, dl, Tmp1, Tmp2, LD->getSrcValue(),
                                         LD->getSrcValueOffset(),
                                         LD->isVolatile(), LD->getAlignment());
            Result = DAG.getNode(ISD::FP_EXTEND, dl,
                                 Node->getValueType(0), Load);
            Tmp1 = LegalizeOp(Result);  // Relegalize new nodes.
            Tmp2 = LegalizeOp(Load.getValue(1));
            break;
          }
          assert(ExtType != ISD::EXTLOAD &&"EXTLOAD should always be supported!");
          // Turn the unsupported load into an EXTLOAD followed by an explicit
          // zero/sign extend inreg.
          Result = DAG.getExtLoad(ISD::EXTLOAD, dl, Node->getValueType(0),
                                  Tmp1, Tmp2, LD->getSrcValue(),
                                  LD->getSrcValueOffset(), SrcVT,
                                  LD->isVolatile(), LD->getAlignment());
          SDValue ValRes;
          if (ExtType == ISD::SEXTLOAD)
            ValRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl,
                                 Result.getValueType(),
                                 Result, DAG.getValueType(SrcVT));
          else
            ValRes = DAG.getZeroExtendInReg(Result, dl, SrcVT);
          Tmp1 = LegalizeOp(ValRes);  // Relegalize new nodes.
          Tmp2 = LegalizeOp(Result.getValue(1));  // Relegalize new nodes.
          break;
        }
      }

      // Since loads produce two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDValue(Node, 0), Tmp1);
      AddLegalizedOperand(SDValue(Node, 1), Tmp2);
      return Op.getResNo() ? Tmp2 : Tmp1;
    }
  }
  case ISD::STORE: {
    StoreSDNode *ST = cast<StoreSDNode>(Node);
    Tmp1 = LegalizeOp(ST->getChain());    // Legalize the chain.
    Tmp2 = LegalizeOp(ST->getBasePtr());  // Legalize the pointer.
    int SVOffset = ST->getSrcValueOffset();
    unsigned Alignment = ST->getAlignment();
    bool isVolatile = ST->isVolatile();

    if (!ST->isTruncatingStore()) {
      // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
      // FIXME: We shouldn't do this for TargetConstantFP's.
      // FIXME: move this to the DAG Combiner!  Note that we can't regress due
      // to phase ordering between legalized code and the dag combiner.  This
      // probably means that we need to integrate dag combiner and legalizer
      // together.
      // We generally can't do this one for long doubles.
      if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(ST->getValue())) {
        if (CFP->getValueType(0) == MVT::f32 &&
            getTypeAction(MVT::i32) == Legal) {
          Tmp3 = DAG.getConstant(CFP->getValueAPF().
                                          bitcastToAPInt().zextOrTrunc(32),
                                  MVT::i32);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                SVOffset, isVolatile, Alignment);
          break;
        } else if (CFP->getValueType(0) == MVT::f64) {
          // If this target supports 64-bit registers, do a single 64-bit store.
          if (getTypeAction(MVT::i64) == Legal) {
            Tmp3 = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                     zextOrTrunc(64), MVT::i64);
            Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            break;
          } else if (getTypeAction(MVT::i32) == Legal && !ST->isVolatile()) {
            // Otherwise, if the target supports 32-bit registers, use 2 32-bit
            // stores.  If the target supports neither 32- nor 64-bits, this
            // xform is certainly not worth it.
            const APInt &IntVal =CFP->getValueAPF().bitcastToAPInt();
            SDValue Lo = DAG.getConstant(APInt(IntVal).trunc(32), MVT::i32);
            SDValue Hi = DAG.getConstant(IntVal.lshr(32).trunc(32), MVT::i32);
            if (TLI.isBigEndian()) std::swap(Lo, Hi);

            Lo = DAG.getStore(Tmp1, dl, Lo, Tmp2, ST->getSrcValue(),
                              SVOffset, isVolatile, Alignment);
            Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                               DAG.getIntPtrConstant(4));
            Hi = DAG.getStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(), SVOffset+4,
                              isVolatile, MinAlign(Alignment, 4U));

            Result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
            break;
          }
        }
      }

      {
        Tmp3 = LegalizeOp(ST->getValue());
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp3, Tmp2,
                                        ST->getOffset());

        MVT VT = Tmp3.getValueType();
        switch (TLI.getOperationAction(ISD::STORE, VT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Legal:
          // If this is an unaligned store and the target doesn't support it,
          // expand it.
          if (!TLI.allowsUnalignedMemoryAccesses()) {
            unsigned ABIAlignment = TLI.getTargetData()->
              getABITypeAlignment(ST->getMemoryVT().getTypeForMVT());
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.getNode()), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.getNode()) Result = Tmp1;
          break;
        case TargetLowering::Promote:
          assert(VT.isVector() && "Unknown legal promote case!");
          Tmp3 = DAG.getNode(ISD::BIT_CONVERT, dl,
                             TLI.getTypeToPromoteTo(ISD::STORE, VT), Tmp3);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2,
                                ST->getSrcValue(), SVOffset, isVolatile,
                                Alignment);
          break;
        }
        break;
      }
    } else {
      Tmp3 = LegalizeOp(ST->getValue());

      MVT StVT = ST->getMemoryVT();
      unsigned StWidth = StVT.getSizeInBits();

      if (StWidth != StVT.getStoreSizeInBits()) {
        // Promote to a byte-sized store with upper bits zero if not
        // storing an integral number of bytes.  For example, promote
        // TRUNCSTORE:i1 X -> TRUNCSTORE:i8 (and X, 1)
        MVT NVT = MVT::getIntegerVT(StVT.getStoreSizeInBits());
        Tmp3 = DAG.getZeroExtendInReg(Tmp3, dl, StVT);
        Result = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                   SVOffset, NVT, isVolatile, Alignment);
      } else if (StWidth & (StWidth - 1)) {
        // If not storing a power-of-2 number of bits, expand as two stores.
        assert(StVT.isExtended() && !StVT.isVector() &&
               "Unsupported truncstore!");
        unsigned RoundWidth = 1 << Log2_32(StWidth);
        assert(RoundWidth < StWidth);
        unsigned ExtraWidth = StWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Store size not an integral number of bytes!");
        MVT RoundVT = MVT::getIntegerVT(RoundWidth);
        MVT ExtraVT = MVT::getIntegerVT(ExtraWidth);
        SDValue Lo, Hi;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 X, TRUNCSTORE@+2:i8 (srl X, 16)
          // Store the bottom RoundWidth bits.
          Lo = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset, RoundVT,
                                 isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getNode(ISD::SRL, dl, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        } else {
          // Big endian - avoid unaligned stores.
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 (srl X, 8), TRUNCSTORE@+2:i8 X
          // Store the top RoundWidth bits.
          Hi = DAG.getNode(ISD::SRL, dl, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(),
                                 SVOffset, RoundVT, isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        }

        // The order of the stores doesn't matter.
        Result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
      } else {
        if (Tmp1 != ST->getChain() || Tmp3 != ST->getValue() ||
            Tmp2 != ST->getBasePtr())
          Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp3, Tmp2,
                                          ST->getOffset());

        switch (TLI.getTruncStoreAction(ST->getValue().getValueType(), StVT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Legal:
          // If this is an unaligned store and the target doesn't support it,
          // expand it.
          if (!TLI.allowsUnalignedMemoryAccesses()) {
            unsigned ABIAlignment = TLI.getTargetData()->
              getABITypeAlignment(ST->getMemoryVT().getTypeForMVT());
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.getNode()), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Result = TLI.LowerOperation(Result, DAG);
          break;
        case Expand:
          // TRUNCSTORE:i16 i32 -> STORE i16
          assert(isTypeLegal(StVT) && "Do not know how to expand this store!");
          Tmp3 = DAG.getNode(ISD::TRUNCATE, dl, StVT, Tmp3);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                SVOffset, isVolatile, Alignment);
          break;
        }
      }
    }
    break;
  }
  case ISD::SELECT:
    Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the condition.
    Tmp2 = LegalizeOp(Node->getOperand(1));   // TrueVal
    Tmp3 = LegalizeOp(Node->getOperand(2));   // FalseVal

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);

    switch (TLI.getOperationAction(ISD::SELECT, Tmp2.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom: {
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    }
    case TargetLowering::Expand:
      if (Tmp1.getOpcode() == ISD::SETCC) {
        Result = DAG.getSelectCC(dl, Tmp1.getOperand(0), Tmp1.getOperand(1),
                              Tmp2, Tmp3,
                              cast<CondCodeSDNode>(Tmp1.getOperand(2))->get());
      } else {
        Result = DAG.getSelectCC(dl, Tmp1,
                                 DAG.getConstant(0, Tmp1.getValueType()),
                                 Tmp2, Tmp3, ISD::SETNE);
      }
      break;
    case TargetLowering::Promote: {
      MVT NVT =
        TLI.getTypeToPromoteTo(ISD::SELECT, Tmp2.getValueType());
      unsigned ExtOp, TruncOp;
      if (Tmp2.getValueType().isVector()) {
        ExtOp   = ISD::BIT_CONVERT;
        TruncOp = ISD::BIT_CONVERT;
      } else if (Tmp2.getValueType().isInteger()) {
        ExtOp   = ISD::ANY_EXTEND;
        TruncOp = ISD::TRUNCATE;
      } else {
        ExtOp   = ISD::FP_EXTEND;
        TruncOp = ISD::FP_ROUND;
      }
      // Promote each of the values to the new type.
      Tmp2 = DAG.getNode(ExtOp, dl, NVT, Tmp2);
      Tmp3 = DAG.getNode(ExtOp, dl, NVT, Tmp3);
      // Perform the larger operation, then round down.
      Result = DAG.getNode(ISD::SELECT, dl, NVT, Tmp1, Tmp2, Tmp3);
      if (TruncOp != ISD::FP_ROUND)
        Result = DAG.getNode(TruncOp, dl, Node->getValueType(0), Result);
      else
        Result = DAG.getNode(TruncOp, dl, Node->getValueType(0), Result,
                             DAG.getIntPtrConstant(0));
      break;
    }
    }
    break;
  case ISD::SELECT_CC: {
    Tmp1 = Node->getOperand(0);               // LHS
    Tmp2 = Node->getOperand(1);               // RHS
    Tmp3 = LegalizeOp(Node->getOperand(2));   // True
    Tmp4 = LegalizeOp(Node->getOperand(3));   // False
    SDValue CC = Node->getOperand(4);

    LegalizeSetCC(TLI.getSetCCResultType(Tmp1.getValueType()),
                  Tmp1, Tmp2, CC, dl);

    // If we didn't get both a LHS and RHS back from LegalizeSetCC,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp2.getNode() == 0) {
      Tmp2 = DAG.getConstant(0, Tmp1.getValueType());
      CC = DAG.getCondCode(ISD::SETNE);
    }
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3, Tmp4, CC);

    // Everything is legal, see if we should expand this op or something.
    switch (TLI.getOperationAction(ISD::SELECT_CC, Tmp3.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    }
    break;
  }
  case ISD::SETCC:
    Tmp1 = Node->getOperand(0);
    Tmp2 = Node->getOperand(1);
    Tmp3 = Node->getOperand(2);
    LegalizeSetCC(Node->getValueType(0), Tmp1, Tmp2, Tmp3, dl);

    // If we had to Expand the SetCC operands into a SELECT node, then it may
    // not always be possible to return a true LHS & RHS.  In this case, just
    // return the value we legalized, returned in the LHS
    if (Tmp2.getNode() == 0) {
      Result = Tmp1;
      break;
    }

    switch (TLI.getOperationAction(ISD::SETCC, Tmp1.getValueType())) {
    default: assert(0 && "Cannot handle this action for SETCC yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH.
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      if (isCustom) {
        Tmp4 = TLI.LowerOperation(Result, DAG);
        if (Tmp4.getNode()) Result = Tmp4;
      }
      break;
    case TargetLowering::Promote: {
      // First step, figure out the appropriate operation to use.
      // Allow SETCC to not be supported for all legal data types
      // Mostly this targets FP
      MVT NewInTy = Node->getOperand(0).getValueType();
      MVT OldVT = NewInTy; OldVT = OldVT;

      // Scan for the appropriate larger type to use.
      while (1) {
        NewInTy = (MVT::SimpleValueType)(NewInTy.getSimpleVT()+1);

        assert(NewInTy.isInteger() == OldVT.isInteger() &&
               "Fell off of the edge of the integer world");
        assert(NewInTy.isFloatingPoint() == OldVT.isFloatingPoint() &&
               "Fell off of the edge of the floating point world");

        // If the target supports SETCC of this type, use it.
        if (TLI.isOperationLegalOrCustom(ISD::SETCC, NewInTy))
          break;
      }
      if (NewInTy.isInteger())
        assert(0 && "Cannot promote Legal Integer SETCC yet");
      else {
        Tmp1 = DAG.getNode(ISD::FP_EXTEND, dl, NewInTy, Tmp1);
        Tmp2 = DAG.getNode(ISD::FP_EXTEND, dl, NewInTy, Tmp2);
      }
      Tmp1 = LegalizeOp(Tmp1);
      Tmp2 = LegalizeOp(Tmp2);
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      Result = LegalizeOp(Result);
      break;
    }
    case TargetLowering::Expand:
      // Expand a setcc node into a select_cc of the same condition, lhs, and
      // rhs that selects between const 1 (true) and const 0 (false).
      MVT VT = Node->getValueType(0);
      Result = DAG.getNode(ISD::SELECT_CC, dl, VT, Tmp1, Tmp2,
                           DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                           Tmp3);
      break;
    }
    break;

    // Binary operators
  case ISD::FCOPYSIGN:  // FCOPYSIGN does not require LHS/RHS to match type!
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the RHS.

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "Operation not supported");
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Legal: break;
    case TargetLowering::Expand: {
      assert((Tmp2.getValueType() == MVT::f32 ||
              Tmp2.getValueType() == MVT::f64) &&
              "Ugly special-cased code!");
      // Get the sign bit of the RHS.
      SDValue SignBit;
      MVT IVT = Tmp2.getValueType() == MVT::f64 ? MVT::i64 : MVT::i32;
      if (isTypeLegal(IVT)) {
        SignBit = DAG.getNode(ISD::BIT_CONVERT, dl, IVT, Tmp2);
      } else {
        assert(isTypeLegal(TLI.getPointerTy()) &&
               (TLI.getPointerTy() == MVT::i32 || 
                TLI.getPointerTy() == MVT::i64) &&
               "Legal type for load?!");
        SDValue StackPtr = DAG.CreateStackTemporary(Tmp2.getValueType());
        SDValue StorePtr = StackPtr, LoadPtr = StackPtr;
        SDValue Ch =
            DAG.getStore(DAG.getEntryNode(), dl, Tmp2, StorePtr, NULL, 0);
        if (Tmp2.getValueType() == MVT::f64 && TLI.isLittleEndian())
          LoadPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(),
                                LoadPtr, DAG.getIntPtrConstant(4));
        SignBit = DAG.getExtLoad(ISD::SEXTLOAD, dl, TLI.getPointerTy(),
                                 Ch, LoadPtr, NULL, 0, MVT::i32);
      }
      SignBit =
          DAG.getSetCC(dl, TLI.getSetCCResultType(SignBit.getValueType()),
                       SignBit, DAG.getConstant(0, SignBit.getValueType()),
                       ISD::SETLT);
      // Get the absolute value of the result.
      SDValue AbsVal = DAG.getNode(ISD::FABS, dl, Tmp1.getValueType(), Tmp1);
      // Select between the nabs and abs value based on the sign bit of
      // the input.
      Result = DAG.getNode(ISD::SELECT, dl, AbsVal.getValueType(), SignBit,
                           DAG.getNode(ISD::FNEG, dl, AbsVal.getValueType(),
                                       AbsVal),
                           AbsVal);
      Result = LegalizeOp(Result);
      break;
    }
    }
    break;
  case ISD::BUILD_PAIR: {
    MVT PairTy = Node->getValueType(0);
    // TODO: handle the case where the Lo and Hi operands are not of legal type
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Lo
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Hi
    switch (TLI.getOperationAction(ISD::BUILD_PAIR, PairTy)) {
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom this yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::BUILD_PAIR, dl, PairTy, Tmp1, Tmp2);
      break;
    case TargetLowering::Expand:
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, PairTy, Tmp1);
      Tmp2 = DAG.getNode(ISD::ANY_EXTEND, dl, PairTy, Tmp2);
      Tmp2 = DAG.getNode(ISD::SHL, dl, PairTy, Tmp2,
                         DAG.getConstant(PairTy.getSizeInBits()/2,
                                         TLI.getShiftAmountTy()));
      Result = DAG.getNode(ISD::OR, dl, PairTy, Tmp1, Tmp2);
      break;
    }
    break;
  }

  case ISD::UREM:
  case ISD::SREM:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Promote: assert(0 && "Cannot promote this yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand: {
      unsigned DivOpc= (Node->getOpcode() == ISD::UREM) ? ISD::UDIV : ISD::SDIV;
      bool isSigned = DivOpc == ISD::SDIV;
      MVT VT = Node->getValueType(0);

      // See if remainder can be lowered using two-result operations.
      SDVTList VTs = DAG.getVTList(VT, VT);
      if (Node->getOpcode() == ISD::SREM &&
          TLI.isOperationLegalOrCustom(ISD::SDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::SDIVREM, dl,
                                     VTs, Tmp1, Tmp2).getNode(), 1);
        break;
      }
      if (Node->getOpcode() == ISD::UREM &&
          TLI.isOperationLegalOrCustom(ISD::UDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::UDIVREM, dl,
                                     VTs, Tmp1, Tmp2).getNode(), 1);
        break;
      }

      if (VT.isInteger() &&
          TLI.getOperationAction(DivOpc, VT) == TargetLowering::Legal) {
        // X % Y -> X-X/Y*Y
        Result = DAG.getNode(DivOpc, dl, VT, Tmp1, Tmp2);
        Result = DAG.getNode(ISD::MUL, dl, VT, Result, Tmp2);
        Result = DAG.getNode(ISD::SUB, dl, VT, Tmp1, Result);
        break;
      }

      // Check to see if we have a libcall for this operator.
      RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
      switch (Node->getOpcode()) {
      default: break;
      case ISD::UREM:
      case ISD::SREM:
       if (VT == MVT::i16)
         LC = (isSigned ? RTLIB::SREM_I16  : RTLIB::UREM_I16);
       else if (VT == MVT::i32)
         LC = (isSigned ? RTLIB::SREM_I32  : RTLIB::UREM_I32);
       else if (VT == MVT::i64)
         LC = (isSigned ? RTLIB::SREM_I64  : RTLIB::UREM_I64);
       else if (VT == MVT::i128)
         LC = (isSigned ? RTLIB::SREM_I128 : RTLIB::UREM_I128);
       break;
      }

      if (LC != RTLIB::UNKNOWN_LIBCALL) {
        Result = ExpandLibCall(LC, Node, isSigned);
        break;
      }

      assert(0 && "Cannot expand this binary operator!");
      break;
    }
    }
    break;
  case ISD::VAARG: {
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));
      Result = Result.getValue(0);
      Tmp1 = Result.getValue(1);

      if (isCustom) {
        Tmp2 = TLI.LowerOperation(Result, DAG);
        if (Tmp2.getNode()) {
          Result = LegalizeOp(Tmp2);
          Tmp1 = LegalizeOp(Tmp2.getValue(1));
        }
      }
      break;
    case TargetLowering::Expand: {
      const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
      SDValue VAList = DAG.getLoad(TLI.getPointerTy(), dl, Tmp1, Tmp2, V, 0);
      // Increment the pointer, VAList, to the next vaarg
      Tmp3 = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), VAList,
                         DAG.getConstant(TLI.getTargetData()->
                                         getTypeAllocSize(VT.getTypeForMVT()),
                                         TLI.getPointerTy()));
      // Store the incremented VAList to the legalized pointer
      Tmp3 = DAG.getStore(VAList.getValue(1), dl, Tmp3, Tmp2, V, 0);
      // Load the actual argument out of the pointer VAList
      Result = DAG.getLoad(VT, dl, Tmp3, VAList, NULL, 0);
      Tmp1 = LegalizeOp(Result.getValue(1));
      Result = LegalizeOp(Result);
      break;
    }
    }
    // Since VAARG produces two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Result);
    AddLegalizedOperand(SDValue(Node, 1), Tmp1);
    return Op.getResNo() ? Tmp1 : Result;
  }
  case ISD::SADDO:
  case ISD::SSUBO: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // FALLTHROUGH
    case TargetLowering::Legal: {
      SDValue LHS = LegalizeOp(Node->getOperand(0));
      SDValue RHS = LegalizeOp(Node->getOperand(1));

      SDValue Sum = DAG.getNode(Node->getOpcode() == ISD::SADDO ?
                                ISD::ADD : ISD::SUB, dl, LHS.getValueType(),
                                LHS, RHS);
      MVT OType = Node->getValueType(1);

      SDValue Zero = DAG.getConstant(0, LHS.getValueType());

      //   LHSSign -> LHS >= 0
      //   RHSSign -> RHS >= 0
      //   SumSign -> Sum >= 0
      //
      //   Add:
      //   Overflow -> (LHSSign == RHSSign) && (LHSSign != SumSign)
      //   Sub:
      //   Overflow -> (LHSSign != RHSSign) && (LHSSign != SumSign)
      //
      SDValue LHSSign = DAG.getSetCC(dl, OType, LHS, Zero, ISD::SETGE);
      SDValue RHSSign = DAG.getSetCC(dl, OType, RHS, Zero, ISD::SETGE);
      SDValue SignsMatch = DAG.getSetCC(dl, OType, LHSSign, RHSSign,
                                        Node->getOpcode() == ISD::SADDO ?
                                        ISD::SETEQ : ISD::SETNE);

      SDValue SumSign = DAG.getSetCC(dl, OType, Sum, Zero, ISD::SETGE);
      SDValue SumSignNE = DAG.getSetCC(dl, OType, LHSSign, SumSign, ISD::SETNE);

      SDValue Cmp = DAG.getNode(ISD::AND, dl, OType, SignsMatch, SumSignNE);

      MVT ValueVTs[] = { LHS.getValueType(), OType };
      SDValue Ops[] = { Sum, Cmp };

      Result = DAG.getNode(ISD::MERGE_VALUES, dl,
                           DAG.getVTList(&ValueVTs[0], 2),
                           &Ops[0], 2);
      SDNode *RNode = Result.getNode();
      DAG.ReplaceAllUsesWith(Node, RNode);
      break;
    }
    }

    break;
  }
  case ISD::UADDO:
  case ISD::USUBO: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // FALLTHROUGH
    case TargetLowering::Legal: {
      SDValue LHS = LegalizeOp(Node->getOperand(0));
      SDValue RHS = LegalizeOp(Node->getOperand(1));

      SDValue Sum = DAG.getNode(Node->getOpcode() == ISD::UADDO ?
                                ISD::ADD : ISD::SUB, dl, LHS.getValueType(),
                                LHS, RHS);
      MVT OType = Node->getValueType(1);
      SDValue Cmp = DAG.getSetCC(dl, OType, Sum, LHS,
                                 Node->getOpcode () == ISD::UADDO ?
                                 ISD::SETULT : ISD::SETUGT);

      MVT ValueVTs[] = { LHS.getValueType(), OType };
      SDValue Ops[] = { Sum, Cmp };

      Result = DAG.getNode(ISD::MERGE_VALUES, dl,
                           DAG.getVTList(&ValueVTs[0], 2),
                           &Ops[0], 2);
      SDNode *RNode = Result.getNode();
      DAG.ReplaceAllUsesWith(Node, RNode);
      break;
    }
    }

    break;
  }
  }

  assert(Result.getValueType() == Op.getValueType() &&
         "Bad legalization!");

  // Make sure that the generated code is itself legal.
  if (Result != Op)
    Result = LegalizeOp(Result);

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

SDValue SelectionDAGLegalize::ExpandExtractFromVectorThroughStack(SDValue Op) {
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  // Store the value to a temporary stack slot, then LOAD the returned part.
  SDValue StackPtr = DAG.CreateStackTemporary(Vec.getValueType());
  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, Vec, StackPtr, NULL, 0);

  // Add the offset to the index.
  unsigned EltSize =
      Vec.getValueType().getVectorElementType().getSizeInBits()/8;
  Idx = DAG.getNode(ISD::MUL, dl, Idx.getValueType(), Idx,
                    DAG.getConstant(EltSize, Idx.getValueType()));

  if (Idx.getValueType().bitsGT(TLI.getPointerTy()))
    Idx = DAG.getNode(ISD::TRUNCATE, dl, TLI.getPointerTy(), Idx);
  else
    Idx = DAG.getNode(ISD::ZERO_EXTEND, dl, TLI.getPointerTy(), Idx);

  StackPtr = DAG.getNode(ISD::ADD, dl, Idx.getValueType(), Idx, StackPtr);

  return DAG.getLoad(Op.getValueType(), dl, Ch, StackPtr, NULL, 0);
}

/// LegalizeSetCCOperands - Attempts to create a legal LHS and RHS for a SETCC
/// with condition CC on the current target.  This usually involves legalizing
/// or promoting the arguments.  In the case where LHS and RHS must be expanded,
/// there may be no choice but to create a new SetCC node to represent the
/// legalized value of setcc lhs, rhs.  In this case, the value is returned in
/// LHS, and the SDValue returned in RHS has a nil SDNode value.
void SelectionDAGLegalize::LegalizeSetCCOperands(SDValue &LHS,
                                                 SDValue &RHS,
                                                 SDValue &CC,
                                                 DebugLoc dl) {
  LHS = LegalizeOp(LHS);
  RHS = LegalizeOp(RHS);
}

/// LegalizeSetCCCondCode - Legalize a SETCC with given LHS and RHS and
/// condition code CC on the current target. This routine assumes LHS and rHS
/// have already been legalized by LegalizeSetCCOperands. It expands SETCC with
/// illegal condition code into AND / OR of multiple SETCC values.
void SelectionDAGLegalize::LegalizeSetCCCondCode(MVT VT,
                                                 SDValue &LHS, SDValue &RHS,
                                                 SDValue &CC,
                                                 DebugLoc dl) {
  MVT OpVT = LHS.getValueType();
  ISD::CondCode CCCode = cast<CondCodeSDNode>(CC)->get();
  switch (TLI.getCondCodeAction(CCCode, OpVT)) {
  default: assert(0 && "Unknown condition code action!");
  case TargetLowering::Legal:
    // Nothing to do.
    break;
  case TargetLowering::Expand: {
    ISD::CondCode CC1 = ISD::SETCC_INVALID, CC2 = ISD::SETCC_INVALID;
    unsigned Opc = 0;
    switch (CCCode) {
    default: assert(0 && "Don't know how to expand this condition!"); abort();
    case ISD::SETOEQ: CC1 = ISD::SETEQ; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOGT: CC1 = ISD::SETGT; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOGE: CC1 = ISD::SETGE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOLT: CC1 = ISD::SETLT; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOLE: CC1 = ISD::SETLE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETONE: CC1 = ISD::SETNE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETUEQ: CC1 = ISD::SETEQ; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUGT: CC1 = ISD::SETGT; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUGE: CC1 = ISD::SETGE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETULT: CC1 = ISD::SETLT; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETULE: CC1 = ISD::SETLE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUNE: CC1 = ISD::SETNE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    // FIXME: Implement more expansions.
    }

    SDValue SetCC1 = DAG.getSetCC(dl, VT, LHS, RHS, CC1);
    SDValue SetCC2 = DAG.getSetCC(dl, VT, LHS, RHS, CC2);
    LHS = DAG.getNode(Opc, dl, VT, SetCC1, SetCC2);
    RHS = SDValue();
    CC  = SDValue();
    break;
  }
  }
}

/// EmitStackConvert - Emit a store/load combination to the stack.  This stores
/// SrcOp to a stack slot of type SlotVT, truncating it if needed.  It then does
/// a load from the stack slot to DestVT, extending it if needed.
/// The resultant code need not be legal.
SDValue SelectionDAGLegalize::EmitStackConvert(SDValue SrcOp,
                                               MVT SlotVT,
                                               MVT DestVT,
                                               DebugLoc dl) {
  // Create the stack frame object.
  unsigned SrcAlign =
    TLI.getTargetData()->getPrefTypeAlignment(SrcOp.getValueType().
                                              getTypeForMVT());
  SDValue FIPtr = DAG.CreateStackTemporary(SlotVT, SrcAlign);

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(FIPtr);
  int SPFI = StackPtrFI->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(SPFI);

  unsigned SrcSize = SrcOp.getValueType().getSizeInBits();
  unsigned SlotSize = SlotVT.getSizeInBits();
  unsigned DestSize = DestVT.getSizeInBits();
  unsigned DestAlign =
    TLI.getTargetData()->getPrefTypeAlignment(DestVT.getTypeForMVT());

  // Emit a store to the stack slot.  Use a truncstore if the input value is
  // later than DestVT.
  SDValue Store;

  if (SrcSize > SlotSize)
    Store = DAG.getTruncStore(DAG.getEntryNode(), dl, SrcOp, FIPtr,
                              SV, 0, SlotVT, false, SrcAlign);
  else {
    assert(SrcSize == SlotSize && "Invalid store");
    Store = DAG.getStore(DAG.getEntryNode(), dl, SrcOp, FIPtr,
                         SV, 0, false, SrcAlign);
  }

  // Result is a load from the stack slot.
  if (SlotSize == DestSize)
    return DAG.getLoad(DestVT, dl, Store, FIPtr, SV, 0, false, DestAlign);

  assert(SlotSize < DestSize && "Unknown extension!");
  return DAG.getExtLoad(ISD::EXTLOAD, dl, DestVT, Store, FIPtr, SV, 0, SlotVT,
                        false, DestAlign);
}

SDValue SelectionDAGLegalize::ExpandSCALAR_TO_VECTOR(SDNode *Node) {
  DebugLoc dl = Node->getDebugLoc();
  // Create a vector sized/aligned stack slot, store the value to element #0,
  // then load the whole vector back out.
  SDValue StackPtr = DAG.CreateStackTemporary(Node->getValueType(0));

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(StackPtr);
  int SPFI = StackPtrFI->getIndex();

  SDValue Ch = DAG.getTruncStore(DAG.getEntryNode(), dl, Node->getOperand(0),
                                 StackPtr,
                                 PseudoSourceValue::getFixedStack(SPFI), 0,
                                 Node->getValueType(0).getVectorElementType());
  return DAG.getLoad(Node->getValueType(0), dl, Ch, StackPtr,
                     PseudoSourceValue::getFixedStack(SPFI), 0);
}


/// ExpandBUILD_VECTOR - Expand a BUILD_VECTOR node on targets that don't
/// support the operation, but do support the resultant vector type.
SDValue SelectionDAGLegalize::ExpandBUILD_VECTOR(SDNode *Node) {
  unsigned NumElems = Node->getNumOperands();
  SDValue SplatValue = Node->getOperand(0);
  DebugLoc dl = Node->getDebugLoc();
  MVT VT = Node->getValueType(0);
  MVT OpVT = SplatValue.getValueType();
  MVT EltVT = VT.getVectorElementType();

  // If the only non-undef value is the low element, turn this into a
  // SCALAR_TO_VECTOR node.  If this is { X, X, X, X }, determine X.
  bool isOnlyLowElement = true;

  // FIXME: it would be far nicer to change this into map<SDValue,uint64_t>
  // and use a bitmask instead of a list of elements.
  // FIXME: this doesn't treat <0, u, 0, u> for example, as a splat.
  std::map<SDValue, std::vector<unsigned> > Values;
  Values[SplatValue].push_back(0);
  bool isConstant = true;
  if (!isa<ConstantFPSDNode>(SplatValue) && !isa<ConstantSDNode>(SplatValue) &&
      SplatValue.getOpcode() != ISD::UNDEF)
    isConstant = false;

  for (unsigned i = 1; i < NumElems; ++i) {
    SDValue V = Node->getOperand(i);
    Values[V].push_back(i);
    if (V.getOpcode() != ISD::UNDEF)
      isOnlyLowElement = false;
    if (SplatValue != V)
      SplatValue = SDValue(0, 0);

    // If this isn't a constant element or an undef, we can't use a constant
    // pool load.
    if (!isa<ConstantFPSDNode>(V) && !isa<ConstantSDNode>(V) &&
        V.getOpcode() != ISD::UNDEF)
      isConstant = false;
  }

  if (isOnlyLowElement) {
    // If the low element is an undef too, then this whole things is an undef.
    if (Node->getOperand(0).getOpcode() == ISD::UNDEF)
      return DAG.getUNDEF(VT);
    // Otherwise, turn this into a scalar_to_vector node.
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Node->getOperand(0));
  }

  // If all elements are constants, create a load from the constant pool.
  if (isConstant) {
    std::vector<Constant*> CV;
    for (unsigned i = 0, e = NumElems; i != e; ++i) {
      if (ConstantFPSDNode *V =
          dyn_cast<ConstantFPSDNode>(Node->getOperand(i))) {
        CV.push_back(const_cast<ConstantFP *>(V->getConstantFPValue()));
      } else if (ConstantSDNode *V =
                 dyn_cast<ConstantSDNode>(Node->getOperand(i))) {
        CV.push_back(const_cast<ConstantInt *>(V->getConstantIntValue()));
      } else {
        assert(Node->getOperand(i).getOpcode() == ISD::UNDEF);
        const Type *OpNTy = OpVT.getTypeForMVT();
        CV.push_back(UndefValue::get(OpNTy));
      }
    }
    Constant *CP = ConstantVector::get(CV);
    SDValue CPIdx = DAG.getConstantPool(CP, TLI.getPointerTy());
    unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
    return DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                       PseudoSourceValue::getConstantPool(), 0,
                       false, Alignment);
  }

  if (SplatValue.getNode()) {   // Splat of one value?
    // Build the shuffle constant vector: <0, 0, 0, 0>
    SmallVector<int, 8> ZeroVec(NumElems, 0);

    // If the target supports VECTOR_SHUFFLE and this shuffle mask, use it.
    if (TLI.isShuffleMaskLegal(ZeroVec, Node->getValueType(0))) {
      // Get the splatted value into the low element of a vector register.
      SDValue LowValVec =
        DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, SplatValue);

      // Return shuffle(LowValVec, undef, <0,0,0,0>)
      return DAG.getVectorShuffle(VT, dl, LowValVec, DAG.getUNDEF(VT),
                                  &ZeroVec[0]);
    }
  }

  // If there are only two unique elements, we may be able to turn this into a
  // vector shuffle.
  if (Values.size() == 2) {
    // Get the two values in deterministic order.
    SDValue Val1 = Node->getOperand(1);
    SDValue Val2;
    std::map<SDValue, std::vector<unsigned> >::iterator MI = Values.begin();
    if (MI->first != Val1)
      Val2 = MI->first;
    else
      Val2 = (++MI)->first;

    // If Val1 is an undef, make sure it ends up as Val2, to ensure that our
    // vector shuffle has the undef vector on the RHS.
    if (Val1.getOpcode() == ISD::UNDEF)
      std::swap(Val1, Val2);

    // Build the shuffle constant vector: e.g. <0, 4, 0, 4>
    SmallVector<int, 8> ShuffleMask(NumElems, -1);

    // Set elements of the shuffle mask for Val1.
    std::vector<unsigned> &Val1Elts = Values[Val1];
    for (unsigned i = 0, e = Val1Elts.size(); i != e; ++i)
      ShuffleMask[Val1Elts[i]] = 0;

    // Set elements of the shuffle mask for Val2.
    std::vector<unsigned> &Val2Elts = Values[Val2];
    for (unsigned i = 0, e = Val2Elts.size(); i != e; ++i)
      if (Val2.getOpcode() != ISD::UNDEF)
        ShuffleMask[Val2Elts[i]] = NumElems;

    // If the target supports SCALAR_TO_VECTOR and this shuffle mask, use it.
    if (TLI.isOperationLegalOrCustom(ISD::SCALAR_TO_VECTOR, VT) &&
        TLI.isShuffleMaskLegal(ShuffleMask, VT)) {
      Val1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Val1);
      Val2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Val2);
      return DAG.getVectorShuffle(VT, dl, Val1, Val2, &ShuffleMask[0]);
    }
  }

  // Otherwise, we can't handle this case efficiently.  Allocate a sufficiently
  // aligned object on the stack, store each element into it, then load
  // the result as a vector.
  // Create the stack frame object.
  SDValue FIPtr = DAG.CreateStackTemporary(VT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(FI);

  // Emit a store of each element to the stack slot.
  SmallVector<SDValue, 8> Stores;
  unsigned TypeByteSize = OpVT.getSizeInBits() / 8;
  // Store (in the right endianness) the elements to memory.
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    // Ignore undef elements.
    if (Node->getOperand(i).getOpcode() == ISD::UNDEF) continue;

    unsigned Offset = TypeByteSize*i;

    SDValue Idx = DAG.getConstant(Offset, FIPtr.getValueType());
    Idx = DAG.getNode(ISD::ADD, dl, FIPtr.getValueType(), FIPtr, Idx);

    Stores.push_back(DAG.getStore(DAG.getEntryNode(), dl, Node->getOperand(i),
                                  Idx, SV, Offset));
  }

  SDValue StoreChain;
  if (!Stores.empty())    // Not all undef elements?
    StoreChain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                             &Stores[0], Stores.size());
  else
    StoreChain = DAG.getEntryNode();

  // Result is a load from the stack slot.
  return DAG.getLoad(VT, dl, StoreChain, FIPtr, SV, 0);
}

// ExpandLibCall - Expand a node into a call to a libcall.  If the result value
// does not fit into a register, return the lo part and set the hi part to the
// by-reg argument.  If it does fit into a single register, return the result
// and leave the Hi part unset.
SDValue SelectionDAGLegalize::ExpandLibCall(RTLIB::Libcall LC, SDNode *Node,
                                            bool isSigned) {
  assert(!IsLegalizingCall && "Cannot overlap legalization of calls!");
  // The input chain to this libcall is the entry node of the function.
  // Legalizing the call will automatically add the previous call to the
  // dependence.
  SDValue InChain = DAG.getEntryNode();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    MVT ArgVT = Node->getOperand(i).getValueType();
    const Type *ArgTy = ArgVT.getTypeForMVT();
    Entry.Node = Node->getOperand(i); Entry.Ty = ArgTy;
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                         TLI.getPointerTy());

  // Splice the libcall in wherever FindInputOutputChains tells us to.
  const Type *RetTy = Node->getValueType(0).getTypeForMVT();
  std::pair<SDValue, SDValue> CallInfo =
    TLI.LowerCallTo(InChain, RetTy, isSigned, !isSigned, false, false,
                    CallingConv::C, false, Callee, Args, DAG,
                    Node->getDebugLoc());

  // Legalize the call sequence, starting with the chain.  This will advance
  // the LastCALLSEQ_END to the legalized version of the CALLSEQ_END node that
  // was added by LowerCallTo (guaranteeing proper serialization of calls).
  LegalizeOp(CallInfo.second);
  return CallInfo.first;
}

SDValue SelectionDAGLegalize::ExpandFPLibCall(SDNode* Node,
                                              RTLIB::Libcall Call_F32,
                                              RTLIB::Libcall Call_F64,
                                              RTLIB::Libcall Call_F80,
                                              RTLIB::Libcall Call_PPCF128) {
  RTLIB::Libcall LC;
  switch (Node->getValueType(0).getSimpleVT()) {
  default: assert(0 && "Unexpected request for libcall!");
  case MVT::f32: LC = Call_F32; break;
  case MVT::f64: LC = Call_F64; break;
  case MVT::f80: LC = Call_F80; break;
  case MVT::ppcf128: LC = Call_PPCF128; break;
  }
  return ExpandLibCall(LC, Node, false);
}

SDValue SelectionDAGLegalize::ExpandIntLibCall(SDNode* Node, bool isSigned,
                                               RTLIB::Libcall Call_I16,
                                               RTLIB::Libcall Call_I32,
                                               RTLIB::Libcall Call_I64,
                                               RTLIB::Libcall Call_I128) {
  RTLIB::Libcall LC;
  switch (Node->getValueType(0).getSimpleVT()) {
  default: assert(0 && "Unexpected request for libcall!");
  case MVT::i16: LC = Call_I16; break;
  case MVT::i32: LC = Call_I32; break;
  case MVT::i64: LC = Call_I64; break;
  case MVT::i128: LC = Call_I128; break;
  }
  return ExpandLibCall(LC, Node, isSigned);
}

/// ExpandLegalINT_TO_FP - This function is responsible for legalizing a
/// INT_TO_FP operation of the specified operand when the target requests that
/// we expand it.  At this point, we know that the result and operand types are
/// legal for the target.
SDValue SelectionDAGLegalize::ExpandLegalINT_TO_FP(bool isSigned,
                                                   SDValue Op0,
                                                   MVT DestVT,
                                                   DebugLoc dl) {
  if (Op0.getValueType() == MVT::i32) {
    // simple 32-bit [signed|unsigned] integer to float/double expansion

    // Get the stack frame index of a 8 byte buffer.
    SDValue StackSlot = DAG.CreateStackTemporary(MVT::f64);

    // word offset constant for Hi/Lo address computation
    SDValue WordOff = DAG.getConstant(sizeof(int), TLI.getPointerTy());
    // set up Hi and Lo (into buffer) address based on endian
    SDValue Hi = StackSlot;
    SDValue Lo = DAG.getNode(ISD::ADD, dl,
                             TLI.getPointerTy(), StackSlot, WordOff);
    if (TLI.isLittleEndian())
      std::swap(Hi, Lo);

    // if signed map to unsigned space
    SDValue Op0Mapped;
    if (isSigned) {
      // constant used to invert sign bit (signed to unsigned mapping)
      SDValue SignBit = DAG.getConstant(0x80000000u, MVT::i32);
      Op0Mapped = DAG.getNode(ISD::XOR, dl, MVT::i32, Op0, SignBit);
    } else {
      Op0Mapped = Op0;
    }
    // store the lo of the constructed double - based on integer input
    SDValue Store1 = DAG.getStore(DAG.getEntryNode(), dl,
                                  Op0Mapped, Lo, NULL, 0);
    // initial hi portion of constructed double
    SDValue InitialHi = DAG.getConstant(0x43300000u, MVT::i32);
    // store the hi of the constructed double - biased exponent
    SDValue Store2=DAG.getStore(Store1, dl, InitialHi, Hi, NULL, 0);
    // load the constructed double
    SDValue Load = DAG.getLoad(MVT::f64, dl, Store2, StackSlot, NULL, 0);
    // FP constant to bias correct the final result
    SDValue Bias = DAG.getConstantFP(isSigned ?
                                     BitsToDouble(0x4330000080000000ULL) :
                                     BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    // subtract the bias
    SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::f64, Load, Bias);
    // final result
    SDValue Result;
    // handle final rounding
    if (DestVT == MVT::f64) {
      // do nothing
      Result = Sub;
    } else if (DestVT.bitsLT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_ROUND, dl, DestVT, Sub,
                           DAG.getIntPtrConstant(0));
    } else if (DestVT.bitsGT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_EXTEND, dl, DestVT, Sub);
    }
    return Result;
  }
  assert(!isSigned && "Legalize cannot Expand SINT_TO_FP for i64 yet");
  SDValue Tmp1 = DAG.getNode(ISD::SINT_TO_FP, dl, DestVT, Op0);

  SDValue SignSet = DAG.getSetCC(dl, TLI.getSetCCResultType(Op0.getValueType()),
                                 Op0, DAG.getConstant(0, Op0.getValueType()),
                                 ISD::SETLT);
  SDValue Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
  SDValue CstOffset = DAG.getNode(ISD::SELECT, dl, Zero.getValueType(),
                                    SignSet, Four, Zero);

  // If the sign bit of the integer is set, the large number will be treated
  // as a negative number.  To counteract this, the dynamic code adds an
  // offset depending on the data type.
  uint64_t FF;
  switch (Op0.getValueType().getSimpleVT()) {
  default: assert(0 && "Unsupported integer type!");
  case MVT::i8 : FF = 0x43800000ULL; break;  // 2^8  (as a float)
  case MVT::i16: FF = 0x47800000ULL; break;  // 2^16 (as a float)
  case MVT::i32: FF = 0x4F800000ULL; break;  // 2^32 (as a float)
  case MVT::i64: FF = 0x5F800000ULL; break;  // 2^64 (as a float)
  }
  if (TLI.isLittleEndian()) FF <<= 32;
  Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

  SDValue CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  CPIdx = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), CPIdx, CstOffset);
  Alignment = std::min(Alignment, 4u);
  SDValue FudgeInReg;
  if (DestVT == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, dl, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0,
                             false, Alignment);
  else {
    FudgeInReg =
      LegalizeOp(DAG.getExtLoad(ISD::EXTLOAD, dl, DestVT,
                                DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                MVT::f32, false, Alignment));
  }

  return DAG.getNode(ISD::FADD, dl, DestVT, Tmp1, FudgeInReg);
}

/// PromoteLegalINT_TO_FP - This function is responsible for legalizing a
/// *INT_TO_FP operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal UINT_TO_FP or SINT_TO_FP
/// operation that takes a larger input.
SDValue SelectionDAGLegalize::PromoteLegalINT_TO_FP(SDValue LegalOp,
                                                    MVT DestVT,
                                                    bool isSigned,
                                                    DebugLoc dl) {
  // First step, figure out the appropriate *INT_TO_FP operation to use.
  MVT NewInTy = LegalOp.getValueType();

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewInTy = (MVT::SimpleValueType)(NewInTy.getSimpleVT()+1);
    assert(NewInTy.isInteger() && "Ran out of possibilities!");

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

  // Okay, we found the operation and type to use.  Zero extend our input to the
  // desired type then run the operation on it.
  return DAG.getNode(OpToUse, dl, DestVT,
                     DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND,
                                 dl, NewInTy, LegalOp));
}

/// PromoteLegalFP_TO_INT - This function is responsible for legalizing a
/// FP_TO_*INT operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal FP_TO_UINT or FP_TO_SINT
/// operation that returns a larger result.
SDValue SelectionDAGLegalize::PromoteLegalFP_TO_INT(SDValue LegalOp,
                                                    MVT DestVT,
                                                    bool isSigned,
                                                    DebugLoc dl) {
  // First step, figure out the appropriate FP_TO*INT operation to use.
  MVT NewOutTy = DestVT;

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewOutTy = (MVT::SimpleValueType)(NewOutTy.getSimpleVT()+1);
    assert(NewOutTy.isInteger() && "Ran out of possibilities!");

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


  // Okay, we found the operation and type to use.
  SDValue Operation = DAG.getNode(OpToUse, dl, NewOutTy, LegalOp);

  // If the operation produces an invalid type, it must be custom lowered.  Use
  // the target lowering hooks to expand it.  Just keep the low part of the
  // expanded operation, we know that we're truncating anyway.
  if (getTypeAction(NewOutTy) == Expand) {
    SmallVector<SDValue, 2> Results;
    TLI.ReplaceNodeResults(Operation.getNode(), Results, DAG);
    assert(Results.size() == 1 && "Incorrect FP_TO_XINT lowering!");
    Operation = Results[0];
  }

  // Truncate the result of the extended FP_TO_*INT operation to the desired
  // size.
  return DAG.getNode(ISD::TRUNCATE, dl, DestVT, Operation);
}

/// ExpandBSWAP - Open code the operations for BSWAP of the specified operation.
///
SDValue SelectionDAGLegalize::ExpandBSWAP(SDValue Op, DebugLoc dl) {
  MVT VT = Op.getValueType();
  MVT SHVT = TLI.getShiftAmountTy();
  SDValue Tmp1, Tmp2, Tmp3, Tmp4, Tmp5, Tmp6, Tmp7, Tmp8;
  switch (VT.getSimpleVT()) {
  default: assert(0 && "Unhandled Expand type in BSWAP!"); abort();
  case MVT::i16:
    Tmp2 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    return DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
  case MVT::i32:
    Tmp4 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::AND, dl, VT, Tmp3, DAG.getConstant(0xFF0000, VT));
    Tmp2 = DAG.getNode(ISD::AND, dl, VT, Tmp2, DAG.getConstant(0xFF00, VT));
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp1);
    return DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp2);
  case MVT::i64:
    Tmp8 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(40, SHVT));
    Tmp6 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp5 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp4 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp3 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(40, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::AND, dl, VT, Tmp7, DAG.getConstant(255ULL<<48, VT));
    Tmp6 = DAG.getNode(ISD::AND, dl, VT, Tmp6, DAG.getConstant(255ULL<<40, VT));
    Tmp5 = DAG.getNode(ISD::AND, dl, VT, Tmp5, DAG.getConstant(255ULL<<32, VT));
    Tmp4 = DAG.getNode(ISD::AND, dl, VT, Tmp4, DAG.getConstant(255ULL<<24, VT));
    Tmp3 = DAG.getNode(ISD::AND, dl, VT, Tmp3, DAG.getConstant(255ULL<<16, VT));
    Tmp2 = DAG.getNode(ISD::AND, dl, VT, Tmp2, DAG.getConstant(255ULL<<8 , VT));
    Tmp8 = DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp7);
    Tmp6 = DAG.getNode(ISD::OR, dl, VT, Tmp6, Tmp5);
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp1);
    Tmp8 = DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp6);
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp2);
    return DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp4);
  }
}

/// ExpandBitCount - Expand the specified bitcount instruction into operations.
///
SDValue SelectionDAGLegalize::ExpandBitCount(unsigned Opc, SDValue Op,
                                             DebugLoc dl) {
  switch (Opc) {
  default: assert(0 && "Cannot expand this yet!");
  case ISD::CTPOP: {
    static const uint64_t mask[6] = {
      0x5555555555555555ULL, 0x3333333333333333ULL,
      0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
      0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
    };
    MVT VT = Op.getValueType();
    MVT ShVT = TLI.getShiftAmountTy();
    unsigned len = VT.getSizeInBits();
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      //x = (x & mask[i][len/8]) + (x >> (1 << i) & mask[i][len/8])
      unsigned EltSize = VT.isVector() ?
        VT.getVectorElementType().getSizeInBits() : len;
      SDValue Tmp2 = DAG.getConstant(APInt(EltSize, mask[i]), VT);
      SDValue Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::ADD, dl, VT,
                       DAG.getNode(ISD::AND, dl, VT, Op, Tmp2),
                       DAG.getNode(ISD::AND, dl, VT,
                                   DAG.getNode(ISD::SRL, dl, VT, Op, Tmp3),
                                   Tmp2));
    }
    return Op;
  }
  case ISD::CTLZ: {
    // for now, we do this:
    // x = x | (x >> 1);
    // x = x | (x >> 2);
    // ...
    // x = x | (x >>16);
    // x = x | (x >>32); // for 64-bit input
    // return popcount(~x);
    //
    // but see also: http://www.hackersdelight.org/HDcode/nlz.cc
    MVT VT = Op.getValueType();
    MVT ShVT = TLI.getShiftAmountTy();
    unsigned len = VT.getSizeInBits();
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      SDValue Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::OR, dl, VT, Op,
                       DAG.getNode(ISD::SRL, dl, VT, Op, Tmp3));
    }
    Op = DAG.getNOT(dl, Op, VT);
    return DAG.getNode(ISD::CTPOP, dl, VT, Op);
  }
  case ISD::CTTZ: {
    // for now, we use: { return popcount(~x & (x - 1)); }
    // unless the target has ctlz but not ctpop, in which case we use:
    // { return 32 - nlz(~x & (x-1)); }
    // see also http://www.hackersdelight.org/HDcode/ntz.cc
    MVT VT = Op.getValueType();
    SDValue Tmp3 = DAG.getNode(ISD::AND, dl, VT,
                               DAG.getNOT(dl, Op, VT),
                               DAG.getNode(ISD::SUB, dl, VT, Op,
                                           DAG.getConstant(1, VT)));
    // If ISD::CTLZ is legal and CTPOP isn't, then do that instead.
    if (!TLI.isOperationLegalOrCustom(ISD::CTPOP, VT) &&
        TLI.isOperationLegalOrCustom(ISD::CTLZ, VT))
      return DAG.getNode(ISD::SUB, dl, VT,
                         DAG.getConstant(VT.getSizeInBits(), VT),
                         DAG.getNode(ISD::CTLZ, dl, VT, Tmp3));
    return DAG.getNode(ISD::CTPOP, dl, VT, Tmp3);
  }
  }
}

void SelectionDAGLegalize::ExpandNode(SDNode *Node,
                                      SmallVectorImpl<SDValue> &Results) {
  DebugLoc dl = Node->getDebugLoc();
  SDValue Tmp1, Tmp2, Tmp3;
  switch (Node->getOpcode()) {
  case ISD::CTPOP:
  case ISD::CTLZ:
  case ISD::CTTZ:
    Tmp1 = ExpandBitCount(Node->getOpcode(), Node->getOperand(0), dl);
    Results.push_back(Tmp1);
    break;
  case ISD::BSWAP:
    Results.push_back(ExpandBSWAP(Node->getOperand(0), dl));
    break;
  case ISD::FRAMEADDR:
  case ISD::RETURNADDR:
  case ISD::FRAME_TO_ARGS_OFFSET:
    Results.push_back(DAG.getConstant(0, Node->getValueType(0)));
    break;
  case ISD::FLT_ROUNDS_:
    Results.push_back(DAG.getConstant(1, Node->getValueType(0)));
    break;
  case ISD::EH_RETURN:
  case ISD::DECLARE:
  case ISD::DBG_LABEL:
  case ISD::EH_LABEL:
  case ISD::PREFETCH:
  case ISD::MEMBARRIER:
  case ISD::VAEND:
    Results.push_back(Node->getOperand(0));
    break;
  case ISD::MERGE_VALUES:
    for (unsigned i = 0; i < Node->getNumValues(); i++)
      Results.push_back(Node->getOperand(i));
    break;
  case ISD::UNDEF: {
    MVT VT = Node->getValueType(0);
    if (VT.isInteger())
      Results.push_back(DAG.getConstant(0, VT));
    else if (VT.isFloatingPoint())
      Results.push_back(DAG.getConstantFP(0, VT));
    else
      assert(0 && "Unknown value type!");
    break;
  }
  case ISD::TRAP: {
    // If this operation is not supported, lower it to 'abort()' call
    TargetLowering::ArgListTy Args;
    std::pair<SDValue, SDValue> CallResult =
      TLI.LowerCallTo(Node->getOperand(0), Type::VoidTy,
                      false, false, false, false, CallingConv::C, false,
                      DAG.getExternalSymbol("abort", TLI.getPointerTy()),
                      Args, DAG, dl);
    Results.push_back(CallResult.second);
    break;
  }
  case ISD::FP_ROUND:
  case ISD::BIT_CONVERT:
    Tmp1 = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                            Node->getValueType(0), dl);
    Results.push_back(Tmp1);
    break;
  case ISD::FP_EXTEND:
    Tmp1 = EmitStackConvert(Node->getOperand(0),
                            Node->getOperand(0).getValueType(),
                            Node->getValueType(0), dl);
    Results.push_back(Tmp1);
    break;
  case ISD::SIGN_EXTEND_INREG: {
    // NOTE: we could fall back on load/store here too for targets without
    // SAR.  However, it is doubtful that any exist.
    MVT ExtraVT = cast<VTSDNode>(Node->getOperand(1))->getVT();
    unsigned BitsDiff = Node->getValueType(0).getSizeInBits() -
                        ExtraVT.getSizeInBits();
    SDValue ShiftCst = DAG.getConstant(BitsDiff, TLI.getShiftAmountTy());
    Tmp1 = DAG.getNode(ISD::SHL, dl, Node->getValueType(0),
                       Node->getOperand(0), ShiftCst);
    Tmp1 = DAG.getNode(ISD::SRA, dl, Node->getValueType(0), Tmp1, ShiftCst);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::FP_ROUND_INREG: {
    // The only way we can lower this is to turn it into a TRUNCSTORE,
    // EXTLOAD pair, targetting a temporary location (a stack slot).

    // NOTE: there is a choice here between constantly creating new stack
    // slots and always reusing the same one.  We currently always create
    // new ones, as reuse may inhibit scheduling.
    MVT ExtraVT = cast<VTSDNode>(Node->getOperand(1))->getVT();
    Tmp1 = EmitStackConvert(Node->getOperand(0), ExtraVT,
                            Node->getValueType(0), dl);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    Tmp1 = ExpandLegalINT_TO_FP(Node->getOpcode() == ISD::SINT_TO_FP,
                                Node->getOperand(0), Node->getValueType(0), dl);
    Results.push_back(Tmp1);
    break;
  case ISD::FP_TO_UINT: {
    SDValue True, False;
    MVT VT =  Node->getOperand(0).getValueType();
    MVT NVT = Node->getValueType(0);
    const uint64_t zero[] = {0, 0};
    APFloat apf = APFloat(APInt(VT.getSizeInBits(), 2, zero));
    APInt x = APInt::getSignBit(NVT.getSizeInBits());
    (void)apf.convertFromAPInt(x, false, APFloat::rmNearestTiesToEven);
    Tmp1 = DAG.getConstantFP(apf, VT);
    Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(VT),
                        Node->getOperand(0),
                        Tmp1, ISD::SETLT);
    True = DAG.getNode(ISD::FP_TO_SINT, dl, NVT, Node->getOperand(0));
    False = DAG.getNode(ISD::FP_TO_SINT, dl, NVT,
                        DAG.getNode(ISD::FSUB, dl, VT,
                                    Node->getOperand(0), Tmp1));
    False = DAG.getNode(ISD::XOR, dl, NVT, False,
                        DAG.getConstant(x, NVT));
    Tmp1 = DAG.getNode(ISD::SELECT, dl, NVT, Tmp2, True, False);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::VACOPY: {
    // This defaults to loading a pointer from the input and storing it to the
    // output, returning the chain.
    const Value *VD = cast<SrcValueSDNode>(Node->getOperand(3))->getValue();
    const Value *VS = cast<SrcValueSDNode>(Node->getOperand(4))->getValue();
    Tmp1 = DAG.getLoad(TLI.getPointerTy(), dl, Node->getOperand(0),
                       Node->getOperand(2), VS, 0);
    Tmp1 = DAG.getStore(Tmp1.getValue(1), dl, Tmp1, Node->getOperand(1), VD, 0);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    if (Node->getOperand(0).getValueType().getVectorNumElements() == 1)
      // This must be an access of the only element.  Return it.
      Tmp1 = DAG.getNode(ISD::BIT_CONVERT, dl, Node->getValueType(0), 
                         Node->getOperand(0));
    else
      Tmp1 = ExpandExtractFromVectorThroughStack(SDValue(Node, 0));
    Results.push_back(Tmp1);
    break;
  case ISD::EXTRACT_SUBVECTOR:
    Results.push_back(ExpandExtractFromVectorThroughStack(SDValue(Node, 0)));
    break;
  case ISD::SCALAR_TO_VECTOR:
    Results.push_back(ExpandSCALAR_TO_VECTOR(Node));
    break;
  case ISD::INSERT_VECTOR_ELT:
    Results.push_back(ExpandINSERT_VECTOR_ELT(Node->getOperand(0),
                                              Node->getOperand(1),
                                              Node->getOperand(2), dl));
    break;
  case ISD::EXTRACT_ELEMENT: {
    MVT OpTy = Node->getOperand(0).getValueType();
    if (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue()) {
      // 1 -> Hi
      Tmp1 = DAG.getNode(ISD::SRL, dl, OpTy, Node->getOperand(0),
                         DAG.getConstant(OpTy.getSizeInBits()/2,
                                         TLI.getShiftAmountTy()));
      Tmp1 = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0), Tmp1);
    } else {
      // 0 -> Lo
      Tmp1 = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0),
                         Node->getOperand(0));
    }
    Results.push_back(Tmp1);
    break;
  }
  case ISD::STACKSAVE:
    // Expand to CopyFromReg if the target set
    // StackPointerRegisterToSaveRestore.
    if (unsigned SP = TLI.getStackPointerRegisterToSaveRestore()) {
      Results.push_back(DAG.getCopyFromReg(Node->getOperand(0), dl, SP,
                                           Node->getValueType(0)));
      Results.push_back(Results[0].getValue(1));
    } else {
      Results.push_back(DAG.getUNDEF(Node->getValueType(0)));
      Results.push_back(Node->getOperand(0));
    }
    break;
  case ISD::STACKRESTORE:
    // Expand to CopyToReg if the target set
    // StackPointerRegisterToSaveRestore.
    if (unsigned SP = TLI.getStackPointerRegisterToSaveRestore()) {
      Results.push_back(DAG.getCopyToReg(Node->getOperand(0), dl, SP,
                                         Node->getOperand(1)));
    } else {
      Results.push_back(Node->getOperand(0));
    }
    break;
  case ISD::FNEG:
    // Expand Y = FNEG(X) ->  Y = SUB -0.0, X
    Tmp1 = DAG.getConstantFP(-0.0, Node->getValueType(0));
    Tmp1 = DAG.getNode(ISD::FSUB, dl, Node->getValueType(0), Tmp1,
                       Node->getOperand(0));
    Results.push_back(Tmp1);
    break;
  case ISD::FABS: {
    // Expand Y = FABS(X) -> Y = (X >u 0.0) ? X : fneg(X).
    MVT VT = Node->getValueType(0);
    Tmp1 = Node->getOperand(0);
    Tmp2 = DAG.getConstantFP(0.0, VT);
    Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(Tmp1.getValueType()),
                        Tmp1, Tmp2, ISD::SETUGT);
    Tmp3 = DAG.getNode(ISD::FNEG, dl, VT, Tmp1);
    Tmp1 = DAG.getNode(ISD::SELECT, dl, VT, Tmp2, Tmp1, Tmp3);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::FSQRT:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::SQRT_F32, RTLIB::SQRT_F64,
                                      RTLIB::SQRT_F80, RTLIB::SQRT_PPCF128));
    break;
  case ISD::FSIN:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::SIN_F32, RTLIB::SIN_F64,
                                      RTLIB::SIN_F80, RTLIB::SIN_PPCF128));
    break;
  case ISD::FCOS:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::COS_F32, RTLIB::COS_F64,
                                      RTLIB::COS_F80, RTLIB::COS_PPCF128));
    break;
  case ISD::FLOG:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::LOG_F32, RTLIB::LOG_F64,
                                      RTLIB::LOG_F80, RTLIB::LOG_PPCF128));
    break;
  case ISD::FLOG2:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::LOG2_F32, RTLIB::LOG2_F64,
                                      RTLIB::LOG2_F80, RTLIB::LOG2_PPCF128));
    break;
  case ISD::FLOG10:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::LOG10_F32, RTLIB::LOG10_F64,
                                      RTLIB::LOG10_F80, RTLIB::LOG10_PPCF128));
    break;
  case ISD::FEXP:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::EXP_F32, RTLIB::EXP_F64,
                                      RTLIB::EXP_F80, RTLIB::EXP_PPCF128));
    break;
  case ISD::FEXP2:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::EXP2_F32, RTLIB::EXP2_F64,
                                      RTLIB::EXP2_F80, RTLIB::EXP2_PPCF128));
    break;
  case ISD::FTRUNC:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::TRUNC_F32, RTLIB::TRUNC_F64,
                                      RTLIB::TRUNC_F80, RTLIB::TRUNC_PPCF128));
    break;
  case ISD::FFLOOR:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::FLOOR_F32, RTLIB::FLOOR_F64,
                                      RTLIB::FLOOR_F80, RTLIB::FLOOR_PPCF128));
    break;
  case ISD::FCEIL:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::CEIL_F32, RTLIB::CEIL_F64,
                                      RTLIB::CEIL_F80, RTLIB::CEIL_PPCF128));
    break;
  case ISD::FRINT:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::RINT_F32, RTLIB::RINT_F64,
                                      RTLIB::RINT_F80, RTLIB::RINT_PPCF128));
    break;
  case ISD::FNEARBYINT:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::NEARBYINT_F32,
                                      RTLIB::NEARBYINT_F64,
                                      RTLIB::NEARBYINT_F80,
                                      RTLIB::NEARBYINT_PPCF128));
    break;
  case ISD::FPOWI:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::POWI_F32, RTLIB::POWI_F64,
                                      RTLIB::POWI_F80, RTLIB::POWI_PPCF128));
    break;
  case ISD::FPOW:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::POW_F32, RTLIB::POW_F64,
                                      RTLIB::POW_F80, RTLIB::POW_PPCF128));
    break;
  case ISD::FDIV:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::DIV_F32, RTLIB::DIV_F64,
                                      RTLIB::DIV_F80, RTLIB::DIV_PPCF128));
    break;
  case ISD::FREM:
    Results.push_back(ExpandFPLibCall(Node, RTLIB::REM_F32, RTLIB::REM_F64,
                                      RTLIB::REM_F80, RTLIB::REM_PPCF128));
    break;
  case ISD::EHSELECTION: {
    unsigned Reg = TLI.getExceptionSelectorRegister();
    assert(Reg && "Can't expand to unknown register!");
    Results.push_back(DAG.getCopyFromReg(Node->getOperand(1), dl, Reg,
                                         Node->getValueType(0)));
    Results.push_back(Results[0].getValue(1));
    break;
  }
  case ISD::EXCEPTIONADDR: {
    unsigned Reg = TLI.getExceptionAddressRegister();
    assert(Reg && "Can't expand to unknown register!");
    Results.push_back(DAG.getCopyFromReg(Node->getOperand(0), dl, Reg,
                                         Node->getValueType(0)));
    Results.push_back(Results[0].getValue(1));
    break;
  }
  case ISD::SUB: {
    MVT VT = Node->getValueType(0);
    assert(TLI.isOperationLegalOrCustom(ISD::ADD, VT) &&
           TLI.isOperationLegalOrCustom(ISD::XOR, VT) &&
           "Don't know how to expand this subtraction!");
    Tmp1 = DAG.getNode(ISD::XOR, dl, VT, Node->getOperand(1),
               DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), VT));
    Tmp1 = DAG.getNode(ISD::ADD, dl, VT, Tmp2, DAG.getConstant(1, VT));
    Results.push_back(DAG.getNode(ISD::ADD, dl, VT, Node->getOperand(0), Tmp1));
    break;
  }
  case ISD::UDIV:
  case ISD::UREM: {
    bool isRem = Node->getOpcode() == ISD::UREM;
    MVT VT = Node->getValueType(0);
    SDVTList VTs = DAG.getVTList(VT, VT);
    if (TLI.isOperationLegalOrCustom(ISD::UDIVREM, VT))
      Tmp1 = DAG.getNode(ISD::UDIVREM, dl, VTs, Node->getOperand(0),
                         Node->getOperand(1)).getValue(isRem);
    else if (isRem)
      Tmp1 = ExpandIntLibCall(Node, false, RTLIB::UREM_I16, RTLIB::UREM_I32,
                              RTLIB::UREM_I64, RTLIB::UREM_I128);
    else
      Tmp1 = ExpandIntLibCall(Node, false, RTLIB::UDIV_I16, RTLIB::UDIV_I32,
                              RTLIB::UDIV_I64, RTLIB::UDIV_I128);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::SDIV:
  case ISD::SREM: {
    bool isRem = Node->getOpcode() == ISD::SREM;
    MVT VT = Node->getValueType(0);
    SDVTList VTs = DAG.getVTList(VT, VT);
    if (TLI.isOperationLegalOrCustom(ISD::SDIVREM, VT))
      Tmp1 = DAG.getNode(ISD::SDIVREM, dl, VTs, Node->getOperand(0),
                         Node->getOperand(1)).getValue(isRem);
    else if (isRem)
      Tmp1 = ExpandIntLibCall(Node, true, RTLIB::SREM_I16, RTLIB::SREM_I32,
                              RTLIB::SREM_I64, RTLIB::SREM_I128);
    else
      Tmp1 = ExpandIntLibCall(Node, true, RTLIB::SDIV_I16, RTLIB::SDIV_I32,
                              RTLIB::SDIV_I64, RTLIB::SDIV_I128);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::MULHU:
  case ISD::MULHS: {
    unsigned ExpandOpcode = Node->getOpcode() == ISD::MULHU ? ISD::UMUL_LOHI :
                                                              ISD::SMUL_LOHI;
    MVT VT = Node->getValueType(0);
    SDVTList VTs = DAG.getVTList(VT, VT);
    assert(TLI.isOperationLegalOrCustom(ExpandOpcode, VT) &&
           "If this wasn't legal, it shouldn't have been created!");
    Tmp1 = DAG.getNode(ExpandOpcode, dl, VTs, Node->getOperand(0),
                       Node->getOperand(1));
    Results.push_back(Tmp1.getValue(1));
    break;
  }
  case ISD::MUL: {
    MVT VT = Node->getValueType(0);
    SDVTList VTs = DAG.getVTList(VT, VT);
    // See if multiply or divide can be lowered using two-result operations.
    // We just need the low half of the multiply; try both the signed
    // and unsigned forms. If the target supports both SMUL_LOHI and
    // UMUL_LOHI, form a preference by checking which forms of plain
    // MULH it supports.
    bool HasSMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::SMUL_LOHI, VT);
    bool HasUMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::UMUL_LOHI, VT);
    bool HasMULHS = TLI.isOperationLegalOrCustom(ISD::MULHS, VT);
    bool HasMULHU = TLI.isOperationLegalOrCustom(ISD::MULHU, VT);
    unsigned OpToUse = 0;
    if (HasSMUL_LOHI && !HasMULHS) {
      OpToUse = ISD::SMUL_LOHI;
    } else if (HasUMUL_LOHI && !HasMULHU) {
      OpToUse = ISD::UMUL_LOHI;
    } else if (HasSMUL_LOHI) {
      OpToUse = ISD::SMUL_LOHI;
    } else if (HasUMUL_LOHI) {
      OpToUse = ISD::UMUL_LOHI;
    }
    if (OpToUse) {
      Results.push_back(DAG.getNode(OpToUse, dl, VTs, Tmp1, Tmp2));
      break;
    }
    Tmp1 = ExpandIntLibCall(Node, false, RTLIB::MUL_I16, RTLIB::MUL_I32,
                            RTLIB::MUL_I64, RTLIB::MUL_I128);
    Results.push_back(Tmp1);
    break;
  }
  case ISD::GLOBAL_OFFSET_TABLE:
  case ISD::GlobalAddress:
  case ISD::GlobalTLSAddress:
  case ISD::ExternalSymbol:
  case ISD::ConstantPool:
  case ISD::JumpTable:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID:
    // FIXME: Custom lowering for these operations shouldn't return null!
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      Results.push_back(SDValue(Node, i));
    break;
  }
}
void SelectionDAGLegalize::PromoteNode(SDNode *Node,
                                       SmallVectorImpl<SDValue> &Results) {
  MVT OVT = Node->getValueType(0);
  if (Node->getOpcode() == ISD::UINT_TO_FP ||
      Node->getOpcode() == ISD::SINT_TO_FP) {
    OVT = Node->getOperand(0).getValueType();
  }
  MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);
  DebugLoc dl = Node->getDebugLoc();
  SDValue Tmp1, Tmp2;
  switch (Node->getOpcode()) {
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
    // Zero extend the argument.
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Node->getOperand(0));
    // Perform the larger operation.
    Tmp1 = DAG.getNode(Node->getOpcode(), dl, Node->getValueType(0), Tmp1);
    if (Node->getOpcode() == ISD::CTTZ) {
      //if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
      Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(Tmp1.getValueType()),
                          Tmp1, DAG.getConstant(NVT.getSizeInBits(), NVT),
                          ISD::SETEQ);
      Tmp1 = DAG.getNode(ISD::SELECT, dl, NVT, Tmp2,
                          DAG.getConstant(OVT.getSizeInBits(), NVT), Tmp1);
    } else if (Node->getOpcode() == ISD::CTLZ) {
      // Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
      Tmp1 = DAG.getNode(ISD::SUB, dl, NVT, Tmp1,
                          DAG.getConstant(NVT.getSizeInBits() -
                                          OVT.getSizeInBits(), NVT));
    }
    Results.push_back(Tmp1);
    break;
  case ISD::BSWAP: {
    unsigned DiffBits = NVT.getSizeInBits() - OVT.getSizeInBits();
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Tmp1);
    Tmp1 = DAG.getNode(ISD::BSWAP, dl, NVT, Tmp1);
    Tmp1 = DAG.getNode(ISD::SRL, dl, NVT, Tmp1,
                          DAG.getConstant(DiffBits, TLI.getShiftAmountTy()));
    Results.push_back(Tmp1);
    break;
  }
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
    Tmp1 = PromoteLegalFP_TO_INT(Node->getOperand(0), Node->getValueType(0),
                                 Node->getOpcode() == ISD::FP_TO_SINT, dl);
    Results.push_back(Tmp1);
    break;
  case ISD::UINT_TO_FP:
  case ISD::SINT_TO_FP:
    Tmp1 = PromoteLegalINT_TO_FP(Node->getOperand(0), Node->getValueType(0),
                                 Node->getOpcode() == ISD::SINT_TO_FP, dl);
    Results.push_back(Tmp1);
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    assert(OVT.isVector() && "Don't know how to promote scalar logic ops");
    // Bit convert each of the values to the new type.
    Tmp1 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(0));
    Tmp2 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(1));
    Tmp1 = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    // Bit convert the result back the original type.
    Results.push_back(DAG.getNode(ISD::BIT_CONVERT, dl, OVT, Tmp1));
    break;
  }
}

// SelectionDAG::Legalize - This is the entry point for the file.
//
void SelectionDAG::Legalize(bool TypesNeedLegalizing,
                            CodeGenOpt::Level OptLevel) {
  /// run - This is the main entry point to this class.
  ///
  SelectionDAGLegalize(*this, OptLevel).LegalizeDAG();
}

