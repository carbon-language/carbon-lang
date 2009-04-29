//===- X86ISelDAGToDAG.cpp - A DAG pattern matching inst selector for X86 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DAG pattern matching instruction selector for X86,
// converting from a legalized dag to a X86 dag.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-isel"
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86ISelLowering.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/CFG.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

#include "llvm/Support/CommandLine.h"
static cl::opt<bool> AvoidDupAddrCompute("x86-avoid-dup-address", cl::Hidden);

STATISTIC(NumLoadMoved, "Number of loads moved below TokenFactor");

//===----------------------------------------------------------------------===//
//                      Pattern Matcher Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// X86ISelAddressMode - This corresponds to X86AddressMode, but uses
  /// SDValue's instead of register numbers for the leaves of the matched
  /// tree.
  struct X86ISelAddressMode {
    enum {
      RegBase,
      FrameIndexBase
    } BaseType;

    struct {            // This is really a union, discriminated by BaseType!
      SDValue Reg;
      int FrameIndex;
    } Base;

    bool isRIPRel;     // RIP as base?
    unsigned Scale;
    SDValue IndexReg; 
    int32_t Disp;
    SDValue Segment;
    GlobalValue *GV;
    Constant *CP;
    const char *ES;
    int JT;
    unsigned Align;    // CP alignment.

    X86ISelAddressMode()
      : BaseType(RegBase), isRIPRel(false), Scale(1), IndexReg(), Disp(0),
        Segment(), GV(0), CP(0), ES(0), JT(-1), Align(0) {
    }

    bool hasSymbolicDisplacement() const {
      return GV != 0 || CP != 0 || ES != 0 || JT != -1;
    }

    void dump() {
      cerr << "X86ISelAddressMode " << this << "\n";
      cerr << "Base.Reg ";
              if (Base.Reg.getNode() != 0) Base.Reg.getNode()->dump(); 
              else cerr << "nul";
      cerr << " Base.FrameIndex " << Base.FrameIndex << "\n";
      cerr << "isRIPRel " << isRIPRel << " Scale" << Scale << "\n";
      cerr << "IndexReg ";
              if (IndexReg.getNode() != 0) IndexReg.getNode()->dump();
              else cerr << "nul"; 
      cerr << " Disp " << Disp << "\n";
      cerr << "GV "; if (GV) GV->dump(); 
                     else cerr << "nul";
      cerr << " CP "; if (CP) CP->dump(); 
                     else cerr << "nul";
      cerr << "\n";
      cerr << "ES "; if (ES) cerr << ES; else cerr << "nul";
      cerr  << " JT" << JT << " Align" << Align << "\n";
    }
  };
}

namespace {
  //===--------------------------------------------------------------------===//
  /// ISel - X86 specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class VISIBILITY_HIDDEN X86DAGToDAGISel : public SelectionDAGISel {
    /// TM - Keep a reference to X86TargetMachine.
    ///
    X86TargetMachine &TM;

    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    X86TargetLowering &X86Lowering;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// CurBB - Current BB being isel'd.
    ///
    MachineBasicBlock *CurBB;

    /// OptForSize - If true, selector should try to optimize for code size
    /// instead of performance.
    bool OptForSize;

  public:
    explicit X86DAGToDAGISel(X86TargetMachine &tm, CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel),
        TM(tm), X86Lowering(*TM.getTargetLowering()),
        Subtarget(&TM.getSubtarget<X86Subtarget>()),
        OptForSize(false) {}

    virtual const char *getPassName() const {
      return "X86 DAG->DAG Instruction Selection";
    }

    /// InstructionSelect - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelect();

    virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF);

    virtual
      bool IsLegalAndProfitableToFold(SDNode *N, SDNode *U, SDNode *Root) const;

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    SDNode *Select(SDValue N);
    SDNode *SelectAtomic64(SDNode *Node, unsigned Opc);

    bool MatchSegmentBaseAddress(SDValue N, X86ISelAddressMode &AM);
    bool MatchLoad(SDValue N, X86ISelAddressMode &AM);
    bool MatchWrapper(SDValue N, X86ISelAddressMode &AM);
    bool MatchAddress(SDValue N, X86ISelAddressMode &AM,
                      unsigned Depth = 0);
    bool MatchAddressBase(SDValue N, X86ISelAddressMode &AM);
    bool SelectAddr(SDValue Op, SDValue N, SDValue &Base,
                    SDValue &Scale, SDValue &Index, SDValue &Disp,
                    SDValue &Segment);
    bool SelectLEAAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Scale, SDValue &Index, SDValue &Disp);
    bool SelectScalarSSELoad(SDValue Op, SDValue Pred,
                             SDValue N, SDValue &Base, SDValue &Scale,
                             SDValue &Index, SDValue &Disp,
                             SDValue &Segment,
                             SDValue &InChain, SDValue &OutChain);
    bool TryFoldLoad(SDValue P, SDValue N,
                     SDValue &Base, SDValue &Scale,
                     SDValue &Index, SDValue &Disp,
                     SDValue &Segment);
    void PreprocessForRMW();
    void PreprocessForFPConvert();

    /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
    /// inline asm expressions.
    virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                              char ConstraintCode,
                                              std::vector<SDValue> &OutOps);
    
    void EmitSpecialCodeForMain(MachineBasicBlock *BB, MachineFrameInfo *MFI);

    inline void getAddressOperands(X86ISelAddressMode &AM, SDValue &Base, 
                                   SDValue &Scale, SDValue &Index,
                                   SDValue &Disp, SDValue &Segment) {
      Base  = (AM.BaseType == X86ISelAddressMode::FrameIndexBase) ?
        CurDAG->getTargetFrameIndex(AM.Base.FrameIndex, TLI.getPointerTy()) :
        AM.Base.Reg;
      Scale = getI8Imm(AM.Scale);
      Index = AM.IndexReg;
      // These are 32-bit even in 64-bit mode since RIP relative offset
      // is 32-bit.
      if (AM.GV)
        Disp = CurDAG->getTargetGlobalAddress(AM.GV, MVT::i32, AM.Disp);
      else if (AM.CP)
        Disp = CurDAG->getTargetConstantPool(AM.CP, MVT::i32,
                                             AM.Align, AM.Disp);
      else if (AM.ES)
        Disp = CurDAG->getTargetExternalSymbol(AM.ES, MVT::i32);
      else if (AM.JT != -1)
        Disp = CurDAG->getTargetJumpTable(AM.JT, MVT::i32);
      else
        Disp = CurDAG->getTargetConstant(AM.Disp, MVT::i32);

      if (AM.Segment.getNode())
        Segment = AM.Segment;
      else
        Segment = CurDAG->getRegister(0, MVT::i32);
    }

    /// getI8Imm - Return a target constant with the specified value, of type
    /// i8.
    inline SDValue getI8Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i8);
    }

    /// getI16Imm - Return a target constant with the specified value, of type
    /// i16.
    inline SDValue getI16Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i16);
    }

    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDValue getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getGlobalBaseReg - Return an SDNode that returns the value of
    /// the global base register. Output instructions required to
    /// initialize the global base register, if necessary.
    ///
    SDNode *getGlobalBaseReg();

#ifndef NDEBUG
    unsigned Indent;
#endif
  };
}

/// findFlagUse - Return use of MVT::Flag value produced by the specified
/// SDNode.
///
static SDNode *findFlagUse(SDNode *N) {
  unsigned FlagResNo = N->getNumValues()-1;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDUse &Use = I.getUse();
    if (Use.getResNo() == FlagResNo)
      return Use.getUser();
  }
  return NULL;
}

/// findNonImmUse - Return true if "Use" is a non-immediate use of "Def".
/// This function recursively traverses up the operand chain, ignoring
/// certain nodes.
static bool findNonImmUse(SDNode *Use, SDNode* Def, SDNode *ImmedUse,
                          SDNode *Root,
                          SmallPtrSet<SDNode*, 16> &Visited) {
  if (Use->getNodeId() < Def->getNodeId() ||
      !Visited.insert(Use))
    return false;

  for (unsigned i = 0, e = Use->getNumOperands(); i != e; ++i) {
    SDNode *N = Use->getOperand(i).getNode();
    if (N == Def) {
      if (Use == ImmedUse || Use == Root)
        continue;  // We are not looking for immediate use.
      assert(N != Root);
      return true;
    }

    // Traverse up the operand chain.
    if (findNonImmUse(N, Def, ImmedUse, Root, Visited))
      return true;
  }
  return false;
}

/// isNonImmUse - Start searching from Root up the DAG to check is Def can
/// be reached. Return true if that's the case. However, ignore direct uses
/// by ImmedUse (which would be U in the example illustrated in
/// IsLegalAndProfitableToFold) and by Root (which can happen in the store
/// case).
/// FIXME: to be really generic, we should allow direct use by any node
/// that is being folded. But realisticly since we only fold loads which
/// have one non-chain use, we only need to watch out for load/op/store
/// and load/op/cmp case where the root (store / cmp) may reach the load via
/// its chain operand.
static inline bool isNonImmUse(SDNode *Root, SDNode *Def, SDNode *ImmedUse) {
  SmallPtrSet<SDNode*, 16> Visited;
  return findNonImmUse(Root, Def, ImmedUse, Root, Visited);
}


bool X86DAGToDAGISel::IsLegalAndProfitableToFold(SDNode *N, SDNode *U,
                                                 SDNode *Root) const {
  if (OptLevel == CodeGenOpt::None) return false;

  if (U == Root)
    switch (U->getOpcode()) {
    default: break;
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR: {
      SDValue Op1 = U->getOperand(1);

      // If the other operand is a 8-bit immediate we should fold the immediate
      // instead. This reduces code size.
      // e.g.
      // movl 4(%esp), %eax
      // addl $4, %eax
      // vs.
      // movl $4, %eax
      // addl 4(%esp), %eax
      // The former is 2 bytes shorter. In case where the increment is 1, then
      // the saving can be 4 bytes (by using incl %eax).
      if (ConstantSDNode *Imm = dyn_cast<ConstantSDNode>(Op1))
        if (Imm->getAPIntValue().isSignedIntN(8))
          return false;

      // If the other operand is a TLS address, we should fold it instead.
      // This produces
      // movl    %gs:0, %eax
      // leal    i@NTPOFF(%eax), %eax
      // instead of
      // movl    $i@NTPOFF, %eax
      // addl    %gs:0, %eax
      // if the block also has an access to a second TLS address this will save
      // a load.
      // FIXME: This is probably also true for non TLS addresses.
      if (Op1.getOpcode() == X86ISD::Wrapper) {
        SDValue Val = Op1.getOperand(0);
        if (Val.getOpcode() == ISD::TargetGlobalTLSAddress)
          return false;
      }
    }
    }

  // If Root use can somehow reach N through a path that that doesn't contain
  // U then folding N would create a cycle. e.g. In the following
  // diagram, Root can reach N through X. If N is folded into into Root, then
  // X is both a predecessor and a successor of U.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^     ^          //
  //         \   /           //
  //          \ /            //
  //         [Root*]         //
  //
  // * indicates nodes to be folded together.
  //
  // If Root produces a flag, then it gets (even more) interesting. Since it
  // will be "glued" together with its flag use in the scheduler, we need to
  // check if it might reach N.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^       ^        //
  //         \       \       //
  //          \      |       //
  //         [Root*] |       //
  //          ^      |       //
  //          f      |       //
  //          |      /       //
  //         [Y]    /        //
  //           ^   /         //
  //           f  /          //
  //           | /           //
  //          [FU]           //
  //
  // If FU (flag use) indirectly reaches N (the load), and Root folds N
  // (call it Fold), then X is a predecessor of FU and a successor of
  // Fold. But since Fold and FU are flagged together, this will create
  // a cycle in the scheduling graph.

  MVT VT = Root->getValueType(Root->getNumValues()-1);
  while (VT == MVT::Flag) {
    SDNode *FU = findFlagUse(Root);
    if (FU == NULL)
      break;
    Root = FU;
    VT = Root->getValueType(Root->getNumValues()-1);
  }

  return !isNonImmUse(Root, N, U);
}

/// MoveBelowTokenFactor - Replace TokenFactor operand with load's chain operand
/// and move load below the TokenFactor. Replace store's chain operand with
/// load's chain result.
static void MoveBelowTokenFactor(SelectionDAG *CurDAG, SDValue Load,
                                 SDValue Store, SDValue TF) {
  SmallVector<SDValue, 4> Ops;
  for (unsigned i = 0, e = TF.getNode()->getNumOperands(); i != e; ++i)
    if (Load.getNode() == TF.getOperand(i).getNode())
      Ops.push_back(Load.getOperand(0));
    else
      Ops.push_back(TF.getOperand(i));
  CurDAG->UpdateNodeOperands(TF, &Ops[0], Ops.size());
  CurDAG->UpdateNodeOperands(Load, TF, Load.getOperand(1), Load.getOperand(2));
  CurDAG->UpdateNodeOperands(Store, Load.getValue(1), Store.getOperand(1),
                             Store.getOperand(2), Store.getOperand(3));
}

/// isRMWLoad - Return true if N is a load that's part of RMW sub-DAG.
/// 
static bool isRMWLoad(SDValue N, SDValue Chain, SDValue Address,
                      SDValue &Load) {
  if (N.getOpcode() == ISD::BIT_CONVERT)
    N = N.getOperand(0);

  LoadSDNode *LD = dyn_cast<LoadSDNode>(N);
  if (!LD || LD->isVolatile())
    return false;
  if (LD->getAddressingMode() != ISD::UNINDEXED)
    return false;

  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType != ISD::NON_EXTLOAD && ExtType != ISD::EXTLOAD)
    return false;

  if (N.hasOneUse() &&
      N.getOperand(1) == Address &&
      N.getNode()->isOperandOf(Chain.getNode())) {
    Load = N;
    return true;
  }
  return false;
}

/// MoveBelowCallSeqStart - Replace CALLSEQ_START operand with load's chain
/// operand and move load below the call's chain operand.
static void MoveBelowCallSeqStart(SelectionDAG *CurDAG, SDValue Load,
                                  SDValue Call, SDValue CallSeqStart) {
  SmallVector<SDValue, 8> Ops;
  SDValue Chain = CallSeqStart.getOperand(0);
  if (Chain.getNode() == Load.getNode())
    Ops.push_back(Load.getOperand(0));
  else {
    assert(Chain.getOpcode() == ISD::TokenFactor &&
           "Unexpected CallSeqStart chain operand");
    for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i)
      if (Chain.getOperand(i).getNode() == Load.getNode())
        Ops.push_back(Load.getOperand(0));
      else
        Ops.push_back(Chain.getOperand(i));
    SDValue NewChain =
      CurDAG->getNode(ISD::TokenFactor, Load.getDebugLoc(),
                      MVT::Other, &Ops[0], Ops.size());
    Ops.clear();
    Ops.push_back(NewChain);
  }
  for (unsigned i = 1, e = CallSeqStart.getNumOperands(); i != e; ++i)
    Ops.push_back(CallSeqStart.getOperand(i));
  CurDAG->UpdateNodeOperands(CallSeqStart, &Ops[0], Ops.size());
  CurDAG->UpdateNodeOperands(Load, Call.getOperand(0),
                             Load.getOperand(1), Load.getOperand(2));
  Ops.clear();
  Ops.push_back(SDValue(Load.getNode(), 1));
  for (unsigned i = 1, e = Call.getNode()->getNumOperands(); i != e; ++i)
    Ops.push_back(Call.getOperand(i));
  CurDAG->UpdateNodeOperands(Call, &Ops[0], Ops.size());
}

/// isCalleeLoad - Return true if call address is a load and it can be
/// moved below CALLSEQ_START and the chains leading up to the call.
/// Return the CALLSEQ_START by reference as a second output.
static bool isCalleeLoad(SDValue Callee, SDValue &Chain) {
  if (Callee.getNode() == Chain.getNode() || !Callee.hasOneUse())
    return false;
  LoadSDNode *LD = dyn_cast<LoadSDNode>(Callee.getNode());
  if (!LD ||
      LD->isVolatile() ||
      LD->getAddressingMode() != ISD::UNINDEXED ||
      LD->getExtensionType() != ISD::NON_EXTLOAD)
    return false;

  // Now let's find the callseq_start.
  while (Chain.getOpcode() != ISD::CALLSEQ_START) {
    if (!Chain.hasOneUse())
      return false;
    Chain = Chain.getOperand(0);
  }
  
  if (Chain.getOperand(0).getNode() == Callee.getNode())
    return true;
  if (Chain.getOperand(0).getOpcode() == ISD::TokenFactor &&
      Callee.getValue(1).isOperandOf(Chain.getOperand(0).getNode()))
    return true;
  return false;
}


/// PreprocessForRMW - Preprocess the DAG to make instruction selection better.
/// This is only run if not in -O0 mode.
/// This allows the instruction selector to pick more read-modify-write
/// instructions. This is a common case:
///
///     [Load chain]
///         ^
///         |
///       [Load]
///       ^    ^
///       |    |
///      /      \-
///     /         |
/// [TokenFactor] [Op]
///     ^          ^
///     |          |
///      \        /
///       \      /
///       [Store]
///
/// The fact the store's chain operand != load's chain will prevent the
/// (store (op (load))) instruction from being selected. We can transform it to:
///
///     [Load chain]
///         ^
///         |
///    [TokenFactor]
///         ^
///         |
///       [Load]
///       ^    ^
///       |    |
///       |     \- 
///       |       | 
///       |     [Op]
///       |       ^
///       |       |
///       \      /
///        \    /
///       [Store]
void X86DAGToDAGISel::PreprocessForRMW() {
  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
         E = CurDAG->allnodes_end(); I != E; ++I) {
    if (I->getOpcode() == X86ISD::CALL) {
      /// Also try moving call address load from outside callseq_start to just
      /// before the call to allow it to be folded.
      ///
      ///     [Load chain]
      ///         ^
      ///         |
      ///       [Load]
      ///       ^    ^
      ///       |    |
      ///      /      \--
      ///     /          |
      ///[CALLSEQ_START] |
      ///     ^          |
      ///     |          |
      /// [LOAD/C2Reg]   |
      ///     |          |
      ///      \        /
      ///       \      /
      ///       [CALL]
      SDValue Chain = I->getOperand(0);
      SDValue Load  = I->getOperand(1);
      if (!isCalleeLoad(Load, Chain))
        continue;
      MoveBelowCallSeqStart(CurDAG, Load, SDValue(I, 0), Chain);
      ++NumLoadMoved;
      continue;
    }

    if (!ISD::isNON_TRUNCStore(I))
      continue;
    SDValue Chain = I->getOperand(0);

    if (Chain.getNode()->getOpcode() != ISD::TokenFactor)
      continue;

    SDValue N1 = I->getOperand(1);
    SDValue N2 = I->getOperand(2);
    if ((N1.getValueType().isFloatingPoint() &&
         !N1.getValueType().isVector()) ||
        !N1.hasOneUse())
      continue;

    bool RModW = false;
    SDValue Load;
    unsigned Opcode = N1.getNode()->getOpcode();
    switch (Opcode) {
    case ISD::ADD:
    case ISD::MUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::VECTOR_SHUFFLE: {
      SDValue N10 = N1.getOperand(0);
      SDValue N11 = N1.getOperand(1);
      RModW = isRMWLoad(N10, Chain, N2, Load);
      if (!RModW)
        RModW = isRMWLoad(N11, Chain, N2, Load);
      break;
    }
    case ISD::SUB:
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
    case ISD::ROTL:
    case ISD::ROTR:
    case ISD::SUBC:
    case ISD::SUBE:
    case X86ISD::SHLD:
    case X86ISD::SHRD: {
      SDValue N10 = N1.getOperand(0);
      RModW = isRMWLoad(N10, Chain, N2, Load);
      break;
    }
    }

    if (RModW) {
      MoveBelowTokenFactor(CurDAG, Load, SDValue(I, 0), Chain);
      ++NumLoadMoved;
    }
  }
}


/// PreprocessForFPConvert - Walk over the dag lowering fpround and fpextend
/// nodes that target the FP stack to be store and load to the stack.  This is a
/// gross hack.  We would like to simply mark these as being illegal, but when
/// we do that, legalize produces these when it expands calls, then expands
/// these in the same legalize pass.  We would like dag combine to be able to
/// hack on these between the call expansion and the node legalization.  As such
/// this pass basically does "really late" legalization of these inline with the
/// X86 isel pass.
void X86DAGToDAGISel::PreprocessForFPConvert() {
  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
       E = CurDAG->allnodes_end(); I != E; ) {
    SDNode *N = I++;  // Preincrement iterator to avoid invalidation issues.
    if (N->getOpcode() != ISD::FP_ROUND && N->getOpcode() != ISD::FP_EXTEND)
      continue;
    
    // If the source and destination are SSE registers, then this is a legal
    // conversion that should not be lowered.
    MVT SrcVT = N->getOperand(0).getValueType();
    MVT DstVT = N->getValueType(0);
    bool SrcIsSSE = X86Lowering.isScalarFPTypeInSSEReg(SrcVT);
    bool DstIsSSE = X86Lowering.isScalarFPTypeInSSEReg(DstVT);
    if (SrcIsSSE && DstIsSSE)
      continue;

    if (!SrcIsSSE && !DstIsSSE) {
      // If this is an FPStack extension, it is a noop.
      if (N->getOpcode() == ISD::FP_EXTEND)
        continue;
      // If this is a value-preserving FPStack truncation, it is a noop.
      if (N->getConstantOperandVal(1))
        continue;
    }
   
    // Here we could have an FP stack truncation or an FPStack <-> SSE convert.
    // FPStack has extload and truncstore.  SSE can fold direct loads into other
    // operations.  Based on this, decide what we want to do.
    MVT MemVT;
    if (N->getOpcode() == ISD::FP_ROUND)
      MemVT = DstVT;  // FP_ROUND must use DstVT, we can't do a 'trunc load'.
    else
      MemVT = SrcIsSSE ? SrcVT : DstVT;
    
    SDValue MemTmp = CurDAG->CreateStackTemporary(MemVT);
    DebugLoc dl = N->getDebugLoc();
    
    // FIXME: optimize the case where the src/dest is a load or store?
    SDValue Store = CurDAG->getTruncStore(CurDAG->getEntryNode(), dl,
                                          N->getOperand(0),
                                          MemTmp, NULL, 0, MemVT);
    SDValue Result = CurDAG->getExtLoad(ISD::EXTLOAD, dl, DstVT, Store, MemTmp,
                                        NULL, 0, MemVT);

    // We're about to replace all uses of the FP_ROUND/FP_EXTEND with the
    // extload we created.  This will cause general havok on the dag because
    // anything below the conversion could be folded into other existing nodes.
    // To avoid invalidating 'I', back it up to the convert node.
    --I;
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
    
    // Now that we did that, the node is dead.  Increment the iterator to the
    // next node to process, then delete N.
    ++I;
    CurDAG->DeleteNode(N);
  }  
}

/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void X86DAGToDAGISel::InstructionSelect() {
  CurBB = BB;  // BB can change as result of isel.
  const Function *F = CurDAG->getMachineFunction().getFunction();
  OptForSize = F->hasFnAttr(Attribute::OptimizeForSize);

  DEBUG(BB->dump());
  if (OptLevel != CodeGenOpt::None)
    PreprocessForRMW();

  // FIXME: This should only happen when not compiled with -O0.
  PreprocessForFPConvert();

  // Codegen the basic block.
#ifndef NDEBUG
  DOUT << "===== Instruction selection begins:\n";
  Indent = 0;
#endif
  SelectRoot(*CurDAG);
#ifndef NDEBUG
  DOUT << "===== Instruction selection ends:\n";
#endif

  CurDAG->RemoveDeadNodes();
}

/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
void X86DAGToDAGISel::EmitSpecialCodeForMain(MachineBasicBlock *BB,
                                             MachineFrameInfo *MFI) {
  const TargetInstrInfo *TII = TM.getInstrInfo();
  if (Subtarget->isTargetCygMing())
    BuildMI(BB, DebugLoc::getUnknownLoc(),
            TII->get(X86::CALLpcrel32)).addExternalSymbol("__main");
}

void X86DAGToDAGISel::EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {
  // If this is main, emit special code for main.
  MachineBasicBlock *BB = MF.begin();
  if (Fn.hasExternalLinkage() && Fn.getName() == "main")
    EmitSpecialCodeForMain(BB, MF.getFrameInfo());
}


bool X86DAGToDAGISel::MatchSegmentBaseAddress(SDValue N,
                                              X86ISelAddressMode &AM) {
  assert(N.getOpcode() == X86ISD::SegmentBaseAddress);
  SDValue Segment = N.getOperand(0);

  if (AM.Segment.getNode() == 0) {
    AM.Segment = Segment;
    return false;
  }

  return true;
}

bool X86DAGToDAGISel::MatchLoad(SDValue N, X86ISelAddressMode &AM) {
  // This optimization is valid because the GNU TLS model defines that
  // gs:0 (or fs:0 on X86-64) contains its own address.
  // For more information see http://people.redhat.com/drepper/tls.pdf

  SDValue Address = N.getOperand(1);
  if (Address.getOpcode() == X86ISD::SegmentBaseAddress &&
      !MatchSegmentBaseAddress (Address, AM))
    return false;

  return true;
}

bool X86DAGToDAGISel::MatchWrapper(SDValue N, X86ISelAddressMode &AM) {
  bool is64Bit = Subtarget->is64Bit();
  DOUT << "Wrapper: 64bit " << is64Bit;
  DOUT << " AM "; DEBUG(AM.dump()); DOUT << "\n";

  // Under X86-64 non-small code model, GV (and friends) are 64-bits.
  if (is64Bit && (TM.getCodeModel() != CodeModel::Small))
    return true;

  // Base and index reg must be 0 in order to use rip as base.
  bool canUsePICRel = !AM.Base.Reg.getNode() && !AM.IndexReg.getNode();
  if (is64Bit && !canUsePICRel && TM.symbolicAddressesAreRIPRel())
    return true;

  if (AM.hasSymbolicDisplacement())
    return true;
  // If value is available in a register both base and index components have
  // been picked, we can't fit the result available in the register in the
  // addressing mode. Duplicate GlobalAddress or ConstantPool as displacement.

  SDValue N0 = N.getOperand(0);
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
    uint64_t Offset = G->getOffset();
    if (!is64Bit || isInt32(AM.Disp + Offset)) {
      GlobalValue *GV = G->getGlobal();
      bool isRIPRel = TM.symbolicAddressesAreRIPRel();
      if (N0.getOpcode() == llvm::ISD::TargetGlobalTLSAddress) {
        TLSModel::Model model =
          getTLSModel (GV, TM.getRelocationModel());
        if (is64Bit && model == TLSModel::InitialExec)
          isRIPRel = true;
      }
      AM.GV = GV;
      AM.Disp += Offset;
      AM.isRIPRel = isRIPRel;
      return false;
    }
  } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
    uint64_t Offset = CP->getOffset();
    if (!is64Bit || isInt32(AM.Disp + Offset)) {
      AM.CP = CP->getConstVal();
      AM.Align = CP->getAlignment();
      AM.Disp += Offset;
      AM.isRIPRel = TM.symbolicAddressesAreRIPRel();
      return false;
    }
  } else if (ExternalSymbolSDNode *S =dyn_cast<ExternalSymbolSDNode>(N0)) {
    AM.ES = S->getSymbol();
    AM.isRIPRel = TM.symbolicAddressesAreRIPRel();
    return false;
  } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
    AM.JT = J->getIndex();
    AM.isRIPRel = TM.symbolicAddressesAreRIPRel();
    return false;
  }

  return true;
}

/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode.
bool X86DAGToDAGISel::MatchAddress(SDValue N, X86ISelAddressMode &AM,
                                   unsigned Depth) {
  bool is64Bit = Subtarget->is64Bit();
  DebugLoc dl = N.getDebugLoc();
  DOUT << "MatchAddress: "; DEBUG(AM.dump());
  // Limit recursion.
  if (Depth > 5)
    return MatchAddressBase(N, AM);
  
  // RIP relative addressing: %rip + 32-bit displacement!
  if (AM.isRIPRel) {
    if (!AM.ES && AM.JT != -1 && N.getOpcode() == ISD::Constant) {
      uint64_t Val = cast<ConstantSDNode>(N)->getSExtValue();
      if (!is64Bit || isInt32(AM.Disp + Val)) {
        AM.Disp += Val;
        return false;
      }
    }
    return true;
  }

  switch (N.getOpcode()) {
  default: break;
  case ISD::Constant: {
    uint64_t Val = cast<ConstantSDNode>(N)->getSExtValue();
    if (!is64Bit || isInt32(AM.Disp + Val)) {
      AM.Disp += Val;
      return false;
    }
    break;
  }

  case X86ISD::SegmentBaseAddress:
    if (!MatchSegmentBaseAddress(N, AM))
      return false;
    break;

  case X86ISD::Wrapper:
    if (!MatchWrapper(N, AM))
      return false;
    break;

  case ISD::LOAD:
    if (!MatchLoad(N, AM))
      return false;
    break;

  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase
        && AM.Base.Reg.getNode() == 0) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::SHL:
    if (AM.IndexReg.getNode() != 0 || AM.Scale != 1 || AM.isRIPRel)
      break;
      
    if (ConstantSDNode
          *CN = dyn_cast<ConstantSDNode>(N.getNode()->getOperand(1))) {
      unsigned Val = CN->getZExtValue();
      if (Val == 1 || Val == 2 || Val == 3) {
        AM.Scale = 1 << Val;
        SDValue ShVal = N.getNode()->getOperand(0);

        // Okay, we know that we have a scale by now.  However, if the scaled
        // value is an add of something and a constant, we can fold the
        // constant into the disp field here.
        if (ShVal.getNode()->getOpcode() == ISD::ADD && ShVal.hasOneUse() &&
            isa<ConstantSDNode>(ShVal.getNode()->getOperand(1))) {
          AM.IndexReg = ShVal.getNode()->getOperand(0);
          ConstantSDNode *AddVal =
            cast<ConstantSDNode>(ShVal.getNode()->getOperand(1));
          uint64_t Disp = AM.Disp + (AddVal->getSExtValue() << Val);
          if (!is64Bit || isInt32(Disp))
            AM.Disp = Disp;
          else
            AM.IndexReg = ShVal;
        } else {
          AM.IndexReg = ShVal;
        }
        return false;
      }
    break;
    }

  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
    // A mul_lohi where we need the low part can be folded as a plain multiply.
    if (N.getResNo() != 0) break;
    // FALL THROUGH
  case ISD::MUL:
  case X86ISD::MUL_IMM:
    // X*[3,5,9] -> X+X*[2,4,8]
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base.Reg.getNode() == 0 &&
        AM.IndexReg.getNode() == 0 &&
        !AM.isRIPRel) {
      if (ConstantSDNode
            *CN = dyn_cast<ConstantSDNode>(N.getNode()->getOperand(1)))
        if (CN->getZExtValue() == 3 || CN->getZExtValue() == 5 ||
            CN->getZExtValue() == 9) {
          AM.Scale = unsigned(CN->getZExtValue())-1;

          SDValue MulVal = N.getNode()->getOperand(0);
          SDValue Reg;

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (MulVal.getNode()->getOpcode() == ISD::ADD && MulVal.hasOneUse() &&
              isa<ConstantSDNode>(MulVal.getNode()->getOperand(1))) {
            Reg = MulVal.getNode()->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(MulVal.getNode()->getOperand(1));
            uint64_t Disp = AM.Disp + AddVal->getSExtValue() *
                                      CN->getZExtValue();
            if (!is64Bit || isInt32(Disp))
              AM.Disp = Disp;
            else
              Reg = N.getNode()->getOperand(0);
          } else {
            Reg = N.getNode()->getOperand(0);
          }

          AM.IndexReg = AM.Base.Reg = Reg;
          return false;
        }
    }
    break;

  case ISD::ADD: {
    X86ISelAddressMode Backup = AM;
    if (!MatchAddress(N.getNode()->getOperand(0), AM, Depth+1) &&
        !MatchAddress(N.getNode()->getOperand(1), AM, Depth+1))
      return false;
    AM = Backup;
    if (!MatchAddress(N.getNode()->getOperand(1), AM, Depth+1) &&
        !MatchAddress(N.getNode()->getOperand(0), AM, Depth+1))
      return false;
    AM = Backup;

    // If we couldn't fold both operands into the address at the same time,
    // see if we can just put each operand into a register and fold at least
    // the add.
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        !AM.Base.Reg.getNode() &&
        !AM.IndexReg.getNode() &&
        !AM.isRIPRel) {
      AM.Base.Reg = N.getNode()->getOperand(0);
      AM.IndexReg = N.getNode()->getOperand(1);
      AM.Scale = 1;
      return false;
    }
    break;
  }

  case ISD::OR:
    // Handle "X | C" as "X + C" iff X is known to have C bits clear.
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      X86ISelAddressMode Backup = AM;
      uint64_t Offset = CN->getSExtValue();
      // Start with the LHS as an addr mode.
      if (!MatchAddress(N.getOperand(0), AM, Depth+1) &&
          // Address could not have picked a GV address for the displacement.
          AM.GV == NULL &&
          // On x86-64, the resultant disp must fit in 32-bits.
          (!is64Bit || isInt32(AM.Disp + Offset)) &&
          // Check to see if the LHS & C is zero.
          CurDAG->MaskedValueIsZero(N.getOperand(0), CN->getAPIntValue())) {
        AM.Disp += Offset;
        return false;
      }
      AM = Backup;
    }
    break;
      
  case ISD::AND: {
    // Perform some heroic transforms on an and of a constant-count shift
    // with a constant to enable use of the scaled offset field.

    SDValue Shift = N.getOperand(0);
    if (Shift.getNumOperands() != 2) break;

    // Scale must not be used already.
    if (AM.IndexReg.getNode() != 0 || AM.Scale != 1) break;

    // Not when RIP is used as the base.
    if (AM.isRIPRel) break;

    SDValue X = Shift.getOperand(0);
    ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N.getOperand(1));
    ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
    if (!C1 || !C2) break;

    // Handle "(X >> (8-C1)) & C2" as "(X >> 8) & 0xff)" if safe. This
    // allows us to convert the shift and and into an h-register extract and
    // a scaled index.
    if (Shift.getOpcode() == ISD::SRL && Shift.hasOneUse()) {
      unsigned ScaleLog = 8 - C1->getZExtValue();
      if (ScaleLog > 0 && ScaleLog < 4 &&
          C2->getZExtValue() == (UINT64_C(0xff) << ScaleLog)) {
        SDValue Eight = CurDAG->getConstant(8, MVT::i8);
        SDValue Mask = CurDAG->getConstant(0xff, N.getValueType());
        SDValue Srl = CurDAG->getNode(ISD::SRL, dl, N.getValueType(),
                                      X, Eight);
        SDValue And = CurDAG->getNode(ISD::AND, dl, N.getValueType(),
                                      Srl, Mask);
        SDValue ShlCount = CurDAG->getConstant(ScaleLog, MVT::i8);
        SDValue Shl = CurDAG->getNode(ISD::SHL, dl, N.getValueType(),
                                      And, ShlCount);

        // Insert the new nodes into the topological ordering.
        if (Eight.getNode()->getNodeId() == -1 ||
            Eight.getNode()->getNodeId() > X.getNode()->getNodeId()) {
          CurDAG->RepositionNode(X.getNode(), Eight.getNode());
          Eight.getNode()->setNodeId(X.getNode()->getNodeId());
        }
        if (Mask.getNode()->getNodeId() == -1 ||
            Mask.getNode()->getNodeId() > X.getNode()->getNodeId()) {
          CurDAG->RepositionNode(X.getNode(), Mask.getNode());
          Mask.getNode()->setNodeId(X.getNode()->getNodeId());
        }
        if (Srl.getNode()->getNodeId() == -1 ||
            Srl.getNode()->getNodeId() > Shift.getNode()->getNodeId()) {
          CurDAG->RepositionNode(Shift.getNode(), Srl.getNode());
          Srl.getNode()->setNodeId(Shift.getNode()->getNodeId());
        }
        if (And.getNode()->getNodeId() == -1 ||
            And.getNode()->getNodeId() > N.getNode()->getNodeId()) {
          CurDAG->RepositionNode(N.getNode(), And.getNode());
          And.getNode()->setNodeId(N.getNode()->getNodeId());
        }
        if (ShlCount.getNode()->getNodeId() == -1 ||
            ShlCount.getNode()->getNodeId() > X.getNode()->getNodeId()) {
          CurDAG->RepositionNode(X.getNode(), ShlCount.getNode());
          ShlCount.getNode()->setNodeId(N.getNode()->getNodeId());
        }
        if (Shl.getNode()->getNodeId() == -1 ||
            Shl.getNode()->getNodeId() > N.getNode()->getNodeId()) {
          CurDAG->RepositionNode(N.getNode(), Shl.getNode());
          Shl.getNode()->setNodeId(N.getNode()->getNodeId());
        }
        CurDAG->ReplaceAllUsesWith(N, Shl);
        AM.IndexReg = And;
        AM.Scale = (1 << ScaleLog);
        return false;
      }
    }

    // Handle "(X << C1) & C2" as "(X & (C2>>C1)) << C1" if safe and if this
    // allows us to fold the shift into this addressing mode.
    if (Shift.getOpcode() != ISD::SHL) break;

    // Not likely to be profitable if either the AND or SHIFT node has more
    // than one use (unless all uses are for address computation). Besides,
    // isel mechanism requires their node ids to be reused.
    if (!N.hasOneUse() || !Shift.hasOneUse())
      break;
    
    // Verify that the shift amount is something we can fold.
    unsigned ShiftCst = C1->getZExtValue();
    if (ShiftCst != 1 && ShiftCst != 2 && ShiftCst != 3)
      break;
    
    // Get the new AND mask, this folds to a constant.
    SDValue NewANDMask = CurDAG->getNode(ISD::SRL, dl, N.getValueType(),
                                         SDValue(C2, 0), SDValue(C1, 0));
    SDValue NewAND = CurDAG->getNode(ISD::AND, dl, N.getValueType(), X, 
                                     NewANDMask);
    SDValue NewSHIFT = CurDAG->getNode(ISD::SHL, dl, N.getValueType(),
                                       NewAND, SDValue(C1, 0));

    // Insert the new nodes into the topological ordering.
    if (C1->getNodeId() > X.getNode()->getNodeId()) {
      CurDAG->RepositionNode(X.getNode(), C1);
      C1->setNodeId(X.getNode()->getNodeId());
    }
    if (NewANDMask.getNode()->getNodeId() == -1 ||
        NewANDMask.getNode()->getNodeId() > X.getNode()->getNodeId()) {
      CurDAG->RepositionNode(X.getNode(), NewANDMask.getNode());
      NewANDMask.getNode()->setNodeId(X.getNode()->getNodeId());
    }
    if (NewAND.getNode()->getNodeId() == -1 ||
        NewAND.getNode()->getNodeId() > Shift.getNode()->getNodeId()) {
      CurDAG->RepositionNode(Shift.getNode(), NewAND.getNode());
      NewAND.getNode()->setNodeId(Shift.getNode()->getNodeId());
    }
    if (NewSHIFT.getNode()->getNodeId() == -1 ||
        NewSHIFT.getNode()->getNodeId() > N.getNode()->getNodeId()) {
      CurDAG->RepositionNode(N.getNode(), NewSHIFT.getNode());
      NewSHIFT.getNode()->setNodeId(N.getNode()->getNodeId());
    }

    CurDAG->ReplaceAllUsesWith(N, NewSHIFT);
    
    AM.Scale = 1 << ShiftCst;
    AM.IndexReg = NewAND;
    return false;
  }
  }

  return MatchAddressBase(N, AM);
}

/// MatchAddressBase - Helper for MatchAddress. Add the specified node to the
/// specified addressing mode without any further recursion.
bool X86DAGToDAGISel::MatchAddressBase(SDValue N, X86ISelAddressMode &AM) {
  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base.Reg.getNode()) {
    // If so, check to see if the scale index register is set.
    if (AM.IndexReg.getNode() == 0 && !AM.isRIPRel) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base.Reg = N;
  return false;
}

/// SelectAddr - returns true if it is able pattern match an addressing mode.
/// It returns the operands which make up the maximal addressing mode it can
/// match by reference.
bool X86DAGToDAGISel::SelectAddr(SDValue Op, SDValue N, SDValue &Base,
                                 SDValue &Scale, SDValue &Index,
                                 SDValue &Disp, SDValue &Segment) {
  X86ISelAddressMode AM;
  bool Done = false;
  if (AvoidDupAddrCompute && !N.hasOneUse()) {
    unsigned Opcode = N.getOpcode();
    if (Opcode != ISD::Constant && Opcode != ISD::FrameIndex &&
        Opcode != X86ISD::Wrapper) {
      // If we are able to fold N into addressing mode, then we'll allow it even
      // if N has multiple uses. In general, addressing computation is used as
      // addresses by all of its uses. But watch out for CopyToReg uses, that
      // means the address computation is liveout. It will be computed by a LEA
      // so we want to avoid computing the address twice.
      for (SDNode::use_iterator UI = N.getNode()->use_begin(),
             UE = N.getNode()->use_end(); UI != UE; ++UI) {
        if (UI->getOpcode() == ISD::CopyToReg) {
          MatchAddressBase(N, AM);
          Done = true;
          break;
        }
      }
    }
  }

  if (!Done && MatchAddress(N, AM))
    return false;

  MVT VT = N.getValueType();
  if (AM.BaseType == X86ISelAddressMode::RegBase) {
    if (!AM.Base.Reg.getNode())
      AM.Base.Reg = CurDAG->getRegister(0, VT);
  }

  if (!AM.IndexReg.getNode())
    AM.IndexReg = CurDAG->getRegister(0, VT);

  getAddressOperands(AM, Base, Scale, Index, Disp, Segment);
  return true;
}

/// SelectScalarSSELoad - Match a scalar SSE load.  In particular, we want to
/// match a load whose top elements are either undef or zeros.  The load flavor
/// is derived from the type of N, which is either v4f32 or v2f64.
bool X86DAGToDAGISel::SelectScalarSSELoad(SDValue Op, SDValue Pred,
                                          SDValue N, SDValue &Base,
                                          SDValue &Scale, SDValue &Index,
                                          SDValue &Disp, SDValue &Segment,
                                          SDValue &InChain,
                                          SDValue &OutChain) {
  if (N.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    InChain = N.getOperand(0).getValue(1);
    if (ISD::isNON_EXTLoad(InChain.getNode()) &&
        InChain.getValue(0).hasOneUse() &&
        N.hasOneUse() &&
        IsLegalAndProfitableToFold(N.getNode(), Pred.getNode(), Op.getNode())) {
      LoadSDNode *LD = cast<LoadSDNode>(InChain);
      if (!SelectAddr(Op, LD->getBasePtr(), Base, Scale, Index, Disp, Segment))
        return false;
      OutChain = LD->getChain();
      return true;
    }
  }

  // Also handle the case where we explicitly require zeros in the top
  // elements.  This is a vector shuffle from the zero vector.
  if (N.getOpcode() == X86ISD::VZEXT_MOVL && N.getNode()->hasOneUse() &&
      // Check to see if the top elements are all zeros (or bitcast of zeros).
      N.getOperand(0).getOpcode() == ISD::SCALAR_TO_VECTOR && 
      N.getOperand(0).getNode()->hasOneUse() &&
      ISD::isNON_EXTLoad(N.getOperand(0).getOperand(0).getNode()) &&
      N.getOperand(0).getOperand(0).hasOneUse()) {
    // Okay, this is a zero extending load.  Fold it.
    LoadSDNode *LD = cast<LoadSDNode>(N.getOperand(0).getOperand(0));
    if (!SelectAddr(Op, LD->getBasePtr(), Base, Scale, Index, Disp, Segment))
      return false;
    OutChain = LD->getChain();
    InChain = SDValue(LD, 1);
    return true;
  }
  return false;
}


/// SelectLEAAddr - it calls SelectAddr and determines if the maximal addressing
/// mode it matches can be cost effectively emitted as an LEA instruction.
bool X86DAGToDAGISel::SelectLEAAddr(SDValue Op, SDValue N,
                                    SDValue &Base, SDValue &Scale,
                                    SDValue &Index, SDValue &Disp) {
  X86ISelAddressMode AM;

  // Set AM.Segment to prevent MatchAddress from using one. LEA doesn't support
  // segments.
  SDValue Copy = AM.Segment;
  SDValue T = CurDAG->getRegister(0, MVT::i32);
  AM.Segment = T;
  if (MatchAddress(N, AM))
    return false;
  assert (T == AM.Segment);
  AM.Segment = Copy;

  MVT VT = N.getValueType();
  unsigned Complexity = 0;
  if (AM.BaseType == X86ISelAddressMode::RegBase)
    if (AM.Base.Reg.getNode())
      Complexity = 1;
    else
      AM.Base.Reg = CurDAG->getRegister(0, VT);
  else if (AM.BaseType == X86ISelAddressMode::FrameIndexBase)
    Complexity = 4;

  if (AM.IndexReg.getNode())
    Complexity++;
  else
    AM.IndexReg = CurDAG->getRegister(0, VT);

  // Don't match just leal(,%reg,2). It's cheaper to do addl %reg, %reg, or with
  // a simple shift.
  if (AM.Scale > 1)
    Complexity++;

  // FIXME: We are artificially lowering the criteria to turn ADD %reg, $GA
  // to a LEA. This is determined with some expermentation but is by no means
  // optimal (especially for code size consideration). LEA is nice because of
  // its three-address nature. Tweak the cost function again when we can run
  // convertToThreeAddress() at register allocation time.
  if (AM.hasSymbolicDisplacement()) {
    // For X86-64, we should always use lea to materialize RIP relative
    // addresses.
    if (Subtarget->is64Bit())
      Complexity = 4;
    else
      Complexity += 2;
  }

  if (AM.Disp && (AM.Base.Reg.getNode() || AM.IndexReg.getNode()))
    Complexity++;

  if (Complexity > 2) {
    SDValue Segment;
    getAddressOperands(AM, Base, Scale, Index, Disp, Segment);
    return true;
  }
  return false;
}

bool X86DAGToDAGISel::TryFoldLoad(SDValue P, SDValue N,
                                  SDValue &Base, SDValue &Scale,
                                  SDValue &Index, SDValue &Disp,
                                  SDValue &Segment) {
  if (ISD::isNON_EXTLoad(N.getNode()) &&
      N.hasOneUse() &&
      IsLegalAndProfitableToFold(N.getNode(), P.getNode(), P.getNode()))
    return SelectAddr(P, N.getOperand(1), Base, Scale, Index, Disp, Segment);
  return false;
}

/// getGlobalBaseReg - Return an SDNode that returns the value of
/// the global base register. Output instructions required to
/// initialize the global base register, if necessary.
///
SDNode *X86DAGToDAGISel::getGlobalBaseReg() {
  MachineFunction *MF = CurBB->getParent();
  unsigned GlobalBaseReg = TM.getInstrInfo()->getGlobalBaseReg(MF);
  return CurDAG->getRegister(GlobalBaseReg, TLI.getPointerTy()).getNode();
}

static SDNode *FindCallStartFromCall(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_START) return Node;
    assert(Node->getOperand(0).getValueType() == MVT::Other &&
         "Node doesn't have a token chain argument!");
  return FindCallStartFromCall(Node->getOperand(0).getNode());
}

SDNode *X86DAGToDAGISel::SelectAtomic64(SDNode *Node, unsigned Opc) {
  SDValue Chain = Node->getOperand(0);
  SDValue In1 = Node->getOperand(1);
  SDValue In2L = Node->getOperand(2);
  SDValue In2H = Node->getOperand(3);
  SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
  if (!SelectAddr(In1, In1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4))
    return NULL;
  SDValue LSI = Node->getOperand(4);    // MemOperand
  const SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, In2L, In2H, LSI, Chain};
  return CurDAG->getTargetNode(Opc, Node->getDebugLoc(),
                               MVT::i32, MVT::i32, MVT::Other, Ops,
                               array_lengthof(Ops));
}

SDNode *X86DAGToDAGISel::Select(SDValue N) {
  SDNode *Node = N.getNode();
  MVT NVT = Node->getValueType(0);
  unsigned Opc, MOpc;
  unsigned Opcode = Node->getOpcode();
  DebugLoc dl = Node->getDebugLoc();
  
#ifndef NDEBUG
  DOUT << std::string(Indent, ' ') << "Selecting: ";
  DEBUG(Node->dump(CurDAG));
  DOUT << "\n";
  Indent += 2;
#endif

  if (Node->isMachineOpcode()) {
#ifndef NDEBUG
    DOUT << std::string(Indent-2, ' ') << "== ";
    DEBUG(Node->dump(CurDAG));
    DOUT << "\n";
    Indent -= 2;
#endif
    return NULL;   // Already selected.
  }

  switch (Opcode) {
    default: break;
    case X86ISD::GlobalBaseReg: 
      return getGlobalBaseReg();

    case X86ISD::ATOMOR64_DAG:
      return SelectAtomic64(Node, X86::ATOMOR6432);
    case X86ISD::ATOMXOR64_DAG:
      return SelectAtomic64(Node, X86::ATOMXOR6432);
    case X86ISD::ATOMADD64_DAG:
      return SelectAtomic64(Node, X86::ATOMADD6432);
    case X86ISD::ATOMSUB64_DAG:
      return SelectAtomic64(Node, X86::ATOMSUB6432);
    case X86ISD::ATOMNAND64_DAG:
      return SelectAtomic64(Node, X86::ATOMNAND6432);
    case X86ISD::ATOMAND64_DAG:
      return SelectAtomic64(Node, X86::ATOMAND6432);
    case X86ISD::ATOMSWAP64_DAG:
      return SelectAtomic64(Node, X86::ATOMSWAP6432);

    case ISD::SMUL_LOHI:
    case ISD::UMUL_LOHI: {
      SDValue N0 = Node->getOperand(0);
      SDValue N1 = Node->getOperand(1);

      bool isSigned = Opcode == ISD::SMUL_LOHI;
      if (!isSigned)
        switch (NVT.getSimpleVT()) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::MUL8r;  MOpc = X86::MUL8m;  break;
        case MVT::i16: Opc = X86::MUL16r; MOpc = X86::MUL16m; break;
        case MVT::i32: Opc = X86::MUL32r; MOpc = X86::MUL32m; break;
        case MVT::i64: Opc = X86::MUL64r; MOpc = X86::MUL64m; break;
        }
      else
        switch (NVT.getSimpleVT()) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::IMUL8r;  MOpc = X86::IMUL8m;  break;
        case MVT::i16: Opc = X86::IMUL16r; MOpc = X86::IMUL16m; break;
        case MVT::i32: Opc = X86::IMUL32r; MOpc = X86::IMUL32m; break;
        case MVT::i64: Opc = X86::IMUL64r; MOpc = X86::IMUL64m; break;
        }

      unsigned LoReg, HiReg;
      switch (NVT.getSimpleVT()) {
      default: assert(0 && "Unsupported VT!");
      case MVT::i8:  LoReg = X86::AL;  HiReg = X86::AH;  break;
      case MVT::i16: LoReg = X86::AX;  HiReg = X86::DX;  break;
      case MVT::i32: LoReg = X86::EAX; HiReg = X86::EDX; break;
      case MVT::i64: LoReg = X86::RAX; HiReg = X86::RDX; break;
      }

      SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
      bool foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
      // multiplty is commmutative
      if (!foldedLoad) {
        foldedLoad = TryFoldLoad(N, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
        if (foldedLoad)
          std::swap(N0, N1);
      }

      SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, LoReg,
                                              N0, SDValue()).getValue(1);

      if (foldedLoad) {
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                          InFlag };
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, dl, MVT::Other, MVT::Flag, Ops,
                                array_lengthof(Ops));
        InFlag = SDValue(CNode, 1);
        // Update the chain.
        ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
      } else {
        InFlag =
          SDValue(CurDAG->getTargetNode(Opc, dl, MVT::Flag, N1, InFlag), 0);
      }

      // Copy the low half of the result, if it is needed.
      if (!N.getValue(0).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                  LoReg, NVT, InFlag);
        InFlag = Result.getValue(2);
        ReplaceUses(N.getValue(0), Result);
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(Result.getNode()->dump(CurDAG));
        DOUT << "\n";
#endif
      }
      // Copy the high half of the result, if it is needed.
      if (!N.getValue(1).use_empty()) {
        SDValue Result;
        if (HiReg == X86::AH && Subtarget->is64Bit()) {
          // Prevent use of AH in a REX instruction by referencing AX instead.
          // Shift it down 8 bits.
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                          X86::AX, MVT::i16, InFlag);
          InFlag = Result.getValue(2);
          Result = SDValue(CurDAG->getTargetNode(X86::SHR16ri, dl, MVT::i16,
                                                 Result,
                                     CurDAG->getTargetConstant(8, MVT::i8)), 0);
          // Then truncate it down to i8.
          SDValue SRIdx = CurDAG->getTargetConstant(X86::SUBREG_8BIT, MVT::i32);
          Result = SDValue(CurDAG->getTargetNode(X86::EXTRACT_SUBREG, dl,
                                                   MVT::i8, Result, SRIdx), 0);
        } else {
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                          HiReg, NVT, InFlag);
          InFlag = Result.getValue(2);
        }
        ReplaceUses(N.getValue(1), Result);
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(Result.getNode()->dump(CurDAG));
        DOUT << "\n";
#endif
      }

#ifndef NDEBUG
      Indent -= 2;
#endif

      return NULL;
    }
      
    case ISD::SDIVREM:
    case ISD::UDIVREM: {
      SDValue N0 = Node->getOperand(0);
      SDValue N1 = Node->getOperand(1);

      bool isSigned = Opcode == ISD::SDIVREM;
      if (!isSigned)
        switch (NVT.getSimpleVT()) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::DIV8r;  MOpc = X86::DIV8m;  break;
        case MVT::i16: Opc = X86::DIV16r; MOpc = X86::DIV16m; break;
        case MVT::i32: Opc = X86::DIV32r; MOpc = X86::DIV32m; break;
        case MVT::i64: Opc = X86::DIV64r; MOpc = X86::DIV64m; break;
        }
      else
        switch (NVT.getSimpleVT()) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::IDIV8r;  MOpc = X86::IDIV8m;  break;
        case MVT::i16: Opc = X86::IDIV16r; MOpc = X86::IDIV16m; break;
        case MVT::i32: Opc = X86::IDIV32r; MOpc = X86::IDIV32m; break;
        case MVT::i64: Opc = X86::IDIV64r; MOpc = X86::IDIV64m; break;
        }

      unsigned LoReg, HiReg;
      unsigned ClrOpcode, SExtOpcode;
      switch (NVT.getSimpleVT()) {
      default: assert(0 && "Unsupported VT!");
      case MVT::i8:
        LoReg = X86::AL;  HiReg = X86::AH;
        ClrOpcode  = 0;
        SExtOpcode = X86::CBW;
        break;
      case MVT::i16:
        LoReg = X86::AX;  HiReg = X86::DX;
        ClrOpcode  = X86::MOV16r0;
        SExtOpcode = X86::CWD;
        break;
      case MVT::i32:
        LoReg = X86::EAX; HiReg = X86::EDX;
        ClrOpcode  = X86::MOV32r0;
        SExtOpcode = X86::CDQ;
        break;
      case MVT::i64:
        LoReg = X86::RAX; HiReg = X86::RDX;
        ClrOpcode  = X86::MOV64r0;
        SExtOpcode = X86::CQO;
        break;
      }

      SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
      bool foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
      bool signBitIsZero = CurDAG->SignBitIsZero(N0);

      SDValue InFlag;
      if (NVT == MVT::i8 && (!isSigned || signBitIsZero)) {
        // Special case for div8, just use a move with zero extension to AX to
        // clear the upper 8 bits (AH).
        SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Move, Chain;
        if (TryFoldLoad(N, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4)) {
          SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N0.getOperand(0) };
          Move =
            SDValue(CurDAG->getTargetNode(X86::MOVZX16rm8, dl, MVT::i16, 
                                          MVT::Other, Ops,
                                          array_lengthof(Ops)), 0);
          Chain = Move.getValue(1);
          ReplaceUses(N0.getValue(1), Chain);
        } else {
          Move =
            SDValue(CurDAG->getTargetNode(X86::MOVZX16rr8, dl, MVT::i16, N0),0);
          Chain = CurDAG->getEntryNode();
        }
        Chain  = CurDAG->getCopyToReg(Chain, dl, X86::AX, Move, SDValue());
        InFlag = Chain.getValue(1);
      } else {
        InFlag =
          CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl,
                               LoReg, N0, SDValue()).getValue(1);
        if (isSigned && !signBitIsZero) {
          // Sign extend the low part into the high part.
          InFlag =
            SDValue(CurDAG->getTargetNode(SExtOpcode, dl, MVT::Flag, InFlag),0);
        } else {
          // Zero out the high part, effectively zero extending the input.
          SDValue ClrNode = SDValue(CurDAG->getTargetNode(ClrOpcode, dl, NVT), 
                                    0);
          InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, HiReg,
                                        ClrNode, InFlag).getValue(1);
        }
      }

      if (foldedLoad) {
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                          InFlag };
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, dl, MVT::Other, MVT::Flag, Ops,
                                array_lengthof(Ops));
        InFlag = SDValue(CNode, 1);
        // Update the chain.
        ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
      } else {
        InFlag =
          SDValue(CurDAG->getTargetNode(Opc, dl, MVT::Flag, N1, InFlag), 0);
      }

      // Copy the division (low) result, if it is needed.
      if (!N.getValue(0).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                  LoReg, NVT, InFlag);
        InFlag = Result.getValue(2);
        ReplaceUses(N.getValue(0), Result);
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(Result.getNode()->dump(CurDAG));
        DOUT << "\n";
#endif
      }
      // Copy the remainder (high) result, if it is needed.
      if (!N.getValue(1).use_empty()) {
        SDValue Result;
        if (HiReg == X86::AH && Subtarget->is64Bit()) {
          // Prevent use of AH in a REX instruction by referencing AX instead.
          // Shift it down 8 bits.
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                          X86::AX, MVT::i16, InFlag);
          InFlag = Result.getValue(2);
          Result = SDValue(CurDAG->getTargetNode(X86::SHR16ri, dl, MVT::i16,
                                        Result,
                                        CurDAG->getTargetConstant(8, MVT::i8)), 
                           0);
          // Then truncate it down to i8.
          SDValue SRIdx = CurDAG->getTargetConstant(X86::SUBREG_8BIT, MVT::i32);
          Result = SDValue(CurDAG->getTargetNode(X86::EXTRACT_SUBREG, dl,
                                                   MVT::i8, Result, SRIdx), 0);
        } else {
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                          HiReg, NVT, InFlag);
          InFlag = Result.getValue(2);
        }
        ReplaceUses(N.getValue(1), Result);
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(Result.getNode()->dump(CurDAG));
        DOUT << "\n";
#endif
      }

#ifndef NDEBUG
      Indent -= 2;
#endif

      return NULL;
    }

    case ISD::DECLARE: {
      // Handle DECLARE nodes here because the second operand may have been
      // wrapped in X86ISD::Wrapper.
      SDValue Chain = Node->getOperand(0);
      SDValue N1 = Node->getOperand(1);
      SDValue N2 = Node->getOperand(2);
      FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(N1);
      
      // FIXME: We need to handle this for VLAs.
      if (!FINode) {
        ReplaceUses(N.getValue(0), Chain);
        return NULL;
      }
      
      if (N2.getOpcode() == ISD::ADD &&
          N2.getOperand(0).getOpcode() == X86ISD::GlobalBaseReg)
        N2 = N2.getOperand(1);
      
      // If N2 is not Wrapper(decriptor) then the llvm.declare is mangled
      // somehow, just ignore it.
      if (N2.getOpcode() != X86ISD::Wrapper) {
        ReplaceUses(N.getValue(0), Chain);
        return NULL;
      }
      GlobalAddressSDNode *GVNode =
        dyn_cast<GlobalAddressSDNode>(N2.getOperand(0));
      if (GVNode == 0) {
        ReplaceUses(N.getValue(0), Chain);
        return NULL;
      }
      SDValue Tmp1 = CurDAG->getTargetFrameIndex(FINode->getIndex(),
                                                 TLI.getPointerTy());
      SDValue Tmp2 = CurDAG->getTargetGlobalAddress(GVNode->getGlobal(),
                                                    TLI.getPointerTy());
      SDValue Ops[] = { Tmp1, Tmp2, Chain };
      return CurDAG->getTargetNode(TargetInstrInfo::DECLARE, dl,
                                   MVT::Other, Ops,
                                   array_lengthof(Ops));
    }
  }

  SDNode *ResNode = SelectCode(N);

#ifndef NDEBUG
  DOUT << std::string(Indent-2, ' ') << "=> ";
  if (ResNode == NULL || ResNode == N.getNode())
    DEBUG(N.getNode()->dump(CurDAG));
  else
    DEBUG(ResNode->dump(CurDAG));
  DOUT << "\n";
  Indent -= 2;
#endif

  return ResNode;
}

bool X86DAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  SDValue Op0, Op1, Op2, Op3, Op4;
  switch (ConstraintCode) {
  case 'o':   // offsetable        ??
  case 'v':   // not offsetable    ??
  default: return true;
  case 'm':   // memory
    if (!SelectAddr(Op, Op, Op0, Op1, Op2, Op3, Op4))
      return true;
    break;
  }
  
  OutOps.push_back(Op0);
  OutOps.push_back(Op1);
  OutOps.push_back(Op2);
  OutOps.push_back(Op3);
  OutOps.push_back(Op4);
  return false;
}

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createX86ISelDag(X86TargetMachine &TM,
                                     llvm::CodeGenOpt::Level OptLevel) {
  return new X86DAGToDAGISel(TM, OptLevel);
}
