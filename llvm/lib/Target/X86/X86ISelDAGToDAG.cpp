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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include <queue>
#include <set>
using namespace llvm;

STATISTIC(NumFPKill   , "Number of FP_REG_KILL instructions added");
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
    unsigned Disp;
    GlobalValue *GV;
    Constant *CP;
    const char *ES;
    int JT;
    unsigned Align;    // CP alignment.

    X86ISelAddressMode()
      : BaseType(RegBase), isRIPRel(false), Scale(1), IndexReg(), Disp(0),
        GV(0), CP(0), ES(0), JT(-1), Align(0) {
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
    /// ContainsFPCode - Every instruction we select that uses or defines a FP
    /// register should set this to true.
    bool ContainsFPCode;

    /// TM - Keep a reference to X86TargetMachine.
    ///
    X86TargetMachine &TM;

    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    X86TargetLowering X86Lowering;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// GlobalBaseReg - keeps track of the virtual register mapped onto global
    /// base register.
    unsigned GlobalBaseReg;

    /// CurBB - Current BB being isel'd.
    ///
    MachineBasicBlock *CurBB;

  public:
    X86DAGToDAGISel(X86TargetMachine &tm, bool fast)
      : SelectionDAGISel(X86Lowering, fast),
        ContainsFPCode(false), TM(tm),
        X86Lowering(*TM.getTargetLowering()),
        Subtarget(&TM.getSubtarget<X86Subtarget>()) {}

    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      return SelectionDAGISel::runOnFunction(Fn);
    }
   
    virtual const char *getPassName() const {
      return "X86 DAG->DAG Instruction Selection";
    }

    /// InstructionSelect - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelect();

    /// InstructionSelectPostProcessing - Post processing of selected and
    /// scheduled basic blocks.
    virtual void InstructionSelectPostProcessing();

    virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF);

    virtual bool CanBeFoldedBy(SDNode *N, SDNode *U, SDNode *Root) const;

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    SDNode *Select(SDValue N);

    bool MatchAddress(SDValue N, X86ISelAddressMode &AM,
                      bool isRoot = true, unsigned Depth = 0);
    bool MatchAddressBase(SDValue N, X86ISelAddressMode &AM,
                          bool isRoot, unsigned Depth);
    bool SelectAddr(SDValue Op, SDValue N, SDValue &Base,
                    SDValue &Scale, SDValue &Index, SDValue &Disp);
    bool SelectLEAAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Scale, SDValue &Index, SDValue &Disp);
    bool SelectScalarSSELoad(SDValue Op, SDValue Pred,
                             SDValue N, SDValue &Base, SDValue &Scale,
                             SDValue &Index, SDValue &Disp,
                             SDValue &InChain, SDValue &OutChain);
    bool TryFoldLoad(SDValue P, SDValue N,
                     SDValue &Base, SDValue &Scale,
                     SDValue &Index, SDValue &Disp);
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
                                   SDValue &Disp) {
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
        Disp = getI32Imm(AM.Disp);
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

    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDNode *getGlobalBaseReg();

    /// getTruncateTo8Bit - return an SDNode that implements a subreg based
    /// truncate of the specified operand to i8. This can be done with tablegen,
    /// except that this code uses MVT::Flag in a tricky way that happens to
    /// improve scheduling in some cases.
    SDNode *getTruncateTo8Bit(SDValue N0);

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
    SDNode *User = *I;
    for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
      SDValue Op = User->getOperand(i);
      if (Op.getNode() == N && Op.getResNo() == FlagResNo)
        return User;
    }
  }
  return NULL;
}

/// findNonImmUse - Return true by reference in "found" if "Use" is an
/// non-immediate use of "Def". This function recursively traversing
/// up the operand chain ignoring certain nodes.
static void findNonImmUse(SDNode *Use, SDNode* Def, SDNode *ImmedUse,
                          SDNode *Root, bool &found,
                          SmallPtrSet<SDNode*, 16> &Visited) {
  if (found ||
      Use->getNodeId() > Def->getNodeId() ||
      !Visited.insert(Use))
    return;
  
  for (unsigned i = 0, e = Use->getNumOperands(); !found && i != e; ++i) {
    SDNode *N = Use->getOperand(i).getNode();
    if (N == Def) {
      if (Use == ImmedUse || Use == Root)
        continue;  // We are not looking for immediate use.
      assert(N != Root);
      found = true;
      break;
    }

    // Traverse up the operand chain.
    findNonImmUse(N, Def, ImmedUse, Root, found, Visited);
  }
}

/// isNonImmUse - Start searching from Root up the DAG to check is Def can
/// be reached. Return true if that's the case. However, ignore direct uses
/// by ImmedUse (which would be U in the example illustrated in
/// CanBeFoldedBy) and by Root (which can happen in the store case).
/// FIXME: to be really generic, we should allow direct use by any node
/// that is being folded. But realisticly since we only fold loads which
/// have one non-chain use, we only need to watch out for load/op/store
/// and load/op/cmp case where the root (store / cmp) may reach the load via
/// its chain operand.
static inline bool isNonImmUse(SDNode *Root, SDNode *Def, SDNode *ImmedUse) {
  SmallPtrSet<SDNode*, 16> Visited;
  bool found = false;
  findNonImmUse(Root, Def, ImmedUse, Root, found, Visited);
  return found;
}


bool X86DAGToDAGISel::CanBeFoldedBy(SDNode *N, SDNode *U, SDNode *Root) const {
  if (Fast) return false;

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
                           SDValue Call, SDValue Chain) {
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0, e = Chain.getNode()->getNumOperands(); i != e; ++i)
    if (Load.getNode() == Chain.getOperand(i).getNode())
      Ops.push_back(Load.getOperand(0));
    else
      Ops.push_back(Chain.getOperand(i));
  CurDAG->UpdateNodeOperands(Chain, &Ops[0], Ops.size());
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
  return Chain.getOperand(0).getNode() == Callee.getNode();
}


/// PreprocessForRMW - Preprocess the DAG to make instruction selection better.
/// This is only run if not in -fast mode (aka -O0).
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
    
    // FIXME: optimize the case where the src/dest is a load or store?
    SDValue Store = CurDAG->getTruncStore(CurDAG->getEntryNode(),
                                          N->getOperand(0),
                                          MemTmp, NULL, 0, MemVT);
    SDValue Result = CurDAG->getExtLoad(ISD::EXTLOAD, DstVT, Store, MemTmp,
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

  DEBUG(BB->dump());
  if (!Fast)
    PreprocessForRMW();

  // FIXME: This should only happen when not -fast.
  PreprocessForFPConvert();

  // Codegen the basic block.
#ifndef NDEBUG
  DOUT << "===== Instruction selection begins:\n";
  Indent = 0;
#endif
  SelectRoot();
#ifndef NDEBUG
  DOUT << "===== Instruction selection ends:\n";
#endif

  CurDAG->RemoveDeadNodes();
}

void X86DAGToDAGISel::InstructionSelectPostProcessing() {
  // If we are emitting FP stack code, scan the basic block to determine if this
  // block defines any FP values.  If so, put an FP_REG_KILL instruction before
  // the terminator of the block.

  // Note that FP stack instructions are used in all modes for long double,
  // so we always need to do this check.
  // Also note that it's possible for an FP stack register to be live across
  // an instruction that produces multiple basic blocks (SSE CMOV) so we
  // must check all the generated basic blocks.

  // Scan all of the machine instructions in these MBBs, checking for FP
  // stores.  (RFP32 and RFP64 will not exist in SSE mode, but RFP80 might.)
  MachineFunction::iterator MBBI = CurBB;
  MachineFunction::iterator EndMBB = BB; ++EndMBB;
  for (; MBBI != EndMBB; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    
    // If this block returns, ignore it.  We don't want to insert an FP_REG_KILL
    // before the return.
    if (!MBB->empty()) {
      MachineBasicBlock::iterator EndI = MBB->end();
      --EndI;
      if (EndI->getDesc().isReturn())
        continue;
    }
    
    bool ContainsFPCode = false;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         !ContainsFPCode && I != E; ++I) {
      if (I->getNumOperands() != 0 && I->getOperand(0).isRegister()) {
        const TargetRegisterClass *clas;
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
          if (I->getOperand(op).isRegister() && I->getOperand(op).isDef() &&
            TargetRegisterInfo::isVirtualRegister(I->getOperand(op).getReg()) &&
              ((clas = RegInfo->getRegClass(I->getOperand(0).getReg())) == 
                 X86::RFP32RegisterClass ||
               clas == X86::RFP64RegisterClass ||
               clas == X86::RFP80RegisterClass)) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    }
    // Check PHI nodes in successor blocks.  These PHI's will be lowered to have
    // a copy of the input value in this block.  In SSE mode, we only care about
    // 80-bit values.
    if (!ContainsFPCode) {
      // Final check, check LLVM BB's that are successors to the LLVM BB
      // corresponding to BB for FP PHI nodes.
      const BasicBlock *LLVMBB = BB->getBasicBlock();
      const PHINode *PN;
      for (succ_const_iterator SI = succ_begin(LLVMBB), E = succ_end(LLVMBB);
           !ContainsFPCode && SI != E; ++SI) {
        for (BasicBlock::const_iterator II = SI->begin();
             (PN = dyn_cast<PHINode>(II)); ++II) {
          if (PN->getType()==Type::X86_FP80Ty ||
              (!Subtarget->hasSSE1() && PN->getType()->isFloatingPoint()) ||
              (!Subtarget->hasSSE2() && PN->getType()==Type::DoubleTy)) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    }
    // Finally, if we found any FP code, emit the FP_REG_KILL instruction.
    if (ContainsFPCode) {
      BuildMI(*MBB, MBBI->getFirstTerminator(),
              TM.getInstrInfo()->get(X86::FP_REG_KILL));
      ++NumFPKill;
    }
  }
}

/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
void X86DAGToDAGISel::EmitSpecialCodeForMain(MachineBasicBlock *BB,
                                             MachineFrameInfo *MFI) {
  const TargetInstrInfo *TII = TM.getInstrInfo();
  if (Subtarget->isTargetCygMing())
    BuildMI(BB, TII->get(X86::CALLpcrel32)).addExternalSymbol("__main");
}

void X86DAGToDAGISel::EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {
  // If this is main, emit special code for main.
  MachineBasicBlock *BB = MF.begin();
  if (Fn.hasExternalLinkage() && Fn.getName() == "main")
    EmitSpecialCodeForMain(BB, MF.getFrameInfo());
}

/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode.
bool X86DAGToDAGISel::MatchAddress(SDValue N, X86ISelAddressMode &AM,
                                   bool isRoot, unsigned Depth) {
DOUT << "MatchAddress: "; DEBUG(AM.dump());
  // Limit recursion.
  if (Depth > 5)
    return MatchAddressBase(N, AM, isRoot, Depth);
  
  // RIP relative addressing: %rip + 32-bit displacement!
  if (AM.isRIPRel) {
    if (!AM.ES && AM.JT != -1 && N.getOpcode() == ISD::Constant) {
      int64_t Val = cast<ConstantSDNode>(N)->getSignExtended();
      if (isInt32(AM.Disp + Val)) {
        AM.Disp += Val;
        return false;
      }
    }
    return true;
  }

  int id = N.getNode()->getNodeId();
  bool AlreadySelected = isSelected(id); // Already selected, not yet replaced.

  switch (N.getOpcode()) {
  default: break;
  case ISD::Constant: {
    int64_t Val = cast<ConstantSDNode>(N)->getSignExtended();
    if (isInt32(AM.Disp + Val)) {
      AM.Disp += Val;
      return false;
    }
    break;
  }

  case X86ISD::Wrapper: {
DOUT << "Wrapper: 64bit " << Subtarget->is64Bit();
DOUT << " AM "; DEBUG(AM.dump()); DOUT << "\n";
DOUT << "AlreadySelected " << AlreadySelected << "\n";
    bool is64Bit = Subtarget->is64Bit();
    // Under X86-64 non-small code model, GV (and friends) are 64-bits.
    // Also, base and index reg must be 0 in order to use rip as base.
    if (is64Bit && (TM.getCodeModel() != CodeModel::Small ||
                    AM.Base.Reg.getNode() || AM.IndexReg.getNode()))
      break;
    if (AM.GV != 0 || AM.CP != 0 || AM.ES != 0 || AM.JT != -1)
      break;
    // If value is available in a register both base and index components have
    // been picked, we can't fit the result available in the register in the
    // addressing mode. Duplicate GlobalAddress or ConstantPool as displacement.
    if (!AlreadySelected || (AM.Base.Reg.getNode() && AM.IndexReg.getNode())) {
      SDValue N0 = N.getOperand(0);
      if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
        GlobalValue *GV = G->getGlobal();
        AM.GV = GV;
        AM.Disp += G->getOffset();
        AM.isRIPRel = TM.getRelocationModel() != Reloc::Static &&
          Subtarget->isPICStyleRIPRel();
        return false;
      } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
        AM.CP = CP->getConstVal();
        AM.Align = CP->getAlignment();
        AM.Disp += CP->getOffset();
        AM.isRIPRel = TM.getRelocationModel() != Reloc::Static &&
          Subtarget->isPICStyleRIPRel();
        return false;
      } else if (ExternalSymbolSDNode *S =dyn_cast<ExternalSymbolSDNode>(N0)) {
        AM.ES = S->getSymbol();
        AM.isRIPRel = TM.getRelocationModel() != Reloc::Static &&
          Subtarget->isPICStyleRIPRel();
        return false;
      } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
        AM.JT = J->getIndex();
        AM.isRIPRel = TM.getRelocationModel() != Reloc::Static &&
          Subtarget->isPICStyleRIPRel();
        return false;
      }
    }
    break;
  }

  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase
        && AM.Base.Reg.getNode() == 0) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::SHL:
    if (AlreadySelected || AM.IndexReg.getNode() != 0
        || AM.Scale != 1 || AM.isRIPRel)
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
          uint64_t Disp = AM.Disp + (AddVal->getZExtValue() << Val);
          if (isInt32(Disp))
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
    // X*[3,5,9] -> X+X*[2,4,8]
    if (!AlreadySelected &&
        AM.BaseType == X86ISelAddressMode::RegBase &&
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
            uint64_t Disp = AM.Disp + AddVal->getZExtValue() *
                                      CN->getZExtValue();
            if (isInt32(Disp))
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

  case ISD::ADD:
    if (!AlreadySelected) {
      X86ISelAddressMode Backup = AM;
      if (!MatchAddress(N.getNode()->getOperand(0), AM, false, Depth+1) &&
          !MatchAddress(N.getNode()->getOperand(1), AM, false, Depth+1))
        return false;
      AM = Backup;
      if (!MatchAddress(N.getNode()->getOperand(1), AM, false, Depth+1) &&
          !MatchAddress(N.getNode()->getOperand(0), AM, false, Depth+1))
        return false;
      AM = Backup;
    }
    break;

  case ISD::OR:
    // Handle "X | C" as "X + C" iff X is known to have C bits clear.
    if (AlreadySelected) break;
      
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      X86ISelAddressMode Backup = AM;
      // Start with the LHS as an addr mode.
      if (!MatchAddress(N.getOperand(0), AM, false) &&
          // Address could not have picked a GV address for the displacement.
          AM.GV == NULL &&
          // On x86-64, the resultant disp must fit in 32-bits.
          isInt32(AM.Disp + CN->getSignExtended()) &&
          // Check to see if the LHS & C is zero.
          CurDAG->MaskedValueIsZero(N.getOperand(0), CN->getAPIntValue())) {
        AM.Disp += CN->getZExtValue();
        return false;
      }
      AM = Backup;
    }
    break;
      
  case ISD::AND: {
    // Handle "(x << C1) & C2" as "(X & (C2>>C1)) << C1" if safe and if this
    // allows us to fold the shift into this addressing mode.
    if (AlreadySelected) break;
    SDValue Shift = N.getOperand(0);
    if (Shift.getOpcode() != ISD::SHL) break;
    
    // Scale must not be used already.
    if (AM.IndexReg.getNode() != 0 || AM.Scale != 1) break;

    // Not when RIP is used as the base.
    if (AM.isRIPRel) break;
      
    ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N.getOperand(1));
    ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
    if (!C1 || !C2) break;

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
    SDValue NewANDMask = CurDAG->getNode(ISD::SRL, N.getValueType(),
                                           SDValue(C2, 0), SDValue(C1, 0));
    SDValue NewAND = CurDAG->getNode(ISD::AND, N.getValueType(),
                                       Shift.getOperand(0), NewANDMask);
    NewANDMask.getNode()->setNodeId(Shift.getNode()->getNodeId());
    NewAND.getNode()->setNodeId(N.getNode()->getNodeId());
    
    AM.Scale = 1 << ShiftCst;
    AM.IndexReg = NewAND;
    return false;
  }
  }

  return MatchAddressBase(N, AM, isRoot, Depth);
}

/// MatchAddressBase - Helper for MatchAddress. Add the specified node to the
/// specified addressing mode without any further recursion.
bool X86DAGToDAGISel::MatchAddressBase(SDValue N, X86ISelAddressMode &AM,
                                       bool isRoot, unsigned Depth) {
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
                                 SDValue &Disp) {
  X86ISelAddressMode AM;
  if (MatchAddress(N, AM))
    return false;

  MVT VT = N.getValueType();
  if (AM.BaseType == X86ISelAddressMode::RegBase) {
    if (!AM.Base.Reg.getNode())
      AM.Base.Reg = CurDAG->getRegister(0, VT);
  }

  if (!AM.IndexReg.getNode())
    AM.IndexReg = CurDAG->getRegister(0, VT);

  getAddressOperands(AM, Base, Scale, Index, Disp);
  return true;
}

/// isZeroNode - Returns true if Elt is a constant zero or a floating point
/// constant +0.0.
static inline bool isZeroNode(SDValue Elt) {
  return ((isa<ConstantSDNode>(Elt) &&
  cast<ConstantSDNode>(Elt)->getZExtValue() == 0) ||
  (isa<ConstantFPSDNode>(Elt) &&
  cast<ConstantFPSDNode>(Elt)->getValueAPF().isPosZero()));
}


/// SelectScalarSSELoad - Match a scalar SSE load.  In particular, we want to
/// match a load whose top elements are either undef or zeros.  The load flavor
/// is derived from the type of N, which is either v4f32 or v2f64.
bool X86DAGToDAGISel::SelectScalarSSELoad(SDValue Op, SDValue Pred,
                                          SDValue N, SDValue &Base,
                                          SDValue &Scale, SDValue &Index,
                                          SDValue &Disp, SDValue &InChain,
                                          SDValue &OutChain) {
  if (N.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    InChain = N.getOperand(0).getValue(1);
    if (ISD::isNON_EXTLoad(InChain.getNode()) &&
        InChain.getValue(0).hasOneUse() &&
        N.hasOneUse() &&
        CanBeFoldedBy(N.getNode(), Pred.getNode(), Op.getNode())) {
      LoadSDNode *LD = cast<LoadSDNode>(InChain);
      if (!SelectAddr(Op, LD->getBasePtr(), Base, Scale, Index, Disp))
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
    if (!SelectAddr(Op, LD->getBasePtr(), Base, Scale, Index, Disp))
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
  if (MatchAddress(N, AM))
    return false;

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
  if (AM.GV || AM.CP || AM.ES || AM.JT != -1) {
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
    getAddressOperands(AM, Base, Scale, Index, Disp);
    return true;
  }
  return false;
}

bool X86DAGToDAGISel::TryFoldLoad(SDValue P, SDValue N,
                                  SDValue &Base, SDValue &Scale,
                                  SDValue &Index, SDValue &Disp) {
  if (ISD::isNON_EXTLoad(N.getNode()) &&
      N.hasOneUse() &&
      CanBeFoldedBy(N.getNode(), P.getNode(), P.getNode()))
    return SelectAddr(P, N.getOperand(1), Base, Scale, Index, Disp);
  return false;
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDNode *X86DAGToDAGISel::getGlobalBaseReg() {
  assert(!Subtarget->is64Bit() && "X86-64 PIC uses RIP relative addressing");
  if (!GlobalBaseReg) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineFunction *MF = BB->getParent();
    MachineBasicBlock &FirstMBB = MF->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    MachineRegisterInfo &RegInfo = MF->getRegInfo();
    unsigned PC = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
    
    const TargetInstrInfo *TII = TM.getInstrInfo();
    // Operand of MovePCtoStack is completely ignored by asm printer. It's
    // only used in JIT code emission as displacement to pc.
    BuildMI(FirstMBB, MBBI, TII->get(X86::MOVPC32r), PC).addImm(0);
    
    // If we're using vanilla 'GOT' PIC style, we should use relative addressing
    // not to pc, but to _GLOBAL_ADDRESS_TABLE_ external
    if (TM.getRelocationModel() == Reloc::PIC_ &&
        Subtarget->isPICStyleGOT()) {
      GlobalBaseReg = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
      BuildMI(FirstMBB, MBBI, TII->get(X86::ADD32ri), GlobalBaseReg)
        .addReg(PC).addExternalSymbol("_GLOBAL_OFFSET_TABLE_");
    } else {
      GlobalBaseReg = PC;
    }
    
  }
  return CurDAG->getRegister(GlobalBaseReg, TLI.getPointerTy()).getNode();
}

static SDNode *FindCallStartFromCall(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_START) return Node;
    assert(Node->getOperand(0).getValueType() == MVT::Other &&
         "Node doesn't have a token chain argument!");
  return FindCallStartFromCall(Node->getOperand(0).getNode());
}

/// getTruncateTo8Bit - return an SDNode that implements a subreg based
/// truncate of the specified operand to i8. This can be done with tablegen,
/// except that this code uses MVT::Flag in a tricky way that happens to
/// improve scheduling in some cases.
SDNode *X86DAGToDAGISel::getTruncateTo8Bit(SDValue N0) {
  assert(!Subtarget->is64Bit() &&
         "getTruncateTo8Bit is only needed on x86-32!");
  SDValue SRIdx = CurDAG->getTargetConstant(1, MVT::i32); // SubRegSet 1

  // Ensure that the source register has an 8-bit subreg on 32-bit targets
  unsigned Opc;
  MVT N0VT = N0.getValueType();
  switch (N0VT.getSimpleVT()) {
  default: assert(0 && "Unknown truncate!");
  case MVT::i16:
    Opc = X86::MOV16to16_;
    break;
  case MVT::i32:
    Opc = X86::MOV32to32_;
    break;
  }

  // The use of MVT::Flag here is not strictly accurate, but it helps
  // scheduling in some cases.
  N0 = SDValue(CurDAG->getTargetNode(Opc, N0VT, MVT::Flag, N0), 0);
  return CurDAG->getTargetNode(X86::EXTRACT_SUBREG,
                               MVT::i8, N0, SRIdx, N0.getValue(1));
}


SDNode *X86DAGToDAGISel::Select(SDValue N) {
  SDNode *Node = N.getNode();
  MVT NVT = Node->getValueType(0);
  unsigned Opc, MOpc;
  unsigned Opcode = Node->getOpcode();

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

    case ISD::ADD: {
      // Turn ADD X, c to MOV32ri X+c. This cannot be done with tblgen'd
      // code and is matched first so to prevent it from being turned into
      // LEA32r X+c.
      // In 64-bit small code size mode, use LEA to take advantage of
      // RIP-relative addressing.
      if (TM.getCodeModel() != CodeModel::Small)
        break;
      MVT PtrVT = TLI.getPointerTy();
      SDValue N0 = N.getOperand(0);
      SDValue N1 = N.getOperand(1);
      if (N.getNode()->getValueType(0) == PtrVT &&
          N0.getOpcode() == X86ISD::Wrapper &&
          N1.getOpcode() == ISD::Constant) {
        unsigned Offset = (unsigned)cast<ConstantSDNode>(N1)->getZExtValue();
        SDValue C(0, 0);
        // TODO: handle ExternalSymbolSDNode.
        if (GlobalAddressSDNode *G =
            dyn_cast<GlobalAddressSDNode>(N0.getOperand(0))) {
          C = CurDAG->getTargetGlobalAddress(G->getGlobal(), PtrVT,
                                             G->getOffset() + Offset);
        } else if (ConstantPoolSDNode *CP =
                   dyn_cast<ConstantPoolSDNode>(N0.getOperand(0))) {
          C = CurDAG->getTargetConstantPool(CP->getConstVal(), PtrVT,
                                            CP->getAlignment(),
                                            CP->getOffset()+Offset);
        }

        if (C.getNode()) {
          if (Subtarget->is64Bit()) {
            SDValue Ops[] = { CurDAG->getRegister(0, PtrVT), getI8Imm(1),
                                CurDAG->getRegister(0, PtrVT), C };
            return CurDAG->SelectNodeTo(N.getNode(), X86::LEA64r,
                                        MVT::i64, Ops, 4);
          } else
            return CurDAG->SelectNodeTo(N.getNode(), X86::MOV32ri, PtrVT, C);
        }
      }

      // Other cases are handled by auto-generated code.
      break;
    }

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

      SDValue Tmp0, Tmp1, Tmp2, Tmp3;
      bool foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3);
      // multiplty is commmutative
      if (!foldedLoad) {
        foldedLoad = TryFoldLoad(N, N0, Tmp0, Tmp1, Tmp2, Tmp3);
        if (foldedLoad)
          std::swap(N0, N1);
      }

      AddToISelQueue(N0);
      SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), LoReg,
                                              N0, SDValue()).getValue(1);

      if (foldedLoad) {
        AddToISelQueue(N1.getOperand(0));
        AddToISelQueue(Tmp0);
        AddToISelQueue(Tmp1);
        AddToISelQueue(Tmp2);
        AddToISelQueue(Tmp3);
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, N1.getOperand(0), InFlag };
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, MVT::Other, MVT::Flag, Ops, 6);
        InFlag = SDValue(CNode, 1);
        // Update the chain.
        ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
      } else {
        AddToISelQueue(N1);
        InFlag =
          SDValue(CurDAG->getTargetNode(Opc, MVT::Flag, N1, InFlag), 0);
      }

      // Copy the low half of the result, if it is needed.
      if (!N.getValue(0).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
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
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
                                          X86::AX, MVT::i16, InFlag);
          InFlag = Result.getValue(2);
          Result = SDValue(CurDAG->getTargetNode(X86::SHR16ri, MVT::i16, Result,
                                     CurDAG->getTargetConstant(8, MVT::i8)), 0);
          // Then truncate it down to i8.
          SDValue SRIdx = CurDAG->getTargetConstant(1, MVT::i32); // SubRegSet 1
          Result = SDValue(CurDAG->getTargetNode(X86::EXTRACT_SUBREG,
                                                   MVT::i8, Result, SRIdx), 0);
        } else {
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
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

      SDValue Tmp0, Tmp1, Tmp2, Tmp3;
      bool foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3);

      SDValue InFlag;
      if (NVT == MVT::i8 && !isSigned) {
        // Special case for div8, just use a move with zero extension to AX to
        // clear the upper 8 bits (AH).
        SDValue Tmp0, Tmp1, Tmp2, Tmp3, Move, Chain;
        if (TryFoldLoad(N, N0, Tmp0, Tmp1, Tmp2, Tmp3)) {
          SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, N0.getOperand(0) };
          AddToISelQueue(N0.getOperand(0));
          AddToISelQueue(Tmp0);
          AddToISelQueue(Tmp1);
          AddToISelQueue(Tmp2);
          AddToISelQueue(Tmp3);
          Move =
            SDValue(CurDAG->getTargetNode(X86::MOVZX16rm8, MVT::i16, MVT::Other,
                                            Ops, 5), 0);
          Chain = Move.getValue(1);
          ReplaceUses(N0.getValue(1), Chain);
        } else {
          AddToISelQueue(N0);
          Move =
            SDValue(CurDAG->getTargetNode(X86::MOVZX16rr8, MVT::i16, N0), 0);
          Chain = CurDAG->getEntryNode();
        }
        Chain  = CurDAG->getCopyToReg(Chain, X86::AX, Move, SDValue());
        InFlag = Chain.getValue(1);
      } else {
        AddToISelQueue(N0);
        InFlag =
          CurDAG->getCopyToReg(CurDAG->getEntryNode(),
                               LoReg, N0, SDValue()).getValue(1);
        if (isSigned) {
          // Sign extend the low part into the high part.
          InFlag =
            SDValue(CurDAG->getTargetNode(SExtOpcode, MVT::Flag, InFlag), 0);
        } else {
          // Zero out the high part, effectively zero extending the input.
          SDValue ClrNode = SDValue(CurDAG->getTargetNode(ClrOpcode, NVT), 0);
          InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), HiReg,
                                        ClrNode, InFlag).getValue(1);
        }
      }

      if (foldedLoad) {
        AddToISelQueue(N1.getOperand(0));
        AddToISelQueue(Tmp0);
        AddToISelQueue(Tmp1);
        AddToISelQueue(Tmp2);
        AddToISelQueue(Tmp3);
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, N1.getOperand(0), InFlag };
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, MVT::Other, MVT::Flag, Ops, 6);
        InFlag = SDValue(CNode, 1);
        // Update the chain.
        ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
      } else {
        AddToISelQueue(N1);
        InFlag =
          SDValue(CurDAG->getTargetNode(Opc, MVT::Flag, N1, InFlag), 0);
      }

      // Copy the division (low) result, if it is needed.
      if (!N.getValue(0).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
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
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
                                          X86::AX, MVT::i16, InFlag);
          InFlag = Result.getValue(2);
          Result = SDValue(CurDAG->getTargetNode(X86::SHR16ri, MVT::i16, Result,
                                     CurDAG->getTargetConstant(8, MVT::i8)), 0);
          // Then truncate it down to i8.
          SDValue SRIdx = CurDAG->getTargetConstant(1, MVT::i32); // SubRegSet 1
          Result = SDValue(CurDAG->getTargetNode(X86::EXTRACT_SUBREG,
                                                   MVT::i8, Result, SRIdx), 0);
        } else {
          Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
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

    case ISD::SIGN_EXTEND_INREG: {
      MVT SVT = cast<VTSDNode>(Node->getOperand(1))->getVT();
      if (SVT == MVT::i8 && !Subtarget->is64Bit()) {
        SDValue N0 = Node->getOperand(0);
        AddToISelQueue(N0);
      
        SDValue TruncOp = SDValue(getTruncateTo8Bit(N0), 0);
        unsigned Opc = 0;
        switch (NVT.getSimpleVT()) {
        default: assert(0 && "Unknown sign_extend_inreg!");
        case MVT::i16:
          Opc = X86::MOVSX16rr8;
          break;
        case MVT::i32:
          Opc = X86::MOVSX32rr8; 
          break;
        }
      
        SDNode *ResNode = CurDAG->getTargetNode(Opc, NVT, TruncOp);
      
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(TruncOp.getNode()->dump(CurDAG));
        DOUT << "\n";
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(ResNode->dump(CurDAG));
        DOUT << "\n";
        Indent -= 2;
#endif
        return ResNode;
      }
      break;
    }
    
    case ISD::TRUNCATE: {
      if (NVT == MVT::i8 && !Subtarget->is64Bit()) {
        SDValue Input = Node->getOperand(0);
        AddToISelQueue(Node->getOperand(0));
        SDNode *ResNode = getTruncateTo8Bit(Input);
      
#ifndef NDEBUG
        DOUT << std::string(Indent-2, ' ') << "=> ";
        DEBUG(ResNode->dump(CurDAG));
        DOUT << "\n";
        Indent -= 2;
#endif
        return ResNode;
      }
      break;
    }

    case ISD::DECLARE: {
      // Handle DECLARE nodes here because the second operand may have been
      // wrapped in X86ISD::Wrapper.
      SDValue Chain = Node->getOperand(0);
      SDValue N1 = Node->getOperand(1);
      SDValue N2 = Node->getOperand(2);
      if (!isa<FrameIndexSDNode>(N1))
        break;
      int FI = cast<FrameIndexSDNode>(N1)->getIndex();
      if (N2.getOpcode() == ISD::ADD &&
          N2.getOperand(0).getOpcode() == X86ISD::GlobalBaseReg)
        N2 = N2.getOperand(1);
      if (N2.getOpcode() == X86ISD::Wrapper &&
          isa<GlobalAddressSDNode>(N2.getOperand(0))) {
        GlobalValue *GV =
          cast<GlobalAddressSDNode>(N2.getOperand(0))->getGlobal();
        SDValue Tmp1 = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
        SDValue Tmp2 = CurDAG->getTargetGlobalAddress(GV, TLI.getPointerTy());
        AddToISelQueue(Chain);
        SDValue Ops[] = { Tmp1, Tmp2, Chain };
        return CurDAG->getTargetNode(TargetInstrInfo::DECLARE,
                                     MVT::Other, Ops, 3);
      }
      break;
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
  SDValue Op0, Op1, Op2, Op3;
  switch (ConstraintCode) {
  case 'o':   // offsetable        ??
  case 'v':   // not offsetable    ??
  default: return true;
  case 'm':   // memory
    if (!SelectAddr(Op, Op, Op0, Op1, Op2, Op3))
      return true;
    break;
  }
  
  OutOps.push_back(Op0);
  OutOps.push_back(Op1);
  OutOps.push_back(Op2);
  OutOps.push_back(Op3);
  AddToISelQueue(Op0);
  AddToISelQueue(Op1);
  AddToISelQueue(Op2);
  AddToISelQueue(Op3);
  return false;
}

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createX86ISelDag(X86TargetMachine &TM, bool Fast) {
  return new X86DAGToDAGISel(TM, Fast);
}
