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
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/CFG.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

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

    // This is really a union, discriminated by BaseType!
    SDValue Base_Reg;
    int Base_FrameIndex;

    unsigned Scale;
    SDValue IndexReg; 
    int32_t Disp;
    SDValue Segment;
    const GlobalValue *GV;
    const Constant *CP;
    const BlockAddress *BlockAddr;
    const char *ES;
    int JT;
    unsigned Align;    // CP alignment.
    unsigned char SymbolFlags;  // X86II::MO_*

    X86ISelAddressMode()
      : BaseType(RegBase), Base_FrameIndex(0), Scale(1), IndexReg(), Disp(0),
        Segment(), GV(0), CP(0), BlockAddr(0), ES(0), JT(-1), Align(0),
        SymbolFlags(X86II::MO_NO_FLAG) {
    }

    bool hasSymbolicDisplacement() const {
      return GV != 0 || CP != 0 || ES != 0 || JT != -1 || BlockAddr != 0;
    }
    
    bool hasBaseOrIndexReg() const {
      return IndexReg.getNode() != 0 || Base_Reg.getNode() != 0;
    }
    
    /// isRIPRelative - Return true if this addressing mode is already RIP
    /// relative.
    bool isRIPRelative() const {
      if (BaseType != RegBase) return false;
      if (RegisterSDNode *RegNode =
            dyn_cast_or_null<RegisterSDNode>(Base_Reg.getNode()))
        return RegNode->getReg() == X86::RIP;
      return false;
    }
    
    void setBaseReg(SDValue Reg) {
      BaseType = RegBase;
      Base_Reg = Reg;
    }

    void dump() {
      dbgs() << "X86ISelAddressMode " << this << '\n';
      dbgs() << "Base_Reg ";
      if (Base_Reg.getNode() != 0)
        Base_Reg.getNode()->dump(); 
      else
        dbgs() << "nul";
      dbgs() << " Base.FrameIndex " << Base_FrameIndex << '\n'
             << " Scale" << Scale << '\n'
             << "IndexReg ";
      if (IndexReg.getNode() != 0)
        IndexReg.getNode()->dump();
      else
        dbgs() << "nul"; 
      dbgs() << " Disp " << Disp << '\n'
             << "GV ";
      if (GV)
        GV->dump();
      else
        dbgs() << "nul";
      dbgs() << " CP ";
      if (CP)
        CP->dump();
      else
        dbgs() << "nul";
      dbgs() << '\n'
             << "ES ";
      if (ES)
        dbgs() << ES;
      else
        dbgs() << "nul";
      dbgs() << " JT" << JT << " Align" << Align << '\n';
    }
  };
}

namespace {
  //===--------------------------------------------------------------------===//
  /// ISel - X86 specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class X86DAGToDAGISel : public SelectionDAGISel {
    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    const X86TargetLowering &X86Lowering;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// OptForSize - If true, selector should try to optimize for code size
    /// instead of performance.
    bool OptForSize;

  public:
    explicit X86DAGToDAGISel(X86TargetMachine &tm, CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel),
        X86Lowering(*tm.getTargetLowering()),
        Subtarget(&tm.getSubtarget<X86Subtarget>()),
        OptForSize(false) {}

    virtual const char *getPassName() const {
      return "X86 DAG->DAG Instruction Selection";
    }

    virtual void EmitFunctionEntryCode();

    virtual bool IsProfitableToFold(SDValue N, SDNode *U, SDNode *Root) const;

    virtual void PreprocessISelDAG();

    inline bool immSext8(SDNode *N) const {
      return isInt<8>(cast<ConstantSDNode>(N)->getSExtValue());
    }

    // i64immSExt32 predicate - True if the 64-bit immediate fits in a 32-bit
    // sign extended field.
    inline bool i64immSExt32(SDNode *N) const {
      uint64_t v = cast<ConstantSDNode>(N)->getZExtValue();
      return (int64_t)v == (int32_t)v;
    }

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    SDNode *Select(SDNode *N);
    SDNode *SelectAtomic64(SDNode *Node, unsigned Opc);
    SDNode *SelectAtomicLoadAdd(SDNode *Node, EVT NVT);
    SDNode *SelectAtomicLoadArith(SDNode *Node, EVT NVT);

    bool FoldOffsetIntoAddress(uint64_t Offset, X86ISelAddressMode &AM);
    bool MatchLoadInAddress(LoadSDNode *N, X86ISelAddressMode &AM);
    bool MatchWrapper(SDValue N, X86ISelAddressMode &AM);
    bool MatchAddress(SDValue N, X86ISelAddressMode &AM);
    bool MatchAddressRecursively(SDValue N, X86ISelAddressMode &AM,
                                 unsigned Depth);
    bool MatchAddressBase(SDValue N, X86ISelAddressMode &AM);
    bool SelectAddr(SDNode *Parent, SDValue N, SDValue &Base,
                    SDValue &Scale, SDValue &Index, SDValue &Disp,
                    SDValue &Segment);
    bool SelectLEAAddr(SDValue N, SDValue &Base,
                       SDValue &Scale, SDValue &Index, SDValue &Disp,
                       SDValue &Segment);
    bool SelectTLSADDRAddr(SDValue N, SDValue &Base,
                           SDValue &Scale, SDValue &Index, SDValue &Disp,
                           SDValue &Segment);
    bool SelectScalarSSELoad(SDNode *Root, SDValue N,
                             SDValue &Base, SDValue &Scale,
                             SDValue &Index, SDValue &Disp,
                             SDValue &Segment,
                             SDValue &NodeWithChain);
    
    bool TryFoldLoad(SDNode *P, SDValue N,
                     SDValue &Base, SDValue &Scale,
                     SDValue &Index, SDValue &Disp,
                     SDValue &Segment);
    
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
        CurDAG->getTargetFrameIndex(AM.Base_FrameIndex, TLI.getPointerTy()) :
        AM.Base_Reg;
      Scale = getI8Imm(AM.Scale);
      Index = AM.IndexReg;
      // These are 32-bit even in 64-bit mode since RIP relative offset
      // is 32-bit.
      if (AM.GV)
        Disp = CurDAG->getTargetGlobalAddress(AM.GV, DebugLoc(),
                                              MVT::i32, AM.Disp,
                                              AM.SymbolFlags);
      else if (AM.CP)
        Disp = CurDAG->getTargetConstantPool(AM.CP, MVT::i32,
                                             AM.Align, AM.Disp, AM.SymbolFlags);
      else if (AM.ES)
        Disp = CurDAG->getTargetExternalSymbol(AM.ES, MVT::i32, AM.SymbolFlags);
      else if (AM.JT != -1)
        Disp = CurDAG->getTargetJumpTable(AM.JT, MVT::i32, AM.SymbolFlags);
      else if (AM.BlockAddr)
        Disp = CurDAG->getBlockAddress(AM.BlockAddr, MVT::i32,
                                       true, AM.SymbolFlags);
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

    /// getTargetMachine - Return a reference to the TargetMachine, casted
    /// to the target-specific type.
    const X86TargetMachine &getTargetMachine() {
      return static_cast<const X86TargetMachine &>(TM);
    }

    /// getInstrInfo - Return a reference to the TargetInstrInfo, casted
    /// to the target-specific type.
    const X86InstrInfo *getInstrInfo() {
      return getTargetMachine().getInstrInfo();
    }
  };
}


bool
X86DAGToDAGISel::IsProfitableToFold(SDValue N, SDNode *U, SDNode *Root) const {
  if (OptLevel == CodeGenOpt::None) return false;

  if (!N.hasOneUse())
    return false;

  if (N.getOpcode() != ISD::LOAD)
    return true;

  // If N is a load, do additional profitability checks.
  if (U == Root) {
    switch (U->getOpcode()) {
    default: break;
    case X86ISD::ADD:
    case X86ISD::SUB:
    case X86ISD::AND:
    case X86ISD::XOR:
    case X86ISD::OR:
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
  }

  return true;
}

/// MoveBelowCallOrigChain - Replace the original chain operand of the call with
/// load's chain operand and move load below the call's chain operand.
static void MoveBelowOrigChain(SelectionDAG *CurDAG, SDValue Load,
                                  SDValue Call, SDValue OrigChain) {
  SmallVector<SDValue, 8> Ops;
  SDValue Chain = OrigChain.getOperand(0);
  if (Chain.getNode() == Load.getNode())
    Ops.push_back(Load.getOperand(0));
  else {
    assert(Chain.getOpcode() == ISD::TokenFactor &&
           "Unexpected chain operand");
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
  for (unsigned i = 1, e = OrigChain.getNumOperands(); i != e; ++i)
    Ops.push_back(OrigChain.getOperand(i));
  CurDAG->UpdateNodeOperands(OrigChain.getNode(), &Ops[0], Ops.size());
  CurDAG->UpdateNodeOperands(Load.getNode(), Call.getOperand(0),
                             Load.getOperand(1), Load.getOperand(2));
  Ops.clear();
  Ops.push_back(SDValue(Load.getNode(), 1));
  for (unsigned i = 1, e = Call.getNode()->getNumOperands(); i != e; ++i)
    Ops.push_back(Call.getOperand(i));
  CurDAG->UpdateNodeOperands(Call.getNode(), &Ops[0], Ops.size());
}

/// isCalleeLoad - Return true if call address is a load and it can be
/// moved below CALLSEQ_START and the chains leading up to the call.
/// Return the CALLSEQ_START by reference as a second output.
/// In the case of a tail call, there isn't a callseq node between the call
/// chain and the load.
static bool isCalleeLoad(SDValue Callee, SDValue &Chain, bool HasCallSeq) {
  if (Callee.getNode() == Chain.getNode() || !Callee.hasOneUse())
    return false;
  LoadSDNode *LD = dyn_cast<LoadSDNode>(Callee.getNode());
  if (!LD ||
      LD->isVolatile() ||
      LD->getAddressingMode() != ISD::UNINDEXED ||
      LD->getExtensionType() != ISD::NON_EXTLOAD)
    return false;

  // Now let's find the callseq_start.
  while (HasCallSeq && Chain.getOpcode() != ISD::CALLSEQ_START) {
    if (!Chain.hasOneUse())
      return false;
    Chain = Chain.getOperand(0);
  }

  if (!Chain.getNumOperands())
    return false;
  if (Chain.getOperand(0).getNode() == Callee.getNode())
    return true;
  if (Chain.getOperand(0).getOpcode() == ISD::TokenFactor &&
      Callee.getValue(1).isOperandOf(Chain.getOperand(0).getNode()) &&
      Callee.getValue(1).hasOneUse())
    return true;
  return false;
}

void X86DAGToDAGISel::PreprocessISelDAG() {
  // OptForSize is used in pattern predicates that isel is matching.
  OptForSize = MF->getFunction()->hasFnAttr(Attribute::OptimizeForSize);
  
  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
       E = CurDAG->allnodes_end(); I != E; ) {
    SDNode *N = I++;  // Preincrement iterator to avoid invalidation issues.

    if (OptLevel != CodeGenOpt::None &&
        (N->getOpcode() == X86ISD::CALL ||
         N->getOpcode() == X86ISD::TC_RETURN)) {
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
      bool HasCallSeq = N->getOpcode() == X86ISD::CALL;
      SDValue Chain = N->getOperand(0);
      SDValue Load  = N->getOperand(1);
      if (!isCalleeLoad(Load, Chain, HasCallSeq))
        continue;
      MoveBelowOrigChain(CurDAG, Load, SDValue(N, 0), Chain);
      ++NumLoadMoved;
      continue;
    }
    
    // Lower fpround and fpextend nodes that target the FP stack to be store and
    // load to the stack.  This is a gross hack.  We would like to simply mark
    // these as being illegal, but when we do that, legalize produces these when
    // it expands calls, then expands these in the same legalize pass.  We would
    // like dag combine to be able to hack on these between the call expansion
    // and the node legalization.  As such this pass basically does "really
    // late" legalization of these inline with the X86 isel pass.
    // FIXME: This should only happen when not compiled with -O0.
    if (N->getOpcode() != ISD::FP_ROUND && N->getOpcode() != ISD::FP_EXTEND)
      continue;
    
    EVT SrcVT = N->getOperand(0).getValueType();
    EVT DstVT = N->getValueType(0);

    // If any of the sources are vectors, no fp stack involved.
    if (SrcVT.isVector() || DstVT.isVector())
      continue;

    // If the source and destination are SSE registers, then this is a legal
    // conversion that should not be lowered.
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
    EVT MemVT;
    if (N->getOpcode() == ISD::FP_ROUND)
      MemVT = DstVT;  // FP_ROUND must use DstVT, we can't do a 'trunc load'.
    else
      MemVT = SrcIsSSE ? SrcVT : DstVT;
    
    SDValue MemTmp = CurDAG->CreateStackTemporary(MemVT);
    DebugLoc dl = N->getDebugLoc();
    
    // FIXME: optimize the case where the src/dest is a load or store?
    SDValue Store = CurDAG->getTruncStore(CurDAG->getEntryNode(), dl,
                                          N->getOperand(0),
                                          MemTmp, MachinePointerInfo(), MemVT,
                                          false, false, 0);
    SDValue Result = CurDAG->getExtLoad(ISD::EXTLOAD, dl, DstVT, Store, MemTmp,
                                        MachinePointerInfo(),
                                        MemVT, false, false, 0);

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


/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
void X86DAGToDAGISel::EmitSpecialCodeForMain(MachineBasicBlock *BB,
                                             MachineFrameInfo *MFI) {
  const TargetInstrInfo *TII = TM.getInstrInfo();
  if (Subtarget->isTargetCygMing()) {
    unsigned CallOp =
      Subtarget->is64Bit() ? X86::WINCALL64pcrel32 : X86::CALLpcrel32;
    BuildMI(BB, DebugLoc(),
            TII->get(CallOp)).addExternalSymbol("__main");
  }
}

void X86DAGToDAGISel::EmitFunctionEntryCode() {
  // If this is main, emit special code for main.
  if (const Function *Fn = MF->getFunction())
    if (Fn->hasExternalLinkage() && Fn->getName() == "main")
      EmitSpecialCodeForMain(MF->begin(), MF->getFrameInfo());
}

static bool isDispSafeForFrameIndex(int64_t Val) {
  // On 64-bit platforms, we can run into an issue where a frame index
  // includes a displacement that, when added to the explicit displacement,
  // will overflow the displacement field. Assuming that the frame index
  // displacement fits into a 31-bit integer  (which is only slightly more
  // aggressive than the current fundamental assumption that it fits into
  // a 32-bit integer), a 31-bit disp should always be safe.
  return isInt<31>(Val);
}

bool X86DAGToDAGISel::FoldOffsetIntoAddress(uint64_t Offset,
                                            X86ISelAddressMode &AM) {
  int64_t Val = AM.Disp + Offset;
  CodeModel::Model M = TM.getCodeModel();
  if (Subtarget->is64Bit()) {
    if (!X86::isOffsetSuitableForCodeModel(Val, M,
                                           AM.hasSymbolicDisplacement()))
      return true;
    // In addition to the checks required for a register base, check that
    // we do not try to use an unsafe Disp with a frame index.
    if (AM.BaseType == X86ISelAddressMode::FrameIndexBase &&
        !isDispSafeForFrameIndex(Val))
      return true;
  }
  AM.Disp = Val;
  return false;

}

bool X86DAGToDAGISel::MatchLoadInAddress(LoadSDNode *N, X86ISelAddressMode &AM){
  SDValue Address = N->getOperand(1);
  
  // load gs:0 -> GS segment register.
  // load fs:0 -> FS segment register.
  //
  // This optimization is valid because the GNU TLS model defines that
  // gs:0 (or fs:0 on X86-64) contains its own address.
  // For more information see http://people.redhat.com/drepper/tls.pdf
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Address))
    if (C->getSExtValue() == 0 && AM.Segment.getNode() == 0 &&
        Subtarget->isTargetELF())
      switch (N->getPointerInfo().getAddrSpace()) {
      case 256:
        AM.Segment = CurDAG->getRegister(X86::GS, MVT::i16);
        return false;
      case 257:
        AM.Segment = CurDAG->getRegister(X86::FS, MVT::i16);
        return false;
      }
  
  return true;
}

/// MatchWrapper - Try to match X86ISD::Wrapper and X86ISD::WrapperRIP nodes
/// into an addressing mode.  These wrap things that will resolve down into a
/// symbol reference.  If no match is possible, this returns true, otherwise it
/// returns false.
bool X86DAGToDAGISel::MatchWrapper(SDValue N, X86ISelAddressMode &AM) {
  // If the addressing mode already has a symbol as the displacement, we can
  // never match another symbol.
  if (AM.hasSymbolicDisplacement())
    return true;

  SDValue N0 = N.getOperand(0);
  CodeModel::Model M = TM.getCodeModel();

  // Handle X86-64 rip-relative addresses.  We check this before checking direct
  // folding because RIP is preferable to non-RIP accesses.
  if (Subtarget->is64Bit() &&
      // Under X86-64 non-small code model, GV (and friends) are 64-bits, so
      // they cannot be folded into immediate fields.
      // FIXME: This can be improved for kernel and other models?
      (M == CodeModel::Small || M == CodeModel::Kernel) &&
      // Base and index reg must be 0 in order to use %rip as base and lowering
      // must allow RIP.
      !AM.hasBaseOrIndexReg() && N.getOpcode() == X86ISD::WrapperRIP) {
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
      X86ISelAddressMode Backup = AM;
      AM.GV = G->getGlobal();
      AM.SymbolFlags = G->getTargetFlags();
      if (FoldOffsetIntoAddress(G->getOffset(), AM)) {
        AM = Backup;
        return true;
      }
    } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
      X86ISelAddressMode Backup = AM;
      AM.CP = CP->getConstVal();
      AM.Align = CP->getAlignment();
      AM.SymbolFlags = CP->getTargetFlags();
      if (FoldOffsetIntoAddress(CP->getOffset(), AM)) {
        AM = Backup;
        return true;
      }
    } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(N0)) {
      AM.ES = S->getSymbol();
      AM.SymbolFlags = S->getTargetFlags();
    } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
      AM.JT = J->getIndex();
      AM.SymbolFlags = J->getTargetFlags();
    } else {
      AM.BlockAddr = cast<BlockAddressSDNode>(N0)->getBlockAddress();
      AM.SymbolFlags = cast<BlockAddressSDNode>(N0)->getTargetFlags();
    }

    if (N.getOpcode() == X86ISD::WrapperRIP)
      AM.setBaseReg(CurDAG->getRegister(X86::RIP, MVT::i64));
    return false;
  }

  // Handle the case when globals fit in our immediate field: This is true for
  // X86-32 always and X86-64 when in -static -mcmodel=small mode.  In 64-bit
  // mode, this results in a non-RIP-relative computation.
  if (!Subtarget->is64Bit() ||
      ((M == CodeModel::Small || M == CodeModel::Kernel) &&
       TM.getRelocationModel() == Reloc::Static)) {
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
      AM.GV = G->getGlobal();
      AM.Disp += G->getOffset();
      AM.SymbolFlags = G->getTargetFlags();
    } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
      AM.CP = CP->getConstVal();
      AM.Align = CP->getAlignment();
      AM.Disp += CP->getOffset();
      AM.SymbolFlags = CP->getTargetFlags();
    } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(N0)) {
      AM.ES = S->getSymbol();
      AM.SymbolFlags = S->getTargetFlags();
    } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
      AM.JT = J->getIndex();
      AM.SymbolFlags = J->getTargetFlags();
    } else {
      AM.BlockAddr = cast<BlockAddressSDNode>(N0)->getBlockAddress();
      AM.SymbolFlags = cast<BlockAddressSDNode>(N0)->getTargetFlags();
    }
    return false;
  }

  return true;
}

/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode.
bool X86DAGToDAGISel::MatchAddress(SDValue N, X86ISelAddressMode &AM) {
  if (MatchAddressRecursively(N, AM, 0))
    return true;

  // Post-processing: Convert lea(,%reg,2) to lea(%reg,%reg), which has
  // a smaller encoding and avoids a scaled-index.
  if (AM.Scale == 2 &&
      AM.BaseType == X86ISelAddressMode::RegBase &&
      AM.Base_Reg.getNode() == 0) {
    AM.Base_Reg = AM.IndexReg;
    AM.Scale = 1;
  }

  // Post-processing: Convert foo to foo(%rip), even in non-PIC mode,
  // because it has a smaller encoding.
  // TODO: Which other code models can use this?
  if (TM.getCodeModel() == CodeModel::Small &&
      Subtarget->is64Bit() &&
      AM.Scale == 1 &&
      AM.BaseType == X86ISelAddressMode::RegBase &&
      AM.Base_Reg.getNode() == 0 &&
      AM.IndexReg.getNode() == 0 &&
      AM.SymbolFlags == X86II::MO_NO_FLAG &&
      AM.hasSymbolicDisplacement())
    AM.Base_Reg = CurDAG->getRegister(X86::RIP, MVT::i64);

  return false;
}

// Transform "(X >> (8-C1)) & C2" to "(X >> 8) & 0xff)" if safe. This
// allows us to convert the shift and and into an h-register extract and
// a scaled index. Returns false if the simplification is performed.
static bool FoldMaskAndShiftToExtract(SelectionDAG &DAG, SDValue N,
                                      uint64_t Mask,
                                      SDValue Shift, SDValue X,
                                      X86ISelAddressMode &AM) {
  if (Shift.getOpcode() != ISD::SRL ||
      !isa<ConstantSDNode>(Shift.getOperand(1)) ||
      !Shift.hasOneUse())
    return true;

  int ScaleLog = 8 - Shift.getConstantOperandVal(1);
  if (ScaleLog <= 0 || ScaleLog >= 4 ||
      Mask != (0xffu << ScaleLog))
    return true;

  EVT VT = N.getValueType();
  DebugLoc DL = N.getDebugLoc();
  SDValue Eight = DAG.getConstant(8, MVT::i8);
  SDValue NewMask = DAG.getConstant(0xff, VT);
  SDValue Srl = DAG.getNode(ISD::SRL, DL, VT, X, Eight);
  SDValue And = DAG.getNode(ISD::AND, DL, VT, Srl, NewMask);
  SDValue ShlCount = DAG.getConstant(ScaleLog, MVT::i8);
  SDValue Shl = DAG.getNode(ISD::SHL, DL, VT, And, ShlCount);

  // Insert the new nodes into the topological ordering.
  if (Eight.getNode()->getNodeId() == -1 ||
      Eight.getNode()->getNodeId() > X.getNode()->getNodeId()) {
    DAG.RepositionNode(X.getNode(), Eight.getNode());
    Eight.getNode()->setNodeId(X.getNode()->getNodeId());
  }
  if (NewMask.getNode()->getNodeId() == -1 ||
      NewMask.getNode()->getNodeId() > X.getNode()->getNodeId()) {
    DAG.RepositionNode(X.getNode(), NewMask.getNode());
    NewMask.getNode()->setNodeId(X.getNode()->getNodeId());
  }
  if (Srl.getNode()->getNodeId() == -1 ||
      Srl.getNode()->getNodeId() > Shift.getNode()->getNodeId()) {
    DAG.RepositionNode(Shift.getNode(), Srl.getNode());
    Srl.getNode()->setNodeId(Shift.getNode()->getNodeId());
  }
  if (And.getNode()->getNodeId() == -1 ||
      And.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), And.getNode());
    And.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  if (ShlCount.getNode()->getNodeId() == -1 ||
      ShlCount.getNode()->getNodeId() > X.getNode()->getNodeId()) {
    DAG.RepositionNode(X.getNode(), ShlCount.getNode());
    ShlCount.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  if (Shl.getNode()->getNodeId() == -1 ||
      Shl.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), Shl.getNode());
    Shl.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  DAG.ReplaceAllUsesWith(N, Shl);
  AM.IndexReg = And;
  AM.Scale = (1 << ScaleLog);
  return false;
}

// Transforms "(X << C1) & C2" to "(X & (C2>>C1)) << C1" if safe and if this
// allows us to fold the shift into this addressing mode. Returns false if the
// transform succeeded.
static bool FoldMaskedShiftToScaledMask(SelectionDAG &DAG, SDValue N,
                                        uint64_t Mask,
                                        SDValue Shift, SDValue X,
                                        X86ISelAddressMode &AM) {
  if (Shift.getOpcode() != ISD::SHL ||
      !isa<ConstantSDNode>(Shift.getOperand(1)))
    return true;

  // Not likely to be profitable if either the AND or SHIFT node has more
  // than one use (unless all uses are for address computation). Besides,
  // isel mechanism requires their node ids to be reused.
  if (!N.hasOneUse() || !Shift.hasOneUse())
    return true;

  // Verify that the shift amount is something we can fold.
  unsigned ShiftAmt = Shift.getConstantOperandVal(1);
  if (ShiftAmt != 1 && ShiftAmt != 2 && ShiftAmt != 3)
    return true;

  EVT VT = N.getValueType();
  DebugLoc DL = N.getDebugLoc();
  SDValue NewMask = DAG.getConstant(Mask >> ShiftAmt, VT);
  SDValue NewAnd = DAG.getNode(ISD::AND, DL, VT, X, NewMask);
  SDValue NewShift = DAG.getNode(ISD::SHL, DL, VT, NewAnd, Shift.getOperand(1));

  // Insert the new nodes into the topological ordering.
  if (NewMask.getNode()->getNodeId() == -1 ||
      NewMask.getNode()->getNodeId() > X.getNode()->getNodeId()) {
    DAG.RepositionNode(X.getNode(), NewMask.getNode());
    NewMask.getNode()->setNodeId(X.getNode()->getNodeId());
  }
  if (NewAnd.getNode()->getNodeId() == -1 ||
      NewAnd.getNode()->getNodeId() > Shift.getNode()->getNodeId()) {
    DAG.RepositionNode(Shift.getNode(), NewAnd.getNode());
    NewAnd.getNode()->setNodeId(Shift.getNode()->getNodeId());
  }
  if (NewShift.getNode()->getNodeId() == -1 ||
      NewShift.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), NewShift.getNode());
    NewShift.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  DAG.ReplaceAllUsesWith(N, NewShift);

  AM.Scale = 1 << ShiftAmt;
  AM.IndexReg = NewAnd;
  return false;
}

// Implement some heroics to detect shifts of masked values where the mask can
// be replaced by extending the shift and undoing that in the addressing mode
// scale. Patterns such as (shl (srl x, c1), c2) are canonicalized into (and
// (srl x, SHIFT), MASK) by DAGCombines that don't know the shl can be done in
// the addressing mode. This results in code such as:
//
//   int f(short *y, int *lookup_table) {
//     ...
//     return *y + lookup_table[*y >> 11];
//   }
//
// Turning into:
//   movzwl (%rdi), %eax
//   movl %eax, %ecx
//   shrl $11, %ecx
//   addl (%rsi,%rcx,4), %eax
//
// Instead of:
//   movzwl (%rdi), %eax
//   movl %eax, %ecx
//   shrl $9, %ecx
//   andl $124, %rcx
//   addl (%rsi,%rcx), %eax
//
static bool FoldMaskAndShiftToScale(SelectionDAG &DAG, SDValue N,
                                    X86ISelAddressMode &AM) {
  // Scale must not be used already.
  if (AM.IndexReg.getNode() != 0 || AM.Scale != 1) return true;

  SDValue Shift = N;
  SDValue And = N.getOperand(0);
  if (N.getOpcode() != ISD::SRL)
    std::swap(Shift, And);
  if (Shift.getOpcode() != ISD::SRL || And.getOpcode() != ISD::AND ||
      !Shift.hasOneUse() ||
      !isa<ConstantSDNode>(Shift.getOperand(1)) ||
      !isa<ConstantSDNode>(And.getOperand(1)))
    return true;
  SDValue X = (N == Shift ? And.getOperand(0) : Shift.getOperand(0));

  // We only handle up to 64-bit values here as those are what matter for
  // addressing mode optimizations.
  if (X.getValueSizeInBits() > 64) return true;

  uint64_t Mask = And.getConstantOperandVal(1);
  unsigned ShiftAmt = Shift.getConstantOperandVal(1);
  unsigned MaskLZ = CountLeadingZeros_64(Mask);
  unsigned MaskTZ = CountTrailingZeros_64(Mask);

  // The amount of shift we're trying to fit into the addressing mode is taken
  // from the trailing zeros of the mask. If the mask is pre-shift, we subtract
  // the shift amount.
  int AMShiftAmt = MaskTZ - (N == Shift ? ShiftAmt : 0);

  // There is nothing we can do here unless the mask is removing some bits.
  // Also, the addressing mode can only represent shifts of 1, 2, or 3 bits.
  if (AMShiftAmt <= 0 || AMShiftAmt > 3) return true;

  // We also need to ensure that mask is a continuous run of bits.
  if (CountTrailingOnes_64(Mask >> MaskTZ) + MaskTZ + MaskLZ != 64) return true;

  // Scale the leading zero count down based on the actual size of the value.
  // Also scale it down based on the size of the shift if it was applied
  // before the mask.
  MaskLZ -= (64 - X.getValueSizeInBits()) + (N == Shift ? 0 : ShiftAmt);

  // The final check is to ensure that any masked out high bits of X are
  // already known to be zero. Otherwise, the mask has a semantic impact
  // other than masking out a couple of low bits. Unfortunately, because of
  // the mask, zero extensions will be removed from operands in some cases.
  // This code works extra hard to look through extensions because we can
  // replace them with zero extensions cheaply if necessary.
  bool ReplacingAnyExtend = false;
  if (X.getOpcode() == ISD::ANY_EXTEND) {
    unsigned ExtendBits =
      X.getValueSizeInBits() - X.getOperand(0).getValueSizeInBits();
    // Assume that we'll replace the any-extend with a zero-extend, and
    // narrow the search to the extended value.
    X = X.getOperand(0);
    MaskLZ = ExtendBits > MaskLZ ? 0 : MaskLZ - ExtendBits;
    ReplacingAnyExtend = true;
  }
  APInt MaskedHighBits = APInt::getHighBitsSet(X.getValueSizeInBits(),
                                               MaskLZ);
  APInt KnownZero, KnownOne;
  DAG.ComputeMaskedBits(X, MaskedHighBits, KnownZero, KnownOne);
  if (MaskedHighBits != KnownZero) return true;

  // We've identified a pattern that can be transformed into a single shift
  // and an addressing mode. Make it so.
  EVT VT = N.getValueType();
  if (ReplacingAnyExtend) {
    assert(X.getValueType() != VT);
    // We looked through an ANY_EXTEND node, insert a ZERO_EXTEND.
    SDValue NewX = DAG.getNode(ISD::ZERO_EXTEND, X.getDebugLoc(), VT, X);
    if (NewX.getNode()->getNodeId() == -1 ||
        NewX.getNode()->getNodeId() > N.getNode()->getNodeId()) {
      DAG.RepositionNode(N.getNode(), NewX.getNode());
      NewX.getNode()->setNodeId(N.getNode()->getNodeId());
    }
    X = NewX;
  }
  DebugLoc DL = N.getDebugLoc();
  SDValue NewSRLAmt = DAG.getConstant(ShiftAmt + AMShiftAmt, MVT::i8);
  SDValue NewSRL = DAG.getNode(ISD::SRL, DL, VT, X, NewSRLAmt);
  SDValue NewSHLAmt = DAG.getConstant(AMShiftAmt, MVT::i8);
  SDValue NewSHL = DAG.getNode(ISD::SHL, DL, VT, NewSRL, NewSHLAmt);
  if (NewSRLAmt.getNode()->getNodeId() == -1 ||
      NewSRLAmt.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), NewSRLAmt.getNode());
    NewSRLAmt.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  if (NewSRL.getNode()->getNodeId() == -1 ||
      NewSRL.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), NewSRL.getNode());
    NewSRL.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  if (NewSHLAmt.getNode()->getNodeId() == -1 ||
      NewSHLAmt.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), NewSHLAmt.getNode());
    NewSHLAmt.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  if (NewSHL.getNode()->getNodeId() == -1 ||
      NewSHL.getNode()->getNodeId() > N.getNode()->getNodeId()) {
    DAG.RepositionNode(N.getNode(), NewSHL.getNode());
    NewSHL.getNode()->setNodeId(N.getNode()->getNodeId());
  }
  DAG.ReplaceAllUsesWith(N, NewSHL);

  AM.Scale = 1 << AMShiftAmt;
  AM.IndexReg = NewSRL;
  return false;
}

bool X86DAGToDAGISel::MatchAddressRecursively(SDValue N, X86ISelAddressMode &AM,
                                              unsigned Depth) {
  DebugLoc dl = N.getDebugLoc();
  DEBUG({
      dbgs() << "MatchAddress: ";
      AM.dump();
    });
  // Limit recursion.
  if (Depth > 5)
    return MatchAddressBase(N, AM);

  // If this is already a %rip relative address, we can only merge immediates
  // into it.  Instead of handling this in every case, we handle it here.
  // RIP relative addressing: %rip + 32-bit displacement!
  if (AM.isRIPRelative()) {
    // FIXME: JumpTable and ExternalSymbol address currently don't like
    // displacements.  It isn't very important, but this should be fixed for
    // consistency.
    if (!AM.ES && AM.JT != -1) return true;

    if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N))
      if (!FoldOffsetIntoAddress(Cst->getSExtValue(), AM))
        return false;
    return true;
  }

  switch (N.getOpcode()) {
  default: break;
  case ISD::Constant: {
    uint64_t Val = cast<ConstantSDNode>(N)->getSExtValue();
    if (!FoldOffsetIntoAddress(Val, AM))
      return false;
    break;
  }

  case X86ISD::Wrapper:
  case X86ISD::WrapperRIP:
    if (!MatchWrapper(N, AM))
      return false;
    break;

  case ISD::LOAD:
    if (!MatchLoadInAddress(cast<LoadSDNode>(N), AM))
      return false;
    break;

  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base_Reg.getNode() == 0 &&
        (!Subtarget->is64Bit() || isDispSafeForFrameIndex(AM.Disp))) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base_FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::SHL:
    if (AM.IndexReg.getNode() != 0 || AM.Scale != 1)
      break;
      
    if (ConstantSDNode
          *CN = dyn_cast<ConstantSDNode>(N.getNode()->getOperand(1))) {
      unsigned Val = CN->getZExtValue();
      // Note that we handle x<<1 as (,x,2) rather than (x,x) here so
      // that the base operand remains free for further matching. If
      // the base doesn't end up getting used, a post-processing step
      // in MatchAddress turns (,x,2) into (x,x), which is cheaper.
      if (Val == 1 || Val == 2 || Val == 3) {
        AM.Scale = 1 << Val;
        SDValue ShVal = N.getNode()->getOperand(0);

        // Okay, we know that we have a scale by now.  However, if the scaled
        // value is an add of something and a constant, we can fold the
        // constant into the disp field here.
        if (CurDAG->isBaseWithConstantOffset(ShVal)) {
          AM.IndexReg = ShVal.getNode()->getOperand(0);
          ConstantSDNode *AddVal =
            cast<ConstantSDNode>(ShVal.getNode()->getOperand(1));
          uint64_t Disp = AddVal->getSExtValue() << Val;
          if (!FoldOffsetIntoAddress(Disp, AM))
            return false;
        }

        AM.IndexReg = ShVal;
        return false;
      }
    break;
    }

  case ISD::SRL:
    // Try to fold the mask and shift into the scale, and return false if we
    // succeed.
    if (!FoldMaskAndShiftToScale(*CurDAG, N, AM))
      return false;
    break;

  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
    // A mul_lohi where we need the low part can be folded as a plain multiply.
    if (N.getResNo() != 0) break;
    // FALL THROUGH
  case ISD::MUL:
  case X86ISD::MUL_IMM:
    // X*[3,5,9] -> X+X*[2,4,8]
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base_Reg.getNode() == 0 &&
        AM.IndexReg.getNode() == 0) {
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
            uint64_t Disp = AddVal->getSExtValue() * CN->getZExtValue();
            if (FoldOffsetIntoAddress(Disp, AM))
              Reg = N.getNode()->getOperand(0);
          } else {
            Reg = N.getNode()->getOperand(0);
          }

          AM.IndexReg = AM.Base_Reg = Reg;
          return false;
        }
    }
    break;

  case ISD::SUB: {
    // Given A-B, if A can be completely folded into the address and
    // the index field with the index field unused, use -B as the index.
    // This is a win if a has multiple parts that can be folded into
    // the address. Also, this saves a mov if the base register has
    // other uses, since it avoids a two-address sub instruction, however
    // it costs an additional mov if the index register has other uses.

    // Add an artificial use to this node so that we can keep track of
    // it if it gets CSE'd with a different node.
    HandleSDNode Handle(N);

    // Test if the LHS of the sub can be folded.
    X86ISelAddressMode Backup = AM;
    if (MatchAddressRecursively(N.getNode()->getOperand(0), AM, Depth+1)) {
      AM = Backup;
      break;
    }
    // Test if the index field is free for use.
    if (AM.IndexReg.getNode() || AM.isRIPRelative()) {
      AM = Backup;
      break;
    }

    int Cost = 0;
    SDValue RHS = Handle.getValue().getNode()->getOperand(1);
    // If the RHS involves a register with multiple uses, this
    // transformation incurs an extra mov, due to the neg instruction
    // clobbering its operand.
    if (!RHS.getNode()->hasOneUse() ||
        RHS.getNode()->getOpcode() == ISD::CopyFromReg ||
        RHS.getNode()->getOpcode() == ISD::TRUNCATE ||
        RHS.getNode()->getOpcode() == ISD::ANY_EXTEND ||
        (RHS.getNode()->getOpcode() == ISD::ZERO_EXTEND &&
         RHS.getNode()->getOperand(0).getValueType() == MVT::i32))
      ++Cost;
    // If the base is a register with multiple uses, this
    // transformation may save a mov.
    if ((AM.BaseType == X86ISelAddressMode::RegBase &&
         AM.Base_Reg.getNode() &&
         !AM.Base_Reg.getNode()->hasOneUse()) ||
        AM.BaseType == X86ISelAddressMode::FrameIndexBase)
      --Cost;
    // If the folded LHS was interesting, this transformation saves
    // address arithmetic.
    if ((AM.hasSymbolicDisplacement() && !Backup.hasSymbolicDisplacement()) +
        ((AM.Disp != 0) && (Backup.Disp == 0)) +
        (AM.Segment.getNode() && !Backup.Segment.getNode()) >= 2)
      --Cost;
    // If it doesn't look like it may be an overall win, don't do it.
    if (Cost >= 0) {
      AM = Backup;
      break;
    }

    // Ok, the transformation is legal and appears profitable. Go for it.
    SDValue Zero = CurDAG->getConstant(0, N.getValueType());
    SDValue Neg = CurDAG->getNode(ISD::SUB, dl, N.getValueType(), Zero, RHS);
    AM.IndexReg = Neg;
    AM.Scale = 1;

    // Insert the new nodes into the topological ordering.
    if (Zero.getNode()->getNodeId() == -1 ||
        Zero.getNode()->getNodeId() > N.getNode()->getNodeId()) {
      CurDAG->RepositionNode(N.getNode(), Zero.getNode());
      Zero.getNode()->setNodeId(N.getNode()->getNodeId());
    }
    if (Neg.getNode()->getNodeId() == -1 ||
        Neg.getNode()->getNodeId() > N.getNode()->getNodeId()) {
      CurDAG->RepositionNode(N.getNode(), Neg.getNode());
      Neg.getNode()->setNodeId(N.getNode()->getNodeId());
    }
    return false;
  }

  case ISD::ADD: {
    // Add an artificial use to this node so that we can keep track of
    // it if it gets CSE'd with a different node.
    HandleSDNode Handle(N);

    X86ISelAddressMode Backup = AM;
    if (!MatchAddressRecursively(N.getOperand(0), AM, Depth+1) &&
        !MatchAddressRecursively(Handle.getValue().getOperand(1), AM, Depth+1))
      return false;
    AM = Backup;
    
    // Try again after commuting the operands.
    if (!MatchAddressRecursively(Handle.getValue().getOperand(1), AM, Depth+1)&&
        !MatchAddressRecursively(Handle.getValue().getOperand(0), AM, Depth+1))
      return false;
    AM = Backup;

    // If we couldn't fold both operands into the address at the same time,
    // see if we can just put each operand into a register and fold at least
    // the add.
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        !AM.Base_Reg.getNode() &&
        !AM.IndexReg.getNode()) {
      N = Handle.getValue();
      AM.Base_Reg = N.getOperand(0);
      AM.IndexReg = N.getOperand(1);
      AM.Scale = 1;
      return false;
    }
    N = Handle.getValue();
    break;
  }

  case ISD::OR:
    // Handle "X | C" as "X + C" iff X is known to have C bits clear.
    if (CurDAG->isBaseWithConstantOffset(N)) {
      X86ISelAddressMode Backup = AM;
      ConstantSDNode *CN = cast<ConstantSDNode>(N.getOperand(1));

      // Start with the LHS as an addr mode.
      if (!MatchAddressRecursively(N.getOperand(0), AM, Depth+1) &&
          !FoldOffsetIntoAddress(CN->getSExtValue(), AM))
        return false;
      AM = Backup;
    }
    break;
      
  case ISD::AND: {
    // Perform some heroic transforms on an and of a constant-count shift
    // with a constant to enable use of the scaled offset field.

    // Scale must not be used already.
    if (AM.IndexReg.getNode() != 0 || AM.Scale != 1) break;

    SDValue Shift = N.getOperand(0);
    if (Shift.getOpcode() != ISD::SRL && Shift.getOpcode() != ISD::SHL) break;
    SDValue X = Shift.getOperand(0);

    // We only handle up to 64-bit values here as those are what matter for
    // addressing mode optimizations.
    if (X.getValueSizeInBits() > 64) break;

    ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N.getOperand(1));
    ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
    if (!C1 || !C2) break;

    // Try to fold the mask and shift into an extract and scale.
    if (!FoldMaskAndShiftToExtract(*CurDAG, N, C2->getZExtValue(),
                                   Shift, X, AM))
      return false;

    // Try to fold the mask and shift directly into the scale.
    if (!FoldMaskAndShiftToScale(*CurDAG, N, AM))
      return false;

    // Try to swap the mask and shift to place shifts which can be done as
    // a scale on the outside of the mask.
    if (!FoldMaskedShiftToScaledMask(*CurDAG, N, C2->getZExtValue(),
                                     Shift, X, AM))
      return false;
    break;
  }
  }

  return MatchAddressBase(N, AM);
}

/// MatchAddressBase - Helper for MatchAddress. Add the specified node to the
/// specified addressing mode without any further recursion.
bool X86DAGToDAGISel::MatchAddressBase(SDValue N, X86ISelAddressMode &AM) {
  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base_Reg.getNode()) {
    // If so, check to see if the scale index register is set.
    if (AM.IndexReg.getNode() == 0) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base_Reg = N;
  return false;
}

/// SelectAddr - returns true if it is able pattern match an addressing mode.
/// It returns the operands which make up the maximal addressing mode it can
/// match by reference.
///
/// Parent is the parent node of the addr operand that is being matched.  It
/// is always a load, store, atomic node, or null.  It is only null when
/// checking memory operands for inline asm nodes.
bool X86DAGToDAGISel::SelectAddr(SDNode *Parent, SDValue N, SDValue &Base,
                                 SDValue &Scale, SDValue &Index,
                                 SDValue &Disp, SDValue &Segment) {
  X86ISelAddressMode AM;
  
  if (Parent &&
      // This list of opcodes are all the nodes that have an "addr:$ptr" operand
      // that are not a MemSDNode, and thus don't have proper addrspace info.
      Parent->getOpcode() != ISD::INTRINSIC_W_CHAIN && // unaligned loads, fixme
      Parent->getOpcode() != ISD::INTRINSIC_VOID && // nontemporal stores
      Parent->getOpcode() != X86ISD::TLSCALL) { // Fixme
    unsigned AddrSpace =
      cast<MemSDNode>(Parent)->getPointerInfo().getAddrSpace();
    // AddrSpace 256 -> GS, 257 -> FS.
    if (AddrSpace == 256)
      AM.Segment = CurDAG->getRegister(X86::GS, MVT::i16);
    if (AddrSpace == 257)
      AM.Segment = CurDAG->getRegister(X86::FS, MVT::i16);
  }
  
  if (MatchAddress(N, AM))
    return false;

  EVT VT = N.getValueType();
  if (AM.BaseType == X86ISelAddressMode::RegBase) {
    if (!AM.Base_Reg.getNode())
      AM.Base_Reg = CurDAG->getRegister(0, VT);
  }

  if (!AM.IndexReg.getNode())
    AM.IndexReg = CurDAG->getRegister(0, VT);

  getAddressOperands(AM, Base, Scale, Index, Disp, Segment);
  return true;
}

/// SelectScalarSSELoad - Match a scalar SSE load.  In particular, we want to
/// match a load whose top elements are either undef or zeros.  The load flavor
/// is derived from the type of N, which is either v4f32 or v2f64.
///
/// We also return:
///   PatternChainNode: this is the matched node that has a chain input and
///   output.
bool X86DAGToDAGISel::SelectScalarSSELoad(SDNode *Root,
                                          SDValue N, SDValue &Base,
                                          SDValue &Scale, SDValue &Index,
                                          SDValue &Disp, SDValue &Segment,
                                          SDValue &PatternNodeWithChain) {
  if (N.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    PatternNodeWithChain = N.getOperand(0);
    if (ISD::isNON_EXTLoad(PatternNodeWithChain.getNode()) &&
        PatternNodeWithChain.hasOneUse() &&
        IsProfitableToFold(N.getOperand(0), N.getNode(), Root) &&
        IsLegalToFold(N.getOperand(0), N.getNode(), Root, OptLevel)) {
      LoadSDNode *LD = cast<LoadSDNode>(PatternNodeWithChain);
      if (!SelectAddr(LD, LD->getBasePtr(), Base, Scale, Index, Disp, Segment))
        return false;
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
      N.getOperand(0).getOperand(0).hasOneUse() &&
      IsProfitableToFold(N.getOperand(0), N.getNode(), Root) &&
      IsLegalToFold(N.getOperand(0), N.getNode(), Root, OptLevel)) {
    // Okay, this is a zero extending load.  Fold it.
    LoadSDNode *LD = cast<LoadSDNode>(N.getOperand(0).getOperand(0));
    if (!SelectAddr(LD, LD->getBasePtr(), Base, Scale, Index, Disp, Segment))
      return false;
    PatternNodeWithChain = SDValue(LD, 0);
    return true;
  }
  return false;
}


/// SelectLEAAddr - it calls SelectAddr and determines if the maximal addressing
/// mode it matches can be cost effectively emitted as an LEA instruction.
bool X86DAGToDAGISel::SelectLEAAddr(SDValue N,
                                    SDValue &Base, SDValue &Scale,
                                    SDValue &Index, SDValue &Disp,
                                    SDValue &Segment) {
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

  EVT VT = N.getValueType();
  unsigned Complexity = 0;
  if (AM.BaseType == X86ISelAddressMode::RegBase)
    if (AM.Base_Reg.getNode())
      Complexity = 1;
    else
      AM.Base_Reg = CurDAG->getRegister(0, VT);
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

  if (AM.Disp && (AM.Base_Reg.getNode() || AM.IndexReg.getNode()))
    Complexity++;

  // If it isn't worth using an LEA, reject it.
  if (Complexity <= 2)
    return false;
  
  getAddressOperands(AM, Base, Scale, Index, Disp, Segment);
  return true;
}

/// SelectTLSADDRAddr - This is only run on TargetGlobalTLSAddress nodes.
bool X86DAGToDAGISel::SelectTLSADDRAddr(SDValue N, SDValue &Base,
                                        SDValue &Scale, SDValue &Index,
                                        SDValue &Disp, SDValue &Segment) {
  assert(N.getOpcode() == ISD::TargetGlobalTLSAddress);
  const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);
    
  X86ISelAddressMode AM;
  AM.GV = GA->getGlobal();
  AM.Disp += GA->getOffset();
  AM.Base_Reg = CurDAG->getRegister(0, N.getValueType());
  AM.SymbolFlags = GA->getTargetFlags();

  if (N.getValueType() == MVT::i32) {
    AM.Scale = 1;
    AM.IndexReg = CurDAG->getRegister(X86::EBX, MVT::i32);
  } else {
    AM.IndexReg = CurDAG->getRegister(0, MVT::i64);
  }
  
  getAddressOperands(AM, Base, Scale, Index, Disp, Segment);
  return true;
}


bool X86DAGToDAGISel::TryFoldLoad(SDNode *P, SDValue N,
                                  SDValue &Base, SDValue &Scale,
                                  SDValue &Index, SDValue &Disp,
                                  SDValue &Segment) {
  if (!ISD::isNON_EXTLoad(N.getNode()) ||
      !IsProfitableToFold(N, P, P) ||
      !IsLegalToFold(N, P, P, OptLevel))
    return false;
  
  return SelectAddr(N.getNode(),
                    N.getOperand(1), Base, Scale, Index, Disp, Segment);
}

/// getGlobalBaseReg - Return an SDNode that returns the value of
/// the global base register. Output instructions required to
/// initialize the global base register, if necessary.
///
SDNode *X86DAGToDAGISel::getGlobalBaseReg() {
  unsigned GlobalBaseReg = getInstrInfo()->getGlobalBaseReg(MF);
  return CurDAG->getRegister(GlobalBaseReg, TLI.getPointerTy()).getNode();
}

SDNode *X86DAGToDAGISel::SelectAtomic64(SDNode *Node, unsigned Opc) {
  SDValue Chain = Node->getOperand(0);
  SDValue In1 = Node->getOperand(1);
  SDValue In2L = Node->getOperand(2);
  SDValue In2H = Node->getOperand(3);
  SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
  if (!SelectAddr(Node, In1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4))
    return NULL;
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemSDNode>(Node)->getMemOperand();
  const SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, In2L, In2H, Chain};
  SDNode *ResNode = CurDAG->getMachineNode(Opc, Node->getDebugLoc(),
                                           MVT::i32, MVT::i32, MVT::Other, Ops,
                                           array_lengthof(Ops));
  cast<MachineSDNode>(ResNode)->setMemRefs(MemOp, MemOp + 1);
  return ResNode;
}

// FIXME: Figure out some way to unify this with the 'or' and other code
// below.
SDNode *X86DAGToDAGISel::SelectAtomicLoadAdd(SDNode *Node, EVT NVT) {
  if (Node->hasAnyUseOfValue(0))
    return 0;

  // Optimize common patterns for __sync_add_and_fetch and
  // __sync_sub_and_fetch where the result is not used. This allows us
  // to use "lock" version of add, sub, inc, dec instructions.
  // FIXME: Do not use special instructions but instead add the "lock"
  // prefix to the target node somehow. The extra information will then be
  // transferred to machine instruction and it denotes the prefix.
  SDValue Chain = Node->getOperand(0);
  SDValue Ptr = Node->getOperand(1);
  SDValue Val = Node->getOperand(2);
  SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
  if (!SelectAddr(Node, Ptr, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4))
    return 0;

  bool isInc = false, isDec = false, isSub = false, isCN = false;
  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Val);
  if (CN && CN->getSExtValue() == (int32_t)CN->getSExtValue()) {
    isCN = true;
    int64_t CNVal = CN->getSExtValue();
    if (CNVal == 1)
      isInc = true;
    else if (CNVal == -1)
      isDec = true;
    else if (CNVal >= 0)
      Val = CurDAG->getTargetConstant(CNVal, NVT);
    else {
      isSub = true;
      Val = CurDAG->getTargetConstant(-CNVal, NVT);
    }
  } else if (Val.hasOneUse() &&
             Val.getOpcode() == ISD::SUB &&
             X86::isZeroNode(Val.getOperand(0))) {
    isSub = true;
    Val = Val.getOperand(1);
  }

  DebugLoc dl = Node->getDebugLoc();
  unsigned Opc = 0;
  switch (NVT.getSimpleVT().SimpleTy) {
  default: return 0;
  case MVT::i8:
    if (isInc)
      Opc = X86::LOCK_INC8m;
    else if (isDec)
      Opc = X86::LOCK_DEC8m;
    else if (isSub) {
      if (isCN)
        Opc = X86::LOCK_SUB8mi;
      else
        Opc = X86::LOCK_SUB8mr;
    } else {
      if (isCN)
        Opc = X86::LOCK_ADD8mi;
      else
        Opc = X86::LOCK_ADD8mr;
    }
    break;
  case MVT::i16:
    if (isInc)
      Opc = X86::LOCK_INC16m;
    else if (isDec)
      Opc = X86::LOCK_DEC16m;
    else if (isSub) {
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_SUB16mi8;
        else
          Opc = X86::LOCK_SUB16mi;
      } else
        Opc = X86::LOCK_SUB16mr;
    } else {
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_ADD16mi8;
        else
          Opc = X86::LOCK_ADD16mi;
      } else
        Opc = X86::LOCK_ADD16mr;
    }
    break;
  case MVT::i32:
    if (isInc)
      Opc = X86::LOCK_INC32m;
    else if (isDec)
      Opc = X86::LOCK_DEC32m;
    else if (isSub) {
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_SUB32mi8;
        else
          Opc = X86::LOCK_SUB32mi;
      } else
        Opc = X86::LOCK_SUB32mr;
    } else {
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_ADD32mi8;
        else
          Opc = X86::LOCK_ADD32mi;
      } else
        Opc = X86::LOCK_ADD32mr;
    }
    break;
  case MVT::i64:
    if (isInc)
      Opc = X86::LOCK_INC64m;
    else if (isDec)
      Opc = X86::LOCK_DEC64m;
    else if (isSub) {
      Opc = X86::LOCK_SUB64mr;
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_SUB64mi8;
        else if (i64immSExt32(Val.getNode()))
          Opc = X86::LOCK_SUB64mi32;
      }
    } else {
      Opc = X86::LOCK_ADD64mr;
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = X86::LOCK_ADD64mi8;
        else if (i64immSExt32(Val.getNode()))
          Opc = X86::LOCK_ADD64mi32;
      }
    }
    break;
  }

  SDValue Undef = SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                 dl, NVT), 0);
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemSDNode>(Node)->getMemOperand();
  if (isInc || isDec) {
    SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Chain };
    SDValue Ret = SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops, 6), 0);
    cast<MachineSDNode>(Ret)->setMemRefs(MemOp, MemOp + 1);
    SDValue RetVals[] = { Undef, Ret };
    return CurDAG->getMergeValues(RetVals, 2, dl).getNode();
  } else {
    SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Val, Chain };
    SDValue Ret = SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops, 7), 0);
    cast<MachineSDNode>(Ret)->setMemRefs(MemOp, MemOp + 1);
    SDValue RetVals[] = { Undef, Ret };
    return CurDAG->getMergeValues(RetVals, 2, dl).getNode();
  }
}

enum AtomicOpc {
  OR,
  AND,
  XOR,
  AtomicOpcEnd
};

enum AtomicSz {
  ConstantI8,
  I8,
  SextConstantI16,
  ConstantI16,
  I16,
  SextConstantI32,
  ConstantI32,
  I32,
  SextConstantI64,
  ConstantI64,
  I64,
  AtomicSzEnd
};

static const unsigned int AtomicOpcTbl[AtomicOpcEnd][AtomicSzEnd] = {
  {
    X86::LOCK_OR8mi,
    X86::LOCK_OR8mr,
    X86::LOCK_OR16mi8,
    X86::LOCK_OR16mi,
    X86::LOCK_OR16mr,
    X86::LOCK_OR32mi8,
    X86::LOCK_OR32mi,
    X86::LOCK_OR32mr,
    X86::LOCK_OR64mi8,
    X86::LOCK_OR64mi32,
    X86::LOCK_OR64mr
  },
  {
    X86::LOCK_AND8mi,
    X86::LOCK_AND8mr,
    X86::LOCK_AND16mi8,
    X86::LOCK_AND16mi,
    X86::LOCK_AND16mr,
    X86::LOCK_AND32mi8,
    X86::LOCK_AND32mi,
    X86::LOCK_AND32mr,
    X86::LOCK_AND64mi8,
    X86::LOCK_AND64mi32,
    X86::LOCK_AND64mr
  },
  {
    X86::LOCK_XOR8mi,
    X86::LOCK_XOR8mr,
    X86::LOCK_XOR16mi8,
    X86::LOCK_XOR16mi,
    X86::LOCK_XOR16mr,
    X86::LOCK_XOR32mi8,
    X86::LOCK_XOR32mi,
    X86::LOCK_XOR32mr,
    X86::LOCK_XOR64mi8,
    X86::LOCK_XOR64mi32,
    X86::LOCK_XOR64mr
  }
};

SDNode *X86DAGToDAGISel::SelectAtomicLoadArith(SDNode *Node, EVT NVT) {
  if (Node->hasAnyUseOfValue(0))
    return 0;
  
  // Optimize common patterns for __sync_or_and_fetch and similar arith
  // operations where the result is not used. This allows us to use the "lock"
  // version of the arithmetic instruction.
  // FIXME: Same as for 'add' and 'sub', try to merge those down here.
  SDValue Chain = Node->getOperand(0);
  SDValue Ptr = Node->getOperand(1);
  SDValue Val = Node->getOperand(2);
  SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
  if (!SelectAddr(Node, Ptr, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4))
    return 0;

  // Which index into the table.
  enum AtomicOpc Op;
  switch (Node->getOpcode()) {
    case ISD::ATOMIC_LOAD_OR:
      Op = OR;
      break;
    case ISD::ATOMIC_LOAD_AND:
      Op = AND;
      break;
    case ISD::ATOMIC_LOAD_XOR:
      Op = XOR;
      break;
    default:
      return 0;
  }
  
  bool isCN = false;
  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Val);
  if (CN && (int32_t)CN->getSExtValue() == CN->getSExtValue()) {
    isCN = true;
    Val = CurDAG->getTargetConstant(CN->getSExtValue(), NVT);
  }
  
  unsigned Opc = 0;
  switch (NVT.getSimpleVT().SimpleTy) {
    default: return 0;
    case MVT::i8:
      if (isCN)
        Opc = AtomicOpcTbl[Op][ConstantI8];
      else
        Opc = AtomicOpcTbl[Op][I8];
      break;
    case MVT::i16:
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = AtomicOpcTbl[Op][SextConstantI16];
        else
          Opc = AtomicOpcTbl[Op][ConstantI16];
      } else
        Opc = AtomicOpcTbl[Op][I16];
      break;
    case MVT::i32:
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = AtomicOpcTbl[Op][SextConstantI32];
        else
          Opc = AtomicOpcTbl[Op][ConstantI32];
      } else
        Opc = AtomicOpcTbl[Op][I32];
      break;
    case MVT::i64:
      Opc = AtomicOpcTbl[Op][I64];
      if (isCN) {
        if (immSext8(Val.getNode()))
          Opc = AtomicOpcTbl[Op][SextConstantI64];
        else if (i64immSExt32(Val.getNode()))
          Opc = AtomicOpcTbl[Op][ConstantI64];
      }
      break;
  }
  
  assert(Opc != 0 && "Invalid arith lock transform!");

  DebugLoc dl = Node->getDebugLoc();
  SDValue Undef = SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                 dl, NVT), 0);
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemSDNode>(Node)->getMemOperand();
  SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Val, Chain };
  SDValue Ret = SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops, 7), 0);
  cast<MachineSDNode>(Ret)->setMemRefs(MemOp, MemOp + 1);
  SDValue RetVals[] = { Undef, Ret };
  return CurDAG->getMergeValues(RetVals, 2, dl).getNode();
}

/// HasNoSignedComparisonUses - Test whether the given X86ISD::CMP node has
/// any uses which require the SF or OF bits to be accurate.
static bool HasNoSignedComparisonUses(SDNode *N) {
  // Examine each user of the node.
  for (SDNode::use_iterator UI = N->use_begin(),
         UE = N->use_end(); UI != UE; ++UI) {
    // Only examine CopyToReg uses.
    if (UI->getOpcode() != ISD::CopyToReg)
      return false;
    // Only examine CopyToReg uses that copy to EFLAGS.
    if (cast<RegisterSDNode>(UI->getOperand(1))->getReg() !=
          X86::EFLAGS)
      return false;
    // Examine each user of the CopyToReg use.
    for (SDNode::use_iterator FlagUI = UI->use_begin(),
           FlagUE = UI->use_end(); FlagUI != FlagUE; ++FlagUI) {
      // Only examine the Flag result.
      if (FlagUI.getUse().getResNo() != 1) continue;
      // Anything unusual: assume conservatively.
      if (!FlagUI->isMachineOpcode()) return false;
      // Examine the opcode of the user.
      switch (FlagUI->getMachineOpcode()) {
      // These comparisons don't treat the most significant bit specially.
      case X86::SETAr: case X86::SETAEr: case X86::SETBr: case X86::SETBEr:
      case X86::SETEr: case X86::SETNEr: case X86::SETPr: case X86::SETNPr:
      case X86::SETAm: case X86::SETAEm: case X86::SETBm: case X86::SETBEm:
      case X86::SETEm: case X86::SETNEm: case X86::SETPm: case X86::SETNPm:
      case X86::JA_4: case X86::JAE_4: case X86::JB_4: case X86::JBE_4:
      case X86::JE_4: case X86::JNE_4: case X86::JP_4: case X86::JNP_4:
      case X86::CMOVA16rr: case X86::CMOVA16rm:
      case X86::CMOVA32rr: case X86::CMOVA32rm:
      case X86::CMOVA64rr: case X86::CMOVA64rm:
      case X86::CMOVAE16rr: case X86::CMOVAE16rm:
      case X86::CMOVAE32rr: case X86::CMOVAE32rm:
      case X86::CMOVAE64rr: case X86::CMOVAE64rm:
      case X86::CMOVB16rr: case X86::CMOVB16rm:
      case X86::CMOVB32rr: case X86::CMOVB32rm:
      case X86::CMOVB64rr: case X86::CMOVB64rm:
      case X86::CMOVBE16rr: case X86::CMOVBE16rm:
      case X86::CMOVBE32rr: case X86::CMOVBE32rm:
      case X86::CMOVBE64rr: case X86::CMOVBE64rm:
      case X86::CMOVE16rr: case X86::CMOVE16rm:
      case X86::CMOVE32rr: case X86::CMOVE32rm:
      case X86::CMOVE64rr: case X86::CMOVE64rm:
      case X86::CMOVNE16rr: case X86::CMOVNE16rm:
      case X86::CMOVNE32rr: case X86::CMOVNE32rm:
      case X86::CMOVNE64rr: case X86::CMOVNE64rm:
      case X86::CMOVNP16rr: case X86::CMOVNP16rm:
      case X86::CMOVNP32rr: case X86::CMOVNP32rm:
      case X86::CMOVNP64rr: case X86::CMOVNP64rm:
      case X86::CMOVP16rr: case X86::CMOVP16rm:
      case X86::CMOVP32rr: case X86::CMOVP32rm:
      case X86::CMOVP64rr: case X86::CMOVP64rm:
        continue;
      // Anything else: assume conservatively.
      default: return false;
      }
    }
  }
  return true;
}

SDNode *X86DAGToDAGISel::Select(SDNode *Node) {
  EVT NVT = Node->getValueType(0);
  unsigned Opc, MOpc;
  unsigned Opcode = Node->getOpcode();
  DebugLoc dl = Node->getDebugLoc();
  
  DEBUG(dbgs() << "Selecting: "; Node->dump(CurDAG); dbgs() << '\n');

  if (Node->isMachineOpcode()) {
    DEBUG(dbgs() << "== ";  Node->dump(CurDAG); dbgs() << '\n');
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

  case ISD::ATOMIC_LOAD_ADD: {
    SDNode *RetVal = SelectAtomicLoadAdd(Node, NVT);
    if (RetVal)
      return RetVal;
    break;
  }
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR: {
    SDNode *RetVal = SelectAtomicLoadArith(Node, NVT);
    if (RetVal)
      return RetVal;
    break;
  }
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {
    // For operations of the form (x << C1) op C2, check if we can use a smaller
    // encoding for C2 by transforming it into (x op (C2>>C1)) << C1.
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    if (N0->getOpcode() != ISD::SHL || !N0->hasOneUse())
      break;

    // i8 is unshrinkable, i16 should be promoted to i32.
    if (NVT != MVT::i32 && NVT != MVT::i64)
      break;

    ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N1);
    ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (!Cst || !ShlCst)
      break;

    int64_t Val = Cst->getSExtValue();
    uint64_t ShlVal = ShlCst->getZExtValue();

    // Make sure that we don't change the operation by removing bits.
    // This only matters for OR and XOR, AND is unaffected.
    if (Opcode != ISD::AND && ((Val >> ShlVal) << ShlVal) != Val)
      break;

    unsigned ShlOp, Op = 0;
    EVT CstVT = NVT;

    // Check the minimum bitwidth for the new constant.
    // TODO: AND32ri is the same as AND64ri32 with zext imm.
    // TODO: MOV32ri+OR64r is cheaper than MOV64ri64+OR64rr
    // TODO: Using 16 and 8 bit operations is also possible for or32 & xor32.
    if (!isInt<8>(Val) && isInt<8>(Val >> ShlVal))
      CstVT = MVT::i8;
    else if (!isInt<32>(Val) && isInt<32>(Val >> ShlVal))
      CstVT = MVT::i32;

    // Bail if there is no smaller encoding.
    if (NVT == CstVT)
      break;

    switch (NVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i32:
      assert(CstVT == MVT::i8);
      ShlOp = X86::SHL32ri;

      switch (Opcode) {
      case ISD::AND: Op = X86::AND32ri8; break;
      case ISD::OR:  Op =  X86::OR32ri8; break;
      case ISD::XOR: Op = X86::XOR32ri8; break;
      }
      break;
    case MVT::i64:
      assert(CstVT == MVT::i8 || CstVT == MVT::i32);
      ShlOp = X86::SHL64ri;

      switch (Opcode) {
      case ISD::AND: Op = CstVT==MVT::i8? X86::AND64ri8 : X86::AND64ri32; break;
      case ISD::OR:  Op = CstVT==MVT::i8?  X86::OR64ri8 :  X86::OR64ri32; break;
      case ISD::XOR: Op = CstVT==MVT::i8? X86::XOR64ri8 : X86::XOR64ri32; break;
      }
      break;
    }

    // Emit the smaller op and the shift.
    SDValue NewCst = CurDAG->getTargetConstant(Val >> ShlVal, CstVT);
    SDNode *New = CurDAG->getMachineNode(Op, dl, NVT, N0->getOperand(0),NewCst);
    return CurDAG->SelectNodeTo(Node, ShlOp, NVT, SDValue(New, 0),
                                getI8Imm(ShlVal));
    break;
  }
  case X86ISD::UMUL: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);
    
    unsigned LoReg;
    switch (NVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i8:  LoReg = X86::AL;  Opc = X86::MUL8r; break;
    case MVT::i16: LoReg = X86::AX;  Opc = X86::MUL16r; break;
    case MVT::i32: LoReg = X86::EAX; Opc = X86::MUL32r; break;
    case MVT::i64: LoReg = X86::RAX; Opc = X86::MUL64r; break;
    }
    
    SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, LoReg,
                                          N0, SDValue()).getValue(1);
    
    SDVTList VTs = CurDAG->getVTList(NVT, NVT, MVT::i32);
    SDValue Ops[] = {N1, InFlag};
    SDNode *CNode = CurDAG->getMachineNode(Opc, dl, VTs, Ops, 2);
    
    ReplaceUses(SDValue(Node, 0), SDValue(CNode, 0));
    ReplaceUses(SDValue(Node, 1), SDValue(CNode, 1));
    ReplaceUses(SDValue(Node, 2), SDValue(CNode, 2));
    return NULL;
  }
      
  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    bool isSigned = Opcode == ISD::SMUL_LOHI;
    if (!isSigned) {
      switch (NVT.getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::MUL8r;  MOpc = X86::MUL8m;  break;
      case MVT::i16: Opc = X86::MUL16r; MOpc = X86::MUL16m; break;
      case MVT::i32: Opc = X86::MUL32r; MOpc = X86::MUL32m; break;
      case MVT::i64: Opc = X86::MUL64r; MOpc = X86::MUL64m; break;
      }
    } else {
      switch (NVT.getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::IMUL8r;  MOpc = X86::IMUL8m;  break;
      case MVT::i16: Opc = X86::IMUL16r; MOpc = X86::IMUL16m; break;
      case MVT::i32: Opc = X86::IMUL32r; MOpc = X86::IMUL32m; break;
      case MVT::i64: Opc = X86::IMUL64r; MOpc = X86::IMUL64m; break;
      }
    }

    unsigned LoReg, HiReg;
    switch (NVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i8:  LoReg = X86::AL;  HiReg = X86::AH;  break;
    case MVT::i16: LoReg = X86::AX;  HiReg = X86::DX;  break;
    case MVT::i32: LoReg = X86::EAX; HiReg = X86::EDX; break;
    case MVT::i64: LoReg = X86::RAX; HiReg = X86::RDX; break;
    }

    SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
    bool foldedLoad = TryFoldLoad(Node, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
    // Multiply is commmutative.
    if (!foldedLoad) {
      foldedLoad = TryFoldLoad(Node, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
      if (foldedLoad)
        std::swap(N0, N1);
    }

    SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, LoReg,
                                            N0, SDValue()).getValue(1);

    if (foldedLoad) {
      SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                        InFlag };
      SDNode *CNode =
        CurDAG->getMachineNode(MOpc, dl, MVT::Other, MVT::Glue, Ops,
                               array_lengthof(Ops));
      InFlag = SDValue(CNode, 1);

      // Update the chain.
      ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
    } else {
      SDNode *CNode = CurDAG->getMachineNode(Opc, dl, MVT::Glue, N1, InFlag);
      InFlag = SDValue(CNode, 0);
    }

    // Prevent use of AH in a REX instruction by referencing AX instead.
    if (HiReg == X86::AH && Subtarget->is64Bit() &&
        !SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              X86::AX, MVT::i16, InFlag);
      InFlag = Result.getValue(2);
      // Get the low part if needed. Don't use getCopyFromReg for aliasing
      // registers.
      if (!SDValue(Node, 0).use_empty())
        ReplaceUses(SDValue(Node, 1),
          CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));

      // Shift AX down 8 bits.
      Result = SDValue(CurDAG->getMachineNode(X86::SHR16ri, dl, MVT::i16,
                                              Result,
                                     CurDAG->getTargetConstant(8, MVT::i8)), 0);
      // Then truncate it down to i8.
      ReplaceUses(SDValue(Node, 1),
        CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));
    }
    // Copy the low half of the result, if it is needed.
    if (!SDValue(Node, 0).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                LoReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 0), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    // Copy the high half of the result, if it is needed.
    if (!SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              HiReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 1), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    
    return NULL;
  }

  case ISD::SDIVREM:
  case ISD::UDIVREM: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    bool isSigned = Opcode == ISD::SDIVREM;
    if (!isSigned) {
      switch (NVT.getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::DIV8r;  MOpc = X86::DIV8m;  break;
      case MVT::i16: Opc = X86::DIV16r; MOpc = X86::DIV16m; break;
      case MVT::i32: Opc = X86::DIV32r; MOpc = X86::DIV32m; break;
      case MVT::i64: Opc = X86::DIV64r; MOpc = X86::DIV64m; break;
      }
    } else {
      switch (NVT.getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::IDIV8r;  MOpc = X86::IDIV8m;  break;
      case MVT::i16: Opc = X86::IDIV16r; MOpc = X86::IDIV16m; break;
      case MVT::i32: Opc = X86::IDIV32r; MOpc = X86::IDIV32m; break;
      case MVT::i64: Opc = X86::IDIV64r; MOpc = X86::IDIV64m; break;
      }
    }

    unsigned LoReg, HiReg, ClrReg;
    unsigned ClrOpcode, SExtOpcode;
    switch (NVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i8:
      LoReg = X86::AL;  ClrReg = HiReg = X86::AH;
      ClrOpcode  = 0;
      SExtOpcode = X86::CBW;
      break;
    case MVT::i16:
      LoReg = X86::AX;  HiReg = X86::DX;
      ClrOpcode  = X86::MOV16r0; ClrReg = X86::DX;
      SExtOpcode = X86::CWD;
      break;
    case MVT::i32:
      LoReg = X86::EAX; ClrReg = HiReg = X86::EDX;
      ClrOpcode  = X86::MOV32r0;
      SExtOpcode = X86::CDQ;
      break;
    case MVT::i64:
      LoReg = X86::RAX; ClrReg = HiReg = X86::RDX;
      ClrOpcode  = X86::MOV64r0;
      SExtOpcode = X86::CQO;
      break;
    }

    SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
    bool foldedLoad = TryFoldLoad(Node, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
    bool signBitIsZero = CurDAG->SignBitIsZero(N0);

    SDValue InFlag;
    if (NVT == MVT::i8 && (!isSigned || signBitIsZero)) {
      // Special case for div8, just use a move with zero extension to AX to
      // clear the upper 8 bits (AH).
      SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Move, Chain;
      if (TryFoldLoad(Node, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4)) {
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N0.getOperand(0) };
        Move =
          SDValue(CurDAG->getMachineNode(X86::MOVZX32rm8, dl, MVT::i32,
                                         MVT::Other, Ops,
                                         array_lengthof(Ops)), 0);
        Chain = Move.getValue(1);
        ReplaceUses(N0.getValue(1), Chain);
      } else {
        Move =
          SDValue(CurDAG->getMachineNode(X86::MOVZX32rr8, dl, MVT::i32, N0),0);
        Chain = CurDAG->getEntryNode();
      }
      Chain  = CurDAG->getCopyToReg(Chain, dl, X86::EAX, Move, SDValue());
      InFlag = Chain.getValue(1);
    } else {
      InFlag =
        CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl,
                             LoReg, N0, SDValue()).getValue(1);
      if (isSigned && !signBitIsZero) {
        // Sign extend the low part into the high part.
        InFlag =
          SDValue(CurDAG->getMachineNode(SExtOpcode, dl, MVT::Glue, InFlag),0);
      } else {
        // Zero out the high part, effectively zero extending the input.
        SDValue ClrNode =
          SDValue(CurDAG->getMachineNode(ClrOpcode, dl, NVT), 0);
        InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, ClrReg,
                                      ClrNode, InFlag).getValue(1);
      }
    }

    if (foldedLoad) {
      SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                        InFlag };
      SDNode *CNode =
        CurDAG->getMachineNode(MOpc, dl, MVT::Other, MVT::Glue, Ops,
                               array_lengthof(Ops));
      InFlag = SDValue(CNode, 1);
      // Update the chain.
      ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
    } else {
      InFlag =
        SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Glue, N1, InFlag), 0);
    }

    // Prevent use of AH in a REX instruction by referencing AX instead.
    // Shift it down 8 bits.
    if (HiReg == X86::AH && Subtarget->is64Bit() &&
        !SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              X86::AX, MVT::i16, InFlag);
      InFlag = Result.getValue(2);

      // If we also need AL (the quotient), get it by extracting a subreg from
      // Result. The fast register allocator does not like multiple CopyFromReg
      // nodes using aliasing registers.
      if (!SDValue(Node, 0).use_empty())
        ReplaceUses(SDValue(Node, 0),
          CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));

      // Shift AX right by 8 bits instead of using AH.
      Result = SDValue(CurDAG->getMachineNode(X86::SHR16ri, dl, MVT::i16,
                                         Result,
                                         CurDAG->getTargetConstant(8, MVT::i8)),
                       0);
      ReplaceUses(SDValue(Node, 1),
        CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));
    }
    // Copy the division (low) result, if it is needed.
    if (!SDValue(Node, 0).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                LoReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 0), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    // Copy the remainder (high) result, if it is needed.
    if (!SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              HiReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 1), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    return NULL;
  }

  case X86ISD::CMP: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    // Look for (X86cmp (and $op, $imm), 0) and see if we can convert it to
    // use a smaller encoding.
    if (N0.getOpcode() == ISD::TRUNCATE && N0.hasOneUse() &&
        HasNoSignedComparisonUses(Node))
      // Look past the truncate if CMP is the only use of it.
      N0 = N0.getOperand(0);
    if ((N0.getNode()->getOpcode() == ISD::AND ||
         (N0.getResNo() == 0 && N0.getNode()->getOpcode() == X86ISD::AND)) &&
        N0.getNode()->hasOneUse() &&
        N0.getValueType() != MVT::i8 &&
        X86::isZeroNode(N1)) {
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getNode()->getOperand(1));
      if (!C) break;

      // For example, convert "testl %eax, $8" to "testb %al, $8"
      if ((C->getZExtValue() & ~UINT64_C(0xff)) == 0 &&
          (!(C->getZExtValue() & 0x80) ||
           HasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(C->getZExtValue(), MVT::i8);
        SDValue Reg = N0.getNode()->getOperand(0);

        // On x86-32, only the ABCD registers have 8-bit subregisters.
        if (!Subtarget->is64Bit()) {
          TargetRegisterClass *TRC = 0;
          switch (N0.getValueType().getSimpleVT().SimpleTy) {
          case MVT::i32: TRC = &X86::GR32_ABCDRegClass; break;
          case MVT::i16: TRC = &X86::GR16_ABCDRegClass; break;
          default: llvm_unreachable("Unsupported TEST operand type!");
          }
          SDValue RC = CurDAG->getTargetConstant(TRC->getID(), MVT::i32);
          Reg = SDValue(CurDAG->getMachineNode(X86::COPY_TO_REGCLASS, dl,
                                               Reg.getValueType(), Reg, RC), 0);
        }

        // Extract the l-register.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl,
                                                        MVT::i8, Reg);

        // Emit a testb.
        return CurDAG->getMachineNode(X86::TEST8ri, dl, MVT::i32, Subreg, Imm);
      }

      // For example, "testl %eax, $2048" to "testb %ah, $8".
      if ((C->getZExtValue() & ~UINT64_C(0xff00)) == 0 &&
          (!(C->getZExtValue() & 0x8000) ||
           HasNoSignedComparisonUses(Node))) {
        // Shift the immediate right by 8 bits.
        SDValue ShiftedImm = CurDAG->getTargetConstant(C->getZExtValue() >> 8,
                                                       MVT::i8);
        SDValue Reg = N0.getNode()->getOperand(0);

        // Put the value in an ABCD register.
        TargetRegisterClass *TRC = 0;
        switch (N0.getValueType().getSimpleVT().SimpleTy) {
        case MVT::i64: TRC = &X86::GR64_ABCDRegClass; break;
        case MVT::i32: TRC = &X86::GR32_ABCDRegClass; break;
        case MVT::i16: TRC = &X86::GR16_ABCDRegClass; break;
        default: llvm_unreachable("Unsupported TEST operand type!");
        }
        SDValue RC = CurDAG->getTargetConstant(TRC->getID(), MVT::i32);
        Reg = SDValue(CurDAG->getMachineNode(X86::COPY_TO_REGCLASS, dl,
                                             Reg.getValueType(), Reg, RC), 0);

        // Extract the h-register.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_8bit_hi, dl,
                                                        MVT::i8, Reg);

        // Emit a testb.  The EXTRACT_SUBREG becomes a COPY that can only
        // target GR8_NOREX registers, so make sure the register class is
        // forced.
        return CurDAG->getMachineNode(X86::TEST8ri_NOREX, dl, MVT::i32,
                                      Subreg, ShiftedImm);
      }

      // For example, "testl %eax, $32776" to "testw %ax, $32776".
      if ((C->getZExtValue() & ~UINT64_C(0xffff)) == 0 &&
          N0.getValueType() != MVT::i16 &&
          (!(C->getZExtValue() & 0x8000) ||
           HasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(C->getZExtValue(), MVT::i16);
        SDValue Reg = N0.getNode()->getOperand(0);

        // Extract the 16-bit subregister.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_16bit, dl,
                                                        MVT::i16, Reg);

        // Emit a testw.
        return CurDAG->getMachineNode(X86::TEST16ri, dl, MVT::i32, Subreg, Imm);
      }

      // For example, "testq %rax, $268468232" to "testl %eax, $268468232".
      if ((C->getZExtValue() & ~UINT64_C(0xffffffff)) == 0 &&
          N0.getValueType() == MVT::i64 &&
          (!(C->getZExtValue() & 0x80000000) ||
           HasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(C->getZExtValue(), MVT::i32);
        SDValue Reg = N0.getNode()->getOperand(0);

        // Extract the 32-bit subregister.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_32bit, dl,
                                                        MVT::i32, Reg);

        // Emit a testl.
        return CurDAG->getMachineNode(X86::TEST32ri, dl, MVT::i32, Subreg, Imm);
      }
    }
    break;
  }
  case ISD::STORE: {
    // The DEC64m tablegen pattern is currently not able to match the case where
    // the EFLAGS on the original DEC are used.
    // we'll need to improve tablegen to allow flags to be transferred from a
    // node in the pattern to the result node.  probably with a new keyword
    // for example, we have this
    // def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
    //  [(store (add (loadi64 addr:$dst), -1), addr:$dst),
    //   (implicit EFLAGS)]>;
    // but maybe need something like this
    // def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
    //  [(store (add (loadi64 addr:$dst), -1), addr:$dst),
    //   (transferrable EFLAGS)]>;
    StoreSDNode *StoreNode = cast<StoreSDNode>(Node);
    SDValue Chain = StoreNode->getOperand(0);
    SDValue StoredVal = StoreNode->getOperand(1);
    SDValue Address = StoreNode->getOperand(2);
    SDValue Undef = StoreNode->getOperand(3);

    if (StoreNode->getMemOperand()->getSize() != 8 ||
        Undef->getOpcode() != ISD::UNDEF ||
        Chain->getOpcode() != ISD::LOAD ||
        StoredVal->getOpcode() != X86ISD::DEC ||
        StoredVal.getResNo() != 0 ||
        StoredVal->getOperand(0).getNode() != Chain.getNode())
      break;

    //OPC_CheckPredicate, 1, // Predicate_nontemporalstore
    if (StoreNode->isNonTemporal())
      break;

    LoadSDNode *LoadNode = cast<LoadSDNode>(Chain.getNode());
    if (LoadNode->getOperand(1) != Address ||
        LoadNode->getOperand(2) != Undef)
      break;

    if (!ISD::isNormalLoad(LoadNode))
      break;

    if (!ISD::isNormalStore(StoreNode))
      break;

    // check load chain has only one use (from the store)
    if (!Chain.hasOneUse())
      break;

    // Merge the input chains if they are not intra-pattern references.
    SDValue InputChain = LoadNode->getOperand(0);

    SDValue Base, Scale, Index, Disp, Segment;
    if (!SelectAddr(LoadNode, LoadNode->getBasePtr(),
                    Base, Scale, Index, Disp, Segment))
      break;

    MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(2);
    MemOp[0] = StoreNode->getMemOperand();
    MemOp[1] = LoadNode->getMemOperand();
    const SDValue Ops[] = { Base, Scale, Index, Disp, Segment, InputChain };
    MachineSDNode *Result = CurDAG->getMachineNode(X86::DEC64m,
                                                   Node->getDebugLoc(),
                                                   MVT::i32, MVT::Other, Ops,
                                                   array_lengthof(Ops));
    Result->setMemRefs(MemOp, MemOp + 2);

    ReplaceUses(SDValue(StoreNode, 0), SDValue(Result, 1));
    ReplaceUses(SDValue(StoredVal.getNode(), 1), SDValue(Result, 0));

    return Result;
  }
  }

  SDNode *ResNode = SelectCode(Node);

  DEBUG(dbgs() << "=> ";
        if (ResNode == NULL || ResNode == Node)
          Node->dump(CurDAG);
        else
          ResNode->dump(CurDAG);
        dbgs() << '\n');

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
    if (!SelectAddr(0, Op, Op0, Op1, Op2, Op3, Op4))
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
