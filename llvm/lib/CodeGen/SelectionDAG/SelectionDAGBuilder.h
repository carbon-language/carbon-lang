//===-- SelectionDAGBuilder.h - Selection-DAG building --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating from LLVM IR into SelectionDAG IR.
//
//===----------------------------------------------------------------------===//

#ifndef SELECTIONDAGBUILDER_H
#define SELECTIONDAGBUILDER_H

#include "llvm/Constants.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>

namespace llvm {

class AliasAnalysis;
class AllocaInst;
class BasicBlock;
class BitCastInst;
class BranchInst;
class CallInst;
class DbgValueInst;
class ExtractElementInst;
class ExtractValueInst;
class FCmpInst;
class FPExtInst;
class FPToSIInst;
class FPToUIInst;
class FPTruncInst;
class Function;
class FunctionLoweringInfo;
class GetElementPtrInst;
class GCFunctionInfo;
class ICmpInst;
class IntToPtrInst;
class IndirectBrInst;
class InvokeInst;
class InsertElementInst;
class InsertValueInst;
class Instruction;
class LoadInst;
class MachineBasicBlock;
class MachineInstr;
class MachineRegisterInfo;
class MDNode;
class PHINode;
class PtrToIntInst;
class ReturnInst;
class SDDbgValue;
class SExtInst;
class SelectInst;
class ShuffleVectorInst;
class SIToFPInst;
class StoreInst;
class SwitchInst;
class TargetData;
class TargetLowering;
class TruncInst;
class UIToFPInst;
class UnreachableInst;
class UnwindInst;
class VAArgInst;
class ZExtInst;

//===----------------------------------------------------------------------===//
/// SelectionDAGBuilder - This is the common target-independent lowering
/// implementation that is parameterized by a TargetLowering object.
///
class SelectionDAGBuilder {
  /// CurDebugLoc - current file + line number.  Changes as we build the DAG.
  DebugLoc CurDebugLoc;

  DenseMap<const Value*, SDValue> NodeMap;
  
  /// UnusedArgNodeMap - Maps argument value for unused arguments. This is used
  /// to preserve debug information for incoming arguments.
  DenseMap<const Value*, SDValue> UnusedArgNodeMap;

  /// DanglingDebugInfo - Helper type for DanglingDebugInfoMap.
  class DanglingDebugInfo {
    const DbgValueInst* DI;
    DebugLoc dl;
    unsigned SDNodeOrder;
  public:
    DanglingDebugInfo() : DI(0), dl(DebugLoc()), SDNodeOrder(0) { }
    DanglingDebugInfo(const DbgValueInst *di, DebugLoc DL, unsigned SDNO) :
      DI(di), dl(DL), SDNodeOrder(SDNO) { }
    const DbgValueInst* getDI() { return DI; }
    DebugLoc getdl() { return dl; }
    unsigned getSDNodeOrder() { return SDNodeOrder; }
  };

  /// DanglingDebugInfoMap - Keeps track of dbg_values for which we have not
  /// yet seen the referent.  We defer handling these until we do see it.
  DenseMap<const Value*, DanglingDebugInfo> DanglingDebugInfoMap;

public:
  /// PendingLoads - Loads are not emitted to the program immediately.  We bunch
  /// them up and then emit token factor nodes when possible.  This allows us to
  /// get simple disambiguation between loads without worrying about alias
  /// analysis.
  SmallVector<SDValue, 8> PendingLoads;
private:

  /// PendingExports - CopyToReg nodes that copy values to virtual registers
  /// for export to other blocks need to be emitted before any terminator
  /// instruction, but they have no other ordering requirements. We bunch them
  /// up and the emit a single tokenfactor for them just before terminator
  /// instructions.
  SmallVector<SDValue, 8> PendingExports;

  /// SDNodeOrder - A unique monotonically increasing number used to order the
  /// SDNodes we create.
  unsigned SDNodeOrder;

  /// Case - A struct to record the Value for a switch case, and the
  /// case's target basic block.
  struct Case {
    Constant* Low;
    Constant* High;
    MachineBasicBlock* BB;

    Case() : Low(0), High(0), BB(0) { }
    Case(Constant* low, Constant* high, MachineBasicBlock* bb) :
      Low(low), High(high), BB(bb) { }
    APInt size() const {
      const APInt &rHigh = cast<ConstantInt>(High)->getValue();
      const APInt &rLow  = cast<ConstantInt>(Low)->getValue();
      return (rHigh - rLow + 1ULL);
    }
  };

  struct CaseBits {
    uint64_t Mask;
    MachineBasicBlock* BB;
    unsigned Bits;

    CaseBits(uint64_t mask, MachineBasicBlock* bb, unsigned bits):
      Mask(mask), BB(bb), Bits(bits) { }
  };

  typedef std::vector<Case>           CaseVector;
  typedef std::vector<CaseBits>       CaseBitsVector;
  typedef CaseVector::iterator        CaseItr;
  typedef std::pair<CaseItr, CaseItr> CaseRange;

  /// CaseRec - A struct with ctor used in lowering switches to a binary tree
  /// of conditional branches.
  struct CaseRec {
    CaseRec(MachineBasicBlock *bb, const Constant *lt, const Constant *ge,
            CaseRange r) :
    CaseBB(bb), LT(lt), GE(ge), Range(r) {}

    /// CaseBB - The MBB in which to emit the compare and branch
    MachineBasicBlock *CaseBB;
    /// LT, GE - If nonzero, we know the current case value must be less-than or
    /// greater-than-or-equal-to these Constants.
    const Constant *LT;
    const Constant *GE;
    /// Range - A pair of iterators representing the range of case values to be
    /// processed at this point in the binary search tree.
    CaseRange Range;
  };

  typedef std::vector<CaseRec> CaseRecVector;

  /// The comparison function for sorting the switch case values in the vector.
  /// WARNING: Case ranges should be disjoint!
  struct CaseCmp {
    bool operator()(const Case &C1, const Case &C2) {
      assert(isa<ConstantInt>(C1.Low) && isa<ConstantInt>(C2.High));
      const ConstantInt* CI1 = cast<const ConstantInt>(C1.Low);
      const ConstantInt* CI2 = cast<const ConstantInt>(C2.High);
      return CI1->getValue().slt(CI2->getValue());
    }
  };

  struct CaseBitsCmp {
    bool operator()(const CaseBits &C1, const CaseBits &C2) {
      return C1.Bits > C2.Bits;
    }
  };

  size_t Clusterify(CaseVector &Cases, const SwitchInst &SI);

  /// CaseBlock - This structure is used to communicate between
  /// SelectionDAGBuilder and SDISel for the code generation of additional basic
  /// blocks needed by multi-case switch statements.
  struct CaseBlock {
    CaseBlock(ISD::CondCode cc, const Value *cmplhs, const Value *cmprhs,
              const Value *cmpmiddle,
              MachineBasicBlock *truebb, MachineBasicBlock *falsebb,
              MachineBasicBlock *me)
      : CC(cc), CmpLHS(cmplhs), CmpMHS(cmpmiddle), CmpRHS(cmprhs),
        TrueBB(truebb), FalseBB(falsebb), ThisBB(me) {}
    // CC - the condition code to use for the case block's setcc node
    ISD::CondCode CC;
    // CmpLHS/CmpRHS/CmpMHS - The LHS/MHS/RHS of the comparison to emit.
    // Emit by default LHS op RHS. MHS is used for range comparisons:
    // If MHS is not null: (LHS <= MHS) and (MHS <= RHS).
    const Value *CmpLHS, *CmpMHS, *CmpRHS;
    // TrueBB/FalseBB - the block to branch to if the setcc is true/false.
    MachineBasicBlock *TrueBB, *FalseBB;
    // ThisBB - the block into which to emit the code for the setcc and branches
    MachineBasicBlock *ThisBB;
  };
  struct JumpTable {
    JumpTable(unsigned R, unsigned J, MachineBasicBlock *M,
              MachineBasicBlock *D): Reg(R), JTI(J), MBB(M), Default(D) {}
  
    /// Reg - the virtual register containing the index of the jump table entry
    //. to jump to.
    unsigned Reg;
    /// JTI - the JumpTableIndex for this jump table in the function.
    unsigned JTI;
    /// MBB - the MBB into which to emit the code for the indirect jump.
    MachineBasicBlock *MBB;
    /// Default - the MBB of the default bb, which is a successor of the range
    /// check MBB.  This is when updating PHI nodes in successors.
    MachineBasicBlock *Default;
  };
  struct JumpTableHeader {
    JumpTableHeader(APInt F, APInt L, const Value *SV, MachineBasicBlock *H,
                    bool E = false):
      First(F), Last(L), SValue(SV), HeaderBB(H), Emitted(E) {}
    APInt First;
    APInt Last;
    const Value *SValue;
    MachineBasicBlock *HeaderBB;
    bool Emitted;
  };
  typedef std::pair<JumpTableHeader, JumpTable> JumpTableBlock;

  struct BitTestCase {
    BitTestCase(uint64_t M, MachineBasicBlock* T, MachineBasicBlock* Tr):
      Mask(M), ThisBB(T), TargetBB(Tr) { }
    uint64_t Mask;
    MachineBasicBlock *ThisBB;
    MachineBasicBlock *TargetBB;
  };

  typedef SmallVector<BitTestCase, 3> BitTestInfo;

  struct BitTestBlock {
    BitTestBlock(APInt F, APInt R, const Value* SV,
                 unsigned Rg, EVT RgVT, bool E,
                 MachineBasicBlock* P, MachineBasicBlock* D,
                 const BitTestInfo& C):
      First(F), Range(R), SValue(SV), Reg(Rg), RegVT(RgVT), Emitted(E),
      Parent(P), Default(D), Cases(C) { }
    APInt First;
    APInt Range;
    const Value *SValue;
    unsigned Reg;
    EVT RegVT;
    bool Emitted;
    MachineBasicBlock *Parent;
    MachineBasicBlock *Default;
    BitTestInfo Cases;
  };

public:
  // TLI - This is information that describes the available target features we
  // need for lowering.  This indicates when operations are unavailable,
  // implemented with a libcall, etc.
  const TargetMachine &TM;
  const TargetLowering &TLI;
  SelectionDAG &DAG;
  const TargetData *TD;
  AliasAnalysis *AA;

  /// SwitchCases - Vector of CaseBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<CaseBlock> SwitchCases;
  /// JTCases - Vector of JumpTable structures used to communicate
  /// SwitchInst code generation information.
  std::vector<JumpTableBlock> JTCases;
  /// BitTestCases - Vector of BitTestBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<BitTestBlock> BitTestCases;

  // Emit PHI-node-operand constants only once even if used by multiple
  // PHI nodes.
  DenseMap<const Constant *, unsigned> ConstantsOut;

  /// FuncInfo - Information about the function as a whole.
  ///
  FunctionLoweringInfo &FuncInfo;

  /// OptLevel - What optimization level we're generating code for.
  /// 
  CodeGenOpt::Level OptLevel;
  
  /// GFI - Garbage collection metadata for the function.
  GCFunctionInfo *GFI;

  /// HasTailCall - This is set to true if a call in the current
  /// block has been translated as a tail call. In this case,
  /// no subsequent DAG nodes should be created.
  ///
  bool HasTailCall;

  LLVMContext *Context;

  SelectionDAGBuilder(SelectionDAG &dag, FunctionLoweringInfo &funcinfo,
                      CodeGenOpt::Level ol)
    : SDNodeOrder(0), TM(dag.getTarget()), TLI(dag.getTargetLoweringInfo()),
      DAG(dag), FuncInfo(funcinfo), OptLevel(ol),
      HasTailCall(false), Context(dag.getContext()) {
  }

  void init(GCFunctionInfo *gfi, AliasAnalysis &aa);

  /// clear - Clear out the current SelectionDAG and the associated
  /// state and prepare this SelectionDAGBuilder object to be used
  /// for a new block. This doesn't clear out information about
  /// additional blocks that are needed to complete switch lowering
  /// or PHI node updating; that information is cleared out as it is
  /// consumed.
  void clear();

  /// clearDanglingDebugInfo - Clear the dangling debug information
  /// map. This function is seperated from the clear so that debug
  /// information that is dangling in a basic block can be properly
  /// resolved in a different basic block. This allows the
  /// SelectionDAG to resolve dangling debug information attached
  /// to PHI nodes.
  void clearDanglingDebugInfo();

  /// getRoot - Return the current virtual root of the Selection DAG,
  /// flushing any PendingLoad items. This must be done before emitting
  /// a store or any other node that may need to be ordered after any
  /// prior load instructions.
  ///
  SDValue getRoot();

  /// getControlRoot - Similar to getRoot, but instead of flushing all the
  /// PendingLoad items, flush all the PendingExports items. It is necessary
  /// to do this before emitting a terminator instruction.
  ///
  SDValue getControlRoot();

  DebugLoc getCurDebugLoc() const { return CurDebugLoc; }

  unsigned getSDNodeOrder() const { return SDNodeOrder; }

  void CopyValueToVirtualRegister(const Value *V, unsigned Reg);

  /// AssignOrderingToNode - Assign an ordering to the node. The order is gotten
  /// from how the code appeared in the source. The ordering is used by the
  /// scheduler to effectively turn off scheduling.
  void AssignOrderingToNode(const SDNode *Node);

  void visit(const Instruction &I);

  void visit(unsigned Opcode, const User &I);

  // resolveDanglingDebugInfo - if we saw an earlier dbg_value referring to V,
  // generate the debug data structures now that we've seen its definition.
  void resolveDanglingDebugInfo(const Value *V, SDValue Val);
  SDValue getValue(const Value *V);
  SDValue getNonRegisterValue(const Value *V);
  SDValue getValueImpl(const Value *V);

  void setValue(const Value *V, SDValue NewN) {
    SDValue &N = NodeMap[V];
    assert(N.getNode() == 0 && "Already set a value for this node!");
    N = NewN;
  }
  
  void setUnusedArgValue(const Value *V, SDValue NewN) {
    SDValue &N = UnusedArgNodeMap[V];
    assert(N.getNode() == 0 && "Already set a value for this node!");
    N = NewN;
  }

  void FindMergedConditions(const Value *Cond, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *CurBB,
                            MachineBasicBlock *SwitchBB, unsigned Opc);
  void EmitBranchForMergedCondition(const Value *Cond, MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    MachineBasicBlock *CurBB,
                                    MachineBasicBlock *SwitchBB);
  bool ShouldEmitAsBranches(const std::vector<CaseBlock> &Cases);
  bool isExportableFromCurrentBlock(const Value *V, const BasicBlock *FromBB);
  void CopyToExportRegsIfNeeded(const Value *V);
  void ExportFromCurrentBlock(const Value *V);
  void LowerCallTo(ImmutableCallSite CS, SDValue Callee, bool IsTailCall,
                   MachineBasicBlock *LandingPad = NULL);

  /// UpdateSplitBlock - When an MBB was split during scheduling, update the
  /// references that ned to refer to the last resulting block.
  void UpdateSplitBlock(MachineBasicBlock *First, MachineBasicBlock *Last);

private:
  // Terminator instructions.
  void visitRet(const ReturnInst &I);
  void visitBr(const BranchInst &I);
  void visitSwitch(const SwitchInst &I);
  void visitIndirectBr(const IndirectBrInst &I);
  void visitUnreachable(const UnreachableInst &I) { /* noop */ }

  // Helpers for visitSwitch
  bool handleSmallSwitchRange(CaseRec& CR,
                              CaseRecVector& WorkList,
                              const Value* SV,
                              MachineBasicBlock* Default,
                              MachineBasicBlock *SwitchBB);
  bool handleJTSwitchCase(CaseRec& CR,
                          CaseRecVector& WorkList,
                          const Value* SV,
                          MachineBasicBlock* Default,
                          MachineBasicBlock *SwitchBB);
  bool handleBTSplitSwitchCase(CaseRec& CR,
                               CaseRecVector& WorkList,
                               const Value* SV,
                               MachineBasicBlock* Default,
                               MachineBasicBlock *SwitchBB);
  bool handleBitTestsSwitchCase(CaseRec& CR,
                                CaseRecVector& WorkList,
                                const Value* SV,
                                MachineBasicBlock* Default,
                                MachineBasicBlock *SwitchBB);

  uint32_t getEdgeWeight(MachineBasicBlock *Src, MachineBasicBlock *Dst);
  void addSuccessorWithWeight(MachineBasicBlock *Src, MachineBasicBlock *Dst);
public:
  void visitSwitchCase(CaseBlock &CB,
                       MachineBasicBlock *SwitchBB);
  void visitBitTestHeader(BitTestBlock &B, MachineBasicBlock *SwitchBB);
  void visitBitTestCase(BitTestBlock &BB,
                        MachineBasicBlock* NextMBB,
                        unsigned Reg,
                        BitTestCase &B,
                        MachineBasicBlock *SwitchBB);
  void visitJumpTable(JumpTable &JT);
  void visitJumpTableHeader(JumpTable &JT, JumpTableHeader &JTH,
                            MachineBasicBlock *SwitchBB);
  
private:
  // These all get lowered before this pass.
  void visitInvoke(const InvokeInst &I);
  void visitUnwind(const UnwindInst &I);

  void visitBinary(const User &I, unsigned OpCode);
  void visitShift(const User &I, unsigned Opcode);
  void visitAdd(const User &I)  { visitBinary(I, ISD::ADD); }
  void visitFAdd(const User &I) { visitBinary(I, ISD::FADD); }
  void visitSub(const User &I)  { visitBinary(I, ISD::SUB); }
  void visitFSub(const User &I);
  void visitMul(const User &I)  { visitBinary(I, ISD::MUL); }
  void visitFMul(const User &I) { visitBinary(I, ISD::FMUL); }
  void visitURem(const User &I) { visitBinary(I, ISD::UREM); }
  void visitSRem(const User &I) { visitBinary(I, ISD::SREM); }
  void visitFRem(const User &I) { visitBinary(I, ISD::FREM); }
  void visitUDiv(const User &I) { visitBinary(I, ISD::UDIV); }
  void visitSDiv(const User &I);
  void visitFDiv(const User &I) { visitBinary(I, ISD::FDIV); }
  void visitAnd (const User &I) { visitBinary(I, ISD::AND); }
  void visitOr  (const User &I) { visitBinary(I, ISD::OR); }
  void visitXor (const User &I) { visitBinary(I, ISD::XOR); }
  void visitShl (const User &I) { visitShift(I, ISD::SHL); }
  void visitLShr(const User &I) { visitShift(I, ISD::SRL); }
  void visitAShr(const User &I) { visitShift(I, ISD::SRA); }
  void visitICmp(const User &I);
  void visitFCmp(const User &I);
  // Visit the conversion instructions
  void visitTrunc(const User &I);
  void visitZExt(const User &I);
  void visitSExt(const User &I);
  void visitFPTrunc(const User &I);
  void visitFPExt(const User &I);
  void visitFPToUI(const User &I);
  void visitFPToSI(const User &I);
  void visitUIToFP(const User &I);
  void visitSIToFP(const User &I);
  void visitPtrToInt(const User &I);
  void visitIntToPtr(const User &I);
  void visitBitCast(const User &I);

  void visitExtractElement(const User &I);
  void visitInsertElement(const User &I);
  void visitShuffleVector(const User &I);

  void visitExtractValue(const ExtractValueInst &I);
  void visitInsertValue(const InsertValueInst &I);

  void visitGetElementPtr(const User &I);
  void visitSelect(const User &I);

  void visitAlloca(const AllocaInst &I);
  void visitLoad(const LoadInst &I);
  void visitStore(const StoreInst &I);
  void visitPHI(const PHINode &I);
  void visitCall(const CallInst &I);
  bool visitMemCmpCall(const CallInst &I);
  
  void visitInlineAsm(ImmutableCallSite CS);
  const char *visitIntrinsicCall(const CallInst &I, unsigned Intrinsic);
  void visitTargetIntrinsic(const CallInst &I, unsigned Intrinsic);

  void visitPow(const CallInst &I);
  void visitExp2(const CallInst &I);
  void visitExp(const CallInst &I);
  void visitLog(const CallInst &I);
  void visitLog2(const CallInst &I);
  void visitLog10(const CallInst &I);

  void visitVAStart(const CallInst &I);
  void visitVAArg(const VAArgInst &I);
  void visitVAEnd(const CallInst &I);
  void visitVACopy(const CallInst &I);

  void visitUserOp1(const Instruction &I) {
    llvm_unreachable("UserOp1 should not exist at instruction selection time!");
  }
  void visitUserOp2(const Instruction &I) {
    llvm_unreachable("UserOp2 should not exist at instruction selection time!");
  }
  
  const char *implVisitBinaryAtomic(const CallInst& I, ISD::NodeType Op);
  const char *implVisitAluOverflow(const CallInst &I, ISD::NodeType Op);

  void HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB);

  /// EmitFuncArgumentDbgValue - If V is an function argument then create
  /// corresponding DBG_VALUE machine instruction for it now. At the end of 
  /// instruction selection, they will be inserted to the entry BB.
  bool EmitFuncArgumentDbgValue(const Value *V, MDNode *Variable,
                                int64_t Offset, const SDValue &N);
};

} // end namespace llvm

#endif
