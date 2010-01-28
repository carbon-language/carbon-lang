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
#ifndef NDEBUG
#include "llvm/ADT/SmallSet.h"
#endif
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>
#include <set>

namespace llvm {

class AliasAnalysis;
class AllocaInst;
class BasicBlock;
class BitCastInst;
class BranchInst;
class CallInst;
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
class MachineFunction;
class MachineInstr;
class MachineRegisterInfo;
class PHINode;
class PtrToIntInst;
class ReturnInst;
class SDISelAsmOperandInfo;
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
/// Also, targets can overload any lowering method.
///
class SelectionDAGBuilder {
  MachineBasicBlock *CurMBB;

  /// CurDebugLoc - current file + line number.  Changes as we build the DAG.
  DebugLoc CurDebugLoc;

  DenseMap<const Value*, SDValue> NodeMap;

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
    CaseRec(MachineBasicBlock *bb, Constant *lt, Constant *ge, CaseRange r) :
    CaseBB(bb), LT(lt), GE(ge), Range(r) {}

    /// CaseBB - The MBB in which to emit the compare and branch
    MachineBasicBlock *CaseBB;
    /// LT, GE - If nonzero, we know the current case value must be less-than or
    /// greater-than-or-equal-to these Constants.
    Constant *LT;
    Constant *GE;
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
    CaseBlock(ISD::CondCode cc, Value *cmplhs, Value *cmprhs, Value *cmpmiddle,
              MachineBasicBlock *truebb, MachineBasicBlock *falsebb,
              MachineBasicBlock *me)
      : CC(cc), CmpLHS(cmplhs), CmpMHS(cmpmiddle), CmpRHS(cmprhs),
        TrueBB(truebb), FalseBB(falsebb), ThisBB(me) {}
    // CC - the condition code to use for the case block's setcc node
    ISD::CondCode CC;
    // CmpLHS/CmpRHS/CmpMHS - The LHS/MHS/RHS of the comparison to emit.
    // Emit by default LHS op RHS. MHS is used for range comparisons:
    // If MHS is not null: (LHS <= MHS) and (MHS <= RHS).
    Value *CmpLHS, *CmpMHS, *CmpRHS;
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
    JumpTableHeader(APInt F, APInt L, Value *SV, MachineBasicBlock *H,
                    bool E = false):
      First(F), Last(L), SValue(SV), HeaderBB(H), Emitted(E) {}
    APInt First;
    APInt Last;
    Value *SValue;
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
    BitTestBlock(APInt F, APInt R, Value* SV,
                 unsigned Rg, bool E,
                 MachineBasicBlock* P, MachineBasicBlock* D,
                 const BitTestInfo& C):
      First(F), Range(R), SValue(SV), Reg(Rg), Emitted(E),
      Parent(P), Default(D), Cases(C) { }
    APInt First;
    APInt Range;
    Value  *SValue;
    unsigned Reg;
    bool Emitted;
    MachineBasicBlock *Parent;
    MachineBasicBlock *Default;
    BitTestInfo Cases;
  };

public:
  // TLI - This is information that describes the available target features we
  // need for lowering.  This indicates when operations are unavailable,
  // implemented with a libcall, etc.
  TargetLowering &TLI;
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

  /// PHINodesToUpdate - A list of phi instructions whose operand list will
  /// be updated after processing the current basic block.
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;

  /// EdgeMapping - If an edge from CurMBB to any MBB is changed (e.g. due to
  /// scheduler custom lowering), track the change here.
  DenseMap<MachineBasicBlock*, MachineBasicBlock*> EdgeMapping;

  // Emit PHI-node-operand constants only once even if used by multiple
  // PHI nodes.
  DenseMap<Constant*, unsigned> ConstantsOut;

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

  SelectionDAGBuilder(SelectionDAG &dag, TargetLowering &tli,
                      FunctionLoweringInfo &funcinfo,
                      CodeGenOpt::Level ol)
    : CurDebugLoc(DebugLoc::getUnknownLoc()), SDNodeOrder(0),
      TLI(tli), DAG(dag), FuncInfo(funcinfo), OptLevel(ol),
      HasTailCall(false),
      Context(dag.getContext()) {
  }

  void init(GCFunctionInfo *gfi, AliasAnalysis &aa);

  /// clear - Clear out the curret SelectionDAG and the associated
  /// state and prepare this SelectionDAGBuilder object to be used
  /// for a new block. This doesn't clear out information about
  /// additional blocks that are needed to complete switch lowering
  /// or PHI node updating; that information is cleared out as it is
  /// consumed.
  void clear();

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
  void setCurDebugLoc(DebugLoc dl) { CurDebugLoc = dl; }

  unsigned getSDNodeOrder() const { return SDNodeOrder; }

  void CopyValueToVirtualRegister(Value *V, unsigned Reg);

  /// AssignOrderingToNode - Assign an ordering to the node. The order is gotten
  /// from how the code appeared in the source. The ordering is used by the
  /// scheduler to effectively turn off scheduling.
  void AssignOrderingToNode(const SDNode *Node);

  void visit(Instruction &I);

  void visit(unsigned Opcode, User &I);

  void setCurrentBasicBlock(MachineBasicBlock *MBB) { CurMBB = MBB; }

  SDValue getValue(const Value *V);

  void setValue(const Value *V, SDValue NewN) {
    SDValue &N = NodeMap[V];
    assert(N.getNode() == 0 && "Already set a value for this node!");
    N = NewN;
  }
  
  void GetRegistersForValue(SDISelAsmOperandInfo &OpInfo,
                            std::set<unsigned> &OutputRegs, 
                            std::set<unsigned> &InputRegs);

  void FindMergedConditions(Value *Cond, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *CurBB,
                            unsigned Opc);
  void EmitBranchForMergedCondition(Value *Cond, MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    MachineBasicBlock *CurBB);
  bool ShouldEmitAsBranches(const std::vector<CaseBlock> &Cases);
  bool isExportableFromCurrentBlock(Value *V, const BasicBlock *FromBB);
  void CopyToExportRegsIfNeeded(Value *V);
  void ExportFromCurrentBlock(Value *V);
  void LowerCallTo(CallSite CS, SDValue Callee, bool IsTailCall,
                   MachineBasicBlock *LandingPad = NULL);

private:
  // Terminator instructions.
  void visitRet(ReturnInst &I);
  void visitBr(BranchInst &I);
  void visitSwitch(SwitchInst &I);
  void visitIndirectBr(IndirectBrInst &I);
  void visitUnreachable(UnreachableInst &I) { /* noop */ }

  // Helpers for visitSwitch
  bool handleSmallSwitchRange(CaseRec& CR,
                              CaseRecVector& WorkList,
                              Value* SV,
                              MachineBasicBlock* Default);
  bool handleJTSwitchCase(CaseRec& CR,
                          CaseRecVector& WorkList,
                          Value* SV,
                          MachineBasicBlock* Default);
  bool handleBTSplitSwitchCase(CaseRec& CR,
                               CaseRecVector& WorkList,
                               Value* SV,
                               MachineBasicBlock* Default);
  bool handleBitTestsSwitchCase(CaseRec& CR,
                                CaseRecVector& WorkList,
                                Value* SV,
                                MachineBasicBlock* Default);  
public:
  void visitSwitchCase(CaseBlock &CB);
  void visitBitTestHeader(BitTestBlock &B);
  void visitBitTestCase(MachineBasicBlock* NextMBB,
                        unsigned Reg,
                        BitTestCase &B);
  void visitJumpTable(JumpTable &JT);
  void visitJumpTableHeader(JumpTable &JT, JumpTableHeader &JTH);
  
private:
  // These all get lowered before this pass.
  void visitInvoke(InvokeInst &I);
  void visitUnwind(UnwindInst &I);

  void visitBinary(User &I, unsigned OpCode);
  void visitShift(User &I, unsigned Opcode);
  void visitAdd(User &I)  { visitBinary(I, ISD::ADD); }
  void visitFAdd(User &I) { visitBinary(I, ISD::FADD); }
  void visitSub(User &I)  { visitBinary(I, ISD::SUB); }
  void visitFSub(User &I);
  void visitMul(User &I)  { visitBinary(I, ISD::MUL); }
  void visitFMul(User &I) { visitBinary(I, ISD::FMUL); }
  void visitURem(User &I) { visitBinary(I, ISD::UREM); }
  void visitSRem(User &I) { visitBinary(I, ISD::SREM); }
  void visitFRem(User &I) { visitBinary(I, ISD::FREM); }
  void visitUDiv(User &I) { visitBinary(I, ISD::UDIV); }
  void visitSDiv(User &I) { visitBinary(I, ISD::SDIV); }
  void visitFDiv(User &I) { visitBinary(I, ISD::FDIV); }
  void visitAnd (User &I) { visitBinary(I, ISD::AND); }
  void visitOr  (User &I) { visitBinary(I, ISD::OR); }
  void visitXor (User &I) { visitBinary(I, ISD::XOR); }
  void visitShl (User &I) { visitShift(I, ISD::SHL); }
  void visitLShr(User &I) { visitShift(I, ISD::SRL); }
  void visitAShr(User &I) { visitShift(I, ISD::SRA); }
  void visitICmp(User &I);
  void visitFCmp(User &I);
  // Visit the conversion instructions
  void visitTrunc(User &I);
  void visitZExt(User &I);
  void visitSExt(User &I);
  void visitFPTrunc(User &I);
  void visitFPExt(User &I);
  void visitFPToUI(User &I);
  void visitFPToSI(User &I);
  void visitUIToFP(User &I);
  void visitSIToFP(User &I);
  void visitPtrToInt(User &I);
  void visitIntToPtr(User &I);
  void visitBitCast(User &I);

  void visitExtractElement(User &I);
  void visitInsertElement(User &I);
  void visitShuffleVector(User &I);

  void visitExtractValue(ExtractValueInst &I);
  void visitInsertValue(InsertValueInst &I);

  void visitGetElementPtr(User &I);
  void visitSelect(User &I);

  void visitAlloca(AllocaInst &I);
  void visitLoad(LoadInst &I);
  void visitStore(StoreInst &I);
  void visitPHI(PHINode &I) { } // PHI nodes are handled specially.
  void visitCall(CallInst &I);
  bool visitMemCmpCall(CallInst &I);
  
  void visitInlineAsm(CallSite CS);
  const char *visitIntrinsicCall(CallInst &I, unsigned Intrinsic);
  void visitTargetIntrinsic(CallInst &I, unsigned Intrinsic);

  void visitPow(CallInst &I);
  void visitExp2(CallInst &I);
  void visitExp(CallInst &I);
  void visitLog(CallInst &I);
  void visitLog2(CallInst &I);
  void visitLog10(CallInst &I);

  void visitVAStart(CallInst &I);
  void visitVAArg(VAArgInst &I);
  void visitVAEnd(CallInst &I);
  void visitVACopy(CallInst &I);

  void visitUserOp1(Instruction &I) {
    llvm_unreachable("UserOp1 should not exist at instruction selection time!");
  }
  void visitUserOp2(Instruction &I) {
    llvm_unreachable("UserOp2 should not exist at instruction selection time!");
  }
  
  const char *implVisitBinaryAtomic(CallInst& I, ISD::NodeType Op);
  const char *implVisitAluOverflow(CallInst &I, ISD::NodeType Op);
};

} // end namespace llvm

#endif
