//===-- llvm/CodeGen/SelectionDAGISel.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAGISel class, which is used as the common
// base class for SelectionDAG-based instruction selectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAG_ISEL_H
#define LLVM_CODEGEN_SELECTIONDAG_ISEL_H

#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Constant.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
  class FastISel;
  class SelectionDAGBuilder;
  class SDValue;
  class MachineRegisterInfo;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineInstr;
  class MachineModuleInfo;
  class DwarfWriter;
  class TargetLowering;
  class TargetInstrInfo;
  class FunctionLoweringInfo;
  class ScheduleHazardRecognizer;
  class GCFunctionInfo;
  class ScheduleDAGSDNodes;
 
/// SelectionDAGISel - This is the common base class used for SelectionDAG-based
/// pattern-matching instruction selectors.
class SelectionDAGISel : public MachineFunctionPass {
public:
  const TargetMachine &TM;
  TargetLowering &TLI;
  FunctionLoweringInfo *FuncInfo;
  MachineFunction *MF;
  MachineRegisterInfo *RegInfo;
  SelectionDAG *CurDAG;
  SelectionDAGBuilder *SDB;
  MachineBasicBlock *BB;
  AliasAnalysis *AA;
  GCFunctionInfo *GFI;
  CodeGenOpt::Level OptLevel;
  static char ID;

  explicit SelectionDAGISel(TargetMachine &tm,
                            CodeGenOpt::Level OL = CodeGenOpt::Default);
  virtual ~SelectionDAGISel();
  
  TargetLowering &getTargetLowering() { return TLI; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual bool runOnMachineFunction(MachineFunction &MF);

  unsigned MakeReg(EVT VT);

  virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {}
  
  /// PreprocessISelDAG - This hook allows targets to hack on the graph before
  /// instruction selection starts.
  virtual void PreprocessISelDAG() {}
  
  /// PostprocessISelDAG() - This hook allows the target to hack on the graph
  /// right after selection.
  virtual void PostprocessISelDAG() {}
  
  /// Select - Main hook targets implement to select a node.
  virtual SDNode *Select(SDNode *N) = 0;
  
  /// SelectInlineAsmMemoryOperand - Select the specified address as a target
  /// addressing mode, according to the specified constraint code.  If this does
  /// not match or is not implemented, return true.  The resultant operands
  /// (which will appear in the machine instruction) should be added to the
  /// OutOps vector.
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps) {
    return true;
  }

  /// IsProfitableToFold - Returns true if it's profitable to fold the specific
  /// operand node N of U during instruction selection that starts at Root.
  virtual bool IsProfitableToFold(SDValue N, SDNode *U, SDNode *Root) const;

  /// IsLegalToFold - Returns true if the specific operand node N of
  /// U can be folded during instruction selection that starts at Root.
  bool IsLegalToFold(SDValue N, SDNode *U, SDNode *Root,
                     bool IgnoreChains = false) const;

  /// CreateTargetHazardRecognizer - Return a newly allocated hazard recognizer
  /// to use for this target when scheduling the DAG.
  virtual ScheduleHazardRecognizer *CreateTargetHazardRecognizer();
  
  
  // Opcodes used by the DAG state machine:
  enum BuiltinOpcodes {
    OPC_Scope,
    OPC_RecordNode,
    OPC_RecordChild0, OPC_RecordChild1, OPC_RecordChild2, OPC_RecordChild3, 
    OPC_RecordChild4, OPC_RecordChild5, OPC_RecordChild6, OPC_RecordChild7,
    OPC_RecordMemRef,
    OPC_CaptureFlagInput,
    OPC_MoveChild,
    OPC_MoveParent,
    OPC_CheckSame,
    OPC_CheckPatternPredicate,
    OPC_CheckPredicate,
    OPC_CheckOpcode,
    OPC_SwitchOpcode,
    OPC_CheckType,
    OPC_SwitchType,
    OPC_CheckChild0Type, OPC_CheckChild1Type, OPC_CheckChild2Type,
    OPC_CheckChild3Type, OPC_CheckChild4Type, OPC_CheckChild5Type,
    OPC_CheckChild6Type, OPC_CheckChild7Type,
    OPC_CheckInteger,
    OPC_CheckCondCode,
    OPC_CheckValueType,
    OPC_CheckComplexPat,
    OPC_CheckAndImm, OPC_CheckOrImm,
    OPC_CheckFoldableChainNode,
    
    OPC_EmitInteger,
    OPC_EmitRegister,
    OPC_EmitConvertToTarget,
    OPC_EmitMergeInputChains,
    OPC_EmitMergeInputChains1_0,
    OPC_EmitMergeInputChains1_1,
    OPC_EmitCopyToReg,
    OPC_EmitNodeXForm,
    OPC_EmitNode,
    OPC_MorphNodeTo,
    OPC_MarkFlagResults,
    OPC_CompleteMatch
  };
  
  enum {
    OPFL_None       = 0,     // Node has no chain or flag input and isn't variadic.
    OPFL_Chain      = 1,     // Node has a chain input.
    OPFL_FlagInput  = 2,     // Node has a flag input.
    OPFL_FlagOutput = 4,     // Node has a flag output.
    OPFL_MemRefs    = 8,     // Node gets accumulated MemRefs.
    OPFL_Variadic0  = 1<<4,  // Node is variadic, root has 0 fixed inputs.
    OPFL_Variadic1  = 2<<4,  // Node is variadic, root has 1 fixed inputs.
    OPFL_Variadic2  = 3<<4,  // Node is variadic, root has 2 fixed inputs.
    OPFL_Variadic3  = 4<<4,  // Node is variadic, root has 3 fixed inputs.
    OPFL_Variadic4  = 5<<4,  // Node is variadic, root has 4 fixed inputs.
    OPFL_Variadic5  = 6<<4,  // Node is variadic, root has 5 fixed inputs.
    OPFL_Variadic6  = 7<<4,  // Node is variadic, root has 6 fixed inputs.
    
    OPFL_VariadicInfo = OPFL_Variadic6
  };
  
  /// getNumFixedFromVariadicInfo - Transform an EmitNode flags word into the
  /// number of fixed arity values that should be skipped when copying from the
  /// root.
  static inline int getNumFixedFromVariadicInfo(unsigned Flags) {
    return ((Flags&OPFL_VariadicInfo) >> 4)-1;
  }
  
  
protected:
  /// DAGSize - Size of DAG being instruction selected.
  ///
  unsigned DAGSize;
  
  /// ISelPosition - Node iterator marking the current position of
  /// instruction selection as it procedes through the topologically-sorted
  /// node list.
  SelectionDAG::allnodes_iterator ISelPosition;

  
  /// ISelUpdater - helper class to handle updates of the 
  /// instruction selection graph.
  class ISelUpdater : public SelectionDAG::DAGUpdateListener {
    SelectionDAG::allnodes_iterator &ISelPosition;
  public:
    explicit ISelUpdater(SelectionDAG::allnodes_iterator &isp)
      : ISelPosition(isp) {}
    
    /// NodeDeleted - Handle nodes deleted from the graph. If the
    /// node being deleted is the current ISelPosition node, update
    /// ISelPosition.
    ///
    virtual void NodeDeleted(SDNode *N, SDNode *E) {
      if (ISelPosition == SelectionDAG::allnodes_iterator(N))
        ++ISelPosition;
    }
    
    /// NodeUpdated - Ignore updates for now.
    virtual void NodeUpdated(SDNode *N) {}
  };
  
  /// ReplaceUses - replace all uses of the old node F with the use
  /// of the new node T.
  void ReplaceUses(SDValue F, SDValue T) {
    ISelUpdater ISU(ISelPosition);
    CurDAG->ReplaceAllUsesOfValueWith(F, T, &ISU);
  }
  
  /// ReplaceUses - replace all uses of the old nodes F with the use
  /// of the new nodes T.
  void ReplaceUses(const SDValue *F, const SDValue *T, unsigned Num) {
    ISelUpdater ISU(ISelPosition);
    CurDAG->ReplaceAllUsesOfValuesWith(F, T, Num, &ISU);
  }
  
  /// ReplaceUses - replace all uses of the old node F with the use
  /// of the new node T.
  void ReplaceUses(SDNode *F, SDNode *T) {
    ISelUpdater ISU(ISelPosition);
    CurDAG->ReplaceAllUsesWith(F, T, &ISU);
  }
  

  /// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
  /// by tblgen.  Others should not call it.
  void SelectInlineAsmMemoryOperands(std::vector<SDValue> &Ops);

  
public:
  // Calls to these predicates are generated by tblgen.
  bool CheckAndMask(SDValue LHS, ConstantSDNode *RHS,
                    int64_t DesiredMaskS) const;
  bool CheckOrMask(SDValue LHS, ConstantSDNode *RHS,
                    int64_t DesiredMaskS) const;
  
  
  /// CheckPatternPredicate - This function is generated by tblgen in the
  /// target.  It runs the specified pattern predicate and returns true if it
  /// succeeds or false if it fails.  The number is a private implementation
  /// detail to the code tblgen produces.
  virtual bool CheckPatternPredicate(unsigned PredNo) const {
    assert(0 && "Tblgen should generate the implementation of this!");
    return 0;
  }

  /// CheckNodePredicate - This function is generated by tblgen in the target.
  /// It runs node predicate number PredNo and returns true if it succeeds or
  /// false if it fails.  The number is a private implementation
  /// detail to the code tblgen produces.
  virtual bool CheckNodePredicate(SDNode *N, unsigned PredNo) const {
    assert(0 && "Tblgen should generate the implementation of this!");
    return 0;
  }
  
  virtual bool CheckComplexPattern(SDNode *Root, SDValue N, unsigned PatternNo,
                                   SmallVectorImpl<SDValue> &Result) {
    assert(0 && "Tblgen should generate the implementation of this!");
    return false;
  }
  
  virtual SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) {
    assert(0 && "Tblgen shoudl generate this!");
    return SDValue();
  }

  SDNode *SelectCodeCommon(SDNode *NodeToMatch,
                           const unsigned char *MatcherTable,
                           unsigned TableSize);
  
private:
  
  // Calls to these functions are generated by tblgen.
  SDNode *Select_INLINEASM(SDNode *N);
  SDNode *Select_UNDEF(SDNode *N);
  void CannotYetSelect(SDNode *N);

private:
  void DoInstructionSelection();
  SDNode *MorphNode(SDNode *Node, unsigned TargetOpc, SDVTList VTs,
                    const SDValue *Ops, unsigned NumOps, unsigned EmitNodeInfo);
  
  void SelectAllBasicBlocks(Function &Fn, MachineFunction &MF,
                            MachineModuleInfo *MMI,
                            DwarfWriter *DW,
                            const TargetInstrInfo &TII);
  void FinishBasicBlock();

  void SelectBasicBlock(BasicBlock *LLVMBB,
                        BasicBlock::iterator Begin,
                        BasicBlock::iterator End,
                        bool &HadTailCall);
  void CodeGenAndEmitDAG();
  void LowerArguments(BasicBlock *BB);
  
  void ShrinkDemandedOps();
  void ComputeLiveOutVRegInfo();

  void HandlePHINodesInSuccessorBlocks(BasicBlock *LLVMBB);

  bool HandlePHINodesInSuccessorBlocksFast(BasicBlock *LLVMBB, FastISel *F);

  /// Create the scheduler. If a specific scheduler was specified
  /// via the SchedulerRegistry, use it, otherwise select the
  /// one preferred by the target.
  ///
  ScheduleDAGSDNodes *CreateScheduler();
  
  /// OpcodeOffset - This is a cache used to dispatch efficiently into isel
  /// state machines that start with a OPC_SwitchOpcode node.
  std::vector<unsigned> OpcodeOffset;
  
  void UpdateChainsAndFlags(SDNode *NodeToMatch, SDValue InputChain,
                            const SmallVectorImpl<SDNode*> &ChainNodesMatched,
                            SDValue InputFlag,const SmallVectorImpl<SDNode*> &F,
                            bool isMorphNodeTo);
    
};

}

#endif /* LLVM_CODEGEN_SELECTIONDAG_ISEL_H */
