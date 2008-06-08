//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Constants.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/CodeGen/Collector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
static cl::opt<bool>
ViewSUnitDAGs("view-sunit-dags", cl::Hidden,
      cl::desc("Pop up a window to show SUnit dags after they are processed"));
#else
static const bool ViewISelDAGs = 0, ViewSchedDAGs = 0, ViewSUnitDAGs = 0;
#endif

//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterScheduler::Registry;

//===---------------------------------------------------------------------===//
///
/// ISHeuristic command line option for instruction schedulers.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterScheduler::FunctionPassCtor, false,
               RegisterPassParser<RegisterScheduler> >
ISHeuristic("pre-RA-sched",
            cl::init(&createDefaultScheduler),
            cl::desc("Instruction schedulers available (before register"
                     " allocation):"));

static RegisterScheduler
defaultListDAGScheduler("default", "  Best scheduler for the target",
                        createDefaultScheduler);

namespace { struct SDISelAsmOperandInfo; }

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
static unsigned ComputeLinearIndex(const TargetLowering &TLI, const Type *Ty,
                                   const unsigned *Indices,
                                   const unsigned *IndicesEnd,
                                   unsigned CurIndex = 0) {
  // Base case: We're done.
  if (Indices == IndicesEnd)
    return CurIndex;

  // Otherwise we need to recurse. A non-negative value is used to
  // indicate the final result value; a negative value carries the
  // complemented position to continue the search.
  CurIndex = ~CurIndex;

  // Given a struct type, recursively traverse the elements.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    for (StructType::element_iterator EI = STy->element_begin(),
                                      EE = STy->element_end();
        EI != EE; ++EI) {
      CurIndex = ComputeLinearIndex(TLI, *EI, Indices+1, IndicesEnd,
                                    ~CurIndex);
      if ((int)CurIndex >= 0)
        return CurIndex;
    }
  }
  // Given an array type, recursively traverse the elements.
  else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = ATy->getElementType();
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i) {
      CurIndex = ComputeLinearIndex(TLI, EltTy, Indices+1, IndicesEnd,
                                    ~CurIndex);
      if ((int)CurIndex >= 0)
        return CurIndex;
    }
  }
  // We haven't found the type we're looking for, so keep searching.
  return CurIndex;
}

/// ComputeValueVTs - Given an LLVM IR type, compute a sequence of
/// MVTs that represent all the individual underlying
/// non-aggregate types that comprise it.
///
/// If Offsets is non-null, it points to a vector to be filled in
/// with the in-memory offsets of each of the individual values.
///
static void ComputeValueVTs(const TargetLowering &TLI, const Type *Ty,
                            SmallVectorImpl<MVT> &ValueVTs,
                            SmallVectorImpl<uint64_t> *Offsets = 0,
                            uint64_t StartingOffset = 0) {
  // Given a struct type, recursively traverse the elements.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = TLI.getTargetData()->getStructLayout(STy);
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
         EI != EE; ++EI)
      ComputeValueVTs(TLI, *EI, ValueVTs, Offsets,
                      StartingOffset + SL->getElementOffset(EI - EB));
    return;
  }
  // Given an array type, recursively traverse the elements.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = ATy->getElementType();
    uint64_t EltSize = TLI.getTargetData()->getABITypeSize(EltTy);
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
      ComputeValueVTs(TLI, EltTy, ValueVTs, Offsets,
                      StartingOffset + i * EltSize);
    return;
  }
  // Base case: we can get an MVT for this LLVM IR type.
  ValueVTs.push_back(TLI.getValueType(Ty));
  if (Offsets)
    Offsets->push_back(StartingOffset);
}

namespace {
  /// RegsForValue - This struct represents the registers (physical or virtual)
  /// that a particular set of values is assigned, and the type information about
  /// the value. The most common situation is to represent one value at a time,
  /// but struct or array values are handled element-wise as multiple values.
  /// The splitting of aggregates is performed recursively, so that we never
  /// have aggregate-typed registers. The values at this point do not necessarily
  /// have legal types, so each value may require one or more registers of some
  /// legal type.
  /// 
  struct VISIBILITY_HIDDEN RegsForValue {
    /// TLI - The TargetLowering object.
    ///
    const TargetLowering *TLI;

    /// ValueVTs - The value types of the values, which may not be legal, and
    /// may need be promoted or synthesized from one or more registers.
    ///
    SmallVector<MVT, 4> ValueVTs;
    
    /// RegVTs - The value types of the registers. This is the same size as
    /// ValueVTs and it records, for each value, what the type of the assigned
    /// register or registers are. (Individual values are never synthesized
    /// from more than one type of register.)
    ///
    /// With virtual registers, the contents of RegVTs is redundant with TLI's
    /// getRegisterType member function, however when with physical registers
    /// it is necessary to have a separate record of the types.
    ///
    SmallVector<MVT, 4> RegVTs;
    
    /// Regs - This list holds the registers assigned to the values.
    /// Each legal or promoted value requires one register, and each
    /// expanded value requires multiple registers.
    ///
    SmallVector<unsigned, 4> Regs;
    
    RegsForValue() : TLI(0) {}
    
    RegsForValue(const TargetLowering &tli,
                 const SmallVector<unsigned, 4> &regs, 
                 MVT regvt, MVT valuevt)
      : TLI(&tli),  ValueVTs(1, valuevt), RegVTs(1, regvt), Regs(regs) {}
    RegsForValue(const TargetLowering &tli,
                 const SmallVector<unsigned, 4> &regs, 
                 const SmallVector<MVT, 4> &regvts,
                 const SmallVector<MVT, 4> &valuevts)
      : TLI(&tli), ValueVTs(valuevts), RegVTs(regvts), Regs(regs) {}
    RegsForValue(const TargetLowering &tli,
                 unsigned Reg, const Type *Ty) : TLI(&tli) {
      ComputeValueVTs(tli, Ty, ValueVTs);

      for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
        MVT ValueVT = ValueVTs[Value];
        unsigned NumRegs = TLI->getNumRegisters(ValueVT);
        MVT RegisterVT = TLI->getRegisterType(ValueVT);
        for (unsigned i = 0; i != NumRegs; ++i)
          Regs.push_back(Reg + i);
        RegVTs.push_back(RegisterVT);
        Reg += NumRegs;
      }
    }
    
    /// append - Add the specified values to this one.
    void append(const RegsForValue &RHS) {
      TLI = RHS.TLI;
      ValueVTs.append(RHS.ValueVTs.begin(), RHS.ValueVTs.end());
      RegVTs.append(RHS.RegVTs.begin(), RHS.RegVTs.end());
      Regs.append(RHS.Regs.begin(), RHS.Regs.end());
    }
    
    
    /// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
    /// this value and returns the result as a ValueVTs value.  This uses 
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    /// If the Flag pointer is NULL, no flag is used.
    SDOperand getCopyFromRegs(SelectionDAG &DAG,
                              SDOperand &Chain, SDOperand *Flag) const;

    /// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
    /// specified value into the registers specified by this object.  This uses 
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    /// If the Flag pointer is NULL, no flag is used.
    void getCopyToRegs(SDOperand Val, SelectionDAG &DAG,
                       SDOperand &Chain, SDOperand *Flag) const;
    
    /// AddInlineAsmOperands - Add this value to the specified inlineasm node
    /// operand list.  This adds the code marker and includes the number of 
    /// values added into it.
    void AddInlineAsmOperands(unsigned Code, SelectionDAG &DAG,
                              std::vector<SDOperand> &Ops) const;
  };
}

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAG* createDefaultScheduler(SelectionDAGISel *IS,
                                      SelectionDAG *DAG,
                                      MachineBasicBlock *BB) {
    TargetLowering &TLI = IS->getTargetLowering();
    
    if (TLI.getSchedulingPreference() == TargetLowering::SchedulingForLatency) {
      return createTDListDAGScheduler(IS, DAG, BB);
    } else {
      assert(TLI.getSchedulingPreference() ==
           TargetLowering::SchedulingForRegPressure && "Unknown sched type!");
      return createBURRListDAGScheduler(IS, DAG, BB);
    }
  }


  //===--------------------------------------------------------------------===//
  /// FunctionLoweringInfo - This contains information that is global to a
  /// function that is used when lowering a region of the function.
  class FunctionLoweringInfo {
  public:
    TargetLowering &TLI;
    Function &Fn;
    MachineFunction &MF;
    MachineRegisterInfo &RegInfo;

    FunctionLoweringInfo(TargetLowering &TLI, Function &Fn,MachineFunction &MF);

    /// MBBMap - A mapping from LLVM basic blocks to their machine code entry.
    std::map<const BasicBlock*, MachineBasicBlock *> MBBMap;

    /// ValueMap - Since we emit code for the function a basic block at a time,
    /// we must remember which virtual registers hold the values for
    /// cross-basic-block values.
    DenseMap<const Value*, unsigned> ValueMap;

    /// StaticAllocaMap - Keep track of frame indices for fixed sized allocas in
    /// the entry block.  This allows the allocas to be efficiently referenced
    /// anywhere in the function.
    std::map<const AllocaInst*, int> StaticAllocaMap;

#ifndef NDEBUG
    SmallSet<Instruction*, 8> CatchInfoLost;
    SmallSet<Instruction*, 8> CatchInfoFound;
#endif

    unsigned MakeReg(MVT VT) {
      return RegInfo.createVirtualRegister(TLI.getRegClassFor(VT));
    }
    
    /// isExportedInst - Return true if the specified value is an instruction
    /// exported from its block.
    bool isExportedInst(const Value *V) {
      return ValueMap.count(V);
    }

    unsigned CreateRegForValue(const Value *V);
    
    unsigned InitializeRegForValue(const Value *V) {
      unsigned &R = ValueMap[V];
      assert(R == 0 && "Already initialized this value register!");
      return R = CreateRegForValue(V);
    }
  };
}

/// isSelector - Return true if this instruction is a call to the
/// eh.selector intrinsic.
static bool isSelector(Instruction *I) {
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    return (II->getIntrinsicID() == Intrinsic::eh_selector_i32 ||
            II->getIntrinsicID() == Intrinsic::eh_selector_i64);
  return false;
}

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it, or used by a 
/// switch or atomic instruction, which may expand to multiple basic blocks.
static bool isUsedOutsideOfDefiningBlock(Instruction *I) {
  if (isa<PHINode>(I)) return true;
  BasicBlock *BB = I->getParent();
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != BB || isa<PHINode>(*UI) ||
        // FIXME: Remove switchinst special case.
        isa<SwitchInst>(*UI))
      return true;
  return false;
}

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.  This includes arguments used by switches, since
/// the switch may expand into multiple basic blocks.
static bool isOnlyUsedInEntryBlock(Argument *A) {
  BasicBlock *Entry = A->getParent()->begin();
  for (Value::use_iterator UI = A->use_begin(), E = A->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != Entry || isa<SwitchInst>(*UI))
      return false;  // Use not in entry block.
  return true;
}

FunctionLoweringInfo::FunctionLoweringInfo(TargetLowering &tli,
                                           Function &fn, MachineFunction &mf)
    : TLI(tli), Fn(fn), MF(mf), RegInfo(MF.getRegInfo()) {

  // Create a vreg for each argument register that is not dead and is used
  // outside of the entry block for the function.
  for (Function::arg_iterator AI = Fn.arg_begin(), E = Fn.arg_end();
       AI != E; ++AI)
    if (!isOnlyUsedInEntryBlock(AI))
      InitializeRegForValue(AI);

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::iterator BB = Fn.begin(), EB = Fn.end();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (ConstantInt *CUI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        const Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getTargetData()->getABITypeSize(Ty);
        unsigned Align = 
          std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty),
                   AI->getAlignment());

        TySize *= CUI->getZExtValue();   // Get total allocated size.
        if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.
        StaticAllocaMap[AI] =
          MF.getFrameInfo()->CreateStackObject(TySize, Align);
      }

  for (; BB != EB; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (!I->use_empty() && isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (BB = Fn.begin(), EB = Fn.end(); BB != EB; ++BB) {
    MachineBasicBlock *MBB = new MachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF.getBasicBlockList().push_back(MBB);

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    PHINode *PN;
    for (BasicBlock::iterator I = BB->begin();(PN = dyn_cast<PHINode>(I)); ++I){
      if (PN->use_empty()) continue;
      
      MVT VT = TLI.getValueType(PN->getType());
      unsigned NumRegisters = TLI.getNumRegisters(VT);
      unsigned PHIReg = ValueMap[PN];
      assert(PHIReg && "PHI node does not have an assigned virtual register!");
      const TargetInstrInfo *TII = TLI.getTargetMachine().getInstrInfo();
      for (unsigned i = 0; i != NumRegisters; ++i)
        BuildMI(MBB, TII->get(TargetInstrInfo::PHI), PHIReg+i);
    }
  }
}

/// CreateRegForValue - Allocate the appropriate number of virtual registers of
/// the correctly promoted or expanded types.  Assign these registers
/// consecutive vreg numbers and return the first assigned number.
///
/// In the case that the given value has struct or array type, this function
/// will assign registers for each member or element.
///
unsigned FunctionLoweringInfo::CreateRegForValue(const Value *V) {
  SmallVector<MVT, 4> ValueVTs;
  ComputeValueVTs(TLI, V->getType(), ValueVTs);

  unsigned FirstReg = 0;
  for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
    MVT ValueVT = ValueVTs[Value];
    MVT RegisterVT = TLI.getRegisterType(ValueVT);

    unsigned NumRegs = TLI.getNumRegisters(ValueVT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      unsigned R = MakeReg(RegisterVT);
      if (!FirstReg) FirstReg = R;
    }
  }
  return FirstReg;
}

//===----------------------------------------------------------------------===//
/// SelectionDAGLowering - This is the common target-independent lowering
/// implementation that is parameterized by a TargetLowering object.
/// Also, targets can overload any lowering method.
///
namespace llvm {
class SelectionDAGLowering {
  MachineBasicBlock *CurMBB;

  DenseMap<const Value*, SDOperand> NodeMap;

  /// PendingLoads - Loads are not emitted to the program immediately.  We bunch
  /// them up and then emit token factor nodes when possible.  This allows us to
  /// get simple disambiguation between loads without worrying about alias
  /// analysis.
  std::vector<SDOperand> PendingLoads;

  /// PendingExports - CopyToReg nodes that copy values to virtual registers
  /// for export to other blocks need to be emitted before any terminator
  /// instruction, but they have no other ordering requirements. We bunch them
  /// up and the emit a single tokenfactor for them just before terminator
  /// instructions.
  std::vector<SDOperand> PendingExports;

  /// Case - A struct to record the Value for a switch case, and the
  /// case's target basic block.
  struct Case {
    Constant* Low;
    Constant* High;
    MachineBasicBlock* BB;

    Case() : Low(0), High(0), BB(0) { }
    Case(Constant* low, Constant* high, MachineBasicBlock* bb) :
      Low(low), High(high), BB(bb) { }
    uint64_t size() const {
      uint64_t rHigh = cast<ConstantInt>(High)->getSExtValue();
      uint64_t rLow  = cast<ConstantInt>(Low)->getSExtValue();
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
    bool operator () (const Case& C1, const Case& C2) {
      assert(isa<ConstantInt>(C1.Low) && isa<ConstantInt>(C2.High));
      const ConstantInt* CI1 = cast<const ConstantInt>(C1.Low);
      const ConstantInt* CI2 = cast<const ConstantInt>(C2.High);
      return CI1->getValue().slt(CI2->getValue());
    }
  };

  struct CaseBitsCmp {
    bool operator () (const CaseBits& C1, const CaseBits& C2) {
      return C1.Bits > C2.Bits;
    }
  };

  unsigned Clusterify(CaseVector& Cases, const SwitchInst &SI);
  
public:
  // TLI - This is information that describes the available target features we
  // need for lowering.  This indicates when operations are unavailable,
  // implemented with a libcall, etc.
  TargetLowering &TLI;
  SelectionDAG &DAG;
  const TargetData *TD;
  AliasAnalysis &AA;

  /// SwitchCases - Vector of CaseBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<SelectionDAGISel::CaseBlock> SwitchCases;
  /// JTCases - Vector of JumpTable structures used to communicate
  /// SwitchInst code generation information.
  std::vector<SelectionDAGISel::JumpTableBlock> JTCases;
  std::vector<SelectionDAGISel::BitTestBlock> BitTestCases;
  
  /// FuncInfo - Information about the function as a whole.
  ///
  FunctionLoweringInfo &FuncInfo;
  
  /// GCI - Garbage collection metadata for the function.
  CollectorMetadata *GCI;

  SelectionDAGLowering(SelectionDAG &dag, TargetLowering &tli,
                       AliasAnalysis &aa,
                       FunctionLoweringInfo &funcinfo,
                       CollectorMetadata *gci)
    : TLI(tli), DAG(dag), TD(DAG.getTarget().getTargetData()), AA(aa),
      FuncInfo(funcinfo), GCI(gci) {
  }

  /// getRoot - Return the current virtual root of the Selection DAG,
  /// flushing any PendingLoad items. This must be done before emitting
  /// a store or any other node that may need to be ordered after any
  /// prior load instructions.
  ///
  SDOperand getRoot() {
    if (PendingLoads.empty())
      return DAG.getRoot();

    if (PendingLoads.size() == 1) {
      SDOperand Root = PendingLoads[0];
      DAG.setRoot(Root);
      PendingLoads.clear();
      return Root;
    }

    // Otherwise, we have to make a token factor node.
    SDOperand Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                 &PendingLoads[0], PendingLoads.size());
    PendingLoads.clear();
    DAG.setRoot(Root);
    return Root;
  }

  /// getControlRoot - Similar to getRoot, but instead of flushing all the
  /// PendingLoad items, flush all the PendingExports items. It is necessary
  /// to do this before emitting a terminator instruction.
  ///
  SDOperand getControlRoot() {
    SDOperand Root = DAG.getRoot();

    if (PendingExports.empty())
      return Root;

    // Turn all of the CopyToReg chains into one factored node.
    if (Root.getOpcode() != ISD::EntryToken) {
      unsigned i = 0, e = PendingExports.size();
      for (; i != e; ++i) {
        assert(PendingExports[i].Val->getNumOperands() > 1);
        if (PendingExports[i].Val->getOperand(0) == Root)
          break;  // Don't add the root if we already indirectly depend on it.
      }
        
      if (i == e)
        PendingExports.push_back(Root);
    }

    Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                       &PendingExports[0],
                       PendingExports.size());
    PendingExports.clear();
    DAG.setRoot(Root);
    return Root;
  }

  void CopyValueToVirtualRegister(Value *V, unsigned Reg);

  void visit(Instruction &I) { visit(I.getOpcode(), I); }

  void visit(unsigned Opcode, User &I) {
    // Note: this doesn't use InstVisitor, because it has to work with
    // ConstantExpr's in addition to instructions.
    switch (Opcode) {
    default: assert(0 && "Unknown instruction type encountered!");
             abort();
      // Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE:return visit##OPCODE((CLASS&)I);
#include "llvm/Instruction.def"
    }
  }

  void setCurrentBasicBlock(MachineBasicBlock *MBB) { CurMBB = MBB; }

  SDOperand getLoadFrom(const Type *Ty, SDOperand Ptr,
                        const Value *SV, SDOperand Root,
                        bool isVolatile, unsigned Alignment);

  SDOperand getValue(const Value *V);

  void setValue(const Value *V, SDOperand NewN) {
    SDOperand &N = NodeMap[V];
    assert(N.Val == 0 && "Already set a value for this node!");
    N = NewN;
  }
  
  void GetRegistersForValue(SDISelAsmOperandInfo &OpInfo, bool HasEarlyClobber,
                            std::set<unsigned> &OutputRegs, 
                            std::set<unsigned> &InputRegs);

  void FindMergedConditions(Value *Cond, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *CurBB,
                            unsigned Opc);
  bool isExportableFromCurrentBlock(Value *V, const BasicBlock *FromBB);
  void ExportFromCurrentBlock(Value *V);
  void LowerCallTo(CallSite CS, SDOperand Callee, bool IsTailCall,
                   MachineBasicBlock *LandingPad = NULL);

  // Terminator instructions.
  void visitRet(ReturnInst &I);
  void visitBr(BranchInst &I);
  void visitSwitch(SwitchInst &I);
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
  void visitSwitchCase(SelectionDAGISel::CaseBlock &CB);
  void visitBitTestHeader(SelectionDAGISel::BitTestBlock &B);
  void visitBitTestCase(MachineBasicBlock* NextMBB,
                        unsigned Reg,
                        SelectionDAGISel::BitTestCase &B);
  void visitJumpTable(SelectionDAGISel::JumpTable &JT);
  void visitJumpTableHeader(SelectionDAGISel::JumpTable &JT,
                            SelectionDAGISel::JumpTableHeader &JTH);
  
  // These all get lowered before this pass.
  void visitInvoke(InvokeInst &I);
  void visitUnwind(UnwindInst &I);

  void visitBinary(User &I, unsigned OpCode);
  void visitShift(User &I, unsigned Opcode);
  void visitAdd(User &I) { 
    if (I.getType()->isFPOrFPVector())
      visitBinary(I, ISD::FADD);
    else
      visitBinary(I, ISD::ADD);
  }
  void visitSub(User &I);
  void visitMul(User &I) {
    if (I.getType()->isFPOrFPVector())
      visitBinary(I, ISD::FMUL);
    else
      visitBinary(I, ISD::MUL);
  }
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
  void visitVICmp(User &I);
  void visitVFCmp(User &I);
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

  void visitMalloc(MallocInst &I);
  void visitFree(FreeInst &I);
  void visitAlloca(AllocaInst &I);
  void visitLoad(LoadInst &I);
  void visitStore(StoreInst &I);
  void visitPHI(PHINode &I) { } // PHI nodes are handled specially.
  void visitCall(CallInst &I);
  void visitInlineAsm(CallSite CS);
  const char *visitIntrinsicCall(CallInst &I, unsigned Intrinsic);
  void visitTargetIntrinsic(CallInst &I, unsigned Intrinsic);

  void visitVAStart(CallInst &I);
  void visitVAArg(VAArgInst &I);
  void visitVAEnd(CallInst &I);
  void visitVACopy(CallInst &I);

  void visitGetResult(GetResultInst &I);

  void visitUserOp1(Instruction &I) {
    assert(0 && "UserOp1 should not exist at instruction selection time!");
    abort();
  }
  void visitUserOp2(Instruction &I) {
    assert(0 && "UserOp2 should not exist at instruction selection time!");
    abort();
  }
  
private:
  inline const char *implVisitBinaryAtomic(CallInst& I, ISD::NodeType Op);

};
} // end namespace llvm


/// getCopyFromParts - Create a value that contains the specified legal parts
/// combined into the value they represent.  If the parts combine to a type
/// larger then ValueVT then AssertOp can be used to specify whether the extra
/// bits are known to be zero (ISD::AssertZext) or sign extended from ValueVT
/// (ISD::AssertSext).
static SDOperand getCopyFromParts(SelectionDAG &DAG,
                                  const SDOperand *Parts,
                                  unsigned NumParts,
                                  MVT PartVT,
                                  MVT ValueVT,
                                  ISD::NodeType AssertOp = ISD::DELETED_NODE) {
  assert(NumParts > 0 && "No parts to assemble!");
  TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDOperand Val = Parts[0];

  if (NumParts > 1) {
    // Assemble the value from multiple parts.
    if (!ValueVT.isVector()) {
      unsigned PartBits = PartVT.getSizeInBits();
      unsigned ValueBits = ValueVT.getSizeInBits();

      // Assemble the power of 2 part.
      unsigned RoundParts = NumParts & (NumParts - 1) ?
        1 << Log2_32(NumParts) : NumParts;
      unsigned RoundBits = PartBits * RoundParts;
      MVT RoundVT = RoundBits == ValueBits ?
        ValueVT : MVT::getIntegerVT(RoundBits);
      SDOperand Lo, Hi;

      if (RoundParts > 2) {
        MVT HalfVT = MVT::getIntegerVT(RoundBits/2);
        Lo = getCopyFromParts(DAG, Parts, RoundParts/2, PartVT, HalfVT);
        Hi = getCopyFromParts(DAG, Parts+RoundParts/2, RoundParts/2,
                              PartVT, HalfVT);
      } else {
        Lo = Parts[0];
        Hi = Parts[1];
      }
      if (TLI.isBigEndian())
        std::swap(Lo, Hi);
      Val = DAG.getNode(ISD::BUILD_PAIR, RoundVT, Lo, Hi);

      if (RoundParts < NumParts) {
        // Assemble the trailing non-power-of-2 part.
        unsigned OddParts = NumParts - RoundParts;
        MVT OddVT = MVT::getIntegerVT(OddParts * PartBits);
        Hi = getCopyFromParts(DAG, Parts+RoundParts, OddParts, PartVT, OddVT);

        // Combine the round and odd parts.
        Lo = Val;
        if (TLI.isBigEndian())
          std::swap(Lo, Hi);
        MVT TotalVT = MVT::getIntegerVT(NumParts * PartBits);
        Hi = DAG.getNode(ISD::ANY_EXTEND, TotalVT, Hi);
        Hi = DAG.getNode(ISD::SHL, TotalVT, Hi,
                         DAG.getConstant(Lo.getValueType().getSizeInBits(),
                                         TLI.getShiftAmountTy()));
        Lo = DAG.getNode(ISD::ZERO_EXTEND, TotalVT, Lo);
        Val = DAG.getNode(ISD::OR, TotalVT, Lo, Hi);
      }
    } else {
      // Handle a multi-element vector.
      MVT IntermediateVT, RegisterVT;
      unsigned NumIntermediates;
      unsigned NumRegs =
        TLI.getVectorTypeBreakdown(ValueVT, IntermediateVT, NumIntermediates,
                                   RegisterVT);
      assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
      NumParts = NumRegs; // Silence a compiler warning.
      assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");
      assert(RegisterVT == Parts[0].getValueType() &&
             "Part type doesn't match part!");

      // Assemble the parts into intermediate operands.
      SmallVector<SDOperand, 8> Ops(NumIntermediates);
      if (NumIntermediates == NumParts) {
        // If the register was not expanded, truncate or copy the value,
        // as appropriate.
        for (unsigned i = 0; i != NumParts; ++i)
          Ops[i] = getCopyFromParts(DAG, &Parts[i], 1,
                                    PartVT, IntermediateVT);
      } else if (NumParts > 0) {
        // If the intermediate type was expanded, build the intermediate operands
        // from the parts.
        assert(NumParts % NumIntermediates == 0 &&
               "Must expand into a divisible number of parts!");
        unsigned Factor = NumParts / NumIntermediates;
        for (unsigned i = 0; i != NumIntermediates; ++i)
          Ops[i] = getCopyFromParts(DAG, &Parts[i * Factor], Factor,
                                    PartVT, IntermediateVT);
      }

      // Build a vector with BUILD_VECTOR or CONCAT_VECTORS from the intermediate
      // operands.
      Val = DAG.getNode(IntermediateVT.isVector() ?
                        ISD::CONCAT_VECTORS : ISD::BUILD_VECTOR,
                        ValueVT, &Ops[0], NumIntermediates);
    }
  }

  // There is now one part, held in Val.  Correct it to match ValueVT.
  PartVT = Val.getValueType();

  if (PartVT == ValueVT)
    return Val;

  if (PartVT.isVector()) {
    assert(ValueVT.isVector() && "Unknown vector conversion!");
    return DAG.getNode(ISD::BIT_CONVERT, ValueVT, Val);
  }

  if (ValueVT.isVector()) {
    assert(ValueVT.getVectorElementType() == PartVT &&
           ValueVT.getVectorNumElements() == 1 &&
           "Only trivial scalar-to-vector conversions should get here!");
    return DAG.getNode(ISD::BUILD_VECTOR, ValueVT, Val);
  }

  if (PartVT.isInteger() &&
      ValueVT.isInteger()) {
    if (ValueVT.bitsLT(PartVT)) {
      // For a truncate, see if we have any information to
      // indicate whether the truncated bits will always be
      // zero or sign-extension.
      if (AssertOp != ISD::DELETED_NODE)
        Val = DAG.getNode(AssertOp, PartVT, Val,
                          DAG.getValueType(ValueVT));
      return DAG.getNode(ISD::TRUNCATE, ValueVT, Val);
    } else {
      return DAG.getNode(ISD::ANY_EXTEND, ValueVT, Val);
    }
  }

  if (PartVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
    if (ValueVT.bitsLT(Val.getValueType()))
      // FP_ROUND's are always exact here.
      return DAG.getNode(ISD::FP_ROUND, ValueVT, Val,
                         DAG.getIntPtrConstant(1));
    return DAG.getNode(ISD::FP_EXTEND, ValueVT, Val);
  }

  if (PartVT.getSizeInBits() == ValueVT.getSizeInBits())
    return DAG.getNode(ISD::BIT_CONVERT, ValueVT, Val);

  assert(0 && "Unknown mismatch!");
  return SDOperand();
}

/// getCopyToParts - Create a series of nodes that contain the specified value
/// split into legal parts.  If the parts contain more bits than Val, then, for
/// integers, ExtendKind can be used to specify how to generate the extra bits.
static void getCopyToParts(SelectionDAG &DAG,
                           SDOperand Val,
                           SDOperand *Parts,
                           unsigned NumParts,
                           MVT PartVT,
                           ISD::NodeType ExtendKind = ISD::ANY_EXTEND) {
  TargetLowering &TLI = DAG.getTargetLoweringInfo();
  MVT PtrVT = TLI.getPointerTy();
  MVT ValueVT = Val.getValueType();
  unsigned PartBits = PartVT.getSizeInBits();
  assert(TLI.isTypeLegal(PartVT) && "Copying to an illegal type!");

  if (!NumParts)
    return;

  if (!ValueVT.isVector()) {
    if (PartVT == ValueVT) {
      assert(NumParts == 1 && "No-op copy with multiple parts!");
      Parts[0] = Val;
      return;
    }

    if (NumParts * PartBits > ValueVT.getSizeInBits()) {
      // If the parts cover more bits than the value has, promote the value.
      if (PartVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
        assert(NumParts == 1 && "Do not know what to promote to!");
        Val = DAG.getNode(ISD::FP_EXTEND, PartVT, Val);
      } else if (PartVT.isInteger() && ValueVT.isInteger()) {
        ValueVT = MVT::getIntegerVT(NumParts * PartBits);
        Val = DAG.getNode(ExtendKind, ValueVT, Val);
      } else {
        assert(0 && "Unknown mismatch!");
      }
    } else if (PartBits == ValueVT.getSizeInBits()) {
      // Different types of the same size.
      assert(NumParts == 1 && PartVT != ValueVT);
      Val = DAG.getNode(ISD::BIT_CONVERT, PartVT, Val);
    } else if (NumParts * PartBits < ValueVT.getSizeInBits()) {
      // If the parts cover less bits than value has, truncate the value.
      if (PartVT.isInteger() && ValueVT.isInteger()) {
        ValueVT = MVT::getIntegerVT(NumParts * PartBits);
        Val = DAG.getNode(ISD::TRUNCATE, ValueVT, Val);
      } else {
        assert(0 && "Unknown mismatch!");
      }
    }

    // The value may have changed - recompute ValueVT.
    ValueVT = Val.getValueType();
    assert(NumParts * PartBits == ValueVT.getSizeInBits() &&
           "Failed to tile the value with PartVT!");

    if (NumParts == 1) {
      assert(PartVT == ValueVT && "Type conversion failed!");
      Parts[0] = Val;
      return;
    }

    // Expand the value into multiple parts.
    if (NumParts & (NumParts - 1)) {
      // The number of parts is not a power of 2.  Split off and copy the tail.
      assert(PartVT.isInteger() && ValueVT.isInteger() &&
             "Do not know what to expand to!");
      unsigned RoundParts = 1 << Log2_32(NumParts);
      unsigned RoundBits = RoundParts * PartBits;
      unsigned OddParts = NumParts - RoundParts;
      SDOperand OddVal = DAG.getNode(ISD::SRL, ValueVT, Val,
                                     DAG.getConstant(RoundBits,
                                                     TLI.getShiftAmountTy()));
      getCopyToParts(DAG, OddVal, Parts + RoundParts, OddParts, PartVT);
      if (TLI.isBigEndian())
        // The odd parts were reversed by getCopyToParts - unreverse them.
        std::reverse(Parts + RoundParts, Parts + NumParts);
      NumParts = RoundParts;
      ValueVT = MVT::getIntegerVT(NumParts * PartBits);
      Val = DAG.getNode(ISD::TRUNCATE, ValueVT, Val);
    }

    // The number of parts is a power of 2.  Repeatedly bisect the value using
    // EXTRACT_ELEMENT.
    Parts[0] = DAG.getNode(ISD::BIT_CONVERT,
                           MVT::getIntegerVT(ValueVT.getSizeInBits()),
                           Val);
    for (unsigned StepSize = NumParts; StepSize > 1; StepSize /= 2) {
      for (unsigned i = 0; i < NumParts; i += StepSize) {
        unsigned ThisBits = StepSize * PartBits / 2;
        MVT ThisVT = MVT::getIntegerVT (ThisBits);
        SDOperand &Part0 = Parts[i];
        SDOperand &Part1 = Parts[i+StepSize/2];

        Part1 = DAG.getNode(ISD::EXTRACT_ELEMENT, ThisVT, Part0,
                            DAG.getConstant(1, PtrVT));
        Part0 = DAG.getNode(ISD::EXTRACT_ELEMENT, ThisVT, Part0,
                            DAG.getConstant(0, PtrVT));

        if (ThisBits == PartBits && ThisVT != PartVT) {
          Part0 = DAG.getNode(ISD::BIT_CONVERT, PartVT, Part0);
          Part1 = DAG.getNode(ISD::BIT_CONVERT, PartVT, Part1);
        }
      }
    }

    if (TLI.isBigEndian())
      std::reverse(Parts, Parts + NumParts);

    return;
  }

  // Vector ValueVT.
  if (NumParts == 1) {
    if (PartVT != ValueVT) {
      if (PartVT.isVector()) {
        Val = DAG.getNode(ISD::BIT_CONVERT, PartVT, Val);
      } else {
        assert(ValueVT.getVectorElementType() == PartVT &&
               ValueVT.getVectorNumElements() == 1 &&
               "Only trivial vector-to-scalar conversions should get here!");
        Val = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, PartVT, Val,
                          DAG.getConstant(0, PtrVT));
      }
    }

    Parts[0] = Val;
    return;
  }

  // Handle a multi-element vector.
  MVT IntermediateVT, RegisterVT;
  unsigned NumIntermediates;
  unsigned NumRegs =
    DAG.getTargetLoweringInfo()
      .getVectorTypeBreakdown(ValueVT, IntermediateVT, NumIntermediates,
                              RegisterVT);
  unsigned NumElements = ValueVT.getVectorNumElements();

  assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
  NumParts = NumRegs; // Silence a compiler warning.
  assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");

  // Split the vector into intermediate operands.
  SmallVector<SDOperand, 8> Ops(NumIntermediates);
  for (unsigned i = 0; i != NumIntermediates; ++i)
    if (IntermediateVT.isVector())
      Ops[i] = DAG.getNode(ISD::EXTRACT_SUBVECTOR,
                           IntermediateVT, Val,
                           DAG.getConstant(i * (NumElements / NumIntermediates),
                                           PtrVT));
    else
      Ops[i] = DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
                           IntermediateVT, Val, 
                           DAG.getConstant(i, PtrVT));

  // Split the intermediate operands into legal parts.
  if (NumParts == NumIntermediates) {
    // If the register was not expanded, promote or copy the value,
    // as appropriate.
    for (unsigned i = 0; i != NumParts; ++i)
      getCopyToParts(DAG, Ops[i], &Parts[i], 1, PartVT);
  } else if (NumParts > 0) {
    // If the intermediate type was expanded, split each the value into
    // legal parts.
    assert(NumParts % NumIntermediates == 0 &&
           "Must expand into a divisible number of parts!");
    unsigned Factor = NumParts / NumIntermediates;
    for (unsigned i = 0; i != NumIntermediates; ++i)
      getCopyToParts(DAG, Ops[i], &Parts[i * Factor], Factor, PartVT);
  }
}


SDOperand SelectionDAGLowering::getValue(const Value *V) {
  SDOperand &N = NodeMap[V];
  if (N.Val) return N;
  
  if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V))) {
    MVT VT = TLI.getValueType(V->getType(), true);
    
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return N = DAG.getConstant(CI->getValue(), VT);

    if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return N = DAG.getGlobalAddress(GV, VT);
    
    if (isa<ConstantPointerNull>(C))
      return N = DAG.getConstant(0, TLI.getPointerTy());
    
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(C))
      return N = DAG.getConstantFP(CFP->getValueAPF(), VT);
    
    if (isa<UndefValue>(C) && !isa<VectorType>(V->getType()) &&
        !V->getType()->isAggregateType())
      return N = DAG.getNode(ISD::UNDEF, VT);

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      visit(CE->getOpcode(), *CE);
      SDOperand N1 = NodeMap[V];
      assert(N1.Val && "visit didn't populate the ValueMap!");
      return N1;
    }
    
    if (isa<ConstantStruct>(C) || isa<ConstantArray>(C)) {
      SmallVector<SDOperand, 4> Constants;
      SmallVector<MVT, 4> ValueVTs;
      for (User::const_op_iterator OI = C->op_begin(), OE = C->op_end();
           OI != OE; ++OI) {
        SDNode *Val = getValue(*OI).Val;
        for (unsigned i = 0, e = Val->getNumValues(); i != e; ++i) {
          Constants.push_back(SDOperand(Val, i));
          ValueVTs.push_back(Val->getValueType(i));
        }
      }
      return DAG.getNode(ISD::MERGE_VALUES,
                         DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                         &Constants[0], Constants.size());
    }

    if (const ArrayType *ATy = dyn_cast<ArrayType>(C->getType())) {
      assert((isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) &&
             "Unknown array constant!");
      unsigned NumElts = ATy->getNumElements();
      MVT EltVT = TLI.getValueType(ATy->getElementType());
      SmallVector<SDOperand, 4> Constants(NumElts);
      SmallVector<MVT, 4> ValueVTs(NumElts, EltVT);
      for (unsigned i = 0, e = NumElts; i != e; ++i) {
        if (isa<UndefValue>(C))
          Constants[i] = DAG.getNode(ISD::UNDEF, EltVT);
        else if (EltVT.isFloatingPoint())
          Constants[i] = DAG.getConstantFP(0, EltVT);
        else
          Constants[i] = DAG.getConstant(0, EltVT);
      }
      return DAG.getNode(ISD::MERGE_VALUES,
                         DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                         &Constants[0], Constants.size());
    }

    if (const StructType *STy = dyn_cast<StructType>(C->getType())) {
      assert((isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) &&
             "Unknown struct constant!");
      unsigned NumElts = STy->getNumElements();
      SmallVector<SDOperand, 4> Constants(NumElts);
      SmallVector<MVT, 4> ValueVTs(NumElts);
      for (unsigned i = 0, e = NumElts; i != e; ++i) {
        MVT EltVT = TLI.getValueType(STy->getElementType(i));
        ValueVTs[i] = EltVT;
        if (isa<UndefValue>(C))
          Constants[i] = DAG.getNode(ISD::UNDEF, EltVT);
        else if (EltVT.isFloatingPoint())
          Constants[i] = DAG.getConstantFP(0, EltVT);
        else
          Constants[i] = DAG.getConstant(0, EltVT);
      }
      return DAG.getNode(ISD::MERGE_VALUES,
                         DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                         &Constants[0], Constants.size());
    }

    const VectorType *VecTy = cast<VectorType>(V->getType());
    unsigned NumElements = VecTy->getNumElements();
    
    // Now that we know the number and type of the elements, get that number of
    // elements into the Ops array based on what kind of constant it is.
    SmallVector<SDOperand, 16> Ops;
    if (ConstantVector *CP = dyn_cast<ConstantVector>(C)) {
      for (unsigned i = 0; i != NumElements; ++i)
        Ops.push_back(getValue(CP->getOperand(i)));
    } else {
      assert((isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) &&
             "Unknown vector constant!");
      MVT EltVT = TLI.getValueType(VecTy->getElementType());

      SDOperand Op;
      if (isa<UndefValue>(C))
        Op = DAG.getNode(ISD::UNDEF, EltVT);
      else if (EltVT.isFloatingPoint())
        Op = DAG.getConstantFP(0, EltVT);
      else
        Op = DAG.getConstant(0, EltVT);
      Ops.assign(NumElements, Op);
    }
    
    // Create a BUILD_VECTOR node.
    return NodeMap[V] = DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
      
  // If this is a static alloca, generate it as the frameindex instead of
  // computation.
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    std::map<const AllocaInst*, int>::iterator SI =
      FuncInfo.StaticAllocaMap.find(AI);
    if (SI != FuncInfo.StaticAllocaMap.end())
      return DAG.getFrameIndex(SI->second, TLI.getPointerTy());
  }
      
  unsigned InReg = FuncInfo.ValueMap[V];
  assert(InReg && "Value not in map!");
  
  RegsForValue RFV(TLI, InReg, V->getType());
  SDOperand Chain = DAG.getEntryNode();
  return RFV.getCopyFromRegs(DAG, Chain, NULL);
}


void SelectionDAGLowering::visitRet(ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other, getControlRoot()));
    return;
  }
  
  SmallVector<SDOperand, 8> NewValues;
  NewValues.push_back(getControlRoot());
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {  
    SDOperand RetOp = getValue(I.getOperand(i));
    MVT VT = RetOp.getValueType();

    // FIXME: C calling convention requires the return type to be promoted to
    // at least 32-bit. But this is not necessary for non-C calling conventions.
    if (VT.isInteger()) {
      MVT MinVT = TLI.getRegisterType(MVT::i32);
      if (VT.bitsLT(MinVT))
        VT = MinVT;
    }

    unsigned NumParts = TLI.getNumRegisters(VT);
    MVT PartVT = TLI.getRegisterType(VT);
    SmallVector<SDOperand, 4> Parts(NumParts);
    ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

    const Function *F = I.getParent()->getParent();
    if (F->paramHasAttr(0, ParamAttr::SExt))
      ExtendKind = ISD::SIGN_EXTEND;
    else if (F->paramHasAttr(0, ParamAttr::ZExt))
      ExtendKind = ISD::ZERO_EXTEND;

    getCopyToParts(DAG, RetOp, &Parts[0], NumParts, PartVT, ExtendKind);

    for (unsigned i = 0; i < NumParts; ++i) {
      NewValues.push_back(Parts[i]);
      NewValues.push_back(DAG.getArgFlags(ISD::ArgFlagsTy()));
    }
  }
  DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other,
                          &NewValues[0], NewValues.size()));
}

/// ExportFromCurrentBlock - If this condition isn't known to be exported from
/// the current basic block, add it to ValueMap now so that we'll get a
/// CopyTo/FromReg.
void SelectionDAGLowering::ExportFromCurrentBlock(Value *V) {
  // No need to export constants.
  if (!isa<Instruction>(V) && !isa<Argument>(V)) return;
  
  // Already exported?
  if (FuncInfo.isExportedInst(V)) return;

  unsigned Reg = FuncInfo.InitializeRegForValue(V);
  CopyValueToVirtualRegister(V, Reg);
}

bool SelectionDAGLowering::isExportableFromCurrentBlock(Value *V,
                                                    const BasicBlock *FromBB) {
  // The operands of the setcc have to be in this block.  We don't know
  // how to export them from some other block.
  if (Instruction *VI = dyn_cast<Instruction>(V)) {
    // Can export from current BB.
    if (VI->getParent() == FromBB)
      return true;
    
    // Is already exported, noop.
    return FuncInfo.isExportedInst(V);
  }
  
  // If this is an argument, we can export it if the BB is the entry block or
  // if it is already exported.
  if (isa<Argument>(V)) {
    if (FromBB == &FromBB->getParent()->getEntryBlock())
      return true;

    // Otherwise, can only export this if it is already exported.
    return FuncInfo.isExportedInst(V);
  }
  
  // Otherwise, constants can always be exported.
  return true;
}

static bool InBlock(const Value *V, const BasicBlock *BB) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() == BB;
  return true;
}

/// FindMergedConditions - If Cond is an expression like 
void SelectionDAGLowering::FindMergedConditions(Value *Cond,
                                                MachineBasicBlock *TBB,
                                                MachineBasicBlock *FBB,
                                                MachineBasicBlock *CurBB,
                                                unsigned Opc) {
  // If this node is not part of the or/and tree, emit it as a branch.
  Instruction *BOp = dyn_cast<Instruction>(Cond);

  if (!BOp || !(isa<BinaryOperator>(BOp) || isa<CmpInst>(BOp)) || 
      (unsigned)BOp->getOpcode() != Opc || !BOp->hasOneUse() ||
      BOp->getParent() != CurBB->getBasicBlock() ||
      !InBlock(BOp->getOperand(0), CurBB->getBasicBlock()) ||
      !InBlock(BOp->getOperand(1), CurBB->getBasicBlock())) {
    const BasicBlock *BB = CurBB->getBasicBlock();
    
    // If the leaf of the tree is a comparison, merge the condition into 
    // the caseblock.
    if ((isa<ICmpInst>(Cond) || isa<FCmpInst>(Cond)) &&
        // The operands of the cmp have to be in this block.  We don't know
        // how to export them from some other block.  If this is the first block
        // of the sequence, no exporting is needed.
        (CurBB == CurMBB ||
         (isExportableFromCurrentBlock(BOp->getOperand(0), BB) &&
          isExportableFromCurrentBlock(BOp->getOperand(1), BB)))) {
      BOp = cast<Instruction>(Cond);
      ISD::CondCode Condition;
      if (ICmpInst *IC = dyn_cast<ICmpInst>(Cond)) {
        switch (IC->getPredicate()) {
        default: assert(0 && "Unknown icmp predicate opcode!");
        case ICmpInst::ICMP_EQ:  Condition = ISD::SETEQ;  break;
        case ICmpInst::ICMP_NE:  Condition = ISD::SETNE;  break;
        case ICmpInst::ICMP_SLE: Condition = ISD::SETLE;  break;
        case ICmpInst::ICMP_ULE: Condition = ISD::SETULE; break;
        case ICmpInst::ICMP_SGE: Condition = ISD::SETGE;  break;
        case ICmpInst::ICMP_UGE: Condition = ISD::SETUGE; break;
        case ICmpInst::ICMP_SLT: Condition = ISD::SETLT;  break;
        case ICmpInst::ICMP_ULT: Condition = ISD::SETULT; break;
        case ICmpInst::ICMP_SGT: Condition = ISD::SETGT;  break;
        case ICmpInst::ICMP_UGT: Condition = ISD::SETUGT; break;
        }
      } else if (FCmpInst *FC = dyn_cast<FCmpInst>(Cond)) {
        ISD::CondCode FPC, FOC;
        switch (FC->getPredicate()) {
        default: assert(0 && "Unknown fcmp predicate opcode!");
        case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
        case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
        case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
        case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
        case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
        case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
        case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
        case FCmpInst::FCMP_ORD:   FOC = FPC = ISD::SETO;   break;
        case FCmpInst::FCMP_UNO:   FOC = FPC = ISD::SETUO;  break;
        case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
        case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
        case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
        case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
        case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
        case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
        case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
        }
        if (FiniteOnlyFPMath())
          Condition = FOC;
        else 
          Condition = FPC;
      } else {
        Condition = ISD::SETEQ; // silence warning.
        assert(0 && "Unknown compare instruction");
      }
      
      SelectionDAGISel::CaseBlock CB(Condition, BOp->getOperand(0), 
                                     BOp->getOperand(1), NULL, TBB, FBB, CurBB);
      SwitchCases.push_back(CB);
      return;
    }
    
    // Create a CaseBlock record representing this branch.
    SelectionDAGISel::CaseBlock CB(ISD::SETEQ, Cond, ConstantInt::getTrue(),
                                   NULL, TBB, FBB, CurBB);
    SwitchCases.push_back(CB);
    return;
  }
  
  
  //  Create TmpBB after CurBB.
  MachineFunction::iterator BBI = CurBB;
  MachineBasicBlock *TmpBB = new MachineBasicBlock(CurBB->getBasicBlock());
  CurBB->getParent()->getBasicBlockList().insert(++BBI, TmpBB);
  
  if (Opc == Instruction::Or) {
    // Codegen X | Y as:
    //   jmp_if_X TBB
    //   jmp TmpBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
  
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TBB, TmpBB, CurBB, Opc);
  
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, Opc);
  } else {
    assert(Opc == Instruction::And && "Unknown merge op!");
    // Codegen X & Y as:
    //   jmp_if_X TmpBB
    //   jmp FBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
    //  This requires creation of TmpBB after CurBB.
    
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TmpBB, FBB, CurBB, Opc);
    
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, Opc);
  }
}

/// If the set of cases should be emitted as a series of branches, return true.
/// If we should emit this as a bunch of and/or'd together conditions, return
/// false.
static bool 
ShouldEmitAsBranches(const std::vector<SelectionDAGISel::CaseBlock> &Cases) {
  if (Cases.size() != 2) return true;
  
  // If this is two comparisons of the same values or'd or and'd together, they
  // will get folded into a single comparison, so don't emit two blocks.
  if ((Cases[0].CmpLHS == Cases[1].CmpLHS &&
       Cases[0].CmpRHS == Cases[1].CmpRHS) ||
      (Cases[0].CmpRHS == Cases[1].CmpLHS &&
       Cases[0].CmpLHS == Cases[1].CmpRHS)) {
    return false;
  }
  
  return true;
}

void SelectionDAGLowering::visitBr(BranchInst &I) {
  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  if (I.isUnconditional()) {
    // Update machine-CFG edges.
    CurMBB->addSuccessor(Succ0MBB);
    
    // If this is not a fall-through branch, emit the branch.
    if (Succ0MBB != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getControlRoot(),
                              DAG.getBasicBlock(Succ0MBB)));
    return;
  }

  // If this condition is one of the special cases we handle, do special stuff
  // now.
  Value *CondVal = I.getCondition();
  MachineBasicBlock *Succ1MBB = FuncInfo.MBBMap[I.getSuccessor(1)];

  // If this is a series of conditions that are or'd or and'd together, emit
  // this as a sequence of branches instead of setcc's with and/or operations.
  // For example, instead of something like:
  //     cmp A, B
  //     C = seteq 
  //     cmp D, E
  //     F = setle 
  //     or C, F
  //     jnz foo
  // Emit:
  //     cmp A, B
  //     je foo
  //     cmp D, E
  //     jle foo
  //
  if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(CondVal)) {
    if (BOp->hasOneUse() && 
        (BOp->getOpcode() == Instruction::And ||
         BOp->getOpcode() == Instruction::Or)) {
      FindMergedConditions(BOp, Succ0MBB, Succ1MBB, CurMBB, BOp->getOpcode());
      // If the compares in later blocks need to use values not currently
      // exported from this block, export them now.  This block should always
      // be the first entry.
      assert(SwitchCases[0].ThisBB == CurMBB && "Unexpected lowering!");
      
      // Allow some cases to be rejected.
      if (ShouldEmitAsBranches(SwitchCases)) {
        for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i) {
          ExportFromCurrentBlock(SwitchCases[i].CmpLHS);
          ExportFromCurrentBlock(SwitchCases[i].CmpRHS);
        }
        
        // Emit the branch for this block.
        visitSwitchCase(SwitchCases[0]);
        SwitchCases.erase(SwitchCases.begin());
        return;
      }
      
      // Okay, we decided not to do this, remove any inserted MBB's and clear
      // SwitchCases.
      for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i)
        CurMBB->getParent()->getBasicBlockList().erase(SwitchCases[i].ThisBB);
      
      SwitchCases.clear();
    }
  }
  
  // Create a CaseBlock record representing this branch.
  SelectionDAGISel::CaseBlock CB(ISD::SETEQ, CondVal, ConstantInt::getTrue(),
                                 NULL, Succ0MBB, Succ1MBB, CurMBB);
  // Use visitSwitchCase to actually insert the fast branch sequence for this
  // cond branch.
  visitSwitchCase(CB);
}

/// visitSwitchCase - Emits the necessary code to represent a single node in
/// the binary search tree resulting from lowering a switch instruction.
void SelectionDAGLowering::visitSwitchCase(SelectionDAGISel::CaseBlock &CB) {
  SDOperand Cond;
  SDOperand CondLHS = getValue(CB.CmpLHS);
  
  // Build the setcc now. 
  if (CB.CmpMHS == NULL) {
    // Fold "(X == true)" to X and "(X == false)" to !X to
    // handle common cases produced by branch lowering.
    if (CB.CmpRHS == ConstantInt::getTrue() && CB.CC == ISD::SETEQ)
      Cond = CondLHS;
    else if (CB.CmpRHS == ConstantInt::getFalse() && CB.CC == ISD::SETEQ) {
      SDOperand True = DAG.getConstant(1, CondLHS.getValueType());
      Cond = DAG.getNode(ISD::XOR, CondLHS.getValueType(), CondLHS, True);
    } else
      Cond = DAG.getSetCC(MVT::i1, CondLHS, getValue(CB.CmpRHS), CB.CC);
  } else {
    assert(CB.CC == ISD::SETLE && "Can handle only LE ranges now");

    uint64_t Low = cast<ConstantInt>(CB.CmpLHS)->getSExtValue();
    uint64_t High  = cast<ConstantInt>(CB.CmpRHS)->getSExtValue();

    SDOperand CmpOp = getValue(CB.CmpMHS);
    MVT VT = CmpOp.getValueType();

    if (cast<ConstantInt>(CB.CmpLHS)->isMinValue(true)) {
      Cond = DAG.getSetCC(MVT::i1, CmpOp, DAG.getConstant(High, VT), ISD::SETLE);
    } else {
      SDOperand SUB = DAG.getNode(ISD::SUB, VT, CmpOp, DAG.getConstant(Low, VT));
      Cond = DAG.getSetCC(MVT::i1, SUB,
                          DAG.getConstant(High-Low, VT), ISD::SETULE);
    }
  }
  
  // Update successor info
  CurMBB->addSuccessor(CB.TrueBB);
  CurMBB->addSuccessor(CB.FalseBB);
  
  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;
  
  // If the lhs block is the next block, invert the condition so that we can
  // fall through to the lhs instead of the rhs block.
  if (CB.TrueBB == NextBlock) {
    std::swap(CB.TrueBB, CB.FalseBB);
    SDOperand True = DAG.getConstant(1, Cond.getValueType());
    Cond = DAG.getNode(ISD::XOR, Cond.getValueType(), Cond, True);
  }
  SDOperand BrCond = DAG.getNode(ISD::BRCOND, MVT::Other, getControlRoot(), Cond,
                                 DAG.getBasicBlock(CB.TrueBB));
  if (CB.FalseBB == NextBlock)
    DAG.setRoot(BrCond);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrCond, 
                            DAG.getBasicBlock(CB.FalseBB)));
}

/// visitJumpTable - Emit JumpTable node in the current MBB
void SelectionDAGLowering::visitJumpTable(SelectionDAGISel::JumpTable &JT) {
  // Emit the code for the jump table
  assert(JT.Reg != -1U && "Should lower JT Header first!");
  MVT PTy = TLI.getPointerTy();
  SDOperand Index = DAG.getCopyFromReg(getControlRoot(), JT.Reg, PTy);
  SDOperand Table = DAG.getJumpTable(JT.JTI, PTy);
  DAG.setRoot(DAG.getNode(ISD::BR_JT, MVT::Other, Index.getValue(1),
                          Table, Index));
  return;
}

/// visitJumpTableHeader - This function emits necessary code to produce index
/// in the JumpTable from switch case.
void SelectionDAGLowering::visitJumpTableHeader(SelectionDAGISel::JumpTable &JT,
                                         SelectionDAGISel::JumpTableHeader &JTH) {
  // Subtract the lowest switch case value from the value being switched on
  // and conditional branch to default mbb if the result is greater than the
  // difference between smallest and largest cases.
  SDOperand SwitchOp = getValue(JTH.SValue);
  MVT VT = SwitchOp.getValueType();
  SDOperand SUB = DAG.getNode(ISD::SUB, VT, SwitchOp,
                              DAG.getConstant(JTH.First, VT));
  
  // The SDNode we just created, which holds the value being switched on
  // minus the the smallest case value, needs to be copied to a virtual
  // register so it can be used as an index into the jump table in a 
  // subsequent basic block.  This value may be smaller or larger than the
  // target's pointer type, and therefore require extension or truncating.
  if (VT.bitsGT(TLI.getPointerTy()))
    SwitchOp = DAG.getNode(ISD::TRUNCATE, TLI.getPointerTy(), SUB);
  else
    SwitchOp = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(), SUB);
  
  unsigned JumpTableReg = FuncInfo.MakeReg(TLI.getPointerTy());
  SDOperand CopyTo = DAG.getCopyToReg(getControlRoot(), JumpTableReg, SwitchOp);
  JT.Reg = JumpTableReg;

  // Emit the range check for the jump table, and branch to the default
  // block for the switch statement if the value being switched on exceeds
  // the largest case in the switch.
  SDOperand CMP = DAG.getSetCC(TLI.getSetCCResultType(SUB), SUB,
                               DAG.getConstant(JTH.Last-JTH.First,VT),
                               ISD::SETUGT);

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  SDOperand BrCond = DAG.getNode(ISD::BRCOND, MVT::Other, CopyTo, CMP,
                                 DAG.getBasicBlock(JT.Default));

  if (JT.MBB == NextBlock)
    DAG.setRoot(BrCond);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrCond, 
                            DAG.getBasicBlock(JT.MBB)));

  return;
}

/// visitBitTestHeader - This function emits necessary code to produce value
/// suitable for "bit tests"
void SelectionDAGLowering::visitBitTestHeader(SelectionDAGISel::BitTestBlock &B) {
  // Subtract the minimum value
  SDOperand SwitchOp = getValue(B.SValue);
  MVT VT = SwitchOp.getValueType();
  SDOperand SUB = DAG.getNode(ISD::SUB, VT, SwitchOp,
                              DAG.getConstant(B.First, VT));

  // Check range
  SDOperand RangeCmp = DAG.getSetCC(TLI.getSetCCResultType(SUB), SUB,
                                    DAG.getConstant(B.Range, VT),
                                    ISD::SETUGT);

  SDOperand ShiftOp;
  if (VT.bitsGT(TLI.getShiftAmountTy()))
    ShiftOp = DAG.getNode(ISD::TRUNCATE, TLI.getShiftAmountTy(), SUB);
  else
    ShiftOp = DAG.getNode(ISD::ZERO_EXTEND, TLI.getShiftAmountTy(), SUB);

  // Make desired shift
  SDOperand SwitchVal = DAG.getNode(ISD::SHL, TLI.getPointerTy(),
                                    DAG.getConstant(1, TLI.getPointerTy()),
                                    ShiftOp);

  unsigned SwitchReg = FuncInfo.MakeReg(TLI.getPointerTy());
  SDOperand CopyTo = DAG.getCopyToReg(getControlRoot(), SwitchReg, SwitchVal);
  B.Reg = SwitchReg;

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  MachineBasicBlock* MBB = B.Cases[0].ThisBB;

  CurMBB->addSuccessor(B.Default);
  CurMBB->addSuccessor(MBB);

  SDOperand BrRange = DAG.getNode(ISD::BRCOND, MVT::Other, CopyTo, RangeCmp,
                                  DAG.getBasicBlock(B.Default));
  
  if (MBB == NextBlock)
    DAG.setRoot(BrRange);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, CopyTo,
                            DAG.getBasicBlock(MBB)));

  return;
}

/// visitBitTestCase - this function produces one "bit test"
void SelectionDAGLowering::visitBitTestCase(MachineBasicBlock* NextMBB,
                                            unsigned Reg,
                                            SelectionDAGISel::BitTestCase &B) {
  // Emit bit tests and jumps
  SDOperand SwitchVal = DAG.getCopyFromReg(getControlRoot(), Reg, TLI.getPointerTy());
  
  SDOperand AndOp = DAG.getNode(ISD::AND, TLI.getPointerTy(),
                                SwitchVal,
                                DAG.getConstant(B.Mask,
                                                TLI.getPointerTy()));
  SDOperand AndCmp = DAG.getSetCC(TLI.getSetCCResultType(AndOp), AndOp,
                                  DAG.getConstant(0, TLI.getPointerTy()),
                                  ISD::SETNE);

  CurMBB->addSuccessor(B.TargetBB);
  CurMBB->addSuccessor(NextMBB);
  
  SDOperand BrAnd = DAG.getNode(ISD::BRCOND, MVT::Other, getControlRoot(),
                                AndCmp, DAG.getBasicBlock(B.TargetBB));

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  if (NextMBB == NextBlock)
    DAG.setRoot(BrAnd);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrAnd,
                            DAG.getBasicBlock(NextMBB)));

  return;
}

void SelectionDAGLowering::visitInvoke(InvokeInst &I) {
  // Retrieve successors.
  MachineBasicBlock *Return = FuncInfo.MBBMap[I.getSuccessor(0)];
  MachineBasicBlock *LandingPad = FuncInfo.MBBMap[I.getSuccessor(1)];

  if (isa<InlineAsm>(I.getCalledValue()))
    visitInlineAsm(&I);
  else
    LowerCallTo(&I, getValue(I.getOperand(0)), false, LandingPad);

  // If the value of the invoke is used outside of its defining block, make it
  // available as a virtual register.
  if (!I.use_empty()) {
    DenseMap<const Value*, unsigned>::iterator VMI = FuncInfo.ValueMap.find(&I);
    if (VMI != FuncInfo.ValueMap.end())
      CopyValueToVirtualRegister(&I, VMI->second);
  }

  // Update successor info
  CurMBB->addSuccessor(Return);
  CurMBB->addSuccessor(LandingPad);

  // Drop into normal successor.
  DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getControlRoot(),
                          DAG.getBasicBlock(Return)));
}

void SelectionDAGLowering::visitUnwind(UnwindInst &I) {
}

/// handleSmallSwitchCaseRange - Emit a series of specific tests (suitable for
/// small case ranges).
bool SelectionDAGLowering::handleSmallSwitchRange(CaseRec& CR,
                                                  CaseRecVector& WorkList,
                                                  Value* SV,
                                                  MachineBasicBlock* Default) {
  Case& BackCase  = *(CR.Range.second-1);
  
  // Size is the number of Cases represented by this range.
  unsigned Size = CR.Range.second - CR.Range.first;
  if (Size > 3)
    return false;  
  
  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  // TODO: If any two of the cases has the same destination, and if one value
  // is the same as the other, but has one bit unset that the other has set,
  // use bit manipulation to do two compares at once.  For example:
  // "if (X == 6 || X == 4)" -> "if ((X|2) == 6)"
    
  // Rearrange the case blocks so that the last one falls through if possible.
  if (NextBlock && Default != NextBlock && BackCase.BB != NextBlock) {
    // The last case block won't fall through into 'NextBlock' if we emit the
    // branches in this order.  See if rearranging a case value would help.
    for (CaseItr I = CR.Range.first, E = CR.Range.second-1; I != E; ++I) {
      if (I->BB == NextBlock) {
        std::swap(*I, BackCase);
        break;
      }
    }
  }
  
  // Create a CaseBlock record representing a conditional branch to
  // the Case's target mbb if the value being switched on SV is equal
  // to C.
  MachineBasicBlock *CurBlock = CR.CaseBB;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++I) {
    MachineBasicBlock *FallThrough;
    if (I != E-1) {
      FallThrough = new MachineBasicBlock(CurBlock->getBasicBlock());
      CurMF->getBasicBlockList().insert(BBI, FallThrough);
    } else {
      // If the last case doesn't match, go to the default block.
      FallThrough = Default;
    }

    Value *RHS, *LHS, *MHS;
    ISD::CondCode CC;
    if (I->High == I->Low) {
      // This is just small small case range :) containing exactly 1 case
      CC = ISD::SETEQ;
      LHS = SV; RHS = I->High; MHS = NULL;
    } else {
      CC = ISD::SETLE;
      LHS = I->Low; MHS = SV; RHS = I->High;
    }
    SelectionDAGISel::CaseBlock CB(CC, LHS, RHS, MHS,
                                   I->BB, FallThrough, CurBlock);
    
    // If emitting the first comparison, just call visitSwitchCase to emit the
    // code into the current block.  Otherwise, push the CaseBlock onto the
    // vector to be later processed by SDISel, and insert the node's MBB
    // before the next MBB.
    if (CurBlock == CurMBB)
      visitSwitchCase(CB);
    else
      SwitchCases.push_back(CB);
    
    CurBlock = FallThrough;
  }

  return true;
}

static inline bool areJTsAllowed(const TargetLowering &TLI) {
  return (TLI.isOperationLegal(ISD::BR_JT, MVT::Other) ||
          TLI.isOperationLegal(ISD::BRIND, MVT::Other));
}
  
/// handleJTSwitchCase - Emit jumptable for current switch case range
bool SelectionDAGLowering::handleJTSwitchCase(CaseRec& CR,
                                              CaseRecVector& WorkList,
                                              Value* SV,
                                              MachineBasicBlock* Default) {
  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  int64_t First = cast<ConstantInt>(FrontCase.Low)->getSExtValue();
  int64_t Last  = cast<ConstantInt>(BackCase.High)->getSExtValue();

  uint64_t TSize = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I)
    TSize += I->size();

  if (!areJTsAllowed(TLI) || TSize <= 3)
    return false;
  
  double Density = (double)TSize / (double)((Last - First) + 1ULL);  
  if (Density < 0.4)
    return false;

  DOUT << "Lowering jump table\n"
       << "First entry: " << First << ". Last entry: " << Last << "\n"
       << "Size: " << TSize << ". Density: " << Density << "\n\n";

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Create a new basic block to hold the code for loading the address
  // of the jump table, and jumping to it.  Update successor information;
  // we will either branch to the default case for the switch, or the jump
  // table.
  MachineBasicBlock *JumpTableBB = new MachineBasicBlock(LLVMBB);
  CurMF->getBasicBlockList().insert(BBI, JumpTableBB);
  CR.CaseBB->addSuccessor(Default);
  CR.CaseBB->addSuccessor(JumpTableBB);
                
  // Build a vector of destination BBs, corresponding to each target
  // of the jump table. If the value of the jump table slot corresponds to
  // a case statement, push the case's BB onto the vector, otherwise, push
  // the default BB.
  std::vector<MachineBasicBlock*> DestBBs;
  int64_t TEI = First;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++TEI) {
    int64_t Low = cast<ConstantInt>(I->Low)->getSExtValue();
    int64_t High = cast<ConstantInt>(I->High)->getSExtValue();
    
    if ((Low <= TEI) && (TEI <= High)) {
      DestBBs.push_back(I->BB);
      if (TEI==High)
        ++I;
    } else {
      DestBBs.push_back(Default);
    }
  }
  
  // Update successor info. Add one edge to each unique successor.
  BitVector SuccsHandled(CR.CaseBB->getParent()->getNumBlockIDs());  
  for (std::vector<MachineBasicBlock*>::iterator I = DestBBs.begin(), 
         E = DestBBs.end(); I != E; ++I) {
    if (!SuccsHandled[(*I)->getNumber()]) {
      SuccsHandled[(*I)->getNumber()] = true;
      JumpTableBB->addSuccessor(*I);
    }
  }
      
  // Create a jump table index for this jump table, or return an existing
  // one.
  unsigned JTI = CurMF->getJumpTableInfo()->getJumpTableIndex(DestBBs);
  
  // Set the jump table information so that we can codegen it as a second
  // MachineBasicBlock
  SelectionDAGISel::JumpTable JT(-1U, JTI, JumpTableBB, Default);
  SelectionDAGISel::JumpTableHeader JTH(First, Last, SV, CR.CaseBB,
                                        (CR.CaseBB == CurMBB));
  if (CR.CaseBB == CurMBB)
    visitJumpTableHeader(JT, JTH);
        
  JTCases.push_back(SelectionDAGISel::JumpTableBlock(JTH, JT));

  return true;
}

/// handleBTSplitSwitchCase - emit comparison and split binary search tree into
/// 2 subtrees.
bool SelectionDAGLowering::handleBTSplitSwitchCase(CaseRec& CR,
                                                   CaseRecVector& WorkList,
                                                   Value* SV,
                                                   MachineBasicBlock* Default) {
  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);
  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Size is the number of Cases represented by this range.
  unsigned Size = CR.Range.second - CR.Range.first;

  int64_t First = cast<ConstantInt>(FrontCase.Low)->getSExtValue();
  int64_t Last  = cast<ConstantInt>(BackCase.High)->getSExtValue();
  double FMetric = 0;
  CaseItr Pivot = CR.Range.first + Size/2;

  // Select optimal pivot, maximizing sum density of LHS and RHS. This will
  // (heuristically) allow us to emit JumpTable's later.
  uint64_t TSize = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I)
    TSize += I->size();

  uint64_t LSize = FrontCase.size();
  uint64_t RSize = TSize-LSize;
  DOUT << "Selecting best pivot: \n"
       << "First: " << First << ", Last: " << Last <<"\n"
       << "LSize: " << LSize << ", RSize: " << RSize << "\n";
  for (CaseItr I = CR.Range.first, J=I+1, E = CR.Range.second;
       J!=E; ++I, ++J) {
    int64_t LEnd = cast<ConstantInt>(I->High)->getSExtValue();
    int64_t RBegin = cast<ConstantInt>(J->Low)->getSExtValue();
    assert((RBegin-LEnd>=1) && "Invalid case distance");
    double LDensity = (double)LSize / (double)((LEnd - First) + 1ULL);
    double RDensity = (double)RSize / (double)((Last - RBegin) + 1ULL);
    double Metric = Log2_64(RBegin-LEnd)*(LDensity+RDensity);
    // Should always split in some non-trivial place
    DOUT <<"=>Step\n"
         << "LEnd: " << LEnd << ", RBegin: " << RBegin << "\n"
         << "LDensity: " << LDensity << ", RDensity: " << RDensity << "\n"
         << "Metric: " << Metric << "\n"; 
    if (FMetric < Metric) {
      Pivot = J;
      FMetric = Metric;
      DOUT << "Current metric set to: " << FMetric << "\n";
    }

    LSize += J->size();
    RSize -= J->size();
  }
  if (areJTsAllowed(TLI)) {
    // If our case is dense we *really* should handle it earlier!
    assert((FMetric > 0) && "Should handle dense range earlier!");
  } else {
    Pivot = CR.Range.first + Size/2;
  }
  
  CaseRange LHSR(CR.Range.first, Pivot);
  CaseRange RHSR(Pivot, CR.Range.second);
  Constant *C = Pivot->Low;
  MachineBasicBlock *FalseBB = 0, *TrueBB = 0;
      
  // We know that we branch to the LHS if the Value being switched on is
  // less than the Pivot value, C.  We use this to optimize our binary 
  // tree a bit, by recognizing that if SV is greater than or equal to the
  // LHS's Case Value, and that Case Value is exactly one less than the 
  // Pivot's Value, then we can branch directly to the LHS's Target,
  // rather than creating a leaf node for it.
  if ((LHSR.second - LHSR.first) == 1 &&
      LHSR.first->High == CR.GE &&
      cast<ConstantInt>(C)->getSExtValue() ==
      (cast<ConstantInt>(CR.GE)->getSExtValue() + 1LL)) {
    TrueBB = LHSR.first->BB;
  } else {
    TrueBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, TrueBB);
    WorkList.push_back(CaseRec(TrueBB, C, CR.GE, LHSR));
  }
  
  // Similar to the optimization above, if the Value being switched on is
  // known to be less than the Constant CR.LT, and the current Case Value
  // is CR.LT - 1, then we can branch directly to the target block for
  // the current Case Value, rather than emitting a RHS leaf node for it.
  if ((RHSR.second - RHSR.first) == 1 && CR.LT &&
      cast<ConstantInt>(RHSR.first->Low)->getSExtValue() ==
      (cast<ConstantInt>(CR.LT)->getSExtValue() - 1LL)) {
    FalseBB = RHSR.first->BB;
  } else {
    FalseBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, FalseBB);
    WorkList.push_back(CaseRec(FalseBB,CR.LT,C,RHSR));
  }

  // Create a CaseBlock record representing a conditional branch to
  // the LHS node if the value being switched on SV is less than C. 
  // Otherwise, branch to LHS.
  SelectionDAGISel::CaseBlock CB(ISD::SETLT, SV, C, NULL,
                                 TrueBB, FalseBB, CR.CaseBB);

  if (CR.CaseBB == CurMBB)
    visitSwitchCase(CB);
  else
    SwitchCases.push_back(CB);

  return true;
}

/// handleBitTestsSwitchCase - if current case range has few destination and
/// range span less, than machine word bitwidth, encode case range into series
/// of masks and emit bit tests with these masks.
bool SelectionDAGLowering::handleBitTestsSwitchCase(CaseRec& CR,
                                                    CaseRecVector& WorkList,
                                                    Value* SV,
                                                    MachineBasicBlock* Default){
  unsigned IntPtrBits = TLI.getPointerTy().getSizeInBits();

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  unsigned numCmps = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I) {
    // Single case counts one, case range - two.
    if (I->Low == I->High)
      numCmps +=1;
    else
      numCmps +=2;
  }
    
  // Count unique destinations
  SmallSet<MachineBasicBlock*, 4> Dests;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    Dests.insert(I->BB);
    if (Dests.size() > 3)
      // Don't bother the code below, if there are too much unique destinations
      return false;
  }
  DOUT << "Total number of unique destinations: " << Dests.size() << "\n"
       << "Total number of comparisons: " << numCmps << "\n";
  
  // Compute span of values.
  Constant* minValue = FrontCase.Low;
  Constant* maxValue = BackCase.High;
  uint64_t range = cast<ConstantInt>(maxValue)->getSExtValue() -
                   cast<ConstantInt>(minValue)->getSExtValue();
  DOUT << "Compare range: " << range << "\n"
       << "Low bound: " << cast<ConstantInt>(minValue)->getSExtValue() << "\n"
       << "High bound: " << cast<ConstantInt>(maxValue)->getSExtValue() << "\n";
  
  if (range>=IntPtrBits ||
      (!(Dests.size() == 1 && numCmps >= 3) &&
       !(Dests.size() == 2 && numCmps >= 5) &&
       !(Dests.size() >= 3 && numCmps >= 6)))
    return false;
  
  DOUT << "Emitting bit tests\n";
  int64_t lowBound = 0;
    
  // Optimize the case where all the case values fit in a
  // word without having to subtract minValue. In this case,
  // we can optimize away the subtraction.
  if (cast<ConstantInt>(minValue)->getSExtValue() >= 0 &&
      cast<ConstantInt>(maxValue)->getSExtValue() <  IntPtrBits) {
    range = cast<ConstantInt>(maxValue)->getSExtValue();
  } else {
    lowBound = cast<ConstantInt>(minValue)->getSExtValue();
  }
    
  CaseBitsVector CasesBits;
  unsigned i, count = 0;

  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    MachineBasicBlock* Dest = I->BB;
    for (i = 0; i < count; ++i)
      if (Dest == CasesBits[i].BB)
        break;
    
    if (i == count) {
      assert((count < 3) && "Too much destinations to test!");
      CasesBits.push_back(CaseBits(0, Dest, 0));
      count++;
    }
    
    uint64_t lo = cast<ConstantInt>(I->Low)->getSExtValue() - lowBound;
    uint64_t hi = cast<ConstantInt>(I->High)->getSExtValue() - lowBound;
    
    for (uint64_t j = lo; j <= hi; j++) {
      CasesBits[i].Mask |=  1ULL << j;
      CasesBits[i].Bits++;
    }
      
  }
  std::sort(CasesBits.begin(), CasesBits.end(), CaseBitsCmp());
  
  SelectionDAGISel::BitTestInfo BTC;

  // Figure out which block is immediately after the current one.
  MachineFunction::iterator BBI = CR.CaseBB;
  ++BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  DOUT << "Cases:\n";
  for (unsigned i = 0, e = CasesBits.size(); i!=e; ++i) {
    DOUT << "Mask: " << CasesBits[i].Mask << ", Bits: " << CasesBits[i].Bits
         << ", BB: " << CasesBits[i].BB << "\n";

    MachineBasicBlock *CaseBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, CaseBB);
    BTC.push_back(SelectionDAGISel::BitTestCase(CasesBits[i].Mask,
                                                CaseBB,
                                                CasesBits[i].BB));
  }
  
  SelectionDAGISel::BitTestBlock BTB(lowBound, range, SV,
                                     -1U, (CR.CaseBB == CurMBB),
                                     CR.CaseBB, Default, BTC);

  if (CR.CaseBB == CurMBB)
    visitBitTestHeader(BTB);
  
  BitTestCases.push_back(BTB);

  return true;
}


/// Clusterify - Transform simple list of Cases into list of CaseRange's
unsigned SelectionDAGLowering::Clusterify(CaseVector& Cases,
                                          const SwitchInst& SI) {
  unsigned numCmps = 0;

  // Start with "simple" cases
  for (unsigned i = 1; i < SI.getNumSuccessors(); ++i) {
    MachineBasicBlock *SMBB = FuncInfo.MBBMap[SI.getSuccessor(i)];
    Cases.push_back(Case(SI.getSuccessorValue(i),
                         SI.getSuccessorValue(i),
                         SMBB));
  }
  std::sort(Cases.begin(), Cases.end(), CaseCmp());

  // Merge case into clusters
  if (Cases.size()>=2)
    // Must recompute end() each iteration because it may be
    // invalidated by erase if we hold on to it
    for (CaseItr I=Cases.begin(), J=++(Cases.begin()); J!=Cases.end(); ) {
      int64_t nextValue = cast<ConstantInt>(J->Low)->getSExtValue();
      int64_t currentValue = cast<ConstantInt>(I->High)->getSExtValue();
      MachineBasicBlock* nextBB = J->BB;
      MachineBasicBlock* currentBB = I->BB;

      // If the two neighboring cases go to the same destination, merge them
      // into a single case.
      if ((nextValue-currentValue==1) && (currentBB == nextBB)) {
        I->High = J->High;
        J = Cases.erase(J);
      } else {
        I = J++;
      }
    }

  for (CaseItr I=Cases.begin(), E=Cases.end(); I!=E; ++I, ++numCmps) {
    if (I->Low != I->High)
      // A range counts double, since it requires two compares.
      ++numCmps;
  }

  return numCmps;
}

void SelectionDAGLowering::visitSwitch(SwitchInst &SI) {  
  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;

  MachineBasicBlock *Default = FuncInfo.MBBMap[SI.getDefaultDest()];

  // If there is only the default destination, branch to it if it is not the
  // next basic block.  Otherwise, just fall through.
  if (SI.getNumOperands() == 2) {
    // Update machine-CFG edges.

    // If this is not a fall-through branch, emit the branch.
    CurMBB->addSuccessor(Default);
    if (Default != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getControlRoot(),
                              DAG.getBasicBlock(Default)));
    
    return;
  }
  
  // If there are any non-default case statements, create a vector of Cases
  // representing each one, and sort the vector so that we can efficiently
  // create a binary search tree from them.
  CaseVector Cases;
  unsigned numCmps = Clusterify(Cases, SI);
  DOUT << "Clusterify finished. Total clusters: " << Cases.size()
       << ". Total compares: " << numCmps << "\n";

  // Get the Value to be switched on and default basic blocks, which will be
  // inserted into CaseBlock records, representing basic blocks in the binary
  // search tree.
  Value *SV = SI.getOperand(0);

  // Push the initial CaseRec onto the worklist
  CaseRecVector WorkList;
  WorkList.push_back(CaseRec(CurMBB,0,0,CaseRange(Cases.begin(),Cases.end())));

  while (!WorkList.empty()) {
    // Grab a record representing a case range to process off the worklist
    CaseRec CR = WorkList.back();
    WorkList.pop_back();

    if (handleBitTestsSwitchCase(CR, WorkList, SV, Default))
      continue;
    
    // If the range has few cases (two or less) emit a series of specific
    // tests.
    if (handleSmallSwitchRange(CR, WorkList, SV, Default))
      continue;
    
    // If the switch has more than 5 blocks, and at least 40% dense, and the 
    // target supports indirect branches, then emit a jump table rather than 
    // lowering the switch to a binary tree of conditional branches.
    if (handleJTSwitchCase(CR, WorkList, SV, Default))
      continue;
          
    // Emit binary tree. We need to pick a pivot, and push left and right ranges
    // onto the worklist. Leafs are handled via handleSmallSwitchRange() call.
    handleBTSplitSwitchCase(CR, WorkList, SV, Default);
  }
}


void SelectionDAGLowering::visitSub(User &I) {
  // -0.0 - X --> fneg
  const Type *Ty = I.getType();
  if (isa<VectorType>(Ty)) {
    if (ConstantVector *CV = dyn_cast<ConstantVector>(I.getOperand(0))) {
      const VectorType *DestTy = cast<VectorType>(I.getType());
      const Type *ElTy = DestTy->getElementType();
      if (ElTy->isFloatingPoint()) {
        unsigned VL = DestTy->getNumElements();
        std::vector<Constant*> NZ(VL, ConstantFP::getNegativeZero(ElTy));
        Constant *CNZ = ConstantVector::get(&NZ[0], NZ.size());
        if (CV == CNZ) {
          SDOperand Op2 = getValue(I.getOperand(1));
          setValue(&I, DAG.getNode(ISD::FNEG, Op2.getValueType(), Op2));
          return;
        }
      }
    }
  }
  if (Ty->isFloatingPoint()) {
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(I.getOperand(0)))
      if (CFP->isExactlyValue(ConstantFP::getNegativeZero(Ty)->getValueAPF())) {
        SDOperand Op2 = getValue(I.getOperand(1));
        setValue(&I, DAG.getNode(ISD::FNEG, Op2.getValueType(), Op2));
        return;
      }
  }

  visitBinary(I, Ty->isFPOrFPVector() ? ISD::FSUB : ISD::SUB);
}

void SelectionDAGLowering::visitBinary(User &I, unsigned OpCode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  
  setValue(&I, DAG.getNode(OpCode, Op1.getValueType(), Op1, Op2));
}

void SelectionDAGLowering::visitShift(User &I, unsigned Opcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  
  if (TLI.getShiftAmountTy().bitsLT(Op2.getValueType()))
    Op2 = DAG.getNode(ISD::TRUNCATE, TLI.getShiftAmountTy(), Op2);
  else if (TLI.getShiftAmountTy().bitsGT(Op2.getValueType()))
    Op2 = DAG.getNode(ISD::ANY_EXTEND, TLI.getShiftAmountTy(), Op2);
  
  setValue(&I, DAG.getNode(Opcode, Op1.getValueType(), Op1, Op2));
}

void SelectionDAGLowering::visitICmp(User &I) {
  ICmpInst::Predicate predicate = ICmpInst::BAD_ICMP_PREDICATE;
  if (ICmpInst *IC = dyn_cast<ICmpInst>(&I))
    predicate = IC->getPredicate();
  else if (ConstantExpr *IC = dyn_cast<ConstantExpr>(&I))
    predicate = ICmpInst::Predicate(IC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode;
  switch (predicate) {
    case ICmpInst::ICMP_EQ  : Opcode = ISD::SETEQ; break;
    case ICmpInst::ICMP_NE  : Opcode = ISD::SETNE; break;
    case ICmpInst::ICMP_UGT : Opcode = ISD::SETUGT; break;
    case ICmpInst::ICMP_UGE : Opcode = ISD::SETUGE; break;
    case ICmpInst::ICMP_ULT : Opcode = ISD::SETULT; break;
    case ICmpInst::ICMP_ULE : Opcode = ISD::SETULE; break;
    case ICmpInst::ICMP_SGT : Opcode = ISD::SETGT; break;
    case ICmpInst::ICMP_SGE : Opcode = ISD::SETGE; break;
    case ICmpInst::ICMP_SLT : Opcode = ISD::SETLT; break;
    case ICmpInst::ICMP_SLE : Opcode = ISD::SETLE; break;
    default:
      assert(!"Invalid ICmp predicate value");
      Opcode = ISD::SETEQ;
      break;
  }
  setValue(&I, DAG.getSetCC(MVT::i1, Op1, Op2, Opcode));
}

void SelectionDAGLowering::visitFCmp(User &I) {
  FCmpInst::Predicate predicate = FCmpInst::BAD_FCMP_PREDICATE;
  if (FCmpInst *FC = dyn_cast<FCmpInst>(&I))
    predicate = FC->getPredicate();
  else if (ConstantExpr *FC = dyn_cast<ConstantExpr>(&I))
    predicate = FCmpInst::Predicate(FC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Condition, FOC, FPC;
  switch (predicate) {
    case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
    case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
    case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
    case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
    case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
    case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
    case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
    case FCmpInst::FCMP_ORD:   FOC = FPC = ISD::SETO;   break;
    case FCmpInst::FCMP_UNO:   FOC = FPC = ISD::SETUO;  break;
    case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
    case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
    case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
    case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
    case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
    case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
    case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
    default:
      assert(!"Invalid FCmp predicate value");
      FOC = FPC = ISD::SETFALSE;
      break;
  }
  if (FiniteOnlyFPMath())
    Condition = FOC;
  else 
    Condition = FPC;
  setValue(&I, DAG.getSetCC(MVT::i1, Op1, Op2, Condition));
}

void SelectionDAGLowering::visitVICmp(User &I) {
  ICmpInst::Predicate predicate = ICmpInst::BAD_ICMP_PREDICATE;
  if (VICmpInst *IC = dyn_cast<VICmpInst>(&I))
    predicate = IC->getPredicate();
  else if (ConstantExpr *IC = dyn_cast<ConstantExpr>(&I))
    predicate = ICmpInst::Predicate(IC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode;
  switch (predicate) {
    case ICmpInst::ICMP_EQ  : Opcode = ISD::SETEQ; break;
    case ICmpInst::ICMP_NE  : Opcode = ISD::SETNE; break;
    case ICmpInst::ICMP_UGT : Opcode = ISD::SETUGT; break;
    case ICmpInst::ICMP_UGE : Opcode = ISD::SETUGE; break;
    case ICmpInst::ICMP_ULT : Opcode = ISD::SETULT; break;
    case ICmpInst::ICMP_ULE : Opcode = ISD::SETULE; break;
    case ICmpInst::ICMP_SGT : Opcode = ISD::SETGT; break;
    case ICmpInst::ICMP_SGE : Opcode = ISD::SETGE; break;
    case ICmpInst::ICMP_SLT : Opcode = ISD::SETLT; break;
    case ICmpInst::ICMP_SLE : Opcode = ISD::SETLE; break;
    default:
      assert(!"Invalid ICmp predicate value");
      Opcode = ISD::SETEQ;
      break;
  }
  setValue(&I, DAG.getVSetCC(Op1.getValueType(), Op1, Op2, Opcode));
}

void SelectionDAGLowering::visitVFCmp(User &I) {
  FCmpInst::Predicate predicate = FCmpInst::BAD_FCMP_PREDICATE;
  if (VFCmpInst *FC = dyn_cast<VFCmpInst>(&I))
    predicate = FC->getPredicate();
  else if (ConstantExpr *FC = dyn_cast<ConstantExpr>(&I))
    predicate = FCmpInst::Predicate(FC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Condition, FOC, FPC;
  switch (predicate) {
    case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
    case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
    case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
    case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
    case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
    case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
    case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
    case FCmpInst::FCMP_ORD:   FOC = FPC = ISD::SETO;   break;
    case FCmpInst::FCMP_UNO:   FOC = FPC = ISD::SETUO;  break;
    case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
    case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
    case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
    case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
    case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
    case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
    case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
    default:
      assert(!"Invalid VFCmp predicate value");
      FOC = FPC = ISD::SETFALSE;
      break;
  }
  if (FiniteOnlyFPMath())
    Condition = FOC;
  else 
    Condition = FPC;
    
  MVT DestVT = TLI.getValueType(I.getType());
    
  setValue(&I, DAG.getVSetCC(DestVT, Op1, Op2, Condition));
}

void SelectionDAGLowering::visitSelect(User &I) {
  SDOperand Cond     = getValue(I.getOperand(0));
  SDOperand TrueVal  = getValue(I.getOperand(1));
  SDOperand FalseVal = getValue(I.getOperand(2));
  setValue(&I, DAG.getNode(ISD::SELECT, TrueVal.getValueType(), Cond,
                           TrueVal, FalseVal));
}


void SelectionDAGLowering::visitTrunc(User &I) {
  // TruncInst cannot be a no-op cast because sizeof(src) > sizeof(dest).
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::TRUNCATE, DestVT, N));
}

void SelectionDAGLowering::visitZExt(User &I) {
  // ZExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // ZExt also can't be a cast to bool for same reason. So, nothing much to do
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitSExt(User &I) {
  // SExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // SExt also can't be a cast to bool for same reason. So, nothing much to do
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SIGN_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitFPTrunc(User &I) {
  // FPTrunc is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_ROUND, DestVT, N, DAG.getIntPtrConstant(0)));
}

void SelectionDAGLowering::visitFPExt(User &I){ 
  // FPTrunc is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitFPToUI(User &I) { 
  // FPToUI is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_UINT, DestVT, N));
}

void SelectionDAGLowering::visitFPToSI(User &I) {
  // FPToSI is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_SINT, DestVT, N));
}

void SelectionDAGLowering::visitUIToFP(User &I) { 
  // UIToFP is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::UINT_TO_FP, DestVT, N));
}

void SelectionDAGLowering::visitSIToFP(User &I){ 
  // UIToFP is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SINT_TO_FP, DestVT, N));
}

void SelectionDAGLowering::visitPtrToInt(User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDOperand N = getValue(I.getOperand(0));
  MVT SrcVT = N.getValueType();
  MVT DestVT = TLI.getValueType(I.getType());
  SDOperand Result;
  if (DestVT.bitsLT(SrcVT))
    Result = DAG.getNode(ISD::TRUNCATE, DestVT, N);
  else 
    // Note: ZERO_EXTEND can handle cases where the sizes are equal too
    Result = DAG.getNode(ISD::ZERO_EXTEND, DestVT, N);
  setValue(&I, Result);
}

void SelectionDAGLowering::visitIntToPtr(User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDOperand N = getValue(I.getOperand(0));
  MVT SrcVT = N.getValueType();
  MVT DestVT = TLI.getValueType(I.getType());
  if (DestVT.bitsLT(SrcVT))
    setValue(&I, DAG.getNode(ISD::TRUNCATE, DestVT, N));
  else 
    // Note: ZERO_EXTEND can handle cases where the sizes are equal too
    setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitBitCast(User &I) { 
  SDOperand N = getValue(I.getOperand(0));
  MVT DestVT = TLI.getValueType(I.getType());

  // BitCast assures us that source and destination are the same size so this 
  // is either a BIT_CONVERT or a no-op.
  if (DestVT != N.getValueType())
    setValue(&I, DAG.getNode(ISD::BIT_CONVERT, DestVT, N)); // convert types
  else
    setValue(&I, N); // noop cast.
}

void SelectionDAGLowering::visitInsertElement(User &I) {
  SDOperand InVec = getValue(I.getOperand(0));
  SDOperand InVal = getValue(I.getOperand(1));
  SDOperand InIdx = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(),
                                getValue(I.getOperand(2)));

  setValue(&I, DAG.getNode(ISD::INSERT_VECTOR_ELT,
                           TLI.getValueType(I.getType()),
                           InVec, InVal, InIdx));
}

void SelectionDAGLowering::visitExtractElement(User &I) {
  SDOperand InVec = getValue(I.getOperand(0));
  SDOperand InIdx = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(),
                                getValue(I.getOperand(1)));
  setValue(&I, DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
                           TLI.getValueType(I.getType()), InVec, InIdx));
}

void SelectionDAGLowering::visitShuffleVector(User &I) {
  SDOperand V1   = getValue(I.getOperand(0));
  SDOperand V2   = getValue(I.getOperand(1));
  SDOperand Mask = getValue(I.getOperand(2));

  setValue(&I, DAG.getNode(ISD::VECTOR_SHUFFLE,
                           TLI.getValueType(I.getType()),
                           V1, V2, Mask));
}

void SelectionDAGLowering::visitInsertValue(InsertValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  const Value *Op1 = I.getOperand(1);
  const Type *AggTy = I.getType();
  const Type *ValTy = Op1->getType();
  bool IntoUndef = isa<UndefValue>(Op0);
  bool FromUndef = isa<UndefValue>(Op1);

  unsigned LinearIndex = ComputeLinearIndex(TLI, AggTy,
                                            I.idx_begin(), I.idx_end());

  SmallVector<MVT, 4> AggValueVTs;
  ComputeValueVTs(TLI, AggTy, AggValueVTs);
  SmallVector<MVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, ValTy, ValValueVTs);

  unsigned NumAggValues = AggValueVTs.size();
  unsigned NumValValues = ValValueVTs.size();
  SmallVector<SDOperand, 4> Values(NumAggValues);

  SDOperand Agg = getValue(Op0);
  SDOperand Val = getValue(Op1);
  unsigned i = 0;
  // Copy the beginning value(s) from the original aggregate.
  for (; i != LinearIndex; ++i)
    Values[i] = IntoUndef ? DAG.getNode(ISD::UNDEF, AggValueVTs[i]) :
                SDOperand(Agg.Val, Agg.ResNo + i);
  // Copy values from the inserted value(s).
  for (; i != LinearIndex + NumValValues; ++i)
    Values[i] = FromUndef ? DAG.getNode(ISD::UNDEF, AggValueVTs[i]) :
                SDOperand(Val.Val, Val.ResNo + i - LinearIndex);
  // Copy remaining value(s) from the original aggregate.
  for (; i != NumAggValues; ++i)
    Values[i] = IntoUndef ? DAG.getNode(ISD::UNDEF, AggValueVTs[i]) :
                SDOperand(Agg.Val, Agg.ResNo + i);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES,
                           DAG.getVTList(&AggValueVTs[0], NumAggValues),
                           &Values[0], NumAggValues));
}

void SelectionDAGLowering::visitExtractValue(ExtractValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  const Type *AggTy = Op0->getType();
  const Type *ValTy = I.getType();
  bool OutOfUndef = isa<UndefValue>(Op0);

  unsigned LinearIndex = ComputeLinearIndex(TLI, AggTy,
                                            I.idx_begin(), I.idx_end());

  SmallVector<MVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, ValTy, ValValueVTs);

  unsigned NumValValues = ValValueVTs.size();
  SmallVector<SDOperand, 4> Values(NumValValues);

  SDOperand Agg = getValue(Op0);
  // Copy out the selected value(s).
  for (unsigned i = LinearIndex; i != LinearIndex + NumValValues; ++i)
    Values[i - LinearIndex] =
      OutOfUndef ? DAG.getNode(ISD::UNDEF, Agg.Val->getValueType(i)) :
                   SDOperand(Agg.Val, Agg.ResNo + i - LinearIndex);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES,
                           DAG.getVTList(&ValValueVTs[0], NumValValues),
                           &Values[0], NumValValues));
}


void SelectionDAGLowering::visitGetElementPtr(User &I) {
  SDOperand N = getValue(I.getOperand(0));
  const Type *Ty = I.getOperand(0)->getType();

  for (GetElementPtrInst::op_iterator OI = I.op_begin()+1, E = I.op_end();
       OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offset = TD->getStructLayout(StTy)->getElementOffset(Field);
        N = DAG.getNode(ISD::ADD, N.getValueType(), N,
                        DAG.getIntPtrConstant(Offset));
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getZExtValue() == 0) continue;
        uint64_t Offs = 
            TD->getABITypeSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        N = DAG.getNode(ISD::ADD, N.getValueType(), N,
                        DAG.getIntPtrConstant(Offs));
        continue;
      }
      
      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD->getABITypeSize(Ty);
      SDOperand IdxN = getValue(Idx);

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      if (IdxN.getValueType().bitsLT(N.getValueType())) {
        IdxN = DAG.getNode(ISD::SIGN_EXTEND, N.getValueType(), IdxN);
      } else if (IdxN.getValueType().bitsGT(N.getValueType()))
        IdxN = DAG.getNode(ISD::TRUNCATE, N.getValueType(), IdxN);

      // If this is a multiply by a power of two, turn it into a shl
      // immediately.  This is a very common case.
      if (isPowerOf2_64(ElementSize)) {
        unsigned Amt = Log2_64(ElementSize);
        IdxN = DAG.getNode(ISD::SHL, N.getValueType(), IdxN,
                           DAG.getConstant(Amt, TLI.getShiftAmountTy()));
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
        continue;
      }
      
      SDOperand Scale = DAG.getIntPtrConstant(ElementSize);
      IdxN = DAG.getNode(ISD::MUL, N.getValueType(), IdxN, Scale);
      N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
    }
  }
  setValue(&I, N);
}

void SelectionDAGLowering::visitAlloca(AllocaInst &I) {
  // If this is a fixed sized alloca in the entry block of the function,
  // allocate it statically on the stack.
  if (FuncInfo.StaticAllocaMap.count(&I))
    return;   // getValue will auto-populate this.

  const Type *Ty = I.getAllocatedType();
  uint64_t TySize = TLI.getTargetData()->getABITypeSize(Ty);
  unsigned Align =
    std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty),
             I.getAlignment());

  SDOperand AllocSize = getValue(I.getArraySize());
  MVT IntPtr = TLI.getPointerTy();
  if (IntPtr.bitsLT(AllocSize.getValueType()))
    AllocSize = DAG.getNode(ISD::TRUNCATE, IntPtr, AllocSize);
  else if (IntPtr.bitsGT(AllocSize.getValueType()))
    AllocSize = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, AllocSize);

  AllocSize = DAG.getNode(ISD::MUL, IntPtr, AllocSize,
                          DAG.getIntPtrConstant(TySize));

  // Handle alignment.  If the requested alignment is less than or equal to
  // the stack alignment, ignore it.  If the size is greater than or equal to
  // the stack alignment, we note this in the DYNAMIC_STACKALLOC node.
  unsigned StackAlign =
    TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
  if (Align <= StackAlign)
    Align = 0;

  // Round the size of the allocation up to the stack alignment size
  // by add SA-1 to the size.
  AllocSize = DAG.getNode(ISD::ADD, AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(StackAlign-1));
  // Mask out the low bits for alignment purposes.
  AllocSize = DAG.getNode(ISD::AND, AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(~(uint64_t)(StackAlign-1)));

  SDOperand Ops[] = { getRoot(), AllocSize, DAG.getIntPtrConstant(Align) };
  const MVT *VTs = DAG.getNodeValueTypes(AllocSize.getValueType(),
                                                    MVT::Other);
  SDOperand DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, VTs, 2, Ops, 3);
  setValue(&I, DSA);
  DAG.setRoot(DSA.getValue(1));

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  CurMBB->getParent()->getFrameInfo()->CreateVariableSizedObject();
}

void SelectionDAGLowering::visitLoad(LoadInst &I) {
  SDOperand Ptr = getValue(I.getOperand(0));

  SDOperand Root;
  if (I.isVolatile())
    Root = getRoot();
  else {
    // Do not serialize non-volatile loads against each other.
    Root = DAG.getRoot();
  }

  setValue(&I, getLoadFrom(I.getType(), Ptr, I.getOperand(0),
                           Root, I.isVolatile(), I.getAlignment()));
}

SDOperand SelectionDAGLowering::getLoadFrom(const Type *Ty, SDOperand Ptr,
                                            const Value *SV, SDOperand Root,
                                            bool isVolatile, 
                                            unsigned Alignment) {
  SmallVector<MVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, Ty, ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();

  SmallVector<SDOperand, 4> Values(NumValues);
  SmallVector<SDOperand, 4> Chains(NumValues);
  MVT PtrVT = Ptr.getValueType();
  for (unsigned i = 0; i != NumValues; ++i) {
    SDOperand L = DAG.getLoad(ValueVTs[i], Root,
                              DAG.getNode(ISD::ADD, PtrVT, Ptr,
                                          DAG.getConstant(Offsets[i], PtrVT)),
                              SV, Offsets[i],
                              isVolatile, Alignment);
    Values[i] = L;
    Chains[i] = L.getValue(1);
  }
  
  SDOperand Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                &Chains[0], NumValues);
  if (isVolatile)
    DAG.setRoot(Chain);
  else
    PendingLoads.push_back(Chain);

  return DAG.getNode(ISD::MERGE_VALUES,
                     DAG.getVTList(&ValueVTs[0], NumValues),
                     &Values[0], NumValues);
}


void SelectionDAGLowering::visitStore(StoreInst &I) {
  Value *SrcV = I.getOperand(0);
  SDOperand Src = getValue(SrcV);
  Value *PtrV = I.getOperand(1);
  SDOperand Ptr = getValue(PtrV);

  SmallVector<MVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, SrcV->getType(), ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();

  SDOperand Root = getRoot();
  SmallVector<SDOperand, 4> Chains(NumValues);
  MVT PtrVT = Ptr.getValueType();
  bool isVolatile = I.isVolatile();
  unsigned Alignment = I.getAlignment();
  for (unsigned i = 0; i != NumValues; ++i)
    Chains[i] = DAG.getStore(Root, SDOperand(Src.Val, Src.ResNo + i),
                             DAG.getNode(ISD::ADD, PtrVT, Ptr,
                                         DAG.getConstant(Offsets[i], PtrVT)),
                             PtrV, Offsets[i],
                             isVolatile, Alignment);

  DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, &Chains[0], NumValues));
}

/// visitTargetIntrinsic - Lower a call of a target intrinsic to an INTRINSIC
/// node.
void SelectionDAGLowering::visitTargetIntrinsic(CallInst &I, 
                                                unsigned Intrinsic) {
  bool HasChain = !I.doesNotAccessMemory();
  bool OnlyLoad = HasChain && I.onlyReadsMemory();

  // Build the operand list.
  SmallVector<SDOperand, 8> Ops;
  if (HasChain) {  // If this intrinsic has side-effects, chainify it.
    if (OnlyLoad) {
      // We don't need to serialize loads against other loads.
      Ops.push_back(DAG.getRoot());
    } else { 
      Ops.push_back(getRoot());
    }
  }
  
  // Add the intrinsic ID as an integer operand.
  Ops.push_back(DAG.getConstant(Intrinsic, TLI.getPointerTy()));

  // Add all operands of the call to the operand list.
  for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i) {
    SDOperand Op = getValue(I.getOperand(i));
    assert(TLI.isTypeLegal(Op.getValueType()) &&
           "Intrinsic uses a non-legal type?");
    Ops.push_back(Op);
  }

  std::vector<MVT> VTs;
  if (I.getType() != Type::VoidTy) {
    MVT VT = TLI.getValueType(I.getType());
    if (VT.isVector()) {
      const VectorType *DestTy = cast<VectorType>(I.getType());
      MVT EltVT = TLI.getValueType(DestTy->getElementType());
      
      VT = MVT::getVectorVT(EltVT, DestTy->getNumElements());
      assert(VT != MVT::Other && "Intrinsic uses a non-legal type?");
    }
    
    assert(TLI.isTypeLegal(VT) && "Intrinsic uses a non-legal type?");
    VTs.push_back(VT);
  }
  if (HasChain)
    VTs.push_back(MVT::Other);

  const MVT *VTList = DAG.getNodeValueTypes(VTs);

  // Create the node.
  SDOperand Result;
  if (!HasChain)
    Result = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, VTList, VTs.size(),
                         &Ops[0], Ops.size());
  else if (I.getType() != Type::VoidTy)
    Result = DAG.getNode(ISD::INTRINSIC_W_CHAIN, VTList, VTs.size(),
                         &Ops[0], Ops.size());
  else
    Result = DAG.getNode(ISD::INTRINSIC_VOID, VTList, VTs.size(),
                         &Ops[0], Ops.size());

  if (HasChain) {
    SDOperand Chain = Result.getValue(Result.Val->getNumValues()-1);
    if (OnlyLoad)
      PendingLoads.push_back(Chain);
    else
      DAG.setRoot(Chain);
  }
  if (I.getType() != Type::VoidTy) {
    if (const VectorType *PTy = dyn_cast<VectorType>(I.getType())) {
      MVT VT = TLI.getValueType(PTy);
      Result = DAG.getNode(ISD::BIT_CONVERT, VT, Result);
    } 
    setValue(&I, Result);
  }
}

/// ExtractTypeInfo - Returns the type info, possibly bitcast, encoded in V.
static GlobalVariable *ExtractTypeInfo (Value *V) {
  V = V->stripPointerCasts();
  GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  assert ((GV || isa<ConstantPointerNull>(V)) &&
          "TypeInfo must be a global variable or NULL");
  return GV;
}

/// addCatchInfo - Extract the personality and type infos from an eh.selector
/// call, and add them to the specified machine basic block.
static void addCatchInfo(CallInst &I, MachineModuleInfo *MMI,
                         MachineBasicBlock *MBB) {
  // Inform the MachineModuleInfo of the personality for this landing pad.
  ConstantExpr *CE = cast<ConstantExpr>(I.getOperand(2));
  assert(CE->getOpcode() == Instruction::BitCast &&
         isa<Function>(CE->getOperand(0)) &&
         "Personality should be a function");
  MMI->addPersonality(MBB, cast<Function>(CE->getOperand(0)));

  // Gather all the type infos for this landing pad and pass them along to
  // MachineModuleInfo.
  std::vector<GlobalVariable *> TyInfo;
  unsigned N = I.getNumOperands();

  for (unsigned i = N - 1; i > 2; --i) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(i))) {
      unsigned FilterLength = CI->getZExtValue();
      unsigned FirstCatch = i + FilterLength + !FilterLength;
      assert (FirstCatch <= N && "Invalid filter length");

      if (FirstCatch < N) {
        TyInfo.reserve(N - FirstCatch);
        for (unsigned j = FirstCatch; j < N; ++j)
          TyInfo.push_back(ExtractTypeInfo(I.getOperand(j)));
        MMI->addCatchTypeInfo(MBB, TyInfo);
        TyInfo.clear();
      }

      if (!FilterLength) {
        // Cleanup.
        MMI->addCleanup(MBB);
      } else {
        // Filter.
        TyInfo.reserve(FilterLength - 1);
        for (unsigned j = i + 1; j < FirstCatch; ++j)
          TyInfo.push_back(ExtractTypeInfo(I.getOperand(j)));
        MMI->addFilterTypeInfo(MBB, TyInfo);
        TyInfo.clear();
      }

      N = i;
    }
  }

  if (N > 3) {
    TyInfo.reserve(N - 3);
    for (unsigned j = 3; j < N; ++j)
      TyInfo.push_back(ExtractTypeInfo(I.getOperand(j)));
    MMI->addCatchTypeInfo(MBB, TyInfo);
  }
}


/// Inlined utility function to implement binary input atomic intrinsics for 
// visitIntrinsicCall: I is a call instruction
//                     Op is the associated NodeType for I
const char *
SelectionDAGLowering::implVisitBinaryAtomic(CallInst& I, ISD::NodeType Op) {
  SDOperand Root = getRoot();   
  SDOperand O2 = getValue(I.getOperand(2));
  SDOperand L = DAG.getAtomic(Op, Root, 
                              getValue(I.getOperand(1)), 
                              O2, O2.getValueType());
  setValue(&I, L);
  DAG.setRoot(L.getValue(1));
  return 0;
}

/// visitIntrinsicCall - Lower the call to the specified intrinsic function.  If
/// we want to emit this as a call to a named external function, return the name
/// otherwise lower it and return null.
const char *
SelectionDAGLowering::visitIntrinsicCall(CallInst &I, unsigned Intrinsic) {
  switch (Intrinsic) {
  default:
    // By default, turn this into a target intrinsic node.
    visitTargetIntrinsic(I, Intrinsic);
    return 0;
  case Intrinsic::vastart:  visitVAStart(I); return 0;
  case Intrinsic::vaend:    visitVAEnd(I); return 0;
  case Intrinsic::vacopy:   visitVACopy(I); return 0;
  case Intrinsic::returnaddress:
    setValue(&I, DAG.getNode(ISD::RETURNADDR, TLI.getPointerTy(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::frameaddress:
    setValue(&I, DAG.getNode(ISD::FRAMEADDR, TLI.getPointerTy(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::setjmp:
    return "_setjmp"+!TLI.usesUnderscoreSetJmp();
    break;
  case Intrinsic::longjmp:
    return "_longjmp"+!TLI.usesUnderscoreLongJmp();
    break;
  case Intrinsic::memcpy_i32:
  case Intrinsic::memcpy_i64: {
    SDOperand Op1 = getValue(I.getOperand(1));
    SDOperand Op2 = getValue(I.getOperand(2));
    SDOperand Op3 = getValue(I.getOperand(3));
    unsigned Align = cast<ConstantInt>(I.getOperand(4))->getZExtValue();
    DAG.setRoot(DAG.getMemcpy(getRoot(), Op1, Op2, Op3, Align, false,
                              I.getOperand(1), 0, I.getOperand(2), 0));
    return 0;
  }
  case Intrinsic::memset_i32:
  case Intrinsic::memset_i64: {
    SDOperand Op1 = getValue(I.getOperand(1));
    SDOperand Op2 = getValue(I.getOperand(2));
    SDOperand Op3 = getValue(I.getOperand(3));
    unsigned Align = cast<ConstantInt>(I.getOperand(4))->getZExtValue();
    DAG.setRoot(DAG.getMemset(getRoot(), Op1, Op2, Op3, Align,
                              I.getOperand(1), 0));
    return 0;
  }
  case Intrinsic::memmove_i32:
  case Intrinsic::memmove_i64: {
    SDOperand Op1 = getValue(I.getOperand(1));
    SDOperand Op2 = getValue(I.getOperand(2));
    SDOperand Op3 = getValue(I.getOperand(3));
    unsigned Align = cast<ConstantInt>(I.getOperand(4))->getZExtValue();

    // If the source and destination are known to not be aliases, we can
    // lower memmove as memcpy.
    uint64_t Size = -1ULL;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op3))
      Size = C->getValue();
    if (AA.alias(I.getOperand(1), Size, I.getOperand(2), Size) ==
        AliasAnalysis::NoAlias) {
      DAG.setRoot(DAG.getMemcpy(getRoot(), Op1, Op2, Op3, Align, false,
                                I.getOperand(1), 0, I.getOperand(2), 0));
      return 0;
    }

    DAG.setRoot(DAG.getMemmove(getRoot(), Op1, Op2, Op3, Align,
                               I.getOperand(1), 0, I.getOperand(2), 0));
    return 0;
  }
  case Intrinsic::dbg_stoppoint: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgStopPointInst &SPI = cast<DbgStopPointInst>(I);
    if (MMI && SPI.getContext() && MMI->Verify(SPI.getContext())) {
      SDOperand Ops[5];

      Ops[0] = getRoot();
      Ops[1] = getValue(SPI.getLineValue());
      Ops[2] = getValue(SPI.getColumnValue());

      DebugInfoDesc *DD = MMI->getDescFor(SPI.getContext());
      assert(DD && "Not a debug information descriptor");
      CompileUnitDesc *CompileUnit = cast<CompileUnitDesc>(DD);
      
      Ops[3] = DAG.getString(CompileUnit->getFileName());
      Ops[4] = DAG.getString(CompileUnit->getDirectory());
      
      DAG.setRoot(DAG.getNode(ISD::LOCATION, MVT::Other, Ops, 5));
    }

    return 0;
  }
  case Intrinsic::dbg_region_start: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgRegionStartInst &RSI = cast<DbgRegionStartInst>(I);
    if (MMI && RSI.getContext() && MMI->Verify(RSI.getContext())) {
      unsigned LabelID = MMI->RecordRegionStart(RSI.getContext());
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                              DAG.getConstant(LabelID, MVT::i32),
                              DAG.getConstant(0, MVT::i32)));
    }

    return 0;
  }
  case Intrinsic::dbg_region_end: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgRegionEndInst &REI = cast<DbgRegionEndInst>(I);
    if (MMI && REI.getContext() && MMI->Verify(REI.getContext())) {
      unsigned LabelID = MMI->RecordRegionEnd(REI.getContext());
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                              DAG.getConstant(LabelID, MVT::i32),
                              DAG.getConstant(0, MVT::i32)));
    }

    return 0;
  }
  case Intrinsic::dbg_func_start: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    if (!MMI) return 0;
    DbgFuncStartInst &FSI = cast<DbgFuncStartInst>(I);
    Value *SP = FSI.getSubprogram();
    if (SP && MMI->Verify(SP)) {
      // llvm.dbg.func.start implicitly defines a dbg_stoppoint which is
      // what (most?) gdb expects.
      DebugInfoDesc *DD = MMI->getDescFor(SP);
      assert(DD && "Not a debug information descriptor");
      SubprogramDesc *Subprogram = cast<SubprogramDesc>(DD);
      const CompileUnitDesc *CompileUnit = Subprogram->getFile();
      unsigned SrcFile = MMI->RecordSource(CompileUnit->getDirectory(),
                                           CompileUnit->getFileName());
      // Record the source line but does create a label. It will be emitted
      // at asm emission time.
      MMI->RecordSourceLine(Subprogram->getLine(), 0, SrcFile);
    }

    return 0;
  }
  case Intrinsic::dbg_declare: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgDeclareInst &DI = cast<DbgDeclareInst>(I);
    Value *Variable = DI.getVariable();
    if (MMI && Variable && MMI->Verify(Variable))
      DAG.setRoot(DAG.getNode(ISD::DECLARE, MVT::Other, getRoot(),
                              getValue(DI.getAddress()), getValue(Variable)));
    return 0;
  }
    
  case Intrinsic::eh_exception: {
    if (!CurMBB->isLandingPad()) {
      // FIXME: Mark exception register as live in.  Hack for PR1508.
      unsigned Reg = TLI.getExceptionAddressRegister();
      if (Reg) CurMBB->addLiveIn(Reg);
    }
    // Insert the EXCEPTIONADDR instruction.
    SDVTList VTs = DAG.getVTList(TLI.getPointerTy(), MVT::Other);
    SDOperand Ops[1];
    Ops[0] = DAG.getRoot();
    SDOperand Op = DAG.getNode(ISD::EXCEPTIONADDR, VTs, Ops, 1);
    setValue(&I, Op);
    DAG.setRoot(Op.getValue(1));
    return 0;
  }

  case Intrinsic::eh_selector_i32:
  case Intrinsic::eh_selector_i64: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    MVT VT = (Intrinsic == Intrinsic::eh_selector_i32 ?
                         MVT::i32 : MVT::i64);
    
    if (MMI) {
      if (CurMBB->isLandingPad())
        addCatchInfo(I, MMI, CurMBB);
      else {
#ifndef NDEBUG
        FuncInfo.CatchInfoLost.insert(&I);
#endif
        // FIXME: Mark exception selector register as live in.  Hack for PR1508.
        unsigned Reg = TLI.getExceptionSelectorRegister();
        if (Reg) CurMBB->addLiveIn(Reg);
      }

      // Insert the EHSELECTION instruction.
      SDVTList VTs = DAG.getVTList(VT, MVT::Other);
      SDOperand Ops[2];
      Ops[0] = getValue(I.getOperand(1));
      Ops[1] = getRoot();
      SDOperand Op = DAG.getNode(ISD::EHSELECTION, VTs, Ops, 2);
      setValue(&I, Op);
      DAG.setRoot(Op.getValue(1));
    } else {
      setValue(&I, DAG.getConstant(0, VT));
    }
    
    return 0;
  }

  case Intrinsic::eh_typeid_for_i32:
  case Intrinsic::eh_typeid_for_i64: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    MVT VT = (Intrinsic == Intrinsic::eh_typeid_for_i32 ?
                         MVT::i32 : MVT::i64);
    
    if (MMI) {
      // Find the type id for the given typeinfo.
      GlobalVariable *GV = ExtractTypeInfo(I.getOperand(1));

      unsigned TypeID = MMI->getTypeIDFor(GV);
      setValue(&I, DAG.getConstant(TypeID, VT));
    } else {
      // Return something different to eh_selector.
      setValue(&I, DAG.getConstant(1, VT));
    }

    return 0;
  }

  case Intrinsic::eh_return: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();

    if (MMI) {
      MMI->setCallsEHReturn(true);
      DAG.setRoot(DAG.getNode(ISD::EH_RETURN,
                              MVT::Other,
                              getControlRoot(),
                              getValue(I.getOperand(1)),
                              getValue(I.getOperand(2))));
    } else {
      setValue(&I, DAG.getConstant(0, TLI.getPointerTy()));
    }

    return 0;
  }

   case Intrinsic::eh_unwind_init: {    
     if (MachineModuleInfo *MMI = DAG.getMachineModuleInfo()) {
       MMI->setCallsUnwindInit(true);
     }

     return 0;
   }

   case Intrinsic::eh_dwarf_cfa: {
     MVT VT = getValue(I.getOperand(1)).getValueType();
     SDOperand CfaArg;
     if (VT.bitsGT(TLI.getPointerTy()))
       CfaArg = DAG.getNode(ISD::TRUNCATE,
                            TLI.getPointerTy(), getValue(I.getOperand(1)));
     else
       CfaArg = DAG.getNode(ISD::SIGN_EXTEND,
                            TLI.getPointerTy(), getValue(I.getOperand(1)));

     SDOperand Offset = DAG.getNode(ISD::ADD,
                                    TLI.getPointerTy(),
                                    DAG.getNode(ISD::FRAME_TO_ARGS_OFFSET,
                                                TLI.getPointerTy()),
                                    CfaArg);
     setValue(&I, DAG.getNode(ISD::ADD,
                              TLI.getPointerTy(),
                              DAG.getNode(ISD::FRAMEADDR,
                                          TLI.getPointerTy(),
                                          DAG.getConstant(0,
                                                          TLI.getPointerTy())),
                              Offset));
     return 0;
  }

  case Intrinsic::sqrt:
    setValue(&I, DAG.getNode(ISD::FSQRT,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::powi:
    setValue(&I, DAG.getNode(ISD::FPOWI,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1)),
                             getValue(I.getOperand(2))));
    return 0;
  case Intrinsic::sin:
    setValue(&I, DAG.getNode(ISD::FSIN,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::cos:
    setValue(&I, DAG.getNode(ISD::FCOS,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::pow:
    setValue(&I, DAG.getNode(ISD::FPOW,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1)),
                             getValue(I.getOperand(2))));
    return 0;
  case Intrinsic::pcmarker: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::PCMARKER, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::readcyclecounter: {
    SDOperand Op = getRoot();
    SDOperand Tmp = DAG.getNode(ISD::READCYCLECOUNTER,
                                DAG.getNodeValueTypes(MVT::i64, MVT::Other), 2,
                                &Op, 1);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::part_select: {
    // Currently not implemented: just abort
    assert(0 && "part_select intrinsic not implemented");
    abort();
  }
  case Intrinsic::part_set: {
    // Currently not implemented: just abort
    assert(0 && "part_set intrinsic not implemented");
    abort();
  }
  case Intrinsic::bswap:
    setValue(&I, DAG.getNode(ISD::BSWAP,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::cttz: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTTZ, Ty, Arg);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::ctlz: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTLZ, Ty, Arg);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::ctpop: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTPOP, Ty, Arg);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::stacksave: {
    SDOperand Op = getRoot();
    SDOperand Tmp = DAG.getNode(ISD::STACKSAVE,
              DAG.getNodeValueTypes(TLI.getPointerTy(), MVT::Other), 2, &Op, 1);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::stackrestore: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::STACKRESTORE, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::var_annotation:
    // Discard annotate attributes
    return 0;

  case Intrinsic::init_trampoline: {
    const Function *F = cast<Function>(I.getOperand(2)->stripPointerCasts());

    SDOperand Ops[6];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getOperand(1));
    Ops[2] = getValue(I.getOperand(2));
    Ops[3] = getValue(I.getOperand(3));
    Ops[4] = DAG.getSrcValue(I.getOperand(1));
    Ops[5] = DAG.getSrcValue(F);

    SDOperand Tmp = DAG.getNode(ISD::TRAMPOLINE,
                                DAG.getNodeValueTypes(TLI.getPointerTy(),
                                                      MVT::Other), 2,
                                Ops, 6);

    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }

  case Intrinsic::gcroot:
    if (GCI) {
      Value *Alloca = I.getOperand(1);
      Constant *TypeMap = cast<Constant>(I.getOperand(2));
      
      FrameIndexSDNode *FI = cast<FrameIndexSDNode>(getValue(Alloca).Val);
      GCI->addStackRoot(FI->getIndex(), TypeMap);
    }
    return 0;

  case Intrinsic::gcread:
  case Intrinsic::gcwrite:
    assert(0 && "Collector failed to lower gcread/gcwrite intrinsics!");
    return 0;

  case Intrinsic::flt_rounds: {
    setValue(&I, DAG.getNode(ISD::FLT_ROUNDS_, MVT::i32));
    return 0;
  }

  case Intrinsic::trap: {
    DAG.setRoot(DAG.getNode(ISD::TRAP, MVT::Other, getRoot()));
    return 0;
  }
  case Intrinsic::prefetch: {
    SDOperand Ops[4];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getOperand(1));
    Ops[2] = getValue(I.getOperand(2));
    Ops[3] = getValue(I.getOperand(3));
    DAG.setRoot(DAG.getNode(ISD::PREFETCH, MVT::Other, &Ops[0], 4));
    return 0;
  }
  
  case Intrinsic::memory_barrier: {
    SDOperand Ops[6];
    Ops[0] = getRoot();
    for (int x = 1; x < 6; ++x)
      Ops[x] = getValue(I.getOperand(x));

    DAG.setRoot(DAG.getNode(ISD::MEMBARRIER, MVT::Other, &Ops[0], 6));
    return 0;
  }
  case Intrinsic::atomic_lcs: {
    SDOperand Root = getRoot();   
    SDOperand O3 = getValue(I.getOperand(3));
    SDOperand L = DAG.getAtomic(ISD::ATOMIC_LCS, Root, 
                                getValue(I.getOperand(1)), 
                                getValue(I.getOperand(2)),
                                O3, O3.getValueType());
    setValue(&I, L);
    DAG.setRoot(L.getValue(1));
    return 0;
  }
  case Intrinsic::atomic_las:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LAS);
  case Intrinsic::atomic_lss:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LSS);
  case Intrinsic::atomic_load_and:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_AND);
  case Intrinsic::atomic_load_or:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_OR);
  case Intrinsic::atomic_load_xor:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_XOR);
  case Intrinsic::atomic_load_min:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_MIN);
  case Intrinsic::atomic_load_max:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_MAX);
  case Intrinsic::atomic_load_umin:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_UMIN);
  case Intrinsic::atomic_load_umax:
      return implVisitBinaryAtomic(I, ISD::ATOMIC_LOAD_UMAX);                                              
  case Intrinsic::atomic_swap:
    return implVisitBinaryAtomic(I, ISD::ATOMIC_SWAP);
  }
}


void SelectionDAGLowering::LowerCallTo(CallSite CS, SDOperand Callee,
                                       bool IsTailCall,
                                       MachineBasicBlock *LandingPad) {
  const PointerType *PT = cast<PointerType>(CS.getCalledValue()->getType());
  const FunctionType *FTy = cast<FunctionType>(PT->getElementType());
  MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
  unsigned BeginLabel = 0, EndLabel = 0;

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Args.reserve(CS.arg_size());
  for (CallSite::arg_iterator i = CS.arg_begin(), e = CS.arg_end();
       i != e; ++i) {
    SDOperand ArgNode = getValue(*i);
    Entry.Node = ArgNode; Entry.Ty = (*i)->getType();

    unsigned attrInd = i - CS.arg_begin() + 1;
    Entry.isSExt  = CS.paramHasAttr(attrInd, ParamAttr::SExt);
    Entry.isZExt  = CS.paramHasAttr(attrInd, ParamAttr::ZExt);
    Entry.isInReg = CS.paramHasAttr(attrInd, ParamAttr::InReg);
    Entry.isSRet  = CS.paramHasAttr(attrInd, ParamAttr::StructRet);
    Entry.isNest  = CS.paramHasAttr(attrInd, ParamAttr::Nest);
    Entry.isByVal = CS.paramHasAttr(attrInd, ParamAttr::ByVal);
    Entry.Alignment = CS.getParamAlignment(attrInd);
    Args.push_back(Entry);
  }

  if (LandingPad && MMI) {
    // Insert a label before the invoke call to mark the try range.  This can be
    // used to detect deletion of the invoke via the MachineModuleInfo.
    BeginLabel = MMI->NextLabelID();
    // Both PendingLoads and PendingExports must be flushed here;
    // this call might not return.
    (void)getRoot();
    DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getControlRoot(),
                            DAG.getConstant(BeginLabel, MVT::i32),
                            DAG.getConstant(1, MVT::i32)));
  }

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), CS.getType(),
                    CS.paramHasAttr(0, ParamAttr::SExt),
                    CS.paramHasAttr(0, ParamAttr::ZExt),
                    FTy->isVarArg(), CS.getCallingConv(), IsTailCall,
                    Callee, Args, DAG);
  if (CS.getType() != Type::VoidTy)
    setValue(CS.getInstruction(), Result.first);
  DAG.setRoot(Result.second);

  if (LandingPad && MMI) {
    // Insert a label at the end of the invoke call to mark the try range.  This
    // can be used to detect deletion of the invoke via the MachineModuleInfo.
    EndLabel = MMI->NextLabelID();
    DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                            DAG.getConstant(EndLabel, MVT::i32),
                            DAG.getConstant(1, MVT::i32)));

    // Inform MachineModuleInfo of range.
    MMI->addInvoke(LandingPad, BeginLabel, EndLabel);
  }
}


void SelectionDAGLowering::visitCall(CallInst &I) {
  const char *RenameFn = 0;
  if (Function *F = I.getCalledFunction()) {
    if (F->isDeclaration()) {
      if (unsigned IID = F->getIntrinsicID()) {
        RenameFn = visitIntrinsicCall(I, IID);
        if (!RenameFn)
          return;
      }
    }

    // Check for well-known libc/libm calls.  If the function is internal, it
    // can't be a library call.
    unsigned NameLen = F->getNameLen();
    if (!F->hasInternalLinkage() && NameLen) {
      const char *NameStr = F->getNameStart();
      if (NameStr[0] == 'c' &&
          ((NameLen == 8 && !strcmp(NameStr, "copysign")) ||
           (NameLen == 9 && !strcmp(NameStr, "copysignf")))) {
        if (I.getNumOperands() == 3 &&   // Basic sanity checks.
            I.getOperand(1)->getType()->isFloatingPoint() &&
            I.getType() == I.getOperand(1)->getType() &&
            I.getType() == I.getOperand(2)->getType()) {
          SDOperand LHS = getValue(I.getOperand(1));
          SDOperand RHS = getValue(I.getOperand(2));
          setValue(&I, DAG.getNode(ISD::FCOPYSIGN, LHS.getValueType(),
                                   LHS, RHS));
          return;
        }
      } else if (NameStr[0] == 'f' &&
                 ((NameLen == 4 && !strcmp(NameStr, "fabs")) ||
                  (NameLen == 5 && !strcmp(NameStr, "fabsf")) ||
                  (NameLen == 5 && !strcmp(NameStr, "fabsl")))) {
        if (I.getNumOperands() == 2 &&   // Basic sanity checks.
            I.getOperand(1)->getType()->isFloatingPoint() &&
            I.getType() == I.getOperand(1)->getType()) {
          SDOperand Tmp = getValue(I.getOperand(1));
          setValue(&I, DAG.getNode(ISD::FABS, Tmp.getValueType(), Tmp));
          return;
        }
      } else if (NameStr[0] == 's' && 
                 ((NameLen == 3 && !strcmp(NameStr, "sin")) ||
                  (NameLen == 4 && !strcmp(NameStr, "sinf")) ||
                  (NameLen == 4 && !strcmp(NameStr, "sinl")))) {
        if (I.getNumOperands() == 2 &&   // Basic sanity checks.
            I.getOperand(1)->getType()->isFloatingPoint() &&
            I.getType() == I.getOperand(1)->getType()) {
          SDOperand Tmp = getValue(I.getOperand(1));
          setValue(&I, DAG.getNode(ISD::FSIN, Tmp.getValueType(), Tmp));
          return;
        }
      } else if (NameStr[0] == 'c' &&
                 ((NameLen == 3 && !strcmp(NameStr, "cos")) ||
                  (NameLen == 4 && !strcmp(NameStr, "cosf")) ||
                  (NameLen == 4 && !strcmp(NameStr, "cosl")))) {
        if (I.getNumOperands() == 2 &&   // Basic sanity checks.
            I.getOperand(1)->getType()->isFloatingPoint() &&
            I.getType() == I.getOperand(1)->getType()) {
          SDOperand Tmp = getValue(I.getOperand(1));
          setValue(&I, DAG.getNode(ISD::FCOS, Tmp.getValueType(), Tmp));
          return;
        }
      }
    }
  } else if (isa<InlineAsm>(I.getOperand(0))) {
    visitInlineAsm(&I);
    return;
  }

  SDOperand Callee;
  if (!RenameFn)
    Callee = getValue(I.getOperand(0));
  else
    Callee = DAG.getExternalSymbol(RenameFn, TLI.getPointerTy());

  LowerCallTo(&I, Callee, I.isTailCall());
}


void SelectionDAGLowering::visitGetResult(GetResultInst &I) {
  if (isa<UndefValue>(I.getOperand(0))) {
    SDOperand Undef = DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType()));
    setValue(&I, Undef);
    return;
  }
  
  // To add support for individual return values with aggregate types,
  // we'd need a way to take a getresult index and determine which
  // values of the Call SDNode are associated with it.
  assert(TLI.getValueType(I.getType(), true) != MVT::Other &&
         "Individual return values must not be aggregates!");

  SDOperand Call = getValue(I.getOperand(0));
  setValue(&I, SDOperand(Call.Val, I.getIndex()));
}


/// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
/// this value and returns the result as a ValueVT value.  This uses 
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
SDOperand RegsForValue::getCopyFromRegs(SelectionDAG &DAG,
                                        SDOperand &Chain,
                                        SDOperand *Flag) const {
  // Assemble the legal parts into the final values.
  SmallVector<SDOperand, 4> Values(ValueVTs.size());
  SmallVector<SDOperand, 8> Parts;
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    // Copy the legal parts from the registers.
    MVT ValueVT = ValueVTs[Value];
    unsigned NumRegs = TLI->getNumRegisters(ValueVT);
    MVT RegisterVT = RegVTs[Value];

    Parts.resize(NumRegs);
    for (unsigned i = 0; i != NumRegs; ++i) {
      SDOperand P;
      if (Flag == 0)
        P = DAG.getCopyFromReg(Chain, Regs[Part+i], RegisterVT);
      else {
        P = DAG.getCopyFromReg(Chain, Regs[Part+i], RegisterVT, *Flag);
        *Flag = P.getValue(2);
      }
      Chain = P.getValue(1);
      Parts[Part+i] = P;
    }
  
    Values[Value] = getCopyFromParts(DAG, &Parts[Part], NumRegs, RegisterVT,
                                     ValueVT);
    Part += NumRegs;
  }
  
  if (ValueVTs.size() == 1)
    return Values[0];
    
  return DAG.getNode(ISD::MERGE_VALUES,
                     DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                     &Values[0], ValueVTs.size());
}

/// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
/// specified value into the registers specified by this object.  This uses 
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
void RegsForValue::getCopyToRegs(SDOperand Val, SelectionDAG &DAG,
                                 SDOperand &Chain, SDOperand *Flag) const {
  // Get the list of the values's legal parts.
  unsigned NumRegs = Regs.size();
  SmallVector<SDOperand, 8> Parts(NumRegs);
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    MVT ValueVT = ValueVTs[Value];
    unsigned NumParts = TLI->getNumRegisters(ValueVT);
    MVT RegisterVT = RegVTs[Value];

    getCopyToParts(DAG, Val.getValue(Val.ResNo + Value),
                   &Parts[Part], NumParts, RegisterVT);
    Part += NumParts;
  }

  // Copy the parts into the registers.
  SmallVector<SDOperand, 8> Chains(NumRegs);
  for (unsigned i = 0; i != NumRegs; ++i) {
    SDOperand Part;
    if (Flag == 0)
      Part = DAG.getCopyToReg(Chain, Regs[i], Parts[i]);
    else {
      Part = DAG.getCopyToReg(Chain, Regs[i], Parts[i], *Flag);
      *Flag = Part.getValue(1);
    }
    Chains[i] = Part.getValue(0);
  }
  
  if (NumRegs == 1 || Flag)
    // If NumRegs > 1 && Flag is used then the use of the last CopyToReg is 
    // flagged to it. That is the CopyToReg nodes and the user are considered
    // a single scheduling unit. If we create a TokenFactor and return it as
    // chain, then the TokenFactor is both a predecessor (operand) of the
    // user as well as a successor (the TF operands are flagged to the user).
    // c1, f1 = CopyToReg
    // c2, f2 = CopyToReg
    // c3     = TokenFactor c1, c2
    // ...
    //        = op c3, ..., f2
    Chain = Chains[NumRegs-1];
  else
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Chains[0], NumRegs);
}

/// AddInlineAsmOperands - Add this value to the specified inlineasm node
/// operand list.  This adds the code marker and includes the number of 
/// values added into it.
void RegsForValue::AddInlineAsmOperands(unsigned Code, SelectionDAG &DAG,
                                        std::vector<SDOperand> &Ops) const {
  MVT IntPtrTy = DAG.getTargetLoweringInfo().getPointerTy();
  Ops.push_back(DAG.getTargetConstant(Code | (Regs.size() << 3), IntPtrTy));
  for (unsigned Value = 0, Reg = 0, e = ValueVTs.size(); Value != e; ++Value) {
    unsigned NumRegs = TLI->getNumRegisters(ValueVTs[Value]);
    MVT RegisterVT = RegVTs[Value];
    for (unsigned i = 0; i != NumRegs; ++i)
      Ops.push_back(DAG.getRegister(Regs[Reg++], RegisterVT));
  }
}

/// isAllocatableRegister - If the specified register is safe to allocate, 
/// i.e. it isn't a stack pointer or some other special register, return the
/// register class for the register.  Otherwise, return null.
static const TargetRegisterClass *
isAllocatableRegister(unsigned Reg, MachineFunction &MF,
                      const TargetLowering &TLI,
                      const TargetRegisterInfo *TRI) {
  MVT FoundVT = MVT::Other;
  const TargetRegisterClass *FoundRC = 0;
  for (TargetRegisterInfo::regclass_iterator RCI = TRI->regclass_begin(),
       E = TRI->regclass_end(); RCI != E; ++RCI) {
    MVT ThisVT = MVT::Other;

    const TargetRegisterClass *RC = *RCI;
    // If none of the the value types for this register class are valid, we 
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
         I != E; ++I) {
      if (TLI.isTypeLegal(*I)) {
        // If we have already found this register in a different register class,
        // choose the one with the largest VT specified.  For example, on
        // PowerPC, we favor f64 register classes over f32.
        if (FoundVT == MVT::Other || FoundVT.bitsLT(*I)) {
          ThisVT = *I;
          break;
        }
      }
    }
    
    if (ThisVT == MVT::Other) continue;
    
    // NOTE: This isn't ideal.  In particular, this might allocate the
    // frame pointer in functions that need it (due to them not being taken
    // out of allocation, because a variable sized allocation hasn't been seen
    // yet).  This is a slight code pessimization, but should still work.
    for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
         E = RC->allocation_order_end(MF); I != E; ++I)
      if (*I == Reg) {
        // We found a matching register class.  Keep looking at others in case
        // we find one with larger registers that this physreg is also in.
        FoundRC = RC;
        FoundVT = ThisVT;
        break;
      }
  }
  return FoundRC;
}    


namespace {
/// AsmOperandInfo - This contains information for each constraint that we are
/// lowering.
struct SDISelAsmOperandInfo : public TargetLowering::AsmOperandInfo {
  /// CallOperand - If this is the result output operand or a clobber
  /// this is null, otherwise it is the incoming operand to the CallInst.
  /// This gets modified as the asm is processed.
  SDOperand CallOperand;

  /// AssignedRegs - If this is a register or register class operand, this
  /// contains the set of register corresponding to the operand.
  RegsForValue AssignedRegs;
  
  explicit SDISelAsmOperandInfo(const InlineAsm::ConstraintInfo &info)
    : TargetLowering::AsmOperandInfo(info), CallOperand(0,0) {
  }
  
  /// MarkAllocatedRegs - Once AssignedRegs is set, mark the assigned registers
  /// busy in OutputRegs/InputRegs.
  void MarkAllocatedRegs(bool isOutReg, bool isInReg,
                         std::set<unsigned> &OutputRegs, 
                         std::set<unsigned> &InputRegs,
                         const TargetRegisterInfo &TRI) const {
    if (isOutReg) {
      for (unsigned i = 0, e = AssignedRegs.Regs.size(); i != e; ++i)
        MarkRegAndAliases(AssignedRegs.Regs[i], OutputRegs, TRI);
    }
    if (isInReg) {
      for (unsigned i = 0, e = AssignedRegs.Regs.size(); i != e; ++i)
        MarkRegAndAliases(AssignedRegs.Regs[i], InputRegs, TRI);
    }
  }
  
private:
  /// MarkRegAndAliases - Mark the specified register and all aliases in the
  /// specified set.
  static void MarkRegAndAliases(unsigned Reg, std::set<unsigned> &Regs, 
                                const TargetRegisterInfo &TRI) {
    assert(TargetRegisterInfo::isPhysicalRegister(Reg) && "Isn't a physreg");
    Regs.insert(Reg);
    if (const unsigned *Aliases = TRI.getAliasSet(Reg))
      for (; *Aliases; ++Aliases)
        Regs.insert(*Aliases);
  }
};
} // end anon namespace.


/// GetRegistersForValue - Assign registers (virtual or physical) for the
/// specified operand.  We prefer to assign virtual registers, to allow the
/// register allocator handle the assignment process.  However, if the asm uses
/// features that we can't model on machineinstrs, we have SDISel do the
/// allocation.  This produces generally horrible, but correct, code.
///
///   OpInfo describes the operand.
///   HasEarlyClobber is true if there are any early clobber constraints (=&r)
///     or any explicitly clobbered registers.
///   Input and OutputRegs are the set of already allocated physical registers.
///
void SelectionDAGLowering::
GetRegistersForValue(SDISelAsmOperandInfo &OpInfo, bool HasEarlyClobber,
                     std::set<unsigned> &OutputRegs, 
                     std::set<unsigned> &InputRegs) {
  // Compute whether this value requires an input register, an output register,
  // or both.
  bool isOutReg = false;
  bool isInReg = false;
  switch (OpInfo.Type) {
  case InlineAsm::isOutput:
    isOutReg = true;
    
    // If this is an early-clobber output, or if there is an input
    // constraint that matches this, we need to reserve the input register
    // so no other inputs allocate to it.
    isInReg = OpInfo.isEarlyClobber || OpInfo.hasMatchingInput;
    break;
  case InlineAsm::isInput:
    isInReg = true;
    isOutReg = false;
    break;
  case InlineAsm::isClobber:
    isOutReg = true;
    isInReg = true;
    break;
  }
  
  
  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<unsigned, 4> Regs;
  
  // If this is a constraint for a single physreg, or a constraint for a
  // register class, find it.
  std::pair<unsigned, const TargetRegisterClass*> PhysReg = 
    TLI.getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                     OpInfo.ConstraintVT);

  unsigned NumRegs = 1;
  if (OpInfo.ConstraintVT != MVT::Other)
    NumRegs = TLI.getNumRegisters(OpInfo.ConstraintVT);
  MVT RegVT;
  MVT ValueVT = OpInfo.ConstraintVT;
  

  // If this is a constraint for a specific physical register, like {r17},
  // assign it now.
  if (PhysReg.first) {
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = *PhysReg.second->vt_begin();
    
    // Get the actual register value type.  This is important, because the user
    // may have asked for (e.g.) the AX register in i32 type.  We need to
    // remember that AX is actually i16 to get the right extension.
    RegVT = *PhysReg.second->vt_begin();
    
    // This is a explicit reference to a physical register.
    Regs.push_back(PhysReg.first);

    // If this is an expanded reference, add the rest of the regs to Regs.
    if (NumRegs != 1) {
      TargetRegisterClass::iterator I = PhysReg.second->begin();
      for (; *I != PhysReg.first; ++I)
        assert(I != PhysReg.second->end() && "Didn't find reg!"); 
      
      // Already added the first reg.
      --NumRegs; ++I;
      for (; NumRegs; --NumRegs, ++I) {
        assert(I != PhysReg.second->end() && "Ran out of registers to allocate!");
        Regs.push_back(*I);
      }
    }
    OpInfo.AssignedRegs = RegsForValue(TLI, Regs, RegVT, ValueVT);
    const TargetRegisterInfo *TRI = DAG.getTarget().getRegisterInfo();
    OpInfo.MarkAllocatedRegs(isOutReg, isInReg, OutputRegs, InputRegs, *TRI);
    return;
  }
  
  // Otherwise, if this was a reference to an LLVM register class, create vregs
  // for this reference.
  std::vector<unsigned> RegClassRegs;
  const TargetRegisterClass *RC = PhysReg.second;
  if (RC) {
    // If this is an early clobber or tied register, our regalloc doesn't know
    // how to maintain the constraint.  If it isn't, go ahead and create vreg
    // and let the regalloc do the right thing.
    if (!OpInfo.hasMatchingInput && !OpInfo.isEarlyClobber &&
        // If there is some other early clobber and this is an input register,
        // then we are forced to pre-allocate the input reg so it doesn't
        // conflict with the earlyclobber.
        !(OpInfo.Type == InlineAsm::isInput && HasEarlyClobber)) {
      RegVT = *PhysReg.second->vt_begin();
      
      if (OpInfo.ConstraintVT == MVT::Other)
        ValueVT = RegVT;

      // Create the appropriate number of virtual registers.
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      for (; NumRegs; --NumRegs)
        Regs.push_back(RegInfo.createVirtualRegister(PhysReg.second));
      
      OpInfo.AssignedRegs = RegsForValue(TLI, Regs, RegVT, ValueVT);
      return;
    }
    
    // Otherwise, we can't allocate it.  Let the code below figure out how to
    // maintain these constraints.
    RegClassRegs.assign(PhysReg.second->begin(), PhysReg.second->end());
    
  } else {
    // This is a reference to a register class that doesn't directly correspond
    // to an LLVM register class.  Allocate NumRegs consecutive, available,
    // registers from the class.
    RegClassRegs = TLI.getRegClassForInlineAsmConstraint(OpInfo.ConstraintCode,
                                                         OpInfo.ConstraintVT);
  }
  
  const TargetRegisterInfo *TRI = DAG.getTarget().getRegisterInfo();
  unsigned NumAllocated = 0;
  for (unsigned i = 0, e = RegClassRegs.size(); i != e; ++i) {
    unsigned Reg = RegClassRegs[i];
    // See if this register is available.
    if ((isOutReg && OutputRegs.count(Reg)) ||   // Already used.
        (isInReg  && InputRegs.count(Reg))) {    // Already used.
      // Make sure we find consecutive registers.
      NumAllocated = 0;
      continue;
    }
    
    // Check to see if this register is allocatable (i.e. don't give out the
    // stack pointer).
    if (RC == 0) {
      RC = isAllocatableRegister(Reg, MF, TLI, TRI);
      if (!RC) {        // Couldn't allocate this register.
        // Reset NumAllocated to make sure we return consecutive registers.
        NumAllocated = 0;
        continue;
      }
    }
    
    // Okay, this register is good, we can use it.
    ++NumAllocated;

    // If we allocated enough consecutive registers, succeed.
    if (NumAllocated == NumRegs) {
      unsigned RegStart = (i-NumAllocated)+1;
      unsigned RegEnd   = i+1;
      // Mark all of the allocated registers used.
      for (unsigned i = RegStart; i != RegEnd; ++i)
        Regs.push_back(RegClassRegs[i]);
      
      OpInfo.AssignedRegs = RegsForValue(TLI, Regs, *RC->vt_begin(), 
                                         OpInfo.ConstraintVT);
      OpInfo.MarkAllocatedRegs(isOutReg, isInReg, OutputRegs, InputRegs, *TRI);
      return;
    }
  }
  
  // Otherwise, we couldn't allocate enough registers for this.
}


/// visitInlineAsm - Handle a call to an InlineAsm object.
///
void SelectionDAGLowering::visitInlineAsm(CallSite CS) {
  InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());

  /// ConstraintOperands - Information about all of the constraints.
  std::vector<SDISelAsmOperandInfo> ConstraintOperands;
  
  SDOperand Chain = getRoot();
  SDOperand Flag;
  
  std::set<unsigned> OutputRegs, InputRegs;

  // Do a prepass over the constraints, canonicalizing them, and building up the
  // ConstraintOperands list.
  std::vector<InlineAsm::ConstraintInfo>
    ConstraintInfos = IA->ParseConstraints();

  // SawEarlyClobber - Keep track of whether we saw an earlyclobber output
  // constraint.  If so, we can't let the register allocator allocate any input
  // registers, because it will not know to avoid the earlyclobbered output reg.
  bool SawEarlyClobber = false;
  
  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  unsigned ResNo = 0;   // ResNo - The result number of the next output.
  for (unsigned i = 0, e = ConstraintInfos.size(); i != e; ++i) {
    ConstraintOperands.push_back(SDISelAsmOperandInfo(ConstraintInfos[i]));
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands.back();
    
    MVT OpVT = MVT::Other;

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      // Indirect outputs just consume an argument.
      if (OpInfo.isIndirect) {
        OpInfo.CallOperandVal = CS.getArgument(ArgNo++);
        break;
      }
      // The return value of the call is this value.  As such, there is no
      // corresponding argument.
      assert(CS.getType() != Type::VoidTy && "Bad inline asm!");
      if (const StructType *STy = dyn_cast<StructType>(CS.getType())) {
        OpVT = TLI.getValueType(STy->getElementType(ResNo));
      } else {
        assert(ResNo == 0 && "Asm only has one result!");
        OpVT = TLI.getValueType(CS.getType());
      }
      ++ResNo;
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = CS.getArgument(ArgNo++);
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    // If this is an input or an indirect output, process the call argument.
    // BasicBlocks are labels, currently appearing only in asm's.
    if (OpInfo.CallOperandVal) {
      if (BasicBlock *BB = dyn_cast<BasicBlock>(OpInfo.CallOperandVal))
        OpInfo.CallOperand = DAG.getBasicBlock(FuncInfo.MBBMap[BB]);
      else {
        OpInfo.CallOperand = getValue(OpInfo.CallOperandVal);
        const Type *OpTy = OpInfo.CallOperandVal->getType();
        // If this is an indirect operand, the operand is a pointer to the
        // accessed type.
        if (OpInfo.isIndirect)
          OpTy = cast<PointerType>(OpTy)->getElementType();

        // If OpTy is not a single value, it may be a struct/union that we
        // can tile with integers.
        if (!OpTy->isSingleValueType() && OpTy->isSized()) {
          unsigned BitSize = TD->getTypeSizeInBits(OpTy);
          switch (BitSize) {
          default: break;
          case 1:
          case 8:
          case 16:
          case 32:
          case 64:
            OpTy = IntegerType::get(BitSize);
            break;
          }
        }

        OpVT = TLI.getValueType(OpTy, true);
      }
    }
    
    OpInfo.ConstraintVT = OpVT;
    
    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, OpInfo.CallOperand, &DAG);

    // Keep track of whether we see an earlyclobber.
    SawEarlyClobber |= OpInfo.isEarlyClobber;
    
    // If we see a clobber of a register, it is an early clobber.
    if (!SawEarlyClobber &&
        OpInfo.Type == InlineAsm::isClobber &&
        OpInfo.ConstraintType == TargetLowering::C_Register) {
      // Note that we want to ignore things that we don't trick here, like
      // dirflag, fpsr, flags, etc.
      std::pair<unsigned, const TargetRegisterClass*> PhysReg = 
        TLI.getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                         OpInfo.ConstraintVT);
      if (PhysReg.first || PhysReg.second) {
        // This is a register we know of.
        SawEarlyClobber = true;
      }
    }
    
    // If this is a memory input, and if the operand is not indirect, do what we
    // need to to provide an address for the memory input.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        !OpInfo.isIndirect) {
      assert(OpInfo.Type == InlineAsm::isInput &&
             "Can only indirectify direct input operands!");
      
      // Memory operands really want the address of the value.  If we don't have
      // an indirect input, put it in the constpool if we can, otherwise spill
      // it to a stack slot.
      
      // If the operand is a float, integer, or vector constant, spill to a
      // constant pool entry to get its address.
      Value *OpVal = OpInfo.CallOperandVal;
      if (isa<ConstantFP>(OpVal) || isa<ConstantInt>(OpVal) ||
          isa<ConstantVector>(OpVal)) {
        OpInfo.CallOperand = DAG.getConstantPool(cast<Constant>(OpVal),
                                                 TLI.getPointerTy());
      } else {
        // Otherwise, create a stack slot and emit a store to it before the
        // asm.
        const Type *Ty = OpVal->getType();
        uint64_t TySize = TLI.getTargetData()->getABITypeSize(Ty);
        unsigned Align  = TLI.getTargetData()->getPrefTypeAlignment(Ty);
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(TySize, Align);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, TLI.getPointerTy());
        Chain = DAG.getStore(Chain, OpInfo.CallOperand, StackSlot, NULL, 0);
        OpInfo.CallOperand = StackSlot;
      }
     
      // There is no longer a Value* corresponding to this operand.
      OpInfo.CallOperandVal = 0;
      // It is now an indirect operand.
      OpInfo.isIndirect = true;
    }
    
    // If this constraint is for a specific register, allocate it before
    // anything else.
    if (OpInfo.ConstraintType == TargetLowering::C_Register)
      GetRegistersForValue(OpInfo, SawEarlyClobber, OutputRegs, InputRegs);
  }
  ConstraintInfos.clear();
  
  
  // Second pass - Loop over all of the operands, assigning virtual or physregs
  // to registerclass operands.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];
    
    // C_Register operands have already been allocated, Other/Memory don't need
    // to be.
    if (OpInfo.ConstraintType == TargetLowering::C_RegisterClass)
      GetRegistersForValue(OpInfo, SawEarlyClobber, OutputRegs, InputRegs);
  }    
  
  // AsmNodeOperands - The operands for the ISD::INLINEASM node.
  std::vector<SDOperand> AsmNodeOperands;
  AsmNodeOperands.push_back(SDOperand());  // reserve space for input chain
  AsmNodeOperands.push_back(
          DAG.getTargetExternalSymbol(IA->getAsmString().c_str(), MVT::Other));
  
  
  // Loop over all of the inputs, copying the operand values into the
  // appropriate registers and processing the output regs.
  RegsForValue RetValRegs;
 
  // IndirectStoresToEmit - The set of stores to emit after the inline asm node.
  std::vector<std::pair<RegsForValue, Value*> > IndirectStoresToEmit;
  
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    switch (OpInfo.Type) {
    case InlineAsm::isOutput: {
      if (OpInfo.ConstraintType != TargetLowering::C_RegisterClass &&
          OpInfo.ConstraintType != TargetLowering::C_Register) {
        // Memory output, or 'other' output (e.g. 'X' constraint).
        assert(OpInfo.isIndirect && "Memory output must be indirect operand");

        // Add information to the INLINEASM node to know about this output.
        unsigned ResOpType = 4/*MEM*/ | (1 << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType, 
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(OpInfo.CallOperand);
        break;
      }

      // Otherwise, this is a register or register class output.

      // Copy the output from the appropriate register.  Find a register that
      // we can use.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        cerr << "Couldn't allocate output reg for contraint '"
             << OpInfo.ConstraintCode << "'!\n";
        exit(1);
      }

      // If this is an indirect operand, store through the pointer after the
      // asm.
      if (OpInfo.isIndirect) {
        IndirectStoresToEmit.push_back(std::make_pair(OpInfo.AssignedRegs,
                                                      OpInfo.CallOperandVal));
      } else {
        // This is the result value of the call.
        assert(CS.getType() != Type::VoidTy && "Bad inline asm!");
        // Concatenate this output onto the outputs list.
        RetValRegs.append(OpInfo.AssignedRegs);
      }
      
      // Add information to the INLINEASM node to know that this register is
      // set.
      OpInfo.AssignedRegs.AddInlineAsmOperands(2 /*REGDEF*/, DAG,
                                               AsmNodeOperands);
      break;
    }
    case InlineAsm::isInput: {
      SDOperand InOperandVal = OpInfo.CallOperand;
      
      if (isdigit(OpInfo.ConstraintCode[0])) {    // Matching constraint?
        // If this is required to match an output register we have already set,
        // just use its register.
        unsigned OperandNo = atoi(OpInfo.ConstraintCode.c_str());
        
        // Scan until we find the definition we already emitted of this operand.
        // When we find it, create a RegsForValue operand.
        unsigned CurOp = 2;  // The first operand.
        for (; OperandNo; --OperandNo) {
          // Advance to the next operand.
          unsigned NumOps = 
            cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getValue();
          assert(((NumOps & 7) == 2 /*REGDEF*/ ||
                  (NumOps & 7) == 4 /*MEM*/) &&
                 "Skipped past definitions?");
          CurOp += (NumOps>>3)+1;
        }

        unsigned NumOps = 
          cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getValue();
        if ((NumOps & 7) == 2 /*REGDEF*/) {
          // Add NumOps>>3 registers to MatchedRegs.
          RegsForValue MatchedRegs;
          MatchedRegs.TLI = &TLI;
          MatchedRegs.ValueVTs.push_back(InOperandVal.getValueType());
          MatchedRegs.RegVTs.push_back(AsmNodeOperands[CurOp+1].getValueType());
          for (unsigned i = 0, e = NumOps>>3; i != e; ++i) {
            unsigned Reg =
              cast<RegisterSDNode>(AsmNodeOperands[++CurOp])->getReg();
            MatchedRegs.Regs.push_back(Reg);
          }
        
          // Use the produced MatchedRegs object to 
          MatchedRegs.getCopyToRegs(InOperandVal, DAG, Chain, &Flag);
          MatchedRegs.AddInlineAsmOperands(1 /*REGUSE*/, DAG, AsmNodeOperands);
          break;
        } else {
          assert((NumOps & 7) == 4/*MEM*/ && "Unknown matching constraint!");
          assert((NumOps >> 3) == 1 && "Unexpected number of operands"); 
          // Add information to the INLINEASM node to know about this input.
          unsigned ResOpType = 4/*MEM*/ | (1 << 3);
          AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                          TLI.getPointerTy()));
          AsmNodeOperands.push_back(AsmNodeOperands[CurOp+1]);
          break;
        }
      }
      
      if (OpInfo.ConstraintType == TargetLowering::C_Other) {
        assert(!OpInfo.isIndirect && 
               "Don't know how to handle indirect other inputs yet!");
        
        std::vector<SDOperand> Ops;
        TLI.LowerAsmOperandForConstraint(InOperandVal, OpInfo.ConstraintCode[0],
                                         Ops, DAG);
        if (Ops.empty()) {
          cerr << "Invalid operand for inline asm constraint '"
               << OpInfo.ConstraintCode << "'!\n";
          exit(1);
        }
        
        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = 3 /*IMM*/ | (Ops.size() << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType, 
                                                        TLI.getPointerTy()));
        AsmNodeOperands.insert(AsmNodeOperands.end(), Ops.begin(), Ops.end());
        break;
      } else if (OpInfo.ConstraintType == TargetLowering::C_Memory) {
        assert(OpInfo.isIndirect && "Operand must be indirect to be a mem!");
        assert(InOperandVal.getValueType() == TLI.getPointerTy() &&
               "Memory operands expect pointer values");
               
        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = 4/*MEM*/ | (1 << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(InOperandVal);
        break;
      }
        
      assert((OpInfo.ConstraintType == TargetLowering::C_RegisterClass ||
              OpInfo.ConstraintType == TargetLowering::C_Register) &&
             "Unknown constraint type!");
      assert(!OpInfo.isIndirect && 
             "Don't know how to handle indirect register inputs yet!");

      // Copy the input into the appropriate registers.
      assert(!OpInfo.AssignedRegs.Regs.empty() &&
             "Couldn't allocate input reg!");

      OpInfo.AssignedRegs.getCopyToRegs(InOperandVal, DAG, Chain, &Flag);
      
      OpInfo.AssignedRegs.AddInlineAsmOperands(1/*REGUSE*/, DAG,
                                               AsmNodeOperands);
      break;
    }
    case InlineAsm::isClobber: {
      // Add the clobbered value to the operand list, so that the register
      // allocator is aware that the physreg got clobbered.
      if (!OpInfo.AssignedRegs.Regs.empty())
        OpInfo.AssignedRegs.AddInlineAsmOperands(2/*REGDEF*/, DAG,
                                                 AsmNodeOperands);
      break;
    }
    }
  }
  
  // Finish up input operands.
  AsmNodeOperands[0] = Chain;
  if (Flag.Val) AsmNodeOperands.push_back(Flag);
  
  Chain = DAG.getNode(ISD::INLINEASM, 
                      DAG.getNodeValueTypes(MVT::Other, MVT::Flag), 2,
                      &AsmNodeOperands[0], AsmNodeOperands.size());
  Flag = Chain.getValue(1);

  // If this asm returns a register value, copy the result from that register
  // and set it as the value of the call.
  if (!RetValRegs.Regs.empty()) {
    SDOperand Val = RetValRegs.getCopyFromRegs(DAG, Chain, &Flag);

    // If any of the results of the inline asm is a vector, it may have the
    // wrong width/num elts.  This can happen for register classes that can
    // contain multiple different value types.  The preg or vreg allocated may
    // not have the same VT as was expected.  Convert it to the right type with
    // bit_convert.
    if (const StructType *ResSTy = dyn_cast<StructType>(CS.getType())) {
      for (unsigned i = 0, e = ResSTy->getNumElements(); i != e; ++i) {
        if (Val.Val->getValueType(i).isVector())
          Val = DAG.getNode(ISD::BIT_CONVERT,
                            TLI.getValueType(ResSTy->getElementType(i)), Val);
      }
    } else {
      if (Val.getValueType().isVector())
        Val = DAG.getNode(ISD::BIT_CONVERT, TLI.getValueType(CS.getType()),
                          Val);
    }

    setValue(CS.getInstruction(), Val);
  }
  
  std::vector<std::pair<SDOperand, Value*> > StoresToEmit;
  
  // Process indirect outputs, first output all of the flagged copies out of
  // physregs.
  for (unsigned i = 0, e = IndirectStoresToEmit.size(); i != e; ++i) {
    RegsForValue &OutRegs = IndirectStoresToEmit[i].first;
    Value *Ptr = IndirectStoresToEmit[i].second;
    SDOperand OutVal = OutRegs.getCopyFromRegs(DAG, Chain, &Flag);
    StoresToEmit.push_back(std::make_pair(OutVal, Ptr));
  }
  
  // Emit the non-flagged stores from the physregs.
  SmallVector<SDOperand, 8> OutChains;
  for (unsigned i = 0, e = StoresToEmit.size(); i != e; ++i)
    OutChains.push_back(DAG.getStore(Chain, StoresToEmit[i].first,
                                    getValue(StoresToEmit[i].second),
                                    StoresToEmit[i].second, 0));
  if (!OutChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &OutChains[0], OutChains.size());
  DAG.setRoot(Chain);
}


void SelectionDAGLowering::visitMalloc(MallocInst &I) {
  SDOperand Src = getValue(I.getOperand(0));

  MVT IntPtr = TLI.getPointerTy();

  if (IntPtr.bitsLT(Src.getValueType()))
    Src = DAG.getNode(ISD::TRUNCATE, IntPtr, Src);
  else if (IntPtr.bitsGT(Src.getValueType()))
    Src = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, Src);

  // Scale the source by the type size.
  uint64_t ElementSize = TD->getABITypeSize(I.getType()->getElementType());
  Src = DAG.getNode(ISD::MUL, Src.getValueType(),
                    Src, DAG.getIntPtrConstant(ElementSize));

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Src;
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Args.push_back(Entry);

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(), false, false, false, CallingConv::C,
                    true, DAG.getExternalSymbol("malloc", IntPtr), Args, DAG);
  setValue(&I, Result.first);  // Pointers always fit in registers
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitFree(FreeInst &I) {
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = getValue(I.getOperand(0));
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Args.push_back(Entry);
  MVT IntPtr = TLI.getPointerTy();
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), Type::VoidTy, false, false, false,
                    CallingConv::C, true,
                    DAG.getExternalSymbol("free", IntPtr), Args, DAG);
  DAG.setRoot(Result.second);
}

// EmitInstrWithCustomInserter - This method should be implemented by targets
// that mark instructions with the 'usesCustomDAGSchedInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and the scheduler passes ownership of it to this method.
MachineBasicBlock *TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                       MachineBasicBlock *MBB) {
  cerr << "If a target marks an instruction with "
       << "'usesCustomDAGSchedInserter', it must implement "
       << "TargetLowering::EmitInstrWithCustomInserter!\n";
  abort();
  return 0;  
}

void SelectionDAGLowering::visitVAStart(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VASTART, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVAArg(VAArgInst &I) {
  SDOperand V = DAG.getVAArg(TLI.getValueType(I.getType()), getRoot(),
                             getValue(I.getOperand(0)),
                             DAG.getSrcValue(I.getOperand(0)));
  setValue(&I, V);
  DAG.setRoot(V.getValue(1));
}

void SelectionDAGLowering::visitVAEnd(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VAEND, MVT::Other, getRoot(),
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVACopy(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VACOPY, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          getValue(I.getOperand(2)),
                          DAG.getSrcValue(I.getOperand(1)),
                          DAG.getSrcValue(I.getOperand(2))));
}

/// TargetLowering::LowerArguments - This is the default LowerArguments
/// implementation, which just inserts a FORMAL_ARGUMENTS node.  FIXME: When all
/// targets are migrated to using FORMAL_ARGUMENTS, this hook should be 
/// integrated into SDISel.
std::vector<SDOperand> 
TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  // Add CC# and isVararg as operands to the FORMAL_ARGUMENTS node.
  std::vector<SDOperand> Ops;
  Ops.push_back(DAG.getRoot());
  Ops.push_back(DAG.getConstant(F.getCallingConv(), getPointerTy()));
  Ops.push_back(DAG.getConstant(F.isVarArg(), getPointerTy()));

  // Add one result value for each formal argument.
  std::vector<MVT> RetVals;
  unsigned j = 1;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++j) {
    MVT VT = getValueType(I->getType());
    ISD::ArgFlagsTy Flags;
    unsigned OriginalAlignment =
      getTargetData()->getABITypeAlignment(I->getType());

    if (F.paramHasAttr(j, ParamAttr::ZExt))
      Flags.setZExt();
    if (F.paramHasAttr(j, ParamAttr::SExt))
      Flags.setSExt();
    if (F.paramHasAttr(j, ParamAttr::InReg))
      Flags.setInReg();
    if (F.paramHasAttr(j, ParamAttr::StructRet))
      Flags.setSRet();
    if (F.paramHasAttr(j, ParamAttr::ByVal)) {
      Flags.setByVal();
      const PointerType *Ty = cast<PointerType>(I->getType());
      const Type *ElementTy = Ty->getElementType();
      unsigned FrameAlign = getByValTypeAlignment(ElementTy);
      unsigned FrameSize  = getTargetData()->getABITypeSize(ElementTy);
      // For ByVal, alignment should be passed from FE.  BE will guess if
      // this info is not there but there are cases it cannot get right.
      if (F.getParamAlignment(j))
        FrameAlign = F.getParamAlignment(j);
      Flags.setByValAlign(FrameAlign);
      Flags.setByValSize(FrameSize);
    }
    if (F.paramHasAttr(j, ParamAttr::Nest))
      Flags.setNest();
    Flags.setOrigAlign(OriginalAlignment);

    MVT RegisterVT = getRegisterType(VT);
    unsigned NumRegs = getNumRegisters(VT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      RetVals.push_back(RegisterVT);
      ISD::ArgFlagsTy MyFlags = Flags;
      if (NumRegs > 1 && i == 0)
        MyFlags.setSplit();
      // if it isn't first piece, alignment must be 1
      else if (i > 0)
        MyFlags.setOrigAlign(1);
      Ops.push_back(DAG.getArgFlags(MyFlags));
    }
  }

  RetVals.push_back(MVT::Other);
  
  // Create the node.
  SDNode *Result = DAG.getNode(ISD::FORMAL_ARGUMENTS,
                               DAG.getVTList(&RetVals[0], RetVals.size()),
                               &Ops[0], Ops.size()).Val;
  
  // Prelower FORMAL_ARGUMENTS.  This isn't required for functionality, but
  // allows exposing the loads that may be part of the argument access to the
  // first DAGCombiner pass.
  SDOperand TmpRes = LowerOperation(SDOperand(Result, 0), DAG);
  
  // The number of results should match up, except that the lowered one may have
  // an extra flag result.
  assert((Result->getNumValues() == TmpRes.Val->getNumValues() ||
          (Result->getNumValues()+1 == TmpRes.Val->getNumValues() &&
           TmpRes.getValue(Result->getNumValues()).getValueType() == MVT::Flag))
         && "Lowering produced unexpected number of results!");
  Result = TmpRes.Val;
  
  unsigned NumArgRegs = Result->getNumValues() - 1;
  DAG.setRoot(SDOperand(Result, NumArgRegs));

  // Set up the return result vector.
  Ops.clear();
  unsigned i = 0;
  unsigned Idx = 1;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; 
      ++I, ++Idx) {
    MVT VT = getValueType(I->getType());
    MVT PartVT = getRegisterType(VT);

    unsigned NumParts = getNumRegisters(VT);
    SmallVector<SDOperand, 4> Parts(NumParts);
    for (unsigned j = 0; j != NumParts; ++j)
      Parts[j] = SDOperand(Result, i++);

    ISD::NodeType AssertOp = ISD::DELETED_NODE;
    if (F.paramHasAttr(Idx, ParamAttr::SExt))
      AssertOp = ISD::AssertSext;
    else if (F.paramHasAttr(Idx, ParamAttr::ZExt))
      AssertOp = ISD::AssertZext;

    Ops.push_back(getCopyFromParts(DAG, &Parts[0], NumParts, PartVT, VT,
                                   AssertOp));
  }
  assert(i == NumArgRegs && "Argument register count mismatch!");
  return Ops;
}


/// TargetLowering::LowerCallTo - This is the default LowerCallTo
/// implementation, which just inserts an ISD::CALL node, which is later custom
/// lowered by the target to something concrete.  FIXME: When all targets are
/// migrated to using ISD::CALL, this hook should be integrated into SDISel.
std::pair<SDOperand, SDOperand>
TargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                            bool RetSExt, bool RetZExt, bool isVarArg,
                            unsigned CallingConv, bool isTailCall,
                            SDOperand Callee,
                            ArgListTy &Args, SelectionDAG &DAG) {
  SmallVector<SDOperand, 32> Ops;
  Ops.push_back(Chain);   // Op#0 - Chain
  Ops.push_back(DAG.getConstant(CallingConv, getPointerTy())); // Op#1 - CC
  Ops.push_back(DAG.getConstant(isVarArg, getPointerTy()));    // Op#2 - VarArg
  Ops.push_back(DAG.getConstant(isTailCall, getPointerTy()));  // Op#3 - Tail
  Ops.push_back(Callee);
  
  // Handle all of the outgoing arguments.
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    MVT VT = getValueType(Args[i].Ty);
    SDOperand Op = Args[i].Node;
    ISD::ArgFlagsTy Flags;
    unsigned OriginalAlignment =
      getTargetData()->getABITypeAlignment(Args[i].Ty);

    if (Args[i].isZExt)
      Flags.setZExt();
    if (Args[i].isSExt)
      Flags.setSExt();
    if (Args[i].isInReg)
      Flags.setInReg();
    if (Args[i].isSRet)
      Flags.setSRet();
    if (Args[i].isByVal) {
      Flags.setByVal();
      const PointerType *Ty = cast<PointerType>(Args[i].Ty);
      const Type *ElementTy = Ty->getElementType();
      unsigned FrameAlign = getByValTypeAlignment(ElementTy);
      unsigned FrameSize  = getTargetData()->getABITypeSize(ElementTy);
      // For ByVal, alignment should come from FE.  BE will guess if this
      // info is not there but there are cases it cannot get right.
      if (Args[i].Alignment)
        FrameAlign = Args[i].Alignment;
      Flags.setByValAlign(FrameAlign);
      Flags.setByValSize(FrameSize);
    }
    if (Args[i].isNest)
      Flags.setNest();
    Flags.setOrigAlign(OriginalAlignment);

    MVT PartVT = getRegisterType(VT);
    unsigned NumParts = getNumRegisters(VT);
    SmallVector<SDOperand, 4> Parts(NumParts);
    ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

    if (Args[i].isSExt)
      ExtendKind = ISD::SIGN_EXTEND;
    else if (Args[i].isZExt)
      ExtendKind = ISD::ZERO_EXTEND;

    getCopyToParts(DAG, Op, &Parts[0], NumParts, PartVT, ExtendKind);

    for (unsigned i = 0; i != NumParts; ++i) {
      // if it isn't first piece, alignment must be 1
      ISD::ArgFlagsTy MyFlags = Flags;
      if (NumParts > 1 && i == 0)
        MyFlags.setSplit();
      else if (i != 0)
        MyFlags.setOrigAlign(1);

      Ops.push_back(Parts[i]);
      Ops.push_back(DAG.getArgFlags(MyFlags));
    }
  }
  
  // Figure out the result value types. We start by making a list of
  // the potentially illegal return value types.
  SmallVector<MVT, 4> LoweredRetTys;
  SmallVector<MVT, 4> RetTys;
  ComputeValueVTs(*this, RetTy, RetTys);

  // Then we translate that to a list of legal types.
  for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
    MVT VT = RetTys[I];
    MVT RegisterVT = getRegisterType(VT);
    unsigned NumRegs = getNumRegisters(VT);
    for (unsigned i = 0; i != NumRegs; ++i)
      LoweredRetTys.push_back(RegisterVT);
  }
  
  LoweredRetTys.push_back(MVT::Other);  // Always has a chain.
  
  // Create the CALL node.
  SDOperand Res = DAG.getNode(ISD::CALL,
                              DAG.getVTList(&LoweredRetTys[0],
                                            LoweredRetTys.size()),
                              &Ops[0], Ops.size());
  Chain = Res.getValue(LoweredRetTys.size() - 1);

  // Gather up the call result into a single value.
  if (RetTy != Type::VoidTy) {
    ISD::NodeType AssertOp = ISD::DELETED_NODE;

    if (RetSExt)
      AssertOp = ISD::AssertSext;
    else if (RetZExt)
      AssertOp = ISD::AssertZext;

    SmallVector<SDOperand, 4> ReturnValues;
    unsigned RegNo = 0;
    for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
      MVT VT = RetTys[I];
      MVT RegisterVT = getRegisterType(VT);
      unsigned NumRegs = getNumRegisters(VT);
      unsigned RegNoEnd = NumRegs + RegNo;
      SmallVector<SDOperand, 4> Results;
      for (; RegNo != RegNoEnd; ++RegNo)
        Results.push_back(Res.getValue(RegNo));
      SDOperand ReturnValue =
        getCopyFromParts(DAG, &Results[0], NumRegs, RegisterVT, VT,
                         AssertOp);
      ReturnValues.push_back(ReturnValue);
    }
    Res = ReturnValues.size() == 1 ? ReturnValues.front() :
          DAG.getNode(ISD::MERGE_VALUES,
                      DAG.getVTList(&RetTys[0], RetTys.size()),
                      &ReturnValues[0], ReturnValues.size());
  }

  return std::make_pair(Res, Chain);
}

SDOperand TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  assert(0 && "LowerOperation not implemented for this target!");
  abort();
  return SDOperand();
}

SDOperand TargetLowering::CustomPromoteOperation(SDOperand Op,
                                                 SelectionDAG &DAG) {
  assert(0 && "CustomPromoteOperation not implemented for this target!");
  abort();
  return SDOperand();
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

unsigned SelectionDAGISel::MakeReg(MVT VT) {
  return RegInfo->createVirtualRegister(TLI.getRegClassFor(VT));
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<CollectorModuleMetadata>();
  AU.setPreservesAll();
}

bool SelectionDAGISel::runOnFunction(Function &Fn) {
  // Get alias analysis for load/store combining.
  AA = &getAnalysis<AliasAnalysis>();

  MachineFunction &MF = MachineFunction::construct(&Fn, TLI.getTargetMachine());
  if (MF.getFunction()->hasCollector())
    GCI = &getAnalysis<CollectorModuleMetadata>().get(*MF.getFunction());
  else
    GCI = 0;
  RegInfo = &MF.getRegInfo();
  DOUT << "\n\n\n=== " << Fn.getName() << "\n";

  FunctionLoweringInfo FuncInfo(TLI, Fn, MF);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    if (InvokeInst *Invoke = dyn_cast<InvokeInst>(I->getTerminator()))
      // Mark landing pad.
      FuncInfo.MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    SelectBasicBlock(I, MF, FuncInfo);

  // Add function live-ins to entry block live-in set.
  BasicBlock *EntryBB = &Fn.getEntryBlock();
  BB = FuncInfo.MBBMap[EntryBB];
  if (!RegInfo->livein_empty())
    for (MachineRegisterInfo::livein_iterator I = RegInfo->livein_begin(),
           E = RegInfo->livein_end(); I != E; ++I)
      BB->addLiveIn(I->first);

#ifndef NDEBUG
  assert(FuncInfo.CatchInfoFound.size() == FuncInfo.CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");
#endif

  return true;
}

void SelectionDAGLowering::CopyValueToVirtualRegister(Value *V, unsigned Reg) {
  SDOperand Op = getValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegisterSDNode>(Op.getOperand(1))->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  assert(!TargetRegisterInfo::isPhysicalRegister(Reg) && "Is a physreg");

  RegsForValue RFV(TLI, Reg, V->getType());
  SDOperand Chain = DAG.getEntryNode();
  RFV.getCopyToRegs(Op, DAG, Chain, 0);
  PendingExports.push_back(Chain);
}

void SelectionDAGISel::
LowerArguments(BasicBlock *LLVMBB, SelectionDAGLowering &SDL) {
  // If this is the entry block, emit arguments.
  Function &F = *LLVMBB->getParent();
  FunctionLoweringInfo &FuncInfo = SDL.FuncInfo;
  SDOperand OldRoot = SDL.DAG.getRoot();
  std::vector<SDOperand> Args = TLI.LowerArguments(F, SDL.DAG);

  unsigned a = 0;
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI, ++a)
    if (!AI->use_empty()) {
      SDL.setValue(AI, Args[a]);

      // If this argument is live outside of the entry block, insert a copy from
      // whereever we got it to the vreg that other BB's will reference it as.
      DenseMap<const Value*, unsigned>::iterator VMI=FuncInfo.ValueMap.find(AI);
      if (VMI != FuncInfo.ValueMap.end()) {
        SDL.CopyValueToVirtualRegister(AI, VMI->second);
      }
    }

  // Finally, if the target has anything special to do, allow it to do so.
  // FIXME: this should insert code into the DAG!
  EmitFunctionEntryCode(F, SDL.DAG.getMachineFunction());
}

static void copyCatchInfo(BasicBlock *SrcBB, BasicBlock *DestBB,
                          MachineModuleInfo *MMI, FunctionLoweringInfo &FLI) {
  for (BasicBlock::iterator I = SrcBB->begin(), E = --SrcBB->end(); I != E; ++I)
    if (isSelector(I)) {
      // Apply the catch info to DestBB.
      addCatchInfo(cast<CallInst>(*I), MMI, FLI.MBBMap[DestBB]);
#ifndef NDEBUG
      if (!FLI.MBBMap[SrcBB]->isLandingPad())
        FLI.CatchInfoFound.insert(I);
#endif
    }
}

/// IsFixedFrameObjectWithPosOffset - Check if object is a fixed frame object and
/// whether object offset >= 0.
static bool
IsFixedFrameObjectWithPosOffset(MachineFrameInfo * MFI, SDOperand Op) {
  if (!isa<FrameIndexSDNode>(Op)) return false;

  FrameIndexSDNode * FrameIdxNode = dyn_cast<FrameIndexSDNode>(Op);
  int FrameIdx =  FrameIdxNode->getIndex();
  return MFI->isFixedObjectIndex(FrameIdx) &&
    MFI->getObjectOffset(FrameIdx) >= 0;
}

/// IsPossiblyOverwrittenArgumentOfTailCall - Check if the operand could
/// possibly be overwritten when lowering the outgoing arguments in a tail
/// call. Currently the implementation of this call is very conservative and
/// assumes all arguments sourcing from FORMAL_ARGUMENTS or a CopyFromReg with
/// virtual registers would be overwritten by direct lowering.
static bool IsPossiblyOverwrittenArgumentOfTailCall(SDOperand Op,
                                                    MachineFrameInfo * MFI) {
  RegisterSDNode * OpReg = NULL;
  if (Op.getOpcode() == ISD::FORMAL_ARGUMENTS ||
      (Op.getOpcode()== ISD::CopyFromReg &&
       (OpReg = dyn_cast<RegisterSDNode>(Op.getOperand(1))) &&
       (OpReg->getReg() >= TargetRegisterInfo::FirstVirtualRegister)) ||
      (Op.getOpcode() == ISD::LOAD &&
       IsFixedFrameObjectWithPosOffset(MFI, Op.getOperand(1))) ||
      (Op.getOpcode() == ISD::MERGE_VALUES &&
       Op.getOperand(Op.ResNo).getOpcode() == ISD::LOAD &&
       IsFixedFrameObjectWithPosOffset(MFI, Op.getOperand(Op.ResNo).
                                       getOperand(1))))
    return true;
  return false;
}

/// CheckDAGForTailCallsAndFixThem - This Function looks for CALL nodes in the
/// DAG and fixes their tailcall attribute operand.
static void CheckDAGForTailCallsAndFixThem(SelectionDAG &DAG, 
                                           TargetLowering& TLI) {
  SDNode * Ret = NULL;
  SDOperand Terminator = DAG.getRoot();

  // Find RET node.
  if (Terminator.getOpcode() == ISD::RET) {
    Ret = Terminator.Val;
  }
 
  // Fix tail call attribute of CALL nodes.
  for (SelectionDAG::allnodes_iterator BE = DAG.allnodes_begin(),
         BI = prior(DAG.allnodes_end()); BI != BE; --BI) {
    if (BI->getOpcode() == ISD::CALL) {
      SDOperand OpRet(Ret, 0);
      SDOperand OpCall(static_cast<SDNode*>(BI), 0);
      bool isMarkedTailCall = 
        cast<ConstantSDNode>(OpCall.getOperand(3))->getValue() != 0;
      // If CALL node has tail call attribute set to true and the call is not
      // eligible (no RET or the target rejects) the attribute is fixed to
      // false. The TargetLowering::IsEligibleForTailCallOptimization function
      // must correctly identify tail call optimizable calls.
      if (!isMarkedTailCall) continue;
      if (Ret==NULL ||
          !TLI.IsEligibleForTailCallOptimization(OpCall, OpRet, DAG)) {
        // Not eligible. Mark CALL node as non tail call.
        SmallVector<SDOperand, 32> Ops;
        unsigned idx=0;
        for(SDNode::op_iterator I =OpCall.Val->op_begin(),
              E = OpCall.Val->op_end(); I != E; I++, idx++) {
          if (idx!=3)
            Ops.push_back(*I);
          else
            Ops.push_back(DAG.getConstant(false, TLI.getPointerTy()));
        }
        DAG.UpdateNodeOperands(OpCall, Ops.begin(), Ops.size());
      } else {
        // Look for tail call clobbered arguments. Emit a series of
        // copyto/copyfrom virtual register nodes to protect them.
        SmallVector<SDOperand, 32> Ops;
        SDOperand Chain = OpCall.getOperand(0), InFlag;
        unsigned idx=0;
        for(SDNode::op_iterator I = OpCall.Val->op_begin(),
              E = OpCall.Val->op_end(); I != E; I++, idx++) {
          SDOperand Arg = *I;
          if (idx > 4 && (idx % 2)) {
            bool isByVal = cast<ARG_FLAGSSDNode>(OpCall.getOperand(idx+1))->
              getArgFlags().isByVal();
            MachineFunction &MF = DAG.getMachineFunction();
            MachineFrameInfo *MFI = MF.getFrameInfo();
            if (!isByVal &&
                IsPossiblyOverwrittenArgumentOfTailCall(Arg, MFI)) {
              MVT VT = Arg.getValueType();
              unsigned VReg = MF.getRegInfo().
                createVirtualRegister(TLI.getRegClassFor(VT));
              Chain = DAG.getCopyToReg(Chain, VReg, Arg, InFlag);
              InFlag = Chain.getValue(1);
              Arg = DAG.getCopyFromReg(Chain, VReg, VT, InFlag);
              Chain = Arg.getValue(1);
              InFlag = Arg.getValue(2);
            }
          }
          Ops.push_back(Arg);
        }
        // Link in chain of CopyTo/CopyFromReg.
        Ops[0] = Chain;
        DAG.UpdateNodeOperands(OpCall, Ops.begin(), Ops.size());
      }
    }
  }
}

void SelectionDAGISel::BuildSelectionDAG(SelectionDAG &DAG, BasicBlock *LLVMBB,
       std::vector<std::pair<MachineInstr*, unsigned> > &PHINodesToUpdate,
                                         FunctionLoweringInfo &FuncInfo) {
  SelectionDAGLowering SDL(DAG, TLI, *AA, FuncInfo, GCI);

  // Lower any arguments needed in this block if this is the entry block.
  if (LLVMBB == &LLVMBB->getParent()->getEntryBlock())
    LowerArguments(LLVMBB, SDL);

  BB = FuncInfo.MBBMap[LLVMBB];
  SDL.setCurrentBasicBlock(BB);

  MachineModuleInfo *MMI = DAG.getMachineModuleInfo();

  if (MMI && BB->isLandingPad()) {
    // Add a label to mark the beginning of the landing pad.  Deletion of the
    // landing pad can thus be detected via the MachineModuleInfo.
    unsigned LabelID = MMI->addLandingPad(BB);
    DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, DAG.getEntryNode(),
                            DAG.getConstant(LabelID, MVT::i32),
                            DAG.getConstant(1, MVT::i32)));

    // Mark exception register as live in.
    unsigned Reg = TLI.getExceptionAddressRegister();
    if (Reg) BB->addLiveIn(Reg);

    // Mark exception selector register as live in.
    Reg = TLI.getExceptionSelectorRegister();
    if (Reg) BB->addLiveIn(Reg);

    // FIXME: Hack around an exception handling flaw (PR1508): the personality
    // function and list of typeids logically belong to the invoke (or, if you
    // like, the basic block containing the invoke), and need to be associated
    // with it in the dwarf exception handling tables.  Currently however the
    // information is provided by an intrinsic (eh.selector) that can be moved
    // to unexpected places by the optimizers: if the unwind edge is critical,
    // then breaking it can result in the intrinsics being in the successor of
    // the landing pad, not the landing pad itself.  This results in exceptions
    // not being caught because no typeids are associated with the invoke.
    // This may not be the only way things can go wrong, but it is the only way
    // we try to work around for the moment.
    BranchInst *Br = dyn_cast<BranchInst>(LLVMBB->getTerminator());

    if (Br && Br->isUnconditional()) { // Critical edge?
      BasicBlock::iterator I, E;
      for (I = LLVMBB->begin(), E = --LLVMBB->end(); I != E; ++I)
        if (isSelector(I))
          break;

      if (I == E)
        // No catch info found - try to extract some from the successor.
        copyCatchInfo(Br->getSuccessor(0), LLVMBB, MMI, FuncInfo);
    }
  }

  // Lower all of the non-terminator instructions.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = --LLVMBB->end();
       I != E; ++I)
    SDL.visit(*I);

  // Ensure that all instructions which are used outside of their defining
  // blocks are available as virtual registers.  Invoke is handled elsewhere.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = LLVMBB->end(); I != E;++I)
    if (!I->use_empty() && !isa<PHINode>(I) && !isa<InvokeInst>(I)) {
      DenseMap<const Value*, unsigned>::iterator VMI =FuncInfo.ValueMap.find(I);
      if (VMI != FuncInfo.ValueMap.end())
        SDL.CopyValueToVirtualRegister(I, VMI->second);
    }

  // Handle PHI nodes in successor blocks.  Emit code into the SelectionDAG to
  // ensure constants are generated when needed.  Remember the virtual registers
  // that need to be added to the Machine PHI nodes as input.  We cannot just
  // directly add them, because expansion might result in multiple MBB's for one
  // BB.  As such, the start of the BB might correspond to a different MBB than
  // the end.
  //
  TerminatorInst *TI = LLVMBB->getTerminator();

  // Emit constants only once even if used by multiple PHI nodes.
  std::map<Constant*, unsigned> ConstantsOut;
  
  // Vector bool would be better, but vector<bool> is really slow.
  std::vector<unsigned char> SuccsHandled;
  if (TI->getNumSuccessors())
    SuccsHandled.resize(BB->getParent()->getNumBlockIDs());
    
  // Check successor nodes' PHI nodes that expect a constant to be available
  // from this block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = FuncInfo.MBBMap[SuccBB];
    
    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    unsigned SuccMBBNo = SuccMBB->getNumber();
    if (SuccsHandled[SuccMBBNo]) continue;
    SuccsHandled[SuccMBBNo] = true;
    
    MachineBasicBlock::iterator MBBI = SuccMBB->begin();
    PHINode *PN;

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::iterator I = SuccBB->begin();
         (PN = dyn_cast<PHINode>(I)); ++I) {
      // Ignore dead phi's.
      if (PN->use_empty()) continue;
      
      unsigned Reg;
      Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);
      
      if (Constant *C = dyn_cast<Constant>(PHIOp)) {
        unsigned &RegOut = ConstantsOut[C];
        if (RegOut == 0) {
          RegOut = FuncInfo.CreateRegForValue(C);
          SDL.CopyValueToVirtualRegister(C, RegOut);
        }
        Reg = RegOut;
      } else {
        Reg = FuncInfo.ValueMap[PHIOp];
        if (Reg == 0) {
          assert(isa<AllocaInst>(PHIOp) &&
                 FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(PHIOp)) &&
                 "Didn't codegen value into a register!??");
          Reg = FuncInfo.CreateRegForValue(PHIOp);
          SDL.CopyValueToVirtualRegister(PHIOp, Reg);
        }
      }

      // Remember that this register needs to added to the machine PHI node as
      // the input for this MBB.
      MVT VT = TLI.getValueType(PN->getType());
      unsigned NumRegisters = TLI.getNumRegisters(VT);
      for (unsigned i = 0, e = NumRegisters; i != e; ++i)
        PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg+i));
    }
  }
  ConstantsOut.clear();

  // Lower the terminator after the copies are emitted.
  SDL.visit(*LLVMBB->getTerminator());

  // Copy over any CaseBlock records that may now exist due to SwitchInst
  // lowering, as well as any jump table information.
  SwitchCases.clear();
  SwitchCases = SDL.SwitchCases;
  JTCases.clear();
  JTCases = SDL.JTCases;
  BitTestCases.clear();
  BitTestCases = SDL.BitTestCases;
    
  // Make sure the root of the DAG is up-to-date.
  DAG.setRoot(SDL.getControlRoot());

  // Check whether calls in this block are real tail calls. Fix up CALL nodes
  // with correct tailcall attribute so that the target can rely on the tailcall
  // attribute indicating whether the call is really eligible for tail call
  // optimization.
  CheckDAGForTailCallsAndFixThem(DAG, TLI);
}

void SelectionDAGISel::CodeGenAndEmitDAG(SelectionDAG &DAG) {
  DOUT << "Lowered selection DAG:\n";
  DEBUG(DAG.dump());

  // Run the DAG combiner in pre-legalize mode.
  DAG.Combine(false, *AA);
  
  DOUT << "Optimized lowered selection DAG:\n";
  DEBUG(DAG.dump());
  
  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
#if 0  // Enable this some day.
  DAG.LegalizeTypes();
  // Someday even later, enable a dag combine pass here.
#endif
  DAG.Legalize();
  
  DOUT << "Legalized selection DAG:\n";
  DEBUG(DAG.dump());
  
  // Run the DAG combiner in post-legalize mode.
  DAG.Combine(true, *AA);
  
  DOUT << "Optimized legalized selection DAG:\n";
  DEBUG(DAG.dump());

  if (ViewISelDAGs) DAG.viewGraph();

  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  InstructionSelectBasicBlock(DAG);
  
  DOUT << "Selected machine code:\n";
  DEBUG(BB->dump());
}  

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB, MachineFunction &MF,
                                        FunctionLoweringInfo &FuncInfo) {
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;
  {
    SelectionDAG DAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &DAG;
  
    // First step, lower LLVM code to some DAG.  This DAG may use operations and
    // types that are not supported by the target.
    BuildSelectionDAG(DAG, LLVMBB, PHINodesToUpdate, FuncInfo);

    // Second step, emit the lowered DAG as machine code.
    CodeGenAndEmitDAG(DAG);
  }

  DOUT << "Total amount of phi nodes to update: "
       << PHINodesToUpdate.size() << "\n";
  DEBUG(for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i)
          DOUT << "Node " << i << " : (" << PHINodesToUpdate[i].first
               << ", " << PHINodesToUpdate[i].second << ")\n";);
  
  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  if (SwitchCases.empty() && JTCases.empty() && BitTestCases.empty()) {
    for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
      MachineInstr *PHI = PHINodesToUpdate[i].first;
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
    return;
  }

  for (unsigned i = 0, e = BitTestCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!BitTestCases[i].Emitted) {
      SelectionDAG HSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &HSDAG;
      SelectionDAGLowering HSDL(HSDAG, TLI, *AA, FuncInfo, GCI);
      // Set the current basic block to the mbb we wish to insert the code into
      BB = BitTestCases[i].Parent;
      HSDL.setCurrentBasicBlock(BB);
      // Emit the code
      HSDL.visitBitTestHeader(BitTestCases[i]);
      HSDAG.setRoot(HSDL.getRoot());
      CodeGenAndEmitDAG(HSDAG);
    }    

    for (unsigned j = 0, ej = BitTestCases[i].Cases.size(); j != ej; ++j) {
      SelectionDAG BSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &BSDAG;
      SelectionDAGLowering BSDL(BSDAG, TLI, *AA, FuncInfo, GCI);
      // Set the current basic block to the mbb we wish to insert the code into
      BB = BitTestCases[i].Cases[j].ThisBB;
      BSDL.setCurrentBasicBlock(BB);
      // Emit the code
      if (j+1 != ej)
        BSDL.visitBitTestCase(BitTestCases[i].Cases[j+1].ThisBB,
                              BitTestCases[i].Reg,
                              BitTestCases[i].Cases[j]);
      else
        BSDL.visitBitTestCase(BitTestCases[i].Default,
                              BitTestCases[i].Reg,
                              BitTestCases[i].Cases[j]);
        
        
      BSDAG.setRoot(BSDL.getRoot());
      CodeGenAndEmitDAG(BSDAG);
    }

    // Update PHI Nodes
    for (unsigned pi = 0, pe = PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // This is "default" BB. We have two jumps to it. From "header" BB and
      // from last "case" BB.
      if (PHIBB == BitTestCases[i].Default) {
        PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(BitTestCases[i].Parent));
        PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(BitTestCases[i].Cases.
                                                  back().ThisBB));
      }
      // One of "cases" BB.
      for (unsigned j = 0, ej = BitTestCases[i].Cases.size(); j != ej; ++j) {
        MachineBasicBlock* cBB = BitTestCases[i].Cases[j].ThisBB;
        if (cBB->succ_end() !=
            std::find(cBB->succ_begin(),cBB->succ_end(), PHIBB)) {
          PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pi].second,
                                                    false));
          PHI->addOperand(MachineOperand::CreateMBB(cBB));
        }
      }
    }
  }

  // If the JumpTable record is filled in, then we need to emit a jump table.
  // Updating the PHI nodes is tricky in this case, since we need to determine
  // whether the PHI is a successor of the range check MBB or the jump table MBB
  for (unsigned i = 0, e = JTCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!JTCases[i].first.Emitted) {
      SelectionDAG HSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &HSDAG;
      SelectionDAGLowering HSDL(HSDAG, TLI, *AA, FuncInfo, GCI);
      // Set the current basic block to the mbb we wish to insert the code into
      BB = JTCases[i].first.HeaderBB;
      HSDL.setCurrentBasicBlock(BB);
      // Emit the code
      HSDL.visitJumpTableHeader(JTCases[i].second, JTCases[i].first);
      HSDAG.setRoot(HSDL.getRoot());
      CodeGenAndEmitDAG(HSDAG);
    }
    
    SelectionDAG JSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &JSDAG;
    SelectionDAGLowering JSDL(JSDAG, TLI, *AA, FuncInfo, GCI);
    // Set the current basic block to the mbb we wish to insert the code into
    BB = JTCases[i].second.MBB;
    JSDL.setCurrentBasicBlock(BB);
    // Emit the code
    JSDL.visitJumpTable(JTCases[i].second);
    JSDAG.setRoot(JSDL.getRoot());
    CodeGenAndEmitDAG(JSDAG);
    
    // Update PHI Nodes
    for (unsigned pi = 0, pe = PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // "default" BB. We can go there only from header BB.
      if (PHIBB == JTCases[i].second.Default) {
        PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(JTCases[i].first.HeaderBB));
      }
      // JT BB. Just iterate over successors here
      if (BB->succ_end() != std::find(BB->succ_begin(),BB->succ_end(), PHIBB)) {
        PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(BB));
      }
    }
  }
  
  // If the switch block involved a branch to one of the actual successors, we
  // need to update PHI nodes in that block.
  for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    if (BB->isSuccessor(PHI->getParent())) {
      PHI->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
  }
  
  // If we generated any switch lowering information, build and codegen any
  // additional DAGs necessary.
  for (unsigned i = 0, e = SwitchCases.size(); i != e; ++i) {
    SelectionDAG SDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &SDAG;
    SelectionDAGLowering SDL(SDAG, TLI, *AA, FuncInfo, GCI);
    
    // Set the current basic block to the mbb we wish to insert the code into
    BB = SwitchCases[i].ThisBB;
    SDL.setCurrentBasicBlock(BB);
    
    // Emit the code
    SDL.visitSwitchCase(SwitchCases[i]);
    SDAG.setRoot(SDL.getRoot());
    CodeGenAndEmitDAG(SDAG);
    
    // Handle any PHI nodes in successors of this chunk, as if we were coming
    // from the original BB before switch expansion.  Note that PHI nodes can
    // occur multiple times in PHINodesToUpdate.  We have to be very careful to
    // handle them the right number of times.
    while ((BB = SwitchCases[i].TrueBB)) {  // Handle LHS and RHS.
      for (MachineBasicBlock::iterator Phi = BB->begin();
           Phi != BB->end() && Phi->getOpcode() == TargetInstrInfo::PHI; ++Phi){
        // This value for this PHI node is recorded in PHINodesToUpdate, get it.
        for (unsigned pn = 0; ; ++pn) {
          assert(pn != PHINodesToUpdate.size() && "Didn't find PHI entry!");
          if (PHINodesToUpdate[pn].first == Phi) {
            Phi->addOperand(MachineOperand::CreateReg(PHINodesToUpdate[pn].
                                                      second, false));
            Phi->addOperand(MachineOperand::CreateMBB(SwitchCases[i].ThisBB));
            break;
          }
        }
      }
      
      // Don't process RHS if same block as LHS.
      if (BB == SwitchCases[i].FalseBB)
        SwitchCases[i].FalseBB = 0;
      
      // If we haven't handled the RHS, do so now.  Otherwise, we're done.
      SwitchCases[i].TrueBB = SwitchCases[i].FalseBB;
      SwitchCases[i].FalseBB = 0;
    }
    assert(SwitchCases[i].TrueBB == 0 && SwitchCases[i].FalseBB == 0);
  }
}


//===----------------------------------------------------------------------===//
/// ScheduleAndEmitDAG - Pick a safe ordering and emit instructions for each
/// target node in the graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &DAG) {
  if (ViewSchedDAGs) DAG.viewGraph();

  RegisterScheduler::FunctionPassCtor Ctor = RegisterScheduler::getDefault();
  
  if (!Ctor) {
    Ctor = ISHeuristic;
    RegisterScheduler::setDefault(Ctor);
  }
  
  ScheduleDAG *SL = Ctor(this, &DAG, BB);
  BB = SL->Run();

  if (ViewSUnitDAGs) SL->viewGraph();

  delete SL;
}


HazardRecognizer *SelectionDAGISel::CreateTargetHazardRecognizer() {
  return new HazardRecognizer();
}

//===----------------------------------------------------------------------===//
// Helper functions used by the generated instruction selector.
//===----------------------------------------------------------------------===//
// Calls to these methods are generated by tblgen.

/// CheckAndMask - The isel is trying to match something like (and X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an AND, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckAndMask(SDOperand LHS, ConstantSDNode *RHS, 
                                    int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  if (CurDAG->MaskedValueIsZero(LHS, NeededMask))
    return true;
  
  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// CheckOrMask - The isel is trying to match something like (or X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an OR, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckOrMask(SDOperand LHS, ConstantSDNode *RHS, 
                                   int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  
  APInt KnownZero, KnownOne;
  CurDAG->ComputeMaskedBits(LHS, NeededMask, KnownZero, KnownOne);
  
  // If all the missing bits in the or are already known to be set, match!
  if ((NeededMask & KnownOne) == NeededMask)
    return true;
  
  // TODO: check to see if missing bits are just not demanded.
  
  // Otherwise, this pattern doesn't match.
  return false;
}


/// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
/// by tblgen.  Others should not call it.
void SelectionDAGISel::
SelectInlineAsmMemoryOperands(std::vector<SDOperand> &Ops, SelectionDAG &DAG) {
  std::vector<SDOperand> InOps;
  std::swap(InOps, Ops);

  Ops.push_back(InOps[0]);  // input chain.
  Ops.push_back(InOps[1]);  // input asm string.

  unsigned i = 2, e = InOps.size();
  if (InOps[e-1].getValueType() == MVT::Flag)
    --e;  // Don't process a flag operand if it is here.
  
  while (i != e) {
    unsigned Flags = cast<ConstantSDNode>(InOps[i])->getValue();
    if ((Flags & 7) != 4 /*MEM*/) {
      // Just skip over this operand, copying the operands verbatim.
      Ops.insert(Ops.end(), InOps.begin()+i, InOps.begin()+i+(Flags >> 3) + 1);
      i += (Flags >> 3) + 1;
    } else {
      assert((Flags >> 3) == 1 && "Memory operand with multiple values?");
      // Otherwise, this is a memory operand.  Ask the target to select it.
      std::vector<SDOperand> SelOps;
      if (SelectInlineAsmMemoryOperand(InOps[i+1], 'm', SelOps, DAG)) {
        cerr << "Could not match memory address.  Inline asm failure!\n";
        exit(1);
      }
      
      // Add this to the output node.
      MVT IntPtrTy = DAG.getTargetLoweringInfo().getPointerTy();
      Ops.push_back(DAG.getTargetConstant(4/*MEM*/ | (SelOps.size() << 3),
                                          IntPtrTy));
      Ops.insert(Ops.end(), SelOps.begin(), SelOps.end());
      i += 2;
    }
  }
  
  // Add the flag input back if present.
  if (e != InOps.size())
    Ops.push_back(InOps.back());
}

char SelectionDAGISel::ID = 0;
