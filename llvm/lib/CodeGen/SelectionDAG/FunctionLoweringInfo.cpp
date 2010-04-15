//===-- FunctionLoweringInfo.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating functions from LLVM IR into
// Machine IR.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "function-lowering-info"
#include "FunctionLoweringInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// of insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
unsigned llvm::ComputeLinearIndex(const TargetLowering &TLI, const Type *Ty,
                                  const unsigned *Indices,
                                  const unsigned *IndicesEnd,
                                  unsigned CurIndex) {
  // Base case: We're done.
  if (Indices && Indices == IndicesEnd)
    return CurIndex;

  // Given a struct type, recursively traverse the elements.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
        EI != EE; ++EI) {
      if (Indices && *Indices == unsigned(EI - EB))
        return ComputeLinearIndex(TLI, *EI, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(TLI, *EI, 0, 0, CurIndex);
    }
    return CurIndex;
  }
  // Given an array type, recursively traverse the elements.
  else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = ATy->getElementType();
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i) {
      if (Indices && *Indices == i)
        return ComputeLinearIndex(TLI, EltTy, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(TLI, EltTy, 0, 0, CurIndex);
    }
    return CurIndex;
  }
  // We haven't found the type we're looking for, so keep searching.
  return CurIndex + 1;
}

/// ComputeValueVTs - Given an LLVM IR type, compute a sequence of
/// EVTs that represent all the individual underlying
/// non-aggregate types that comprise it.
///
/// If Offsets is non-null, it points to a vector to be filled in
/// with the in-memory offsets of each of the individual values.
///
void llvm::ComputeValueVTs(const TargetLowering &TLI, const Type *Ty,
                           SmallVectorImpl<EVT> &ValueVTs,
                           SmallVectorImpl<uint64_t> *Offsets,
                           uint64_t StartingOffset) {
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
    uint64_t EltSize = TLI.getTargetData()->getTypeAllocSize(EltTy);
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
      ComputeValueVTs(TLI, EltTy, ValueVTs, Offsets,
                      StartingOffset + i * EltSize);
    return;
  }
  // Interpret void as zero return values.
  if (Ty->isVoidTy())
    return;
  // Base case: we can get an EVT for this LLVM IR type.
  ValueVTs.push_back(TLI.getValueType(Ty));
  if (Offsets)
    Offsets->push_back(StartingOffset);
}

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it, or used by a
/// switch or atomic instruction, which may expand to multiple basic blocks.
static bool isUsedOutsideOfDefiningBlock(Instruction *I) {
  if (isa<PHINode>(I)) return true;
  BasicBlock *BB = I->getParent();
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != BB || isa<PHINode>(*UI))
      return true;
  return false;
}

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.  This includes arguments used by switches, since
/// the switch may expand into multiple basic blocks.
static bool isOnlyUsedInEntryBlock(Argument *A, bool EnableFastISel) {
  // With FastISel active, we may be splitting blocks, so force creation
  // of virtual registers for all non-dead arguments.
  // Don't force virtual registers for byval arguments though, because
  // fast-isel can't handle those in all cases.
  if (EnableFastISel && !A->hasByValAttr())
    return A->use_empty();

  BasicBlock *Entry = A->getParent()->begin();
  for (Value::use_iterator UI = A->use_begin(), E = A->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != Entry || isa<SwitchInst>(*UI))
      return false;  // Use not in entry block.
  return true;
}

FunctionLoweringInfo::FunctionLoweringInfo(TargetLowering &tli)
  : TLI(tli) {
}

void FunctionLoweringInfo::set(Function &fn, MachineFunction &mf,
                               bool EnableFastISel) {
  Fn = &fn;
  MF = &mf;
  RegInfo = &MF->getRegInfo();

  // Create a vreg for each argument register that is not dead and is used
  // outside of the entry block for the function.
  for (Function::arg_iterator AI = Fn->arg_begin(), E = Fn->arg_end();
       AI != E; ++AI)
    if (!isOnlyUsedInEntryBlock(AI, EnableFastISel))
      InitializeRegForValue(AI);

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::iterator BB = Fn->begin(), EB = Fn->end();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (ConstantInt *CUI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        const Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getTargetData()->getTypeAllocSize(Ty);
        unsigned Align =
          std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty),
                   AI->getAlignment());

        TySize *= CUI->getZExtValue();   // Get total allocated size.
        if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.
        StaticAllocaMap[AI] =
          MF->getFrameInfo()->CreateStackObject(TySize, Align, false);
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
  for (BB = Fn->begin(); BB != EB; ++BB) {
    MachineBasicBlock *MBB = mf.CreateMachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF->push_back(MBB);

    // Transfer the address-taken flag. This is necessary because there could
    // be multiple MachineBasicBlocks corresponding to one BasicBlock, and only
    // the first one should be marked.
    if (BB->hasAddressTaken())
      MBB->setHasAddressTaken();

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    PHINode *PN;
    DebugLoc DL;
    for (BasicBlock::iterator
           I = BB->begin(), E = BB->end(); I != E; ++I) {

      PN = dyn_cast<PHINode>(I);
      if (!PN || PN->use_empty()) continue;

      unsigned PHIReg = ValueMap[PN];
      assert(PHIReg && "PHI node does not have an assigned virtual register!");

      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(TLI, PN->getType(), ValueVTs);
      for (unsigned vti = 0, vte = ValueVTs.size(); vti != vte; ++vti) {
        EVT VT = ValueVTs[vti];
        unsigned NumRegisters = TLI.getNumRegisters(Fn->getContext(), VT);
        const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
        for (unsigned i = 0; i != NumRegisters; ++i)
          BuildMI(MBB, DL, TII->get(TargetOpcode::PHI), PHIReg + i);
        PHIReg += NumRegisters;
      }
    }
  }

  // Mark landing pad blocks.
  for (BB = Fn->begin(); BB != EB; ++BB)
    if (InvokeInst *Invoke = dyn_cast<InvokeInst>(BB->getTerminator()))
      MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();
}

/// clear - Clear out all the function-specific state. This returns this
/// FunctionLoweringInfo to an empty state, ready to be used for a
/// different function.
void FunctionLoweringInfo::clear() {
  assert(CatchInfoFound.size() == CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");

  MBBMap.clear();
  ValueMap.clear();
  StaticAllocaMap.clear();
#ifndef NDEBUG
  CatchInfoLost.clear();
  CatchInfoFound.clear();
#endif
  LiveOutRegInfo.clear();
}

unsigned FunctionLoweringInfo::MakeReg(EVT VT) {
  return RegInfo->createVirtualRegister(TLI.getRegClassFor(VT));
}

/// CreateRegForValue - Allocate the appropriate number of virtual registers of
/// the correctly promoted or expanded types.  Assign these registers
/// consecutive vreg numbers and return the first assigned number.
///
/// In the case that the given value has struct or array type, this function
/// will assign registers for each member or element.
///
unsigned FunctionLoweringInfo::CreateRegForValue(const Value *V) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, V->getType(), ValueVTs);

  unsigned FirstReg = 0;
  for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    EVT RegisterVT = TLI.getRegisterType(V->getContext(), ValueVT);

    unsigned NumRegs = TLI.getNumRegisters(V->getContext(), ValueVT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      unsigned R = MakeReg(RegisterVT);
      if (!FirstReg) FirstReg = R;
    }
  }
  return FirstReg;
}

/// ExtractTypeInfo - Returns the type info, possibly bitcast, encoded in V.
GlobalVariable *llvm::ExtractTypeInfo(Value *V) {
  V = V->stripPointerCasts();
  GlobalVariable *GV = dyn_cast<GlobalVariable>(V);

  if (GV && GV->getName() == ".llvm.eh.catch.all.value") {
    assert(GV->hasInitializer() &&
           "The EH catch-all value must have an initializer");
    Value *Init = GV->getInitializer();
    GV = dyn_cast<GlobalVariable>(Init);
    if (!GV) V = cast<ConstantPointerNull>(Init);
  }

  assert((GV || isa<ConstantPointerNull>(V)) &&
         "TypeInfo must be a global variable or NULL");
  return GV;
}

/// AddCatchInfo - Extract the personality and type infos from an eh.selector
/// call, and add them to the specified machine basic block.
void llvm::AddCatchInfo(const CallInst &I, MachineModuleInfo *MMI,
                        MachineBasicBlock *MBB) {
  // Inform the MachineModuleInfo of the personality for this landing pad.
  const ConstantExpr *CE = cast<ConstantExpr>(I.getOperand(2));
  assert(CE->getOpcode() == Instruction::BitCast &&
         isa<Function>(CE->getOperand(0)) &&
         "Personality should be a function");
  MMI->addPersonality(MBB, cast<Function>(CE->getOperand(0)));

  // Gather all the type infos for this landing pad and pass them along to
  // MachineModuleInfo.
  std::vector<const GlobalVariable *> TyInfo;
  unsigned N = I.getNumOperands();

  for (unsigned i = N - 1; i > 2; --i) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(i))) {
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

void llvm::CopyCatchInfo(const BasicBlock *SrcBB, const BasicBlock *DestBB,
                         MachineModuleInfo *MMI, FunctionLoweringInfo &FLI) {
  for (BasicBlock::const_iterator I = SrcBB->begin(), E = --SrcBB->end();
       I != E; ++I)
    if (const EHSelectorInst *EHSel = dyn_cast<EHSelectorInst>(I)) {
      // Apply the catch info to DestBB.
      AddCatchInfo(*EHSel, MMI, FLI.MBBMap[DestBB]);
#ifndef NDEBUG
      if (!FLI.MBBMap[SrcBB]->isLandingPad())
        FLI.CatchInfoFound.insert(EHSel);
#endif
    }
}

/// hasInlineAsmMemConstraint - Return true if the inline asm instruction being
/// processed uses a memory 'm' constraint.
bool
llvm::hasInlineAsmMemConstraint(std::vector<InlineAsm::ConstraintInfo> &CInfos,
                                const TargetLowering &TLI) {
  for (unsigned i = 0, e = CInfos.size(); i != e; ++i) {
    InlineAsm::ConstraintInfo &CI = CInfos[i];
    for (unsigned j = 0, ee = CI.Codes.size(); j != ee; ++j) {
      TargetLowering::ConstraintType CType = TLI.getConstraintType(CI.Codes[j]);
      if (CType == TargetLowering::C_Memory)
        return true;
    }

    // Indirect operand accesses access memory.
    if (CI.isIndirect)
      return true;
  }

  return false;
}

/// getFCmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR floating-point condition code.  This includes
/// consideration of global floating-point math flags.
///
ISD::CondCode llvm::getFCmpCondCode(FCmpInst::Predicate Pred) {
  ISD::CondCode FPC, FOC;
  switch (Pred) {
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
    llvm_unreachable("Invalid FCmp predicate opcode!");
    FOC = FPC = ISD::SETFALSE;
    break;
  }
  if (FiniteOnlyFPMath())
    return FOC;
  else
    return FPC;
}

/// getICmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR integer condition code.
///
ISD::CondCode llvm::getICmpCondCode(ICmpInst::Predicate Pred) {
  switch (Pred) {
  case ICmpInst::ICMP_EQ:  return ISD::SETEQ;
  case ICmpInst::ICMP_NE:  return ISD::SETNE;
  case ICmpInst::ICMP_SLE: return ISD::SETLE;
  case ICmpInst::ICMP_ULE: return ISD::SETULE;
  case ICmpInst::ICMP_SGE: return ISD::SETGE;
  case ICmpInst::ICMP_UGE: return ISD::SETUGE;
  case ICmpInst::ICMP_SLT: return ISD::SETLT;
  case ICmpInst::ICMP_ULT: return ISD::SETULT;
  case ICmpInst::ICMP_SGT: return ISD::SETGT;
  case ICmpInst::ICMP_UGT: return ISD::SETUGT;
  default:
    llvm_unreachable("Invalid ICmp predicate opcode!");
    return ISD::SETNE;
  }
}
