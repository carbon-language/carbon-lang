//===-- SPIRVPreLegalizer.cpp - prepare IR for legalization -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass prepares IR for legalization: it assigns SPIR-V types to registers
// and removes intrinsics which holded these types during IR translation.
// Also it processes constants and registers them in GR to avoid duplication.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Target/TargetIntrinsicInfo.h"

#define DEBUG_TYPE "spirv-prelegalizer"

using namespace llvm;

namespace {
class SPIRVPreLegalizer : public MachineFunctionPass {
public:
  static char ID;
  SPIRVPreLegalizer() : MachineFunctionPass(ID) {
    initializeSPIRVPreLegalizerPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

static bool isSpvIntrinsic(MachineInstr &MI, Intrinsic::ID IntrinsicID) {
  if (MI.getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS &&
      MI.getIntrinsicID() == IntrinsicID)
    return true;
  return false;
}

static void foldConstantsIntoIntrinsics(MachineFunction &MF) {
  SmallVector<MachineInstr *, 10> ToErase;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned AssignNameOperandShift = 2;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_assign_name))
        continue;
      unsigned NumOp = MI.getNumExplicitDefs() + AssignNameOperandShift;
      while (MI.getOperand(NumOp).isReg()) {
        MachineOperand &MOp = MI.getOperand(NumOp);
        MachineInstr *ConstMI = MRI.getVRegDef(MOp.getReg());
        assert(ConstMI->getOpcode() == TargetOpcode::G_CONSTANT);
        MI.removeOperand(NumOp);
        MI.addOperand(MachineOperand::CreateImm(
            ConstMI->getOperand(1).getCImm()->getZExtValue()));
        if (MRI.use_empty(ConstMI->getOperand(0).getReg()))
          ToErase.push_back(ConstMI);
      }
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
}

static void insertBitcasts(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                           MachineIRBuilder MIB) {
  SmallVector<MachineInstr *, 10> ToErase;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_bitcast))
        continue;
      assert(MI.getOperand(2).isReg());
      MIB.setInsertPt(*MI.getParent(), MI);
      MIB.buildBitcast(MI.getOperand(0).getReg(), MI.getOperand(2).getReg());
      ToErase.push_back(&MI);
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
}

// Translating GV, IRTranslator sometimes generates following IR:
//   %1 = G_GLOBAL_VALUE
//   %2 = COPY %1
//   %3 = G_ADDRSPACE_CAST %2
// New registers have no SPIRVType and no register class info.
//
// Set SPIRVType for GV, propagate it from GV to other instructions,
// also set register classes.
static SPIRVType *propagateSPIRVType(MachineInstr *MI, SPIRVGlobalRegistry *GR,
                                     MachineRegisterInfo &MRI,
                                     MachineIRBuilder &MIB) {
  SPIRVType *SpirvTy = nullptr;
  assert(MI && "Machine instr is expected");
  if (MI->getOperand(0).isReg()) {
    Register Reg = MI->getOperand(0).getReg();
    SpirvTy = GR->getSPIRVTypeForVReg(Reg);
    if (!SpirvTy) {
      switch (MI->getOpcode()) {
      case TargetOpcode::G_CONSTANT: {
        MIB.setInsertPt(*MI->getParent(), MI);
        Type *Ty = MI->getOperand(1).getCImm()->getType();
        SpirvTy = GR->getOrCreateSPIRVType(Ty, MIB);
        break;
      }
      case TargetOpcode::G_GLOBAL_VALUE: {
        MIB.setInsertPt(*MI->getParent(), MI);
        Type *Ty = MI->getOperand(1).getGlobal()->getType();
        SpirvTy = GR->getOrCreateSPIRVType(Ty, MIB);
        break;
      }
      case TargetOpcode::G_TRUNC:
      case TargetOpcode::G_ADDRSPACE_CAST:
      case TargetOpcode::COPY: {
        MachineOperand &Op = MI->getOperand(1);
        MachineInstr *Def = Op.isReg() ? MRI.getVRegDef(Op.getReg()) : nullptr;
        if (Def)
          SpirvTy = propagateSPIRVType(Def, GR, MRI, MIB);
        break;
      }
      default:
        break;
      }
      if (SpirvTy)
        GR->assignSPIRVTypeToVReg(SpirvTy, Reg, MIB.getMF());
      if (!MRI.getRegClassOrNull(Reg))
        MRI.setRegClass(Reg, &SPIRV::IDRegClass);
    }
  }
  return SpirvTy;
}

// Insert ASSIGN_TYPE instuction between Reg and its definition, set NewReg as
// a dst of the definition, assign SPIRVType to both registers. If SpirvTy is
// provided, use it as SPIRVType in ASSIGN_TYPE, otherwise create it from Ty.
// TODO: maybe move to SPIRVUtils.
static Register insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                                  SPIRVGlobalRegistry *GR,
                                  MachineIRBuilder &MIB,
                                  MachineRegisterInfo &MRI) {
  MachineInstr *Def = MRI.getVRegDef(Reg);
  assert((Ty || SpirvTy) && "Either LLVM or SPIRV type is expected.");
  MIB.setInsertPt(*Def->getParent(),
                  (Def->getNextNode() ? Def->getNextNode()->getIterator()
                                      : Def->getParent()->end()));
  Register NewReg = MRI.createGenericVirtualRegister(MRI.getType(Reg));
  if (auto *RC = MRI.getRegClassOrNull(Reg))
    MRI.setRegClass(NewReg, RC);
  SpirvTy = SpirvTy ? SpirvTy : GR->getOrCreateSPIRVType(Ty, MIB);
  GR->assignSPIRVTypeToVReg(SpirvTy, Reg, MIB.getMF());
  // This is to make it convenient for Legalizer to get the SPIRVType
  // when processing the actual MI (i.e. not pseudo one).
  GR->assignSPIRVTypeToVReg(SpirvTy, NewReg, MIB.getMF());
  MIB.buildInstr(SPIRV::ASSIGN_TYPE)
      .addDef(Reg)
      .addUse(NewReg)
      .addUse(GR->getSPIRVTypeID(SpirvTy));
  Def->getOperand(0).setReg(NewReg);
  MRI.setRegClass(Reg, &SPIRV::ANYIDRegClass);
  return NewReg;
}

static void generateAssignInstrs(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                                 MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 10> ToErase;

  for (MachineBasicBlock *MBB : post_order(&MF)) {
    if (MBB->empty())
      continue;

    bool ReachedBegin = false;
    for (auto MII = std::prev(MBB->end()), Begin = MBB->begin();
         !ReachedBegin;) {
      MachineInstr &MI = *MII;

      if (isSpvIntrinsic(MI, Intrinsic::spv_assign_type)) {
        Register Reg = MI.getOperand(1).getReg();
        Type *Ty = getMDOperandAsType(MI.getOperand(2).getMetadata(), 0);
        MachineInstr *Def = MRI.getVRegDef(Reg);
        assert(Def && "Expecting an instruction that defines the register");
        // G_GLOBAL_VALUE already has type info.
        if (Def->getOpcode() != TargetOpcode::G_GLOBAL_VALUE)
          insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MF.getRegInfo());
        ToErase.push_back(&MI);
      } else if (MI.getOpcode() == TargetOpcode::G_CONSTANT ||
                 MI.getOpcode() == TargetOpcode::G_FCONSTANT ||
                 MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
        // %rc = G_CONSTANT ty Val
        // ===>
        // %cty = OpType* ty
        // %rctmp = G_CONSTANT ty Val
        // %rc = ASSIGN_TYPE %rctmp, %cty
        Register Reg = MI.getOperand(0).getReg();
        if (MRI.hasOneUse(Reg)) {
          MachineInstr &UseMI = *MRI.use_instr_begin(Reg);
          if (isSpvIntrinsic(UseMI, Intrinsic::spv_assign_type) ||
              isSpvIntrinsic(UseMI, Intrinsic::spv_assign_name))
            continue;
        }
        Type *Ty = nullptr;
        if (MI.getOpcode() == TargetOpcode::G_CONSTANT)
          Ty = MI.getOperand(1).getCImm()->getType();
        else if (MI.getOpcode() == TargetOpcode::G_FCONSTANT)
          Ty = MI.getOperand(1).getFPImm()->getType();
        else {
          assert(MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR);
          Type *ElemTy = nullptr;
          MachineInstr *ElemMI = MRI.getVRegDef(MI.getOperand(1).getReg());
          assert(ElemMI);

          if (ElemMI->getOpcode() == TargetOpcode::G_CONSTANT)
            ElemTy = ElemMI->getOperand(1).getCImm()->getType();
          else if (ElemMI->getOpcode() == TargetOpcode::G_FCONSTANT)
            ElemTy = ElemMI->getOperand(1).getFPImm()->getType();
          else
            llvm_unreachable("Unexpected opcode");
          unsigned NumElts =
              MI.getNumExplicitOperands() - MI.getNumExplicitDefs();
          Ty = VectorType::get(ElemTy, NumElts, false);
        }
        insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MRI);
      } else if (MI.getOpcode() == TargetOpcode::G_TRUNC ||
                 MI.getOpcode() == TargetOpcode::G_GLOBAL_VALUE ||
                 MI.getOpcode() == TargetOpcode::COPY ||
                 MI.getOpcode() == TargetOpcode::G_ADDRSPACE_CAST) {
        propagateSPIRVType(&MI, GR, MRI, MIB);
      }

      if (MII == Begin)
        ReachedBegin = true;
      else
        --MII;
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
}

static std::pair<Register, unsigned>
createNewIdReg(Register ValReg, unsigned Opcode, MachineRegisterInfo &MRI,
               const SPIRVGlobalRegistry &GR) {
  LLT NewT = LLT::scalar(32);
  SPIRVType *SpvType = GR.getSPIRVTypeForVReg(ValReg);
  assert(SpvType && "VReg is expected to have SPIRV type");
  bool IsFloat = SpvType->getOpcode() == SPIRV::OpTypeFloat;
  bool IsVectorFloat =
      SpvType->getOpcode() == SPIRV::OpTypeVector &&
      GR.getSPIRVTypeForVReg(SpvType->getOperand(1).getReg())->getOpcode() ==
          SPIRV::OpTypeFloat;
  IsFloat |= IsVectorFloat;
  auto GetIdOp = IsFloat ? SPIRV::GET_fID : SPIRV::GET_ID;
  auto DstClass = IsFloat ? &SPIRV::fIDRegClass : &SPIRV::IDRegClass;
  if (MRI.getType(ValReg).isPointer()) {
    NewT = LLT::pointer(0, 32);
    GetIdOp = SPIRV::GET_pID;
    DstClass = &SPIRV::pIDRegClass;
  } else if (MRI.getType(ValReg).isVector()) {
    NewT = LLT::fixed_vector(2, NewT);
    GetIdOp = IsFloat ? SPIRV::GET_vfID : SPIRV::GET_vID;
    DstClass = IsFloat ? &SPIRV::vfIDRegClass : &SPIRV::vIDRegClass;
  }
  Register IdReg = MRI.createGenericVirtualRegister(NewT);
  MRI.setRegClass(IdReg, DstClass);
  return {IdReg, GetIdOp};
}

static void processInstr(MachineInstr &MI, MachineIRBuilder &MIB,
                         MachineRegisterInfo &MRI, SPIRVGlobalRegistry *GR) {
  unsigned Opc = MI.getOpcode();
  assert(MI.getNumDefs() > 0 && MRI.hasOneUse(MI.getOperand(0).getReg()));
  MachineInstr &AssignTypeInst =
      *(MRI.use_instr_begin(MI.getOperand(0).getReg()));
  auto NewReg = createNewIdReg(MI.getOperand(0).getReg(), Opc, MRI, *GR).first;
  AssignTypeInst.getOperand(1).setReg(NewReg);
  MI.getOperand(0).setReg(NewReg);
  MIB.setInsertPt(*MI.getParent(),
                  (MI.getNextNode() ? MI.getNextNode()->getIterator()
                                    : MI.getParent()->end()));
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || Op.isDef())
      continue;
    auto IdOpInfo = createNewIdReg(Op.getReg(), Opc, MRI, *GR);
    MIB.buildInstr(IdOpInfo.second).addDef(IdOpInfo.first).addUse(Op.getReg());
    Op.setReg(IdOpInfo.first);
  }
}

// Defined in SPIRVLegalizerInfo.cpp.
extern bool isTypeFoldingSupported(unsigned Opcode);

static void processInstrsWithTypeFolding(MachineFunction &MF,
                                         SPIRVGlobalRegistry *GR,
                                         MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isTypeFoldingSupported(MI.getOpcode()))
        processInstr(MI, MIB, MRI, GR);
    }
  }
}

static void processSwitches(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                            MachineIRBuilder MIB) {
  DenseMap<Register, SmallDenseMap<uint64_t, MachineBasicBlock *>>
      SwitchRegToMBB;
  DenseMap<Register, MachineBasicBlock *> DefaultMBBs;
  DenseSet<Register> SwitchRegs;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  // Before IRTranslator pass, spv_switch calls are inserted before each
  // switch instruction. IRTranslator lowers switches to ICMP+CBr+Br triples.
  // A switch with two cases may be translated to this MIR sequesnce:
  //   intrinsic(@llvm.spv.switch), %CmpReg, %Const0, %Const1
  //   %Dst0 = G_ICMP intpred(eq), %CmpReg, %Const0
  //   G_BRCOND %Dst0, %bb.2
  //   G_BR %bb.5
  // bb.5.entry:
  //   %Dst1 = G_ICMP intpred(eq), %CmpReg, %Const1
  //   G_BRCOND %Dst1, %bb.3
  //   G_BR %bb.4
  // bb.2.sw.bb:
  //   ...
  // bb.3.sw.bb1:
  //   ...
  // bb.4.sw.epilog:
  //   ...
  // Walk MIs and collect information about destination MBBs to update
  // spv_switch call. We assume that all spv_switch precede corresponding ICMPs.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isSpvIntrinsic(MI, Intrinsic::spv_switch)) {
        assert(MI.getOperand(1).isReg());
        Register Reg = MI.getOperand(1).getReg();
        SwitchRegs.insert(Reg);
        // Set the first successor as default MBB to support empty switches.
        DefaultMBBs[Reg] = *MBB.succ_begin();
      }
      // Process only ICMPs that relate to spv_switches.
      if (MI.getOpcode() == TargetOpcode::G_ICMP && MI.getOperand(2).isReg() &&
          SwitchRegs.contains(MI.getOperand(2).getReg())) {
        assert(MI.getOperand(0).isReg() && MI.getOperand(1).isPredicate() &&
               MI.getOperand(3).isReg());
        Register Dst = MI.getOperand(0).getReg();
        // Set type info for destination register of switch's ICMP instruction.
        if (GR->getSPIRVTypeForVReg(Dst) == nullptr) {
          MIB.setInsertPt(*MI.getParent(), MI);
          Type *LLVMTy = IntegerType::get(MF.getFunction().getContext(), 1);
          SPIRVType *SpirvTy = GR->getOrCreateSPIRVType(LLVMTy, MIB);
          MRI.setRegClass(Dst, &SPIRV::IDRegClass);
          GR->assignSPIRVTypeToVReg(SpirvTy, Dst, MIB.getMF());
        }
        Register CmpReg = MI.getOperand(2).getReg();
        MachineOperand &PredOp = MI.getOperand(1);
        const auto CC = static_cast<CmpInst::Predicate>(PredOp.getPredicate());
        assert(CC == CmpInst::ICMP_EQ && MRI.hasOneUse(Dst) &&
               MRI.hasOneDef(CmpReg));
        uint64_t Val = getIConstVal(MI.getOperand(3).getReg(), &MRI);
        MachineInstr *CBr = MRI.use_begin(Dst)->getParent();
        assert(CBr->getOpcode() == SPIRV::G_BRCOND &&
               CBr->getOperand(1).isMBB());
        SwitchRegToMBB[CmpReg][Val] = CBr->getOperand(1).getMBB();
        // The next MI is always BR to either the next case or the default.
        MachineInstr *NextMI = CBr->getNextNode();
        assert(NextMI->getOpcode() == SPIRV::G_BR &&
               NextMI->getOperand(0).isMBB());
        MachineBasicBlock *NextMBB = NextMI->getOperand(0).getMBB();
        assert(NextMBB != nullptr);
        // The default MBB is not started by ICMP with switch's cmp register.
        if (NextMBB->front().getOpcode() != SPIRV::G_ICMP ||
            (NextMBB->front().getOperand(2).isReg() &&
             NextMBB->front().getOperand(2).getReg() != CmpReg))
          DefaultMBBs[CmpReg] = NextMBB;
      }
    }
  }
  // Modify spv_switch's operands by collected values. For the example above,
  // the result will be like this:
  //   intrinsic(@llvm.spv.switch), %CmpReg, %bb.4, i32 0, %bb.2, i32 1, %bb.3
  // Note that ICMP+CBr+Br sequences are not removed, but ModuleAnalysis marks
  // them as skipped and AsmPrinter does not output them.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_switch))
        continue;
      assert(MI.getOperand(1).isReg());
      Register Reg = MI.getOperand(1).getReg();
      unsigned NumOp = MI.getNumExplicitOperands();
      SmallVector<const ConstantInt *, 3> Vals;
      SmallVector<MachineBasicBlock *, 3> MBBs;
      for (unsigned i = 2; i < NumOp; i++) {
        Register CReg = MI.getOperand(i).getReg();
        uint64_t Val = getIConstVal(CReg, &MRI);
        MachineInstr *ConstInstr = getDefInstrMaybeConstant(CReg, &MRI);
        Vals.push_back(ConstInstr->getOperand(1).getCImm());
        MBBs.push_back(SwitchRegToMBB[Reg][Val]);
      }
      for (unsigned i = MI.getNumExplicitOperands() - 1; i > 1; i--)
        MI.removeOperand(i);
      MI.addOperand(MachineOperand::CreateMBB(DefaultMBBs[Reg]));
      for (unsigned i = 0; i < Vals.size(); i++) {
        MI.addOperand(MachineOperand::CreateCImm(Vals[i]));
        MI.addOperand(MachineOperand::CreateMBB(MBBs[i]));
      }
    }
  }
}

bool SPIRVPreLegalizer::runOnMachineFunction(MachineFunction &MF) {
  // Initialize the type registry.
  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  GR->setCurrentFunc(MF);
  MachineIRBuilder MIB(MF);
  foldConstantsIntoIntrinsics(MF);
  insertBitcasts(MF, GR, MIB);
  generateAssignInstrs(MF, GR, MIB);
  processInstrsWithTypeFolding(MF, GR, MIB);
  processSwitches(MF, GR, MIB);

  return true;
}

INITIALIZE_PASS(SPIRVPreLegalizer, DEBUG_TYPE, "SPIRV pre legalizer", false,
                false)

char SPIRVPreLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPreLegalizerPass() {
  return new SPIRVPreLegalizer();
}
