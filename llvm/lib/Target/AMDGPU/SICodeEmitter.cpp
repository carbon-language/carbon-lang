//===-- SICodeEmitter.cpp - SI Code Emitter -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The SI code emitter produces machine code that can be executed directly on
// the GPU device.
//
//===----------------------------------------------------------------------===//


#include "AMDGPU.h"
#include "AMDGPUUtil.h"
#include "AMDILCodeEmitter.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"

#include <stdio.h>

#define LITERAL_REG 255
#define VGPR_BIT(src_idx) (1ULL << (9 * src_idx - 1))
using namespace llvm;

namespace {

  class SICodeEmitter : public MachineFunctionPass, public AMDILCodeEmitter {

  private:
    static char ID;
    formatted_raw_ostream &_OS;
    const TargetMachine *TM;
    void emitState(MachineFunction & MF);
    void emitInstr(MachineInstr &MI);

    void outputBytes(uint64_t value, unsigned bytes);
    unsigned GPRAlign(const MachineInstr &MI, unsigned OpNo, unsigned shift)
                                                                      const;

  public:
    SICodeEmitter(formatted_raw_ostream &OS) : MachineFunctionPass(ID),
        _OS(OS), TM(NULL) { }
    const char *getPassName() const { return "SI Code Emitter"; }
    bool runOnMachineFunction(MachineFunction &MF);

    /// getMachineOpValue - Return the encoding for MO
    virtual uint64_t getMachineOpValue(const MachineInstr &MI,
                                       const MachineOperand &MO) const;

    /// GPR4AlignEncode - Encoding for when 4 consectuive registers are used 
    virtual unsigned GPR4AlignEncode(const MachineInstr  &MI, unsigned OpNo)
                                                                      const;

    /// GPR2AlignEncode - Encoding for when 2 consecutive registers are used
    virtual unsigned GPR2AlignEncode(const MachineInstr &MI, unsigned OpNo)
                                                                      const;
    /// i32LiteralEncode - Encode an i32 literal this is used as an operand
    /// for an instruction in place of a register.
    virtual uint64_t i32LiteralEncode(const MachineInstr &MI, unsigned OpNo)
                                                                      const;
    /// SMRDmemriEncode - Encoding for SMRD indexed loads
    virtual uint32_t SMRDmemriEncode(const MachineInstr &MI, unsigned OpNo)
                                                                     const;

    /// VOPPostEncode - Post-Encoder method for VOP instructions 
    virtual uint64_t VOPPostEncode(const MachineInstr &MI,
                                   uint64_t Value) const;
  };
}

char SICodeEmitter::ID = 0;

FunctionPass *llvm::createSICodeEmitterPass(formatted_raw_ostream &OS) {
  return new SICodeEmitter(OS);
}

void SICodeEmitter::emitState(MachineFunction & MF)
{
  unsigned maxSGPR = 0;
  unsigned maxVGPR = 0;
  bool VCCUsed = false;
  const SIRegisterInfo * RI =
                static_cast<const SIRegisterInfo*>(TM->getRegisterInfo());
  SIMachineFunctionInfo * MFI = MF.getInfo<SIMachineFunctionInfo>();

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E; ++I) {
      MachineInstr &MI = *I;
      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        MachineOperand & MO = MI.getOperand(op_idx);
        unsigned maxUsed;
        unsigned width = 0;
        bool isSGPR = false;
        unsigned reg;
        unsigned hwReg;
        if (!MO.isReg()) {
          continue;
        }
        reg = MO.getReg();
        if (reg == AMDGPU::VCC) {
          VCCUsed = true;
          continue;
        }
        if (AMDGPU::SReg_32RegClass.contains(reg)) {
          isSGPR = true;
          width = 1;
        } else if (AMDGPU::VReg_32RegClass.contains(reg)) {
          isSGPR = false;
          width = 1;
        } else if (AMDGPU::SReg_64RegClass.contains(reg)) {
          isSGPR = true;
          width = 2;
        } else if (AMDGPU::VReg_64RegClass.contains(reg)) {
          isSGPR = false;
          width = 2;
        } else if (AMDGPU::SReg_128RegClass.contains(reg)) {
          isSGPR = true;
          width = 4;
        } else if (AMDGPU::VReg_128RegClass.contains(reg)) {
          isSGPR = false;
          width = 4;
        } else if (AMDGPU::SReg_256RegClass.contains(reg)) {
          isSGPR = true;
          width = 8;
        } else {
          assert("!Unknown register class");
        }
        hwReg = RI->getEncodingValue(reg);
        maxUsed = ((hwReg + 1) * width) - 1;
        if (isSGPR) {
          maxSGPR = maxUsed > maxSGPR ? maxUsed : maxSGPR;
        } else {
          maxVGPR = maxUsed > maxVGPR ? maxUsed : maxVGPR;
        }
      }
    }
  }
  if (VCCUsed) {
    maxSGPR += 2;
  }
  outputBytes(maxSGPR + 1, 4);
  outputBytes(maxVGPR + 1, 4);
  outputBytes(MFI->spi_ps_input_addr, 4);
}

bool SICodeEmitter::runOnMachineFunction(MachineFunction &MF)
{
  TM = &MF.getTarget();
  const AMDILSubtarget &STM = TM->getSubtarget<AMDILSubtarget>();

  if (STM.dumpCode()) {
    MF.dump();
  }

  emitState(MF);

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.getOpcode() != AMDGPU::KILL && MI.getOpcode() != AMDGPU::RETURN) {
        emitInstr(MI);
      }
    }
  }
  // Emit S_END_PGM
  MachineInstr * End = BuildMI(MF, DebugLoc(),
                               TM->getInstrInfo()->get(AMDGPU::S_ENDPGM));
  emitInstr(*End);
  return false;
}

void SICodeEmitter::emitInstr(MachineInstr &MI)
{
  const SIInstrInfo * SII = static_cast<const SIInstrInfo*>(TM->getInstrInfo());

  uint64_t hwInst = getBinaryCodeForInstr(MI);

  if ((hwInst & 0xffffffff) == 0xffffffff) {
    fprintf(stderr, "Unsupported Instruction: \n");
    MI.dump();
    abort();
  }

  unsigned bytes = SII->getEncodingBytes(MI);
  outputBytes(hwInst, bytes);
}

uint64_t SICodeEmitter::getMachineOpValue(const MachineInstr &MI,
                                          const MachineOperand &MO) const
{
  const SIRegisterInfo * RI =
                static_cast<const SIRegisterInfo*>(TM->getRegisterInfo());

  switch(MO.getType()) {
  case MachineOperand::MO_Register:
    return RI->getEncodingValue(MO.getReg());

  case MachineOperand::MO_Immediate:
    return MO.getImm();

  case MachineOperand::MO_FPImmediate:
    // XXX: Not all instructions can use inline literals
    // XXX: We should make sure this is a 32-bit constant
    return LITERAL_REG | (MO.getFPImm()->getValueAPF().bitcastToAPInt().getZExtValue() << 32);
  default:
    llvm_unreachable("Encoding of this operand type is not supported yet.");
    break;
  }
}

unsigned SICodeEmitter::GPRAlign(const MachineInstr &MI, unsigned OpNo,
    unsigned shift) const
{
  const SIRegisterInfo * RI =
                static_cast<const SIRegisterInfo*>(TM->getRegisterInfo());
  unsigned regCode = RI->getEncodingValue(MI.getOperand(OpNo).getReg());
  return regCode >> shift;
}

unsigned SICodeEmitter::GPR4AlignEncode(const MachineInstr &MI,
    unsigned OpNo) const
{
  return GPRAlign(MI, OpNo, 2);
}

unsigned SICodeEmitter::GPR2AlignEncode(const MachineInstr &MI,
    unsigned OpNo) const
{
  return GPRAlign(MI, OpNo, 1);
}

uint64_t SICodeEmitter::i32LiteralEncode(const MachineInstr &MI,
    unsigned OpNo) const
{
  return LITERAL_REG | (MI.getOperand(OpNo).getImm() << 32);
}

#define SMRD_OFFSET_MASK 0xff
#define SMRD_IMM_SHIFT 8
#define SMRD_SBASE_MASK 0x3f
#define SMRD_SBASE_SHIFT 9
/// SMRDmemriEncode - This function is responsibe for encoding the offset
/// and the base ptr for SMRD instructions it should return a bit string in
/// this format:
///
/// OFFSET = bits{7-0}
/// IMM    = bits{8}
/// SBASE  = bits{14-9}
///
uint32_t SICodeEmitter::SMRDmemriEncode(const MachineInstr &MI,
    unsigned OpNo) const
{
  uint32_t encoding;

  const MachineOperand &OffsetOp = MI.getOperand(OpNo + 1);

  //XXX: Use this function for SMRD loads with register offsets
  assert(OffsetOp.isImm());

  encoding =
      (getMachineOpValue(MI, OffsetOp) & SMRD_OFFSET_MASK)
    | (1 << SMRD_IMM_SHIFT) //XXX If the Offset is a register we shouldn't set this bit
    | ((GPR2AlignEncode(MI, OpNo) & SMRD_SBASE_MASK) << SMRD_SBASE_SHIFT)
    ;

  return encoding;
}

/// Set the "VGPR" bit for VOP args that can take either a VGPR or a SGPR.
/// XXX: It would be nice if we could handle this without a PostEncode function.
uint64_t SICodeEmitter::VOPPostEncode(const MachineInstr &MI,
    uint64_t Value) const
{
  const SIInstrInfo * SII = static_cast<const SIInstrInfo*>(TM->getInstrInfo());
  unsigned encodingType = SII->getEncodingType(MI);
  unsigned numSrcOps;
  unsigned vgprBitOffset;

  if (encodingType == SIInstrEncodingType::VOP3) {
    numSrcOps = 3;
    vgprBitOffset = 32;
  } else {
    numSrcOps = 1;
    vgprBitOffset = 0;
  }

  // Add one to skip over the destination reg operand.
  for (unsigned opIdx = 1; opIdx < numSrcOps + 1; opIdx++) {
    if (!MI.getOperand(opIdx).isReg()) {
      continue;
    }
    unsigned reg = MI.getOperand(opIdx).getReg();
    if (AMDGPU::VReg_32RegClass.contains(reg)
        || AMDGPU::VReg_64RegClass.contains(reg)) {
      Value |= (VGPR_BIT(opIdx)) << vgprBitOffset;
    }
  }
  return Value;
}


void SICodeEmitter::outputBytes(uint64_t value, unsigned bytes)
{
  for (unsigned i = 0; i < bytes; i++) {
    _OS.write((uint8_t) ((value >> (8 * i)) & 0xff));
  }
}
