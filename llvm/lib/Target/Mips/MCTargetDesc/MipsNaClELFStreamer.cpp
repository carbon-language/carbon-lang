//===-- MipsNaClELFStreamer.cpp - ELF Object Output for Mips NaCl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements MCELFStreamer for Mips NaCl.  It emits .o object files
// as required by NaCl's SFI sandbox.  It inserts address-masking instructions
// before dangerous control-flow instructions.  It aligns on bundle size all
// functions and all targets of indirect branches.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-mc-nacl"

#include "Mips.h"
#include "MipsMCNaCl.h"
#include "llvm/MC/MCELFStreamer.h"

using namespace llvm;

namespace {

const unsigned IndirectBranchMaskReg = Mips::T6;

/// Extend the generic MCELFStreamer class so that it can mask dangerous
/// instructions.

class MipsNaClELFStreamer : public MCELFStreamer {
public:
  MipsNaClELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &OS,
                      MCCodeEmitter *Emitter)
    : MCELFStreamer(Context, TAB, OS, Emitter) {}

  ~MipsNaClELFStreamer() {}

private:
  bool isIndirectJump(const MCInst &MI) {
    return MI.getOpcode() == Mips::JR || MI.getOpcode() == Mips::RET;
  }

  void emitMask(unsigned AddrReg, unsigned MaskReg,
                const MCSubtargetInfo &STI) {
    MCInst MaskInst;
    MaskInst.setOpcode(Mips::AND);
    MaskInst.addOperand(MCOperand::CreateReg(AddrReg));
    MaskInst.addOperand(MCOperand::CreateReg(AddrReg));
    MaskInst.addOperand(MCOperand::CreateReg(MaskReg));
    MCELFStreamer::EmitInstruction(MaskInst, STI);
  }

  // Sandbox indirect branch or return instruction by inserting mask operation
  // before it.
  void sandboxIndirectJump(const MCInst &MI, const MCSubtargetInfo &STI) {
    unsigned AddrReg = MI.getOperand(0).getReg();

    EmitBundleLock(false);
    emitMask(AddrReg, IndirectBranchMaskReg, STI);
    MCELFStreamer::EmitInstruction(MI, STI);
    EmitBundleUnlock();
  }

public:
  /// This function is the one used to emit instruction data into the ELF
  /// streamer.  We override it to mask dangerous instructions.
  virtual void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) {
    if (isIndirectJump(Inst))
      sandboxIndirectJump(Inst, STI);
    else
      MCELFStreamer::EmitInstruction(Inst, STI);
  }
};

} // end anonymous namespace

namespace llvm {

MCELFStreamer *createMipsNaClELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                         raw_ostream &OS,
                                         MCCodeEmitter *Emitter, bool RelaxAll,
                                         bool NoExecStack) {
  MipsNaClELFStreamer *S = new MipsNaClELFStreamer(Context, TAB, OS, Emitter);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  if (NoExecStack)
    S->getAssembler().setNoExecStack(true);

  // Set bundle-alignment as required by the NaCl ABI for the target.
  S->EmitBundleAlignMode(MIPS_NACL_BUNDLE_ALIGN);

  return S;
}

}
