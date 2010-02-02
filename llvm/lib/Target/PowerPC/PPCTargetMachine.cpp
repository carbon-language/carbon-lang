//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCMCAsmInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

static const MCAsmInfo *createMCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);
  bool isPPC64 = TheTriple.getArch() == Triple::ppc64;
  if (TheTriple.getOS() == Triple::Darwin)
    return new PPCMCAsmInfoDarwin(isPPC64);
  return new PPCLinuxMCAsmInfo(isPPC64);
  
}

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);  
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
  
  RegisterAsmInfoFn C(ThePPC32Target, createMCAsmInfo);
  RegisterAsmInfoFn D(ThePPC64Target, createMCAsmInfo);
}


PPCTargetMachine::PPCTargetMachine(const Target &T, const std::string &TT,
                                   const std::string &FS, bool is64Bit)
  : LLVMTargetMachine(T, TT),
    Subtarget(TT, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameInfo(*this, is64Bit), JITInfo(*this, is64Bit), TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {

  if (getRelocationModel() == Reloc::Default) {
    if (Subtarget.isDarwin())
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::Static);
  }
}

/// Override this for PowerPC.  Tail merging happily breaks up instruction issue
/// groups, which typically degrades performance.
bool PPCTargetMachine::getEnableTailMergeDefault() const { return false; }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, const std::string &TT, 
                                       const std::string &FS) 
  : PPCTargetMachine(T, TT, FS, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Target &T, const std::string &TT, 
                                       const std::string &FS)
  : PPCTargetMachine(T, TT, FS, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PPCTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));
  return false;
}

bool PPCTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  // Must run branch selection immediately preceding the asm printer.
  PM.add(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      MachineCodeEmitter &MCE) {
  // The JIT should use the static relocation model in ppc32 mode, PIC in ppc64.
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64()) {
    // We use PIC codegen in ppc64 mode, because otherwise we'd have to use many
    // instructions to materialize arbitrary global variable + function +
    // constant pool addresses.
    setRelocationModel(Reloc::PIC_);
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    DisableJumpTables = true;      
  } else {
    setRelocationModel(Reloc::Static);
  }
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));

  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      JITCodeEmitter &JCE) {
  // The JIT should use the static relocation model in ppc32 mode, PIC in ppc64.
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64()) {
    // We use PIC codegen in ppc64 mode, because otherwise we'd have to use many
    // instructions to materialize arbitrary global variable + function +
    // constant pool addresses.
    setRelocationModel(Reloc::PIC_);
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    DisableJumpTables = true;      
  } else {
    setRelocationModel(Reloc::Static);
  }
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));

  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      ObjectCodeEmitter &OCE) {
  // The JIT should use the static relocation model in ppc32 mode, PIC in ppc64.
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64()) {
    // We use PIC codegen in ppc64 mode, because otherwise we'd have to use many
    // instructions to materialize arbitrary global variable + function +
    // constant pool addresses.
    setRelocationModel(Reloc::PIC_);
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    DisableJumpTables = true;      
  } else {
    setRelocationModel(Reloc::Static);
  }
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCObjectCodeEmitterPass(*this, OCE));

  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            MachineCodeEmitter &MCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));
  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            JITCodeEmitter &JCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));
  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            ObjectCodeEmitter &OCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCObjectCodeEmitterPass(*this, OCE));
  return false;
}

/// getLSDAEncoding - Returns the LSDA pointer encoding. The choices are 4-byte,
/// 8-byte, and target default. The CIE is hard-coded to indicate that the LSDA
/// pointer in the FDE section is an "sdata4", and should be encoded as a 4-byte
/// pointer by default. However, some systems may require a different size due
/// to bugs or other conditions. We will default to a 4-byte encoding unless the
/// system tells us otherwise.
///
/// The issue is when the CIE says their is an LSDA. That mandates that every
/// FDE have an LSDA slot. But if the function does not need an LSDA. There
/// needs to be some way to signify there is none. The LSDA is encoded as
/// pc-rel. But you don't look for some magic value after adding the pc. You
/// have to look for a zero before adding the pc. The problem is that the size
/// of the zero to look for depends on the encoding. The unwinder bug in SL is
/// that it always checks for a pointer-size zero. So on x86_64 it looks for 8
/// bytes of zero. If you have an LSDA, it works fine since the 8-bytes are
/// non-zero so it goes ahead and then reads the value based on the encoding.
/// But if you use sdata4 and there is no LSDA, then the test for zero gives a
/// false negative and the unwinder thinks there is an LSDA.
///
/// FIXME: This call-back isn't good! We should be using the correct encoding
/// regardless of the system. However, there are some systems which have bugs
/// that prevent this from occuring.
DwarfLSDAEncoding::Encoding PPCTargetMachine::getLSDAEncoding() const {
  if (Subtarget.isDarwin() && Subtarget.getDarwinVers() != 10)
    return DwarfLSDAEncoding::Default;

  return DwarfLSDAEncoding::EightByte;
}
