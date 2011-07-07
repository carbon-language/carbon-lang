//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86MCAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

static MCAsmInfo *createMCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO) {
    if (TheTriple.getArch() == Triple::x86_64)
      return new X86_64MCAsmInfoDarwin(TheTriple);
    else
      return new X86MCAsmInfoDarwin(TheTriple);
  }

  if (TheTriple.isOSWindows())
    return new X86MCAsmInfoCOFF(TheTriple);

  return new X86ELFMCAsmInfo(TheTriple);
}

static MCStreamer *createMCStreamer(const Target &T, const std::string &TT,
                                    MCContext &Ctx, TargetAsmBackend &TAB,
                                    raw_ostream &_OS,
                                    MCCodeEmitter *_Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO)
    return createMachOStreamer(Ctx, TAB, _OS, _Emitter, RelaxAll);

  if (TheTriple.isOSWindows())
    return createWinCOFFStreamer(Ctx, TAB, *_Emitter, _OS, RelaxAll);

  return createELFStreamer(Ctx, TAB, _OS, _Emitter, RelaxAll, NoExecStack);
}

extern "C" void LLVMInitializeX86Target() {
  // Register the target.
  RegisterTargetMachine<X86_32TargetMachine> X(TheX86_32Target);
  RegisterTargetMachine<X86_64TargetMachine> Y(TheX86_64Target);

  // Register the target asm info.
  RegisterAsmInfoFn A(TheX86_32Target, createMCAsmInfo);
  RegisterAsmInfoFn B(TheX86_64Target, createMCAsmInfo);

  // Register the code emitter.
  TargetRegistry::RegisterCodeEmitter(TheX86_32Target,
                                      createX86_32MCCodeEmitter);
  TargetRegistry::RegisterCodeEmitter(TheX86_64Target,
                                      createX86_64MCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterAsmBackend(TheX86_32Target,
                                     createX86_32AsmBackend);
  TargetRegistry::RegisterAsmBackend(TheX86_64Target,
                                     createX86_64AsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterObjectStreamer(TheX86_32Target,
                                         createMCStreamer);
  TargetRegistry::RegisterObjectStreamer(TheX86_64Target,
                                         createMCStreamer);
}


X86_32TargetMachine::X86_32TargetMachine(const Target &T, const std::string &TT,
                                         const std::string &CPU,
                                         const std::string &FS)
  : X86TargetMachine(T, TT, CPU, FS, false),
    DataLayout(getSubtargetImpl()->isTargetDarwin() ?
               "e-p:32:32-f64:32:64-i64:32:64-f80:128:128-f128:128:128-n8:16:32" :
               (getSubtargetImpl()->isTargetCygMing() ||
                getSubtargetImpl()->isTargetWindows()) ?
               "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-f128:128:128-n8:16:32" :
               "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-f128:128:128-n8:16:32"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this) {
}


X86_64TargetMachine::X86_64TargetMachine(const Target &T, const std::string &TT,
                                         const std::string &CPU, 
                                         const std::string &FS)
  : X86TargetMachine(T, TT, CPU, FS, true),
    DataLayout("e-p:64:64-s:64-f64:64:64-i64:64:64-f80:128:128-f128:128:128-n8:16:32:64"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this) {
}

/// X86TargetMachine ctor - Create an X86 target.
///
X86TargetMachine::X86TargetMachine(const Target &T, const std::string &TT,
                                   const std::string &CPU,
                                   const std::string &FS, bool is64Bit)
  : LLVMTargetMachine(T, TT),
    Subtarget(TT, CPU, FS, StackAlignmentOverride),
    FrameLowering(*this, Subtarget),
    ELFWriterInfo(is64Bit, true) {
  DefRelocModel = getRelocationModel();

  // If no relocation model was picked, default as appropriate for the target.
  if (getRelocationModel() == Reloc::Default) {
    // Darwin defaults to PIC in 64 bit mode and dynamic-no-pic in 32 bit mode.
    // Win64 requires rip-rel addressing, thus we force it to PIC. Otherwise we
    // use static relocation model by default.
    if (Subtarget.isTargetDarwin()) {
      if (Subtarget.is64Bit())
        setRelocationModel(Reloc::PIC_);
      else
        setRelocationModel(Reloc::DynamicNoPIC);
    } else if (Subtarget.isTargetWin64())
      setRelocationModel(Reloc::PIC_);
    else
      setRelocationModel(Reloc::Static);
  }

  assert(getRelocationModel() != Reloc::Default &&
         "Relocation mode not picked");

  // ELF and X86-64 don't have a distinct DynamicNoPIC model.  DynamicNoPIC
  // is defined as a model for code which may be used in static or dynamic
  // executables but not necessarily a shared library. On X86-32 we just
  // compile in -static mode, in x86-64 we use PIC.
  if (getRelocationModel() == Reloc::DynamicNoPIC) {
    if (is64Bit)
      setRelocationModel(Reloc::PIC_);
    else if (!Subtarget.isTargetDarwin())
      setRelocationModel(Reloc::Static);
  }

  // If we are on Darwin, disallow static relocation model in X86-64 mode, since
  // the Mach-O file format doesn't support it.
  if (getRelocationModel() == Reloc::Static &&
      Subtarget.isTargetDarwin() &&
      is64Bit)
    setRelocationModel(Reloc::PIC_);

  // Determine the PICStyle based on the target selected.
  if (getRelocationModel() == Reloc::Static) {
    // Unless we're in PIC or DynamicNoPIC mode, set the PIC style to None.
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.is64Bit()) {
    // PIC in 64 bit mode is always rip-rel.
    Subtarget.setPICStyle(PICStyles::RIPRel);
  } else if (Subtarget.isTargetCygMing()) {
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.isTargetDarwin()) {
    if (getRelocationModel() == Reloc::PIC_)
      Subtarget.setPICStyle(PICStyles::StubPIC);
    else {
      assert(getRelocationModel() == Reloc::DynamicNoPIC);
      Subtarget.setPICStyle(PICStyles::StubDynamicNoPIC);
    }
  } else if (Subtarget.isTargetELF()) {
    Subtarget.setPICStyle(PICStyles::GOT);
  }

  // Finally, if we have "none" as our PIC style, force to static mode.
  if (Subtarget.getPICStyle() == PICStyles::None)
    setRelocationModel(Reloc::Static);

  // default to hard float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Hard;    
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool X86TargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createX86ISelDag(*this, OptLevel));

  // For 32-bit, prepend instructions to set the "global base reg" for PIC.
  if (!Subtarget.is64Bit())
    PM.add(createGlobalBaseRegPass());

  return false;
}

bool X86TargetMachine::addPreRegAlloc(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  PM.add(createX86MaxStackAlignmentHeuristicPass());
  return false;  // -print-machineinstr shouldn't print after this.
}

bool X86TargetMachine::addPostRegAlloc(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  PM.add(createX86FloatingPointStackifierPass());
  return true;  // -print-machineinstr should print after this.
}

bool X86TargetMachine::addPreEmitPass(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  if (OptLevel != CodeGenOpt::None && Subtarget.hasSSE2()) {
    PM.add(createSSEDomainFixPass());
    return true;
  }
  return false;
}

bool X86TargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      JITCodeEmitter &JCE) {
  // FIXME: Move this to TargetJITInfo!
  // On Darwin, do not override 64-bit setting made in X86TargetMachine().
  if (DefRelocModel == Reloc::Default &&
      (!Subtarget.isTargetDarwin() || !Subtarget.is64Bit())) {
    setRelocationModel(Reloc::Static);
    Subtarget.setPICStyle(PICStyles::None);
  }


  PM.add(createX86JITCodeEmitterPass(*this, JCE));

  return false;
}

void X86TargetMachine::setCodeModelForStatic() {

    if (getCodeModel() != CodeModel::Default) return;

    // For static codegen, if we're not already set, use Small codegen.
    setCodeModel(CodeModel::Small);
}


void X86TargetMachine::setCodeModelForJIT() {

  if (getCodeModel() != CodeModel::Default) return;

  // 64-bit JIT places everything in the same buffer except external functions.
  if (Subtarget.is64Bit())
    setCodeModel(CodeModel::Large);
  else
    setCodeModel(CodeModel::Small);
}
