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
#include "PPCTargetAsmInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

/// PowerPCTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int PowerPCTargetMachineModule;
int PowerPCTargetMachineModule = 0;

// Register the targets
extern Target ThePPC32Target;
static RegisterTarget<PPC32TargetMachine>
X(ThePPC32Target, "ppc32", "PowerPC 32");

extern Target ThePPC64Target;
static RegisterTarget<PPC64TargetMachine>
Y(ThePPC64Target, "ppc64", "PowerPC 64");

// Force static initialization.
extern "C" void LLVMInitializePowerPCTarget() { }

const TargetAsmInfo *PPCTargetMachine::createTargetAsmInfo() const {
  if (Subtarget.isDarwin())
    return new PPCDarwinTargetAsmInfo(*this);
  else
    return new PPCLinuxTargetAsmInfo(*this);
}

PPCTargetMachine::PPCTargetMachine(const Target&T, const Module &M, 
                                   const std::string &FS, bool is64Bit)
  : LLVMTargetMachine(T),
    Subtarget(*this, M, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameInfo(*this, is64Bit), JITInfo(*this, is64Bit), TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()), MachOWriterInfo(*this) {

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

PPC32TargetMachine::PPC32TargetMachine(const Target &T, const Module &M, 
                                       const std::string &FS) 
  : PPCTargetMachine(T, M, FS, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Target &T, const Module &M, 
                                       const std::string &FS)
  : PPCTargetMachine(T, M, FS, true) {
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

bool PPCTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel,
                                          bool Verbose,
                                          formatted_raw_ostream &Out) {
  FunctionPass *Printer = getTarget().createAsmPrinter(Out, *this, Verbose);
  if (!Printer)
    llvm_report_error("unable to create assembly printer");
  PM.add(Printer);
  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      bool DumpAsm, MachineCodeEmitter &MCE) {
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
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      bool DumpAsm, JITCodeEmitter &JCE) {
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
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      bool DumpAsm, ObjectCodeEmitter &OCE) {
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
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool DumpAsm, 
                                            MachineCodeEmitter &MCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool DumpAsm,
                                            JITCodeEmitter &JCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool DumpAsm,
                                            ObjectCodeEmitter &OCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCObjectCodeEmitterPass(*this, OCE));
  if (DumpAsm)
    addAssemblyEmitter(PM, OptLevel, true, ferrs());

  return false;
}


