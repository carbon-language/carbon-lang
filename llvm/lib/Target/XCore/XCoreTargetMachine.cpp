//===-- XCoreTargetMachine.cpp - Define TargetMachine for XCore -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetAsmInfo.h"
#include "XCoreTargetMachine.h"
#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// XCoreTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int XCoreTargetMachineModule;
int XCoreTargetMachineModule = 0;

extern Target TheXCoreTarget;
namespace {
  // Register the target.
  RegisterTarget<XCoreTargetMachine> X(TheXCoreTarget, "xcore", "XCore");
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTarget() { 
  TargetRegistry::RegisterAsmPrinter(TheXCoreTarget,
                                     &createXCoreCodePrinterPass);
}

const TargetAsmInfo *XCoreTargetMachine::createTargetAsmInfo() const {
  return new XCoreTargetAsmInfo(*this);
}

/// XCoreTargetMachine ctor - Create an ILP32 architecture model
///
XCoreTargetMachine::XCoreTargetMachine(const Target &T, const Module &M, 
                                       const std::string &FS)
  : LLVMTargetMachine(T),
    Subtarget(*this, M, FS),
    DataLayout("e-p:32:32:32-a0:0:32-f32:32:32-f64:32:32-i1:8:32-i8:8:32-"
               "i16:16:32-i32:32:32-i64:32:32"),
    InstrInfo(),
    FrameInfo(*this),
    TLInfo(*this) {
}

bool XCoreTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createXCoreISelDag(*this));
  return false;
}

bool XCoreTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool Verbose,
                                            formatted_raw_ostream &Out) {
  FunctionPass *Printer = getTarget().createAsmPrinter(Out, *this, Verbose);
  if (!Printer)
    llvm_report_error("unable to create assembly printer");
  PM.add(Printer);
  return false;
}
