//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

//---------------------------------------------------------------------------
// Command-line options that tend to be useful on more than one back-end.
//

namespace llvm {
  bool StrongPHIElim;
  bool HasDivModLibcall;
  bool AsmVerbosityDefault(false);
}

static cl::opt<bool>
DataSections("fdata-sections",
  cl::desc("Emit data into separate sections"),
  cl::init(false));
static cl::opt<bool>
FunctionSections("ffunction-sections",
  cl::desc("Emit functions into separate sections"),
  cl::init(false));
                         
//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const Target &T,
                             StringRef TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options)
  : TheTarget(T), TargetTriple(TT), TargetCPU(CPU), TargetFS(FS),
    CodeGenInfo(0), AsmInfo(0),
    MCRelaxAll(false),
    MCNoExecStack(false),
    MCSaveTempLabels(false),
    MCUseLoc(true),
    MCUseCFI(true),
    MCUseDwarfDirectory(false),
    Options(Options) {
}

TargetMachine::~TargetMachine() {
  delete CodeGenInfo;
  delete AsmInfo;
}

/// getRelocationModel - Returns the code generation relocation model. The
/// choices are static, PIC, and dynamic-no-pic, and target default.
Reloc::Model TargetMachine::getRelocationModel() const {
  if (!CodeGenInfo)
    return Reloc::Default;
  return CodeGenInfo->getRelocationModel();
}

/// getCodeModel - Returns the code model. The choices are small, kernel,
/// medium, large, and target default.
CodeModel::Model TargetMachine::getCodeModel() const {
  if (!CodeGenInfo)
    return CodeModel::Default;
  return CodeGenInfo->getCodeModel();
}

/// getOptLevel - Returns the optimization level: None, Less,
/// Default, or Aggressive.
CodeGenOpt::Level TargetMachine::getOptLevel() const {
  if (!CodeGenInfo)
    return CodeGenOpt::Default;
  return CodeGenInfo->getOptLevel();
}

bool TargetMachine::getAsmVerbosityDefault() {
  return AsmVerbosityDefault;
}

void TargetMachine::setAsmVerbosityDefault(bool V) {
  AsmVerbosityDefault = V;
}

bool TargetMachine::getFunctionSections() {
  return FunctionSections;
}

bool TargetMachine::getDataSections() {
  return DataSections;
}

void TargetMachine::setFunctionSections(bool V) {
  FunctionSections = V;
}

void TargetMachine::setDataSections(bool V) {
  DataSections = V;
}

