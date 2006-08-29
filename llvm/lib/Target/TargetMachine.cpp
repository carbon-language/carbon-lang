//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Type.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

//---------------------------------------------------------------------------
// Command-line options that tend to be useful on more than one back-end.
//

namespace llvm {
  bool PrintMachineCode;
  bool NoFramePointerElim;
  bool NoExcessFPPrecision;
  bool UnsafeFPMath;
  bool FiniteOnlyFPMathOption;
  Reloc::Model RelocationModel;
  CodeModel::Model CMModel;
}
namespace {
  cl::opt<bool, true> PrintCode("print-machineinstrs",
    cl::desc("Print generated machine code"),
    cl::location(PrintMachineCode), cl::init(false));

  cl::opt<bool, true>
    DisableFPElim("disable-fp-elim",
                  cl::desc("Disable frame pointer elimination optimization"),
                  cl::location(NoFramePointerElim),
                  cl::init(false));
  cl::opt<bool, true>
  DisableExcessPrecision("disable-excess-fp-precision",
               cl::desc("Disable optimizations that may increase FP precision"),
               cl::location(NoExcessFPPrecision),
               cl::init(false));
  cl::opt<bool, true>
  EnableUnsafeFPMath("enable-unsafe-fp-math",
               cl::desc("Enable optimizations that may decrease FP precision"),
               cl::location(UnsafeFPMath),
               cl::init(false));
  cl::opt<bool, true>
  EnableFiniteOnltFPMath("enable-finite-only-fp-math",
               cl::desc("Enable optimizations that assumes non- NaNs / +-Infs"),
               cl::location(FiniteOnlyFPMathOption),
               cl::init(false));
  cl::opt<llvm::Reloc::Model, true>
  DefRelocationModel(
    "relocation-model",
    cl::desc("Choose relocation model"),
    cl::location(RelocationModel),
    cl::init(Reloc::Default),
    cl::values(
      clEnumValN(Reloc::Default, "default",
                 "  Target default relocation model"),
      clEnumValN(Reloc::Static, "static",
                 "  Non-relocatable code"),
      clEnumValN(Reloc::PIC_, "pic",
                 "  Fully relocatable, position independent code"),
      clEnumValN(Reloc::DynamicNoPIC, "dynamic-no-pic",
                 "  Relocatable external references, non-relocatable code"),
      clEnumValEnd));
  cl::opt<llvm::CodeModel::Model, true>
  DefCodeModel(
    "code-model",
    cl::desc("Choose relocation model"),
    cl::location(CMModel),
    cl::init(CodeModel::Default),
    cl::values(
      clEnumValN(CodeModel::Default, "default",
                 "  Target default code model"),
      clEnumValN(CodeModel::Small, "small",
                 "  Small code model"),
      clEnumValN(CodeModel::Kernel, "kernel",
                 "  Kernel code model"),
      clEnumValN(CodeModel::Medium, "medium",
                 "  Medium code model"),
      clEnumValN(CodeModel::Large, "large",
                 "  Large code model"),
      clEnumValEnd));
}

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const std::string &name, const Module &M)
  : Name(name) {
}

TargetMachine::~TargetMachine() {
}

/// getRelocationModel - Returns the code generation relocation model. The
/// choices are static, PIC, and dynamic-no-pic, and target default.
Reloc::Model TargetMachine::getRelocationModel() {
  return RelocationModel;
}

/// setRelocationModel - Sets the code generation relocation model.
void TargetMachine::setRelocationModel(Reloc::Model Model) {
  RelocationModel = Model;
}

/// getCodeModel - Returns the code model. The choices are small, kernel,
/// medium, large, and target default.
CodeModel::Model TargetMachine::getCodeModel() {
  return CMModel;
}

/// setCodeModel - Sets the code model.
void TargetMachine::setCodeModel(CodeModel::Model Model) {
  CMModel = Model;
}

namespace llvm {
  /// FiniteOnlyFPMath - This returns true when the -enable-finite-only-fp-math
  /// option is specified on the command line. If this returns false (default),
  /// the code generator is not allowed to assume that FP arithmetic arguments
  /// and results are never NaNs or +-Infs.
  bool FiniteOnlyFPMath() { return UnsafeFPMath || FiniteOnlyFPMathOption; }
}
