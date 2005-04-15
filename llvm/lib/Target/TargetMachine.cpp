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
#include "llvm/Type.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

//---------------------------------------------------------------------------
// Command-line options that tend to be useful on more than one back-end.
//

namespace llvm {
  bool PrintMachineCode;
  bool NoFramePointerElim;
  bool NoExcessFPPrecision;
  int  PatternISelTriState;
};
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
  cl::opt<int, true> PatternISel("enable-pattern-isel",
                    cl::desc("sets the pattern ISel off(0), on(1), default(2)"),
                    cl::location(PatternISelTriState),
                    cl::init(2));
};

//---------------------------------------------------------------------------
// TargetMachine Class
//
TargetMachine::TargetMachine(const std::string &name, IntrinsicLowering *il,
                             bool LittleEndian,
                             unsigned char PtrSize, unsigned char PtrAl,
                             unsigned char DoubleAl, unsigned char FloatAl,
                             unsigned char LongAl, unsigned char IntAl,
                             unsigned char ShortAl, unsigned char ByteAl,
                             unsigned char BoolAl)
  : Name(name), DataLayout(name, LittleEndian,
                           PtrSize, PtrAl, DoubleAl, FloatAl, LongAl,
                           IntAl, ShortAl, ByteAl, BoolAl) {
  IL = il ? il : new DefaultIntrinsicLowering();
}

TargetMachine::TargetMachine(const std::string &name, IntrinsicLowering *il,
                             const TargetData &TD)
  : Name(name), DataLayout(TD) {
  IL = il ? il : new DefaultIntrinsicLowering();
}

TargetMachine::TargetMachine(const std::string &name, IntrinsicLowering *il,
                             const Module &M)
  : Name(name), DataLayout(name, &M) {
  IL = il ? il : new DefaultIntrinsicLowering();
}

TargetMachine::~TargetMachine() {
  delete IL;
}

