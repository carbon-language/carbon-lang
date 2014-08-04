//===-- CPPTargetMachine.h - TargetMachine for the C++ backend --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TargetMachine that is used by the C++ backend.
//
//===----------------------------------------------------------------------===//

#ifndef CPPTARGETMACHINE_H
#define CPPTARGETMACHINE_H

#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

namespace llvm {

class formatted_raw_ostream;

class CPPSubtarget : public TargetSubtargetInfo {
};

struct CPPTargetMachine : public TargetMachine {
  CPPTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL)
    : TargetMachine(T, TT, CPU, FS, Options), Subtarget() {}
private:
  CPPSubtarget Subtarget;

public:
  const CPPSubtarget *getSubtargetImpl() const override { return &Subtarget; }
  bool addPassesToEmitFile(PassManagerBase &PM, formatted_raw_ostream &Out,
                           CodeGenFileType FileType, bool DisableVerify,
                           AnalysisID StartAfter,
                           AnalysisID StopAfter) override;
};

extern Target TheCppBackendTarget;

} // End llvm namespace


#endif
