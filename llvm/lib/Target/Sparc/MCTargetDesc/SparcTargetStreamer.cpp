//===-- SparcTargetStreamer.cpp - Sparc Target Streamer Methods -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sparc specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "SparcTargetStreamer.h"
#include "InstPrinter/SparcInstPrinter.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

// pin vtable to this file
void SparcTargetStreamer::anchor() {}

SparcTargetAsmStreamer::SparcTargetAsmStreamer(formatted_raw_ostream &OS)
    : OS(OS) {}

void SparcTargetAsmStreamer::emitSparcRegisterIgnore(unsigned reg) {
  OS << "\t.register "
     << "%" << StringRef(SparcInstPrinter::getRegisterName(reg)).lower()
     << ", #ignore\n";
}

void SparcTargetAsmStreamer::emitSparcRegisterScratch(unsigned reg) {
  OS << "\t.register "
     << "%" << StringRef(SparcInstPrinter::getRegisterName(reg)).lower()
     << ", #scratch\n";
}

MCELFStreamer &SparcTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(*Streamer);
}
