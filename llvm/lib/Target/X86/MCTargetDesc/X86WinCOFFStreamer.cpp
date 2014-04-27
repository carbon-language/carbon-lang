//===-- X86WinCOFFStreamer.cpp - X86 Target WinCOFF Streamer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86MCTargetDesc.h"
#include "llvm/MC/MCWinCOFFStreamer.h"

using namespace llvm;

namespace {
class X86WinCOFFStreamer : public MCWinCOFFStreamer {
public:
  X86WinCOFFStreamer(MCContext &C, MCAsmBackend &AB, MCCodeEmitter *CE,
                     raw_ostream &OS)
    : MCWinCOFFStreamer(C, AB, *CE, OS) { }

  void EmitWin64EHHandlerData() override;
  void FinishImpl() override;
};

void X86WinCOFFStreamer::EmitWin64EHHandlerData() {
  MCStreamer::EmitWin64EHHandlerData();

  // We have to emit the unwind info now, because this directive
  // actually switches to the .xdata section!
  MCWin64EHUnwindEmitter::EmitUnwindInfo(*this, getCurrentW64UnwindInfo());
}

void X86WinCOFFStreamer::FinishImpl() {
  EmitFrames(nullptr, true);
  EmitW64Tables();

  MCWinCOFFStreamer::FinishImpl();
}
}

namespace llvm {
MCStreamer *createX86WinCOFFStreamer(MCContext &C, MCAsmBackend &AB,
                                     MCCodeEmitter *CE, raw_ostream &OS,
                                     bool RelaxAll) {
  X86WinCOFFStreamer *S = new X86WinCOFFStreamer(C, AB, CE, OS);
  S->getAssembler().setRelaxAll(RelaxAll);
  return S;
}
}

