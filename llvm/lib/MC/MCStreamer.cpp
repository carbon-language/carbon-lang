//===- lib/MC/MCStreamer.cpp - Streaming Machine Code Output --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

using namespace llvm;

MCStreamer::MCStreamer(MCContext &_Context) : Context(_Context), CurSection(0) {
}

MCStreamer::~MCStreamer() {
}
