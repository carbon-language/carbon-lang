//===-- XCoreTargetStreamer.h - XCore Target Streamer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETSTREAMER_H
#define XCORETARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class XCoreTargetStreamer : public MCTargetStreamer {
public:
  XCoreTargetStreamer(MCStreamer &S);
  virtual ~XCoreTargetStreamer();
  virtual void emitCCTopData(StringRef Name) = 0;
  virtual void emitCCTopFunction(StringRef Name) = 0;
  virtual void emitCCBottomData(StringRef Name) = 0;
  virtual void emitCCBottomFunction(StringRef Name) = 0;
};
}

#endif
