//====-- PTXSubtarget.h - Define Subtarget for the PTX ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_SUBTARGET_H
#define PTX_SUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

namespace llvm {
  class PTXSubtarget : public TargetSubtarget {
    private:
      bool is_sm20;

    public:
      PTXSubtarget(const std::string &TT, const std::string &FS);

      std::string ParseSubtargetFeatures(const std::string &FS,
                                         const std::string &CPU);
  }; // class PTXSubtarget
} // namespace llvm

#endif // PTX_SUBTARGET_H
