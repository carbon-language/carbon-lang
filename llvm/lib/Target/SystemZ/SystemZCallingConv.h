//===-- SystemZCallingConv.h - Calling conventions for SystemZ --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZCALLINGCONV_H
#define SYSTEMZCALLINGCONV_H

namespace llvm {
  namespace SystemZ {
    const unsigned NumArgGPRs = 5;
    extern const unsigned ArgGPRs[NumArgGPRs];

    const unsigned NumArgFPRs = 4;
    extern const unsigned ArgFPRs[NumArgFPRs];
  }
}

#endif
