//===- X86RegisterInfo.h - X86 Register Information Impl ----------*-C++-*-===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"

struct X86RegisterInfo : public MRegisterInfo {
  X86RegisterInfo();

};

#endif
