//===- X86RegisterInfo.cpp - X86 Register Information ---------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86RegisterInfo.h"

// X86Regs - Turn the X86RegisterInfo.def file into a bunch of register
// descriptors
//
static const MRegisterDesc X86Regs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS) { NAME, FLAGS, TSFLAGS },
#include "X86RegisterInfo.def"
};

X86RegisterInfo::X86RegisterInfo()
  : MRegisterInfo(X86Regs, sizeof(X86Regs)/sizeof(X86Regs[0])) {
}
