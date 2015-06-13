//===-- MCTargetDesc/AMDGPUMCAsmInfo.h - AMDGPU MCAsm Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCASMINFO_H
#define LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
namespace llvm {

class Triple;

// If you need to create another MCAsmInfo class, which inherits from MCAsmInfo,
// you will need to make sure your new class sets PrivateGlobalPrefix to
// a prefix that won't appeary in a fuction name.  The default value
// for PrivateGlobalPrefix is 'L', so it will consider any function starting
// with 'L' as a local symbol.
class AMDGPUMCAsmInfo : public MCAsmInfoELF {
public:
  explicit AMDGPUMCAsmInfo(const Triple &TT);
};
} // namespace llvm
#endif
