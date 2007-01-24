//===-- PPCMachOWriterInfo.h - Mach-O Writer Info for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Mach-O writer information for the PowerPC backend.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_MACHO_WRITER_INFO_H
#define PPC_MACHO_WRITER_INFO_H

#include "llvm/Target/TargetMachOWriterInfo.h"

namespace llvm {

  // Forward declarations
  class PPCTargetMachine;

  struct PPCMachOWriterInfo : public TargetMachOWriterInfo {
    PPCMachOWriterInfo(const PPCTargetMachine &TM);
    virtual ~PPCMachOWriterInfo() {}

    virtual const char *getPassName() const {
      return "PowerPC Mach-O Writer";
    }
  };

} // end llvm namespace

#endif // PPC_MACHO_WRITER_INFO_H
