//===-- llvm/Support/MachO.h - The MachO file format ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the MachO object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MACHO_H
#define LLVM_SUPPORT_MACHO_H

// NOTE: The enums in this file are intentially named to be different than those
// in the headers in /usr/include/mach (on darwin systems) to avoid conflicts
// with those macros.
namespace llvm {
  namespace MachO {
    // Enums from <mach/machine.h>
    enum {
      // Capability bits used in the definition of cpu_type.
      CPUArchMask = 0xff000000,   // Mask for architecture bits
      CPUArchABI64 = 0x01000000,  // 64 bit ABI
      
      // Constants for the cputype field.
      CPUTypeI386      = 7,
      CPUTypeX86_64    = CPUTypeI386 | CPUArchABI64,
      CPUTypeARM       = 12,
      CPUTypeSPARC     = 14,
      CPUTypePowerPC   = 18,
      CPUTypePowerPC64 = CPUTypePowerPC | CPUArchABI64,


      // Constants for the cpusubtype field.
      
      // X86
      CPUSubType_I386_ALL    = 3,
      CPUSubType_X86_64_ALL  = 3,
      
      // ARM
      CPUSubType_ARM_ALL     = 0,
      CPUSubType_ARM_V4T     = 5,
      CPUSubType_ARM_V6      = 6,

      // PowerPC
      CPUSubType_POWERPC_ALL = 0,
      
      CPUSubType_SPARC_ALL   = 0
    };
  } // end namespace MachO
} // end namespace llvm

#endif
