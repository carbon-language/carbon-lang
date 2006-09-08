//===-- PPCMachOWriter.cpp - Emit a Mach-O file for the PowerPC backend ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Mach-O writer for the PowerPC backend.  The public
// interface to this file is the createPPCMachOObjectWriterPass function.
//
//===----------------------------------------------------------------------===//

#include "PPCRelocations.h"
#include "PPCTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachOWriter.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN PPCMachOWriter : public MachOWriter {
  public:
    PPCMachOWriter(std::ostream &O, PPCTargetMachine &TM) : MachOWriter(O, TM) {
      if (TM.getTargetData()->getPointerSizeInBits() == 64) {
        Header.cputype = MachOHeader::CPU_TYPE_POWERPC64;
      } else {
        Header.cputype = MachOHeader::CPU_TYPE_POWERPC;
      }
      Header.cpusubtype = MachOHeader::CPU_SUBTYPE_POWERPC_ALL;
    }

    virtual void GetTargetRelocation(MachOSection &MOS, MachineRelocation &MR,
                                     uint64_t Addr);
    
    // Constants for the relocation r_type field.
    // see <mach-o/ppc/reloc.h>
    enum { PPC_RELOC_VANILLA, // generic relocation
           PPC_RELOC_PAIR,    // the second relocation entry of a pair
           PPC_RELOC_BR14,    // 14 bit branch displacement to word address
           PPC_RELOC_BR24,    // 24 bit branch displacement to word address
           PPC_RELOC_HI16,    // a PAIR follows with the low 16 bits
           PPC_RELOC_LO16,    // a PAIR follows with the high 16 bits
           PPC_RELOC_HA16,    // a PAIR follows, which is sign extended to 32b
           PPC_RELOC_LO14     // LO16 with low 2 bits implicitly zero
    };
  };
}

/// addPPCMachOObjectWriterPass - Returns a pass that outputs the generated code
/// as a Mach-O object file.
///
void llvm::addPPCMachOObjectWriterPass(FunctionPassManager &FPM,
                                       std::ostream &O, PPCTargetMachine &TM) {
  PPCMachOWriter *EW = new PPCMachOWriter(O, TM);
  FPM.add(EW);
  FPM.add(createPPCCodeEmitterPass(TM, EW->getMachineCodeEmitter()));
}

/// GetTargetRelocation - For the MachineRelocation MR, convert it to one or
/// more PowerPC MachORelocation(s), add the new relocations to the
/// MachOSection, and rewrite the instruction at the section offset if required 
/// by that relocation type.
void PPCMachOWriter::GetTargetRelocation(MachOSection &MOS,
                                         MachineRelocation &MR,
                                         uint64_t Addr) {
  // Keep track of whether or not this is an externally defined relocation.
  uint32_t index = MOS.Index;
  bool     isExtern = false;
  
  // Get the address of the instruction to rewrite
  unsigned char *RelocPos = &MOS.SectionData[0] + MR.getMachineCodeOffset();
  
  // Get the address of whatever it is we're relocating, if possible.
  if (MR.isGlobalValue()) {
    // determine whether or not its external and then figure out what section
    // we put it in if it's a locally defined symbol.
  } else if (MR.isString()) {
    // lookup in global values?
  } else {
    assert((MR.isConstantPoolIndex() || MR.isJumpTableIndex()) &&
           "Unhandled MachineRelocation type!");
  }
  
  switch ((PPC::RelocationType)MR.getRelocationType()) {
  default: assert(0 && "Unknown PPC relocation type!");
  case PPC::reloc_pcrel_bx:
  case PPC::reloc_pcrel_bcx:
  case PPC::reloc_absolute_low_ix:
    assert(0 && "Unhandled PPC relocation type!");
    break;
  case PPC::reloc_absolute_high:
    {
      MachORelocation HA16(MR.getMachineCodeOffset(), index, false, 2, isExtern, 
                           PPC_RELOC_HA16);
      MachORelocation PAIR(Addr & 0xFFFF, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      outword(RelocBuffer, HA16.r_address);
      outword(RelocBuffer, HA16.getPackedFields());
      outword(RelocBuffer, PAIR.r_address);
      outword(RelocBuffer, PAIR.getPackedFields());
    }
    MOS.nreloc += 2;
    Addr += 0x8000;
    *(unsigned *)RelocPos &= 0xFFFF0000;
    *(unsigned *)RelocPos |= ((Addr >> 16) & 0xFFFF);
    break;
  case PPC::reloc_absolute_low:
    {
      MachORelocation LO16(MR.getMachineCodeOffset(), index, false, 2, isExtern, 
                           PPC_RELOC_LO16);
      MachORelocation PAIR(Addr >> 16, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      outword(RelocBuffer, LO16.r_address);
      outword(RelocBuffer, LO16.getPackedFields());
      outword(RelocBuffer, PAIR.r_address);
      outword(RelocBuffer, PAIR.getPackedFields());
    }
    MOS.nreloc += 2;
    *(unsigned *)RelocPos &= 0xFFFF0000;
    *(unsigned *)RelocPos |= (Addr & 0xFFFF);
    break;
  }
}
