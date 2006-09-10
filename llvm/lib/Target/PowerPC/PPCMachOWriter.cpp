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

    virtual void GetTargetRelocation(MachineRelocation &MR, MachOSection &MOS,
                                     unsigned ToIndex);
    virtual MachineRelocation GetJTRelocation(unsigned Offset,
                                              MachineBasicBlock *MBB);
    
    virtual const char *getPassName() const {
      return "PowerPC Mach-O Writer";
    }

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
  PPCMachOWriter *MOW = new PPCMachOWriter(O, TM);
  FPM.add(MOW);
  FPM.add(createPPCCodeEmitterPass(TM, MOW->getMachineCodeEmitter()));
}

/// GetTargetRelocation - For the MachineRelocation MR, convert it to one or
/// more PowerPC MachORelocation(s), add the new relocations to the
/// MachOSection, and rewrite the instruction at the section offset if required 
/// by that relocation type.
void PPCMachOWriter::GetTargetRelocation(MachineRelocation &MR,
                                         MachOSection &MOS,
                                         unsigned ToIndex) {
  uint64_t Addr = 0;
  
  // Keep track of whether or not this is an externally defined relocation.
  bool     isExtern = false;
  
  // Get the address of whatever it is we're relocating, if possible.
  if (!isExtern)
    Addr = (uintptr_t)MR.getResultPointer();
    
  switch ((PPC::RelocationType)MR.getRelocationType()) {
  default: assert(0 && "Unknown PPC relocation type!");
  case PPC::reloc_absolute_low_ix:
    assert(0 && "Unhandled PPC relocation type!");
    break;
  case PPC::reloc_vanilla:
    {
      // FIXME: need to handle 64 bit vanilla relocs
      MachORelocation VANILLA(MR.getMachineCodeOffset(), ToIndex, false, 2, 
                              isExtern, PPC_RELOC_VANILLA);
      outword(MOS.RelocBuffer, VANILLA.r_address);
      outword(MOS.RelocBuffer, VANILLA.getPackedFields());
    }
    MOS.nreloc += 1;
    fixword(MOS.SectionData, Addr, MR.getMachineCodeOffset());
    break;
  case PPC::reloc_pcrel_bx:
    Addr -= MR.getMachineCodeOffset();
    Addr >>= 2;
    Addr & 0xFFFFFF;
    Addr <<= 2;
    Addr |= (MOS.SectionData[MR.getMachineCodeOffset()] << 24);
    fixword(MOS.SectionData, Addr, MR.getMachineCodeOffset());
    break;
  case PPC::reloc_pcrel_bcx:
    Addr -= MR.getMachineCodeOffset();
    Addr &= 0xFFFC;
    fixhalf(MOS.SectionData, Addr, MR.getMachineCodeOffset() + 2);
    break;
  case PPC::reloc_absolute_high:
    {
      MachORelocation HA16(MR.getMachineCodeOffset(), ToIndex, false, 2,
                           isExtern, PPC_RELOC_HA16);
      MachORelocation PAIR(Addr & 0xFFFF, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      outword(MOS.RelocBuffer, HA16.r_address);
      outword(MOS.RelocBuffer, HA16.getPackedFields());
      outword(MOS.RelocBuffer, PAIR.r_address);
      outword(MOS.RelocBuffer, PAIR.getPackedFields());
    }
    printf("ha16: %x\n", (unsigned)Addr);
    MOS.nreloc += 2;
    Addr += 0x8000;
    fixhalf(MOS.SectionData, Addr >> 16, MR.getMachineCodeOffset() + 2);
    break;
  case PPC::reloc_absolute_low:
    {
      MachORelocation LO16(MR.getMachineCodeOffset(), ToIndex, false, 2,
                           isExtern, PPC_RELOC_LO16);
      MachORelocation PAIR(Addr >> 16, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      outword(MOS.RelocBuffer, LO16.r_address);
      outword(MOS.RelocBuffer, LO16.getPackedFields());
      outword(MOS.RelocBuffer, PAIR.r_address);
      outword(MOS.RelocBuffer, PAIR.getPackedFields());
    }
    printf("lo16: %x\n", (unsigned)Addr);
    MOS.nreloc += 2;
    fixhalf(MOS.SectionData, Addr, MR.getMachineCodeOffset() + 2);
    break;
  }
}

MachineRelocation PPCMachOWriter::GetJTRelocation(unsigned Offset,
                                                  MachineBasicBlock *MBB) {
  // FIXME: do something about PIC
  return MachineRelocation::getBB(Offset, PPC::reloc_vanilla, MBB);
}

