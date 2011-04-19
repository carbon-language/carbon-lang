//===-- PPCAsmBackend.cpp - PPC Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmBackend.h"
#include "PPC.h"
#include "PPCFixupKinds.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

namespace {
class PPCMachObjectWriter : public MCMachObjectTargetWriter {
public:
  PPCMachObjectWriter(bool Is64Bit, uint32_t CPUType,
                      uint32_t CPUSubtype)
    : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype) {}
};

class PPCAsmBackend : public TargetAsmBackend {
const Target &TheTarget;
public:
  PPCAsmBackend(const Target &T) : TargetAsmBackend(), TheTarget(T) {}

  unsigned getNumFixupKinds() const { return PPC::NumTargetFixupKinds; }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[PPC::NumTargetFixupKinds] = {
      // name                    offset  bits  flags
      { "fixup_ppc_br24",        6,      24,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_brcond14",    16,     14,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_lo16",        16,     16,   0 },
      { "fixup_ppc_ha16",        16,     16,   0 },
      { "fixup_ppc_lo14",        16,     14,   0 }
    };
  
    if (Kind < FirstTargetFixupKind)
      return TargetAsmBackend::getFixupKindInfo(Kind);
  
    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }
  
  bool MayNeedRelaxation(const MCInst &Inst) const {
    // FIXME.
    return false;
  }
  
  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
    // FIXME.
    assert(0 && "RelaxInstruction() unimplemented");
  }
  
  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
    // FIXME: Zero fill for now. That's not right, but at least will get the
    // section size right.
    for (uint64_t i = 0; i != Count; ++i)
      OW->Write8(0);
    return true;
  }      
  
  unsigned getPointerSize() const {
    StringRef Name = TheTarget.getName();
    if (Name == "ppc64") return 8;
    assert(Name == "ppc32" && "Unknown target name!");
    return 4;
  }
};
} // end anonymous namespace


// FIXME: This should be in a separate file.
namespace {
  class DarwinPPCAsmBackend : public PPCAsmBackend {
  public:
    DarwinPPCAsmBackend(const Target &T) : PPCAsmBackend(T) { }
    
    void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                    uint64_t Value) const {
      assert(0 && "UNIMP");
    }
    
    MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
      bool is64 = getPointerSize() == 8;
      return createMachObjectWriter(new PPCMachObjectWriter(
                                      /*Is64Bit=*/is64,
                                      (is64 ? object::mach::CTM_PowerPC64 :
                                       object::mach::CTM_PowerPC),
                                      object::mach::CSPPC_ALL),
                                    OS, /*IsLittleEndian=*/false);
    }
    
    virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
      return false;
    }
  };
} // end anonymous namespace




TargetAsmBackend *llvm::createPPCAsmBackend(const Target &T,
                                            const std::string &TT) {
  if (Triple(TT).isOSDarwin())
    return new DarwinPPCAsmBackend(T);

  return 0;
}
