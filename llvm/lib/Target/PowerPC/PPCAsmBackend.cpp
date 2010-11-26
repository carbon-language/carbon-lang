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
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/MachO.h"
using namespace llvm;

namespace {
  class PPCAsmBackend : public TargetAsmBackend {
  const Target &TheTarget;
  public:
    PPCAsmBackend(const Target &T) : TargetAsmBackend(), TheTarget(T) {}
    
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
    MCMachOObjectFormat Format;
  public:
    DarwinPPCAsmBackend(const Target &T) : PPCAsmBackend(T) {
      HasScatteredSymbols = true;
    }
    
    virtual const MCObjectFormat &getObjectFormat() const {
      return Format;
    }
    
    void ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                    uint64_t Value) const {
      assert(0 && "UNIMP");
    }
    
    MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
      bool is64 = getPointerSize() == 8;
      return createMachObjectWriter(OS, /*Is64Bit=*/is64,
                                    is64 ? MachO::CPUTypePowerPC64 : 
                                    MachO::CPUTypePowerPC64,
                                    MachO::CPUSubType_POWERPC_ALL,
                                    /*IsLittleEndian=*/false);
    }
    
    virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
      return false;
    }
  };
} // end anonymous namespace




TargetAsmBackend *llvm::createPPCAsmBackend(const Target &T,
                                            const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinPPCAsmBackend(T);
  default:
    return 0;
  }
}
