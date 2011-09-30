#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class MipsELFObjectWriter : public MCELFObjectTargetWriter {
public:
  MipsELFObjectWriter(bool is64Bit, Triple::OSType OSType, uint16_t EMachine,
                      bool HasRelocationAddend)
    : MCELFObjectTargetWriter(is64Bit, OSType, EMachine,
                              HasRelocationAddend) {}
};

class MipsAsmBackend : public MCAsmBackend {
public:
  MipsAsmBackend(const Target &T)
    : MCAsmBackend() {}

  unsigned getNumFixupKinds() const {
    return 1;   //tbd
  }

  /// ApplyFixup - Apply the \arg Value for given \arg Fixup into the provided
  /// data fragment, at the offset specified by the fixup and following the
  /// fixup kind as appropriate.
  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
  }

  /// @name Target Relaxation Interfaces
  /// @{

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \param Inst - The instruction to test.
  bool MayNeedRelaxation(const MCInst &Inst) const {
    return false;
  }

  /// RelaxInstruction - Relax the instruction in the given fragment to the next
  /// wider instruction.
  ///
  /// \param Inst - The instruction to relax, which may be the same as the
  /// output.
  /// \parm Res [output] - On return, the relaxed instruction.
  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
  }
  
  /// @}

  /// WriteNopData - Write an (optimal) nop sequence of Count bytes to the given
  /// output. If the target cannot generate such a sequence, it should return an
  /// error.
  ///
  /// \return - True on success.
  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
    return false;
  }
};

class MipsEB_AsmBackend : public MipsAsmBackend {
public:
  Triple::OSType OSType;

  MipsEB_AsmBackend(const Target &T, Triple::OSType _OSType)
    : MipsAsmBackend(T), OSType(_OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(createELFObjectTargetWriter(),
                                 OS, /*IsLittleEndian*/ false);
  }

  MCELFObjectTargetWriter *createELFObjectTargetWriter() const {
    return new MipsELFObjectWriter(false, OSType, ELF::EM_MIPS, false);
  }
};

class MipsEL_AsmBackend : public MipsAsmBackend {
public:
  Triple::OSType OSType;

  MipsEL_AsmBackend(const Target &T, Triple::OSType _OSType)
    : MipsAsmBackend(T), OSType(_OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(createELFObjectTargetWriter(),
                                 OS, /*IsLittleEndian*/ true);
  }

  MCELFObjectTargetWriter *createELFObjectTargetWriter() const {
    return new MipsELFObjectWriter(false, OSType, ELF::EM_MIPS, false);
  }
};
}

MCAsmBackend *llvm::createMipsAsmBackend(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  // just return little endian for now
  //
  return new MipsEL_AsmBackend(T, Triple(TT).getOS());
}
