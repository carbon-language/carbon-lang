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
