//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetHandler.h"
#include "X86LinkingContext.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

/// \brief R_386_32 - word32:  S + A
static int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}

/// \brief R_386_PC32 - word32: S + A - P
static int relocPC32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result +
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}



const Registry::KindStrings X86TargetHandler::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(R_386_NONE),
  LLD_KIND_STRING_ENTRY(R_386_32),
  LLD_KIND_STRING_ENTRY(R_386_PC32),
  LLD_KIND_STRING_ENTRY(R_386_GOT32),
  LLD_KIND_STRING_ENTRY(R_386_PLT32),
  LLD_KIND_STRING_ENTRY(R_386_COPY),
  LLD_KIND_STRING_ENTRY(R_386_GLOB_DAT),
  LLD_KIND_STRING_ENTRY(R_386_JUMP_SLOT),
  LLD_KIND_STRING_ENTRY(R_386_RELATIVE),
  LLD_KIND_STRING_ENTRY(R_386_GOTOFF),
  LLD_KIND_STRING_ENTRY(R_386_GOTPC),
  LLD_KIND_STRING_ENTRY(R_386_32PLT),
  LLD_KIND_STRING_ENTRY(R_386_TLS_TPOFF),
  LLD_KIND_STRING_ENTRY(R_386_TLS_IE),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GOTIE),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LE),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GD),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDM),
  LLD_KIND_STRING_ENTRY(R_386_16),
  LLD_KIND_STRING_ENTRY(R_386_PC16),
  LLD_KIND_STRING_ENTRY(R_386_8),
  LLD_KIND_STRING_ENTRY(R_386_PC8),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GD_32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GD_PUSH),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GD_CALL),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GD_POP),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDM_32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDM_PUSH),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDM_CALL),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDM_POP),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LDO_32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_IE_32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_LE_32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_DTPMOD32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_DTPOFF32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_TPOFF32),
  LLD_KIND_STRING_ENTRY(R_386_TLS_GOTDESC),
  LLD_KIND_STRING_ENTRY(R_386_TLS_DESC_CALL),
  LLD_KIND_STRING_ENTRY(R_386_TLS_DESC),
  LLD_KIND_STRING_ENTRY(R_386_IRELATIVE),
  LLD_KIND_STRING_ENTRY(R_386_NUM),
  LLD_KIND_STRING_END
};

void X86TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, 
                        Reference::KindArch::x86, 
                        kindStrings);
}

error_code X86TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return error_code::success();
  assert(ref.kindArch() == Reference::KindArch::x86);
  switch (ref.kindValue()) {
  case R_386_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_386_PC32:
    relocPC32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  default : {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "Unhandled I386 relocation # " << ref.kindValue();
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return error_code::success();
}

X86TargetHandler::X86TargetHandler(X86LinkingContext &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo) {}
