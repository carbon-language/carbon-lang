//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFFile.h"
#include "MipsLinkingContext.h"
#include "MipsTargetHandler.h"
#include "MipsExecutableWriter.h"
#include "MipsDynamicLibraryWriter.h"

using namespace lld;
using namespace elf;

typedef llvm::object::ELFType<llvm::support::little, 2, false> Mips32ElELFType;

MipsTargetHandler::MipsTargetHandler(MipsLinkingContext &context)
    : DefaultTargetHandler(context), _mipsLinkingContext(context),
      _mipsRuntimeFile(new MipsRuntimeFile<Mips32ElELFType>(context)),
      _mipsTargetLayout(new MipsTargetLayout<Mips32ElELFType>(context)),
      _mipsRelocationHandler(
          new MipsTargetRelocationHandler(context, *_mipsTargetLayout.get())) {}

std::unique_ptr<Writer> MipsTargetHandler::getWriter() {
  switch (_mipsLinkingContext.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(
        new elf::MipsExecutableWriter<Mips32ElELFType>(
            _mipsLinkingContext, *_mipsTargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(
        new elf::MipsDynamicLibraryWriter<Mips32ElELFType>(
            _mipsLinkingContext, *_mipsTargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

void MipsTargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::Mips, kindStrings);
}

const Registry::KindStrings MipsTargetHandler::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(R_MIPS_NONE),
  LLD_KIND_STRING_ENTRY(R_MIPS_32),
  LLD_KIND_STRING_ENTRY(R_MIPS_26),
  LLD_KIND_STRING_ENTRY(R_MIPS_HI16),
  LLD_KIND_STRING_ENTRY(R_MIPS_LO16),
  LLD_KIND_STRING_ENTRY(R_MIPS_GOT16),
  LLD_KIND_STRING_ENTRY(R_MIPS_CALL16),
  LLD_KIND_STRING_ENTRY(R_MIPS_JALR),
  LLD_KIND_STRING_ENTRY(R_MIPS_COPY),
  LLD_KIND_STRING_ENTRY(R_MIPS_JUMP_SLOT),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_GOT),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_GOT16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_26),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_LO16),
  LLD_KIND_STRING_END
};
