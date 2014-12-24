//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFFile.h"
#include "MipsDynamicLibraryWriter.h"
#include "MipsExecutableWriter.h"
#include "MipsLinkingContext.h"
#include "MipsTargetHandler.h"

using namespace lld;
using namespace elf;

typedef llvm::object::ELFType<llvm::support::little, 2, false> Mips32ElELFType;

MipsTargetHandler::MipsTargetHandler(MipsLinkingContext &ctx)
    : DefaultTargetHandler(ctx), _ctx(ctx),
      _runtimeFile(new MipsRuntimeFile<Mips32ElELFType>(ctx)),
      _targetLayout(new MipsTargetLayout<Mips32ElELFType>(ctx)),
      _relocationHandler(new MipsTargetRelocationHandler(*_targetLayout, ctx)) {}

std::unique_ptr<Writer> MipsTargetHandler::getWriter() {
  switch (_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new MipsExecutableWriter<Mips32ElELFType>(
        _ctx, *_targetLayout, _elfFlagsMerger));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(
        new MipsDynamicLibraryWriter<Mips32ElELFType>(_ctx, *_targetLayout,
                                                      _elfFlagsMerger));
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

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings MipsTargetHandler::kindStrings[] = {
#include "llvm/Support/ELFRelocs/Mips.def"
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_GOT),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_32_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_26),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_LO16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_STO_PLT),
  LLD_KIND_STRING_ENTRY(LLD_R_MICROMIPS_GLOBAL_26_S1),
  LLD_KIND_STRING_END
};

#undef ELF_RELOC
