//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "X86_64DynamicLibraryWriter.h"
#include "X86_64ExecutableWriter.h"
#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"

using namespace lld;
using namespace elf;

X86_64TargetHandler::X86_64TargetHandler(X86_64LinkingContext &context)
    : DefaultTargetHandler(context), _context(context),
      _x86_64TargetLayout(new X86_64TargetLayout<X86_64ELFType>(context)),
      _x86_64RelocationHandler(
          new X86_64TargetRelocationHandler(*_x86_64TargetLayout.get())) {}

void X86_64TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::x86_64, kindStrings);
}

std::unique_ptr<Writer> X86_64TargetHandler::getWriter() {
  switch (this->_context.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new X86_64ExecutableWriter<X86_64ELFType>(
        _context, *_x86_64TargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(
        new X86_64DynamicLibraryWriter<X86_64ELFType>(
            _context, *_x86_64TargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

const Registry::KindStrings X86_64TargetHandler::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(R_X86_64_NONE),
  LLD_KIND_STRING_ENTRY(R_X86_64_64),
  LLD_KIND_STRING_ENTRY(R_X86_64_PC32),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOT32),
  LLD_KIND_STRING_ENTRY(R_X86_64_PLT32),
  LLD_KIND_STRING_ENTRY(R_X86_64_COPY),
  LLD_KIND_STRING_ENTRY(R_X86_64_GLOB_DAT),
  LLD_KIND_STRING_ENTRY(R_X86_64_JUMP_SLOT),
  LLD_KIND_STRING_ENTRY(R_X86_64_RELATIVE),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPCREL),
  LLD_KIND_STRING_ENTRY(R_X86_64_32),
  LLD_KIND_STRING_ENTRY(R_X86_64_32S),
  LLD_KIND_STRING_ENTRY(R_X86_64_16),
  LLD_KIND_STRING_ENTRY(R_X86_64_PC16),
  LLD_KIND_STRING_ENTRY(R_X86_64_8),
  LLD_KIND_STRING_ENTRY(R_X86_64_PC8),
  LLD_KIND_STRING_ENTRY(R_X86_64_DTPMOD64),
  LLD_KIND_STRING_ENTRY(R_X86_64_DTPOFF64),
  LLD_KIND_STRING_ENTRY(R_X86_64_TPOFF64),
  LLD_KIND_STRING_ENTRY(R_X86_64_TLSGD),
  LLD_KIND_STRING_ENTRY(R_X86_64_TLSLD),
  LLD_KIND_STRING_ENTRY(R_X86_64_DTPOFF32),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTTPOFF),
  LLD_KIND_STRING_ENTRY(R_X86_64_TPOFF32),
  LLD_KIND_STRING_ENTRY(R_X86_64_PC64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTOFF64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPC32),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOT64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPCREL64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPC64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPLT64),
  LLD_KIND_STRING_ENTRY(R_X86_64_PLTOFF64),
  LLD_KIND_STRING_ENTRY(R_X86_64_SIZE32),
  LLD_KIND_STRING_ENTRY(R_X86_64_SIZE64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPC32_TLSDESC),
  LLD_KIND_STRING_ENTRY(R_X86_64_TLSDESC_CALL),
  LLD_KIND_STRING_ENTRY(R_X86_64_TLSDESC),
  LLD_KIND_STRING_ENTRY(R_X86_64_IRELATIVE),
  LLD_KIND_STRING_ENTRY(LLD_R_X86_64_GOTRELINDEX),
  LLD_KIND_STRING_END
};
