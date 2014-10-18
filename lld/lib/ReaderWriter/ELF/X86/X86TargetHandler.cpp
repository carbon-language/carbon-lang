//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetHandler.h"
#include "X86DynamicLibraryWriter.h"
#include "X86ExecutableWriter.h"
#include "X86LinkingContext.h"
#include "X86RelocationHandler.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

std::unique_ptr<Writer> X86TargetHandler::getWriter() {
  switch (_x86LinkingContext.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new X86ExecutableWriter<X86ELFType>(
        _x86LinkingContext, *_x86TargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(new X86DynamicLibraryWriter<X86ELFType>(
        _x86LinkingContext, *_x86TargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
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
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::x86,
                        kindStrings);
}

X86TargetHandler::X86TargetHandler(X86LinkingContext &context)
    : DefaultTargetHandler(context), _x86LinkingContext(context),
      _x86TargetLayout(new X86TargetLayout<X86ELFType>(context)),
      _x86RelocationHandler(
          new X86TargetRelocationHandler(*_x86TargetLayout.get())) {}
