//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "X86_64TargetHandler.h"
#include "X86_64LinkingContext.h"

using namespace lld;
using namespace elf;

X86_64TargetHandler::X86_64TargetHandler(X86_64LinkingContext &context)
    : DefaultTargetHandler(context), _gotFile(new GOTFile(context)),
      _relocationHandler(context), _targetLayout(context) {}

bool X86_64TargetHandler::createImplicitFiles(
    std::vector<std::unique_ptr<File> > &result) {
  _gotFile->addAtom(*new (_gotFile->_alloc) GLOBAL_OFFSET_TABLEAtom(*_gotFile));
  _gotFile->addAtom(*new (_gotFile->_alloc) TLSGETADDRAtom(*_gotFile));
  if (_context.isDynamic())
    _gotFile->addAtom(*new (_gotFile->_alloc) DYNAMICAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
  return true;
}

void X86_64TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, 
                        Reference::KindArch::x86_64, 
                        kindStrings);
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
  LLD_KIND_STRING_ENTRY(R_X86_64_TPOFF32),
  LLD_KIND_STRING_ENTRY(R_X86_64_PC64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTOFF64),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOTPC32),
  LLD_KIND_STRING_ENTRY(R_X86_64_GOT64),
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

