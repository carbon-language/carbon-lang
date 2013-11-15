//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationPass.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares the relocation processing pass for x86-64. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_X86_64_RELOCATION_PASS_H
#define LLD_READER_WRITER_ELF_X86_64_X86_64_RELOCATION_PASS_H

#include <memory>

namespace lld {
class Pass;
namespace elf {
class X86_64LinkingContext;

/// \brief Create x86-64 relocation pass for the given linking context.
std::unique_ptr<Pass>
createX86_64RelocationPass(const X86_64LinkingContext &);
}
}

#endif
