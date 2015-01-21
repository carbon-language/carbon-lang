//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationPass.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares the relocation processing pass for ARM. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_RELOCATION_PASS_H
#define LLD_READER_WRITER_ELF_ARM_ARM_RELOCATION_PASS_H

#include <memory>

namespace lld {
class Pass;
namespace elf {
class ARMLinkingContext;

/// \brief Create ARM relocation pass for the given linking context.
std::unique_ptr<Pass> createARMRelocationPass(const ARMLinkingContext &);
}
}

#endif
