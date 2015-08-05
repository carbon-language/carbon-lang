//===- Writer.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_WRITER_H
#define LLD_ELF_WRITER_H

namespace lld {
namespace elf2 {

class OutputSection;
class SymbolTable;

template <class ELFT>
void writeResult(SymbolTable *Symtab, StringRef Path);

}
}

#endif
