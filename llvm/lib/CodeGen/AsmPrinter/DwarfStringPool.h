//===-- llvm/CodeGen/DwarfStringPool.h - Dwarf Debug Framework -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Allocator.h"

#include <utility>

namespace llvm {

class MCSymbol;
class MCSection;
class StringRef;

// Collection of strings for this unit and assorted symbols.
// A String->Symbol mapping of strings used by indirect
// references.
class DwarfStringPool {
  StringMap<std::pair<MCSymbol *, unsigned>, BumpPtrAllocator &> Pool;
  StringRef Prefix;

public:
  DwarfStringPool(BumpPtrAllocator &A, AsmPrinter &Asm, StringRef Prefix)
      : Pool(A), Prefix(Prefix) {}

  void emit(AsmPrinter &Asm, const MCSection *StrSection,
            const MCSection *OffsetSection = nullptr);

  /// \brief Returns an entry into the string pool with the given
  /// string text.
  MCSymbol *getSymbol(AsmPrinter &Asm, StringRef Str);

  /// \brief Returns the index into the string pool with the given
  /// string text.
  unsigned getIndex(AsmPrinter &Asm, StringRef Str);

  bool empty() const { return Pool.empty(); }
};
}
#endif
