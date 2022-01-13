//===- MinGW.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_MINGW_H
#define LLD_COFF_MINGW_H

#include "Config.h"
#include "Symbols.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/ArgList.h"
#include <vector>

namespace lld {
namespace coff {

// Logic for deciding what symbols to export, when exporting all
// symbols for MinGW.
class AutoExporter {
public:
  AutoExporter();

  void addWholeArchive(StringRef path);

  llvm::StringSet<> excludeSymbols;
  llvm::StringSet<> excludeSymbolPrefixes;
  llvm::StringSet<> excludeSymbolSuffixes;
  llvm::StringSet<> excludeLibs;
  llvm::StringSet<> excludeObjects;

  bool shouldExport(Defined *sym) const;
};

void writeDefFile(StringRef name);

// The -wrap option is a feature to rename symbols so that you can write
// wrappers for existing functions. If you pass `-wrap:foo`, all
// occurrences of symbol `foo` are resolved to `__wrap_foo` (so, you are
// expected to write `__wrap_foo` function as a wrapper). The original
// symbol becomes accessible as `__real_foo`, so you can call that from your
// wrapper.
//
// This data structure is instantiated for each -wrap option.
struct WrappedSymbol {
  Symbol *sym;
  Symbol *real;
  Symbol *wrap;
};

std::vector<WrappedSymbol> addWrappedSymbols(llvm::opt::InputArgList &args);

void wrapSymbols(ArrayRef<WrappedSymbol> wrapped);

} // namespace coff
} // namespace lld

#endif
