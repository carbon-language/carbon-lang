//===- lib/ReaderWriter/PECOFF/LinkerGeneratedSymbolFile.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LinkerGeneratedSymbolFile.h"

namespace lld {
namespace pecoff {

// Find decorated symbol, namely /sym@[0-9]+/ or /\?sym@@.+/.
bool findDecoratedSymbol(PECOFFLinkingContext *ctx, ResolvableSymbols *syms,
                         std::string sym, std::string &res) {
  const std::set<std::string> &defined = syms->defined();
  // Search for /sym@[0-9]+/
  {
    std::string s = sym + '@';
    auto it = defined.lower_bound(s);
    for (auto e = defined.end(); it != e; ++it) {
      if (!StringRef(*it).startswith(s))
        break;
      if (it->size() == s.size())
        continue;
      StringRef suffix = StringRef(*it).substr(s.size());
      if (suffix.find_first_not_of("0123456789") != StringRef::npos)
        continue;
      res = *it;
      return true;
    }
  }
  // Search for /\?sym@@.+/
  {
    std::string s = "?" + ctx->undecorateSymbol(sym).str() + "@@";
    auto it = defined.lower_bound(s);
    if (it != defined.end() && StringRef(*it).startswith(s)) {
      res = *it;
      return true;
    }
  }
  return false;
}

} // namespace pecoff
} // namespace lld
