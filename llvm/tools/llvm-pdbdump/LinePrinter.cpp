//===- LinePrinter.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LinePrinter.h"

#include "llvm-pdbdump.h"

#include "llvm/Support/Regex.h"

#include <algorithm>

namespace {
template <class T, class Pred> bool any_of_range(T &&R, Pred P) {
  return std::any_of(R.begin(), R.end(), P);
}

bool IsItemExcluded(llvm::StringRef Item,
                    std::list<llvm::Regex> &IncludeFilters,
                    std::list<llvm::Regex> &ExcludeFilters) {
  if (Item.empty())
    return false;

  auto match_pred = [Item](llvm::Regex &R) { return R.match(Item); };

  // Include takes priority over exclude.  If the user specified include
  // filters, and none of them include this item, them item is gone.
  if (!IncludeFilters.empty() && !any_of_range(IncludeFilters, match_pred))
    return true;

  if (any_of_range(ExcludeFilters, match_pred))
    return true;

  return false;
}
}

using namespace llvm;

LinePrinter::LinePrinter(int Indent, llvm::raw_ostream &Stream)
    : OS(Stream), IndentSpaces(Indent), CurrentIndent(0) {
  SetFilters(ExcludeTypeFilters, opts::ExcludeTypes.begin(),
             opts::ExcludeTypes.end());
  SetFilters(ExcludeSymbolFilters, opts::ExcludeSymbols.begin(),
             opts::ExcludeSymbols.end());
  SetFilters(ExcludeCompilandFilters, opts::ExcludeCompilands.begin(),
             opts::ExcludeCompilands.end());

  SetFilters(IncludeTypeFilters, opts::IncludeTypes.begin(),
             opts::IncludeTypes.end());
  SetFilters(IncludeSymbolFilters, opts::IncludeSymbols.begin(),
             opts::IncludeSymbols.end());
  SetFilters(IncludeCompilandFilters, opts::IncludeCompilands.begin(),
             opts::IncludeCompilands.end());
}

void LinePrinter::Indent() { CurrentIndent += IndentSpaces; }

void LinePrinter::Unindent() {
  CurrentIndent = std::max(0, CurrentIndent - IndentSpaces);
}

void LinePrinter::NewLine() {
  OS << "\n";
  OS.indent(CurrentIndent);
}

bool LinePrinter::IsTypeExcluded(llvm::StringRef TypeName) {
  return IsItemExcluded(TypeName, IncludeTypeFilters, ExcludeTypeFilters);
}

bool LinePrinter::IsSymbolExcluded(llvm::StringRef SymbolName) {
  return IsItemExcluded(SymbolName, IncludeSymbolFilters, ExcludeSymbolFilters);
}

bool LinePrinter::IsCompilandExcluded(llvm::StringRef CompilandName) {
  return IsItemExcluded(CompilandName, IncludeCompilandFilters,
                        ExcludeCompilandFilters);
}

WithColor::WithColor(LinePrinter &P, PDB_ColorItem C) : OS(P.OS) {
  if (C == PDB_ColorItem::None)
    OS.resetColor();
  else {
    raw_ostream::Colors Color;
    bool Bold;
    translateColor(C, Color, Bold);
    OS.changeColor(Color, Bold);
  }
}

WithColor::~WithColor() { OS.resetColor(); }

void WithColor::translateColor(PDB_ColorItem C, raw_ostream::Colors &Color,
                               bool &Bold) const {
  switch (C) {
  case PDB_ColorItem::Address:
    Color = raw_ostream::YELLOW;
    Bold = true;
    return;
  case PDB_ColorItem::Keyword:
    Color = raw_ostream::MAGENTA;
    Bold = true;
    return;
  case PDB_ColorItem::Register:
  case PDB_ColorItem::Offset:
    Color = raw_ostream::YELLOW;
    Bold = false;
    return;
  case PDB_ColorItem::Type:
    Color = raw_ostream::CYAN;
    Bold = true;
    return;
  case PDB_ColorItem::Identifier:
    Color = raw_ostream::CYAN;
    Bold = false;
    return;
  case PDB_ColorItem::Path:
    Color = raw_ostream::CYAN;
    Bold = false;
    return;
  case PDB_ColorItem::SectionHeader:
    Color = raw_ostream::RED;
    Bold = true;
    return;
  case PDB_ColorItem::LiteralValue:
    Color = raw_ostream::GREEN;
    Bold = true;
  default:
    return;
  }
}
