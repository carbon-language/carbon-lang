//===- PrettyClassDefinitionDumper.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyClassDefinitionDumper.h"

#include "LinePrinter.h"
#include "PrettyEnumDumper.h"
#include "PrettyFunctionDumper.h"
#include "PrettyTypedefDumper.h"
#include "PrettyVariableDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/UDTLayout.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

ClassDefinitionDumper::ClassDefinitionDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

void ClassDefinitionDumper::start(const PDBSymbolTypeUDT &Class) {
  assert(opts::pretty::ClassFormat !=
         opts::pretty::ClassDefinitionFormat::None);

  uint32_t Size = Class.getLength();

  ClassLayout Layout(Class.clone());

  if (opts::pretty::OnlyPaddingClasses && (Layout.shallowPaddingSize() == 0))
    return;

  Printer.NewLine();

  WithColor(Printer, PDB_ColorItem::Keyword).get() << Class.getUdtKind() << " ";
  WithColor(Printer, PDB_ColorItem::Type).get() << Class.getName();
  WithColor(Printer, PDB_ColorItem::Comment).get() << " [sizeof = " << Size
                                                   << "]";

  auto Bases = Class.findAllChildren<PDBSymbolTypeBaseClass>();
  if (Bases->getChildCount() > 0) {
    Printer.Indent();
    Printer.NewLine();
    Printer << ":";
    uint32_t BaseIndex = 0;
    while (auto Base = Bases->getNext()) {
      Printer << " ";
      WithColor(Printer, PDB_ColorItem::Keyword).get() << Base->getAccess();
      if (Base->isVirtualBaseClass())
        WithColor(Printer, PDB_ColorItem::Keyword).get() << " virtual";
      WithColor(Printer, PDB_ColorItem::Type).get() << " " << Base->getName();
      if (++BaseIndex < Bases->getChildCount()) {
        Printer.NewLine();
        Printer << ",";
      }
    }
    Printer.Unindent();
  }

  Printer << " {";
  Printer.Indent();

  // Dump non-layout items first, but only if we're not in layout-only mode.
  if (opts::pretty::ClassFormat !=
      opts::pretty::ClassDefinitionFormat::Layout) {
    for (auto &Other : Layout.other_items())
      Other->dump(*this);
  }

  const BitVector &UseMap = Layout.usedBytes();
  int NextUnusedByte = Layout.usedBytes().find_first_unset();
  // Next dump items which affect class layout.
  for (auto &LayoutItem : Layout.layout_items()) {
    if (NextUnusedByte >= 0) {
      // If there are padding bytes remaining, see if this field is the first to
      // cross a padding boundary, and print a padding field indicator if so.
      int Off = LayoutItem->getOffsetInParent();
      if (Off > NextUnusedByte) {
        uint32_t Amount = Off - NextUnusedByte;
        Printer.NewLine();
        WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> ("
                                                         << Amount << " bytes)";
        assert(UseMap.find_next(NextUnusedByte) == Off);
        NextUnusedByte = UseMap.find_next_unset(Off);
      }
    }
    LayoutItem->getSymbol().dump(*this);
  }

  if (NextUnusedByte >= 0 && Layout.getClassSize() > 1) {
    uint32_t Amount = Layout.getClassSize() - NextUnusedByte;
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> (" << Amount
                                                     << " bytes)";
    DumpedAnything = true;
  }

  Printer.Unindent();
  if (DumpedAnything)
    Printer.NewLine();
  Printer << "}";
  Printer.NewLine();
  if (Layout.deepPaddingSize() > 0) {
    APFloat Pct(100.0 * (double)Layout.deepPaddingSize() / (double)Size);
    SmallString<8> PctStr;
    Pct.toString(PctStr, 4);
    WithColor(Printer, PDB_ColorItem::Padding).get()
        << "Total padding " << Layout.deepPaddingSize() << " bytes (" << PctStr
        << "% of class size)";
    Printer.NewLine();
  }
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeBaseClass &Symbol) {}

void ClassDefinitionDumper::dump(const PDBSymbolData &Symbol) {
  VariableDumper Dumper(Printer);
  Dumper.start(Symbol);
  DumpedAnything = true;
}

void ClassDefinitionDumper::dump(const PDBSymbolFunc &Symbol) {
  if (Printer.IsSymbolExcluded(Symbol.getName()))
    return;
  if (Symbol.isCompilerGenerated() && opts::pretty::ExcludeCompilerGenerated)
    return;
  if (Symbol.getLength() == 0 && !Symbol.isPureVirtual() &&
      !Symbol.isIntroVirtualFunction())
    return;

  DumpedAnything = true;
  Printer.NewLine();
  FunctionDumper Dumper(Printer);
  Dumper.start(Symbol, FunctionDumper::PointerType::None);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeVTable &Symbol) {
  VariableDumper Dumper(Printer);
  Dumper.start(Symbol);
  DumpedAnything = true;
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  DumpedAnything = true;
  Printer.NewLine();
  EnumDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  DumpedAnything = true;
  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeUDT &Symbol) {}
