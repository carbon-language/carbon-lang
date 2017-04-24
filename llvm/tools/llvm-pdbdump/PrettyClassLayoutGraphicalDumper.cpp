//===- PrettyClassLayoutGraphicalDumper.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyClassLayoutGraphicalDumper.h"

#include "LinePrinter.h"
#include "PrettyClassDefinitionDumper.h"
#include "PrettyVariableDumper.h"

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/UDTLayout.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

PrettyClassLayoutGraphicalDumper::PrettyClassLayoutGraphicalDumper(
    LinePrinter &P, uint32_t InitialOffset)
    : PDBSymDumper(true), Printer(P), ClassOffsetZero(InitialOffset),
      CurrentAbsoluteOffset(InitialOffset) {}

bool PrettyClassLayoutGraphicalDumper::start(const UDTLayoutBase &Layout) {
  const BitVector &UseMap = Layout.usedBytes();
  int NextPaddingByte = UseMap.find_first_unset();

  for (auto &Item : Layout.layout_items()) {
    // Calculate the absolute offset of the first byte of the next field.
    uint32_t RelativeOffset = Item->getOffsetInParent();
    CurrentAbsoluteOffset = ClassOffsetZero + RelativeOffset;

    // Since there is storage there, it should be set!  However, this might
    // be an empty base, in which case it could extend outside the bounds of
    // the parent class.
    if (RelativeOffset < UseMap.size() && (Item->getSize() > 0)) {
      assert(UseMap.test(RelativeOffset));

      // If there is any remaining padding in this class, and the offset of the
      // new item is after the padding, then we must have just jumped over some
      // padding.  Print a padding row and then look for where the next block
      // of padding begins.
      if ((NextPaddingByte >= 0) &&
          (RelativeOffset > uint32_t(NextPaddingByte))) {
        printPaddingRow(RelativeOffset - NextPaddingByte);
        NextPaddingByte = UseMap.find_next_unset(RelativeOffset);
      }
    }

    CurrentItem = Item;
    if (Item->isVBPtr()) {
      VTableLayoutItem &Layout = static_cast<VTableLayoutItem &>(*CurrentItem);

      VariableDumper VarDumper(Printer);
      VarDumper.startVbptr(CurrentAbsoluteOffset, Layout.getSize());
    } else {
      if (auto Sym = Item->getSymbol())
        Sym->dump(*this);
    }

    if (Item->getLayoutSize() > 0) {
      uint32_t Prev = RelativeOffset + Item->getLayoutSize() - 1;
      NextPaddingByte = UseMap.find_next_unset(Prev);
    }
  }

  auto TailPadding = Layout.tailPadding();
  if (TailPadding > 0) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> ("
      << TailPadding << " bytes)";
    DumpedAnything = true;
  }

  return DumpedAnything;
}

void PrettyClassLayoutGraphicalDumper::printPaddingRow(uint32_t Amount) {
  if (Amount == 0)
    return;

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> (" << Amount
                                                   << " bytes)";
  DumpedAnything = true;
}

void PrettyClassLayoutGraphicalDumper::dump(
    const PDBSymbolTypeBaseClass &Symbol) {
  assert(CurrentItem != nullptr);

  Printer.NewLine();
  BaseClassLayout &Layout = static_cast<BaseClassLayout &>(*CurrentItem);

  std::string Label = "base";
  if (Layout.isVirtualBase()) {
    Label.insert(Label.begin(), 'v');
    if (Layout.getBase().isIndirectVirtualBaseClass())
      Label.insert(Label.begin(), 'i');
  }
  Printer << Label << " ";

  uint32_t Size = Layout.isEmptyBase() ? 1 : Layout.getLayoutSize();

  WithColor(Printer, PDB_ColorItem::Offset).get()
      << "+" << format_hex(CurrentAbsoluteOffset, 4) << " [sizeof=" << Size
      << "] ";

  WithColor(Printer, PDB_ColorItem::Identifier).get() << Layout.getName();

  Printer.Indent();
  uint32_t ChildOffsetZero = ClassOffsetZero + Layout.getOffsetInParent();
  PrettyClassLayoutGraphicalDumper BaseDumper(Printer, ChildOffsetZero);
  BaseDumper.start(Layout);
  Printer.Unindent();

  DumpedAnything = true;
}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolData &Symbol) {
  assert(CurrentItem != nullptr);

  DataMemberLayoutItem &Layout =
      static_cast<DataMemberLayoutItem &>(*CurrentItem);

  VariableDumper VarDumper(Printer);
  VarDumper.start(Symbol, ClassOffsetZero);

  if (Layout.hasUDTLayout()) {
    Printer.Indent();
    PrettyClassLayoutGraphicalDumper TypeDumper(Printer, ClassOffsetZero);
    TypeDumper.start(Layout.getUDTLayout());
    Printer.Unindent();
  }

  DumpedAnything = true;
}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolTypeVTable &Symbol) {
  assert(CurrentItem != nullptr);

  VariableDumper VarDumper(Printer);
  VarDumper.start(Symbol, ClassOffsetZero);

  DumpedAnything = true;
}
