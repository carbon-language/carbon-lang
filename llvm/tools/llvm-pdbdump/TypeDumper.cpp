//===- TypeDumper.cpp - PDBSymDumper implementation for types *----- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeDumper.h"

#include "ClassDefinitionDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"
#include "TypedefDumper.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;

TypeDumper::TypeDumper(LinePrinter &P, bool ClassDefs)
    : PDBSymDumper(true), Printer(P), FullClassDefs(ClassDefs) {}

void TypeDumper::start(const PDBSymbolExe &Exe, raw_ostream &OS, int Indent) {
  auto Enums = Exe.findAllChildren<PDBSymbolTypeEnum>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Enums";
  Printer << ": (" << Enums->getChildCount() << " items)";
  Printer.Indent();
  while (auto Enum = Enums->getNext())
    Enum->dump(OS, Indent + 2, *this);
  Printer.Unindent();

  auto Typedefs = Exe.findAllChildren<PDBSymbolTypeTypedef>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Typedefs";
  Printer << ": (" << Typedefs->getChildCount() << " items)";
  Printer.Indent();
  while (auto Typedef = Typedefs->getNext())
    Typedef->dump(OS, Indent + 2, *this);
  Printer.Unindent();

  auto Classes = Exe.findAllChildren<PDBSymbolTypeUDT>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Classes";
  Printer << ": (" << Classes->getChildCount() << " items)";
  Printer.Indent();
  while (auto Class = Classes->getNext())
    Class->dump(OS, Indent + 2, *this);
  Printer.Unindent();
}

void TypeDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                      int Indent) {
  if (Symbol.getUnmodifiedTypeId() != 0)
    return;
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;
  Printer.NewLine();

  WithColor(Printer, PDB_ColorItem::Keyword).get() << "enum ";
  WithColor(Printer, PDB_ColorItem::Identifier).get() << Symbol.getName();
}

void TypeDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                      int Indent) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol, OS, Indent);
}

void TypeDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                      int Indent) {
  if (Symbol.getUnmodifiedTypeId() != 0)
    return;
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();

  if (FullClassDefs) {
    ClassDefinitionDumper Dumper(Printer);
    Dumper.start(Symbol, OS, Indent);
  } else {
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "class ";
    WithColor(Printer, PDB_ColorItem::Identifier).get() << Symbol.getName();
  }
}
