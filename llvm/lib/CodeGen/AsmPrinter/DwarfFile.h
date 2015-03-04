//===-- llvm/CodeGen/DwarfFile.h - Dwarf Debug Framework -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H

#include "AddressPool.h"
#include "DwarfStringPool.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class AsmPrinter;
class DbgVariable;
class DwarfUnit;
class DIEAbbrev;
class MCSymbol;
class DIE;
class DISubprogram;
class LexicalScope;
class StringRef;
class DwarfDebug;
class MCSection;
class DwarfFile {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  // Used to uniquely define abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  // A list of all the unique abbreviations in use.
  std::vector<DIEAbbrev *> Abbreviations;

  // A pointer to all units in the section.
  SmallVector<std::unique_ptr<DwarfUnit>, 1> CUs;

  DwarfStringPool StrPool;

  // Collection of dbg variables of a scope.
  DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8>> ScopeVariables;

  // Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;

  /// Maps MDNodes for type system with the corresponding DIEs. These DIEs can
  /// be shared across CUs, that is why we keep the map here instead
  /// of in DwarfCompileUnit.
  DenseMap<const MDNode *, DIE *> MDTypeNodeToDieMap;

public:
  DwarfFile(AsmPrinter *AP, StringRef Pref, BumpPtrAllocator &DA);

  ~DwarfFile();

  const SmallVectorImpl<std::unique_ptr<DwarfUnit>> &getUnits() { return CUs; }

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE &Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Define a unique number for the abbreviation.
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// \brief Add a unit to the list of CUs.
  void addUnit(std::unique_ptr<DwarfUnit> U);

  /// \brief Emit all of the units to the section listed with the given
  /// abbreviation section.
  void emitUnits(const MCSymbol *ASectionSym);

  /// \brief Emit a set of abbreviations to the specific section.
  void emitAbbrevs(const MCSection *);

  /// \brief Emit all of the strings to the section given.
  void emitStrings(const MCSection *StrSection,
                   const MCSection *OffsetSection = nullptr);

  /// \brief Returns the string pool.
  DwarfStringPool &getStringPool() { return StrPool; }

  /// \returns false if the variable was merged with a previous one.
  bool addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8>> &getScopeVariables() {
    return ScopeVariables;
  }

  DenseMap<const MDNode *, DIE *> &getAbstractSPDies() {
    return AbstractSPDies;
  }

  void insertDIE(const MDNode *TypeMD, DIE *Die) {
    MDTypeNodeToDieMap.insert(std::make_pair(TypeMD, Die));
  }
  DIE *getDIE(const MDNode *TypeMD) {
    return MDTypeNodeToDieMap.lookup(TypeMD);
  }
};
}
#endif
