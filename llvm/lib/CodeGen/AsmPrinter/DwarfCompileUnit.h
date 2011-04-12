//===-- llvm/CodeGen/DwarfCompileUnit.h - Dwarf Compile Unit ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
#define CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H

#include "DIE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {

//===----------------------------------------------------------------------===//
/// CompileUnit - This dwarf writer support class manages information associate
/// with a source file.
class CompileUnit {
  /// ID - File identifier for source.
  ///
  unsigned ID;

  /// Die - Compile unit debug information entry.
  ///
  const OwningPtr<DIE> CUDie;

  /// IndexTyDie - An anonymous type for index type.  Owned by CUDie.
  DIE *IndexTyDie;

  /// MDNodeToDieMap - Tracks the mapping of unit level debug informaton
  /// variables to debug information entries.
  DenseMap<const MDNode *, DIE *> MDNodeToDieMap;

  /// MDNodeToDIEEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEEntry proxy.
  DenseMap<const MDNode *, DIEEntry *> MDNodeToDIEEntryMap;

  /// Globals - A map of globally visible named entities for this unit.
  ///
  StringMap<DIE*> Globals;

  /// GlobalTypes - A map of globally visible types for this unit.
  ///
  StringMap<DIE*> GlobalTypes;

public:
  CompileUnit(unsigned I, DIE *D)
    : ID(I), CUDie(D), IndexTyDie(0) {}

  // Accessors.
  unsigned getID()                  const { return ID; }
  DIE* getCUDie()                   const { return CUDie.get(); }
  const StringMap<DIE*> &getGlobals()     const { return Globals; }
  const StringMap<DIE*> &getGlobalTypes() const { return GlobalTypes; }

  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const { return !CUDie->getChildren().empty(); }

  /// addGlobal - Add a new global entity to the compile unit.
  ///
  void addGlobal(StringRef Name, DIE *Die) { Globals[Name] = Die; }

  /// addGlobalType - Add a new global type to the compile unit.
  ///
  void addGlobalType(StringRef Name, DIE *Die) {
    GlobalTypes[Name] = Die;
  }

  /// getDIE - Returns the debug information entry map slot for the
  /// specified debug variable.
  DIE *getDIE(const MDNode *N) { return MDNodeToDieMap.lookup(N); }

  /// insertDIE - Insert DIE into the map.
  void insertDIE(const MDNode *N, DIE *D) {
    MDNodeToDieMap.insert(std::make_pair(N, D));
  }

  /// getDIEEntry - Returns the debug information entry for the speciefied
  /// debug variable.
  DIEEntry *getDIEEntry(const MDNode *N) {
    DenseMap<const MDNode *, DIEEntry *>::iterator I =
      MDNodeToDIEEntryMap.find(N);
    if (I == MDNodeToDIEEntryMap.end())
      return NULL;
    return I->second;
  }

  /// insertDIEEntry - Insert debug information entry into the map.
  void insertDIEEntry(const MDNode *N, DIEEntry *E) {
    MDNodeToDIEEntryMap.insert(std::make_pair(N, E));
  }

  /// addDie - Adds or interns the DIE to the compile unit.
  ///
  void addDie(DIE *Buffer) {
    this->CUDie->addChild(Buffer);
  }

  // getIndexTyDie - Get an anonymous type for index type.
  DIE *getIndexTyDie() {
    return IndexTyDie;
  }

  // setIndexTyDie - Set D as anonymous type for index which can be reused
  // later.
  void setIndexTyDie(DIE *D) {
    IndexTyDie = D;
  }

};

} // end llvm namespace
#endif
