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

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H

#include "DwarfUnit.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfo.h"

namespace llvm {

class AsmPrinter;
class DIE;
class DwarfDebug;
class DwarfFile;
class MCSymbol;
class LexicalScope;

class DwarfCompileUnit : public DwarfUnit {
  /// The attribute index of DW_AT_stmt_list in the compile unit DIE, avoiding
  /// the need to search for it in applyStmtList.
  unsigned stmtListIndex;

  /// \brief Construct a DIE for the given DbgVariable without initializing the
  /// DbgVariable's DIE reference.
  std::unique_ptr<DIE> constructVariableDIEImpl(const DbgVariable &DV,
                                                bool Abstract);

public:
  DwarfCompileUnit(unsigned UID, DICompileUnit Node, AsmPrinter *A,
                   DwarfDebug *DW, DwarfFile *DWU);

  void initStmtList(MCSymbol *DwarfLineSectionSym);

  /// Apply the DW_AT_stmt_list from this compile unit to the specified DIE.
  void applyStmtList(DIE &D);

  /// getOrCreateGlobalVariableDIE - get or create global variable DIE.
  DIE *getOrCreateGlobalVariableDIE(DIGlobalVariable GV);

  /// addLabelAddress - Add a dwarf label attribute data and value using
  /// either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label);

  /// addLocalLabelAddress - Add a dwarf label attribute data and value using
  /// DW_FORM_addr only.
  void addLocalLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                            const MCSymbol *Label);

  /// addSectionDelta - Add a label delta attribute data and value.
  void addSectionDelta(DIE &Die, dwarf::Attribute Attribute, const MCSymbol *Hi,
                       const MCSymbol *Lo);

  DwarfCompileUnit &getCU() override { return *this; }

  unsigned getOrCreateSourceID(StringRef FileName, StringRef DirName) override;

  /// addRange - Add an address range to the list of ranges for this unit.
  void addRange(RangeSpan Range);

  void attachLowHighPC(DIE &D, const MCSymbol *Begin, const MCSymbol *End);

  /// addSectionLabel - Add a Dwarf section label attribute data and value.
  ///
  void addSectionLabel(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label, const MCSymbol *Sec);

  /// \brief Find DIE for the given subprogram and attach appropriate
  /// DW_AT_low_pc and DW_AT_high_pc attributes. If there are global
  /// variables in this scope then create and insert DIEs for these
  /// variables.
  DIE &updateSubprogramScopeDIE(DISubprogram SP);

  void constructScopeDIE(LexicalScope *Scope,
                         SmallVectorImpl<std::unique_ptr<DIE>> &FinalChildren);

  /// \brief A helper function to construct a RangeSpanList for a given
  /// lexical scope.
  void addScopeRangeList(DIE &ScopeDIE,
                         const SmallVectorImpl<InsnRange> &Range);

  void attachRangesOrLowHighPC(DIE &D,
                               const SmallVectorImpl<InsnRange> &Ranges);

  /// \brief This scope represents inlined body of a function. Construct
  /// DIE to represent this concrete inlined copy of the function.
  std::unique_ptr<DIE> constructInlinedScopeDIE(LexicalScope *Scope);

  /// \brief Construct new DW_TAG_lexical_block for this scope and
  /// attach DW_AT_low_pc/DW_AT_high_pc labels.
  std::unique_ptr<DIE> constructLexicalScopeDIE(LexicalScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  std::unique_ptr<DIE> constructVariableDIE(DbgVariable &DV,
                                            bool Abstract = false);

  std::unique_ptr<DIE> constructVariableDIE(DbgVariable &DV,
                                            const LexicalScope &Scope,
                                            DIE *&ObjectPointer);

  /// A helper function to create children of a Scope DIE.
  DIE *createScopeChildrenDIE(LexicalScope *Scope,
                              SmallVectorImpl<std::unique_ptr<DIE>> &Children,
                              unsigned *ChildScopeCount = nullptr);

  /// \brief Construct a DIE for this subprogram scope.
  void constructSubprogramScopeDIE(LexicalScope *Scope);

  DIE *createAndAddScopeChildren(LexicalScope *Scope, DIE &ScopeDIE);

  void constructAbstractSubprogramScopeDIE(LexicalScope *Scope);

  /// \brief Construct import_module DIE.
  std::unique_ptr<DIE>
  constructImportedEntityDIE(const DIImportedEntity &Module);

  void finishSubprogramDefinition(DISubprogram SP);
};

} // end llvm namespace

#endif
