//===-- llvm/CodeGen/DwarfUnit.h - Dwarf Compile Unit ---*- C++ -*--===//
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
#include "DwarfDebug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCDwarf.h"

namespace llvm {

class MachineLocation;
class MachineOperand;
class ConstantInt;
class ConstantFP;
class DbgVariable;
class DwarfCompileUnit;

// Data structure to hold a range for range lists.
class RangeSpan {
public:
  RangeSpan(MCSymbol *S, MCSymbol *E) : Start(S), End(E) {}
  const MCSymbol *getStart() const { return Start; }
  const MCSymbol *getEnd() const { return End; }
  void setEnd(const MCSymbol *E) { End = E; }

private:
  const MCSymbol *Start, *End;
};

class RangeSpanList {
private:
  // Index for locating within the debug_range section this particular span.
  MCSymbol *RangeSym;
  // List of ranges.
  SmallVector<RangeSpan, 2> Ranges;

public:
  RangeSpanList(MCSymbol *Sym) : RangeSym(Sym) {}
  MCSymbol *getSym() const { return RangeSym; }
  const SmallVectorImpl<RangeSpan> &getRanges() const { return Ranges; }
  void addRange(RangeSpan Range) { Ranges.push_back(Range); }
};

//===----------------------------------------------------------------------===//
/// Unit - This dwarf writer support class manages information associated
/// with a source file.
class DwarfUnit {
protected:
  /// UniqueID - a numeric ID unique among all CUs in the module
  unsigned UniqueID;

  /// Node - MDNode for the compile unit.
  DICompileUnit CUNode;

  /// Unit debug information entry.
  const std::unique_ptr<DIE> UnitDie;

  /// Offset of the UnitDie from beginning of debug info section.
  unsigned DebugInfoOffset;

  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  // Holders for some common dwarf information.
  DwarfDebug *DD;
  DwarfFile *DU;

  /// IndexTyDie - An anonymous type for index type.  Owned by UnitDie.
  DIE *IndexTyDie;

  /// MDNodeToDieMap - Tracks the mapping of unit level debug information
  /// variables to debug information entries.
  DenseMap<const MDNode *, DIE *> MDNodeToDieMap;

  /// MDNodeToDIEEntryMap - Tracks the mapping of unit level debug information
  /// descriptors to debug information entries using a DIEEntry proxy.
  DenseMap<const MDNode *, DIEEntry *> MDNodeToDIEEntryMap;

  /// GlobalNames - A map of globally visible named entities for this unit.
  StringMap<const DIE *> GlobalNames;

  /// GlobalTypes - A map of globally visible types for this unit.
  StringMap<const DIE *> GlobalTypes;

  /// DIEBlocks - A list of all the DIEBlocks in use.
  std::vector<DIEBlock *> DIEBlocks;
  
  /// DIELocs - A list of all the DIELocs in use.
  std::vector<DIELoc *> DIELocs;

  /// ContainingTypeMap - This map is used to keep track of subprogram DIEs that
  /// need DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, const MDNode *> ContainingTypeMap;

  // List of ranges for a given compile unit.
  SmallVector<RangeSpan, 1> CURanges;

  // List of range lists for a given compile unit, separate from the ranges for
  // the CU itself.
  SmallVector<RangeSpanList, 1> CURangeLists;

  // DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  // DIEIntegerOne - A preallocated DIEValue because 1 is used frequently.
  DIEInteger *DIEIntegerOne;

  /// The section this unit will be emitted in.
  const MCSection *Section;

  /// A label at the start of the non-dwo section related to this unit.
  MCSymbol *SectionSym;

  /// The start of the unit within its section.
  MCSymbol *LabelBegin;

  /// The end of the unit within its section.
  MCSymbol *LabelEnd;

  /// The label for the start of the range sets for the elements of this unit.
  MCSymbol *LabelRange;

  /// Skeleton unit associated with this unit.
  DwarfUnit *Skeleton;

  DwarfUnit(unsigned UID, DIE *D, DICompileUnit CU, AsmPrinter *A,
            DwarfDebug *DW, DwarfFile *DWU);

public:
  virtual ~DwarfUnit();

  /// Set the skeleton unit associated with this unit.
  void setSkeleton(DwarfUnit &Skel) { Skeleton = &Skel; }

  /// Get the skeleton unit associated with this unit.
  DwarfUnit *getSkeleton() const { return Skeleton; }

  /// Pass in the SectionSym even though we could recreate it in every compile
  /// unit (type units will have actually distinct symbols once they're in
  /// comdat sections).
  void initSection(const MCSection *Section, MCSymbol *SectionSym) {
    assert(!this->Section);
    this->Section = Section;
    this->SectionSym = SectionSym;
    this->LabelBegin =
        Asm->GetTempSymbol(Section->getLabelBeginName(), getUniqueID());
    this->LabelEnd =
        Asm->GetTempSymbol(Section->getLabelEndName(), getUniqueID());
    this->LabelRange = Asm->GetTempSymbol("gnu_ranges", getUniqueID());
  }

  const MCSection *getSection() const {
    assert(Section);
    return Section;
  }

  /// If there's a skeleton then return the section symbol for the skeleton
  /// unit, otherwise return the section symbol for this unit.
  MCSymbol *getLocalSectionSym() const {
    if (Skeleton)
      return Skeleton->getSectionSym();
    return getSectionSym();
  }

  MCSymbol *getSectionSym() const {
    assert(Section);
    return SectionSym;
  }

  /// If there's a skeleton then return the begin label for the skeleton unit,
  /// otherwise return the local label for this unit.
  MCSymbol *getLocalLabelBegin() const {
    if (Skeleton)
      return Skeleton->getLabelBegin();
    return getLabelBegin();
  }

  MCSymbol *getLabelBegin() const {
    assert(Section);
    return LabelBegin;
  }

  MCSymbol *getLabelEnd() const {
    assert(Section);
    return LabelEnd;
  }

  MCSymbol *getLabelRange() const {
    assert(Section);
    return LabelRange;
  }

  // Accessors.
  unsigned getUniqueID() const { return UniqueID; }
  uint16_t getLanguage() const { return CUNode.getLanguage(); }
  DICompileUnit getCUNode() const { return CUNode; }
  DIE &getUnitDie() const { return *UnitDie; }
  const StringMap<const DIE *> &getGlobalNames() const { return GlobalNames; }
  const StringMap<const DIE *> &getGlobalTypes() const { return GlobalTypes; }

  unsigned getDebugInfoOffset() const { return DebugInfoOffset; }
  void setDebugInfoOffset(unsigned DbgInfoOff) { DebugInfoOffset = DbgInfoOff; }

  /// hasContent - Return true if this compile unit has something to write out.
  bool hasContent() const { return !UnitDie->getChildren().empty(); }

  /// addRange - Add an address range to the list of ranges for this unit.
  void addRange(RangeSpan Range);

  /// getRanges - Get the list of ranges for this unit.
  const SmallVectorImpl<RangeSpan> &getRanges() const { return CURanges; }
  SmallVectorImpl<RangeSpan> &getRanges() { return CURanges; }

  /// addRangeList - Add an address range list to the list of range lists.
  void addRangeList(RangeSpanList Ranges) { CURangeLists.push_back(Ranges); }

  /// getRangeLists - Get the vector of range lists.
  const SmallVectorImpl<RangeSpanList> &getRangeLists() const {
    return CURangeLists;
  }
  SmallVectorImpl<RangeSpanList> &getRangeLists() { return CURangeLists; }

  /// getParentContextString - Get a string containing the language specific
  /// context for a global name.
  std::string getParentContextString(DIScope Context) const;

  /// addGlobalName - Add a new global entity to the compile unit.
  ///
  void addGlobalName(StringRef Name, DIE &Die, DIScope Context);

  /// addAccelNamespace - Add a new name to the namespace accelerator table.
  void addAccelNamespace(StringRef Name, const DIE &Die);

  /// getDIE - Returns the debug information entry map slot for the
  /// specified debug variable. We delegate the request to DwarfDebug
  /// when the MDNode can be part of the type system, since DIEs for
  /// the type system can be shared across CUs and the mappings are
  /// kept in DwarfDebug.
  DIE *getDIE(DIDescriptor D) const;

  /// getDIELoc - Returns a fresh newly allocated DIELoc.
  DIELoc *getDIELoc() { return new (DIEValueAllocator) DIELoc(); }

  /// insertDIE - Insert DIE into the map. We delegate the request to DwarfDebug
  /// when the MDNode can be part of the type system, since DIEs for
  /// the type system can be shared across CUs and the mappings are
  /// kept in DwarfDebug.
  void insertDIE(DIDescriptor Desc, DIE *D);

  /// addFlag - Add a flag that is true to the DIE.
  void addFlag(DIE &Die, dwarf::Attribute Attribute);

  /// addUInt - Add an unsigned integer attribute data and value.
  void addUInt(DIE &Die, dwarf::Attribute Attribute, Optional<dwarf::Form> Form,
               uint64_t Integer);

  void addUInt(DIE &Block, dwarf::Form Form, uint64_t Integer);

  /// addSInt - Add an signed integer attribute data and value.
  void addSInt(DIE &Die, dwarf::Attribute Attribute, Optional<dwarf::Form> Form,
               int64_t Integer);

  void addSInt(DIELoc &Die, Optional<dwarf::Form> Form, int64_t Integer);

  /// addString - Add a string attribute data and value.
  void addString(DIE &Die, dwarf::Attribute Attribute, const StringRef Str);

  /// addLocalString - Add a string attribute data and value.
  void addLocalString(DIE &Die, dwarf::Attribute Attribute,
                      const StringRef Str);

  /// addExpr - Add a Dwarf expression attribute data and value.
  void addExpr(DIELoc &Die, dwarf::Form Form, const MCExpr *Expr);

  /// addLabel - Add a Dwarf label attribute data and value.
  void addLabel(DIE &Die, dwarf::Attribute Attribute, dwarf::Form Form,
                const MCSymbol *Label);

  void addLabel(DIELoc &Die, dwarf::Form Form, const MCSymbol *Label);

  /// addLocationList - Add a Dwarf loclistptr attribute data and value.
  void addLocationList(DIE &Die, dwarf::Attribute Attribute, unsigned Index);

  /// addSectionLabel - Add a Dwarf section label attribute data and value.
  ///
  void addSectionLabel(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label);

  /// addSectionOffset - Add an offset into a section attribute data and value.
  ///
  void addSectionOffset(DIE &Die, dwarf::Attribute Attribute, uint64_t Integer);

  /// addOpAddress - Add a dwarf op address data and value using the
  /// form given and an op of either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addOpAddress(DIELoc &Die, const MCSymbol *Label);

  /// addSectionDelta - Add a label delta attribute data and value.
  void addSectionDelta(DIE &Die, dwarf::Attribute Attribute, const MCSymbol *Hi,
                       const MCSymbol *Lo);

  /// addLabelDelta - Add a label delta attribute data and value.
  void addLabelDelta(DIE &Die, dwarf::Attribute Attribute, const MCSymbol *Hi,
                     const MCSymbol *Lo);

  /// addDIEEntry - Add a DIE attribute data and value.
  void addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIE &Entry);

  /// addDIEEntry - Add a DIE attribute data and value.
  void addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIEEntry *Entry);

  void addDIETypeSignature(DIE &Die, const DwarfTypeUnit &Type);

  /// addBlock - Add block data.
  void addBlock(DIE &Die, dwarf::Attribute Attribute, DIELoc *Block);

  /// addBlock - Add block data.
  void addBlock(DIE &Die, dwarf::Attribute Attribute, DIEBlock *Block);

  /// addSourceLine - Add location information to specified debug information
  /// entry.
  void addSourceLine(DIE &Die, unsigned Line, StringRef File,
                     StringRef Directory);
  void addSourceLine(DIE &Die, DIVariable V);
  void addSourceLine(DIE &Die, DIGlobalVariable G);
  void addSourceLine(DIE &Die, DISubprogram SP);
  void addSourceLine(DIE &Die, DIType Ty);
  void addSourceLine(DIE &Die, DINameSpace NS);
  void addSourceLine(DIE &Die, DIObjCProperty Ty);

  /// addAddress - Add an address attribute to a die based on the location
  /// provided.
  void addAddress(DIE &Die, dwarf::Attribute Attribute,
                  const MachineLocation &Location, bool Indirect = false);

  /// addConstantValue - Add constant value entry in variable DIE.
  void addConstantValue(DIE &Die, const MachineOperand &MO, DIType Ty);
  void addConstantValue(DIE &Die, const ConstantInt *CI, bool Unsigned);
  void addConstantValue(DIE &Die, const APInt &Val, bool Unsigned);

  /// addConstantFPValue - Add constant value entry in variable DIE.
  void addConstantFPValue(DIE &Die, const MachineOperand &MO);
  void addConstantFPValue(DIE &Die, const ConstantFP *CFP);

  /// addTemplateParams - Add template parameters in buffer.
  void addTemplateParams(DIE &Buffer, DIArray TParams);

  /// addRegisterOp - Add register operand.
  void addRegisterOp(DIELoc &TheDie, unsigned Reg);

  /// addRegisterOffset - Add register offset.
  void addRegisterOffset(DIELoc &TheDie, unsigned Reg, int64_t Offset);

  /// addComplexAddress - Start with the address based on the location provided,
  /// and generate the DWARF information necessary to find the actual variable
  /// (navigating the extra location information encoded in the type) based on
  /// the starting location.  Add the DWARF information to the die.
  void addComplexAddress(const DbgVariable &DV, DIE &Die,
                         dwarf::Attribute Attribute,
                         const MachineLocation &Location);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// addBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use addComplexAddress instead.
  void addBlockByrefAddress(const DbgVariable &DV, DIE &Die,
                            dwarf::Attribute Attribute,
                            const MachineLocation &Location);

  /// addVariableAddress - Add DW_AT_location attribute for a
  /// DbgVariable based on provided MachineLocation.
  void addVariableAddress(const DbgVariable &DV, DIE &Die,
                          MachineLocation Location);

  /// addType - Add a new type attribute to the specified entity. This takes
  /// and attribute parameter because DW_AT_friend attributes are also
  /// type references.
  void addType(DIE &Entity, DIType Ty,
               dwarf::Attribute Attribute = dwarf::DW_AT_type);

  /// getOrCreateNameSpace - Create a DIE for DINameSpace.
  DIE *getOrCreateNameSpace(DINameSpace NS);

  /// getOrCreateSubprogramDIE - Create new DIE using SP.
  DIE *getOrCreateSubprogramDIE(DISubprogram SP);

  /// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
  /// given DIType.
  DIE *getOrCreateTypeDIE(const MDNode *N);

  /// getOrCreateContextDIE - Get context owner's DIE.
  DIE *createTypeDIE(DICompositeType Ty);

  /// getOrCreateContextDIE - Get context owner's DIE.
  DIE *getOrCreateContextDIE(DIScope Context);

  /// constructContainingTypeDIEs - Construct DIEs for types that contain
  /// vtables.
  void constructContainingTypeDIEs();

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  std::unique_ptr<DIE> constructVariableDIE(DbgVariable &DV,
                                            bool isScopeAbstract);

  /// constructSubprogramArguments - Construct function argument DIEs.
  void constructSubprogramArguments(DIE &Buffer, DIArray Args);

  /// Create a DIE with the given Tag, add the DIE to its parent, and
  /// call insertDIE if MD is not null.
  DIE &createAndAddDIE(unsigned Tag, DIE &Parent,
                       DIDescriptor N = DIDescriptor());

  /// Compute the size of a header for this unit, not including the initial
  /// length field.
  virtual unsigned getHeaderSize() const {
    return sizeof(int16_t) + // DWARF version number
           sizeof(int32_t) + // Offset Into Abbrev. Section
           sizeof(int8_t);   // Pointer Size (in bytes)
  }

  /// Emit the header for this unit, not including the initial length field.
  virtual void emitHeader(const MCSymbol *ASectionSym) const;

  virtual DwarfCompileUnit &getCU() = 0;

  /// constructTypeDIE - Construct type DIE from DICompositeType.
  void constructTypeDIE(DIE &Buffer, DICompositeType CTy);

protected:
  /// getOrCreateStaticMemberDIE - Create new static data member DIE.
  DIE *getOrCreateStaticMemberDIE(DIDerivedType DT);

  /// Look up the source ID with the given directory and source file names. If
  /// none currently exists, create a new ID and insert it in the line table.
  virtual unsigned getOrCreateSourceID(StringRef File, StringRef Directory) = 0;

private:
  /// \brief Construct a DIE for the given DbgVariable without initializing the
  /// DbgVariable's DIE reference.
  std::unique_ptr<DIE> constructVariableDIEImpl(const DbgVariable &DV,
                                                bool isScopeAbstract);

  /// constructTypeDIE - Construct basic type die from DIBasicType.
  void constructTypeDIE(DIE &Buffer, DIBasicType BTy);

  /// constructTypeDIE - Construct derived type die from DIDerivedType.
  void constructTypeDIE(DIE &Buffer, DIDerivedType DTy);

  /// constructSubrangeDIE - Construct subrange DIE from DISubrange.
  void constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy);

  /// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
  void constructArrayTypeDIE(DIE &Buffer, DICompositeType CTy);

  /// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
  void constructEnumTypeDIE(DIE &Buffer, DICompositeType CTy);

  /// constructMemberDIE - Construct member DIE from DIDerivedType.
  void constructMemberDIE(DIE &Buffer, DIDerivedType DT);

  /// constructTemplateTypeParameterDIE - Construct new DIE for the given
  /// DITemplateTypeParameter.
  void constructTemplateTypeParameterDIE(DIE &Buffer,
                                         DITemplateTypeParameter TP);

  /// constructTemplateValueParameterDIE - Construct new DIE for the given
  /// DITemplateValueParameter.
  void constructTemplateValueParameterDIE(DIE &Buffer,
                                          DITemplateValueParameter TVP);

  /// getLowerBoundDefault - Return the default lower bound for an array. If the
  /// DWARF version doesn't handle the language, return -1.
  int64_t getDefaultLowerBound() const;

  /// getDIEEntry - Returns the debug information entry for the specified
  /// debug variable.
  DIEEntry *getDIEEntry(const MDNode *N) const {
    return MDNodeToDIEEntryMap.lookup(N);
  }

  /// insertDIEEntry - Insert debug information entry into the map.
  void insertDIEEntry(const MDNode *N, DIEEntry *E) {
    MDNodeToDIEEntryMap.insert(std::make_pair(N, E));
  }

  // getIndexTyDie - Get an anonymous type for index type.
  DIE *getIndexTyDie() { return IndexTyDie; }

  // setIndexTyDie - Set D as anonymous type for index which can be reused
  // later.
  void setIndexTyDie(DIE *D) { IndexTyDie = D; }

  /// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *createDIEEntry(DIE &Entry);

  /// resolve - Look in the DwarfDebug map for the MDNode that
  /// corresponds to the reference.
  template <typename T> T resolve(DIRef<T> Ref) const {
    return DD->resolve(Ref);
  }

  /// If this is a named finished type then include it in the list of types for
  /// the accelerator tables.
  void updateAcceleratorTables(DIScope Context, DIType Ty, const DIE &TyDIE);
};

class DwarfCompileUnit : public DwarfUnit {
  /// The attribute index of DW_AT_stmt_list in the compile unit DIE, avoiding
  /// the need to search for it in applyStmtList.
  unsigned stmtListIndex;

public:
  DwarfCompileUnit(unsigned UID, DIE *D, DICompileUnit Node, AsmPrinter *A,
                   DwarfDebug *DW, DwarfFile *DWU);

  void initStmtList(MCSymbol *DwarfLineSectionSym);

  /// Apply the DW_AT_stmt_list from this compile unit to the specified DIE.
  void applyStmtList(DIE &D);

  /// createGlobalVariableDIE - create global variable DIE.
  void createGlobalVariableDIE(DIGlobalVariable GV);

  /// addLabelAddress - Add a dwarf label attribute data and value using
  /// either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label);

  /// addLocalLabelAddress - Add a dwarf label attribute data and value using
  /// DW_FORM_addr only.
  void addLocalLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                            const MCSymbol *Label);

  DwarfCompileUnit &getCU() override { return *this; }

  unsigned getOrCreateSourceID(StringRef FileName, StringRef DirName) override;
};

class DwarfTypeUnit : public DwarfUnit {
private:
  uint64_t TypeSignature;
  const DIE *Ty;
  DwarfCompileUnit &CU;
  MCDwarfDwoLineTable *SplitLineTable;

public:
  DwarfTypeUnit(unsigned UID, DIE *D, DwarfCompileUnit &CU, AsmPrinter *A,
                DwarfDebug *DW, DwarfFile *DWU,
                MCDwarfDwoLineTable *SplitLineTable = nullptr);

  void setTypeSignature(uint64_t Signature) { TypeSignature = Signature; }
  uint64_t getTypeSignature() const { return TypeSignature; }
  void setType(const DIE *Ty) { this->Ty = Ty; }

  /// Emit the header for this unit, not including the initial length field.
  void emitHeader(const MCSymbol *ASectionSym) const override;
  unsigned getHeaderSize() const override {
    return DwarfUnit::getHeaderSize() + sizeof(uint64_t) + // Type Signature
           sizeof(uint32_t);                               // Type DIE Offset
  }
  void initSection(const MCSection *Section);
  DwarfCompileUnit &getCU() override { return CU; }

protected:
  unsigned getOrCreateSourceID(StringRef File, StringRef Directory) override;
};
} // end llvm namespace
#endif
