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

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFUNIT_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFUNIT_H

#include "DwarfDebug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"

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
  RangeSpanList(MCSymbol *Sym, SmallVector<RangeSpan, 2> Ranges)
      : RangeSym(Sym), Ranges(std::move(Ranges)) {}
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
  DIE UnitDie;

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

  /// DIEBlocks - A list of all the DIEBlocks in use.
  std::vector<DIEBlock *> DIEBlocks;
  
  /// DIELocs - A list of all the DIELocs in use.
  std::vector<DIELoc *> DIELocs;

  /// ContainingTypeMap - This map is used to keep track of subprogram DIEs that
  /// need DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, const MDNode *> ContainingTypeMap;

  // DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  // DIEIntegerOne - A preallocated DIEValue because 1 is used frequently.
  DIEInteger *DIEIntegerOne;

  /// The section this unit will be emitted in.
  const MCSection *Section;

  DwarfUnit(unsigned UID, dwarf::Tag, DICompileUnit CU, AsmPrinter *A,
            DwarfDebug *DW, DwarfFile *DWU);

  void initSection(const MCSection *Section);

  /// Add a string attribute data and value.
  void addLocalString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  void addIndexedString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  bool applySubprogramDefinitionAttributes(DISubprogram SP, DIE &SPDie);

public:
  virtual ~DwarfUnit();

  const MCSection *getSection() const {
    assert(Section);
    return Section;
  }

  // Accessors.
  AsmPrinter* getAsmPrinter() const { return Asm; }
  unsigned getUniqueID() const { return UniqueID; }
  uint16_t getLanguage() const { return CUNode.getLanguage(); }
  DICompileUnit getCUNode() const { return CUNode; }
  DIE &getUnitDie() { return UnitDie; }

  unsigned getDebugInfoOffset() const { return DebugInfoOffset; }
  void setDebugInfoOffset(unsigned DbgInfoOff) { DebugInfoOffset = DbgInfoOff; }

  /// hasContent - Return true if this compile unit has something to write out.
  bool hasContent() const { return !UnitDie.getChildren().empty(); }

  /// getParentContextString - Get a string containing the language specific
  /// context for a global name.
  std::string getParentContextString(DIScope Context) const;

  /// Add a new global name to the compile unit.
  virtual void addGlobalName(StringRef Name, DIE &Die, DIScope Context) {}

  /// Add a new global type to the compile unit.
  virtual void addGlobalType(DIType Ty, const DIE &Die, DIScope Context) {}

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
  void addString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  /// addLabel - Add a Dwarf label attribute data and value.
  void addLabel(DIE &Die, dwarf::Attribute Attribute, dwarf::Form Form,
                const MCSymbol *Label);

  void addLabel(DIELoc &Die, dwarf::Form Form, const MCSymbol *Label);

  /// addSectionOffset - Add an offset into a section attribute data and value.
  ///
  void addSectionOffset(DIE &Die, dwarf::Attribute Attribute, uint64_t Integer);

  /// addOpAddress - Add a dwarf op address data and value using the
  /// form given and an op of either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addOpAddress(DIELoc &Die, const MCSymbol *Label);

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

  /// addConstantValue - Add constant value entry in variable DIE.
  void addConstantValue(DIE &Die, const MachineOperand &MO, DIType Ty);
  void addConstantValue(DIE &Die, const ConstantInt *CI, DIType Ty);
  void addConstantValue(DIE &Die, const APInt &Val, DIType Ty);
  void addConstantValue(DIE &Die, const APInt &Val, bool Unsigned);
  void addConstantValue(DIE &Die, bool Unsigned, uint64_t Val);

  /// addConstantFPValue - Add constant value entry in variable DIE.
  void addConstantFPValue(DIE &Die, const MachineOperand &MO);
  void addConstantFPValue(DIE &Die, const ConstantFP *CFP);

  /// addTemplateParams - Add template parameters in buffer.
  void addTemplateParams(DIE &Buffer, DIArray TParams);

  /// \brief Add register operand.
  /// \returns false if the register does not exist, e.g., because it was never
  /// materialized.
  bool addRegisterOpPiece(DIELoc &TheDie, unsigned Reg,
                          unsigned SizeInBits = 0, unsigned OffsetInBits = 0);

  /// \brief Add register offset.
  /// \returns false if the register does not exist, e.g., because it was never
  /// materialized.
  bool addRegisterOffset(DIELoc &TheDie, unsigned Reg, int64_t Offset);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// addBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use addComplexAddress instead.
  void addBlockByrefAddress(const DbgVariable &DV, DIE &Die,
                            dwarf::Attribute Attribute,
                            const MachineLocation &Location);

  /// addType - Add a new type attribute to the specified entity. This takes
  /// and attribute parameter because DW_AT_friend attributes are also
  /// type references.
  void addType(DIE &Entity, DIType Ty,
               dwarf::Attribute Attribute = dwarf::DW_AT_type);

  /// getOrCreateNameSpace - Create a DIE for DINameSpace.
  DIE *getOrCreateNameSpace(DINameSpace NS);

  /// getOrCreateSubprogramDIE - Create new DIE using SP.
  DIE *getOrCreateSubprogramDIE(DISubprogram SP, bool Minimal = false);

  void applySubprogramAttributes(DISubprogram SP, DIE &SPDie,
                                 bool Minimal = false);

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

  /// constructSubprogramArguments - Construct function argument DIEs.
  void constructSubprogramArguments(DIE &Buffer, DITypeArray Args);

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
  virtual void emitHeader(const MCSymbol *ASectionSym);

  virtual DwarfCompileUnit &getCU() = 0;

  /// constructTypeDIE - Construct type DIE from DICompositeType.
  void constructTypeDIE(DIE &Buffer, DICompositeType CTy);

protected:
  /// getOrCreateStaticMemberDIE - Create new static data member DIE.
  DIE *getOrCreateStaticMemberDIE(DIDerivedType DT);

  /// Look up the source ID with the given directory and source file names. If
  /// none currently exists, create a new ID and insert it in the line table.
  virtual unsigned getOrCreateSourceID(StringRef File, StringRef Directory) = 0;

  /// resolve - Look in the DwarfDebug map for the MDNode that
  /// corresponds to the reference.
  template <typename T> T resolve(DIRef<T> Ref) const {
    return DD->resolve(Ref);
  }

private:
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
  DIE *getIndexTyDie();

  // setIndexTyDie - Set D as anonymous type for index which can be reused
  // later.
  void setIndexTyDie(DIE *D) { IndexTyDie = D; }

  /// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *createDIEEntry(DIE &Entry);

  /// If this is a named finished type then include it in the list of types for
  /// the accelerator tables.
  void updateAcceleratorTables(DIScope Context, DIType Ty, const DIE &TyDIE);

  virtual bool isDwoUnit() const = 0;
};

class DwarfTypeUnit : public DwarfUnit {
  uint64_t TypeSignature;
  const DIE *Ty;
  DwarfCompileUnit &CU;
  MCDwarfDwoLineTable *SplitLineTable;

  unsigned getOrCreateSourceID(StringRef File, StringRef Directory) override;
  bool isDwoUnit() const override;

public:
  DwarfTypeUnit(unsigned UID, DwarfCompileUnit &CU, AsmPrinter *A,
                DwarfDebug *DW, DwarfFile *DWU,
                MCDwarfDwoLineTable *SplitLineTable = nullptr);

  void setTypeSignature(uint64_t Signature) { TypeSignature = Signature; }
  uint64_t getTypeSignature() const { return TypeSignature; }
  void setType(const DIE *Ty) { this->Ty = Ty; }

  /// Emit the header for this unit, not including the initial length field.
  void emitHeader(const MCSymbol *ASectionSym) override;
  unsigned getHeaderSize() const override {
    return DwarfUnit::getHeaderSize() + sizeof(uint64_t) + // Type Signature
           sizeof(uint32_t);                               // Type DIE Offset
  }
  using DwarfUnit::initSection;
  DwarfCompileUnit &getCU() override { return CU; }
};
} // end llvm namespace
#endif
