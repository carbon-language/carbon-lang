//===- MCAssembler.h - Object File Generation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASSEMBLER_H
#define LLVM_MC_MCASSEMBLER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/Casting.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/System/DataTypes.h"
#include <vector> // FIXME: Shouldn't be needed.

namespace llvm {
class raw_ostream;
class MCAsmLayout;
class MCAssembler;
class MCContext;
class MCCodeEmitter;
class MCExpr;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCSectionData;
class MCSymbol;
class MCValue;
class TargetAsmBackend;

/// MCAsmFixup - Represent a fixed size region of bytes inside some fragment
/// which needs to be rewritten. This region will either be rewritten by the
/// assembler or cause a relocation entry to be generated.
//
// FIXME: This should probably just be merged with MCFixup.
class MCAsmFixup {
public:
  /// Offset - The offset inside the fragment which needs to be rewritten.
  uint64_t Offset;

  /// Value - The expression to eventually write into the fragment.
  const MCExpr *Value;

  /// Kind - The fixup kind.
  MCFixupKind Kind;

public:
  MCAsmFixup(uint64_t _Offset, const MCExpr &_Value, MCFixupKind _Kind)
    : Offset(_Offset), Value(&_Value), Kind(_Kind) {}
};

class MCFragment : public ilist_node<MCFragment> {
  friend class MCAsmLayout;

  MCFragment(const MCFragment&);     // DO NOT IMPLEMENT
  void operator=(const MCFragment&); // DO NOT IMPLEMENT

public:
  enum FragmentType {
    FT_Align,
    FT_Data,
    FT_Fill,
    FT_Inst,
    FT_Org,
    FT_ZeroFill
  };

private:
  FragmentType Kind;

  /// Parent - The data for the section this fragment is in.
  MCSectionData *Parent;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// Offset - The offset of this fragment in its section. This is ~0 until
  /// initialized.
  uint64_t Offset;

  /// EffectiveSize - The compute size of this section. This is ~0 until
  /// initialized.
  uint64_t EffectiveSize;

  /// Ordinal - The global index of this fragment. This is the index across all
  /// sections, not just the parent section.
  unsigned Ordinal;

  /// @}

protected:
  MCFragment(FragmentType _Kind, MCSectionData *_Parent = 0);

public:
  // Only for sentinel.
  MCFragment();
  virtual ~MCFragment();

  FragmentType getKind() const { return Kind; }

  MCSectionData *getParent() const { return Parent; }
  void setParent(MCSectionData *Value) { Parent = Value; }

  unsigned getOrdinal() const { return Ordinal; }
  void setOrdinal(unsigned Value) { Ordinal = Value; }

  static bool classof(const MCFragment *O) { return true; }

  virtual void dump();
};

class MCDataFragment : public MCFragment {
  SmallString<32> Contents;

  /// Fixups - The list of fixups in this fragment.
  std::vector<MCAsmFixup> Fixups;

public:
  typedef std::vector<MCAsmFixup>::const_iterator const_fixup_iterator;
  typedef std::vector<MCAsmFixup>::iterator fixup_iterator;

public:
  MCDataFragment(MCSectionData *SD = 0) : MCFragment(FT_Data, SD) {}

  /// @name Accessors
  /// @{

  SmallString<32> &getContents() { return Contents; }
  const SmallString<32> &getContents() const { return Contents; }

  /// @}
  /// @name Fixup Access
  /// @{

  void addFixup(MCAsmFixup Fixup) {
    // Enforce invariant that fixups are in offset order.
    assert((Fixups.empty() || Fixup.Offset > Fixups.back().Offset) &&
           "Fixups must be added in order!");
    Fixups.push_back(Fixup);
  }

  std::vector<MCAsmFixup> &getFixups() { return Fixups; }
  const std::vector<MCAsmFixup> &getFixups() const { return Fixups; }

  fixup_iterator fixup_begin() { return Fixups.begin(); }
  const_fixup_iterator fixup_begin() const { return Fixups.begin(); }

  fixup_iterator fixup_end() {return Fixups.end();}
  const_fixup_iterator fixup_end() const {return Fixups.end();}

  size_t fixup_size() const { return Fixups.size(); }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Data;
  }
  static bool classof(const MCDataFragment *) { return true; }

  virtual void dump();
};

// FIXME: This current incarnation of MCInstFragment doesn't make much sense, as
// it is almost entirely a duplicate of MCDataFragment. If we decide to stick
// with this approach (as opposed to making MCInstFragment a very light weight
// object with just the MCInst and a code size, then we should just change
// MCDataFragment to have an optional MCInst at its end.
class MCInstFragment : public MCFragment {
  /// Inst - The instruction this is a fragment for.
  MCInst Inst;

  /// InstSize - The size of the currently encoded instruction.
  SmallString<8> Code;

  /// Fixups - The list of fixups in this fragment.
  SmallVector<MCAsmFixup, 1> Fixups;

public:
  typedef SmallVectorImpl<MCAsmFixup>::const_iterator const_fixup_iterator;
  typedef SmallVectorImpl<MCAsmFixup>::iterator fixup_iterator;

public:
  MCInstFragment(MCInst _Inst, MCSectionData *SD = 0)
    : MCFragment(FT_Inst, SD), Inst(_Inst) {
  }

  /// @name Accessors
  /// @{

  SmallVectorImpl<char> &getCode() { return Code; }
  const SmallVectorImpl<char> &getCode() const { return Code; }

  unsigned getInstSize() const { return Code.size(); }

  MCInst &getInst() { return Inst; }
  const MCInst &getInst() const { return Inst; }

  void setInst(MCInst Value) { Inst = Value; }

  /// @}
  /// @name Fixup Access
  /// @{

  SmallVectorImpl<MCAsmFixup> &getFixups() { return Fixups; }
  const SmallVectorImpl<MCAsmFixup> &getFixups() const { return Fixups; }

  fixup_iterator fixup_begin() { return Fixups.begin(); }
  const_fixup_iterator fixup_begin() const { return Fixups.begin(); }

  fixup_iterator fixup_end() {return Fixups.end();}
  const_fixup_iterator fixup_end() const {return Fixups.end();}

  size_t fixup_size() const { return Fixups.size(); }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Inst;
  }
  static bool classof(const MCInstFragment *) { return true; }

  virtual void dump();
};

class MCAlignFragment : public MCFragment {
  /// Alignment - The alignment to ensure, in bytes.
  unsigned Alignment;

  /// Value - Value to use for filling padding bytes.
  int64_t Value;

  /// ValueSize - The size of the integer (in bytes) of \arg Value.
  unsigned ValueSize;

  /// MaxBytesToEmit - The maximum number of bytes to emit; if the alignment
  /// cannot be satisfied in this width then this fragment is ignored.
  unsigned MaxBytesToEmit;

  /// EmitNops - true when aligning code and optimal nops to be used for
  /// filling.
  bool EmitNops;

public:
  MCAlignFragment(unsigned _Alignment, int64_t _Value, unsigned _ValueSize,
                  unsigned _MaxBytesToEmit, bool _EmitNops,
		  MCSectionData *SD = 0)
    : MCFragment(FT_Align, SD), Alignment(_Alignment),
      Value(_Value),ValueSize(_ValueSize),
      MaxBytesToEmit(_MaxBytesToEmit), EmitNops(_EmitNops) {}

  /// @name Accessors
  /// @{

  unsigned getAlignment() const { return Alignment; }

  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  unsigned getMaxBytesToEmit() const { return MaxBytesToEmit; }

  unsigned getEmitNops() const { return EmitNops; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Align;
  }
  static bool classof(const MCAlignFragment *) { return true; }

  virtual void dump();
};

class MCFillFragment : public MCFragment {
  /// Value - Value to use for filling bytes.
  int64_t Value;

  /// ValueSize - The size (in bytes) of \arg Value to use when filling.
  unsigned ValueSize;

  /// Count - The number of copies of \arg Value to insert.
  uint64_t Count;

public:
  MCFillFragment(int64_t _Value, unsigned _ValueSize, uint64_t _Count,
                 MCSectionData *SD = 0)
    : MCFragment(FT_Fill, SD),
      Value(_Value), ValueSize(_ValueSize), Count(_Count) {}

  /// @name Accessors
  /// @{

  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  uint64_t getCount() const { return Count; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Fill;
  }
  static bool classof(const MCFillFragment *) { return true; }

  virtual void dump();
};

class MCOrgFragment : public MCFragment {
  /// Offset - The offset this fragment should start at.
  const MCExpr *Offset;

  /// Value - Value to use for filling bytes.
  int8_t Value;

public:
  MCOrgFragment(const MCExpr &_Offset, int8_t _Value, MCSectionData *SD = 0)
    : MCFragment(FT_Org, SD),
      Offset(&_Offset), Value(_Value) {}

  /// @name Accessors
  /// @{

  const MCExpr &getOffset() const { return *Offset; }

  uint8_t getValue() const { return Value; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Org;
  }
  static bool classof(const MCOrgFragment *) { return true; }

  virtual void dump();
};

/// MCZeroFillFragment - Represent data which has a fixed size and alignment,
/// but requires no physical space in the object file.
class MCZeroFillFragment : public MCFragment {
  /// Size - The size of this fragment.
  uint64_t Size;

  /// Alignment - The alignment for this fragment.
  unsigned Alignment;

public:
  MCZeroFillFragment(uint64_t _Size, unsigned _Alignment, MCSectionData *SD = 0)
    : MCFragment(FT_ZeroFill, SD),
      Size(_Size), Alignment(_Alignment) {}

  /// @name Accessors
  /// @{

  uint64_t getSize() const { return Size; }

  unsigned getAlignment() const { return Alignment; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_ZeroFill;
  }
  static bool classof(const MCZeroFillFragment *) { return true; }

  virtual void dump();
};

// FIXME: Should this be a separate class, or just merged into MCSection? Since
// we anticipate the fast path being through an MCAssembler, the only reason to
// keep it out is for API abstraction.
class MCSectionData : public ilist_node<MCSectionData> {
  friend class MCAsmLayout;

  MCSectionData(const MCSectionData&);  // DO NOT IMPLEMENT
  void operator=(const MCSectionData&); // DO NOT IMPLEMENT

public:
  typedef iplist<MCFragment> FragmentListType;

  typedef FragmentListType::const_iterator const_iterator;
  typedef FragmentListType::iterator iterator;

  typedef FragmentListType::const_reverse_iterator const_reverse_iterator;
  typedef FragmentListType::reverse_iterator reverse_iterator;

private:
  iplist<MCFragment> Fragments;
  const MCSection *Section;

  /// Ordinal - The section index in the assemblers section list.
  unsigned Ordinal;

  /// Alignment - The maximum alignment seen in this section.
  unsigned Alignment;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// Address - The computed address of this section. This is ~0 until
  /// initialized.
  uint64_t Address;

  /// Size - The content size of this section. This is ~0 until initialized.
  uint64_t Size;

  /// FileSize - The size of this section in the object file. This is ~0 until
  /// initialized.
  uint64_t FileSize;

  /// HasInstructions - Whether this section has had instructions emitted into
  /// it.
  unsigned HasInstructions : 1;

  /// @}

public:
  // Only for use as sentinel.
  MCSectionData();
  MCSectionData(const MCSection &Section, MCAssembler *A = 0);

  const MCSection &getSection() const { return *Section; }

  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Value) { Alignment = Value; }

  bool hasInstructions() const { return HasInstructions; }
  void setHasInstructions(bool Value) { HasInstructions = Value; }

  unsigned getOrdinal() const { return Ordinal; }
  void setOrdinal(unsigned Value) { Ordinal = Value; }

  /// @name Fragment Access
  /// @{

  const FragmentListType &getFragmentList() const { return Fragments; }
  FragmentListType &getFragmentList() { return Fragments; }

  iterator begin() { return Fragments.begin(); }
  const_iterator begin() const { return Fragments.begin(); }

  iterator end() { return Fragments.end(); }
  const_iterator end() const { return Fragments.end(); }

  reverse_iterator rbegin() { return Fragments.rbegin(); }
  const_reverse_iterator rbegin() const { return Fragments.rbegin(); }

  reverse_iterator rend() { return Fragments.rend(); }
  const_reverse_iterator rend() const { return Fragments.rend(); }

  size_t size() const { return Fragments.size(); }

  bool empty() const { return Fragments.empty(); }

  void dump();

  /// @}
};

// FIXME: Same concerns as with SectionData.
class MCSymbolData : public ilist_node<MCSymbolData> {
public:
  const MCSymbol *Symbol;

  /// Fragment - The fragment this symbol's value is relative to, if any.
  MCFragment *Fragment;

  /// Offset - The offset to apply to the fragment address to form this symbol's
  /// value.
  uint64_t Offset;

  /// IsExternal - True if this symbol is visible outside this translation
  /// unit.
  unsigned IsExternal : 1;

  /// IsPrivateExtern - True if this symbol is private extern.
  unsigned IsPrivateExtern : 1;

  /// CommonSize - The size of the symbol, if it is 'common', or 0.
  //
  // FIXME: Pack this in with other fields? We could put it in offset, since a
  // common symbol can never get a definition.
  uint64_t CommonSize;

  /// CommonAlign - The alignment of the symbol, if it is 'common'.
  //
  // FIXME: Pack this in with other fields?
  unsigned CommonAlign;

  /// Flags - The Flags field is used by object file implementations to store
  /// additional per symbol information which is not easily classified.
  uint32_t Flags;

  /// Index - Index field, for use by the object file implementation.
  uint64_t Index;

public:
  // Only for use as sentinel.
  MCSymbolData();
  MCSymbolData(const MCSymbol &_Symbol, MCFragment *_Fragment, uint64_t _Offset,
               MCAssembler *A = 0);

  /// @name Accessors
  /// @{

  const MCSymbol &getSymbol() const { return *Symbol; }

  MCFragment *getFragment() const { return Fragment; }
  void setFragment(MCFragment *Value) { Fragment = Value; }

  uint64_t getOffset() const { return Offset; }
  void setOffset(uint64_t Value) { Offset = Value; }

  /// @}
  /// @name Symbol Attributes
  /// @{

  bool isExternal() const { return IsExternal; }
  void setExternal(bool Value) { IsExternal = Value; }

  bool isPrivateExtern() const { return IsPrivateExtern; }
  void setPrivateExtern(bool Value) { IsPrivateExtern = Value; }

  /// isCommon - Is this a 'common' symbol.
  bool isCommon() const { return CommonSize != 0; }

  /// setCommon - Mark this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  void setCommon(uint64_t Size, unsigned Align) {
    CommonSize = Size;
    CommonAlign = Align;
  }

  /// getCommonSize - Return the size of a 'common' symbol.
  uint64_t getCommonSize() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonSize;
  }

  /// getCommonAlignment - Return the alignment of a 'common' symbol.
  unsigned getCommonAlignment() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonAlign;
  }

  /// getFlags - Get the (implementation defined) symbol flags.
  uint32_t getFlags() const { return Flags; }

  /// setFlags - Set the (implementation defined) symbol flags.
  void setFlags(uint32_t Value) { Flags = Value; }

  /// getIndex - Get the (implementation defined) index.
  uint64_t getIndex() const { return Index; }

  /// setIndex - Set the (implementation defined) index.
  void setIndex(uint64_t Value) { Index = Value; }

  /// @}

  void dump();
};

// FIXME: This really doesn't belong here. See comments below.
struct IndirectSymbolData {
  MCSymbol *Symbol;
  MCSectionData *SectionData;
};

class MCAssembler {
  friend class MCAsmLayout;

public:
  typedef iplist<MCSectionData> SectionDataListType;
  typedef iplist<MCSymbolData> SymbolDataListType;

  typedef SectionDataListType::const_iterator const_iterator;
  typedef SectionDataListType::iterator iterator;

  typedef SymbolDataListType::const_iterator const_symbol_iterator;
  typedef SymbolDataListType::iterator symbol_iterator;

  typedef std::vector<IndirectSymbolData>::const_iterator
    const_indirect_symbol_iterator;
  typedef std::vector<IndirectSymbolData>::iterator indirect_symbol_iterator;

private:
  MCAssembler(const MCAssembler&);    // DO NOT IMPLEMENT
  void operator=(const MCAssembler&); // DO NOT IMPLEMENT

  MCContext &Context;

  TargetAsmBackend &Backend;

  MCCodeEmitter &Emitter;

  raw_ostream &OS;

  iplist<MCSectionData> Sections;

  iplist<MCSymbolData> Symbols;

  /// The map of sections to their associated assembler backend data.
  //
  // FIXME: Avoid this indirection?
  DenseMap<const MCSection*, MCSectionData*> SectionMap;

  /// The map of symbols to their associated assembler backend data.
  //
  // FIXME: Avoid this indirection?
  DenseMap<const MCSymbol*, MCSymbolData*> SymbolMap;

  std::vector<IndirectSymbolData> IndirectSymbols;

  unsigned RelaxAll : 1;
  unsigned SubsectionsViaSymbols : 1;

private:
  /// Evaluate a fixup to a relocatable expression and the value which should be
  /// placed into the fixup.
  ///
  /// \param Layout The layout to use for evaluation.
  /// \param Fixup The fixup to evaluate.
  /// \param DF The fragment the fixup is inside.
  /// \param Target [out] On return, the relocatable expression the fixup
  /// evaluates to.
  /// \param Value [out] On return, the value of the fixup as currently layed
  /// out.
  /// \return Whether the fixup value was fully resolved. This is true if the
  /// \arg Value result is fixed, otherwise the value may change due to
  /// relocation.
  bool EvaluateFixup(const MCAsmLayout &Layout,
                     const MCAsmFixup &Fixup, const MCFragment *DF,
                     MCValue &Target, uint64_t &Value) const;

  /// Check whether a fixup can be satisfied, or whether it needs to be relaxed
  /// (increased in size, in order to hold its value correctly).
  bool FixupNeedsRelaxation(const MCAsmFixup &Fixup, const MCFragment *DF,
                            const MCAsmLayout &Layout) const;

  /// Check whether the given fragment needs relaxation.
  bool FragmentNeedsRelaxation(const MCInstFragment *IF,
                               const MCAsmLayout &Layout) const;

  /// LayoutSection - Assign the section the given \arg StartAddress, and then
  /// assign offsets and sizes to the fragments in the section \arg SD, and
  /// update the section size.
  ///
  /// \return The address at the end of the section, for use in laying out the
  /// succeeding section.
  uint64_t LayoutSection(MCSectionData &SD, MCAsmLayout &Layout,
                         uint64_t StartAddress);

  /// LayoutOnce - Perform one layout iteration and return true if any offsets
  /// were adjusted.
  bool LayoutOnce(MCAsmLayout &Layout);

  /// FinishLayout - Finalize a layout, including fragment lowering.
  void FinishLayout(MCAsmLayout &Layout);

public:
  /// Find the symbol which defines the atom containing given address, inside
  /// the given section, or null if there is no such symbol.
  //
  // FIXME-PERF: Eliminate this, it is very slow.
  const MCSymbolData *getAtomForAddress(const MCAsmLayout &Layout,
                                        const MCSectionData *Section,
                                        uint64_t Address) const;

  /// Find the symbol which defines the atom containing the given symbol, or
  /// null if there is no such symbol.
  //
  // FIXME-PERF: Eliminate this, it is very slow.
  const MCSymbolData *getAtom(const MCAsmLayout &Layout,
                              const MCSymbolData *Symbol) const;

  /// Check whether a particular symbol is visible to the linker and is required
  /// in the symbol table, or whether it can be discarded by the assembler. This
  /// also effects whether the assembler treats the label as potentially
  /// defining a separate atom.
  bool isSymbolLinkerVisible(const MCSymbolData *SD) const;

  /// Emit the section contents using the given object writer.
  //
  // FIXME: Should MCAssembler always have a reference to the object writer?
  void WriteSectionData(const MCSectionData *Section, const MCAsmLayout &Layout,
                        MCObjectWriter *OW) const;

public:
  /// Construct a new assembler instance.
  ///
  /// \arg OS - The stream to output to.
  //
  // FIXME: How are we going to parameterize this? Two obvious options are stay
  // concrete and require clients to pass in a target like object. The other
  // option is to make this abstract, and have targets provide concrete
  // implementations as we do with AsmParser.
  MCAssembler(MCContext &_Context, TargetAsmBackend &_Backend,
              MCCodeEmitter &_Emitter, raw_ostream &OS);
  ~MCAssembler();

  MCContext &getContext() const { return Context; }

  TargetAsmBackend &getBackend() const { return Backend; }

  MCCodeEmitter &getEmitter() const { return Emitter; }

  /// Finish - Do final processing and write the object to the output stream.
  void Finish();

  // FIXME: This does not belong here.
  bool getSubsectionsViaSymbols() const {
    return SubsectionsViaSymbols;
  }
  void setSubsectionsViaSymbols(bool Value) {
    SubsectionsViaSymbols = Value;
  }

  bool getRelaxAll() const { return RelaxAll; }
  void setRelaxAll(bool Value) { RelaxAll = Value; }

  /// @name Section List Access
  /// @{

  const SectionDataListType &getSectionList() const { return Sections; }
  SectionDataListType &getSectionList() { return Sections; }

  iterator begin() { return Sections.begin(); }
  const_iterator begin() const { return Sections.begin(); }

  iterator end() { return Sections.end(); }
  const_iterator end() const { return Sections.end(); }

  size_t size() const { return Sections.size(); }

  /// @}
  /// @name Symbol List Access
  /// @{

  const SymbolDataListType &getSymbolList() const { return Symbols; }
  SymbolDataListType &getSymbolList() { return Symbols; }

  symbol_iterator symbol_begin() { return Symbols.begin(); }
  const_symbol_iterator symbol_begin() const { return Symbols.begin(); }

  symbol_iterator symbol_end() { return Symbols.end(); }
  const_symbol_iterator symbol_end() const { return Symbols.end(); }

  size_t symbol_size() const { return Symbols.size(); }

  /// @}
  /// @name Indirect Symbol List Access
  /// @{

  // FIXME: This is a total hack, this should not be here. Once things are
  // factored so that the streamer has direct access to the .o writer, it can
  // disappear.
  std::vector<IndirectSymbolData> &getIndirectSymbols() {
    return IndirectSymbols;
  }

  indirect_symbol_iterator indirect_symbol_begin() {
    return IndirectSymbols.begin();
  }
  const_indirect_symbol_iterator indirect_symbol_begin() const {
    return IndirectSymbols.begin();
  }

  indirect_symbol_iterator indirect_symbol_end() {
    return IndirectSymbols.end();
  }
  const_indirect_symbol_iterator indirect_symbol_end() const {
    return IndirectSymbols.end();
  }

  size_t indirect_symbol_size() const { return IndirectSymbols.size(); }

  /// @}
  /// @name Backend Data Access
  /// @{

  MCSectionData &getSectionData(const MCSection &Section) const {
    MCSectionData *Entry = SectionMap.lookup(&Section);
    assert(Entry && "Missing section data!");
    return *Entry;
  }

  MCSectionData &getOrCreateSectionData(const MCSection &Section,
                                        bool *Created = 0) {
    MCSectionData *&Entry = SectionMap[&Section];

    if (Created) *Created = !Entry;
    if (!Entry)
      Entry = new MCSectionData(Section, this);

    return *Entry;
  }

  MCSymbolData &getSymbolData(const MCSymbol &Symbol) const {
    MCSymbolData *Entry = SymbolMap.lookup(&Symbol);
    assert(Entry && "Missing symbol data!");
    return *Entry;
  }

  MCSymbolData &getOrCreateSymbolData(const MCSymbol &Symbol,
                                      bool *Created = 0) {
    MCSymbolData *&Entry = SymbolMap[&Symbol];

    if (Created) *Created = !Entry;
    if (!Entry)
      Entry = new MCSymbolData(Symbol, 0, 0, this);

    return *Entry;
  }

  /// @}

  void dump();
};

} // end namespace llvm

#endif
