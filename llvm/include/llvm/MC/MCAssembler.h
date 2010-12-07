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
#include "llvm/Support/DataTypes.h"
#include <vector> // FIXME: Shouldn't be needed.

namespace llvm {
class raw_ostream;
class MCAsmLayout;
class MCAssembler;
class MCBinaryExpr;
class MCContext;
class MCCodeEmitter;
class MCExpr;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCSectionData;
class MCSymbol;
class MCSymbolData;
class MCValue;
class TargetAsmBackend;

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
    FT_Dwarf,
    FT_LEB
  };

private:
  FragmentType Kind;

  /// Parent - The data for the section this fragment is in.
  MCSectionData *Parent;

  /// Atom - The atom this fragment is in, as represented by it's defining
  /// symbol. Atom's are only used by backends which set
  /// \see MCAsmBackend::hasReliableSymbolDifference().
  MCSymbolData *Atom;

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

  /// LayoutOrder - The global layout order of this fragment. This is the index
  /// across all fragments in the file, not just within the section.
  unsigned LayoutOrder;

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

  MCSymbolData *getAtom() const { return Atom; }
  void setAtom(MCSymbolData *Value) { Atom = Value; }

  unsigned getLayoutOrder() const { return LayoutOrder; }
  void setLayoutOrder(unsigned Value) { LayoutOrder = Value; }

  static bool classof(const MCFragment *O) { return true; }

  void dump();
};

class MCDataFragment : public MCFragment {
  SmallString<32> Contents;

  /// Fixups - The list of fixups in this fragment.
  std::vector<MCFixup> Fixups;

public:
  typedef std::vector<MCFixup>::const_iterator const_fixup_iterator;
  typedef std::vector<MCFixup>::iterator fixup_iterator;

public:
  MCDataFragment(MCSectionData *SD = 0) : MCFragment(FT_Data, SD) {}

  /// @name Accessors
  /// @{

  SmallString<32> &getContents() { return Contents; }
  const SmallString<32> &getContents() const { return Contents; }

  /// @}
  /// @name Fixup Access
  /// @{

  void addFixup(MCFixup Fixup) {
    // Enforce invariant that fixups are in offset order.
    assert((Fixups.empty() || Fixup.getOffset() > Fixups.back().getOffset()) &&
           "Fixups must be added in order!");
    Fixups.push_back(Fixup);
  }

  std::vector<MCFixup> &getFixups() { return Fixups; }
  const std::vector<MCFixup> &getFixups() const { return Fixups; }

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
};

// FIXME: This current incarnation of MCInstFragment doesn't make much sense, as
// it is almost entirely a duplicate of MCDataFragment. If we decide to stick
// with this approach (as opposed to making MCInstFragment a very light weight
// object with just the MCInst and a code size, then we should just change
// MCDataFragment to have an optional MCInst at its end.
class MCInstFragment : public MCFragment {
  /// Inst - The instruction this is a fragment for.
  MCInst Inst;

  /// Code - Binary data for the currently encoded instruction.
  SmallString<8> Code;

  /// Fixups - The list of fixups in this fragment.
  SmallVector<MCFixup, 1> Fixups;

public:
  typedef SmallVectorImpl<MCFixup>::const_iterator const_fixup_iterator;
  typedef SmallVectorImpl<MCFixup>::iterator fixup_iterator;

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

  SmallVectorImpl<MCFixup> &getFixups() { return Fixups; }
  const SmallVectorImpl<MCFixup> &getFixups() const { return Fixups; }

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

  /// EmitNops - Flag to indicate that (optimal) NOPs should be emitted instead
  /// of using the provided value. The exact interpretation of this flag is
  /// target dependent.
  bool EmitNops : 1;

public:
  MCAlignFragment(unsigned _Alignment, int64_t _Value, unsigned _ValueSize,
                  unsigned _MaxBytesToEmit, MCSectionData *SD = 0)
    : MCFragment(FT_Align, SD), Alignment(_Alignment),
      Value(_Value),ValueSize(_ValueSize),
      MaxBytesToEmit(_MaxBytesToEmit), EmitNops(false) {}

  /// @name Accessors
  /// @{

  unsigned getAlignment() const { return Alignment; }

  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  unsigned getMaxBytesToEmit() const { return MaxBytesToEmit; }

  bool hasEmitNops() const { return EmitNops; }
  void setEmitNops(bool Value) { EmitNops = Value; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Align;
  }
  static bool classof(const MCAlignFragment *) { return true; }
};

class MCFillFragment : public MCFragment {
  /// Value - Value to use for filling bytes.
  int64_t Value;

  /// ValueSize - The size (in bytes) of \arg Value to use when filling, or 0 if
  /// this is a virtual fill fragment.
  unsigned ValueSize;

  /// Size - The number of bytes to insert.
  uint64_t Size;

public:
  MCFillFragment(int64_t _Value, unsigned _ValueSize, uint64_t _Size,
                 MCSectionData *SD = 0)
    : MCFragment(FT_Fill, SD),
      Value(_Value), ValueSize(_ValueSize), Size(_Size) {
    assert((!ValueSize || (Size % ValueSize) == 0) &&
           "Fill size must be a multiple of the value size!");
  }

  /// @name Accessors
  /// @{

  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  uint64_t getSize() const { return Size; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Fill;
  }
  static bool classof(const MCFillFragment *) { return true; }
};

class MCOrgFragment : public MCFragment {
  /// Offset - The offset this fragment should start at.
  const MCExpr *Offset;

  /// Value - Value to use for filling bytes.
  int8_t Value;

  /// Size - The current estimate of the size.
  unsigned Size;

public:
  MCOrgFragment(const MCExpr &_Offset, int8_t _Value, MCSectionData *SD = 0)
    : MCFragment(FT_Org, SD),
      Offset(&_Offset), Value(_Value), Size(0) {}

  /// @name Accessors
  /// @{

  const MCExpr &getOffset() const { return *Offset; }

  uint8_t getValue() const { return Value; }

  unsigned getSize() const { return Size; }

  void setSize(unsigned Size_) { Size = Size_; }
  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Org;
  }
  static bool classof(const MCOrgFragment *) { return true; }
};

class MCLEBFragment : public MCFragment {
  /// Value - The value this fragment should contain.
  const MCExpr *Value;

  /// IsSigned - True if this is a sleb128, false if uleb128.
  bool IsSigned;

  SmallString<8> Contents;
public:
  MCLEBFragment(const MCExpr &Value_, bool IsSigned_, MCSectionData *SD)
    : MCFragment(FT_LEB, SD),
      Value(&Value_), IsSigned(IsSigned_) { Contents.push_back(0); }

  /// @name Accessors
  /// @{

  const MCExpr &getValue() const { return *Value; }

  bool isSigned() const { return IsSigned; }

  SmallString<8> &getContents() { return Contents; }
  const SmallString<8> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_LEB;
  }
  static bool classof(const MCLEBFragment *) { return true; }
};

class MCDwarfLineAddrFragment : public MCFragment {
  /// LineDelta - the value of the difference between the two line numbers
  /// between two .loc dwarf directives.
  int64_t LineDelta;

  /// AddrDelta - The expression for the difference of the two symbols that
  /// make up the address delta between two .loc dwarf directives.
  const MCExpr *AddrDelta;

  SmallString<8> Contents;

public:
  MCDwarfLineAddrFragment(int64_t _LineDelta, const MCExpr &_AddrDelta,
                      MCSectionData *SD = 0)
    : MCFragment(FT_Dwarf, SD),
      LineDelta(_LineDelta), AddrDelta(&_AddrDelta) { Contents.push_back(0); }

  /// @name Accessors
  /// @{

  int64_t getLineDelta() const { return LineDelta; }

  const MCExpr &getAddrDelta() const { return *AddrDelta; }

  SmallString<8> &getContents() { return Contents; }
  const SmallString<8> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Dwarf;
  }
  static bool classof(const MCDwarfLineAddrFragment *) { return true; }
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
  FragmentListType Fragments;
  const MCSection *Section;

  /// Ordinal - The section index in the assemblers section list.
  unsigned Ordinal;

  /// LayoutOrder - The index of this section in the layout order.
  unsigned LayoutOrder;

  /// Alignment - The maximum alignment seen in this section.
  unsigned Alignment;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

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

  unsigned getLayoutOrder() const { return LayoutOrder; }
  void setLayoutOrder(unsigned Value) { LayoutOrder = Value; }

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

  /// SymbolSize - An expression describing how to calculate the size of
  /// a symbol. If a symbol has no size this field will be NULL.
  const MCExpr *SymbolSize;

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

  void setSize(const MCExpr *SS) {
    SymbolSize = SS;
  }

  const MCExpr *getSize() const {
    return SymbolSize;
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

  /// modifyFlags - Modify the flags via a mask
  void modifyFlags(uint32_t Value, uint32_t Mask) {
    Flags = (Flags & ~Mask) | Value;
  }

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
  bool EvaluateFixup(const MCObjectWriter &Writer, const MCAsmLayout &Layout,
                     const MCFixup &Fixup, const MCFragment *DF,
                     MCValue &Target, uint64_t &Value) const;

  /// Check whether a fixup can be satisfied, or whether it needs to be relaxed
  /// (increased in size, in order to hold its value correctly).
  bool FixupNeedsRelaxation(const MCObjectWriter &Writer,
                            const MCFixup &Fixup, const MCFragment *DF,
                            const MCAsmLayout &Layout) const;

  /// Check whether the given fragment needs relaxation.
  bool FragmentNeedsRelaxation(const MCObjectWriter &Writer,
                               const MCInstFragment *IF,
                               const MCAsmLayout &Layout) const;

  /// Compute the effective fragment size assuming it is layed out at the given
  /// \arg SectionAddress and \arg FragmentOffset.
  uint64_t ComputeFragmentSize(const MCFragment &F,
                               uint64_t FragmentOffset) const;

  /// LayoutOnce - Perform one layout iteration and return true if any offsets
  /// were adjusted.
  bool LayoutOnce(const MCObjectWriter &Writer, MCAsmLayout &Layout);

  bool RelaxInstruction(const MCObjectWriter &Writer, MCAsmLayout &Layout,
                        MCInstFragment &IF);

  bool RelaxOrg(const MCObjectWriter &Writer, MCAsmLayout &Layout,
                MCOrgFragment &OF);

  bool RelaxLEB(const MCObjectWriter &Writer, MCAsmLayout &Layout,
                MCLEBFragment &IF);

  bool RelaxDwarfLineAddr(const MCObjectWriter &Writer, MCAsmLayout &Layout,
			  MCDwarfLineAddrFragment &DF);

  /// FinishLayout - Finalize a layout, including fragment lowering.
  void FinishLayout(MCAsmLayout &Layout);

  uint64_t HandleFixup(MCObjectWriter &Writer, const MCAsmLayout &Layout,
                       MCFragment &F, const MCFixup &Fixup);

public:
  /// Find the symbol which defines the atom containing the given symbol, or
  /// null if there is no such symbol.
  const MCSymbolData *getAtom(const MCSymbolData *Symbol) const;

  /// Check whether a particular symbol is visible to the linker and is required
  /// in the symbol table, or whether it can be discarded by the assembler. This
  /// also effects whether the assembler treats the label as potentially
  /// defining a separate atom.
  bool isSymbolLinkerVisible(const MCSymbol &SD) const;

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
  /// \arg Writer is used for custom object writer (as the MCJIT does),
  /// if not specified it is automatically created from backend.
  void Finish(MCObjectWriter *Writer = 0);

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
