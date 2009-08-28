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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include <vector> // FIXME: Shouldn't be needed.

namespace llvm {
class raw_ostream;
class MCAssembler;
class MCSection;
class MCSectionData;

class MCFragment : public ilist_node<MCFragment> {
  MCFragment(const MCFragment&);     // DO NOT IMPLEMENT
  void operator=(const MCFragment&); // DO NOT IMPLEMENT

public:
  enum FragmentType {
    FT_Data,
    FT_Align,
    FT_Fill,
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

  /// FileSize - The file size of this section. This is ~0 until initialized.
  uint64_t FileSize;

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

  // FIXME: This should be abstract, fix sentinel.
  virtual uint64_t getMaxFileSize() const {
    assert(0 && "Invalid getMaxFileSize call!");
    return 0;
  };

  /// @name Assembler Backend Support
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  uint64_t getAddress() const;

  uint64_t getFileSize() const { 
    assert(FileSize != ~UINT64_C(0) && "File size not set!");
    return FileSize;
  }
  void setFileSize(uint64_t Value) {
    assert(Value <= getMaxFileSize() && "Invalid file size!");
    FileSize = Value;
  }

  uint64_t getOffset() const {
    assert(Offset != ~UINT64_C(0) && "File offset not set!");
    return Offset;
  }
  void setOffset(uint64_t Value) { Offset = Value; }

  /// @}

  static bool classof(const MCFragment *O) { return true; }
};

class MCDataFragment : public MCFragment {
  SmallString<32> Contents;

public:
  MCDataFragment(MCSectionData *SD = 0) : MCFragment(FT_Data, SD) {}

  /// @name Accessors
  /// @{

  uint64_t getMaxFileSize() const {
    return Contents.size();
  }

  SmallString<32> &getContents() { return Contents; }
  const SmallString<32> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) { 
    return F->getKind() == MCFragment::FT_Data; 
  }
  static bool classof(const MCDataFragment *) { return true; }
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

public:
  MCAlignFragment(unsigned _Alignment, int64_t _Value, unsigned _ValueSize,
                  unsigned _MaxBytesToEmit, MCSectionData *SD = 0)
    : MCFragment(FT_Align, SD), Alignment(_Alignment),
      Value(_Value),ValueSize(_ValueSize),
      MaxBytesToEmit(_MaxBytesToEmit) {}

  /// @name Accessors
  /// @{

  uint64_t getMaxFileSize() const {
    return std::max(Alignment - 1, MaxBytesToEmit);
  }

  unsigned getAlignment() const { return Alignment; }
  
  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  unsigned getMaxBytesToEmit() const { return MaxBytesToEmit; }

  /// @}

  static bool classof(const MCFragment *F) { 
    return F->getKind() == MCFragment::FT_Align; 
  }
  static bool classof(const MCAlignFragment *) { return true; }
};

class MCFillFragment : public MCFragment {
  /// Value - Value to use for filling bytes.
  MCValue Value;

  /// ValueSize - The size (in bytes) of \arg Value to use when filling.
  unsigned ValueSize;

  /// Count - The number of copies of \arg Value to insert.
  uint64_t Count;

public:
  MCFillFragment(MCValue _Value, unsigned _ValueSize, uint64_t _Count,
                 MCSectionData *SD = 0) 
    : MCFragment(FT_Fill, SD),
      Value(_Value), ValueSize(_ValueSize), Count(_Count) {}

  /// @name Accessors
  /// @{

  uint64_t getMaxFileSize() const {
    return ValueSize * Count;
  }

  MCValue getValue() const { return Value; }
  
  unsigned getValueSize() const { return ValueSize; }

  uint64_t getCount() const { return Count; }

  /// @}

  static bool classof(const MCFragment *F) { 
    return F->getKind() == MCFragment::FT_Fill; 
  }
  static bool classof(const MCFillFragment *) { return true; }
};

class MCOrgFragment : public MCFragment {
  /// Offset - The offset this fragment should start at.
  MCValue Offset;

  /// Value - Value to use for filling bytes.  
  int8_t Value;

public:
  MCOrgFragment(MCValue _Offset, int8_t _Value, MCSectionData *SD = 0)
    : MCFragment(FT_Org, SD),
      Offset(_Offset), Value(_Value) {}

  /// @name Accessors
  /// @{

  uint64_t getMaxFileSize() const {
    // FIXME: This doesn't make much sense.
    return ~UINT64_C(0);
  }

  MCValue getOffset() const { return Offset; }
  
  uint8_t getValue() const { return Value; }

  /// @}

  static bool classof(const MCFragment *F) { 
    return F->getKind() == MCFragment::FT_Org; 
  }
  static bool classof(const MCOrgFragment *) { return true; }
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

  uint64_t getMaxFileSize() const {
    // FIXME: This also doesn't make much sense, this method is misnamed.
    return ~UINT64_C(0);
  }

  uint64_t getSize() const { return Size; }
  
  unsigned getAlignment() const { return Alignment; }

  /// @}

  static bool classof(const MCFragment *F) { 
    return F->getKind() == MCFragment::FT_ZeroFill; 
  }
  static bool classof(const MCZeroFillFragment *) { return true; }
};

// FIXME: Should this be a separate class, or just merged into MCSection? Since
// we anticipate the fast path being through an MCAssembler, the only reason to
// keep it out is for API abstraction.
class MCSectionData : public ilist_node<MCSectionData> {
  MCSectionData(const MCSectionData&);  // DO NOT IMPLEMENT
  void operator=(const MCSectionData&); // DO NOT IMPLEMENT

public:
  /// Fixup - Represent a fixed size region of bytes inside some fragment which
  /// needs to be rewritten. This region will either be rewritten by the
  /// assembler or cause a relocation entry to be generated.
  struct Fixup {
    /// Fragment - The fragment containing the fixup.
    MCFragment *Fragment;
    
    /// Offset - The offset inside the fragment which needs to be rewritten.
    uint64_t Offset;

    /// Value - The expression to eventually write into the fragment.
    //
    // FIXME: We could probably get away with requiring the client to pass in an
    // owned reference whose lifetime extends past that of the fixup.
    MCValue Value;

    /// Size - The fixup size.
    unsigned Size;

    /// FixedValue - The value to replace the fix up by.
    //
    // FIXME: This should not be here.
    uint64_t FixedValue;

  public:
    Fixup(MCFragment &_Fragment, uint64_t _Offset, const MCValue &_Value, 
          unsigned _Size) 
      : Fragment(&_Fragment), Offset(_Offset), Value(_Value), Size(_Size),
        FixedValue(0) {}
  };

  typedef iplist<MCFragment> FragmentListType;

  typedef FragmentListType::const_iterator const_iterator;
  typedef FragmentListType::iterator iterator;

  typedef std::vector<Fixup>::const_iterator const_fixup_iterator;
  typedef std::vector<Fixup>::iterator fixup_iterator;

private:
  iplist<MCFragment> Fragments;
  const MCSection *Section;

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

  /// LastFixupLookup - Cache for the last looked up fixup.
  mutable unsigned LastFixupLookup;

  /// Fixups - The list of fixups in this section.
  std::vector<Fixup> Fixups;
  
  /// @}

public:    
  // Only for use as sentinel.
  MCSectionData();
  MCSectionData(const MCSection &Section, MCAssembler *A = 0);

  const MCSection &getSection() const { return *Section; }

  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Value) { Alignment = Value; }

  /// @name Fragment Access
  /// @{

  const FragmentListType &getFragmentList() const { return Fragments; }
  FragmentListType &getFragmentList() { return Fragments; }

  iterator begin() { return Fragments.begin(); }
  const_iterator begin() const { return Fragments.begin(); }

  iterator end() { return Fragments.end(); }
  const_iterator end() const { return Fragments.end(); }

  size_t size() const { return Fragments.size(); }

  bool empty() const { return Fragments.empty(); }

  /// @}
  /// @name Fixup Access
  /// @{

  std::vector<Fixup> &getFixups() {
    return Fixups;
  }

  fixup_iterator fixup_begin() {
    return Fixups.begin();
  }

  fixup_iterator fixup_end() {
    return Fixups.end();
  }

  size_t fixup_size() const { return Fixups.size(); }

  /// @}
  /// @name Assembler Backend Support
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// LookupFixup - Look up the fixup for the given \arg Fragment and \arg
  /// Offset.
  ///
  /// If multiple fixups exist for the same fragment and offset it is undefined
  /// which one is returned.
  //
  // FIXME: This isn't horribly slow in practice, but there are much nicer
  // solutions to applying the fixups.
  const Fixup *LookupFixup(const MCFragment *Fragment, uint64_t Offset) const;

  uint64_t getAddress() const { 
    assert(Address != ~UINT64_C(0) && "Address not set!");
    return Address;
  }
  void setAddress(uint64_t Value) { Address = Value; }

  uint64_t getSize() const { 
    assert(Size != ~UINT64_C(0) && "File size not set!");
    return Size;
  }
  void setSize(uint64_t Value) { Size = Value; }

  uint64_t getFileSize() const { 
    assert(FileSize != ~UINT64_C(0) && "File size not set!");
    return FileSize;
  }
  void setFileSize(uint64_t Value) { FileSize = Value; }  

  /// @}
};

// FIXME: Same concerns as with SectionData.
class MCSymbolData : public ilist_node<MCSymbolData> {
public:
  MCSymbol &Symbol;

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

  /// Flags - The Flags field is used by object file implementations to store
  /// additional per symbol information which is not easily classified.
  uint32_t Flags;

  /// Index - Index field, for use by the object file implementation.
  uint64_t Index;

public:
  // Only for use as sentinel.
  MCSymbolData();
  MCSymbolData(MCSymbol &_Symbol, MCFragment *_Fragment, uint64_t _Offset,
               MCAssembler *A = 0);

  /// @name Accessors
  /// @{

  MCSymbol &getSymbol() const { return Symbol; }

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
  
  /// getFlags - Get the (implementation defined) symbol flags.
  uint32_t getFlags() const { return Flags; }

  /// setFlags - Set the (implementation defined) symbol flags.
  void setFlags(uint32_t Value) { Flags = Value; }
  
  /// getIndex - Get the (implementation defined) index.
  uint64_t getIndex() const { return Index; }

  /// setIndex - Set the (implementation defined) index.
  void setIndex(uint64_t Value) { Index = Value; }
  
  /// @}  
};

// FIXME: This really doesn't belong here. See comments below.
struct IndirectSymbolData {
  MCSymbol *Symbol;
  MCSectionData *SectionData;
};

class MCAssembler {
public:
  typedef iplist<MCSectionData> SectionDataListType;
  typedef iplist<MCSymbolData> SymbolDataListType;

  typedef SectionDataListType::const_iterator const_iterator;
  typedef SectionDataListType::iterator iterator;

  typedef SymbolDataListType::const_iterator const_symbol_iterator;
  typedef SymbolDataListType::iterator symbol_iterator;

  typedef std::vector<IndirectSymbolData>::iterator indirect_symbol_iterator;

private:
  MCAssembler(const MCAssembler&);    // DO NOT IMPLEMENT
  void operator=(const MCAssembler&); // DO NOT IMPLEMENT

  raw_ostream &OS;
  
  iplist<MCSectionData> Sections;

  iplist<MCSymbolData> Symbols;

  std::vector<IndirectSymbolData> IndirectSymbols;

  unsigned SubsectionsViaSymbols : 1;

private:
  /// LayoutSection - Assign offsets and sizes to the fragments in the section
  /// \arg SD, and update the section size. The section file offset should
  /// already have been computed.
  void LayoutSection(MCSectionData &SD);

public:
  /// Construct a new assembler instance.
  ///
  /// \arg OS - The stream to output to.
  //
  // FIXME: How are we going to parameterize this? Two obvious options are stay
  // concrete and require clients to pass in a target like object. The other
  // option is to make this abstract, and have targets provide concrete
  // implementations as we do with AsmParser.
  MCAssembler(raw_ostream &OS);
  ~MCAssembler();

  /// Finish - Do final processing and write the object to the output stream.
  void Finish();

  // FIXME: This does not belong here.
  bool getSubsectionsViaSymbols() const {
    return SubsectionsViaSymbols;
  }
  void setSubsectionsViaSymbols(bool Value) {
    SubsectionsViaSymbols = Value;
  }

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

  indirect_symbol_iterator indirect_symbol_end() {
    return IndirectSymbols.end();
  }

  size_t indirect_symbol_size() const { return IndirectSymbols.size(); }

  /// @}
};

} // end namespace llvm

#endif
