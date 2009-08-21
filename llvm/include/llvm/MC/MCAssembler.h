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
    FT_Org
  };

private:
  FragmentType Kind;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// FileOffset - The offset of this section in the object file. This is ~0
  /// until initialized.
  uint64_t FileOffset;

  /// FileSize - The size of this section in the object file. This is ~0 until
  /// initialized.
  uint64_t FileSize;

  /// @}

protected:
  MCFragment(FragmentType _Kind, MCSectionData *SD = 0);

public:
  // Only for sentinel.
  MCFragment();
  virtual ~MCFragment();

  FragmentType getKind() const { return Kind; }

  // FIXME: This should be abstract, fix sentinel.
  virtual uint64_t getMaxFileSize() const {
    assert(0 && "Invalid getMaxFileSize call !");
    return 0;
  };

  /// @name Assembler Backend Support
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  unsigned getFileSize() const { 
    assert(FileSize != ~UINT64_C(0) && "File size not set!");
    return FileSize;
  }
  void setFileSize(uint64_t Value) {
    assert(Value <= getMaxFileSize() && "Invalid file size!");
    FileSize = Value;
  }

  uint64_t getFileOffset() const {
    assert(FileOffset != ~UINT64_C(0) && "File offset not set!");
    return FileOffset;
  }
  void setFileOffset(uint64_t Value) { FileOffset = Value; }

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

// FIXME: Should this be a separate class, or just merged into MCSection? Since
// we anticipate the fast path being through an MCAssembler, the only reason to
// keep it out is for API abstraction.
class MCSectionData : public ilist_node<MCSectionData> {
  MCSectionData(const MCSectionData&);  // DO NOT IMPLEMENT
  void operator=(const MCSectionData&); // DO NOT IMPLEMENT

public:
  typedef iplist<MCFragment> FragmentListType;

  typedef FragmentListType::const_iterator const_iterator;
  typedef FragmentListType::iterator iterator;

private:
  iplist<MCFragment> Fragments;
  const MCSection &Section;

  /// Alignment - The maximum alignment seen in this section.
  unsigned Alignment;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// FileOffset - The offset of this section in the object file. This is ~0
  /// until initialized.
  uint64_t FileOffset;

  /// FileSize - The size of this section in the object file. This is ~0 until
  /// initialized.
  uint64_t FileSize;

  /// @}

public:    
  // Only for use as sentinel.
  MCSectionData();
  MCSectionData(const MCSection &Section, MCAssembler *A = 0);

  const MCSection &getSection() const { return Section; }

  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Value) { Alignment = Value; }


  /// @name Section List Access
  /// @{

  const FragmentListType &getFragmentList() const { return Fragments; }
  FragmentListType &getFragmentList() { return Fragments; }

  iterator begin() { return Fragments.begin(); }
  const_iterator begin() const { return Fragments.begin(); }

  iterator end() { return Fragments.end(); }
  const_iterator end() const { return Fragments.end(); }

  size_t size() const { return Fragments.size(); }

  /// @}
  /// @name Assembler Backend Support
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  unsigned getFileSize() const { 
    assert(FileSize != ~UINT64_C(0) && "File size not set!");
    return FileSize;
  }
  void setFileSize(uint64_t Value) { FileSize = Value; }

  uint64_t getFileOffset() const {
    assert(FileOffset != ~UINT64_C(0) && "File offset not set!");
    return FileOffset;
  }
  void setFileOffset(uint64_t Value) { FileOffset = Value; }

  /// @}
};

class MCAssembler {
public:
  typedef iplist<MCSectionData> SectionDataListType;

  typedef SectionDataListType::const_iterator const_iterator;
  typedef SectionDataListType::iterator iterator;

private:
  MCAssembler(const MCAssembler&);    // DO NOT IMPLEMENT
  void operator=(const MCAssembler&); // DO NOT IMPLEMENT

  raw_ostream &OS;
  
  iplist<MCSectionData> Sections;

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
};

} // end namespace llvm

#endif
