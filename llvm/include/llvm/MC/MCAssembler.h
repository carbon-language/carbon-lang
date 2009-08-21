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

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
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
  MCFragment(MCSectionData *SD = 0);
};

// FIXME: Should this be a separate class, or just merged into MCSection? Since
// we anticipate the fast path being through an MCAssembler, the only reason to
// keep it out is for API abstraction.
class MCSectionData : public ilist_node<MCSectionData> {
  MCSectionData(const MCSectionData&);  // DO NOT IMPLEMENT
  void operator=(const MCSectionData&); // DO NOT IMPLEMENT

public:
  typedef iplist<MCFragment> FragmentListType;

private:
  iplist<MCFragment> Fragments;
  const MCSection &Section;

  /// Alignment - The maximum alignment seen in this section.
  unsigned Alignment;

  /// @name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// FileOffset - The offset of this section in the object file.
  uint64_t FileOffset;

  /// FileSize - The size of this section in the object file.
  uint64_t FileSize;

  /// @}

public:    
  // Only for use as sentinel.
  MCSectionData();
  MCSectionData(const MCSection &Section, MCAssembler *A = 0);

  const FragmentListType &getFragmentList() const { return Fragments; }
  FragmentListType &getFragmentList() { return Fragments; }

  const MCSection &getSection() const { return Section; }

  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Value) { Alignment = Value; }

  /// @name Assembler Backend Support
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  unsigned getFileSize() const { return FileSize; }

  uint64_t getFileOffset() const { return FileOffset; }
  void setFileOffset(uint64_t Value) { FileOffset = Value; }

  void WriteFileData(raw_ostream &OS) const;

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
