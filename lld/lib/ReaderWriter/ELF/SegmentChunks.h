//===- lib/ReaderWriter/ELF/SegmentChunks.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_SEGMENT_CHUNKS_H
#define LLD_READER_WRITER_ELF_SEGMENT_CHUNKS_H

#include "Chunk.h"
#include "Layout.h"
#include "SectionChunks.h"
#include "Writer.h"

#include "lld/Core/range.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

namespace lld {
namespace elf {
/// \brief A segment can be divided into segment slices
///        depending on how the segments can be split
template<class ELFT>
class SegmentSlice {
public:
  typedef typename std::vector<Chunk<ELFT> *>::iterator SectionIter;

  SegmentSlice() { }

  /// Set the segment slice so that it begins at the offset specified
  /// by file offset and set the start of the slice to be s and the end
  /// of the slice to be e
  void set(uint64_t fileoffset, int32_t s, int e) {
    _startSection = s;
    _endSection = e + 1;
    _offset = fileoffset;
  }

  // Set the segment slice start and end iterators. This is used to walk through
  // the sections that are part of the Segment slice
  inline void setSections(range<SectionIter> sections) {
    _sections = sections;
  }

  // Return the fileOffset of the slice
  inline uint64_t fileOffset() const { return _offset; }

  // Return the size of the slice
  inline uint64_t fileSize() const { return _size; }

  // Return the start of the slice
  inline int32_t startSection() const { return _startSection; }

  // Return the start address of the slice
  inline uint64_t virtualAddr() const { return _addr; }

  // Return the memory size of the slice
  inline uint64_t memSize() const { return _memSize; }

  // Return the alignment of the slice
  inline uint64_t align2() const { return _align2; }

  inline void setSize(uint64_t sz) { _size = sz; }

  inline void setMemSize(uint64_t memsz) { _memSize = memsz; }

  inline void setVAddr(uint64_t addr) { _addr = addr; }

  inline void setAlign(uint64_t align) { _align2 = align; }

  static bool compare_slices(SegmentSlice<ELFT> *a, SegmentSlice<ELFT> *b) {
    return a->startSection() < b->startSection();
  }

  inline range<SectionIter> sections() {
    return _sections;
  }

private:
  int32_t _startSection;
  int32_t _endSection;
  range<SectionIter> _sections;
  uint64_t _addr;
  uint64_t _offset;
  uint64_t _size;
  uint64_t _align2;
  uint64_t _memSize;
};

/// \brief A segment contains a set of sections, that have similiar properties
//  the sections are already seperated based on different flags and properties
//  the segment is just a way to concatenate sections to segments
template<class ELFT>
class Segment : public Chunk<ELFT> {
public:
  typedef typename std::vector<SegmentSlice<ELFT> *>::iterator SliceIter;
  typedef typename std::vector<Chunk<ELFT> *>::iterator SectionIter;

  Segment(const ELFTargetInfo &ti, StringRef name,
          const Layout::SegmentType type);

  enum SegmentOrder {
    permUnknown,
    permRWX,
    permRX,
    permR,
    permRWL,
    permRW,
    permNonAccess
  };

  /// append a section to a segment
  virtual void append(Section<ELFT> *chunk);

  /// append a chunk to a segment, this function
  /// is used by the ProgramHeader segment
  virtual void append(Chunk<ELFT> *chunk) {}

  /// Sort segments depending on the property
  /// If we have a Program Header segment, it should appear first
  /// If we have a INTERP segment, that should appear after the Program Header
  /// All Loadable segments appear next in this order
  /// All Read Write Execute segments follow
  /// All Read Execute segments appear next
  /// All Read only segments appear first
  /// All Write execute segments follow
  static bool compareSegments(Segment<ELFT> *sega, Segment<ELFT> *segb);

  /// \brief Start assigning file offset to the segment chunks The fileoffset
  /// needs to be page at the start of the segment and in addition the
  /// fileoffset needs to be aligned to the max section alignment within the
  /// segment. This is required so that the ELF property p_poffset % p_align =
  /// p_vaddr mod p_align holds true.
  /// The algorithm starts off by assigning the startOffset thats passed in as
  /// parameter to the first section in the segment, if the difference between
  /// the newly computed offset is greater than a page, then we create a segment
  /// slice, as it would be a waste of virtual memory just to be filled with
  /// zeroes
  void assignOffsets(uint64_t startOffset);

  /// \brief Assign virtual addresses to the slices
  void assignVirtualAddress(uint64_t &addr);

  // Write the Segment
  void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  int64_t flags() const;

  /// Prepend a generic chunk to the segment.
  void prepend(Chunk<ELFT> *c) {
    _sections.insert(_sections.begin(), c);
  }

  /// Finalize the segment before assigning File Offsets / Virtual addresses
  inline void doPreFlight() {}

  /// Finalize the segment, before we want to write the segment header
  /// information
  inline void finalize() {
    // We want to finalize the segment values for now only for non loadable
    // segments, since those values are not set in the Layout
    if (_segmentType == llvm::ELF::PT_LOAD)
      return;
    // The size is the difference of the
    // last section to the first section, especially for TLS because
    // the TLS segment contains both .tdata/.tbss
    this->setFileOffset(_sections.front()->fileOffset());
    this->setVAddr(_sections.front()->virtualAddr());
    size_t startFileOffset = _sections.front()->fileOffset();
    size_t startAddr = _sections.front()->virtualAddr();
    for (auto ai : _sections) {
      this->_fsize = ai->fileOffset() + ai->fileSize() - startFileOffset;
      this->_msize = ai->virtualAddr() + ai->memSize() - startAddr;
    }
  }

  // For LLVM RTTI
  static inline bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::K_ELFSegment;
  }

  // Getters
  inline int32_t sectionCount() const {
    return _sections.size();
  }

  inline Layout::SegmentType segmentType() { return _segmentType; }

  inline int pageSize() const { return this->_targetInfo.getPageSize(); }

  inline int rawflags() const { return _atomflags; }

  inline int64_t atomflags() const {
    switch (_atomflags) {

    case DefinedAtom::permUnknown:
      return permUnknown;

    case DefinedAtom::permRWX:
      return permRWX;

    case DefinedAtom::permR_X:
      return permRX;

    case DefinedAtom::permR__:
      return permR;

    case DefinedAtom::permRW_L:
      return permRWL;

    case DefinedAtom::permRW_:
      return permRW;

    case DefinedAtom::perm___:
    default:
      return permNonAccess;
    }
  }

  inline int64_t numSlices() const { return _segmentSlices.size(); }

  inline range<SliceIter> slices() { return _segmentSlices; }

  // These two accessors are still needed for a call to std::stable_sort.
  // Consider adding wrappers for two iterator algorithms.
  inline SliceIter slices_begin() {
    return _segmentSlices.begin();
  }

  inline SliceIter slices_end() {
    return _segmentSlices.end();
  }

protected:
  /// \brief Section or some other chunk type.
  std::vector<Chunk<ELFT> *> _sections;
  std::vector<SegmentSlice<ELFT> *> _segmentSlices;
  Layout::SegmentType _segmentType;
  uint64_t _flags;
  int64_t _atomflags;
  llvm::BumpPtrAllocator _segmentAllocate;
};

/// \brief A Program Header segment contains a set of chunks instead of sections
/// The segment doesnot contain any slice
template <class ELFT> class ProgramHeaderSegment : public Segment<ELFT> {
public:
  ProgramHeaderSegment(const ELFTargetInfo &ti)
      : Segment<ELFT>(ti, "PHDR", llvm::ELF::PT_PHDR) {
    this->_align2 = 8;
    this->_flags = (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR);
  }

  /// append a section to a segment
  void append(Chunk<ELFT> *chunk) { _sections.push_back(chunk); }

  /// Finalize the segment, before we want to write the segment header
  /// information
  inline void finalize() {
    // If the segment is of type Program Header, then the values fileOffset
    // and the fileSize need to be picked up from the last section, the first
    // section points to the ELF header and the second chunk points to the
    // actual program headers
    this->setFileOffset(_sections.back()->fileOffset());
    this->setVAddr(_sections.back()->virtualAddr());
    this->_fsize = _sections.back()->fileSize();
    this->_msize = _sections.back()->memSize();
  }

protected:
  /// \brief Section or some other chunk type.
  std::vector<Chunk<ELFT> *> _sections;
};

template <class ELFT>
Segment<ELFT>::Segment(const ELFTargetInfo &ti, StringRef name,
                       const Layout::SegmentType type)
    : Chunk<ELFT>(name, Chunk<ELFT>::K_ELFSegment, ti), _segmentType(type),
      _flags(0), _atomflags(0) {
  this->_align2 = 0;
  this->_fsize = 0;
}

// This function actually is used, but not in all instantiations of Segment.
LLVM_ATTRIBUTE_UNUSED
static DefinedAtom::ContentPermissions toAtomPerms(uint64_t flags) {
  switch (flags & (SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR)) {
  case SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR:
    return DefinedAtom::permRWX;
  case SHF_ALLOC | SHF_EXECINSTR:
    return DefinedAtom::permR_X;
  case SHF_ALLOC:
    return DefinedAtom::permR__;
  case SHF_ALLOC | SHF_WRITE:
    return DefinedAtom::permRW_;
  default:
    return DefinedAtom::permUnknown;
  }
}

template <class ELFT> void Segment<ELFT>::append(Section<ELFT> *section) {
  _sections.push_back(section);
  if (_flags < section->getFlags())
    _flags = section->getFlags();
  if (_atomflags < toAtomPerms(_flags))
    _atomflags = toAtomPerms(_flags);
  if (this->_align2 < section->align2())
    this->_align2 = section->align2();
}

template <class ELFT>
bool Segment<ELFT>::compareSegments(Segment<ELFT> *sega, Segment<ELFT> *segb) {
  int64_t type1 = sega->segmentType();
  int64_t type2 = segb->segmentType();

  // The single PT_PHDR segment is required to precede any loadable
  // segment.  We simply make it always first.
  if (type1 == llvm::ELF::PT_PHDR)
    return true;
  if (type2 == llvm::ELF::PT_PHDR)
    return false;

  // The single PT_INTERP segment is required to precede any loadable
  // segment.  We simply make it always second.
  if (type1 == llvm::ELF::PT_INTERP)
    return true;
  if (type2 == llvm::ELF::PT_INTERP)
    return false;

  // We then put PT_LOAD segments before any other segments.
  if (type1 == llvm::ELF::PT_LOAD && type2 != llvm::ELF::PT_LOAD)
    return true;
  if (type2 == llvm::ELF::PT_LOAD && type1 != llvm::ELF::PT_LOAD)
    return false;

  // We put the PT_TLS segment last except for the PT_GNU_RELRO
  // segment, because that is where the dynamic linker expects to find
  if (type1 == llvm::ELF::PT_TLS && type2 != llvm::ELF::PT_TLS &&
      type2 != llvm::ELF::PT_GNU_RELRO)
    return false;
  if (type2 == llvm::ELF::PT_TLS && type1 != llvm::ELF::PT_TLS &&
      type1 != llvm::ELF::PT_GNU_RELRO)
    return true;

  // We put the PT_GNU_RELRO segment last, because that is where the
  // dynamic linker expects to find it
  if (type1 == llvm::ELF::PT_GNU_RELRO && type2 != llvm::ELF::PT_GNU_RELRO)
    return false;
  if (type2 == llvm::ELF::PT_GNU_RELRO && type1 != llvm::ELF::PT_GNU_RELRO)
    return true;

  if (type1 == type2)
    return sega->atomflags() < segb->atomflags();
  return false;
}

template <class ELFT> void Segment<ELFT>::assignOffsets(uint64_t startOffset) {
  int startSection = 0;
  int currSection = 0;
  SectionIter startSectionIter, endSectionIter;
  // slice align is set to the max alignment of the chunks that are
  // contained in the slice
  uint64_t sliceAlign = 0;
  // Current slice size
  uint64_t curSliceSize = 0;
  // Current Slice File Offset
  uint64_t curSliceFileOffset = 0;

  startSectionIter = _sections.begin();
  endSectionIter = _sections.end();
  startSection = 0;
  bool isFirstSection = true;
  for (auto si = _sections.begin(); si != _sections.end(); ++si) {
    if (isFirstSection) {
      // align the startOffset to the section alignment
      uint64_t newOffset =
        llvm::RoundUpToAlignment(startOffset, (*si)->align2());
      curSliceFileOffset = newOffset;
      sliceAlign = (*si)->align2();
      this->setFileOffset(startOffset);
      (*si)->setFileOffset(newOffset);
      curSliceSize = (*si)->fileSize();
      isFirstSection = false;
    } else {
      uint64_t curOffset = curSliceFileOffset + curSliceSize;
      uint64_t newOffset =
        llvm::RoundUpToAlignment(curOffset, (*si)->align2());
      SegmentSlice<ELFT> *slice = nullptr;
      // If the newOffset computed is more than a page away, lets create
      // a seperate segment, so that memory is not used up while running
      if ((newOffset - curOffset) > this->_targetInfo.getPageSize()) {
        // TODO: use std::find here
        for (auto s : slices()) {
          if (s->startSection() == startSection) {
            slice = s;
            break;
          }
        }
        if (!slice) {
          slice = new (_segmentAllocate.Allocate<SegmentSlice<ELFT>>())
            SegmentSlice<ELFT>();
          _segmentSlices.push_back(slice);
        }
        slice->set(curSliceFileOffset, startSection, currSection);
        slice->setSections(make_range(startSectionIter, endSectionIter));
        slice->setSize(curSliceSize);
        slice->setAlign(sliceAlign);
        uint64_t newPageOffset = llvm::RoundUpToAlignment(
            curOffset, this->_targetInfo.getPageSize());
        newOffset = llvm::RoundUpToAlignment(newPageOffset, (*si)->align2());
        curSliceFileOffset = newOffset;
        startSectionIter = endSectionIter;
        startSection = currSection;
        (*si)->setFileOffset(curSliceFileOffset);
        curSliceSize = newOffset - curSliceFileOffset + (*si)->fileSize();
        sliceAlign = (*si)->align2();
      } else {
        if (sliceAlign < (*si)->align2())
          sliceAlign = (*si)->align2();
        (*si)->setFileOffset(newOffset);
        curSliceSize = newOffset - curSliceFileOffset + (*si)->fileSize();
      }
    }
    currSection++;
    endSectionIter = si;
  }
  SegmentSlice<ELFT> *slice = nullptr;
  for (auto s : slices()) {
    // TODO: add std::find
    if (s->startSection() == startSection) {
      slice = s;
      break;
    }
  }
  if (!slice) {
    slice = new (_segmentAllocate.Allocate<SegmentSlice<ELFT>>())
      SegmentSlice<ELFT>();
    _segmentSlices.push_back(slice);
  }
  slice->set(curSliceFileOffset, startSection, currSection);
  slice->setSections(make_range(startSectionIter, _sections.end()));
  slice->setSize(curSliceSize);
  slice->setAlign(sliceAlign);
  this->_fsize = curSliceFileOffset - startOffset + curSliceSize;
  std::stable_sort(slices_begin(), slices_end(),
                   SegmentSlice<ELFT>::compare_slices);
}

/// \brief Assign virtual addresses to the slices
template <class ELFT> void Segment<ELFT>::assignVirtualAddress(uint64_t &addr) {
  bool isTLSSegment = false;
  uint64_t tlsStartAddr = 0;

  for (auto slice : slices()) {
    // Align to a page
    addr = llvm::RoundUpToAlignment(addr, this->_targetInfo.getPageSize());
    // Align to the slice alignment
    addr = llvm::RoundUpToAlignment(addr, slice->align2());

    bool virtualAddressSet = false;
    for (auto section : slice->sections()) {
      // Align the section address
      addr = llvm::RoundUpToAlignment(addr, section->align2());
      // Check if the segment is of type TLS
      // The sections that belong to the TLS segment have their
      // virtual addresses that are relative To TP
      Section<ELFT> *currentSection = llvm::dyn_cast<Section<ELFT> >(section);
      if (currentSection)
        isTLSSegment = (currentSection->getSegmentType() == llvm::ELF::PT_TLS);

      tlsStartAddr = (isTLSSegment)
                     ? llvm::RoundUpToAlignment(tlsStartAddr, section->align2())
                     : 0;
      if (!virtualAddressSet) {
        slice->setVAddr(addr);
        virtualAddressSet = true;
      }
      section->setVAddr(addr);
      if (auto s = dyn_cast<Section<ELFT> >(section)) {
        if (isTLSSegment)
          s->assignVirtualAddress(tlsStartAddr);
        else
          s->assignVirtualAddress(addr);
      }
      if (isTLSSegment)
        tlsStartAddr += section->memSize();
      addr += section->memSize();
      section->setMemSize(addr - section->virtualAddr());
    }
    slice->setMemSize(addr - slice->virtualAddr());
  }
}

// Write the Segment
template <class ELFT>
void Segment<ELFT>::write(ELFWriter *writer, llvm::FileOutputBuffer &buffer) {
  for (auto slice : slices())
    for (auto section : slice->sections())
      section->write(writer, buffer);
}

template<class ELFT>
int64_t
Segment<ELFT>::flags() const {
  int64_t fl = 0;
  if (_flags & llvm::ELF::SHF_ALLOC)
    fl |= llvm::ELF::PF_R;
  if (_flags & llvm::ELF::SHF_WRITE)
    fl |= llvm::ELF::PF_W;
  if (_flags & llvm::ELF::SHF_EXECINSTR)
    fl |= llvm::ELF::PF_X;
  return fl;
}
} // end namespace elf
} // end namespace lld

#endif
