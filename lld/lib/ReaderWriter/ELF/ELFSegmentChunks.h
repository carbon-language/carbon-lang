//===- lib/ReaderWriter/ELF/ELFSegmentChunks.h -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_SEGMENT_CHUNKS_H_
#define LLD_READER_WRITER_ELF_SEGMENT_CHUNKS_H_

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

#include "ELFChunk.h"
#include "ELFLayout.h"
#include "ELFSectionChunks.h"
#include "ELFWriter.h"

/// \brief A segment can be divided into segment slices
///        depending on how the segments can be split
namespace lld {
namespace elf {

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

  Segment(const ELFTargetInfo &ti, const StringRef name,
          const ELFLayout::SegmentType type);

  /// append a section to a segment
  void append(Section<ELFT> *section);

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

  // Finalize the segment, before we want to write to the output file
  inline void finalize() { }

  // For LLVM RTTI
  static inline bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::K_ELFSegment;
  }

  // Getters
  inline int32_t sectionCount() const {
    return _sections.size();
  }

  inline ELFLayout::SegmentType segmentType() { return _segmentType; }

  inline int pageSize() const { return this->_targetInfo.getPageSize(); }

  inline int64_t atomflags() const { return _atomflags; }

  inline int64_t numSlices() const {
    return _segmentSlices.size();
  }

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
  ELFLayout::SegmentType _segmentType;
  int64_t _flags;
  int64_t _atomflags;
  llvm::BumpPtrAllocator _segmentAllocate;
};

template <class ELFT>
Segment<ELFT>::Segment(const ELFTargetInfo &ti, const StringRef name,
                       const ELFLayout::SegmentType type)
    : Chunk<ELFT>(name, Chunk<ELFT>::K_ELFSegment, ti), _segmentType(type),
      _flags(0), _atomflags(0) {
  this->_align2 = 0;
  this->_fsize = 0;
}

template<class ELFT>
void 
Segment<ELFT>::append(Section<ELFT> *section) {
  _sections.push_back(section);
  if (_flags < section->flags())
    _flags = section->flags();
  if (_atomflags < section->atomflags())
    _atomflags = section->atomflags();
  if (this->_align2 < section->align2())
    this->_align2 = section->align2();
}

template<class ELFT>
bool 
Segment<ELFT>::compareSegments(Segment<ELFT> *sega, Segment<ELFT> *segb) {
  if (sega->atomflags() < segb->atomflags())
    return false;
  return true;
}

template<class ELFT>
void 
Segment<ELFT>::assignOffsets(uint64_t startOffset) {
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
template<class ELFT>
void 
Segment<ELFT>::assignVirtualAddress(uint64_t &addr) {
  for (auto slice : slices()) {
    // Align to a page
    addr = llvm::RoundUpToAlignment(addr, this->_targetInfo.getPageSize());
    // Align to the slice alignment
    addr = llvm::RoundUpToAlignment(addr, slice->align2());

    bool virtualAddressSet = false;
    for (auto section : slice->sections()) {
      // Align the section address
      addr = llvm::RoundUpToAlignment(addr, section->align2());
      if (!virtualAddressSet) {
        slice->setVAddr(addr);
        virtualAddressSet = true;
      }
      section->setVAddr(addr);
      if (auto s = dyn_cast<Section<ELFT>>(section))
        s->assignVirtualAddress(addr);
      else
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

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_SEGMENT_CHUNKS_H_
