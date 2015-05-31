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
#include "SectionChunks.h"
#include "Writer.h"
#include "lld/Core/range.h"
#include "lld/Core/Writer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

namespace lld {
namespace elf {

template <typename ELFT> class TargetLayout;

/// \brief A segment can be divided into segment slices
///        depending on how the segments can be split
template<class ELFT>
class SegmentSlice {
public:
  typedef typename std::vector<Chunk<ELFT> *>::iterator SectionIter;

  /// Set the start of the slice.
  void setStart(int32_t s) { _startSection = s; }

  // Set the segment slice start and end iterators. This is used to walk through
  // the sections that are part of the Segment slice
  void setSections(range<SectionIter> sections) { _sections = sections; }

  // Return the fileOffset of the slice
  uint64_t fileOffset() const { return _offset; }
  void setFileOffset(uint64_t offset) { _offset = offset; }

  // Return the size of the slice
  uint64_t fileSize() const { return _fsize; }
  void setFileSize(uint64_t filesz) { _fsize = filesz; }

  // Return the start of the slice
  int32_t startSection() const { return _startSection; }

  // Return the start address of the slice
  uint64_t virtualAddr() const { return _addr; }

  // Return the memory size of the slice
  uint64_t memSize() const { return _memSize; }

  // Return the alignment of the slice
  uint64_t alignment() const { return _alignment; }

  void setMemSize(uint64_t memsz) { _memSize = memsz; }

  void setVirtualAddr(uint64_t addr) { _addr = addr; }

  void setAlign(uint64_t align) { _alignment = align; }

  static bool compare_slices(SegmentSlice<ELFT> *a, SegmentSlice<ELFT> *b);

  range<SectionIter> sections() { return _sections; }

private:
  range<SectionIter> _sections;
  int32_t _startSection;
  uint64_t _addr;
  uint64_t _offset;
  uint64_t _alignment;
  uint64_t _fsize;
  uint64_t _memSize;
};

/// \brief A segment contains a set of sections, that have similar properties
//  the sections are already separated based on different flags and properties
//  the segment is just a way to concatenate sections to segments
template<class ELFT>
class Segment : public Chunk<ELFT> {
public:
  typedef typename std::vector<SegmentSlice<ELFT> *>::iterator SliceIter;
  typedef typename std::vector<Chunk<ELFT> *>::iterator SectionIter;

  Segment(const ELFLinkingContext &ctx, StringRef name,
          const typename TargetLayout<ELFT>::SegmentType type);

  /// \brief the Order of segments that appear in the output file
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
  virtual void append(Chunk<ELFT> *chunk);

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
  void assignFileOffsets(uint64_t startOffset);

  /// \brief Assign virtual addresses to the slices
  void assignVirtualAddress(uint64_t addr);

  // Write the Segment
  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer);

  int64_t flags() const;

  /// Prepend a generic chunk to the segment.
  void prepend(Chunk<ELFT> *c) {
    _sections.insert(_sections.begin(), c);
  }

  /// Finalize the segment, before we want to write the segment header
  /// information
  void finalize();

  // For LLVM RTTI
  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::ELFSegment;
  }

  // Getters
  int32_t sectionCount() const { return _sections.size(); }

  /// \brief, this function returns the type of segment (PT_*)
  typename TargetLayout<ELFT>::SegmentType segmentType() const {
    return _segmentType;
  }

  /// \brief return the segment type depending on the content,
  /// If the content corresponds to Code, this will return Segment::Code
  /// If the content corresponds to Data, this will return Segment::Data
  /// If the content corresponds to TLS, this will return Segment::TLS
  virtual int getContentType() const;

  int pageSize() const { return this->_ctx.getPageSize(); }
  int rawflags() const { return _atomflags; }
  int64_t atomflags() const;
  int64_t numSlices() const { return _segmentSlices.size(); }
  range<SliceIter> slices() { return _segmentSlices; }
  Chunk<ELFT> *firstSection() { return _sections[0]; }

private:
  /// \brief Check if the chunk needs to be aligned
  bool needAlign(Chunk<ELFT> *chunk) const;

  // Cached value of outputMagic
  ELFLinkingContext::OutputMagic _outputMagic;

protected:
  /// \brief Section or some other chunk type.
  std::vector<Chunk<ELFT> *> _sections;
  std::vector<SegmentSlice<ELFT> *> _segmentSlices;
  typename TargetLayout<ELFT>::SegmentType _segmentType;
  uint64_t _flags;
  int64_t _atomflags;
  llvm::BumpPtrAllocator _segmentAllocate;
};

/// This chunk represents a linker script expression that needs to be calculated
/// at the time the virtual addresses for the parent segment are being assigned.
template <class ELFT> class ExpressionChunk : public Chunk<ELFT> {
public:
  ExpressionChunk(ELFLinkingContext &ctx, const script::SymbolAssignment *expr)
      : Chunk<ELFT>(StringRef(), Chunk<ELFT>::Kind::Expression, ctx),
        _expr(expr), _linkerScriptSema(ctx.linkerScriptSema()) {
    this->_alignment = 1;
  }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::Expression;
  }

  int getContentType() const override {
    return Chunk<ELFT>::ContentType::Unknown;
  }

  void write(ELFWriter *, TargetLayout<ELFT> &,
             llvm::FileOutputBuffer &) override {}

  std::error_code evalExpr(uint64_t &curPos) {
    return _linkerScriptSema.evalExpr(_expr, curPos);
  }

private:
  const script::SymbolAssignment *_expr;
  script::Sema &_linkerScriptSema;
};

/// \brief A Program Header segment contains a set of chunks instead of sections
/// The segment doesn't contain any slice
template <class ELFT> class ProgramHeaderSegment : public Segment<ELFT> {
public:
  ProgramHeaderSegment(const ELFLinkingContext &ctx)
      : Segment<ELFT>(ctx, "PHDR", llvm::ELF::PT_PHDR) {
    this->_alignment = 8;
    this->_flags = (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR);
  }

  /// Finalize the segment, before we want to write the segment header
  /// information
  void finalize();
};

} // end namespace elf
} // end namespace lld

#endif
