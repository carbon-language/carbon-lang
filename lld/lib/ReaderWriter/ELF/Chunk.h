//===- lib/ReaderWriter/ELF/Chunks.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_CHUNKS_H
#define LLD_READER_WRITER_ELF_CHUNKS_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

namespace lld {
class ELFLinkingContext;

namespace elf {
class ELFWriter;
template <class ELFT> class TargetLayout;

/// \brief A chunk is a contiguous region of space
template <class ELFT> class Chunk {
public:
  /// \brief Describes the type of Chunk
  enum Kind : uint8_t {
    ELFHeader,     ///< ELF Header
    ProgramHeader, ///< Program Header
    SectionHeader, ///< Section header
    ELFSegment,    ///< Segment
    ELFSection,    ///< Section
    AtomSection,   ///< A section containing atoms
    Expression     ///< A linker script expression
  };

  /// \brief the ContentType of the chunk
  enum ContentType : uint8_t { Unknown, Header, Code, Data, Note, TLS };

  Chunk(StringRef name, Kind kind, const ELFLinkingContext &ctx)
      : _name(name), _kind(kind), _ctx(ctx) {}

  virtual ~Chunk() {}

  // The name of the chunk
  StringRef name() const { return _name; }

  // Kind of chunk
  Kind kind() const { return _kind; }
  virtual uint64_t fileSize() const { return _fsize; }
  virtual void setFileSize(uint64_t sz) { _fsize = sz; }
  virtual void setAlign(uint64_t align) { _alignment = align; }
  virtual uint64_t alignment() const { return _alignment; }

  // The ordinal value of the chunk
  uint64_t ordinal() const { return _ordinal; }
  void setOrdinal(uint64_t newVal) { _ordinal = newVal; }

  // The order in which the chunk would appear in the output file
  uint64_t order() const { return _order; }
  void setOrder(uint32_t order) { _order = order; }

  // Output file offset of the chunk
  uint64_t fileOffset() const { return _fileoffset; }
  void setFileOffset(uint64_t offset) { _fileoffset = offset; }

  // Output start address of the chunk
  virtual void setVirtualAddr(uint64_t start) { _start = start; }
  virtual uint64_t virtualAddr() const { return _start; }

  // Memory size of the chunk
  uint64_t memSize() const { return _msize; }
  void setMemSize(uint64_t msize) { _msize = msize; }

  // Returns the ContentType of the chunk
  virtual int getContentType() const = 0;

  // Writer the chunk
  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer) = 0;

  // Finalize the chunk before assigning offsets/virtual addresses
  virtual void doPreFlight() {}

  // Finalize the chunk before writing
  virtual void finalize() {}

protected:
  StringRef _name;
  Kind _kind;
  const ELFLinkingContext &_ctx;
  uint64_t _fsize = 0;
  uint64_t _msize = 0;
  uint64_t _alignment = 1;
  uint32_t _order = 0;
  uint64_t _ordinal = 1;
  uint64_t _start = 0;
  uint64_t _fileoffset = 0;
};

} // end namespace elf
} // end namespace lld

#endif
