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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

namespace lld {
class ELFTargetInfo;

namespace elf {
class ELFWriter;

/// \brief A chunk is a contiguous region of space
template<class ELFT>
class Chunk {
public:

  /// \brief Describes the type of Chunk
  enum Kind {
    K_Header, ///< ELF Header
    K_ProgramHeader, ///< Program Header
    K_ELFSegment, ///< Segment
    K_ELFSection, ///< Section
    K_AtomSection, ///< A section containing atoms.
    K_SectionHeader ///< Section header
  };
  Chunk(StringRef name, Kind kind, const ELFTargetInfo &ti)
      : _name(name), _kind(kind), _fsize(0), _msize(0), _align2(0), _order(0),
        _ordinal(1), _start(0), _fileoffset(0), _targetInfo(ti) {
  }
  virtual ~Chunk() {}
  // Does the chunk occupy disk space
  virtual bool occupiesNoDiskSpace() const { return false; }
  // The name of the chunk
  StringRef name() const { return _name; }
  // Kind of chunk
  Kind kind() const { return _kind; }
  uint64_t            fileSize() const { return _fsize; }
  uint64_t            align2() const { return _align2; }

  // The ordinal value of the chunk
  uint64_t            ordinal() const { return _ordinal;}
  void               setOrdinal(uint64_t newVal) { _ordinal = newVal;}
  // The order in which the chunk would appear in the output file
  uint64_t            order() const { return _order; }
  void               setOrder(uint32_t order) { _order = order; }
  // Output file offset of the chunk
  uint64_t            fileOffset() const { return _fileoffset; }
  void               setFileOffset(uint64_t offset) { _fileoffset = offset; }
  // Output start address of the chunk
  void               setVAddr(uint64_t start) { _start = start; }
  uint64_t            virtualAddr() const { return _start; }
  // Does the chunk occupy memory during execution ?
  uint64_t            memSize() const { return _msize; }
  void setMemSize(uint64_t msize) { _msize = msize; }
  // Writer the chunk
  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer) = 0;
  // Finalize the chunk before assigning offsets/virtual addresses
  virtual void doPreFlight() = 0;
  // Finalize the chunk before writing
  virtual void finalize() = 0;

protected:
  StringRef _name;
  Kind _kind;
  uint64_t _fsize;
  uint64_t _msize;
  uint64_t _align2;
  uint32_t _order;
  uint64_t _ordinal;
  uint64_t _start;
  uint64_t _fileoffset;
  const ELFTargetInfo &_targetInfo;
};

} // end namespace elf
} // end namespace lld

#endif
