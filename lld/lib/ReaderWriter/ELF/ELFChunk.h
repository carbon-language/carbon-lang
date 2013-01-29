//===- lib/ReaderWriter/ELF/ELFChunks.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_CHUNKS_H_
#define LLD_READER_WRITER_ELF_CHUNKS_H_

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

class ELFWriter;

/// \brief A chunk is a contiguous region of space
template<class ELFT>
class Chunk {
public:

  /// \brief Describes the type of Chunk
  enum Kind {
    K_ELFHeader, // ELF Header
    K_ELFProgramHeader, // Program Header
    K_ELFSegment, // Segment
    K_ELFSection, // Section
    K_ELFSectionHeader // Section header
  };
  Chunk(llvm::StringRef name, Kind kind, const ELFTargetInfo &ti)
      : _name(name), _kind(kind), _fsize(0), _msize(0), _align2(0), _order(0),
        _ordinal(1), _start(0), _fileoffset(0), _targetInfo(ti) {
  }
  virtual             ~Chunk() {}
  // Does the chunk occupy disk space
  virtual bool        occupiesNoDiskSpace() const {
    return false;
  }
  // The name of the chunk
  llvm::StringRef name() const { return _name; }
  // Kind of chunk
  Kind kind() const { return _kind; }
  uint64_t            fileSize() const { return _fsize; }
  uint64_t            align2() const { return _align2; }
  void                appendAtom() const;

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
  void               setMemSize(uint64_t msize) { _msize = msize; }
  // Writer the chunk
  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer) = 0;
  // Finalize the chunk before writing
  virtual void       finalize() = 0;

protected:
  llvm::StringRef _name;
  Kind _kind;
  uint64_t _fsize;
  uint64_t _msize;
  uint64_t _align2;
  uint32_t  _order;
  uint64_t _ordinal;
  uint64_t _start;
  uint64_t _fileoffset;
  const ELFTargetInfo &_targetInfo;
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_CHUNKS_H_
