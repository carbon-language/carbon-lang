//===- ReaderWriter/WriterELF.h - ELF File Format Writing Interface -------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_ELF_H_
#define LLD_READERWRITER_WRITER_ELF_H_

#include "lld/ReaderWriter/Writer.h"

#include "lld/Core/LLVM.h"

#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"

namespace lld {

/// \brief The WriterOptionsELF class encapsulates options needed to process ELF
/// files.
///
/// You can create an WriterOptionsELF instance from command line arguments or
/// by subclassing and setting the instance variables in the subclass's
/// constructor.
class WriterOptionsELF {
public:
  WriterOptionsELF()
    : _is64Bit(false)
    , _endianness(llvm::support::little)
    , _type(llvm::ELF::ET_EXEC)
    , _pointerWidth(4)
    , _machine(llvm::ELF::EM_386)
    , _baseAddress(0x400000)
    , _pageSize(0x1000)
    , _entryPoint("start") {}

  /// \brief Create a specific instance of an architecture.
  ///
  /// \param[in] Is64Bit Is this a ELF64 file or ELF32 file.
  /// \param[in] Endianness Is this an ELFDATA2LSB or ELFDATA2MSB file.
  /// \param[in] Type The e_type of the file. (Relocatable, Executable, etc...).
  /// \param[in] Machine The e_machine of the file. (EM_386, EM_X86_86, etc...).
  WriterOptionsELF(const bool Is64Bit,
                   const llvm::support::endianness endian,
                   const uint16_t Type,
                   const uint16_t Machine,
                   uint64_t pointerWidth = 4,
                   uint64_t baseAddress = 0x400000,
                   uint64_t pageSize = 0x1000)
  : _is64Bit(Is64Bit)
  , _endianness(endian)
  , _type(Type)
  , _pointerWidth(pointerWidth)
  , _machine(Machine)
  , _baseAddress(baseAddress)
  , _pageSize(pageSize)
  , _entryPoint("start") {}

  bool is64Bit() const { return _is64Bit; }
  llvm::support::endianness endianness() const { return _endianness; }
  uint16_t type() const { return _type; }
  uint16_t machine() const { return _machine; }
  uint16_t pointerWidth() const { return _pointerWidth; }
  uint64_t baseAddress() const { return _baseAddress; }
  uint64_t pageSize() const { return _pageSize; }
  void setEntryPoint(StringRef name) { _entryPoint = name; }

  /// \brief Get the entry point if type() is ET_EXEC. Empty otherwise.
  StringRef entryPoint() const;

protected:
  bool                      _is64Bit;
  llvm::support::endianness _endianness;
  uint16_t                  _type;
  uint16_t                  _pointerWidth;
  uint16_t                  _machine;
  uint64_t                  _baseAddress;
  uint64_t                  _pageSize;
  StringRef                 _entryPoint;
};

/// \brief Create a WriterELF using the given options.
///
/// The only way to instantiate a WriterELF object is via this function. The
/// Writer object created retains a reference to the WriterOptionsELF object
/// supplied, so it must not be destroyed before the Writer object. 
Writer *createWriterELF(const WriterOptionsELF &Options);

} // namespace lld

#endif // LLD_READERWRITER_WRITER_ELF_H_
