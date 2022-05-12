//===- GsymReader.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReader.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReader::GsymReader(std::unique_ptr<MemoryBuffer> Buffer) :
    MemBuffer(std::move(Buffer)),
    Endian(support::endian::system_endianness()) {}

  GsymReader::GsymReader(GsymReader &&RHS) = default;

GsymReader::~GsymReader() = default;

llvm::Expected<GsymReader> GsymReader::openFile(StringRef Filename) {
  // Open the input file and return an appropriate error if needed.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  auto Err = BuffOrErr.getError();
  if (Err)
    return llvm::errorCodeToError(Err);
  return create(BuffOrErr.get());
}

llvm::Expected<GsymReader> GsymReader::copyBuffer(StringRef Bytes) {
  auto MemBuffer = MemoryBuffer::getMemBufferCopy(Bytes, "GSYM bytes");
  return create(MemBuffer);
}

llvm::Expected<llvm::gsym::GsymReader>
GsymReader::create(std::unique_ptr<MemoryBuffer> &MemBuffer) {
  if (!MemBuffer.get())
    return createStringError(std::errc::invalid_argument,
                             "invalid memory buffer");
  GsymReader GR(std::move(MemBuffer));
  llvm::Error Err = GR.parse();
  if (Err)
    return std::move(Err);
  return std::move(GR);
}

llvm::Error
GsymReader::parse() {
  BinaryStreamReader FileData(MemBuffer->getBuffer(),
                              support::endian::system_endianness());
  // Check for the magic bytes. This file format is designed to be mmap'ed
  // into a process and accessed as read only. This is done for performance
  // and efficiency for symbolicating and parsing GSYM data.
  if (FileData.readObject(Hdr))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM header");

  const auto HostByteOrder = support::endian::system_endianness();
  switch (Hdr->Magic) {
    case GSYM_MAGIC:
      Endian = HostByteOrder;
      break;
    case GSYM_CIGAM:
      // This is a GSYM file, but not native endianness.
      Endian = sys::IsBigEndianHost ? support::little : support::big;
      Swap.reset(new SwappedData);
      break;
    default:
      return createStringError(std::errc::invalid_argument,
                               "not a GSYM file");
  }

  bool DataIsLittleEndian = HostByteOrder != support::little;
  // Read a correctly byte swapped header if we need to.
  if (Swap) {
    DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);
    if (auto ExpectedHdr = Header::decode(Data))
      Swap->Hdr = ExpectedHdr.get();
    else
      return ExpectedHdr.takeError();
    Hdr = &Swap->Hdr;
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and a valid UUID size.
  if (Error Err = Hdr->checkForError())
    return Err;

  if (!Swap) {
    // This is the native endianness case that is most common and optimized for
    // efficient lookups. Here we just grab pointers to the native data and
    // use ArrayRef objects to allow efficient read only access.

    // Read the address offsets.
    if (FileData.padToAlignment(Hdr->AddrOffSize) ||
        FileData.readArray(AddrOffsets,
                           Hdr->NumAddresses * Hdr->AddrOffSize))
      return createStringError(std::errc::invalid_argument,
                              "failed to read address table");

    // Read the address info offsets.
    if (FileData.padToAlignment(4) ||
        FileData.readArray(AddrInfoOffsets, Hdr->NumAddresses))
      return createStringError(std::errc::invalid_argument,
                              "failed to read address info offsets table");

    // Read the file table.
    uint32_t NumFiles = 0;
    if (FileData.readInteger(NumFiles) || FileData.readArray(Files, NumFiles))
      return createStringError(std::errc::invalid_argument,
                              "failed to read file table");

    // Get the string table.
    FileData.setOffset(Hdr->StrtabOffset);
    if (FileData.readFixedString(StrTab.Data, Hdr->StrtabSize))
      return createStringError(std::errc::invalid_argument,
                              "failed to read string table");
} else {
  // This is the non native endianness case that is not common and not
  // optimized for lookups. Here we decode the important tables into local
  // storage and then set the ArrayRef objects to point to these swapped
  // copies of the read only data so lookups can be as efficient as possible.
  DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);

  // Read the address offsets.
  uint64_t Offset = alignTo(sizeof(Header), Hdr->AddrOffSize);
  Swap->AddrOffsets.resize(Hdr->NumAddresses * Hdr->AddrOffSize);
  switch (Hdr->AddrOffSize) {
    case 1:
      if (!Data.getU8(&Offset, Swap->AddrOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 2:
      if (!Data.getU16(&Offset,
                        reinterpret_cast<uint16_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 4:
      if (!Data.getU32(&Offset,
                        reinterpret_cast<uint32_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 8:
      if (!Data.getU64(&Offset,
                        reinterpret_cast<uint64_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);

    // Read the address info offsets.
    Offset = alignTo(Offset, 4);
    Swap->AddrInfoOffsets.resize(Hdr->NumAddresses);
    if (Data.getU32(&Offset, Swap->AddrInfoOffsets.data(), Hdr->NumAddresses))
      AddrInfoOffsets = ArrayRef<uint32_t>(Swap->AddrInfoOffsets);
    else
      return createStringError(std::errc::invalid_argument,
                               "failed to read address table");
    // Read the file table.
    const uint32_t NumFiles = Data.getU32(&Offset);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (Data.getU32(&Offset, &Swap->Files[0].Dir, NumFiles*2))
        Files = ArrayRef<FileEntry>(Swap->Files);
      else
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
    }
    // Get the string table.
    StrTab.Data = MemBuffer->getBuffer().substr(Hdr->StrtabOffset,
                                                Hdr->StrtabSize);
    if (StrTab.Data.empty())
      return createStringError(std::errc::invalid_argument,
                               "failed to read string table");
  }
  return Error::success();

}

const Header &GsymReader::getHeader() const {
  // The only way to get a GsymReader is from GsymReader::openFile(...) or
  // GsymReader::copyBuffer() and the header must be valid and initialized to
  // a valid pointer value, so the assert below should not trigger.
  assert(Hdr);
  return *Hdr;
}

Optional<uint64_t> GsymReader::getAddress(size_t Index) const {
  switch (Hdr->AddrOffSize) {
  case 1: return addressForIndex<uint8_t>(Index);
  case 2: return addressForIndex<uint16_t>(Index);
  case 4: return addressForIndex<uint32_t>(Index);
  case 8: return addressForIndex<uint64_t>(Index);
  }
  return llvm::None;
}

Optional<uint64_t> GsymReader::getAddressInfoOffset(size_t Index) const {
  const auto NumAddrInfoOffsets = AddrInfoOffsets.size();
  if (Index < NumAddrInfoOffsets)
    return AddrInfoOffsets[Index];
  return llvm::None;
}

Expected<uint64_t>
GsymReader::getAddressIndex(const uint64_t Addr) const {
  if (Addr >= Hdr->BaseAddress) {
    const uint64_t AddrOffset = Addr - Hdr->BaseAddress;
    Optional<uint64_t> AddrOffsetIndex;
    switch (Hdr->AddrOffSize) {
    case 1:
      AddrOffsetIndex = getAddressOffsetIndex<uint8_t>(AddrOffset);
      break;
    case 2:
      AddrOffsetIndex = getAddressOffsetIndex<uint16_t>(AddrOffset);
      break;
    case 4:
      AddrOffsetIndex = getAddressOffsetIndex<uint32_t>(AddrOffset);
      break;
    case 8:
      AddrOffsetIndex = getAddressOffsetIndex<uint64_t>(AddrOffset);
      break;
    default:
      return createStringError(std::errc::invalid_argument,
                               "unsupported address offset size %u",
                               Hdr->AddrOffSize);
    }
    if (AddrOffsetIndex)
      return *AddrOffsetIndex;
  }
  return createStringError(std::errc::invalid_argument,
                           "address 0x%" PRIx64 " is not in GSYM", Addr);

}

llvm::Expected<FunctionInfo> GsymReader::getFunctionInfo(uint64_t Addr) const {
  Expected<uint64_t> AddressIndex = getAddressIndex(Addr);
  if (!AddressIndex)
    return AddressIndex.takeError();
  // Address info offsets size should have been checked in parse().
  assert(*AddressIndex < AddrInfoOffsets.size());
  auto AddrInfoOffset = AddrInfoOffsets[*AddressIndex];
  DataExtractor Data(MemBuffer->getBuffer().substr(AddrInfoOffset), Endian, 4);
  if (Optional<uint64_t> OptAddr = getAddress(*AddressIndex)) {
    auto ExpectedFI = FunctionInfo::decode(Data, *OptAddr);
    if (ExpectedFI) {
      if (ExpectedFI->Range.contains(Addr) || ExpectedFI->Range.size() == 0)
        return ExpectedFI;
      return createStringError(std::errc::invalid_argument,
                                "address 0x%" PRIx64 " is not in GSYM", Addr);
    }
  }
  return createStringError(std::errc::invalid_argument,
                           "failed to extract address[%" PRIu64 "]",
                           *AddressIndex);
}

llvm::Expected<LookupResult> GsymReader::lookup(uint64_t Addr) const {
  Expected<uint64_t> AddressIndex = getAddressIndex(Addr);
  if (!AddressIndex)
    return AddressIndex.takeError();
  // Address info offsets size should have been checked in parse().
  assert(*AddressIndex < AddrInfoOffsets.size());
  auto AddrInfoOffset = AddrInfoOffsets[*AddressIndex];
  DataExtractor Data(MemBuffer->getBuffer().substr(AddrInfoOffset), Endian, 4);
  if (Optional<uint64_t> OptAddr = getAddress(*AddressIndex))
    return FunctionInfo::lookup(Data, *this, *OptAddr, Addr);
  return createStringError(std::errc::invalid_argument,
                           "failed to extract address[%" PRIu64 "]",
                           *AddressIndex);
}

void GsymReader::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  // Dump the GSYM header.
  OS << Header << "\n";
  // Dump the address table.
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET";

  switch (Hdr->AddrOffSize) {
  case 1: OS << "8 "; break;
  case 2: OS << "16"; break;
  case 4: OS << "32"; break;
  case 8: OS << "64"; break;
  default: OS << "??"; break;
  }
  OS << " (ADDRESS)\n";
  OS << "====== =============================== \n";
  for (uint32_t I = 0; I < Header.NumAddresses; ++I) {
    OS << format("[%4u] ", I);
    switch (Hdr->AddrOffSize) {
    case 1: OS << HEX8(getAddrOffsets<uint8_t>()[I]); break;
    case 2: OS << HEX16(getAddrOffsets<uint16_t>()[I]); break;
    case 4: OS << HEX32(getAddrOffsets<uint32_t>()[I]); break;
    case 8: OS << HEX32(getAddrOffsets<uint64_t>()[I]); break;
    default: break;
    }
    OS << " (" << HEX64(*getAddress(I)) << ")\n";
  }
  // Dump the address info offsets table.
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < Header.NumAddresses; ++I)
    OS << format("[%4u] ", I) << HEX32(AddrInfoOffsets[I]) << "\n";
  // Dump the file table.
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== ==============================\n";
  for (uint32_t I = 0; I < Files.size(); ++I) {
    OS << format("[%4u] ", I) << HEX32(Files[I].Dir) << ' '
       << HEX32(Files[I].Base) << ' ';
    dump(OS, getFile(I));
    OS << "\n";
  }
  OS << "\n" << StrTab << "\n";

  for (uint32_t I = 0; I < Header.NumAddresses; ++I) {
    OS << "FunctionInfo @ " << HEX32(AddrInfoOffsets[I]) << ": ";
    if (auto FI = getFunctionInfo(*getAddress(I)))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}

void GsymReader::dump(raw_ostream &OS, const FunctionInfo &FI) {
  OS << FI.Range << " \"" << getString(FI.Name) << "\"\n";
  if (FI.OptLineTable)
    dump(OS, *FI.OptLineTable);
  if (FI.Inline)
    dump(OS, *FI.Inline);
}

void GsymReader::dump(raw_ostream &OS, const LineTable &LT) {
  OS << "LineTable:\n";
  for (auto &LE: LT) {
    OS << "  " << HEX64(LE.Addr) << ' ';
    if (LE.File)
      dump(OS, getFile(LE.File));
    OS << ':' << LE.Line << '\n';
  }
}

void GsymReader::dump(raw_ostream &OS, const InlineInfo &II, uint32_t Indent) {
  if (Indent == 0)
    OS << "InlineInfo:\n";
  else
    OS.indent(Indent);
  OS << II.Ranges << ' ' << getString(II.Name);
  if (II.CallFile != 0) {
    if (auto File = getFile(II.CallFile)) {
      OS << " called from ";
      dump(OS, File);
      OS << ':' << II.CallLine;
    }
  }
  OS << '\n';
  for (const auto &ChildII: II.Children)
    dump(OS, ChildII, Indent + 2);
}

void GsymReader::dump(raw_ostream &OS, Optional<FileEntry> FE) {
  if (FE) {
    // IF we have the file from index 0, then don't print anything
    if (FE->Dir == 0 && FE->Base == 0)
      return;
    StringRef Dir = getString(FE->Dir);
    StringRef Base = getString(FE->Base);
    if (!Dir.empty()) {
      OS << Dir;
      if (Dir.contains('\\') && !Dir.contains('/'))
        OS << '\\';
      else
        OS << '/';
    }
    if (!Base.empty()) {
      OS << Base;
    }
    if (!Dir.empty() || !Base.empty())
      return;
  }
  OS << "<invalid-file>";
}
