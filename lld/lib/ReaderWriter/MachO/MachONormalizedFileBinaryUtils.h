//===- lib/ReaderWriter/MachO/MachONormalizedFileBinaryUtils.h ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "MachONormalizedFile.h"
#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include <system_error>

#ifndef LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H
#define LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H

namespace lld {
namespace mach_o {
namespace normalized {

using llvm::sys::swapByteOrder;
using llvm::sys::getSwappedBytes;

inline void swapStruct(llvm::MachO::mach_header &mh) {
  swapByteOrder(mh.magic);
  swapByteOrder(mh.cputype);
  swapByteOrder(mh.cpusubtype);
  swapByteOrder(mh.filetype);
  swapByteOrder(mh.ncmds);
  swapByteOrder(mh.sizeofcmds);
  swapByteOrder(mh.flags);
}

inline void swapStruct(llvm::MachO::load_command &lc) {
  swapByteOrder(lc.cmd);
  swapByteOrder(lc.cmdsize);
}

inline void swapStruct(llvm::MachO::symtab_command &lc) {
  swapByteOrder(lc.cmd);
  swapByteOrder(lc.cmdsize);
  swapByteOrder(lc.symoff);
  swapByteOrder(lc.nsyms);
  swapByteOrder(lc.stroff);
  swapByteOrder(lc.strsize);
}

inline void swapStruct(llvm::MachO::segment_command_64 &seg) {
  swapByteOrder(seg.cmd);
  swapByteOrder(seg.cmdsize);
  swapByteOrder(seg.vmaddr);
  swapByteOrder(seg.vmsize);
  swapByteOrder(seg.fileoff);
  swapByteOrder(seg.filesize);
  swapByteOrder(seg.maxprot);
  swapByteOrder(seg.initprot);
  swapByteOrder(seg.nsects);
  swapByteOrder(seg.flags);
}

inline void swapStruct(llvm::MachO::segment_command &seg) {
  swapByteOrder(seg.cmd);
  swapByteOrder(seg.cmdsize);
  swapByteOrder(seg.vmaddr);
  swapByteOrder(seg.vmsize);
  swapByteOrder(seg.fileoff);
  swapByteOrder(seg.filesize);
  swapByteOrder(seg.maxprot);
  swapByteOrder(seg.initprot);
  swapByteOrder(seg.nsects);
  swapByteOrder(seg.flags);
}

inline void swapStruct(llvm::MachO::section_64 &sect) {
  swapByteOrder(sect.addr);
  swapByteOrder(sect.size);
  swapByteOrder(sect.offset);
  swapByteOrder(sect.align);
  swapByteOrder(sect.reloff);
  swapByteOrder(sect.nreloc);
  swapByteOrder(sect.flags);
  swapByteOrder(sect.reserved1);
  swapByteOrder(sect.reserved2);
}

inline void swapStruct(llvm::MachO::section &sect) {
  swapByteOrder(sect.addr);
  swapByteOrder(sect.size);
  swapByteOrder(sect.offset);
  swapByteOrder(sect.align);
  swapByteOrder(sect.reloff);
  swapByteOrder(sect.nreloc);
  swapByteOrder(sect.flags);
  swapByteOrder(sect.reserved1);
  swapByteOrder(sect.reserved2);
}

inline void swapStruct(llvm::MachO::dyld_info_command &info) {
  swapByteOrder(info.cmd);
  swapByteOrder(info.cmdsize);
  swapByteOrder(info.rebase_off);
  swapByteOrder(info.rebase_size);
  swapByteOrder(info.bind_off);
  swapByteOrder(info.bind_size);
  swapByteOrder(info.weak_bind_off);
  swapByteOrder(info.weak_bind_size);
  swapByteOrder(info.lazy_bind_off);
  swapByteOrder(info.lazy_bind_size);
  swapByteOrder(info.export_off);
  swapByteOrder(info.export_size);
}

inline void swapStruct(llvm::MachO::dylib_command &d) {
  swapByteOrder(d.cmd);
  swapByteOrder(d.cmdsize);
  swapByteOrder(d.dylib.name);
  swapByteOrder(d.dylib.timestamp);
  swapByteOrder(d.dylib.current_version);
  swapByteOrder(d.dylib.compatibility_version);
}

inline void swapStruct(llvm::MachO::dylinker_command &d) {
  swapByteOrder(d.cmd);
  swapByteOrder(d.cmdsize);
  swapByteOrder(d.name);
}

inline void swapStruct(llvm::MachO::entry_point_command &e) {
  swapByteOrder(e.cmd);
  swapByteOrder(e.cmdsize);
  swapByteOrder(e.entryoff);
  swapByteOrder(e.stacksize);
}

inline void swapStruct(llvm::MachO::dysymtab_command &dst) {
  swapByteOrder(dst.cmd);
  swapByteOrder(dst.cmdsize);
  swapByteOrder(dst.ilocalsym);
  swapByteOrder(dst.nlocalsym);
  swapByteOrder(dst.iextdefsym);
  swapByteOrder(dst.nextdefsym);
  swapByteOrder(dst.iundefsym);
  swapByteOrder(dst.nundefsym);
  swapByteOrder(dst.tocoff);
  swapByteOrder(dst.ntoc);
  swapByteOrder(dst.modtaboff);
  swapByteOrder(dst.nmodtab);
  swapByteOrder(dst.extrefsymoff);
  swapByteOrder(dst.nextrefsyms);
  swapByteOrder(dst.indirectsymoff);
  swapByteOrder(dst.nindirectsyms);
  swapByteOrder(dst.extreloff);
  swapByteOrder(dst.nextrel);
  swapByteOrder(dst.locreloff);
  swapByteOrder(dst.nlocrel);
}


inline void swapStruct(llvm::MachO::any_relocation_info &reloc) {
  swapByteOrder(reloc.r_word0);
  swapByteOrder(reloc.r_word1);
}

inline void swapStruct(llvm::MachO::nlist &sym) {
  swapByteOrder(sym.n_strx);
  swapByteOrder(sym.n_desc);
  swapByteOrder(sym.n_value);
}

inline void swapStruct(llvm::MachO::nlist_64 &sym) {
  swapByteOrder(sym.n_strx);
  swapByteOrder(sym.n_desc);
  swapByteOrder(sym.n_value);
}



inline uint16_t read16(bool swap, uint16_t value) {
    return (swap ? getSwappedBytes(value) : value);
}

inline uint32_t read32(bool swap, uint32_t value) {
    return (swap ? getSwappedBytes(value) : value);
}

inline uint64_t read64(bool swap, uint64_t value) {
    return (swap ? getSwappedBytes(value) : value);
}



inline uint32_t
bitFieldExtract(uint32_t value, bool isBigEndianBigField, uint8_t firstBit,
                                                          uint8_t bitCount) {
  const uint32_t mask = ((1<<bitCount)-1);
  const uint8_t shift = isBigEndianBigField ? (32-firstBit-bitCount) : firstBit;
  return (value >> shift) & mask;
}

inline void
bitFieldSet(uint32_t &bits, bool isBigEndianBigField, uint32_t newBits,
                            uint8_t firstBit, uint8_t bitCount) {
  const uint32_t mask = ((1<<bitCount)-1);
  assert((newBits & mask) == newBits);
  const uint8_t shift = isBigEndianBigField ? (32-firstBit-bitCount) : firstBit;
  bits &= ~(mask << shift);
  bits |= (newBits << shift);
}

inline Relocation
unpackRelocation(const llvm::MachO::any_relocation_info &r, bool swap,
                                                            bool isBigEndian) {
  uint32_t r0 = read32(swap, r.r_word0);
  uint32_t r1 = read32(swap, r.r_word1);

  Relocation result;
  if (r0 & llvm::MachO::R_SCATTERED) {
    // scattered relocation record always laid out like big endian bit field
    result.offset     = bitFieldExtract(r0, true, 8, 24);
    result.scattered  = true;
    result.type       = (RelocationInfoType)
                        bitFieldExtract(r0, true, 4, 4);
    result.length     = bitFieldExtract(r0, true, 2, 2);
    result.pcRel      = bitFieldExtract(r0, true, 1, 1);
    result.isExtern   = false;
    result.value      = r1;
    result.symbol     = 0;
  } else {
    result.offset     = r0;
    result.scattered  = false;
    result.type       = (RelocationInfoType)
                        bitFieldExtract(r1, isBigEndian, 28, 4);
    result.length     = bitFieldExtract(r1, isBigEndian, 25, 2);
    result.pcRel      = bitFieldExtract(r1, isBigEndian, 24, 1);
    result.isExtern   = bitFieldExtract(r1, isBigEndian, 27, 1);
    result.value      = 0;
    result.symbol     = bitFieldExtract(r1, isBigEndian, 0, 24);
  }
  return result;
}


inline llvm::MachO::any_relocation_info
packRelocation(const Relocation &r, bool swap, bool isBigEndian) {
  uint32_t r0 = 0;
  uint32_t r1 = 0;

  if (r.scattered) {
    r1 = r.value;
    bitFieldSet(r0, true, r.offset,    8, 24);
    bitFieldSet(r0, true, r.type,      4, 4);
    bitFieldSet(r0, true, r.length,    2, 2);
    bitFieldSet(r0, true, r.pcRel,     1, 1);
    bitFieldSet(r0, true, r.scattered, 0, 1); // R_SCATTERED
  } else {
    r0 = r.offset;
    bitFieldSet(r1, isBigEndian, r.type,     28, 4);
    bitFieldSet(r1, isBigEndian, r.isExtern, 27, 1);
    bitFieldSet(r1, isBigEndian, r.length,   25, 2);
    bitFieldSet(r1, isBigEndian, r.pcRel,    24, 1);
    bitFieldSet(r1, isBigEndian, r.symbol,   0,  24);
  }

  llvm::MachO::any_relocation_info result;
  result.r_word0 = swap ? getSwappedBytes(r0) : r0;
  result.r_word1 = swap ? getSwappedBytes(r1) : r1;
  return result;
}

inline StringRef getString16(const char s[16]) {
  StringRef x = s;
  if ( x.size() > 16 )
    return x.substr(0, 16);
  else
    return x;
}

inline void setString16(StringRef str, char s[16]) {
  memset(s, 0, 16);
  memcpy(s, str.begin(), (str.size() > 16) ? 16: str.size());
}

// Implemented in normalizedToAtoms() and used by normalizedFromAtoms() so
// that the same table can be used to map mach-o sections to and from
// DefinedAtom::ContentType.
void relocatableSectionInfoForContentType(DefinedAtom::ContentType atomType,
                                          StringRef &segmentName,
                                          StringRef &sectionName,
                                          SectionType &sectionType,
                                          SectionAttr &sectionAttrs);

} // namespace normalized
} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H
