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

using llvm::sys::getSwappedBytes;

inline void swapStruct(llvm::MachO::mach_header &mh) {
  mh.magic      = getSwappedBytes(mh.magic);
  mh.cputype    = getSwappedBytes(mh.cputype);
  mh.cpusubtype = getSwappedBytes(mh.cpusubtype);
  mh.filetype   = getSwappedBytes(mh.filetype);
  mh.ncmds      = getSwappedBytes(mh.ncmds);
  mh.sizeofcmds = getSwappedBytes(mh.sizeofcmds);
  mh.flags      = getSwappedBytes(mh.flags);
}

inline void swapStruct(llvm::MachO::load_command &lc) {
  lc.cmd      = getSwappedBytes(lc.cmd);
  lc.cmdsize  = getSwappedBytes(lc.cmdsize);
}

inline void swapStruct(llvm::MachO::symtab_command &lc) {
  lc.cmd      = getSwappedBytes(lc.cmd);
  lc.cmdsize  = getSwappedBytes(lc.cmdsize);
  lc.symoff   = getSwappedBytes(lc.symoff);
  lc.nsyms    = getSwappedBytes(lc.nsyms);
  lc.stroff   = getSwappedBytes(lc.stroff);
  lc.strsize  = getSwappedBytes(lc.strsize);
}

inline void swapStruct(llvm::MachO::segment_command_64 &seg) {
  seg.cmd      = getSwappedBytes(seg.cmd);
  seg.cmdsize  = getSwappedBytes(seg.cmdsize);
  seg.vmaddr   = getSwappedBytes(seg.vmaddr);
  seg.vmsize   = getSwappedBytes(seg.vmsize);
  seg.fileoff  = getSwappedBytes(seg.fileoff);
  seg.filesize = getSwappedBytes(seg.filesize);
  seg.maxprot  = getSwappedBytes(seg.maxprot);
  seg.initprot = getSwappedBytes(seg.initprot);
  seg.nsects   = getSwappedBytes(seg.nsects);
  seg.flags    = getSwappedBytes(seg.flags);
}

inline void swapStruct(llvm::MachO::segment_command &seg) {
  seg.cmd      = getSwappedBytes(seg.cmd);
  seg.cmdsize  = getSwappedBytes(seg.cmdsize);
  seg.vmaddr   = getSwappedBytes(seg.vmaddr);
  seg.vmsize   = getSwappedBytes(seg.vmsize);
  seg.fileoff  = getSwappedBytes(seg.fileoff);
  seg.filesize = getSwappedBytes(seg.filesize);
  seg.maxprot  = getSwappedBytes(seg.maxprot);
  seg.initprot = getSwappedBytes(seg.initprot);
  seg.nsects   = getSwappedBytes(seg.nsects);
  seg.flags    = getSwappedBytes(seg.flags);
}

inline void swapStruct(llvm::MachO::section_64 &sect) {
  sect.addr      = getSwappedBytes(sect.addr);
  sect.size      = getSwappedBytes(sect.size);
  sect.offset    = getSwappedBytes(sect.offset);
  sect.align     = getSwappedBytes(sect.align);
  sect.reloff    = getSwappedBytes(sect.reloff);
  sect.nreloc    = getSwappedBytes(sect.nreloc);
  sect.flags     = getSwappedBytes(sect.flags);
  sect.reserved1 = getSwappedBytes(sect.reserved1);
  sect.reserved2 = getSwappedBytes(sect.reserved2);
}

inline void swapStruct(llvm::MachO::section &sect) {
  sect.addr      = getSwappedBytes(sect.addr);
  sect.size      = getSwappedBytes(sect.size);
  sect.offset    = getSwappedBytes(sect.offset);
  sect.align     = getSwappedBytes(sect.align);
  sect.reloff    = getSwappedBytes(sect.reloff);
  sect.nreloc    = getSwappedBytes(sect.nreloc);
  sect.flags     = getSwappedBytes(sect.flags);
  sect.reserved1 = getSwappedBytes(sect.reserved1);
  sect.reserved2 = getSwappedBytes(sect.reserved2);
}

inline void swapStruct(llvm::MachO::dyld_info_command &info) {
  info.cmd            = getSwappedBytes(info.cmd);
  info.cmdsize        = getSwappedBytes(info.cmdsize);
  info.rebase_off     = getSwappedBytes(info.rebase_off);
  info.rebase_size    = getSwappedBytes(info.rebase_size);
  info.bind_off       = getSwappedBytes(info.bind_off);
  info.bind_size      = getSwappedBytes(info.bind_size);
  info.weak_bind_off  = getSwappedBytes(info.weak_bind_off);
  info.weak_bind_size = getSwappedBytes(info.weak_bind_size);
  info.lazy_bind_off  = getSwappedBytes(info.lazy_bind_off);
  info.lazy_bind_size = getSwappedBytes(info.lazy_bind_size);
  info.export_off     = getSwappedBytes(info.export_off);
  info.export_size    = getSwappedBytes(info.export_size);
}

inline void swapStruct(llvm::MachO::dylib_command &d) {
  d.cmd                         = getSwappedBytes(d.cmd);
  d.cmdsize                     = getSwappedBytes(d.cmdsize);
  d.dylib.name                  = getSwappedBytes(d.dylib.name);
  d.dylib.timestamp             = getSwappedBytes(d.dylib.timestamp);
  d.dylib.current_version       = getSwappedBytes(d.dylib.current_version);
  d.dylib.compatibility_version = getSwappedBytes(d.dylib.compatibility_version);
}

inline void swapStruct(llvm::MachO::dylinker_command &d) {
  d.cmd       = getSwappedBytes(d.cmd);
  d.cmdsize   = getSwappedBytes(d.cmdsize);
  d.name      = getSwappedBytes(d.name);
}

inline void swapStruct(llvm::MachO::entry_point_command &e) {
  e.cmd        = getSwappedBytes(e.cmd);
  e.cmdsize    = getSwappedBytes(e.cmdsize);
  e.entryoff   = getSwappedBytes(e.entryoff);
  e.stacksize  = getSwappedBytes(e.stacksize);
}

inline void swapStruct(llvm::MachO::dysymtab_command &dst) {
  dst.cmd             = getSwappedBytes(dst.cmd);
  dst.cmdsize         = getSwappedBytes(dst.cmdsize);
  dst.ilocalsym       = getSwappedBytes(dst.ilocalsym);
  dst.nlocalsym       = getSwappedBytes(dst.nlocalsym);
  dst.iextdefsym      = getSwappedBytes(dst.iextdefsym);
  dst.nextdefsym      = getSwappedBytes(dst.nextdefsym);
  dst.iundefsym       = getSwappedBytes(dst.iundefsym);
  dst.nundefsym       = getSwappedBytes(dst.nundefsym);
  dst.tocoff          = getSwappedBytes(dst.tocoff);
  dst.ntoc            = getSwappedBytes(dst.ntoc);
  dst.modtaboff       = getSwappedBytes(dst.modtaboff);
  dst.nmodtab         = getSwappedBytes(dst.nmodtab);
  dst.extrefsymoff    = getSwappedBytes(dst.extrefsymoff);
  dst.nextrefsyms     = getSwappedBytes(dst.nextrefsyms);
  dst.indirectsymoff  = getSwappedBytes(dst.indirectsymoff);
  dst.nindirectsyms   = getSwappedBytes(dst.nindirectsyms);
  dst.extreloff       = getSwappedBytes(dst.extreloff);
  dst.nextrel         = getSwappedBytes(dst.nextrel);
  dst.locreloff       = getSwappedBytes(dst.locreloff);
  dst.nlocrel         = getSwappedBytes(dst.nlocrel);
}


inline void swapStruct(llvm::MachO::any_relocation_info &reloc) {
  reloc.r_word0 = getSwappedBytes(reloc.r_word0);
  reloc.r_word1 = getSwappedBytes(reloc.r_word1);
}

inline void swapStruct(llvm::MachO::nlist &sym) {
  sym.n_strx = getSwappedBytes(sym.n_strx);
  sym.n_desc = getSwappedBytes(sym.n_desc);
  sym.n_value = getSwappedBytes(sym.n_value);
}

inline void swapStruct(llvm::MachO::nlist_64 &sym) {
  sym.n_strx = getSwappedBytes(sym.n_strx);
  sym.n_desc = getSwappedBytes(sym.n_desc);
  sym.n_value = getSwappedBytes(sym.n_value);
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
