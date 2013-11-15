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
#include "llvm/Support/system_error.h"

#ifndef LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H
#define LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H

namespace lld {
namespace mach_o {
namespace normalized {

using llvm::sys::SwapByteOrder;

inline void swapStruct(llvm::MachO::mach_header &mh) {
  mh.magic      = SwapByteOrder(mh.magic);
  mh.cputype    = SwapByteOrder(mh.cputype);
  mh.cpusubtype = SwapByteOrder(mh.cpusubtype);
  mh.filetype   = SwapByteOrder(mh.filetype);
  mh.ncmds      = SwapByteOrder(mh.ncmds);
  mh.sizeofcmds = SwapByteOrder(mh.sizeofcmds);
  mh.flags      = SwapByteOrder(mh.flags);
}

inline void swapStruct(llvm::MachO::load_command &lc) {
  lc.cmd      = SwapByteOrder(lc.cmd);
  lc.cmdsize  = SwapByteOrder(lc.cmdsize);
}

inline void swapStruct(llvm::MachO::symtab_command &lc) {
  lc.cmd      = SwapByteOrder(lc.cmd);
  lc.cmdsize  = SwapByteOrder(lc.cmdsize);
  lc.symoff   = SwapByteOrder(lc.symoff);
  lc.nsyms    = SwapByteOrder(lc.nsyms);
  lc.stroff   = SwapByteOrder(lc.stroff);
  lc.strsize  = SwapByteOrder(lc.strsize);
}

inline void swapStruct(llvm::MachO::segment_command_64 &seg) {
  seg.cmd      = SwapByteOrder(seg.cmd);
  seg.cmdsize  = SwapByteOrder(seg.cmdsize);
  seg.vmaddr   = SwapByteOrder(seg.vmaddr);
  seg.vmsize   = SwapByteOrder(seg.vmsize);
  seg.fileoff  = SwapByteOrder(seg.fileoff);
  seg.filesize = SwapByteOrder(seg.filesize);
  seg.maxprot  = SwapByteOrder(seg.maxprot);
  seg.initprot = SwapByteOrder(seg.initprot);
  seg.nsects   = SwapByteOrder(seg.nsects);
  seg.flags    = SwapByteOrder(seg.flags);
}

inline void swapStruct(llvm::MachO::segment_command &seg) {
  seg.cmd      = SwapByteOrder(seg.cmd);
  seg.cmdsize  = SwapByteOrder(seg.cmdsize);
  seg.vmaddr   = SwapByteOrder(seg.vmaddr);
  seg.vmsize   = SwapByteOrder(seg.vmsize);
  seg.fileoff  = SwapByteOrder(seg.fileoff);
  seg.filesize = SwapByteOrder(seg.filesize);
  seg.maxprot  = SwapByteOrder(seg.maxprot);
  seg.initprot = SwapByteOrder(seg.initprot);
  seg.nsects   = SwapByteOrder(seg.nsects);
  seg.flags    = SwapByteOrder(seg.flags);
}

inline void swapStruct(llvm::MachO::section_64 &sect) {
  sect.addr      = SwapByteOrder(sect.addr);
  sect.size      = SwapByteOrder(sect.size);
  sect.offset    = SwapByteOrder(sect.offset);
  sect.align     = SwapByteOrder(sect.align);
  sect.reloff    = SwapByteOrder(sect.reloff);
  sect.nreloc    = SwapByteOrder(sect.nreloc);
  sect.flags     = SwapByteOrder(sect.flags);
  sect.reserved1 = SwapByteOrder(sect.reserved1);
  sect.reserved2 = SwapByteOrder(sect.reserved2);
}

inline void swapStruct(llvm::MachO::section &sect) {
  sect.addr      = SwapByteOrder(sect.addr);
  sect.size      = SwapByteOrder(sect.size);
  sect.offset    = SwapByteOrder(sect.offset);
  sect.align     = SwapByteOrder(sect.align);
  sect.reloff    = SwapByteOrder(sect.reloff);
  sect.nreloc    = SwapByteOrder(sect.nreloc);
  sect.flags     = SwapByteOrder(sect.flags);
  sect.reserved1 = SwapByteOrder(sect.reserved1);
  sect.reserved2 = SwapByteOrder(sect.reserved2);
}

inline void swapStruct(llvm::MachO::dyld_info_command &info) {
  info.cmd            = SwapByteOrder(info.cmd);
  info.cmdsize        = SwapByteOrder(info.cmdsize);
  info.rebase_off     = SwapByteOrder(info.rebase_off);
  info.rebase_size    = SwapByteOrder(info.rebase_size);
  info.bind_off       = SwapByteOrder(info.bind_off);
  info.bind_size      = SwapByteOrder(info.bind_size);
  info.weak_bind_off  = SwapByteOrder(info.weak_bind_off);
  info.weak_bind_size = SwapByteOrder(info.weak_bind_size);
  info.lazy_bind_off  = SwapByteOrder(info.lazy_bind_off);
  info.lazy_bind_size = SwapByteOrder(info.lazy_bind_size);
  info.export_off     = SwapByteOrder(info.export_off);
  info.export_size    = SwapByteOrder(info.export_size);
}

inline void swapStruct(llvm::MachO::dylib_command &d) {
  d.cmd                         = SwapByteOrder(d.cmd);
  d.cmdsize                     = SwapByteOrder(d.cmdsize);
  d.dylib.name                  = SwapByteOrder(d.dylib.name);
  d.dylib.timestamp             = SwapByteOrder(d.dylib.timestamp);
  d.dylib.current_version       = SwapByteOrder(d.dylib.current_version);
  d.dylib.compatibility_version = SwapByteOrder(d.dylib.compatibility_version);
}

inline void swapStruct(llvm::MachO::dylinker_command &d) {
  d.cmd       = SwapByteOrder(d.cmd);
  d.cmdsize   = SwapByteOrder(d.cmdsize);
  d.name      = SwapByteOrder(d.name);
}

inline void swapStruct(llvm::MachO::entry_point_command &e) {
  e.cmd        = SwapByteOrder(e.cmd);
  e.cmdsize    = SwapByteOrder(e.cmdsize);
  e.entryoff   = SwapByteOrder(e.entryoff);
  e.stacksize  = SwapByteOrder(e.stacksize);
}

inline void swapStruct(llvm::MachO::dysymtab_command &dst) {
  dst.cmd             = SwapByteOrder(dst.cmd);
  dst.cmdsize         = SwapByteOrder(dst.cmdsize);
  dst.ilocalsym       = SwapByteOrder(dst.ilocalsym);
  dst.nlocalsym       = SwapByteOrder(dst.nlocalsym);
  dst.iextdefsym      = SwapByteOrder(dst.iextdefsym);
  dst.nextdefsym      = SwapByteOrder(dst.nextdefsym);
  dst.iundefsym       = SwapByteOrder(dst.iundefsym);
  dst.nundefsym       = SwapByteOrder(dst.nundefsym);
  dst.tocoff          = SwapByteOrder(dst.tocoff);
  dst.ntoc            = SwapByteOrder(dst.ntoc);
  dst.modtaboff       = SwapByteOrder(dst.modtaboff);
  dst.nmodtab         = SwapByteOrder(dst.nmodtab);
  dst.extrefsymoff    = SwapByteOrder(dst.extrefsymoff);
  dst.nextrefsyms     = SwapByteOrder(dst.nextrefsyms);
  dst.indirectsymoff  = SwapByteOrder(dst.indirectsymoff);
  dst.nindirectsyms   = SwapByteOrder(dst.nindirectsyms);
  dst.extreloff       = SwapByteOrder(dst.extreloff);
  dst.nextrel         = SwapByteOrder(dst.nextrel);
  dst.locreloff       = SwapByteOrder(dst.locreloff);
  dst.nlocrel         = SwapByteOrder(dst.nlocrel);
}


inline void swapStruct(llvm::MachO::any_relocation_info &reloc) {
  reloc.r_word0 = SwapByteOrder(reloc.r_word0);
  reloc.r_word1 = SwapByteOrder(reloc.r_word1);
}

inline void swapStruct(llvm::MachO::nlist &sym) {
  sym.n_strx = SwapByteOrder(sym.n_strx);
  sym.n_desc = SwapByteOrder(sym.n_desc);
  sym.n_value = SwapByteOrder(sym.n_value);
}

inline void swapStruct(llvm::MachO::nlist_64 &sym) {
  sym.n_strx = SwapByteOrder(sym.n_strx);
  sym.n_desc = SwapByteOrder(sym.n_desc);
  sym.n_value = SwapByteOrder(sym.n_value);
}




inline uint32_t read32(bool swap, uint32_t value) {
    return (swap ? SwapByteOrder(value) : value);
}

inline uint64_t read64(bool swap, uint64_t value) {
    return (swap ? SwapByteOrder(value) : value);
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
  result.r_word0 = swap ? SwapByteOrder(r0) : r0;
  result.r_word1 = swap ? SwapByteOrder(r1) : r1;
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


} // namespace normalized
} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_NORMALIZED_FILE_BINARY_UTILS_H
