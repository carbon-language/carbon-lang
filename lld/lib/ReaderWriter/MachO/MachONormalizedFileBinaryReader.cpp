//===- lib/ReaderWriter/MachO/MachONormalizedFileBinaryReader.cpp ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file For mach-o object files, this implementation converts from
/// mach-o on-disk binary format to in-memory normalized mach-o.
///
///                 +---------------+
///                 | binary mach-o |
///                 +---------------+
///                        |
///                        |
///                        v
///                  +------------+
///                  | normalized |
///                  +------------+

#include "MachONormalizedFile.h"
#include "MachONormalizedFileBinaryUtils.h"
#include "ReferenceKinds.h"

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <functional>

using namespace llvm::MachO;

namespace lld {
namespace mach_o {
namespace normalized {

// Utility to call a lambda expression on each load command.
static error_code 
forEachLoadCommand(StringRef lcRange, unsigned lcCount, bool swap, bool is64,
                   std::function<bool (uint32_t cmd, uint32_t size, 
                                                      const char* lc)> func) {
  const char* p = lcRange.begin();
  for (unsigned i=0; i < lcCount; ++i) {
    const load_command *lc = reinterpret_cast<const load_command*>(p);
    load_command lcCopy;
    const load_command *slc = lc;
    if (swap) {
      memcpy(&lcCopy, lc, sizeof(load_command));
      swapStruct(lcCopy);
      slc = &lcCopy;
    }
    if ( (p + slc->cmdsize) > lcRange.end() )
      return llvm::make_error_code(llvm::errc::executable_format_error);
  
    if (func(slc->cmd, slc->cmdsize, p))
      return error_code::success();
  
    p += slc->cmdsize;
  } 
  
  return error_code::success();
}


static error_code 
appendRelocations(Relocations &relocs, StringRef buffer, bool swap, 
                             bool bigEndian, uint32_t reloff, uint32_t nreloc) {
  if ((reloff + nreloc*8) > buffer.size())
    return llvm::make_error_code(llvm::errc::executable_format_error);
  const any_relocation_info* relocsArray = 
            reinterpret_cast<const any_relocation_info*>(buffer.begin()+reloff); 
  
  for(uint32_t i=0; i < nreloc; ++i) {
    relocs.push_back(unpackRelocation(relocsArray[i], swap, bigEndian));
  }
  return error_code::success();
}



/// Reads a mach-o file and produces an in-memory normalized view.
ErrorOr<std::unique_ptr<NormalizedFile>> 
readBinary(std::unique_ptr<MemoryBuffer> &mb) {
  // Make empty NormalizedFile.
  std::unique_ptr<NormalizedFile> f(new NormalizedFile());

  // Determine endianness and pointer size for mach-o file.
  const mach_header *mh = reinterpret_cast<const mach_header*>
                                                      (mb->getBufferStart());
  bool is64, swap;
  switch (mh->magic) {
  case llvm::MachO::MH_MAGIC:
    is64 = false;
    swap = false;
    break;
  case llvm::MachO::MH_MAGIC_64:
    is64 = true;
    swap = false;
    break;
  case llvm::MachO::MH_CIGAM:
    is64 = false;
    swap = true;
    break;
  case llvm::MachO::MH_CIGAM_64:
    is64 = true;
    swap = true;
    break;
  default:
    return llvm::make_error_code(llvm::errc::executable_format_error);
  }

  // Endian swap header, if needed.
  mach_header headerCopy;
  const mach_header *smh = mh;
  if (swap) {
    memcpy(&headerCopy, mh, sizeof(mach_header));
    swapStruct(headerCopy);
    smh = &headerCopy;
  }

  // Validate head and load commands fit in buffer.
  const uint32_t lcCount = smh->ncmds;
  const char* lcStart = mb->getBufferStart() + (is64 ? sizeof(mach_header_64) 
                                                     : sizeof(mach_header));
  StringRef lcRange(lcStart, smh->sizeofcmds);
  if (lcRange.end() > mb->getBufferEnd())
    return llvm::make_error_code(llvm::errc::executable_format_error);

  // Normalize architecture
  f->arch = MachOLinkingContext::archFromCpuType(smh->cputype, smh->cpusubtype);
  bool isBigEndianArch = MachOLinkingContext::isBigEndian(f->arch);
  // Copy file type and flags
  f->fileType = HeaderFileType(smh->filetype);
  f->flags = smh->flags;


  // Walk load commands looking for segments/sections and the symbol table.
  error_code ec = forEachLoadCommand(lcRange, lcCount, swap, is64, 
                    [&] (uint32_t cmd, uint32_t size, const char* lc) -> bool {
    if (is64) {
      if (cmd == LC_SEGMENT_64) {
        const segment_command_64 *seg = 
                              reinterpret_cast<const segment_command_64*>(lc);
        const unsigned sectionCount = (swap ? SwapByteOrder(seg->nsects)
                                            : seg->nsects);
        const section_64 *sects = reinterpret_cast<const section_64*>
                                  (lc + sizeof(segment_command_64));
        const unsigned lcSize = sizeof(segment_command_64) 
                                              + sectionCount*sizeof(section_64);
        // Verify sections don't extend beyond end of segment load command.
        if (lcSize > size) 
          return llvm::make_error_code(llvm::errc::executable_format_error);
        for (unsigned i=0; i < sectionCount; ++i) {
          const section_64 *sect = &sects[i];
          Section section;
          section.segmentName = getString16(sect->segname);
          section.sectionName = getString16(sect->sectname);
          section.type        = (SectionType)(read32(swap, sect->flags) 
                                                                & SECTION_TYPE);
          section.attributes  = read32(swap, sect->flags) & SECTION_ATTRIBUTES;
          section.alignment   = read32(swap, sect->align);
          section.address     = read64(swap, sect->addr);
          const char *content = mb->getBufferStart() 
                                           + read32(swap, sect->offset);
          size_t contentSize = read64(swap, sect->size);
          // Note: this assign() is copying the content bytes.  Ideally,
          // we can use a custom allocator for vector to avoid the copy.
          section.content.assign(content, content+contentSize);
          appendRelocations(section.relocations, mb->getBuffer(), 
                            swap, isBigEndianArch, read32(swap, sect->reloff), 
                                                   read32(swap, sect->nreloc));
          f->sections.push_back(section);
        }
      }
    } else {
      if (cmd == LC_SEGMENT) {
        const segment_command *seg = 
                              reinterpret_cast<const segment_command*>(lc);
        const unsigned sectionCount = (swap ? SwapByteOrder(seg->nsects)
                                            : seg->nsects);
        const section *sects = reinterpret_cast<const section*>
                                  (lc + sizeof(segment_command));
        const unsigned lcSize = sizeof(segment_command) 
                                              + sectionCount*sizeof(section);
        // Verify sections don't extend beyond end of segment load command.
        if (lcSize > size) 
          return llvm::make_error_code(llvm::errc::executable_format_error);
        for (unsigned i=0; i < sectionCount; ++i) {
          const section *sect = &sects[i];
          Section section;
          section.segmentName = getString16(sect->segname);
          section.sectionName = getString16(sect->sectname);
          section.type        = (SectionType)(read32(swap, sect->flags) 
                                                                & SECTION_TYPE);
          section.attributes  = read32(swap, sect->flags) & SECTION_ATTRIBUTES;
          section.alignment   = read32(swap, sect->align);
          section.address     = read32(swap, sect->addr);
          const char *content = mb->getBufferStart() 
                                           + read32(swap, sect->offset);
          size_t contentSize = read32(swap, sect->size);
          // Note: this assign() is copying the content bytes.  Ideally,
          // we can use a custom allocator for vector to avoid the copy.
          section.content.assign(content, content+contentSize);
          appendRelocations(section.relocations, mb->getBuffer(), 
                            swap, isBigEndianArch, read32(swap, sect->reloff), 
                                                   read32(swap, sect->nreloc));
          f->sections.push_back(section);
        }
      }
    }
    if (cmd == LC_SYMTAB) {
      const symtab_command *st = reinterpret_cast<const symtab_command*>(lc);
      const char* strings = mb->getBufferStart() + read32(swap, st->stroff);
      const uint32_t strSize = read32(swap, st->strsize);
      // Validate string pool and symbol table all in buffer.
      if ( read32(swap, st->stroff)+read32(swap, st->strsize) 
                                                        > mb->getBufferSize() )
        return llvm::make_error_code(llvm::errc::executable_format_error);
      if (is64) {
        const uint32_t symOffset = read32(swap, st->symoff);
        const uint32_t symCount = read32(swap, st->nsyms);
        if ( symOffset+(symCount*sizeof(nlist_64)) > mb->getBufferSize())
          return llvm::make_error_code(llvm::errc::executable_format_error);
        const nlist_64* symbols = reinterpret_cast<const nlist_64*> 
                                            (mb->getBufferStart() + symOffset);
        // Convert each nlist_64 to a lld::mach_o::normalized::Symbol.
        for(uint32_t i=0; i < symCount; ++i) {
          const nlist_64 *sin = &symbols[i];
          nlist_64 tempSym;
          if (swap) {
            tempSym = *sin; swapStruct(tempSym); sin = &tempSym;
          }
          Symbol sout;
          if (sin->n_strx > strSize) 
            return llvm::make_error_code(llvm::errc::executable_format_error);
          sout.name  = &strings[sin->n_strx];
          sout.type  = (NListType)(sin->n_type & N_TYPE);
          sout.scope = (sin->n_type & (N_PEXT|N_EXT));
          sout.sect  = sin->n_sect;
          sout.desc  = sin->n_desc;
          sout.value = sin->n_value;
          if (sout.type == N_UNDF)
            f->undefinedSymbols.push_back(sout);
          else if (sout.scope == (SymbolScope)N_EXT)
            f->globalSymbols.push_back(sout);
          else
            f->localSymbols.push_back(sout);
        }
      } else { 
        const uint32_t symOffset = read32(swap, st->symoff);
        const uint32_t symCount = read32(swap, st->nsyms);
        if ( symOffset+(symCount*sizeof(nlist)) > mb->getBufferSize())
          return llvm::make_error_code(llvm::errc::executable_format_error);
        const nlist* symbols = reinterpret_cast<const nlist*> 
                                            (mb->getBufferStart() + symOffset);
        // Convert each nlist to a lld::mach_o::normalized::Symbol.
        for(uint32_t i=0; i < symCount; ++i) {
          const nlist *sin = &symbols[i];
          nlist tempSym;
          if (swap) {
            tempSym = *sin; swapStruct(tempSym); sin = &tempSym;
          }
          Symbol sout;
          if (sin->n_strx > strSize) 
            return llvm::make_error_code(llvm::errc::executable_format_error);
          sout.name  = &strings[sin->n_strx];
          sout.type  = (NListType)(sin->n_type & N_TYPE);
          sout.scope = (sin->n_type & (N_PEXT|N_EXT));
          sout.sect  = sin->n_sect;
          sout.desc  = sin->n_desc;
          sout.value = sin->n_value;
          if (sout.type == N_UNDF)
            f->undefinedSymbols.push_back(sout);
          else if (sout.scope == (SymbolScope)N_EXT)
            f->globalSymbols.push_back(sout);
          else
            f->localSymbols.push_back(sout);
        }
      }
    } else if (cmd == LC_DYSYMTAB) {
      // TODO: indirect symbols 
    }

    return false;
  });
  if (ec) 
    return ec;

  return std::move(f);
}


} // namespace normalized
} // namespace mach_o

void Registry::addSupportMachOObjects(StringRef archName) {
  MachOLinkingContext::Arch arch = MachOLinkingContext::archFromName(archName);
  switch (arch) {
  case MachOLinkingContext::arch_x86_64:
    addKindTable(Reference::KindNamespace::mach_o, Reference::KindArch::x86_64,
                 mach_o::KindHandler_x86_64::kindStrings);
    break;
  case MachOLinkingContext::arch_x86:
    addKindTable(Reference::KindNamespace::mach_o, Reference::KindArch::x86,
                 mach_o::KindHandler_x86::kindStrings);
    break;
  case MachOLinkingContext::arch_armv6:
  case MachOLinkingContext::arch_armv7:
  case MachOLinkingContext::arch_armv7s:
    addKindTable(Reference::KindNamespace::mach_o, Reference::KindArch::ARM,
                 mach_o::KindHandler_arm::kindStrings);
    break;
  default:
    llvm_unreachable("mach-o arch not supported");
  }
}

} // namespace lld

