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

#include "ArchHandler.h"
#include "MachONormalizedFileBinaryUtils.h"

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/SharedLibraryFile.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <system_error>

using namespace llvm::MachO;

namespace lld {
namespace mach_o {
namespace normalized {

// Utility to call a lambda expression on each load command.
static std::error_code forEachLoadCommand(
    StringRef lcRange, unsigned lcCount, bool swap, bool is64,
    std::function<bool(uint32_t cmd, uint32_t size, const char *lc)> func) {
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
      return make_error_code(llvm::errc::executable_format_error);

    if (func(slc->cmd, slc->cmdsize, p))
      return std::error_code();

    p += slc->cmdsize;
  }

  return std::error_code();
}

static std::error_code appendRelocations(Relocations &relocs, StringRef buffer,
                                         bool swap, bool bigEndian,
                                         uint32_t reloff, uint32_t nreloc) {
  if ((reloff + nreloc*8) > buffer.size())
    return make_error_code(llvm::errc::executable_format_error);
  const any_relocation_info* relocsArray =
            reinterpret_cast<const any_relocation_info*>(buffer.begin()+reloff);

  for(uint32_t i=0; i < nreloc; ++i) {
    relocs.push_back(unpackRelocation(relocsArray[i], swap, bigEndian));
  }
  return std::error_code();
}

static std::error_code
appendIndirectSymbols(IndirectSymbols &isyms, StringRef buffer, bool swap,
                      bool bigEndian, uint32_t istOffset, uint32_t istCount,
                      uint32_t startIndex, uint32_t count) {
  if ((istOffset + istCount*4) > buffer.size())
    return make_error_code(llvm::errc::executable_format_error);
  if (startIndex+count  > istCount)
    return make_error_code(llvm::errc::executable_format_error);
  const uint32_t *indirectSymbolArray =
            reinterpret_cast<const uint32_t*>(buffer.begin()+istOffset);

  for(uint32_t i=0; i < count; ++i) {
    isyms.push_back(read32(swap, indirectSymbolArray[startIndex+i]));
  }
  return std::error_code();
}


template <typename T> static T readBigEndian(T t) {
  if (llvm::sys::IsLittleEndianHost)
    llvm::sys::swapByteOrder(t);
  return t;
}

/// Reads a mach-o file and produces an in-memory normalized view.
ErrorOr<std::unique_ptr<NormalizedFile>>
readBinary(std::unique_ptr<MemoryBuffer> &mb,
           const MachOLinkingContext::Arch arch) {
  // Make empty NormalizedFile.
  std::unique_ptr<NormalizedFile> f(new NormalizedFile());

  const char *start = mb->getBufferStart();
  size_t objSize = mb->getBufferSize();

  // Determine endianness and pointer size for mach-o file.
  const mach_header *mh = reinterpret_cast<const mach_header *>(start);
  bool isFat = mh->magic == llvm::MachO::FAT_CIGAM ||
               mh->magic == llvm::MachO::FAT_MAGIC;
  if (isFat) {
    uint32_t cputype = MachOLinkingContext::cpuTypeFromArch(arch);
    uint32_t cpusubtype = MachOLinkingContext::cpuSubtypeFromArch(arch);
    const fat_header *fh = reinterpret_cast<const fat_header *>(start);
    uint32_t nfat_arch = readBigEndian(fh->nfat_arch);
    const fat_arch *fa =
        reinterpret_cast<const fat_arch *>(start + sizeof(fat_header));
    bool foundArch = false;
    while (nfat_arch-- > 0) {
      if (readBigEndian(fa->cputype) == cputype &&
          readBigEndian(fa->cpusubtype) == cpusubtype) {
        foundArch = true;
        break;
      }
      fa++;
    }
    if (!foundArch) {
      return make_dynamic_error_code(Twine("file does not contain required"
                                    " architecture ("
                                    + MachOLinkingContext::nameFromArch(arch)
                                    + ")" ));
    }
    objSize = readBigEndian(fa->size);
    uint32_t offset = readBigEndian(fa->offset);
    if ((offset + objSize) > mb->getBufferSize())
      return make_error_code(llvm::errc::executable_format_error);
    start += offset;
    mh = reinterpret_cast<const mach_header *>(start);
  }

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
    return make_error_code(llvm::errc::executable_format_error);
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
  const char *lcStart =
      start + (is64 ? sizeof(mach_header_64) : sizeof(mach_header));
  StringRef lcRange(lcStart, smh->sizeofcmds);
  if (lcRange.end() > (start + objSize))
    return make_error_code(llvm::errc::executable_format_error);

  // Get architecture from mach_header.
  f->arch = MachOLinkingContext::archFromCpuType(smh->cputype, smh->cpusubtype);
  if (f->arch != arch) {
    return make_dynamic_error_code(Twine("file is wrong architecture. Expected "
                                  "(" + MachOLinkingContext::nameFromArch(arch)
                                  + ") found ("
                                  + MachOLinkingContext::nameFromArch(f->arch)
                                  + ")" ));
  }
  bool isBigEndianArch = MachOLinkingContext::isBigEndian(f->arch);
  // Copy file type and flags
  f->fileType = HeaderFileType(smh->filetype);
  f->flags = smh->flags;


  // Pre-scan load commands looking for indirect symbol table.
  uint32_t indirectSymbolTableOffset = 0;
  uint32_t indirectSymbolTableCount = 0;
  std::error_code ec = forEachLoadCommand(lcRange, lcCount, swap, is64,
                                          [&](uint32_t cmd, uint32_t size,
                                              const char *lc) -> bool {
    if (cmd == LC_DYSYMTAB) {
      const dysymtab_command *d = reinterpret_cast<const dysymtab_command*>(lc);
      indirectSymbolTableOffset = read32(swap, d->indirectsymoff);
      indirectSymbolTableCount = read32(swap, d->nindirectsyms);
      return true;
    }
    return false;
  });
  if (ec)
    return ec;

  // Walk load commands looking for segments/sections and the symbol table.
  const data_in_code_entry *dataInCode = nullptr;
  uint32_t dataInCodeSize = 0;
  ec = forEachLoadCommand(lcRange, lcCount, swap, is64,
                    [&] (uint32_t cmd, uint32_t size, const char* lc) -> bool {
    switch(cmd) {
    case LC_SEGMENT_64:
      if (is64) {
        const segment_command_64 *seg =
                              reinterpret_cast<const segment_command_64*>(lc);
        const unsigned sectionCount = (swap
                                       ? llvm::sys::getSwappedBytes(seg->nsects)
                                       : seg->nsects);
        const section_64 *sects = reinterpret_cast<const section_64*>
                                  (lc + sizeof(segment_command_64));
        const unsigned lcSize = sizeof(segment_command_64)
                                              + sectionCount*sizeof(section_64);
        // Verify sections don't extend beyond end of segment load command.
        if (lcSize > size)
          return true;
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
          const uint8_t *content =
              (uint8_t *)start + read32(swap, sect->offset);
          size_t contentSize = read64(swap, sect->size);
          // Note: this assign() is copying the content bytes.  Ideally,
          // we can use a custom allocator for vector to avoid the copy.
          section.content = llvm::makeArrayRef(content, contentSize);
          appendRelocations(section.relocations, mb->getBuffer(),
                            swap, isBigEndianArch, read32(swap, sect->reloff),
                                                   read32(swap, sect->nreloc));
          if (section.type == S_NON_LAZY_SYMBOL_POINTERS) {
            appendIndirectSymbols(section.indirectSymbols, mb->getBuffer(),
                                  swap, isBigEndianArch,
                                  indirectSymbolTableOffset,
                                  indirectSymbolTableCount,
                                  read32(swap, sect->reserved1), contentSize/4);
          }
          f->sections.push_back(section);
        }
      }
      break;
    case LC_SEGMENT:
      if (!is64) {
        const segment_command *seg =
                              reinterpret_cast<const segment_command*>(lc);
        const unsigned sectionCount = (swap
                                       ? llvm::sys::getSwappedBytes(seg->nsects)
                                       : seg->nsects);
        const section *sects = reinterpret_cast<const section*>
                                  (lc + sizeof(segment_command));
        const unsigned lcSize = sizeof(segment_command)
                                              + sectionCount*sizeof(section);
        // Verify sections don't extend beyond end of segment load command.
        if (lcSize > size)
          return true;
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
          const uint8_t *content =
              (uint8_t *)start + read32(swap, sect->offset);
          size_t contentSize = read32(swap, sect->size);
          // Note: this assign() is copying the content bytes.  Ideally,
          // we can use a custom allocator for vector to avoid the copy.
          section.content = llvm::makeArrayRef(content, contentSize);
          appendRelocations(section.relocations, mb->getBuffer(),
                            swap, isBigEndianArch, read32(swap, sect->reloff),
                                                   read32(swap, sect->nreloc));
          if (section.type == S_NON_LAZY_SYMBOL_POINTERS) {
            appendIndirectSymbols(section.indirectSymbols, mb->getBuffer(),
                                  swap, isBigEndianArch,
                                  indirectSymbolTableOffset,
                                  indirectSymbolTableCount,
                                  read32(swap, sect->reserved1), contentSize/4);
          }
          f->sections.push_back(section);
        }
      }
      break;
    case LC_SYMTAB: {
      const symtab_command *st = reinterpret_cast<const symtab_command*>(lc);
      const char *strings = start + read32(swap, st->stroff);
      const uint32_t strSize = read32(swap, st->strsize);
      // Validate string pool and symbol table all in buffer.
      if ( read32(swap, st->stroff)+read32(swap, st->strsize)
                                                        > objSize )
        return true;
      if (is64) {
        const uint32_t symOffset = read32(swap, st->symoff);
        const uint32_t symCount = read32(swap, st->nsyms);
        if ( symOffset+(symCount*sizeof(nlist_64)) > objSize)
          return true;
        const nlist_64 *symbols =
            reinterpret_cast<const nlist_64 *>(start + symOffset);
        // Convert each nlist_64 to a lld::mach_o::normalized::Symbol.
        for(uint32_t i=0; i < symCount; ++i) {
          const nlist_64 *sin = &symbols[i];
          nlist_64 tempSym;
          if (swap) {
            tempSym = *sin; swapStruct(tempSym); sin = &tempSym;
          }
          Symbol sout;
          if (sin->n_strx > strSize)
            return true;
          sout.name  = &strings[sin->n_strx];
          sout.type  = (NListType)(sin->n_type & N_TYPE);
          sout.scope = (sin->n_type & (N_PEXT|N_EXT));
          sout.sect  = sin->n_sect;
          sout.desc  = sin->n_desc;
          sout.value = sin->n_value;
          if (sout.type == N_UNDF)
            f->undefinedSymbols.push_back(sout);
          else if (sin->n_type & N_EXT)
            f->globalSymbols.push_back(sout);
          else
            f->localSymbols.push_back(sout);
        }
      } else {
        const uint32_t symOffset = read32(swap, st->symoff);
        const uint32_t symCount = read32(swap, st->nsyms);
        if ( symOffset+(symCount*sizeof(nlist)) > objSize)
          return true;
        const nlist *symbols =
            reinterpret_cast<const nlist *>(start + symOffset);
        // Convert each nlist to a lld::mach_o::normalized::Symbol.
        for(uint32_t i=0; i < symCount; ++i) {
          const nlist *sin = &symbols[i];
          nlist tempSym;
          if (swap) {
            tempSym = *sin; swapStruct(tempSym); sin = &tempSym;
          }
          Symbol sout;
          if (sin->n_strx > strSize)
            return true;
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
      }
      break;
    case LC_ID_DYLIB: {
      const dylib_command *dl = reinterpret_cast<const dylib_command*>(lc);
      f->installName = lc + read32(swap, dl->dylib.name);
      }
      break;
    case LC_DATA_IN_CODE: {
      const linkedit_data_command *ldc =
                            reinterpret_cast<const linkedit_data_command*>(lc);
      dataInCode = reinterpret_cast<const data_in_code_entry*>(
                                            start + read32(swap, ldc->dataoff));
      dataInCodeSize = read32(swap, ldc->datasize);
      }
    case LC_LOAD_DYLIB:
    case LC_LOAD_WEAK_DYLIB:
    case LC_REEXPORT_DYLIB:
    case LC_LOAD_UPWARD_DYLIB: {
      const dylib_command *dl = reinterpret_cast<const dylib_command*>(lc);
      DependentDylib entry;
      entry.path = lc + read32(swap, dl->dylib.name);
      entry.kind = LoadCommandType(cmd);
      f->dependentDylibs.push_back(entry);
      }
      break;
    }
    return false;
  });
  if (ec)
    return ec;

  if (dataInCode) {
    // Convert on-disk data_in_code_entry array to DataInCode vector.
    for (unsigned i=0; i < dataInCodeSize/sizeof(data_in_code_entry); ++i) {
      DataInCode entry;
      entry.offset = read32(swap, dataInCode[i].offset);
      entry.length = read16(swap, dataInCode[i].length);
      entry.kind   = (DataRegionType)read16(swap, dataInCode[i].kind);
      f->dataInCode.push_back(entry);
    }
  }

  return std::move(f);
}



class MachOReader : public Reader {
public:
  MachOReader(MachOLinkingContext &ctx) : _ctx(ctx) {}

  bool canParse(file_magic magic, StringRef ext,
                const MemoryBuffer &mb) const override {
    if (magic != llvm::sys::fs::file_magic::macho_object &&
        magic != llvm::sys::fs::file_magic::macho_universal_binary &&
        magic != llvm::sys::fs::file_magic::macho_dynamically_linked_shared_lib)
      return false;
    return (mb.getBufferSize() > 32);
  }

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const Registry &registry,
            std::vector<std::unique_ptr<File>> &result) const override {
    // Convert binary file to normalized mach-o.
    auto normFile = readBinary(mb, _ctx.arch());
    if (std::error_code ec = normFile.getError())
      return ec;
    // Convert normalized mach-o to atoms.
    auto file = normalizedToAtoms(**normFile, mb->getBufferIdentifier(), false);
    if (std::error_code ec = file.getError())
      return ec;

    result.push_back(std::move(*file));

    return std::error_code();
  }
private:
  MachOLinkingContext &_ctx;
};


} // namespace normalized
} // namespace mach_o

void Registry::addSupportMachOObjects(MachOLinkingContext &ctx) {
  MachOLinkingContext::Arch arch = ctx.arch();
  add(std::unique_ptr<Reader>(new mach_o::normalized::MachOReader(ctx)));
  addKindTable(Reference::KindNamespace::mach_o, ctx.archHandler().kindArch(), 
               ctx.archHandler().kindStrings());
  add(std::unique_ptr<YamlIOTaggedDocumentHandler>(
                           new mach_o::MachOYamlIOTaggedDocumentHandler(arch)));
}

} // namespace lld

