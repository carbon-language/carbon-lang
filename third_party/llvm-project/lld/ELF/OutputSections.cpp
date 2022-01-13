//===- OutputSections.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "Config.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/TimeProfiler.h"
#include <regex>
#include <unordered_set>

using namespace llvm;
using namespace llvm::dwarf;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

uint8_t *Out::bufferStart;
uint8_t Out::first;
PhdrEntry *Out::tlsPhdr;
OutputSection *Out::elfHeader;
OutputSection *Out::programHeaders;
OutputSection *Out::preinitArray;
OutputSection *Out::initArray;
OutputSection *Out::finiArray;

std::vector<OutputSection *> elf::outputSections;

uint32_t OutputSection::getPhdrFlags() const {
  uint32_t ret = 0;
  if (config->emachine != EM_ARM || !(flags & SHF_ARM_PURECODE))
    ret |= PF_R;
  if (flags & SHF_WRITE)
    ret |= PF_W;
  if (flags & SHF_EXECINSTR)
    ret |= PF_X;
  return ret;
}

template <class ELFT>
void OutputSection::writeHeaderTo(typename ELFT::Shdr *shdr) {
  shdr->sh_entsize = entsize;
  shdr->sh_addralign = alignment;
  shdr->sh_type = type;
  shdr->sh_offset = offset;
  shdr->sh_flags = flags;
  shdr->sh_info = info;
  shdr->sh_link = link;
  shdr->sh_addr = addr;
  shdr->sh_size = size;
  shdr->sh_name = shName;
}

OutputSection::OutputSection(StringRef name, uint32_t type, uint64_t flags)
    : BaseCommand(OutputSectionKind),
      SectionBase(Output, name, flags, /*Entsize*/ 0, /*Alignment*/ 1, type,
                  /*Info*/ 0, /*Link*/ 0) {}

// We allow sections of types listed below to merged into a
// single progbits section. This is typically done by linker
// scripts. Merging nobits and progbits will force disk space
// to be allocated for nobits sections. Other ones don't require
// any special treatment on top of progbits, so there doesn't
// seem to be a harm in merging them.
//
// NOTE: clang since rL252300 emits SHT_X86_64_UNWIND .eh_frame sections. Allow
// them to be merged into SHT_PROGBITS .eh_frame (GNU as .cfi_*).
static bool canMergeToProgbits(unsigned type) {
  return type == SHT_NOBITS || type == SHT_PROGBITS || type == SHT_INIT_ARRAY ||
         type == SHT_PREINIT_ARRAY || type == SHT_FINI_ARRAY ||
         type == SHT_NOTE ||
         (type == SHT_X86_64_UNWIND && config->emachine == EM_X86_64);
}

// Record that isec will be placed in the OutputSection. isec does not become
// permanent until finalizeInputSections() is called. The function should not be
// used after finalizeInputSections() is called. If you need to add an
// InputSection post finalizeInputSections(), then you must do the following:
//
// 1. Find or create an InputSectionDescription to hold InputSection.
// 2. Add the InputSection to the InputSectionDescription::sections.
// 3. Call commitSection(isec).
void OutputSection::recordSection(InputSectionBase *isec) {
  partition = isec->partition;
  isec->parent = this;
  if (sectionCommands.empty() ||
      !isa<InputSectionDescription>(sectionCommands.back()))
    sectionCommands.push_back(make<InputSectionDescription>(""));
  auto *isd = cast<InputSectionDescription>(sectionCommands.back());
  isd->sectionBases.push_back(isec);
}

// Update fields (type, flags, alignment, etc) according to the InputSection
// isec. Also check whether the InputSection flags and type are consistent with
// other InputSections.
void OutputSection::commitSection(InputSection *isec) {
  if (!hasInputSections) {
    // If IS is the first section to be added to this section,
    // initialize type, entsize and flags from isec.
    hasInputSections = true;
    type = isec->type;
    entsize = isec->entsize;
    flags = isec->flags;
  } else {
    // Otherwise, check if new type or flags are compatible with existing ones.
    if ((flags ^ isec->flags) & SHF_TLS)
      error("incompatible section flags for " + name + "\n>>> " + toString(isec) +
            ": 0x" + utohexstr(isec->flags) + "\n>>> output section " + name +
            ": 0x" + utohexstr(flags));

    if (type != isec->type) {
      if (!canMergeToProgbits(type) || !canMergeToProgbits(isec->type))
        error("section type mismatch for " + isec->name + "\n>>> " +
              toString(isec) + ": " +
              getELFSectionTypeName(config->emachine, isec->type) +
              "\n>>> output section " + name + ": " +
              getELFSectionTypeName(config->emachine, type));
      type = SHT_PROGBITS;
    }
  }
  if (noload)
    type = SHT_NOBITS;

  isec->parent = this;
  uint64_t andMask =
      config->emachine == EM_ARM ? (uint64_t)SHF_ARM_PURECODE : 0;
  uint64_t orMask = ~andMask;
  uint64_t andFlags = (flags & isec->flags) & andMask;
  uint64_t orFlags = (flags | isec->flags) & orMask;
  flags = andFlags | orFlags;
  if (nonAlloc)
    flags &= ~(uint64_t)SHF_ALLOC;

  alignment = std::max(alignment, isec->alignment);

  // If this section contains a table of fixed-size entries, sh_entsize
  // holds the element size. If it contains elements of different size we
  // set sh_entsize to 0.
  if (entsize != isec->entsize)
    entsize = 0;
}

// This function scans over the InputSectionBase list sectionBases to create
// InputSectionDescription::sections.
//
// It removes MergeInputSections from the input section array and adds
// new synthetic sections at the location of the first input section
// that it replaces. It then finalizes each synthetic section in order
// to compute an output offset for each piece of each input section.
void OutputSection::finalizeInputSections() {
  std::vector<MergeSyntheticSection *> mergeSections;
  for (BaseCommand *base : sectionCommands) {
    auto *cmd = dyn_cast<InputSectionDescription>(base);
    if (!cmd)
      continue;
    cmd->sections.reserve(cmd->sectionBases.size());
    for (InputSectionBase *s : cmd->sectionBases) {
      MergeInputSection *ms = dyn_cast<MergeInputSection>(s);
      if (!ms) {
        cmd->sections.push_back(cast<InputSection>(s));
        continue;
      }

      // We do not want to handle sections that are not alive, so just remove
      // them instead of trying to merge.
      if (!ms->isLive())
        continue;

      auto i = llvm::find_if(mergeSections, [=](MergeSyntheticSection *sec) {
        // While we could create a single synthetic section for two different
        // values of Entsize, it is better to take Entsize into consideration.
        //
        // With a single synthetic section no two pieces with different Entsize
        // could be equal, so we may as well have two sections.
        //
        // Using Entsize in here also allows us to propagate it to the synthetic
        // section.
        //
        // SHF_STRINGS section with different alignments should not be merged.
        return sec->flags == ms->flags && sec->entsize == ms->entsize &&
               (sec->alignment == ms->alignment || !(sec->flags & SHF_STRINGS));
      });
      if (i == mergeSections.end()) {
        MergeSyntheticSection *syn =
            createMergeSynthetic(name, ms->type, ms->flags, ms->alignment);
        mergeSections.push_back(syn);
        i = std::prev(mergeSections.end());
        syn->entsize = ms->entsize;
        cmd->sections.push_back(syn);
      }
      (*i)->addSection(ms);
    }

    // sectionBases should not be used from this point onwards. Clear it to
    // catch misuses.
    cmd->sectionBases.clear();

    // Some input sections may be removed from the list after ICF.
    for (InputSection *s : cmd->sections)
      commitSection(s);
  }
  for (auto *ms : mergeSections)
    ms->finalizeContents();
}

static void sortByOrder(MutableArrayRef<InputSection *> in,
                        llvm::function_ref<int(InputSectionBase *s)> order) {
  std::vector<std::pair<int, InputSection *>> v;
  for (InputSection *s : in)
    v.push_back({order(s), s});
  llvm::stable_sort(v, less_first());

  for (size_t i = 0; i < v.size(); ++i)
    in[i] = v[i].second;
}

uint64_t elf::getHeaderSize() {
  if (config->oFormatBinary)
    return 0;
  return Out::elfHeader->size + Out::programHeaders->size;
}

bool OutputSection::classof(const BaseCommand *c) {
  return c->kind == OutputSectionKind;
}

void OutputSection::sort(llvm::function_ref<int(InputSectionBase *s)> order) {
  assert(isLive());
  for (BaseCommand *b : sectionCommands)
    if (auto *isd = dyn_cast<InputSectionDescription>(b))
      sortByOrder(isd->sections, order);
}

static void nopInstrFill(uint8_t *buf, size_t size) {
  if (size == 0)
    return;
  unsigned i = 0;
  if (size == 0)
    return;
  std::vector<std::vector<uint8_t>> nopFiller = *target->nopInstrs;
  unsigned num = size / nopFiller.back().size();
  for (unsigned c = 0; c < num; ++c) {
    memcpy(buf + i, nopFiller.back().data(), nopFiller.back().size());
    i += nopFiller.back().size();
  }
  unsigned remaining = size - i;
  if (!remaining)
    return;
  assert(nopFiller[remaining - 1].size() == remaining);
  memcpy(buf + i, nopFiller[remaining - 1].data(), remaining);
}

// Fill [Buf, Buf + Size) with Filler.
// This is used for linker script "=fillexp" command.
static void fill(uint8_t *buf, size_t size,
                 const std::array<uint8_t, 4> &filler) {
  size_t i = 0;
  for (; i + 4 < size; i += 4)
    memcpy(buf + i, filler.data(), 4);
  memcpy(buf + i, filler.data(), size - i);
}

// Compress section contents if this section contains debug info.
template <class ELFT> void OutputSection::maybeCompress() {
  using Elf_Chdr = typename ELFT::Chdr;

  // Compress only DWARF debug sections.
  if (!config->compressDebugSections || (flags & SHF_ALLOC) ||
      !name.startswith(".debug_"))
    return;

  llvm::TimeTraceScope timeScope("Compress debug sections");

  // Create a section header.
  zDebugHeader.resize(sizeof(Elf_Chdr));
  auto *hdr = reinterpret_cast<Elf_Chdr *>(zDebugHeader.data());
  hdr->ch_type = ELFCOMPRESS_ZLIB;
  hdr->ch_size = size;
  hdr->ch_addralign = alignment;

  // Write section contents to a temporary buffer and compress it.
  std::vector<uint8_t> buf(size);
  writeTo<ELFT>(buf.data());
  // We chose 1 as the default compression level because it is the fastest. If
  // -O2 is given, we use level 6 to compress debug info more by ~15%. We found
  // that level 7 to 9 doesn't make much difference (~1% more compression) while
  // they take significant amount of time (~2x), so level 6 seems enough.
  if (Error e = zlib::compress(toStringRef(buf), compressedData,
                               config->optimize >= 2 ? 6 : 1))
    fatal("compress failed: " + llvm::toString(std::move(e)));

  // Update section headers.
  size = sizeof(Elf_Chdr) + compressedData.size();
  flags |= SHF_COMPRESSED;
}

static void writeInt(uint8_t *buf, uint64_t data, uint64_t size) {
  if (size == 1)
    *buf = data;
  else if (size == 2)
    write16(buf, data);
  else if (size == 4)
    write32(buf, data);
  else if (size == 8)
    write64(buf, data);
  else
    llvm_unreachable("unsupported Size argument");
}

template <class ELFT> void OutputSection::writeTo(uint8_t *buf) {
  if (type == SHT_NOBITS)
    return;

  // If -compress-debug-section is specified and if this is a debug section,
  // we've already compressed section contents. If that's the case,
  // just write it down.
  if (!compressedData.empty()) {
    memcpy(buf, zDebugHeader.data(), zDebugHeader.size());
    memcpy(buf + zDebugHeader.size(), compressedData.data(),
           compressedData.size());
    return;
  }

  // Write leading padding.
  std::vector<InputSection *> sections = getInputSections(this);
  std::array<uint8_t, 4> filler = getFiller();
  bool nonZeroFiller = read32(filler.data()) != 0;
  if (nonZeroFiller)
    fill(buf, sections.empty() ? size : sections[0]->outSecOff, filler);

  parallelForEachN(0, sections.size(), [&](size_t i) {
    InputSection *isec = sections[i];
    isec->writeTo<ELFT>(buf);

    // Fill gaps between sections.
    if (nonZeroFiller) {
      uint8_t *start = buf + isec->outSecOff + isec->getSize();
      uint8_t *end;
      if (i + 1 == sections.size())
        end = buf + size;
      else
        end = buf + sections[i + 1]->outSecOff;
      if (isec->nopFiller) {
        assert(target->nopInstrs);
        nopInstrFill(start, end - start);
      } else
        fill(start, end - start, filler);
    }
  });

  // Linker scripts may have BYTE()-family commands with which you
  // can write arbitrary bytes to the output. Process them if any.
  for (BaseCommand *base : sectionCommands)
    if (auto *data = dyn_cast<ByteCommand>(base))
      writeInt(buf + data->offset, data->expression().getValue(), data->size);
}

static void finalizeShtGroup(OutputSection *os,
                             InputSection *section) {
  assert(config->relocatable);

  // sh_link field for SHT_GROUP sections should contain the section index of
  // the symbol table.
  os->link = in.symTab->getParent()->sectionIndex;

  // sh_info then contain index of an entry in symbol table section which
  // provides signature of the section group.
  ArrayRef<Symbol *> symbols = section->file->getSymbols();
  os->info = in.symTab->getSymbolIndex(symbols[section->info]);

  // Some group members may be combined or discarded, so we need to compute the
  // new size. The content will be rewritten in InputSection::copyShtGroup.
  std::unordered_set<uint32_t> seen;
  ArrayRef<InputSectionBase *> sections = section->file->getSections();
  for (const uint32_t &idx : section->getDataAs<uint32_t>().slice(1))
    if (OutputSection *osec = sections[read32(&idx)]->getOutputSection())
      seen.insert(osec->sectionIndex);
  os->size = (1 + seen.size()) * sizeof(uint32_t);
}

void OutputSection::finalize() {
  InputSection *first = getFirstInputSection(this);

  if (flags & SHF_LINK_ORDER) {
    // We must preserve the link order dependency of sections with the
    // SHF_LINK_ORDER flag. The dependency is indicated by the sh_link field. We
    // need to translate the InputSection sh_link to the OutputSection sh_link,
    // all InputSections in the OutputSection have the same dependency.
    if (auto *ex = dyn_cast<ARMExidxSyntheticSection>(first))
      link = ex->getLinkOrderDep()->getParent()->sectionIndex;
    else if (first->flags & SHF_LINK_ORDER)
      if (auto *d = first->getLinkOrderDep())
        link = d->getParent()->sectionIndex;
  }

  if (type == SHT_GROUP) {
    finalizeShtGroup(this, first);
    return;
  }

  if (!config->copyRelocs || (type != SHT_RELA && type != SHT_REL))
    return;

  // Skip if 'first' is synthetic, i.e. not a section created by --emit-relocs.
  // Normally 'type' was changed by 'first' so 'first' should be non-null.
  // However, if the output section is .rela.dyn, 'type' can be set by the empty
  // synthetic .rela.plt and first can be null.
  if (!first || isa<SyntheticSection>(first))
    return;

  link = in.symTab->getParent()->sectionIndex;
  // sh_info for SHT_REL[A] sections should contain the section header index of
  // the section to which the relocation applies.
  InputSectionBase *s = first->getRelocatedSection();
  info = s->getOutputSection()->sectionIndex;
  flags |= SHF_INFO_LINK;
}

// Returns true if S is in one of the many forms the compiler driver may pass
// crtbegin files.
//
// Gcc uses any of crtbegin[<empty>|S|T].o.
// Clang uses Gcc's plus clang_rt.crtbegin[<empty>|S|T][-<arch>|<empty>].o.

static bool isCrtbegin(StringRef s) {
  static std::regex re(R"((clang_rt\.)?crtbegin[ST]?(-.*)?\.o)");
  s = sys::path::filename(s);
  return std::regex_match(s.begin(), s.end(), re);
}

static bool isCrtend(StringRef s) {
  static std::regex re(R"((clang_rt\.)?crtend[ST]?(-.*)?\.o)");
  s = sys::path::filename(s);
  return std::regex_match(s.begin(), s.end(), re);
}

// .ctors and .dtors are sorted by this order:
//
// 1. .ctors/.dtors in crtbegin (which contains a sentinel value -1).
// 2. The section is named ".ctors" or ".dtors" (priority: 65536).
// 3. The section has an optional priority value in the form of ".ctors.N" or
//    ".dtors.N" where N is a number in the form of %05u (priority: 65535-N).
// 4. .ctors/.dtors in crtend (which contains a sentinel value 0).
//
// For 2 and 3, the sections are sorted by priority from high to low, e.g.
// .ctors (65536), .ctors.00100 (65436), .ctors.00200 (65336).  In GNU ld's
// internal linker scripts, the sorting is by string comparison which can
// achieve the same goal given the optional priority values are of the same
// length.
//
// In an ideal world, we don't need this function because .init_array and
// .ctors are duplicate features (and .init_array is newer.) However, there
// are too many real-world use cases of .ctors, so we had no choice to
// support that with this rather ad-hoc semantics.
static bool compCtors(const InputSection *a, const InputSection *b) {
  bool beginA = isCrtbegin(a->file->getName());
  bool beginB = isCrtbegin(b->file->getName());
  if (beginA != beginB)
    return beginA;
  bool endA = isCrtend(a->file->getName());
  bool endB = isCrtend(b->file->getName());
  if (endA != endB)
    return endB;
  return getPriority(a->name) > getPriority(b->name);
}

// Sorts input sections by the special rules for .ctors and .dtors.
// Unfortunately, the rules are different from the one for .{init,fini}_array.
// Read the comment above.
void OutputSection::sortCtorsDtors() {
  assert(sectionCommands.size() == 1);
  auto *isd = cast<InputSectionDescription>(sectionCommands[0]);
  llvm::stable_sort(isd->sections, compCtors);
}

// If an input string is in the form of "foo.N" where N is a number, return N
// (65535-N if .ctors.N or .dtors.N). Otherwise, returns 65536, which is one
// greater than the lowest priority.
int elf::getPriority(StringRef s) {
  size_t pos = s.rfind('.');
  if (pos == StringRef::npos)
    return 65536;
  int v = 65536;
  if (to_integer(s.substr(pos + 1), v, 10) &&
      (pos == 6 && (s.startswith(".ctors") || s.startswith(".dtors"))))
    v = 65535 - v;
  return v;
}

InputSection *elf::getFirstInputSection(const OutputSection *os) {
  for (BaseCommand *base : os->sectionCommands)
    if (auto *isd = dyn_cast<InputSectionDescription>(base))
      if (!isd->sections.empty())
        return isd->sections[0];
  return nullptr;
}

std::vector<InputSection *> elf::getInputSections(const OutputSection *os) {
  std::vector<InputSection *> ret;
  for (BaseCommand *base : os->sectionCommands)
    if (auto *isd = dyn_cast<InputSectionDescription>(base))
      ret.insert(ret.end(), isd->sections.begin(), isd->sections.end());
  return ret;
}

// Sorts input sections by section name suffixes, so that .foo.N comes
// before .foo.M if N < M. Used to sort .{init,fini}_array.N sections.
// We want to keep the original order if the priorities are the same
// because the compiler keeps the original initialization order in a
// translation unit and we need to respect that.
// For more detail, read the section of the GCC's manual about init_priority.
void OutputSection::sortInitFini() {
  // Sort sections by priority.
  sort([](InputSectionBase *s) { return getPriority(s->name); });
}

std::array<uint8_t, 4> OutputSection::getFiller() {
  if (filler)
    return *filler;
  if (flags & SHF_EXECINSTR)
    return target->trapInstr;
  return {0, 0, 0, 0};
}

void OutputSection::checkDynRelAddends(const uint8_t *bufStart) {
  assert(config->writeAddends && config->checkDynamicRelocs);
  assert(type == SHT_REL || type == SHT_RELA);
  std::vector<InputSection *> sections = getInputSections(this);
  parallelForEachN(0, sections.size(), [&](size_t i) {
    // When linking with -r or --emit-relocs we might also call this function
    // for input .rel[a].<sec> sections which we simply pass through to the
    // output. We skip over those and only look at the synthetic relocation
    // sections created during linking.
    const auto *sec = dyn_cast<RelocationBaseSection>(sections[i]);
    if (!sec)
      return;
    for (const DynamicReloc &rel : sec->relocs) {
      int64_t addend = rel.computeAddend();
      const OutputSection *relOsec = rel.inputSec->getOutputSection();
      assert(relOsec != nullptr && "missing output section for relocation");
      const uint8_t *relocTarget =
          bufStart + relOsec->offset + rel.inputSec->getOffset(rel.offsetInSec);
      // For SHT_NOBITS the written addend is always zero.
      int64_t writtenAddend =
          relOsec->type == SHT_NOBITS
              ? 0
              : target->getImplicitAddend(relocTarget, rel.type);
      if (addend != writtenAddend)
        internalLinkerError(
            getErrorLocation(relocTarget),
            "wrote incorrect addend value 0x" + utohexstr(writtenAddend) +
                " instead of 0x" + utohexstr(addend) +
                " for dynamic relocation " + toString(rel.type) +
                " at offset 0x" + utohexstr(rel.getOffset()) +
                (rel.sym ? " against symbol " + toString(*rel.sym) : ""));
    }
  });
}

template void OutputSection::writeHeaderTo<ELF32LE>(ELF32LE::Shdr *Shdr);
template void OutputSection::writeHeaderTo<ELF32BE>(ELF32BE::Shdr *Shdr);
template void OutputSection::writeHeaderTo<ELF64LE>(ELF64LE::Shdr *Shdr);
template void OutputSection::writeHeaderTo<ELF64BE>(ELF64BE::Shdr *Shdr);

template void OutputSection::writeTo<ELF32LE>(uint8_t *Buf);
template void OutputSection::writeTo<ELF32BE>(uint8_t *Buf);
template void OutputSection::writeTo<ELF64LE>(uint8_t *Buf);
template void OutputSection::writeTo<ELF64BE>(uint8_t *Buf);

template void OutputSection::maybeCompress<ELF32LE>();
template void OutputSection::maybeCompress<ELF32BE>();
template void OutputSection::maybeCompress<ELF64LE>();
template void OutputSection::maybeCompress<ELF64BE>();
