//===- UnwindInfoSection.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindInfoSection.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Parallel.h"

#include <numeric>

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

#define COMMON_ENCODINGS_MAX 127
#define COMPACT_ENCODINGS_MAX 256

#define SECOND_LEVEL_PAGE_BYTES 4096
#define SECOND_LEVEL_PAGE_WORDS (SECOND_LEVEL_PAGE_BYTES / sizeof(uint32_t))
#define REGULAR_SECOND_LEVEL_ENTRIES_MAX                                       \
  ((SECOND_LEVEL_PAGE_BYTES -                                                  \
    sizeof(unwind_info_regular_second_level_page_header)) /                    \
   sizeof(unwind_info_regular_second_level_entry))
#define COMPRESSED_SECOND_LEVEL_ENTRIES_MAX                                    \
  ((SECOND_LEVEL_PAGE_BYTES -                                                  \
    sizeof(unwind_info_compressed_second_level_page_header)) /                 \
   sizeof(uint32_t))

#define COMPRESSED_ENTRY_FUNC_OFFSET_BITS 24
#define COMPRESSED_ENTRY_FUNC_OFFSET_MASK                                      \
  UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET(~0)

// Compact Unwind format is a Mach-O evolution of DWARF Unwind that
// optimizes space and exception-time lookup.  Most DWARF unwind
// entries can be replaced with Compact Unwind entries, but the ones
// that cannot are retained in DWARF form.
//
// This comment will address macro-level organization of the pre-link
// and post-link compact unwind tables. For micro-level organization
// pertaining to the bitfield layout of the 32-bit compact unwind
// entries, see libunwind/include/mach-o/compact_unwind_encoding.h
//
// Important clarifying factoids:
//
// * __LD,__compact_unwind is the compact unwind format for compiler
// output and linker input. It is never a final output. It could be
// an intermediate output with the `-r` option which retains relocs.
//
// * __TEXT,__unwind_info is the compact unwind format for final
// linker output. It is never an input.
//
// * __TEXT,__eh_frame is the DWARF format for both linker input and output.
//
// * __TEXT,__unwind_info entries are divided into 4 KiB pages (2nd
// level) by ascending address, and the pages are referenced by an
// index (1st level) in the section header.
//
// * Following the headers in __TEXT,__unwind_info, the bulk of the
// section contains a vector of compact unwind entries
// `{functionOffset, encoding}` sorted by ascending `functionOffset`.
// Adjacent entries with the same encoding can be folded to great
// advantage, achieving a 3-order-of-magnitude reduction in the
// number of entries.
//
// * The __TEXT,__unwind_info format can accommodate up to 127 unique
// encodings for the space-efficient compressed format. In practice,
// fewer than a dozen unique encodings are used by C++ programs of
// all sizes. Therefore, we don't even bother implementing the regular
// non-compressed format. Time will tell if anyone in the field ever
// overflows the 127-encodings limit.
//
// Refer to the definition of unwind_info_section_header in
// compact_unwind_encoding.h for an overview of the format we are encoding
// here.

// TODO(gkm): prune __eh_frame entries superseded by __unwind_info, PR50410
// TODO(gkm): how do we align the 2nd-level pages?

template <class Ptr> struct CompactUnwindEntry {
  Ptr functionAddress;
  uint32_t functionLength;
  compact_unwind_encoding_t encoding;
  Ptr personality;
  Ptr lsda;
};

using EncodingMap = DenseMap<compact_unwind_encoding_t, size_t>;

struct SecondLevelPage {
  uint32_t kind;
  size_t entryIndex;
  size_t entryCount;
  size_t byteCount;
  std::vector<compact_unwind_encoding_t> localEncodings;
  EncodingMap localEncodingIndexes;
};

template <class Ptr>
class UnwindInfoSectionImpl final : public UnwindInfoSection {
public:
  void prepareRelocations(ConcatInputSection *) override;
  void relocateCompactUnwind(std::vector<CompactUnwindEntry<Ptr>> &);
  Reloc *findLsdaReloc(ConcatInputSection *) const;
  void encodePersonalities();
  void finalize() override;
  void writeTo(uint8_t *buf) const override;

private:
  std::vector<std::pair<compact_unwind_encoding_t, size_t>> commonEncodings;
  EncodingMap commonEncodingIndexes;
  // The entries here will be in the same order as their originating symbols
  // in symbolsVec.
  std::vector<CompactUnwindEntry<Ptr>> cuEntries;
  // Indices into the cuEntries vector.
  std::vector<size_t> cuIndices;
  // Indices of personality functions within the GOT.
  std::vector<Ptr> personalities;
  SmallDenseMap<std::pair<InputSection *, uint64_t /* addend */>, Symbol *>
      personalityTable;
  // Indices into cuEntries for CUEs with a non-null LSDA.
  std::vector<size_t> entriesWithLsda;
  // Map of cuEntries index to an index within the LSDA array.
  DenseMap<size_t, uint32_t> lsdaIndex;
  std::vector<SecondLevelPage> secondLevelPages;
  uint64_t level2PagesOffset = 0;
};

UnwindInfoSection::UnwindInfoSection()
    : SyntheticSection(segment_names::text, section_names::unwindInfo) {
  align = 4;
}

void UnwindInfoSection::prepareRelocations() {
  // This iteration needs to be deterministic, since prepareRelocations may add
  // entries to the GOT. Hence the use of a MapVector for
  // UnwindInfoSection::symbols.
  for (const Defined *d : make_second_range(symbols))
    if (d->unwindEntry)
      prepareRelocations(d->unwindEntry);
}

// Record function symbols that may need entries emitted in __unwind_info, which
// stores unwind data for address ranges.
//
// Note that if several adjacent functions have the same unwind encoding, LSDA,
// and personality function, they share one unwind entry. For this to work,
// functions without unwind info need explicit "no unwind info" unwind entries
// -- else the unwinder would think they have the unwind info of the closest
// function with unwind info right before in the image. Thus, we add function
// symbols for each unique address regardless of whether they have associated
// unwind info.
void UnwindInfoSection::addSymbol(const Defined *d) {
  if (d->unwindEntry)
    allEntriesAreOmitted = false;
  // We don't yet know the final output address of this symbol, but we know that
  // they are uniquely determined by a combination of the isec and value, so
  // we use that as the key here.
  auto p = symbols.insert({{d->isec, d->value}, d});
  // If we have multiple symbols at the same address, only one of them can have
  // an associated CUE.
  if (!p.second && d->unwindEntry) {
    assert(!p.first->second->unwindEntry);
    p.first->second = d;
  }
}

// Compact unwind relocations have different semantics, so we handle them in a
// separate code path from regular relocations. First, we do not wish to add
// rebase opcodes for __LD,__compact_unwind, because that section doesn't
// actually end up in the final binary. Second, personality pointers always
// reside in the GOT and must be treated specially.
template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::prepareRelocations(ConcatInputSection *isec) {
  assert(!isec->shouldOmitFromOutput() &&
         "__compact_unwind section should not be omitted");

  // FIXME: Make this skip relocations for CompactUnwindEntries that
  // point to dead-stripped functions. That might save some amount of
  // work. But since there are usually just few personality functions
  // that are referenced from many places, at least some of them likely
  // live, it wouldn't reduce number of got entries.
  for (size_t i = 0; i < isec->relocs.size(); ++i) {
    Reloc &r = isec->relocs[i];
    assert(target->hasAttr(r.type, RelocAttrBits::UNSIGNED));

    // Functions and LSDA entries always reside in the same object file as the
    // compact unwind entries that references them, and thus appear as section
    // relocs. There is no need to prepare them. We only prepare relocs for
    // personality functions.
    if (r.offset % sizeof(CompactUnwindEntry<Ptr>) !=
        offsetof(CompactUnwindEntry<Ptr>, personality))
      continue;

    if (auto *s = r.referent.dyn_cast<Symbol *>()) {
      // Personality functions are nearly always system-defined (e.g.,
      // ___gxx_personality_v0 for C++) and relocated as dylib symbols.  When an
      // application provides its own personality function, it might be
      // referenced by an extern Defined symbol reloc, or a local section reloc.
      if (auto *defined = dyn_cast<Defined>(s)) {
        // XXX(vyng) This is a a special case for handling duplicate personality
        // symbols. Note that LD64's behavior is a bit different and it is
        // inconsistent with how symbol resolution usually work
        //
        // So we've decided not to follow it. Instead, simply pick the symbol
        // with the same name from the symbol table to replace the local one.
        //
        // (See discussions/alternatives already considered on D107533)
        if (!defined->isExternal())
          if (Symbol *sym = symtab->find(defined->getName()))
            if (!sym->isLazy())
              r.referent = s = sym;
      }
      if (auto *undefined = dyn_cast<Undefined>(s)) {
        treatUndefinedSymbol(*undefined);
        // treatUndefinedSymbol() can replace s with a DylibSymbol; re-check.
        if (isa<Undefined>(s))
          continue;
      }

      if (auto *defined = dyn_cast<Defined>(s)) {
        // Check if we have created a synthetic symbol at the same address.
        Symbol *&personality =
            personalityTable[{defined->isec, defined->value}];
        if (personality == nullptr) {
          personality = defined;
          in.got->addEntry(defined);
        } else if (personality != defined) {
          r.referent = personality;
        }
        continue;
      }
      assert(isa<DylibSymbol>(s));
      in.got->addEntry(s);
      continue;
    }

    if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
      assert(!isCoalescedWeak(referentIsec));
      // Personality functions can be referenced via section relocations
      // if they live in the same object file. Create placeholder synthetic
      // symbols for them in the GOT.
      Symbol *&s = personalityTable[{referentIsec, r.addend}];
      if (s == nullptr) {
        // This runs after dead stripping, so the noDeadStrip argument does not
        // matter.
        s = make<Defined>("<internal>", /*file=*/nullptr, referentIsec,
                          r.addend, /*size=*/0, /*isWeakDef=*/false,
                          /*isExternal=*/false, /*isPrivateExtern=*/false,
                          /*isThumb=*/false, /*isReferencedDynamically=*/false,
                          /*noDeadStrip=*/false);
        in.got->addEntry(s);
      }
      r.referent = s;
      r.addend = 0;
    }
  }
}

// Unwind info lives in __DATA, and finalization of __TEXT will occur before
// finalization of __DATA. Moreover, the finalization of unwind info depends on
// the exact addresses that it references. So it is safe for compact unwind to
// reference addresses in __TEXT, but not addresses in any other segment.
static ConcatInputSection *checkTextSegment(InputSection *isec) {
  if (isec->getSegName() != segment_names::text)
    error("compact unwind references address in " + toString(isec) +
          " which is not in segment __TEXT");
  // __text should always be a ConcatInputSection.
  return cast<ConcatInputSection>(isec);
}

// We need to apply the relocations to the pre-link compact unwind section
// before converting it to post-link form. There should only be absolute
// relocations here: since we are not emitting the pre-link CU section, there
// is no source address to make a relative location meaningful.
template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::relocateCompactUnwind(
    std::vector<CompactUnwindEntry<Ptr>> &cuEntries) {
  parallelForEachN(0, symbolsVec.size(), [&](size_t i) {
    uint8_t *buf = reinterpret_cast<uint8_t *>(cuEntries.data()) +
                   i * sizeof(CompactUnwindEntry<Ptr>);
    const Defined *d = symbolsVec[i].second;
    // Write the functionAddress.
    writeAddress(buf, d->getVA(), sizeof(Ptr) == 8 ? 3 : 2);
    if (!d->unwindEntry)
      return;

    // Write the rest of the CUE.
    memcpy(buf + sizeof(Ptr), d->unwindEntry->data.data(),
           d->unwindEntry->data.size());
    for (const Reloc &r : d->unwindEntry->relocs) {
      uint64_t referentVA = 0;
      if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
        if (!isa<Undefined>(referentSym)) {
          if (auto *defined = dyn_cast<Defined>(referentSym))
            checkTextSegment(defined->isec);
          // At this point in the link, we may not yet know the final address of
          // the GOT, so we just encode the index. We make it a 1-based index so
          // that we can distinguish the null pointer case.
          referentVA = referentSym->gotIndex + 1;
        }
      } else {
        auto *referentIsec = r.referent.get<InputSection *>();
        checkTextSegment(referentIsec);
        referentVA = referentIsec->getVA(r.addend);
      }
      writeAddress(buf + r.offset, referentVA, r.length);
    }
  });
}

// There should only be a handful of unique personality pointers, so we can
// encode them as 2-bit indices into a small array.
template <class Ptr> void UnwindInfoSectionImpl<Ptr>::encodePersonalities() {
  for (size_t idx : cuIndices) {
    CompactUnwindEntry<Ptr> &cu = cuEntries[idx];
    if (cu.personality == 0)
      continue;
    // Linear search is fast enough for a small array.
    auto it = find(personalities, cu.personality);
    uint32_t personalityIndex; // 1-based index
    if (it != personalities.end()) {
      personalityIndex = std::distance(personalities.begin(), it) + 1;
    } else {
      personalities.push_back(cu.personality);
      personalityIndex = personalities.size();
    }
    cu.encoding |=
        personalityIndex << countTrailingZeros(
            static_cast<compact_unwind_encoding_t>(UNWIND_PERSONALITY_MASK));
  }
  if (personalities.size() > 3)
    error("too many personalities (" + std::to_string(personalities.size()) +
          ") for compact unwind to encode");
}

static bool canFoldEncoding(compact_unwind_encoding_t encoding) {
  // From compact_unwind_encoding.h:
  //  UNWIND_X86_64_MODE_STACK_IND:
  //  A "frameless" (RBP not used as frame pointer) function large constant
  //  stack size.  This case is like the previous, except the stack size is too
  //  large to encode in the compact unwind encoding.  Instead it requires that
  //  the function contains "subq $nnnnnnnn,RSP" in its prolog.  The compact
  //  encoding contains the offset to the nnnnnnnn value in the function in
  //  UNWIND_X86_64_FRAMELESS_STACK_SIZE.
  // Since this means the unwinder has to look at the `subq` in the function
  // of the unwind info's unwind address, two functions that have identical
  // unwind info can't be folded if it's using this encoding since both
  // entries need unique addresses.
  static_assert(UNWIND_X86_64_MODE_MASK == UNWIND_X86_MODE_MASK, "");
  static_assert(UNWIND_X86_64_MODE_STACK_IND == UNWIND_X86_MODE_STACK_IND, "");
  if ((target->cpuType == CPU_TYPE_X86_64 || target->cpuType == CPU_TYPE_X86) &&
      (encoding & UNWIND_X86_64_MODE_MASK) == UNWIND_X86_64_MODE_STACK_IND) {
    // FIXME: Consider passing in the two function addresses and getting
    // their two stack sizes off the `subq` and only returning false if they're
    // actually different.
    return false;
  }
  return true;
}

template <class Ptr>
Reloc *
UnwindInfoSectionImpl<Ptr>::findLsdaReloc(ConcatInputSection *isec) const {
  if (isec == nullptr)
    return nullptr;
  auto it = llvm::find_if(isec->relocs, [](const Reloc &r) {
    return r.offset % sizeof(CompactUnwindEntry<Ptr>) ==
           offsetof(CompactUnwindEntry<Ptr>, lsda);
  });
  if (it == isec->relocs.end())
    return nullptr;
  return &*it;
}

// Scan the __LD,__compact_unwind entries and compute the space needs of
// __TEXT,__unwind_info and __TEXT,__eh_frame
template <class Ptr> void UnwindInfoSectionImpl<Ptr>::finalize() {
  if (symbols.empty())
    return;

  // At this point, the address space for __TEXT,__text has been
  // assigned, so we can relocate the __LD,__compact_unwind entries
  // into a temporary buffer. Relocation is necessary in order to sort
  // the CU entries by function address. Sorting is necessary so that
  // we can fold adjacent CU entries with identical
  // encoding+personality+lsda. Folding is necessary because it reduces
  // the number of CU entries by as much as 3 orders of magnitude!
  cuEntries.resize(symbols.size());
  // The "map" part of the symbols MapVector was only needed for deduplication
  // in addSymbol(). Now that we are done adding, move the contents to a plain
  // std::vector for indexed access.
  symbolsVec = symbols.takeVector();
  relocateCompactUnwind(cuEntries);

  // Rather than sort & fold the 32-byte entries directly, we create a
  // vector of indices to entries and sort & fold that instead.
  cuIndices.resize(cuEntries.size());
  std::iota(cuIndices.begin(), cuIndices.end(), 0);
  llvm::sort(cuIndices, [&](size_t a, size_t b) {
    return cuEntries[a].functionAddress < cuEntries[b].functionAddress;
  });

  // Fold adjacent entries with matching encoding+personality+lsda
  // We use three iterators on the same cuIndices to fold in-situ:
  // (1) `foldBegin` is the first of a potential sequence of matching entries
  // (2) `foldEnd` is the first non-matching entry after `foldBegin`.
  // The semi-open interval [ foldBegin .. foldEnd ) contains a range
  // entries that can be folded into a single entry and written to ...
  // (3) `foldWrite`
  auto foldWrite = cuIndices.begin();
  for (auto foldBegin = cuIndices.begin(); foldBegin < cuIndices.end();) {
    auto foldEnd = foldBegin;
    while (++foldEnd < cuIndices.end() &&
           cuEntries[*foldBegin].encoding == cuEntries[*foldEnd].encoding &&
           cuEntries[*foldBegin].personality ==
               cuEntries[*foldEnd].personality &&
           canFoldEncoding(cuEntries[*foldEnd].encoding)) {
      // In most cases, we can just compare the values of cuEntries[*].lsda.
      // However, it is possible for -rename_section to cause the LSDA section
      // (__gcc_except_tab) to be finalized after the unwind info section. In
      // that case, we don't yet have unique addresses for the LSDA entries.
      // So we check their relocations instead.
      // FIXME: should we account for an LSDA at an absolute address? ld64 seems
      // to support it, but it seems unlikely to be used in practice.
      Reloc *lsda1 = findLsdaReloc(symbolsVec[*foldBegin].second->unwindEntry);
      Reloc *lsda2 = findLsdaReloc(symbolsVec[*foldEnd].second->unwindEntry);
      if (lsda1 == nullptr && lsda2 == nullptr)
        continue;
      if (lsda1 == nullptr || lsda2 == nullptr)
        break;
      if (lsda1->referent != lsda2->referent)
        break;
      if (lsda1->addend != lsda2->addend)
        break;
    }
    *foldWrite++ = *foldBegin;
    foldBegin = foldEnd;
  }
  cuIndices.erase(foldWrite, cuIndices.end());

  encodePersonalities();

  // Count frequencies of the folded encodings
  EncodingMap encodingFrequencies;
  for (size_t idx : cuIndices)
    encodingFrequencies[cuEntries[idx].encoding]++;

  // Make a vector of encodings, sorted by descending frequency
  for (const auto &frequency : encodingFrequencies)
    commonEncodings.emplace_back(frequency);
  llvm::sort(commonEncodings,
             [](const std::pair<compact_unwind_encoding_t, size_t> &a,
                const std::pair<compact_unwind_encoding_t, size_t> &b) {
               if (a.second == b.second)
                 // When frequencies match, secondarily sort on encoding
                 // to maintain parity with validate-unwind-info.py
                 return a.first > b.first;
               return a.second > b.second;
             });

  // Truncate the vector to 127 elements.
  // Common encoding indexes are limited to 0..126, while encoding
  // indexes 127..255 are local to each second-level page
  if (commonEncodings.size() > COMMON_ENCODINGS_MAX)
    commonEncodings.resize(COMMON_ENCODINGS_MAX);

  // Create a map from encoding to common-encoding-table index
  for (size_t i = 0; i < commonEncodings.size(); i++)
    commonEncodingIndexes[commonEncodings[i].first] = i;

  // Split folded encodings into pages, where each page is limited by ...
  // (a) 4 KiB capacity
  // (b) 24-bit difference between first & final function address
  // (c) 8-bit compact-encoding-table index,
  //     for which 0..126 references the global common-encodings table,
  //     and 127..255 references a local per-second-level-page table.
  // First we try the compact format and determine how many entries fit.
  // If more entries fit in the regular format, we use that.
  for (size_t i = 0; i < cuIndices.size();) {
    size_t idx = cuIndices[i];
    secondLevelPages.emplace_back();
    SecondLevelPage &page = secondLevelPages.back();
    page.entryIndex = i;
    uintptr_t functionAddressMax =
        cuEntries[idx].functionAddress + COMPRESSED_ENTRY_FUNC_OFFSET_MASK;
    size_t n = commonEncodings.size();
    size_t wordsRemaining =
        SECOND_LEVEL_PAGE_WORDS -
        sizeof(unwind_info_compressed_second_level_page_header) /
            sizeof(uint32_t);
    while (wordsRemaining >= 1 && i < cuIndices.size()) {
      idx = cuIndices[i];
      const CompactUnwindEntry<Ptr> *cuPtr = &cuEntries[idx];
      if (cuPtr->functionAddress >= functionAddressMax) {
        break;
      } else if (commonEncodingIndexes.count(cuPtr->encoding) ||
                 page.localEncodingIndexes.count(cuPtr->encoding)) {
        i++;
        wordsRemaining--;
      } else if (wordsRemaining >= 2 && n < COMPACT_ENCODINGS_MAX) {
        page.localEncodings.emplace_back(cuPtr->encoding);
        page.localEncodingIndexes[cuPtr->encoding] = n++;
        i++;
        wordsRemaining -= 2;
      } else {
        break;
      }
    }
    page.entryCount = i - page.entryIndex;

    // If this is not the final page, see if it's possible to fit more
    // entries by using the regular format. This can happen when there
    // are many unique encodings, and we we saturated the local
    // encoding table early.
    if (i < cuIndices.size() &&
        page.entryCount < REGULAR_SECOND_LEVEL_ENTRIES_MAX) {
      page.kind = UNWIND_SECOND_LEVEL_REGULAR;
      page.entryCount = std::min(REGULAR_SECOND_LEVEL_ENTRIES_MAX,
                                 cuIndices.size() - page.entryIndex);
      i = page.entryIndex + page.entryCount;
    } else {
      page.kind = UNWIND_SECOND_LEVEL_COMPRESSED;
    }
  }

  for (size_t idx : cuIndices) {
    lsdaIndex[idx] = entriesWithLsda.size();
    const Defined *d = symbolsVec[idx].second;
    if (findLsdaReloc(d->unwindEntry))
      entriesWithLsda.push_back(idx);
  }

  // compute size of __TEXT,__unwind_info section
  level2PagesOffset = sizeof(unwind_info_section_header) +
                      commonEncodings.size() * sizeof(uint32_t) +
                      personalities.size() * sizeof(uint32_t) +
                      // The extra second-level-page entry is for the sentinel
                      (secondLevelPages.size() + 1) *
                          sizeof(unwind_info_section_header_index_entry) +
                      entriesWithLsda.size() *
                          sizeof(unwind_info_section_header_lsda_index_entry);
  unwindInfoSize =
      level2PagesOffset + secondLevelPages.size() * SECOND_LEVEL_PAGE_BYTES;
}

// All inputs are relocated and output addresses are known, so write!

template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::writeTo(uint8_t *buf) const {
  assert(!cuIndices.empty() && "call only if there is unwind info");

  // section header
  auto *uip = reinterpret_cast<unwind_info_section_header *>(buf);
  uip->version = 1;
  uip->commonEncodingsArraySectionOffset = sizeof(unwind_info_section_header);
  uip->commonEncodingsArrayCount = commonEncodings.size();
  uip->personalityArraySectionOffset =
      uip->commonEncodingsArraySectionOffset +
      (uip->commonEncodingsArrayCount * sizeof(uint32_t));
  uip->personalityArrayCount = personalities.size();
  uip->indexSectionOffset = uip->personalityArraySectionOffset +
                            (uip->personalityArrayCount * sizeof(uint32_t));
  uip->indexCount = secondLevelPages.size() + 1;

  // Common encodings
  auto *i32p = reinterpret_cast<uint32_t *>(&uip[1]);
  for (const auto &encoding : commonEncodings)
    *i32p++ = encoding.first;

  // Personalities
  for (Ptr personality : personalities)
    *i32p++ =
        in.got->addr + (personality - 1) * target->wordSize - in.header->addr;

  // Level-1 index
  uint32_t lsdaOffset =
      uip->indexSectionOffset +
      uip->indexCount * sizeof(unwind_info_section_header_index_entry);
  uint64_t l2PagesOffset = level2PagesOffset;
  auto *iep = reinterpret_cast<unwind_info_section_header_index_entry *>(i32p);
  for (const SecondLevelPage &page : secondLevelPages) {
    size_t idx = cuIndices[page.entryIndex];
    iep->functionOffset = cuEntries[idx].functionAddress - in.header->addr;
    iep->secondLevelPagesSectionOffset = l2PagesOffset;
    iep->lsdaIndexArraySectionOffset =
        lsdaOffset + lsdaIndex.lookup(idx) *
                         sizeof(unwind_info_section_header_lsda_index_entry);
    iep++;
    l2PagesOffset += SECOND_LEVEL_PAGE_BYTES;
  }
  // Level-1 sentinel
  const CompactUnwindEntry<Ptr> &cuEnd = cuEntries[cuIndices.back()];
  iep->functionOffset =
      cuEnd.functionAddress - in.header->addr + cuEnd.functionLength;
  iep->secondLevelPagesSectionOffset = 0;
  iep->lsdaIndexArraySectionOffset =
      lsdaOffset + entriesWithLsda.size() *
                       sizeof(unwind_info_section_header_lsda_index_entry);
  iep++;

  // LSDAs
  auto *lep =
      reinterpret_cast<unwind_info_section_header_lsda_index_entry *>(iep);
  for (size_t idx : entriesWithLsda) {
    const CompactUnwindEntry<Ptr> &cu = cuEntries[idx];
    const Defined *d = symbolsVec[idx].second;
    if (Reloc *r = findLsdaReloc(d->unwindEntry)) {
      uint64_t va;
      if (auto *isec = r->referent.dyn_cast<InputSection *>()) {
        va = isec->getVA(r->addend);
      } else {
        auto *sym = r->referent.get<Symbol *>();
        va = sym->getVA() + r->addend;
      }
      lep->lsdaOffset = va - in.header->addr;
    }
    lep->functionOffset = cu.functionAddress - in.header->addr;
    lep++;
  }

  // Level-2 pages
  auto *pp = reinterpret_cast<uint32_t *>(lep);
  for (const SecondLevelPage &page : secondLevelPages) {
    if (page.kind == UNWIND_SECOND_LEVEL_COMPRESSED) {
      uintptr_t functionAddressBase =
          cuEntries[cuIndices[page.entryIndex]].functionAddress;
      auto *p2p =
          reinterpret_cast<unwind_info_compressed_second_level_page_header *>(
              pp);
      p2p->kind = page.kind;
      p2p->entryPageOffset =
          sizeof(unwind_info_compressed_second_level_page_header);
      p2p->entryCount = page.entryCount;
      p2p->encodingsPageOffset =
          p2p->entryPageOffset + p2p->entryCount * sizeof(uint32_t);
      p2p->encodingsCount = page.localEncodings.size();
      auto *ep = reinterpret_cast<uint32_t *>(&p2p[1]);
      for (size_t i = 0; i < page.entryCount; i++) {
        const CompactUnwindEntry<Ptr> &cue =
            cuEntries[cuIndices[page.entryIndex + i]];
        auto it = commonEncodingIndexes.find(cue.encoding);
        if (it == commonEncodingIndexes.end())
          it = page.localEncodingIndexes.find(cue.encoding);
        *ep++ = (it->second << COMPRESSED_ENTRY_FUNC_OFFSET_BITS) |
                (cue.functionAddress - functionAddressBase);
      }
      if (!page.localEncodings.empty())
        memcpy(ep, page.localEncodings.data(),
               page.localEncodings.size() * sizeof(uint32_t));
    } else {
      auto *p2p =
          reinterpret_cast<unwind_info_regular_second_level_page_header *>(pp);
      p2p->kind = page.kind;
      p2p->entryPageOffset =
          sizeof(unwind_info_regular_second_level_page_header);
      p2p->entryCount = page.entryCount;
      auto *ep = reinterpret_cast<uint32_t *>(&p2p[1]);
      for (size_t i = 0; i < page.entryCount; i++) {
        const CompactUnwindEntry<Ptr> &cue =
            cuEntries[cuIndices[page.entryIndex + i]];
        *ep++ = cue.functionAddress;
        *ep++ = cue.encoding;
      }
    }
    pp += SECOND_LEVEL_PAGE_WORDS;
  }
}

UnwindInfoSection *macho::makeUnwindInfoSection() {
  if (target->wordSize == 8)
    return make<UnwindInfoSectionImpl<uint64_t>>();
  else
    return make<UnwindInfoSectionImpl<uint32_t>>();
}
