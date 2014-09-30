//===- lib/ReaderWriter/MachO/CompactUnwindPass.cpp -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
//===----------------------------------------------------------------------===//

#include "ArchHandler.h"
#include "File.h"
#include "MachOPasses.h"
#include "MachONormalizedFileBinaryUtils.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Simple.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "macho-compact-unwind"

namespace lld {
namespace mach_o {

namespace {
struct CompactUnwindEntry {
  const Atom *rangeStart;
  const Atom *personalityFunction;
  const Atom *lsdaLocation;

  uint32_t rangeLength;
  uint32_t encoding;
};

struct UnwindInfoPage {
  std::vector<CompactUnwindEntry> entries;
};
}

class UnwindInfoAtom : public SimpleDefinedAtom {
public:
  UnwindInfoAtom(ArchHandler &archHandler, const File &file, bool swap,
                 std::vector<uint32_t> commonEncodings,
                 std::vector<const Atom *> personalities,
                 std::vector<UnwindInfoPage> pages, uint32_t numLSDAs)
      : SimpleDefinedAtom(file), _archHandler(archHandler),
        _commonEncodingsOffset(7 * sizeof(uint32_t)),
        _personalityArrayOffset(_commonEncodingsOffset +
                                commonEncodings.size() * sizeof(uint32_t)),
        _topLevelIndexOffset(_personalityArrayOffset +
                             personalities.size() * sizeof(uint32_t)),
        _lsdaIndexOffset(_topLevelIndexOffset +
                         3 * (pages.size() + 1) * sizeof(uint32_t)),
        _firstPageOffset(_lsdaIndexOffset + 2 * numLSDAs * sizeof(uint32_t)),
        _swap(swap) {

    addHeader(commonEncodings.size(), personalities.size(), pages.size());
    addCommonEncodings(commonEncodings);
    addPersonalityFunctions(personalities);
    addTopLevelIndexes(pages);
    addLSDAIndexes(pages, numLSDAs);
    addSecondLevelPages(pages);
  }

  ContentType contentType() const override {
    return DefinedAtom::typeProcessedUnwindInfo;
  }

  Alignment alignment() const override { return Alignment(2); }

  uint64_t size() const override { return _contents.size(); }

  ContentPermissions permissions() const override {
    return DefinedAtom::permR__;
  }

  ArrayRef<uint8_t> rawContent() const override { return _contents; }

  void addHeader(uint32_t numCommon, uint32_t numPersonalities,
                 uint32_t numPages) {
    using normalized::write32;

    uint32_t headerSize = 7 * sizeof(uint32_t);
    _contents.resize(headerSize);

    int32_t *headerEntries = (int32_t *)_contents.data();
    // version
    write32(headerEntries[0], _swap, 1);
    // commonEncodingsArraySectionOffset
    write32(headerEntries[1], _swap, _commonEncodingsOffset);
    // commonEncodingsArrayCount
    write32(headerEntries[2], _swap, numCommon);
    // personalityArraySectionOffset
    write32(headerEntries[3], _swap, _personalityArrayOffset);
    // personalityArrayCount
    write32(headerEntries[4], _swap, numPersonalities);
    // indexSectionOffset
    write32(headerEntries[5], _swap, _topLevelIndexOffset);
    // indexCount
    write32(headerEntries[6], _swap, numPages + 1);
  }

  /// Add the list of common encodings to the section; this is simply an array
  /// of uint32_t compact values. Size has already been specified in the header.
  void addCommonEncodings(std::vector<uint32_t> &commonEncodings) {
    using normalized::write32;

    _contents.resize(_commonEncodingsOffset +
                     commonEncodings.size() * sizeof(uint32_t));
    int32_t *commonEncodingsArea =
        (int32_t *)&_contents[_commonEncodingsOffset];

    for (uint32_t encoding : commonEncodings)
      write32(*commonEncodingsArea++, _swap, encoding);
  }

  void addPersonalityFunctions(std::vector<const Atom *> personalities) {
    _contents.resize(_personalityArrayOffset +
                     personalities.size() * sizeof(uint32_t));

    for (unsigned i = 0; i < personalities.size(); ++i)
      addImageReferenceIndirect(_personalityArrayOffset + i * sizeof(uint32_t),
                                personalities[i]);
  }

  void addTopLevelIndexes(std::vector<UnwindInfoPage> &pages) {
    using normalized::write32;

    uint32_t numIndexes = pages.size() + 1;
    _contents.resize(_topLevelIndexOffset + numIndexes * 3 * sizeof(uint32_t));

    uint32_t pageLoc = _firstPageOffset;

    // The most difficult job here is calculating the LSDAs; everything else
    // follows fairly naturally, but we can't state where the first
    int32_t *indexData = (int32_t *)&_contents[_topLevelIndexOffset];
    uint32_t numLSDAs = 0;
    for (unsigned i = 0; i < pages.size(); ++i) {
      // functionOffset
      addImageReference(_topLevelIndexOffset + 3 * i * sizeof(uint32_t),
                        pages[i].entries[0].rangeStart);
      // secondLevelPagesSectionOffset
      write32(indexData[3 * i + 1], _swap, pageLoc);
      write32(indexData[3 * i + 2], _swap,
              _lsdaIndexOffset + numLSDAs * 2 * sizeof(uint32_t));

      for (auto &entry : pages[i].entries)
        if (entry.lsdaLocation)
          ++numLSDAs;
    }

    // Finally, write out the final sentinel index
    CompactUnwindEntry &finalEntry = pages[pages.size() - 1].entries.back();
    addImageReference(_topLevelIndexOffset +
                          3 * pages.size() * sizeof(uint32_t),
                      finalEntry.rangeStart, finalEntry.rangeLength);
    // secondLevelPagesSectionOffset => 0
    indexData[3 * pages.size() + 2] =
        _lsdaIndexOffset + numLSDAs * 2 * sizeof(uint32_t);
  }

  void addLSDAIndexes(std::vector<UnwindInfoPage> &pages, uint32_t numLSDAs) {
    _contents.resize(_lsdaIndexOffset + numLSDAs * 2 * sizeof(uint32_t));

    uint32_t curOffset = _lsdaIndexOffset;
    for (auto &page : pages) {
      for (auto &entry : page.entries) {
        if (!entry.lsdaLocation)
          continue;

        addImageReference(curOffset, entry.rangeStart);
        addImageReference(curOffset + sizeof(uint32_t), entry.lsdaLocation);
        curOffset += 2 * sizeof(uint32_t);
      }
    }
  }

  void addSecondLevelPages(std::vector<UnwindInfoPage> &pages) {
    for (auto &page : pages) {
      addRegularSecondLevelPage(page);
    }
  }

  void addRegularSecondLevelPage(const UnwindInfoPage &page) {
    uint32_t curPageOffset = _contents.size();
    const int16_t headerSize = sizeof(uint32_t) + 2 * sizeof(uint16_t);
    uint32_t curPageSize =
        headerSize + 2 * page.entries.size() * sizeof(uint32_t);
    _contents.resize(curPageOffset + curPageSize);

    using normalized::write32;
    using normalized::write16;
    // 2 => regular page
    write32(*(int32_t *)&_contents[curPageOffset], _swap, 2);
    // offset of 1st entry
    write16(*(int16_t *)&_contents[curPageOffset + 4], _swap, headerSize);
    write16(*(int16_t *)&_contents[curPageOffset + 6], _swap,
            page.entries.size());

    uint32_t pagePos = curPageOffset + headerSize;
    for (auto &entry : page.entries) {
      addImageReference(pagePos, entry.rangeStart);
      write32(reinterpret_cast<int32_t *>(_contents.data() + pagePos)[1], _swap,
              entry.encoding);
      pagePos += 2 * sizeof(uint32_t);
    }
  }

  void addImageReference(uint32_t offset, const Atom *dest,
                         Reference::Addend addend = 0) {
    addReference(Reference::KindNamespace::mach_o, _archHandler.kindArch(),
                 _archHandler.imageOffsetKind(), offset, dest, addend);
  }

  void addImageReferenceIndirect(uint32_t offset, const Atom *dest) {
    addReference(Reference::KindNamespace::mach_o, _archHandler.kindArch(),
                 _archHandler.imageOffsetKindIndirect(), offset, dest, 0);
  }

private:
  mach_o::ArchHandler &_archHandler;
  std::vector<uint8_t> _contents;
  uint32_t _commonEncodingsOffset;
  uint32_t _personalityArrayOffset;
  uint32_t _topLevelIndexOffset;
  uint32_t _lsdaIndexOffset;
  uint32_t _firstPageOffset;
  bool _swap;
};

/// Pass for instantiating and optimizing GOT slots.
///
class CompactUnwindPass : public Pass {
public:
  CompactUnwindPass(const MachOLinkingContext &context)
      : _context(context), _archHandler(_context.archHandler()),
        _file("<mach-o Compact Unwind Pass>"),
        _swap(!MachOLinkingContext::isHostEndian(_context.arch())) {}

private:
  void perform(std::unique_ptr<MutableFile> &mergedFile) override {
    DEBUG(llvm::dbgs() << "MachO Compact Unwind pass\n");

    // First collect all __compact_unwind entries, addressable by the function
    // it's referring to.
    std::map<const Atom *, CompactUnwindEntry> unwindLocs;
    std::vector<const Atom *> personalities;
    uint32_t numLSDAs = 0;

    collectCompactUnwindEntries(mergedFile, unwindLocs, personalities,
                                numLSDAs);

    // FIXME: if there are more than 4 personality functions then we need to
    // defer to DWARF info for the ones we don't put in the list. They should
    // also probably be sorted by frequency.
    assert(personalities.size() <= 4);

    // Now sort the entries by final address and fixup the compact encoding to
    // its final form (i.e. set personality function bits & create DWARF
    // references where needed).
    std::vector<CompactUnwindEntry> unwindInfos =
        createUnwindInfoEntries(mergedFile, unwindLocs, personalities);

    // Finally, we can start creating pages based on these entries.

    DEBUG(llvm::dbgs() << "  Splitting entries into pages\n");
    // FIXME: we split the entries into pages naively: lots of 4k pages followed
    // by a small one. ld64 tried to minimize space and align them to real 4k
    // boundaries. That might be worth doing, or perhaps we could perform some
    // minor balancing for expected number of lookups.
    std::vector<UnwindInfoPage> pages;
    unsigned pageStart = 0;
    do {
      pages.push_back(UnwindInfoPage());

      // FIXME: we only create regular pages at the moment. These can hold up to
      // 1021 entries according to the documentation.
      unsigned entriesInPage =
          std::min(1021U, (unsigned)unwindInfos.size() - pageStart);

      std::copy(unwindInfos.begin() + pageStart,
                unwindInfos.begin() + pageStart + entriesInPage,
                std::back_inserter(pages.back().entries));
      pageStart += entriesInPage;

      DEBUG(llvm::dbgs()
            << "    Page from " << pages.back().entries[0].rangeStart->name()
            << " to " << pages.back().entries.back().rangeStart->name() << " + "
            << llvm::format("0x%x", pages.back().entries.back().rangeLength)
            << " has " << entriesInPage << " entries\n");
    } while (pageStart < unwindInfos.size());

    // FIXME: we should also erase all compact-unwind atoms; their job is done.
    UnwindInfoAtom *unwind = new (_file.allocator())
        UnwindInfoAtom(_archHandler, _file, _swap, std::vector<uint32_t>(),
                       personalities, pages, numLSDAs);
    mergedFile->addAtom(*unwind);
  }

  void collectCompactUnwindEntries(
      std::unique_ptr<MutableFile> &mergedFile,
      std::map<const Atom *, CompactUnwindEntry> &unwindLocs,
      std::vector<const Atom *> &personalities, uint32_t &numLSDAs) {
    DEBUG(llvm::dbgs() << "  Collecting __compact_unwind entries\n");

    for (const DefinedAtom *atom : mergedFile->defined()) {
      if (atom->contentType() != DefinedAtom::typeCompactUnwindInfo)
        continue;

      auto unwindEntry = extractCompactUnwindEntry(atom);
      unwindLocs.insert(std::make_pair(unwindEntry.rangeStart, unwindEntry));

      DEBUG(llvm::dbgs() << "    Entry for " << unwindEntry.rangeStart->name()
                         << ", encoding="
                         << llvm::format("0x%08x", unwindEntry.encoding));
      if (unwindEntry.personalityFunction)
        DEBUG(llvm::dbgs() << ", personality="
                           << unwindEntry.personalityFunction->name()
                           << ", lsdaLoc=" << unwindEntry.lsdaLocation->name());
      DEBUG(llvm::dbgs() << '\n');

      // Count number of LSDAs we see, since we need to know how big the index
      // will be while laying out the section.
      if (unwindEntry.lsdaLocation)
        ++numLSDAs;

      // Gather the personality functions now, so that they're in deterministic
      // order (derived from the DefinedAtom order).
      if (unwindEntry.personalityFunction) {
        auto pFunc = std::find(personalities.begin(), personalities.end(),
                               unwindEntry.personalityFunction);
        if (pFunc == personalities.end())
          personalities.push_back(unwindEntry.personalityFunction);
      }
    }
  }

  CompactUnwindEntry extractCompactUnwindEntry(const DefinedAtom *atom) {
    CompactUnwindEntry entry = {nullptr, nullptr, nullptr, 0, 0};

    for (const Reference *ref : *atom) {
      switch (ref->offsetInAtom()) {
      case 0:
        // FIXME: there could legitimately be functions with multiple encoding
        // entries. However, nothing produces them at the moment.
        assert(ref->addend() == 0 && "unexpected offset into function");
        entry.rangeStart = ref->target();
        break;
      case 0x10:
        assert(ref->addend() == 0 && "unexpected offset into personality fn");
        entry.personalityFunction = ref->target();
        break;
      case 0x18:
        assert(ref->addend() == 0 && "unexpected offset into LSDA atom");
        entry.lsdaLocation = ref->target();
        break;
      }
    }

    using normalized::read32;
    entry.rangeLength =
        read32(_swap, ((uint32_t *)atom->rawContent().data())[2]);
    entry.encoding = read32(_swap, ((uint32_t *)atom->rawContent().data())[3]);
    return entry;
  }

  /// Every atom defined in __TEXT,__text needs an entry in the final
  /// __unwind_info section (in order). These comes from two sources:
  ///   + Input __compact_unwind sections where possible (after adding the
  ///      personality function offset which is only known now).
  ///   + A synthesised reference to __eh_frame if there's no __compact_unwind
  ///     or too many personality functions to be accommodated.
  std::vector<CompactUnwindEntry> createUnwindInfoEntries(
      const std::unique_ptr<MutableFile> &mergedFile,
      const std::map<const Atom *, CompactUnwindEntry> &unwindLocs,
      const std::vector<const Atom *> &personalities) {
    std::vector<CompactUnwindEntry> unwindInfos;

    DEBUG(llvm::dbgs() << "  Creating __unwind_info entries\n");
    // The final order in the __unwind_info section must be derived from the
    // order of typeCode atoms, since that's how they'll be put into the object
    // file eventually (yuck!).
    for (const DefinedAtom *atom : mergedFile->defined()) {
      if (atom->contentType() != DefinedAtom::typeCode)
        continue;

      unwindInfos.push_back(
          finalizeUnwindInfoEntryForAtom(atom, unwindLocs, personalities));

      DEBUG(llvm::dbgs() << "    Entry for " << atom->name()
                         << ", final encoding="
                         << llvm::format("0x%08x", unwindInfos.back().encoding)
                         << '\n');
    }

    return unwindInfos;
  }

  CompactUnwindEntry finalizeUnwindInfoEntryForAtom(
      const DefinedAtom *function,
      const std::map<const Atom *, CompactUnwindEntry> &unwindLocs,
      const std::vector<const Atom *> &personalities) {
    auto unwindLoc = unwindLocs.find(function);

    // FIXME: we should synthesize a DWARF compact unwind entry before claiming
    // there's no unwind if a __compact_unwind atom doesn't exist.
    if (unwindLoc == unwindLocs.end()) {
      CompactUnwindEntry entry;
      memset(&entry, 0, sizeof(CompactUnwindEntry));
      entry.rangeStart = function;
      entry.rangeLength = function->size();
      return entry;
    }

    CompactUnwindEntry entry = unwindLoc->second;
    auto personality = std::find(personalities.begin(), personalities.end(),
                                 entry.personalityFunction);
    uint32_t personalityIdx = personality == personalities.end()
                                  ? 0
                                  : personality - personalities.begin() + 1;

    // FIXME: We should also use DWARF when there isn't enough room for the
    // personality function in the compact encoding.
    assert(personalityIdx < 4 && "too many personality functions");

    entry.encoding |= personalityIdx << 28;

    if (entry.lsdaLocation)
      entry.encoding |= 1U << 30;

    return entry;
  }

  const MachOLinkingContext &_context;
  mach_o::ArchHandler &_archHandler;
  MachOFile _file;
  bool _swap;
};

void addCompactUnwindPass(PassManager &pm, const MachOLinkingContext &ctx) {
  assert(ctx.needsCompactUnwindPass());
  pm.add(std::unique_ptr<Pass>(new CompactUnwindPass(ctx)));
}

} // end namesapce mach_o
} // end namesapce lld
