//===- lib/ReaderWriter/ELF/TargetLayout.h --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_DEFAULT_LAYOUT_H
#define LLD_READER_WRITER_ELF_DEFAULT_LAYOUT_H

#include "Atoms.h"
#include "HeaderChunks.h"
#include "SectionChunks.h"
#include "SegmentChunks.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <unordered_map>

namespace lld {
namespace elf {

/// \brief The TargetLayout class is used by the Writer to arrange
///        sections and segments in the order determined by the target ELF
///        format. The writer creates a single instance of the TargetLayout
///        class
template <class ELFT> class TargetLayout {
public:
  typedef uint32_t SectionOrder;
  typedef uint32_t SegmentType;

  // The order in which the sections appear in the output file
  // If its determined, that the layout needs to change
  // just changing the order of enumerations would essentially
  // change the layout in the output file
  // Change the enumerations so that Target can override and stick
  // a section anywhere it wants to
  enum DefaultSectionOrder {
    ORDER_NOT_DEFINED = 0,
    ORDER_INTERP = 10,
    ORDER_RO_NOTE = 15,
    ORDER_HASH = 30,
    ORDER_DYNAMIC_SYMBOLS = 40,
    ORDER_DYNAMIC_STRINGS = 50,
    ORDER_DYNAMIC_RELOCS = 52,
    ORDER_DYNAMIC_PLT_RELOCS = 54,
    ORDER_INIT = 60,
    ORDER_PLT = 70,
    ORDER_TEXT = 80,
    ORDER_FINI = 90,
    ORDER_REL = 95,
    ORDER_RODATA = 100,
    ORDER_EH_FRAME = 110,
    ORDER_EH_FRAMEHDR = 120,
    ORDER_TDATA = 124,
    ORDER_TBSS = 128,
    ORDER_CTORS = 130,
    ORDER_DTORS = 140,
    ORDER_INIT_ARRAY = 150,
    ORDER_FINI_ARRAY = 160,
    ORDER_DYNAMIC = 170,
    ORDER_GOT = 180,
    ORDER_GOT_PLT = 190,
    ORDER_DATA = 200,
    ORDER_RW_NOTE = 205,
    ORDER_BSS = 210,
    ORDER_NOALLOC = 215,
    ORDER_OTHER = 220,
    ORDER_SECTION_STRINGS = 230,
    ORDER_SYMBOL_TABLE = 240,
    ORDER_STRING_TABLE = 250,
    ORDER_SECTION_HEADERS = 260
  };

public:

  // The Key used for creating Sections
  // The sections are created using
  // SectionName, contentPermissions
  struct SectionKey {
    SectionKey(StringRef name, DefinedAtom::ContentPermissions perm,
               StringRef path)
        : _name(name), _perm(perm), _path(path) {}

    // Data members
    StringRef _name;
    DefinedAtom::ContentPermissions _perm;
    StringRef _path;
  };

  struct SectionKeyHash {
    int64_t operator()(const SectionKey &k) const {
      return llvm::hash_combine(k._name, k._perm, k._path);
    }
  };

  struct SectionKeyEq {
    bool operator()(const SectionKey &lhs, const SectionKey &rhs) const {
      return ((lhs._name == rhs._name) && (lhs._perm == rhs._perm) &&
              (lhs._path == rhs._path));
    }
  };

  typedef typename std::vector<Chunk<ELFT> *>::iterator ChunkIter;
  typedef typename std::vector<Segment<ELFT> *>::iterator SegmentIter;

  // The additional segments are used to figure out
  // if there is a segment by that type already created
  // For example : PT_TLS, we have two sections .tdata/.tbss
  // that are part of PT_TLS, we need to create this additional
  // segment only once
  typedef std::pair<int64_t, int64_t> AdditionalSegmentKey;
  // The segments are created using
  // SegmentName, Segment flags
  typedef std::pair<StringRef, int64_t> SegmentKey;

  // HashKey for the Segment
  class SegmentHashKey {
  public:
    int64_t operator() (const SegmentKey &k) const {
      // k.first = SegmentName
      // k.second = SegmentFlags
      return llvm::hash_combine(k.first, k.second);
    }
  };

  class AdditionalSegmentHashKey {
  public:
    int64_t operator()(const AdditionalSegmentKey &k) const {
      // k.first = SegmentName
      // k.second = SegmentFlags
      return llvm::hash_combine(k.first, k.second);
    }
  };

  // Output Sections contain the map of Sectionnames to a vector of sections,
  // that have been merged to form a single section
  typedef llvm::StringMap<OutputSection<ELFT> *> OutputSectionMapT;
  typedef
      typename std::vector<OutputSection<ELFT> *>::iterator OutputSectionIter;

  typedef std::unordered_map<SectionKey, AtomSection<ELFT> *, SectionKeyHash,
                             SectionKeyEq> SectionMapT;
  typedef std::unordered_map<AdditionalSegmentKey, Segment<ELFT> *,
                             AdditionalSegmentHashKey> AdditionalSegmentMapT;
  typedef std::unordered_map<SegmentKey, Segment<ELFT> *, SegmentHashKey>
  SegmentMapT;

  typedef typename std::vector<AtomLayout *>::iterator AbsoluteAtomIterT;

  typedef llvm::DenseSet<const Atom *> AtomSetT;

  TargetLayout(ELFLinkingContext &ctx)
      : _ctx(ctx), _linkerScriptSema(ctx.linkerScriptSema()) {}

  virtual ~TargetLayout() = default;

  /// \brief Return the section order for a input section
  virtual SectionOrder getSectionOrder(StringRef name, int32_t contentType,
                                       int32_t contentPermissions);

  /// \brief Return the name of the input section by decoding the input
  /// sectionChoice.
  virtual StringRef getInputSectionName(const DefinedAtom *da) const;

  /// \brief Return the name of the output section from the input section.
  virtual StringRef getOutputSectionName(StringRef archivePath,
                                         StringRef memberPath,
                                         StringRef inputSectionName) const;

  /// \brief Gets or creates a section.
  AtomSection<ELFT> *
  getSection(StringRef name, int32_t contentType,
             DefinedAtom::ContentPermissions contentPermissions,
             const DefinedAtom *da);

  /// \brief Gets the segment for a output section
  virtual SegmentType getSegmentType(Section<ELFT> *section) const;

  /// \brief Returns true/false depending on whether the section has a Output
  //         segment or not
  static bool hasOutputSegment(Section<ELFT> *section);

  /// \brief Append the Atom to the layout and create appropriate sections.
  /// \returns A reference to the atom layout or an error. The atom layout will
  /// be updated as linking progresses.
  virtual ErrorOr<const AtomLayout *> addAtom(const Atom *atom);

  /// \brief Find an output Section given a section name.
  OutputSection<ELFT> *findOutputSection(StringRef name) {
    auto iter = _outputSectionMap.find(name);
    if (iter == _outputSectionMap.end())
      return nullptr;
    return iter->second;
  }

  /// \brief find a absolute atom given a name
  AtomLayout *findAbsoluteAtom(StringRef name) {
    auto iter = std::find_if(
        _absoluteAtoms.begin(), _absoluteAtoms.end(),
        [=](const AtomLayout *a) { return a->_atom->name() == name; });
    if (iter == _absoluteAtoms.end())
      return nullptr;
    return *iter;
  }

  // Output sections with the same name into a OutputSection
  void createOutputSections();

  /// \brief Sort the sections by their order as defined by the layout,
  /// preparing all sections to be assigned to a segment.
  virtual void sortInputSections();

  /// \brief Add extra chunks to a segment just before including the input
  /// section given by <archivePath, memberPath, sectionName>. This
  /// is used to add linker script expressions before each section.
  virtual void addExtraChunksToSegment(Segment<ELFT> *segment,
                                       StringRef archivePath,
                                       StringRef memberPath,
                                       StringRef sectionName);

  /// \brief associates a section to a segment
  virtual void assignSectionsToSegments();

  /// \brief associates a virtual address to the segment, section, and the atom
  virtual void assignVirtualAddress();

  void assignFileOffsetsForMiscSections();

  range<AbsoluteAtomIterT> absoluteAtoms() { return _absoluteAtoms; }

  void addSection(Chunk<ELFT> *c) { _sections.push_back(c); }

  void finalize() {
    ScopedTask task(getDefaultDomain(), "Finalize layout");
    for (auto &si : _sections)
      si->finalize();
  }

  void doPreFlight() {
    for (auto &si : _sections)
      si->doPreFlight();
  }

  /// \brief find the Atom in the current layout
  virtual const AtomLayout *findAtomLayoutByName(StringRef name) const;

  void setHeader(ELFHeader<ELFT> *elfHeader) { _elfHeader = elfHeader; }

  void setProgramHeader(ProgramHeader<ELFT> *p) {
    _programHeader = p;
  }

  range<OutputSectionIter> outputSections() { return _outputSections; }

  range<ChunkIter> sections() { return _sections; }

  range<SegmentIter> segments() { return _segments; }

  ELFHeader<ELFT> *getHeader() { return _elfHeader; }

  bool hasDynamicRelocationTable() const { return !!_dynamicRelocationTable; }

  bool hasPLTRelocationTable() const { return !!_pltRelocationTable; }

  /// \brief Get or create the dynamic relocation table. All relocations in this
  /// table are processed at startup.
  RelocationTable<ELFT> *getDynamicRelocationTable();

  /// \brief Get or create the PLT relocation table. Referenced by DT_JMPREL.
  RelocationTable<ELFT> *getPLTRelocationTable();

  uint64_t getTLSSize() const;

  bool isReferencedByDefinedAtom(const Atom *a) const {
    return _referencedDynAtoms.count(a);
  }

  bool isCopied(const SharedLibraryAtom *sla) const {
    return _copiedDynSymNames.count(sla->name());
  }

protected:
  /// \brief TargetLayouts may use these functions to reorder the input sections
  /// in a order defined by their ABI.
  virtual void finalizeOutputSectionLayout() {}

  /// \brief Allocate a new section.
  virtual AtomSection<ELFT> *createSection(
      StringRef name, int32_t contentType,
      DefinedAtom::ContentPermissions contentPermissions,
      SectionOrder sectionOrder);

  /// \brief Create a new relocation table.
  virtual unique_bump_ptr<RelocationTable<ELFT>>
  createRelocationTable(StringRef name, int32_t order) {
    return unique_bump_ptr<RelocationTable<ELFT>>(
        new (_allocator) RelocationTable<ELFT>(_ctx, name, order));
  }

  virtual uint64_t getLookupSectionFlags(const OutputSection<ELFT> *os) const;

protected:
  llvm::BumpPtrAllocator _allocator;
  SectionMapT _sectionMap;
  OutputSectionMapT _outputSectionMap;
  AdditionalSegmentMapT _additionalSegmentMap;
  SegmentMapT _segmentMap;
  std::vector<Chunk<ELFT> *> _sections;
  std::vector<Segment<ELFT> *> _segments;
  std::vector<OutputSection<ELFT> *> _outputSections;
  ELFHeader<ELFT> *_elfHeader;
  ProgramHeader<ELFT> *_programHeader;
  unique_bump_ptr<RelocationTable<ELFT>> _dynamicRelocationTable;
  unique_bump_ptr<RelocationTable<ELFT>> _pltRelocationTable;
  std::vector<AtomLayout *> _absoluteAtoms;
  AtomSetT _referencedDynAtoms;
  llvm::StringSet<> _copiedDynSymNames;
  ELFLinkingContext &_ctx;
  script::Sema &_linkerScriptSema;
};

} // end namespace elf
} // end namespace lld

#endif
