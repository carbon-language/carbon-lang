//===- lib/ReaderWriter/ELF/DefaultLayout.h -------------------------------===//
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
#include "Chunk.h"
#include "HeaderChunks.h"
#include "Layout.h"
#include "SectionChunks.h"
#include "SegmentChunks.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/STDExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Format.h"
#include <map>
#include <unordered_map>

namespace lld {
namespace elf {
/// \brief The DefaultLayout class is used by the Writer to arrange
///        sections and segments in the order determined by the target ELF
///        format. The writer creates a single instance of the DefaultLayout
///        class
template<class ELFT>
class DefaultLayout : public Layout {
public:

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

  /// \brief find a absolute atom pair given a absolute atom name
  struct FindByName {
    const std::string _name;
    FindByName(StringRef name) : _name(name) {}
    bool operator()(const lld::AtomLayout *j) { return j->_atom->name() == _name; }
  };

  typedef typename std::vector<lld::AtomLayout *>::iterator AbsoluteAtomIterT;

  typedef llvm::DenseSet<const Atom *> AtomSetT;

  DefaultLayout(ELFLinkingContext &context)
      : _context(context), _linkerScriptSema(context.linkerScriptSema()) {}

  /// \brief Return the section order for a input section
  SectionOrder getSectionOrder(StringRef name, int32_t contentType,
                               int32_t contentPermissions) override;

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
  virtual Layout::SegmentType getSegmentType(Section<ELFT> *section) const;

  /// \brief Returns true/false depending on whether the section has a Output
  //         segment or not
  static bool hasOutputSegment(Section<ELFT> *section);

  // Adds an atom to the section
  ErrorOr<const lld::AtomLayout *> addAtom(const Atom *atom) override;

  /// \brief Find an output Section given a section name.
  OutputSection<ELFT> *findOutputSection(StringRef name) {
    auto iter = _outputSectionMap.find(name);
    if (iter == _outputSectionMap.end())
      return nullptr;
    return iter->second;
  }

  /// \brief find a absolute atom given a name
  AbsoluteAtomIterT findAbsoluteAtom(StringRef name) {
    return std::find_if(_absoluteAtoms.begin(), _absoluteAtoms.end(),
                        FindByName(name));
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

  void assignSectionsToSegments() override;

  void assignVirtualAddress() override;

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

  const AtomLayout *findAtomLayoutByName(StringRef name) const override {
    for (auto sec : _sections)
      if (auto section = dyn_cast<Section<ELFT>>(sec))
        if (auto *al = section->findAtomLayoutByName(name))
          return al;
    return nullptr;
  }

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
  RelocationTable<ELFT> *getDynamicRelocationTable() {
    if (!_dynamicRelocationTable) {
      _dynamicRelocationTable = std::move(createRelocationTable(
          _context.isRelaOutputFormat() ? ".rela.dyn" : ".rel.dyn",
          ORDER_DYNAMIC_RELOCS));
      addSection(_dynamicRelocationTable.get());
    }
    return _dynamicRelocationTable.get();
  }

  /// \brief Get or create the PLT relocation table. Referenced by DT_JMPREL.
  RelocationTable<ELFT> *getPLTRelocationTable() {
    if (!_pltRelocationTable) {
      _pltRelocationTable = std::move(createRelocationTable(
          _context.isRelaOutputFormat() ? ".rela.plt" : ".rel.plt",
          ORDER_DYNAMIC_PLT_RELOCS));
      addSection(_pltRelocationTable.get());
    }
    return _pltRelocationTable.get();
  }

  uint64_t getTLSSize() const {
    for (const auto &phdr : *_programHeader)
      if (phdr->p_type == llvm::ELF::PT_TLS)
        return phdr->p_memsz;
    return 0;
  }

  bool isReferencedByDefinedAtom(const Atom *a) const {
    return _referencedDynAtoms.count(a);
  }

  bool isCopied(const SharedLibraryAtom *sla) const {
    return _copiedDynSymNames.count(sla->name());
  }

  /// \brief Handle SORT_BY_PRIORITY.
  void sortOutputSectionByPriority(StringRef outputSectionName,
                                   StringRef prefix);

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
        new (_allocator) RelocationTable<ELFT>(_context, name, order));
  }

private:
  /// Helper function that returns the priority value from an input section.
  uint32_t getPriorityFromSectionName(StringRef sectionName) const;

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
  std::vector<lld::AtomLayout *> _absoluteAtoms;
  AtomSetT _referencedDynAtoms;
  llvm::StringSet<> _copiedDynSymNames;
  ELFLinkingContext &_context;
  script::Sema &_linkerScriptSema;
};

template <class ELFT>
Layout::SectionOrder DefaultLayout<ELFT>::getSectionOrder(
    StringRef name, int32_t contentType, int32_t contentPermissions) {
  switch (contentType) {
  case DefinedAtom::typeResolver:
  case DefinedAtom::typeCode:
    return llvm::StringSwitch<Layout::SectionOrder>(name)
        .StartsWith(".eh_frame_hdr", ORDER_EH_FRAMEHDR)
        .StartsWith(".eh_frame", ORDER_EH_FRAME)
        .StartsWith(".init", ORDER_INIT)
        .StartsWith(".fini", ORDER_FINI)
        .StartsWith(".hash", ORDER_HASH)
        .Default(ORDER_TEXT);

  case DefinedAtom::typeConstant:
    return ORDER_RODATA;

  case DefinedAtom::typeData:
  case DefinedAtom::typeDataFast:
    return llvm::StringSwitch<Layout::SectionOrder>(name)
        .StartsWith(".init_array", ORDER_INIT_ARRAY)
        .StartsWith(".fini_array", ORDER_FINI_ARRAY)
        .StartsWith(".dynamic", ORDER_DYNAMIC)
        .StartsWith(".ctors", ORDER_CTORS)
        .StartsWith(".dtors", ORDER_DTORS)
        .Default(ORDER_DATA);

  case DefinedAtom::typeZeroFill:
  case DefinedAtom::typeZeroFillFast:
    return ORDER_BSS;

  case DefinedAtom::typeGOT:
    return llvm::StringSwitch<Layout::SectionOrder>(name)
        .StartsWith(".got.plt", ORDER_GOT_PLT)
        .Default(ORDER_GOT);

  case DefinedAtom::typeStub:
    return ORDER_PLT;

  case DefinedAtom::typeRONote:
      return ORDER_RO_NOTE;

  case DefinedAtom::typeRWNote:
      return ORDER_RW_NOTE;

  case DefinedAtom::typeNoAlloc:
    return ORDER_NOALLOC;

  case DefinedAtom::typeThreadData:
    return ORDER_TDATA;
  case DefinedAtom::typeThreadZeroFill:
    return ORDER_TBSS;
  default:
    // If we get passed in a section push it to OTHER
    if (contentPermissions == DefinedAtom::perm___)
      return ORDER_OTHER;

    return ORDER_NOT_DEFINED;
  }
}

/// \brief This maps the input sections to the output section names
template <class ELFT>
StringRef
DefaultLayout<ELFT>::getInputSectionName(const DefinedAtom *da) const {
  if (da->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
    switch (da->contentType()) {
    case DefinedAtom::typeCode:
      return ".text";
    case DefinedAtom::typeData:
      return ".data";
    case DefinedAtom::typeConstant:
      return ".rodata";
    case DefinedAtom::typeZeroFill:
      return ".bss";
    case DefinedAtom::typeThreadData:
      return ".tdata";
    case DefinedAtom::typeThreadZeroFill:
      return ".tbss";
    default:
      break;
    }
  }
  return da->customSectionName();
}

/// \brief This maps the input sections to the output section names.
template <class ELFT>
StringRef
DefaultLayout<ELFT>::getOutputSectionName(StringRef archivePath,
                                          StringRef memberPath,
                                          StringRef inputSectionName) const {
  StringRef outputSectionName;
  if (_linkerScriptSema.hasLayoutCommands()) {
    script::Sema::SectionKey key = {archivePath, memberPath, inputSectionName};
    outputSectionName = _linkerScriptSema.getOutputSection(key);
    if (!outputSectionName.empty())
      return outputSectionName;
  }
  return llvm::StringSwitch<StringRef>(inputSectionName)
      .StartsWith(".text", ".text")
      .StartsWith(".ctors", ".ctors")
      .StartsWith(".dtors", ".dtors")
      .StartsWith(".rodata", ".rodata")
      .StartsWith(".gcc_except_table", ".gcc_except_table")
      .StartsWith(".data.rel.ro", ".data.rel.ro")
      .StartsWith(".data.rel.local", ".data.rel.local")
      .StartsWith(".data", ".data")
      .StartsWith(".tdata", ".tdata")
      .StartsWith(".tbss", ".tbss")
      .StartsWith(".init_array", ".init_array")
      .StartsWith(".fini_array", ".fini_array")
      .Default(inputSectionName);
}

/// \brief Gets the segment for a output section
template <class ELFT>
Layout::SegmentType DefaultLayout<ELFT>::getSegmentType(
    Section<ELFT> *section) const {

  switch (section->order()) {
  case ORDER_INTERP:
    return llvm::ELF::PT_INTERP;

  case ORDER_TEXT:
  case ORDER_HASH:
  case ORDER_DYNAMIC_SYMBOLS:
  case ORDER_DYNAMIC_STRINGS:
  case ORDER_DYNAMIC_RELOCS:
  case ORDER_DYNAMIC_PLT_RELOCS:
  case ORDER_REL:
  case ORDER_INIT:
  case ORDER_PLT:
  case ORDER_FINI:
  case ORDER_RODATA:
  case ORDER_EH_FRAME:
  case ORDER_CTORS:
  case ORDER_DTORS:
    return llvm::ELF::PT_LOAD;

  case ORDER_RO_NOTE:
  case ORDER_RW_NOTE:
    return llvm::ELF::PT_NOTE;

  case ORDER_DYNAMIC:
    return llvm::ELF::PT_DYNAMIC;

  case ORDER_EH_FRAMEHDR:
    return llvm::ELF::PT_GNU_EH_FRAME;

  case ORDER_GOT:
  case ORDER_GOT_PLT:
  case ORDER_DATA:
  case ORDER_BSS:
  case ORDER_INIT_ARRAY:
  case ORDER_FINI_ARRAY:
    return llvm::ELF::PT_LOAD;

  case ORDER_TDATA:
  case ORDER_TBSS:
    return llvm::ELF::PT_TLS;

  default:
    return llvm::ELF::PT_NULL;
  }
}

template <class ELFT>
bool DefaultLayout<ELFT>::hasOutputSegment(Section<ELFT> *section) {
  switch (section->order()) {
  case ORDER_INTERP:
  case ORDER_HASH:
  case ORDER_DYNAMIC_SYMBOLS:
  case ORDER_DYNAMIC_STRINGS:
  case ORDER_DYNAMIC_RELOCS:
  case ORDER_DYNAMIC_PLT_RELOCS:
  case ORDER_REL:
  case ORDER_INIT:
  case ORDER_PLT:
  case ORDER_TEXT:
  case ORDER_FINI:
  case ORDER_RODATA:
  case ORDER_EH_FRAME:
  case ORDER_EH_FRAMEHDR:
  case ORDER_TDATA:
  case ORDER_TBSS:
  case ORDER_RO_NOTE:
  case ORDER_RW_NOTE:
  case ORDER_DYNAMIC:
  case ORDER_CTORS:
  case ORDER_DTORS:
  case ORDER_GOT:
  case ORDER_GOT_PLT:
  case ORDER_DATA:
  case ORDER_INIT_ARRAY:
  case ORDER_FINI_ARRAY:
  case ORDER_BSS:
  case ORDER_NOALLOC:
    return true;
  default:
    return section->hasOutputSegment();
  }
}

template <class ELFT>
AtomSection<ELFT> *DefaultLayout<ELFT>::createSection(
    StringRef sectionName, int32_t contentType,
    DefinedAtom::ContentPermissions permissions, SectionOrder sectionOrder) {
  return new (_allocator) AtomSection<ELFT>(_context, sectionName, contentType,
                                            permissions, sectionOrder);
}

template <class ELFT>
AtomSection<ELFT> *
DefaultLayout<ELFT>::getSection(StringRef sectionName, int32_t contentType,
                                DefinedAtom::ContentPermissions permissions,
                                const DefinedAtom *da) {
  const SectionKey sectionKey(sectionName, permissions, da->file().path());
  SectionOrder sectionOrder = getSectionOrder(sectionName, contentType, permissions);
  auto sec = _sectionMap.find(sectionKey);
  if (sec != _sectionMap.end())
    return sec->second;
  AtomSection<ELFT> *newSec =
      createSection(sectionName, contentType, permissions, sectionOrder);

  newSec->setOutputSectionName(getOutputSectionName(
      da->file().archivePath(), da->file().memberPath(), sectionName));
  newSec->setOrder(sectionOrder);
  newSec->setArchiveNameOrPath(da->file().archivePath());
  newSec->setMemberNameOrPath(da->file().memberPath());
  _sections.push_back(newSec);
  _sectionMap.insert(std::make_pair(sectionKey, newSec));
  return newSec;
}

template <class ELFT>
ErrorOr<const lld::AtomLayout *>
DefaultLayout<ELFT>::addAtom(const Atom *atom) {
  if (const DefinedAtom *definedAtom = dyn_cast<DefinedAtom>(atom)) {
    // HACK: Ignore undefined atoms. We need to adjust the interface so that
    // undefined atoms can still be included in the output symbol table for
    // -noinhibit-exec.
    if (definedAtom->contentType() == DefinedAtom::typeUnknown)
      return make_error_code(llvm::errc::invalid_argument);
    const DefinedAtom::ContentPermissions permissions =
        definedAtom->permissions();
    const DefinedAtom::ContentType contentType = definedAtom->contentType();

    StringRef sectionName = getInputSectionName(definedAtom);
    AtomSection<ELFT> *section =
        getSection(sectionName, contentType, permissions, definedAtom);

    // Add runtime relocations to the .rela section.
    for (const auto &reloc : *definedAtom) {
      bool isLocalReloc = true;
      if (_context.isDynamicRelocation(*reloc)) {
        getDynamicRelocationTable()->addRelocation(*definedAtom, *reloc);
        isLocalReloc = false;
      } else if (_context.isPLTRelocation(*reloc)) {
        getPLTRelocationTable()->addRelocation(*definedAtom, *reloc);
        isLocalReloc = false;
      }

      if (!reloc->target())
        continue;

      //Ignore undefined atoms that are not target of dynamic relocations
      if (isa<UndefinedAtom>(reloc->target()) && isLocalReloc)
        continue;

      if (_context.isCopyRelocation(*reloc)) {
        _copiedDynSymNames.insert(definedAtom->name());
        continue;
      }

      _referencedDynAtoms.insert(reloc->target());
    }

    return section->appendAtom(atom);
  } else if (const AbsoluteAtom *absoluteAtom = dyn_cast<AbsoluteAtom>(atom)) {
    // Absolute atoms are not part of any section, they are global for the whole
    // link
    _absoluteAtoms.push_back(new (_allocator)
        lld::AtomLayout(absoluteAtom, 0, absoluteAtom->value()));
    return _absoluteAtoms.back();
  } else {
    llvm_unreachable("Only absolute / defined atoms can be added here");
  }
}

/// Output sections with the same name into a OutputSection
template <class ELFT> void DefaultLayout<ELFT>::createOutputSections() {
  OutputSection<ELFT> *outputSection;

  for (auto &si : _sections) {
    Section<ELFT> *section = dyn_cast<Section<ELFT>>(si);
    if (!section)
      continue;
    const std::pair<StringRef, OutputSection<ELFT> *> currentOutputSection(
        section->outputSectionName(), nullptr);
    std::pair<typename OutputSectionMapT::iterator, bool> outputSectionInsert(
        _outputSectionMap.insert(currentOutputSection));
    if (!outputSectionInsert.second) {
      outputSection = outputSectionInsert.first->second;
    } else {
      outputSection = new (_allocator.Allocate<OutputSection<ELFT>>())
          OutputSection<ELFT>(section->outputSectionName());
      _outputSections.push_back(outputSection);
      outputSectionInsert.first->second = outputSection;
    }
    outputSection->appendSection(si);
  }
}

template <class ELFT>
uint32_t
DefaultLayout<ELFT>::getPriorityFromSectionName(StringRef sectionName) const {
  StringRef priority = sectionName.drop_front().rsplit('.').second;
  uint32_t prio;
  if (priority.getAsInteger(10, prio))
    return std::numeric_limits<uint32_t>::max();
  return prio;
}

template <class ELFT>
void DefaultLayout<ELFT>::sortOutputSectionByPriority(
    StringRef outputSectionName, StringRef prefix) {
  OutputSection<ELFT> *outputSection = findOutputSection(outputSectionName);
  if (!outputSection)
    return;

  auto sections = outputSection->sections();

  std::sort(sections.begin(), sections.end(),
            [&](Chunk<ELFT> *lhs, Chunk<ELFT> *rhs) {
              Section<ELFT> *lhsSection = dyn_cast<Section<ELFT>>(lhs);
              Section<ELFT> *rhsSection = dyn_cast<Section<ELFT>>(rhs);
              if (!lhsSection || !rhsSection)
                return false;
              StringRef lhsSectionName = lhsSection->inputSectionName();
              StringRef rhsSectionName = rhsSection->inputSectionName();

              if (!prefix.empty()) {
                if (!lhsSectionName.startswith(prefix) ||
                    !rhsSectionName.startswith(prefix))
                  return false;
              }
              return getPriorityFromSectionName(lhsSectionName) <
                     getPriorityFromSectionName(rhsSectionName);
            });
}

template <class ELFT> void DefaultLayout<ELFT>::assignSectionsToSegments() {
  ScopedTask task(getDefaultDomain(), "assignSectionsToSegments");
  ELFLinkingContext::OutputMagic outputMagic = _context.getOutputMagic();
  // sort the sections by their order as defined by the layout
  sortInputSections();

  // Create output sections.
  createOutputSections();

  // Finalize output section layout.
  finalizeOutputSectionLayout();

  // Set the ordinal after sorting the sections
  int ordinal = 1;
  for (auto osi : _outputSections) {
    osi->setOrdinal(ordinal);
    for (auto ai : osi->sections()) {
      ai->setOrdinal(ordinal);
    }
    ++ordinal;
  }
  for (auto osi : _outputSections) {
    for (auto ai : osi->sections()) {
      if (auto section = dyn_cast<Section<ELFT> >(ai)) {
        if (!hasOutputSegment(section))
          continue;

        osi->setLoadableSection(section->isLoadableSection());

        // Get the segment type for the section
        int64_t segmentType = getSegmentType(section);

        osi->setHasSegment();
        section->setSegmentType(segmentType);
        StringRef segmentName = section->segmentKindToStr();

        int64_t lookupSectionFlag = osi->flags();
        if ((!(lookupSectionFlag & llvm::ELF::SHF_WRITE)) &&
            (_context.mergeRODataToTextSegment()))
          lookupSectionFlag &= ~llvm::ELF::SHF_EXECINSTR;

        // Merge string sections into Data segment itself
        lookupSectionFlag &= ~(llvm::ELF::SHF_STRINGS | llvm::ELF::SHF_MERGE);

        // Merge the TLS section into the DATA segment itself
        lookupSectionFlag &= ~(llvm::ELF::SHF_TLS);

        Segment<ELFT> *segment;
        // We need a separate segment for sections that don't have
        // the segment type to be PT_LOAD
        if (segmentType != llvm::ELF::PT_LOAD) {
          const AdditionalSegmentKey key(segmentType, lookupSectionFlag);
          const std::pair<AdditionalSegmentKey, Segment<ELFT> *>
          additionalSegment(key, nullptr);
          std::pair<typename AdditionalSegmentMapT::iterator, bool>
          additionalSegmentInsert(
              _additionalSegmentMap.insert(additionalSegment));
          if (!additionalSegmentInsert.second) {
            segment = additionalSegmentInsert.first->second;
          } else {
            segment = new (_allocator)
                Segment<ELFT>(_context, segmentName, segmentType);
            additionalSegmentInsert.first->second = segment;
            _segments.push_back(segment);
          }
          segment->append(section);
        }
        if (segmentType == llvm::ELF::PT_NULL)
          continue;

        // If the output magic is set to OutputMagic::NMAGIC or
        // OutputMagic::OMAGIC, Place the data alongside text in one single
        // segment
        if (outputMagic == ELFLinkingContext::OutputMagic::NMAGIC ||
            outputMagic == ELFLinkingContext::OutputMagic::OMAGIC)
          lookupSectionFlag = llvm::ELF::SHF_EXECINSTR | llvm::ELF::SHF_ALLOC |
                              llvm::ELF::SHF_WRITE;

        // Use the flags of the merged Section for the segment
        const SegmentKey key("PT_LOAD", lookupSectionFlag);
        const std::pair<SegmentKey, Segment<ELFT> *> currentSegment(key,
                                                                    nullptr);
        std::pair<typename SegmentMapT::iterator, bool> segmentInsert(
            _segmentMap.insert(currentSegment));
        if (!segmentInsert.second) {
          segment = segmentInsert.first->second;
        } else {
          segment = new (_allocator)
              Segment<ELFT>(_context, "PT_LOAD", llvm::ELF::PT_LOAD);
          segmentInsert.first->second = segment;
          _segments.push_back(segment);
        }
        // Insert chunks with linker script expressions that occur at this
        // point, just before appending a new input section
        addExtraChunksToSegment(segment, section->archivePath(),
                                section->memberPath(),
                                section->inputSectionName());
        segment->append(section);
      }
    }
  }
  if (_context.isDynamic() && !_context.isDynamicLibrary()) {
    Segment<ELFT> *segment =
        new (_allocator) ProgramHeaderSegment<ELFT>(_context);
    _segments.push_back(segment);
    segment->append(_elfHeader);
    segment->append(_programHeader);
  }
}

template<class ELFT>
void
DefaultLayout<ELFT>::assignVirtualAddress() {
  if (_segments.empty())
    return;

  std::sort(_segments.begin(), _segments.end(), Segment<ELFT>::compareSegments);

  uint64_t baseAddress = _context.getBaseAddress();

  // HACK: This is a super dirty hack. The elf header and program header are
  // not part of a section, but we need them to be loaded at the base address
  // so that AT_PHDR is set correctly by the loader and so they are accessible
  // at runtime. To do this we simply prepend them to the first loadable Segment
  // and let the layout logic take care of it.
  Segment<ELFT> *firstLoadSegment = nullptr;
  for (auto si : _segments) {
    if (si->segmentType() == llvm::ELF::PT_LOAD) {
      firstLoadSegment = si;
      si->firstSection()->setAlign(si->alignment());
      break;
    }
  }
  assert(firstLoadSegment != nullptr && "No loadable segment!");
  firstLoadSegment->prepend(_programHeader);
  firstLoadSegment->prepend(_elfHeader);
  bool newSegmentHeaderAdded = true;
  bool virtualAddressAssigned = false;
  bool fileOffsetAssigned = false;
  while (true) {
    for (auto si : _segments) {
      si->finalize();
      // Don't add PT_NULL segments into the program header
      if (si->segmentType() != llvm::ELF::PT_NULL)
        newSegmentHeaderAdded = _programHeader->addSegment(si);
    }
    if (!newSegmentHeaderAdded && virtualAddressAssigned)
      break;
    uint64_t address = baseAddress;
    // start assigning virtual addresses
    for (auto &si : _segments) {
      if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
          (si->segmentType() != llvm::ELF::PT_NULL))
        continue;

      if (si->segmentType() == llvm::ELF::PT_NULL) {
        si->assignVirtualAddress(0 /*non loadable*/);
      } else {
        if (virtualAddressAssigned && (address != baseAddress) &&
            (address == si->virtualAddr()))
          break;
        si->assignVirtualAddress(address);
      }
      address = si->virtualAddr() + si->memSize();
    }
    uint64_t baseFileOffset = 0;
    uint64_t fileoffset = baseFileOffset;
    for (auto &si : _segments) {
      if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
          (si->segmentType() != llvm::ELF::PT_NULL))
        continue;
      if (fileOffsetAssigned && (fileoffset != baseFileOffset) &&
          (fileoffset == si->fileOffset()))
        break;
      si->assignFileOffsets(fileoffset);
      fileoffset = si->fileOffset() + si->fileSize();
    }
    virtualAddressAssigned = true;
    fileOffsetAssigned = true;
    _programHeader->resetProgramHeaders();
  }
  Section<ELFT> *section;
  // Fix the offsets of all the atoms within a section
  for (auto &si : _sections) {
    section = dyn_cast<Section<ELFT>>(si);
    if (section && DefaultLayout<ELFT>::hasOutputSegment(section))
      section->assignFileOffsets(section->fileOffset());
  }
  // Set the size of the merged Sections
  for (auto osi : _outputSections) {
    uint64_t sectionfileoffset = 0;
    uint64_t startFileOffset = 0;
    uint64_t sectionsize = 0;
    bool isFirstSection = true;
    for (auto si : osi->sections()) {
      if (isFirstSection) {
        startFileOffset = si->fileOffset();
        isFirstSection = false;
      }
      sectionfileoffset = si->fileOffset();
      sectionsize = si->fileSize();
    }
    sectionsize = (sectionfileoffset - startFileOffset) + sectionsize;
    osi->setFileOffset(startFileOffset);
    osi->setSize(sectionsize);
  }
  // Set the virtual addr of the merged Sections
  for (auto osi : _outputSections) {
    uint64_t sectionstartaddr = 0;
    uint64_t startaddr = 0;
    uint64_t sectionsize = 0;
    bool isFirstSection = true;
    for (auto si : osi->sections()) {
      if (isFirstSection) {
        startaddr = si->virtualAddr();
        isFirstSection = false;
      }
      sectionstartaddr = si->virtualAddr();
      sectionsize = si->memSize();
    }
    sectionsize = (sectionstartaddr - startaddr) + sectionsize;
    osi->setMemSize(sectionsize);
    osi->setAddr(startaddr);
  }
}

template <class ELFT>
void DefaultLayout<ELFT>::assignFileOffsetsForMiscSections() {
  uint64_t fileoffset = 0;
  uint64_t size = 0;
  for (auto si : _segments) {
    // Don't calculate offsets from non loadable segments
    if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
        (si->segmentType() != llvm::ELF::PT_NULL))
      continue;
    fileoffset = si->fileOffset();
    size = si->fileSize();
  }
  fileoffset = fileoffset + size;
  Section<ELFT> *section;
  for (auto si : _sections) {
    section = dyn_cast<Section<ELFT>>(si);
    if (section && DefaultLayout<ELFT>::hasOutputSegment(section))
      continue;
    fileoffset = llvm::RoundUpToAlignment(fileoffset, si->alignment());
    si->setFileOffset(fileoffset);
    si->setVirtualAddr(0);
    fileoffset += si->fileSize();
  }
}

template <class ELFT> void DefaultLayout<ELFT>::sortInputSections() {
  // First, sort according to default layout's order
  std::stable_sort(
      _sections.begin(), _sections.end(),
      [](Chunk<ELFT> *A, Chunk<ELFT> *B) { return A->order() < B->order(); });

  if (!_linkerScriptSema.hasLayoutCommands())
    return;

  // Sort the sections by their order as defined by the linker script
  std::stable_sort(this->_sections.begin(), this->_sections.end(),
                   [this](Chunk<ELFT> *A, Chunk<ELFT> *B) {
                     auto *a = dyn_cast<Section<ELFT>>(A);
                     auto *b = dyn_cast<Section<ELFT>>(B);

                     if (a == nullptr)
                       return false;
                     if (b == nullptr)
                       return true;

                     return _linkerScriptSema.less(
                         {a->archivePath(), a->memberPath(),
                          a->inputSectionName()},
                         {b->archivePath(), b->memberPath(),
                          b->inputSectionName()});
                   });
  // Now try to arrange sections with no mapping rules to sections with
  // similar content
  auto p = this->_sections.begin();
  // Find first section that has no assigned rule id
  while (p != this->_sections.end()) {
    auto *sect = dyn_cast<AtomSection<ELFT>>(*p);
    if (!sect)
      break;

    if (!_linkerScriptSema.hasMapping({sect->archivePath(),
                                       sect->memberPath(),
                                       sect->inputSectionName()}))
      break;

    ++p;
  }
  // For all sections that have no assigned rule id, try to move them near a
  // section with similar contents
  if (p != this->_sections.begin()) {
    for (; p != this->_sections.end(); ++p) {
      auto q = p;
      --q;
      while (q != this->_sections.begin() &&
             (*q)->getContentType() != (*p)->getContentType())
        --q;
      if ((*q)->getContentType() != (*p)->getContentType())
        continue;
      ++q;
      for (auto i = p; i != q;) {
        auto next = i--;
        std::iter_swap(i, next);
      }
    }
  }
}

template <class ELFT>
void DefaultLayout<ELFT>::addExtraChunksToSegment(Segment<ELFT> *segment,
                                                  StringRef archivePath,
                                                  StringRef memberPath,
                                                  StringRef sectionName) {
  if (!_linkerScriptSema.hasLayoutCommands())
    return;

  std::vector<const script::SymbolAssignment *> exprs =
      _linkerScriptSema.getExprs({archivePath, memberPath, sectionName});
  for (auto expr : exprs) {
    auto expChunk =
        new (this->_allocator) ExpressionChunk<ELFT>(this->_context, expr);
    segment->append(expChunk);
  }
}

} // end namespace elf
} // end namespace lld

#endif
