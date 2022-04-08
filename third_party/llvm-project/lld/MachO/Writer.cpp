//===- Writer.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "MapFile.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SectionPriorities.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "UnwindInfoSection.h"

#include "lld/Common/Arrays.h"
#include "lld/Common/CommonLinkerContext.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

namespace {
class LCUuid;

class Writer {
public:
  Writer() : buffer(errorHandler().outputBuffer) {}

  void treatSpecialUndefineds();
  void scanRelocations();
  void scanSymbols();
  template <class LP> void createOutputSections();
  template <class LP> void createLoadCommands();
  void finalizeAddresses();
  void finalizeLinkEditSegment();
  void assignAddresses(OutputSegment *);

  void openFile();
  void writeSections();
  void writeUuid();
  void writeCodeSignature();
  void writeOutputFile();

  template <class LP> void run();

  ThreadPool threadPool;
  std::unique_ptr<FileOutputBuffer> &buffer;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  MachHeaderSection *header = nullptr;
  StringTableSection *stringTableSection = nullptr;
  SymtabSection *symtabSection = nullptr;
  IndirectSymtabSection *indirectSymtabSection = nullptr;
  CodeSignatureSection *codeSignatureSection = nullptr;
  DataInCodeSection *dataInCodeSection = nullptr;
  FunctionStartsSection *functionStartsSection = nullptr;

  LCUuid *uuidCommand = nullptr;
  OutputSegment *linkEditSegment = nullptr;
};

// LC_DYLD_INFO_ONLY stores the offsets of symbol import/export information.
class LCDyldInfo final : public LoadCommand {
public:
  LCDyldInfo(RebaseSection *rebaseSection, BindingSection *bindingSection,
             WeakBindingSection *weakBindingSection,
             LazyBindingSection *lazyBindingSection,
             ExportSection *exportSection)
      : rebaseSection(rebaseSection), bindingSection(bindingSection),
        weakBindingSection(weakBindingSection),
        lazyBindingSection(lazyBindingSection), exportSection(exportSection) {}

  uint32_t getSize() const override { return sizeof(dyld_info_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dyld_info_command *>(buf);
    c->cmd = LC_DYLD_INFO_ONLY;
    c->cmdsize = getSize();
    if (rebaseSection->isNeeded()) {
      c->rebase_off = rebaseSection->fileOff;
      c->rebase_size = rebaseSection->getFileSize();
    }
    if (bindingSection->isNeeded()) {
      c->bind_off = bindingSection->fileOff;
      c->bind_size = bindingSection->getFileSize();
    }
    if (weakBindingSection->isNeeded()) {
      c->weak_bind_off = weakBindingSection->fileOff;
      c->weak_bind_size = weakBindingSection->getFileSize();
    }
    if (lazyBindingSection->isNeeded()) {
      c->lazy_bind_off = lazyBindingSection->fileOff;
      c->lazy_bind_size = lazyBindingSection->getFileSize();
    }
    if (exportSection->isNeeded()) {
      c->export_off = exportSection->fileOff;
      c->export_size = exportSection->getFileSize();
    }
  }

  RebaseSection *rebaseSection;
  BindingSection *bindingSection;
  WeakBindingSection *weakBindingSection;
  LazyBindingSection *lazyBindingSection;
  ExportSection *exportSection;
};

class LCSubFramework final : public LoadCommand {
public:
  LCSubFramework(StringRef umbrella) : umbrella(umbrella) {}

  uint32_t getSize() const override {
    return alignTo(sizeof(sub_framework_command) + umbrella.size() + 1,
                   target->wordSize);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<sub_framework_command *>(buf);
    buf += sizeof(sub_framework_command);

    c->cmd = LC_SUB_FRAMEWORK;
    c->cmdsize = getSize();
    c->umbrella = sizeof(sub_framework_command);

    memcpy(buf, umbrella.data(), umbrella.size());
    buf[umbrella.size()] = '\0';
  }

private:
  const StringRef umbrella;
};

class LCFunctionStarts final : public LoadCommand {
public:
  explicit LCFunctionStarts(FunctionStartsSection *functionStartsSection)
      : functionStartsSection(functionStartsSection) {}

  uint32_t getSize() const override { return sizeof(linkedit_data_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<linkedit_data_command *>(buf);
    c->cmd = LC_FUNCTION_STARTS;
    c->cmdsize = getSize();
    c->dataoff = functionStartsSection->fileOff;
    c->datasize = functionStartsSection->getFileSize();
  }

private:
  FunctionStartsSection *functionStartsSection;
};

class LCDataInCode final : public LoadCommand {
public:
  explicit LCDataInCode(DataInCodeSection *dataInCodeSection)
      : dataInCodeSection(dataInCodeSection) {}

  uint32_t getSize() const override { return sizeof(linkedit_data_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<linkedit_data_command *>(buf);
    c->cmd = LC_DATA_IN_CODE;
    c->cmdsize = getSize();
    c->dataoff = dataInCodeSection->fileOff;
    c->datasize = dataInCodeSection->getFileSize();
  }

private:
  DataInCodeSection *dataInCodeSection;
};

class LCDysymtab final : public LoadCommand {
public:
  LCDysymtab(SymtabSection *symtabSection,
             IndirectSymtabSection *indirectSymtabSection)
      : symtabSection(symtabSection),
        indirectSymtabSection(indirectSymtabSection) {}

  uint32_t getSize() const override { return sizeof(dysymtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dysymtab_command *>(buf);
    c->cmd = LC_DYSYMTAB;
    c->cmdsize = getSize();

    c->ilocalsym = 0;
    c->iextdefsym = c->nlocalsym = symtabSection->getNumLocalSymbols();
    c->nextdefsym = symtabSection->getNumExternalSymbols();
    c->iundefsym = c->iextdefsym + c->nextdefsym;
    c->nundefsym = symtabSection->getNumUndefinedSymbols();

    c->indirectsymoff = indirectSymtabSection->fileOff;
    c->nindirectsyms = indirectSymtabSection->getNumSymbols();
  }

  SymtabSection *symtabSection;
  IndirectSymtabSection *indirectSymtabSection;
};

template <class LP> class LCSegment final : public LoadCommand {
public:
  LCSegment(StringRef name, OutputSegment *seg) : name(name), seg(seg) {}

  uint32_t getSize() const override {
    return sizeof(typename LP::segment_command) +
           seg->numNonHiddenSections() * sizeof(typename LP::section);
  }

  void writeTo(uint8_t *buf) const override {
    using SegmentCommand = typename LP::segment_command;
    using SectionHeader = typename LP::section;

    auto *c = reinterpret_cast<SegmentCommand *>(buf);
    buf += sizeof(SegmentCommand);

    c->cmd = LP::segmentLCType;
    c->cmdsize = getSize();
    memcpy(c->segname, name.data(), name.size());
    c->fileoff = seg->fileOff;
    c->maxprot = seg->maxProt;
    c->initprot = seg->initProt;

    c->vmaddr = seg->addr;
    c->vmsize = seg->vmSize;
    c->filesize = seg->fileSize;
    c->nsects = seg->numNonHiddenSections();

    for (const OutputSection *osec : seg->getSections()) {
      if (osec->isHidden())
        continue;

      auto *sectHdr = reinterpret_cast<SectionHeader *>(buf);
      buf += sizeof(SectionHeader);

      memcpy(sectHdr->sectname, osec->name.data(), osec->name.size());
      memcpy(sectHdr->segname, name.data(), name.size());

      sectHdr->addr = osec->addr;
      sectHdr->offset = osec->fileOff;
      sectHdr->align = Log2_32(osec->align);
      sectHdr->flags = osec->flags;
      sectHdr->size = osec->getSize();
      sectHdr->reserved1 = osec->reserved1;
      sectHdr->reserved2 = osec->reserved2;
    }
  }

private:
  StringRef name;
  OutputSegment *seg;
};

class LCMain final : public LoadCommand {
  uint32_t getSize() const override {
    return sizeof(structs::entry_point_command);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<structs::entry_point_command *>(buf);
    c->cmd = LC_MAIN;
    c->cmdsize = getSize();

    if (config->entry->isInStubs())
      c->entryoff =
          in.stubs->fileOff + config->entry->stubsIndex * target->stubSize;
    else
      c->entryoff = config->entry->getVA() - in.header->addr;

    c->stacksize = 0;
  }
};

class LCSymtab final : public LoadCommand {
public:
  LCSymtab(SymtabSection *symtabSection, StringTableSection *stringTableSection)
      : symtabSection(symtabSection), stringTableSection(stringTableSection) {}

  uint32_t getSize() const override { return sizeof(symtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<symtab_command *>(buf);
    c->cmd = LC_SYMTAB;
    c->cmdsize = getSize();
    c->symoff = symtabSection->fileOff;
    c->nsyms = symtabSection->getNumSymbols();
    c->stroff = stringTableSection->fileOff;
    c->strsize = stringTableSection->getFileSize();
  }

  SymtabSection *symtabSection = nullptr;
  StringTableSection *stringTableSection = nullptr;
};

// There are several dylib load commands that share the same structure:
//   * LC_LOAD_DYLIB
//   * LC_ID_DYLIB
//   * LC_REEXPORT_DYLIB
class LCDylib final : public LoadCommand {
public:
  LCDylib(LoadCommandType type, StringRef path,
          uint32_t compatibilityVersion = 0, uint32_t currentVersion = 0)
      : type(type), path(path), compatibilityVersion(compatibilityVersion),
        currentVersion(currentVersion) {
    instanceCount++;
  }

  uint32_t getSize() const override {
    return alignTo(sizeof(dylib_command) + path.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylib_command *>(buf);
    buf += sizeof(dylib_command);

    c->cmd = type;
    c->cmdsize = getSize();
    c->dylib.name = sizeof(dylib_command);
    c->dylib.timestamp = 0;
    c->dylib.compatibility_version = compatibilityVersion;
    c->dylib.current_version = currentVersion;

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

  static uint32_t getInstanceCount() { return instanceCount; }
  static void resetInstanceCount() { instanceCount = 0; }

private:
  LoadCommandType type;
  StringRef path;
  uint32_t compatibilityVersion;
  uint32_t currentVersion;
  static uint32_t instanceCount;
};

uint32_t LCDylib::instanceCount = 0;

class LCLoadDylinker final : public LoadCommand {
public:
  uint32_t getSize() const override {
    return alignTo(sizeof(dylinker_command) + path.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylinker_command *>(buf);
    buf += sizeof(dylinker_command);

    c->cmd = LC_LOAD_DYLINKER;
    c->cmdsize = getSize();
    c->name = sizeof(dylinker_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

private:
  // Recent versions of Darwin won't run any binary that has dyld at a
  // different location.
  const StringRef path = "/usr/lib/dyld";
};

class LCRPath final : public LoadCommand {
public:
  explicit LCRPath(StringRef path) : path(path) {}

  uint32_t getSize() const override {
    return alignTo(sizeof(rpath_command) + path.size() + 1, target->wordSize);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<rpath_command *>(buf);
    buf += sizeof(rpath_command);

    c->cmd = LC_RPATH;
    c->cmdsize = getSize();
    c->path = sizeof(rpath_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

private:
  StringRef path;
};

class LCMinVersion final : public LoadCommand {
public:
  explicit LCMinVersion(const PlatformInfo &platformInfo)
      : platformInfo(platformInfo) {}

  uint32_t getSize() const override { return sizeof(version_min_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<version_min_command *>(buf);
    switch (platformInfo.target.Platform) {
    case PLATFORM_MACOS:
      c->cmd = LC_VERSION_MIN_MACOSX;
      break;
    case PLATFORM_IOS:
    case PLATFORM_IOSSIMULATOR:
      c->cmd = LC_VERSION_MIN_IPHONEOS;
      break;
    case PLATFORM_TVOS:
    case PLATFORM_TVOSSIMULATOR:
      c->cmd = LC_VERSION_MIN_TVOS;
      break;
    case PLATFORM_WATCHOS:
    case PLATFORM_WATCHOSSIMULATOR:
      c->cmd = LC_VERSION_MIN_WATCHOS;
      break;
    default:
      llvm_unreachable("invalid platform");
      break;
    }
    c->cmdsize = getSize();
    c->version = encodeVersion(platformInfo.minimum);
    c->sdk = encodeVersion(platformInfo.sdk);
  }

private:
  const PlatformInfo &platformInfo;
};

class LCBuildVersion final : public LoadCommand {
public:
  explicit LCBuildVersion(const PlatformInfo &platformInfo)
      : platformInfo(platformInfo) {}

  const int ntools = 1;

  uint32_t getSize() const override {
    return sizeof(build_version_command) + ntools * sizeof(build_tool_version);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<build_version_command *>(buf);
    c->cmd = LC_BUILD_VERSION;
    c->cmdsize = getSize();
    c->platform = static_cast<uint32_t>(platformInfo.target.Platform);
    c->minos = encodeVersion(platformInfo.minimum);
    c->sdk = encodeVersion(platformInfo.sdk);
    c->ntools = ntools;
    auto *t = reinterpret_cast<build_tool_version *>(&c[1]);
    t->tool = TOOL_LD;
    t->version = encodeVersion(VersionTuple(
        LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH));
  }

private:
  const PlatformInfo &platformInfo;
};

// Stores a unique identifier for the output file based on an MD5 hash of its
// contents. In order to hash the contents, we must first write them, but
// LC_UUID itself must be part of the written contents in order for all the
// offsets to be calculated correctly. We resolve this circular paradox by
// first writing an LC_UUID with an all-zero UUID, then updating the UUID with
// its real value later.
class LCUuid final : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(uuid_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<uuid_command *>(buf);
    c->cmd = LC_UUID;
    c->cmdsize = getSize();
    uuidBuf = c->uuid;
  }

  void writeUuid(uint64_t digest) const {
    // xxhash only gives us 8 bytes, so put some fixed data in the other half.
    static_assert(sizeof(uuid_command::uuid) == 16, "unexpected uuid size");
    memcpy(uuidBuf, "LLD\xa1UU1D", 8);
    memcpy(uuidBuf + 8, &digest, 8);

    // RFC 4122 conformance. We need to fix 4 bits in byte 6 and 2 bits in
    // byte 8. Byte 6 is already fine due to the fixed data we put in. We don't
    // want to lose bits of the digest in byte 8, so swap that with a byte of
    // fixed data that happens to have the right bits set.
    std::swap(uuidBuf[3], uuidBuf[8]);

    // Claim that this is an MD5-based hash. It isn't, but this signals that
    // this is not a time-based and not a random hash. MD5 seems like the least
    // bad lie we can put here.
    assert((uuidBuf[6] & 0xf0) == 0x30 && "See RFC 4122 Sections 4.2.2, 4.1.3");
    assert((uuidBuf[8] & 0xc0) == 0x80 && "See RFC 4122 Section 4.2.2");
  }

  mutable uint8_t *uuidBuf;
};

template <class LP> class LCEncryptionInfo final : public LoadCommand {
public:
  uint32_t getSize() const override {
    return sizeof(typename LP::encryption_info_command);
  }

  void writeTo(uint8_t *buf) const override {
    using EncryptionInfo = typename LP::encryption_info_command;
    auto *c = reinterpret_cast<EncryptionInfo *>(buf);
    buf += sizeof(EncryptionInfo);
    c->cmd = LP::encryptionInfoLCType;
    c->cmdsize = getSize();
    c->cryptoff = in.header->getSize();
    auto it = find_if(outputSegments, [](const OutputSegment *seg) {
      return seg->name == segment_names::text;
    });
    assert(it != outputSegments.end());
    c->cryptsize = (*it)->fileSize - c->cryptoff;
  }
};

class LCCodeSignature final : public LoadCommand {
public:
  LCCodeSignature(CodeSignatureSection *section) : section(section) {}

  uint32_t getSize() const override { return sizeof(linkedit_data_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<linkedit_data_command *>(buf);
    c->cmd = LC_CODE_SIGNATURE;
    c->cmdsize = getSize();
    c->dataoff = static_cast<uint32_t>(section->fileOff);
    c->datasize = section->getSize();
  }

  CodeSignatureSection *section;
};

} // namespace

void Writer::treatSpecialUndefineds() {
  if (config->entry)
    if (auto *undefined = dyn_cast<Undefined>(config->entry))
      treatUndefinedSymbol(*undefined, "the entry point");

  // FIXME: This prints symbols that are undefined both in input files and
  // via -u flag twice.
  for (const Symbol *sym : config->explicitUndefineds) {
    if (const auto *undefined = dyn_cast<Undefined>(sym))
      treatUndefinedSymbol(*undefined, "-u");
  }
  // Literal exported-symbol names must be defined, but glob
  // patterns need not match.
  for (const CachedHashStringRef &cachedName :
       config->exportedSymbols.literals) {
    if (const Symbol *sym = symtab->find(cachedName))
      if (const auto *undefined = dyn_cast<Undefined>(sym))
        treatUndefinedSymbol(*undefined, "-exported_symbol(s_list)");
  }
}

// Add stubs and bindings where necessary (e.g. if the symbol is a
// DylibSymbol.)
static void prepareBranchTarget(Symbol *sym) {
  if (auto *dysym = dyn_cast<DylibSymbol>(sym)) {
    if (in.stubs->addEntry(dysym)) {
      if (sym->isWeakDef()) {
        in.binding->addEntry(dysym, in.lazyPointers->isec,
                             sym->stubsIndex * target->wordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers->isec,
                                 sym->stubsIndex * target->wordSize);
      } else {
        in.lazyBinding->addEntry(dysym);
      }
    }
  } else if (auto *defined = dyn_cast<Defined>(sym)) {
    if (defined->isExternalWeakDef()) {
      if (in.stubs->addEntry(sym)) {
        in.rebase->addEntry(in.lazyPointers->isec,
                            sym->stubsIndex * target->wordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers->isec,
                                 sym->stubsIndex * target->wordSize);
      }
    }
  } else {
    llvm_unreachable("invalid branch target symbol type");
  }
}

// Can a symbol's address can only be resolved at runtime?
static bool needsBinding(const Symbol *sym) {
  if (isa<DylibSymbol>(sym))
    return true;
  if (const auto *defined = dyn_cast<Defined>(sym))
    return defined->isExternalWeakDef();
  return false;
}

static void prepareSymbolRelocation(Symbol *sym, const InputSection *isec,
                                    const lld::macho::Reloc &r) {
  assert(sym->isLive());
  const RelocAttrs &relocAttrs = target->getRelocAttrs(r.type);

  if (relocAttrs.hasAttr(RelocAttrBits::BRANCH)) {
    prepareBranchTarget(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::GOT)) {
    if (relocAttrs.hasAttr(RelocAttrBits::POINTER) || needsBinding(sym))
      in.got->addEntry(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::TLV)) {
    if (needsBinding(sym))
      in.tlvPointers->addEntry(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::UNSIGNED)) {
    // References from thread-local variable sections are treated as offsets
    // relative to the start of the referent section, and therefore have no
    // need of rebase opcodes.
    if (!(isThreadLocalVariables(isec->getFlags()) && isa<Defined>(sym)))
      addNonLazyBindingEntries(sym, isec, r.offset, r.addend);
  }
}

void Writer::scanRelocations() {
  TimeTraceScope timeScope("Scan relocations");

  // This can't use a for-each loop: It calls treatUndefinedSymbol(), which can
  // add to inputSections, which invalidates inputSections's iterators.
  for (size_t i = 0; i < inputSections.size(); ++i) {
    ConcatInputSection *isec = inputSections[i];

    if (isec->shouldOmitFromOutput())
      continue;

    for (auto it = isec->relocs.begin(); it != isec->relocs.end(); ++it) {
      lld::macho::Reloc &r = *it;
      if (target->hasAttr(r.type, RelocAttrBits::SUBTRAHEND)) {
        // Skip over the following UNSIGNED relocation -- it's just there as the
        // minuend, and doesn't have the usual UNSIGNED semantics. We don't want
        // to emit rebase opcodes for it.
        it++;
        continue;
      }
      if (auto *sym = r.referent.dyn_cast<Symbol *>()) {
        if (auto *undefined = dyn_cast<Undefined>(sym))
          treatUndefinedSymbol(*undefined);
        // treatUndefinedSymbol() can replace sym with a DylibSymbol; re-check.
        if (!isa<Undefined>(sym) && validateSymbolRelocation(sym, isec, r))
          prepareSymbolRelocation(sym, isec, r);
      } else {
        // Canonicalize the referent so that later accesses in Writer won't
        // have to worry about it. Perhaps we should do this for Defined::isec
        // too...
        auto *referentIsec = r.referent.get<InputSection *>();
        r.referent = referentIsec->canonical();
        if (!r.pcrel)
          in.rebase->addEntry(isec, r.offset);
      }
    }
  }

  in.unwindInfo->prepareRelocations();
}

void Writer::scanSymbols() {
  TimeTraceScope timeScope("Scan symbols");
  for (Symbol *sym : symtab->getSymbols()) {
    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (!defined->isLive())
        continue;
      defined->canonicalize();
      if (defined->overridesWeakDef)
        in.weakBinding->addNonWeakDefinition(defined);
      if (!defined->isAbsolute() && isCodeSection(defined->isec))
        in.unwindInfo->addSymbol(defined);
    } else if (const auto *dysym = dyn_cast<DylibSymbol>(sym)) {
      // This branch intentionally doesn't check isLive().
      if (dysym->isDynamicLookup())
        continue;
      dysym->getFile()->refState =
          std::max(dysym->getFile()->refState, dysym->getRefState());
    }
  }

  for (const InputFile *file : inputFiles) {
    if (auto *objFile = dyn_cast<ObjFile>(file))
      for (Symbol *sym : objFile->symbols) {
        if (auto *defined = dyn_cast_or_null<Defined>(sym)) {
          if (!defined->isLive())
            continue;
          defined->canonicalize();
          if (!defined->isExternal() && !defined->isAbsolute() &&
              isCodeSection(defined->isec))
            in.unwindInfo->addSymbol(defined);
        }
      }
  }
}

// TODO: ld64 enforces the old load commands in a few other cases.
static bool useLCBuildVersion(const PlatformInfo &platformInfo) {
  static const std::vector<std::pair<PlatformType, VersionTuple>> minVersion = {
      {PLATFORM_MACOS, VersionTuple(10, 14)},
      {PLATFORM_IOS, VersionTuple(12, 0)},
      {PLATFORM_IOSSIMULATOR, VersionTuple(13, 0)},
      {PLATFORM_TVOS, VersionTuple(12, 0)},
      {PLATFORM_TVOSSIMULATOR, VersionTuple(13, 0)},
      {PLATFORM_WATCHOS, VersionTuple(5, 0)},
      {PLATFORM_WATCHOSSIMULATOR, VersionTuple(6, 0)}};
  auto it = llvm::find_if(minVersion, [&](const auto &p) {
    return p.first == platformInfo.target.Platform;
  });
  return it == minVersion.end() ? true : platformInfo.minimum >= it->second;
}

template <class LP> void Writer::createLoadCommands() {
  uint8_t segIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    in.header->addLoadCommand(make<LCSegment<LP>>(seg->name, seg));
    seg->index = segIndex++;
  }

  in.header->addLoadCommand(make<LCDyldInfo>(
      in.rebase, in.binding, in.weakBinding, in.lazyBinding, in.exports));
  in.header->addLoadCommand(make<LCSymtab>(symtabSection, stringTableSection));
  in.header->addLoadCommand(
      make<LCDysymtab>(symtabSection, indirectSymtabSection));
  if (!config->umbrella.empty())
    in.header->addLoadCommand(make<LCSubFramework>(config->umbrella));
  if (config->emitEncryptionInfo)
    in.header->addLoadCommand(make<LCEncryptionInfo<LP>>());
  for (StringRef path : config->runtimePaths)
    in.header->addLoadCommand(make<LCRPath>(path));

  switch (config->outputType) {
  case MH_EXECUTE:
    in.header->addLoadCommand(make<LCLoadDylinker>());
    break;
  case MH_DYLIB:
    in.header->addLoadCommand(make<LCDylib>(LC_ID_DYLIB, config->installName,
                                            config->dylibCompatibilityVersion,
                                            config->dylibCurrentVersion));
    break;
  case MH_BUNDLE:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  uuidCommand = make<LCUuid>();
  in.header->addLoadCommand(uuidCommand);

  if (useLCBuildVersion(config->platformInfo))
    in.header->addLoadCommand(make<LCBuildVersion>(config->platformInfo));
  else
    in.header->addLoadCommand(make<LCMinVersion>(config->platformInfo));

  // This is down here to match ld64's load command order.
  if (config->outputType == MH_EXECUTE)
    in.header->addLoadCommand(make<LCMain>());

  int64_t dylibOrdinal = 1;
  DenseMap<StringRef, int64_t> ordinalForInstallName;
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      if (dylibFile->isBundleLoader) {
        dylibFile->ordinal = BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE;
        // Shortcut since bundle-loader does not re-export the symbols.

        dylibFile->reexport = false;
        continue;
      }

      // Don't emit load commands for a dylib that is not referenced if:
      // - it was added implicitly (via a reexport, an LC_LOAD_DYLINKER --
      //   if it's on the linker command line, it's explicit)
      // - or it's marked MH_DEAD_STRIPPABLE_DYLIB
      // - or the flag -dead_strip_dylibs is used
      // FIXME: `isReferenced()` is currently computed before dead code
      // stripping, so references from dead code keep a dylib alive. This
      // matches ld64, but it's something we should do better.
      if (!dylibFile->isReferenced() && !dylibFile->forceNeeded &&
          (!dylibFile->explicitlyLinked || dylibFile->deadStrippable ||
           config->deadStripDylibs))
        continue;

      // Several DylibFiles can have the same installName. Only emit a single
      // load command for that installName and give all these DylibFiles the
      // same ordinal.
      // This can happen in several cases:
      // - a new framework could change its installName to an older
      //   framework name via an $ld$ symbol depending on platform_version
      // - symlinks (for example, libpthread.tbd is a symlink to libSystem.tbd;
      //   Foo.framework/Foo.tbd is usually a symlink to
      //   Foo.framework/Versions/Current/Foo.tbd, where
      //   Foo.framework/Versions/Current is usually a symlink to
      //   Foo.framework/Versions/A)
      // - a framework can be linked both explicitly on the linker
      //   command line and implicitly as a reexport from a different
      //   framework. The re-export will usually point to the tbd file
      //   in Foo.framework/Versions/A/Foo.tbd, while the explicit link will
      //   usually find Foo.framework/Foo.tbd. These are usually symlinks,
      //   but in a --reproduce archive they will be identical but distinct
      //   files.
      // In the first case, *semantically distinct* DylibFiles will have the
      // same installName.
      int64_t &ordinal = ordinalForInstallName[dylibFile->installName];
      if (ordinal) {
        dylibFile->ordinal = ordinal;
        continue;
      }

      ordinal = dylibFile->ordinal = dylibOrdinal++;
      LoadCommandType lcType =
          dylibFile->forceWeakImport || dylibFile->refState == RefState::Weak
              ? LC_LOAD_WEAK_DYLIB
              : LC_LOAD_DYLIB;
      in.header->addLoadCommand(make<LCDylib>(lcType, dylibFile->installName,
                                              dylibFile->compatibilityVersion,
                                              dylibFile->currentVersion));

      if (dylibFile->reexport)
        in.header->addLoadCommand(
            make<LCDylib>(LC_REEXPORT_DYLIB, dylibFile->installName));
    }
  }

  if (functionStartsSection)
    in.header->addLoadCommand(make<LCFunctionStarts>(functionStartsSection));
  if (dataInCodeSection)
    in.header->addLoadCommand(make<LCDataInCode>(dataInCodeSection));
  if (codeSignatureSection)
    in.header->addLoadCommand(make<LCCodeSignature>(codeSignatureSection));

  const uint32_t MACOS_MAXPATHLEN = 1024;
  config->headerPad = std::max(
      config->headerPad, (config->headerPadMaxInstallNames
                              ? LCDylib::getInstanceCount() * MACOS_MAXPATHLEN
                              : 0));
}

// Sorting only can happen once all outputs have been collected. Here we sort
// segments, output sections within each segment, and input sections within each
// output segment.
static void sortSegmentsAndSections() {
  TimeTraceScope timeScope("Sort segments and sections");
  sortOutputSegments();

  DenseMap<const InputSection *, size_t> isecPriorities =
      buildInputSectionPriorities();

  uint32_t sectionIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    seg->sortOutputSections();
    // References from thread-local variable sections are treated as offsets
    // relative to the start of the thread-local data memory area, which
    // is initialized via copying all the TLV data sections (which are all
    // contiguous). If later data sections require a greater alignment than
    // earlier ones, the offsets of data within those sections won't be
    // guaranteed to aligned unless we normalize alignments. We therefore use
    // the largest alignment for all TLV data sections.
    uint32_t tlvAlign = 0;
    for (const OutputSection *osec : seg->getSections())
      if (isThreadLocalData(osec->flags) && osec->align > tlvAlign)
        tlvAlign = osec->align;

    for (OutputSection *osec : seg->getSections()) {
      // Now that the output sections are sorted, assign the final
      // output section indices.
      if (!osec->isHidden())
        osec->index = ++sectionIndex;
      if (isThreadLocalData(osec->flags)) {
        if (!firstTLVDataSection)
          firstTLVDataSection = osec;
        osec->align = tlvAlign;
      }

      if (!isecPriorities.empty()) {
        if (auto *merged = dyn_cast<ConcatOutputSection>(osec)) {
          llvm::stable_sort(merged->inputs,
                            [&](InputSection *a, InputSection *b) {
                              return isecPriorities[a] > isecPriorities[b];
                            });
        }
      }
    }
  }
}

template <class LP> void Writer::createOutputSections() {
  TimeTraceScope timeScope("Create output sections");
  // First, create hidden sections
  stringTableSection = make<StringTableSection>();
  symtabSection = makeSymtabSection<LP>(*stringTableSection);
  indirectSymtabSection = make<IndirectSymtabSection>();
  if (config->adhocCodesign)
    codeSignatureSection = make<CodeSignatureSection>();
  if (config->emitDataInCodeInfo)
    dataInCodeSection = make<DataInCodeSection>();
  if (config->emitFunctionStarts)
    functionStartsSection = make<FunctionStartsSection>();
  if (config->emitBitcodeBundle)
    make<BitcodeBundleSection>();

  switch (config->outputType) {
  case MH_EXECUTE:
    make<PageZeroSection>();
    break;
  case MH_DYLIB:
  case MH_BUNDLE:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  // Then add input sections to output sections.
  for (ConcatInputSection *isec : inputSections) {
    if (isec->shouldOmitFromOutput())
      continue;
    ConcatOutputSection *osec = cast<ConcatOutputSection>(isec->parent);
    osec->addInput(isec);
    osec->inputOrder =
        std::min(osec->inputOrder, static_cast<int>(isec->outSecOff));
  }

  // Once all the inputs are added, we can finalize the output section
  // properties and create the corresponding output segments.
  for (const auto &it : concatOutputSections) {
    StringRef segname = it.first.first;
    ConcatOutputSection *osec = it.second;
    assert(segname != segment_names::ld);
    if (osec->isNeeded())
      getOrCreateOutputSegment(segname)->addOutputSection(osec);
  }

  for (SyntheticSection *ssec : syntheticSections) {
    auto it = concatOutputSections.find({ssec->segname, ssec->name});
    // We add all LinkEdit sections here because we don't know if they are
    // needed until their finalizeContents() methods get called later. While
    // this means that we add some redundant sections to __LINKEDIT, there is
    // is no redundancy in the output, as we do not emit section headers for
    // any LinkEdit sections.
    if (ssec->isNeeded() || ssec->segname == segment_names::linkEdit) {
      if (it == concatOutputSections.end()) {
        getOrCreateOutputSegment(ssec->segname)->addOutputSection(ssec);
      } else {
        fatal("section from " +
              toString(it->second->firstSection()->getFile()) +
              " conflicts with synthetic section " + ssec->segname + "," +
              ssec->name);
      }
    }
  }

  // dyld requires __LINKEDIT segment to always exist (even if empty).
  linkEditSegment = getOrCreateOutputSegment(segment_names::linkEdit);
}

void Writer::finalizeAddresses() {
  TimeTraceScope timeScope("Finalize addresses");
  uint64_t pageSize = target->getPageSize();
  // Ensure that segments (and the sections they contain) are allocated
  // addresses in ascending order, which dyld requires.
  //
  // Note that at this point, __LINKEDIT sections are empty, but we need to
  // determine addresses of other segments/sections before generating its
  // contents.
  for (OutputSegment *seg : outputSegments) {
    if (seg == linkEditSegment)
      continue;
    seg->addr = addr;
    assignAddresses(seg);
    // codesign / libstuff checks for segment ordering by verifying that
    // `fileOff + fileSize == next segment fileOff`. So we call alignTo() before
    // (instead of after) computing fileSize to ensure that the segments are
    // contiguous. We handle addr / vmSize similarly for the same reason.
    fileOff = alignTo(fileOff, pageSize);
    addr = alignTo(addr, pageSize);
    seg->vmSize = addr - seg->addr;
    seg->fileSize = fileOff - seg->fileOff;
    seg->assignAddressesToStartEndSymbols();
  }
}

void Writer::finalizeLinkEditSegment() {
  TimeTraceScope timeScope("Finalize __LINKEDIT segment");
  // Fill __LINKEDIT contents.
  std::vector<LinkEditSection *> linkEditSections{
      in.rebase,
      in.binding,
      in.weakBinding,
      in.lazyBinding,
      in.exports,
      symtabSection,
      indirectSymtabSection,
      dataInCodeSection,
      functionStartsSection,
  };
  SmallVector<std::shared_future<void>> threadFutures;
  threadFutures.reserve(linkEditSections.size());
  for (LinkEditSection *osec : linkEditSections)
    if (osec)
      threadFutures.emplace_back(threadPool.async(
          [](LinkEditSection *osec) { osec->finalizeContents(); }, osec));
  for (std::shared_future<void> &future : threadFutures)
    future.wait();

  // Now that __LINKEDIT is filled out, do a proper calculation of its
  // addresses and offsets.
  linkEditSegment->addr = addr;
  assignAddresses(linkEditSegment);
  // No need to page-align fileOff / addr here since this is the last segment.
  linkEditSegment->vmSize = addr - linkEditSegment->addr;
  linkEditSegment->fileSize = fileOff - linkEditSegment->fileOff;
}

void Writer::assignAddresses(OutputSegment *seg) {
  seg->fileOff = fileOff;

  for (OutputSection *osec : seg->getSections()) {
    if (!osec->isNeeded())
      continue;
    addr = alignTo(addr, osec->align);
    fileOff = alignTo(fileOff, osec->align);
    osec->addr = addr;
    osec->fileOff = isZeroFill(osec->flags) ? 0 : fileOff;
    osec->finalize();
    osec->assignAddressesToStartEndSymbols();

    addr += osec->getSize();
    fileOff += osec->getFileSize();
  }
}

void Writer::openFile() {
  Expected<std::unique_ptr<FileOutputBuffer>> bufferOrErr =
      FileOutputBuffer::create(config->outputFile, fileOff,
                               FileOutputBuffer::F_executable);

  if (!bufferOrErr)
    error("failed to open " + config->outputFile + ": " +
          llvm::toString(bufferOrErr.takeError()));
  else
    buffer = std::move(*bufferOrErr);
}

void Writer::writeSections() {
  uint8_t *buf = buffer->getBufferStart();
  for (const OutputSegment *seg : outputSegments)
    for (const OutputSection *osec : seg->getSections())
      osec->writeTo(buf + osec->fileOff);
}

// In order to utilize multiple cores, we first split the buffer into chunks,
// compute a hash for each chunk, and then compute a hash value of the hash
// values.
void Writer::writeUuid() {
  TimeTraceScope timeScope("Computing UUID");

  ArrayRef<uint8_t> data{buffer->getBufferStart(), buffer->getBufferEnd()};
  unsigned chunkCount = parallel::strategy.compute_thread_count() * 10;
  // Round-up integer division
  size_t chunkSize = (data.size() + chunkCount - 1) / chunkCount;
  std::vector<ArrayRef<uint8_t>> chunks = split(data, chunkSize);
  std::vector<uint64_t> hashes(chunks.size());
  SmallVector<std::shared_future<void>> threadFutures;
  threadFutures.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i)
    threadFutures.emplace_back(threadPool.async(
        [&](size_t j) { hashes[j] = xxHash64(chunks[j]); }, i));
  for (std::shared_future<void> &future : threadFutures)
    future.wait();

  uint64_t digest = xxHash64({reinterpret_cast<uint8_t *>(hashes.data()),
                              hashes.size() * sizeof(uint64_t)});
  uuidCommand->writeUuid(digest);
}

void Writer::writeCodeSignature() {
  if (codeSignatureSection)
    codeSignatureSection->writeHashes(buffer->getBufferStart());
}

void Writer::writeOutputFile() {
  TimeTraceScope timeScope("Write output file");
  openFile();
  if (errorCount())
    return;
  writeSections();
  writeUuid();
  writeCodeSignature();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

template <class LP> void Writer::run() {
  treatSpecialUndefineds();
  if (config->entry && !isa<Undefined>(config->entry))
    prepareBranchTarget(config->entry);

  // Canonicalization of all pointers to InputSections should be handled by
  // these two scan* methods. I.e. from this point onward, for all live
  // InputSections, we should have `isec->canonical() == isec`.
  scanSymbols();
  scanRelocations();

  // Do not proceed if there was an undefined symbol.
  if (errorCount())
    return;

  if (in.stubHelper->isNeeded())
    in.stubHelper->setup();
  // At this point, we should know exactly which output sections are needed,
  // courtesy of scanSymbols() and scanRelocations().
  createOutputSections<LP>();

  // After this point, we create no new segments; HOWEVER, we might
  // yet create branch-range extension thunks for architectures whose
  // hardware call instructions have limited range, e.g., ARM(64).
  // The thunks are created as InputSections interspersed among
  // the ordinary __TEXT,_text InputSections.
  sortSegmentsAndSections();
  createLoadCommands<LP>();
  finalizeAddresses();
  threadPool.async([&] {
    if (LLVM_ENABLE_THREADS && config->timeTraceEnabled)
      timeTraceProfilerInitialize(config->timeTraceGranularity, "writeMapFile");
    writeMapFile();
    if (LLVM_ENABLE_THREADS && config->timeTraceEnabled)
      timeTraceProfilerFinishThread();
  });
  finalizeLinkEditSegment();
  writeOutputFile();
}

template <class LP> void macho::writeResult() { Writer().run<LP>(); }

void macho::resetWriter() { LCDylib::resetInstanceCount(); }

void macho::createSyntheticSections() {
  in.header = make<MachHeaderSection>();
  if (config->dedupLiterals)
    in.cStringSection = make<DeduplicatedCStringSection>();
  else
    in.cStringSection = make<CStringSection>();
  in.wordLiteralSection =
      config->dedupLiterals ? make<WordLiteralSection>() : nullptr;
  in.rebase = make<RebaseSection>();
  in.binding = make<BindingSection>();
  in.weakBinding = make<WeakBindingSection>();
  in.lazyBinding = make<LazyBindingSection>();
  in.exports = make<ExportSection>();
  in.got = make<GotSection>();
  in.tlvPointers = make<TlvPointerSection>();
  in.lazyPointers = make<LazyPointerSection>();
  in.stubs = make<StubsSection>();
  in.stubHelper = make<StubHelperSection>();
  in.unwindInfo = makeUnwindInfoSection();

  // This section contains space for just a single word, and will be used by
  // dyld to cache an address to the image loader it uses.
  uint8_t *arr = bAlloc().Allocate<uint8_t>(target->wordSize);
  memset(arr, 0, target->wordSize);
  in.imageLoaderCache = makeSyntheticInputSection(
      segment_names::data, section_names::data, S_REGULAR,
      ArrayRef<uint8_t>{arr, target->wordSize},
      /*align=*/target->wordSize);
  // References from dyld are not visible to us, so ensure this section is
  // always treated as live.
  in.imageLoaderCache->live = true;
}

OutputSection *macho::firstTLVDataSection = nullptr;

template void macho::writeResult<LP64>();
template void macho::writeResult<ILP32>();
