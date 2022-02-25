//===---- DebugObjectManagerPlugin.h - JITLink debug objects ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;
using namespace llvm::object;

namespace llvm {
namespace orc {

class DebugObjectSection {
public:
  virtual void setTargetMemoryRange(SectionRange Range) = 0;
  virtual void dump(raw_ostream &OS, StringRef Name) {}
  virtual ~DebugObjectSection() {}
};

template <typename ELFT>
class ELFDebugObjectSection : public DebugObjectSection {
public:
  // BinaryFormat ELF is not meant as a mutable format. We can only make changes
  // that don't invalidate the file structure.
  ELFDebugObjectSection(const typename ELFT::Shdr *Header)
      : Header(const_cast<typename ELFT::Shdr *>(Header)) {}

  void setTargetMemoryRange(SectionRange Range) override;
  void dump(raw_ostream &OS, StringRef Name) override;

  Error validateInBounds(StringRef Buffer, const char *Name) const;

private:
  typename ELFT::Shdr *Header;

  bool isTextOrDataSection() const;
};

template <typename ELFT>
void ELFDebugObjectSection<ELFT>::setTargetMemoryRange(SectionRange Range) {
  // Only patch load-addresses for executable and data sections.
  if (isTextOrDataSection()) {
    Header->sh_addr = static_cast<typename ELFT::uint>(Range.getStart());
  }
}

template <typename ELFT>
bool ELFDebugObjectSection<ELFT>::isTextOrDataSection() const {
  switch (Header->sh_type) {
  case ELF::SHT_PROGBITS:
  case ELF::SHT_X86_64_UNWIND:
    return Header->sh_flags & (ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
  }
  return false;
}

template <typename ELFT>
Error ELFDebugObjectSection<ELFT>::validateInBounds(StringRef Buffer,
                                                    const char *Name) const {
  const uint8_t *Start = Buffer.bytes_begin();
  const uint8_t *End = Buffer.bytes_end();
  const uint8_t *HeaderPtr = reinterpret_cast<uint8_t *>(Header);
  if (HeaderPtr < Start || HeaderPtr + sizeof(typename ELFT::Shdr) > End)
    return make_error<StringError>(
        formatv("{0} section header at {1:x16} not within bounds of the "
                "given debug object buffer [{2:x16} - {3:x16}]",
                Name, &Header->sh_addr, Start, End),
        inconvertibleErrorCode());
  if (Header->sh_offset + Header->sh_size > Buffer.size())
    return make_error<StringError>(
        formatv("{0} section data [{1:x16} - {2:x16}] not within bounds of "
                "the given debug object buffer [{3:x16} - {4:x16}]",
                Name, Start + Header->sh_offset,
                Start + Header->sh_offset + Header->sh_size, Start, End),
        inconvertibleErrorCode());
  return Error::success();
}

template <typename ELFT>
void ELFDebugObjectSection<ELFT>::dump(raw_ostream &OS, StringRef Name) {
  if (auto Addr = static_cast<JITTargetAddress>(Header->sh_addr)) {
    OS << formatv("  {0:x16} {1}\n", Addr, Name);
  } else {
    OS << formatv("                     {0}\n", Name);
  }
}

static constexpr sys::Memory::ProtectionFlags ReadOnly =
    static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ);

enum class Requirement {
  // Request final target memory load-addresses for all sections.
  ReportFinalSectionLoadAddresses,
};

/// The plugin creates a debug object from JITLinkContext when JITLink starts
/// processing the corresponding LinkGraph. It provides access to the pass
/// configuration of the LinkGraph and calls the finalization function, once
/// the resulting link artifact was emitted.
///
class DebugObject {
public:
  DebugObject(JITLinkContext &Ctx, ExecutionSession &ES) : Ctx(Ctx), ES(ES) {}

  void set(Requirement Req) { Reqs.insert(Req); }
  bool has(Requirement Req) const { return Reqs.count(Req) > 0; }

  using FinalizeContinuation = std::function<void(Expected<sys::MemoryBlock>)>;
  void finalizeAsync(FinalizeContinuation OnFinalize);

  virtual ~DebugObject() {
    if (Alloc)
      if (Error Err = Alloc->deallocate())
        ES.reportError(std::move(Err));
  }

  virtual void reportSectionTargetMemoryRange(StringRef Name,
                                              SectionRange TargetMem) {}

protected:
  using Allocation = JITLinkMemoryManager::Allocation;

  virtual Expected<std::unique_ptr<Allocation>>
  finalizeWorkingMemory(JITLinkContext &Ctx) = 0;

private:
  JITLinkContext &Ctx;
  ExecutionSession &ES;
  std::set<Requirement> Reqs;
  std::unique_ptr<Allocation> Alloc{nullptr};
};

// Finalize working memory and take ownership of the resulting allocation. Start
// copying memory over to the target and pass on the result once we're done.
// Ownership of the allocation remains with us for the rest of our lifetime.
void DebugObject::finalizeAsync(FinalizeContinuation OnFinalize) {
  assert(Alloc == nullptr && "Cannot finalize more than once");

  auto AllocOrErr = finalizeWorkingMemory(Ctx);
  if (!AllocOrErr)
    OnFinalize(AllocOrErr.takeError());
  Alloc = std::move(*AllocOrErr);

  Alloc->finalizeAsync([this, OnFinalize](Error Err) {
    if (Err)
      OnFinalize(std::move(Err));
    else
      OnFinalize(sys::MemoryBlock(
          jitTargetAddressToPointer<void *>(Alloc->getTargetMemory(ReadOnly)),
          Alloc->getWorkingMemory(ReadOnly).size()));
  });
}

/// The current implementation of ELFDebugObject replicates the approach used in
/// RuntimeDyld: It patches executable and data section headers in the given
/// object buffer with load-addresses of their corresponding sections in target
/// memory.
///
class ELFDebugObject : public DebugObject {
public:
  static Expected<std::unique_ptr<DebugObject>>
  Create(MemoryBufferRef Buffer, JITLinkContext &Ctx, ExecutionSession &ES);

  void reportSectionTargetMemoryRange(StringRef Name,
                                      SectionRange TargetMem) override;

  StringRef getBuffer() const { return Buffer->getMemBufferRef().getBuffer(); }

protected:
  Expected<std::unique_ptr<Allocation>>
  finalizeWorkingMemory(JITLinkContext &Ctx) override;

  template <typename ELFT>
  Error recordSection(StringRef Name,
                      std::unique_ptr<ELFDebugObjectSection<ELFT>> Section);
  DebugObjectSection *getSection(StringRef Name);

private:
  template <typename ELFT>
  static Expected<std::unique_ptr<ELFDebugObject>>
  CreateArchType(MemoryBufferRef Buffer, JITLinkContext &Ctx,
                 ExecutionSession &ES);

  static std::unique_ptr<WritableMemoryBuffer>
  CopyBuffer(MemoryBufferRef Buffer, Error &Err);

  ELFDebugObject(std::unique_ptr<WritableMemoryBuffer> Buffer,
                 JITLinkContext &Ctx, ExecutionSession &ES)
      : DebugObject(Ctx, ES), Buffer(std::move(Buffer)) {
    set(Requirement::ReportFinalSectionLoadAddresses);
  }

  std::unique_ptr<WritableMemoryBuffer> Buffer;
  StringMap<std::unique_ptr<DebugObjectSection>> Sections;
};

static const std::set<StringRef> DwarfSectionNames = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

static bool isDwarfSection(StringRef SectionName) {
  return DwarfSectionNames.count(SectionName) == 1;
}

std::unique_ptr<WritableMemoryBuffer>
ELFDebugObject::CopyBuffer(MemoryBufferRef Buffer, Error &Err) {
  ErrorAsOutParameter _(&Err);
  size_t Size = Buffer.getBufferSize();
  StringRef Name = Buffer.getBufferIdentifier();
  if (auto Copy = WritableMemoryBuffer::getNewUninitMemBuffer(Size, Name)) {
    memcpy(Copy->getBufferStart(), Buffer.getBufferStart(), Size);
    return Copy;
  }

  Err = errorCodeToError(make_error_code(errc::not_enough_memory));
  return nullptr;
}

template <typename ELFT>
Expected<std::unique_ptr<ELFDebugObject>>
ELFDebugObject::CreateArchType(MemoryBufferRef Buffer, JITLinkContext &Ctx,
                               ExecutionSession &ES) {
  using SectionHeader = typename ELFT::Shdr;

  Error Err = Error::success();
  std::unique_ptr<ELFDebugObject> DebugObj(
      new ELFDebugObject(CopyBuffer(Buffer, Err), Ctx, ES));
  if (Err)
    return std::move(Err);

  Expected<ELFFile<ELFT>> ObjRef = ELFFile<ELFT>::create(DebugObj->getBuffer());
  if (!ObjRef)
    return ObjRef.takeError();

  // TODO: Add support for other architectures.
  uint16_t TargetMachineArch = ObjRef->getHeader().e_machine;
  if (TargetMachineArch != ELF::EM_X86_64)
    return nullptr;

  Expected<ArrayRef<SectionHeader>> Sections = ObjRef->sections();
  if (!Sections)
    return Sections.takeError();

  bool HasDwarfSection = false;
  for (const SectionHeader &Header : *Sections) {
    Expected<StringRef> Name = ObjRef->getSectionName(Header);
    if (!Name)
      return Name.takeError();
    if (Name->empty())
      continue;
    HasDwarfSection |= isDwarfSection(*Name);

    auto Wrapped = std::make_unique<ELFDebugObjectSection<ELFT>>(&Header);
    if (Error Err = DebugObj->recordSection(*Name, std::move(Wrapped)))
      return std::move(Err);
  }

  if (!HasDwarfSection) {
    LLVM_DEBUG(dbgs() << "Aborting debug registration for LinkGraph \""
                      << DebugObj->Buffer->getBufferIdentifier()
                      << "\": input object contains no debug info\n");
    return nullptr;
  }

  return std::move(DebugObj);
}

Expected<std::unique_ptr<DebugObject>>
ELFDebugObject::Create(MemoryBufferRef Buffer, JITLinkContext &Ctx,
                       ExecutionSession &ES) {
  unsigned char Class, Endian;
  std::tie(Class, Endian) = getElfArchType(Buffer.getBuffer());

  if (Class == ELF::ELFCLASS32) {
    if (Endian == ELF::ELFDATA2LSB)
      return CreateArchType<ELF32LE>(Buffer, Ctx, ES);
    if (Endian == ELF::ELFDATA2MSB)
      return CreateArchType<ELF32BE>(Buffer, Ctx, ES);
    return nullptr;
  }
  if (Class == ELF::ELFCLASS64) {
    if (Endian == ELF::ELFDATA2LSB)
      return CreateArchType<ELF64LE>(Buffer, Ctx, ES);
    if (Endian == ELF::ELFDATA2MSB)
      return CreateArchType<ELF64BE>(Buffer, Ctx, ES);
    return nullptr;
  }
  return nullptr;
}

Expected<std::unique_ptr<DebugObject::Allocation>>
ELFDebugObject::finalizeWorkingMemory(JITLinkContext &Ctx) {
  LLVM_DEBUG({
    dbgs() << "Section load-addresses in debug object for \""
           << Buffer->getBufferIdentifier() << "\":\n";
    for (const auto &KV : Sections)
      KV.second->dump(dbgs(), KV.first());
  });

  // TODO: This works, but what actual alignment requirements do we have?
  unsigned Alignment = sys::Process::getPageSizeEstimate();
  JITLinkMemoryManager &MemMgr = Ctx.getMemoryManager();
  const JITLinkDylib *JD = Ctx.getJITLinkDylib();
  size_t Size = Buffer->getBufferSize();

  // Allocate working memory for debug object in read-only segment.
  JITLinkMemoryManager::SegmentsRequestMap SingleReadOnlySegment;
  SingleReadOnlySegment[ReadOnly] =
      JITLinkMemoryManager::SegmentRequest(Alignment, Size, 0);

  auto AllocOrErr = MemMgr.allocate(JD, SingleReadOnlySegment);
  if (!AllocOrErr)
    return AllocOrErr.takeError();

  // Initialize working memory with a copy of our object buffer.
  // TODO: Use our buffer as working memory directly.
  std::unique_ptr<Allocation> Alloc = std::move(*AllocOrErr);
  MutableArrayRef<char> WorkingMem = Alloc->getWorkingMemory(ReadOnly);
  memcpy(WorkingMem.data(), Buffer->getBufferStart(), Size);
  Buffer.reset();

  return std::move(Alloc);
}

void ELFDebugObject::reportSectionTargetMemoryRange(StringRef Name,
                                                    SectionRange TargetMem) {
  if (auto *DebugObjSection = getSection(Name))
    DebugObjSection->setTargetMemoryRange(TargetMem);
}

template <typename ELFT>
Error ELFDebugObject::recordSection(
    StringRef Name, std::unique_ptr<ELFDebugObjectSection<ELFT>> Section) {
  if (Error Err = Section->validateInBounds(this->getBuffer(), Name.data()))
    return Err;
  auto ItInserted = Sections.try_emplace(Name, std::move(Section));
  if (!ItInserted.second)
    return make_error<StringError>("Duplicate section",
                                   inconvertibleErrorCode());
  return Error::success();
}

DebugObjectSection *ELFDebugObject::getSection(StringRef Name) {
  auto It = Sections.find(Name);
  return It == Sections.end() ? nullptr : It->second.get();
}

/// Creates a debug object based on the input object file from
/// ObjectLinkingLayerJITLinkContext.
///
static Expected<std::unique_ptr<DebugObject>>
createDebugObjectFromBuffer(ExecutionSession &ES, LinkGraph &G,
                            JITLinkContext &Ctx, MemoryBufferRef ObjBuffer) {
  switch (G.getTargetTriple().getObjectFormat()) {
  case Triple::ELF:
    return ELFDebugObject::Create(ObjBuffer, Ctx, ES);

  default:
    // TODO: Once we add support for other formats, we might want to split this
    // into multiple files.
    return nullptr;
  }
}

DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutionSession &ES, std::unique_ptr<DebugObjectRegistrar> Target)
    : ES(ES), Target(std::move(Target)) {}

DebugObjectManagerPlugin::~DebugObjectManagerPlugin() = default;

void DebugObjectManagerPlugin::notifyMaterializing(
    MaterializationResponsibility &MR, LinkGraph &G, JITLinkContext &Ctx,
    MemoryBufferRef ObjBuffer) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  assert(PendingObjs.count(&MR) == 0 &&
         "Cannot have more than one pending debug object per "
         "MaterializationResponsibility");

  if (auto DebugObj = createDebugObjectFromBuffer(ES, G, Ctx, ObjBuffer)) {
    // Not all link artifacts allow debugging.
    if (*DebugObj != nullptr)
      PendingObjs[&MR] = std::move(*DebugObj);
  } else {
    ES.reportError(DebugObj.takeError());
  }
}

void DebugObjectManagerPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {
  // Not all link artifacts have associated debug objects.
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  if (It == PendingObjs.end())
    return;

  DebugObject &DebugObj = *It->second;
  if (DebugObj.has(Requirement::ReportFinalSectionLoadAddresses)) {
    PassConfig.PostAllocationPasses.push_back(
        [&DebugObj](LinkGraph &Graph) -> Error {
          for (const Section &GraphSection : Graph.sections())
            DebugObj.reportSectionTargetMemoryRange(GraphSection.getName(),
                                                    SectionRange(GraphSection));
          return Error::success();
        });
  }
}

Error DebugObjectManagerPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  if (It == PendingObjs.end())
    return Error::success();

  // During finalization the debug object is registered with the target.
  // Materialization must wait for this process to finish. Otherwise we might
  // start running code before the debugger processed the corresponding debug
  // info.
  std::promise<MSVCPError> FinalizePromise;
  std::future<MSVCPError> FinalizeErr = FinalizePromise.get_future();

  It->second->finalizeAsync(
      [this, &FinalizePromise, &MR](Expected<sys::MemoryBlock> TargetMem) {
        // Any failure here will fail materialization.
        if (!TargetMem) {
          FinalizePromise.set_value(TargetMem.takeError());
          return;
        }
        if (Error Err = Target->registerDebugObject(*TargetMem)) {
          FinalizePromise.set_value(std::move(Err));
          return;
        }

        // Once our tracking info is updated, notifyEmitted() can return and
        // finish materialization.
        FinalizePromise.set_value(MR.withResourceKeyDo([&](ResourceKey K) {
          assert(PendingObjs.count(&MR) && "We still hold PendingObjsLock");
          std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
          RegisteredObjs[K].push_back(std::move(PendingObjs[&MR]));
          PendingObjs.erase(&MR);
        }));
      });

  return FinalizeErr.get();
}

Error DebugObjectManagerPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

void DebugObjectManagerPlugin::notifyTransferringResources(ResourceKey DstKey,
                                                           ResourceKey SrcKey) {
  // Debug objects are stored by ResourceKey only after registration.
  // Thus, pending objects don't need to be updated here.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  auto SrcIt = RegisteredObjs.find(SrcKey);
  if (SrcIt != RegisteredObjs.end()) {
    // Resources from distinct MaterializationResponsibilitys can get merged
    // after emission, so we can have multiple debug objects per resource key.
    for (std::unique_ptr<DebugObject> &DebugObj : SrcIt->second)
      RegisteredObjs[DstKey].push_back(std::move(DebugObj));
    RegisteredObjs.erase(SrcIt);
  }
}

Error DebugObjectManagerPlugin::notifyRemovingResources(ResourceKey Key) {
  // Removing the resource for a pending object fails materialization, so they
  // get cleaned up in the notifyFailed() handler.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  RegisteredObjs.erase(Key);

  // TODO: Implement unregister notifications.
  return Error::success();
}

} // namespace orc
} // namespace llvm
