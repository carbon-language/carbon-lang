//===---- ELF_x86_64.cpp -JIT linker implementation for ELF/x86-64 ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/x86-64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_x86_64.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/TableManager.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#include "DefineExternalSectionStartAndEndSymbols.h"
#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::ELF_x86_64_Edges;

namespace {

constexpr StringRef ELFGOTSymbolName = "_GLOBAL_OFFSET_TABLE_";
constexpr StringRef ELFTLSInfoSectionName = "$__TLSINFO";

class TLSInfoTableManager_ELF_x86_64
    : public TableManager<TLSInfoTableManager_ELF_x86_64> {
public:
  static const uint8_t TLSInfoEntryContent[16];

  static StringRef getSectionName() { return ELFTLSInfoSectionName; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (E.getKind() == x86_64::RequestTLSDescInGOTAndTransformToDelta32) {
      LLVM_DEBUG({
        dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
               << formatv("{0:x}", B->getFixupAddress(E)) << " ("
               << formatv("{0:x}", B->getAddress()) << " + "
               << formatv("{0:x}", E.getOffset()) << ")\n";
      });
      E.setKind(x86_64::Delta32);
      E.setTarget(getEntryForTarget(G, E.getTarget()));
      return true;
    }
    return false;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    // the TLS Info entry's key value will be written by the fixTLVSectionByName
    // pass, so create mutable content.
    auto &TLSInfoEntry = G.createMutableContentBlock(
        getTLSInfoSection(G), G.allocateContent(getTLSInfoEntryContent()),
        orc::ExecutorAddr(), 8, 0);
    TLSInfoEntry.addEdge(x86_64::Pointer64, 8, Target, 0);
    return G.addAnonymousSymbol(TLSInfoEntry, 0, 16, false, false);
  }

private:
  Section &getTLSInfoSection(LinkGraph &G) {
    if (!TLSInfoTable)
      TLSInfoTable = &G.createSection(ELFTLSInfoSectionName, MemProt::Read);
    return *TLSInfoTable;
  }

  ArrayRef<char> getTLSInfoEntryContent() const {
    return {reinterpret_cast<const char *>(TLSInfoEntryContent),
            sizeof(TLSInfoEntryContent)};
  }

  Section *TLSInfoTable = nullptr;
};

const uint8_t TLSInfoTableManager_ELF_x86_64::TLSInfoEntryContent[16] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /*pthread key */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  /*data address*/
};

Error buildTables_ELF_x86_64(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Visiting edges in graph:\n");

  x86_64::GOTTableManager GOT;
  x86_64::PLTTableManager PLT(GOT);
  TLSInfoTableManager_ELF_x86_64 TLSInfo;
  visitExistingEdges(G, GOT, PLT, TLSInfo);
  return Error::success();
}
} // namespace

static const char *getELFX86_64RelocName(uint32_t Type) {
  switch (Type) {
#define ELF_RELOC(Name, Number)                                                \
  case Number:                                                                 \
    return #Name;
#include "llvm/BinaryFormat/ELFRelocs/x86_64.def"
#undef ELF_RELOC
  }
  return "Unrecognized ELF/x86-64 relocation type";
}

namespace llvm {
namespace jitlink {

// This should become a template as the ELFFile is so a lot of this could become
// generic
class ELFLinkGraphBuilder_x86_64 : public ELFLinkGraphBuilder<object::ELF64LE> {
private:
  using ELFT = object::ELF64LE;

  static Expected<ELF_x86_64_Edges::ELFX86RelocationKind>
  getRelocationKind(const uint32_t Type) {
    switch (Type) {
    case ELF::R_X86_64_32S:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Pointer32Signed;
    case ELF::R_X86_64_PC32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32;
    case ELF::R_X86_64_PC64:
    case ELF::R_X86_64_GOTPC64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Delta64;
    case ELF::R_X86_64_64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Pointer64;
    case ELF::R_X86_64_GOTPCREL:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32GOTLoad;
    case ELF::R_X86_64_GOTPCRELX:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32GOTLoadRelaxable;
    case ELF::R_X86_64_REX_GOTPCRELX:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32REXGOTLoadRelaxable;
    case ELF::R_X86_64_GOTPCREL64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel64GOT;
    case ELF::R_X86_64_GOT64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::GOT64;
    case ELF::R_X86_64_GOTOFF64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::GOTOFF64;
    case ELF::R_X86_64_PLT32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Branch32;
    case ELF::R_X86_64_TLSGD:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32TLV;
    }
    return make_error<JITLinkError>("Unsupported x86-64 relocation type " +
                                    formatv("{0:d}: ", Type) +
                                    getELFX86_64RelocName(Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_x86_64;
    for (const auto &RelSect : Base::Sections) {
      // Validate the section to read relocation entries from.
      if (RelSect.sh_type == ELF::SHT_REL)
        return make_error<StringError>(
            "No SHT_REL in valid x64 ELF object files",
            inconvertibleErrorCode());

      if (Error Err = Base::forEachRelocation(RelSect, this,
                                              &Self::addSingleRelocation))
        return Err;
    }

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rela &Rel,
                            const typename ELFT::Shdr &FixupSection,
                            Block &BlockToFix) {
    using Base = ELFLinkGraphBuilder<ELFT>;

    uint32_t SymbolIndex = Rel.getSymbol(false);
    auto ObjSymbol = Base::Obj.getRelocationSymbol(Rel, Base::SymTabSec);
    if (!ObjSymbol)
      return ObjSymbol.takeError();

    Symbol *GraphSymbol = Base::getGraphSymbol(SymbolIndex);
    if (!GraphSymbol)
      return make_error<StringError>(
          formatv("Could not find symbol at given index, did you add it to "
                  "JITSymbolTable? index: {0}, shndx: {1} Size of table: {2}",
                  SymbolIndex, (*ObjSymbol)->st_shndx,
                  Base::GraphSymbols.size()),
          inconvertibleErrorCode());

    // Validate the relocation kind.
    auto ELFRelocKind = getRelocationKind(Rel.getType(false));
    if (!ELFRelocKind)
      return ELFRelocKind.takeError();

    int64_t Addend = Rel.r_addend;
    Edge::Kind Kind = Edge::Invalid;
    switch (*ELFRelocKind) {
    case PCRel32:
      Kind = x86_64::Delta32;
      break;
    case Delta64:
      Kind = x86_64::Delta64;
      break;
    case Pointer32Signed:
      Kind = x86_64::Pointer32Signed;
      break;
    case Pointer64:
      Kind = x86_64::Pointer64;
      break;
    case PCRel32GOTLoad: {
      Kind = x86_64::RequestGOTAndTransformToDelta32;
      break;
    }
    case PCRel32REXGOTLoadRelaxable: {
      Kind = x86_64::RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable;
      Addend = 0;
      break;
    }
    case PCRel32TLV: {
      Kind = x86_64::RequestTLSDescInGOTAndTransformToDelta32;
      break;
    }
    case PCRel32GOTLoadRelaxable: {
      Kind = x86_64::RequestGOTAndTransformToPCRel32GOTLoadRelaxable;
      Addend = 0;
      break;
    }
    case PCRel64GOT: {
      Kind = x86_64::RequestGOTAndTransformToDelta64;
      break;
    }
    case GOT64: {
      Kind = x86_64::RequestGOTAndTransformToDelta64FromGOT;
      break;
    }
    case GOTOFF64: {
      Kind = x86_64::Delta64FromGOT;
      break;
    }
    case Branch32: {
      Kind = x86_64::BranchPCRel32;
      // BranchPCRel32 implicitly handles the '-4' PC adjustment, so we have to
      // adjust the addend by '+4' to compensate.
      Addend += 4;
      break;
    }
    }

    auto FixupAddress = orc::ExecutorAddr(FixupSection.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, x86_64::getEdgeKindName(Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_x86_64(StringRef FileName,
                             const object::ELFFile<object::ELF64LE> &Obj)
      : ELFLinkGraphBuilder(Obj, Triple("x86_64-unknown-linux"), FileName,
                            x86_64::getEdgeKindName) {}
};

class ELFJITLinker_x86_64 : public JITLinker<ELFJITLinker_x86_64> {
  friend class JITLinker<ELFJITLinker_x86_64>;

public:
  ELFJITLinker_x86_64(std::unique_ptr<JITLinkContext> Ctx,
                      std::unique_ptr<LinkGraph> G,
                      PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    getPassConfig().PostAllocationPasses.push_back(
        [this](LinkGraph &G) { return getOrCreateGOTSymbol(G); });
  }

private:
  Symbol *GOTSymbol = nullptr;

  Error getOrCreateGOTSymbol(LinkGraph &G) {
    auto DefineExternalGOTSymbolIfPresent =
        createDefineExternalSectionStartAndEndSymbolsPass(
            [&](LinkGraph &LG, Symbol &Sym) -> SectionRangeSymbolDesc {
              if (Sym.getName() == ELFGOTSymbolName)
                if (auto *GOTSection = G.findSectionByName(
                        x86_64::GOTTableManager::getSectionName())) {
                  GOTSymbol = &Sym;
                  return {*GOTSection, true};
                }
              return {};
            });

    // Try to attach _GLOBAL_OFFSET_TABLE_ to the GOT if it's defined as an
    // external.
    if (auto Err = DefineExternalGOTSymbolIfPresent(G))
      return Err;

    // If we succeeded then we're done.
    if (GOTSymbol)
      return Error::success();

    // Otherwise look for a GOT section: If it already has a start symbol we'll
    // record it, otherwise we'll create our own.
    // If there's a GOT section but we didn't find an external GOT symbol...
    if (auto *GOTSection =
            G.findSectionByName(x86_64::GOTTableManager::getSectionName())) {

      // Check for an existing defined symbol.
      for (auto *Sym : GOTSection->symbols())
        if (Sym->getName() == ELFGOTSymbolName) {
          GOTSymbol = Sym;
          return Error::success();
        }

      // If there's no defined symbol then create one.
      SectionRange SR(*GOTSection);
      if (SR.empty())
        GOTSymbol =
            &G.addAbsoluteSymbol(ELFGOTSymbolName, orc::ExecutorAddr(), 0,
                                 Linkage::Strong, Scope::Local, true);
      else
        GOTSymbol =
            &G.addDefinedSymbol(*SR.getFirstBlock(), 0, ELFGOTSymbolName, 0,
                                Linkage::Strong, Scope::Local, false, true);
    }

    return Error::success();
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return x86_64::applyFixup(G, B, E, GOTSymbol);
  }
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_x86_64(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
  return ELFLinkGraphBuilder_x86_64((*ELFObj)->getFileName(),
                                    ELFObjFile.getELFFile())
      .buildGraph();
}

static SectionRangeSymbolDesc
identifyELFSectionStartAndEndSymbols(LinkGraph &G, Symbol &Sym) {
  constexpr StringRef StartSymbolPrefix = "__start";
  constexpr StringRef EndSymbolPrefix = "__end";

  auto SymName = Sym.getName();
  if (SymName.startswith(StartSymbolPrefix)) {
    if (auto *Sec =
            G.findSectionByName(SymName.drop_front(StartSymbolPrefix.size())))
      return {*Sec, true};
  } else if (SymName.startswith(EndSymbolPrefix)) {
    if (auto *Sec =
            G.findSectionByName(SymName.drop_front(EndSymbolPrefix.size())))
      return {*Sec, false};
  }
  return {};
}

void link_ELF_x86_64(std::unique_ptr<LinkGraph> G,
                     std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;

  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {

    Config.PrePrunePasses.push_back(EHFrameSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer(".eh_frame", x86_64::PointerSize, x86_64::Delta64,
                         x86_64::Delta32, x86_64::NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));

    // Construct a JITLinker and run the link function.
    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT/Stubs/TLSInfoEntry build pass.
    Config.PostPrunePasses.push_back(buildTables_ELF_x86_64);

    // Resolve any external section start / end symbols.
    Config.PostAllocationPasses.push_back(
        createDefineExternalSectionStartAndEndSymbolsPass(
            identifyELFSectionStartAndEndSymbols));

    // Add GOT/Stubs optimizer pass.
    Config.PreFixupPasses.push_back(x86_64::optimizeGOTAndStubAccesses);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_x86_64::link(std::move(Ctx), std::move(G), std::move(Config));
}
const char *getELFX86RelocationKindName(Edge::Kind R) {
  switch (R) {
  case Branch32:
    return "Branch32";
  case Pointer32Signed:
    return "Pointer32Signed";
  case Pointer64:
    return "Pointer64";
  case PCRel32:
    return "PCRel32";
  case PCRel32GOTLoad:
    return "PCRel32GOTLoad";
  case PCRel32GOTLoadRelaxable:
    return "PCRel32GOTLoadRelaxable";
  case PCRel32REXGOTLoadRelaxable:
    return "PCRel32REXGOTLoad";
  case PCRel64GOT:
    return "PCRel64GOT";
  case Delta64:
    return "Delta64";
  case GOT64:
    return "GOT64";
  case GOTOFF64:
    return "GOTOFF64";
  }
  return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
}
} // end namespace jitlink
} // end namespace llvm
