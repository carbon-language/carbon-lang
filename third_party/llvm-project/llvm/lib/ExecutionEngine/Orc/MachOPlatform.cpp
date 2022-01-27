//===------ MachOPlatform.cpp - Utilities for executing MachO in Orc ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class MachOHeaderMaterializationUnit : public MaterializationUnit {
public:
  MachOHeaderMaterializationUnit(MachOPlatform &MOP,
                                 const SymbolStringPtr &HeaderStartSymbol)
      : MaterializationUnit(createHeaderInterface(MOP, HeaderStartSymbol)),
        MOP(MOP) {}

  StringRef getName() const override { return "MachOHeaderMU"; }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    unsigned PointerSize;
    support::endianness Endianness;
    const auto &TT =
        MOP.getExecutionSession().getExecutorProcessControl().getTargetTriple();

    switch (TT.getArch()) {
    case Triple::aarch64:
    case Triple::x86_64:
      PointerSize = 8;
      Endianness = support::endianness::little;
      break;
    default:
      llvm_unreachable("Unrecognized architecture");
    }

    auto G = std::make_unique<jitlink::LinkGraph>(
        "<MachOHeaderMU>", TT, PointerSize, Endianness,
        jitlink::getGenericEdgeKindName);
    auto &HeaderSection = G->createSection("__header", jitlink::MemProt::Read);
    auto &HeaderBlock = createHeaderBlock(*G, HeaderSection);

    // Init symbol is header-start symbol.
    G->addDefinedSymbol(HeaderBlock, 0, *R->getInitializerSymbol(),
                        HeaderBlock.getSize(), jitlink::Linkage::Strong,
                        jitlink::Scope::Default, false, true);
    for (auto &HS : AdditionalHeaderSymbols)
      G->addDefinedSymbol(HeaderBlock, HS.Offset, HS.Name,
                          HeaderBlock.getSize(), jitlink::Linkage::Strong,
                          jitlink::Scope::Default, false, true);

    MOP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

private:
  struct HeaderSymbol {
    const char *Name;
    uint64_t Offset;
  };

  static constexpr HeaderSymbol AdditionalHeaderSymbols[] = {
      {"___mh_executable_header", 0}};

  static jitlink::Block &createHeaderBlock(jitlink::LinkGraph &G,
                                           jitlink::Section &HeaderSection) {
    MachO::mach_header_64 Hdr;
    Hdr.magic = MachO::MH_MAGIC_64;
    switch (G.getTargetTriple().getArch()) {
    case Triple::aarch64:
      Hdr.cputype = MachO::CPU_TYPE_ARM64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_ARM64_ALL;
      break;
    case Triple::x86_64:
      Hdr.cputype = MachO::CPU_TYPE_X86_64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_X86_64_ALL;
      break;
    default:
      llvm_unreachable("Unrecognized architecture");
    }
    Hdr.filetype = MachO::MH_DYLIB; // Custom file type?
    Hdr.ncmds = 0;
    Hdr.sizeofcmds = 0;
    Hdr.flags = 0;
    Hdr.reserved = 0;

    if (G.getEndianness() != support::endian::system_endianness())
      MachO::swapStruct(Hdr);

    auto HeaderContent = G.allocateString(
        StringRef(reinterpret_cast<const char *>(&Hdr), sizeof(Hdr)));

    return G.createContentBlock(HeaderSection, HeaderContent, ExecutorAddr(), 8,
                                0);
  }

  static MaterializationUnit::Interface
  createHeaderInterface(MachOPlatform &MOP,
                        const SymbolStringPtr &HeaderStartSymbol) {
    SymbolFlagsMap HeaderSymbolFlags;

    HeaderSymbolFlags[HeaderStartSymbol] = JITSymbolFlags::Exported;
    for (auto &HS : AdditionalHeaderSymbols)
      HeaderSymbolFlags[MOP.getExecutionSession().intern(HS.Name)] =
          JITSymbolFlags::Exported;

    return MaterializationUnit::Interface(std::move(HeaderSymbolFlags),
                                          HeaderStartSymbol);
  }

  MachOPlatform &MOP;
};

constexpr MachOHeaderMaterializationUnit::HeaderSymbol
    MachOHeaderMaterializationUnit::AdditionalHeaderSymbols[];

StringRef EHFrameSectionName = "__TEXT,__eh_frame";
StringRef ModInitFuncSectionName = "__DATA,__mod_init_func";
StringRef ObjCClassListSectionName = "__DATA,__objc_classlist";
StringRef ObjCImageInfoSectionName = "__DATA,__objc_image_info";
StringRef ObjCSelRefsSectionName = "__DATA,__objc_selrefs";
StringRef Swift5ProtoSectionName = "__TEXT,__swift5_proto";
StringRef Swift5ProtosSectionName = "__TEXT,__swift5_protos";
StringRef Swift5TypesSectionName = "__TEXT,__swift5_types";
StringRef ThreadBSSSectionName = "__DATA,__thread_bss";
StringRef ThreadDataSectionName = "__DATA,__thread_data";
StringRef ThreadVarsSectionName = "__DATA,__thread_vars";

StringRef InitSectionNames[] = {
    ModInitFuncSectionName, ObjCSelRefsSectionName, ObjCClassListSectionName,
    Swift5ProtosSectionName, Swift5ProtoSectionName, Swift5TypesSectionName};

} // end anonymous namespace

namespace llvm {
namespace orc {

Expected<std::unique_ptr<MachOPlatform>>
MachOPlatform::Create(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
                      JITDylib &PlatformJD, const char *OrcRuntimePath,
                      Optional<SymbolAliasMap> RuntimeAliases) {

  auto &EPC = ES.getExecutorProcessControl();

  // If the target is not supported then bail out immediately.
  if (!supportedTarget(EPC.getTargetTriple()))
    return make_error<StringError>("Unsupported MachOPlatform triple: " +
                                       EPC.getTargetTriple().str(),
                                   inconvertibleErrorCode());

  // Create default aliases if the caller didn't supply any.
  if (!RuntimeAliases)
    RuntimeAliases = standardPlatformAliases(ES);

  // Define the aliases.
  if (auto Err = PlatformJD.define(symbolAliases(std::move(*RuntimeAliases))))
    return std::move(Err);

  // Add JIT-dispatch function support symbols.
  if (auto Err = PlatformJD.define(absoluteSymbols(
          {{ES.intern("___orc_rt_jit_dispatch"),
            {EPC.getJITDispatchInfo().JITDispatchFunction.getValue(),
             JITSymbolFlags::Exported}},
           {ES.intern("___orc_rt_jit_dispatch_ctx"),
            {EPC.getJITDispatchInfo().JITDispatchContext.getValue(),
             JITSymbolFlags::Exported}}})))
    return std::move(Err);

  // Create a generator for the ORC runtime archive.
  auto OrcRuntimeArchiveGenerator = StaticLibraryDefinitionGenerator::Load(
      ObjLinkingLayer, OrcRuntimePath, EPC.getTargetTriple());
  if (!OrcRuntimeArchiveGenerator)
    return OrcRuntimeArchiveGenerator.takeError();

  // Create the instance.
  Error Err = Error::success();
  auto P = std::unique_ptr<MachOPlatform>(
      new MachOPlatform(ES, ObjLinkingLayer, PlatformJD,
                        std::move(*OrcRuntimeArchiveGenerator), Err));
  if (Err)
    return std::move(Err);
  return std::move(P);
}

Error MachOPlatform::setupJITDylib(JITDylib &JD) {
  return JD.define(std::make_unique<MachOHeaderMaterializationUnit>(
      *this, MachOHeaderStartSymbol));
}

Error MachOPlatform::notifyAdding(ResourceTracker &RT,
                                  const MaterializationUnit &MU) {
  auto &JD = RT.getJITDylib();
  const auto &InitSym = MU.getInitializerSymbol();
  if (!InitSym)
    return Error::success();

  RegisteredInitSymbols[&JD].add(InitSym,
                                 SymbolLookupFlags::WeaklyReferencedSymbol);
  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Registered init symbol " << *InitSym << " for MU "
           << MU.getName() << "\n";
  });
  return Error::success();
}

Error MachOPlatform::notifyRemoving(ResourceTracker &RT) {
  llvm_unreachable("Not supported yet");
}

static void addAliases(ExecutionSession &ES, SymbolAliasMap &Aliases,
                       ArrayRef<std::pair<const char *, const char *>> AL) {
  for (auto &KV : AL) {
    auto AliasName = ES.intern(KV.first);
    assert(!Aliases.count(AliasName) && "Duplicate symbol name in alias map");
    Aliases[std::move(AliasName)] = {ES.intern(KV.second),
                                     JITSymbolFlags::Exported};
  }
}

SymbolAliasMap MachOPlatform::standardPlatformAliases(ExecutionSession &ES) {
  SymbolAliasMap Aliases;
  addAliases(ES, Aliases, requiredCXXAliases());
  addAliases(ES, Aliases, standardRuntimeUtilityAliases());
  return Aliases;
}

ArrayRef<std::pair<const char *, const char *>>
MachOPlatform::requiredCXXAliases() {
  static const std::pair<const char *, const char *> RequiredCXXAliases[] = {
      {"___cxa_atexit", "___orc_rt_macho_cxa_atexit"}};

  return ArrayRef<std::pair<const char *, const char *>>(RequiredCXXAliases);
}

ArrayRef<std::pair<const char *, const char *>>
MachOPlatform::standardRuntimeUtilityAliases() {
  static const std::pair<const char *, const char *>
      StandardRuntimeUtilityAliases[] = {
          {"___orc_rt_run_program", "___orc_rt_macho_run_program"},
          {"___orc_rt_log_error", "___orc_rt_log_error_to_stderr"}};

  return ArrayRef<std::pair<const char *, const char *>>(
      StandardRuntimeUtilityAliases);
}

bool MachOPlatform::isInitializerSection(StringRef SegName,
                                         StringRef SectName) {
  for (auto &Name : InitSectionNames) {
    if (Name.startswith(SegName) && Name.substr(7) == SectName)
      return true;
  }
  return false;
}

bool MachOPlatform::supportedTarget(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::aarch64:
  case Triple::x86_64:
    return true;
  default:
    return false;
  }
}

MachOPlatform::MachOPlatform(
    ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
    JITDylib &PlatformJD,
    std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator, Error &Err)
    : ES(ES), ObjLinkingLayer(ObjLinkingLayer),
      MachOHeaderStartSymbol(ES.intern("___dso_handle")) {
  ErrorAsOutParameter _(&Err);

  ObjLinkingLayer.addPlugin(std::make_unique<MachOPlatformPlugin>(*this));

  PlatformJD.addGenerator(std::move(OrcRuntimeGenerator));

  // Force linking of eh-frame registration functions.
  if (auto Err2 = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder(&PlatformJD),
          {{ES.intern("___orc_rt_macho_register_ehframe_section"),
            &orc_rt_macho_register_ehframe_section},
           {ES.intern("___orc_rt_macho_deregister_ehframe_section"),
            &orc_rt_macho_deregister_ehframe_section}})) {
    Err = std::move(Err2);
    return;
  }

  State = BootstrapPhase2;

  // PlatformJD hasn't been 'set-up' by the platform yet (since we're creating
  // the platform now), so set it up.
  if (auto E2 = setupJITDylib(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  RegisteredInitSymbols[&PlatformJD].add(
      MachOHeaderStartSymbol, SymbolLookupFlags::WeaklyReferencedSymbol);

  // Associate wrapper function tags with JIT-side function implementations.
  if (auto E2 = associateRuntimeSupportFunctions(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  // Lookup addresses of runtime functions callable by the platform,
  // call the platform bootstrap function to initialize the platform-state
  // object in the executor.
  if (auto E2 = bootstrapMachORuntime(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  State = Initialized;
}

Error MachOPlatform::associateRuntimeSupportFunctions(JITDylib &PlatformJD) {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;

  using GetInitializersSPSSig =
      SPSExpected<SPSMachOJITDylibInitializerSequence>(SPSString);
  WFs[ES.intern("___orc_rt_macho_get_initializers_tag")] =
      ES.wrapAsyncWithSPS<GetInitializersSPSSig>(
          this, &MachOPlatform::rt_getInitializers);

  using GetDeinitializersSPSSig =
      SPSExpected<SPSMachOJITDylibDeinitializerSequence>(SPSExecutorAddr);
  WFs[ES.intern("___orc_rt_macho_get_deinitializers_tag")] =
      ES.wrapAsyncWithSPS<GetDeinitializersSPSSig>(
          this, &MachOPlatform::rt_getDeinitializers);

  using LookupSymbolSPSSig =
      SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
  WFs[ES.intern("___orc_rt_macho_symbol_lookup_tag")] =
      ES.wrapAsyncWithSPS<LookupSymbolSPSSig>(this,
                                              &MachOPlatform::rt_lookupSymbol);

  return ES.registerJITDispatchHandlers(PlatformJD, std::move(WFs));
}

void MachOPlatform::getInitializersBuildSequencePhase(
    SendInitializerSequenceFn SendResult, JITDylib &JD,
    std::vector<JITDylibSP> DFSLinkOrder) {
  MachOJITDylibInitializerSequence FullInitSeq;
  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    for (auto &InitJD : reverse(DFSLinkOrder)) {
      LLVM_DEBUG({
        dbgs() << "MachOPlatform: Appending inits for \"" << InitJD->getName()
               << "\" to sequence\n";
      });
      auto ISItr = InitSeqs.find(InitJD.get());
      if (ISItr != InitSeqs.end()) {
        FullInitSeq.emplace_back(std::move(ISItr->second));
        InitSeqs.erase(ISItr);
      }
    }
  }

  SendResult(std::move(FullInitSeq));
}

void MachOPlatform::getInitializersLookupPhase(
    SendInitializerSequenceFn SendResult, JITDylib &JD) {

  auto DFSLinkOrder = JD.getDFSLinkOrder();
  DenseMap<JITDylib *, SymbolLookupSet> NewInitSymbols;
  ES.runSessionLocked([&]() {
    for (auto &InitJD : DFSLinkOrder) {
      auto RISItr = RegisteredInitSymbols.find(InitJD.get());
      if (RISItr != RegisteredInitSymbols.end()) {
        NewInitSymbols[InitJD.get()] = std::move(RISItr->second);
        RegisteredInitSymbols.erase(RISItr);
      }
    }
  });

  // If there are no further init symbols to look up then move on to the next
  // phase.
  if (NewInitSymbols.empty()) {
    getInitializersBuildSequencePhase(std::move(SendResult), JD,
                                      std::move(DFSLinkOrder));
    return;
  }

  // Otherwise issue a lookup and re-run this phase when it completes.
  lookupInitSymbolsAsync(
      [this, SendResult = std::move(SendResult), &JD](Error Err) mutable {
        if (Err)
          SendResult(std::move(Err));
        else
          getInitializersLookupPhase(std::move(SendResult), JD);
      },
      ES, std::move(NewInitSymbols));
}

void MachOPlatform::rt_getInitializers(SendInitializerSequenceFn SendResult,
                                       StringRef JDName) {
  LLVM_DEBUG({
    dbgs() << "MachOPlatform::rt_getInitializers(\"" << JDName << "\")\n";
  });

  JITDylib *JD = ES.getJITDylibByName(JDName);
  if (!JD) {
    LLVM_DEBUG({
      dbgs() << "  No such JITDylib \"" << JDName << "\". Sending error.\n";
    });
    SendResult(make_error<StringError>("No JITDylib named " + JDName,
                                       inconvertibleErrorCode()));
    return;
  }

  getInitializersLookupPhase(std::move(SendResult), *JD);
}

void MachOPlatform::rt_getDeinitializers(SendDeinitializerSequenceFn SendResult,
                                         ExecutorAddr Handle) {
  LLVM_DEBUG({
    dbgs() << "MachOPlatform::rt_getDeinitializers(\""
           << formatv("{0:x}", Handle.getValue()) << "\")\n";
  });

  JITDylib *JD = nullptr;

  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HeaderAddrToJITDylib.find(Handle);
    if (I != HeaderAddrToJITDylib.end())
      JD = I->second;
  }

  if (!JD) {
    LLVM_DEBUG({
      dbgs() << "  No JITDylib for handle "
             << formatv("{0:x}", Handle.getValue()) << "\n";
    });
    SendResult(make_error<StringError>("No JITDylib associated with handle " +
                                           formatv("{0:x}", Handle.getValue()),
                                       inconvertibleErrorCode()));
    return;
  }

  SendResult(MachOJITDylibDeinitializerSequence());
}

void MachOPlatform::rt_lookupSymbol(SendSymbolAddressFn SendResult,
                                    ExecutorAddr Handle, StringRef SymbolName) {
  LLVM_DEBUG({
    dbgs() << "MachOPlatform::rt_lookupSymbol(\""
           << formatv("{0:x}", Handle.getValue()) << "\")\n";
  });

  JITDylib *JD = nullptr;

  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HeaderAddrToJITDylib.find(Handle);
    if (I != HeaderAddrToJITDylib.end())
      JD = I->second;
  }

  if (!JD) {
    LLVM_DEBUG({
      dbgs() << "  No JITDylib for handle "
             << formatv("{0:x}", Handle.getValue()) << "\n";
    });
    SendResult(make_error<StringError>("No JITDylib associated with handle " +
                                           formatv("{0:x}", Handle.getValue()),
                                       inconvertibleErrorCode()));
    return;
  }

  // Use functor class to work around XL build compiler issue on AIX.
  class RtLookupNotifyComplete {
  public:
    RtLookupNotifyComplete(SendSymbolAddressFn &&SendResult)
        : SendResult(std::move(SendResult)) {}
    void operator()(Expected<SymbolMap> Result) {
      if (Result) {
        assert(Result->size() == 1 && "Unexpected result map count");
        SendResult(ExecutorAddr(Result->begin()->second.getAddress()));
      } else {
        SendResult(Result.takeError());
      }
    }

  private:
    SendSymbolAddressFn SendResult;
  };

  // FIXME: Proper mangling.
  auto MangledName = ("_" + SymbolName).str();
  ES.lookup(
      LookupKind::DLSym, {{JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(ES.intern(MangledName)), SymbolState::Ready,
      RtLookupNotifyComplete(std::move(SendResult)), NoDependenciesToRegister);
}

Error MachOPlatform::bootstrapMachORuntime(JITDylib &PlatformJD) {
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder(&PlatformJD),
          {{ES.intern("___orc_rt_macho_platform_bootstrap"),
            &orc_rt_macho_platform_bootstrap},
           {ES.intern("___orc_rt_macho_platform_shutdown"),
            &orc_rt_macho_platform_shutdown},
           {ES.intern("___orc_rt_macho_register_thread_data_section"),
            &orc_rt_macho_register_thread_data_section},
           {ES.intern("___orc_rt_macho_deregister_thread_data_section"),
            &orc_rt_macho_deregister_thread_data_section},
           {ES.intern("___orc_rt_macho_create_pthread_key"),
            &orc_rt_macho_create_pthread_key}}))
    return Err;

  return ES.callSPSWrapper<void()>(orc_rt_macho_platform_bootstrap);
}

Error MachOPlatform::registerInitInfo(
    JITDylib &JD, ExecutorAddr ObjCImageInfoAddr,
    ArrayRef<jitlink::Section *> InitSections) {

  std::unique_lock<std::mutex> Lock(PlatformMutex);

  MachOJITDylibInitializers *InitSeq = nullptr;
  {
    auto I = InitSeqs.find(&JD);
    if (I == InitSeqs.end()) {
      // If there's no init sequence entry yet then we need to look up the
      // header symbol to force creation of one.
      Lock.unlock();

      auto SearchOrder =
          JD.withLinkOrderDo([](const JITDylibSearchOrder &SO) { return SO; });
      if (auto Err = ES.lookup(SearchOrder, MachOHeaderStartSymbol).takeError())
        return Err;

      Lock.lock();
      I = InitSeqs.find(&JD);
      assert(I != InitSeqs.end() &&
             "Entry missing after header symbol lookup?");
    }
    InitSeq = &I->second;
  }

  InitSeq->ObjCImageInfoAddress = ObjCImageInfoAddr;

  for (auto *Sec : InitSections) {
    // FIXME: Avoid copy here.
    jitlink::SectionRange R(*Sec);
    InitSeq->InitSections[Sec->getName()].push_back(
        {ExecutorAddr(R.getStart()), ExecutorAddr(R.getEnd())});
  }

  return Error::success();
}

Expected<uint64_t> MachOPlatform::createPThreadKey() {
  if (!orc_rt_macho_create_pthread_key)
    return make_error<StringError>(
        "Attempting to create pthread key in target, but runtime support has "
        "not been loaded yet",
        inconvertibleErrorCode());

  Expected<uint64_t> Result(0);
  if (auto Err = ES.callSPSWrapper<SPSExpected<uint64_t>(void)>(
          orc_rt_macho_create_pthread_key, Result))
    return std::move(Err);
  return Result;
}

void MachOPlatform::MachOPlatformPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, jitlink::LinkGraph &LG,
    jitlink::PassConfiguration &Config) {

  auto PS = MP.State.load();

  // --- Handle Initializers ---
  if (auto InitSymbol = MR.getInitializerSymbol()) {

    // If the initializer symbol is the MachOHeader start symbol then just
    // register it and then bail out -- the header materialization unit
    // definitely doesn't need any other passes.
    if (InitSymbol == MP.MachOHeaderStartSymbol) {
      Config.PostAllocationPasses.push_back([this, &MR](jitlink::LinkGraph &G) {
        return associateJITDylibHeaderSymbol(G, MR);
      });
      return;
    }

    // If the object contains an init symbol other than the header start symbol
    // then add passes to preserve, process and register the init
    // sections/symbols.
    Config.PrePrunePasses.push_back([this, &MR](jitlink::LinkGraph &G) {
      if (auto Err = preserveInitSections(G, MR))
        return Err;
      return processObjCImageInfo(G, MR);
    });

    Config.PostFixupPasses.push_back(
        [this, &JD = MR.getTargetJITDylib()](jitlink::LinkGraph &G) {
          return registerInitSections(G, JD);
        });
  }

  // --- Add passes for eh-frame and TLV support ---
  if (PS == MachOPlatform::BootstrapPhase1) {
    Config.PostFixupPasses.push_back(
        [this](jitlink::LinkGraph &G) { return registerEHSectionsPhase1(G); });
    return;
  }

  // Insert TLV lowering at the start of the PostPrunePasses, since we want
  // it to run before GOT/PLT lowering.
  Config.PostPrunePasses.insert(
      Config.PostPrunePasses.begin(),
      [this, &JD = MR.getTargetJITDylib()](jitlink::LinkGraph &G) {
        return fixTLVSectionsAndEdges(G, JD);
      });

  // Add a pass to register the final addresses of the eh-frame and TLV sections
  // with the runtime.
  Config.PostFixupPasses.push_back(
      [this](jitlink::LinkGraph &G) { return registerEHAndTLVSections(G); });
}

ObjectLinkingLayer::Plugin::SyntheticSymbolDependenciesMap
MachOPlatform::MachOPlatformPlugin::getSyntheticSymbolDependencies(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PluginMutex);
  auto I = InitSymbolDeps.find(&MR);
  if (I != InitSymbolDeps.end()) {
    SyntheticSymbolDependenciesMap Result;
    Result[MR.getInitializerSymbol()] = std::move(I->second);
    InitSymbolDeps.erase(&MR);
    return Result;
  }
  return SyntheticSymbolDependenciesMap();
}

Error MachOPlatform::MachOPlatformPlugin::associateJITDylibHeaderSymbol(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  auto I = llvm::find_if(G.defined_symbols(), [this](jitlink::Symbol *Sym) {
    return Sym->getName() == *MP.MachOHeaderStartSymbol;
  });
  assert(I != G.defined_symbols().end() && "Missing MachO header start symbol");

  auto &JD = MR.getTargetJITDylib();
  std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
  auto HeaderAddr = (*I)->getAddress();
  MP.HeaderAddrToJITDylib[HeaderAddr] = &JD;
  assert(!MP.InitSeqs.count(&JD) && "InitSeq entry for JD already exists");
  MP.InitSeqs.insert(
      std::make_pair(&JD, MachOJITDylibInitializers(JD.getName(), HeaderAddr)));
  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::preserveInitSections(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  JITLinkSymbolSet InitSectionSymbols;
  for (auto &InitSectionName : InitSectionNames) {
    // Skip non-init sections.
    auto *InitSection = G.findSectionByName(InitSectionName);
    if (!InitSection)
      continue;

    // Make a pass over live symbols in the section: those blocks are already
    // preserved.
    DenseSet<jitlink::Block *> AlreadyLiveBlocks;
    for (auto &Sym : InitSection->symbols()) {
      auto &B = Sym->getBlock();
      if (Sym->isLive() && Sym->getOffset() == 0 &&
          Sym->getSize() == B.getSize() && !AlreadyLiveBlocks.count(&B)) {
        InitSectionSymbols.insert(Sym);
        AlreadyLiveBlocks.insert(&B);
      }
    }

    // Add anonymous symbols to preserve any not-already-preserved blocks.
    for (auto *B : InitSection->blocks())
      if (!AlreadyLiveBlocks.count(B))
        InitSectionSymbols.insert(
            &G.addAnonymousSymbol(*B, 0, B->getSize(), false, true));
  }

  if (!InitSectionSymbols.empty()) {
    std::lock_guard<std::mutex> Lock(PluginMutex);
    InitSymbolDeps[&MR] = std::move(InitSectionSymbols);
  }

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::processObjCImageInfo(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  // If there's an ObjC imagine info then either
  //   (1) It's the first __objc_imageinfo we've seen in this JITDylib. In
  //       this case we name and record it.
  // OR
  //   (2) We already have a recorded __objc_imageinfo for this JITDylib,
  //       in which case we just verify it.
  auto *ObjCImageInfo = G.findSectionByName(ObjCImageInfoSectionName);
  if (!ObjCImageInfo)
    return Error::success();

  auto ObjCImageInfoBlocks = ObjCImageInfo->blocks();

  // Check that the section is not empty if present.
  if (llvm::empty(ObjCImageInfoBlocks))
    return make_error<StringError>("Empty " + ObjCImageInfoSectionName +
                                       " section in " + G.getName(),
                                   inconvertibleErrorCode());

  // Check that there's only one block in the section.
  if (std::next(ObjCImageInfoBlocks.begin()) != ObjCImageInfoBlocks.end())
    return make_error<StringError>("Multiple blocks in " +
                                       ObjCImageInfoSectionName +
                                       " section in " + G.getName(),
                                   inconvertibleErrorCode());

  // Check that the __objc_imageinfo section is unreferenced.
  // FIXME: We could optimize this check if Symbols had a ref-count.
  for (auto &Sec : G.sections()) {
    if (&Sec != ObjCImageInfo)
      for (auto *B : Sec.blocks())
        for (auto &E : B->edges())
          if (E.getTarget().isDefined() &&
              &E.getTarget().getBlock().getSection() == ObjCImageInfo)
            return make_error<StringError>(ObjCImageInfoSectionName +
                                               " is referenced within file " +
                                               G.getName(),
                                           inconvertibleErrorCode());
  }

  auto &ObjCImageInfoBlock = **ObjCImageInfoBlocks.begin();
  auto *ObjCImageInfoData = ObjCImageInfoBlock.getContent().data();
  auto Version = support::endian::read32(ObjCImageInfoData, G.getEndianness());
  auto Flags =
      support::endian::read32(ObjCImageInfoData + 4, G.getEndianness());

  // Lock the mutex while we verify / update the ObjCImageInfos map.
  std::lock_guard<std::mutex> Lock(PluginMutex);

  auto ObjCImageInfoItr = ObjCImageInfos.find(&MR.getTargetJITDylib());
  if (ObjCImageInfoItr != ObjCImageInfos.end()) {
    // We've already registered an __objc_imageinfo section. Verify the
    // content of this new section matches, then delete it.
    if (ObjCImageInfoItr->second.first != Version)
      return make_error<StringError>(
          "ObjC version in " + G.getName() +
              " does not match first registered version",
          inconvertibleErrorCode());
    if (ObjCImageInfoItr->second.second != Flags)
      return make_error<StringError>("ObjC flags in " + G.getName() +
                                         " do not match first registered flags",
                                     inconvertibleErrorCode());

    // __objc_imageinfo is valid. Delete the block.
    for (auto *S : ObjCImageInfo->symbols())
      G.removeDefinedSymbol(*S);
    G.removeBlock(ObjCImageInfoBlock);
  } else {
    // We haven't registered an __objc_imageinfo section yet. Register and
    // move on. The section should already be marked no-dead-strip.
    ObjCImageInfos[&MR.getTargetJITDylib()] = std::make_pair(Version, Flags);
  }

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::registerInitSections(
    jitlink::LinkGraph &G, JITDylib &JD) {

  ExecutorAddr ObjCImageInfoAddr;
  SmallVector<jitlink::Section *> InitSections;

  if (auto *ObjCImageInfoSec = G.findSectionByName(ObjCImageInfoSectionName)) {
    if (auto Addr = jitlink::SectionRange(*ObjCImageInfoSec).getStart())
      ObjCImageInfoAddr = Addr;
  }

  for (auto InitSectionName : InitSectionNames)
    if (auto *Sec = G.findSectionByName(InitSectionName))
      InitSections.push_back(Sec);

  // Dump the scraped inits.
  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Scraped " << G.getName() << " init sections:\n";
    if (ObjCImageInfoAddr)
      dbgs() << "  " << ObjCImageInfoSectionName << ": "
             << formatv("{0:x}", ObjCImageInfoAddr.getValue()) << "\n";
    for (auto *Sec : InitSections) {
      jitlink::SectionRange R(*Sec);
      dbgs() << "  " << Sec->getName() << ": "
             << formatv("[ {0:x} -- {1:x} ]", R.getStart(), R.getEnd()) << "\n";
    }
  });

  return MP.registerInitInfo(JD, ObjCImageInfoAddr, InitSections);
}

Error MachOPlatform::MachOPlatformPlugin::fixTLVSectionsAndEdges(
    jitlink::LinkGraph &G, JITDylib &JD) {

  // Rename external references to __tlv_bootstrap to ___orc_rt_tlv_get_addr.
  for (auto *Sym : G.external_symbols())
    if (Sym->getName() == "__tlv_bootstrap") {
      Sym->setName("___orc_rt_macho_tlv_get_addr");
      break;
    }

  // Store key in __thread_vars struct fields.
  if (auto *ThreadDataSec = G.findSectionByName(ThreadVarsSectionName)) {
    Optional<uint64_t> Key;
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto I = MP.JITDylibToPThreadKey.find(&JD);
      if (I != MP.JITDylibToPThreadKey.end())
        Key = I->second;
    }

    if (!Key) {
      if (auto KeyOrErr = MP.createPThreadKey())
        Key = *KeyOrErr;
      else
        return KeyOrErr.takeError();
    }

    uint64_t PlatformKeyBits =
        support::endian::byte_swap(*Key, G.getEndianness());

    for (auto *B : ThreadDataSec->blocks()) {
      if (B->getSize() != 3 * G.getPointerSize())
        return make_error<StringError>("__thread_vars block at " +
                                           formatv("{0:x}", B->getAddress()) +
                                           " has unexpected size",
                                       inconvertibleErrorCode());

      auto NewBlockContent = G.allocateBuffer(B->getSize());
      llvm::copy(B->getContent(), NewBlockContent.data());
      memcpy(NewBlockContent.data() + G.getPointerSize(), &PlatformKeyBits,
             G.getPointerSize());
      B->setContent(NewBlockContent);
    }
  }

  // Transform any TLV edges into GOT edges.
  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() ==
          jitlink::x86_64::RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable)
        E.setKind(jitlink::x86_64::
                      RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable);

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::registerEHAndTLVSections(
    jitlink::LinkGraph &G) {

  // Add a pass to register the final addresses of the eh-frame and TLV sections
  // with the runtime.
  if (auto *EHFrameSection = G.findSectionByName(EHFrameSectionName)) {
    jitlink::SectionRange R(*EHFrameSection);
    if (!R.empty())
      G.allocActions().push_back(
          {cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   MP.orc_rt_macho_register_ehframe_section, R.getRange())),
           cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   MP.orc_rt_macho_deregister_ehframe_section, R.getRange()))});
  }

  // Get a pointer to the thread data section if there is one. It will be used
  // below.
  jitlink::Section *ThreadDataSection =
      G.findSectionByName(ThreadDataSectionName);

  // Handle thread BSS section if there is one.
  if (auto *ThreadBSSSection = G.findSectionByName(ThreadBSSSectionName)) {
    // If there's already a thread data section in this graph then merge the
    // thread BSS section content into it, otherwise just treat the thread
    // BSS section as the thread data section.
    if (ThreadDataSection)
      G.mergeSections(*ThreadDataSection, *ThreadBSSSection);
    else
      ThreadDataSection = ThreadBSSSection;
  }

  // Having merged thread BSS (if present) and thread data (if present),
  // record the resulting section range.
  if (ThreadDataSection) {
    jitlink::SectionRange R(*ThreadDataSection);
    if (!R.empty()) {
      if (MP.State != MachOPlatform::Initialized)
        return make_error<StringError>("__thread_data section encountered, but "
                                       "MachOPlatform has not finished booting",
                                       inconvertibleErrorCode());

      G.allocActions().push_back(
          {cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   MP.orc_rt_macho_register_thread_data_section, R.getRange())),
           cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   MP.orc_rt_macho_deregister_thread_data_section,
                   R.getRange()))});
    }
  }
  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::registerEHSectionsPhase1(
    jitlink::LinkGraph &G) {

  // If there's no eh-frame there's nothing to do.
  auto *EHFrameSection = G.findSectionByName(EHFrameSectionName);
  if (!EHFrameSection)
    return Error::success();

  // If the eh-frame section is empty there's nothing to do.
  jitlink::SectionRange R(*EHFrameSection);
  if (R.empty())
    return Error::success();

  // Since we're linking the object containing the registration code now the
  // addresses won't be ready in the platform. We'll have to find them in this
  // graph instead.
  ExecutorAddr orc_rt_macho_register_ehframe_section;
  ExecutorAddr orc_rt_macho_deregister_ehframe_section;
  for (auto *Sym : G.defined_symbols()) {
    if (!Sym->hasName())
      continue;
    if (Sym->getName() == "___orc_rt_macho_register_ehframe_section")
      orc_rt_macho_register_ehframe_section = ExecutorAddr(Sym->getAddress());
    else if (Sym->getName() == "___orc_rt_macho_deregister_ehframe_section")
      orc_rt_macho_deregister_ehframe_section = ExecutorAddr(Sym->getAddress());

    if (orc_rt_macho_register_ehframe_section &&
        orc_rt_macho_deregister_ehframe_section)
      break;
  }

  // If we failed to find the required functions then bail out.
  if (!orc_rt_macho_register_ehframe_section ||
      !orc_rt_macho_deregister_ehframe_section)
    return make_error<StringError>("Could not find eh-frame registration "
                                   "functions during platform bootstrap",
                                   inconvertibleErrorCode());

  // Otherwise, add allocation actions to the graph to register eh-frames for
  // this object.
  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
           orc_rt_macho_register_ehframe_section, R.getRange())),
       cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
           orc_rt_macho_deregister_ehframe_section, R.getRange()))});

  return Error::success();
}

} // End namespace orc.
} // End namespace llvm.
