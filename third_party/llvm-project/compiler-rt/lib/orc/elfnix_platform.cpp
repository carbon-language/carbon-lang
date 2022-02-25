//===- elfnix_platform.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code required to load the rest of the ELF-on-*IX runtime.
//
//===----------------------------------------------------------------------===//

#include "elfnix_platform.h"
#include "common.h"
#include "error.h"
#include "wrapper_function_utils.h"

#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace __orc_rt;
using namespace __orc_rt::elfnix;

// Declare function tags for functions in the JIT process.
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_elfnix_get_initializers_tag)
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_elfnix_get_deinitializers_tag)
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_elfnix_symbol_lookup_tag)

// eh-frame registration functions.
// We expect these to be available for all processes.
extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

namespace {

Error validatePointerSectionExtent(const char *SectionName,
                                   const ExecutorAddressRange &SE) {
  if (SE.size().getValue() % sizeof(uintptr_t)) {
    std::ostringstream ErrMsg;
    ErrMsg << std::hex << "Size of " << SectionName << " 0x"
           << SE.StartAddress.getValue() << " -- 0x" << SE.EndAddress.getValue()
           << " is not a pointer multiple";
    return make_error<StringError>(ErrMsg.str());
  }
  return Error::success();
}

Error runInitArray(const std::vector<ExecutorAddressRange> &InitArraySections,
                   const ELFNixJITDylibInitializers &MOJDIs) {

  for (const auto &ModInits : InitArraySections) {
    if (auto Err = validatePointerSectionExtent(".init_array", ModInits))
      return Err;

    using InitFunc = void (*)();
    for (auto *Init : ModInits.toSpan<InitFunc>())
      (*Init)();
  }

  return Error::success();
}

class ELFNixPlatformRuntimeState {
private:
  struct AtExitEntry {
    void (*Func)(void *);
    void *Arg;
  };

  using AtExitsVector = std::vector<AtExitEntry>;

  struct PerJITDylibState {
    void *Header = nullptr;
    size_t RefCount = 0;
    bool AllowReinitialization = false;
    AtExitsVector AtExits;
  };

public:
  static void initialize();
  static ELFNixPlatformRuntimeState &get();
  static void destroy();

  ELFNixPlatformRuntimeState() = default;

  // Delete copy and move constructors.
  ELFNixPlatformRuntimeState(const ELFNixPlatformRuntimeState &) = delete;
  ELFNixPlatformRuntimeState &
  operator=(const ELFNixPlatformRuntimeState &) = delete;
  ELFNixPlatformRuntimeState(ELFNixPlatformRuntimeState &&) = delete;
  ELFNixPlatformRuntimeState &operator=(ELFNixPlatformRuntimeState &&) = delete;

  Error registerObjectSections(ELFNixPerObjectSectionsToRegister POSR);
  Error deregisterObjectSections(ELFNixPerObjectSectionsToRegister POSR);

  const char *dlerror();
  void *dlopen(string_view Name, int Mode);
  int dlclose(void *DSOHandle);
  void *dlsym(void *DSOHandle, string_view Symbol);

  int registerAtExit(void (*F)(void *), void *Arg, void *DSOHandle);
  void runAtExits(void *DSOHandle);

private:
  PerJITDylibState *getJITDylibStateByHeaderAddr(void *DSOHandle);
  PerJITDylibState *getJITDylibStateByName(string_view Path);
  PerJITDylibState &
  getOrCreateJITDylibState(ELFNixJITDylibInitializers &MOJDIs);

  Expected<ExecutorAddress> lookupSymbolInJITDylib(void *DSOHandle,
                                                   string_view Symbol);

  Expected<ELFNixJITDylibInitializerSequence>
  getJITDylibInitializersByName(string_view Path);
  Expected<void *> dlopenInitialize(string_view Path, int Mode);
  Error initializeJITDylib(ELFNixJITDylibInitializers &MOJDIs);

  static ELFNixPlatformRuntimeState *MOPS;

  using InitSectionHandler =
      Error (*)(const std::vector<ExecutorAddressRange> &Sections,
                const ELFNixJITDylibInitializers &MOJDIs);
  const std::vector<std::pair<const char *, InitSectionHandler>> InitSections =
      {{".init_array", runInitArray}};

  // FIXME: Move to thread-state.
  std::string DLFcnError;

  std::recursive_mutex JDStatesMutex;
  std::unordered_map<void *, PerJITDylibState> JDStates;
  std::unordered_map<std::string, void *> JDNameToHeader;
};

ELFNixPlatformRuntimeState *ELFNixPlatformRuntimeState::MOPS = nullptr;

void ELFNixPlatformRuntimeState::initialize() {
  assert(!MOPS && "ELFNixPlatformRuntimeState should be null");
  MOPS = new ELFNixPlatformRuntimeState();
}

ELFNixPlatformRuntimeState &ELFNixPlatformRuntimeState::get() {
  assert(MOPS && "ELFNixPlatformRuntimeState not initialized");
  return *MOPS;
}

void ELFNixPlatformRuntimeState::destroy() {
  assert(MOPS && "ELFNixPlatformRuntimeState not initialized");
  delete MOPS;
}

Error ELFNixPlatformRuntimeState::registerObjectSections(
    ELFNixPerObjectSectionsToRegister POSR) {
  if (POSR.EHFrameSection.StartAddress)
    __register_frame(POSR.EHFrameSection.StartAddress.toPtr<const char *>());

  // TODO: Register thread data sections.

  return Error::success();
}

Error ELFNixPlatformRuntimeState::deregisterObjectSections(
    ELFNixPerObjectSectionsToRegister POSR) {
  if (POSR.EHFrameSection.StartAddress)
    __deregister_frame(POSR.EHFrameSection.StartAddress.toPtr<const char *>());

  return Error::success();
}

const char *ELFNixPlatformRuntimeState::dlerror() { return DLFcnError.c_str(); }

void *ELFNixPlatformRuntimeState::dlopen(string_view Path, int Mode) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);

  // Use fast path if all JITDylibs are already loaded and don't require
  // re-running initializers.
  if (auto *JDS = getJITDylibStateByName(Path)) {
    if (!JDS->AllowReinitialization) {
      ++JDS->RefCount;
      return JDS->Header;
    }
  }

  auto H = dlopenInitialize(Path, Mode);
  if (!H) {
    DLFcnError = toString(H.takeError());
    return nullptr;
  }

  return *H;
}

int ELFNixPlatformRuntimeState::dlclose(void *DSOHandle) {
  runAtExits(DSOHandle);
  return 0;
}

void *ELFNixPlatformRuntimeState::dlsym(void *DSOHandle, string_view Symbol) {
  auto Addr = lookupSymbolInJITDylib(DSOHandle, Symbol);
  if (!Addr) {
    DLFcnError = toString(Addr.takeError());
    return 0;
  }

  return Addr->toPtr<void *>();
}

int ELFNixPlatformRuntimeState::registerAtExit(void (*F)(void *), void *Arg,
                                               void *DSOHandle) {
  // FIXME: Handle out-of-memory errors, returning -1 if OOM.
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeaderAddr(DSOHandle);
  assert(JDS && "JITDylib state not initialized");
  JDS->AtExits.push_back({F, Arg});
  return 0;
}

void ELFNixPlatformRuntimeState::runAtExits(void *DSOHandle) {
  // FIXME: Should atexits be allowed to run concurrently with access to
  // JDState?
  AtExitsVector V;
  {
    std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
    auto *JDS = getJITDylibStateByHeaderAddr(DSOHandle);
    assert(JDS && "JITDlybi state not initialized");
    std::swap(V, JDS->AtExits);
  }

  while (!V.empty()) {
    auto &AE = V.back();
    AE.Func(AE.Arg);
    V.pop_back();
  }
}

ELFNixPlatformRuntimeState::PerJITDylibState *
ELFNixPlatformRuntimeState::getJITDylibStateByHeaderAddr(void *DSOHandle) {
  auto I = JDStates.find(DSOHandle);
  if (I == JDStates.end())
    return nullptr;
  return &I->second;
}

ELFNixPlatformRuntimeState::PerJITDylibState *
ELFNixPlatformRuntimeState::getJITDylibStateByName(string_view Name) {
  // FIXME: Avoid creating string copy here.
  auto I = JDNameToHeader.find(std::string(Name.data(), Name.size()));
  if (I == JDNameToHeader.end())
    return nullptr;
  void *H = I->second;
  auto J = JDStates.find(H);
  assert(J != JDStates.end() &&
         "JITDylib has name map entry but no header map entry");
  return &J->second;
}

ELFNixPlatformRuntimeState::PerJITDylibState &
ELFNixPlatformRuntimeState::getOrCreateJITDylibState(
    ELFNixJITDylibInitializers &MOJDIs) {
  void *Header = MOJDIs.DSOHandleAddress.toPtr<void *>();

  auto &JDS = JDStates[Header];

  // If this entry hasn't been created yet.
  if (!JDS.Header) {
    assert(!JDNameToHeader.count(MOJDIs.Name) &&
           "JITDylib has header map entry but no name map entry");
    JDNameToHeader[MOJDIs.Name] = Header;
    JDS.Header = Header;
  }

  return JDS;
}

Expected<ExecutorAddress>
ELFNixPlatformRuntimeState::lookupSymbolInJITDylib(void *DSOHandle,
                                                   string_view Sym) {
  Expected<ExecutorAddress> Result((ExecutorAddress()));
  if (auto Err = WrapperFunction<SPSExpected<SPSExecutorAddress>(
          SPSExecutorAddress,
          SPSString)>::call(&__orc_rt_elfnix_symbol_lookup_tag, Result,
                            ExecutorAddress::fromPtr(DSOHandle), Sym))
    return std::move(Err);
  return Result;
}

Expected<ELFNixJITDylibInitializerSequence>
ELFNixPlatformRuntimeState::getJITDylibInitializersByName(string_view Path) {
  Expected<ELFNixJITDylibInitializerSequence> Result(
      (ELFNixJITDylibInitializerSequence()));
  std::string PathStr(Path.data(), Path.size());
  if (auto Err =
          WrapperFunction<SPSExpected<SPSELFNixJITDylibInitializerSequence>(
              SPSString)>::call(&__orc_rt_elfnix_get_initializers_tag, Result,
                                Path))
    return std::move(Err);
  return Result;
}

Expected<void *> ELFNixPlatformRuntimeState::dlopenInitialize(string_view Path,
                                                              int Mode) {
  // Either our JITDylib wasn't loaded, or it or one of its dependencies allows
  // reinitialization. We need to call in to the JIT to see if there's any new
  // work pending.
  auto InitSeq = getJITDylibInitializersByName(Path);
  if (!InitSeq)
    return InitSeq.takeError();

  // Init sequences should be non-empty.
  if (InitSeq->empty())
    return make_error<StringError>(
        "__orc_rt_elfnix_get_initializers returned an "
        "empty init sequence");

  // Otherwise register and run initializers for each JITDylib.
  for (auto &MOJDIs : *InitSeq)
    if (auto Err = initializeJITDylib(MOJDIs))
      return std::move(Err);

  // Return the header for the last item in the list.
  auto *JDS = getJITDylibStateByHeaderAddr(
      InitSeq->back().DSOHandleAddress.toPtr<void *>());
  assert(JDS && "Missing state entry for JD");
  return JDS->Header;
}

Error ELFNixPlatformRuntimeState::initializeJITDylib(
    ELFNixJITDylibInitializers &MOJDIs) {

  auto &JDS = getOrCreateJITDylibState(MOJDIs);
  ++JDS.RefCount;

  for (auto &KV : InitSections) {
    const auto &Name = KV.first;
    const auto &Handler = KV.second;
    auto I = MOJDIs.InitSections.find(Name);
    if (I != MOJDIs.InitSections.end()) {
      if (auto Err = Handler(I->second, MOJDIs))
        return Err;
    }
  }

  return Error::success();
}

} // end anonymous namespace

//------------------------------------------------------------------------------
//                             JIT entry points
//------------------------------------------------------------------------------

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_platform_bootstrap(char *ArgData, size_t ArgSize) {
  ELFNixPlatformRuntimeState::initialize();
  return WrapperFunctionResult().release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_platform_shutdown(char *ArgData, size_t ArgSize) {
  ELFNixPlatformRuntimeState::destroy();
  return WrapperFunctionResult().release();
}

/// Wrapper function for registering metadata on a per-object basis.
ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_register_object_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSELFNixPerObjectSectionsToRegister)>::
      handle(ArgData, ArgSize,
             [](ELFNixPerObjectSectionsToRegister &POSR) {
               return ELFNixPlatformRuntimeState::get().registerObjectSections(
                   std::move(POSR));
             })
          .release();
}

/// Wrapper for releasing per-object metadat.
ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_deregister_object_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSELFNixPerObjectSectionsToRegister)>::
      handle(ArgData, ArgSize,
             [](ELFNixPerObjectSectionsToRegister &POSR) {
               return ELFNixPlatformRuntimeState::get()
                   .deregisterObjectSections(std::move(POSR));
             })
          .release();
}

//------------------------------------------------------------------------------
//                           cxa_atexit support
//------------------------------------------------------------------------------

int __orc_rt_elfnix_cxa_atexit(void (*func)(void *), void *arg,
                               void *dso_handle) {
  return ELFNixPlatformRuntimeState::get().registerAtExit(func, arg,
                                                          dso_handle);
}

void __orc_rt_elfnix_cxa_finalize(void *dso_handle) {
  ELFNixPlatformRuntimeState::get().runAtExits(dso_handle);
}

//------------------------------------------------------------------------------
//                        JIT'd dlfcn alternatives.
//------------------------------------------------------------------------------

const char *__orc_rt_elfnix_jit_dlerror() {
  return ELFNixPlatformRuntimeState::get().dlerror();
}

void *__orc_rt_elfnix_jit_dlopen(const char *path, int mode) {
  return ELFNixPlatformRuntimeState::get().dlopen(path, mode);
}

int __orc_rt_elfnix_jit_dlclose(void *dso_handle) {
  return ELFNixPlatformRuntimeState::get().dlclose(dso_handle);
}

void *__orc_rt_elfnix_jit_dlsym(void *dso_handle, const char *symbol) {
  return ELFNixPlatformRuntimeState::get().dlsym(dso_handle, symbol);
}

//------------------------------------------------------------------------------
//                             ELFNix Run Program
//------------------------------------------------------------------------------

ORC_RT_INTERFACE int64_t __orc_rt_elfnix_run_program(
    const char *JITDylibName, const char *EntrySymbolName, int argc,
    char *argv[]) {
  using MainTy = int (*)(int, char *[]);

  void *H = __orc_rt_elfnix_jit_dlopen(JITDylibName,
                                       __orc_rt::elfnix::ORC_RT_RTLD_LAZY);
  if (!H) {
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());
    return -1;
  }

  auto *Main =
      reinterpret_cast<MainTy>(__orc_rt_elfnix_jit_dlsym(H, EntrySymbolName));

  if (!Main) {
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());
    return -1;
  }

  int Result = Main(argc, argv);

  if (__orc_rt_elfnix_jit_dlclose(H) == -1)
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());

  return Result;
}
