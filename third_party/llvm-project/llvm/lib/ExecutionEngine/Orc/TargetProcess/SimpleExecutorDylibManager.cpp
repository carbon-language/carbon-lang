//===--- SimpleExecutorDylibManager.cpp - Executor-side dylib management --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorDylibManager.h"

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {
namespace rt_bootstrap {

SimpleExecutorDylibManager::~SimpleExecutorDylibManager() {
  assert(Dylibs.empty() && "shutdown not called?");
}

Expected<tpctypes::DylibHandle>
SimpleExecutorDylibManager::open(const std::string &Path, uint64_t Mode) {
  if (Mode != 0)
    return make_error<StringError>("open: non-zero mode bits not yet supported",
                                   inconvertibleErrorCode());

  const char *PathCStr = Path.empty() ? nullptr : Path.c_str();
  std::string ErrMsg;

  auto DL = sys::DynamicLibrary::getPermanentLibrary(PathCStr, &ErrMsg);
  if (!DL.isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());

  std::lock_guard<std::mutex> Lock(M);
  Dylibs[NextId] = std::move(DL);
  return NextId++;
}

Expected<std::vector<ExecutorAddr>>
SimpleExecutorDylibManager::lookup(tpctypes::DylibHandle H,
                                   const RemoteSymbolLookupSet &L) {
  std::vector<ExecutorAddr> Result;

  std::lock_guard<std::mutex> Lock(M);
  auto I = Dylibs.find(H);
  if (I == Dylibs.end())
    return make_error<StringError>("No dylib for handle " + formatv("{0:x}", H),
                                   inconvertibleErrorCode());
  auto &DL = I->second;

  for (const auto &E : L) {

    if (E.Name.empty()) {
      if (E.Required)
        return make_error<StringError>("Required address for empty symbol \"\"",
                                       inconvertibleErrorCode());
      else
        Result.push_back(ExecutorAddr());
    } else {

      const char *DemangledSymName = E.Name.c_str();
#ifdef __APPLE__
      if (E.Name.front() != '_')
        return make_error<StringError>(Twine("MachO symbol \"") + E.Name +
                                           "\" missing leading '_'",
                                       inconvertibleErrorCode());
      ++DemangledSymName;
#endif

      void *Addr = DL.getAddressOfSymbol(DemangledSymName);
      if (!Addr && E.Required)
        return make_error<StringError>(Twine("Missing definition for ") +
                                           DemangledSymName,
                                       inconvertibleErrorCode());

      Result.push_back(ExecutorAddr::fromPtr(Addr));
    }
  }

  return Result;
}

Error SimpleExecutorDylibManager::shutdown() {

  DylibsMap DM;
  {
    std::lock_guard<std::mutex> Lock(M);
    std::swap(DM, Dylibs);
  }

  // There is no removal of dylibs at the moment, so nothing to do here.
  return Error::success();
}

void SimpleExecutorDylibManager::addBootstrapSymbols(
    StringMap<ExecutorAddr> &M) {
  M[rt::SimpleExecutorDylibManagerInstanceName] = ExecutorAddr::fromPtr(this);
  M[rt::SimpleExecutorDylibManagerOpenWrapperName] =
      ExecutorAddr::fromPtr(&openWrapper);
  M[rt::SimpleExecutorDylibManagerLookupWrapperName] =
      ExecutorAddr::fromPtr(&lookupWrapper);
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorDylibManager::openWrapper(const char *ArgData, size_t ArgSize) {
  return shared::
      WrapperFunction<rt::SPSSimpleExecutorDylibManagerOpenSignature>::handle(
             ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorDylibManager::open))
          .release();
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorDylibManager::lookupWrapper(const char *ArgData, size_t ArgSize) {
  return shared::
      WrapperFunction<rt::SPSSimpleExecutorDylibManagerLookupSignature>::handle(
             ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorDylibManager::lookup))
          .release();
}

} // namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
