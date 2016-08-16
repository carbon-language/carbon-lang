//===-- StreamExecutor.cpp - StreamExecutor implementation ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of StreamExecutor class internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/StreamExecutor.h"

#include <cassert>

#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Stream.h"

#include "llvm/ADT/STLExtras.h"

namespace streamexecutor {

StreamExecutor::StreamExecutor(PlatformStreamExecutor *PlatformExecutor)
    : PlatformExecutor(PlatformExecutor) {}

StreamExecutor::~StreamExecutor() = default;

Expected<std::unique_ptr<Stream>> StreamExecutor::createStream() {
  Expected<std::unique_ptr<PlatformStreamHandle>> MaybePlatformStream =
      PlatformExecutor->createStream();
  if (!MaybePlatformStream) {
    return MaybePlatformStream.takeError();
  }
  assert((*MaybePlatformStream)->getExecutor() == PlatformExecutor &&
         "an executor created a stream with a different stored executor");
  return llvm::make_unique<Stream>(std::move(*MaybePlatformStream));
}

} // namespace streamexecutor
