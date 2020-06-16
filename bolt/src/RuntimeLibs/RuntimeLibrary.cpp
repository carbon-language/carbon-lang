//===--- RuntimeLibrary.cpp - The Runtime Library -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "RuntimeLibrary.h"
#include "Utils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Path.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-rtlib"

using namespace llvm;
using namespace bolt;

std::string RuntimeLibrary::getLibPath(StringRef ToolPath,
                                       StringRef LibFileName) {
  auto Dir = llvm::sys::path::parent_path(ToolPath);
  SmallString<128> LibPath = llvm::sys::path::parent_path(Dir);
  llvm::sys::path::append(LibPath, "lib");
  if (!llvm::sys::fs::exists(LibPath)) {
    // In some cases we install bolt binary into one level deeper in bin/,
    // we need to go back one more level to find lib directory.
    LibPath =
        llvm::sys::path::parent_path(llvm::sys::path::parent_path(LibPath));
    llvm::sys::path::append(LibPath, "lib");
  }
  llvm::sys::path::append(LibPath, LibFileName);
  if (!llvm::sys::fs::exists(LibPath)) {
    errs() << "BOLT-ERROR: library not found: " << LibPath << "\n";
    exit(1);
  }
  return LibPath.str();
}

void RuntimeLibrary::loadLibraryToOLT(StringRef LibPath,
                                      orc::ExecutionSession &ES,
                                      orc::RTDyldObjectLinkingLayer &OLT) {
  OLT.setProcessAllSections(false);
  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeBuf =
      MemoryBuffer::getFile(LibPath, -1, false);
  check_error(MaybeBuf.getError(), LibPath);
  std::unique_ptr<MemoryBuffer> B = std::move(MaybeBuf.get());
  file_magic Magic = identify_magic(B->getBuffer());

  if (Magic == file_magic::archive) {
    Error Err = Error::success();
    object::Archive Archive(B.get()->getMemBufferRef(), Err);
    for (auto &C : Archive.children(Err)) {
      auto ChildKey = ES.allocateVModule();
      auto ChildBuf =
          MemoryBuffer::getMemBuffer(cantFail(C.getMemoryBufferRef()));
      cantFail(OLT.addObject(ChildKey, std::move(ChildBuf)));
      cantFail(OLT.emitAndFinalize(ChildKey));
    }
    check_error(std::move(Err), B->getBufferIdentifier());
  } else if (Magic == file_magic::elf_relocatable ||
             Magic == file_magic::elf_shared_object) {
    auto K2 = ES.allocateVModule();
    cantFail(OLT.addObject(K2, std::move(B)));
    cantFail(OLT.emitAndFinalize(K2));
  } else {
    errs() << "BOLT-ERROR: unrecognized library format: " << LibPath << "\n";
    exit(1);
  }
}
