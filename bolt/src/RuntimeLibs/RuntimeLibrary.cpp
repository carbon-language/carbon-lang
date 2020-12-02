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
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Path.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-rtlib"

using namespace llvm;
using namespace bolt;

void RuntimeLibrary::anchor() {}

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
  return std::string(LibPath.str());
}

void RuntimeLibrary::loadLibrary(StringRef LibPath, RuntimeDyld &RTDyld) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeBuf =
      MemoryBuffer::getFile(LibPath, -1, false);
  check_error(MaybeBuf.getError(), LibPath);
  std::unique_ptr<MemoryBuffer> B = std::move(MaybeBuf.get());
  file_magic Magic = identify_magic(B->getBuffer());

  if (Magic == file_magic::archive) {
    Error Err = Error::success();
    object::Archive Archive(B.get()->getMemBufferRef(), Err);
    for (auto &C : Archive.children(Err)) {
      std::unique_ptr<object::Binary> Bin = cantFail(C.getAsBinary());
      if (auto *Obj = dyn_cast<object::ObjectFile>(&*Bin)) {
        RTDyld.loadObject(*Obj);
      }
    }
    check_error(std::move(Err), B->getBufferIdentifier());
  } else if (Magic == file_magic::elf_relocatable ||
             Magic == file_magic::elf_shared_object) {
    std::unique_ptr<object::ObjectFile> Obj = cantFail(
      object::ObjectFile::createObjectFile(B.get()->getMemBufferRef()),
      "error creating in-memory object");
    RTDyld.loadObject(*Obj);
  } else {
    errs() << "BOLT-ERROR: unrecognized library format: " << LibPath << "\n";
    exit(1);
  }
}
