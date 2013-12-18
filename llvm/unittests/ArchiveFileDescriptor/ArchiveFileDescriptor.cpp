//===- llvm/unittest/ArchiveFileDescriptor/ArchiveFileDescriptor.cpp ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static void failIfError(error_code EC) {
  if (!EC)
    return;

  errs() << "ERROR: " << EC.message() << "\n";
  exit(1);
}

TEST(ArchiveFileDescriptor, Test1) {
  int FD;

  error_code EC = sys::fs::openFileForRead("ArchiveFileDescriptor", FD);
  failIfError(EC);

  OwningPtr<MemoryBuffer> MemoryBuffer;
  EC = MemoryBuffer::getOpenFile(FD, "Dummy Filename",
                                 MemoryBuffer,
                                 /* FileSize */ -1,
                                 /* RequiresNullTerminator */ false);
  failIfError(EC);

  // Attempt to open the binary.
  OwningPtr<Binary> Binary;
  EC = createBinary(MemoryBuffer.take(), Binary);
  failIfError(EC);

  if (Archive *Arc = dyn_cast<Archive>(Binary.get())) {
    (void)Arc;
    errs() << "ERROR: Loaded archive, was expecting object file\n";
  } else if (ObjectFile *Obj = dyn_cast<ObjectFile>(Binary.get())) {
    (void)Obj;
  } else {
    outs() << "ERROR: Unknown file type\n";
    exit(1);
  }
}
