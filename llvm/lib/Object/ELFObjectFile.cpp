//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
using namespace object;

static ErrorOr<ObjectFile *> createELFObjectFileAux(MemoryBuffer *Obj,
                                                    bool BufferOwned) {
  std::pair<unsigned char, unsigned char> Ident = getElfArchType(Obj);
  std::size_t MaxAlignment =
    1ULL << countTrailingZeros(uintptr_t(Obj->getBufferStart()));

  std::error_code EC;
  std::unique_ptr<ObjectFile> R;
  if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB)
#if !LLVM_IS_UNALIGNED_ACCESS_FAST
    if (MaxAlignment >= 4)
      R.reset(new ELFObjectFile<ELFType<support::little, 4, false> >(
          Obj, EC, BufferOwned));
    else
#endif
    if (MaxAlignment >= 2)
      R.reset(new ELFObjectFile<ELFType<support::little, 2, false> >(
          Obj, EC, BufferOwned));
    else
      return object_error::parse_failed;
  else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB)
#if !LLVM_IS_UNALIGNED_ACCESS_FAST
    if (MaxAlignment >= 4)
      R.reset(new ELFObjectFile<ELFType<support::big, 4, false> >(Obj, EC,
                                                                  BufferOwned));
    else
#endif
    if (MaxAlignment >= 2)
      R.reset(new ELFObjectFile<ELFType<support::big, 2, false> >(Obj, EC,
                                                                  BufferOwned));
    else
      return object_error::parse_failed;
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB)
#if !LLVM_IS_UNALIGNED_ACCESS_FAST
    if (MaxAlignment >= 8)
      R.reset(new ELFObjectFile<ELFType<support::big, 8, true> >(Obj, EC,
                                                                 BufferOwned));
    else
#endif
    if (MaxAlignment >= 2)
      R.reset(new ELFObjectFile<ELFType<support::big, 2, true> >(Obj, EC,
                                                                 BufferOwned));
    else
      return object_error::parse_failed;
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB) {
#if !LLVM_IS_UNALIGNED_ACCESS_FAST
    if (MaxAlignment >= 8)
      R.reset(new ELFObjectFile<ELFType<support::little, 8, true> >(
          Obj, EC, BufferOwned));
    else
#endif
    if (MaxAlignment >= 2)
      R.reset(new ELFObjectFile<ELFType<support::little, 2, true> >(
          Obj, EC, BufferOwned));
    else
      return object_error::parse_failed;
  }
  else
    llvm_unreachable("Buffer is not an ELF object file!");

  if (EC)
    return EC;
  return R.release();
}

ErrorOr<ObjectFile *> ObjectFile::createELFObjectFile(MemoryBuffer *Obj,
                                                      bool BufferOwned) {
  ErrorOr<ObjectFile *> Ret = createELFObjectFileAux(Obj, BufferOwned);
  if (BufferOwned && Ret.getError())
    delete Obj;
  return Ret;
}

} // end namespace llvm
