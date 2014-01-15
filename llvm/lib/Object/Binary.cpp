//===- Binary.cpp - A generic binary file -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Binary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

// Include headers for createBinary.
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace object;

Binary::~Binary() {
  delete Data;
}

Binary::Binary(unsigned int Type, MemoryBuffer *Source)
  : TypeID(Type)
  , Data(Source) {}

StringRef Binary::getData() const {
  return Data->getBuffer();
}

StringRef Binary::getFileName() const {
  return Data->getBufferIdentifier();
}

ErrorOr<Binary *> object::createBinary(MemoryBuffer *Source) {
  OwningPtr<MemoryBuffer> scopedSource(Source);
  sys::fs::file_magic type = sys::fs::identify_magic(Source->getBuffer());
  error_code EC;
  switch (type) {
    case sys::fs::file_magic::archive: {
      OwningPtr<Binary> Ret(new Archive(scopedSource.take(), EC));
      if (EC)
        return EC;
      return Ret.take();
    }
    case sys::fs::file_magic::elf_relocatable:
    case sys::fs::file_magic::elf_executable:
    case sys::fs::file_magic::elf_shared_object:
    case sys::fs::file_magic::elf_core: {
      OwningPtr<Binary> Ret(
          ObjectFile::createELFObjectFile(scopedSource.take()));
      if (!Ret)
        return object_error::invalid_file_type;
      return Ret.take();
    }
    case sys::fs::file_magic::macho_object:
    case sys::fs::file_magic::macho_executable:
    case sys::fs::file_magic::macho_fixed_virtual_memory_shared_lib:
    case sys::fs::file_magic::macho_core:
    case sys::fs::file_magic::macho_preload_executable:
    case sys::fs::file_magic::macho_dynamically_linked_shared_lib:
    case sys::fs::file_magic::macho_dynamic_linker:
    case sys::fs::file_magic::macho_bundle:
    case sys::fs::file_magic::macho_dynamically_linked_shared_lib_stub:
    case sys::fs::file_magic::macho_dsym_companion: {
      OwningPtr<Binary> Ret(
          ObjectFile::createMachOObjectFile(scopedSource.take()));
      if (!Ret)
        return object_error::invalid_file_type;
      return Ret.take();
    }
    case sys::fs::file_magic::macho_universal_binary: {
      OwningPtr<Binary> Ret(new MachOUniversalBinary(scopedSource.take(), EC));
      if (EC)
        return EC;
      return Ret.take();
    }
    case sys::fs::file_magic::coff_object:
    case sys::fs::file_magic::coff_import_library:
    case sys::fs::file_magic::pecoff_executable: {
      OwningPtr<Binary> Ret(
          ObjectFile::createCOFFObjectFile(scopedSource.take()));
      if (!Ret)
        return object_error::invalid_file_type;
      return Ret.take();
    }
    case sys::fs::file_magic::unknown:
    case sys::fs::file_magic::bitcode:
    case sys::fs::file_magic::windows_resource: {
      // Unrecognized object file format.
      return object_error::invalid_file_type;
    }
  }
  llvm_unreachable("Unexpected Binary File Type");
}

ErrorOr<Binary *> object::createBinary(StringRef Path) {
  OwningPtr<MemoryBuffer> File;
  if (error_code EC = MemoryBuffer::getFileOrSTDIN(Path, File))
    return EC;
  return createBinary(File.take());
}
