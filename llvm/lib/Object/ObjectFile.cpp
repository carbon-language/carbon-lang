//===- ObjectFile.cpp - File format independent object file -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a file format independent ObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

using namespace llvm;
using namespace object;

void ObjectFile::anchor() { }

ObjectFile::ObjectFile(unsigned int Type, MemoryBuffer *source)
  : Binary(Type, source) {
}

error_code ObjectFile::getSymbolAlignment(DataRefImpl DRI,
                                          uint32_t &Result) const {
  Result = 0;
  return object_error::success;
}

section_iterator ObjectFile::getRelocatedSection(DataRefImpl Sec) const {
  return section_iterator(SectionRef(Sec, this));
}

ObjectFile *ObjectFile::createObjectFile(MemoryBuffer *Object) {
  if (Object->getBufferSize() < 64) {
    delete Object;
    return 0;
  }

  sys::fs::file_magic Type = sys::fs::identify_magic(Object->getBuffer());
  switch (Type) {
  case sys::fs::file_magic::unknown:
  case sys::fs::file_magic::bitcode:
  case sys::fs::file_magic::archive:
  case sys::fs::file_magic::macho_universal_binary:
  case sys::fs::file_magic::windows_resource:
    delete Object;
    return 0;
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::elf_executable:
  case sys::fs::file_magic::elf_shared_object:
  case sys::fs::file_magic::elf_core:
    return createELFObjectFile(Object);
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::macho_executable:
  case sys::fs::file_magic::macho_fixed_virtual_memory_shared_lib:
  case sys::fs::file_magic::macho_core:
  case sys::fs::file_magic::macho_preload_executable:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib:
  case sys::fs::file_magic::macho_dynamic_linker:
  case sys::fs::file_magic::macho_bundle:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib_stub:
  case sys::fs::file_magic::macho_dsym_companion:
    return createMachOObjectFile(Object);
  case sys::fs::file_magic::coff_object:
  case sys::fs::file_magic::coff_import_library:
  case sys::fs::file_magic::pecoff_executable:
    return createCOFFObjectFile(Object);
  }
  llvm_unreachable("Unexpected Object File Type");
}

ObjectFile *ObjectFile::createObjectFile(StringRef ObjectPath) {
  OwningPtr<MemoryBuffer> File;
  if (MemoryBuffer::getFile(ObjectPath, File))
    return NULL;
  return createObjectFile(File.take());
}
