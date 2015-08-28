//===- SymbolicFile.cpp - Interface that only provides symbols --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a file format independent SymbolicFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

// COFF short import file is a special kind of file which contains
// only symbol names for DLL-exported symbols. This class implements
// SymbolicFile interface for the file.
namespace {
class COFFImportFile : public SymbolicFile {
public:
  COFFImportFile(MemoryBufferRef Source)
      : SymbolicFile(sys::fs::file_magic::coff_import_library, Source) {}

  void moveSymbolNext(DataRefImpl &Symb) const override { ++Symb.p; }

  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override {
    if (Symb.p == 1)
      OS << "__imp_";
    OS << StringRef(Data.getBufferStart() + sizeof(coff_import_header));
    return std::error_code();
  }

  uint32_t getSymbolFlags(DataRefImpl Symb) const override {
    return SymbolRef::SF_Global;
  }

  basic_symbol_iterator symbol_begin_impl() const override {
    return BasicSymbolRef(DataRefImpl(), this);
  }

  basic_symbol_iterator symbol_end_impl() const override {
    DataRefImpl Symb;
    Symb.p = isCode() ? 2 : 1;
    return BasicSymbolRef(Symb, this);
  }

private:
  bool isCode() const {
    auto *Import = reinterpret_cast<const coff_import_header *>(
        Data.getBufferStart());
    return Import->getType() == llvm::COFF::IMPORT_CODE;
  }
};
} // anonymous namespace

SymbolicFile::SymbolicFile(unsigned int Type, MemoryBufferRef Source)
    : Binary(Type, Source) {}

SymbolicFile::~SymbolicFile() {}

ErrorOr<std::unique_ptr<SymbolicFile>> SymbolicFile::createSymbolicFile(
    MemoryBufferRef Object, sys::fs::file_magic Type, LLVMContext *Context) {
  StringRef Data = Object.getBuffer();
  if (Type == sys::fs::file_magic::unknown)
    Type = sys::fs::identify_magic(Data);

  switch (Type) {
  case sys::fs::file_magic::bitcode:
    if (Context)
      return IRObjectFile::create(Object, *Context);
  // Fallthrough
  case sys::fs::file_magic::unknown:
  case sys::fs::file_magic::archive:
  case sys::fs::file_magic::macho_universal_binary:
  case sys::fs::file_magic::windows_resource:
    return object_error::invalid_file_type;
  case sys::fs::file_magic::elf:
  case sys::fs::file_magic::elf_executable:
  case sys::fs::file_magic::elf_shared_object:
  case sys::fs::file_magic::elf_core:
  case sys::fs::file_magic::macho_executable:
  case sys::fs::file_magic::macho_fixed_virtual_memory_shared_lib:
  case sys::fs::file_magic::macho_core:
  case sys::fs::file_magic::macho_preload_executable:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib:
  case sys::fs::file_magic::macho_dynamic_linker:
  case sys::fs::file_magic::macho_bundle:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib_stub:
  case sys::fs::file_magic::macho_dsym_companion:
  case sys::fs::file_magic::macho_kext_bundle:
  case sys::fs::file_magic::pecoff_executable:
    return ObjectFile::createObjectFile(Object, Type);
  case sys::fs::file_magic::coff_import_library:
    return std::unique_ptr<SymbolicFile>(new COFFImportFile(Object));
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::coff_object: {
    ErrorOr<std::unique_ptr<ObjectFile>> Obj =
        ObjectFile::createObjectFile(Object, Type);
    if (!Obj || !Context)
      return std::move(Obj);

    ErrorOr<MemoryBufferRef> BCData =
        IRObjectFile::findBitcodeInObject(*Obj->get());
    if (!BCData)
      return std::move(Obj);

    return IRObjectFile::create(
        MemoryBufferRef(BCData->getBuffer(), Object.getBufferIdentifier()),
        *Context);
  }
  }
  llvm_unreachable("Unexpected Binary File Type");
}
