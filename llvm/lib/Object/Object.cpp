//===- Object.cpp - C bindings to the object file library--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the C bindings to the file-format-independent object
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm-c/Object.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace object;

inline OwningBinary<ObjectFile> *unwrap(LLVMObjectFileRef OF) {
  return reinterpret_cast<OwningBinary<ObjectFile> *>(OF);
}

inline LLVMObjectFileRef wrap(const OwningBinary<ObjectFile> *OF) {
  return reinterpret_cast<LLVMObjectFileRef>(
      const_cast<OwningBinary<ObjectFile> *>(OF));
}

inline section_iterator *unwrap(LLVMSectionIteratorRef SI) {
  return reinterpret_cast<section_iterator*>(SI);
}

inline LLVMSectionIteratorRef
wrap(const section_iterator *SI) {
  return reinterpret_cast<LLVMSectionIteratorRef>
    (const_cast<section_iterator*>(SI));
}

inline symbol_iterator *unwrap(LLVMSymbolIteratorRef SI) {
  return reinterpret_cast<symbol_iterator*>(SI);
}

inline LLVMSymbolIteratorRef
wrap(const symbol_iterator *SI) {
  return reinterpret_cast<LLVMSymbolIteratorRef>
    (const_cast<symbol_iterator*>(SI));
}

inline relocation_iterator *unwrap(LLVMRelocationIteratorRef SI) {
  return reinterpret_cast<relocation_iterator*>(SI);
}

inline LLVMRelocationIteratorRef
wrap(const relocation_iterator *SI) {
  return reinterpret_cast<LLVMRelocationIteratorRef>
    (const_cast<relocation_iterator*>(SI));
}

// ObjectFile creation
LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf) {
  std::unique_ptr<MemoryBuffer> Buf(unwrap(MemBuf));
  ErrorOr<std::unique_ptr<ObjectFile>> ObjOrErr(
      ObjectFile::createObjectFile(Buf->getMemBufferRef()));
  std::unique_ptr<ObjectFile> Obj;
  if (!ObjOrErr)
    return nullptr;

  auto *Ret = new OwningBinary<ObjectFile>(std::move(ObjOrErr.get()), std::move(Buf));
  return wrap(Ret);
}

void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile) {
  delete unwrap(ObjectFile);
}

// ObjectFile Section iterators
LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef OF) {
  OwningBinary<ObjectFile> *OB = unwrap(OF);
  section_iterator SI = OB->getBinary()->section_begin();
  return wrap(new section_iterator(SI));
}

void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef OF,
                                    LLVMSectionIteratorRef SI) {
  OwningBinary<ObjectFile> *OB = unwrap(OF);
  return (*unwrap(SI) == OB->getBinary()->section_end()) ? 1 : 0;
}

void LLVMMoveToNextSection(LLVMSectionIteratorRef SI) {
  ++(*unwrap(SI));
}

void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect,
                                 LLVMSymbolIteratorRef Sym) {
  if (std::error_code ec = (*unwrap(Sym))->getSection(*unwrap(Sect)))
    report_fatal_error(ec.message());
}

// ObjectFile Symbol iterators
LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef OF) {
  OwningBinary<ObjectFile> *OB = unwrap(OF);
  symbol_iterator SI = OB->getBinary()->symbol_begin();
  return wrap(new symbol_iterator(SI));
}

void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef OF,
                                   LLVMSymbolIteratorRef SI) {
  OwningBinary<ObjectFile> *OB = unwrap(OF);
  return (*unwrap(SI) == OB->getBinary()->symbol_end()) ? 1 : 0;
}

void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI) {
  ++(*unwrap(SI));
}

// SectionRef accessors
const char *LLVMGetSectionName(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (std::error_code ec = (*unwrap(SI))->getName(ret))
   report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (std::error_code ec = (*unwrap(SI))->getContents(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

LLVMBool LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI,
                                 LLVMSymbolIteratorRef Sym) {
  bool ret;
  if (std::error_code ec = (*unwrap(SI))->containsSymbol(**unwrap(Sym), ret))
    report_fatal_error(ec.message());
  return ret;
}

// Section Relocation iterators
LLVMRelocationIteratorRef LLVMGetRelocations(LLVMSectionIteratorRef Section) {
  relocation_iterator SI = (*unwrap(Section))->relocation_begin();
  return wrap(new relocation_iterator(SI));
}

void LLVMDisposeRelocationIterator(LLVMRelocationIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsRelocationIteratorAtEnd(LLVMSectionIteratorRef Section,
                                       LLVMRelocationIteratorRef SI) {
  return (*unwrap(SI) == (*unwrap(Section))->relocation_end()) ? 1 : 0;
}

void LLVMMoveToNextRelocation(LLVMRelocationIteratorRef SI) {
  ++(*unwrap(SI));
}


// SymbolRef accessors
const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI) {
  StringRef ret;
  if (std::error_code ec = (*unwrap(SI))->getName(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

// RelocationRef accessors
uint64_t LLVMGetRelocationAddress(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(RI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetRelocationOffset(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(RI))->getOffset(ret))
    report_fatal_error(ec.message());
  return ret;
}

LLVMSymbolIteratorRef LLVMGetRelocationSymbol(LLVMRelocationIteratorRef RI) {
  symbol_iterator ret = (*unwrap(RI))->getSymbol();
  return wrap(new symbol_iterator(ret));
}

uint64_t LLVMGetRelocationType(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (std::error_code ec = (*unwrap(RI))->getType(ret))
    report_fatal_error(ec.message());
  return ret;
}

// NOTE: Caller takes ownership of returned string.
const char *LLVMGetRelocationTypeName(LLVMRelocationIteratorRef RI) {
  SmallVector<char, 0> ret;
  if (std::error_code ec = (*unwrap(RI))->getTypeName(ret))
    report_fatal_error(ec.message());

  char *str = static_cast<char*>(malloc(ret.size()));
  std::copy(ret.begin(), ret.end(), str);
  return str;
}

// NOTE: Caller takes ownership of returned string.
const char *LLVMGetRelocationValueString(LLVMRelocationIteratorRef RI) {
  SmallVector<char, 0> ret;
  if (std::error_code ec = (*unwrap(RI))->getValueString(ret))
    report_fatal_error(ec.message());

  char *str = static_cast<char*>(malloc(ret.size()));
  std::copy(ret.begin(), ret.end(), str);
  return str;
}

