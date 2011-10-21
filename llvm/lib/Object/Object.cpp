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

#include "llvm/Object/ObjectFile.h"
#include "llvm-c/Object.h"

using namespace llvm;
using namespace object;

// ObjectFile creation
LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf) {
  return wrap(ObjectFile::createObjectFile(unwrap(MemBuf)));
}

void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile) {
  delete unwrap(ObjectFile);
}

// ObjectFile Section iterators
LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile) {
  section_iterator SI = unwrap(ObjectFile)->begin_sections();
  return wrap(new section_iterator(SI));
}

void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSectionIteratorRef SI) {
  return (*unwrap(SI) == unwrap(ObjectFile)->end_sections()) ? 1 : 0;
}

void LLVMMoveToNextSection(LLVMSectionIteratorRef SI) {
  error_code ec;
  unwrap(SI)->increment(ec);
  if (ec) report_fatal_error("LLVMMoveToNextSection failed: " + ec.message());
}

// ObjectFile Symbol iterators
LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile) {
  symbol_iterator SI = unwrap(ObjectFile)->begin_symbols();
  return wrap(new symbol_iterator(SI));
}

void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSymbolIteratorRef SI) {
  return (*unwrap(SI) == unwrap(ObjectFile)->end_symbols()) ? 1 : 0;
}

void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI) {
  error_code ec;
  unwrap(SI)->increment(ec);
  if (ec) report_fatal_error("LLVMMoveToNextSymbol failed: " + ec.message());
}

// SectionRef accessors
const char *LLVMGetSectionName(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getName(ret))
   report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getContents(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

int LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI,
                                 LLVMSymbolIteratorRef Sym) {
  bool ret;
  if (error_code ec = (*unwrap(SI))->containsSymbol(**unwrap(Sym), ret))
    report_fatal_error(ec.message());
  return ret;
}

// SymbolRef accessors
const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getName(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetSymbolOffset(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getOffset(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

