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

LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf) {
  return wrap(ObjectFile::createObjectFile(unwrap(MemBuf)));
}

void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile) {
  delete unwrap(ObjectFile);
}

LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile) {
  ObjectFile::section_iterator SI = unwrap(ObjectFile)->begin_sections();
  return wrap(new ObjectFile::section_iterator(SI));
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
