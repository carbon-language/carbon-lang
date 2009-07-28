//===-- XCoreTargetObjectFile.cpp - XCore object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetObjectFile.h"
using namespace llvm;


XCoreTargetObjectFile::XCoreTargetObjectFile(bool isXS1A) {
  TextSection = getOrCreateSection("\t.text", true, SectionKind::Text);
  DataSection = getOrCreateSection("\t.dp.data", false, SectionKind::DataRel);
  BSSSection_ = getOrCreateSection("\t.dp.bss", false, SectionKind::BSS);
  
  // TLS globals are lowered in the backend to arrays indexed by the current
  // thread id. After lowering they require no special handling by the linker
  // and can be placed in the standard data / bss sections.
  TLSDataSection = DataSection;
  TLSBSSSection = BSSSection_;
  
  if (isXS1A)
    // FIXME: Why is this writable ("datarel")???
    ReadOnlySection = getOrCreateSection("\t.dp.rodata", false,
                                         SectionKind::DataRel);
  else
    ReadOnlySection = getOrCreateSection("\t.cp.rodata", false,
                                         SectionKind::ReadOnly);
}