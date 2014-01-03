//===- lld/unittest/MachOTests/MachONormalizedFileToAtomsTests.cpp --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <llvm/Support/MachO.h>
#include "../../lib/ReaderWriter/MachO/MachONormalizedFile.h"

#include <assert.h>
#include <vector>

using llvm::ErrorOr;

using namespace lld::mach_o::normalized;
using namespace llvm::MachO;

unsigned countDefinedAtoms(const lld::File &file) {
  unsigned count = 0;
  for (const auto &d : file.defined()) {
    (void)d;
    ++count;
  }
  return count;
}

TEST(ToAtomsTest, empty_obj_x86_64) {
  NormalizedFile f;
  f.arch = lld::MachOLinkingContext::arch_x86_64;
  ErrorOr<std::unique_ptr<const lld::File>> atom_f = normalizedToAtoms(f, "");
  EXPECT_FALSE(!atom_f);
  EXPECT_EQ(0U, countDefinedAtoms(**atom_f));
}

TEST(ToAtomsTest, basic_obj_x86_64) {
  NormalizedFile f;
  f.arch = lld::MachOLinkingContext::arch_x86_64;
  Section textSection;
  static const uint8_t contentBytes[] = { 0x55, 0x48, 0x89, 0xE5,
                                          0x31, 0xC0, 0x5D, 0xC3 };
  const unsigned contentSize = sizeof(contentBytes) / sizeof(contentBytes[0]);
  textSection.content.insert(textSection.content.begin(), contentBytes,
                             &contentBytes[contentSize]);
  f.sections.push_back(textSection);
  Symbol mainSymbol;
  mainSymbol.name = "_main";
  mainSymbol.type = N_SECT;
  mainSymbol.sect = 1;
  f.globalSymbols.push_back(mainSymbol);
  ErrorOr<std::unique_ptr<const lld::File>> atom_f = normalizedToAtoms(f, "");
  EXPECT_FALSE(!atom_f);
  EXPECT_EQ(1U, countDefinedAtoms(**atom_f));
  const lld::DefinedAtom *singleAtom = *(*atom_f)->defined().begin();
  llvm::ArrayRef<uint8_t> atomContent(singleAtom->rawContent());
  EXPECT_EQ(0, memcmp(atomContent.data(), contentBytes, contentSize));
}
