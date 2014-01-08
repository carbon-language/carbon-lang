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

TEST(ToAtomsTest, empty_obj_x86_64) {
  NormalizedFile f;
  f.arch = lld::MachOLinkingContext::arch_x86_64;
  ErrorOr<std::unique_ptr<const lld::File>> atom_f = normalizedToAtoms(f, "");
  EXPECT_FALSE(!atom_f);
  EXPECT_EQ(0U, (*atom_f)->defined().size());
}

TEST(ToAtomsTest, basic_obj_x86_64) {
  NormalizedFile f;
  f.arch = lld::MachOLinkingContext::arch_x86_64;
  Section textSection;
  static const uint8_t contentBytes[] = { 0x90, 0xC3, 0xC3 };
  const unsigned contentSize = sizeof(contentBytes) / sizeof(contentBytes[0]);
  textSection.content.insert(textSection.content.begin(), contentBytes,
                             &contentBytes[contentSize]);
  f.sections.push_back(textSection);
  Symbol fooSymbol;
  fooSymbol.name = "_foo";
  fooSymbol.type = N_SECT;
  fooSymbol.sect = 1;
  fooSymbol.value = 0;
  f.globalSymbols.push_back(fooSymbol);
  Symbol barSymbol;
  barSymbol.name = "_bar";
  barSymbol.type = N_SECT;
  barSymbol.sect = 1;
  barSymbol.value = 2;
  f.globalSymbols.push_back(barSymbol);
  
  ErrorOr<std::unique_ptr<const lld::File>> atom_f = normalizedToAtoms(f, "");
  EXPECT_FALSE(!atom_f);
  const lld::File &file = **atom_f;
  EXPECT_EQ(2U, file.defined().size());
  lld::File::defined_iterator it = file.defined().begin();
  const lld::DefinedAtom *atom1 = *it;
  ++it;
  const lld::DefinedAtom *atom2 = *it;
  EXPECT_TRUE(atom1->name().equals("_foo"));
  EXPECT_EQ(2U, atom1->rawContent().size());
  EXPECT_EQ(0x90, atom1->rawContent()[0]);
  EXPECT_EQ(0xC3, atom1->rawContent()[1]);
  
  EXPECT_TRUE(atom2->name().equals("_bar"));
  EXPECT_EQ(1U, atom2->rawContent().size());
  EXPECT_EQ(0xC3, atom2->rawContent()[0]);
}
