//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/AsmPrinter/DIE.h"
#include "../lib/CodeGen/AsmPrinter/DIEHash.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(Data1, DIEHash) {
  DIEHash Hash;
  DIE Die(dwarf::DW_TAG_base_type);
  DIEInteger Size(4);
  Die.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Size);
  uint64_t MD5Res = Hash.computeTypeSignature(&Die);
  ASSERT_EQ(0x1AFE116E83701108ULL, MD5Res);
}

TEST(TrivialType, DIEHash) {
  // A complete, but simple, type containing no members and defined on the first
  // line of a file.
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  // Line and file number are ignored.
  Unnamed.addValue(dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, &One);
  Unnamed.addValue(dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, &One);
  uint64_t MD5Res = DIEHash().computeTypeSignature(&Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x715305ce6cfd9ad1ULL, MD5Res);
}

TEST(NamedType, DIEHash) {
  // A complete named type containing no members and defined on the first line
  // of a file.
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString FooStr(&One, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  // Line and file number are ignored.
  Foo.addValue(dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, &One);
  Foo.addValue(dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, &One);
  uint64_t MD5Res = DIEHash().computeTypeSignature(&Foo);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xd566dbd2ca5265ffULL, MD5Res);
}

TEST(NamespacedType, DIEHash) {
  // A complete named type containing no members and defined on the first line
  // of a file.
  DIE CU(dwarf::DW_TAG_compile_unit);

  DIE *Space = new DIE(dwarf::DW_TAG_namespace);
  DIEInteger One(1);
  DIEString SpaceStr(&One, "space");
  Space->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &SpaceStr);
  // DW_AT_declaration is ignored.
  Space->addValue(dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present, &One);
  // sibling?

  DIE *Foo = new DIE(dwarf::DW_TAG_structure_type);
  DIEString FooStr(&One, "foo");
  Foo->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);
  Foo->addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  // Line and file number are ignored.
  Foo->addValue(dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, &One);
  Foo->addValue(dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, &One);

  Space->addChild(Foo);
  CU.addChild(Space);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x7b80381fd17f1e33ULL, MD5Res);
}
}
