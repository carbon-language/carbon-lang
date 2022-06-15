//===-- DWARFASTParserClangTests.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;

namespace {
class DWARFASTParserClangTests : public testing::Test {};

class DWARFASTParserClangStub : public DWARFASTParserClang {
public:
  using DWARFASTParserClang::DWARFASTParserClang;
  using DWARFASTParserClang::LinkDeclContextToDIE;

  std::vector<const clang::DeclContext *> GetDeclContextToDIEMapKeys() {
    std::vector<const clang::DeclContext *> keys;
    for (const auto &it : m_decl_ctx_to_die)
      keys.push_back(it.first);
    return keys;
  }
};
} // namespace

// If your implementation needs to dereference the dummy pointers we are
// defining here, causing this test to fail, feel free to delete it.
TEST_F(DWARFASTParserClangTests,
       EnsureAllDIEsInDeclContextHaveBeenParsedParsesOnlyMatchingEntries) {

  /// Auxiliary debug info.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_encoding
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000008 # DW_ATE_unsigned_char
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  TypeSystemClang ast_ctx("dummy ASTContext", HostInfoBase::GetTargetTriple());
  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFUnit *unit = t.GetDwarfUnit();
  const DWARFDebugInfoEntry *die_first = unit->DIE().GetDIE();
  const DWARFDebugInfoEntry *die_child0 = die_first->GetFirstChild();
  const DWARFDebugInfoEntry *die_child1 = die_child0->GetSibling();
  const DWARFDebugInfoEntry *die_child2 = die_child1->GetSibling();
  const DWARFDebugInfoEntry *die_child3 = die_child2->GetSibling();
  std::vector<DWARFDIE> dies = {
      DWARFDIE(unit, die_child0), DWARFDIE(unit, die_child1),
      DWARFDIE(unit, die_child2), DWARFDIE(unit, die_child3)};
  std::vector<clang::DeclContext *> decl_ctxs = {
      (clang::DeclContext *)1LL, (clang::DeclContext *)2LL,
      (clang::DeclContext *)2LL, (clang::DeclContext *)3LL};
  for (int i = 0; i < 4; ++i)
    ast_parser.LinkDeclContextToDIE(decl_ctxs[i], dies[i]);
  ast_parser.EnsureAllDIEsInDeclContextHaveBeenParsed(
      CompilerDeclContext(nullptr, decl_ctxs[1]));

  EXPECT_THAT(ast_parser.GetDeclContextToDIEMapKeys(),
              testing::UnorderedElementsAre(decl_ctxs[0], decl_ctxs[3]));
}

TEST_F(DWARFASTParserClangTests, TestCallingConventionParsing) {
  // Tests parsing DW_AT_calling_convention values.

  // The DWARF below just declares a list of function types with
  // DW_AT_calling_convention on them.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - func1
    - func2
    - func3
    - func4
    - func5
    - func6
    - func7
    - func8
    - func9
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_calling_convention
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_external
              Form:            DW_FORM_flag_present
  debug_info:
    - Version:         4
      AddrSize:        4
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0xC
        - AbbrCode:        0x2
          Values:
            - Value:           0x0
            - Value:           0x5
            - Value:           0x00
            - Value:           0xCB
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x10
            - Value:           0x5
            - Value:           0x06
            - Value:           0xB3
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x20
            - Value:           0x5
            - Value:           0x0C
            - Value:           0xB1
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x30
            - Value:           0x5
            - Value:           0x12
            - Value:           0xC0
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x40
            - Value:           0x5
            - Value:           0x18
            - Value:           0xB2
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x50
            - Value:           0x5
            - Value:           0x1E
            - Value:           0xC1
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x60
            - Value:           0x5
            - Value:           0x24
            - Value:           0xC2
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x70
            - Value:           0x5
            - Value:           0x2a
            - Value:           0xEE
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x80
            - Value:           0x5
            - Value:           0x30
            - Value:           0x01
            - Value:           0x1
        - AbbrCode:        0x0
...
)";
  YAMLModuleTester t(yamldata);

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  DWARFDIE cu_die(unit, cu_entry);

  TypeSystemClang ast_ctx("dummy ASTContext", HostInfoBase::GetTargetTriple());
  DWARFASTParserClangStub ast_parser(ast_ctx);

  std::vector<std::string> found_function_types;
  // The DWARF above is just a list of functions. Parse all of them to
  // extract the function types and their calling convention values.
  for (DWARFDIE func : cu_die.children()) {
    ASSERT_EQ(func.Tag(), DW_TAG_subprogram);
    SymbolContext sc;
    bool new_type = false;
    lldb::TypeSP type = ast_parser.ParseTypeFromDWARF(sc, func, &new_type);
    found_function_types.push_back(
        type->GetForwardCompilerType().GetTypeName().AsCString());
  }

  // Compare the parsed function types against the expected list of types.
  const std::vector<std::string> expected_function_types = {
      "void () __attribute__((regcall))",
      "void () __attribute__((fastcall))",
      "void () __attribute__((stdcall))",
      "void () __attribute__((vectorcall))",
      "void () __attribute__((pascal))",
      "void () __attribute__((ms_abi))",
      "void () __attribute__((sysv_abi))",
      "void ()", // invalid calling convention.
      "void ()", // DW_CC_normal -> no attribute
  };
  ASSERT_EQ(found_function_types, expected_function_types);
}
