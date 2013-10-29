//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARFFormValue.h"
#include "llvm/Support/Dwarf.h"
#include "gtest/gtest.h"
using namespace llvm;
using namespace dwarf;

namespace {

TEST(DWARFFormValue, FixedFormSizes) {
  // Size of DW_FORM_addr and DW_FORM_ref_addr are equal in DWARF2,
  // DW_FORM_ref_addr is always 4 bytes in DWARF32 starting from DWARF3.
  ArrayRef<uint8_t> sizes = DWARFFormValue::getFixedFormSizes(4, 2);
  EXPECT_EQ(sizes[DW_FORM_addr], sizes[DW_FORM_ref_addr]);
  sizes = DWARFFormValue::getFixedFormSizes(8, 2);
  EXPECT_EQ(sizes[DW_FORM_addr], sizes[DW_FORM_ref_addr]);
  sizes = DWARFFormValue::getFixedFormSizes(8, 3);
  EXPECT_EQ(4, sizes[DW_FORM_ref_addr]);
  // Check that we don't have fixed form sizes for weird address sizes.
  EXPECT_EQ(0U, DWARFFormValue::getFixedFormSizes(16, 2).size());
}

bool isFormClass(uint16_t Form, DWARFFormValue::FormClass FC) {
  return DWARFFormValue(Form).isFormClass(FC);
}

TEST(DWARFFormValue, FormClass) {
  EXPECT_TRUE(isFormClass(DW_FORM_addr, DWARFFormValue::FC_Address));
  EXPECT_FALSE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_Address));
  EXPECT_TRUE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_Constant));
  EXPECT_TRUE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_SectionOffset));
  EXPECT_TRUE(
      isFormClass(DW_FORM_sec_offset, DWARFFormValue::FC_SectionOffset));
  EXPECT_TRUE(isFormClass(DW_FORM_GNU_str_index, DWARFFormValue::FC_String));
  EXPECT_TRUE(isFormClass(DW_FORM_GNU_addr_index, DWARFFormValue::FC_Address));
  EXPECT_FALSE(isFormClass(DW_FORM_ref_addr, DWARFFormValue::FC_Address));
  EXPECT_TRUE(isFormClass(DW_FORM_ref_addr, DWARFFormValue::FC_Reference));
  EXPECT_TRUE(isFormClass(DW_FORM_ref_sig8, DWARFFormValue::FC_Reference));
}

} // end anonymous namespace
