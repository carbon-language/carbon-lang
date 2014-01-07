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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(DIEHashTest, Data1) {
  DIEHash Hash;
  DIE Die(dwarf::DW_TAG_base_type);
  DIEInteger Size(4);
  Die.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Size);
  uint64_t MD5Res = Hash.computeTypeSignature(Die);
  ASSERT_EQ(0x1AFE116E83701108ULL, MD5Res);
}

// struct {};
TEST(DIEHashTest, TrivialType) {
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  // Line and file number are ignored.
  Unnamed.addValue(dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, &One);
  Unnamed.addValue(dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, &One);
  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x715305ce6cfd9ad1ULL, MD5Res);
}

// struct foo { };
TEST(DIEHashTest, NamedType) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString FooStr(&One, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xd566dbd2ca5265ffULL, MD5Res);
}

// namespace space { struct foo { }; }
TEST(DIEHashTest, NamespacedType) {
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

  Space->addChild(Foo);
  CU.addChild(Space);

  uint64_t MD5Res = DIEHash().computeTypeSignature(*Foo);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x7b80381fd17f1e33ULL, MD5Res);
}

// struct { int member; };
TEST(DIEHashTest, TypeWithMember) {
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger Four(4);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Four);

  DIE *Member = new DIE(dwarf::DW_TAG_member);
  DIEString MemberStr(&Four, "member");
  Member->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemberStr);
  DIEInteger Zero(0);
  Member->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  Unnamed.addChild(Member);

  DIE Int(dwarf::DW_TAG_base_type);
  DIEString IntStr(&Four, "int");
  Int.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &IntStr);
  Int.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Four);
  DIEInteger Five(5);
  Int.addValue(dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, &Five);

  DIEEntry IntRef(&Int);
  Member->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &IntRef);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  ASSERT_EQ(0x5646aa436b7e07c6ULL, MD5Res);
}

// struct foo { int mem1, mem2; };
TEST(DIEHashTest, ReusedType) {
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);

  DIE *Mem1 = new DIE(dwarf::DW_TAG_member);
  DIEInteger Four(4);
  DIEString Mem1Str(&Four, "mem1");
  Mem1->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &Mem1Str);
  DIEInteger Zero(0);
  Mem1->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  Unnamed.addChild(Mem1);

  DIE *Mem2 = new DIE(dwarf::DW_TAG_member);
  DIEString Mem2Str(&Four, "mem2");
  Mem2->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &Mem2Str);
  Mem2->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Four);

  Unnamed.addChild(Mem2);

  DIE Int(dwarf::DW_TAG_base_type);
  DIEString IntStr(&Four, "int");
  Int.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &IntStr);
  Int.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Four);
  DIEInteger Five(5);
  Int.addValue(dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, &Five);

  DIEEntry IntRef(&Int);
  Mem1->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &IntRef);
  Mem2->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &IntRef);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  ASSERT_EQ(0x3a7dc3ed7b76b2f8ULL, MD5Res);
}

// struct foo { static foo f; };
TEST(DIEHashTest, RecursiveType) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);
  DIEString FooStr(&One, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  DIEString MemStr(&One, "mem");
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  DIEEntry FooRef(&Foo);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRef);
  // DW_AT_external and DW_AT_declaration are ignored anyway, so skip them.

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x73d8b25aef227b06ULL, MD5Res);
}

// struct foo { foo *mem; };
TEST(DIEHashTest, Pointer) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEString FooStr(&Eight, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  DIEString MemStr(&Eight, "mem");
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  DIEInteger Zero(0);
  Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  DIE FooPtr(dwarf::DW_TAG_pointer_type);
  FooPtr.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEEntry FooRef(&Foo);
  FooPtr.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRef);

  DIEEntry FooPtrRef(&FooPtr);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooPtrRef);

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x74ea73862e8708d2ULL, MD5Res);
}

// struct foo { foo &mem; };
TEST(DIEHashTest, Reference) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEString FooStr(&Eight, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  DIEString MemStr(&Eight, "mem");
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  DIEInteger Zero(0);
  Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  DIE FooRef(dwarf::DW_TAG_reference_type);
  FooRef.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEEntry FooEntry(&Foo);
  FooRef.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooEntry);

  DIE FooRefConst(dwarf::DW_TAG_const_type);
  DIEEntry FooRefRef(&FooRef);
  FooRefConst.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRefRef);

  DIEEntry FooRefConstRef(&FooRefConst);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRefConstRef);

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0xa0b15f467ad4525bULL, MD5Res);
}

// struct foo { foo &&mem; };
TEST(DIEHashTest, RValueReference) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEString FooStr(&Eight, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  DIEString MemStr(&Eight, "mem");
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  DIEInteger Zero(0);
  Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  DIE FooRef(dwarf::DW_TAG_rvalue_reference_type);
  FooRef.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEEntry FooEntry(&Foo);
  FooRef.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooEntry);

  DIE FooRefConst(dwarf::DW_TAG_const_type);
  DIEEntry FooRefRef(&FooRef);
  FooRefConst.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRefRef);

  DIEEntry FooRefConstRef(&FooRefConst);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooRefConstRef);

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0xad211c8c3b31e57ULL, MD5Res);
}

// struct foo { foo foo::*mem; };
TEST(DIEHashTest, PtrToMember) {
  DIE Foo(dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEString FooStr(&Eight, "foo");
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  DIEString MemStr(&Eight, "mem");
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  DIEInteger Zero(0);
  Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  DIE PtrToFooMem(dwarf::DW_TAG_ptr_to_member_type);
  DIEEntry FooEntry(&Foo);
  PtrToFooMem.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &FooEntry);
  PtrToFooMem.addValue(dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4, &FooEntry);

  DIEEntry PtrToFooMemRef(&PtrToFooMem);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &PtrToFooMemRef);

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x852e0c9ff7c04ebULL, MD5Res);
}

// Check that the hash for a pointer-to-member matches regardless of whether the
// pointed-to type is a declaration or a definition.
//
//   struct bar; // { };
//   struct foo { bar foo::*mem; };
TEST(DIEHashTest, PtrToMemberDeclDefMatch) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr(&Eight, "foo");
  DIEString BarStr(&Eight, "bar");
  DIEString MemStr(&Eight, "mem");
  uint64_t MD5ResDecl;
  {
    DIE Bar(dwarf::DW_TAG_structure_type);
    Bar.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &BarStr);
    Bar.addValue(dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present, &One);

    DIE Foo(dwarf::DW_TAG_structure_type);
    Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
    Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

    DIE *Mem = new DIE(dwarf::DW_TAG_member);
    Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
    Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                  &Zero);

    DIE PtrToFooMem(dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(&Bar);
    PtrToFooMem.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &BarEntry);
    DIEEntry FooEntry(&Foo);
    PtrToFooMem.addValue(dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                         &FooEntry);

    DIEEntry PtrToFooMemRef(&PtrToFooMem);
    Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &PtrToFooMemRef);

    Foo.addChild(Mem);

    MD5ResDecl = DIEHash().computeTypeSignature(Foo);
  }
  uint64_t MD5ResDef;
  {
    DIE Bar(dwarf::DW_TAG_structure_type);
    Bar.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &BarStr);
    Bar.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

    DIE Foo(dwarf::DW_TAG_structure_type);
    Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
    Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

    DIE *Mem = new DIE(dwarf::DW_TAG_member);
    Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
    Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                  &Zero);

    DIE PtrToFooMem(dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(&Bar);
    PtrToFooMem.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &BarEntry);
    DIEEntry FooEntry(&Foo);
    PtrToFooMem.addValue(dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                         &FooEntry);

    DIEEntry PtrToFooMemRef(&PtrToFooMem);
    Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &PtrToFooMemRef);

    Foo.addChild(Mem);

    MD5ResDef = DIEHash().computeTypeSignature(Foo);
  }
  ASSERT_EQ(MD5ResDef, MD5ResDecl);
}

// Check that the hash for a pointer-to-member matches regardless of whether the
// pointed-to type is a declaration or a definition.
//
//   struct bar; // { };
//   struct foo { bar bar::*mem; };
TEST(DIEHashTest, PtrToMemberDeclDefMisMatch) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr(&Eight, "foo");
  DIEString BarStr(&Eight, "bar");
  DIEString MemStr(&Eight, "mem");
  uint64_t MD5ResDecl;
  {
    DIE Bar(dwarf::DW_TAG_structure_type);
    Bar.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &BarStr);
    Bar.addValue(dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present, &One);

    DIE Foo(dwarf::DW_TAG_structure_type);
    Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
    Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

    DIE *Mem = new DIE(dwarf::DW_TAG_member);
    Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
    Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                  &Zero);

    DIE PtrToFooMem(dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(&Bar);
    PtrToFooMem.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &BarEntry);
    PtrToFooMem.addValue(dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                         &BarEntry);

    DIEEntry PtrToFooMemRef(&PtrToFooMem);
    Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &PtrToFooMemRef);

    Foo.addChild(Mem);

    MD5ResDecl = DIEHash().computeTypeSignature(Foo);
  }
  uint64_t MD5ResDef;
  {
    DIE Bar(dwarf::DW_TAG_structure_type);
    Bar.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &BarStr);
    Bar.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

    DIE Foo(dwarf::DW_TAG_structure_type);
    Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
    Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

    DIE *Mem = new DIE(dwarf::DW_TAG_member);
    Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
    Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                  &Zero);

    DIE PtrToFooMem(dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(&Bar);
    PtrToFooMem.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &BarEntry);
    PtrToFooMem.addValue(dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                         &BarEntry);

    DIEEntry PtrToFooMemRef(&PtrToFooMem);
    Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &PtrToFooMemRef);

    Foo.addChild(Mem);

    MD5ResDef = DIEHash().computeTypeSignature(Foo);
  }
  // FIXME: This seems to be a bug in the DWARF type hashing specification that
  // only uses the brief name hashing for types referenced via DW_AT_type. In
  // this case the type is referenced via DW_AT_containing_type and full hashing
  // causes a hash to differ when the containing type is a declaration in one TU
  // and a definition in another.
  ASSERT_NE(MD5ResDef, MD5ResDecl);
}

// struct { } a;
// struct foo { decltype(a) mem; };
TEST(DIEHashTest, RefUnnamedType) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr(&Zero, "foo");
  DIEString MemStr(&Zero, "mem");

  DIE Unnamed(dwarf::DW_TAG_structure_type);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  DIE Foo(dwarf::DW_TAG_structure_type);
  Foo.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  Foo.addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);

  DIE *Mem = new DIE(dwarf::DW_TAG_member);
  Mem->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &MemStr);
  Mem->addValue(dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1, &Zero);

  DIE UnnamedPtr(dwarf::DW_TAG_pointer_type);
  UnnamedPtr.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &Eight);
  DIEEntry UnnamedRef(&Unnamed);
  UnnamedPtr.addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &UnnamedRef);

  DIEEntry UnnamedPtrRef(&UnnamedPtr);
  Mem->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, &UnnamedPtrRef);

  Foo.addChild(Mem);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x954e026f01c02529ULL, MD5Res);
}

// struct { struct bar { }; };
TEST(DIEHashTest, NestedType) {
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  DIE *Foo = new DIE(dwarf::DW_TAG_structure_type);
  DIEString FooStr(&One, "foo");
  Foo->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FooStr);
  Foo->addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  Unnamed.addChild(Foo);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xde8a3b7b43807f4aULL, MD5Res);
}

// struct { static void func(); };
TEST(DIEHashTest, MemberFunc) {
  DIE Unnamed(dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, &One);

  DIE *Func = new DIE(dwarf::DW_TAG_subprogram);
  DIEString FuncStr(&One, "func");
  Func->addValue(dwarf::DW_AT_name, dwarf::DW_FORM_strp, &FuncStr);

  Unnamed.addChild(Func);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xd36a1b6dfb604ba0ULL, MD5Res);
}
}
