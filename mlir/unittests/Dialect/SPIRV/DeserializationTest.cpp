//===- DeserializationTest.cpp - SPIR-V Deserialization Tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of this file is to provide negative deserialization tests.
// For positive deserialization tests, please use serialization and
// deserialization for roundtripping.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "gmock/gmock.h"

#include <memory>

using namespace mlir;

using ::testing::StrEq;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

/// A deserialization test fixture providing minimal SPIR-V building and
/// diagnostic checking utilities.
class DeserializationTest : public ::testing::Test {
protected:
  DeserializationTest() {
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
    // Register a diagnostic handler to capture the diagnostic so that we can
    // check it later.
    context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
      diagnostic.reset(new Diagnostic(std::move(diag)));
    });
  }

  /// Performs deserialization and returns the constructed spv.module op.
  OwningOpRef<spirv::ModuleOp> deserialize() {
    return spirv::deserialize(binary, &context);
  }

  /// Checks there is a diagnostic generated with the given `errorMessage`.
  void expectDiagnostic(StringRef errorMessage) {
    ASSERT_NE(nullptr, diagnostic.get());

    // TODO: check error location too.
    EXPECT_THAT(diagnostic->str(), StrEq(std::string(errorMessage)));
  }

  //===--------------------------------------------------------------------===//
  // SPIR-V builder methods
  //===--------------------------------------------------------------------===//

  /// Adds the SPIR-V module header to `binary`.
  void addHeader() {
    spirv::appendModuleHeader(binary, spirv::Version::V_1_0, /*idBound=*/0);
  }

  /// Adds the SPIR-V instruction into `binary`.
  void addInstruction(spirv::Opcode op, ArrayRef<uint32_t> operands) {
    uint32_t wordCount = 1 + operands.size();
    binary.push_back(spirv::getPrefixedOpcode(wordCount, op));
    binary.append(operands.begin(), operands.end());
  }

  uint32_t addVoidType() {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeVoid, {id});
    return id;
  }

  uint32_t addIntType(uint32_t bitwidth) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeInt, {id, bitwidth, /*signedness=*/1});
    return id;
  }

  uint32_t addStructType(ArrayRef<uint32_t> memberTypes) {
    auto id = nextID++;
    SmallVector<uint32_t, 2> words;
    words.push_back(id);
    words.append(memberTypes.begin(), memberTypes.end());
    addInstruction(spirv::Opcode::OpTypeStruct, words);
    return id;
  }

  uint32_t addFunctionType(uint32_t retType, ArrayRef<uint32_t> paramTypes) {
    auto id = nextID++;
    SmallVector<uint32_t, 4> operands;
    operands.push_back(id);
    operands.push_back(retType);
    operands.append(paramTypes.begin(), paramTypes.end());
    addInstruction(spirv::Opcode::OpTypeFunction, operands);
    return id;
  }

  uint32_t addFunction(uint32_t retType, uint32_t fnType) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpFunction,
                   {retType, id,
                    static_cast<uint32_t>(spirv::FunctionControl::None),
                    fnType});
    return id;
  }

  void addFunctionEnd() { addInstruction(spirv::Opcode::OpFunctionEnd, {}); }

  void addReturn() { addInstruction(spirv::Opcode::OpReturn, {}); }

protected:
  SmallVector<uint32_t, 5> binary;
  uint32_t nextID = 1;
  MLIRContext context;
  std::unique_ptr<Diagnostic> diagnostic;
};

//===----------------------------------------------------------------------===//
// Basics
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, EmptyModuleFailure) {
  ASSERT_FALSE(deserialize());
  expectDiagnostic("SPIR-V binary module must have a 5-word header");
}

TEST_F(DeserializationTest, WrongMagicNumberFailure) {
  addHeader();
  binary.front() = 0xdeadbeef; // Change to a wrong magic number
  ASSERT_FALSE(deserialize());
  expectDiagnostic("incorrect magic number");
}

TEST_F(DeserializationTest, OnlyHeaderSuccess) {
  addHeader();
  EXPECT_TRUE(deserialize());
}

TEST_F(DeserializationTest, ZeroWordCountFailure) {
  addHeader();
  binary.push_back(0); // OpNop with zero word count

  ASSERT_FALSE(deserialize());
  expectDiagnostic("word count cannot be zero");
}

TEST_F(DeserializationTest, InsufficientWordFailure) {
  addHeader();
  binary.push_back((2u << 16) |
                   static_cast<uint32_t>(spirv::Opcode::OpTypeVoid));
  // Missing word for type <id>.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("insufficient words for the last instruction");
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, IntTypeMissingSignednessFailure) {
  addHeader();
  addInstruction(spirv::Opcode::OpTypeInt, {nextID++, 32});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpTypeInt must have bitwidth and signedness parameters");
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, OpMemberNameSuccess) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto structType = addStructType({int32Type, int32Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands1 = {structType, 0};
  (void)spirv::encodeStringLiteralInto(operands1, "i1");
  addInstruction(spirv::Opcode::OpMemberName, operands1);

  SmallVector<uint32_t, 5> operands2 = {structType, 1};
  (void)spirv::encodeStringLiteralInto(operands2, "i2");
  addInstruction(spirv::Opcode::OpMemberName, operands2);

  binary.append(typeDecl.begin(), typeDecl.end());
  EXPECT_TRUE(deserialize());
}

TEST_F(DeserializationTest, OpMemberNameMissingOperands) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto int64Type = addIntType(64);
  auto structType = addStructType({int32Type, int64Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands1 = {structType};
  addInstruction(spirv::Opcode::OpMemberName, operands1);

  binary.append(typeDecl.begin(), typeDecl.end());
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpMemberName must have at least 3 operands");
}

TEST_F(DeserializationTest, OpMemberNameExcessOperands) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto structType = addStructType({int32Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands = {structType, 0};
  (void)spirv::encodeStringLiteralInto(operands, "int32");
  operands.push_back(42);
  addInstruction(spirv::Opcode::OpMemberName, operands);

  binary.append(typeDecl.begin(), typeDecl.end());
  ASSERT_FALSE(deserialize());
  expectDiagnostic("unexpected trailing words in OpMemberName instruction");
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, FunctionMissingEndFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  // Missing OpFunctionEnd.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("expected OpFunctionEnd instruction");
}

TEST_F(DeserializationTest, FunctionMissingParameterFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto i32Type = addIntType(32);
  auto fnType = addFunctionType(voidType, {i32Type});
  addFunction(voidType, fnType);
  // Missing OpFunctionParameter.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("expected OpFunctionParameter instruction");
}

TEST_F(DeserializationTest, FunctionMissingLabelForFirstBlockFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  // Missing OpLabel.
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("a basic block must start with OpLabel");
}

TEST_F(DeserializationTest, FunctionMalformedLabelFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  addInstruction(spirv::Opcode::OpLabel, {}); // Malformed OpLabel
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpLabel should only have result <id>");
}
