//===- TestAttributes.cpp - MLIR Test Dialect Attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains attributes defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestAttributes.h"
#include "TestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// AttrWithSelfTypeParamAttr
//===----------------------------------------------------------------------===//

Attribute AttrWithSelfTypeParamAttr::parse(MLIRContext *context,
                                           DialectAsmParser &parser,
                                           Type type) {
  Type selfType;
  if (parser.parseType(selfType))
    return Attribute();
  return get(context, selfType);
}

void AttrWithSelfTypeParamAttr::print(DialectAsmPrinter &printer) const {
  printer << "attr_with_self_type_param " << getType();
}

//===----------------------------------------------------------------------===//
// AttrWithTypeBuilderAttr
//===----------------------------------------------------------------------===//

Attribute AttrWithTypeBuilderAttr::parse(MLIRContext *context,
                                         DialectAsmParser &parser, Type type) {
  IntegerAttr element;
  if (parser.parseAttribute(element))
    return Attribute();
  return get(context, element);
}

void AttrWithTypeBuilderAttr::print(DialectAsmPrinter &printer) const {
  printer << "attr_with_type_builder " << getAttr();
}

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute CompoundAAttr::parse(MLIRContext *context, DialectAsmParser &parser,
                               Type type) {
  int widthOfSomething;
  Type oneType;
  SmallVector<int, 4> arrayOfInts;
  if (parser.parseLess() || parser.parseInteger(widthOfSomething) ||
      parser.parseComma() || parser.parseType(oneType) || parser.parseComma() ||
      parser.parseLSquare())
    return Attribute();

  int intVal;
  while (!*parser.parseOptionalInteger(intVal)) {
    arrayOfInts.push_back(intVal);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Attribute();
  return get(context, widthOfSomething, oneType, arrayOfInts);
}

void CompoundAAttr::print(DialectAsmPrinter &printer) const {
  printer << "cmpnd_a<" << getWidthOfSomething() << ", " << getOneType()
          << ", [";
  llvm::interleaveComma(getArrayOfInts(), printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#include "TestAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestAttrDefs.cpp.inc"
      >();
}

Attribute TestDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  {
    Attribute attr;
    auto parseResult =
        generatedAttributeParser(getContext(), parser, attrTag, type, attr);
    if (parseResult.hasValue())
      return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown test attribute");
  return Attribute();
}

void TestDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
