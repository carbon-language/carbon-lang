//===- DialectImplementation.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities classes for implementing dialect attributes and
// types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTIMPLEMENTATION_H
#define MLIR_IR_DIALECTIMPLEMENTATION_H

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class Builder;

//===----------------------------------------------------------------------===//
// DialectAsmPrinter
//===----------------------------------------------------------------------===//

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom printAttribute/printType() method on a
/// dialect.
class DialectAsmPrinter {
public:
  DialectAsmPrinter() {}
  virtual ~DialectAsmPrinter();
  virtual raw_ostream &getStream() const = 0;

  /// Print the given attribute to the stream.
  virtual void printAttribute(Attribute attr) = 0;

  /// Print the given floating point value in a stabilized form that can be
  /// roundtripped through the IR. This is the companion to the 'parseFloat'
  /// hook on the DialectAsmParser.
  virtual void printFloat(const APFloat &value) = 0;

  /// Print the given type to the stream.
  virtual void printType(Type type) = 0;

private:
  DialectAsmPrinter(const DialectAsmPrinter &) = delete;
  void operator=(const DialectAsmPrinter &) = delete;
};

// Make the implementations convenient to use.
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, Attribute attr) {
  p.printAttribute(attr);
  return p;
}

inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p,
                                     const APFloat &value) {
  p.printFloat(value);
  return p;
}
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, float value) {
  return p << APFloat(value);
}
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, double value) {
  return p << APFloat(value);
}

inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, Type type) {
  p.printType(type);
  return p;
}

// Support printing anything that isn't convertible to one of the above types,
// even if it isn't exactly one of them.  For example, we want to print
// FunctionType with the Type version above, not have it match this.
template <typename T, typename std::enable_if<
                          !std::is_convertible<T &, Attribute &>::value &&
                              !std::is_convertible<T &, Type &>::value &&
                              !std::is_convertible<T &, APFloat &>::value &&
                              !llvm::is_one_of<T, double, float>::value,
                          T>::type * = nullptr>
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, const T &other) {
  p.getStream() << other;
  return p;
}

//===----------------------------------------------------------------------===//
// DialectAsmParser
//===----------------------------------------------------------------------===//

/// The DialectAsmParser has methods for interacting with the asm parser:
/// parsing things from it, emitting errors etc.  It has an intentionally
/// high-level API that is designed to reduce/constrain syntax innovation in
/// individual attributes or types.
class DialectAsmParser {
public:
  virtual ~DialectAsmParser();

  /// Emit a diagnostic at the specified location and return failure.
  virtual InFlightDiagnostic emitError(llvm::SMLoc loc,
                                       const Twine &message = {}) = 0;

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  virtual Builder &getBuilder() const = 0;

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  virtual llvm::SMLoc getCurrentLocation() = 0;
  ParseResult getCurrentLocation(llvm::SMLoc *loc) {
    *loc = getCurrentLocation();
    return success();
  }

  /// Return the location of the original name token.
  virtual llvm::SMLoc getNameLoc() const = 0;

  /// Re-encode the given source location as an MLIR location and return it.
  /// Note: This method should only be used when a `Location` is necessary, as
  /// the encoding process is not efficient. In other cases a more suitable
  /// alternative should be used, such as the `getChecked` methods defined
  /// below.
  virtual Location getEncodedSourceLoc(llvm::SMLoc loc) = 0;

  /// Returns the full specification of the symbol being parsed. This allows for
  /// using a separate parser if necessary.
  virtual StringRef getFullSymbolSpec() const = 0;

  // These methods emit an error and return failure or success. This allows
  // these to be chained together into a linear sequence of || expressions in
  // many cases.

  /// Parse a floating point value from the stream.
  virtual ParseResult parseFloat(double &result) = 0;

  /// Parse an integer value from the stream.
  template <typename IntT>
  ParseResult parseInteger(IntT &result) {
    auto loc = getCurrentLocation();
    OptionalParseResult parseResult = parseOptionalInteger(result);
    if (!parseResult.hasValue())
      return emitError(loc, "expected integer value");
    return *parseResult;
  }

  /// Parse an optional integer value from the stream.
  virtual OptionalParseResult parseOptionalInteger(APInt &result) = 0;

  template <typename IntT>
  OptionalParseResult parseOptionalInteger(IntT &result) {
    auto loc = getCurrentLocation();

    // Parse the unsigned variant.
    APInt uintResult;
    OptionalParseResult parseResult = parseOptionalInteger(uintResult);
    if (!parseResult.hasValue() || failed(*parseResult))
      return parseResult;

    // Try to convert to the provided integer type.  sextOrTrunc is correct even
    // for unsigned types because parseOptionalInteger ensures the sign bit is
    // zero for non-negated integers.
    result =
        (IntT)uintResult.sextOrTrunc(sizeof(IntT) * CHAR_BIT).getLimitedValue();
    if (APInt(uintResult.getBitWidth(), result) != uintResult)
      return emitError(loc, "integer value too large");
    return success();
  }

  /// Invoke the `getChecked` method of the given Attribute or Type class, using
  /// the provided location to emit errors in the case of failure. Note that
  /// unlike `OpBuilder::getType`, this method does not implicitly insert a
  /// context parameter.
  template <typename T, typename... ParamsT>
  T getChecked(llvm::SMLoc loc, ParamsT &&... params) {
    return T::getChecked([&] { return emitError(loc); },
                         std::forward<ParamsT>(params)...);
  }
  /// A variant of `getChecked` that uses the result of `getNameLoc` to emit
  /// errors.
  template <typename T, typename... ParamsT>
  T getChecked(ParamsT &&... params) {
    return T::getChecked([&] { return emitError(getNameLoc()); },
                         std::forward<ParamsT>(params)...);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a '->' token.
  virtual ParseResult parseArrow() = 0;

  /// Parse a '->' token if present
  virtual ParseResult parseOptionalArrow() = 0;

  /// Parse a '{' token.
  virtual ParseResult parseLBrace() = 0;

  /// Parse a '{' token if present
  virtual ParseResult parseOptionalLBrace() = 0;

  /// Parse a `}` token.
  virtual ParseResult parseRBrace() = 0;

  /// Parse a `}` token if present
  virtual ParseResult parseOptionalRBrace() = 0;

  /// Parse a `:` token.
  virtual ParseResult parseColon() = 0;

  /// Parse a `:` token if present.
  virtual ParseResult parseOptionalColon() = 0;

  /// Parse a `,` token.
  virtual ParseResult parseComma() = 0;

  /// Parse a `,` token if present.
  virtual ParseResult parseOptionalComma() = 0;

  /// Parse a `=` token.
  virtual ParseResult parseEqual() = 0;

  /// Parse a `=` token if present.
  virtual ParseResult parseOptionalEqual() = 0;

  /// Parse a quoted string token.
  ParseResult parseString(std::string *string) {
    auto loc = getCurrentLocation();
    if (parseOptionalString(string))
      return emitError(loc, "expected string");
    return success();
  }

  /// Parse a quoted string token if present.
  virtual ParseResult parseOptionalString(std::string *string) = 0;

  /// Parse a given keyword.
  ParseResult parseKeyword(StringRef keyword, const Twine &msg = "") {
    auto loc = getCurrentLocation();
    if (parseOptionalKeyword(keyword))
      return emitError(loc, "expected '") << keyword << "'" << msg;
    return success();
  }

  /// Parse a keyword into 'keyword'.
  ParseResult parseKeyword(StringRef *keyword) {
    auto loc = getCurrentLocation();
    if (parseOptionalKeyword(keyword))
      return emitError(loc, "expected valid keyword");
    return success();
  }

  /// Parse the given keyword if present.
  virtual ParseResult parseOptionalKeyword(StringRef keyword) = 0;

  /// Parse a keyword, if present, into 'keyword'.
  virtual ParseResult parseOptionalKeyword(StringRef *keyword) = 0;

  /// Parse a '<' token.
  virtual ParseResult parseLess() = 0;

  /// Parse a `<` token if present.
  virtual ParseResult parseOptionalLess() = 0;

  /// Parse a '>' token.
  virtual ParseResult parseGreater() = 0;

  /// Parse a `>` token if present.
  virtual ParseResult parseOptionalGreater() = 0;

  /// Parse a `(` token.
  virtual ParseResult parseLParen() = 0;

  /// Parse a `(` token if present.
  virtual ParseResult parseOptionalLParen() = 0;

  /// Parse a `)` token.
  virtual ParseResult parseRParen() = 0;

  /// Parse a `)` token if present.
  virtual ParseResult parseOptionalRParen() = 0;

  /// Parse a `[` token.
  virtual ParseResult parseLSquare() = 0;

  /// Parse a `[` token if present.
  virtual ParseResult parseOptionalLSquare() = 0;

  /// Parse a `]` token.
  virtual ParseResult parseRSquare() = 0;

  /// Parse a `]` token if present.
  virtual ParseResult parseOptionalRSquare() = 0;

  /// Parse a `...` token if present;
  virtual ParseResult parseOptionalEllipsis() = 0;

  /// Parse a `?` token.
  virtual ParseResult parseOptionalQuestion() = 0;

  /// Parse a `*` token.
  virtual ParseResult parseOptionalStar() = 0;

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute and return it in result.
  virtual ParseResult parseAttribute(Attribute &result, Type type = {}) = 0;

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, Type type = {}) {
    llvm::SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseAttribute(attr, type))
      return failure();

    // Check for the right kind of attribute.
    result = attr.dyn_cast<AttrType>();
    if (!result)
      return emitError(loc, "invalid kind of attribute specified");
    return success();
  }

  /// Parse an affine map instance into 'map'.
  virtual ParseResult parseAffineMap(AffineMap &map) = 0;

  /// Parse an integer set instance into 'set'.
  virtual ParseResult printIntegerSet(IntegerSet &set) = 0;

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a type.
  virtual ParseResult parseType(Type &result) = 0;

  /// Parse a type of a specific kind, e.g. a FunctionType.
  template <typename TypeType>
  ParseResult parseType(TypeType &result) {
    llvm::SMLoc loc = getCurrentLocation();

    // Parse any kind of type.
    Type type;
    if (parseType(type))
      return failure();

    // Check for the right kind of attribute.
    result = type.dyn_cast<TypeType>();
    if (!result)
      return emitError(loc, "invalid kind of type specified");
    return success();
  }

  /// Parse a type if present.
  virtual OptionalParseResult parseOptionalType(Type &result) = 0;

  /// Parse a 'x' separated dimension list. This populates the dimension list,
  /// using -1 for the `?` dimensions if `allowDynamic` is set and errors out on
  /// `?` otherwise.
  ///
  ///   dimension-list ::= (dimension `x`)*
  ///   dimension ::= `?` | integer
  ///
  /// When `allowDynamic` is not set, this is used to parse:
  ///
  ///   static-dimension-list ::= (integer `x`)*
  virtual ParseResult parseDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                         bool allowDynamic = true) = 0;

  /// Parse an 'x' token in a dimension list, handling the case where the x is
  /// juxtaposed with an element type, as in "xf32", leaving the "f32" as the
  /// next token.
  virtual ParseResult parseXInDimensionList() = 0;
};

} // end namespace mlir

#endif
