//===- AttributeParser.cpp - MLIR Attribute Parser Implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the MLIR Types.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Parser/AsmParserState.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"

using namespace mlir;
using namespace mlir::detail;

/// Parse an arbitrary attribute.
///
///  attribute-value ::= `unit`
///                    | bool-literal
///                    | integer-literal (`:` (index-type | integer-type))?
///                    | float-literal (`:` float-type)?
///                    | string-literal (`:` type)?
///                    | type
///                    | `[` (attribute-value (`,` attribute-value)*)? `]`
///                    | `{` (attribute-entry (`,` attribute-entry)*)? `}`
///                    | symbol-ref-id (`::` symbol-ref-id)*
///                    | `dense` `<` attribute-value `>` `:`
///                      (tensor-type | vector-type)
///                    | `sparse` `<` attribute-value `,` attribute-value `>`
///                      `:` (tensor-type | vector-type)
///                    | `opaque` `<` dialect-namespace  `,` hex-string-literal
///                      `>` `:` (tensor-type | vector-type)
///                    | extended-attribute
///
Attribute Parser::parseAttribute(Type type) {
  switch (getToken().getKind()) {
  // Parse an AffineMap or IntegerSet attribute.
  case Token::kw_affine_map: {
    consumeToken(Token::kw_affine_map);

    AffineMap map;
    if (parseToken(Token::less, "expected '<' in affine map") ||
        parseAffineMapReference(map) ||
        parseToken(Token::greater, "expected '>' in affine map"))
      return Attribute();
    return AffineMapAttr::get(map);
  }
  case Token::kw_affine_set: {
    consumeToken(Token::kw_affine_set);

    IntegerSet set;
    if (parseToken(Token::less, "expected '<' in integer set") ||
        parseIntegerSetReference(set) ||
        parseToken(Token::greater, "expected '>' in integer set"))
      return Attribute();
    return IntegerSetAttr::get(set);
  }

  // Parse an array attribute.
  case Token::l_square: {
    consumeToken(Token::l_square);

    SmallVector<Attribute, 4> elements;
    auto parseElt = [&]() -> ParseResult {
      elements.push_back(parseAttribute());
      return elements.back() ? success() : failure();
    };

    if (parseCommaSeparatedListUntil(Token::r_square, parseElt))
      return nullptr;
    return builder.getArrayAttr(elements);
  }

  // Parse a boolean attribute.
  case Token::kw_false:
    consumeToken(Token::kw_false);
    return builder.getBoolAttr(false);
  case Token::kw_true:
    consumeToken(Token::kw_true);
    return builder.getBoolAttr(true);

  // Parse a dense elements attribute.
  case Token::kw_dense:
    return parseDenseElementsAttr(type);

  // Parse a dictionary attribute.
  case Token::l_brace: {
    NamedAttrList elements;
    if (parseAttributeDict(elements))
      return nullptr;
    return elements.getDictionary(getContext());
  }

  // Parse an extended attribute, i.e. alias or dialect attribute.
  case Token::hash_identifier:
    return parseExtendedAttr(type);

  // Parse floating point and integer attributes.
  case Token::floatliteral:
    return parseFloatAttr(type, /*isNegative=*/false);
  case Token::integer:
    return parseDecOrHexAttr(type, /*isNegative=*/false);
  case Token::minus: {
    consumeToken(Token::minus);
    if (getToken().is(Token::integer))
      return parseDecOrHexAttr(type, /*isNegative=*/true);
    if (getToken().is(Token::floatliteral))
      return parseFloatAttr(type, /*isNegative=*/true);

    return (emitError("expected constant integer or floating point value"),
            nullptr);
  }

  // Parse a location attribute.
  case Token::kw_loc: {
    consumeToken(Token::kw_loc);

    LocationAttr locAttr;
    if (parseToken(Token::l_paren, "expected '(' in inline location") ||
        parseLocationInstance(locAttr) ||
        parseToken(Token::r_paren, "expected ')' in inline location"))
      return Attribute();
    return locAttr;
  }

  // Parse an opaque elements attribute.
  case Token::kw_opaque:
    return parseOpaqueElementsAttr(type);

  // Parse a sparse elements attribute.
  case Token::kw_sparse:
    return parseSparseElementsAttr(type);

  // Parse a string attribute.
  case Token::string: {
    auto val = getToken().getStringValue();
    consumeToken(Token::string);
    // Parse the optional trailing colon type if one wasn't explicitly provided.
    if (!type && consumeIf(Token::colon) && !(type = parseType()))
      return Attribute();

    return type ? StringAttr::get(val, type)
                : StringAttr::get(getContext(), val);
  }

  // Parse a symbol reference attribute.
  case Token::at_identifier: {
    // When populating the parser state, this is a list of locations for all of
    // the nested references.
    SmallVector<llvm::SMRange> referenceLocations;
    if (state.asmState)
      referenceLocations.push_back(getToken().getLocRange());

    // Parse the top-level reference.
    std::string nameStr = getToken().getSymbolReference();
    consumeToken(Token::at_identifier);

    // Parse any nested references.
    std::vector<FlatSymbolRefAttr> nestedRefs;
    while (getToken().is(Token::colon)) {
      // Check for the '::' prefix.
      const char *curPointer = getToken().getLoc().getPointer();
      consumeToken(Token::colon);
      if (!consumeIf(Token::colon)) {
        state.lex.resetPointer(curPointer);
        consumeToken();
        break;
      }
      // Parse the reference itself.
      auto curLoc = getToken().getLoc();
      if (getToken().isNot(Token::at_identifier)) {
        emitError(curLoc, "expected nested symbol reference identifier");
        return Attribute();
      }

      // If we are populating the assembly state, add the location for this
      // reference.
      if (state.asmState)
        referenceLocations.push_back(getToken().getLocRange());

      std::string nameStr = getToken().getSymbolReference();
      consumeToken(Token::at_identifier);
      nestedRefs.push_back(SymbolRefAttr::get(getContext(), nameStr));
    }
    SymbolRefAttr symbolRefAttr = builder.getSymbolRefAttr(nameStr, nestedRefs);

    // If we are populating the assembly state, record this symbol reference.
    if (state.asmState)
      state.asmState->addUses(symbolRefAttr, referenceLocations);
    return symbolRefAttr;
  }

  // Parse a 'unit' attribute.
  case Token::kw_unit:
    consumeToken(Token::kw_unit);
    return builder.getUnitAttr();

  default:
    // Parse a type attribute.
    if (Type type = parseType())
      return TypeAttr::get(type);
    return nullptr;
  }
}

/// Parse an optional attribute with the provided type.
OptionalParseResult Parser::parseOptionalAttribute(Attribute &attribute,
                                                   Type type) {
  switch (getToken().getKind()) {
  case Token::at_identifier:
  case Token::floatliteral:
  case Token::integer:
  case Token::hash_identifier:
  case Token::kw_affine_map:
  case Token::kw_affine_set:
  case Token::kw_dense:
  case Token::kw_false:
  case Token::kw_loc:
  case Token::kw_opaque:
  case Token::kw_sparse:
  case Token::kw_true:
  case Token::kw_unit:
  case Token::l_brace:
  case Token::l_square:
  case Token::minus:
  case Token::string:
    attribute = parseAttribute(type);
    return success(attribute != nullptr);

  default:
    // Parse an optional type attribute.
    Type type;
    OptionalParseResult result = parseOptionalType(type);
    if (result.hasValue() && succeeded(*result))
      attribute = TypeAttr::get(type);
    return result;
  }
}
OptionalParseResult Parser::parseOptionalAttribute(ArrayAttr &attribute,
                                                   Type type) {
  return parseOptionalAttributeWithToken(Token::l_square, attribute, type);
}
OptionalParseResult Parser::parseOptionalAttribute(StringAttr &attribute,
                                                   Type type) {
  return parseOptionalAttributeWithToken(Token::string, attribute, type);
}

/// Attribute dictionary.
///
///   attribute-dict ::= `{` `}`
///                    | `{` attribute-entry (`,` attribute-entry)* `}`
///   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
///
ParseResult Parser::parseAttributeDict(NamedAttrList &attributes) {
  if (parseToken(Token::l_brace, "expected '{' in attribute dictionary"))
    return failure();

  llvm::SmallDenseSet<Identifier> seenKeys;
  auto parseElt = [&]() -> ParseResult {
    // The name of an attribute can either be a bare identifier, or a string.
    Optional<Identifier> nameId;
    if (getToken().is(Token::string))
      nameId = builder.getIdentifier(getToken().getStringValue());
    else if (getToken().isAny(Token::bare_identifier, Token::inttype) ||
             getToken().isKeyword())
      nameId = builder.getIdentifier(getTokenSpelling());
    else
      return emitError("expected attribute name");
    if (!seenKeys.insert(*nameId).second)
      return emitError("duplicate key '")
             << *nameId << "' in dictionary attribute";
    consumeToken();

    // Lazy load a dialect in the context if there is a possible namespace.
    auto splitName = nameId->strref().split('.');
    if (!splitName.second.empty())
      getContext()->getOrLoadDialect(splitName.first);

    // Try to parse the '=' for the attribute value.
    if (!consumeIf(Token::equal)) {
      // If there is no '=', we treat this as a unit attribute.
      attributes.push_back({*nameId, builder.getUnitAttr()});
      return success();
    }

    auto attr = parseAttribute();
    if (!attr)
      return failure();
    attributes.push_back({*nameId, attr});
    return success();
  };

  if (parseCommaSeparatedListUntil(Token::r_brace, parseElt))
    return failure();

  return success();
}

/// Parse a float attribute.
Attribute Parser::parseFloatAttr(Type type, bool isNegative) {
  auto val = getToken().getFloatingPointValue();
  if (!val.hasValue())
    return (emitError("floating point value too large for attribute"), nullptr);
  consumeToken(Token::floatliteral);
  if (!type) {
    // Default to F64 when no type is specified.
    if (!consumeIf(Token::colon))
      type = builder.getF64Type();
    else if (!(type = parseType()))
      return nullptr;
  }
  if (!type.isa<FloatType>())
    return (emitError("floating point value not valid for specified type"),
            nullptr);
  return FloatAttr::get(type, isNegative ? -val.getValue() : val.getValue());
}

/// Construct an APint from a parsed value, a known attribute type and
/// sign.
static Optional<APInt> buildAttributeAPInt(Type type, bool isNegative,
                                           StringRef spelling) {
  // Parse the integer value into an APInt that is big enough to hold the value.
  APInt result;
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return llvm::None;

  // Extend or truncate the bitwidth to the right size.
  unsigned width = type.isIndex() ? IndexType::kInternalStorageBitWidth
                                  : type.getIntOrFloatBitWidth();

  // APInt cannot hold a zero bit value.
  if (width == 0)
    return llvm::None;

  if (width > result.getBitWidth()) {
    result = result.zext(width);
  } else if (width < result.getBitWidth()) {
    // The parser can return an unnecessarily wide result with leading zeros.
    // This isn't a problem, but truncating off bits is bad.
    if (result.countLeadingZeros() < result.getBitWidth() - width)
      return llvm::None;

    result = result.trunc(width);
  }

  if (isNegative) {
    // The value is negative, we have an overflow if the sign bit is not set
    // in the negated apInt.
    result.negate();
    if (!result.isSignBitSet())
      return llvm::None;
  } else if ((type.isSignedInteger() || type.isIndex()) &&
             result.isSignBitSet()) {
    // The value is a positive signed integer or index,
    // we have an overflow if the sign bit is set.
    return llvm::None;
  }

  return result;
}

/// Parse a decimal or a hexadecimal literal, which can be either an integer
/// or a float attribute.
Attribute Parser::parseDecOrHexAttr(Type type, bool isNegative) {
  Token tok = getToken();
  StringRef spelling = tok.getSpelling();
  llvm::SMLoc loc = tok.getLoc();

  consumeToken(Token::integer);
  if (!type) {
    // Default to i64 if not type is specified.
    if (!consumeIf(Token::colon))
      type = builder.getIntegerType(64);
    else if (!(type = parseType()))
      return nullptr;
  }

  if (auto floatType = type.dyn_cast<FloatType>()) {
    Optional<APFloat> result;
    if (failed(parseFloatFromIntegerLiteral(result, tok, isNegative,
                                            floatType.getFloatSemantics(),
                                            floatType.getWidth())))
      return Attribute();
    return FloatAttr::get(floatType, *result);
  }

  if (!type.isa<IntegerType, IndexType>())
    return emitError(loc, "integer literal not valid for specified type"),
           nullptr;

  if (isNegative && type.isUnsignedInteger()) {
    emitError(loc,
              "negative integer literal not valid for unsigned integer type");
    return nullptr;
  }

  Optional<APInt> apInt = buildAttributeAPInt(type, isNegative, spelling);
  if (!apInt)
    return emitError(loc, "integer constant out of range for attribute"),
           nullptr;
  return builder.getIntegerAttr(type, *apInt);
}

//===----------------------------------------------------------------------===//
// TensorLiteralParser
//===----------------------------------------------------------------------===//

/// Parse elements values stored within a hex string. On success, the values are
/// stored into 'result'.
static ParseResult parseElementAttrHexValues(Parser &parser, Token tok,
                                             std::string &result) {
  if (Optional<std::string> value = tok.getHexStringValue()) {
    result = std::move(*value);
    return success();
  }
  return parser.emitError(
      tok.getLoc(), "expected string containing hex digits starting with `0x`");
}

namespace {
/// This class implements a parser for TensorLiterals. A tensor literal is
/// either a single element (e.g, 5) or a multi-dimensional list of elements
/// (e.g., [[5, 5]]).
class TensorLiteralParser {
public:
  TensorLiteralParser(Parser &p) : p(p) {}

  /// Parse the elements of a tensor literal. If 'allowHex' is true, the parser
  /// may also parse a tensor literal that is store as a hex string.
  ParseResult parse(bool allowHex);

  /// Build a dense attribute instance with the parsed elements and the given
  /// shaped type.
  DenseElementsAttr getAttr(llvm::SMLoc loc, ShapedType type);

  ArrayRef<int64_t> getShape() const { return shape; }

private:
  /// Get the parsed elements for an integer attribute.
  ParseResult getIntAttrElements(llvm::SMLoc loc, Type eltTy,
                                 std::vector<APInt> &intValues);

  /// Get the parsed elements for a float attribute.
  ParseResult getFloatAttrElements(llvm::SMLoc loc, FloatType eltTy,
                                   std::vector<APFloat> &floatValues);

  /// Build a Dense String attribute for the given type.
  DenseElementsAttr getStringAttr(llvm::SMLoc loc, ShapedType type, Type eltTy);

  /// Build a Dense attribute with hex data for the given type.
  DenseElementsAttr getHexAttr(llvm::SMLoc loc, ShapedType type);

  /// Parse a single element, returning failure if it isn't a valid element
  /// literal. For example:
  /// parseElement(1) -> Success, 1
  /// parseElement([1]) -> Failure
  ParseResult parseElement();

  /// Parse a list of either lists or elements, returning the dimensions of the
  /// parsed sub-tensors in dims. For example:
  ///   parseList([1, 2, 3]) -> Success, [3]
  ///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
  ///   parseList([[1, 2], 3]) -> Failure
  ///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
  ParseResult parseList(SmallVectorImpl<int64_t> &dims);

  /// Parse a literal that was printed as a hex string.
  ParseResult parseHexElements();

  Parser &p;

  /// The shape inferred from the parsed elements.
  SmallVector<int64_t, 4> shape;

  /// Storage used when parsing elements, this is a pair of <is_negated, token>.
  std::vector<std::pair<bool, Token>> storage;

  /// Storage used when parsing elements that were stored as hex values.
  Optional<Token> hexStorage;
};
} // end anonymous namespace

/// Parse the elements of a tensor literal. If 'allowHex' is true, the parser
/// may also parse a tensor literal that is store as a hex string.
ParseResult TensorLiteralParser::parse(bool allowHex) {
  // If hex is allowed, check for a string literal.
  if (allowHex && p.getToken().is(Token::string)) {
    hexStorage = p.getToken();
    p.consumeToken(Token::string);
    return success();
  }
  // Otherwise, parse a list or an individual element.
  if (p.getToken().is(Token::l_square))
    return parseList(shape);
  return parseElement();
}

/// Build a dense attribute instance with the parsed elements and the given
/// shaped type.
DenseElementsAttr TensorLiteralParser::getAttr(llvm::SMLoc loc,
                                               ShapedType type) {
  Type eltType = type.getElementType();

  // Check to see if we parse the literal from a hex string.
  if (hexStorage.hasValue() &&
      (eltType.isIntOrFloat() || eltType.isa<ComplexType>()))
    return getHexAttr(loc, type);

  // Check that the parsed storage size has the same number of elements to the
  // type, or is a known splat.
  if (!shape.empty() && getShape() != type.getShape()) {
    p.emitError(loc) << "inferred shape of elements literal ([" << getShape()
                     << "]) does not match type ([" << type.getShape() << "])";
    return nullptr;
  }

  // Handle the case where no elements were parsed.
  if (!hexStorage.hasValue() && storage.empty() && type.getNumElements()) {
    p.emitError(loc) << "parsed zero elements, but type (" << type
                     << ") expected at least 1";
    return nullptr;
  }

  // Handle complex types in the specific element type cases below.
  bool isComplex = false;
  if (ComplexType complexTy = eltType.dyn_cast<ComplexType>()) {
    eltType = complexTy.getElementType();
    isComplex = true;
  }

  // Handle integer and index types.
  if (eltType.isIntOrIndex()) {
    std::vector<APInt> intValues;
    if (failed(getIntAttrElements(loc, eltType, intValues)))
      return nullptr;
    if (isComplex) {
      // If this is a complex, treat the parsed values as complex values.
      auto complexData = llvm::makeArrayRef(
          reinterpret_cast<std::complex<APInt> *>(intValues.data()),
          intValues.size() / 2);
      return DenseElementsAttr::get(type, complexData);
    }
    return DenseElementsAttr::get(type, intValues);
  }
  // Handle floating point types.
  if (FloatType floatTy = eltType.dyn_cast<FloatType>()) {
    std::vector<APFloat> floatValues;
    if (failed(getFloatAttrElements(loc, floatTy, floatValues)))
      return nullptr;
    if (isComplex) {
      // If this is a complex, treat the parsed values as complex values.
      auto complexData = llvm::makeArrayRef(
          reinterpret_cast<std::complex<APFloat> *>(floatValues.data()),
          floatValues.size() / 2);
      return DenseElementsAttr::get(type, complexData);
    }
    return DenseElementsAttr::get(type, floatValues);
  }

  // Other types are assumed to be string representations.
  return getStringAttr(loc, type, type.getElementType());
}

/// Build a Dense Integer attribute for the given type.
ParseResult
TensorLiteralParser::getIntAttrElements(llvm::SMLoc loc, Type eltTy,
                                        std::vector<APInt> &intValues) {
  intValues.reserve(storage.size());
  bool isUintType = eltTy.isUnsignedInteger();
  for (const auto &signAndToken : storage) {
    bool isNegative = signAndToken.first;
    const Token &token = signAndToken.second;
    auto tokenLoc = token.getLoc();

    if (isNegative && isUintType) {
      return p.emitError(tokenLoc)
             << "expected unsigned integer elements, but parsed negative value";
    }

    // Check to see if floating point values were parsed.
    if (token.is(Token::floatliteral)) {
      return p.emitError(tokenLoc)
             << "expected integer elements, but parsed floating-point";
    }

    assert(token.isAny(Token::integer, Token::kw_true, Token::kw_false) &&
           "unexpected token type");
    if (token.isAny(Token::kw_true, Token::kw_false)) {
      if (!eltTy.isInteger(1)) {
        return p.emitError(tokenLoc)
               << "expected i1 type for 'true' or 'false' values";
      }
      APInt apInt(1, token.is(Token::kw_true), /*isSigned=*/false);
      intValues.push_back(apInt);
      continue;
    }

    // Create APInt values for each element with the correct bitwidth.
    Optional<APInt> apInt =
        buildAttributeAPInt(eltTy, isNegative, token.getSpelling());
    if (!apInt)
      return p.emitError(tokenLoc, "integer constant out of range for type");
    intValues.push_back(*apInt);
  }
  return success();
}

/// Build a Dense Float attribute for the given type.
ParseResult
TensorLiteralParser::getFloatAttrElements(llvm::SMLoc loc, FloatType eltTy,
                                          std::vector<APFloat> &floatValues) {
  floatValues.reserve(storage.size());
  for (const auto &signAndToken : storage) {
    bool isNegative = signAndToken.first;
    const Token &token = signAndToken.second;

    // Handle hexadecimal float literals.
    if (token.is(Token::integer) && token.getSpelling().startswith("0x")) {
      Optional<APFloat> result;
      if (failed(p.parseFloatFromIntegerLiteral(result, token, isNegative,
                                                eltTy.getFloatSemantics(),
                                                eltTy.getWidth())))
        return failure();

      floatValues.push_back(*result);
      continue;
    }

    // Check to see if any decimal integers or booleans were parsed.
    if (!token.is(Token::floatliteral))
      return p.emitError()
             << "expected floating-point elements, but parsed integer";

    // Build the float values from tokens.
    auto val = token.getFloatingPointValue();
    if (!val.hasValue())
      return p.emitError("floating point value too large for attribute");

    APFloat apVal(isNegative ? -*val : *val);
    if (!eltTy.isF64()) {
      bool unused;
      apVal.convert(eltTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                    &unused);
    }
    floatValues.push_back(apVal);
  }
  return success();
}

/// Build a Dense String attribute for the given type.
DenseElementsAttr TensorLiteralParser::getStringAttr(llvm::SMLoc loc,
                                                     ShapedType type,
                                                     Type eltTy) {
  if (hexStorage.hasValue()) {
    auto stringValue = hexStorage.getValue().getStringValue();
    return DenseStringElementsAttr::get(type, {stringValue});
  }

  std::vector<std::string> stringValues;
  std::vector<StringRef> stringRefValues;
  stringValues.reserve(storage.size());
  stringRefValues.reserve(storage.size());

  for (auto val : storage) {
    stringValues.push_back(val.second.getStringValue());
    stringRefValues.push_back(stringValues.back());
  }

  return DenseStringElementsAttr::get(type, stringRefValues);
}

/// Build a Dense attribute with hex data for the given type.
DenseElementsAttr TensorLiteralParser::getHexAttr(llvm::SMLoc loc,
                                                  ShapedType type) {
  Type elementType = type.getElementType();
  if (!elementType.isIntOrIndexOrFloat() && !elementType.isa<ComplexType>()) {
    p.emitError(loc)
        << "expected floating-point, integer, or complex element type, got "
        << elementType;
    return nullptr;
  }

  std::string data;
  if (parseElementAttrHexValues(p, hexStorage.getValue(), data))
    return nullptr;

  ArrayRef<char> rawData(data.data(), data.size());
  bool detectedSplat = false;
  if (!DenseElementsAttr::isValidRawBuffer(type, rawData, detectedSplat)) {
    p.emitError(loc) << "elements hex data size is invalid for provided type: "
                     << type;
    return nullptr;
  }

  if (llvm::support::endian::system_endianness() ==
      llvm::support::endianness::big) {
    // Convert endianess in big-endian(BE) machines. `rawData` is
    // little-endian(LE) because HEX in raw data of dense element attribute
    // is always LE format. It is converted into BE here to be used in BE
    // machines.
    SmallVector<char, 64> outDataVec(rawData.size());
    MutableArrayRef<char> convRawData(outDataVec);
    DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
        rawData, convRawData, type);
    return DenseElementsAttr::getFromRawBuffer(type, convRawData,
                                               detectedSplat);
  }

  return DenseElementsAttr::getFromRawBuffer(type, rawData, detectedSplat);
}

ParseResult TensorLiteralParser::parseElement() {
  switch (p.getToken().getKind()) {
  // Parse a boolean element.
  case Token::kw_true:
  case Token::kw_false:
  case Token::floatliteral:
  case Token::integer:
    storage.emplace_back(/*isNegative=*/false, p.getToken());
    p.consumeToken();
    break;

  // Parse a signed integer or a negative floating-point element.
  case Token::minus:
    p.consumeToken(Token::minus);
    if (!p.getToken().isAny(Token::floatliteral, Token::integer))
      return p.emitError("expected integer or floating point literal");
    storage.emplace_back(/*isNegative=*/true, p.getToken());
    p.consumeToken();
    break;

  case Token::string:
    storage.emplace_back(/*isNegative=*/false, p.getToken());
    p.consumeToken();
    break;

  // Parse a complex element of the form '(' element ',' element ')'.
  case Token::l_paren:
    p.consumeToken(Token::l_paren);
    if (parseElement() ||
        p.parseToken(Token::comma, "expected ',' between complex elements") ||
        parseElement() ||
        p.parseToken(Token::r_paren, "expected ')' after complex elements"))
      return failure();
    break;

  default:
    return p.emitError("expected element literal of primitive type");
  }

  return success();
}

/// Parse a list of either lists or elements, returning the dimensions of the
/// parsed sub-tensors in dims. For example:
///   parseList([1, 2, 3]) -> Success, [3]
///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
///   parseList([[1, 2], 3]) -> Failure
///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
ParseResult TensorLiteralParser::parseList(SmallVectorImpl<int64_t> &dims) {
  p.consumeToken(Token::l_square);

  auto checkDims = [&](const SmallVectorImpl<int64_t> &prevDims,
                       const SmallVectorImpl<int64_t> &newDims) -> ParseResult {
    if (prevDims == newDims)
      return success();
    return p.emitError("tensor literal is invalid; ranks are not consistent "
                       "between elements");
  };

  bool first = true;
  SmallVector<int64_t, 4> newDims;
  unsigned size = 0;
  auto parseCommaSeparatedList = [&]() -> ParseResult {
    SmallVector<int64_t, 4> thisDims;
    if (p.getToken().getKind() == Token::l_square) {
      if (parseList(thisDims))
        return failure();
    } else if (parseElement()) {
      return failure();
    }
    ++size;
    if (!first)
      return checkDims(newDims, thisDims);
    newDims = thisDims;
    first = false;
    return success();
  };
  if (p.parseCommaSeparatedListUntil(Token::r_square, parseCommaSeparatedList))
    return failure();

  // Return the sublists' dimensions with 'size' prepended.
  dims.clear();
  dims.push_back(size);
  dims.append(newDims.begin(), newDims.end());
  return success();
}

//===----------------------------------------------------------------------===//
// ElementsAttr Parser
//===----------------------------------------------------------------------===//

/// Parse a dense elements attribute.
Attribute Parser::parseDenseElementsAttr(Type attrType) {
  auto attribLoc = getToken().getLoc();
  consumeToken(Token::kw_dense);
  if (parseToken(Token::less, "expected '<' after 'dense'"))
    return nullptr;

  // Parse the literal data if necessary.
  TensorLiteralParser literalParser(*this);
  if (!consumeIf(Token::greater)) {
    if (literalParser.parse(/*allowHex=*/true) ||
        parseToken(Token::greater, "expected '>'"))
      return nullptr;
  }

  // If the type is specified `parseElementsLiteralType` will not parse a type.
  // Use the attribute location as the location for error reporting in that
  // case.
  auto loc = attrType ? attribLoc : getToken().getLoc();
  auto type = parseElementsLiteralType(attrType);
  if (!type)
    return nullptr;
  return literalParser.getAttr(loc, type);
}

/// Parse an opaque elements attribute.
Attribute Parser::parseOpaqueElementsAttr(Type attrType) {
  consumeToken(Token::kw_opaque);
  if (parseToken(Token::less, "expected '<' after 'opaque'"))
    return nullptr;

  if (getToken().isNot(Token::string))
    return (emitError("expected dialect namespace"), nullptr);

  std::string name = getToken().getStringValue();
  consumeToken(Token::string);

  if (parseToken(Token::comma, "expected ','"))
    return nullptr;

  Token hexTok = getToken();
  if (parseToken(Token::string, "elements hex string should start with '0x'") ||
      parseToken(Token::greater, "expected '>'"))
    return nullptr;
  auto type = parseElementsLiteralType(attrType);
  if (!type)
    return nullptr;

  std::string data;
  if (parseElementAttrHexValues(*this, hexTok, data))
    return nullptr;
  return OpaqueElementsAttr::get(builder.getIdentifier(name), type, data);
}

/// Shaped type for elements attribute.
///
///   elements-literal-type ::= vector-type | ranked-tensor-type
///
/// This method also checks the type has static shape.
ShapedType Parser::parseElementsLiteralType(Type type) {
  // If the user didn't provide a type, parse the colon type for the literal.
  if (!type) {
    if (parseToken(Token::colon, "expected ':'"))
      return nullptr;
    if (!(type = parseType()))
      return nullptr;
  }

  if (!type.isa<RankedTensorType, VectorType>()) {
    emitError("elements literal must be a ranked tensor or vector type");
    return nullptr;
  }

  auto sType = type.cast<ShapedType>();
  if (!sType.hasStaticShape())
    return (emitError("elements literal type must have static shape"), nullptr);

  return sType;
}

/// Parse a sparse elements attribute.
Attribute Parser::parseSparseElementsAttr(Type attrType) {
  consumeToken(Token::kw_sparse);
  if (parseToken(Token::less, "Expected '<' after 'sparse'"))
    return nullptr;

  // Check for the case where all elements are sparse. The indices are
  // represented by a 2-dimensional shape where the second dimension is the rank
  // of the type.
  Type indiceEltType = builder.getIntegerType(64);
  if (consumeIf(Token::greater)) {
    ShapedType type = parseElementsLiteralType(attrType);
    if (!type)
      return nullptr;

    // Construct the sparse elements attr using zero element indice/value
    // attributes.
    ShapedType indicesType =
        RankedTensorType::get({0, type.getRank()}, indiceEltType);
    ShapedType valuesType = RankedTensorType::get({0}, type.getElementType());
    return SparseElementsAttr::get(
        type, DenseElementsAttr::get(indicesType, ArrayRef<Attribute>()),
        DenseElementsAttr::get(valuesType, ArrayRef<Attribute>()));
  }

  /// Parse the indices. We don't allow hex values here as we may need to use
  /// the inferred shape.
  auto indicesLoc = getToken().getLoc();
  TensorLiteralParser indiceParser(*this);
  if (indiceParser.parse(/*allowHex=*/false))
    return nullptr;

  if (parseToken(Token::comma, "expected ','"))
    return nullptr;

  /// Parse the values.
  auto valuesLoc = getToken().getLoc();
  TensorLiteralParser valuesParser(*this);
  if (valuesParser.parse(/*allowHex=*/true))
    return nullptr;

  if (parseToken(Token::greater, "expected '>'"))
    return nullptr;

  auto type = parseElementsLiteralType(attrType);
  if (!type)
    return nullptr;

  // If the indices are a splat, i.e. the literal parser parsed an element and
  // not a list, we set the shape explicitly. The indices are represented by a
  // 2-dimensional shape where the second dimension is the rank of the type.
  // Given that the parsed indices is a splat, we know that we only have one
  // indice and thus one for the first dimension.
  ShapedType indicesType;
  if (indiceParser.getShape().empty()) {
    indicesType = RankedTensorType::get({1, type.getRank()}, indiceEltType);
  } else {
    // Otherwise, set the shape to the one parsed by the literal parser.
    indicesType = RankedTensorType::get(indiceParser.getShape(), indiceEltType);
  }
  auto indices = indiceParser.getAttr(indicesLoc, indicesType);

  // If the values are a splat, set the shape explicitly based on the number of
  // indices. The number of indices is encoded in the first dimension of the
  // indice shape type.
  auto valuesEltType = type.getElementType();
  ShapedType valuesType =
      valuesParser.getShape().empty()
          ? RankedTensorType::get({indicesType.getDimSize(0)}, valuesEltType)
          : RankedTensorType::get(valuesParser.getShape(), valuesEltType);
  auto values = valuesParser.getAttr(valuesLoc, valuesType);

  /// Sanity check.
  if (valuesType.getRank() != 1)
    return (emitError("expected 1-d tensor for values"), nullptr);

  auto sameShape = (indicesType.getRank() == 1) ||
                   (type.getRank() == indicesType.getDimSize(1));
  auto sameElementNum = indicesType.getDimSize(0) == valuesType.getDimSize(0);
  if (!sameShape || !sameElementNum) {
    emitError() << "expected shape ([" << type.getShape()
                << "]); inferred shape of indices literal (["
                << indicesType.getShape()
                << "]); inferred shape of values literal (["
                << valuesType.getShape() << "])";
    return nullptr;
  }

  // Build the sparse elements attribute by the indices and values.
  return SparseElementsAttr::get(type, indices, values);
}
