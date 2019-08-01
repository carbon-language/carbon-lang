// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "intrinsics.h"
#include "common.h"
#include "expression.h"
#include "fold.h"
#include "shape.h"
#include "tools.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/enum-set.h"
#include "../common/idioms.h"
#include <algorithm>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

class FoldingContext;

// This file defines the supported intrinsic procedures and implements
// their recognition and validation.  It is largely table-driven.  See
// documentation/intrinsics.md and section 16 of the Fortran 2018 standard
// for full details on each of the intrinsics.  Be advised, they have
// complicated details, and the design of these tables has to accommodate
// that complexity.

// Dummy arguments to generic intrinsic procedures are each specified by
// their keyword name (rarely used, but always defined), allowable type
// categories, a kind pattern, a rank pattern, and information about
// optionality and defaults.  The kind and rank patterns are represented
// here with code values that are significant to the matching/validation engine.

// These are small bit-sets of type category enumerators.
// Note that typeless (BOZ literal) values don't have a distinct type category.
// These typeless arguments are represented in the tables as if they were
// INTEGER with a special "typeless" kind code.  Arguments of intrinsic types
// that can also be typeless values are encoded with an "elementalOrBOZ"
// rank pattern.
// Assumed-type (TYPE(*)) dummy arguments can be forwarded along to some
// intrinsic functions that accept AnyType + Rank::anyOrAssumedRank.
using CategorySet = common::EnumSet<TypeCategory, 8>;
static constexpr CategorySet IntType{TypeCategory::Integer};
static constexpr CategorySet RealType{TypeCategory::Real};
static constexpr CategorySet ComplexType{TypeCategory::Complex};
static constexpr CategorySet CharType{TypeCategory::Character};
static constexpr CategorySet LogicalType{TypeCategory::Logical};
static constexpr CategorySet IntOrRealType{IntType | RealType};
static constexpr CategorySet FloatingType{RealType | ComplexType};
static constexpr CategorySet NumericType{IntType | RealType | ComplexType};
static constexpr CategorySet RelatableType{IntType | RealType | CharType};
static constexpr CategorySet IntrinsicType{
    IntType | RealType | ComplexType | CharType | LogicalType};
static constexpr CategorySet AnyType{
    IntrinsicType | CategorySet{TypeCategory::Derived}};

ENUM_CLASS(KindCode, none, defaultIntegerKind,
    defaultRealKind,  // is also the default COMPLEX kind
    doublePrecision, defaultCharKind, defaultLogicalKind,
    any,  // matches any kind value; each instance is independent
    same,  // match any kind, but all "same" kinds must be equal
    operand,  // match any kind, with promotion (non-standard)
    typeless,  // BOZ literals are INTEGER with this kind
    teamType,  // TEAM_TYPE from module ISO_FORTRAN_ENV (for coarrays)
    kindArg,  // this argument is KIND=
    effectiveKind,  // for function results: "kindArg" value, possibly defaulted
    dimArg,  // this argument is DIM=
    likeMultiply,  // for DOT_PRODUCT and MATMUL
    subscript,  // address-sized integer
)

struct TypePattern {
  CategorySet categorySet;
  KindCode kindCode{KindCode::none};
  std::ostream &Dump(std::ostream &) const;
};

// Abbreviations for argument and result patterns in the intrinsic prototypes:

// Match specific kinds of intrinsic types
static constexpr TypePattern DefaultInt{IntType, KindCode::defaultIntegerKind};
static constexpr TypePattern DefaultReal{RealType, KindCode::defaultRealKind};
static constexpr TypePattern DefaultComplex{
    ComplexType, KindCode::defaultRealKind};
static constexpr TypePattern DefaultChar{CharType, KindCode::defaultCharKind};
static constexpr TypePattern DefaultLogical{
    LogicalType, KindCode::defaultLogicalKind};
static constexpr TypePattern BOZ{IntType, KindCode::typeless};
static constexpr TypePattern TEAM_TYPE{IntType, KindCode::teamType};
static constexpr TypePattern DoublePrecision{
    RealType, KindCode::doublePrecision};
static constexpr TypePattern DoublePrecisionComplex{
    ComplexType, KindCode::doublePrecision};
static constexpr TypePattern SubscriptInt{IntType, KindCode::subscript};

// Match any kind of some intrinsic or derived types
static constexpr TypePattern AnyInt{IntType, KindCode::any};
static constexpr TypePattern AnyReal{RealType, KindCode::any};
static constexpr TypePattern AnyIntOrReal{IntOrRealType, KindCode::any};
static constexpr TypePattern AnyComplex{ComplexType, KindCode::any};
static constexpr TypePattern AnyFloating{FloatingType, KindCode::any};
static constexpr TypePattern AnyNumeric{NumericType, KindCode::any};
static constexpr TypePattern AnyChar{CharType, KindCode::any};
static constexpr TypePattern AnyLogical{LogicalType, KindCode::any};
static constexpr TypePattern AnyRelatable{RelatableType, KindCode::any};
static constexpr TypePattern AnyIntrinsic{IntrinsicType, KindCode::any};
static constexpr TypePattern Anything{AnyType, KindCode::any};

// Match some kind of some intrinsic type(s); all "Same" values must match,
// even when not in the same category (e.g., SameComplex and SameReal).
// Can be used to specify a result so long as at least one argument is
// a "Same".
static constexpr TypePattern SameInt{IntType, KindCode::same};
static constexpr TypePattern SameReal{RealType, KindCode::same};
static constexpr TypePattern SameIntOrReal{IntOrRealType, KindCode::same};
static constexpr TypePattern SameComplex{ComplexType, KindCode::same};
static constexpr TypePattern SameFloating{FloatingType, KindCode::same};
static constexpr TypePattern SameNumeric{NumericType, KindCode::same};
static constexpr TypePattern SameChar{CharType, KindCode::same};
static constexpr TypePattern SameLogical{LogicalType, KindCode::same};
static constexpr TypePattern SameRelatable{RelatableType, KindCode::same};
static constexpr TypePattern SameIntrinsic{IntrinsicType, KindCode::same};
static constexpr TypePattern SameDerivedType{
    CategorySet{TypeCategory::Derived}, KindCode::same};
static constexpr TypePattern SameType{AnyType, KindCode::same};

// Match some kind of some INTEGER or REAL type(s); when argument types
// &/or kinds differ, their values are converted as if they were operands to
// an intrinsic operation like addition.  This is a nonstandard but nearly
// universal extension feature.
static constexpr TypePattern OperandReal{RealType, KindCode::operand};
static constexpr TypePattern OperandIntOrReal{IntOrRealType, KindCode::operand};

// For DOT_PRODUCT and MATMUL, the result type depends on the arguments
static constexpr TypePattern ResultLogical{LogicalType, KindCode::likeMultiply};
static constexpr TypePattern ResultNumeric{NumericType, KindCode::likeMultiply};

// Result types with known category and KIND=
static constexpr TypePattern KINDInt{IntType, KindCode::effectiveKind};
static constexpr TypePattern KINDReal{RealType, KindCode::effectiveKind};
static constexpr TypePattern KINDComplex{ComplexType, KindCode::effectiveKind};
static constexpr TypePattern KINDChar{CharType, KindCode::effectiveKind};
static constexpr TypePattern KINDLogical{LogicalType, KindCode::effectiveKind};

// The default rank pattern for dummy arguments and function results is
// "elemental".
ENUM_CLASS(Rank,
    elemental,  // scalar, or array that conforms with other array arguments
    elementalOrBOZ,  // elemental, or typeless BOZ literal scalar
    scalar, vector,
    shape,  // INTEGER vector of known length and no negative element
    matrix,
    array,  // not scalar, rank is known and greater than zero
    known,  // rank is known and can be scalar
    anyOrAssumedRank,  // rank can be unknown; assumed-type TYPE(*) allowed
    conformable,  // scalar, or array of same rank & shape as "array" argument
    reduceOperation,  // a pure function with constraints for REDUCE
    dimReduced,  // scalar if no DIM= argument, else rank(array)-1
    dimRemoved,  // scalar, or rank(array)-1
    rankPlus1,  // rank(known)+1
    shaped,  // rank is length of SHAPE vector
)

ENUM_CLASS(Optionality, required, optional,
    defaultsToSameKind,  // for MatchingDefaultKIND
    defaultsToDefaultForResult,  // for DefaultingKIND
    defaultsToSubscriptKind,  // for SubscriptDefaultKIND
    repeats,  // for MAX/MIN and their several variants
)

struct IntrinsicDummyArgument {
  const char *keyword{nullptr};
  TypePattern typePattern;
  Rank rank{Rank::elemental};
  Optionality optionality{Optionality::required};
  std::ostream &Dump(std::ostream &) const;
};

// constexpr abbreviations for popular arguments:
// DefaultingKIND is a KIND= argument whose default value is the appropriate
// KIND(0), KIND(0.0), KIND(''), &c. value for the function result.
static constexpr IntrinsicDummyArgument DefaultingKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar,
    Optionality::defaultsToDefaultForResult};
// MatchingDefaultKIND is a KIND= argument whose default value is the
// kind of any "Same" function argument (viz., the one whose kind pattern is
// "same").
static constexpr IntrinsicDummyArgument MatchingDefaultKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar,
    Optionality::defaultsToSameKind};
// SubscriptDefaultKind is a KIND= argument whose default value is
// the kind of INTEGER used for address calculations.
static constexpr IntrinsicDummyArgument SubscriptDefaultKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar,
    Optionality::defaultsToSubscriptKind};
static constexpr IntrinsicDummyArgument RequiredDIM{
    "dim", {IntType, KindCode::dimArg}, Rank::scalar, Optionality::required};
static constexpr IntrinsicDummyArgument OptionalDIM{
    "dim", {IntType, KindCode::dimArg}, Rank::scalar, Optionality::optional};
static constexpr IntrinsicDummyArgument OptionalMASK{
    "mask", AnyLogical, Rank::conformable, Optionality::optional};

struct IntrinsicInterface {
  static constexpr int maxArguments{7};  // if not a MAX/MIN(...)
  const char *name{nullptr};
  IntrinsicDummyArgument dummy[maxArguments];
  TypePattern result;
  Rank rank{Rank::elemental};
  std::optional<SpecificCall> Match(const CallCharacteristics &,
      const common::IntrinsicTypeDefaultKinds &, ActualArguments &,
      FoldingContext &context) const;
  int CountArguments() const;
  std::ostream &Dump(std::ostream &) const;
};

int IntrinsicInterface::CountArguments() const {
  int n{0};
  while (n < maxArguments && dummy[n].keyword != nullptr) {
    ++n;
  }
  return n;
}

// GENERIC INTRINSIC FUNCTION INTERFACES
// Each entry in this table defines a pattern.  Some intrinsic
// functions have more than one such pattern.  Besides the name
// of the intrinsic function, each pattern has specifications for
// the dummy arguments and for the result of the function.
// The dummy argument patterns each have a name (these are from the
// standard, but rarely appear in actual code), a type and kind
// pattern, allowable ranks, and optionality indicators.
// Be advised, the default rank pattern is "elemental".
static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", SameIntOrReal}}, SameIntOrReal},
    {"abs", {{"a", SameComplex}}, SameReal},
    {"achar", {{"i", AnyInt, Rank::elementalOrBOZ}, DefaultingKIND}, KINDChar},
    {"acos", {{"x", SameFloating}}, SameFloating},
    {"acosh", {{"x", SameFloating}}, SameFloating},
    {"adjustl", {{"string", SameChar}}, SameChar},
    {"adjustr", {{"string", SameChar}}, SameChar},
    {"aimag", {{"x", SameComplex}}, SameReal},
    {"aint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"all", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"allocated", {{"array", Anything, Rank::array}}, DefaultLogical},
    {"allocated", {{"scalar", Anything, Rank::scalar}}, DefaultLogical},
    {"anint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"any", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"asin", {{"x", SameFloating}}, SameFloating},
    {"asinh", {{"x", SameFloating}}, SameFloating},
    {"associated",
        {{"pointer", Anything, Rank::known},
            {"target", Anything, Rank::known, Optionality::optional}},
        DefaultLogical},
    {"atan", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atan2", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atanh", {{"x", SameFloating}}, SameFloating},
    {"bessel_j0", {{"x", SameReal}}, SameReal},
    {"bessel_j1", {{"x", SameReal}}, SameReal},
    {"bessel_jn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_jn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector},
    {"bessel_y0", {{"x", SameReal}}, SameReal},
    {"bessel_y1", {{"x", SameReal}}, SameReal},
    {"bessel_yn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_yn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector},
    {"bge",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"bgt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"bit_size", {{"i", SameInt, Rank::anyOrAssumedRank}}, SameInt,
        Rank::scalar},
    {"ble",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"blt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"btest", {{"i", AnyInt, Rank::elementalOrBOZ}, {"pos", AnyInt}},
        DefaultLogical},
    {"ceiling", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"char", {{"i", AnyInt, Rank::elementalOrBOZ}, DefaultingKIND}, KINDChar},
    {"cmplx", {{"x", AnyComplex}, DefaultingKIND}, KINDComplex},
    {"cmplx",
        {{"x", AnyIntOrReal, Rank::elementalOrBOZ},
            {"y", AnyIntOrReal, Rank::elementalOrBOZ, Optionality::optional},
            DefaultingKIND},
        KINDComplex},
    {"command_argument_count", {}, DefaultInt, Rank::scalar},
    {"conjg", {{"z", SameComplex}}, SameComplex},
    {"cos", {{"x", SameFloating}}, SameFloating},
    {"cosh", {{"x", SameFloating}}, SameFloating},
    {"count", {{"mask", AnyLogical, Rank::array}, OptionalDIM, DefaultingKIND},
        KINDInt, Rank::dimReduced},
    {"cshift",
        {{"array", SameType, Rank::array}, {"shift", AnyInt, Rank::dimRemoved},
            OptionalDIM},
        SameType, Rank::conformable},
    {"dble", {{"a", AnyNumeric, Rank::elementalOrBOZ}}, DoublePrecision},
    {"dim", {{"x", OperandIntOrReal}, {"y", OperandIntOrReal}},
        OperandIntOrReal},
    {"dot_product",
        {{"vector_a", AnyLogical, Rank::vector},
            {"vector_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::scalar},
    {"dot_product",
        {{"vector_a", AnyComplex, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar},  // conjugates vector_a
    {"dot_product",
        {{"vector_a", AnyIntOrReal, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar},
    {"dprod", {{"x", DefaultReal}, {"y", DefaultReal}}, DoublePrecision},
    {"dshiftl",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftl", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"dshiftr",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftr", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"eoshift",
        {{"array", SameIntrinsic, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameIntrinsic, Rank::dimRemoved,
                Optionality::optional},
            OptionalDIM},
        SameIntrinsic, Rank::conformable},
    {"eoshift",
        {{"array", SameDerivedType, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameDerivedType, Rank::dimRemoved}, OptionalDIM},
        SameDerivedType, Rank::conformable},
    {"epsilon", {{"x", SameReal, Rank::anyOrAssumedRank}}, SameReal,
        Rank::scalar},
    {"erf", {{"x", SameReal}}, SameReal},
    {"erfc", {{"x", SameReal}}, SameReal},
    {"erfc_scaled", {{"x", SameReal}}, SameReal},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"exponent", {{"x", AnyReal}}, DefaultInt},
    {"findloc",
        {{"array", AnyNumeric, Rank::array},
            {"value", AnyNumeric, Rank::scalar}, RequiredDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimRemoved},
    {"findloc",
        {{"array", AnyNumeric, Rank::array},
            {"value", AnyNumeric, Rank::scalar}, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector},
    {"findloc",
        {{"array", SameChar, Rank::array}, {"value", SameChar, Rank::scalar},
            RequiredDIM, OptionalMASK, SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimRemoved},
    {"findloc",
        {{"array", SameChar, Rank::array}, {"value", SameChar, Rank::scalar},
            OptionalMASK, SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, RequiredDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimRemoved},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector},
    {"floor", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"fraction", {{"x", SameReal}}, SameReal},
    {"gamma", {{"x", SameReal}}, SameReal},
    {"huge", {{"x", SameIntOrReal, Rank::anyOrAssumedRank}}, SameIntOrReal,
        Rank::scalar},
    {"hypot", {{"x", OperandReal}, {"y", OperandReal}}, OperandReal},
    {"iachar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"iall", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iany", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iparity", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iand", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"iand", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ibclr", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ibits", {{"i", SameInt}, {"pos", AnyInt}, {"len", AnyInt}}, SameInt},
    {"ibset", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ichar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"ieor", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ieor", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"image_status",
        {{"image", SameInt},
            {"team", TEAM_TYPE, Rank::scalar, Optionality::optional}},
        DefaultInt},
    {"index",
        {{"string", SameChar}, {"substring", SameChar},
            {"back", AnyLogical, Rank::scalar, Optionality::optional},
            SubscriptDefaultKIND},
        KINDInt},
    {"int", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND}, KINDInt},
    {"ior", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ior", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ishft", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"ishftc",
        {{"i", SameInt}, {"shift", AnyInt},
            {"size", AnyInt, Rank::elemental, Optionality::optional}},
        SameInt},
    {"is_iostat_end", {{"i", AnyInt}}, DefaultLogical},
    {"is_iostat_eor", {{"i", AnyInt}}, DefaultLogical},
    {"kind", {{"x", AnyIntrinsic}}, DefaultInt},
    {"lbound",
        {{"array", Anything, Rank::anyOrAssumedRank}, RequiredDIM,
            SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
    {"lbound",
        {{"array", Anything, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::vector},
    {"leadz", {{"i", AnyInt}}, DefaultInt},
    {"len", {{"string", AnyChar, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
    {"len_trim", {{"string", AnyChar}, SubscriptDefaultKIND}, KINDInt},
    {"lge", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"lgt", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"lle", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"llt", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"loc", {{"x", Anything, Rank::anyOrAssumedRank}}, SubscriptInt,
        Rank::scalar},
    {"log", {{"x", SameFloating}}, SameFloating},
    {"log10", {{"x", SameReal}}, SameReal},
    {"logical", {{"l", AnyLogical}, DefaultingKIND}, KINDLogical},
    {"log_gamma", {{"x", SameReal}}, SameReal},
    {"matmul",
        {{"array_a", AnyLogical, Rank::vector},
            {"array_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::vector},
    {"matmul",
        {{"array_a", AnyLogical, Rank::matrix},
            {"array_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::vector},
    {"matmul",
        {{"array_a", AnyLogical, Rank::matrix},
            {"array_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::matrix},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::vector},
            {"array_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::vector},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::matrix},
            {"array_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::vector},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::matrix},
            {"array_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::matrix},
    {"maskl", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maskr", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"max",
        {{"a1", OperandIntOrReal}, {"a2", OperandIntOrReal},
            {"a3", OperandIntOrReal, Rank::elemental, Optionality::repeats}},
        OperandIntOrReal},
    {"max",
        {{"a1", SameChar}, {"a2", SameChar},
            {"a3", SameChar, Rank::elemental, Optionality::repeats}},
        SameChar},
    {"maxexponent", {{"x", AnyReal, Rank::anyOrAssumedRank}}, DefaultInt,
        Rank::scalar},
    {"maxloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"maxval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"merge",
        {{"tsource", SameType}, {"fsource", SameType}, {"mask", AnyLogical}},
        SameType},
    {"merge_bits",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"merge_bits",
        {{"i", BOZ}, {"j", SameInt}, {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"min",
        {{"a1", OperandIntOrReal}, {"a2", OperandIntOrReal},
            {"a3", OperandIntOrReal, Rank::elemental, Optionality::repeats}},
        OperandIntOrReal},
    {"min",
        {{"a1", SameChar}, {"a2", SameChar},
            {"a3", SameChar, Rank::elemental, Optionality::repeats}},
        SameChar},
    {"minexponent", {{"x", AnyReal, Rank::anyOrAssumedRank}}, DefaultInt,
        Rank::scalar},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"minval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"mod", {{"a", OperandIntOrReal}, {"p", OperandIntOrReal}},
        OperandIntOrReal},
    {"modulo", {{"a", OperandIntOrReal}, {"p", OperandIntOrReal}},
        OperandIntOrReal},
    {"nearest", {{"x", SameReal}, {"s", AnyReal}}, SameReal},
    {"nint", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"norm2", {{"x", SameReal, Rank::array}, OptionalDIM}, SameReal,
        Rank::dimReduced},
    {"not", {{"i", SameInt}}, SameInt},
    // NULL() is a special case handled in Probe() below
    {"out_of_range",
        {{"x", AnyIntOrReal}, {"mold", AnyIntOrReal, Rank::scalar}},
        DefaultLogical},
    {"out_of_range",
        {{"x", AnyReal}, {"mold", AnyInt, Rank::scalar},
            {"round", AnyLogical, Rank::scalar, Optionality::optional}},
        DefaultLogical},
    {"out_of_range", {{"x", AnyReal}, {"mold", AnyReal}}, DefaultLogical},
    {"pack",
        {{"array", SameType, Rank::array},
            {"mask", AnyLogical, Rank::conformable},
            {"vector", SameType, Rank::vector, Optionality::optional}},
        SameType, Rank::vector},
    {"parity", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"popcnt", {{"i", AnyInt}}, DefaultInt},
    {"poppar", {{"i", AnyInt}}, DefaultInt},
    {"product",
        {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"precision", {{"x", AnyFloating, Rank::anyOrAssumedRank}}, DefaultInt},
    {"present", {{"a", Anything, Rank::anyOrAssumedRank}}, DefaultLogical},
    {"radix", {{"x", AnyNumeric, Rank::anyOrAssumedRank}}, DefaultInt},
    {"range", {{"x", AnyNumeric, Rank::anyOrAssumedRank}}, DefaultInt,
        Rank::scalar},
    {"rank", {{"a", Anything, Rank::anyOrAssumedRank}}, DefaultInt},
    {"real", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND},
        KINDReal},
    {"reduce",
        {{"array", SameType, Rank::array},
            {"operation", SameType, Rank::reduceOperation}, OptionalDIM,
            OptionalMASK, {"identity", SameType, Rank::scalar},
            {"ordered", AnyLogical, Rank::scalar, Optionality::optional}},
        SameType, Rank::dimReduced},
    {"repeat", {{"string", SameChar, Rank::scalar}, {"ncopies", AnyInt}},
        SameChar, Rank::scalar},
    {"reshape",
        {{"source", SameType, Rank::array}, {"shape", AnyInt, Rank::shape},
            {"pad", SameType, Rank::array, Optionality::optional},
            {"order", AnyInt, Rank::vector, Optionality::optional}},
        SameType, Rank::shaped},
    {"rrspacing", {{"x", SameReal}}, SameReal},
    {"scale", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"scan",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            SubscriptDefaultKIND},
        KINDInt},
    {"selected_char_kind", {{"name", DefaultChar, Rank::scalar}}, DefaultInt,
        Rank::scalar},
    {"selected_int_kind", {{"r", AnyInt, Rank::scalar}}, DefaultInt,
        Rank::scalar},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar},
            {"r", AnyInt, Rank::scalar, Optionality::optional},
            {"radix", AnyInt, Rank::scalar, Optionality::optional}},
        DefaultInt, Rank::scalar},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar, Optionality::optional},
            {"r", AnyInt, Rank::scalar},
            {"radix", AnyInt, Rank::scalar, Optionality::optional}},
        DefaultInt, Rank::scalar},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar, Optionality::optional},
            {"r", AnyInt, Rank::scalar, Optionality::optional},
            {"radix", AnyInt, Rank::scalar}},
        DefaultInt, Rank::scalar},
    {"set_exponent", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"shape",
        {{"source", Anything, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::vector},
    {"shifta", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftl", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftr", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"sign", {{"a", SameIntOrReal}, {"b", SameIntOrReal}}, SameIntOrReal},
    {"sin", {{"x", SameFloating}}, SameFloating},
    {"sinh", {{"x", SameFloating}}, SameFloating},
    {"size",
        {{"array", Anything, Rank::anyOrAssumedRank}, OptionalDIM,
            SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
    {"spacing", {{"x", SameReal}}, SameReal},
    {"spread",
        {{"source", SameType, Rank::known}, RequiredDIM,
            {"ncopies", AnyInt, Rank::scalar}},
        SameType, Rank::rankPlus1},
    {"sqrt", {{"x", SameFloating}}, SameFloating},
    {"sum", {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"tan", {{"x", SameFloating}}, SameFloating},
    {"tanh", {{"x", SameFloating}}, SameFloating},
    {"tiny", {{"x", SameReal, Rank::anyOrAssumedRank}}, SameReal, Rank::scalar},
    {"trailz", {{"i", AnyInt}}, DefaultInt},
    {"transfer",
        {{"source", Anything, Rank::known}, {"mold", SameType, Rank::scalar}},
        SameType, Rank::scalar},
    {"transfer",
        {{"source", Anything, Rank::known}, {"mold", SameType, Rank::array}},
        SameType, Rank::vector},
    {"transfer",
        {{"source", Anything, Rank::anyOrAssumedRank},
            {"mold", SameType, Rank::anyOrAssumedRank},
            {"size", AnyInt, Rank::scalar}},
        SameType, Rank::vector},
    {"transpose", {{"matrix", SameType, Rank::matrix}}, SameType, Rank::matrix},
    {"trim", {{"string", SameChar, Rank::scalar}}, SameChar, Rank::scalar},
    {"ubound",
        {{"array", Anything, Rank::anyOrAssumedRank}, RequiredDIM,
            SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
    {"ubound",
        {{"array", Anything, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::vector},
    {"unpack",
        {{"vector", SameType, Rank::vector}, {"mask", AnyLogical, Rank::array},
            {"field", SameType, Rank::conformable}},
        SameType, Rank::conformable},
    {"verify",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            SubscriptDefaultKIND},
        KINDInt},
};

// TODO: Coarray intrinsic functions
//   LCOBOUND, UCOBOUND, FAILED_IMAGES, GET_TEAM, IMAGE_INDEX,
//   NUM_IMAGES, STOPPED_IMAGES, TEAM_NUMBER, THIS_IMAGE,
//   COSHAPE
// TODO: Object characteristic inquiry functions
//   EXTENDS_TYPE_OF, IS_CONTIGUOUS, SAME_TYPE, STORAGE_SIZE
// TODO: Type inquiry intrinsic functions - these return constants
//  DIGITS, NEW_LINE
// TODO: Non-standard intrinsic functions
//  AND, OR, XOR, LSHIFT, RSHIFT, SHIFT, ZEXT, IZEXT,
//  COSD, SIND, TAND, ACOSD, ASIND, ATAND, ATAN2D, COMPL,
//  DCMPLX, EQV, NEQV, INT8, JINT, JNINT, KNINT,
//  QCMPLX, DREAL, DFLOAT, QEXT, QFLOAT, QREAL, DNUM,
//  INUM, JNUM, KNUM, QNUM, RNUM, RAN, RANF, ILEN, SIZEOF,
//  MCLOCK, SECNDS, COTAN, IBCHNG, ISHA, ISHC, ISHL, IXOR
//  IARG, IARGC, NARGS, NUMARG, BADDRESS, IADDR, CACHESIZE,
//  EOF, FP_CLASS, INT_PTR_KIND, ISNAN, MALLOC
//  probably more (these are PGI + Intel, possibly incomplete)
// TODO: Optionally warn on use of non-standard intrinsics:
//  LOC, probably others
// TODO: Optionally warn on operand promotion extension

// The following table contains the intrinsic functions listed in
// Tables 16.2 and 16.3 in Fortran 2018.  The "unrestricted" functions
// in Table 16.2 can be used as actual arguments, PROCEDURE() interfaces,
// and procedure pointer targets.
struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
  bool isRestrictedSpecific{false};
  bool forceResultType{false};
};

static const SpecificIntrinsicInterface specificIntrinsicFunction[]{
    {{"abs", {{"a", DefaultReal}}, DefaultReal}},
    {{"acos", {{"x", DefaultReal}}, DefaultReal}},
    {{"aimag", {{"z", DefaultComplex}}, DefaultReal}},
    {{"aint", {{"a", DefaultReal}}, DefaultReal}},
    {{"alog", {{"x", DefaultReal}}, DefaultReal}, "log"},
    {{"alog10", {{"x", DefaultReal}}, DefaultReal}, "log10"},
    {{"amax0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "max", true, true},
    {{"amax1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "max", true, true},
    {{"amin0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true, true},
    {{"amin1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true, true},
    {{"amod", {{"a", DefaultReal}, {"p", DefaultReal}}, DefaultReal}, "mod"},
    {{"anint", {{"a", DefaultReal}}, DefaultReal}},
    {{"asin", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan2", {{"y", DefaultReal}, {"x", DefaultReal}}, DefaultReal}},
    {{"cabs", {{"a", DefaultComplex}}, DefaultReal}, "abs"},
    {{"ccos", {{"a", DefaultComplex}}, DefaultComplex}, "cos"},
    {{"cdabs", {{"a", DoublePrecisionComplex}}, DoublePrecision}, "abs"},
    {{"cdcos", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex}, "cos"},
    {{"cdexp", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex}, "exp"},
    {{"cdlog", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex}, "log"},
    {{"cdsin", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex}, "sin"},
    {{"cdsqrt", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex},
        "sqrt"},
    {{"cexp", {{"a", DefaultComplex}}, DefaultComplex}, "exp"},
    {{"clog", {{"a", DefaultComplex}}, DefaultComplex}, "log"},
    {{"conjg", {{"a", DefaultComplex}}, DefaultComplex}},
    {{"cos", {{"x", DefaultReal}}, DefaultReal}},
    {{"csin", {{"a", DefaultComplex}}, DefaultComplex}, "sin"},
    {{"csqrt", {{"a", DefaultComplex}}, DefaultComplex}, "sqrt"},
    {{"ctan", {{"a", DefaultComplex}}, DefaultComplex}, "tan"},
    {{"dabs", {{"a", DoublePrecision}}, DoublePrecision}, "abs"},
    {{"dacos", {{"x", DoublePrecision}}, DoublePrecision}, "acos"},
    {{"dasin", {{"x", DoublePrecision}}, DoublePrecision}, "asin"},
    {{"datan", {{"x", DoublePrecision}}, DoublePrecision}, "atan"},
    {{"datan2", {{"y", DoublePrecision}, {"x", DoublePrecision}},
         DoublePrecision},
        "atan2"},
    {{"dconjg", {{"a", DoublePrecisionComplex}}, DoublePrecisionComplex},
        "conjg"},
    {{"dcos", {{"x", DoublePrecision}}, DoublePrecision}, "cos"},
    {{"dcosh", {{"x", DoublePrecision}}, DoublePrecision}, "cosh"},
    {{"ddim", {{"x", DoublePrecision}, {"y", DoublePrecision}},
         DoublePrecision},
        "dim"},
    {{"dexp", {{"x", DoublePrecision}}, DoublePrecision}, "exp"},
    {{"dim", {{"x", DefaultReal}, {"y", DefaultReal}}, DefaultReal}},
    {{"dint", {{"a", DoublePrecision}}, DoublePrecision}, "aint"},
    {{"dlog", {{"x", DoublePrecision}}, DoublePrecision}, "log"},
    {{"dlog10", {{"x", DoublePrecision}}, DoublePrecision}, "log10"},
    {{"dmax1",
         {{"a1", DoublePrecision}, {"a2", DoublePrecision},
             {"a3", DoublePrecision, Rank::elemental, Optionality::repeats}},
         DoublePrecision},
        "max", true},
    {{"dmin1",
         {{"a1", DoublePrecision}, {"a2", DoublePrecision},
             {"a3", DoublePrecision, Rank::elemental, Optionality::repeats}},
         DoublePrecision},
        "min", true},
    {{"dmod", {{"a", DoublePrecision}, {"p", DoublePrecision}},
         DoublePrecision},
        "mod"},
    {{"dnint", {{"a", DoublePrecision}}, DoublePrecision}, "anint"},
    {{"dprod", {{"x", DefaultReal}, {"y", DefaultReal}}, DoublePrecision}},
    {{"dsign", {{"a", DoublePrecision}, {"b", DoublePrecision}},
         DoublePrecision},
        "sign"},
    {{"dsin", {{"x", DoublePrecision}}, DoublePrecision}, "sin"},
    {{"dsinh", {{"x", DoublePrecision}}, DoublePrecision}, "sinh"},
    {{"dsqrt", {{"x", DoublePrecision}}, DoublePrecision}, "sqrt"},
    {{"dtan", {{"x", DoublePrecision}}, DoublePrecision}, "tan"},
    {{"dtanh", {{"x", DoublePrecision}}, DoublePrecision}, "tanh"},
    {{"exp", {{"x", DefaultReal}}, DefaultReal}},
    {{"float", {{"i", DefaultInt}}, DefaultReal}, "real", true},
    {{"iabs", {{"a", DefaultInt}}, DefaultInt}, "abs"},
    {{"idim", {{"x", DefaultInt}, {"y", DefaultInt}}, DefaultInt}, "dim"},
    {{"idint", {{"a", DoublePrecision}}, DefaultInt}, "int", true},
    {{"idnint", {{"a", DoublePrecision}}, DefaultInt}, "nint"},
    {{"ifix", {{"a", DefaultReal}}, DefaultInt}, "int", true},
    {{"index", {{"string", DefaultChar}, {"substring", DefaultChar}},
        SubscriptInt}},
    {{"isign", {{"a", DefaultInt}, {"b", DefaultInt}}, DefaultInt}, "sign"},
    {{"len", {{"string", DefaultChar, Rank::anyOrAssumedRank}}, SubscriptInt,
        Rank::scalar}},
    {{"lge", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
        DefaultLogical}},
    {{"lgt", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
        DefaultLogical}},
    {{"lle", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
        DefaultLogical}},
    {{"llt", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
        DefaultLogical}},
    {{"log", {{"x", DefaultReal}}, DefaultReal}},
    {{"log10", {{"x", DefaultReal}}, DefaultReal}},
    {{"max0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true, true},
    {{"max1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true, true},
    {{"min0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true, true},
    {{"min1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true, true},
    {{"mod", {{"a", DefaultInt}, {"p", DefaultInt}}, DefaultInt}},
    {{"nint", {{"a", DefaultReal}}, DefaultInt}},
    {{"sign", {{"a", DefaultReal}, {"b", DefaultReal}}, DefaultReal}},
    {{"sin", {{"x", DefaultReal}}, DefaultReal}},
    {{"sinh", {{"x", DefaultReal}}, DefaultReal}},
    {{"sngl", {{"a", DoublePrecision}}, DefaultReal}, "real", true},
    {{"sqrt", {{"x", DefaultReal}}, DefaultReal}},
    {{"tan", {{"x", DefaultReal}}, DefaultReal}},
    {{"tanh", {{"x", DefaultReal}}, DefaultReal}},
};

// TODO: Intrinsic subroutines
//   MVBITS (elemental), CPU_TIME, DATE_AND_TIME, EVENT_QUERY,
//   EXECUTE_COMMAND_LINE, GET_COMMAND, GET_COMMAND_ARGUMENT,
//   GET_ENVIRONMENT_VARIABLE, MOVE_ALLOC, RANDOM_INIT, RANDOM_NUMBER,
//   RANDOM_SEED, SYSTEM_CLOCK
// TODO: Atomic intrinsic subroutines: ATOMIC_ADD &al.
// TODO: Collective intrinsic subroutines: CO_BROADCAST &al.

// Intrinsic interface matching against the arguments of a particular
// procedure reference.
std::optional<SpecificCall> IntrinsicInterface::Match(
    const CallCharacteristics &call,
    const common::IntrinsicTypeDefaultKinds &defaults,
    ActualArguments &arguments, FoldingContext &context) const {
  auto &messages{context.messages()};
  // Attempt to construct a 1-1 correspondence between the dummy arguments in
  // a particular intrinsic procedure's generic interface and the actual
  // arguments in a procedure reference.
  std::size_t dummyArgPatterns{0};
  for (; dummyArgPatterns < maxArguments &&
       dummy[dummyArgPatterns].keyword != nullptr;
       ++dummyArgPatterns) {
  }
  std::vector<ActualArgument *> actualForDummy(dummyArgPatterns, nullptr);
  // MAX and MIN (and others that map to them) allow their last argument to
  // be repeated indefinitely.  The actualForDummy vector is sized
  // and null-initialized to the non-repeated dummy argument count,
  // but additional actual argument pointers can be pushed on it
  // when this flag is set.
  bool repeatLastDummy{dummyArgPatterns > 0 &&
      dummy[dummyArgPatterns - 1].optionality == Optionality::repeats};
  int missingActualArguments{0};
  for (std::optional<ActualArgument> &arg : arguments) {
    if (!arg.has_value()) {
      ++missingActualArguments;
    } else {
      if (arg->isAlternateReturn) {
        messages.Say(
            "alternate return specifier not acceptable on call to intrinsic '%s'"_err_en_US,
            name);
        return std::nullopt;
      }
      bool found{false};
      int slot{missingActualArguments};
      for (std::size_t j{0}; j < dummyArgPatterns && !found; ++j) {
        if (arg->keyword.has_value()) {
          found = *arg->keyword == dummy[j].keyword;
          if (found) {
            if (const auto *previous{actualForDummy[j]}) {
              if (previous->keyword.has_value()) {
                messages.Say(*arg->keyword,
                    "repeated keyword argument to intrinsic '%s'"_err_en_US,
                    name);
              } else {
                messages.Say(*arg->keyword,
                    "keyword argument to intrinsic '%s' was supplied "
                    "positionally by an earlier actual argument"_err_en_US,
                    name);
              }
              return std::nullopt;
            }
          }
        } else {
          found = actualForDummy[j] == nullptr && slot-- == 0;
        }
        if (found) {
          actualForDummy[j] = &*arg;
        }
      }
      if (!found) {
        if (repeatLastDummy && !arg->keyword.has_value()) {
          // MAX/MIN argument after the 2nd
          actualForDummy.push_back(&*arg);
        } else {
          if (arg->keyword.has_value()) {
            messages.Say(*arg->keyword,
                "unknown keyword argument to intrinsic '%s'"_err_en_US, name);
          } else {
            messages.Say(
                "too many actual arguments for intrinsic '%s'"_err_en_US, name);
          }
          return std::nullopt;
        }
      }
    }
  }

  std::size_t dummies{actualForDummy.size()};

  // Check types and kinds of the actual arguments against the intrinsic's
  // interface.  Ensure that two or more arguments that have to have the same
  // (or compatible) type and kind do so.  Check for missing non-optional
  // arguments now, too.
  const ActualArgument *sameArg{nullptr};
  const ActualArgument *operandArg{nullptr};
  const IntrinsicDummyArgument *kindDummyArg{nullptr};
  const ActualArgument *kindArg{nullptr};
  bool hasDimArg{false};
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (d.typePattern.kindCode == KindCode::kindArg) {
      CHECK(kindDummyArg == nullptr);
      kindDummyArg = &d;
    }
    const ActualArgument *arg{actualForDummy[j]};
    if (!arg) {
      if (d.optionality == Optionality::required) {
        messages.Say("missing mandatory '%s=' argument"_err_en_US, d.keyword);
        return std::nullopt;  // missing non-OPTIONAL argument
      } else {
        continue;
      }
    }
    if (arg->GetAssumedTypeDummy()) {
      // TYPE(*) assumed-type dummy argument forwarded to intrinsic
      if (d.typePattern.categorySet == AnyType &&
          d.typePattern.kindCode == KindCode::any &&
          d.rank == Rank::anyOrAssumedRank) {
        continue;
      } else {
        messages.Say("Assumed type TYPE(*) dummy argument not allowed "
                     "for '%s=' intrinsic argument"_err_en_US,
            d.keyword);
        return std::nullopt;
      }
    }
    std::optional<DynamicType> type{arg->GetType()};
    if (!type.has_value()) {
      CHECK(arg->Rank() == 0);
      const Expr<SomeType> *expr{arg->UnwrapExpr()};
      CHECK(expr != nullptr);
      if (std::holds_alternative<BOZLiteralConstant>(expr->u)) {
        if (d.typePattern.kindCode == KindCode::typeless ||
            d.rank == Rank::elementalOrBOZ) {
          continue;
        } else {
          messages.Say(
              "Typeless (BOZ) not allowed for '%s=' argument"_err_en_US,
              d.keyword);
        }
      } else {
        // NULL(), pointer to subroutine, &c.
        if ("loc"s != name) {
          messages.Say("Typeless item not allowed for '%s=' argument"_err_en_US,
              d.keyword);
        }
      }
      return std::nullopt;
    } else if (!d.typePattern.categorySet.test(type->category())) {
      messages.Say("Actual argument for '%s=' has bad type '%s'"_err_en_US,
          d.keyword, type->AsFortran());
      return std::nullopt;  // argument has invalid type category
    }
    bool argOk{false};
    switch (d.typePattern.kindCode) {
    case KindCode::none:
    case KindCode::typeless:
    case KindCode::teamType:  // TODO: TEAM_TYPE
      argOk = false;
      break;
    case KindCode::defaultIntegerKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Integer);
      break;
    case KindCode::defaultRealKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Real);
      break;
    case KindCode::doublePrecision:
      argOk = type->kind() == defaults.doublePrecisionKind();
      break;
    case KindCode::defaultCharKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Character);
      break;
    case KindCode::defaultLogicalKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Logical);
      break;
    case KindCode::any: argOk = true; break;
    case KindCode::kindArg:
      CHECK(type->category() == TypeCategory::Integer);
      CHECK(kindArg == nullptr);
      kindArg = arg;
      argOk = true;
      break;
    case KindCode::dimArg:
      CHECK(type->category() == TypeCategory::Integer);
      hasDimArg = true;
      argOk = true;
      break;
    case KindCode::same:
      if (sameArg == nullptr) {
        sameArg = arg;
      }
      argOk = type->IsTkCompatibleWith(sameArg->GetType().value());
      break;
    case KindCode::operand:
      if (operandArg == nullptr) {
        operandArg = arg;
      } else if (auto prev{operandArg->GetType()}) {
        if (type->category() == prev->category()) {
          if (type->kind() > prev->kind()) {
            operandArg = arg;
          }
        } else if (prev->category() == TypeCategory::Integer) {
          operandArg = arg;
        }
      }
      argOk = true;
      break;
    case KindCode::effectiveKind:
      common::die("INTERNAL: KindCode::effectiveKind appears on argument '%s' "
                  "for intrinsic '%s'",
          d.keyword, name);
      break;
    default: CRASH_NO_CASE;
    }
    if (!argOk) {
      messages.Say(
          "Actual argument for '%s=' has bad type or kind '%s'"_err_en_US,
          d.keyword, type->AsFortran());
      return std::nullopt;
    }
  }

  // Check the ranks of the arguments against the intrinsic's interface.
  const ActualArgument *arrayArg{nullptr};
  const ActualArgument *knownArg{nullptr};
  std::optional<int> shapeArgSize;
  int elementalRank{0};
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (const ActualArgument * arg{actualForDummy[j]}) {
      if (IsAssumedRank(*arg) && d.rank != Rank::anyOrAssumedRank) {
        messages.Say("Assumed-rank array cannot be forwarded to "
                     "'%s=' argument"_err_en_US,
            d.keyword);
        return std::nullopt;
      }
      int rank{arg->Rank()};
      bool argOk{false};
      switch (d.rank) {
      case Rank::elemental:
      case Rank::elementalOrBOZ:
        if (elementalRank == 0) {
          elementalRank = rank;
        }
        argOk = rank == 0 || rank == elementalRank;
        break;
      case Rank::scalar: argOk = rank == 0; break;
      case Rank::vector: argOk = rank == 1; break;
      case Rank::shape:
        CHECK(!shapeArgSize.has_value());
        if (rank == 1) {
          if (auto shape{GetShape(context, *arg)}) {
            if (auto constShape{AsConstantShape(*shape)}) {
              shapeArgSize = constShape->At(ConstantSubscripts{1}).ToInt64();
              CHECK(shapeArgSize >= 0);
              argOk = true;
            }
          }
        }
        if (!argOk) {
          messages.Say(
              "'shape=' argument must be a vector of known size"_err_en_US);
          return std::nullopt;
        }
        break;
      case Rank::matrix: argOk = rank == 2; break;
      case Rank::array:
        argOk = rank > 0;
        if (!arrayArg) {
          arrayArg = arg;
        } else {
          argOk &= rank == arrayArg->Rank();
        }
        break;
      case Rank::known:
        if (knownArg == nullptr) {
          knownArg = arg;
        }
        argOk = rank == knownArg->Rank();
        break;
      case Rank::anyOrAssumedRank: argOk = true; break;
      case Rank::conformable:
        CHECK(arrayArg != nullptr);
        argOk = rank == 0 || rank == arrayArg->Rank();
        break;
      case Rank::dimRemoved:
        CHECK(arrayArg != nullptr);
        if (hasDimArg) {
          argOk = rank == 0 || rank + 1 == arrayArg->Rank();
        } else {
          argOk = rank == 0;
        }
        break;
      case Rank::reduceOperation:
        // TODO: Confirm that the argument is a pure function
        // of two arguments with several constraints
        CHECK(arrayArg != nullptr);
        argOk = rank == 0;
        break;
      case Rank::dimReduced:
      case Rank::rankPlus1:
      case Rank::shaped:
        common::die("INTERNAL: result-only rank code appears on argument '%s' "
                    "for intrinsic '%s'",
            d.keyword, name);
      default: CRASH_NO_CASE;
      }
      if (!argOk) {
        messages.Say("'%s=' argument has unacceptable rank %d"_err_en_US,
            d.keyword, rank);
        return std::nullopt;
      }
    }
  }

  // Calculate the characteristics of the function result, if any
  std::optional<DynamicType> resultType;
  if (auto category{result.categorySet.LeastElement()}) {
    // The intrinsic is not a subroutine.
    if (call.isSubroutineCall) {
      return std::nullopt;
    }
    switch (result.kindCode) {
    case KindCode::defaultIntegerKind:
      CHECK(result.categorySet == IntType);
      CHECK(*category == TypeCategory::Integer);
      resultType = DynamicType{TypeCategory::Integer,
          defaults.GetDefaultKind(TypeCategory::Integer)};
      break;
    case KindCode::defaultRealKind:
      CHECK(result.categorySet == CategorySet{*category});
      CHECK(FloatingType.test(*category));
      resultType =
          DynamicType{*category, defaults.GetDefaultKind(TypeCategory::Real)};
      break;
    case KindCode::doublePrecision:
      CHECK(result.categorySet == RealType);
      CHECK(*category == TypeCategory::Real);
      resultType =
          DynamicType{TypeCategory::Real, defaults.doublePrecisionKind()};
      break;
    case KindCode::defaultCharKind:
      CHECK(result.categorySet == CharType);
      CHECK(*category == TypeCategory::Character);
      resultType = DynamicType{TypeCategory::Character,
          defaults.GetDefaultKind(TypeCategory::Character)};
      break;
    case KindCode::defaultLogicalKind:
      CHECK(result.categorySet == LogicalType);
      CHECK(*category == TypeCategory::Logical);
      resultType = DynamicType{TypeCategory::Logical,
          defaults.GetDefaultKind(TypeCategory::Logical)};
      break;
    case KindCode::same:
      CHECK(sameArg != nullptr);
      if (std::optional<DynamicType> aType{sameArg->GetType()}) {
        if (result.categorySet.test(aType->category())) {
          resultType = *aType;
        } else {
          resultType = DynamicType{*category, aType->kind()};
        }
      }
      break;
    case KindCode::operand:
      CHECK(operandArg != nullptr);
      resultType = operandArg->GetType();
      CHECK(!resultType || result.categorySet.test(resultType->category()));
      break;
    case KindCode::effectiveKind:
      CHECK(kindDummyArg != nullptr);
      CHECK(result.categorySet == CategorySet{*category});
      if (kindArg != nullptr) {
        if (auto *expr{kindArg->UnwrapExpr()}) {
          CHECK(expr->Rank() == 0);
          if (auto code{ToInt64(*expr)}) {
            if (IsValidKindOfIntrinsicType(*category, *code)) {
              resultType = DynamicType{*category, static_cast<int>(*code)};
              break;
            }
          }
        }
        messages.Say("'kind=' argument must be a constant scalar integer "
                     "whose value is a supported kind for the "
                     "intrinsic result type"_err_en_US);
        return std::nullopt;
      } else if (kindDummyArg->optionality == Optionality::defaultsToSameKind) {
        CHECK(sameArg != nullptr);
        resultType = *sameArg->GetType();
      } else if (kindDummyArg->optionality ==
          Optionality::defaultsToSubscriptKind) {
        CHECK(*category == TypeCategory::Integer);
        resultType =
            DynamicType{TypeCategory::Integer, defaults.subscriptIntegerKind()};
      } else {
        CHECK(kindDummyArg->optionality ==
            Optionality::defaultsToDefaultForResult);
        resultType = DynamicType{*category, defaults.GetDefaultKind(*category)};
      }
      break;
    case KindCode::likeMultiply:
      CHECK(dummies >= 2);
      CHECK(actualForDummy[0] != nullptr);
      CHECK(actualForDummy[1] != nullptr);
      resultType = actualForDummy[0]->GetType()->ResultTypeForMultiply(
          *actualForDummy[1]->GetType());
      break;
    case KindCode::subscript:
      CHECK(result.categorySet == IntType);
      CHECK(*category == TypeCategory::Integer);
      resultType =
          DynamicType{TypeCategory::Integer, defaults.subscriptIntegerKind()};
      break;
    case KindCode::typeless:
    case KindCode::teamType:
    case KindCode::any:
    case KindCode::kindArg:
    case KindCode::dimArg:
      common::die(
          "INTERNAL: bad KindCode appears on intrinsic '%s' result", name);
      break;
    default: CRASH_NO_CASE;
    }
  } else {
    if (!call.isSubroutineCall) {
      return std::nullopt;
    }
    CHECK(result.kindCode == KindCode::none);
  }

  // At this point, the call is acceptable.
  // Determine the rank of the function result.
  int resultRank{0};
  switch (rank) {
  case Rank::elemental: resultRank = elementalRank; break;
  case Rank::scalar: resultRank = 0; break;
  case Rank::vector: resultRank = 1; break;
  case Rank::matrix: resultRank = 2; break;
  case Rank::conformable:
    CHECK(arrayArg != nullptr);
    resultRank = arrayArg->Rank();
    break;
  case Rank::dimReduced:
    CHECK(arrayArg != nullptr);
    resultRank = hasDimArg ? arrayArg->Rank() - 1 : 0;
    break;
  case Rank::dimRemoved:
    CHECK(arrayArg != nullptr);
    resultRank = arrayArg->Rank() - 1;
    break;
  case Rank::rankPlus1:
    CHECK(knownArg != nullptr);
    resultRank = knownArg->Rank() + 1;
    break;
  case Rank::shaped:
    CHECK(shapeArgSize.has_value());
    resultRank = *shapeArgSize;
    break;
  case Rank::elementalOrBOZ:
  case Rank::shape:
  case Rank::array:
  case Rank::known:
  case Rank::anyOrAssumedRank:
  case Rank::reduceOperation:
    common::die("INTERNAL: bad Rank code on intrinsic '%s' result", name);
    break;
  default: CRASH_NO_CASE;
  }
  CHECK(resultRank >= 0);

  // Rearrange the actual arguments into dummy argument order.
  ActualArguments rearranged(dummies);
  for (std::size_t j{0}; j < dummies; ++j) {
    if (ActualArgument * arg{actualForDummy[j]}) {
      rearranged[j] = std::move(*arg);
    }
  }

  // Characterize the specific intrinsic function.
  characteristics::TypeAndShape typeAndShape{resultType.value(), resultRank};
  characteristics::FunctionResult funcResult{std::move(typeAndShape)};
  characteristics::DummyArguments dummyArgs;
  std::optional<int> sameDummyArg;
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (const auto &arg{rearranged[j]}) {
      if (const Expr<SomeType> *expr{arg->UnwrapExpr()}) {
        std::optional<characteristics::TypeAndShape> typeAndShape;
        if (auto type{expr->GetType()}) {
          if (auto shape{GetShape(context, *expr)}) {
            typeAndShape.emplace(*type, std::move(*shape));
          } else {
            typeAndShape.emplace(*type);
          }
        } else {
          typeAndShape.emplace(DynamicType::TypelessIntrinsicArgument());
        }
        dummyArgs.emplace_back(std::string{d.keyword},
            characteristics::DummyDataObject{std::move(typeAndShape.value())});
        if (d.typePattern.kindCode == KindCode::same &&
            !sameDummyArg.has_value()) {
          sameDummyArg = j;
        }
      } else {
        CHECK(arg->GetAssumedTypeDummy() != nullptr);
      }
    } else {
      // optional argument is absent
      CHECK(d.optionality != Optionality::required);
      if (d.typePattern.kindCode == KindCode::same) {
        dummyArgs.emplace_back(dummyArgs[sameDummyArg.value()]);
      } else {
        auto category{d.typePattern.categorySet.LeastElement().value()};
        characteristics::TypeAndShape typeAndShape{
            DynamicType{category, defaults.GetDefaultKind(category)}};
        dummyArgs.emplace_back(std::string{d.keyword},
            characteristics::DummyDataObject{std::move(typeAndShape)});
      }
      dummyArgs.back().SetOptional();
    }
  }
  characteristics::Procedure::Attrs attrs;
  if (elementalRank > 0) {
    attrs.set(characteristics::Procedure::Attr::Elemental);
  }
  characteristics::Procedure chars{
      std::move(funcResult), std::move(dummyArgs), attrs};

  return SpecificCall{
      SpecificIntrinsic{name, std::move(chars)}, std::move(rearranged)};
}

class IntrinsicProcTable::Implementation {
public:
  explicit Implementation(const common::IntrinsicTypeDefaultKinds &dfts)
    : defaults_{dfts} {
    for (const IntrinsicInterface &f : genericIntrinsicFunction) {
      genericFuncs_.insert(std::make_pair(std::string{f.name}, &f));
    }
    for (const SpecificIntrinsicInterface &f : specificIntrinsicFunction) {
      specificFuncs_.insert(std::make_pair(std::string{f.name}, &f));
    }
  }

  bool IsIntrinsic(const std::string &) const;

  std::optional<SpecificCall> Probe(const CallCharacteristics &,
      ActualArguments &, FoldingContext &, const IntrinsicProcTable &) const;

  std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
  IsUnrestrictedSpecificIntrinsicFunction(const std::string &) const;

  std::ostream &Dump(std::ostream &) const;

private:
  DynamicType GetSpecificType(const TypePattern &) const;
  SpecificCall HandleNull(
      ActualArguments &, FoldingContext &, const IntrinsicProcTable &) const;

  common::IntrinsicTypeDefaultKinds defaults_;
  std::multimap<std::string, const IntrinsicInterface *> genericFuncs_;
  std::multimap<std::string, const SpecificIntrinsicInterface *> specificFuncs_;
};

bool IntrinsicProcTable::Implementation::IsIntrinsic(
    const std::string &name) const {
  auto specificRange{specificFuncs_.equal_range(name)};
  if (specificRange.first != specificRange.second) {
    return true;
  }
  auto genericRange{genericFuncs_.equal_range(name)};
  if (genericRange.first != genericRange.second) {
    return true;
  }
  // special cases
  return name == "null";  // TODO more
}

// The NULL() intrinsic is a special case.
SpecificCall IntrinsicProcTable::Implementation::HandleNull(
    ActualArguments &arguments, FoldingContext &context,
    const IntrinsicProcTable &intrinsics) const {
  if (!arguments.empty()) {
    if (arguments.size() > 1) {
      context.messages().Say("Too many arguments to NULL()"_err_en_US);
    } else if (arguments[0].has_value() && arguments[0]->keyword.has_value() &&
        arguments[0]->keyword->ToString() != "mold") {
      context.messages().Say("Unknown argument '%s' to NULL()"_err_en_US,
          arguments[0]->keyword->ToString());
    } else {
      if (Expr<SomeType> * mold{arguments[0]->UnwrapExpr()}) {
        if (IsAllocatableOrPointer(*mold)) {
          characteristics::DummyArguments args;
          std::optional<characteristics::FunctionResult> fResult;
          if (IsProcedurePointer(*mold)) {
            // MOLD= procedure pointer
            const Symbol *last{GetLastSymbol(*mold)};
            CHECK(last != nullptr);
            auto procPointer{
                characteristics::Procedure::Characterize(*last, intrinsics)};
            characteristics::DummyProcedure dp{
                common::Clone(procPointer.value())};
            args.emplace_back("mold"s, std::move(dp));
            fResult.emplace(std::move(procPointer.value()));
          } else if (auto type{mold->GetType()}) {
            // MOLD= object pointer
            std::optional<characteristics::TypeAndShape> typeAndShape;
            if (auto shape{GetShape(context, *mold)}) {
              typeAndShape.emplace(*type, std::move(*shape));
            } else {
              typeAndShape.emplace(*type);
            }
            characteristics::DummyDataObject ddo{typeAndShape.value()};
            args.emplace_back("mold"s, std::move(ddo));
            fResult.emplace(std::move(*typeAndShape));
          } else {
            context.messages().Say(
                "MOLD= argument to NULL() lacks type"_err_en_US);
          }
          fResult->attrs.set(characteristics::FunctionResult::Attr::Pointer);
          characteristics::Procedure::Attrs attrs;
          attrs.set(characteristics::Procedure::Attr::NullPointer);
          characteristics::Procedure chars{
              std::move(*fResult), std::move(args), attrs};
          return SpecificCall{SpecificIntrinsic{"null"s, std::move(chars)},
              std::move(arguments)};
        }
      }
      context.messages().Say(
          "MOLD= argument to NULL() must be a pointer or allocatable"_err_en_US);
    }
  }
  characteristics::Procedure::Attrs attrs;
  attrs.set(characteristics::Procedure::Attr::NullPointer);
  arguments.clear();
  return SpecificCall{
      SpecificIntrinsic{"null"s,
          characteristics::Procedure{characteristics::DummyArguments{}, attrs}},
      std::move(arguments)};
}

// Applies any semantic checks peculiar to an intrinsic.
static bool ApplySpecificChecks(
    SpecificCall &call, parser::ContextualMessages &messages) {
  bool ok{true};
  const std::string &name{call.specificIntrinsic.name};
  if (name == "allocated") {
    if (const auto &arg{call.arguments[0]}) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const Symbol * symbol{GetLastSymbol(*expr)}) {
          ok = symbol->attrs().test(semantics::Attr::ALLOCATABLE);
        }
      }
    }
    if (!ok) {
      messages.Say(
          "Argument of ALLOCATED() must be an ALLOCATABLE object or component"_err_en_US);
    }
  } else if (name == "associated") {
    if (const auto &arg{call.arguments[0]}) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const Symbol * symbol{GetLastSymbol(*expr)}) {
          ok = symbol->attrs().test(semantics::Attr::POINTER);
          // TODO: validate the TARGET= argument vs. the pointer
        }
      }
    }
    if (!ok) {
      messages.Say(
          "Arguments of ASSOCIATED() must be a POINTER and an optional valid target"_err_en_US);
    }
  } else if (name == "loc") {
    if (const auto &arg{call.arguments[0]}) {
      ok = arg->GetAssumedTypeDummy() != nullptr ||
          GetLastSymbol(arg->UnwrapExpr()) != nullptr;
    }
    if (!ok) {
      messages.Say(
          "Argument of LOC() must be an object or procedure"_err_en_US);
    }
  } else if (name == "present") {
    if (const auto &arg{call.arguments[0]}) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const Symbol * symbol{UnwrapWholeSymbolDataRef(*expr)}) {
          ok = symbol->attrs().test(semantics::Attr::OPTIONAL);
        }
      }
    }
    if (!ok) {
      messages.Say(
          "Argument of PRESENT() must be the name of an OPTIONAL dummy argument"_err_en_US);
    }
  }
  return ok;
}

// Probe the configured intrinsic procedure pattern tables in search of a
// match for a given procedure reference.
std::optional<SpecificCall> IntrinsicProcTable::Implementation::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    FoldingContext &context, const IntrinsicProcTable &intrinsics) const {
  if (call.isSubroutineCall) {
    return std::nullopt;  // TODO
  }
  parser::Messages *finalBuffer{context.messages().messages()};
  // Special case: NULL()
  // All special cases handled here before the table probes below must
  // also be caught as special names in IsIntrinsic().
  if (call.name == "null") {
    parser::Messages nullBuffer;
    parser::ContextualMessages nullErrors{
        call.name, finalBuffer ? &nullBuffer : nullptr};
    FoldingContext nullContext{context, nullErrors};
    auto result{HandleNull(arguments, nullContext, intrinsics)};
    if (finalBuffer != nullptr) {
      finalBuffer->Annex(std::move(nullBuffer));
    }
    return result;
  }
  // Probe the generic intrinsic function table first.
  parser::Messages localBuffer;
  parser::ContextualMessages localMessages{
      call.name, finalBuffer ? &localBuffer : nullptr};
  FoldingContext localContext{context, localMessages};
  std::string name{call.name.ToString()};
  parser::Messages genericBuffer;
  auto genericRange{genericFuncs_.equal_range(name)};
  for (auto iter{genericRange.first}; iter != genericRange.second; ++iter) {
    if (auto specificCall{
            iter->second->Match(call, defaults_, arguments, localContext)}) {
      ApplySpecificChecks(*specificCall, localMessages);
      if (finalBuffer != nullptr) {
        finalBuffer->Annex(std::move(localBuffer));
      }
      return specificCall;
    } else if (genericBuffer.empty()) {
      genericBuffer.Annex(std::move(localBuffer));
    } else {
      localBuffer.clear();
    }
  }
  // Probe the specific intrinsic function table next.
  // Each specific intrinsic maps to a generic intrinsic.
  parser::Messages specificBuffer;
  auto specificRange{specificFuncs_.equal_range(name)};
  for (auto specIter{specificRange.first}; specIter != specificRange.second;
       ++specIter) {
    // We only need to check the cases with distinct generic names.
    if (const char *genericName{specIter->second->generic}) {
      auto genericRange{genericFuncs_.equal_range(genericName)};
      for (auto genIter{genericRange.first}; genIter != genericRange.second;
           ++genIter) {
        if (auto specificCall{genIter->second->Match(
                call, defaults_, arguments, localContext)}) {
          specificCall->specificIntrinsic.name = genericName;
          specificCall->specificIntrinsic.isRestrictedSpecific =
              specIter->second->isRestrictedSpecific;
          if (finalBuffer != nullptr) {
            finalBuffer->Annex(std::move(localBuffer));
          }
          if (specIter->second->forceResultType) {
            // Force the result type on AMAX0/1, MIN0/1, &c.
            TypeCategory category{TypeCategory::Integer};
            switch (specIter->second->result.kindCode) {
            case KindCode::defaultIntegerKind: break;
            case KindCode::defaultRealKind:
              category = TypeCategory::Real;
              break;
            default: CRASH_NO_CASE;
            }
            DynamicType newType{category, defaults_.GetDefaultKind(category)};
            specificCall->specificIntrinsic.characteristics.value()
                .functionResult.value()
                .SetType(newType);
          }
          // TODO test feature AdditionalIntrinsics, warn on nonstandard
          // specifics with DoublePrecisionComplex arguments.
          return specificCall;
        } else if (specificBuffer.empty()) {
          specificBuffer.Annex(std::move(localBuffer));
        } else {
          specificBuffer.clear();
        }
      }
    }
  }
  // No match; report the right errors, if any
  if (finalBuffer != nullptr) {
    if (specificBuffer.empty()) {
      finalBuffer->Annex(std::move(genericBuffer));
    } else {
      finalBuffer->Annex(std::move(specificBuffer));
    }
  }
  return std::nullopt;
}

std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
IntrinsicProcTable::Implementation::IsUnrestrictedSpecificIntrinsicFunction(
    const std::string &name) const {
  auto specificRange{specificFuncs_.equal_range(name)};
  for (auto iter{specificRange.first}; iter != specificRange.second; ++iter) {
    const SpecificIntrinsicInterface &specific{*iter->second};
    if (!specific.isRestrictedSpecific) {
      std::string genericName{name};
      if (specific.generic != nullptr) {
        genericName = std::string(specific.generic);
      }
      characteristics::FunctionResult fResult{GetSpecificType(specific.result)};
      characteristics::DummyArguments args;
      int dummies{specific.CountArguments()};
      for (int j{0}; j < dummies; ++j) {
        characteristics::DummyDataObject dummy{
            GetSpecificType(specific.dummy[j].typePattern)};
        dummy.intent = common::Intent::In;
        args.emplace_back(
            std::string{specific.dummy[j].keyword}, std::move(dummy));
      }
      characteristics::Procedure::Attrs attrs;
      attrs.set(characteristics::Procedure::Attr::Pure)
          .set(characteristics::Procedure::Attr::Elemental);
      characteristics::Procedure chars{
          std::move(fResult), std::move(args), attrs};
      return UnrestrictedSpecificIntrinsicFunctionInterface{
          std::move(chars), genericName};
    }
  }
  return std::nullopt;
}

DynamicType IntrinsicProcTable::Implementation::GetSpecificType(
    const TypePattern &pattern) const {
  const CategorySet &set{pattern.categorySet};
  CHECK(set.count() == 1);
  TypeCategory category{set.LeastElement().value()};
  return DynamicType{category, defaults_.GetDefaultKind(category)};
}

IntrinsicProcTable::~IntrinsicProcTable() {
  // Discard the configured tables.
  delete impl_;
  impl_ = nullptr;
}

IntrinsicProcTable IntrinsicProcTable::Configure(
    const common::IntrinsicTypeDefaultKinds &defaults) {
  IntrinsicProcTable result;
  result.impl_ = new IntrinsicProcTable::Implementation(defaults);
  return result;
}

bool IntrinsicProcTable::IsIntrinsic(const std::string &name) const {
  CHECK(impl_ != nullptr || !"IntrinsicProcTable: not configured");
  return impl_->IsIntrinsic(name);
}

std::optional<SpecificCall> IntrinsicProcTable::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    FoldingContext &context) const {
  CHECK(impl_ != nullptr || !"IntrinsicProcTable: not configured");
  return impl_->Probe(call, arguments, context, *this);
}

std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
IntrinsicProcTable::IsUnrestrictedSpecificIntrinsicFunction(
    const std::string &name) const {
  CHECK(impl_ != nullptr || !"IntrinsicProcTable: not configured");
  return impl_->IsUnrestrictedSpecificIntrinsicFunction(name);
}

std::ostream &TypePattern::Dump(std::ostream &o) const {
  if (categorySet == AnyType) {
    o << "any type";
  } else {
    const char *sep = "";
    auto set{categorySet};
    while (auto least{set.LeastElement()}) {
      o << sep << EnumToString(*least);
      sep = " or ";
      set.reset(*least);
    }
  }
  o << '(' << EnumToString(kindCode) << ')';
  return o;
}

std::ostream &IntrinsicDummyArgument::Dump(std::ostream &o) const {
  if (keyword) {
    o << keyword << '=';
  }
  return typePattern.Dump(o)
      << ' ' << EnumToString(rank) << ' ' << EnumToString(optionality);
}

std::ostream &IntrinsicInterface::Dump(std::ostream &o) const {
  o << name;
  char sep{'('};
  for (const auto &d : dummy) {
    if (d.typePattern.kindCode == KindCode::none) {
      break;
    }
    d.Dump(o << sep);
    sep = ',';
  }
  if (sep == '(') {
    o << "()";
  }
  return result.Dump(o << " -> ") << ' ' << EnumToString(rank);
}

std::ostream &IntrinsicProcTable::Implementation::Dump(std::ostream &o) const {
  o << "generic intrinsic functions:\n";
  for (const auto &iter : genericFuncs_) {
    iter.second->Dump(o << iter.first << ": ") << '\n';
  }
  o << "specific intrinsic functions:\n";
  for (const auto &iter : specificFuncs_) {
    iter.second->Dump(o << iter.first << ": ");
    if (const char *g{iter.second->generic}) {
      o << " -> " << g;
    }
    o << '\n';
  }
  return o;
}

std::ostream &IntrinsicProcTable::Dump(std::ostream &o) const {
  return impl_->Dump(o);
}
}
