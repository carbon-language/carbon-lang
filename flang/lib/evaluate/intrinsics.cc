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
#include "expression.h"
#include "fold.h"
#include "tools.h"
#include "type.h"
#include "../common/enum-set.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include <algorithm>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

using common::TypeCategory;

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
// that can also be be typeless values are encoded with an "elementalOrBOZ"
// rank pattern.
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
    typeless,  // BOZ literals are INTEGER with this kind
    teamType,  // TEAM_TYPE from module ISO_FORTRAN_ENV (for coarrays)
    kindArg,  // this argument is KIND=
    effectiveKind,  // for function results: same "kindArg", possibly defaulted
    dimArg,  // this argument is DIM=
    same,  // match any kind; all "same" kinds must be equal
    likeMultiply,  // for DOT_PRODUCT and MATMUL
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

// Match any kind of some intrinsic or derived types
static constexpr TypePattern AnyInt{IntType, KindCode::any};
static constexpr TypePattern AnyReal{RealType, KindCode::any};
static constexpr TypePattern AnyIntOrReal{IntOrRealType, KindCode::any};
static constexpr TypePattern AnyComplex{ComplexType, KindCode::any};
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
    anyOrAssumedRank,  // rank can be unknown
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
      parser::ContextualMessages &messages) const;
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
// The dummy argument patterns each have a name (this are from the
// standard, but rarely appear in actual code), a type and kind
// pattern, allowable ranks, and optionality indicators.
// Be advised, the default rank pattern is "elemental".
static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", SameIntOrReal}}, SameIntOrReal},
    {"abs", {{"a", SameComplex}}, SameReal},
    {"achar", {{"i", AnyInt}, DefaultingKIND}, KINDChar},
    {"acos", {{"x", SameFloating}}, SameFloating},
    {"acosh", {{"x", SameFloating}}, SameFloating},
    {"adjustl", {{"string", SameChar}}, SameChar},
    {"adjustr", {{"string", SameChar}}, SameChar},
    {"aimag", {{"x", SameComplex}}, SameReal},
    {"aint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"all", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"anint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"any", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"asin", {{"x", SameFloating}}, SameFloating},
    {"asinh", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"y", SameReal}, {"x", SameReal}}, SameReal},
    {"atan2", {{"y", SameReal}, {"x", SameReal}}, SameReal},
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
    {"ble",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"blt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"btest", {{"i", AnyInt}, {"pos", AnyInt}}, DefaultLogical},
    {"ceiling", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"char", {{"i", AnyInt}, DefaultingKIND}, KINDChar},
    {"cmplx", {{"x", AnyComplex}, DefaultingKIND}, KINDComplex},
    {"cmplx",
        {{"x", SameIntOrReal, Rank::elementalOrBOZ},
            {"y", SameIntOrReal, Rank::elementalOrBOZ}, DefaultingKIND},
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
        SameType, Rank::array},
    {"dble", {{"a", AnyNumeric, Rank::elementalOrBOZ}}, DoublePrecision},
    {"dim", {{"x", SameIntOrReal}, {"y", SameIntOrReal}}, SameIntOrReal},
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
        SameIntrinsic, Rank::array},
    {"eoshift",
        {{"array", SameDerivedType, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameDerivedType, Rank::dimRemoved}, OptionalDIM},
        SameDerivedType, Rank::array},
    {"erf", {{"x", SameReal}}, SameReal},
    {"erfc", {{"x", SameReal}}, SameReal},
    {"erfc_scaled", {{"x", SameReal}}, SameReal},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"exponent", {{"x", AnyReal}}, DefaultInt},
    {"findloc",
        {{"array", AnyNumeric, Rank::array},
            {"value", AnyNumeric, Rank::scalar}, OptionalDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", SameChar, Rank::array}, {"value", SameChar, Rank::scalar},
            OptionalDIM, OptionalMASK, SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, OptionalDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"floor", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"fraction", {{"x", SameReal}}, SameReal},
    {"gamma", {{"x", SameReal}}, SameReal},
    {"hypot", {{"x", SameReal}, {"y", SameReal}}, SameReal},
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
        {{"array", Anything, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::vector},
    {"lbound",
        {{"array", Anything, Rank::anyOrAssumedRank},
            {"dim", {IntType, KindCode::dimArg}, Rank::scalar},
            SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
    {"leadz", {{"i", AnyInt}}, DefaultInt},
    {"len", {{"string", AnyChar}, SubscriptDefaultKIND}, KINDInt},
    {"len_trim", {{"string", AnyChar}, SubscriptDefaultKIND}, KINDInt},
    {"lge", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"lgt", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"lle", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
    {"llt", {{"string_a", SameChar}, {"string_b", SameChar}}, DefaultLogical},
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
        {{"a1", SameRelatable}, {"a2", SameRelatable},
            {"a3", SameRelatable, Rank::elemental, Optionality::repeats}},
        SameRelatable},
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
        {{"a1", SameRelatable}, {"a2", SameRelatable},
            {"a3", SameRelatable, Rank::elemental, Optionality::repeats}},
        SameRelatable},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            SubscriptDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"minval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"mod", {{"a", SameIntOrReal}, {"p", SameIntOrReal}}, SameIntOrReal},
    {"modulo", {{"a", SameIntOrReal}, {"p", SameIntOrReal}}, SameIntOrReal},
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
        {{"source", SameType, Rank::known},
            {"dim", {IntType, KindCode::dimArg}, Rank::scalar /*not optional*/},
            {"ncopies", AnyInt, Rank::scalar}},
        SameType, Rank::rankPlus1},
    {"sqrt", {{"x", SameFloating}}, SameFloating},
    {"sum", {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"tan", {{"x", SameFloating}}, SameFloating},
    {"tanh", {{"x", SameFloating}}, SameFloating},
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
        {{"array", Anything, Rank::anyOrAssumedRank}, SubscriptDefaultKIND},
        KINDInt, Rank::vector},
    {"ubound",
        {{"array", Anything, Rank::anyOrAssumedRank},
            {"dim", {IntType, KindCode::dimArg}, Rank::scalar},
            SubscriptDefaultKIND},
        KINDInt, Rank::scalar},
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
//   ALLOCATED, ASSOCIATED, EXTENDS_TYPE_OF, IS_CONTIGUOUS,
//   PRESENT, RANK, SAME_TYPE, STORAGE_SIZE
// TODO: Type inquiry intrinsic functions - these return constants
//  BIT_SIZE, DIGITS, EPSILON, HUGE, KIND, MAXEXPONENT, MINEXPONENT,
//  NEW_LINE, PRECISION, RADIX, RANGE, TINY
// TODO: Non-standard intrinsic functions
//  AND, OR, XOR, LSHIFT, RSHIFT, SHIFT, ZEXT, IZEXT,
//  COSD, SIND, TAND, ACOSD, ASIND, ATAND, ATAN2D, COMPL,
//  DCMPLX, EQV, NEQV, INT8, JINT, JNINT, KNINT, LOC,
//  QCMPLX, DREAL, DFLOAT, QEXT, QFLOAT, QREAL, DNUM,
//  INUM, JNUM, KNUM, QNUM, RNUM, RAN, RANF, ILEN, SIZEOF,
//  MCLOCK, SECNDS, COTAN, IBCHNG, ISHA, ISHC, ISHL, IXOR
//  IARG, IARGC, NARGS, NUMARG, BADDRESS, IADDR, CACHESIZE,
//  EOF, FP_CLASS, INT_PTR_KIND, ISNAN, MALLOC
//  probably more (these are PGI + Intel, possibly incomplete)

// The following table contains the intrinsic functions listed in
// Tables 16.2 and 16.3 in Fortran 2018.  The "unrestricted" functions
// in Table 16.2 can be used as actual arguments, PROCEDURE() interfaces,
// and procedure pointer targets.
struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
  bool isRestrictedSpecific{false};
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
        "max", true},
    {{"amax1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "max", true},
    {{"amin0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true},
    {{"amin1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true},
    {{"amod", {{"a", DefaultReal}, {"p", DefaultReal}}, DefaultReal}, "mod"},
    {{"anint", {{"a", DefaultReal}}, DefaultReal}},
    {{"asin", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan2", {{"y", DefaultReal}, {"x", DefaultReal}}, DefaultReal}},
    {{"cabs", {{"a", DefaultComplex}}, DefaultReal}, "abs"},
    {{"ccos", {{"a", DefaultComplex}}, DefaultComplex}, "cos"},
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
        DefaultInt}},
    {{"isign", {{"a", DefaultInt}, {"b", DefaultInt}}, DefaultInt}, "sign"},
    {{"len", {{"string", DefaultChar}}, DefaultInt}},
    {{"log", {{"x", DefaultReal}}, DefaultReal}},
    {{"log10", {{"x", DefaultReal}}, DefaultReal}},
    {{"max0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true},
    {{"max1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true},
    {{"min0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true},
    {{"min1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true},
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
    ActualArguments &arguments, parser::ContextualMessages &messages) const {
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
  // type and kind do so.  Check for missing non-optional arguments now, too.
  const ActualArgument *sameArg{nullptr};
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
    std::optional<DynamicType> type{arg->GetType()};
    if (!type.has_value()) {
      CHECK(arg->Rank() == 0);
      if (d.typePattern.kindCode == KindCode::typeless ||
          d.rank == Rank::elementalOrBOZ) {
        continue;
      }
      messages.Say(
          "typeless (BOZ) not allowed for '%s=' argument"_err_en_US, d.keyword);
      return std::nullopt;
    } else if (!d.typePattern.categorySet.test(type->category)) {
      messages.Say("actual argument for '%s=' has bad type '%s'"_err_en_US,
          d.keyword, type->AsFortran().data());
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
      argOk = type->kind == defaults.GetDefaultKind(TypeCategory::Integer);
      break;
    case KindCode::defaultRealKind:
      argOk = type->kind == defaults.GetDefaultKind(TypeCategory::Real);
      break;
    case KindCode::doublePrecision:
      argOk = type->kind == defaults.doublePrecisionKind();
      break;
    case KindCode::defaultCharKind:
      argOk = type->kind == defaults.GetDefaultKind(TypeCategory::Character);
      break;
    case KindCode::defaultLogicalKind:
      argOk = type->kind == defaults.GetDefaultKind(TypeCategory::Logical);
      break;
    case KindCode::any: argOk = true; break;
    case KindCode::kindArg:
      CHECK(type->category == TypeCategory::Integer);
      CHECK(kindArg == nullptr);
      kindArg = arg;
      argOk = true;
      break;
    case KindCode::dimArg:
      CHECK(type->category == TypeCategory::Integer);
      hasDimArg = true;
      argOk = true;
      break;
    case KindCode::same:
      if (sameArg == nullptr) {
        sameArg = arg;
      }
      argOk = type.value() == sameArg->GetType();
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
          "actual argument for '%s=' has bad type or kind '%s'"_err_en_US,
          d.keyword, type->AsFortran().data());
      return std::nullopt;
    }
  }

  // Check the ranks of the arguments against the intrinsic's interface.
  const ActualArgument *arrayArg{nullptr};
  const ActualArgument *knownArg{nullptr};
  const ActualArgument *shapeArg{nullptr};
  int elementalRank{0};
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (const ActualArgument * arg{actualForDummy[j]}) {
      if (IsAssumedRank(*arg->value) && d.rank != Rank::anyOrAssumedRank) {
        messages.Say("assumed-rank array cannot be forwarded to "
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
        CHECK(shapeArg == nullptr);
        shapeArg = arg;
        argOk = rank == 1 && arg->VectorSize().has_value();
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
        CHECK(knownArg == nullptr);
        knownArg = arg;
        argOk = true;
        break;
      case Rank::anyOrAssumedRank: argOk = true; break;
      case Rank::conformable:
        CHECK(arrayArg != nullptr);
        argOk = rank == 0 || rank == arrayArg->Rank();
        break;
      case Rank::dimRemoved:
        CHECK(arrayArg != nullptr);
        if (hasDimArg) {
          argOk = rank + 1 == arrayArg->Rank();
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
  if (result.categorySet.empty()) {
    if (!call.isSubroutineCall) {
      return std::nullopt;
    }
    CHECK(result.kindCode == KindCode::none);
  } else {
    // Determine the result type.
    if (call.isSubroutineCall) {
      return std::nullopt;
    }
    resultType = DynamicType{result.categorySet.LeastElement().value(), 0};
    switch (result.kindCode) {
    case KindCode::defaultIntegerKind:
      CHECK(result.categorySet == IntType);
      CHECK(resultType->category == TypeCategory::Integer);
      resultType->kind = defaults.GetDefaultKind(TypeCategory::Integer);
      break;
    case KindCode::defaultRealKind:
      CHECK(result.categorySet == CategorySet{resultType->category});
      CHECK(FloatingType.test(resultType->category));
      resultType->kind = defaults.GetDefaultKind(TypeCategory::Real);
      break;
    case KindCode::doublePrecision:
      CHECK(result.categorySet == RealType);
      CHECK(resultType->category == TypeCategory::Real);
      resultType->kind = defaults.doublePrecisionKind();
      break;
    case KindCode::defaultCharKind:
      CHECK(result.categorySet == CharType);
      CHECK(resultType->category == TypeCategory::Character);
      resultType->kind = defaults.GetDefaultKind(TypeCategory::Character);
      break;
    case KindCode::defaultLogicalKind:
      CHECK(result.categorySet == LogicalType);
      CHECK(resultType->category == TypeCategory::Logical);
      resultType->kind = defaults.GetDefaultKind(TypeCategory::Logical);
      break;
    case KindCode::same:
      CHECK(sameArg != nullptr);
      if (std::optional<DynamicType> aType{sameArg->GetType()}) {
        if (result.categorySet.test(aType->category)) {
          resultType = *aType;
        } else {
          resultType->kind = aType->kind;
        }
      }
      break;
    case KindCode::effectiveKind:
      CHECK(kindDummyArg != nullptr);
      CHECK(result.categorySet == CategorySet{resultType->category});
      if (kindArg != nullptr) {
        auto &expr{*kindArg->value};
        CHECK(expr.Rank() == 0);
        if (auto code{ToInt64(expr)}) {
          if (IsValidKindOfIntrinsicType(resultType->category, *code)) {
            resultType->kind = *code;
            break;
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
        CHECK(resultType->category == TypeCategory::Integer);
        resultType->kind = defaults.subscriptIntegerKind();
      } else {
        CHECK(kindDummyArg->optionality ==
            Optionality::defaultsToDefaultForResult);
        resultType->kind = defaults.GetDefaultKind(resultType->category);
      }
      break;
    case KindCode::likeMultiply:
      CHECK(dummies >= 2);
      CHECK(actualForDummy[0] != nullptr);
      CHECK(actualForDummy[1] != nullptr);
      resultType = actualForDummy[0]->GetType()->ResultTypeForMultiply(
          *actualForDummy[1]->GetType());
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
  case Rank::rankPlus1:
    CHECK(knownArg != nullptr);
    resultRank = knownArg->Rank() + 1;
    break;
  case Rank::shaped:
    CHECK(shapeArg != nullptr);
    resultRank = shapeArg->VectorSize().value();
    break;
  case Rank::elementalOrBOZ:
  case Rank::shape:
  case Rank::array:
  case Rank::known:
  case Rank::anyOrAssumedRank:
  case Rank::dimRemoved:
  case Rank::reduceOperation:
    common::die("INTERNAL: bad Rank code on intrinsic '%s' result", name);
    break;
  default: CRASH_NO_CASE;
  }
  CHECK(resultRank >= 0);

  semantics::Attrs attrs;
  if (elementalRank > 0) {
    attrs.set(semantics::Attr::ELEMENTAL);
  }

  // Rearrange the actual arguments into dummy argument order.
  ActualArguments rearranged(dummies);
  for (std::size_t j{0}; j < dummies; ++j) {
    if (ActualArgument * arg{actualForDummy[j]}) {
      rearranged[j] = std::move(*arg);
    }
  }

  return {SpecificCall{
      SpecificIntrinsic{name, std::move(resultType), resultRank, attrs},
      std::move(rearranged)}};
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

  std::optional<SpecificCall> Probe(const CallCharacteristics &,
      ActualArguments &, parser::ContextualMessages *) const;

  std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
  IsUnrestrictedSpecificIntrinsicFunction(const std::string &) const;

  std::ostream &Dump(std::ostream &) const;

private:
  common::IntrinsicTypeDefaultKinds defaults_;
  std::multimap<std::string, const IntrinsicInterface *> genericFuncs_;
  std::multimap<std::string, const SpecificIntrinsicInterface *> specificFuncs_;

  DynamicType GetSpecificType(const TypePattern &) const;
};

// Probe the configured intrinsic procedure pattern tables in search of a
// match for a given procedure reference.
std::optional<SpecificCall> IntrinsicProcTable::Implementation::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    parser::ContextualMessages *messages) const {
  if (call.isSubroutineCall) {
    return std::nullopt;  // TODO
  }
  parser::Messages *finalBuffer{messages ? messages->messages() : nullptr};
  // Probe the specific intrinsic function table first.
  parser::Messages specificBuffer;
  parser::ContextualMessages specificErrors{
      messages ? messages->at() : call.name,
      finalBuffer ? &specificBuffer : nullptr};
  std::string name{call.name.ToString()};
  auto specificRange{specificFuncs_.equal_range(name)};
  for (auto iter{specificRange.first}; iter != specificRange.second; ++iter) {
    if (auto specificCall{
            iter->second->Match(call, defaults_, arguments, specificErrors)}) {
      if (const char *genericName{iter->second->generic}) {
        specificCall->specificIntrinsic.name = genericName;
      }
      specificCall->specificIntrinsic.isRestrictedSpecific =
          iter->second->isRestrictedSpecific;
      return specificCall;
    }
  }
  // Probe the generic intrinsic function table next.
  parser::Messages genericBuffer;
  parser::ContextualMessages genericErrors{
      messages ? messages->at() : call.name,
      finalBuffer ? &genericBuffer : nullptr};
  auto genericRange{genericFuncs_.equal_range(name)};
  for (auto iter{genericRange.first}; iter != genericRange.second; ++iter) {
    if (auto specificCall{
            iter->second->Match(call, defaults_, arguments, genericErrors)}) {
      return specificCall;
    }
  }
  // Special cases of intrinsic functions
  if (call.name.ToString() == "null") {
    if (arguments.size() == 0) {
      // TODO: NULL() result type is determined by context
      // Can pass that context in, or return a token distinguishing
      // NULL, or represent NULL as a new kind of top-level expression
    } else if (arguments.size() > 1) {
      genericErrors.Say("too many arguments to NULL()"_err_en_US);
    } else if (arguments[0].has_value() && arguments[0]->keyword.has_value() &&
        arguments[0]->keyword->ToString() != "mold") {
      genericErrors.Say("unknown argument '%s' to NULL()"_err_en_US,
          arguments[0]->keyword->ToString().data());
    } else {
      // TODO: Argument must be pointer, procedure pointer, or allocatable.
      // Characteristics, including dynamic length type parameter values,
      // must be taken from the MOLD argument.
      // TODO: set Attr::POINTER on NULL result
    }
  }
  // No match
  if (finalBuffer) {
    if (genericBuffer.empty()) {
      finalBuffer->Annex(std::move(specificBuffer));
    } else {
      finalBuffer->Annex(std::move(genericBuffer));
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
      UnrestrictedSpecificIntrinsicFunctionInterface result;
      if (specific.generic != nullptr) {
        result.genericName = std::string(specific.generic);
      } else {
        result.genericName = name;
      }
      result.numArguments = specific.CountArguments();
      result.argumentType = GetSpecificType(specific.dummy[0].typePattern);
      result.resultType = GetSpecificType(specific.result);
      return result;
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

std::optional<SpecificCall> IntrinsicProcTable::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    parser::ContextualMessages *messages) const {
  CHECK(impl_ != nullptr || !"IntrinsicProcTable: not configured");
  return impl_->Probe(call, arguments, messages);
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
