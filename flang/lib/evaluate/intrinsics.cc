// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#include "type.h"
#include "../common/enum-set.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../semantics/expression.h"
#include <map>
#include <string>
#include <utility>

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
// categories, a kind pattern, a rank pattern, and an optional special
// note code.  The kind and rank patterns are represented here with code
// values that are significant to the matching/validation engine.

// These are small bit-sets of type category enumerators.
// Note that typeless (BOZ literal) values don't have a distinct type category.
// These typeless arguments are represented in the tables as if they were
// INTEGER with a special "typeless" kind code.  Arguments of intrinsic types
// that can also be be typeless values are encoded with a "BOZisOK" note code.
using CategorySet = common::EnumSet<TypeCategory, 8>;
static constexpr CategorySet Int{TypeCategory::Integer};
static constexpr CategorySet Real{TypeCategory::Real};
static constexpr CategorySet Complex{TypeCategory::Complex};
static constexpr CategorySet Char{TypeCategory::Character};
static constexpr CategorySet Logical{TypeCategory::Logical};
static constexpr CategorySet IntOrReal{Int | Real};
static constexpr CategorySet Floating{Real | Complex};
static constexpr CategorySet Numeric{Int | Real | Complex};
static constexpr CategorySet Relatable{Int | Real | Char};
static constexpr CategorySet IntrinsicType{
    Int | Real | Complex | Char | Logical};
static constexpr CategorySet AnyType{
    IntrinsicType | CategorySet{TypeCategory::Derived}};

enum class KindCode {
  none,
  defaultIntegerKind,
  defaultRealKind,  // is also the default COMPLEX kind
  doublePrecision,
  defaultCharKind,
  defaultLogicalKind,
  any,  // matches any kind value; each instance is independent
  typeless,  // BOZ literals are INTEGER with this kind
  teamType,  // TEAM_TYPE from module ISO_FORTRAN_ENV (for coarrays)
  kindArg,  // this argument is KIND=
  effectiveKind,  // for function results: same "kindArg", possibly defaulted
  dimArg,  // this argument is DIM=
  same,  // match any kind; all "same" kinds must be equal
};

struct TypePattern {
  CategorySet categorySet;
  KindCode kindCode{KindCode::none};
};

// Abbreviations for argument and result patterns in the intrinsic prototypes:

// Match specific kinds of intrinsic types
static constexpr TypePattern DftInt{Int, KindCode::defaultIntegerKind};
static constexpr TypePattern DftReal{Real, KindCode::defaultRealKind};
static constexpr TypePattern DftComplex{Complex, KindCode::defaultRealKind};
static constexpr TypePattern DftChar{Char, KindCode::defaultCharKind};
static constexpr TypePattern DftLogical{Logical, KindCode::defaultLogicalKind};
static constexpr TypePattern BOZ{Int, KindCode::typeless};
static constexpr TypePattern TEAM_TYPE{Int, KindCode::teamType};
static constexpr TypePattern DoublePrecision{Real, KindCode::doublePrecision};

// Match any kind of some intrinsic or derived types
static constexpr TypePattern AnyInt{Int, KindCode::any};
static constexpr TypePattern AnyReal{Real, KindCode::any};
static constexpr TypePattern AnyIntOrReal{IntOrReal, KindCode::any};
static constexpr TypePattern AnyComplex{Complex, KindCode::any};
static constexpr TypePattern AnyNumeric{Numeric, KindCode::any};
static constexpr TypePattern AnyChar{Char, KindCode::any};
static constexpr TypePattern AnyLogical{Logical, KindCode::any};
static constexpr TypePattern AnyRelatable{Relatable, KindCode::any};

// Match some kind of some intrinsic type(s); all "Same" values must match,
// even when not in the same category (e.g., SameComplex and SameReal).
// Can be used to specify a result so long as at least one argument is
// a "Same".
static constexpr TypePattern SameInt{Int, KindCode::same};
static constexpr TypePattern SameReal{Real, KindCode::same};
static constexpr TypePattern SameIntOrReal{IntOrReal, KindCode::same};
static constexpr TypePattern SameComplex{Complex, KindCode::same};
static constexpr TypePattern SameFloating{Floating, KindCode::same};
static constexpr TypePattern SameNumeric{Numeric, KindCode::same};
static constexpr TypePattern SameChar{Char, KindCode::same};
static constexpr TypePattern SameLogical{Logical, KindCode::same};
static constexpr TypePattern SameRelatable{Relatable, KindCode::same};
static constexpr TypePattern SameIntrinsic{IntrinsicType, KindCode::same};
static constexpr TypePattern SameDerivedType{
    CategorySet{TypeCategory::Derived}, KindCode::same};
static constexpr TypePattern SameType{AnyType, KindCode::same};

// Result types with known category and KIND=
static constexpr TypePattern KINDInt{Int, KindCode::effectiveKind};
static constexpr TypePattern KINDReal{Real, KindCode::effectiveKind};
static constexpr TypePattern KINDComplex{Complex, KindCode::effectiveKind};
static constexpr TypePattern KINDChar{Char, KindCode::effectiveKind};
static constexpr TypePattern KINDLogical{Logical, KindCode::effectiveKind};

// The default rank pattern for dummy arguments and function results is
// "elemental".
enum class Rank {
  elemental,  // scalar, or array that conforms with other array arguments
  scalar,
  vector,
  shape,  // INTEGER vector of known length and no negative element
  matrix,
  array,  // not scalar, rank is known and greater than zero
  known,  // rank is known and can be scalar
  anyOrAssumedRank,  // rank can be unknown
  conformable,  // scalar, or array of same rank & shape as "array" argument
  dimReduced,  // scalar if no DIM= argument, else rank(array)-1
  dimRemoved,  // scalar, or rank(array)-1
  rankPlus1,  // rank(known)+1
  shaped,  // rank is length of SHAPE vector
};

enum SpecialNote {
  none = 0,
  BOZisOK,  // typeless BOZ literal actual argument is also acceptable
  optional,
  defaultsToSameKind,  // SameInt, &c.; OPTIONAL also implied
  defaultsToDefaultForResult,  // OPTIONAL also implied
};

struct IntrinsicDummyArgument {
  const char *keyword{nullptr};
  TypePattern typePattern;
  Rank rank{Rank::elemental};
  enum SpecialNote note { none };
};

// constexpr abbreviations for popular arguments:
// DefaultingKIND is a KIND= argument whose default value is the appropriate
// KIND(0), KIND(0.0), KIND(''), &c. value for the function result.
static constexpr IntrinsicDummyArgument DefaultingKIND{
    "kind", {Int, KindCode::kindArg}, Rank::scalar, defaultsToDefaultForResult};
// MatchingDefaultKIND is a KIND= argument whose default value is the
// kind of any "Same" function argument (viz., the one whose kind pattern is
// "same").
static constexpr IntrinsicDummyArgument MatchingDefaultKIND{
    "kind", {Int, KindCode::kindArg}, Rank::scalar, defaultsToSameKind};
static constexpr IntrinsicDummyArgument OptionalDIM{
    "dim", {Int, KindCode::dimArg}, Rank::scalar, optional};
static constexpr IntrinsicDummyArgument OptionalMASK{
    "mask", AnyLogical, Rank::conformable, optional};

struct IntrinsicInterface {
  static constexpr int maxArguments{7};
  const char *name{nullptr};
  IntrinsicDummyArgument dummy[maxArguments];
  TypePattern result;
  Rank rank{Rank::elemental};
  std::optional<SpecificIntrinsic> Match(const CallCharacteristics &,
      const semantics::IntrinsicTypeDefaultKinds &) const;
};

static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", SameIntOrReal}}, SameIntOrReal},
    {"abs", {{"a", SameComplex}}, SameReal},
    {"achar", {{"i", SameInt}, DefaultingKIND}, KINDChar},
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
    {"bessel_y0", {{"x", SameReal}}, SameReal},
    {"bessel_y1", {{"x", SameReal}}, SameReal},
    {"bessel_yn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bge",
        {{"i", AnyInt, Rank::elemental, BOZisOK},
            {"j", AnyInt, Rank::elemental, BOZisOK}},
        DftLogical},
    {"bgt",
        {{"i", AnyInt, Rank::elemental, BOZisOK},
            {"j", AnyInt, Rank::elemental, BOZisOK}},
        DftLogical},
    {"ble",
        {{"i", AnyInt, Rank::elemental, BOZisOK},
            {"j", AnyInt, Rank::elemental, BOZisOK}},
        DftLogical},
    {"blt",
        {{"i", AnyInt, Rank::elemental, BOZisOK},
            {"j", AnyInt, Rank::elemental, BOZisOK}},
        DftLogical},
    {"btest", {{"i", AnyInt}, {"pos", AnyInt}}, DftLogical},
    {"ceiling", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"char", {{"i", AnyInt}, DefaultingKIND}, KINDChar},
    {"cmplx", {{"x", AnyComplex}, DefaultingKIND}, KINDComplex},
    {"cmplx",
        {{"x", SameIntOrReal, Rank::elemental, BOZisOK},
            {"y", SameIntOrReal, Rank::elemental, BOZisOK}, DefaultingKIND},
        KINDComplex},
    {"conjg", {{"z", SameComplex}}, SameComplex},
    {"cos", {{"x", SameFloating}}, SameFloating},
    {"cosh", {{"x", SameFloating}}, SameFloating},
    {"count", {{"mask", AnyLogical, Rank::array}, OptionalDIM, DefaultingKIND},
        KINDInt, Rank::dimReduced},
    {"cshift",
        {{"array", SameType, Rank::array}, {"shift", AnyInt, Rank::dimRemoved},
            OptionalDIM},
        SameType, Rank::array},
    {"dim", {{"x", SameIntOrReal}, {"y", SameIntOrReal}}, SameIntOrReal},
    {"dprod", {{"x", DftReal}, {"y", DftReal}}, DoublePrecision},
    {"dshiftl",
        {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftl", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"dshiftr",
        {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftr", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"eoshift",
        {{"array", SameIntrinsic, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameIntrinsic, Rank::dimRemoved, optional},
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
    {"exponent", {{"x", AnyReal}}, DftInt},
    {"findloc",
        {{"array", SameNumeric, Rank::array},
            {"value", SameNumeric, Rank::scalar}, OptionalDIM, OptionalMASK,
            DefaultingKIND, {"back", AnyLogical, Rank::scalar, optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", SameChar, Rank::array}, {"value", SameChar, Rank::scalar},
            OptionalDIM, OptionalMASK, DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, OptionalDIM, OptionalMASK,
            DefaultingKIND, {"back", AnyLogical, Rank::scalar, optional}},
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
    {"iand", {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK}},
        SameInt},
    {"iand", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ibclr", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ibits", {{"i", SameInt}, {"pos", AnyInt}, {"len", AnyInt}}, SameInt},
    {"ibset", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ichar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"ieor", {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK}},
        SameInt},
    {"ieor", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"image_status",
        {{"image", SameInt}, {"team", TEAM_TYPE, Rank::scalar, optional}},
        DftInt},
    {"index",
        {{"string", SameChar}, {"substring", SameChar},
            {"back", AnyLogical, Rank::scalar, optional}, DefaultingKIND},
        KINDInt},
    {"int", {{"a", AnyNumeric, Rank::elemental, BOZisOK}, DefaultingKIND},
        KINDInt},
    {"ior", {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK}},
        SameInt},
    {"ior", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ishft", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"ishftc",
        {{"i", SameInt}, {"shift", AnyInt},
            {"size", AnyInt, Rank::elemental, optional}},
        SameInt},
    {"is_iostat_end", {{"i", AnyInt}}, DftLogical},
    {"is_iostat_eor", {{"i", AnyInt}}, DftLogical},
    {"leadz", {{"i", AnyInt}}, DftInt},
    {"len", {{"string", AnyChar}, DefaultingKIND}, KINDInt},
    {"len_trim", {{"string", AnyChar}, DefaultingKIND}, KINDInt},
    {"lge", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"lgt", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"lle", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"llt", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"log", {{"x", SameFloating}}, SameFloating},
    {"log10", {{"x", SameReal}}, SameReal},
    {"logical", {{"l", AnyLogical}, DefaultingKIND}, KINDLogical},
    {"log_gamma", {{"x", SameReal}}, SameReal},
    {"maskl", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maskr", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maxloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            DefaultingKIND, {"back", AnyLogical, Rank::scalar, optional}},
        KINDInt, Rank::dimReduced},
    {"maxval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"merge_bits",
        {{"i", SameInt}, {"j", SameInt, Rank::elemental, BOZisOK},
            {"mask", SameInt, Rank::elemental, BOZisOK}},
        SameInt},
    {"merge_bits",
        {{"i", BOZ}, {"j", SameInt},
            {"mask", SameInt, Rank::elemental, BOZisOK}},
        SameInt},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            DefaultingKIND, {"back", AnyLogical, Rank::scalar, optional}},
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
    {"out_of_range",
        {{"x", SameIntOrReal}, {"mold", AnyIntOrReal, Rank::scalar}},
        DftLogical},
    {"out_of_range",
        {{"x", AnyReal}, {"mold", AnyInt, Rank::scalar},
            {"round", AnyLogical, Rank::scalar, optional}},
        DftLogical},
    {"out_of_range", {{"x", AnyReal}, {"mold", AnyReal}}, DftLogical},
    {"pack",
        {{"array", SameType, Rank::array},
            {"mask", AnyLogical, Rank::conformable},
            {"vector", SameType, Rank::vector, optional}},
        SameType, Rank::vector},
    {"parity", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"popcnt", {{"i", AnyInt}}, DftInt},
    {"poppar", {{"i", AnyInt}}, DftInt},
    {"product",
        {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"real", {{"a", AnyNumeric, Rank::elemental, BOZisOK}, DefaultingKIND},
        KINDReal},
    {"reshape",
        {{"source", SameType, Rank::array}, {"shape", AnyInt, Rank::shape},
            {"pad", SameType, Rank::array, optional},
            {"order", AnyInt, Rank::vector, optional}},
        SameType, Rank::shaped},
    {"rrspacing", {{"x", SameReal}}, SameReal},
    {"scale", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"scan",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, optional}, DefaultingKIND},
        KINDInt},
    {"set_exponent", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"shifta", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftl", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftr", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"sign", {{"a", SameIntOrReal}, {"b", SameIntOrReal}}, SameIntOrReal},
    {"sin", {{"x", SameFloating}}, SameFloating},
    {"sinh", {{"x", SameFloating}}, SameFloating},
    {"spacing", {{"x", SameReal}}, SameReal},
    {"spread",
        {{"source", SameType, Rank::known},
            {"dim", {Int, KindCode::dimArg}, Rank::scalar /*not optional*/},
            {"ncopies", AnyInt, Rank::scalar}},
        SameType, Rank::rankPlus1},
    {"sqrt", {{"x", SameFloating}}, SameFloating},
    {"sum", {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"tan", {{"x", SameFloating}}, SameFloating},
    {"tanh", {{"x", SameFloating}}, SameFloating},
    {"trailz", {{"i", AnyInt}}, DftInt},
    // TODO: pmk: continue here with TRANSFER
    {"verify",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, optional}, DefaultingKIND},
        KINDInt},
};

// Not covered by the table above:
// MAX, MIN, MERGE

struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
};

static const SpecificIntrinsicInterface specificIntrinsicFunction[]{
    {{"abs", {{"a", DftReal}}, DftReal}},
    {{"acos", {{"x", DftReal}}, DftReal}},
    {{"aimag", {{"z", DftComplex}}, DftReal}},
    {{"aint", {{"a", DftReal}}, DftReal}},
    {{"alog", {{"x", DftReal}}, DftReal}, "log"},
    {{"alog10", {{"x", DftReal}}, DftReal}, "log10"},
    {{"amod", {{"a", DftReal}, {"p", DftReal}}, DftReal}, "mod"},
    {{"anint", {{"a", DftReal}}, DftReal}},
    {{"asin", {{"x", DftReal}}, DftReal}},
    {{"atan", {{"x", DftReal}}, DftReal}},
    {{"atan2", {{"y", DftReal}, {"x", DftReal}}, DftReal}},
    {{"cabs", {{"a", DftComplex}}, DftReal}, "abs"},
    {{"ccos", {{"a", DftComplex}}, DftComplex}, "cos"},
    {{"cexp", {{"a", DftComplex}}, DftComplex}, "exp"},
    {{"clog", {{"a", DftComplex}}, DftComplex}, "log"},
    {{"conjg", {{"a", DftComplex}}, DftComplex}},
    {{"cos", {{"x", DftReal}}, DftReal}},
    {{"csin", {{"a", DftComplex}}, DftComplex}, "sin"},
    {{"csqrt", {{"a", DftComplex}}, DftComplex}, "sqrt"},
    {{"ctan", {{"a", DftComplex}}, DftComplex}, "tan"},
    {{"dabs", {{"a", DoublePrecision}}, DoublePrecision}, "abs"},
    {{"dacos", {{"x", DoublePrecision}}, DoublePrecision}, "acos"},
    {{"dasin", {{"x", DoublePrecision}}, DoublePrecision}, "asin"},
    {{"datan", {{"x", DoublePrecision}}, DoublePrecision}, "atan"},
    {{"datan2", {{"y", DoublePrecision}, {"x", DoublePrecision}},
         DoublePrecision},
        "atan2"},
    {{"dble", {{"a", DftReal}, DefaultingKIND}, DoublePrecision}, "real"},
    {{"dcos", {{"x", DoublePrecision}}, DoublePrecision}, "cos"},
    {{"dcosh", {{"x", DoublePrecision}}, DoublePrecision}, "cosh"},
    {{"ddim", {{"x", DoublePrecision}, {"y", DoublePrecision}},
         DoublePrecision},
        "dim"},
    {{"dexp", {{"x", DoublePrecision}}, DoublePrecision}, "exp"},
    {{"dim", {{"x", DftReal}, {"y", DftReal}}, DftReal}},
    {{"dint", {{"a", DoublePrecision}}, DoublePrecision}, "aint"},
    {{"dlog", {{"x", DoublePrecision}}, DoublePrecision}, "log"},
    {{"dlog10", {{"x", DoublePrecision}}, DoublePrecision}, "log10"},
    {{"dmod", {{"a", DoublePrecision}, {"p", DoublePrecision}},
         DoublePrecision},
        "mod"},
    {{"dnint", {{"a", DoublePrecision}}, DoublePrecision}, "anint"},
    {{"dprod", {{"x", DftReal}, {"y", DftReal}}, DoublePrecision}},
    {{"dsign", {{"a", DoublePrecision}, {"b", DoublePrecision}},
         DoublePrecision},
        "sign"},
    {{"dsin", {{"x", DoublePrecision}}, DoublePrecision}, "sin"},
    {{"dsinh", {{"x", DoublePrecision}}, DoublePrecision}, "sinh"},
    {{"dsqrt", {{"x", DoublePrecision}}, DoublePrecision}, "sqrt"},
    {{"dtan", {{"x", DoublePrecision}}, DoublePrecision}, "tan"},
    {{"dtanh", {{"x", DoublePrecision}}, DoublePrecision}, "tanh"},
    {{"exp", {{"x", DftReal}}, DftReal}},
    {{"float", {{"i", DftInt}}, DftReal}, "real"},
    {{"iabs", {{"a", DftInt}}, DftInt}, "abs"},
    {{"idim", {{"x", DftInt}, {"y", DftInt}}, DftInt}, "dim"},
    {{"idint", {{"a", DoublePrecision}}, DftInt}, "int"},
    {{"idnint", {{"a", DoublePrecision}}, DftInt}, "nint"},
    {{"ifix", {{"a", DftReal}}, DftInt}, "int"},
    {{"index", {{"string", DftChar}, {"substring", DftChar}}, DftInt}},
    {{"isign", {{"a", DftInt}, {"b", DftInt}}, DftInt}, "sign"},
    {{"len", {{"string", DftChar}}, DftInt}},
    {{"log", {{"x", DftReal}}, DftReal}},
    {{"log10", {{"x", DftReal}}, DftReal}},
    {{"mod", {{"a", DftInt}, {"p", DftInt}}, DftInt}},
    {{"nint", {{"a", DftReal}}, DftInt}},
    {{"sign", {{"a", DftReal}, {"b", DftReal}}, DftReal}},
    {{"sin", {{"x", DftReal}}, DftReal}},
    {{"sinh", {{"x", DftReal}}, DftReal}},
    {{"sngl", {{"a", DoublePrecision}}, DftReal}, "real"},
    {{"sqrt", {{"x", DftReal}}, DftReal}},
    {{"tan", {{"x", DftReal}}, DftReal}},
    {{"tanh", {{"x", DftReal}}, DftReal}},
};

// Some entries in the table above are "restricted" specifics:
//   DBLE, FLOAT, IDINT, IFIX, SNGL
// Additional "restricted" specifics not covered by the table above:
//   AMAX0, AMAX1, AMIN0, AMIN1, DMAX1, DMIN1, MAX0, MAX1, MIN0, MIN1

// Intrinsic interface matching against the arguments of a particular
// procedure reference.
// TODO: return error message rather than just a std::nullopt on failure.
std::optional<SpecificIntrinsic> IntrinsicInterface::Match(
    const CallCharacteristics &call,
    const semantics::IntrinsicTypeDefaultKinds &defaults) const {
  // Attempt to construct a 1-1 correspondence between the dummy arguments in
  // a particular intrinsic procedure's generic interface and the actual
  // arguments in a procedure reference.
  const ActualArgumentCharacteristics *actualForDummy[maxArguments];
  int dummies{0};
  for (; dummies < maxArguments && dummy[dummies].keyword != nullptr;
       ++dummies) {
    actualForDummy[dummies] = nullptr;
  }
  for (const ActualArgumentCharacteristics &arg : call.argument) {
    bool found{false};
    for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
      if (actualForDummy[dummyArgIndex] == nullptr) {
        if (!arg.keyword.has_value() ||
            *arg.keyword == dummy[dummyArgIndex].keyword) {
          actualForDummy[dummyArgIndex] = &arg;
          found = true;
          break;
        }
      }
      if (!found) {
        return std::nullopt;
      }
    }
  }

  // Check types and kinds of the actual arguments against the intrinsic's
  // interface.  Ensure that two or more arguments that have to have the same
  // type and kind do so.  Check for missing non-optional arguments now, too.
  const ActualArgumentCharacteristics *sameArg{nullptr};
  const IntrinsicDummyArgument *kindDummyArg{nullptr};
  const ActualArgumentCharacteristics *kindArg{nullptr};
  bool hasDimArg{false};
  for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
    const IntrinsicDummyArgument &d{dummy[dummyArgIndex]};
    if (d.typePattern.kindCode == KindCode::kindArg) {
      CHECK(kindDummyArg == nullptr);
      kindDummyArg = &d;
    }
    const ActualArgumentCharacteristics *arg{actualForDummy[dummyArgIndex]};
    if (!arg) {
      if (d.note >= optional) {
        continue;  // missing OPTIONAL argument is ok
      } else {
        return std::nullopt;  // missing non-OPTIONAL argument
      }
    }
    if (arg->isBOZ) {
      if (d.typePattern.kindCode == KindCode::typeless || d.note == BOZisOK) {
        continue;
      }
      return std::nullopt;  // typeless argument not allowed here
    } else if (!d.typePattern.categorySet.test(arg->type.category)) {
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
      argOk = arg->type.kind == defaults.defaultIntegerKind;
      break;
    case KindCode::defaultRealKind:
      argOk = arg->type.kind == defaults.defaultRealKind;
      break;
    case KindCode::doublePrecision:
      argOk = arg->type.kind == defaults.defaultDoublePrecisionKind;
      break;
    case KindCode::defaultCharKind:
      argOk = arg->type.kind == defaults.defaultCharacterKind;
      break;
    case KindCode::defaultLogicalKind:
      argOk = arg->type.kind == defaults.defaultLogicalKind;
      break;
    case KindCode::any: argOk = true; break;
    case KindCode::kindArg:
      CHECK(kindArg == nullptr);
      kindArg = arg;
      argOk = arg->intValue.has_value();
      break;
    case KindCode::dimArg:
      hasDimArg = true;
      argOk = true;
      break;
    case KindCode::same:
      if (sameArg == nullptr) {
        sameArg = arg;
      }
      argOk = arg->type == sameArg->type;
      break;
    case KindCode::effectiveKind:
      common::die("INTERNAL: KindCode::effectiveKind appears on argument '%s' "
                  "for intrinsic '%s'",
          d.keyword, name);
      break;
    default: CRASH_NO_CASE;
    }
    if (!argOk) {
      return std::nullopt;
    }
  }

  // Check the ranks of the arguments against the intrinsic's interface.
  const ActualArgumentCharacteristics *arrayArg{nullptr};
  const ActualArgumentCharacteristics *knownArg{nullptr};
  const ActualArgumentCharacteristics *shapeArg{nullptr};
  int elementalRank{0};
  for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
    const IntrinsicDummyArgument &d{dummy[dummyArgIndex]};
    if (const ActualArgumentCharacteristics *
        arg{actualForDummy[dummyArgIndex]}) {
      if (arg->isAssumedRank && d.rank != Rank::anyOrAssumedRank) {
        return std::nullopt;
      }
      bool argOk{false};
      switch (d.rank) {
      case Rank::elemental:
        if (elementalRank == 0) {
          elementalRank = arg->rank;
        }
        argOk = arg->rank == 0 || arg->rank == elementalRank;
        break;
      case Rank::scalar: argOk = arg->rank == 0; break;
      case Rank::vector: argOk = arg->rank == 1; break;
      case Rank::shape:
        CHECK(shapeArg == nullptr);
        shapeArg = arg;
        argOk = arg->rank == 1 && arg->vectorSize.has_value();
        break;
      case Rank::matrix: argOk = arg->rank == 2; break;
      case Rank::array:
        argOk = arg->rank > 0;
        if (!arrayArg) {
          arrayArg = arg;
        } else {
          argOk &= arg->rank == arrayArg->rank;
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
        argOk = arg->rank == 0 || arg->rank == arrayArg->rank;
        break;
      case Rank::dimRemoved:
        CHECK(arrayArg != nullptr);
        if (hasDimArg) {
          argOk = arg->rank + 1 == arrayArg->rank;
        } else {
          argOk = arg->rank == 0;
        }
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
        return std::nullopt;
      }
    }
  }

  // At this point, the call is acceptable.
  // Calculate the characteristics of the function result, if any
  if (result.categorySet.empty()) {
    CHECK(result.kindCode == KindCode::none);
    return std::make_optional<SpecificIntrinsic>(name);
  }
  // Determine the result type.
  DynamicType resultType{*result.categorySet.LeastElement(), 0};
  switch (result.kindCode) {
  case KindCode::defaultIntegerKind:
    CHECK(result.categorySet == Int);
    CHECK(resultType.category == TypeCategory::Integer);
    resultType.kind = defaults.defaultIntegerKind;
    break;
  case KindCode::defaultRealKind:
    CHECK(result.categorySet == CategorySet{resultType.category});
    CHECK(Floating.test(resultType.category));
    resultType.kind = defaults.defaultRealKind;
    break;
  case KindCode::doublePrecision:
    CHECK(result.categorySet == Real);
    CHECK(resultType.category == TypeCategory::Real);
    resultType.kind = defaults.defaultDoublePrecisionKind;
    break;
  case KindCode::defaultCharKind:
    CHECK(result.categorySet == Char);
    CHECK(resultType.category == TypeCategory::Character);
    resultType.kind = defaults.defaultCharacterKind;
    break;
  case KindCode::defaultLogicalKind:
    CHECK(result.categorySet == Logical);
    CHECK(resultType.category == TypeCategory::Logical);
    resultType.kind = defaults.defaultLogicalKind;
    break;
  case KindCode::same:
    CHECK(sameArg != nullptr);
    CHECK(result.categorySet.test(sameArg->type.category));
    resultType = sameArg->type;
    break;
  case KindCode::effectiveKind:
    CHECK(kindDummyArg != nullptr);
    CHECK(result.categorySet == CategorySet{resultType.category});
    if (kindArg != nullptr) {
      CHECK(kindArg->intValue.has_value());
      resultType.kind = *kindArg->intValue;
      // TODO pmk: validate the kind!!
    } else if (kindDummyArg->note == defaultsToSameKind) {
      CHECK(sameArg != nullptr);
      resultType = sameArg->type;
    } else {
      CHECK(kindDummyArg->note == defaultsToDefaultForResult);
      resultType.kind = defaults.DefaultKind(resultType.category);
    }
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

  // Determine the rank of the function result.
  int resultRank{0};
  switch (rank) {
  case Rank::elemental: resultRank = elementalRank; break;
  case Rank::scalar: resultRank = 0; break;
  case Rank::vector: resultRank = 1; break;
  case Rank::matrix: resultRank = 2; break;
  case Rank::dimReduced:
    CHECK(arrayArg != nullptr);
    resultRank = hasDimArg ? arrayArg->rank - 1 : 0;
    break;
  case Rank::rankPlus1:
    CHECK(knownArg != nullptr);
    resultRank = knownArg->rank + 1;
    break;
  case Rank::shaped:
    CHECK(shapeArg != nullptr);
    CHECK(shapeArg->vectorSize.has_value());
    resultRank = *shapeArg->vectorSize;
    break;
  case Rank::shape:
  case Rank::array:
  case Rank::known:
  case Rank::anyOrAssumedRank:
  case Rank::conformable:
  case Rank::dimRemoved:
    common::die("INTERNAL: bad Rank code on intrinsic '%s' result", name);
    break;
  default: CRASH_NO_CASE;
  }
  CHECK(resultRank >= 0);

  return std::make_optional<SpecificIntrinsic>(
      name, elementalRank > 0, resultType, resultRank);
}

struct IntrinsicTable::Implementation {
  explicit Implementation(const semantics::IntrinsicTypeDefaultKinds &dfts)
    : defaults{dfts} {
    for (const IntrinsicInterface &f : genericIntrinsicFunction) {
      genericFuncs.insert(std::make_pair(std::string{f.name}, &f));
    }
    for (const SpecificIntrinsicInterface &f : specificIntrinsicFunction) {
      specificFuncs.insert(std::make_pair(std::string{f.name}, &f));
    }
  }

  semantics::IntrinsicTypeDefaultKinds defaults;
  std::multimap<std::string, const IntrinsicInterface *> genericFuncs;
  std::multimap<std::string, const SpecificIntrinsicInterface *> specificFuncs;
};

IntrinsicTable::~IntrinsicTable() {
  delete impl_;
  impl_ = nullptr;
}

IntrinsicTable IntrinsicTable::Configure(
    const semantics::IntrinsicTypeDefaultKinds &defaults) {
  IntrinsicTable result;
  result.impl_ = new IntrinsicTable::Implementation(defaults);
  return result;
}

std::optional<SpecificIntrinsic> IntrinsicTable::Probe(
    const CallCharacteristics &call) {
  CHECK(impl_ != nullptr || !"IntrinsicTable: not configured");
  if (call.isSubroutineCall) {
    return std::nullopt;  // TODO
  }
}
}  // namespace Fortran::evaluate
