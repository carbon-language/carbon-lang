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
#include "../semantics/expression.h"

namespace Fortran::evaluate {

using common::TypeCategory;

// Dummy arguments to generic intrinsic procedures are each specified by
// name, allowable type categories, a kind pattern, and a rank pattern.

using CategorySet = common::EnumSet<TypeCategory, 32>;

static constexpr CategorySet Int{TypeCategory::Integer},
    Real{TypeCategory::Real}, Complex{TypeCategory::Complex},
    Char{TypeCategory::Character}, Logical{TypeCategory::Logical},
    IntOrReal{Int | Real}, Floating{Real | Complex},
    Numeric{Int | Real | Complex}, Relatable{Int | Real | Char},
    IntrinsicType{Int | Real | Complex | Char | Logical},
    AnyType{IntrinsicType | CategorySet{TypeCategory::Derived}};

enum class KindCode {
  none,
  defaultIntegerKind,
  defaultRealKind,
  doublePrecision,
  defaultCharacterKind,
  defaultLogicalKind,
  any,  // for AnyType
  typeless,  // BOZ literal
  teamType,  // coarray TEAM_TYPE
  kindArg,  // this is the KIND= argument
  effectiveKind,  // value of KIND=, possibly defaulted, for function result
  dimArg,  // this is the DIM= argument
  varA,  // pattern variables match any supported actual kind, must match
  varB,
  varC,
  varD,
  varE,
};

struct TypePattern {
  CategorySet categorySet;
  KindCode kindCode{KindCode::none};
};

// Abbreviations for arguments and results in the prototype specifications
// in the tables below
static constexpr TypePattern Id{Int, KindCode::defaultIntegerKind},
    IA{Int, KindCode::varA}, IB{Int, KindCode::varB}, IC{Int, KindCode::varC},
    KIND{Int, KindCode::kindArg}, BOZ{Int, KindCode::typeless},
    TEAM_TYPE{Int, KindCode::teamType}, IK{Int, KindCode::effectiveKind},
    Rd{Real, KindCode::defaultRealKind}, DP{Real, KindCode::doublePrecision},
    RA{Real, KindCode::varA}, RB{Real, KindCode::varB},
    RK{Real, KindCode::effectiveKind}, Zd{Complex, KindCode::defaultRealKind},
    ZA{Complex, KindCode::varA}, ZK{Complex, KindCode::effectiveKind},
    Chd{Char, KindCode::defaultCharacterKind}, ChA{Char, KindCode::varA},
    ChK{Char, KindCode::effectiveKind},
    Ld{Logical, KindCode::defaultLogicalKind}, LA{Logical, KindCode::varA},
    LB{Logical, KindCode::varB}, LC{Logical, KindCode::varC},
    LD{Logical, KindCode::varD}, LE{Logical, KindCode::varE},
    LK{Logical, KindCode::effectiveKind},
    DT{CategorySet{TypeCategory::Derived}, KindCode::any},
    IntOrRealA{IntOrReal, KindCode::varA},
    IntOrRealB{IntOrReal, KindCode::varB}, NumericA{Numeric, KindCode::varA},
    NumericB{Numeric, KindCode::varB}, FloatingA{Floating, KindCode::varA},
    RelatableA{Relatable, KindCode::varA},
    IntrinsicA{IntrinsicType, KindCode::varA}, Anything{AnyType, KindCode::any};

// The default rank pattern for dummy arguments and function results is
// "elemental".
enum RankPattern {
  elemental = 0,  // scalar, or array that conforms with other array arguments
  scalar,
  vector,
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
  BOZOK,  // typeless BOZ literal actual argument is also acceptable
  optional,
  kindDefaultsToA,  // OPTIONAL also implied
  kindDefaultsToDefaultForResult,  // OPTIONAL also implied
};

struct IntrinsicDummyArgument {
  const char *keyword{nullptr};
  TypePattern typePattern;
  RankPattern rank{elemental};
  enum SpecialNote note { none };
};

static constexpr IntrinsicDummyArgument DefaultingKind{
    "kind", KIND, scalar, kindDefaultsToDefaultForResult},
    MatchingDefaultKind{"kind", KIND, scalar, kindDefaultsToA};

struct IntrinsicInterface {
  static constexpr int maxArguments{7};
  const char *name{nullptr};
  IntrinsicDummyArgument dummy[maxArguments];
  TypePattern result;
  RankPattern rank{elemental};
};

// This table's entries must be ordered alphabetically.
static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", IntOrRealA}}, IntOrRealA},
    {"abs", {{"a", ZA}}, RA},
    {"achar", {{"i", IA}, DefaultingKind}, ChK},
    {"acos", {{"x", FloatingA}}, FloatingA},
    {"acosh", {{"x", FloatingA}}, FloatingA},
    {"adjustl", {{"string", ChA}}, ChA},
    {"adjustr", {{"string", ChA}}, ChA},
    {"aimag", {{"x", ZA}}, RA},
    {"aint", {{"a", RA}, MatchingDefaultKind}, RK},
    {"all", {{"mask", LA, array}, {"dim", IB, scalar, optional}}, LA,
        dimReduced},
    {"anint", {{"a", RA}, MatchingDefaultKind}, RK},
    {"any", {{"mask", LA, array}, {"dim", IB, scalar, optional}}, LA,
        dimReduced},
    {"asin", {{"x", FloatingA}}, FloatingA},
    {"asinh", {{"x", FloatingA}}, FloatingA},
    {"atan", {{"x", FloatingA}}, FloatingA},
    {"atan", {{"y", RA}, {"x", RA}}, RA},
    {"atan2", {{"y", RA}, {"x", RA}}, RA},
    {"atanh", {{"x", FloatingA}}, FloatingA},
    {"bessel_j0", {{"x", RA}}, RA},
    {"bessel_j1", {{"x", RA}}, RA},
    {"bessel_jn", {{"n", IA}, {"x", RB}}, RB},
    {"bessel_y0", {{"x", RA}}, RA},
    {"bessel_y1", {{"x", RA}}, RA},
    {"bessel_yn", {{"n", IA}, {"x", RB}}, RB},
    {"bge", {{"i", IA, elemental, BOZOK}, {"j", IB, elemental, BOZOK}}, Ld},
    {"bgt", {{"i", IA, elemental, BOZOK}, {"j", IB, elemental, BOZOK}}, Ld},
    {"ble", {{"i", IA, elemental, BOZOK}, {"j", IB, elemental, BOZOK}}, Ld},
    {"blt", {{"i", IA, elemental, BOZOK}, {"j", IB, elemental, BOZOK}}, Ld},
    {"btest", {{"i", IA}, {"pos", IB}}, Ld},
    {"ceiling", {{"a", RA}, DefaultingKind}, IK},
    {"char", {{"i", IA}, DefaultingKind}, ChK},
    {"cmplx", {{"x", ZA}, DefaultingKind}, ZK},
    {"cmplx",
        {{"x", IntOrRealA, elemental, BOZOK},
            {"y", IntOrRealA, elemental, BOZOK}, DefaultingKind},
        ZK},
    {"conjg", {{"z", ZA}}, ZA},
    {"cos", {{"x", FloatingA}}, FloatingA},
    {"cosh", {{"x", FloatingA}}, FloatingA},
    {"count",
        {{"mask", LA, array}, {"dim", IB, scalar, optional}, DefaultingKind},
        IK, dimReduced},
    {"cshift",
        {{"array", Anything, array}, {"shift", IA, dimRemoved},
            {"dim", IB, scalar, optional}},
        Anything, array},
    {"dim", {{"x", IntOrRealA}, {"y", IntOrRealA}}, IntOrRealA},
    {"dprod", {{"x", Rd}, {"y", Rd}}, DP},
    {"dshiftl", {{"i", IA}, {"j", IA, elemental, BOZOK}, {"shift", IC}}, IA},
    {"dshiftl", {{"i", BOZ}, {"j", IA}, {"shift", IB}}, IA},
    {"dshiftr", {{"i", IA}, {"j", IA, elemental, BOZOK}, {"shift", IC}}, IA},
    {"dshiftr", {{"i", BOZ}, {"j", IA}, {"shift", IB}}, IA},
    {"eoshift",
        {{"array", IntrinsicA, array}, {"shift", IB, dimRemoved},
            {"boundary", IntrinsicA, dimRemoved, optional},
            {"dim", IC, scalar, optional}},
        IntrinsicA, array},
    {"eoshift",
        {{"array", DT, array}, {"shift", IA, dimRemoved},
            {"boundary", DT, dimRemoved}, {"dim", IB, scalar, optional}},
        DT, array},
    {"erf", {{"x", RA}}, RA},
    {"erfc", {{"x", RA}}, RA},
    {"erfc_scaled", {{"x", RA}}, RA},
    {"exp", {{"x", FloatingA}}, FloatingA},
    {"exponent", {{"x", RA}}, Id},
    {"findloc",
        {{"array", NumericA, array}, {"value", NumericB, scalar},
            {"dim", IC, scalar, optional}, {"mask", LD, conformable, optional},
            DefaultingKind, {"back", LE, scalar, optional}},
        IK, dimReduced},
    {"findloc",
        {{"array", ChA, array}, {"value", ChA, scalar},
            {"dim", IB, scalar, optional}, {"mask", LC, conformable, optional},
            DefaultingKind, {"back", LD, scalar, optional}},
        IK, dimReduced},
    {"findloc",
        {{"array", LA, array}, {"value", LB, scalar},
            {"dim", IC, scalar, optional}, {"mask", LD, conformable, optional},
            DefaultingKind, {"back", LE, scalar, optional}},
        IK, dimReduced},
    {"floor", {{"a", RA}, DefaultingKind}, IK},
    {"fraction", {{"x", RA}}, RA},
    {"gamma", {{"x", RA}}, RA},
    {"hypot", {{"x", RA}, {"y", RA}}, RA},
    {"iachar", {{"c", ChA}, DefaultingKind}, IK},
    {"iall",
        {{"array", IA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        IA, dimReduced},
    {"iany",
        {{"array", IA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        IA, dimReduced},
    {"iparity",
        {{"array", IA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        IA, dimReduced},
    {"iand", {{"i", IA}, {"j", IA, elemental, BOZOK}}, IA},
    {"iand", {{"i", BOZ}, {"j", IA}}, IA},
    {"ibclr", {{"i", IA}, {"pos", IB}}, IA},
    {"ibits", {{"i", IA}, {"pos", IB}, {"len", IC}}, IA},
    {"ibset", {{"i", IA}, {"pos", IB}}, IA},
    {"ichar", {{"c", ChA}, DefaultingKind}, IK},
    {"ieor", {{"i", IA}, {"j", IA, elemental, BOZOK}}, IA},
    {"ieor", {{"i", BOZ}, {"j", IA}}, IA},
    {"image_status", {{"image", IA}, {"team", TEAM_TYPE, scalar, optional}},
        Id},
    {"index",
        {{"string", ChA}, {"substring", ChA}, {"back", LB, scalar, optional},
            DefaultingKind},
        IK},
    {"int", {{"a", NumericA, elemental, BOZOK}, DefaultingKind}, IK},
    {"ior", {{"i", IA}, {"j", IA, elemental, BOZOK}}, IA},
    {"ior", {{"i", BOZ}, {"j", IA}}, IA},
    {"ishft", {{"i", IA}, {"shift", IB}}, IA},
    {"ishftc", {{"i", IA}, {"shift", IB}, {"size", IC, elemental, optional}},
        IA},
    {"is_iostat_end", {{"i", IA}}, Ld},
    {"is_iostat_eor", {{"i", IA}}, Ld},
    {"leadz", {{"i", IA}}, Id},
    {"len", {{"string", ChA}, DefaultingKind}, IK},
    {"len_trim", {{"string", ChA}, DefaultingKind}, IK},
    {"lge", {{"string_a", ChA}, {"string_b", ChA}}, Ld},
    {"lgt", {{"string_a", ChA}, {"string_b", ChA}}, Ld},
    {"lle", {{"string_a", ChA}, {"string_b", ChA}}, Ld},
    {"llt", {{"string_a", ChA}, {"string_b", ChA}}, Ld},
    {"log", {{"x", FloatingA}}, FloatingA},
    {"log10", {{"x", RA}}, RA},
    {"logical", {{"l", LA}, DefaultingKind}, LK},
    {"log_gamma", {{"x", RA}}, RA},
    {"maskl", {{"i", IA}, DefaultingKind}, IK},
    {"maskr", {{"i", IA}, DefaultingKind}, IK},
    {"maxloc",
        {{"array", RelatableA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}, DefaultingKind,
            {"back", LD, scalar, optional}},
        IK, dimReduced},
    {"maxval",
        {{"array", RelatableA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        RelatableA, dimReduced},
    {"merge_bits",
        {{"i", IA}, {"j", IA, elemental, BOZOK},
            {"mask", IA, elemental, BOZOK}},
        IA},
    {"merge_bits", {{"i", BOZ}, {"j", IA}, {"mask", IA, elemental, BOZOK}}, IA},
    {"minloc",
        {{"array", RelatableA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}, DefaultingKind,
            {"back", LD, scalar, optional}},
        IK, dimReduced},
    {"minval",
        {{"array", RelatableA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        RelatableA, dimReduced},
    {"mod", {{"a", IntOrRealA}, {"p", IntOrRealA}}, IntOrRealA},
    {"modulo", {{"a", IntOrRealA}, {"p", IntOrRealA}}, IntOrRealA},
    {"nearest", {{"x", RA}, {"s", RB}}, RA},
    {"nint", {{"a", RA}, DefaultingKind}, IK},
    {"norm2", {{"x", RA, array}, {"dim", IB, scalar, optional}}, RA,
        dimReduced},
    {"not", {{"i", IA}}, IA},
    {"out_of_range", {{"x", IntOrRealA}, {"mold", IntOrRealB, scalar}}, Ld},
    {"out_of_range",
        {{"x", RA}, {"mold", IB, scalar}, {"round", LC, scalar, optional}}, Ld},
    {"out_of_range", {{"x", RA}, {"mold", RB}}, Ld},
    {"pack",
        {{"array", Anything, array}, {"mask", LA, conformable},
            {"vector", Anything, vector, optional}},
        Anything, vector},
    {"parity", {{"mask", LA, array}, {"dim", IB, scalar, optional}}, LA,
        dimReduced},
    {"popcnt", {{"i", IA}}, Id},
    {"poppar", {{"i", IA}}, Id},
    {"product",
        {{"array", NumericA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        NumericA, dimReduced},
    {"real", {{"a", NumericA, elemental, BOZOK}, DefaultingKind}, RK},
    {"reshape",
        {{"source", Anything, array}, {"shape", IB, vector},
            {"pad", Anything, array, optional},
            {"order", IC, vector, optional}},
        Anything, shaped},
    {"rrspacing", {{"x", RA}}, RA},
    {"scale", {{"x", RA}, {"i", IB}}, RA},
    {"scan",
        {{"string", ChA}, {"set", ChA}, {"back", LB, elemental, optional},
            DefaultingKind},
        IK},
    {"set_exponent", {{"x", RA}, {"i", IB}}, RA},
    {"shifta", {{"i", IA}, {"shift", IB}}, IA},
    {"shiftl", {{"i", IA}, {"shift", IB}}, IA},
    {"shiftr", {{"i", IA}, {"shift", IB}}, IA},
    {"sign", {{"a", IntOrRealA}, {"b", IntOrRealA}}, IntOrRealA},
    {"sin", {{"x", FloatingA}}, FloatingA},
    {"sinh", {{"x", FloatingA}}, FloatingA},
    {"spacing", {{"x", RA}}, RA},
    {"spread",
        {{"source", Anything, known}, {"dim", IB, scalar},
            {"ncopies", IC, scalar}},
        Anything, rankPlus1},
    {"sqrt", {{"x", FloatingA}}, FloatingA},
    {"sum",
        {{"array", NumericA, array}, {"dim", IB, scalar, optional},
            {"mask", LC, conformable, optional}},
        NumericA, dimReduced},
    {"tan", {{"x", FloatingA}}, FloatingA},
    {"tanh", {{"x", FloatingA}}, FloatingA},
    {"trailz", {{"i", IA}}, Id},
    // pmk continue here with TRANSFER
    {"verify",
        {{"string", ChA}, {"set", ChA}, {"back", LB, elemental, optional},
            DefaultingKind},
        IK},
};

// Not covered by the table above:
// MAX, MIN, MERGE

struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
};

static const SpecificIntrinsicInterface specificIntrinsicFunction[]{
    {{"abs", {{"a", Rd}}, Rd}},
    {{"acos", {{"x", Rd}}, Rd}},
    {{"aimag", {{"z", Zd}}, Rd}},
    {{"aint", {{"a", Rd}}, Rd}},
    {{"alog", {{"x", Rd}}, Rd, elemental}, "log"},
    {{"alog10", {{"x", Rd}}, Rd, elemental}, "log10"},
    {{"amod", {{"a", Rd}, {"p", Rd}}, Rd, elemental}, "mod"},
    {{"anint", {{"a", Rd}}, Rd}},
    {{"asin", {{"x", Rd}}, Rd}},
    {{"atan", {{"x", Rd}}, Rd}},
    {{"atan2", {{"y", Rd}, {"x", Rd}}, Rd}},
    {{"cabs", {{"a", Zd}}, Rd}, "abs"},
    {{"ccos", {{"a", Zd}}, Zd, elemental}, "cos"},
    {{"cexp", {{"a", Zd}}, Zd, elemental}, "exp"},
    {{"clog", {{"a", Zd}}, Zd, elemental}, "log"},
    {{"conjg", {{"a", Zd}}, Zd}},
    {{"cos", {{"x", Rd}}, Rd}},
    {{"csin", {{"a", Zd}}, Zd, elemental}, "sin"},
    {{"csqrt", {{"a", Zd}}, Zd, elemental}, "sqrt"},
    {{"ctan", {{"a", Zd}}, Zd, elemental}, "tan"},
    {{"dabs", {{"a", DP}}, DP, elemental}, "abs"},
    {{"dacos", {{"x", DP}}, DP, elemental}, "acos"},
    {{"dasin", {{"x", DP}}, DP, elemental}, "asin"},
    {{"datan", {{"x", DP}}, DP, elemental}, "atan"},
    {{"datan2", {{"y", DP}, {"x", DP}}, DP, elemental}, "atan2"},
    {{"dble", {{"a", Rd}, DefaultingKind}, DP, elemental}, "real"},
    {{"dcos", {{"x", DP}}, DP, elemental}, "cos"},
    {{"dcosh", {{"x", DP}}, DP, elemental}, "cosh"},
    {{"ddim", {{"x", DP}, {"y", DP}}, DP, elemental}, "dim"},
    {{"dexp", {{"x", DP}}, DP, elemental}, "exp"},
    {{"dim", {{"x", Rd}, {"y", Rd}}, Rd}},
    {{"dint", {{"a", DP}}, DP, elemental}, "aint"},
    {{"dlog", {{"x", DP}}, DP, elemental}, "log"},
    {{"dlog10", {{"x", DP}}, DP, elemental}, "log10"},
    {{"dmod", {{"a", DP}, {"p", DP}}, DP, elemental}, "mod"},
    {{"dnint", {{"a", DP}}, DP, elemental}, "anint"},
    {{"dprod", {{"x", Rd}, {"y", Rd}}, DP}},
    {{"dsign", {{"a", DP}, {"b", DP}}, DP, elemental}, "sign"},
    {{"dsin", {{"x", DP}}, DP, elemental}, "sin"},
    {{"dsinh", {{"x", DP}}, DP, elemental}, "sinh"},
    {{"dsqrt", {{"x", DP}}, DP, elemental}, "sqrt"},
    {{"dtan", {{"x", DP}}, DP, elemental}, "tan"},
    {{"dtanh", {{"x", DP}}, DP, elemental}, "tanh"},
    {{"exp", {{"x", Rd}}, Rd}},
    {{"float", {{"i", Id}}, Rd, elemental}, "real"},
    {{"iabs", {{"a", Id}}, Id, elemental}, "abs"},
    {{"idim", {{"x", Id}, {"y", Id}}, Id, elemental}, "dim"},
    {{"idint", {{"a", DP}}, Id, elemental}, "int"},
    {{"idnint", {{"a", DP}}, Id, elemental}, "nint"},
    {{"ifix", {{"a", Rd}}, Id, elemental}, "int"},
    {{"index", {{"string", Chd}, {"substring", Chd}}, Id}},
    {{"isign", {{"a", Id}, {"b", Id}}, Id, elemental}, "sign"},
    {{"len", {{"string", Chd}}, Id}},
    {{"log", {{"x", Rd}}, Rd}},
    {{"log10", {{"x", Rd}}, Rd}},
    {{"mod", {{"a", Id}, {"p", Id}}, Id}},
    {{"nint", {{"a", Rd}}, Id}},
    {{"sign", {{"a", Rd}, {"b", Rd}}, Rd}},
    {{"sin", {{"x", Rd}}, Rd}},
    {{"sinh", {{"x", Rd}}, Rd}},
    {{"sngl", {{"a", DP}}, Rd, elemental}, "real"},
    {{"sqrt", {{"x", Rd}}, Rd}},
    {{"tan", {{"x", Rd}}, Rd}},
    {{"tanh", {{"x", Rd}}, Rd}},
};

// Some entries in the table above are "restricted" specifics:
//   DBLE, FLOAT, IDINT, IFIX, SNGL
// Additional "restricted" specifics not covered by the table above:
//   AMAX0, AMAX1, AMIN0, AMIN1, DMAX1, DMIN1, MAX0, MAX1, MIN0, MIN1

struct IntrinsicTable::Implementation {
  explicit Implementation(const semantics::IntrinsicTypeDefaultKinds &dfts)
    : defaults{dfts} {}

  semantics::IntrinsicTypeDefaultKinds defaults;
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

const SpecificIntrinsic *IntrinsicTable::Probe(
    const CallCharacteristics &call) {
  return nullptr;
}
}  // namespace Fortran::evaluate
