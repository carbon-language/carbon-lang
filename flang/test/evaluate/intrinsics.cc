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

#include "../../lib/evaluate/intrinsics.h"
#include "testing.h"
#include "../../lib/evaluate/common.h"
#include "../../lib/evaluate/expression.h"
#include "../../lib/evaluate/tools.h"
#include "../../lib/parser/provenance.h"
#include <initializer_list>
#include <iostream>
#include <map>
#include <string>

namespace Fortran::evaluate {

class CookedStrings {
public:
  CookedStrings() {}
  explicit CookedStrings(const std::initializer_list<std::string> &ss) {
    for (const auto &s : ss) {
      Save(s);
    }
    Marshal();
  }
  void Save(const std::string &s) {
    offsets_[s] = cooked_.Put(s);
    cooked_.PutProvenance(cooked_.allSources().AddCompilerInsertion(s));
  }
  void Marshal() { cooked_.Marshal(); }
  parser::CharBlock operator()(const std::string &s) {
    return {cooked_.data().data() + offsets_[s], s.size()};
  }
  parser::ContextualMessages Messages(parser::Messages &buffer) {
    return parser::ContextualMessages{cooked_.data(), &buffer};
  }
  void Emit(std::ostream &o, const parser::Messages &messages) {
    messages.Emit(o, cooked_);
  }

private:
  parser::AllSources allSources_;
  parser::CookedSource cooked_{allSources_};
  std::map<std::string, std::size_t> offsets_;
};

template<typename A> auto Const(A &&x) -> Constant<TypeOf<A>> {
  return Constant<TypeOf<A>>{std::move(x)};
}

template<typename A> struct NamedArg {
  std::string keyword;
  A value;
};

template<typename A> static NamedArg<A> Named(std::string kw, A &&x) {
  return {kw, std::move(x)};
}

struct TestCall {
  TestCall(const IntrinsicProcTable &t, std::string n) : table{t}, name{n} {}
  template<typename A> TestCall &Push(A &&x) {
    args.emplace_back(AsGenericExpr(std::move(x)));
    keywords.push_back("");
    return *this;
  }
  template<typename A> TestCall &Push(NamedArg<A> &&x) {
    args.emplace_back(AsGenericExpr(std::move(x.value)));
    keywords.push_back(x.keyword);
    strings.Save(x.keyword);
    return *this;
  }
  template<typename A, typename... As> TestCall &Push(A &&x, As &&... xs) {
    Push(std::move(x));
    return Push(std::move(xs)...);
  }
  void Marshal() {
    strings.Save(name);
    strings.Marshal();
    std::size_t j{0};
    for (auto &kw : keywords) {
      if (!kw.empty()) {
        args[j]->keyword = strings(kw);
      }
      ++j;
    }
  }
  void DoCall(std::optional<DynamicType> resultType = std::nullopt,
      int rank = 0, bool isElemental = false) {
    Marshal();
    parser::CharBlock fName{strings(name)};
    std::cout << "function: " << fName.ToString();
    char sep{'('};
    for (const auto &a : args) {
      std::cout << sep;
      sep = ',';
      a->AsFortran(std::cout);
    }
    if (sep == '(') {
      std::cout << '(';
    }
    std::cout << ')' << std::endl;
    CallCharacteristics call{fName};
    auto messages{strings.Messages(buffer)};
    common::IntrinsicTypeDefaultKinds defaults;
    FoldingContext context{messages, defaults};
    std::optional<SpecificCall> si{table.Probe(call, args, context)};
    if (resultType.has_value()) {
      TEST(si.has_value());
      TEST(buffer.empty());
      if (si) {
        const auto &proc{si->specificIntrinsic.characteristics.value()};
        const auto &fr{proc.functionResult};
        TEST(fr.has_value());
        if (fr) {
          const auto *ts{fr->GetTypeAndShape()};
          TEST(ts != nullptr);
          if (ts) {
            TEST(*resultType == ts->type());
            MATCH(rank, ts->Rank());
          }
        }
        MATCH(isElemental,
            proc.attrs.test(characteristics::Procedure::Attr::Elemental));
      }
    } else {
      TEST(!si.has_value());
      TEST(!buffer.empty() || name == "bad");
    }
    strings.Emit(std::cout, buffer);
  }

  const IntrinsicProcTable &table;
  CookedStrings strings;
  parser::Messages buffer;
  ActualArguments args;
  std::string name;
  std::vector<std::string> keywords;
};

void TestIntrinsics() {
  common::IntrinsicTypeDefaultKinds defaults;
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Integer));
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Real));
  IntrinsicProcTable table{IntrinsicProcTable::Configure(defaults)};
  table.Dump(std::cout);

  using Int1 = Type<TypeCategory::Integer, 1>;
  using Int4 = Type<TypeCategory::Integer, 4>;
  using Int8 = Type<TypeCategory::Integer, 8>;
  using Real4 = Type<TypeCategory::Real, 4>;
  using Real8 = Type<TypeCategory::Real, 8>;
  using Complex4 = Type<TypeCategory::Complex, 4>;
  using Complex8 = Type<TypeCategory::Complex, 8>;
  using Char = Type<TypeCategory::Character, 1>;
  using Log4 = Type<TypeCategory::Logical, 4>;

  TestCall{table, "bad"}
      .Push(Const(Scalar<Int4>{}))
      .DoCall();  // bad intrinsic name
  TestCall{table, "abs"}
      .Push(Named("a", Const(Scalar<Int4>{})))
      .DoCall(Int4::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Int4>{})).DoCall(Int4::GetType());
  TestCall{table, "abs"}
      .Push(Named("bad", Const(Scalar<Int4>{})))
      .DoCall();  // bad keyword
  TestCall{table, "abs"}.DoCall();  // insufficient args
  TestCall{table, "abs"}
      .Push(Const(Scalar<Int4>{}))
      .Push(Const(Scalar<Int4>{}))
      .DoCall();  // too many args
  TestCall{table, "abs"}
      .Push(Const(Scalar<Int4>{}))
      .Push(Named("a", Const(Scalar<Int4>{})))
      .DoCall();
  TestCall{table, "abs"}
      .Push(Named("a", Const(Scalar<Int4>{})))
      .Push(Const(Scalar<Int4>{}))
      .DoCall();
  TestCall{table, "abs"}.Push(Const(Scalar<Int1>{})).DoCall(Int1::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Int4>{})).DoCall(Int4::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Int8>{})).DoCall(Int8::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Real4>{})).DoCall(Real4::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Real8>{})).DoCall(Real8::GetType());
  TestCall{table, "abs"}
      .Push(Const(Scalar<Complex4>{}))
      .DoCall(Real4::GetType());
  TestCall{table, "abs"}
      .Push(Const(Scalar<Complex8>{}))
      .DoCall(Real8::GetType());
  TestCall{table, "abs"}.Push(Const(Scalar<Char>{})).DoCall();
  TestCall{table, "abs"}.Push(Const(Scalar<Log4>{})).DoCall();

  TestCall maxCallR{table, "max"}, maxCallI{table, "min"},
      max0Call{table, "max0"}, max1Call{table, "max1"},
      amin0Call{table, "amin0"}, amin1Call{table, "amin1"},
      max0WrongCall{table, "max0"}, amin1WrongCall{table, "amin1"};
  for (int j{0}; j < 10; ++j) {
    maxCallR.Push(Const(Scalar<Real4>{}));
    maxCallI.Push(Const(Scalar<Int4>{}));
    max0Call.Push(Const(Scalar<Int4>{}));
    max0WrongCall.Push(Const(Scalar<Real4>{}));
    max1Call.Push(Const(Scalar<Real4>{}));
    amin0Call.Push(Const(Scalar<Int4>{}));
    amin1WrongCall.Push(Const(Scalar<Int4>{}));
    amin1Call.Push(Const(Scalar<Real4>{}));
  }
  maxCallR.DoCall(Real4::GetType());
  maxCallI.DoCall(Int4::GetType());
  max0Call.DoCall(Int4::GetType());
  max0WrongCall.DoCall();
  max1Call.DoCall(Int4::GetType());
  amin0Call.DoCall(Real4::GetType());
  amin1Call.DoCall(Real4::GetType());
  amin1WrongCall.DoCall();

  // TODO: test other intrinsics
}
}

int main() {
  Fortran::evaluate::TestIntrinsics();
  return testing::Complete();
}
