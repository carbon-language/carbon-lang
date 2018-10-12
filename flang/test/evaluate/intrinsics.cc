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

#include "../../lib/evaluate/intrinsics.h"
#include "testing.h"
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
  parser::CookedSource cooked_;
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
        args[j].keyword = strings(kw);
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
      a.Dump(std::cout);
    }
    if (sep == '(') {
      std::cout << '(';
    }
    std::cout << ")\n";
    CallCharacteristics call{fName, args};
    auto messages{strings.Messages(buffer)};
    std::optional<SpecificIntrinsic> si{table.Probe(call, &messages)};
    if (resultType.has_value()) {
      TEST(si.has_value());
      TEST(buffer.empty());
      TEST(*resultType == si->type);
      MATCH(rank, si->rank);
      MATCH(isElemental, si->isElemental);
    } else {
      TEST(!si.has_value());
      TEST(!buffer.empty() || name == "bad");
    }
    strings.Emit(std::cout, buffer);
  }

  const IntrinsicProcTable &table;
  CookedStrings strings;
  parser::Messages buffer;
  Arguments args;
  std::string name;
  std::vector<std::string> keywords;
};

template<typename A> void Push(Arguments &args, A &&x) {
  args.emplace_back(AsGenericExpr(std::move(x)));
}
template<typename A, typename... As>
void Push(Arguments &args, A &&x, As &&... xs) {
  args.emplace_back(AsGenericExpr(std::move(x)));
  Push(args, std::move(xs)...);
}
template<typename... As> Arguments Args(As &&... xs) {
  Arguments args;
  Push(args, std::move(xs)...);
  return args;
}

void TestIntrinsics() {
  IntrinsicTypeDefaultKinds defaults;
  MATCH(4, defaults.defaultIntegerKind);
  MATCH(4, defaults.defaultRealKind);
  IntrinsicProcTable table{IntrinsicProcTable::Configure(defaults)};
  table.Dump(std::cout);

  using Int4 = Type<TypeCategory::Integer, 4>;

  TestCall{table, "bad"}
      .Push(Const(Scalar<Int4>{1}))
      .DoCall();  // bad intrinsic name
  TestCall{table, "abs"}
      .Push(Named("a", Const(Scalar<Int4>{1})))
      .DoCall(Int4::dynamicType);
  TestCall{table, "abs"}.Push(Const(Scalar<Int4>{1})).DoCall(Int4::dynamicType);
  TestCall{table, "abs"}
      .Push(Named("bad", Const(Scalar<Int4>{1})))
      .DoCall();  // bad keyword
  TestCall{table, "abs"}.DoCall();  // insufficient args
  TestCall{table, "abs"}
      .Push(Const(Scalar<Int4>{1}))
      .Push(Const(Scalar<Int4>{2}))
      .DoCall();  // too many args
  TestCall{table, "abs"}
      .Push(Const(Scalar<Int4>{1}))
      .Push(Named("a", Const(Scalar<Int4>{2})))
      .DoCall();
  TestCall{table, "abs"}
      .Push(Named("a", Const(Scalar<Int4>{1})))
      .Push(Const(Scalar<Int4>{2}))
      .DoCall();
}
}  // namespace Fortran::evaluate

int main() {
  Fortran::evaluate::TestIntrinsics();
  return testing::Complete();
}
