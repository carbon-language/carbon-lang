// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_TRAVERSE_H_
#define FORTRAN_EVALUATE_TRAVERSE_H_

// TODO pmk documentation

#include "expression.h"
#include "../semantics/symbol.h"
#include "../semantics/type.h"

namespace Fortran::evaluate {
template<typename Visitor, typename Result = bool> class Traverse {
public:
  explicit Traverse(Visitor &v) : visitor_{v} {}

  // Packaging
  template<typename A, bool C>
  Result operator()(const common::Indirection<A, C> &x) const {
    return visitor_(x.value());
  }
  template<typename A> Result operator()(const std::unique_ptr<A> &x) const {
    return visitor_(x.get());
  }
  template<typename A> Result operator()(const std::shared_ptr<A> &x) const {
    return visitor_(x.get());
  }
  template<typename A> Result operator()(const A *x) const {
    if (x != nullptr) {
      return visitor_(*x);
    } else {
      return visitor_.defaultResult;
    }
  }
  template<typename A> Result operator()(const std::optional<A> &x) const {
    if (x.has_value()) {
      return visitor_(*x);
    } else {
      return visitor_.defaultResult;
    }
  }
  template<typename... A> Result operator()(const std::variant<A...> &u) const {
    return std::visit(visitor_, u);
  }
  template<typename A> Result operator()(const std::vector<A> &x) const {
    return CombineContents(x);
  }

  // Leaves
  Result operator()(const BOZLiteralConstant &) const {
    return visitor_.defaultResult;
  }
  Result operator()(const NullPointer &) const {
    return visitor_.defaultResult;
  }
  template<typename T> Result operator()(const Constant<T> &) const {
    return visitor_.defaultResult;
  }
  Result operator()(const semantics::Symbol &) const {
    return visitor_.defaultResult;
  }
  Result operator()(const StaticDataObject &) const {
    return visitor_.defaultResult;
  }
  Result operator()(const ImpliedDoIndex &) const {
    return visitor_.defaultResult;
  }

  // Variables
  Result operator()(const BaseObject &x) const { return visitor_(x.u); }
  Result operator()(const Component &x) const {
    return Combine(x.base(), x.GetLastSymbol());
  }
  Result operator()(const NamedEntity &x) const {
    if (const Component * component{x.UnwrapComponent()}) {
      return visitor_(*component);
    } else {
      return visitor_(x.GetFirstSymbol());
    }
  }
  template<int KIND> Result operator()(const TypeParamInquiry<KIND> &x) const {
    return visitor_(x.base());
  }
  Result operator()(const Triplet &x) const {
    return Combine(x.lower(), x.upper(), x.stride());
  }
  Result operator()(const Subscript &x) const { return visitor_(x.u); }
  Result operator()(const ArrayRef &x) const {
    return Combine(x.base(), x.subscript());
  }
  Result operator()(const CoarrayRef &x) const {
    return Combine(
        x.base(), x.subscript(), x.cosubscript(), x.stat(), x.team());
  }
  Result operator()(const DataRef &x) const { return visitor_(x.u); }
  Result operator()(const Substring &x) const {
    return Combine(x.parent(), x.lower(), x.upper());
  }
  Result operator()(const ComplexPart &x) const {
    return visitor_(x.complex());
  }
  template<typename T> Result operator()(const Designator<T> &x) const {
    return visitor_(x.u);
  }
  template<typename T> Result operator()(const Variable<T> &x) const {
    return visitor_(x.u);
  }
  Result operator()(const DescriptorInquiry &x) const {
    return visitor_(x.base());
  }

  // Calls
  Result operator()(const ProcedureDesignator &x) const {
    if (const Component * component{x.GetComponent()}) {
      return visitor_(*component);
    } else if (const semantics::Symbol * symbol{x.GetSymbol()}) {
      return visitor_(*symbol);
    } else {
      return visitor_(DEREF(x.GetSpecificIntrinsic()));
    }
  }
  Result operator()(const ActualArgument &x) const {
    if (const auto *symbol{x.GetAssumedTypeDummy()}) {
      return visitor_(*symbol);
    } else {
      return visitor_(x.UnwrapExpr());
    }
  }
  Result operator()(const ProcedureRef &x) const {
    return Combine(x.proc(), x.arguments());
  }
  template<typename T> Result operator()(const FunctionRef<T> &x) const {
    return visitor_(static_cast<const ProcedureRef &>(x));
  }

  // Other primaries
  template<typename T>
  Result operator()(const ArrayConstructorValue<T> &x) const {
    return visitor_(x.u);
  }
  template<typename T>
  Result operator()(const ArrayConstructorValues<T> &x) const {
    return CombineContents(x);
  }
  template<typename T> Result operator()(const ImpliedDo<T> &x) const {
    return Combine(x.lower(), x.upper(), x.stride(), x.values());
  }
  Result operator()(const semantics::ParamValue &x) const {
    return visitor_(x.GetExplicit());
  }
  Result operator()(
      const semantics::DerivedTypeSpec::ParameterMapType::value_type &x) const {
    return visitor_(x.second);
  }
  Result operator()(const semantics::DerivedTypeSpec &x) const {
    return CombineContents(x.parameters());
  }
  Result operator()(const StructureConstructorValues::value_type &x) const {
    return visitor_(x.second);
  }
  Result operator()(const StructureConstructor &x) const {
    return visitor_.Combine(visitor_(x.derivedTypeSpec()), CombineContents(x));
  }

  // Operations and wrappers
  template<typename D, typename R, typename O>
  Result operator()(const Operation<D, R, O> &op) const {
    return visitor_(op.left());
  }
  template<typename D, typename R, typename LO, typename RO>
  Result operator()(const Operation<D, R, LO, RO> &op) const {
    return Combine(op.left(), op.right());
  }
  Result operator()(const Relational<SomeType> &x) const {
    return visitor_(x.u);
  }
  template<typename T> Result operator()(const Expr<T> &x) const {
    return visitor_(x.u);
  }

private:
  template<typename ITER> Result CombineRange(ITER iter, ITER end) const {
    if (iter == end) {
      return visitor_.defaultResult;
    } else {
      Result result{visitor_(*iter++)};
      for (; iter != end; ++iter) {
        result = visitor_.Combine(std::move(result), visitor_(*iter));
      }
      return result;
    }
  }

  template<typename A> Result CombineContents(const A &x) const {
    return CombineRange(x.begin(), x.end());
  }

  template<typename A, typename... Bs>
  Result Combine(const A &x, const Bs &... ys) const {
    if constexpr (sizeof...(Bs) == 0) {
      return visitor_(x);
    } else {
      return visitor_.Combine(visitor_(x), Combine(ys...));
    }
  }

  Visitor &visitor_;
};

template<typename Visitor> class PredicateTraverse {
public:
  explicit PredicateTraverse(Visitor &v) : traverse_{v} {}
  static constexpr bool defaultResult{true};
  static constexpr bool Combine(bool x, bool y) { return x && y; }

  template<typename A> bool operator()(const A &x) { return traverse_(x); }

private:
  Traverse<Visitor> traverse_;
};
}
#endif
