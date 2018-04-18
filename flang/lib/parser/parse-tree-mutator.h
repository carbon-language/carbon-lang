#ifndef FORTRAN_PARSER_PARSE_TREE_MUTATOR_H_
#define FORTRAN_PARSER_PARSE_TREE_MUTATOR_H_

#include "parse-tree.h"
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

/// Parse tree mutator
/// Call Walk(x, mutator) to visit x and, by default, each node under x,
/// optionally rewriting it in place.
///
/// mutator.Pre(x) is called before visiting x and its children are not
/// visited if it returns false.
///
/// mutator.Post(x) is called after visiting x.

namespace Fortran {
namespace parser {

// Default case for visitation of non-class data members and strings
template<typename A, typename M>
typename std::enable_if<!std::is_class_v<A> ||
    std::is_same_v<std::string, A>>::type
Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}

template<typename M> void Walk(format::ControlEditDesc &, M &);
template<typename M> void Walk(format::DerivedTypeDataEditDesc &, M &);
template<typename M> void Walk(format::FormatItem &, M &);
template<typename M> void Walk(format::FormatSpecification &, M &);
template<typename M> void Walk(format::IntrinsicTypeDataEditDesc &, M &);

// Traversal of needed STL template classes (optional, list, tuple, variant)
template<typename T, typename M>
void Walk(std::optional<T> &x, M &mutator) {
  if (x) {
    Walk(*x, mutator);
  }
}
template<typename T, typename M> void Walk(std::list<T> &x, M &mutator) {
  for (auto &elem : x) {
    Walk(elem, mutator);
  }
}
template<std::size_t I = 0, typename Func, typename T>
void ForEachInTuple(T &tuple, Func func) {
  if constexpr (I < std::tuple_size_v<T>) {
    func(std::get<I>(tuple));
    ForEachInTuple<I + 1>(tuple, func);
  }
}
template<typename M, typename... A>
void Walk(std::tuple<A...> &x, M &mutator) {
  if (mutator.Pre(x)) {
    ForEachInTuple(x, [&](auto &y) { Walk(y, mutator); });
    mutator.Post(x);
  }
}
template<typename M, typename... A>
void Walk(std::variant<A...> &x, M &mutator) {
  if (mutator.Pre(x)) {
    std::visit([&](auto &y) { Walk(y, mutator); }, x);
    mutator.Post(x);
  }
}
template<typename A, typename B, typename M>
void Walk(std::pair<A, B> &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.first, mutator);
    Walk(x.second, mutator);
  }
}

// Trait-determined traversal of empty, tuple, union, and wrapper classes.
template<typename A, typename M>
typename std::enable_if<EmptyTrait<A>>::type Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}

template<typename A, typename M>
typename std::enable_if<TupleTrait<A>>::type Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.t, mutator);
    mutator.Post(x);
  }
}

template<typename A, typename M>
typename std::enable_if<UnionTrait<A>>::type Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}

template<typename A, typename M>
typename std::enable_if<WrapperTrait<A>>::type Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.v, mutator);
    mutator.Post(x);
  }
}

template<typename T, typename M>
void Walk(Indirection<T> &x, M &mutator) {
  Walk(*x, mutator);
}

// Walk a class with a single field 'thing'.
template<typename T, typename M> void Walk(Scalar<T> &x, M &mutator) {
  Walk(x.thing, mutator);
}
template<typename T, typename M> void Walk(Constant<T> &x, M &mutator) {
  Walk(x.thing, mutator);
}
template<typename T, typename M> void Walk(Integer<T> &x, M &mutator) {
  Walk(x.thing, mutator);
}
template<typename T, typename M> void Walk(Logical<T> &x, M &mutator) {
  Walk(x.thing, mutator);
}
template<typename T, typename M>
void Walk(DefaultChar<T> &x, M &mutator) {
  Walk(x.thing, mutator);
}

template<typename T, typename M> void Walk(Statement<T> &x, M &mutator) {
  if (mutator.Pre(x)) {
    // N.B. the label is not traversed
    Walk(x.statement, mutator);
    mutator.Post(x);
  }
}

template<typename M> void Walk(Name &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}

template<typename M> void Walk(AcSpec &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.type, mutator);
    Walk(x.values, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(ArrayElement &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.subscripts, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(CharSelector::LengthAndKind &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.length, mutator);
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(CaseValueRange::Range &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.lower, mutator);
    Walk(x.upper, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(CoindexedNamedObject &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.imageSelector, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(DeclarationTypeSpec::Class &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.derived, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(DeclarationTypeSpec::Type &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.derived, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(ImportStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.names, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(IntrinsicTypeSpec::Character &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.selector, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(IntrinsicTypeSpec::Complex &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(IntrinsicTypeSpec::Logical &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(IntrinsicTypeSpec::Real &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename T, typename M> void Walk(LoopBounds<T> &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.name, mutator);
    Walk(x.lower, mutator);
    Walk(x.upper, mutator);
    Walk(x.step, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(PartRef &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.name, mutator);
    Walk(x.subscripts, mutator);
    Walk(x.imageSelector, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(ReadStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.iounit, mutator);
    Walk(x.format, mutator);
    Walk(x.controls, mutator);
    Walk(x.items, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(RealLiteralConstant &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.real, mutator);
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(RealLiteralConstant::Real &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}
template<typename M> void Walk(StructureComponent &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.component, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(Suffix &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.binding, mutator);
    Walk(x.resultName, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(TypeBoundProcedureStmt::WithInterface &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.interfaceName, mutator);
    Walk(x.attributes, mutator);
    Walk(x.bindingNames, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(TypeBoundProcedureStmt::WithoutInterface &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.attributes, mutator);
    Walk(x.declarations, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(UseStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.nature, mutator);
    Walk(x.moduleName, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(WriteStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.iounit, mutator);
    Walk(x.format, mutator);
    Walk(x.controls, mutator);
    Walk(x.items, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(format::ControlEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(format::DerivedTypeDataEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.type, mutator);
    Walk(x.parameters, mutator);
    mutator.Post(x);
  }
}
template<typename M> void Walk(format::FormatItem &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.repeatCount, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(format::FormatSpecification &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.items, mutator);
    Walk(x.unlimitedItems, mutator);
    mutator.Post(x);
  }
}
template<typename M>
void Walk(format::IntrinsicTypeDataEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    Walk(x.width, mutator);
    Walk(x.digits, mutator);
    Walk(x.exponentWidth, mutator);
    mutator.Post(x);
  }
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSE_TREE_MUTATOR_H_
