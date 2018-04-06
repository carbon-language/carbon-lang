#ifndef FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
#define FORTRAN_PARSER_PARSE_TREE_VISITOR_H_

#include "parse-tree.h"
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

/// Parse tree visitor
/// Call Walk(x, visitor) to visit x and, by default, each node under x.
///
/// visitor.Pre(x) is called before visiting x and its children are not
/// visited if it returns false.
///
/// visitor.Post(x) is called after visiting x.

namespace Fortran {
namespace parser {

// Default case for visitation of non-class data members and strings
template<typename A, typename V>
typename std::enable_if<!std::is_class_v<A> ||
    std::is_same_v<std::string, A>>::type
Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}

template<typename V> void Walk(const format::ControlEditDesc &, V &);
template<typename V> void Walk(const format::DerivedTypeDataEditDesc &, V &);
template<typename V> void Walk(const format::FormatItem &, V &);
template<typename V> void Walk(const format::FormatSpecification &, V &);
template<typename V> void Walk(const format::IntrinsicTypeDataEditDesc &, V &);

// Traversal of needed STL template classes (optional, list, tuple, variant)
template<typename T, typename V>
void Walk(const std::optional<T> &x, V &visitor) {
  if (x) {
    Walk(*x, visitor);
  }
}
template<typename T, typename V> void Walk(const std::list<T> &x, V &visitor) {
  for (const auto &elem : x) {
    Walk(elem, visitor);
  }
}
template<std::size_t I = 0, typename Func, typename T>
void ForEachInTuple(const T &tuple, Func func) {
  if constexpr (I < std::tuple_size_v<T>) {
    func(std::get<I>(tuple));
    ForEachInTuple<I + 1>(tuple, func);
  }
}
template<typename V, typename... A>
void Walk(const std::tuple<A...> &x, V &visitor) {
  if (visitor.Pre(x)) {
    ForEachInTuple(x, [&](const auto &y) { Walk(y, visitor); });
    visitor.Post(x);
  }
}
template<typename V, typename... A>
void Walk(const std::variant<A...> &x, V &visitor) {
  if (visitor.Pre(x)) {
    std::visit([&](const auto &y) { Walk(y, visitor); }, x);
    visitor.Post(x);
  }
}
template<typename A, typename B, typename V>
void Walk(const std::pair<A, B> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.first, visitor);
    Walk(x.second, visitor);
  }
}

// Trait-determined traversal of empty, tuple, union, and wrapper classes.
template<typename A, typename V>
typename std::enable_if<EmptyTrait<A>>::type Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}

template<typename A, typename V>
typename std::enable_if<TupleTrait<A>>::type Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.t, visitor);
    visitor.Post(x);
  }
}

template<typename A, typename V>
typename std::enable_if<UnionTrait<A>>::type Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}

template<typename A, typename V>
typename std::enable_if<WrapperTrait<A>>::type Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.v, visitor);
    visitor.Post(x);
  }
}

template<typename T, typename V>
void Walk(const Indirection<T> &x, V &visitor) {
  Walk(*x, visitor);
}

// Walk a class with a single field 'thing'.
template<typename T, typename V> void Walk(const Scalar<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Constant<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Integer<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Logical<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V>
void Walk(const DefaultChar<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}

template<typename T, typename V> void Walk(const Statement<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    // N.B. the label is not traversed
    Walk(x.statement, visitor);
    visitor.Post(x);
  }
}

template<typename V> void Walk(const Name &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}

template<typename V> void Walk(const AcSpec &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.values, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ArrayElement &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.subscripts, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const CharSelector::LengthAndKind &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.length, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const CaseValueRange::Range &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.lower, visitor);
    Walk(x.upper, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const CoindexedNamedObject &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const DeclarationTypeSpec::Class &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const DeclarationTypeSpec::Type &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ImportStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.names, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Character &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.selector, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Complex &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Logical &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const IntrinsicTypeSpec::Real &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename T, typename V> void Walk(const LoopBounds<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.lower, visitor);
    Walk(x.upper, visitor);
    Walk(x.step, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const PartRef &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.subscripts, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ReadStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const RealLiteralConstant &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.real, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const RealLiteralConstant::Real &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}
template<typename V> void Walk(const StructureComponent &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.component, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Suffix &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.binding, visitor);
    Walk(x.resultName, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const TypeBoundProcedureStmt::WithInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.interfaceName, visitor);
    Walk(x.attributes, visitor);
    Walk(x.bindingNames, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const TypeBoundProcedureStmt::WithoutInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.attributes, visitor);
    Walk(x.declarations, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const UseStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.nature, visitor);
    Walk(x.moduleName, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const WriteStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const format::ControlEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const format::DerivedTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.parameters, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const format::FormatItem &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.repeatCount, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const format::FormatSpecification &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.items, visitor);
    Walk(x.unlimitedItems, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const format::IntrinsicTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    Walk(x.width, visitor);
    Walk(x.digits, visitor);
    Walk(x.exponentWidth, visitor);
    visitor.Post(x);
  }
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
