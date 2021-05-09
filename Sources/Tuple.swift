// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Identifier for a tuple element or parameter to a particular function type.
///
/// Fields of carbon tuples may be named or positional.
enum FieldID: Hashable {
  /// field identified by label (syntactically, ".<label> = ...")
  case label(Identifier)
  /// field identified by its offset in the sequence of positional fields.
  case position(Int)
}

/// A tuple value (or type, which is also a tuple value).
struct Tuple<T> {
  /// The number of fields in `self`.
  var count: Int { fields.count }

  /// Returns a Tuple containing the keys of `self` with the corresponding
  /// values mapped through `transform`.
  func mapValues<U>(_ transform: (T) throws -> U) rethrows -> Tuple<U> {
    return .init(try fields.mapValues(transform))
  }

  /// Returns a Tuple containing the keys of `self` with the corresponding
  /// values mapped through `transform`, except for any `nil` results.
  func compactMapValues<U>(_ transform: (T) throws -> U?) rethrows -> Tuple<U> {
    return .init(try fields.compactMapValues(transform))
  }

  /// Creates an instance using the given underlying storage.
  fileprivate init(_ fields: [FieldID: T]) { self.fields = fields }

  /// Returns the field with the given name, or `nil` if no such name exists.
  subscript(fieldName: Identifier) -> T? { fields[.label(fieldName)] }

  /// Returns the given positional field.
  subscript(position: Int) -> T? { fields[.position(position)] }

  private let fields: [FieldID: T]
}
extension Tuple: Equatable where T: Equatable {}

typealias TupleType = Tuple<Type>
typealias TupleValue = Tuple<Value>

extension TupleValue: CarbonInterpreter.Value {
  var type: Type {
    .tuple(self.mapValues { $0.type })
  }
}

extension TupleType {
  static let void: Self = .init([:])
}

extension TupleSyntax {
  /// Returns a form of this AST node that is agnostic to the ordering of
  /// labeled fields, adding an error to errors if there are any duplicate
  /// fields.
  func fields(
    reportingDuplicatesIn errors: inout ErrorLog
  ) -> Tuple<Payload> {
    var r: [FieldID: Payload] = [:]
    var positionalCount = 0
    for e in elements {
      let key: FieldID
        = e.label.map { .label($0) } ?? .position(positionalCount)
      if case .position = key { positionalCount += 1 }
      if let other = r[key] {
        errors.append(
          CompileError(
            "Duplicate label \(e.label!)", at: e.label!.site,
            notes: [("other definition", other.site)]))
      }
      else {
        r[key] = e.payload
      }
    }
    return Tuple(r)
  }
}
