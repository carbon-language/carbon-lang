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

/// A tuple value (or type, which is also a tuple value) of the given `Field`.
///
/// The `fields` of a tuple are instances of `Field`, while the The `element`s
/// are `(FieldID, Field)` pairs.
struct Tuple<Field> : FieldAccess {
  /// The number of fields in `self`.
  var count: Int { elements.count }

  /// True iff `count == 0`
  var isEmpty: Bool { elements.isEmpty }

  /// Returns a Tuple with the field IDs of `self` and the corresponding
  /// fields mapped through `transform`.
  func mapFields<U>(_ transform: (Field) throws -> U) rethrows -> Tuple<U> {
    return .init(try elements.mapValues(transform))
  }

  /// Returns a Tuple with the field IDs of `self` and the corresponding
  /// elements mapped through `transform`.
  func mapElements<U>(
    _ transform: (Elements.Element) throws -> U
  ) rethrows -> Tuple<U> {
    let newContents = zip(elements.keys, try elements.map(transform))
    return Tuple<U>(Dictionary(uniqueKeysWithValues: newContents))
  }

  /// Returns a Tuple with the field IDs of `self` and the corresponding values
  /// mapped through `transform`, dropping any `nil` results.
  func compactMapFields<U>(
    _ transform: (Field) throws -> U?) rethrows -> Tuple<U>
  {
    return .init(try elements.compactMapValues(transform))
  }

  /// Creates an instance using the given underlying storage.
  init(_ storage: [FieldID: Field] = [:]) { self.elements = storage }

  /// Accesses the field with the given id
  subscript(k: FieldID) -> Field? {
    get { elements[k] }
    set {
      sanityCheck(newValue != nil)
      sanityCheck(elements[k] != nil)
      elements[k] = newValue
    }
  }

  typealias Elements = [FieldID: Field]

  /// An arbitrarily-ordered collection of the contained fields, sans ID.
  var fields: Elements.Values { elements.values }

  /// The arbitrarily-ordered collection of (ID, field) pairs.
  private(set) var elements: Elements
}
extension Tuple: Equatable where Field: Equatable {}

extension Tuple {
  /// Returns `true` iff `self` has the same set of fieldIDs as `other`.
  func isCongruent<OtherPayload>(to other: Tuple<OtherPayload>) -> Bool {
    count == other.count && elements.keys.allSatisfy {
      other.elements[$0] != nil
    }
  }
}
extension Tuple: CustomStringConvertible {
  var description: String {
    let labeled = elements.lazy.compactMap { (k, v) -> (String, Field)? in
      guard case let .label(l) = k else { return nil }
      return (l.text, v)
    }
      .sorted { $0.0 < $1.0 }.map { ".\($0.0) = \($0.1)" }

    let positional = elements.lazy.compactMap { (k, v) -> (Int, Field)? in
      guard case let .position(p) = k else { return nil }
      return (p, v)
    }
      .sorted { $0.0 < $1.0 }.map { "\($0.1)" }

    return "(\((labeled + positional).joined(separator: ", ")))"
  }
}

typealias TupleType = Tuple<Type>
typealias TupleValue = Tuple<Value>

extension TupleValue: Value, CompoundValue {
  var dynamic_type: Type {
    .tuple(self.mapFields { $0.dynamic_type })
  }

  var parts: Tuple<Value> { self }
  init(parts: Tuple<Value>) { self = parts }
}

extension TupleType {
  static let void: Self = .init([:])

  /// Accesses the `Value` corresponding to `self`.
  ///
  /// Writing anything other than a tuple of types into this property is a
  /// precondition violation.
  var upcastToValue: Value {
    get { self.mapFields { $0 } }
    set { self = (newValue as! TupleValue).mapFields { Type($0)! } }
  }
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
          CarbonError(
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
