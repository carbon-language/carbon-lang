// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A Dictionary with a nominal `Element` type, that can conform to things.
public struct KnownDictionary<Key: Hashable, Value> {
  /// The underlying Swift dictionary type.
  public typealias Base = [Key : Value]

  /// A view of a dictionary's keys.
  public typealias Keys = Base.Keys

  /// A view of a dictionary's values.
  public typealias Values = Base.Values

  /// The position of a key-value pair in a dictionary.
  public typealias Index = Base.Index

  /// The underlying Swift Dictionary
  public var base: Base

  /// Creates an instance equivalent to `base`.
  public init(_ base: Base) {
    self.base = base
  }

  /// The element type.
  public typealias Element = Base.Element

  /// Creates an empty dictionary.
  public init() { base = .init() }

  /// Creates an empty dictionary with preallocated space for at least the
  /// specified number of elements.
  public init(minimumCapacity: Int) {
    base = .init(minimumCapacity: minimumCapacity)
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence.
  public init<S>(
    uniqueKeysWithValues keysAndValues: S)
    where S : Sequence, S.Element == Element
  {
    base = .init(uniqueKeysWithValues: keysAndValues.lazy.map { $0 })
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence,
  /// using a combining closure to determine the value for any duplicate keys.
  public init<S>(
    _ keysAndValues: S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S : Sequence, S.Element == Element {
    try base = .init(keysAndValues.lazy.map { $0 }, uniquingKeysWith: combine)
  }

  /// Creates a new dictionary whose keys are the groupings returned by the
  /// given closure and whose values are arrays of the elements that returned
  /// each key.
  public init<S>(
    grouping values: S, by keyForValue: (S.Element) throws -> Key)
    rethrows where Value == [S.Element], S : Sequence
  {
    try base = .init(grouping: values, by: keyForValue)
  }

  /// Returns a new dictionary containing the key-value pairs of the dictionary
  /// that satisfy the given predicate.
  @available(swift 4.0)
  public func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> KnownDictionary {
    try .init(base.filter(isIncluded))
  }

  /// Accesses the value associated with the given key, producing `nil` when the
  /// value of a key not in the dictionary is read, and erasing the key if `nil`
  /// is written.
  public subscript(key: Key) -> Value {
    get {
      base[key]!
    }
    set {
      sanityCheck(base[key] != nil)
      base[key] = newValue
    }
  }

  public mutating func insert(_ newElement: Element) {
    sanityCheck(base[newElement.key] == nil)
    base[newElement.key] = newElement.value
  }

  /// Accesses the value for `key`, or `defaultValue` no such key exists in the
  /// dictionary, on write first inserting `key` with value `defaultValue` if it
  /// does not exist in the dictionary.
  public subscript(
    key: Key, default defaultValue: @autoclosure () -> Value
  ) -> Value {
    get {
      base[key, default: defaultValue()]
    }
    _modify {
      yield &base[key, default: defaultValue()]
    }
  }

  /// Returns a new dictionary containing the keys of this dictionary with the
  /// values transformed by the given closure.
  public func mapValues<T>(
    _ transform: (Value) throws -> T
  ) rethrows -> KnownDictionary<Key, T> {
    try .init(base.mapValues(transform))
  }

  /// Returns a new dictionary containing only the key-value pairs that have
  /// non-`nil` values as the result of transformation by the given closure.
  public func compactMapValues<T>(
    _ transform: (Value) throws -> T?
  ) rethrows -> [Key : T] { try base.compactMapValues(transform) }

  /// Updates the value stored in the dictionary for the given key and returns
  /// the old value, or adds a new key-value pair if the key does not exist and
  /// returns nil .
  public mutating func updateValue(
    _ value: Value, forKey key: Key
  ) -> Value? {
    base.updateValue(value, forKey: key)
  }

  /// Merges the key-value pairs in the given sequence into the dictionary,
  /// using a combining closure to determine the value for any duplicate keys.
  public mutating func merge<S>(
    _ other: S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S : Sequence, S.Element == Element
  {
    try base.merge(other.lazy.map { $0 }, uniquingKeysWith: combine)
  }

  /// Merges the given dictionary into this dictionary, using a combining
  /// closure to determine the value for any duplicate keys.
  public mutating func merge(
    _ other: Self,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    try base.merge(other.base, uniquingKeysWith: combine)
  }

  /// Creates a dictionary by merging key-value pairs in a sequence into the
  /// dictionary, using a combining closure to determine the value for
  /// duplicate keys.
  public func merging<S>(
    _ other: S, uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self where S : Sequence, S.Element == Element {
    try .init(base.merging(other.lazy.map { $0 }, uniquingKeysWith: combine))
  }

  /// Creates a dictionary by merging the given dictionary into this
  /// dictionary, using a combining closure to determine the value for
  /// duplicate keys.
  public func merging(
    _ other: Self, uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self {
    try .init(base.merging(other.base, uniquingKeysWith: combine))
  }

  /// Removes and returns the key-value pair at the specified index.
  public mutating func remove(at index: Index) -> Element {
    base.remove(at: index)
  }

  /// Removes the given key and its associated value from the dictionary.
  public mutating func removeValue(forKey key: Key) -> Value? {
    base.removeValue(forKey: key)
  }

  /// Removes all key-value pairs from the dictionary.
  public mutating func removeAll(
    keepingCapacity keepCapacity: Bool = false
  ) {
    base.removeAll(keepingCapacity: keepCapacity)
  }

  /// A collection containing just the keys of the dictionary.
  @available(swift 4.0)
  public var keys: Keys { base.keys }

  /// A collection containing just the values of the dictionary.
  @available(swift 4.0)
  public var values: Values {
    get {
      base.values
    }
    _modify {
      yield &base.values
    }
  }


  /// An iterator over the members of a `KnownDictionary<Key, Value>`.
  public typealias Iterator = Base.Iterator

  /// Removes and returns the first key-value pair of the dictionary if the
  /// dictionary isn't empty.
  public mutating func popFirst() -> Element? {
    base.popFirst()
  }

  /// The total number of key-value pairs that the dictionary can contain without
  /// allocating new storage.
  public var capacity: Int { base.capacity }

  /// Reserves enough space to store the specified number of key-value pairs.
  public mutating func reserveCapacity(_ minimumCapacity: Int) {
    base.reserveCapacity(minimumCapacity)
  }
}

extension KnownDictionary : Collection {
  /// The position of the first element in a nonempty dictionary.
  public var startIndex: Index { base.startIndex }

  /// The dictionary's "past the end" position---that is, the position one
  /// greater than the last valid subscript argument.
  public var endIndex: Index { base.endIndex }

  /// Returns the position immediately after the given index.
  public func index(after i: Index) -> Index {
    base.index(after: i)
  }

  /// Replaces the given index with its successor.
  public func formIndex(after i: inout Index) {
    base.formIndex(after: &i)
  }

  /// Returns the index for the given key.
  public func index(forKey key: Key) -> Index? {
    base.index(forKey: key)
  }

  /// Accesses the key-value pair at the specified position.
  public subscript(position: Index) -> Element {
    base[position]
  }

  /// The number of key-value pairs in the dictionary.
  public var count: Int { base.count }

  /// A Boolean value that indicates whether the dictionary is empty.
  public var isEmpty: Bool { base.isEmpty }
}

extension KnownDictionary : Sequence {
  /// Returns an iterator over the dictionary's key-value pairs.
  public func makeIterator() -> Iterator {
    return base.makeIterator()
  }
}

extension KnownDictionary : CustomReflectable {
  /// A mirror that reflects the dictionary.
  public var customMirror: Mirror { base.customMirror }
}

extension KnownDictionary
  : CustomStringConvertible, CustomDebugStringConvertible
{
  /// A string that represents the contents of the dictionary.
  public var description: String { base.description }

  /// A string that represents the contents of the dictionary, suitable for
  /// debugging.
  public var debugDescription: String { base.debugDescription }
}

extension KnownDictionary : Hashable where Value : Hashable {}
extension KnownDictionary : Equatable where Value : Equatable {}

extension KnownDictionary
  : Decodable where Key : Decodable, Value : Decodable
{
  /// Creates a new dictionary by decoding from the given decoder.
  public init(from decoder: Decoder) throws {
    try base = .init(from: decoder)
  }
}

extension KnownDictionary : Encodable
  where Key : Encodable, Value : Encodable
{
  /// Encodes the contents of this dictionary into the given encoder.
  public func encode(to encoder: Encoder) throws {
    try base.encode(to: encoder)
  }
}
