// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Address: Hashable, CustomStringConvertible {
  fileprivate init(_ offset: Int) { self.offset = offset }
  var description: String { "@\(offset)" }
  private let offset: Int
}

/// Stops the program with an error exit code and the given message.
///
/// - Note: only works in contexts where the return type can be deduced,
///   e.g. dict[k] ?? fatal("key not found").
func fatal<R>(
  file: StaticString = #filePath, line: UInt = #line, _ msg: String
) -> R {
  // Add a newline so the message can also contain a source region (in carbon
  // code) that's recognized by IDEs.
  fatalError("\n" + msg, file: (file), line: line)
}

struct Memory {
  /// An allocated element of memory.
  private struct Location {
    enum Storage {
      /// The value is stored directly.
      case atom(AtomicValue)

      /// The parts of the value are stored indirectly
      case compound(type: Type, parts: Tuple<Address>)
    }

    /// A representation of the stored value, with respect to the rest of memory.
    var occupant: Storage?

    var atom: AtomicValue? {
      if case let .atom(r) = occupant { return r }
      else { return nil }
    }

    var compound: (type: Type, parts: Tuple<Address>)? {
      if case let .compound(type: t, parts: p) = occupant { return (t, p) }
      else { return nil }
    }
    let mutable: Bool
  }

  private var storage: [Address: Location] = [:]
  private(set) var nextOffset = 0
}

extension Memory {
  /// Returns an uninitialized address.
  ///
  /// - Parameter mutable: `true` iff mutations of the Value at this address
  ///   will be allowed.
  mutating func allocate(mutable: Bool = false) -> Address {
    defer { nextOffset += 1 }
    storage[Address(nextOffset)] = Location(mutable: mutable)
    return Address(nextOffset)
  }

  /// Initializes the value at `a` to a new instance of `type` with the given
  /// substructure.
  ///
  /// Use this function to adopt storage of already-computed parts into
  /// newly-initialized values.
  ///
  /// - Note: initialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is an allocated address.
  mutating func initialize(
    _ a: Address, as type: Type, adoptingParts parts: Tuple<Address>
  ) {
    storage[a]!.occupant = .compound(type: type, parts: parts)
  }

  /// Initializes the value at `a` to `v`.
  ///
  /// - Note: initialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is an allocated address.
  mutating func initialize(_ a: Address, to v: Value) {
    let i = storage.index(forKey: a)
      ?? fatal("initializing unallocated address \(a).")

    if storage.values[i].occupant != nil {
      fatalError("address \(a) already initialized to \(self[a]).")
    }

    if let atom = v as? AtomicValue {
      // Can't use v.parts.isEmpty as a key because empty tuples have no parts.
      sanityCheck(v.parts.isEmpty)
      storage.values[i].occupant = .atom(atom)
    }
    else {
      let isMutable = storage.values[i].mutable

      // This creates new dictionary entries so will invalidate the index i
      storage[a]!.occupant = Location.Storage.compound(
        type: v.dynamic_type,
        parts: v.parts.mapFields {
          let l = allocate(mutable: isMutable) // NOTE: SIDE-EFFECTS
          initialize(l, to: $0) // NOTE: SIDE-EFFECTS
          return l
        })
      
    }
  }

  /// Deinitializes the storage at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is the address of an initialized value.
  mutating func deinitialize(_ a: Address) {
    let i = storage.index(forKey: a)
      ?? fatal("deinitializing unallocated address \(a).")

    switch storage[i].value.occupant
      ?? fatal("deinitializing uninitialized address \(a).")
    {
    case .atom: break
    case let .compound(type: _, parts: structure):
      for a1 in structure.fields { deinitialize(a1) }
    }
    storage.values[i].occupant = nil
  }

  /// Deallocates the storage at `a`.
  ///
  /// - Requires: `a` is an uninitialized address.
  mutating func deallocate(_ a: Address) {
    let v = storage[a] ?? fatal("deallocating unallocated address \(a).")
    sanityCheck(v.occupant == nil, "deallocating initialized address \(a)")
    storage[a] = nil
  }

  /// Deintializes and then deallocates the memory at `a`
  ///
  /// - Requires: `a` is the address of an initialized value.
  mutating func delete(_ a: Address) {
    deinitialize(a)
    deallocate(a)
  }

  /// Accesses the value at `a`.
  ///
  /// - Requires: The value at `a` is initialized.
  /// - Requires: (`set`) The type of the new value must match that of
  ///   the existing value at `a`.
  /// - Requires: (`set`) `a` was allocated with `mutable = true`.
  subscript(a: Address) -> Value {
    let i = storage.index(forKey: a)
      ?? fatal("reading from unallocated address \(a).")

    switch storage[i].value.occupant
      ?? fatal("reading from uninitialized address \(a).")
    {
    case let .atom(x):
      return x
    case let .compound(type: type, _):
      return type.swiftType.init(from: a, in: self)
    }
  }

  mutating func assign(from source: Address, into target: Address) {
    
    // Only check the top level type because choices of a single type can have
    // different payload types.
    func uncheckedAssign(from source: Address, into target: Address) {
      sanityCheck(storage[target]!.mutable)

      let sourceOccupant = storage[source]!.occupant!
      switch (sourceOccupant, storage[target]!.occupant!) {
      case let (.atom(s), .atom(t)):
        sanityCheck(s.dynamic_type == t.dynamic_type)
        storage[target]!.occupant = sourceOccupant

      case let (.compound(type: sourceType, parts: sourceMap),
                .compound(type: targetType, parts: targetMap)):
        sanityCheck(sourceType == targetType)

        // Choices of a single type can have different payload types, so if the
        // parts are not congruent, we need to deinitialize/reinitialize.
        if sourceMap.count == targetMap.count
             && sourceMap.elements.keys.allSatisfy({ targetMap[$0] != nil })
        {
          for (field, s) in sourceMap.elements {
            assign(from: s, into: targetMap[field]!)
          }
        }
        else {
          deinitialize(target)
          initialize(target, to: self[source])
        }
      default: UNREACHABLE()
      }
    }

    sanityCheck(self[source].dynamic_type == self[target].dynamic_type)
    uncheckedAssign(from: source, into: target)
  }

  /// Returns the value at `a` or nil if `a` is not an initialized address.
  func value(at a: Address) -> Value? {
    guard storage.index(forKey: a) != nil else { return nil }
    return self[a]
  }

  /// Returns the substructure of the value stored at `a`
  func substructure(at a: Address) -> Tuple<Address> {
    switch storage[a]!.occupant! {
    case .atom: return .init()
    case .compound(type: _, let r): return r
    }
  }

  /// Returns the value at `a`.
  func atom(at a: Address) -> AtomicValue {
    return storage[a]!.atom!
  }
}

// TODO: Stop using tuples of addresses as a storage substrate (I think).
// Alternative, StructDefinition, and ChoiceDefinition can then stop modeling
// Value.
