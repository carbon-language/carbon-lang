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
  /// Returns an uninitialized address.
  ///
  /// - Parameter mutable: `true` iff mutations of the Value at this address
  ///   will be allowed.
  mutating func allocate(mutable: Bool = false) -> Address {
    defer { nextOffset += 1 }
    storage[Address(nextOffset)] = Location(mutable: mutable)
    return Address(nextOffset)
  }

  /// Initializes the value at `a` to `v`.
  ///
  /// - Note: initialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is an allocated address.
  mutating func initialize(_ a: Address, to v: Value) {
    let i = storage.index(forKey: a)
      ?? fatal("initializing unallocated address \(a).")

    if let x = storage.values[i].content {
      fatalError("address \(a) already initialized to \(x).")
    }

    storage.values[i].content = v
    let isMutable = storage.values[i].mutable
    storage[a]!.substructure = v.parts.mapFields {
      let l = allocate(mutable: isMutable)
      initialize(l, to: $0)
      return l
    }
  }

  /// Deinitializes the storage at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is the address of an initialized value.
  mutating func deinitialize(_ a: Address) {
    let i = storage.index(forKey: a)
      ?? fatal("deinitializing unallocated address \(a).")
    // The following precondition dies on an empty tuple. See fun4.6c.
    // I added the if statement. -Jeremy
    //precondition(storage[i].value.content != nil)
    if storage[i].value.content != nil {
      for a1 in storage.values[i].substructure.fields { deinitialize(a1) }
      storage.values[i].content = nil
    }
  }

  /// Deallocates the storage at `a`.
  ///
  /// - Requires: `a` is an uninitialized address.
  mutating func deallocate(_ a: Address) {
    let v = storage[a] ?? fatal("deallocating unallocated address \(a).")
    precondition(v.content == nil, "deallocating initialized address \(a)")
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
    let r = storage[i].value.content
      ?? fatal("reading from uninitialized address \(a).")
    return r
  }

  mutating func assign(from source: Address, into target: Address) {
    precondition(self[source].type == self[target].type)

    // Only check the top level type because choices of a single type can have
    // different payload types.
    func uncheckedAssign(from source: Address, into target: Address) {
      sanityCheck(storage[target]!.mutable)
      let sourceMap = substructure(at: source).elements
      let targetMap = substructure(at: target).elements
      if !sourceMap.isEmpty && sourceMap.count == targetMap.count
           && sourceMap.keys.allSatisfy({ targetMap[$0] != nil })
      {
        for (field, s) in sourceMap {
          assign(from: s, into: targetMap[field]!)
        }
        storage[target]!.content = storage[source]!.content
      }
      else {
        deinitialize(target)
        initialize(target, to: self[source])
      }
    }

    uncheckedAssign(from: source, into: target)
  }

  /// Returns the value at `a` or nil if `a` is not an initialized address.
  func value(at a: Address) -> Value? {
    guard let i = storage.index(forKey: a) else { return nil }
    return storage[i].value.content
  }

  /// Returns the substructure of the value stored at `a`
  func substructure(at a: Address) -> Tuple<Address> {
    storage[a]!.substructure
  }

  /// An allocated element of memory.
  private struct Location {
    /// The value stored in this location, if initialized.
    var content: Value?

    /// The addresses of subparts of this value.
    var substructure = Tuple<Address>()

    /// True iff the value at this location can be mutated.
    let mutable: Bool
  }

  private var storage: [Address: Location] = [:]
  private(set) var nextOffset = 0
}

// TODO: initialize memory with a Value type and a Tuple<Address>.
