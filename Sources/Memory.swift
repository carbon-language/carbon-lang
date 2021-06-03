// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Address: Hashable, CustomStringConvertible {
  let allocation: Int
  let part: WritableKeyPath<Value, Value>
  let description: String
}

extension Address {
  static func .^ (l: Address, r: FieldID) -> Address {
    Address(
      allocation: l.allocation,
      part: l.part.appending(path: \.self[r]!),
      description: l.description + {
        switch r {
        case let .position(x): return "[\(x)]"
        case let .label(x): return ".\(x.text)"
        }
      }()
    )
  }

  static func .^ (l: Address, r: Identifier) -> Address {
    l .^ .label(r)
  }

  static func .^ (l: Address, r: Int) -> Address {
    l .^ .position(r)
  }

  static func .^ <T: Value>(
    l: Address, r: WritableKeyPath<T, Value>
  ) -> Address {
    Address(
      allocation: l.allocation,
      part: l.part
        .appending(path: \.self[downcastTo: TypeID<T>()]!)
        .appending(path: r), description: l.description)
  }
}

/// Stops the program with an error exit code and the given message.
///
/// - Note: use this in contexts where the return type can be deduced,
///   e.g. dict[k] ?? fatal("key not found").
func fatal<R>(
  file: StaticString = #filePath, line: UInt = #line, _ msg: String
) -> R {
  // Add a newline so the message can also contain a source region (in carbon
  // code) that's recognized by IDEs.
  fatalError("\n" + msg, file: (file), line: line)
}

/// Stops the program with an error exit code and the given message.
func fatal(
  file: StaticString = #filePath, line: UInt = #line, _ msg: String
) -> Never {
  // Add a newline so the message can also contain a source region (in carbon
  // code) that's recognized by IDEs.
  fatalError("\n" + msg, file: (file), line: line)
}

struct Memory {
  /// An allocated element of memory.
  typealias Storage = [Int: Value]

  private var storage: Storage = .init()
  private var mutableAllocations: Set<Int> = []
  private(set) var nextAllocation = 0
}

extension Memory {
  func isInitialized(at a: Address) -> Bool {
    !(storage[a.allocation]![keyPath: a.part] is Uninitialized)
  }

  func isMutable(at a: Address) -> Bool {
    mutableAllocations.contains(a.allocation)
  }

  func boundType(at a: Address) -> Type? {
    storage[a.allocation]![keyPath: a.part].dynamic_type
  }

  /// Returns an uninitialized address earmarked for storing instances of the
  /// given type.
  ///
  /// - Parameter mutable: `true` iff mutations of the value at this address
  ///   will be allowed.
  mutating func allocate(boundTo type: Type, mutable: Bool = false) -> Address {
    defer { nextAllocation += 1 }
    storage[nextAllocation] = Uninitialized(dynamic_type: type)
    if mutable { mutableAllocations.insert(nextAllocation) }
    return Address(
      allocation: nextAllocation, part: \.self,
      description: "@\(nextAllocation)")
  }

  /// Initializes the value at `a` to `v`.
  ///
  /// - Note: initialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is an allocated address.
  mutating func initialize(_ a: Address, to v: Value) {
    sanityCheck(!isInitialized(at: a))
    sanityCheck(boundType(at: a) == v.dynamic_type,
                "\(boundType(at: a)!) != \(v.dynamic_type)")
    storage[a.allocation]![keyPath: a.part] = v
  }

  /// Deinitializes the storage at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is the address of an initialized value.
  mutating func deinitialize(_ a: Address) {
    sanityCheck(isInitialized(at: a))
    storage[a.allocation]![keyPath: a.part]
      = Uninitialized(dynamic_type: boundType(at: a)!)
  }

  /// Deallocates the storage at `a`.
  ///
  /// - Requires: `a` is an uninitialized address.
  mutating func deallocate(_ a: Address) {
    sanityCheck(
      !isInitialized(at: a), "\(self[a]) at \(a) must be deinitialized")
    sanityCheck(a.part == \.self, "Can't deallocate subpart \(a.part)")
    _ = storage.removeValue(forKey: a.allocation)
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
  subscript(a: Address) -> Value {
    storage[a.allocation]![keyPath: a.part]
  }

  /// Copies the value at `target` into `source`.
  ///
  /// - Requires: The type of the new value must match that of
  ///   the existing value at `a`.
  /// - Requires: `a` is in memory allocated with `mutable = true`.
  mutating func assign(from source: Address, into target: Address) {
    sanityCheck(
      mutableAllocations.contains(target.allocation),
      "Assigning into immutable address \(target)"
    )
    sanityCheck(isInitialized(at: source))
    sanityCheck(isInitialized(at: target))
    sanityCheck(boundType(at: source) == boundType(at: target))
    storage[target.allocation]![keyPath: target.part] = self[source]
  }
}
