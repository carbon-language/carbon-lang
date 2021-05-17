// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typealias Address = Int

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
  /// - Parameter site: the region of the code that triggered the allocation.
  /// - Parameter mutable: `true` iff mutations of the Value at this address
  ///   will be allowed.
  mutating func allocate(
    from site: SourceRegion, mutable: Bool = false
  ) -> Address {
    defer { nextAddress += 1 }
    storage[nextAddress] = Location(site: site, mutable: mutable)
    return nextAddress
  }

  /// Initializes the value at `a` to `v`.
  ///
  /// - Note: initialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is an allocated address bound to `v.type`.
  mutating func initialize(_ a: Address, to v: Value) {
    let i = storage.index(forKey: a)
      ?? fatal("initializing unallocated address \(a).")
    storage.values[i].content = v
  }

  /// Deinitializes the value at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is the address of an initialized value.
  mutating func deinitialize(_ a: Address) {
    let i = storage.index(forKey: a)
      ?? fatal("deinitializing unallocated address \(a).")
    precondition(storage[i].value.content != nil)
    storage.values[i].content = nil
  }

  /// Deallocates the storage at `a`.
  ///
  /// - Requires: `a` is an uninitialized address.
  mutating func deallocate(_ a: Address) {
    let v = storage[a] ?? fatal("deallocating unallocated address \(a).")
    precondition(v.content == nil, "deallocating initialized address \(a)")
    storage[a] = nil
  }

  /// Accesses the value at `a`.
  ///
  /// - Requires: The value at `a` is initialized.
  /// - Requires: (`set`) The type of the new value must match that of
  ///   the existing value at `a`.
  /// - Requires: (`set`) `a` was allocated with `mutable = true`.
  subscript(a: Address) -> Value {
    get {
      let i = storage.index(forKey: a)
        ?? fatal("reading from unallocated address \(a).")
      let r = storage[i].value.content
        ?? fatal("reading from uninitialized address \(a).")
      return r
    }
    set {
      let i = storage.index(forKey: a)
        ?? fatal("writing to unallocated address \(a).")
      precondition(
        storage[i].value.content != nil,
        "modifying uninitialized address \(a).")
      storage.values[i].content = newValue
    }
  }

  /// An allocated element of memory.
  private struct Location {
    /// The value stored in this location, if initialized.
    var content: Value? = nil

    /// Where the storage was declared (if a variable), computed (if a
    /// temporary), or dynamically allocated.
    let site: SourceRegion

    /// True iff the value at this location can be mutated.
    let mutable: Bool
  }

  private var storage: [Address: Location] = [:]
  private(set) var nextAddress = 0
}
