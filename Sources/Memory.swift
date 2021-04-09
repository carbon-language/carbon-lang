// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typealias Address = Int

protocol Value {
  var type: Type { get }
}

/// Stops the program with an error exit code and the given message.
///
/// - Note: only works in contexts where the return type can be deduced,
///   e.g. dict[k] ?? fatal("key not found").
func fatal<R>(_ msg: String) -> R {
  fatalError(msg)
}

struct Memory {
  /// Returns the next uninitialized address bound to `t`.
  ///
  /// - Parameter `mutable`: `true` iff mutations of the Value at this address
  ///   will be allowed.
  ///
  /// - Note: if the last address allocated was `x`, the next one will be `x +
  ///   1`.
  mutating func allocate(
    boundTo t: Type, from site: SourceRegion, mutable: Bool = false
  ) -> Address {
    defer { nextAddress += 1 }
    storage[nextAddress] = Location(boundType: t, site: site, mutable: mutable)
    return nextAddress
  }

  /// Initializes the value at `a`.
  ///
  /// - Note: initialization is not considered a mutation
  /// - Requires: `a` is an allocated address bound to `newValue.type`.
  mutating func initialize(_ a: Address, to content: Value) {
    let i = storage.index(forKey: a)
      ?? fatal("initializing unallocated address \(a).")
    precondition(
      storage[i].value.boundType == content.type,
      "initializing location \(a) bound to \(storage[i].value.boundType) with "
        + "value of type \(content.type)."
    )
    storage.values[i].content = content
  }

  /// Deinitializes the value at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation
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
    let i = storage.index(forKey: a)
      ?? fatal("deallocating unallocated address \(a).")
    precondition(
      storage[i].value.content == nil, "deallocating initialized address \(a)")
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
    ///
    /// - Invariant: `stored == nil || stored!.type == boundType`.
    var content: Value? = nil

    /// The type that this location is bound to.
    let boundType: Type

    /// Where the storage was declared (if a variable), computed (if a
    /// temporary), or dynamically allocated.
    let site: SourceRegion

    /// True iff the value at this location can be mutated.
    let mutable: Bool
  }

  private var storage: [Address: Location] = [:]
  private var nextAddress = 0
}
