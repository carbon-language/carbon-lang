// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The location of a `Value` in `Memory`, often a sub-part of some other
/// `Value`.
struct Address: Hashable, CustomStringConvertible {
  /// The allocation that created this address.
  let allocation: Int

  /// Which part of the Value at the top level of the allocation the address
  /// refers to.
  let subObject: WritableKeyPath<Value, Value>

  /// A string representation for diagnostic purposes.
  let description: String
}

/// Address composition.
extension Address {
  /// Returns an address that accesses the given field of the value at `base`.
  static func .^ (base: Address, field: FieldID) -> Address {
    Address(
      allocation: base.allocation,
      subObject: base.subObject.appending(path: \.self[field]!),
      description: base.description + {
        switch field {
        case let .position(x): return "[\(x)]"
        case let .label(x): return ".\(x.text)"
        }
      }()
    )
  }

  /// Returns an address that accesses the given field of the value at `base`.
  static func .^ (base: Address, field: Identifier) -> Address {
    base .^ .label(field)
  }

  /// Returns an address that accesses the given field of the value at `base`.
  static func .^ (base: Address, field: Int) -> Address {
    base .^ .position(field)
  }

  /// Assuming the value at `self` has concrete type `T`, returns the address of
  /// the given part of that value, using `partDescription` to extend the
  /// textual representation.
  ///
  /// Used to access parts that can't be addressed directly in carbon, e.g. the
  /// `returnType` part of a `FunctionType`, which has no carbon name, but whose
  /// address must be bound to variables as part of pattern matching.
  func addresseePart<T: Value>(
    _ part: WritableKeyPath<T, Value>, _ partDescription: String
  ) -> Address {
    Address(
      allocation: self.allocation,
      subObject: self.subObject
        .appending(path: \.self[downcastTo: TypeID<T>()]!)
        .appending(path: part),
      description: self.description + partDescription)
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

/// Storage for every value handled by the interpreter.
struct Memory {
  /// Everything stored in memory, accessed via its allocation value.
  private var allocations: [Int: Value] = [:]

  /// The allocated addresses that contain mutable values.
  private var mutableAllocations: Set<Int> = []

  /// The base of the next address to be allocated.
  private var nextAllocation = 0
}

extension Memory {
  func isInitialized(at a: Address) -> Bool {
    !(allocations[a.allocation]![keyPath: a.subObject] is Uninitialized)
  }

  func isMutable(at a: Address) -> Bool {
    mutableAllocations.contains(a.allocation)
  }

  func boundType(at a: Address) -> Type? {
    allocations[a.allocation]![keyPath: a.subObject].dynamic_type
  }

  /// Returns an uninitialized address earmarked for storing instances of the
  /// given type.
  ///
  /// - Parameter mutable: `true` iff mutations of the value at this address
  ///   will be allowed.
  mutating func allocate(boundTo type: Type, mutable: Bool = false) -> Address {
    defer { nextAllocation += 1 }
    allocations[nextAllocation] = Uninitialized(dynamic_type: type)
    if mutable { mutableAllocations.insert(nextAllocation) }
    return Address(
      allocation: nextAllocation, subObject: \.self,
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
    allocations[a.allocation]![keyPath: a.subObject] = v
  }

  /// Deinitializes the storage at `a`, returning it to an uninitialized state.
  ///
  /// - Note: deinitialization is not considered a mutation of `a`'s value.
  /// - Requires: `a` is the address of an initialized value.
  mutating func deinitialize(_ a: Address) {
    sanityCheck(isInitialized(at: a))
    allocations[a.allocation]![keyPath: a.subObject]
      = Uninitialized(dynamic_type: boundType(at: a)!)
  }

  /// Deallocates the storage at `a`.
  ///
  /// - Requires: `a` is an uninitialized address.
  mutating func deallocate(_ a: Address) {
    sanityCheck(
      !isInitialized(at: a), "\(self[a]) at \(a) must be deinitialized")
    sanityCheck(
      a.subObject == \.self, "Can't deallocate subObject \(a.subObject)")
    _ = allocations.removeValue(forKey: a.allocation)
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
    allocations[a.allocation]![keyPath: a.subObject]
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
    allocations[target.allocation]![keyPath: target.subObject] = self[source]
  }
}
