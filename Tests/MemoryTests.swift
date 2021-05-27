// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class MemoryTests: XCTestCase {
  /// Useful values
  var o: ASTSite { .init(devaluing: SourceRegion.empty) }
  var foo: Identifier { .init(text: "foo", site: o) }
  var bar: Identifier { .init(text: "bar", site: o) }

  func testStoreInt() {
    var m = Memory()
    let a = m.allocate()
    m.initialize(a, to: 3)
    if let x = checkNonNil(m[a] as? Int) {
      XCTAssertEqual(x, 3)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreBool() {
    var m = Memory()
    let a = m.allocate()
    m.initialize(a, to: false)
    if let x = checkNonNil(m[a] as? Bool) {
      XCTAssertEqual(x, false)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreFunctionValue() {
    var m = Memory()
    let a = m.allocate()

    let v = FunctionValue(
      dynamic_type: .function(parameterTypes: Tuple(), returnType: .void),
      code: FunctionDefinition(
        name: Identifier(text: "main", site: o),
        parameters: TupleSyntax([], o),
        returnType: .expression(TypeExpression(.intType(o))),
        body: .block([], o),
        site: o))

    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? FunctionValue) {
      XCTAssertEqual(x, v)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreTupleValue() {
    var m = Memory()
    let a = m.allocate()

    let v = TupleValue([.label(foo): 3, .label(bar): false, .position(0): 7])

    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? TupleValue) {
      XCTAssertEqual(x[foo] as? Int, 3)
      XCTAssertEqual(x[bar] as? Bool, false)
      XCTAssertEqual(x[0] as? Int, 7)
      let s = m.substructure(at: a)
      XCTAssertEqual(s.count, 3)
      guard let fooPartAddress = checkNonNil(s[foo]) else { return }
      guard let barPartAddress = checkNonNil(s[bar]) else { return }
      guard let zeroPartAddress = checkNonNil(s[0]) else { return }
      XCTAssertEqual(m[fooPartAddress] as? Int, 3)
      XCTAssertEqual(m[barPartAddress] as? Bool, false)
      XCTAssertEqual(m[zeroPartAddress] as? Int, 7)
    }
  }

  func testStoreIntType() {
    var m = Memory()
    let a = m.allocate()

    let v = Type.int
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreBoolType() {
    var m = Memory()
    let a = m.allocate()

    let v = Type.bool
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreTypeType() {
    var m = Memory()
    let a = m.allocate()

    let v = Type.type
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      XCTAssertEqual(m.substructure(at: a), Tuple())
    }
  }

  func testStoreTupleType() {
    var m = Memory()
    let a = m.allocate()

    let v = Type.tuple(Tuple([.position(0): .int, .label(foo): .bool]))
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      let s = m.substructure(at: a)
      XCTAssertEqual(s.count, 2)
      guard let fooPartAddress = checkNonNil(s[foo]) else { return }
      guard let zeroPartAddress = checkNonNil(s[0]) else { return }
      XCTAssertEqual(m[fooPartAddress] as? Type, .bool)
      XCTAssertEqual(m.substructure(at: fooPartAddress).count, 0)
      XCTAssertEqual(m[zeroPartAddress] as? Type, .int)
      XCTAssertEqual(m.substructure(at: zeroPartAddress).count, 0)
    }
  }

  func testStoreFunctionType() {
    var m = Memory()
    let a = m.allocate()

    let p: TupleType = .init([.position(0): .int])
    let r = Type.bool
    let v = Type.function(parameterTypes: p, returnType: r)

    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)

      let s0 = m.substructure(at: a)
      XCTAssertEqual(s0.count, 2)
      guard let parameterPartAddress = checkNonNil(s0[0]) else { return }

      guard let parameterPart
        = checkNonNil(m[parameterPartAddress] as? TupleValue) else { return }

      XCTAssertEqual(parameterPart.count, 1)
      XCTAssertEqual(parameterPart[0] as? Type, .int)

      guard let returnPartAddress = checkNonNil(s0[1]) else { return }
      XCTAssertEqual(m[returnPartAddress] as? Type, r)
      XCTAssertEqual(m.substructure(at: returnPartAddress).count, 0)

      let s1 = m.substructure(at: parameterPartAddress)
      XCTAssertEqual(s1.count, 1)
      guard let p0Part = checkNonNil(s1[0]) else { return }
      XCTAssertEqual(m[p0Part] as? Type, .int)
      XCTAssertEqual(m.substructure(at: p0Part).count, 0)
    }
  }
}
