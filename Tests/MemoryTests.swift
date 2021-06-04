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
    let a = m.allocate(boundTo: .int)
    m.initialize(a, to: 3)
    XCTAssertEqual(m[a] as? Int, 3)
  }

  func testStoreBool() {
    var m = Memory()
    let a = m.allocate(boundTo: .bool)
    m.initialize(a, to: false)
    XCTAssertEqual(m[a] as? Bool, false)
  }

  func testStoreFunctionValue() {
    var m = Memory()
    let t = Type.function(
      .init(
        parameterTypes: Tuple([.position(0): .int, .position(1): .bool]),
        returnType: .void))
    let a = m.allocate(boundTo: t)

    let v = FunctionValue(
      dynamic_type: t,
      code: FunctionDefinition(
        name: Identifier(text: "main", site: o),
        parameters: TupleSyntax([], o),
        returnType: .expression(TypeExpression(.intType(o))),
        body: .block([], o),
        site: o))

    m.initialize(a, to: v)

    if let x = checkNonNil(m[a] as? FunctionValue) {
      XCTAssertEqual(x, v)
    }
  }

  func testStoreTupleValue() {
    var m = Memory()
    let t = TupleType(
      [.label(foo): .int, .label(bar): .bool, .position(0): .int])
    let a = m.allocate(boundTo: .tuple(t))
    let v = TupleValue([.label(foo): 3, .label(bar): false, .position(0): 7])

    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? TupleValue) {
      XCTAssertEqual(x[foo] as? Int, 3)
      XCTAssertEqual(x[bar] as? Bool, false)
      XCTAssertEqual(x[0] as? Int, 7)

      XCTAssertEqual(m[a.^foo] as? Int, 3)
      XCTAssertEqual(m[a.^bar] as? Bool, false)
      XCTAssertEqual(m[a.^0] as? Int, 7)
    }
  }

  func testStoreIntType() {
    var m = Memory()
    let a = m.allocate(boundTo: .type)

    let v = Type.int
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
    }
  }

  func testStoreBoolType() {
    var m = Memory()
    let a = m.allocate(boundTo: .type)

    let v = Type.bool
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
    }
  }

  func testStoreTypeType() {
    var m = Memory()
    let a = m.allocate(boundTo: .type)

    let v = Type.type
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
    }
  }

  func testStoreTupleType() {
    var m = Memory()
    let a = m.allocate(boundTo: .type)

    let v = Type.tuple(Tuple([.position(0): .int, .label(foo): .bool, .label(bar): .type]))
    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      XCTAssertNotNil(x.tuple)

      XCTAssertEqual(m[a.^foo] as? Type, .bool)
      XCTAssertEqual(m[a.^bar] as? Type, .type)
      XCTAssertEqual(m[a.^0] as? Type, .int)
    }
  }

  func testStoreFunctionType() {
    var m = Memory()
    let a = m.allocate(boundTo: .type)

    let p: TupleType = .init([.position(0): .int])
    let r = Type.bool
    let v = Type.function(.init(parameterTypes: p, returnType: r))

    m.initialize(a, to: v)
    if let x = checkNonNil(m[a] as? Type) {
      XCTAssertEqual(x, v)
      let a1 = a.addresseePart(
        \Type.function!.parameterTypes.upcastToValue, ".<parameterTypes>")

      let p1 = m[a1]
      XCTAssertEqual(Type(p1), .tuple(p))

      let a2 = a.addresseePart(
        \Type.function!.returnType.upcastToValue, ".<returnType>")

      XCTAssertEqual(m[a2] as! Type, r)
    }
  }
}
