# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testAffineExprCapsule
def testAffineExprCapsule():
  with Context() as ctx:
    affine_expr = AffineExpr.get_constant(42)

  affine_expr_capsule = affine_expr._CAPIPtr
  # CHECK: capsule object
  # CHECK: mlir.ir.AffineExpr._CAPIPtr
  print(affine_expr_capsule)

  affine_expr_2 = AffineExpr._CAPICreate(affine_expr_capsule)
  assert affine_expr == affine_expr_2
  assert affine_expr_2.context == ctx

run(testAffineExprCapsule)


# CHECK-LABEL: TEST: testAffineExprEq
def testAffineExprEq():
  with Context():
    a1 = AffineExpr.get_constant(42)
    a2 = AffineExpr.get_constant(42)
    a3 = AffineExpr.get_constant(43)
    # CHECK: True
    print(a1 == a1)
    # CHECK: True
    print(a1 == a2)
    # CHECK: False
    print(a1 == a3)
    # CHECK: False
    print(a1 == None)
    # CHECK: False
    print(a1 == "foo")

run(testAffineExprEq)


# CHECK-LABEL: TEST: testAffineExprContext
def testAffineExprContext():
  with Context():
    a1 = AffineExpr.get_constant(42)
  with Context():
    a2 = AffineExpr.get_constant(42)

  # CHECK: False
  print(a1 == a2)

run(testAffineExprContext)


# CHECK-LABEL: TEST: testAffineExprConstant
def testAffineExprConstant():
  with Context():
    a1 = AffineExpr.get_constant(42)
    # CHECK: 42
    print(a1.value)
    # CHECK: 42
    print(a1)

    a2 = AffineConstantExpr.get(42)
    # CHECK: 42
    print(a2.value)
    # CHECK: 42
    print(a2)

    assert a1 == a2

run(testAffineExprConstant)


# CHECK-LABEL: TEST: testAffineExprDim
def testAffineExprDim():
  with Context():
    d1 = AffineExpr.get_dim(1)
    d11 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)

    # CHECK: 1
    print(d1.position)
    # CHECK: d1
    print(d1)

    # CHECK: 2
    print(d2.position)
    # CHECK: d2
    print(d2)

    assert d1 == d11
    assert d1 != d2

run(testAffineExprDim)


# CHECK-LABEL: TEST: testAffineExprSymbol
def testAffineExprSymbol():
  with Context():
    s1 = AffineExpr.get_symbol(1)
    s11 = AffineSymbolExpr.get(1)
    s2 = AffineSymbolExpr.get(2)

    # CHECK: 1
    print(s1.position)
    # CHECK: s1
    print(s1)

    # CHECK: 2
    print(s2.position)
    # CHEKC: s2
    print(s2)

    assert s1 == s11
    assert s1 != s2

run(testAffineExprSymbol)


# CHECK-LABEL: TEST: testAffineAddExpr
def testAffineAddExpr():
  with Context():
    d1 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)
    d12 = AffineExpr.get_add(d1, d2)
    # CHECK: d1 + d2
    print(d12)

    d12op = d1 + d2
    # CHECK: d1 + d2
    print(d12op)

    assert d12 == d12op
    assert d12.lhs == d1
    assert d12.rhs == d2

run(testAffineAddExpr)


# CHECK-LABEL: TEST: testAffineMulExpr
def testAffineMulExpr():
  with Context():
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    expr = AffineExpr.get_mul(d1, c2)
    # CHECK: d1 * 2
    print(expr)

    # CHECK: d1 * 2
    op = d1 * c2
    print(op)

    assert expr == op
    assert expr.lhs == d1
    assert expr.rhs == c2

run(testAffineMulExpr)


# CHECK-LABEL: TEST: testAffineModExpr
def testAffineModExpr():
  with Context():
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    expr = AffineExpr.get_mod(d1, c2)
    # CHECK: d1 mod 2
    print(expr)

    # CHECK: d1 mod 2
    op = d1 % c2
    print(op)

    assert expr == op
    assert expr.lhs == d1
    assert expr.rhs == c2

run(testAffineModExpr)


# CHECK-LABEL: TEST: testAffineFloorDivExpr
def testAffineFloorDivExpr():
  with Context():
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    expr = AffineExpr.get_floor_div(d1, c2)
    # CHECK: d1 floordiv 2
    print(expr)

    assert expr.lhs == d1
    assert expr.rhs == c2

run(testAffineFloorDivExpr)


# CHECK-LABEL: TEST: testAffineCeilDivExpr
def testAffineCeilDivExpr():
  with Context():
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    expr = AffineExpr.get_ceil_div(d1, c2)
    # CHECK: d1 ceildiv 2
    print(expr)

    assert expr.lhs == d1
    assert expr.rhs == c2

run(testAffineCeilDivExpr)


# CHECK-LABEL: TEST: testAffineExprSub
def testAffineExprSub():
  with Context():
    d1 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)
    expr = d1 - d2
    # CHECK: d1 - d2
    print(expr)

    assert expr.lhs == d1
    rhs = AffineMulExpr(expr.rhs)
    # CHECK: d2
    print(rhs.lhs)
    # CHECK: -1
    print(rhs.rhs)

run(testAffineExprSub)


def testClassHierarchy():
  with Context():
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    add = AffineAddExpr.get(d1, c2)
    mul = AffineMulExpr.get(d1, c2)
    mod = AffineModExpr.get(d1, c2)
    floor_div = AffineFloorDivExpr.get(d1, c2)
    ceil_div = AffineCeilDivExpr.get(d1, c2)

    # CHECK: False
    print(isinstance(d1, AffineBinaryExpr))
    # CHECK: False
    print(isinstance(c2, AffineBinaryExpr))
    # CHECK: True
    print(isinstance(add, AffineBinaryExpr))
    # CHECK: True
    print(isinstance(mul, AffineBinaryExpr))
    # CHECK: True
    print(isinstance(mod, AffineBinaryExpr))
    # CHECK: True
    print(isinstance(floor_div, AffineBinaryExpr))
    # CHECK: True
    print(isinstance(ceil_div, AffineBinaryExpr))

    try:
      AffineBinaryExpr(d1)
    except ValueError as e:
      # CHECK: Cannot cast affine expression to AffineBinaryExpr
      print(e)

    try:
      AffineBinaryExpr(c2)
    except ValueError as e:
      # CHECK: Cannot cast affine expression to AffineBinaryExpr
      print(e)

run(testClassHierarchy)
