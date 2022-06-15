# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.python_test as test

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: testAttributes
@run
def testAttributes():
  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True

    #
    # Check op construction with attributes.
    #

    i32 = IntegerType.get_signless(32)
    one = IntegerAttr.get(i32, 1)
    two = IntegerAttr.get(i32, 2)
    unit = UnitAttr.get()

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: mandatory_i32 = 1 : i32
    # CHECK-DAG: optional_i32 = 2 : i32
    # CHECK-DAG: unit
    # CHECK: }
    op = test.AttributedOp(one, optional_i32=two, unit=unit)
    print(f"{op}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK: mandatory_i32 = 2 : i32
    # CHECK: }
    op2 = test.AttributedOp(two)
    print(f"{op2}")

    #
    # Check generic "attributes" access and mutation.
    #

    assert "additional" not in op.attributes

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: additional = 1 : i32
    # CHECK-DAG: mandatory_i32 = 2 : i32
    # CHECK: }
    op2.attributes["additional"] = one
    print(f"{op2}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: additional = 2 : i32
    # CHECK-DAG: mandatory_i32 = 2 : i32
    # CHECK: }
    op2.attributes["additional"] = two
    print(f"{op2}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK-NOT: additional = 2 : i32
    # CHECK:     mandatory_i32 = 2 : i32
    # CHECK: }
    del op2.attributes["additional"]
    print(f"{op2}")

    try:
      print(op.attributes["additional"])
    except KeyError:
      pass
    else:
      assert False, "expected KeyError on unknown attribute key"

    #
    # Check accessors to defined attributes.
    #

    # CHECK: Mandatory: 1
    # CHECK: Optional: 2
    # CHECK: Unit: True
    print(f"Mandatory: {op.mandatory_i32.value}")
    print(f"Optional: {op.optional_i32.value}")
    print(f"Unit: {op.unit}")

    # CHECK: Mandatory: 2
    # CHECK: Optional: None
    # CHECK: Unit: False
    print(f"Mandatory: {op2.mandatory_i32.value}")
    print(f"Optional: {op2.optional_i32}")
    print(f"Unit: {op2.unit}")

    # CHECK: Mandatory: 2
    # CHECK: Optional: None
    # CHECK: Unit: False
    op.mandatory_i32 = two
    op.optional_i32 = None
    op.unit = False
    print(f"Mandatory: {op.mandatory_i32.value}")
    print(f"Optional: {op.optional_i32}")
    print(f"Unit: {op.unit}")
    assert "optional_i32" not in op.attributes
    assert "unit" not in op.attributes

    try:
      op.mandatory_i32 = None
    except ValueError:
      pass
    else:
      assert False, "expected ValueError on setting a mandatory attribute to None"

    # CHECK: Optional: 2
    op.optional_i32 = two
    print(f"Optional: {op.optional_i32.value}")

    # CHECK: Optional: None
    del op.optional_i32
    print(f"Optional: {op.optional_i32}")

    # CHECK: Unit: False
    op.unit = None
    print(f"Unit: {op.unit}")
    assert "unit" not in op.attributes

    # CHECK: Unit: True
    op.unit = True
    print(f"Unit: {op.unit}")

    # CHECK: Unit: False
    del op.unit
    print(f"Unit: {op.unit}")


# CHECK-LABEL: TEST: inferReturnTypes
@run
def inferReturnTypes():
  with Context() as ctx, Location.unknown(ctx):
    test.register_python_test_dialect(ctx)
    module = Module.create()
    with InsertionPoint(module.body):
      op = test.InferResultsOp()
      dummy = test.DummyOp()

    # CHECK: [Type(i32), Type(i64)]
    iface = InferTypeOpInterface(op)
    print(iface.inferReturnTypes())

    # CHECK: [Type(i32), Type(i64)]
    iface_static = InferTypeOpInterface(test.InferResultsOp)
    print(iface.inferReturnTypes())

    assert isinstance(iface.opview, test.InferResultsOp)
    assert iface.opview == iface.operation.opview

    try:
      iface_static.opview
    except TypeError:
      pass
    else:
      assert False, ("not expected to be able to obtain an opview from a static"
                     " interface")

    try:
      InferTypeOpInterface(dummy)
    except ValueError:
      pass
    else:
      assert False, "not expected dummy op to implement the interface"

    try:
      InferTypeOpInterface(test.DummyOp)
    except ValueError:
      pass
    else:
      assert False, "not expected dummy op class to implement the interface"


# CHECK-LABEL: TEST: resultTypesDefinedByTraits
@run
def resultTypesDefinedByTraits():
  with Context() as ctx, Location.unknown(ctx):
    test.register_python_test_dialect(ctx)
    module = Module.create()
    with InsertionPoint(module.body):
      inferred = test.InferResultsOp()
      same = test.SameOperandAndResultTypeOp([inferred.results[0]])
      # CHECK-COUNT-2: i32
      print(same.one.type)
      print(same.two.type)

      first_type_attr = test.FirstAttrDeriveTypeAttrOp(
          inferred.results[1], TypeAttr.get(IndexType.get()))
      # CHECK-COUNT-2: index
      print(first_type_attr.one.type)
      print(first_type_attr.two.type)

      first_attr = test.FirstAttrDeriveAttrOp(
          FloatAttr.get(F32Type.get(), 3.14))
      # CHECK-COUNT-3: f32
      print(first_attr.one.type)
      print(first_attr.two.type)
      print(first_attr.three.type)

      implied = test.InferResultsImpliedOp()
      # CHECK: i32
      print(implied.integer.type)
      # CHECK: f64
      print(implied.flt.type)
      # CHECK: index
      print(implied.index.type)


# CHECK-LABEL: TEST: testOptionalOperandOp
@run
def testOptionalOperandOp():
  with Context() as ctx, Location.unknown():
    test.register_python_test_dialect(ctx)

    module = Module.create()
    with InsertionPoint(module.body):

      op1 = test.OptionalOperandOp()
      # CHECK: op1.input is None: True
      print(f"op1.input is None: {op1.input is None}")

      op2 = test.OptionalOperandOp(input=op1)
      # CHECK: op2.input is None: False
      print(f"op2.input is None: {op2.input is None}")


# CHECK-LABEL: TEST: testCustomAttribute
@run
def testCustomAttribute():
  with Context() as ctx:
    test.register_python_test_dialect(ctx)
    a = test.TestAttr.get()
    # CHECK: #python_test.test_attr
    print(a)

    # The following cast must not assert.
    b = test.TestAttr(a)

    unit = UnitAttr.get()
    try:
      test.TestAttr(unit)
    except ValueError as e:
      assert "Cannot cast attribute to TestAttr" in str(e)
    else:
      raise

    # The following must trigger a TypeError from our adaptors and must not
    # crash.
    try:
      test.TestAttr(42)
    except TypeError as e:
      assert "Expected an MLIR object" in str(e)
    else:
      raise

    # The following must trigger a TypeError from pybind (therefore, not
    # checking its message) and must not crash.
    try:
      test.TestAttr(42, 56)
    except TypeError:
      pass
    else:
      raise


@run
def testCustomType():
  with Context() as ctx:
    test.register_python_test_dialect(ctx)
    a = test.TestType.get()
    # CHECK: !python_test.test_type
    print(a)

    # The following cast must not assert.
    b = test.TestType(a)

    i8 = IntegerType.get_signless(8)
    try:
      test.TestType(i8)
    except ValueError as e:
      assert "Cannot cast type to TestType" in str(e)
    else:
      raise

    # The following must trigger a TypeError from our adaptors and must not
    # crash.
    try:
      test.TestType(42)
    except TypeError as e:
      assert "Expected an MLIR object" in str(e)
    else:
      raise

    # The following must trigger a TypeError from pybind (therefore, not
    # checking its message) and must not crash.
    try:
      test.TestType(42, 56)
    except TypeError:
      pass
    else:
      raise
