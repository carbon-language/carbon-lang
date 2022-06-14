#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Dict, List, Sequence, Tuple, Union

from .....ir import *

from .... import func
from .... import linalg
from .... import math
from .... import arith
from ...._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values

from .scalar_expr import *
from .config import *
from .comprehension import *
import numpy as np

__all__ = [
    "emit_generic_structured_op",
    "emit_named_structured_op",
    "ValueList",
]

# Type aliases.
ValueList = Union[Sequence[Value], OpResultList]


def isa(cls: Type, ty: Type):
  try:
    cls(ty)
    return True
  except ValueError:
    return False


def prepare_common_structured_op(op_config: LinalgStructuredOpConfig,
                                 *ins: Value, outs: ValueList,
                                 **attrs: Union[Sequence[int], TypeFnType]):
  all_arg_defs = op_config.ordered_operands
  in_arg_defs = [
      d for d in all_arg_defs
      if d.kind in [OperandKind.SCALAR, OperandKind.INPUT_TENSOR]
  ]
  out_arg_defs = [
      d for d in all_arg_defs if d.kind == OperandKind.OUTPUT_TENSOR
  ]
  index_attr_arg_defs = [
      d for d in all_arg_defs if d.kind == OperandKind.INDEX_ATTR
  ]
  fn_attr_arg_defs = [
      d for d in all_arg_defs if d.kind in [
          OperandKind.UNARY_FN_ATTR, OperandKind.BINARY_FN_ATTR,
          OperandKind.TYPE_FN_ATTR
      ]
  ]

  # Verify outs is a sequence or a list of results.
  if not isinstance(outs, (Sequence, OpResultList)):
    raise ValueError(f"Expected named argument outs to have type Sequence or "
                     f"OpResultLis but got {type(outs)}")

  # Arity validation.
  if len(ins) != len(in_arg_defs):
    raise ValueError(f"Expected {len(in_arg_defs)} inputs but got "
                     f"{len(ins)} for {op_config}")
  if outs and len(outs) != len(out_arg_defs):
    raise ValueError(f"Expected {len(out_arg_defs)} outputs but got "
                     f"{len(outs)} for {op_config}")

  # Compute a replacement list for all index attribute symbols.
  expressions = []  # type: Sequence[AffineExpr]
  replacements = []  # type: Sequence[AffineExpr]
  for index_attr in index_attr_arg_defs:
    index_attr_vals = index_attr.operand_def.default_indices
    if index_attr.name in attrs:
      index_attr_vals = attrs.get(index_attr.name)
    assert index_attr_vals, "Index attribute has no value"
    if not all(isinstance(value, int) for value in index_attr_vals):
      raise ValueError(f"Attribute {index_attr.name} needs to be of type "
                       f"Sequence[int] but got {type(index_attr_vals)}")
    results = index_attr.index_attr_map.results  # type: AffineExprList
    if len(index_attr_vals) != len(results):
      raise ValueError(f"Attribute {index_attr.name} has length {len(results)} "
                       f"but got {len(index_attr_vals)} values")
    for expr, value in zip(results, index_attr_vals):
      expressions.append(expr)
      replacements.append(AffineConstantExpr.get(value))

  # Replace all index attribute symbols by their value.
  # TODO: Add support for shape symbols.
  indexing_maps = []  # type: Sequence[AffineMap]
  for curr in op_config.indexing_maps:
    for expression, replacement in zip(expressions, replacements):
      curr = curr.replace(expression, replacement, curr.n_dims, curr.n_symbols)
    indexing_maps.append(curr)

  # TODO: Linalg verification does not currently allow symbols.
  # Compress them for now and verify none are left.
  indexing_maps = AffineMap.compress_unused_symbols(indexing_maps,
                                                    Context.current)
  if any(indexing_map.n_symbols != 0 for indexing_map in indexing_maps):
    raise ValueError(f"Expected indexing_maps to use no symbols after "
                     f"replacement and compression but got {indexing_maps}")

  outs, out_types = _infer_structured_outs(op_config, in_arg_defs, ins,
                                           out_arg_defs, outs)

  result_types = [t for t in out_types if isa(RankedTensorType, t)]

  # Initialize the type dictionary with the predefined types.
  type_mapping = dict()  # type: Dict[str, Type]
  type_mapping["F32"] = F32Type.get()
  type_mapping["F64"] = F64Type.get()
  type_mapping["I32"] = IntegerType.get_signless(32)
  type_mapping["I64"] = IntegerType.get_signless(64)

  # Extract type vars for input/output based types.
  block_arg_types = list()  # type: List[Type]
  for arg_def, arg_element_type in zip(in_arg_defs + out_arg_defs,
                                       _get_types_from_values(*ins, *outs)):
    _add_type_mapping(arg_def, arg_element_type, type_mapping, block_arg_types)

  # Emit the generic op.
  # TODO: Support emission of pure memref form.
  indexing_maps_attr = ArrayAttr.get(
      [AffineMapAttr.get(am) for am in indexing_maps])
  iterator_types_attr = ArrayAttr.get(
      [StringAttr.get(s) for s in op_config.iterator_types])

  # Compute the index attributes used when emitting a named structured op.
  index_attrs = {}  # type: Dict[str, DenseElementAttr]
  for index_attr in index_attr_arg_defs:
    index_attr_vals = attrs.get(index_attr.name)
    # Only forward attributes set to a non-default value.
    if index_attr_vals:
      array = np.array(index_attr_vals, dtype=np.int64)
      index_attrs[index_attr.name] = DenseElementsAttr.get(array)

  # Compute the function attribute mapping.
  fn_attr_mapping = {}
  for fn_attr in fn_attr_arg_defs:
    attr_val = fn_attr.operand_def.default_fn
    attr_kind = fn_attr.kind
    if fn_attr.name in attrs:
      fn = attrs.get(fn_attr.name)
      if attr_kind == OperandKind.UNARY_FN_ATTR:
        if not isinstance(fn, UnaryFnType):
          raise ValueError(f"Attribute {fn_attr.name} needs to be of type "
                           f"UnaryFnType but got {type(attr_val)}")
      elif attr_kind == OperandKind.BINARY_FN_ATTR:
        if not isinstance(fn, BinaryFnType):
          raise ValueError(f"Attribute {fn_attr.name} needs to be of type "
                           f"BinaryFnType but got {type(attr_val)}")
      else:
        if not isinstance(fn, TypeFnType):
          raise ValueError(f"Attribute {fn_attr.name} needs to be of type "
                           f"TypeFnType but got {type(attr_val)}")
      attr_val = fn.fn_name
    assert attr_val, "Function attribute has no value"
    fn_attr_mapping[fn_attr.name] = (attr_val, attr_kind)

  return (all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types,
          type_mapping, indexing_maps_attr, iterator_types_attr, index_attrs,
          fn_attr_mapping, block_arg_types)


def emit_generic_structured_op(op_config: LinalgStructuredOpConfig, *ins: Value,
                               outs: ValueList, **attrs: Sequence[int]):
  all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types, type_mapping, \
  indexing_maps_attr, iterator_types_attr, index_attrs, fn_attr_mapping, \
  block_arg_types = \
     prepare_common_structured_op(op_config, *ins, outs = outs, **attrs)

  # An operation that accesses only scalars and scalar/rank zero tensors is
  # rank polymorhpic. We implement rank polymorphism by generating different
  # indexing maps and iterators that match the rank of the first output tensor.
  # An operation is rank polymorphic if the iteration domain has rank zero.
  if not iterator_types_attr:
    rank = ShapedType(outs[0].type).rank
    iterator_types_attr = ArrayAttr.get([StringAttr.get("parallel")] * rank)
    scalar_map = AffineMap.get(rank, 0, [])
    tensor_map = AffineMap.get_identity(rank)
    indexing_maps = []
    for arg_def in all_arg_defs:
      if arg_def.operand_def.kind == OperandKind.SCALAR:
        indexing_maps.append(scalar_map)
      if arg_def.operand_def.is_tensor():
        idx = arg_def.operand_def.registered_index
        if idx < len(ins) and ShapedType(ins[idx].type).rank == 0:
          indexing_maps.append(scalar_map)
        else:
          indexing_maps.append(tensor_map)
    indexing_maps_attr = ArrayAttr.get(
        [AffineMapAttr.get(am) for am in indexing_maps])

  generic_op = linalg.GenericOp(
      result_tensors=result_types,
      inputs=ins,
      outputs=outs,
      indexing_maps=indexing_maps_attr,
      iterator_types=iterator_types_attr,
      doc=None,  # TODO: Make optional.
      library_call=None)  # TODO: Make optional.

  # Construct the body.
  block_arg_names = _get_operand_def_names(*in_arg_defs, *out_arg_defs)
  block = generic_op.regions[0].blocks.append(*block_arg_types)
  block_arg_mapping = dict(zip(block_arg_names, block.arguments))
  with InsertionPoint(block):
    body_builder = _BodyBuilder(type_mapping, block_arg_mapping,
                                fn_attr_mapping)
    for assignment in op_config.assignments:
      body_builder.assign(assignment)
    body_builder.yield_outputs(*_get_operand_def_names(*out_arg_defs))

  if len(result_types) == 1:
    return generic_op.result
  else:
    return generic_op.results


def emit_named_structured_op(op_config: LinalgStructuredOpConfig, op_name: str,
                             op_class_name: str, *ins: Value, outs: ValueList,
                             **attrs: Sequence[int]):
  all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types, type_mapping, \
  indexing_maps_attr, iterator_types_attr, index_attrs, fn_attr_mapping, \
  block_arg_types = \
     prepare_common_structured_op(op_config, *ins, outs = outs, **attrs)

  # If we get here, there must exist a builtin class `op_class_name`.
  ctx = Context.current
  fully_qualified_name = "linalg." + op_name
  if (not ctx.is_registered_operation(fully_qualified_name) or
      not op_class_name in linalg.__dict__.keys()):
    raise NotImplementedError(
        f"Unknown named op_name / op_class_name: {op_name} / {op_class_name}")

  # Set the index attributes used to compute the indexing maps.
  named_op = getattr(linalg, op_class_name)(ins, outs, result_types)
  for name, value in index_attrs.items():
    named_op.operation.attributes[name] = value

  # Compute the function attributes by combining operand kind and function name.
  for name, (fn_name, kind) in fn_attr_mapping.items():
    assert kind.name.lower().endswith("_attr")
    enum_name = kind.name.lower()[:-5]
    named_op.operation.attributes[name] = Attribute.parse(
        f"#linalg.{enum_name}<{fn_name}>")

  linalg.fill_builtin_region(named_op.operation)

  if len(result_types) == 1:
    return named_op.result
  else:
    return named_op.results


class _BodyBuilder:
  """Constructs a structured op body by evaluating assignments."""

  def __init__(self, type_mapping: Dict[str, Type],
               block_arg_mapping: Dict[str, Value], fn_attr_mapping: Dict[str,
                                                                          str]):
    self.type_mapping = type_mapping
    self.block_arg_mapping = block_arg_mapping
    self.fn_attr_mapping = fn_attr_mapping
    self.yield_mapping = dict()  # type: Dict[str, Value]

  def assign(self, assignment: ScalarAssign):
    if assignment.arg in self.yield_mapping:
      raise ValueError(
          f"Multiple assignments to the same argument are forbidden: "
          f"{assignment}")
    self.yield_mapping[assignment.arg] = self.expression(assignment.value)

  def expression(self, expr: ScalarExpression) -> Value:
    if expr.scalar_arg:
      try:
        return self.block_arg_mapping[expr.scalar_arg.arg]
      except KeyError:
        raise ValueError(f"Argument {expr.scalar_arg.arg} is not bound for "
                         f"this structured op.")
    elif expr.scalar_const:
      value_attr = Attribute.parse(expr.scalar_const.value)
      return arith.ConstantOp(value_attr.type, value_attr).result
    elif expr.scalar_index:
      dim_attr = IntegerAttr.get(
          IntegerType.get_signless(64), expr.scalar_index.dim)
      return linalg.IndexOp(dim_attr).result
    elif expr.scalar_fn:
      kind = expr.scalar_fn.kind.name.lower()
      fn_name = expr.scalar_fn.fn_name
      if expr.scalar_fn.attr_name:
        fn_name, _ = self.fn_attr_mapping[expr.scalar_fn.attr_name]
      fn = self._get_function(f"_{kind}_{fn_name}")
      operand_values = [
          self.expression(operand) for operand in expr.scalar_fn.operands
      ]
      if expr.scalar_fn.kind == FunctionKind.TYPE:
        operand_values = [expr.scalar_fn.type_var.name] + operand_values
      return fn(*operand_values)
    raise NotImplementedError(f"Unimplemented scalar body expression: {expr}")

  def yield_outputs(self, *output_names: str):
    output_values = []
    for n in output_names:
      try:
        output_values.append(self.yield_mapping[n])
      except KeyError:
        raise ValueError(f"Body assignments do not assign all outputs: "
                         f"missing '{n}'")
    linalg.YieldOp(output_values)

  def _get_function(self, fn_name: str) -> Callable:
    try:
      fn = getattr(self, f"{fn_name}")
    except AttributeError:
      raise ValueError(f"Function '{fn_name}' is not a known function")
    return fn

  def _cast(self,
            type_var_name: str,
            operand: Value,
            is_unsigned_cast: bool = False) -> Value:
    try:
      to_type = self.type_mapping[type_var_name]
    except KeyError:
      raise ValueError(f"Unbound type variable '{type_var_name}' ("
                       f"expected one of {self.type_mapping.keys()}")
    if operand.type == to_type:
      return operand
    if _is_integer_type(to_type):
      return self._cast_to_integer(to_type, operand, is_unsigned_cast)
    elif _is_floating_point_type(to_type):
      return self._cast_to_floating_point(to_type, operand, is_unsigned_cast)

  def _cast_to_integer(self, to_type: Type, operand: Value,
                       is_unsigned_cast: bool) -> Value:
    to_width = IntegerType(to_type).width
    operand_type = operand.type
    if _is_floating_point_type(operand_type):
      if is_unsigned_cast:
        return arith.FPToUIOp(to_type, operand).result
      return arith.FPToSIOp(to_type, operand).result
    if _is_index_type(operand_type):
      return arith.IndexCastOp(to_type, operand).result
    # Assume integer.
    from_width = IntegerType(operand_type).width
    if to_width > from_width:
      if is_unsigned_cast:
        return arith.ExtUIOp(to_type, operand).result
      return arith.ExtSIOp(to_type, operand).result
    elif to_width < from_width:
      return arith.TruncIOp(to_type, operand).result
    raise ValueError(f"Unable to cast body expression from {operand_type} to "
                     f"{to_type}")

  def _cast_to_floating_point(self, to_type: Type, operand: Value,
                              is_unsigned_cast: bool) -> Value:
    operand_type = operand.type
    if _is_integer_type(operand_type):
      if is_unsigned_cast:
        return arith.UIToFPOp(to_type, operand).result
      return arith.SIToFPOp(to_type, operand).result
    # Assume FloatType.
    to_width = _get_floating_point_width(to_type)
    from_width = _get_floating_point_width(operand_type)
    if to_width > from_width:
      return arith.ExtFOp(to_type, operand).result
    elif to_width < from_width:
      return arith.TruncFOp(to_type, operand).result
    raise ValueError(f"Unable to cast body expression from {operand_type} to "
                     f"{to_type}")

  def _type_cast_signed(self, type_var_name: str, operand: Value) -> Value:
    return self._cast(type_var_name, operand, False)

  def _type_cast_unsigned(self, type_var_name: str, operand: Value) -> Value:
    return self._cast(type_var_name, operand, True)

  def _unary_exp(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return math.ExpOp(x).result
    raise NotImplementedError("Unsupported 'exp' operand: {x}")

  def _unary_log(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return math.LogOp(x).result
    raise NotImplementedError("Unsupported 'log' operand: {x}")

  def _unary_abs(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return math.AbsOp(x).result
    raise NotImplementedError("Unsupported 'abs' operand: {x}")

  def _unary_ceil(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return math.CeilOp(x).result
    raise NotImplementedError("Unsupported 'ceil' operand: {x}")

  def _unary_floor(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return math.FloorOp(x).result
    raise NotImplementedError("Unsupported 'floor' operand: {x}")

  def _unary_negf(self, x: Value) -> Value:
    if _is_floating_point_type(x.type):
      return arith.NegFOp(x).result
    raise NotImplementedError("Unsupported 'negf' operand: {x}")

  def _binary_add(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.AddFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.AddIOp(lhs, rhs).result
    raise NotImplementedError("Unsupported 'add' operands: {lhs}, {rhs}")

  def _binary_sub(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.SubFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.SubIOp(lhs, rhs).result
    raise NotImplementedError("Unsupported 'sub' operands: {lhs}, {rhs}")

  def _binary_mul(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.MulFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.MulIOp(lhs, rhs).result
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")

  def _binary_max_signed(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.MaxFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.MaxSIOp(lhs, rhs).result
    raise NotImplementedError("Unsupported 'max' operands: {lhs}, {rhs}")

  def _binary_max_unsigned(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.MaxFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.MaxUIOp(lhs, rhs).result
    raise NotImplementedError(
        "Unsupported 'max_unsigned' operands: {lhs}, {rhs}")

  def _binary_min_signed(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.MinFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.MinSIOp(lhs, rhs).result
    raise NotImplementedError("Unsupported 'min' operands: {lhs}, {rhs}")

  def _binary_min_unsigned(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return arith.MinFOp(lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return arith.MinUIOp(lhs, rhs).result
    raise NotImplementedError(
        "Unsupported 'min_unsigned' operands: {lhs}, {rhs}")


def _infer_structured_outs(
    op_config: LinalgStructuredOpConfig,
    in_arg_defs: Sequence[OperandDefConfig], ins: Sequence[Value],
    out_arg_defs: Sequence[OperandDefConfig],
    outs: Union[Sequence[Value], OpResultList]) -> Tuple[ValueList, List[Type]]:
  """Infers implicit outs and output types.

  Respects existing contents of outs if not empty.

  Returns:
    normalized outs, output types
  """
  # If outs were explicitly provided, we accept them verbatim.
  if outs:
    return outs, [out.type for out in outs]

  raise NotImplementedError(f"Output tensor inference not yet supported for "
                            "structured ops")


def _get_types_from_values(*values: Value) -> Sequence[Type]:
  types = []
  for v in values:
    types.append(v.type)
  return types


def _get_operand_def_names(*operand_configs: OperandDefConfig) -> Sequence[str]:
  return [odc.operand_def.name for odc in operand_configs]


def _add_type_mapping(operand_config: OperandDefConfig, operand_type: Type,
                      type_mapping: Dict[str, Type],
                      block_arg_types: Sequence[Type]):
  element_or_self_type = operand_type
  # Get the element type for tensor operands and the type itself for scalars.
  if operand_config.shape_map:
    try:
      element_or_self_type = ShapedType(operand_type).element_type
    except Exception as e:
      raise ValueError(f"Expected ShapedType but got {operand_type}") from e
  name = operand_config.type_var.name
  if name in type_mapping:
    if type_mapping[name] != element_or_self_type:
      raise ValueError(f"Cannot overwrite type mapping {name} = "
                       f"{type_mapping[name]} by type {element_or_self_type}")
  type_mapping[name] = element_or_self_type
  block_arg_types.append(element_or_self_type)


def _is_floating_point_type(t: Type) -> bool:
  # TODO: Create a FloatType in the Python API and implement the switch
  # there.
  return (F64Type.isinstance(t) or F32Type.isinstance(t) or
          F16Type.isinstance(t) or BF16Type.isinstance(t))


def _is_integer_type(t: Type) -> bool:
  return IntegerType.isinstance(t)


def _is_index_type(t: Type) -> bool:
  return IndexType.isinstance(t)


def _get_floating_point_width(t: Type) -> int:
  # TODO: Create a FloatType in the Python API and implement the switch
  # there.
  if F64Type.isinstance(t):
    return 64
  if F32Type.isinstance(t):
    return 32
  if F16Type.isinstance(t):
    return 16
  if BF16Type.isinstance(t):
    return 16
  raise NotImplementedError(f"Unhandled floating point type switch {t}")
