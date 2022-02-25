"""GDB pretty printers for MLIR types."""

import gdb.printing


class StoragePrinter:
  """Prints bases of a struct and its fields."""

  def __init__(self, val):
    self.val = val

  def children(self):
    for field in self.val.type.fields():
      if field.is_base_class:
        yield '<%s>' % field.name, self.val.cast(field.type)
      else:
        yield field.name, self.val[field.name]


class TupleTypeStoragePrinter(StoragePrinter):

  def children(self):
    for child in StoragePrinter.children(self):
      yield child
    pointer_type = gdb.lookup_type('mlir::Type').pointer()
    elements = (self.val.address + 1).cast(pointer_type)
    for i in range(self.val['numElements']):
      yield 'elements[%u]' % i, elements[i]


class FusedLocationStoragePrinter(StoragePrinter):

  def children(self):
    for child in StoragePrinter.children(self):
      yield child
    pointer_type = gdb.lookup_type('mlir::Location').pointer()
    elements = (self.val.address + 1).cast(pointer_type)
    for i in range(self.val['numLocs']):
      yield 'locs[%u]' % i, elements[i]


class StorageTypeMap:
  """Maps a TypeID to the corresponding concrete type.

  Types need to be registered by name before the first lookup.
  """

  def __init__(self):
    self.map = None
    self.type_names = []

  def register_type(self, type_name):
    assert not self.map, 'register_type called after __getitem__'
    self.type_names += [type_name]

  def _init_map(self):
    """Lazy initialization  of self.map."""
    if self.map:
      return
    self.map = {}
    for type_name in self.type_names:
      concrete_type = gdb.lookup_type(type_name)
      try:
        storage = gdb.parse_and_eval(
            "&'mlir::detail::TypeIDExported::get<%s>()::instance'" % type_name)
      except gdb.error:
        # Skip when TypeID instance cannot be found in current context.
        continue
      if concrete_type and storage:
        self.map[int(storage)] = concrete_type

  def __getitem__(self, type_id):
    self._init_map()
    return self.map.get(int(type_id['storage']))


storage_type_map = StorageTypeMap()


def get_type_id_printer(val):
  """Returns a printer of the name of a mlir::TypeID."""

  class TypeIdPrinter:

    def __init__(self, string):
      self.string = string

    def to_string(self):
      return self.string

  concrete_type = storage_type_map[val]
  if not concrete_type:
    return None
  return TypeIdPrinter('mlir::TypeID::get<%s>()' % concrete_type)


def get_attr_or_type_printer(val, get_type_id):
  """Returns a printer for mlir::Attribute or mlir::Type."""

  class AttrOrTypePrinter:

    def __init__(self, type_id, impl):
      self.type_id = type_id
      self.impl = impl

    def children(self):
      yield 'typeID', self.type_id
      yield 'cast<%s>(impl)' % self.impl.type, self.impl

  if not val['impl']:
    return None
  impl = val['impl'].dereference()
  type_id = get_type_id(impl)
  concrete_type = storage_type_map[type_id]
  if not concrete_type:
    return None
  # 3rd template argument of StorageUserBase is the storage type.
  storage_type = concrete_type.fields()[0].type.template_argument(2)
  if not storage_type:
    return None
  return AttrOrTypePrinter(type_id, impl.cast(storage_type))


class ImplPrinter:
  """Printer for an instance with a single 'impl' member pointer."""

  def __init__(self, val):
    self.impl = val['impl']

  def children(self):
    yield 'impl', (self.impl.dereference() if self.impl else self.impl)


# Printers of types deriving from Attribute::AttrBase or Type::TypeBase.
for name in [
    # mlir/IR/Attributes.h
    'ArrayAttr',
    'DictionaryAttr',
    'FloatAttr',
    'IntegerAttr',
    'IntegerSetAttr',
    'OpaqueAttr',
    'StringAttr',
    'SymbolRefAttr',
    'TypeAttr',
    'UnitAttr',
    'DenseStringElementsAttr',
    'DenseIntOrFPElementsAttr',
    'OpaqueElementsAttr',
    'SparseElementsAttr',
    # mlir/IR/BuiltinTypes.h
    'ComplexType',
    'IndexType',
    'IntegerType',
    'Float16Type',
    'Float32Type',
    'Float64Type',
    'Float80Type',
    'Float128Type',
    'NoneType',
    'VectorType',
    'RankedTensorType',
    'UnrankedTensorType',
    'MemRefType',
    'UnrankedMemRefType',
    'TupleType',
    # mlir/IR/Location.h
    'CallSiteLoc',
    'FileLineColLoc',
    'FusedLoc',
    'NameLoc',
    'OpaqueLoc',
    'UnknownLoc'
]:
  storage_type_map.register_type('mlir::%s' % name)  # Register for upcasting.
storage_type_map.register_type('void')  # Register default.


pp = gdb.printing.RegexpCollectionPrettyPrinter('MLIRSupport')

pp.add_printer('mlir::OperationName', '^mlir::OperationName$', ImplPrinter)
pp.add_printer('mlir::Value', '^mlir::Value$', ImplPrinter)

# Printers for types deriving from AttributeStorage or TypeStorage.
pp.add_printer('mlir::detail::FusedLocationStorage',
               '^mlir::detail::FusedLocationStorage',
               FusedLocationStoragePrinter)
pp.add_printer('mlir::detail::TupleTypeStorage',
               '^mlir::detail::TupleTypeStorage$', TupleTypeStoragePrinter)

pp.add_printer('mlir::TypeID', '^mlir::TypeID$', get_type_id_printer)


def add_attr_or_type_printers(name):
  """Adds printers for mlir::Attribute or mlir::Type and their Storage type."""
  get_type_id = lambda val: val['abstract%s' % name]['typeID']
  pp.add_printer('mlir::%s' % name, '^mlir::%s$' % name,
                 lambda val: get_attr_or_type_printer(val, get_type_id))


# Upcasting printers of mlir::Attribute and mlir::Type.
for name in ['Attribute', 'Type']:
  add_attr_or_type_printers(name)

gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
