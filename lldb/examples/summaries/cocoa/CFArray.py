# synthetic children provider for NSArray
import lldb
import ctypes
import objc_runtime
import metrics

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# much less functional than the other two cases below
# just runs code to get to the count and then returns
# no children
class NSArrayKVC_SynthProvider:

	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	def num_children(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		num_children_vo = self.valobj.CreateValueFromExpression("count","(int)[" + stream.GetData() + " count]");
		return num_children_vo.GetValueAsUnsigned(0)

	def get_child_index(self,name):
		if name == "len":
			return self.num_children();
		else:
			return None

	def get_child_at_index(self, index):
		return None



# much less functional than the other two cases below
# just runs code to get to the count and then returns
# no children
class NSArrayCF_SynthProvider:

	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()
		self.cfruntime_size = self.size_of_cfruntime_base()

	# CFRuntimeBase is defined as having an additional
	# 4 bytes (padding?) on LP64 architectures
	# to get its size we add up sizeof(pointer)+4
	# and then add 4 more bytes if we are on a 64bit system
	def size_of_cfruntime_base(self):
		if self.is_64_bit == True:
			return 8+4+4;
		else:
			return 4+4;

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	def num_children(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.cfruntime_size,
							self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong))
		return num_children_vo.GetValueAsUnsigned(0)

	def get_child_index(self,name):
		if name == "len":
			return self.num_children();
		else:
			return None

	def get_child_at_index(self, index):
		return None


class NSArrayI_SynthProvider:

	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	# skip the isa pointer and get at the size
	def num_children(self):
		offset = self.pointer_size;
		datatype = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong)
		count = self.valobj.CreateChildAtOffset("count",
				offset,
				datatype);
		return int(count.GetValue(), 0)

	def get_child_index(self,name):
		if name == "len":
			return self.num_children();
		else:
			return int(name.lstrip('[').rstrip(']'), 0)

	def get_child_at_index(self, index):
		if index == self.num_children():
			return self.valobj.CreateValueFromExpression("len",
				str(index))
		offset = 2 * self.pointer_size + self.id_type.GetByteSize()*index
		return self.valobj.CreateChildAtOffset('[' + str(index) + ']',
				offset,
				self.id_type)


class NSArrayM_SynthProvider:

	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	# skip the isa pointer and get at the size
	def num_children(self):
		offset = self.pointer_size;
		datatype = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong)
		count = self.valobj.CreateChildAtOffset("count",
				offset,
				datatype);
		return int(count.GetValue(), 0)

	def get_child_index(self,name):
		if name == "len":
			return self.num_children();
		else:
			return int(name.lstrip('[').rstrip(']'), 0)

	def data_offset(self):
		offset = self.pointer_size; # isa
		offset += self.pointer_size; # _used
		offset += self.pointer_size; # _doHardRetain, _doWeakAccess, _size
		offset += self.pointer_size; # _hasObjects, _hasStrongReferences, _offset
		offset += self.pointer_size; # _mutations
		return offset;

	# the _offset field is used to calculate the actual offset
	# when reading a value out of the array. we need to read it
	# to do so we read a whole pointer_size of data from the
	# right spot, and then zero out the two LSB
	def read_offset_field(self):
		disp = self.pointer_size;  # isa
		disp += self.pointer_size; # _used
		disp += self.pointer_size; # _doHardRetain, _doWeakAccess, _size
		offset = self.valobj.CreateChildAtOffset("offset",
					disp,
					self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong))
		offset_value = int(offset.GetValue(), 0)
		offset_value = ctypes.c_uint32((offset_value & 0xFFFFFFFC) >> 2).value
		return offset_value

	# the _used field tells how many items are in the array
	# but since this is a mutable array, it allocates more space
	# for performance reasons. we need to get the real _size of
	# the array to calculate the actual offset of each element
	# in get_child_at_index() (see NSArray.m for details)
	def read_size_field(self):
		disp = self.pointer_size;  # isa
		disp += self.pointer_size; # _used
		size = self.valobj.CreateChildAtOffset("size",
					disp,
					self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong))
		size_value = int(size.GetValue(), 0)
		size_value = ctypes.c_uint32((size_value & 0xFFFFFFFA) >> 2).value
		return size_value

	def get_child_at_index(self, index):
		if index == self.num_children():
			return self.valobj.CreateValueFromExpression("len",
				str(index))
		size = self.read_size_field()
		offset = self.read_offset_field()
		phys_idx = offset + index
		if size <= phys_idx:
			phys_idx -=size;
		# we still need to multiply by element size to do a correct pointer read
		phys_idx *= self.id_type.GetByteSize()
		list_ptr = self.valobj.CreateChildAtOffset("_list",
            self.data_offset(),
            self.id_type.GetBasicType(lldb.eBasicTypeUnsignedLongLong))
		list_addr = int(list_ptr.GetValue(), 0)
		return self.valobj.CreateValueFromAddress('[' + str(index) + ']',
				list_addr + phys_idx,
				self.id_type)

# this is the actual synth provider, but is just a wrapper that checks
# whether valobj is an instance of __NSArrayI or __NSArrayM and sets up an
# appropriate backend layer to do the computations 
class NSArray_SynthProvider:

	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.adjust_for_architecture()
		self.wrapper = self.make_wrapper(valobj,dict)
		self.invalid = (self.wrapper == None)

	def get_child_at_index(self, index):
		if self.wrapper == None:
			return None;
		return self.wrapper.get_child_at_index(index)

	def get_child_index(self,name):
		if self.wrapper == None:
			return None;
		return self.wrapper.get_child_index(name)

	def num_children(self):
		if self.wrapper == None:
			return 0;
		return self.wrapper.num_children()

	def update(self):
		if self.wrapper == None:
			return None;
		return self.wrapper.update()

	def read_ascii(self, pointer):
		process = self.valobj.GetTarget().GetProcess()
		error = lldb.SBError()
		pystr = ''
		# cannot do the read at once because there is no length byte
		while True:
			content = process.ReadMemory(pointer, 1, error)
			new_bytes = bytearray(content)
			b0 = new_bytes[0]
			pointer = pointer + 1
			if b0 == 0:
				break
			pystr = pystr + chr(b0)
		return pystr

	# this code acts as our defense against NULL and unitialized
	# NSArray pointers, which makes it much longer than it would be otherwise
	def make_wrapper(self,valobj,dict):
		global statistics
		class_data = objc_runtime.ObjCRuntime(valobj)
		if class_data.is_valid() == False:
			statistics.metric_hit('invalid_pointer',valobj)
			wrapper = None
			return
		class_data = class_data.read_class_data()
		if class_data.is_valid() == False:
			statistics.metric_hit('invalid_isa',valobj)
			wrapper = None
			return
		if class_data.is_kvo():
			class_data = class_data.get_superclass()
		if class_data.is_valid() == False:
			statistics.metric_hit('invalid_isa',valobj)
			wrapper = None
			return
		
		name_string = class_data.class_name()
		if name_string == '__NSArrayI':
			wrapper = NSArrayI_SynthProvider(valobj, dict)
			statistics.metric_hit('code_notrun',valobj)
		elif name_string == '__NSArrayM':
			wrapper = NSArrayM_SynthProvider(valobj, dict)
			statistics.metric_hit('code_notrun',valobj)
		elif name_string == '__NSCFArray':
			wrapper = NSArrayCF_SynthProvider(valobj, dict)
			statistics.metric_hit('code_notrun',valobj)
		else:
			wrapper = NSArrayKVC_SynthProvider(valobj, dict)
			statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
		return wrapper;

def CFArray_SummaryProvider (valobj,dict):
	provider = NSArray_SynthProvider(valobj,dict);
	if provider.invalid == False:
	    try:
	        summary = str(provider.num_children());
	    except:
	        summary = None
	    if summary == None:
	        summary = 'no valid array here'
	    return summary + " objects"
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFArray.CFArray_SummaryProvider NSArray CFArrayRef CFMutableArrayRef")
