# summary provider for CF(Mutable)BitVector
import lldb
import ctypes
import objc_runtime
import metrics

# first define some utility functions
def byte_index(abs_pos):
	return abs_pos/8

def bit_index(abs_pos):
	return abs_pos & 7

def get_bit(byte,index):
	if index < 0 or index > 7:
		return None
	return (byte >> (7-index)) & 1

def grab_array_item_data(pointer,index):
	return pointer.GetPointeeData(index,1)

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# despite the similary to synthetic children providers, these classes are not
# trying to provide anything but a summary for a CF*BitVector, so they need not
# obey the interface specification for synthetic children providers
class CFBitVectorKnown_SummaryProvider:
	def adjust_for_architecture(self):
		self.is_64_bit = self.sys_params.is_64_bit
		self.is_little = self.sys_params.is_little
		self.pointer_size = self.sys_params.pointer_size
		self.cfruntime_size = 16 if self.is_64_bit else 8

	def __init__(self, valobj, params):
		self.valobj = valobj;
		self.sys_params = params
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)
		if self.is_64_bit:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)
		self.charptr_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType()
		self.uiint_size = self.NSUInteger.GetByteSize()

	# we skip the CFRuntimeBase
	# then the next CFIndex is the count
	# then we skip another CFIndex and then we get at a byte array
	# that wraps the individual bits

	def contents(self):
		count_vo = self.valobj.CreateChildAtOffset("count",self.cfruntime_size,
													self.NSUInteger)
		count = count_vo.GetValueAsUnsigned(0)
		if count == 0:
			return '(empty)'
		
		array_vo = self.valobj.CreateChildAtOffset("data",
													self.cfruntime_size+2*self.uiint_size,
													self.charptr_type)
		
		data_list = []
		cur_byte_pos = None
		for i in range(0,count):
			if cur_byte_pos == None:
				cur_byte_pos = byte_index(i)
				cur_byte = grab_array_item_data(array_vo,cur_byte_pos)
				cur_byte_val = cur_byte.uint8[0]
			else:
				byte_pos = byte_index(i)
				# do not fetch the pointee data every single time through
				if byte_pos != cur_byte_pos:
					cur_byte_pos = byte_pos
					cur_byte = grab_array_item_data(array_vo,cur_byte_pos)
					cur_byte_val = cur_byte.uint8[0]
			bit = get_bit(cur_byte_val,bit_index(i))
			if (i % 4) == 0:
				data_list.append(' ')
			if bit == 1:
				data_list.append('1')
			else:
				data_list.append('0')
		return ''.join(data_list)


class CFBitVectorUnknown_SummaryProvider:
	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
		self.valobj = valobj;
		self.update()

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

	def contents(self):
		return '*** unknown class *** very bad thing *** find out my name ***'


def GetSummary_Impl(valobj):
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
	if name_string == '__NSCFType':
		# CFBitVectorRef does not expose an actual NSWrapper type, so we have to check that this is
		# an NSCFType and then check we are a pointer-to CFBitVectorRef
		valobj_type = valobj.GetType()
		if valobj_type.IsValid() and valobj_type.IsPointerType():
			pointee_type = valobj_type.GetPointeeType()
			if pointee_type.GetName() == '__CFBitVector' or pointee_type.GetName() == '__CFMutableBitVector':
				wrapper = CFBitVectorKnown_SummaryProvider(valobj, class_data.sys_params)
				statistics.metric_hit('code_notrun',valobj)
			else:
				wrapper = CFBitVectorUnknown_SummaryProvider(valobj)
				print pointee_type.GetName()
	else:
		wrapper = CFBitVectorUnknown_SummaryProvider(valobj)
		print name_string
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def CFBitVector_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    #try:
	    summary = provider.contents();
	    #except:
	    #    summary = None
	    if summary == None or summary == '':
	        summary = 'no valid bitvector here'
	    return summary
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFBitVector.CFBitVector_SummaryProvider CFBitVectorRef CFMutableBitVectorRef")
