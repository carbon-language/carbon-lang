# summary provider for CFBag
import lldb
import ctypes
import objc_runtime
import metrics

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# despite the similary to synthetic children providers, these classes are not
# trying to provide anything but the length for an CFBag, so they need not
# obey the interface specification for synthetic children providers
class CFBagRef_SummaryProvider:
	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
		self.valobj = valobj;
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)
		if self.is_64_bit:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)

	# 12 bytes on i386
	# 20 bytes on x64
	# most probably 2 pointers and 4 bytes of data
	def offset(self):
		if self.is_64_bit:
			return 20
		else:
			return 12

	def length(self):
		size = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		return size.GetValueAsUnsigned(0)


class CFBagUnknown_SummaryProvider:
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

	def length(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		num_children_vo = self.valobj.CreateValueFromExpression("count","(int)CFBagGetCount(" + stream.GetData() + " )");
		return num_children_vo.GetValueAsUnsigned(0)


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
	actual_name = name_string
	if name_string == '__NSCFType':
		# CFBag does not expose an actual NSWrapper type, so we have to check that this is
		# an NSCFType and then check we are a pointer-to __CFBag
		valobj_type = valobj.GetType()
		if valobj_type.IsValid() and valobj_type.IsPointerType():
			pointee_type = valobj_type.GetPointeeType()
			actual_name = pointee_type.GetName()
			if actual_name == '__CFBag' or \
			   actual_name == 'const struct __CFBag':
				wrapper = CFBagRef_SummaryProvider(valobj)
				statistics.metric_hit('code_notrun',valobj)
				return wrapper
	wrapper = CFBagUnknown_SummaryProvider(valobj)
	statistics.metric_hit('unknown_class',str(valobj) + " seen as " + actual_name)
	return wrapper;

def CFBag_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
		try:
			summary = provider.length();
		except:
			summary = None
		# for some reason, one needs to clear some bits for the count
		# to be correct when using CF(Mutable)BagRef on x64
		# the bit mask was derived through experimentation
		# (if counts start looking weird, then most probably
		#  the mask needs to be changed)
		if summary == None:
			summary = 'no valid set here'
		else:
			if provider.is_64_bit:
				summary = summary & ~0x1fff000000000000
		if summary == 1:
			return '1 item'
		return str(summary) + " items"
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFBag.CFBag_SummaryProvider CFBagRef CFMutableBagRef")
