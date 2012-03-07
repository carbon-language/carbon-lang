# summary provider for CFBinaryHeap
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
# trying to provide anything but the length for an CFBinaryHeap, so they need not
# obey the interface specification for synthetic children providers
class CFBinaryHeapRef_SummaryProvider:
	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, params):
		self.valobj = valobj;
		self.sys_params = params
		if not(self.sys_params.types_cache.NSUInteger):
			if self.sys_params.is_64_bit:
				self.sys_params.types_cache.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
			else:
				self.sys_params.types_cache.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)
		self.update();

	def update(self):
		self.adjust_for_architecture();

	# 8 bytes on i386
	# 16 bytes on x64
	# most probably 2 pointers
	def offset(self):
		return 2 * self.sys_params.pointer_size

	def length(self):
		size = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.sys_params.types_cache.NSUInteger)
		return size.GetValueAsUnsigned(0)


class CFBinaryHeapUnknown_SummaryProvider:
	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, params):
		self.valobj = valobj;
		self.sys_params = params
		self.update();

	def update(self):
		self.adjust_for_architecture();

	def length(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		num_children_vo = self.valobj.CreateValueFromExpression("count","(int)CFBinaryHeapGetCount(" + stream.GetData() + " )");
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
	if name_string == '__NSCFType':
		# CFBinaryHeap does not expose an actual NSWrapper type, so we have to check that this is
		# an NSCFType and then check we are a pointer-to CFBinaryHeap
		valobj_type = valobj.GetType()
		if valobj_type.IsValid() and valobj_type.IsPointerType():
			pointee_type = valobj_type.GetPointeeType()
			if pointee_type.GetName() == '__CFBinaryHeap':
				wrapper = CFBinaryHeapRef_SummaryProvider(valobj, class_data.sys_params)
				statistics.metric_hit('code_notrun',valobj)
				return wrapper
	wrapper = CFBinaryHeapUnknown_SummaryProvider(valobj, class_data.sys_params)
	statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def CFBinaryHeap_SummaryProvider (valobj,dict):
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
			if provider.sys_params.is_64_bit:
				summary = summary & ~0x1fff000000000000
			if summary == 1:
				return '1 item'
			else:
				summary = str(summary) + ' items'
		return summary
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFBinaryHeap.CFBinaryHeap_SummaryProvider CFBinaryHeapRef")
