# summary provider for NSData
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
# trying to provide anything but the length for an NSData, so they need not
# obey the interface specification for synthetic children providers
class NSConcreteData_SummaryProvider:
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

	# one pointer is the ISA
	# then there are 32 bit worth of flags and other data
	# however, on 64bit systems these are padded to be a full
	# machine word long, which means we actually have two pointers
	# worth of data to skip
	def offset(self):
		return 2 * self.sys_params.pointer_size

	def length(self):
		size = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.sys_params.types_cache.NSUInteger)
		return size.GetValueAsUnsigned(0)


class NSDataUnknown_SummaryProvider:
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
		num_children_vo = self.valobj.CreateValueFromExpression("count","(int)[" + stream.GetData() + " length]");
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
	if name_string == 'NSConcreteData' or \
	   name_string == 'NSConcreteMutableData' or \
	   name_string == '__NSCFData':
		wrapper = NSConcreteData_SummaryProvider(valobj, class_data.sys_params)
		statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSDataUnknown_SummaryProvider(valobj, class_data.sys_params)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def NSData_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    try:
	        summary = provider.length();
	    except:
	        summary = None
	    if summary == None:
	        summary = 'no valid data here'
	    else:
	        if summary == 1:
	           summary = '1 byte'
	        else:
	           summary = str(summary) + ' bytes'
	    return summary
	return ''

def NSData_SummaryProvider2 (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    try:
	        summary = provider.length();
	    except:
	        summary = None
	    if summary == None:
	        summary = 'no valid data here'
	    else:
	        if summary == 1:
	           summary = '@"1 byte"'
	        else:
	           summary = '@"' + str(summary) + ' bytes"'
	    return summary
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F NSData.NSData_SummaryProvider NSData")
	debugger.HandleCommand("type summary add -F NSData.NSData_SummaryProvider2 CFDataRef CFMutableDataRef")
