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
		pass

	def __init__(self, valobj, dict, params):
		self.valobj = valobj;
		self.update()

	def update(self):
		self.adjust_for_architecture();

	def num_children(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		num_children_vo = self.valobj.CreateValueFromExpression("count","(int)[" + stream.GetData() + " count]");
		return num_children_vo.GetValueAsUnsigned(0)

# much less functional than the other two cases below
# just runs code to get to the count and then returns
# no children
class NSArrayCF_SynthProvider:

	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, dict, params):
		self.valobj = valobj;
		self.sys_params = params
		if not (self.sys_params.types_cache.ulong):
			self.sys_params.types_cache.ulong = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		self.update()

	def update(self):
		self.adjust_for_architecture();

	def num_children(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.sys_params.cfruntime_size,
							self.sys_params.types_cache.ulong)
		return num_children_vo.GetValueAsUnsigned(0)

class NSArrayI_SynthProvider:
	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, dict, params):
		self.valobj = valobj;
		self.sys_params = params
		if not(self.sys_params.types_cache.long):
			self.sys_params.types_cache.long = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong)
		self.update()

	def update(self):
		self.adjust_for_architecture();

	# skip the isa pointer and get at the size
	def num_children(self):
		count = self.valobj.CreateChildAtOffset("count",
				self.sys_params.pointer_size,
				self.sys_params.types_cache.long);
		return count.GetValueAsUnsigned(0)

class NSArrayM_SynthProvider:
	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, dict, params):
		self.valobj = valobj;
		self.sys_params = params
		if not(self.sys_params.types_cache.long):
			self.sys_params.types_cache.long = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong)
		self.update()

	def update(self):
		self.adjust_for_architecture();

	# skip the isa pointer and get at the size
	def num_children(self):
		count = self.valobj.CreateChildAtOffset("count",
				self.sys_params.pointer_size,
				self.sys_params.types_cache.long);
		return count.GetValueAsUnsigned(0)

# this is the actual synth provider, but is just a wrapper that checks
# whether valobj is an instance of __NSArrayI or __NSArrayM and sets up an
# appropriate backend layer to do the computations 
class NSArray_SynthProvider:
	def adjust_for_architecture(self):
		pass

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.adjust_for_architecture()
		self.wrapper = self.make_wrapper(valobj,dict)
		self.invalid = (self.wrapper == None)

	def num_children(self):
		if self.wrapper == None:
			return 0;
		return self.wrapper.num_children()

	def update(self):
		if self.wrapper == None:
			return
		self.wrapper.update()

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
			wrapper = NSArrayI_SynthProvider(valobj, dict, class_data.sys_params)
			statistics.metric_hit('code_notrun',valobj)
		elif name_string == '__NSArrayM':
			wrapper = NSArrayM_SynthProvider(valobj, dict, class_data.sys_params)
			statistics.metric_hit('code_notrun',valobj)
		elif name_string == '__NSCFArray':
			wrapper = NSArrayCF_SynthProvider(valobj, dict, class_data.sys_params)
			statistics.metric_hit('code_notrun',valobj)
		else:
			wrapper = NSArrayKVC_SynthProvider(valobj, dict, class_data.sys_params)
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
