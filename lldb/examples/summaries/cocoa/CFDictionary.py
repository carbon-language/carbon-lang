# summary provider for NSDictionary
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
# trying to provide anything but the count for an NSDictionary, so they need not
# obey the interface specification for synthetic children providers
class NSCFDictionary_SummaryProvider:
	def adjust_for_architecture(self):
		self.lp64 = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
		self.valobj = valobj;
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)
		if self.lp64:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)

	# empirically determined on both 32 and 64bit desktop Mac OS X
	# probably boils down to 2 pointers and 4 bytes of data, but
	# the description of __CFDictionary is not readily available so most
	# of this is guesswork, plain and simple
	def offset(self):
		if self.lp64:
			return 20
		else:
			return 12

	def num_children(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		return num_children_vo.GetValueAsUnsigned(0)


class NSDictionaryI_SummaryProvider:
	def adjust_for_architecture(self):
		self.lp64 = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
		self.valobj = valobj;
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)
		if self.lp64:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)

	# we just need to skip the ISA and the count immediately follows
	def offset(self):
		if self.lp64:
			return 8
		else:
			return 4

	def num_children(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		value = num_children_vo.GetValueAsUnsigned(0)
		if value != None:
			# the MSB on immutable dictionaries seems to be taken by the LSB of capacity
			# not sure if it is a bug or some weird sort of feature, but masking it out
			# gets the count right (unless, of course, someone's dictionaries grow
			#                       too large - but I have not tested this)
			if self.lp64:
				value = value & ~0xFF00000000000000
			else:
				value = value & ~0xFF000000
		return value

class NSDictionaryM_SummaryProvider:
	def adjust_for_architecture(self):
		self.lp64 = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
		self.valobj = valobj;
		self.update();

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)
		if self.lp64:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)

	# we just need to skip the ISA and the count immediately follows
	def offset(self):
		if self.lp64:
			return 8
		else:
			return 4

	def num_children(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		return num_children_vo.GetValueAsUnsigned(0)


class NSDictionaryUnknown_SummaryProvider:
	def adjust_for_architecture(self):
		self.lp64 = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj):
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
	if name_string == '__NSCFDictionary':
		wrapper = NSCFDictionary_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	elif name_string == '__NSDictionaryI':
		wrapper = NSDictionaryI_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	elif name_string == '__NSDictionaryM':
		wrapper = NSDictionaryM_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSDictionaryUnknown_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def CFDictionary_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    try:
	        summary = str(provider.num_children());
	    except:
	        summary = None
	    if summary == None:
	        summary = 'no valid dictionary here'
	    return summary + " key/value pairs"
	return ''

def CFDictionary_SummaryProvider2 (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
		try:
			summary = (provider.num_children());
		except:
			summary = None
		if summary == None:
			summary = 'no valid dictionary here'
		# needed on OSX Mountain Lion
		elif provider.lp64:
			summary = int(summary) & ~0x0f1f000000000000
		return str(summary) + " key/value pairs"
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFDictionary.CFDictionary_SummaryProvider NSDictionary")
	debugger.HandleCommand("type summary add -F CFDictionary.CFDictionary_SummaryProvider2 CFDictionaryRef CFMutableDictionaryRef")
