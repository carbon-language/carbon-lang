# summary provider for NSSet
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
# trying to provide anything but the port number of an NSMachPort, so they need not
# obey the interface specification for synthetic children providers
class NSCFSet_SummaryProvider:
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
		self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)

	# one pointer is the ISA
	# then we have one other internal pointer, plus
	# 4 bytes worth of flags. hence, these values
	def offset(self):
		if self.lp64:
			return 20
		else:
			return 12

	def count(self):
		vcount = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		return vcount.GetValueAsUnsigned(0)


class NSSetUnknown_SummaryProvider:
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

	def count(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		expr = "(int)[" + stream.GetData() + " count]"
		num_children_vo = self.valobj.CreateValueFromExpression("count",expr);
		return num_children_vo.GetValueAsUnsigned(0)

class NSSetI_SummaryProvider:
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

	def count(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
		value = num_children_vo.GetValueAsUnsigned(0)
		if value != None:
			# the MSB on immutable sets seems to be taken by some other data
			# not sure if it is a bug or some weird sort of feature, but masking it out
			# gets the count right (unless, of course, someone's dictionaries grow
			#                       too large - but I have not tested this)
			if self.lp64:
				value = value & ~0xFF00000000000000
			else:
				value = value & ~0xFF000000
		return value

class NSSetM_SummaryProvider:
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

	def count(self):
		num_children_vo = self.valobj.CreateChildAtOffset("count",
							self.offset(),
							self.NSUInteger)
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
	if name_string == '__NSCFSet':
		wrapper = NSCFSet_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	elif name_string == '__NSSetI':
		wrapper = NSSetI_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	elif name_string == '__NSSetM':
		wrapper = NSSetM_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSSetUnknown_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;


def NSSet_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    #try:
	    summary = provider.count();
	    #except:
	    #    summary = None
	    if summary == None:
	        summary = 'no valid set here'
	    return str(summary) + ' objects'
	return ''

def NSSet_SummaryProvider2 (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
		try:
			summary = provider.count();
		except:
			summary = None
		# for some reason, one needs to clear some bits for the count returned
		# to be correct when using directly CF*SetRef as compared to NS*Set
		# this only happens on 64bit, and the bit mask was derived through
		# experimentation (if counts start looking weird, then most probably
		#                  the mask needs to be changed)
		if summary == None:
			summary = 'no valid set here'
		else:
			if provider.lp64:
				summary = int(summary) & ~0x1fff000000000000
		return str(summary) + ' objects'
	return ''


def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F NSSet.NSSet_SummaryProvider NSSet")
	debugger.HandleCommand("type summary add -F NSSet.NSSet_SummaryProvider2 CFSetRef CFMutableSetRef")
