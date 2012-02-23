# summary provider for NSURL
import lldb
import ctypes
import objc_runtime
import metrics
import CFString

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# despite the similary to synthetic children providers, these classes are not
# trying to provide anything but a summary for an NSURL, so they need not
# obey the interface specification for synthetic children providers
class NSURLKnown_SummaryProvider:
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
			self.pointer_size = 8
		else:
			self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedInt)
			self.pointer_size = 4
		self.NSString = self.valobj.GetTarget().FindFirstType('NSString')
		self.NSURL = self.valobj.GetTarget().FindFirstType('NSURL')

	# one pointer is the ISA
	# then there is one more pointer and 8 bytes of plain data
	# (which are also present on a 32-bit system)
	# plus another pointer, and then the real data
	def offset(self):
		if self.lp64:
			return 24
		else:
			return 16

	def url_text(self):
		text = self.valobj.CreateChildAtOffset("text",
							self.offset(),
							self.NSString.GetPointerType())
		base = self.valobj.CreateChildAtOffset("base",
							self.offset()+self.pointer_size,
							self.NSURL.GetPointerType())
		my_string = CFString.CFString_SummaryProvider(text,None)
		if base.GetValueAsUnsigned(0) != 0:
			my_string = my_string + " (base path: " + NSURL_SummaryProvider(base,None) + ")"
		return my_string


class NSURLUnknown_SummaryProvider:
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

	def url_text(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		url_text_vo = self.valobj.CreateValueFromExpression("url","(NSString*)[" + stream.GetData() + " description]");
		return CFString.CFString_SummaryProvider(url_text_vo,None)


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
	if name_string == 'NSURL':
		wrapper = NSURLKnown_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSURLUnknown_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def NSURL_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    try:
	        summary = provider.url_text();
	    except:
	        summary = None
	    if summary == None or summary == '':
	        summary = 'no valid NSURL here'
	    return summary
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F NSURL.NSURL_SummaryProvider NSURL CFURLRef")
