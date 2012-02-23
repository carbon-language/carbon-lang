# summary provider for class NSException
import objc_runtime
import metrics
import CFString
import lldb

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

class NSKnownException_SummaryProvider:
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

	# skip the ISA and go to the name pointer
	def offset(self):
		if self.lp64:
			return 8
		else:
			return 4

	def description(self):
		name_ptr = self.valobj.CreateChildAtOffset("name",
							self.offset(),
							self.id_type)
		reason_ptr = self.valobj.CreateChildAtOffset("reason",
							2*self.offset(),
							self.id_type)
		return CFString.CFString_SummaryProvider(name_ptr,None) + " " + CFString.CFString_SummaryProvider(reason_ptr,None)


class NSUnknownException_SummaryProvider:
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

	def description(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		name_vo = self.valobj.CreateValueFromExpression("name","(NSString*)[" + stream.GetData() + " name]");
		reason_vo = self.valobj.CreateValueFromExpression("reason","(NSString*)[" + stream.GetData() + " reason]");
		return CFString.CFString_SummaryProvider(name_vo,None) + ' ' + CFString.CFString_SummaryProvider(reason_vo,None)


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
	if name_string == 'NSException':
		wrapper = NSKnownException_SummaryProvider(valobj)
		statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSUnknownException_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;

def NSException_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    try:
	        summary = provider.description();
	    except:
	        summary = None
	    if summary == None:
	        summary = 'no valid exception here'
	    return str(summary)
	return ''

def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F NSException.NSException_SummaryProvider NSException")
