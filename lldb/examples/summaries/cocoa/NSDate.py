# summary provider for NSDate
import lldb
import ctypes
import objc_runtime
import metrics
import struct
import time
import datetime

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# Python promises to start counting time at midnight on Jan 1st on the epoch year
# hence, all we need to know is the epoch year
python_epoch = time.gmtime(0).tm_year

osx_epoch = datetime.date(2001,1,1).timetuple()

def mkgmtime(t):
    return time.mktime(t)-time.timezone

osx_epoch = mkgmtime(osx_epoch)

def osx_to_python_time(osx):
	if python_epoch <= 2011:
		return osx + osx_epoch
	else:
		return osx - osx_epoch


# despite the similary to synthetic children providers, these classes are not
# trying to provide anything but the port number of an NSDate, so they need not
# obey the interface specification for synthetic children providers
class NSTaggedDate_SummaryProvider:
	def adjust_for_architecture(self):
		self.is_64_bit = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()

	def __init__(self, valobj, info_bits, data):
		self.valobj = valobj;
		self.update();
		self.info_bits = info_bits
		self.data = data

	def update(self):
		self.adjust_for_architecture();
		self.id_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeObjCID)

		self.char = self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar)
		self.short = self.valobj.GetType().GetBasicType(lldb.eBasicTypeShort)
		self.ushort = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedShort)
		self.int = self.valobj.GetType().GetBasicType(lldb.eBasicTypeInt)
		self.long = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLong)
		self.ulong = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		self.longlong = self.valobj.GetType().GetBasicType(lldb.eBasicTypeLongLong)
		self.ulonglong = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLongLong)
		self.float = self.valobj.GetType().GetBasicType(lldb.eBasicTypeFloat)
		self.double = self.valobj.GetType().GetBasicType(lldb.eBasicTypeDouble)

	def value(self):
		# the value of the date-time object is wrapped into the pointer value
		# unfortunately, it is made as a time-delta after Jan 1 2011 midnight GMT
		# while all Python knows about is the "epoch", which is a platform-dependent
		# year (1970 of *nix) whose Jan 1 at midnight is taken as reference
		return time.ctime(osx_to_python_time(self.data))


class NSUntaggedDate_SummaryProvider:
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
		self.NSUInteger = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		self.double = self.valobj.GetType().GetBasicType(lldb.eBasicTypeDouble)

	def offset(self):
		if self.is_64_bit:
			return 8
		else:
			return 4


	def value(self):
		value = self.valobj.CreateChildAtOffset("value",
							self.offset(),
							self.double)
		value_double = struct.unpack('d', struct.pack('Q', value.GetValueAsUnsigned(0)))[0]
		return time.ctime(osx_to_python_time(value_double))

class NSUnknownDate_SummaryProvider:
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

	def value(self):
		stream = lldb.SBStream()
		self.valobj.GetExpressionPath(stream)
		expr = "(NSString*)[" + stream.GetData() + " description]"
		num_children_vo = self.valobj.CreateValueFromExpression("str",expr);
		return num_children_vo.GetSummary()

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
	if name_string == 'NSDate' or name_string == '__NSDate' or name_string == '__NSTaggedDate':
		if class_data.is_tagged():
			wrapper = NSTaggedDate_SummaryProvider(valobj,class_data.info_bits(),class_data.value())
			statistics.metric_hit('code_notrun',valobj)
		else:
			wrapper = NSUntaggedDate_SummaryProvider(valobj)
			statistics.metric_hit('code_notrun',valobj)
	else:
		wrapper = NSUnknownDate_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;


def NSDate_SummaryProvider (valobj,dict):
	provider = GetSummary_Impl(valobj);
	if provider != None:
	    #try:
	    summary = provider.value();
	    #except:
	    #    summary = None
	    if summary == None:
	        summary = 'no valid number here'
	    return str(summary)
	return ''


def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F NSDate.NSDate_SummaryProvider NSDate")

