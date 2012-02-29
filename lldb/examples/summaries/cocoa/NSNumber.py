# summary provider for NSNumber
import lldb
import ctypes
import objc_runtime
import metrics
import struct

statistics = metrics.Metrics()
statistics.add_metric('invalid_isa')
statistics.add_metric('invalid_pointer')
statistics.add_metric('unknown_class')
statistics.add_metric('code_notrun')

# despite the similary to synthetic children providers, these classes are not
# trying to provide anything but the port number of an NSNumber, so they need not
# obey the interface specification for synthetic children providers
class NSTaggedNumber_SummaryProvider:
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
		# in spite of the plenty of types made available by the public NSNumber API
		# only a bunch of these are actually used in the internal implementation
		# unfortunately, the original type information appears to be lost
		# so we try to at least recover the proper magnitude of the data
		if self.info_bits == 0:
			return '(char)' + str(self.data % 256)
		if self.info_bits == 4:
			return '(short)' + str(self.data % (256*256))
		if self.info_bits == 8:
			return '(int)' + str(self.data % (256*256*256*256))
		if self.info_bits == 12:
			return '(long)' + str(self.data)
		else:
			return 'absurd value:(info=' + str(self.info_bits) + ", value = " + str(self.data) + ')'


class NSUntaggedNumber_SummaryProvider:
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
		global statistics
		# we need to skip the ISA, then the next byte tells us what to read
		# we then skip one other full pointer worth of data and then fetch the contents
		# if we are fetching an int64 value, one more pointer must be skipped to get at our data
		data_type_vo = self.valobj.CreateChildAtOffset("dt",
							self.pointer_size,
							self.char)
		data_type = ((data_type_vo.GetValueAsUnsigned(0) % 256) & 0x1F)
		data_offset = 2 * self.pointer_size
		if data_type == 0B00001:
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.char)
			statistics.metric_hit('code_notrun',self.valobj)
			return '(char)' + str(data_vo.GetValueAsUnsigned(0))
		elif data_type == 0B0010:
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.short)
			statistics.metric_hit('code_notrun',self.valobj)
			return '(short)' + str(data_vo.GetValueAsUnsigned(0) % (256*256))
		# IF tagged pointers are possible on 32bit+v2 runtime
		# (of which the only existing instance should be iOS)
		# then values of this type might be tagged
		elif data_type == 0B0011:
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.int)
			statistics.metric_hit('code_notrun',self.valobj)
			return '(int)' + str(data_vo.GetValueAsUnsigned(0) % (256*256*256*256))
		# apparently, on is_64_bit architectures, these are the only values that will ever
		# be represented by a non tagged pointers
		elif data_type == 0B10001 or data_type == 0B0100:
			data_offset = data_offset + self.pointer_size
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.longlong)
			statistics.metric_hit('code_notrun',self.valobj)
			return '(long)' + str(data_vo.GetValueAsUnsigned(0))
		elif data_type == 0B0101:
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.longlong)
			data_plain = int(str(data_vo.GetValueAsUnsigned(0) & 0x00000000FFFFFFFF))
			packed = struct.pack('I', data_plain)
			data_float = struct.unpack('f', packed)[0]
			statistics.metric_hit('code_notrun',self.valobj)
			return '(float)' + str(data_float)
		elif data_type == 0B0110:
			data_vo = self.valobj.CreateChildAtOffset("data",
								data_offset,
								self.longlong)
			data_plain = data_vo.GetValueAsUnsigned(0)
			data_double = struct.unpack('d', struct.pack('Q', data_plain))[0]
			statistics.metric_hit('code_notrun',self.valobj)
			return '(double)' + str(data_double)
		statistics.metric_hit('unknown_class',str(self.valobj) + " had unknown data_type " + str(data_type))
		return 'absurd: dt = ' + str(data_type)


class NSUnknownNumber_SummaryProvider:
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
		expr = "(NSString*)[" + stream.GetData() + " stringValue]"
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
	if name_string == 'NSNumber' or name_string == '__NSCFNumber':
		if class_data.is_tagged():
			wrapper = NSTaggedNumber_SummaryProvider(valobj,class_data.info_bits(),class_data.value())
			statistics.metric_hit('code_notrun',valobj)
		else:
			# the wrapper might be unable to decipher what is into the NSNumber
			# and then have to run code on it
			wrapper = NSUntaggedNumber_SummaryProvider(valobj)
	else:
		wrapper = NSUnknownNumber_SummaryProvider(valobj)
		statistics.metric_hit('unknown_class',str(valobj) + " seen as " + name_string)
	return wrapper;


def NSNumber_SummaryProvider (valobj,dict):
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
	debugger.HandleCommand("type summary add -F NSNumber.NSNumber_SummaryProvider NSNumber")

