"""
LLDB AppKit formatters

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
# synthetic children and summary provider for CFString
# (and related NSString class)
import lldb
import objc_runtime
import Logger

def CFString_SummaryProvider (valobj,dict):
	logger = Logger.Logger()
	provider = CFStringSynthProvider(valobj,dict);
	if provider.invalid == False:
		try:
			summary = provider.get_child_at_index(provider.get_child_index("content"))
			if type(summary) == lldb.SBValue:
				summary = summary.GetSummary()
			else:
				summary = '"' + summary + '"'
		except:
			summary = None
		if summary == None:
			summary = '<variable is not NSString>'
		return '@'+summary
	return ''

def CFAttributedString_SummaryProvider (valobj,dict):
	logger = Logger.Logger()
	offset = valobj.GetTarget().GetProcess().GetAddressByteSize()
	pointee = valobj.GetValueAsUnsigned(0)
	summary = '<variable is not NSAttributedString>'
	if pointee != None and pointee != 0:
		pointee = pointee + offset
		child_ptr = valobj.CreateValueFromAddress("string_ptr",pointee,valobj.GetType())
		child = child_ptr.CreateValueFromAddress("string_data",child_ptr.GetValueAsUnsigned(),valobj.GetType()).AddressOf()
		provider = CFStringSynthProvider(child,dict);
		if provider.invalid == False:
			try:
				summary = provider.get_child_at_index(provider.get_child_index("content")).GetSummary();
			except:
				summary = '<variable is not NSAttributedString>'
	if summary == None:
		summary = '<variable is not NSAttributedString>'
	return '@'+summary


def __lldb_init_module(debugger,dict):
	debugger.HandleCommand("type summary add -F CFString.CFString_SummaryProvider NSString CFStringRef CFMutableStringRef")
	debugger.HandleCommand("type summary add -F CFString.CFAttributedString_SummaryProvider NSAttributedString")

class CFStringSynthProvider:
	def __init__(self,valobj,dict):
		logger = Logger.Logger()
		self.valobj = valobj;
		self.update()

	# children other than "content" are for debugging only and must not be used in production code
	def num_children(self):
		logger = Logger.Logger()
		if self.invalid:
			return 0;
		return 6;

	def read_unicode(self, pointer):
		logger = Logger.Logger()
		process = self.valobj.GetTarget().GetProcess()
		error = lldb.SBError()
		pystr = u''
		# cannot do the read at once because the length value has
		# a weird encoding. better play it safe here
		while True:
			content = process.ReadMemory(pointer, 2, error)
			new_bytes = bytearray(content)
			b0 = new_bytes[0]
			b1 = new_bytes[1]
			pointer = pointer + 2
			if b0 == 0 and b1 == 0:
				break
			# rearrange bytes depending on endianness
			# (do we really need this or is Cocoa going to
			#  use Windows-compatible little-endian even
			#  if the target is big endian?)
			if self.is_little:
				value = b1 * 256 + b0
			else:
				value = b0 * 256 + b1
			pystr = pystr + unichr(value)
		return pystr

	# handle the special case strings
	# only use the custom code for the tested LP64 case
	def handle_special(self):
		logger = Logger.Logger()
		if self.is_64_bit == False:
			# for 32bit targets, use safe ObjC code
			return self.handle_unicode_string_safe()
		offset = 12
		pointer = self.valobj.GetValueAsUnsigned(0) + offset
		pystr = self.read_unicode(pointer)
		return self.valobj.CreateValueFromExpression("content",
			"(char*)\"" + pystr.encode('utf-8') + "\"")

	# last resort call, use ObjC code to read; the final aim is to
	# be able to strip this call away entirely and only do the read
	# ourselves
	def handle_unicode_string_safe(self):
		return self.valobj.CreateValueFromExpression("content",
			"(char*)\"" + self.valobj.GetObjectDescription() + "\"");

	def handle_unicode_string(self):
		logger = Logger.Logger()
		# step 1: find offset
		if self.inline:
			pointer = self.valobj.GetValueAsUnsigned(0) + self.size_of_cfruntime_base();
			if self.explicit == False:
				# untested, use the safe code path
				return self.handle_unicode_string_safe();
			else:
				# a full pointer is skipped here before getting to the live data
				pointer = pointer + self.pointer_size
		else:
			pointer = self.valobj.GetValueAsUnsigned(0) + self.size_of_cfruntime_base()
			# read 8 bytes here and make an address out of them
			try:
				char_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType()
				vopointer = self.valobj.CreateValueFromAddress("dummy",pointer,char_type);
				pointer = vopointer.GetValueAsUnsigned(0)
			except:
				return self.valobj.CreateValueFromExpression("content",
                                                             '(char*)"@\"invalid NSString\""')
		# step 2: read Unicode data at pointer
		pystr = self.read_unicode(pointer)
		# step 3: return it
		return pystr.encode('utf-8')

	def handle_inline_explicit(self):
		logger = Logger.Logger()
		offset = 3*self.pointer_size
		offset = offset + self.valobj.GetValueAsUnsigned(0)
		return self.valobj.CreateValueFromExpression("content",
				"(char*)(" + str(offset) + ")")

	def handle_mutable_string(self):
		logger = Logger.Logger()
		offset = 2 * self.pointer_size
		data = self.valobj.CreateChildAtOffset("content",
			offset, self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType());
		data_value = data.GetValueAsUnsigned(0)
		data_value = data_value + 1
		return self.valobj.CreateValueFromExpression("content", "(char*)(" + str(data_value) + ")")

	def handle_UTF8_inline(self):
		logger = Logger.Logger()
		offset = self.valobj.GetValueAsUnsigned(0) + self.size_of_cfruntime_base();
		if self.explicit == False:
			offset = offset + 1;
		return self.valobj.CreateValueFromAddress("content",
				offset, self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar)).AddressOf();

	def handle_UTF8_not_inline(self):
		logger = Logger.Logger()
		offset = self.size_of_cfruntime_base();
		return self.valobj.CreateChildAtOffset("content",
				offset,self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType());

	def get_child_at_index(self,index):
		logger = Logger.Logger()
		logger >> "Querying for child [" + str(index) + "]"
		if index == 0:
			return self.valobj.CreateValueFromExpression("mutable",
				str(int(self.mutable)));
		if index == 1:
			return self.valobj.CreateValueFromExpression("inline",
				str(int(self.inline)));
		if index == 2:
			return self.valobj.CreateValueFromExpression("explicit",
				str(int(self.explicit)));
		if index == 3:
			return self.valobj.CreateValueFromExpression("unicode",
				str(int(self.unicode)));
		if index == 4:
			return self.valobj.CreateValueFromExpression("special",
				str(int(self.special)));
		if index == 5:
			# we are handling the several possible combinations of flags.
			# for each known combination we have a function that knows how to
			# go fetch the data from memory instead of running code. if a string is not
			# correctly displayed, one should start by finding a combination of flags that
			# makes it different from these known cases, and provide a new reader function
			# if this is not possible, a new flag might have to be made up (like the "special" flag
			# below, which is not a real flag in CFString), or alternatively one might need to use
			# the ObjC runtime helper to detect the new class and deal with it accordingly
			#print 'mutable = ' + str(self.mutable)
			#print 'inline = ' + str(self.inline)
			#print 'explicit = ' + str(self.explicit)
			#print 'unicode = ' + str(self.unicode)
			#print 'special = ' + str(self.special)
			if self.mutable == True:
				return self.handle_mutable_string()
			elif self.inline == True and self.explicit == True and \
			   self.unicode == False and self.special == False and \
			   self.mutable == False:
				return self.handle_inline_explicit()
			elif self.unicode == True:
				return self.handle_unicode_string();
			elif self.special == True:
				return self.handle_special();
			elif self.inline == True:
				return self.handle_UTF8_inline();
			else:
				return self.handle_UTF8_not_inline();

	def get_child_index(self,name):
		logger = Logger.Logger()
		logger >> "Querying for child ['" + str(name) + "']"
		if name == "content":
			return self.num_children() - 1;
		if name == "mutable":
			return 0;
		if name == "inline":
			return 1;
		if name == "explicit":
			return 2;
		if name == "unicode":
			return 3;
		if name == "special":
			return 4;

	# CFRuntimeBase is defined as having an additional
	# 4 bytes (padding?) on LP64 architectures
	# to get its size we add up sizeof(pointer)+4
	# and then add 4 more bytes if we are on a 64bit system
	def size_of_cfruntime_base(self):
		logger = Logger.Logger()
		return self.pointer_size+4+(4 if self.is_64_bit else 0)

	# the info bits are part of the CFRuntimeBase structure
	# to get at them we have to skip a uintptr_t and then get
	# at the least-significant byte of a 4 byte array. If we are
	# on big-endian this means going to byte 3, if we are on
	# little endian (OSX & iOS), this means reading byte 0
	def offset_of_info_bits(self):
		logger = Logger.Logger()
		offset = self.pointer_size
		if self.is_little == False:
			offset = offset + 3;
		return offset;

	def read_info_bits(self):
		logger = Logger.Logger()
		cfinfo = self.valobj.CreateChildAtOffset("cfinfo",
					self.offset_of_info_bits(),
					self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar));
		cfinfo.SetFormat(11)
		info = cfinfo.GetValue();
		if info != None:
			self.invalid = False;
			return int(info,0);
		else:
			self.invalid = True;
			return None;

	# calculating internal flag bits of the CFString object
	# this stuff is defined and discussed in CFString.c
	def is_mutable(self):
		logger = Logger.Logger()
		return (self.info_bits & 1) == 1;

	def is_inline(self):
		logger = Logger.Logger()
		return (self.info_bits & 0x60) == 0;

	# this flag's name is ambiguous, it turns out
	# we must skip a length byte to get at the data
	# when this flag is False
	def has_explicit_length(self):
		logger = Logger.Logger()
		return (self.info_bits & (1 | 4)) != 4;

	# probably a subclass of NSString. obtained this from [str pathExtension]
	# here info_bits = 0 and Unicode data at the start of the padding word
	# in the long run using the isa value might be safer as a way to identify this
	# instead of reading the info_bits
	def is_special_case(self):
		logger = Logger.Logger()
		return self.info_bits == 0;

	def is_unicode(self):
		logger = Logger.Logger()
		return (self.info_bits & 0x10) == 0x10;

	# preparing ourselves to read into memory
	# by adjusting architecture-specific info
	def adjust_for_architecture(self):
		logger = Logger.Logger()
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()
		self.is_64_bit = self.pointer_size == 8
		self.is_little = self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle

	# reading info bits out of the CFString and computing
	# useful values to get at the real data
	def compute_flags(self):
		logger = Logger.Logger()
		self.info_bits = self.read_info_bits();
		if self.info_bits == None:
			return;
		self.mutable = self.is_mutable();
		self.inline = self.is_inline();
		self.explicit = self.has_explicit_length();
		self.unicode = self.is_unicode();
		self.special = self.is_special_case();

	def update(self):
		logger = Logger.Logger()
		self.adjust_for_architecture();
		self.compute_flags();
