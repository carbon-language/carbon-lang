# synthetic children provider for CFString
# (and related NSString class)
import lldb

class CFStringSynthProvider:
	def __init__(self,valobj,dict):
		self.valobj = valobj;
		self.update()

	# children other than "content" are for debugging only and must not be used in production code
	def num_children(self):
		if self.invalid:
			return 0;
		return 6;

	def read_unicode(self, pointer):
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
		if self.lp64 == False:
			# for 32bit targets, use safe ObjC code
			return self.handle_unicode_string_safe()
		offset = 12
		pointer = int(self.valobj.GetValue(), 0) + offset
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
		# step 1: find offset
		if self.inline:
			pointer = int(self.valobj.GetValue(), 0) + self.size_of_cfruntime_base();
			if self.explicit == False:
				# untested, use the safe code path
				return self.handle_unicode_string_safe();
			else:
				# not sure why 8 bytes are skipped here
				# (lldb) mem read -c 50 0x00000001001154f0
				# 0x1001154f0: 98 1a 85 71 ff 7f 00 00 90 07 00 00 01 00 00 00  ...q?...........
				# 0x100115500: 03 00 00 00 00 00 00 00 *c3 03 78 00 78 00 00 00  ........?.x.x...
				# 0x100115510: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
				# 0x100115520: 00 00                                            ..
				# content begins at * (i.e. 8 bytes into variants, skipping void* buffer in
				# __notInlineImmutable1 entirely, while the length byte is correctly located
				# for an inline string)
				pointer = pointer + 8;
		else:
			pointer = int(self.valobj.GetValue(), 0) + self.size_of_cfruntime_base();
			# read 8 bytes here and make an address out of them
			vopointer = self.valobj.CreateChildAtOffset("dummy",
				pointer,self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType());
			pointer = int(vopointer.GetValue(), 0)
		# step 2: read Unicode data at pointer
		pystr = self.read_unicode(pointer)
		# step 3: return it
		return self.valobj.CreateValueFromExpression("content",
			"(char*)\"" + pystr.encode('utf-8') + "\"")

	# we read at "the right place" into the __CFString object instead of running code
	# we are replicating the functionality of __CFStrContents in CFString.c here
	def handle_UTF8_inline(self):
		offset = int(self.valobj.GetValue(), 0) + self.size_of_cfruntime_base();
		if self.explicit == False:
			offset = offset + 1;
		return self.valobj.CreateValueFromAddress("content",
				offset, self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar));

	def handle_UTF8_not_inline(self):
		offset = self.size_of_cfruntime_base();
		return self.valobj.CreateChildAtOffset("content",
				offset,self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar).GetPointerType());

	def get_child_at_index(self,index):
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
			if self.unicode == True:
				return self.handle_unicode_string();
			elif self.special == True:
				return self.handle_special();
			elif self.inline == True:
				return self.handle_UTF8_inline();
			else:
				return self.handle_UTF8_not_inline();

	def get_child_index(self,name):
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

	def is_64bit(self):
		return self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8

	def is_little_endian(self):
		return self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle

	# CFRuntimeBase is defined as having an additional
	# 4 bytes (padding?) on LP64 architectures
	# to get its size we add up sizeof(pointer)+4
	# and then add 4 more bytes if we are on a 64bit system
	def size_of_cfruntime_base(self):
		if self.lp64 == True:
			return 8+4+4;
		else:
			return 4+4;

	# the info bits are part of the CFRuntimeBase structure
	# to get at them we have to skip a uintptr_t and then get
	# at the least-significant byte of a 4 byte array. If we are
	# on big-endian this means going to byte 3, if we are on
	# little endian (OSX & iOS), this means reading byte 0
	def offset_of_info_bits(self):
		if self.lp64 == True:
			offset = 8;
		else:
			offset = 4;
		if self.is_little == False:
			offset = offset + 3;
		return offset;

	def read_info_bits(self):
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
		return (self.info_bits & 1) == 1;

	def is_inline(self):
		return (self.info_bits & 0x60) == 0;

	# this flag's name is ambiguous, it turns out
	# we must skip a length byte to get at the data
	# when this flag is False
	def has_explicit_length(self):
		return (self.info_bits & (1 | 4)) != 4;

	# probably a subclass of NSString. obtained this from [str pathExtension]
	# here info_bits = 0 and Unicode data at the start of the padding word
	# in the long run using the isa value might be safer as a way to identify this
	# instead of reading the info_bits
	def is_special_case(self):
		return self.info_bits == 0;

	def is_unicode(self):
		return (self.info_bits & 0x10) == 0x10;

	# preparing ourselves to read into memory
	# by adjusting architecture-specific info
	def adjust_for_architecture(self):
		self.lp64 = self.is_64bit();
		self.is_little = self.is_little_endian();

	# reading info bits out of the CFString and computing
	# useful values to get at the real data
	def compute_flags(self):
		self.info_bits = self.read_info_bits();
		if self.info_bits == None:
			return;
		self.mutable = self.is_mutable();
		self.inline = self.is_inline();
		self.explicit = self.has_explicit_length();
		self.unicode = self.is_unicode();
		self.special = self.is_special_case();

	def update(self):
		self.adjust_for_architecture();
		self.compute_flags();

def CFString_SummaryProvider (valobj,dict):
	provider = CFStringSynthProvider(valobj,dict);
	if provider.invalid == False:
	    return provider.get_child_at_index(provider.get_child_index("content")).GetSummary();
	return ''
	
