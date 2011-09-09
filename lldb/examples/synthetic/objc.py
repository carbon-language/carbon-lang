"""
Objective-C runtime wrapper - Replicates the behavior of AppleObjCRuntimeV2.cpp in Python code
for the benefit of synthetic children providers and Python summaries

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
import lldb

class ObjCRuntime:

	def __init__(self,valobj = None):
		self.valobj = valobj;
		self.adjust_for_architecture() 

	def adjust_for_architecture(self):
		self.lp64 = (self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
		self.is_little = (self.valobj.GetTarget().GetProcess().GetByteOrder() == lldb.eByteOrderLittle)
		self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()
		self.addr_type = self.valobj.GetType().GetBasicType(lldb.eBasicTypeUnsignedLong)
		self.addr_ptr_type = self.addr_type.GetPointerType()

	def is_tagged(self):
		if valobj is None:
			return None
		ptr_value = self.valobj.GetPointerValue()
		if (ptr_value % 2) == 1:
			return True
		else:
			return False

	def read_ascii(self, pointer):
		process = self.valobj.GetTarget().GetProcess()
		error = lldb.SBError()
		pystr = ''
		# cannot do the read at once because there is no length byte
		while True:
			content = process.ReadMemory(pointer, 1, error)
			new_bytes = bytearray(content)
			b0 = new_bytes[0]
			pointer = pointer + 1
			if b0 == 0:
				break
			pystr = pystr + chr(b0)
		return pystr

	def read_isa(self):
		# read ISA pointer
		isa_pointer = self.valobj.CreateChildAtOffset("cfisa",
			0,
			self.addr_ptr_type)
		if isa_pointer == None or isa_pointer.IsValid() == False:
			return None;
		if isa_pointer.GetValue() == None:
			return None;
		isa = int(isa_pointer.GetValue(), 0)
		if isa == 0 or isa == None:
			return None;
		return isa
		

	def get_parent_class(self, isa = None):
		if isa is None:
			isa = self.read_isa()
		# read superclass pointer
		rw_pointer = isa + self.pointer_size
		rw_object = self.valobj.CreateValueFromAddress("parent_isa",
			rw_pointer,
			self.addr_type)
		if rw_object == None or rw_object.IsValid() == False:
			return None;
		if rw_object.GetValue() == None:
			return None;
		rw = int(rw_object.GetValue(), 0)
		if rw == 0 or rw == None:
			return None;
		return rw

	def get_class_name(self, isa = None):
		if isa is None:
			isa = self.read_isa()
		# read rw pointer
		rw_pointer = isa + 4 * self.pointer_size
		rw_object = self.valobj.CreateValueFromAddress("rw",
			rw_pointer,
			self.addr_type)
		if rw_object == None or rw_object.IsValid() == False:
			return None;
		if rw_object.GetValue() == None:
			return None;
		rw = int(rw_object.GetValue(), 0)
		if rw == 0 or rw == None:
			return None;

		# read data pointer
		data_pointer = rw + 8
		data_object = self.valobj.CreateValueFromAddress("data",
			data_pointer,
			self.addr_type)
		if data_object == None or data_object.IsValid() == False:
			return None;
		if data_object.GetValue() == None:
			return None;
		data = int(data_object.GetValue(), 0)
		if data == 0 or data == None:
			return None;

		# read ro pointer
		ro_pointer = data + 12 + self.pointer_size
		if self.lp64:
			ro_pointer += 4
		ro_object = self.valobj.CreateValueFromAddress("ro",
			ro_pointer,
			self.addr_type)
		if ro_object == None or ro_object.IsValid() == False:
			return None;
		if ro_object.GetValue() == None:
			return None;
		name_pointer = int(ro_object.GetValue(), 0)
		if name_pointer == 0 or name_pointer == None:
			return None;

		# now read the actual name and compare it to known stuff
		name_string = self.read_ascii(name_pointer)
		if (name_string.startswith("NSKVONotify")):
			return self.get_class_name(self.get_parent_class())
		return name_string
