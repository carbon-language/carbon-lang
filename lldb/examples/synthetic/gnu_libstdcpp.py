import re

# C++ STL formatters for LLDB
# These formatters are based upon the version of the GNU libstdc++
# as it ships with Mac OS X 10.6.8 and 10.7.0
# You are encouraged to look at the STL implementation for your platform
# before relying on these formatters to do the right thing for your setup

class StdListSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj
		self.update()

	def num_children(self):
		try:
			next_val = self.next.GetValueAsUnsigned(0)
			prev_val = self.prev.GetValueAsUnsigned(0)
			# After a std::list has been initialized, both next and prev will be non-NULL
			if next_val == 0 or prev_val == 0:
				return 0
			if next_val == self.node_address:
				return 0
			if next_val == prev_val:
				return 1
			size = 2
			current = self.next
			while current.GetChildMemberWithName('_M_next').GetValueAsUnsigned(0) != self.node_address:
				size = size + 1
				current = current.GetChildMemberWithName('_M_next')
			return (size - 1)
		except:
			return 0;

	def get_child_index(self,name):
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		if index < 0:
			return None;
		if index >= self.num_children():
			return None;
		try:
			offset = index
			current = self.next
			while offset > 0:
				current = current.GetChildMemberWithName('_M_next')
				offset = offset - 1
			return current.CreateChildAtOffset('['+str(index)+']',2*current.GetType().GetByteSize(),self.data_type)
		except:
			return None

	def extract_type_name(self,name):
		self.type_name = name[16:]
		index = 2
		count_of_template = 1
		while index < len(self.type_name):
			if self.type_name[index] == '<':
				count_of_template = count_of_template + 1
			elif self.type_name[index] == '>':
				count_of_template = count_of_template - 1
			elif self.type_name[index] == ',' and count_of_template == 1:
				self.type_name = self.type_name[:index]
				break
			index = index + 1
		self.type_name_nospaces = self.type_name.replace(", ", ",")

	def update(self):
		try:
			impl = self.valobj.GetChildMemberWithName('_M_impl')
			node = impl.GetChildMemberWithName('_M_node')
			self.extract_type_name(impl.GetType().GetName())
			self.node_address = self.valobj.AddressOf().GetValueAsUnsigned(0)
			self.next = node.GetChildMemberWithName('_M_next')
			self.prev = node.GetChildMemberWithName('_M_prev')
			self.data_type = node.GetTarget().FindFirstType(self.type_name)
			# tries to fight against a difference in formatting type names between gcc and clang
			if self.data_type.IsValid() == False:
				self.data_type = node.GetTarget().FindFirstType(self.type_name_nospaces)
			self.data_size = self.data_type.GetByteSize()
		except:
			pass

class StdVectorSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def num_children(self):
		try:
			start_val = self.start.GetValueAsUnsigned(0)
			finish_val = self.finish.GetValueAsUnsigned(0)
			end_val  = self.end.GetValueAsUnsigned(0)
			# Before a vector has been constructed, it will contain bad values
			# so we really need to be careful about the length we return since
			# unitialized data can cause us to return a huge number. We need
			# to also check for any of the start, finish or end of storage values
			# being zero (NULL). If any are, then this vector has not been 
			# initialized yet and we should return zero

			# Make sure nothing is NULL
			if start_val == 0 or finish_val == 0 or end_val == 0:
				return 0
			# Make sure start is less than finish
			if start_val >= finish_val:
				return 0
			# Make sure finish is less than or equal to end of storage
			if finish_val > end_val:
				return 0

			num_children = (finish_val-start_val)/self.data_size
			return num_children
		except:
			return 0;

	def get_child_index(self,name):
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		if index < 0:
			return None;
		if index >= self.num_children():
			return None;
		try:
			offset = index * self.data_size
			return self.start.CreateChildAtOffset('['+str(index)+']',offset,self.data_type)
		except:
			return None

	def update(self):
		try:
			impl = self.valobj.GetChildMemberWithName('_M_impl')
			self.start = impl.GetChildMemberWithName('_M_start')
			self.finish = impl.GetChildMemberWithName('_M_finish')
			self.end = impl.GetChildMemberWithName('_M_end_of_storage')
			self.data_type = self.start.GetType().GetPointeeType()
			self.data_size = self.data_type.GetByteSize()
		except:
			pass


class StdMapSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def update(self):
		try:
			self.Mt = self.valobj.GetChildMemberWithName('_M_t')
			self.Mimpl = self.Mt.GetChildMemberWithName('_M_impl')
			self.Mheader = self.Mimpl.GetChildMemberWithName('_M_header')
			# from libstdc++ implementation of _M_root for rbtree
			self.Mroot = self.Mheader.GetChildMemberWithName('_M_parent')
			# the stuff into the tree is actually a std::pair<const key, value>
			# life would be much easier if gcc had a coherent way to print out
			# template names in debug info
			self.expand_clang_type_name()
			self.expand_gcc_type_name()
			self.data_type = self.Mt.GetTarget().FindFirstType(self.clang_type_name)
			if self.data_type.IsValid() == False:
				self.data_type = self.Mt.GetTarget().FindFirstType(self.gcc_type_name)
			self.data_size = self.data_type.GetByteSize()
			self.skip_size = self.Mheader.GetType().GetByteSize()
		except:
			pass

	def expand_clang_type_name(self):
		type_name = self.Mimpl.GetType().GetName()
		index = type_name.find("std::pair<")
		type_name = type_name[index+5:]
		index = 6
		template_count = 1
		while index < len(type_name):
			if type_name[index] == '<':
				template_count = template_count + 1
			elif type_name[index] == '>' and template_count == 1:
				type_name = type_name[:index+1]
				break
			elif type_name[index] == '>':
				template_count = template_count - 1
			index = index + 1;
		self.clang_type_name = type_name

	def expand_gcc_type_name(self):
		type_name = self.Mt.GetType().GetName()
		index = type_name.find("std::pair<")
		type_name = type_name[index+5:]
		index = 6
		template_count = 1
		while index < len(type_name):
			if type_name[index] == '<':
				template_count = template_count + 1
			elif type_name[index] == '>' and template_count == 1:
				type_name = type_name[:index+1]
				break
			elif type_name[index] == '>':
				template_count = template_count - 1
			elif type_name[index] == ' ' and template_count == 1 and type_name[index-1] == ',':
			    type_name = type_name[0:index] + type_name[index+1:]
			    index = index - 1
			index = index + 1;
		self.gcc_type_name = type_name

	def num_children(self):
		try:
			root_ptr_val = self.node_ptr_value(self.Mroot)
			if root_ptr_val == 0:
				return 0;
			return self.Mimpl.GetChildMemberWithName('_M_node_count').GetValueAsUnsigned(0)
		except:
			return 0;

	def get_child_index(self,name):
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		if index < 0:
			return None
		if index >= self.num_children():
			return None;
		try:
			offset = index
			current = self.left(self.Mheader);
			while offset > 0:
				current = self.increment_node(current)
				offset = offset - 1;
			# skip all the base stuff and get at the data
			return current.CreateChildAtOffset('['+str(index)+']',self.skip_size,self.data_type)
		except:
			return None

	# utility functions
	def node_ptr_value(self,node):
		return node.GetValueAsUnsigned(0)

	def right(self,node):
		return node.GetChildMemberWithName("_M_right");

	def left(self,node):
		return node.GetChildMemberWithName("_M_left");

	def parent(self,node):
		return node.GetChildMemberWithName("_M_parent");

	# from libstdc++ implementation of iterator for rbtree
	def increment_node(self,node):
		if self.node_ptr_value(self.right(node)) != 0:
			x = self.right(node);
			while self.node_ptr_value(self.left(x)) != 0:
				x = self.left(x);
			return x;
		else:
			x = node;
			y = self.parent(x)
			while(self.node_ptr_value(x) == self.node_ptr_value(self.right(y))):
				x = y;
				y = self.parent(y);
			if self.node_ptr_value(self.right(x)) != self.node_ptr_value(y):
				x = y;
			return x;

