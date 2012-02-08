import re

# C++ STL formatters for LLDB
# These formatters are based upon the version of the GNU libstdc++
# as it ships with Mac OS X 10.6.8 thru 10.7.3
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

	def extract_type(self):
		list_type = self.valobj.GetType().GetUnqualifiedType()
		if list_type.GetNumberOfTemplateArguments() > 0:
			data_type = list_type.GetTemplateArgumentType(0)
		else:
			data_type = None
		return data_type

	def update(self):
		try:
			impl = self.valobj.GetChildMemberWithName('_M_impl')
			node = impl.GetChildMemberWithName('_M_node')
			self.node_address = self.valobj.AddressOf().GetValueAsUnsigned(0)
			self.next = node.GetChildMemberWithName('_M_next')
			self.prev = node.GetChildMemberWithName('_M_prev')
			self.data_type = self.extract_type()
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
		
	# we need this function as a temporary workaround for rdar://problem/10801549
	# which prevents us from extracting the std::pair<K,V> SBType out of the template
	# arguments for _Rep_Type _M_t in the map itself - because we have to make up the
	# typename and then find it, we may hit the situation were std::string has multiple
	# names but only one is actually referenced in the debug information. hence, we need
	# to replace the longer versions of std::string with the shorter one in order to be able
	# to find the type name
	def fixup_class_name(self, class_name):
		if class_name == 'std::basic_string<char, class std::char_traits<char>, class std::allocator<char> >':
			return 'std::basic_string<char>'
		if class_name == 'basic_string<char, class std::char_traits<char>, class std::allocator<char> >':
			return 'std::basic_string<char>'
		if class_name == 'std::basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>'
		if class_name == 'basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>'
		return class_name

	def update(self):
		try:
			self.Mt = self.valobj.GetChildMemberWithName('_M_t')
			self.Mimpl = self.Mt.GetChildMemberWithName('_M_impl')
			self.Mheader = self.Mimpl.GetChildMemberWithName('_M_header')
			
			map_arg_0 = str(self.valobj.GetType().GetTemplateArgumentType(0).GetName())
			map_arg_1 = str(self.valobj.GetType().GetTemplateArgumentType(1).GetName())
			
			map_arg_0 = self.fixup_class_name(map_arg_0)
			map_arg_1 = self.fixup_class_name(map_arg_1)
			
			map_arg_type = "std::pair<const " + map_arg_0 + ", " + map_arg_1
			if map_arg_1[-1] == '>':
				map_arg_type = map_arg_type + " >"
			else:
				map_arg_type = map_arg_type + ">"
			
			self.data_type = self.valobj.GetTarget().FindFirstType(map_arg_type)
			
			# from libstdc++ implementation of _M_root for rbtree
			self.Mroot = self.Mheader.GetChildMemberWithName('_M_parent')
			self.data_size = self.data_type.GetByteSize()
			self.skip_size = self.Mheader.GetType().GetByteSize()
		except:
			pass

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

