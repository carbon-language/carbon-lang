import re

class StdMapSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def update(self):
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
		root_ptr_val = self.node_ptr_value(self.Mroot)
		if root_ptr_val == 0:
			return 0;
		return self.Mimpl.GetChildMemberWithName('_M_node_count').GetValueAsUnsigned(0)

	def get_child_index(self,name):
		return int(name.lstrip('[').rstrip(']'))

	def get_child_at_index(self,index):
		if index >= self.num_children():
			return None;
		offset = index
		current = self.left(self.Mheader);
		while offset > 0:
			current = self.increment_node(current)
			offset = offset - 1;
		# skip all the base stuff and get at the data
		return current.CreateChildAtOffset('['+str(index)+']',self.skip_size,self.data_type)

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

