class StdVectorSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update() # initialize this provider

	def num_children(self):
		start_val = self.start.GetValueAsUnsigned(0) # read _M_start
		finish_val = self.finish.GetValueAsUnsigned(0) # read _M_finish
		end_val  = self.end.GetValueAsUnsigned(0) # read _M_end_of_storage
		
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

		# pointer arithmetic: (_M_finish - _M_start) would return the number of
		# items of type T contained in the vector. because Python has no way to know
		# that we want to subtract two pointers instead of two integers, we have to divide
		# by sizeof(T) to be equivalent to the C++ pointer expression
		num_children = (finish_val-start_val)/self.data_size
		return num_children

	# we assume we are getting children named [0] thru [N-1]
	# if for some reason our child name is not in this format,
	# do not bother to show it, and return an invalid value
	def get_child_index(self,name):
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1;

	def get_child_at_index(self,index):
		# LLDB itself should never query for children < 0, but this might come
		# from someone asking for a nonexisting child and getting -1 as index
		if index < 0:
			return None
		if index >= self.num_children():
			return None;
		# *(_M_start + index), or equivalently _M_start[index] is C++ code to
		# read the index-th item of the vector. in Python we must make an offset
		# that is index * sizeof(T), and then grab the value at that offset from
		# _M_start
		offset = index * self.data_size # index * sizeof(T)
		return self.start.CreateChildAtOffset('['+str(index)+']',offset,self.data_type) # *(_M_start + index)

	# an std::vector contains an object named _M_impl, which in turn contains
	# three pointers, _M_start, _M_end and _M_end_of_storage. _M_start points to the
	# beginning of the data area, _M_finish points to where the current vector elements
	# finish, and _M_end_of_storage is the end of the currently alloc'ed memory portion
	# (to allow resizing, a vector may allocate more memory than required)
	def update(self):
		impl = self.valobj.GetChildMemberWithName('_M_impl')
		self.start = impl.GetChildMemberWithName('_M_start')
		self.finish = impl.GetChildMemberWithName('_M_finish')
		self.end = impl.GetChildMemberWithName('_M_end_of_storage')
		self.data_type = self.start.GetType().GetPointeeType() # _M_start is defined as a T*
		self.data_size = self.data_type.GetByteSize() # sizeof(T)

